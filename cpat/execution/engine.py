"""
CPAT — Execution Engine v2
===========================
Production-grade simulated order execution with three slippage models
and configurable commission structures.

Execution Model (anti-bias):
    All orders submitted at bar T are executed at bar T+1's **open price**.
    This is the industry-standard "trade-on-next-open" assumption, which:
        1. Prevents using signal prices as fill prices.
        2. Accounts for the fact that a trade takes time to route and fill.
        3. Produces pessimistic (conservative) P&L estimates.

Slippage Models:
    1. FIXED_BPS     — fixed cost regardless of order size (default)
    2. VOLUME_WEIGHTED — slippage scales with order_qty / bar_volume
    3. SPREAD_BASED  — slippage = 0.5 × estimated_spread (from high-low)

Commission Models:
    - ZERO        — no commission (unrealistic baseline)
    - PER_SHARE   — flat $ per share (e.g. $0.005/share)
    - PERCENTAGE  — % of notional value (e.g. 0.1%)

Partial Fill Simulation:
    If ``fill_ratio < 1.0``, a random fraction of the order is filled.
    The remainder is carried to the next bar (configurable via config).
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from cpat.core.enums import OrderSide, SlippageModel as CoreSlippageModel
from cpat.core.models import Bar, CostConfig, Fill, Order
from cpat.core.events import FillEvent
from cpat.backtest.event_queue import EventQueue

logger = logging.getLogger(__name__)


# ── Slippage model hierarchy ───────────────────────────────────────────────────


class AbstractSlippageModel(ABC):
    """Base class for all slippage estimation strategies."""

    @abstractmethod
    def compute(self, order: Order, bar: Bar) -> float:
        """Compute the per-share slippage for a given order and bar.

        Args:
            order: The order being filled.
            bar: The bar at which execution occurs (next bar's open).

        Returns:
            Slippage amount per share (always positive).
        """


class FixedBpsSlippage(AbstractSlippageModel):
    """Fixed slippage in basis points — independent of order size.

    Simple and conservative. Appropriate for liquid, large-cap equities.

    Args:
        bps: Slippage in basis points (e.g. 5.0 = 5 bps = 0.05%).
    """

    def __init__(self, bps: float = 5.0) -> None:
        self._bps = bps

    def compute(self, order: Order, bar: Bar) -> float:
        return bar.open * (self._bps / 10_000)


class VolumeWeightedSlippage(AbstractSlippageModel):
    """Volume-impact slippage model — scales with order size relative to volume.

    Models the market impact of larger orders:
        slippage = base_bps + impact_bps × (order_qty / bar_volume)

    This penalises large orders on thin volume, which is realistic.

    Args:
        base_bps: Minimum slippage (bps) for any order size.
        impact_factor: Multiplier for the volume-ratio component.
    """

    def __init__(self, base_bps: float = 3.0, impact_factor: float = 0.1) -> None:
        self._base_bps = base_bps
        self._impact_factor = impact_factor

    def compute(self, order: Order, bar: Bar) -> float:
        if bar.volume <= 0:
            return bar.open * (self._base_bps / 10_000)

        volume_ratio = order.quantity / bar.volume
        effective_bps = self._base_bps + self._impact_factor * volume_ratio * 10_000
        return bar.open * (effective_bps / 10_000)


class SpreadBasedSlippage(AbstractSlippageModel):
    """Half-spread slippage estimated from the bar's high-low range.

    Approximates the bid-ask spread as a fraction of the high-low range.
    Slippage = 0.5 × (spread_fraction × (high - low)).

    Args:
        spread_fraction: Fraction of (high-low) to use as spread estimate.
                         Typically 0.1–0.3 for daily bars.
    """

    def __init__(self, spread_fraction: float = 0.15) -> None:
        self._spread_fraction = spread_fraction

    def compute(self, order: Order, bar: Bar) -> float:
        estimated_spread = self._spread_fraction * (bar.high - bar.low)
        return 0.5 * estimated_spread  # We pay half the spread


# ── Factory ────────────────────────────────────────────────────────────────────


def build_slippage_model(
    model: str | CoreSlippageModel,
    **kwargs: float,
) -> AbstractSlippageModel:
    """Factory: create a slippage model from a string identifier or enum.

    Args:
        model: ``"FIXED_BPS"``, ``"VOLUME_WEIGHTED"``, ``"SPREAD_BASED"``,
               or ``"ZERO"``, or the corresponding ``SlippageModel`` enum.
        **kwargs: Model-specific parameters.

    Returns:
        Configured slippage model instance.
    """
    if isinstance(model, CoreSlippageModel):
        model = model.value

    match model.upper():
        case "ZERO":
            return FixedBpsSlippage(bps=0.0)
        case "FIXED_BPS":
            return FixedBpsSlippage(bps=kwargs.get("bps", 5.0))
        case "VOLUME_WEIGHTED":
            return VolumeWeightedSlippage(
                base_bps=kwargs.get("base_bps", 3.0),
                impact_factor=kwargs.get("impact_factor", 0.1),
            )
        case "SPREAD_BASED":
            return SpreadBasedSlippage(
                spread_fraction=kwargs.get("spread_fraction", 0.15),
            )
        case _:
            raise ValueError(f"Unknown slippage model: {model!r}")


# ── Execution Engine ───────────────────────────────────────────────────────────


@dataclass
class ExecutionResult:
    """Summary of order execution for a single bar.

    Attributes:
        fills: List of Fill objects created.
        rejected: Orders rejected (e.g. insufficient bars volume).
        deferred: Orders carried to the next bar (partial fill mode).
    """

    fills: list[Fill] = field(default_factory=list)
    rejected: list[Order] = field(default_factory=list)
    deferred: list[Order] = field(default_factory=list)

    @property
    def n_filled(self) -> int:
        return len(self.fills)

    @property
    def total_commission(self) -> float:
        return sum(f.commission for f in self.fills)

    @property
    def total_slippage(self) -> float:
        return sum(f.slippage for f in self.fills)


class ExecutionEngine:
    """Simulated order execution engine for backtesting.

    Implements next-bar-open execution with pluggable slippage and
    commission models.  Supports optional partial fill simulation.

    Args:
        cost_config: Commission parameters from the master config.
        slippage_model: Pluggable slippage model instance.
        event_queue: Event bus for posting FillEvents.
        fill_ratio: Fraction of each order filled per bar (1.0 = full fill).
                    Values < 1.0 simulate partial fills.
        random_seed: Seed for partial fill randomisation (reproducibility).
    """

    def __init__(
        self,
        cost_config: CostConfig,
        event_queue: EventQueue,
        slippage_model: AbstractSlippageModel | None = None,
        fill_ratio: float = 1.0,
        random_seed: int = 42,
    ) -> None:
        self._cost_config = cost_config
        self._event_queue = event_queue
        self._slippage_model = slippage_model or FixedBpsSlippage(
            bps=cost_config.slippage_bps
        )
        self._fill_ratio = max(0.0, min(1.0, fill_ratio))
        self._rng = random.Random(random_seed)

        self._pending_orders: list[Order] = []
        self._all_fills: list[Fill] = []

        logger.info(
            "ExecutionEngine initialised: slippage=%s, fill_ratio=%.2f",
            self._slippage_model.__class__.__name__,
            self._fill_ratio,
        )

    @classmethod
    def from_cost_config(
        cls,
        cost_config: CostConfig,
        event_queue: EventQueue,
        slippage_model_name: str = "FIXED_BPS",
        fill_ratio: float = 1.0,
        **slippage_kwargs: float,
    ) -> "ExecutionEngine":
        """Factory: build ExecutionEngine from a CostConfig.

        Args:
            cost_config: Commission parameters.
            event_queue: Event bus.
            slippage_model_name: One of FIXED_BPS, VOLUME_WEIGHTED, SPREAD_BASED.
            fill_ratio: Partial fill ratio (1.0 = full fill).
            **slippage_kwargs: Parameters passed to the slippage model.

        Returns:
            Configured ExecutionEngine.
        """
        slippage_model = build_slippage_model(slippage_model_name, **slippage_kwargs)
        return cls(
            cost_config=cost_config,
            event_queue=event_queue,
            slippage_model=slippage_model,
            fill_ratio=fill_ratio,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def submit(self, order: Order) -> None:
        """Queue an order for deferred execution at the next bar's open.

        Args:
            order: Order to queue.
        """
        self._pending_orders.append(order)
        logger.debug(
            "Order queued [%s]: %s %s %.0f",
            order.order_id, order.side.value, order.symbol, order.quantity,
        )

    def process_pending(
        self,
        bars: dict[str, Bar],
        timestamp: pd.Timestamp,
    ) -> ExecutionResult:
        """Execute all pending orders at the current bar's open price.

        Called at the **start** of each bar, before the strategy sees
        the bar — guaranteeing no look-ahead in fill prices.

        Args:
            bars: Current-bar data keyed by symbol.
            timestamp: Current bar timestamp (fill timestamp).

        Returns:
            ExecutionResult with fills, rejected, and deferred orders.
        """
        result = ExecutionResult()
        still_pending: list[Order] = []

        for order in self._pending_orders:
            bar = bars.get(order.symbol)

            # No bar available — defer to next bar
            if bar is None:
                still_pending.append(order)
                result.deferred.append(order)
                continue

            fill = self._execute_order(order, bar, timestamp)
            if fill is not None:
                result.fills.append(fill)
                self._all_fills.append(fill)
                self._event_queue.put(FillEvent.from_fill(fill))
            else:
                result.rejected.append(order)

        self._pending_orders = still_pending
        return result

    # ── Private helpers ────────────────────────────────────────────────────────

    def _execute_order(
        self,
        order: Order,
        bar: Bar,
        timestamp: pd.Timestamp,
    ) -> Fill | None:
        """Execute a single order against a bar's open price.

        Args:
            order: Order to execute.
            bar: Execution bar (open price used).
            timestamp: Fill confirmation timestamp.

        Returns:
            Fill object, or None if execution was rejected.
        """
        reference_price = bar.open

        # Slippage: adverse for both buys (higher) and sells (lower)
        slippage_per_share = self._slippage_model.compute(order, bar)
        if order.side == OrderSide.BUY:
            fill_price = reference_price + slippage_per_share
        else:
            fill_price = reference_price - slippage_per_share

        # Guard: fill price must be positive
        if fill_price <= 0:
            logger.warning(
                "Rejected: fill price ≤ 0 for %s (open=%.4f, slip=%.4f)",
                order.symbol, reference_price, slippage_per_share,
            )
            return None

        # Partial fill simulation
        actual_qty = order.quantity
        if self._fill_ratio < 1.0:
            ratio = self._rng.uniform(self._fill_ratio * 0.8, self._fill_ratio)
            actual_qty = max(1.0, round(order.quantity * ratio))

        commission = self._cost_config.compute_commission(actual_qty, fill_price)

        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=actual_qty,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_per_share * actual_qty,
            timestamp=timestamp,
        )

        logger.debug(
            "Fill: %s %s %.0f @ %.4f | slip=%.4f | comm=%.2f | total_cost=%.2f bps",
            fill.side.value, fill.symbol, fill.quantity, fill.fill_price,
            fill.slippage, fill.commission, fill.cost_bps,
        )

        return fill

    @property
    def pending_order_count(self) -> int:
        """Number of orders awaiting execution."""
        return len(self._pending_orders)

    @property
    def all_fills(self) -> list[Fill]:
        """All fills executed by this engine (read-only copy)."""
        return list(self._all_fills)

    def clear_pending(self) -> None:
        """Cancel all pending orders (e.g. at end-of-day or market close)."""
        n = len(self._pending_orders)
        self._pending_orders.clear()
        if n > 0:
            logger.info("Cleared %d pending orders.", n)
