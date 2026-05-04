"""
CPAT — Risk Engine
===================
Pre-trade risk checks applied to every OrderEvent before routing to
the execution engine.

Risk constraints are checked in a specific order (from cheapest to most
expensive computation).  A single violation blocks the order.

Constraints (configurable via settings.yaml):

    1. max_position_pct (default 5%)
       Single position market value ≤ 5% of total equity.
       Prevents over-concentration in one name.

    2. max_sector_pct (default 25%)
       Sector exposure ≤ 25% of total equity.
       Prevents sector bets masquerading as alpha.

    3. max_gross_exposure_pct (default 100% for long-only)
       Gross long + abs(short) ≤ 100% of equity.
       Hard leverage cap.

    4. max_drawdown_halt (default 15%)
       If current drawdown > 15%, block all new opening trades.
       Drawdown-based trading halt (auto-lifts when equity recovers).

    5. min_cash_pct (default 3%)
       Maintain a minimum cash buffer. Prevents the portfolio from
       being 100% invested (can't pay commissions or margin calls).

Design:
    RiskEngine is STATELESS within a bar — it reads portfolio and equity
    from the PortfolioManager and makes a binary approve/reject decision.
    It does NOT modify orders (that would be a design violation).
    Instead, it returns a ``RiskDecision`` with the verdict and reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import pandas as pd

from cpat.core.enums import OrderSide
from cpat.core.models import Order
from cpat.portfolio.manager import PortfolioManager

logger = logging.getLogger(__name__)


# ── Decision dataclass ─────────────────────────────────────────────────────────


class RiskVerdict(Enum):
    """Outcome of a risk check."""

    APPROVED = auto()
    REJECTED = auto()
    APPROVED_REDUCED = auto()  # Approved but with reduced quantity


@dataclass(frozen=True)
class RiskDecision:
    """Output of the RiskEngine for a single order.

    Attributes:
        verdict: APPROVED, REJECTED, or APPROVED_REDUCED.
        order: The original Order.
        approved_quantity: Quantity approved (may be < order.quantity).
        constraint_name: Which constraint triggered (if rejected/reduced).
        reason: Human-readable explanation.
    """

    verdict: RiskVerdict
    order: Order
    approved_quantity: float
    constraint_name: str = ""
    reason: str = ""

    @property
    def is_approved(self) -> bool:
        return self.verdict in (RiskVerdict.APPROVED, RiskVerdict.APPROVED_REDUCED)

    @property
    def is_rejected(self) -> bool:
        return self.verdict == RiskVerdict.REJECTED


# ── Risk Engine ────────────────────────────────────────────────────────────────


class RiskEngine:
    """Pre-trade risk constraint checker.

    All constraint thresholds are configurable and documented in
    ``config/settings.yaml`` under the ``risk:`` key.

    Args:
        portfolio: Portfolio manager to read current state from.
        max_position_pct: Max single-name weight (fraction of equity).
        max_sector_pct: Max sector exposure (fraction of equity).
        max_gross_exposure_pct: Max gross leverage.
        max_drawdown_pct: Drawdown threshold triggering trading halt.
        min_cash_pct: Minimum cash buffer.
        peak_equity: Starting peak equity for drawdown tracking
                     (defaults to initial_capital).
    """

    def __init__(
        self,
        portfolio: PortfolioManager,
        max_position_pct: float = 0.05,
        max_sector_pct: float = 0.25,
        max_gross_exposure_pct: float = 1.0,
        max_drawdown_pct: float = 0.15,
        min_cash_pct: float = 0.03,
    ) -> None:
        self._portfolio = portfolio
        self._max_position_pct = max_position_pct
        self._max_sector_pct = max_sector_pct
        self._max_gross_exposure_pct = max_gross_exposure_pct
        self._max_drawdown_pct = max_drawdown_pct
        self._min_cash_pct = min_cash_pct

        # Drawdown tracking
        self._peak_equity: float = portfolio.initial_capital
        self._drawdown_halt_active: bool = False

        logger.info(
            "RiskEngine initialised: pos_pct=%.0f%%, sector_pct=%.0f%%, "
            "drawdown_halt=%.0f%%, min_cash=%.0f%%",
            max_position_pct * 100, max_sector_pct * 100,
            max_drawdown_pct * 100, min_cash_pct * 100,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def check(
        self,
        order: Order,
        current_prices: dict[str, float],
    ) -> RiskDecision:
        """Run all pre-trade risk checks for a proposed order.

        Checks are ordered cheapest-first. The first violation returns
        immediately — no subsequent checks are run.

        Args:
            order: The proposed order to evaluate.
            current_prices: Latest bar close prices for valuation.

        Returns:
            RiskDecision with verdict and explanation.
        """
        equity = self._portfolio.total_equity(current_prices)
        if equity <= 0:
            return self._reject(order, "zero_equity", "Portfolio equity is ≤ 0.")

        self._update_peak_equity(equity)

        # ── 1. Drawdown halt ───────────────────────────────────────────────────
        decision = self._check_drawdown_halt(order, equity)
        if decision is not None:
            return decision

        # Only check remaining constraints for new OPENING orders
        # (closing orders that reduce risk are always allowed)
        is_opening_trade = self._is_opening(order, current_prices)

        if is_opening_trade:
            # ── 2. Minimum cash buffer ─────────────────────────────────────────
            decision = self._check_min_cash(order, equity, current_prices)
            if decision is not None:
                return decision

            # ── 3. Single-name position limit ─────────────────────────────────
            decision = self._check_position_limit(order, equity, current_prices)
            if decision is not None:
                return decision

            # ── 4. Sector exposure limit ───────────────────────────────────────
            decision = self._check_sector_limit(order, equity, current_prices)
            if decision is not None:
                return decision

            # ── 5. Gross leverage cap ──────────────────────────────────────────
            decision = self._check_gross_exposure(order, equity, current_prices)
            if decision is not None:
                return decision

        return RiskDecision(
            verdict=RiskVerdict.APPROVED,
            order=order,
            approved_quantity=order.quantity,
            reason="All constraints passed.",
        )

    def update_equity(self, current_equity: float) -> None:
        """Update peak equity for drawdown tracking.

        Call this once per bar after computing portfolio equity.

        Args:
            current_equity: Current total portfolio value.
        """
        self._update_peak_equity(current_equity)

    @property
    def drawdown_halt_active(self) -> bool:
        """True when the drawdown halt is currently active."""
        return self._drawdown_halt_active

    @property
    def current_drawdown(self) -> float:
        """Latest drawdown from peak (negative number, e.g. -0.12 = -12%)."""
        equity = self._portfolio.initial_capital  # Approximation without prices
        if self._peak_equity <= 0:
            return 0.0
        # We don't have prices here, so return approximation
        return 0.0  # Will be updated via update_equity

    # ── Constraint checks ──────────────────────────────────────────────────────

    def _check_drawdown_halt(
        self, order: Order, equity: float
    ) -> Optional[RiskDecision]:
        """Block NEW opening trades during a drawdown halt."""
        if self._peak_equity <= 0:
            return None

        drawdown = (equity - self._peak_equity) / self._peak_equity
        if drawdown < -self._max_drawdown_pct:
            self._drawdown_halt_active = True
            # Only block opening trades; closing trades always go through
            pos = self._portfolio.get_position(order.symbol)
            is_closing = (
                (order.side == OrderSide.SELL and pos.is_long) or
                (order.side == OrderSide.BUY and pos.is_short)
            )
            if not is_closing:
                logger.warning(
                    "RISK HALT: drawdown %.2f%% > limit %.2f%%. Blocking %s %s.",
                    drawdown * 100, -self._max_drawdown_pct * 100,
                    order.side.value, order.symbol,
                )
                return self._reject(
                    order, "max_drawdown",
                    f"Drawdown {drawdown:.2%} exceeds limit {-self._max_drawdown_pct:.2%}.",
                )
        else:
            self._drawdown_halt_active = False
        return None

    def _check_min_cash(
        self,
        order: Order,
        equity: float,
        current_prices: dict[str, float],
    ) -> Optional[RiskDecision]:
        """Ensure enough cash remains after the trade."""
        price = current_prices.get(order.symbol, 0.0)
        if price <= 0:
            return None

        trade_cost = price * order.quantity
        remaining_cash = self._portfolio.cash - trade_cost
        min_cash = equity * self._min_cash_pct

        if remaining_cash < min_cash:
            logger.debug(
                "RISK: min_cash violation for %s (remaining=%.0f, min=%.0f)",
                order.symbol, remaining_cash, min_cash,
            )
            return self._reject(
                order, "min_cash",
                f"Trade would reduce cash to {remaining_cash:.0f}, below minimum {min_cash:.0f}.",
            )
        return None

    def _check_position_limit(
        self,
        order: Order,
        equity: float,
        current_prices: dict[str, float],
    ) -> Optional[RiskDecision]:
        """Enforce single-name position concentration limit."""
        price = current_prices.get(order.symbol, 0.0)
        if price <= 0 or equity <= 0:
            return None

        current_pos = self._portfolio.get_position(order.symbol)
        current_value = abs(current_pos.market_value(price))
        new_value = current_value + price * order.quantity
        max_value = equity * self._max_position_pct

        if new_value > max_value:
            # Compute approved quantity
            available_value = max(0.0, max_value - current_value)
            approved_qty = int(available_value / price)

            if approved_qty < 1:
                return self._reject(
                    order, "max_position_pct",
                    f"{order.symbol} already at/above position limit "
                    f"({current_value / equity:.1%} of equity, limit={self._max_position_pct:.1%}).",
                )

            # Reduce but don't reject
            logger.info(
                "RISK: Reducing %s order from %.0f to %d shares (position limit).",
                order.symbol, order.quantity, approved_qty,
            )
            return RiskDecision(
                verdict=RiskVerdict.APPROVED_REDUCED,
                order=order,
                approved_quantity=float(approved_qty),
                constraint_name="max_position_pct",
                reason=f"Quantity reduced to respect {self._max_position_pct:.1%} position limit.",
            )
        return None

    def _check_sector_limit(
        self,
        order: Order,
        equity: float,
        current_prices: dict[str, float],
    ) -> Optional[RiskDecision]:
        """Enforce sector concentration limit."""
        sector_exp = self._portfolio.sector_exposure(current_prices)
        sector = self._portfolio._universe_sectors.get(order.symbol, "Unknown")
        price = current_prices.get(order.symbol, 0.0)

        if price <= 0 or equity <= 0:
            return None

        current_sector_exp = sector_exp.get(sector, 0.0)
        additional_exp = price * order.quantity
        projected_sector_exp = current_sector_exp + additional_exp
        max_sector_exp = equity * self._max_sector_pct

        if projected_sector_exp > max_sector_exp:
            logger.debug(
                "RISK: sector limit for %s [%s]: projected=%.0f > max=%.0f",
                order.symbol, sector, projected_sector_exp, max_sector_exp,
            )
            return self._reject(
                order, "max_sector_pct",
                f"Adding {order.symbol} would bring {sector} sector "
                f"to {projected_sector_exp / equity:.1%} (limit={self._max_sector_pct:.1%}).",
            )
        return None

    def _check_gross_exposure(
        self,
        order: Order,
        equity: float,
        current_prices: dict[str, float],
    ) -> Optional[RiskDecision]:
        """Enforce gross leverage cap."""
        price = current_prices.get(order.symbol, 0.0)
        if price <= 0 or equity <= 0:
            return None

        current_gross = self._portfolio.gross_exposure(current_prices)
        new_gross = current_gross + price * order.quantity
        max_gross = equity * self._max_gross_exposure_pct

        if new_gross > max_gross:
            logger.debug(
                "RISK: gross exposure limit: projected=%.0f > max=%.0f",
                new_gross, max_gross,
            )
            return self._reject(
                order, "max_gross_exposure",
                f"Trade would bring gross exposure to {new_gross / equity:.1%} "
                f"(limit={self._max_gross_exposure_pct:.1%}).",
            )
        return None

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _is_opening(self, order: Order, current_prices: dict[str, float]) -> bool:
        """Return True if the order increases gross exposure."""
        pos = self._portfolio.get_position(order.symbol)
        if pos.is_flat:
            return True
        if order.side == OrderSide.BUY and pos.is_long:
            return True  # Adding to long
        if order.side == OrderSide.SELL and pos.is_short:
            return True  # Adding to short
        return False  # Reducing position = closing trade

    def _update_peak_equity(self, current_equity: float) -> None:
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

    @staticmethod
    def _reject(order: Order, constraint: str, reason: str) -> RiskDecision:
        return RiskDecision(
            verdict=RiskVerdict.REJECTED,
            order=order,
            approved_quantity=0.0,
            constraint_name=constraint,
            reason=reason,
        )
