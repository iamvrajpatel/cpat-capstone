"""
CPAT — Core Domain Models
==========================
Frozen, immutable dataclasses representing every first-class entity in
the trading system. Immutability is enforced at runtime (``frozen=True``)
to prevent accidental mutation across event-handler boundaries.

All monetary values are in the instrument's native currency.
All timestamps are UTC-aware ``pandas.Timestamp`` objects.

Design notes:
    - ``frozen=True`` makes instances hashable (safe as dict keys / set members).
    - ``__slots__`` is implicitly set by ``@dataclass(slots=True)`` (Python 3.10+),
      giving a memory footprint ~40% smaller than plain dict-backed instances.
    - No business logic lives here — models are pure data containers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

import pandas as pd

from cpat.core.enums import (
    AssetClass,
    CommissionModel,
    Interval,
    OrderSide,
    OrderStatus,
    OrderType,
    SignalDirection,
    SlippageModel,
)


# ── Market Data ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Bar:
    """OHLCV bar for a single instrument at a specific interval.

    Attributes:
        symbol: Ticker symbol (e.g. ``"AAPL"``).
        timestamp: UTC bar open timestamp.
        open: Opening price.
        high: Session high price.
        low: Session low price.
        close: Unadjusted closing price.
        volume: Traded volume (shares / contracts).
        adj_close: Split- and dividend-adjusted close.
        interval: Bar frequency.
        vwap: Volume-weighted average price (optional; provider-dependent).
    """

    symbol: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_close: float
    interval: Interval = Interval.ONE_DAY
    vwap: Optional[float] = None

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(
                f"Bar invariant violated for {self.symbol}@{self.timestamp}: "
                f"high ({self.high}) < low ({self.low})"
            )
        if any(p <= 0 for p in (self.open, self.high, self.low, self.close, self.adj_close)):
            raise ValueError(
                f"Non-positive price detected for {self.symbol}@{self.timestamp}"
            )
        if self.volume < 0:
            raise ValueError(
                f"Negative volume for {self.symbol}@{self.timestamp}: {self.volume}"
            )

    @property
    def mid(self) -> float:
        """Midpoint of the bar's high-low range."""
        return (self.high + self.low) / 2.0

    @property
    def body(self) -> float:
        """Absolute candle body size (|close - open|)."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        """True when close > open."""
        return self.close > self.open


@dataclass(frozen=True, slots=True)
class Tick:
    """Level-1 quote / trade tick.

    Attributes:
        symbol: Ticker symbol.
        timestamp: UTC tick timestamp.
        bid: Best bid price.
        ask: Best ask price.
        last: Last trade price.
        volume: Tick volume (shares).
    """

    symbol: str
    timestamp: pd.Timestamp
    bid: float
    ask: float
    last: float
    volume: float

    @property
    def spread(self) -> float:
        """Absolute bid-ask spread."""
        return self.ask - self.bid

    @property
    def mid(self) -> float:
        """Mid-market price."""
        return (self.bid + self.ask) / 2.0


# ── Signal ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Signal:
    """Trading signal produced by a strategy.

    A signal is a *recommendation*, not a commitment.  The portfolio manager
    and risk engine decide whether to act on it and at what size.

    Attributes:
        signal_id: Unique identifier (auto-generated UUID4).
        strategy_id: Identifier of the strategy that generated this signal.
        symbol: Target instrument.
        direction: LONG, SHORT, or FLAT.
        strength: Normalised signal strength in [-1.0, 1.0].
        timestamp: UTC timestamp when the signal was generated.
        metadata: Arbitrary key-value annotations (e.g. indicator values).
    """

    strategy_id: str
    symbol: str
    direction: SignalDirection
    timestamp: pd.Timestamp
    strength: float = 0.0
    signal_id: UUID = field(default_factory=uuid4)
    metadata: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not -1.0 <= self.strength <= 1.0:
            raise ValueError(
                f"Signal strength must be in [-1, 1], got {self.strength}"
            )


# ── Order ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Order:
    """Instruction to the execution engine to trade a quantity of an instrument.

    Attributes:
        order_id: Unique identifier (auto-generated UUID4).
        symbol: Target instrument.
        side: BUY or SELL.
        order_type: MARKET, LIMIT, STOP, or STOP_LIMIT.
        quantity: Number of shares / contracts (always positive).
        limit_price: Required for LIMIT and STOP_LIMIT orders.
        stop_price: Required for STOP and STOP_LIMIT orders.
        timestamp: UTC order creation timestamp.
        strategy_id: Originating strategy identifier.
        signal_id: UUID of the signal that triggered this order.
    """

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    timestamp: pd.Timestamp
    strategy_id: str
    order_id: UUID = field(default_factory=uuid4)
    signal_id: Optional[UUID] = None
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {self.quantity}")
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and self.limit_price is None:
            raise ValueError(f"LIMIT order requires limit_price")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError(f"STOP order requires stop_price")


# ── Fill ───────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Fill:
    """Confirmed execution of an order (or partial execution).

    Attributes:
        fill_id: Unique identifier.
        order_id: Reference to the originating Order.
        symbol: Traded instrument.
        side: BUY or SELL.
        quantity: Executed quantity (may be < order quantity for partials).
        fill_price: Achieved execution price (inclusive of slippage).
        commission: Total brokerage commission charged.
        slippage: Absolute slippage from mid-market (informational).
        timestamp: UTC fill confirmation timestamp.
        exchange: Execution venue (optional).
    """

    order_id: UUID
    symbol: str
    side: OrderSide
    quantity: float
    fill_price: float
    commission: float
    slippage: float
    timestamp: pd.Timestamp
    fill_id: UUID = field(default_factory=uuid4)
    exchange: str = "SIMULATED"

    @property
    def gross_value(self) -> float:
        """Gross trade value (fill_price × quantity)."""
        return self.fill_price * self.quantity

    @property
    def net_value(self) -> float:
        """Net trade value including commission cost."""
        return self.gross_value + self.commission

    @property
    def cost_bps(self) -> float:
        """Total transaction cost in basis points of gross value."""
        if self.gross_value == 0:
            return 0.0
        return (self.commission + abs(self.slippage * self.quantity)) / self.gross_value * 10_000


# ── Position ───────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Position:
    """Mutable representation of a current holding in a single instrument.

    Unlike other models, Position is **mutable** because it is updated
    incrementally with each fill.  It is intentionally NOT frozen.

    Attributes:
        symbol: Held instrument.
        quantity: Signed holding (positive = long, negative = short).
        avg_cost: Volume-weighted average cost basis per share.
        realised_pnl: Cumulative realised profit-and-loss.
        open_timestamp: UTC timestamp of the first fill that opened the position.
    """

    symbol: str
    quantity: float = 0.0
    avg_cost: float = 0.0
    realised_pnl: float = 0.0
    open_timestamp: Optional[pd.Timestamp] = None

    def apply_fill(self, fill: Fill) -> None:
        """Update position with a new fill using FIFO cost accounting.

        Args:
            fill: The confirmed fill to apply.
        """
        signed_qty = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity

        if self.quantity == 0:
            # Opening a new position
            self.quantity = signed_qty
            self.avg_cost = fill.fill_price
            self.open_timestamp = fill.timestamp
        elif (self.quantity > 0 and signed_qty > 0) or (self.quantity < 0 and signed_qty < 0):
            # Adding to existing position — update VWAC
            total_cost = self.avg_cost * abs(self.quantity) + fill.fill_price * fill.quantity
            self.quantity += signed_qty
            self.avg_cost = total_cost / abs(self.quantity)
        else:
            # Reducing or flipping position
            closed_qty = min(abs(signed_qty), abs(self.quantity))
            pnl_per_share = (fill.fill_price - self.avg_cost) * (1 if self.quantity > 0 else -1)
            self.realised_pnl += pnl_per_share * closed_qty - fill.commission
            self.quantity += signed_qty
            if abs(self.quantity) < 1e-9:
                self.quantity = 0.0
                self.avg_cost = 0.0
                self.open_timestamp = None
            elif (self.quantity > 0 and signed_qty > 0) or (self.quantity < 0 and signed_qty < 0):
                # Flipped — reset avg cost
                self.avg_cost = fill.fill_price

    def unrealised_pnl(self, current_price: float) -> float:
        """Compute mark-to-market unrealised P&L.

        Args:
            current_price: Current market price of the instrument.

        Returns:
            Unrealised P&L in currency units.
        """
        if self.quantity == 0:
            return 0.0
        return (current_price - self.avg_cost) * self.quantity

    def market_value(self, current_price: float) -> float:
        """Current market value of the position.

        Args:
            current_price: Current market price.

        Returns:
            Signed market value (negative for short positions).
        """
        return current_price * self.quantity

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0


# ── Instrument ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Instrument:
    """Static metadata for a tradeable instrument.

    Attributes:
        symbol: Primary ticker symbol.
        name: Human-readable name.
        asset_class: Broad asset classification.
        exchange: Primary listing exchange.
        currency: ISO 4217 currency code.
        sector: GICS sector (equities only).
        lot_size: Minimum tradeable lot (default 1 share).
        tick_size: Minimum price increment.
    """

    symbol: str
    name: str
    asset_class: AssetClass
    exchange: str
    currency: str = "USD"
    sector: Optional[str] = None
    lot_size: float = 1.0
    tick_size: float = 0.01


# ── Cost Models ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CostConfig:
    """Transaction cost parameters used by the simulated execution engine.

    Attributes:
        commission_model: How commission is calculated.
        commission_value: Flat rate per share OR percentage (0.001 = 0.1%).
        slippage_model: How slippage is estimated.
        slippage_bps: Fixed slippage in basis points (for FIXED_BPS model).
        min_commission: Minimum per-order commission (e.g. $1.00).
    """

    commission_model: CommissionModel = CommissionModel.PERCENTAGE
    commission_value: float = 0.0005  # 5 bps (0.05%)
    slippage_model: SlippageModel = SlippageModel.FIXED_BPS
    slippage_bps: float = 5.0  # 5 bps one-way
    min_commission: float = 1.0  # USD

    def compute_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for a single order.

        Args:
            quantity: Number of shares.
            price: Execution price per share.

        Returns:
            Commission in currency units.
        """
        match self.commission_model:
            case CommissionModel.ZERO:
                return 0.0
            case CommissionModel.PER_SHARE:
                return max(self.commission_value * quantity, self.min_commission)
            case CommissionModel.PERCENTAGE:
                return max(self.commission_value * quantity * price, self.min_commission)
            case _:
                raise NotImplementedError(f"Commission model {self.commission_model} not implemented")

    def compute_slippage(self, price: float) -> float:
        """Calculate per-share slippage amount.

        Args:
            price: Reference execution price.

        Returns:
            Slippage amount per share (always positive).
        """
        match self.slippage_model:
            case SlippageModel.ZERO:
                return 0.0
            case SlippageModel.FIXED_BPS:
                return price * (self.slippage_bps / 10_000)
            case _:
                raise NotImplementedError(f"Slippage model {self.slippage_model} not implemented")
