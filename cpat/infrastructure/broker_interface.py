"""
CPAT — Broker Interface Abstraction (Week 5)
=============================================
Defines the Protocol that every broker adapter must implement, plus the
canonical data structures exchanged between the live engine and brokers.

Architecture (Adapter Pattern)
-------------------------------
                    ┌──────────────────────┐
                    │   BrokerInterface    │  ← Protocol (structural typing)
                    └────────┬─────────────┘
                             │ implemented by
               ┌─────────────┼────────────────┐
               │             │                │
        PaperBroker     DhanBroker      (future)
        (in-memory)   (HTTP/REST)     ZerodhaBroker

All live-trading components depend ONLY on BrokerInterface — broker
swap requires zero changes to strategy, risk, or portfolio code.

Key design decisions
---------------------
* Methods return typed dataclasses, never raw dicts.
* Every method can raise ``BrokerError`` — callers must handle it.
* Credentials are NEVER stored inside the adapter; they are passed via
  environment variables and read at construction time.
* Retry logic lives in the callers (``LiveExecutionEngine``), not here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Protocol, runtime_checkable
from uuid import UUID, uuid4

import pandas as pd

from cpat.core.enums import OrderSide, OrderStatus, OrderType
from cpat.core.models import Order

logger = logging.getLogger(__name__)


# ── Broker Error ───────────────────────────────────────────────────────────────


class BrokerError(Exception):
    """Raised when a broker operation fails.

    Attributes:
        code   : Broker-specific error code (string or int).
        retryable: True if the caller should retry the operation.
    """

    def __init__(self, message: str, code: str = "", retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class BrokerConnectionError(BrokerError):
    """Network / connectivity failure."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="CONNECTION_ERROR", retryable=True)


class BrokerAuthError(BrokerError):
    """Authentication / authorisation failure — do NOT retry."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="AUTH_ERROR", retryable=False)


class BrokerOrderError(BrokerError):
    """Order-level rejection (insufficient funds, invalid symbol, etc.)."""

    def __init__(self, message: str, code: str = "ORDER_ERROR", retryable: bool = False) -> None:
        super().__init__(message, code=code, retryable=retryable)


# ── Broker Data Structures ─────────────────────────────────────────────────────


@dataclass
class BrokerOrderStatus:
    """Live status of an order as reported by the broker.

    Attributes:
        broker_order_id : Broker-assigned order identifier.
        internal_order_id: Our internal ``Order.order_id``.
        status          : Canonical ``OrderStatus`` enum.
        filled_qty      : Quantity filled so far.
        avg_fill_price  : Volume-weighted average fill price.
        remaining_qty   : Quantity still pending.
        rejected_reason : Rejection message (populated only on REJECTED).
        timestamp       : Broker-side update timestamp.
    """

    broker_order_id: str
    internal_order_id: UUID
    status: OrderStatus
    filled_qty: float
    avg_fill_price: float
    remaining_qty: float
    rejected_reason: str = ""
    timestamp: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.utcnow())

    @property
    def is_terminal(self) -> bool:
        """True when the order can no longer change state."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )


@dataclass
class BrokerPosition:
    """A single open position as reported by the broker.

    Attributes:
        symbol      : Instrument ticker.
        quantity    : Net signed quantity (positive = long).
        avg_cost    : Average entry cost per share.
        current_price: Latest market price (broker-provided).
        unrealised_pnl: Broker-computed unrealised P&L.
    """

    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    unrealised_pnl: float = 0.0


@dataclass
class BrokerAccountInfo:
    """Snapshot of account balances and margin.

    Attributes:
        cash_balance    : Available cash.
        portfolio_value : Total equity (cash + positions).
        buying_power    : Deployable buying power.
        currency        : ISO 4217 code.
        timestamp       : Snapshot time.
    """

    cash_balance: float
    portfolio_value: float
    buying_power: float
    currency: str = "INR"
    timestamp: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.utcnow())


@dataclass
class BrokerQuote:
    """Real-time or delayed quote for one instrument.

    Attributes:
        symbol   : Ticker symbol.
        bid      : Best bid.
        ask      : Best ask.
        last     : Last traded price.
        volume   : Day volume so far.
        timestamp: Quote timestamp.
    """

    symbol: str
    bid: float
    ask: float
    last: float
    volume: float = 0.0
    timestamp: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.utcnow())

    @property
    def mid(self) -> float:
        """Mid-market price."""
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        """Absolute bid-ask spread."""
        return self.ask - self.bid


# ── Broker Protocol ────────────────────────────────────────────────────────────


@runtime_checkable
class BrokerInterface(Protocol):
    """Structural protocol that every broker adapter must satisfy.

    All methods may raise subclasses of ``BrokerError``.  Callers are
    responsible for retry/fallback logic.

    Implementation note: adapters should be *stateless* between calls where
    possible.  State (e.g. session tokens) may be cached but must be
    re-validated lazily.
    """

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Establish / validate connection to the broker.

        Raises:
            BrokerConnectionError: If network is unreachable.
            BrokerAuthError: If credentials are invalid.
        """
        ...

    def disconnect(self) -> None:
        """Gracefully close the broker session."""
        ...

    @property
    def is_connected(self) -> bool:
        """True if the broker session is active."""
        ...

    # ── Orders ─────────────────────────────────────────────────────────────────

    def place_order(self, order: Order) -> str:
        """Submit an order to the broker.

        Args:
            order: The ``Order`` domain object to submit.

        Returns:
            Broker-assigned order ID string.

        Raises:
            BrokerOrderError: If the order is rejected immediately.
            BrokerConnectionError: If the network call fails.
        """
        ...

    def cancel_order(self, broker_order_id: str) -> bool:
        """Request cancellation of a pending order.

        Args:
            broker_order_id: The broker-side order identifier.

        Returns:
            True if the cancellation was accepted (order may still fill
            before cancellation reaches the exchange).

        Raises:
            BrokerError: On failure.
        """
        ...

    def get_order_status(self, broker_order_id: str) -> BrokerOrderStatus:
        """Poll the current status of a single order.

        Args:
            broker_order_id: Broker-side order ID.

        Returns:
            ``BrokerOrderStatus`` with latest state.

        Raises:
            BrokerError: On network/API failure.
        """
        ...

    # ── Market Data ────────────────────────────────────────────────────────────

    def get_market_data(self, symbol: str) -> BrokerQuote:
        """Fetch the latest quote for an instrument.

        Args:
            symbol: Ticker symbol.

        Returns:
            ``BrokerQuote`` with bid, ask, last.

        Raises:
            BrokerError: On failure or stale/missing data.
        """
        ...

    def get_quote(self, symbol: str) -> BrokerQuote:
        """Backward-compatible alias for ``get_market_data()``."""
        ...

    # ── Account ────────────────────────────────────────────────────────────────

    def get_account_info(self) -> BrokerAccountInfo:
        """Retrieve current account balances.

        Returns:
            ``BrokerAccountInfo`` snapshot.

        Raises:
            BrokerError: On failure.
        """
        ...

    def get_positions(self) -> list[BrokerPosition]:
        """Retrieve all open positions.

        Returns:
            List of ``BrokerPosition`` objects (empty list = no positions).

        Raises:
            BrokerError: On failure.
        """
        ...
