"""
CPAT — Event System
====================
Typed event dataclasses that flow through the event bus during backtesting
and live trading.  Every event carries an ``event_type`` discriminator for
efficient dispatch without isinstance chains.

Design:
    - All events are ``frozen=True`` for thread safety and hashability.
    - ``EventType`` enum (from enums.py) is the canonical discriminator.
    - Handlers receive a Union[MarketEvent, SignalEvent, OrderEvent, FillEvent];
      they should match on ``event.event_type`` or use structural pattern matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pandas as pd

from cpat.core.enums import EventType, OrderSide, OrderType, SignalDirection
from cpat.core.models import Bar, Fill, Order, Signal


# ── Base ───────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class BaseEvent:
    """Abstract base for all system events.

    Attributes:
        event_id: Unique event identifier.
        event_type: Discriminator for fast dispatch.
        timestamp: UTC event creation time.
    """

    event_id: UUID
    event_type: EventType
    timestamp: pd.Timestamp


# ── Market Event ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MarketEvent(BaseEvent):
    """Emitted when a new bar of market data arrives.

    This is the primary stimulus for strategy ``on_bar`` handlers.

    Attributes:
        bars: Mapping of symbol → Bar for all symbols updated in this tick.
               A single MarketEvent may carry bars for multiple symbols
               when the data feed delivers them simultaneously.
    """

    bars: dict[str, Bar]

    @classmethod
    def from_bars(
        cls,
        bars: dict[str, Bar],
        timestamp: pd.Timestamp,
    ) -> "MarketEvent":
        """Factory method for creating a MarketEvent from a bar dict.

        Args:
            bars: Symbol-to-Bar mapping.
            timestamp: Event timestamp (typically the bar's open timestamp).

        Returns:
            A new MarketEvent instance.
        """
        return cls(
            event_id=uuid4(),
            event_type=EventType.MARKET,
            timestamp=timestamp,
            bars=bars,
        )


# ── Signal Event ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SignalEvent(BaseEvent):
    """Emitted by a strategy when it has a directional view on an instrument.

    Attributes:
        signal: The Signal domain object containing all signal metadata.
    """

    signal: Signal

    @classmethod
    def from_signal(cls, signal: Signal) -> "SignalEvent":
        """Factory method for wrapping a Signal in a SignalEvent.

        Args:
            signal: The Signal produced by a strategy.

        Returns:
            A new SignalEvent.
        """
        return cls(
            event_id=uuid4(),
            event_type=EventType.SIGNAL,
            timestamp=signal.timestamp,
            signal=signal,
        )


# ── Order Event ────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class OrderEvent(BaseEvent):
    """Emitted by the portfolio manager to instruct execution.

    Attributes:
        order: The Order domain object detailing what to execute.
    """

    order: Order

    @classmethod
    def from_order(cls, order: Order) -> "OrderEvent":
        """Factory method for wrapping an Order in an OrderEvent.

        Args:
            order: The Order to be executed.

        Returns:
            A new OrderEvent.
        """
        return cls(
            event_id=uuid4(),
            event_type=EventType.ORDER,
            timestamp=order.timestamp,
            order=order,
        )


# ── Fill Event ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FillEvent(BaseEvent):
    """Emitted by the execution engine when an order is (partially) filled.

    Attributes:
        fill: The Fill domain object with execution details.
    """

    fill: Fill

    @classmethod
    def from_fill(cls, fill: Fill) -> "FillEvent":
        """Factory method for wrapping a Fill in a FillEvent.

        Args:
            fill: The confirmed Fill.

        Returns:
            A new FillEvent.
        """
        return cls(
            event_id=uuid4(),
            event_type=EventType.FILL,
            timestamp=fill.timestamp,
            fill=fill,
        )


# ── Risk Event ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RiskEvent(BaseEvent):
    """Emitted by the risk engine to signal constraint violations.

    Attributes:
        constraint_name: Name of the violated risk constraint.
        severity: Severity level (``"WARNING"`` or ``"BREACH"``).
        affected_symbol: Symbol that triggered the constraint (if applicable).
        message: Human-readable description of the violation.
    """

    constraint_name: str
    severity: str
    message: str
    affected_symbol: str | None = None

    @classmethod
    def breach(
        cls,
        constraint_name: str,
        message: str,
        timestamp: pd.Timestamp,
        affected_symbol: str | None = None,
    ) -> "RiskEvent":
        """Convenience factory for a BREACH-level risk event."""
        return cls(
            event_id=uuid4(),
            event_type=EventType.RISK,
            timestamp=timestamp,
            constraint_name=constraint_name,
            severity="BREACH",
            message=message,
            affected_symbol=affected_symbol,
        )


# ── System Event ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SystemEvent(BaseEvent):
    """Lifecycle event (START, STOP, HEARTBEAT, etc.).

    Attributes:
        message: Description of the system state change.
    """

    message: str

    @classmethod
    def start(cls, timestamp: pd.Timestamp) -> "SystemEvent":
        """Factory for engine start event."""
        return cls(
            event_id=uuid4(),
            event_type=EventType.SYSTEM,
            timestamp=timestamp,
            message="ENGINE_START",
        )

    @classmethod
    def stop(cls, timestamp: pd.Timestamp) -> "SystemEvent":
        """Factory for engine stop event."""
        return cls(
            event_id=uuid4(),
            event_type=EventType.SYSTEM,
            timestamp=timestamp,
            message="ENGINE_STOP",
        )


# ── Type Alias ─────────────────────────────────────────────────────────────────

AnyEvent = MarketEvent | SignalEvent | OrderEvent | FillEvent | RiskEvent | SystemEvent
