"""
CPAT — Event Handler Protocols
================================
Defines the Protocol (structural typing) interfaces that all event handlers
must satisfy.  Using Protocol instead of ABC allows duck-typed handlers
and enables easier mocking in tests.

Handler registration is managed by the BacktestEngine; these interfaces
are the contract between the engine and each component.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cpat.core.events import FillEvent, MarketEvent, OrderEvent, RiskEvent, SignalEvent


@runtime_checkable
class MarketEventHandler(Protocol):
    """Component that reacts to new market data (typically a Strategy)."""

    def on_market(self, event: MarketEvent) -> None:
        """Process a new bar of market data.

        Args:
            event: MarketEvent containing the latest bars for all symbols.
        """
        ...


@runtime_checkable
class SignalEventHandler(Protocol):
    """Component that reacts to trading signals (typically the Portfolio Manager)."""

    def on_signal(self, event: SignalEvent) -> None:
        """Process a trading signal from a strategy.

        Args:
            event: SignalEvent wrapping the Signal domain object.
        """
        ...


@runtime_checkable
class OrderEventHandler(Protocol):
    """Component that reacts to order requests (typically the Execution Engine)."""

    def on_order(self, event: OrderEvent) -> None:
        """Process an order instruction.

        Args:
            event: OrderEvent wrapping the Order domain object.
        """
        ...


@runtime_checkable
class FillEventHandler(Protocol):
    """Component that reacts to fill confirmations (Strategy + Portfolio Manager)."""

    def on_fill(self, event: FillEvent) -> None:
        """Process a fill confirmation.

        Args:
            event: FillEvent wrapping the Fill domain object.
        """
        ...


@runtime_checkable
class RiskEventHandler(Protocol):
    """Component that reacts to risk constraint violations."""

    def on_risk(self, event: RiskEvent) -> None:
        """Handle a risk constraint breach or warning.

        Args:
            event: RiskEvent with constraint details.
        """
        ...
