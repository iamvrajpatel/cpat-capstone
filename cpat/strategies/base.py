"""
CPAT — Abstract Strategy Base
================================
Defines the AbstractStrategy interface that every trading strategy must
implement.  The Strategy Pattern is used here: the BacktestEngine holds
a reference to AbstractStrategy instances and calls ``on_market``/``on_fill``
without knowing the concrete implementation.

Design contract:
    - Strategies MUST NOT access future data.
    - Strategies MUST NOT directly interact with the execution engine.
    - Strategies produce ``SignalEvent`` objects which are queued for
      processing by the portfolio manager.
    - All rolling computations must use only data available at bar T.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from cpat.core.enums import SignalDirection
from cpat.core.events import FillEvent, MarketEvent, SignalEvent
from cpat.core.models import Signal
from cpat.backtest.event_queue import EventQueue

logger = logging.getLogger(__name__)


class AbstractStrategy(ABC):
    """Base class for all CPAT trading strategies.

    Subclasses must implement ``on_market`` and optionally ``on_fill``.

    Args:
        strategy_id: Unique name for this strategy instance.
        event_queue: Event bus to post SignalEvents to.
        symbols: List of symbols this strategy trades (empty = all universe).
    """

    def __init__(
        self,
        strategy_id: str,
        event_queue: EventQueue,
        symbols: Optional[list[str]] = None,
    ) -> None:
        self.strategy_id = strategy_id
        self._event_queue = event_queue
        self._symbols = symbols or []

        # Bar history buffer: symbol → list of past bars (newest last)
        self._bar_history: dict[str, list[dict]] = {}

        # Internal bar counter for warmup tracking
        self._bar_count: int = 0

        logger.info("Strategy '%s' initialised.", strategy_id)

    # ── Interface ──────────────────────────────────────────────────────────────

    @abstractmethod
    def on_market(self, event: MarketEvent) -> None:
        """Process a new bar of market data.

        Called once per bar for every MarketEvent.  Implementations should:
            1. Update internal bar history.
            2. Compute indicators using ONLY past bars (no look-ahead).
            3. Emit SignalEvents via ``self._emit_signal()``.

        Args:
            event: MarketEvent containing the latest bars.
        """

    def on_fill(self, event: FillEvent) -> None:
        """React to a fill confirmation (optional override).

        Default implementation logs the fill.  Override to update internal
        position tracking or risk state.

        Args:
            event: FillEvent with fill details.
        """
        logger.debug(
            "[%s] Fill received: %s %s %.0f @ %.4f",
            self.strategy_id,
            event.fill.side.value,
            event.fill.symbol,
            event.fill.quantity,
            event.fill.fill_price,
        )

    @property
    @abstractmethod
    def min_bars_required(self) -> int:
        """Minimum number of bars required before generating signals.

        Returns:
            Integer number of bars (lookback period).
        """

    # ── Protected helpers ──────────────────────────────────────────────────────

    def _update_history(self, event: MarketEvent) -> None:
        """Append current bars to the per-symbol history buffer.

        Args:
            event: Current MarketEvent.
        """
        for symbol, bar in event.bars.items():
            if self._symbols and symbol not in self._symbols:
                continue
            if symbol not in self._bar_history:
                self._bar_history[symbol] = []
            self._bar_history[symbol].append({
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "adj_close": bar.adj_close,
                "volume": bar.volume,
            })
        self._bar_count += 1

    def _get_price_series(self, symbol: str, column: str = "adj_close") -> pd.Series:
        """Return a price series for a symbol from the bar history buffer.

        Args:
            symbol: Ticker symbol.
            column: OHLCV column name (default: ``adj_close``).

        Returns:
            Pandas Series with DatetimeIndex, newest value last.
        """
        history = self._bar_history.get(symbol, [])
        if not history:
            return pd.Series(dtype=float)
        df = pd.DataFrame(history).set_index("timestamp")
        return df[column].astype(float)

    def _emit_signal(
        self,
        symbol: str,
        direction: SignalDirection,
        strength: float = 0.0,
        timestamp: Optional[pd.Timestamp] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Create and queue a SignalEvent.

        Args:
            symbol: Target instrument.
            direction: LONG, SHORT, or FLAT.
            strength: Normalised signal strength in [-1, 1].
            timestamp: Signal timestamp (defaults to current time).
            metadata: Optional indicator values for logging/analysis.
        """
        ts = timestamp or pd.Timestamp.now(tz="UTC")
        signal = Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            direction=direction,
            timestamp=ts,
            strength=max(-1.0, min(1.0, strength)),
            metadata=metadata or {},
        )
        signal_event = SignalEvent.from_signal(signal)
        self._event_queue.put(signal_event)
        logger.debug(
            "[%s] Signal: %s %s (strength=%.3f)",
            self.strategy_id, direction.name, symbol, strength,
        )

    def _has_sufficient_history(self, symbol: str) -> bool:
        """Return True if the symbol has enough bars for signal generation.

        Args:
            symbol: Ticker symbol.

        Returns:
            True if bar count >= min_bars_required.
        """
        return len(self._bar_history.get(symbol, [])) >= self.min_bars_required
