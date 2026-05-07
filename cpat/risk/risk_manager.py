"""
CPAT — Trade-Level & Portfolio-Level Risk Manager (Week 4)
===========================================================
Two components work in tandem:

StopLossTracker
    Owns *per-position* stop levels (stop, take-profit, trailing stop).
    Called once per bar *before* market events to check whether any
    open position has breached its stop or hit its take-profit.
    When triggered it emits FLAT SignalEvents into the event queue
    so the engine closes the position via the normal execution path.

PortfolioRiskManager
    Portfolio-level guardrails *beyond* those in the existing RiskEngine:
        1. max_open_positions   — cap the number of simultaneous trades.
        2. max_daily_loss_pct   — halt all new opens if today's P&L < −N%.

    These two checks plug into the existing RiskEngine/Translator flow:
    ``PortfolioRiskManager.can_open_new_position()`` is called by the
    BacktestEngine before creating an order.

Design
------
* StopLossTracker is purely additive — it does not modify the RiskEngine.
* All thresholds are config-driven; nothing is hardcoded.
* Trailing stops are updated each bar if enabled.

Usage::

    from cpat.risk.risk_manager import StopLossTracker, PortfolioRiskManager

    tracker = StopLossTracker(event_queue, trailing_stop=True)
    tracker.register(symbol="RELIANCE.NS", stop_price=2400.0, take_profit_price=2700.0)

    # Per-bar call (before market event processing):
    tracker.check_stops(current_prices, timestamp)

    pm = PortfolioRiskManager(max_open_positions=20, max_daily_loss_pct=0.02)
    if pm.can_open(portfolio, equity, daily_start_equity):
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from cpat.backtest.event_queue import EventQueue
from cpat.core.enums import SignalDirection
from cpat.core.events import SignalEvent
from cpat.core.models import Bar, Signal
from cpat.portfolio.manager import PortfolioManager

logger = logging.getLogger(__name__)


# ── Stop Level ─────────────────────────────────────────────────────────────────


@dataclass
class StopLevel:
    """Per-position stop-loss state.

    Attributes:
        symbol          : Ticker.
        entry_price     : Price at which the position was opened.
        stop_price      : Current hard stop-loss price.
        take_profit_price: Optional take-profit price.
        trailing_stop   : If True, stop_price ratchets up with price moves.
        trailing_distance: Distance (in price) to maintain between high-water-mark
                           and the trailing stop.  Initialised as entry − stop.
        high_water_mark : Highest price seen since entry (for trailing).
        strategy_id     : Which strategy owns this position.
        triggered       : True once the stop/TP has fired (prevents double-fire).
    """

    symbol: str
    entry_price: float
    stop_price: float
    take_profit_price: Optional[float]
    trailing_stop: bool
    trailing_distance: float
    strategy_id: str
    high_water_mark: float = field(init=False)
    triggered: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.high_water_mark = self.entry_price

    def update_trailing(self, current_price: float) -> None:
        """Ratchet trailing stop up if price has moved favourably.

        Only moves the stop upward — never downward.

        Args:
            current_price: Latest bar close or high price.
        """
        if not self.trailing_stop or self.triggered:
            return
        if current_price > self.high_water_mark:
            self.high_water_mark = current_price
            new_stop = self.high_water_mark - self.trailing_distance
            if new_stop > self.stop_price:
                old_stop = self.stop_price
                self.stop_price = new_stop
                logger.debug(
                    "[trailing] %s: stop ratcheted %.4f → %.4f (hwm=%.4f)",
                    self.symbol, old_stop, self.stop_price, self.high_water_mark,
                )

    def is_stop_breached(self, current_price: float) -> bool:
        """True if current_price is at or below the stop."""
        return not self.triggered and current_price <= self.stop_price

    def is_tp_hit(self, current_price: float) -> bool:
        """True if current_price is at or above the take-profit."""
        return (
            not self.triggered
            and self.take_profit_price is not None
            and current_price >= self.take_profit_price
        )


# ── Stop-Loss Tracker ──────────────────────────────────────────────────────────


class StopLossTracker:
    """Monitors open positions for stop-loss and take-profit triggers.

    Each bar the engine calls ``check_stops()`` *before* draining the
    market event queue.  Any triggered stop emits a FLAT SignalEvent,
    which is processed normally through the translator → risk engine →
    execution engine pipeline.

    Args:
        event_queue   : Engine event bus.
        trailing_stop : Default trailing mode for new registrations.
                        Can be overridden per-position in ``register()``.
    """

    def __init__(
        self,
        event_queue: EventQueue,
        trailing_stop: bool = False,
    ) -> None:
        self._queue = event_queue
        self._default_trailing = trailing_stop
        self._levels: dict[str, StopLevel] = {}  # symbol → StopLevel

    # ── Public API ─────────────────────────────────────────────────────────────

    def register(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        strategy_id: str,
        take_profit_price: Optional[float] = None,
        trailing_stop: Optional[bool] = None,
    ) -> None:
        """Register a new stop level for an opened position.

        Args:
            symbol          : Ticker.
            entry_price     : Actual fill price.
            stop_price      : Initial stop-loss price.
            strategy_id     : Strategy that owns this position.
            take_profit_price: Optional take-profit price.
            trailing_stop   : Override default trailing mode.
        """
        use_trailing = trailing_stop if trailing_stop is not None else self._default_trailing
        trailing_dist = entry_price - stop_price  # distance in price units

        level = StopLevel(
            symbol=symbol,
            entry_price=entry_price,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            trailing_stop=use_trailing,
            trailing_distance=max(trailing_dist, 0.01),  # floor at 1 tick
            strategy_id=strategy_id,
        )
        self._levels[symbol] = level
        logger.info(
            "[stops] Register %s | entry=%.4f | stop=%.4f | tp=%s | trailing=%s",
            symbol, entry_price, stop_price,
            f"{take_profit_price:.4f}" if take_profit_price else "None",
            use_trailing,
        )

    def deregister(self, symbol: str) -> None:
        """Remove a stop level when a position is closed externally.

        Args:
            symbol: Ticker to remove.
        """
        self._levels.pop(symbol, None)

    def check_stops(
        self,
        current_bars: dict[str, float | Bar],
        timestamp: pd.Timestamp,
    ) -> list[str]:
        """Check all registered stops against current prices.

        For each breached stop / hit take-profit, emits a FLAT SignalEvent
        and marks the level as triggered.

        Args:
            current_prices: Latest close prices (symbol → price).
            timestamp     : Current bar timestamp.

        Returns:
            List of symbols whose stops triggered this bar.
        """
        triggered_symbols: list[str] = []

        for symbol, level in list(self._levels.items()):
            if level.triggered:
                continue

            bar_or_price = current_bars.get(symbol)
            if bar_or_price is None:
                continue

            if isinstance(bar_or_price, Bar):
                high_price = float(bar_or_price.high)
                low_price = float(bar_or_price.low)
                mark_price = float(bar_or_price.close)
            else:
                mark_price = float(bar_or_price)
                high_price = mark_price
                low_price = mark_price

            if mark_price <= 0:
                continue

            # Update trailing stop first
            level.update_trailing(high_price)

            reason: Optional[str] = None
            if not level.triggered and low_price <= level.stop_price:
                reason = (
                    f"stop_loss (low={low_price:.4f} ≤ stop={level.stop_price:.4f})"
                )
            elif (
                not level.triggered
                and level.take_profit_price is not None
                and high_price >= level.take_profit_price
            ):
                reason = (
                    f"take_profit (high={high_price:.4f} ≥ tp={level.take_profit_price:.4f})"
                )

            if reason:
                level.triggered = True
                triggered_symbols.append(symbol)
                self._emit_flat(symbol, level.strategy_id, timestamp, reason)

        # Clean up triggered levels
        for sym in triggered_symbols:
            del self._levels[sym]

        return triggered_symbols

    def update_trailing_stops(self, current_prices: dict[str, float]) -> None:
        """Update trailing stops without triggering exits.

        Call this at bar close to ratchet trailing stops before the
        next bar's open check.

        Args:
            current_prices: Latest prices.
        """
        for symbol, level in self._levels.items():
            price = current_prices.get(symbol)
            if price and price > 0:
                level.update_trailing(price)

    @property
    def active_levels(self) -> dict[str, StopLevel]:
        """Read-only snapshot of all active stop levels."""
        return dict(self._levels)

    @property
    def n_active(self) -> int:
        """Number of active (non-triggered) stop levels."""
        return sum(1 for lv in self._levels.values() if not lv.triggered)

    def get_stop_price(self, symbol: str) -> Optional[float]:
        """Return current stop price for a symbol, or None if not tracked.

        Args:
            symbol: Ticker to query.
        """
        level = self._levels.get(symbol)
        return level.stop_price if level and not level.triggered else None

    # ── Private ────────────────────────────────────────────────────────────────

    def _emit_flat(
        self,
        symbol: str,
        strategy_id: str,
        timestamp: pd.Timestamp,
        reason: str,
    ) -> None:
        """Emit a FLAT SignalEvent to close the position via normal flow."""
        signal = Signal(
            symbol=symbol,
            direction=SignalDirection.FLAT,
            timestamp=timestamp,
            strategy_id=strategy_id,
            metadata={"reason": reason, "source": "stop_loss_tracker"},
        )
        event = SignalEvent.from_signal(signal)
        self._queue.put(event)
        logger.info("[stops] TRIGGERED %s → FLAT | %s", symbol, reason)


# ── Portfolio Risk Manager ─────────────────────────────────────────────────────


class PortfolioRiskManager:
    """Portfolio-level risk guardrails beyond the existing RiskEngine.

    Adds two additional constraints:
        1. ``max_open_positions`` — cap the number of concurrent trades.
        2. ``max_daily_loss_pct`` — halt new opens if today's realised + unrealised
           loss exceeds the daily limit.

    These are checked BEFORE the signal reaches the translator.

    Args:
        max_open_positions: Maximum concurrent open positions (default 20).
        max_daily_loss_pct: Daily loss limit as fraction of start-of-day equity.
                            E.g. 0.02 = 2% drawdown within one day halts trading.
    """

    def __init__(
        self,
        max_open_positions: int = 20,
        max_daily_loss_pct: float = 0.02,
        timezone: str = "UTC",
    ) -> None:
        self._max_positions   = max_open_positions
        self._max_daily_loss  = max_daily_loss_pct
        self._timezone = timezone
        self._daily_start_equity: Optional[float] = None
        self._current_bar_date: Optional[str] = None
        self._daily_halt: bool = False

    # ── Daily reset ────────────────────────────────────────────────────────────

    def update_daily_equity(
        self, timestamp: pd.Timestamp, current_equity: float
    ) -> None:
        """Record start-of-day equity for daily loss tracking.

        Call once per bar. Automatically resets on date change.

        Args:
            timestamp      : Current bar timestamp.
            current_equity : Current total portfolio equity.
        """
        local_ts = timestamp.tz_convert(self._timezone) if timestamp.tzinfo else timestamp.tz_localize(self._timezone)
        bar_date = str(local_ts.date())
        if bar_date != self._current_bar_date:
            self._current_bar_date = bar_date
            self._daily_start_equity = current_equity
            self._daily_halt = False
            logger.debug("[risk_mgr] New day %s | start_equity=%.2f", bar_date, current_equity)

    # ── Guard checks ───────────────────────────────────────────────────────────

    def can_open_new_position(
        self,
        portfolio: PortfolioManager,
        current_equity: float,
    ) -> tuple[bool, str]:
        """Check whether a new long position can be opened.

        Args:
            portfolio      : Current portfolio state.
            current_equity : Latest mark-to-market equity.

        Returns:
            (allowed, reason) — if not allowed, reason explains why.
        """
        # 1. Max open positions
        n_open = len(portfolio.open_positions)
        if n_open >= self._max_positions:
            return False, (
                f"max_open_positions={self._max_positions} reached "
                f"(currently {n_open} open)."
            )

        # 2. Daily loss halt
        if self._daily_halt:
            return False, "Daily loss limit active — no new opens."

        if self._daily_start_equity and self._daily_start_equity > 0:
            daily_return = (current_equity - self._daily_start_equity) / self._daily_start_equity
            if daily_return < -self._max_daily_loss:
                self._daily_halt = True
                logger.warning(
                    "[risk_mgr] DAILY HALT: return=%.2f%% < -%.2f%%",
                    daily_return * 100, self._max_daily_loss * 100,
                )
                return False, (
                    f"Daily loss {daily_return:.2%} exceeds limit "
                    f"{-self._max_daily_loss:.2%}."
                )

        return True, "ok"

    @property
    def daily_halt_active(self) -> bool:
        """True if the daily loss halt is currently active."""
        return self._daily_halt

    @property
    def max_open_positions(self) -> int:
        """Configured maximum simultaneous positions."""
        return self._max_positions
