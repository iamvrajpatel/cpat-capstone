"""
CPAT — Mean Reversion Strategy
================================
Implements a single-instrument mean reversion strategy based on:
    - Bollinger Band z-score for entry/exit signals
    - RSI confirmation filter to avoid trend-following entries

    Hypothesis:
        Prices exhibit mean-reversion on short time horizons (days to weeks).
        When a stock is >2 standard deviations below its rolling mean AND
        RSI < 30, it is likely to revert — generating a LONG signal.

    Mathematical formulation:
        μ(t) = rolling_mean(close, window=N)  [strictly past bars]
        σ(t) = rolling_std(close, window=N)   [strictly past bars]
        z(t) = (close(t) - μ(t)) / σ(t)

        RSI(t) = 100 - 100/(1 + RS(t))
        RS(t) = avg_gain(N_rsi) / avg_loss(N_rsi)

        Entry: z(t) < -entry_threshold AND RSI(t) < rsi_oversold  → LONG
        Exit:  z(t) > -exit_threshold                              → FLAT

    Bias prevention:
        - All rolling windows use data strictly up to bar T.
        - No future close prices are used.
        - adj_close used to avoid distortions from dividends/splits.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from cpat.config.loader import MeanReversionStrategyConfig
from cpat.core.enums import SignalDirection
from cpat.core.events import FillEvent, MarketEvent
from cpat.backtest.event_queue import EventQueue
from cpat.strategies.base import AbstractStrategy

logger = logging.getLogger(__name__)


def _compute_rsi(prices: pd.Series, period: int) -> float:
    """Compute the Wilder RSI for a price series.

    Uses Wilder's smoothing method (exponential moving average with
    alpha = 1/period) which is the canonical RSI implementation.

    Args:
        prices: Price series (newest value last).
        period: RSI lookback period.

    Returns:
        RSI value in [0, 100], or 50.0 if insufficient data.
    """
    if len(prices) < period + 1:
        return 50.0

    deltas = prices.diff().dropna()
    gains = deltas.clip(lower=0)
    losses = (-deltas).clip(lower=0)

    # Initial averages
    avg_gain = gains.iloc[:period].mean()
    avg_loss = losses.iloc[:period].mean()

    # Wilder's smoothing for remaining periods
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period

    if avg_loss < 1e-10:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


class MeanReversionStrategy(AbstractStrategy):
    """Bollinger Band + RSI mean reversion strategy.

    Operates on each symbol independently (time-series, not cross-sectional).
    Generates a signal per bar when conditions are met.

    Args:
        config: Mean reversion strategy parameters.
        event_queue: Event bus.
        symbols: Symbols to trade.
    """

    def __init__(
        self,
        config: MeanReversionStrategyConfig,
        event_queue: EventQueue,
        symbols: list[str] | None = None,
    ) -> None:
        super().__init__(
            strategy_id="mean_reversion_v1",
            event_queue=event_queue,
            symbols=symbols,
        )
        self._cfg = config
        # Track current position state per symbol to avoid repeated signals
        self._in_position: dict[str, bool] = {}

    @property
    def min_bars_required(self) -> int:
        """Require enough bars for rolling stats + RSI."""
        return max(self._cfg.lookback_window, self._cfg.rsi_period) + 5

    def on_market(self, event: MarketEvent) -> None:
        """Update history and evaluate mean reversion signals for each symbol."""
        self._update_history(event)

        for symbol in list(self._bar_history.keys()):
            if not self._has_sufficient_history(symbol):
                continue
            self._evaluate_symbol(symbol, event.timestamp)

    def _evaluate_symbol(self, symbol: str, timestamp: pd.Timestamp) -> None:
        """Compute z-score and RSI; emit signal if entry/exit conditions met.

        Args:
            symbol: Symbol to evaluate.
            timestamp: Current bar timestamp.
        """
        prices = self._get_price_series(symbol, "adj_close")
        n = self._cfg.lookback_window

        if len(prices) < n:
            return

        # Rolling statistics — strictly past data (all computations on history up to T)
        window = prices.iloc[-n:]
        mu = float(window.mean())
        sigma = float(window.std(ddof=1))

        if sigma < 1e-10:
            return

        current_price = float(prices.iloc[-1])
        z_score = (current_price - mu) / sigma

        # RSI computation
        rsi_prices = prices.iloc[-(self._cfg.rsi_period + 10):]
        rsi = _compute_rsi(rsi_prices, self._cfg.rsi_period)

        in_pos = self._in_position.get(symbol, False)

        # Entry condition: oversold + low RSI
        if (not in_pos
                and z_score < -self._cfg.entry_zscore
                and rsi < self._cfg.rsi_oversold):
            # Strength: how far below entry threshold (clamped to [-1, 0])
            strength = max(-1.0, z_score / (self._cfg.entry_zscore * 2))
            self._emit_signal(
                symbol=symbol,
                direction=SignalDirection.LONG,
                strength=abs(strength),  # Positive = long conviction
                timestamp=timestamp,
                metadata={"z_score": z_score, "rsi": rsi, "mu": mu, "sigma": sigma},
            )
            self._in_position[symbol] = True

        # Exit condition: price reverts toward mean
        elif in_pos and z_score > -self._cfg.exit_zscore:
            self._emit_signal(
                symbol=symbol,
                direction=SignalDirection.FLAT,
                strength=0.0,
                timestamp=timestamp,
                metadata={"z_score": z_score, "rsi": rsi},
            )
            self._in_position[symbol] = False

        # Short-side (only if configured)
        elif (self._cfg.allow_short
              and not in_pos
              and z_score > self._cfg.entry_zscore
              and rsi > self._cfg.rsi_overbought):
            strength = min(1.0, z_score / (self._cfg.entry_zscore * 2))
            self._emit_signal(
                symbol=symbol,
                direction=SignalDirection.SHORT,
                strength=strength,
                timestamp=timestamp,
                metadata={"z_score": z_score, "rsi": rsi, "mu": mu, "sigma": sigma},
            )
            self._in_position[symbol] = True

    def on_fill(self, event: FillEvent) -> None:
        """Log fill; no additional state update needed here."""
        super().on_fill(event)
