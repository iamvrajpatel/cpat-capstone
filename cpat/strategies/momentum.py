"""
CPAT — Cross-Sectional Momentum Strategy
==========================================
Implements the Jegadeesh & Titman (1993) cross-sectional momentum effect:

    Hypothesis:
        Stocks that performed best over the past J months (excluding the
        most recent K months) continue to outperform over the next M months.

    Implementation:
        - Lookback: 12-month formation period (252 trading days)
        - Skip: 1-month (21 days) to avoid the short-term reversal effect
        - Signal: z-score of (12-1)-month return within the cross-section
        - Rebalance: monthly (every 21 trading days)
        - Long: top 20% by momentum score
        - Flat: bottom 80% (no shorting in this config)

    Mathematical formulation:
        r_i(t) = log(P_i(t - skip) / P_i(t - lookback_long))
        z_i(t) = (r_i(t) - μ_r(t)) / σ_r(t)
        signal_i(t) = LONG if z_i(t) > z_threshold else FLAT

    Bias prevention:
        - Formation period returns use only data up to bar T.
        - adj_close prices used to account for dividends and splits.
        - z-scores computed cross-sectionally at each rebalance bar.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from cpat.config.loader import MomentumStrategyConfig
from cpat.core.enums import SignalDirection
from cpat.core.events import FillEvent, MarketEvent
from cpat.backtest.event_queue import EventQueue
from cpat.strategies.base import AbstractStrategy

logger = logging.getLogger(__name__)


class MomentumStrategy(AbstractStrategy):
    """Cross-sectional momentum strategy (Jegadeesh & Titman, 1993).

    Signals are generated at rebalance dates only. Between rebalances,
    the strategy holds its last signal (i.e., no intra-month churn).

    Args:
        config: Momentum strategy parameters.
        event_queue: Event bus.
        symbols: Symbols to trade (subset of universe).
    """

    def __init__(
        self,
        config: MomentumStrategyConfig,
        event_queue: EventQueue,
        symbols: list[str] | None = None,
    ) -> None:
        super().__init__(
            strategy_id="momentum_v1",
            event_queue=event_queue,
            symbols=symbols,
        )
        self._cfg = config
        self._bars_since_rebalance: int = 0
        self._last_signals: dict[str, SignalDirection] = {}

    @property
    def min_bars_required(self) -> int:
        """Require enough bars to compute the full formation period."""
        return self._cfg.lookback_long + self._cfg.skip_most_recent + 1

    def on_market(self, event: MarketEvent) -> None:
        """Update history and emit signals at rebalance dates."""
        self._update_history(event)
        self._bars_since_rebalance += 1

        if self._bars_since_rebalance < self._cfg.rebalance_frequency:
            return

        # Check all tracked symbols have sufficient history
        eligible = [
            sym for sym in self._bar_history
            if self._has_sufficient_history(sym)
        ]
        if len(eligible) < 5:
            return

        self._bars_since_rebalance = 0
        self._generate_signals(event.timestamp, eligible)

    def _generate_signals(
        self,
        timestamp: pd.Timestamp,
        eligible_symbols: list[str],
    ) -> None:
        """Compute cross-sectional momentum scores and emit signals.

        Args:
            timestamp: Current bar timestamp.
            eligible_symbols: Symbols with sufficient history.
        """
        scores: dict[str, float] = {}

        for symbol in eligible_symbols:
            prices = self._get_price_series(symbol, "adj_close")
            if len(prices) < self.min_bars_required:
                continue

            # Formation return: from (lookback_long + skip) days ago to skip days ago
            # This is strictly historical — no future data
            end_idx = -(self._cfg.skip_most_recent) if self._cfg.skip_most_recent > 0 else len(prices)
            start_idx = -(self._cfg.lookback_long + self._cfg.skip_most_recent)

            price_end = prices.iloc[end_idx - 1] if self._cfg.skip_most_recent > 0 else prices.iloc[-1]
            price_start = prices.iloc[start_idx]

            if price_start <= 0:
                continue

            # Log return (more statistically robust than simple return)
            log_ret = np.log(price_end / price_start)
            scores[symbol] = log_ret

        if len(scores) < 3:
            logger.debug("[momentum] Insufficient scores (%d) for cross-sectional ranking.", len(scores))
            return

        # Cross-sectional z-score normalisation
        score_series = pd.Series(scores)
        mean_r = score_series.mean()
        std_r = score_series.std()

        if std_r < 1e-10:
            logger.debug("[momentum] Cross-sectional standard deviation near zero; skipping.")
            return

        z_scores = (score_series - mean_r) / std_r

        # Rank-based thresholds
        n = len(z_scores)
        top_n = max(1, int(n * self._cfg.top_n_pct))
        bottom_n = max(1, int(n * self._cfg.bottom_n_pct)) if self._cfg.allow_short else 0

        sorted_symbols = z_scores.sort_values(ascending=False)
        long_symbols = set(sorted_symbols.head(top_n).index)
        short_symbols = set(sorted_symbols.tail(bottom_n).index) if bottom_n > 0 else set()

        # Emit signals for all tracked symbols
        for symbol in eligible_symbols:
            if symbol not in z_scores.index:
                continue

            z = float(z_scores[symbol])

            if symbol in long_symbols:
                direction = SignalDirection.LONG
            elif symbol in short_symbols:
                direction = SignalDirection.SHORT
            else:
                direction = SignalDirection.FLAT

            # Normalise strength to [-1, 1] using z-score clamped to ±3
            strength = max(-1.0, min(1.0, z / 3.0))

            self._emit_signal(
                symbol=symbol,
                direction=direction,
                strength=strength,
                timestamp=timestamp,
                metadata={"z_score": z, "log_return": float(scores.get(symbol, 0.0))},
            )
            self._last_signals[symbol] = direction

        logger.info(
            "[momentum] Rebalance at %s: LONG=%d, SHORT=%d, FLAT=%d",
            timestamp.date(),
            len(long_symbols),
            len(short_symbols),
            n - len(long_symbols) - len(short_symbols),
        )

    def on_fill(self, event: FillEvent) -> None:
        """Log fills for monitoring."""
        super().on_fill(event)
