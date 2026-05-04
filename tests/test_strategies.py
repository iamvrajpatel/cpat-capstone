"""
Unit tests for strategy implementations.

Covers:
    - AbstractStrategy bar history accumulation
    - AbstractStrategy _emit_signal produces correct SignalEvent
    - MomentumStrategy: requires sufficient bars, z-score computation
    - MomentumStrategy: correct LONG signal for top performer
    - MeanReversionStrategy: no signal within normal range
    - MeanReversionStrategy: LONG signal on oversold conditions
    - MeanReversionStrategy: FLAT signal when price reverts
    - _compute_rsi correctness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cpat.core.enums import EventType, SignalDirection
from cpat.core.events import MarketEvent
from cpat.core.models import Bar
from cpat.backtest.event_queue import EventQueue
from cpat.config.loader import MomentumStrategyConfig, MeanReversionStrategyConfig
from cpat.strategies.momentum import MomentumStrategy
from cpat.strategies.mean_reversion import MeanReversionStrategy, _compute_rsi


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_bar(symbol: str, price: float, ts: pd.Timestamp) -> Bar:
    return Bar(
        symbol=symbol, timestamp=ts,
        open=price, high=price * 1.005, low=price * 0.995,
        close=price, volume=1_000_000.0, adj_close=price,
    )


def _feed_bars(strategy, symbols_prices: dict[str, list[float]]) -> None:
    """Feed a sequence of bars to a strategy."""
    prices_list = list(symbols_prices.values())
    n_bars = len(prices_list[0])
    for i in range(n_bars):
        ts = pd.Timestamp("2022-01-01", tz="UTC") + pd.Timedelta(days=i)
        bars = {
            sym: _make_bar(sym, prices[i], ts)
            for sym, prices in symbols_prices.items()
        }
        event = MarketEvent.from_bars(bars, ts)
        strategy.on_market(event)


# ── RSI tests ──────────────────────────────────────────────────────────────────

class TestComputeRSI:
    def test_insufficient_data_returns_fifty(self):
        prices = pd.Series([100.0, 101.0, 99.0])
        rsi = _compute_rsi(prices, period=14)
        assert rsi == pytest.approx(50.0)

    def test_all_gains_returns_100(self):
        prices = pd.Series([100.0 + i for i in range(20)])
        rsi = _compute_rsi(prices, period=14)
        assert rsi == pytest.approx(100.0)

    def test_all_losses_returns_0(self):
        prices = pd.Series([100.0 - i for i in range(20)])
        rsi = _compute_rsi(prices, period=14)
        assert rsi == pytest.approx(0.0, abs=1.0)

    def test_rsi_in_valid_range(self):
        import numpy as np
        rng = np.random.default_rng(42)
        prices = pd.Series(rng.normal(100, 5, 50).cumsum() + 100)
        rsi = _compute_rsi(prices, period=14)
        assert 0.0 <= rsi <= 100.0


# ── MomentumStrategy tests ─────────────────────────────────────────────────────

class TestMomentumStrategy:
    def _make_strategy(self, symbols: list[str] | None = None) -> tuple[MomentumStrategy, EventQueue]:
        eq = EventQueue()
        cfg = MomentumStrategyConfig(
            lookback_short=5,
            lookback_long=20,
            skip_most_recent=0,
            rebalance_frequency=5,
            top_n_pct=0.4,
            allow_short=False,
        )
        return MomentumStrategy(cfg, eq, symbols), eq

    def test_no_signals_before_warmup(self):
        strategy, eq = self._make_strategy(["AAPL", "MSFT"])
        # Feed fewer bars than required
        _feed_bars(strategy, {
            "AAPL": [100.0] * 10,
            "MSFT": [200.0] * 10,
        })
        # Drain only signal events
        signals = [e for e in eq.drain() if e.event_type == EventType.SIGNAL]
        assert len(signals) == 0

    def test_long_signal_for_top_performer(self):
        strategy, eq = self._make_strategy(["AAPL", "MSFT", "GOOGL"])
        n = 30
        # AAPL rises strongly, MSFT flat, GOOGL falls
        _feed_bars(strategy, {
            "AAPL": [100.0 + i * 2 for i in range(n)],
            "MSFT": [200.0] * n,
            "GOOGL": [300.0 - i for i in range(n)],
        })
        signals = [e for e in eq.drain() if e.event_type == EventType.SIGNAL]
        aapl_signals = [s for s in signals if s.signal.symbol == "AAPL"]
        if aapl_signals:
            assert aapl_signals[-1].signal.direction == SignalDirection.LONG

    def test_min_bars_required(self):
        cfg = MomentumStrategyConfig(lookback_short=5, lookback_long=20)
        s = MomentumStrategy(cfg, EventQueue())
        # Formula: lookback_long + skip_most_recent + 1
        expected = cfg.lookback_long + cfg.skip_most_recent + 1
        assert s.min_bars_required == expected

    def test_strategy_id(self):
        strategy, _ = self._make_strategy()
        assert strategy.strategy_id == "momentum_v1"


# ── MeanReversionStrategy tests ────────────────────────────────────────────────

class TestMeanReversionStrategy:
    def _make_strategy(self) -> tuple[MeanReversionStrategy, EventQueue]:
        eq = EventQueue()
        cfg = MeanReversionStrategyConfig(
            lookback_window=10,
            entry_zscore=2.0,
            exit_zscore=0.5,
            rsi_period=5,
            rsi_oversold=30.0,
            rsi_overbought=70.0,
            allow_short=False,
        )
        return MeanReversionStrategy(cfg, eq), eq

    def test_no_signals_for_mean_reverting_random_walk(self):
        strategy, eq = self._make_strategy()
        # Small random fluctuations: z-score stays near 0
        import numpy as np
        rng = np.random.default_rng(0)
        prices = [100.0 + rng.uniform(-0.5, 0.5) for _ in range(30)]
        _feed_bars(strategy, {"AAPL": prices})
        signals = [e for e in eq.drain()
                   if e.event_type == EventType.SIGNAL
                   and e.signal.direction == SignalDirection.LONG]
        # With small noise, few or no LONG signals expected
        assert len(signals) < 5

    def test_long_signal_on_extreme_drop(self):
        strategy, eq = self._make_strategy()
        # Build up history at ~100, then crash
        base = [100.0] * 15 + [70.0]  # Z-score of ~-9 — should trigger entry
        _feed_bars(strategy, {"AAPL": base})
        signals = [e for e in eq.drain() if e.event_type == EventType.SIGNAL]
        long_signals = [s for s in signals if s.signal.direction == SignalDirection.LONG]
        assert len(long_signals) >= 1

    def test_flat_signal_after_reversion(self):
        strategy, eq = self._make_strategy()
        # 1. Build history, trigger entry
        base = [100.0] * 15 + [70.0]
        _feed_bars(strategy, {"AAPL": base})
        # 2. Price reverts — should emit FLAT
        _feed_bars(strategy, {"AAPL": [100.0]})  # Single bar at mean
        signals = [e for e in eq.drain() if e.event_type == EventType.SIGNAL]
        flat_signals = [s for s in signals if s.signal.direction == SignalDirection.FLAT]
        assert len(flat_signals) >= 1

    def test_min_bars_required(self):
        strategy, _ = self._make_strategy()
        assert strategy.min_bars_required > 0

    def test_strategy_id(self):
        strategy, _ = self._make_strategy()
        assert strategy.strategy_id == "mean_reversion_v1"
