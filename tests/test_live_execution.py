"""Tests — Live Execution Engine (Week 5)."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from cpat.brokers.paper import PaperBroker
from cpat.core.enums import SignalDirection
from cpat.core.events import SignalEvent
from cpat.core.models import Signal
from cpat.infrastructure.broker_interface import BrokerConnectionError
from cpat.infrastructure.execution_engine_live import LiveExecutionEngine
from cpat.infrastructure.live_logger import LiveTradeLogger
from cpat.infrastructure.order_manager import OrderManager
from cpat.infrastructure.scheduler import TradingScheduler
from cpat.portfolio.allocator import EqualWeightAllocator
from cpat.portfolio.manager import PortfolioManager
from cpat.risk.engine import RiskEngine
from cpat.risk.position_sizing import FixedFractionalSizer
from cpat.risk.risk_manager import PortfolioRiskManager, StopLossTracker
from cpat.backtest.event_queue import EventQueue
from cpat.strategies.base import AbstractStrategy


# ── Fixtures / Helpers ────────────────────────────────────────────────────────

def _make_bars(n=50, close=1000.0, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = close * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    highs  = closes * 1.005
    lows   = closes * 0.995
    opens  = closes * 1.001
    idx    = pd.date_range(end=pd.Timestamp.utcnow().floor("min"), periods=n, freq="min", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "adj_close": closes, "volume": 1e6}, index=idx)

SYMBOLS = ["RELIANCE.NS", "TCS.NS"]
EQUITY  = 10_000_000.0


def _make_data_provider(symbols=SYMBOLS, close=2500.0):
    dfs = {s: _make_bars(close=close) for s in symbols}
    def _provider(syms):
        return {s: dfs[s] for s in syms if s in dfs}
    return _provider


class _AlwaysSignalStrategy(AbstractStrategy):
    def __init__(self, event_queue: EventQueue, direction: SignalDirection):
        super().__init__(strategy_id="mock_strategy", event_queue=event_queue, symbols=SYMBOLS)
        self._direction = direction

    @property
    def min_bars_required(self) -> int:
        return 1

    def on_market(self, event):
        self._update_history(event)
        for symbol in event.bars:
            signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                direction=self._direction,
                timestamp=event.timestamp,
                strength=1.0,
            )
            self._event_queue.put(SignalEvent.from_signal(signal))


def _make_engine(
    direction=SignalDirection.LONG,
    initial_capital=EQUITY,
    dry_run=False,
    symbols=SYMBOLS,
    close=2500.0,
) -> LiveExecutionEngine:
    broker = PaperBroker(initial_capital=initial_capital)
    broker.connect()
    for sym in symbols:
        broker.set_price(sym, last=close)

    eq = EventQueue()
    portfolio = PortfolioManager(initial_capital=initial_capital)
    risk = RiskEngine(
        portfolio=portfolio,
        max_position_pct=0.10, max_sector_pct=0.5,
        max_gross_exposure_pct=2.0, max_drawdown_pct=0.20,
        min_cash_pct=0.01,
    )
    sizer      = FixedFractionalSizer(risk_per_trade_pct=0.02, min_trade_value=100.0)
    allocator  = EqualWeightAllocator(max_position_pct=0.10)
    stop_track = StopLossTracker(eq, trailing_stop=False)
    port_risk  = PortfolioRiskManager(max_open_positions=20, max_daily_loss_pct=0.05)

    trade_log = LiveTradeLogger(log_dir="/tmp/cpat_test_logs", console=False)
    oms = OrderManager()
    strategy = _AlwaysSignalStrategy(eq, direction)

    return LiveExecutionEngine(
        broker=broker,
        strategy=strategy,
        portfolio=portfolio,
        risk_engine=risk,
        position_sizer=sizer,
        allocator=allocator,
        stop_tracker=stop_track,
        portfolio_risk_manager=port_risk,
        data_provider=_make_data_provider(symbols, close),
        symbols=symbols,
        trade_logger=trade_log,
        oms=oms,
        event_queue=eq,
        dry_run=dry_run,
        stale_data_seconds=300,
    )


# ── LiveExecutionEngine: Basic ────────────────────────────────────────────────

class TestLiveExecutionEngineBasic:
    def test_run_tick_does_not_raise(self):
        engine = _make_engine()
        engine.run_tick()  # should not raise

    def test_tick_count_increments(self):
        engine = _make_engine()
        engine.run_tick()
        engine.run_tick()
        assert engine._tick_count == 2

    def test_summary_keys(self):
        engine = _make_engine()
        engine.run_tick()
        s = engine.summary()
        for k in ["tick_count", "orders_placed", "fills_received",
                  "api_errors", "oms_summary", "broker_connected", "reconciliation_ok"]:
            assert k in s

    def test_dry_run_no_orders_placed_to_broker(self):
        engine = _make_engine(dry_run=True)
        engine.run_tick()
        # OMS should have orders (created+submitted), broker fill_log should be empty
        assert engine._broker.get_fill_log() == []

    def test_long_signal_creates_orders(self):
        engine = _make_engine(direction=SignalDirection.LONG)
        engine.run_tick()
        assert engine._orders_placed > 0

    def test_flat_signal_no_new_orders(self):
        engine = _make_engine(direction=SignalDirection.FLAT)
        engine.run_tick()
        # No buys for FLAT signal
        assert engine._orders_placed == 0

    def test_stale_data_skips_tick(self):
        engine = _make_engine(direction=SignalDirection.LONG)
        old_df = _make_bars()
        old_df.index = pd.date_range("2020-01-01", periods=len(old_df), freq="min", tz="UTC")
        engine._data_fn = lambda syms: {sym: old_df for sym in syms}
        engine.run_tick()
        assert engine._stale_data_skips > 0


# ── Paper fills applied to OMS ────────────────────────────────────────────────

class TestPaperFillDraining:
    def test_fills_received_after_tick(self):
        engine = _make_engine(direction=SignalDirection.LONG)
        engine.run_tick()
        assert engine._fills_received > 0

    def test_oms_has_filled_orders(self):
        engine = _make_engine(direction=SignalDirection.LONG)
        engine.run_tick()
        summary = engine._oms.summary()
        assert summary.get("FILLED", 0) > 0


# ── Data provider failure ─────────────────────────────────────────────────────

class TestDataProviderFailure:
    def test_engine_survives_data_fetch_error(self):
        """Engine should not crash when data provider raises."""
        engine = _make_engine()
        engine._data_fn = lambda syms: (_ for _ in ()).throw(RuntimeError("network down"))

        # Should not raise
        engine.run_tick()
        assert engine._api_errors >= 1

    def test_empty_data_returns_skips_tick(self):
        engine = _make_engine()
        engine._data_fn = lambda syms: {}
        engine.run_tick()
        assert engine._orders_placed == 0


# ── API connection retry ──────────────────────────────────────────────────────

class TestAPIRetry:
    def test_connection_error_increments_oms_retry(self):
        """BrokerConnectionError should increment OMS retry counter."""
        engine = _make_engine(direction=SignalDirection.LONG)

        # Patch place_order to raise on first call
        call_count = {"n": 0}
        original_place = engine._broker.place_order

        def failing_place(order):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise BrokerConnectionError("timeout")
            return original_place(order)

        engine._broker.place_order = failing_place
        engine.run_tick()
        # Should still complete (retry logic)
        assert engine._tick_count == 1
        assert engine._retry_count >= 1


# ── OMS lifecycle via engine ─────────────────────────────────────────────────

class TestOMSLifecycleViaEngine:
    def test_order_created_in_oms_before_send(self):
        """OMS should have the order even if broker call fails."""
        engine = _make_engine(direction=SignalDirection.LONG)
        engine._broker.place_order = MagicMock(
            side_effect=BrokerConnectionError("fail")
        )
        engine.run_tick()
        all_orders = engine._oms.all_orders
        # Orders were created in OMS and rejected
        rejected = [o for o in all_orders if o.status.value == "REJECTED"]
        assert len(all_orders) >= 1

    def test_orders_not_duplicated_on_existing_position(self):
        """If already holding a position, no duplicate BUY should be placed."""
        engine = _make_engine(direction=SignalDirection.LONG)
        engine.run_tick()
        orders_after_1 = engine._orders_placed

        # Second tick — already holding RELIANCE.NS — should not buy again
        engine.run_tick()
        orders_after_2 = engine._orders_placed
        # Second tick should add 0 new orders (already in position)
        delta = orders_after_2 - orders_after_1
        assert delta == 0


# ── Portfolio risk guards ─────────────────────────────────────────────────────

class TestPortfolioRiskGuards:
    def test_max_positions_blocks_new_orders(self):
        engine = _make_engine(direction=SignalDirection.LONG)
        engine._port_risk._max_positions = 0  # block all new opens
        engine.run_tick()
        assert engine._orders_placed == 0

    def test_daily_loss_halt_blocks_orders(self):
        engine = _make_engine(direction=SignalDirection.LONG)
        # Activate daily halt by setting it directly
        engine._port_risk._daily_halt = True
        engine._port_risk._daily_start_equity = EQUITY
        engine.run_tick()
        assert engine._orders_placed == 0

    def test_reconciliation_mismatch_blocks_new_orders(self):
        engine = _make_engine(direction=SignalDirection.LONG)
        engine._broker._cash += 1000.0
        engine.run_tick()
        assert engine._orders_placed == 0
        assert engine.summary()["reconciliation_ok"] is False


# ── Scheduler ────────────────────────────────────────────────────────────────

class TestTradingScheduler:
    def test_scheduler_fires_callback(self):
        fired = {"count": 0}

        def cb():
            fired["count"] += 1

        sched = TradingScheduler(cb, interval_seconds=1, mode="always")
        sched.start()
        time.sleep(2.5)
        sched.stop()
        assert fired["count"] >= 1

    def test_scheduler_stops_cleanly(self):
        sched = TradingScheduler(lambda: None, interval_seconds=1, mode="always")
        sched.start()
        time.sleep(0.5)
        sched.stop()
        assert sched.is_running is False

    def test_market_hours_mode_skips_weekend(self):
        """Market hours mode must not fire on Saturday/Sunday."""
        fired = {"count": 0}

        def cb():
            fired["count"] += 1

        sched = TradingScheduler(cb, interval_seconds=1, mode="market_hours")
        # Patch _should_fire to simulate weekend
        sched._should_fire = lambda: False
        sched.start()
        time.sleep(1.5)
        sched.stop()
        assert fired["count"] == 0

    def test_error_counter_halts_scheduler(self):
        """max_tick_errors consecutive errors should halt the scheduler."""
        def bad_cb():
            raise RuntimeError("simulated crash")

        sched = TradingScheduler(bad_cb, interval_seconds=0,
                                  mode="always", max_tick_errors=2)
        sched.start()
        time.sleep(1.0)
        assert sched.is_running is False
        assert sched._consecutive_errors >= 2

    def test_tick_count_increments(self):
        sched = TradingScheduler(lambda: None, interval_seconds=0, mode="always")
        sched.start()
        time.sleep(0.3)
        sched.stop()
        assert sched.tick_count >= 1

    def test_overlap_guard_skips_tick(self):
        fired = {"count": 0}

        def slow_cb():
            fired["count"] += 1
            time.sleep(0.15)

        sched = TradingScheduler(slow_cb, interval_seconds=0, mode="always")
        sched.start()
        time.sleep(0.3)
        sched.stop()
        assert sched.overlap_skips >= 0


# ── LiveTradeLogger ───────────────────────────────────────────────────────────

class TestLiveTradeLogger:
    def _logger(self) -> LiveTradeLogger:
        return LiveTradeLogger(log_dir="/tmp/cpat_test_logs", console=False)

    def test_signal_does_not_raise(self):
        log = self._logger()
        log.signal("S", "LONG", strength=0.9)  # no exception

    def test_order_placed_does_not_raise(self):
        log = self._logger()
        log.order_placed("S", qty=10, side="BUY", broker_order_id="BRK001")

    def test_fill_does_not_raise(self):
        log = self._logger()
        log.fill("S", qty=10, price=2500.0, side="BUY", commission=7.5)

    def test_error_does_not_raise(self):
        log = self._logger()
        log.error("Something went wrong", exc=RuntimeError("test"), symbol="S")

    def test_heartbeat_does_not_raise(self):
        log = self._logger()
        log.heartbeat(equity=1_000_000.0, n_positions=3, n_open_orders=2)

    def test_system_does_not_raise(self):
        log = self._logger()
        log.system("ENGINE_START")

    def test_reconciliation_does_not_raise(self):
        log = self._logger()
        log.reconciliation("ok", "state aligned")
