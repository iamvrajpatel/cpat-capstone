"""Tests — Risk Manager: StopLossTracker + PortfolioRiskManager (Week 4)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock
from cpat.risk.risk_manager import StopLevel, StopLossTracker, PortfolioRiskManager
from cpat.backtest.event_queue import EventQueue
from cpat.core.events import SignalEvent
from cpat.core.enums import SignalDirection
from cpat.core.models import Bar


def _queue() -> EventQueue:
    return EventQueue()

def _tracker(trailing=False) -> StopLossTracker:
    return StopLossTracker(_queue(), trailing_stop=trailing)

def _ts(day="2021-01-04") -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")

def _portfolio(n=0) -> MagicMock:
    m = MagicMock()
    m.open_positions = {f"SYM{i}": MagicMock() for i in range(n)}
    return m

def _level(entry=1000.0, stop=950.0, tp=None, trailing=False) -> StopLevel:
    return StopLevel(symbol="S", entry_price=entry, stop_price=stop,
                     take_profit_price=tp, trailing_stop=trailing,
                     trailing_distance=entry-stop, strategy_id="m")


def _bar(price=1000.0, high=None, low=None) -> Bar:
    return Bar(
        symbol="S",
        timestamp=_ts(),
        open=price,
        high=high if high is not None else price,
        low=low if low is not None else price,
        close=price,
        volume=1_000_000.0,
        adj_close=price,
    )


# ── StopLevel ──────────────────────────────────────────────────────────────────
class TestStopLevel:
    def test_stop_breached_below(self):
        assert _level(stop=950.0).is_stop_breached(940.0) is True

    def test_stop_breached_at_stop(self):
        assert _level(stop=950.0).is_stop_breached(950.0) is True

    def test_stop_not_breached_above(self):
        assert _level(stop=950.0).is_stop_breached(960.0) is False

    def test_tp_hit(self):
        assert _level(tp=1100.0).is_tp_hit(1100.0) is True

    def test_tp_not_hit_below(self):
        assert _level(tp=1100.0).is_tp_hit(1080.0) is False

    def test_no_tp_never_triggers(self):
        assert _level(tp=None).is_tp_hit(9999.0) is False

    def test_trailing_ratchets_up(self):
        sl = _level(trailing=True)
        sl.update_trailing(1100.0)
        assert sl.stop_price > 950.0
        assert sl.high_water_mark == 1100.0

    def test_trailing_never_drops(self):
        sl = _level(trailing=True)
        sl.update_trailing(1100.0)
        ratcheted = sl.stop_price
        sl.update_trailing(900.0)
        assert sl.stop_price == ratcheted

    def test_triggered_prevents_breach(self):
        sl = _level(stop=950.0)
        sl.triggered = True
        assert sl.is_stop_breached(900.0) is False

    def test_non_trailing_unchanged(self):
        sl = _level(stop=950.0, trailing=False)
        original = sl.stop_price
        sl.update_trailing(1100.0)
        assert sl.stop_price == original


# ── StopLossTracker ────────────────────────────────────────────────────────────
class TestStopLossTracker:
    def test_register_creates_level(self):
        t = _tracker()
        t.register("S", 1000.0, 950.0, "m")
        assert "S" in t.active_levels

    def test_n_active(self):
        t = _tracker()
        t.register("A", 1000.0, 950.0, "m")
        t.register("B", 500.0, 470.0, "m")
        assert t.n_active == 2

    def test_deregister(self):
        t = _tracker()
        t.register("S", 1000.0, 950.0, "m")
        t.deregister("S")
        assert "S" not in t.active_levels

    def test_check_stops_triggers_on_breach(self):
        t = _tracker()
        t.register("S", 1000.0, 950.0, "m")
        triggered = t.check_stops({"S": 940.0}, _ts())
        assert "S" in triggered

    def test_triggered_emits_flat_signal(self):
        q = _queue()
        t = StopLossTracker(q)
        t.register("S", 1000.0, 950.0, "m")
        t.check_stops({"S": 940.0}, _ts())
        events = []
        while not q.empty():
            events.append(q.get())
        flat = [e for e in events if isinstance(e, SignalEvent)
                and e.signal.direction == SignalDirection.FLAT]
        assert len(flat) == 1 and flat[0].signal.symbol == "S"

    def test_no_trigger_above_stop(self):
        t = _tracker()
        t.register("S", 1000.0, 950.0, "m")
        assert t.check_stops({"S": 980.0}, _ts()) == []

    def test_tp_trigger(self):
        t = _tracker()
        t.register("S", 1000.0, 950.0, "m", take_profit_price=1100.0)
        assert "S" in t.check_stops({"S": 1100.0}, _ts())

    def test_level_removed_after_trigger(self):
        t = _tracker()
        t.register("S", 1000.0, 950.0, "m")
        t.check_stops({"S": 900.0}, _ts())
        assert "S" not in t.active_levels

    def test_bar_low_triggers_stop(self):
        t = _tracker()
        t.register("S", 1000.0, 950.0, "m")
        triggered = t.check_stops({"S": _bar(price=980.0, low=940.0, high=990.0)}, _ts())
        assert triggered == ["S"]

    def test_bar_high_triggers_take_profit(self):
        t = _tracker()
        t.register("S", 1000.0, 950.0, "m", take_profit_price=1100.0)
        triggered = t.check_stops({"S": _bar(price=1080.0, high=1110.0, low=1070.0)}, _ts())
        assert triggered == ["S"]

    def test_no_double_trigger(self):
        q = _queue()
        t = StopLossTracker(q)
        t.register("S", 1000.0, 950.0, "m")
        t.check_stops({"S": 900.0}, _ts("2021-01-04"))
        t.check_stops({"S": 800.0}, _ts("2021-01-05"))
        flat = []
        while not q.empty():
            e = q.get()
            if isinstance(e, SignalEvent) and e.signal.direction == SignalDirection.FLAT:
                flat.append(e)
        assert len(flat) == 1

    def test_missing_price_no_trigger(self):
        t = _tracker()
        t.register("S", 1000.0, 950.0, "m")
        assert t.check_stops({}, _ts()) == []

    def test_get_stop_price(self):
        t = _tracker()
        t.register("S", 1000.0, 950.0, "m")
        assert t.get_stop_price("S") == 950.0

    def test_get_stop_price_none_unknown(self):
        assert _tracker().get_stop_price("UNKNOWN") is None

    def test_trailing_default_from_tracker(self):
        t = StopLossTracker(_queue(), trailing_stop=True)
        t.register("S", 1000.0, 950.0, "m")
        assert t.active_levels["S"].trailing_stop is True

    def test_trailing_override_per_registration(self):
        t = StopLossTracker(_queue(), trailing_stop=True)
        t.register("S", 1000.0, 950.0, "m", trailing_stop=False)
        assert t.active_levels["S"].trailing_stop is False


# ── PortfolioRiskManager ───────────────────────────────────────────────────────
class TestPortfolioRiskManager:
    def test_allows_below_max(self):
        pm = PortfolioRiskManager(max_open_positions=5)
        ok, _ = pm.can_open_new_position(_portfolio(n=4), 1_000_000.0)
        assert ok is True

    def test_blocks_at_max(self):
        pm = PortfolioRiskManager(max_open_positions=5)
        ok, _ = pm.can_open_new_position(_portfolio(n=5), 1_000_000.0)
        assert ok is False

    def test_daily_loss_halt(self):
        pm = PortfolioRiskManager(max_open_positions=20, max_daily_loss_pct=0.02)
        pm.update_daily_equity(pd.Timestamp("2021-01-04", tz="UTC"), 1_000_000.0)
        ok, reason = pm.can_open_new_position(_portfolio(), 970_000.0)
        assert ok is False and "Daily loss" in reason

    def test_within_daily_limit(self):
        pm = PortfolioRiskManager(max_open_positions=20, max_daily_loss_pct=0.02)
        pm.update_daily_equity(pd.Timestamp("2021-01-04", tz="UTC"), 1_000_000.0)
        ok, _ = pm.can_open_new_position(_portfolio(), 990_000.0)
        assert ok is True

    def test_halt_resets_next_day(self):
        pm = PortfolioRiskManager(max_open_positions=20, max_daily_loss_pct=0.02)
        pm.update_daily_equity(pd.Timestamp("2021-01-04", tz="UTC"), 1_000_000.0)
        pm.can_open_new_position(_portfolio(), 970_000.0)
        assert pm.daily_halt_active is True
        pm.update_daily_equity(pd.Timestamp("2021-01-05", tz="UTC"), 970_000.0)
        assert pm.daily_halt_active is False

    def test_no_daily_equity_allows(self):
        pm = PortfolioRiskManager(max_open_positions=20, max_daily_loss_pct=0.02)
        ok, _ = pm.can_open_new_position(_portfolio(), 1_000_000.0)
        assert ok is True

    def test_halt_active_property_default(self):
        assert PortfolioRiskManager().daily_halt_active is False

    def test_reason_contains_constraint(self):
        pm = PortfolioRiskManager(max_open_positions=3)
        _, reason = pm.can_open_new_position(_portfolio(n=3), 1_000_000.0)
        assert "max_open_positions" in reason

    def test_max_open_positions_property(self):
        assert PortfolioRiskManager(max_open_positions=15).max_open_positions == 15
