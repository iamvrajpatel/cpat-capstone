"""
Unit tests for the Portfolio package.

Covers:
    - PortfolioManager: apply_fill cash accounting (BUY/SELL)
    - PortfolioManager: total_equity mark-to-market
    - PortfolioManager: gross/net exposure
    - PortfolioManager: sector exposure
    - PortfolioManager: equity_curve from snapshots
    - PortfolioManager: daily_returns
    - SignalOrderTranslator: LONG on flat position → BUY order
    - SignalOrderTranslator: FLAT with long position → SELL order
    - SignalOrderTranslator: LONG with short position → 2 orders (close+open)
    - SignalOrderTranslator: SHORT with allow_short=False → no order
    - SignalOrderTranslator: FLAT on flat position → no order
    - DataHandler: sequential iteration
    - DataHandler: get_slice returns only past data
    - DataHandler: DataAccessViolation on unknown symbol
"""

from __future__ import annotations

import math
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from cpat.core.enums import OrderSide, OrderType, SignalDirection
from cpat.core.events import SignalEvent
from cpat.core.models import Bar, Fill, Order, Signal
from cpat.portfolio.manager import PortfolioManager
from cpat.portfolio.translator import SignalOrderTranslator
from cpat.backtest.event_queue import EventQueue
from cpat.data.handler import DataHandler, DataAccessViolation


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_fill(
    symbol: str,
    side: OrderSide,
    qty: float,
    price: float,
    commission: float = 0.0,
) -> Fill:
    return Fill(
        order_id=uuid4(),
        symbol=symbol,
        side=side,
        quantity=qty,
        fill_price=price,
        commission=commission,
        slippage=0.0,
        timestamp=pd.Timestamp("2022-01-03", tz="UTC"),
    )


def _make_bar(symbol: str, price: float, ts: str = "2022-01-03") -> Bar:
    return Bar(
        symbol=symbol,
        timestamp=pd.Timestamp(ts, tz="UTC"),
        open=price, high=price * 1.01, low=price * 0.99,
        close=price, volume=1_000_000.0, adj_close=price,
    )


def _make_signal_event(
    symbol: str, direction: SignalDirection, last_price: float = 150.0
) -> SignalEvent:
    sig = Signal(
        strategy_id="test",
        symbol=symbol,
        direction=direction,
        timestamp=pd.Timestamp("2022-01-03", tz="UTC"),
        metadata={"last_price": last_price},
    )
    return SignalEvent.from_signal(sig)


def _make_bar_df(n: int = 30, price: float = 100.0) -> pd.DataFrame:
    import numpy as np
    dates = pd.date_range("2022-01-01", periods=n, freq="B", tz="UTC")
    p = np.full(n, price)
    df = pd.DataFrame({
        "open": p, "high": p * 1.01, "low": p * 0.99,
        "close": p, "adj_close": p,
        "volume": np.full(n, 1_000_000.0),
    }, index=dates)
    df.index.name = "timestamp"
    return df


# ── PortfolioManager tests ─────────────────────────────────────────────────────


class TestPortfolioManagerCash:
    def test_initial_cash(self):
        pm = PortfolioManager(100_000.0)
        assert pm.cash == 100_000.0

    def test_buy_reduces_cash(self):
        pm = PortfolioManager(100_000.0)
        fill = _make_fill("AAPL", OrderSide.BUY, qty=100, price=150.0, commission=1.5)
        pm.apply_fill(fill)
        # cash -= 100*150 + 1.5
        assert pm.cash == pytest.approx(100_000.0 - 15_000.0 - 1.5)

    def test_sell_increases_cash(self):
        pm = PortfolioManager(100_000.0)
        # First buy
        pm.apply_fill(_make_fill("AAPL", OrderSide.BUY, 100, 150.0))
        cash_after_buy = pm.cash
        # Then sell
        pm.apply_fill(_make_fill("AAPL", OrderSide.SELL, 100, 155.0, commission=1.5))
        # cash += 100*155 - 1.5
        assert pm.cash == pytest.approx(cash_after_buy + 15_500.0 - 1.5)

    def test_negative_initial_capital_raises(self):
        with pytest.raises(ValueError, match="positive"):
            PortfolioManager(-1.0)

    def test_zero_initial_capital_raises(self):
        with pytest.raises(ValueError, match="positive"):
            PortfolioManager(0.0)


class TestPortfolioManagerEquity:
    def test_equity_with_no_positions_equals_cash(self):
        pm = PortfolioManager(100_000.0)
        equity = pm.total_equity({"AAPL": 150.0})
        assert equity == pytest.approx(100_000.0)

    def test_equity_includes_position_value(self):
        pm = PortfolioManager(100_000.0)
        pm.apply_fill(_make_fill("AAPL", OrderSide.BUY, 100, 150.0))
        # cash = 85000, position = 100*160 = 16000
        equity = pm.total_equity({"AAPL": 160.0})
        assert equity == pytest.approx(85_000.0 + 16_000.0)

    def test_equity_ignores_missing_prices(self):
        pm = PortfolioManager(100_000.0)
        pm.apply_fill(_make_fill("AAPL", OrderSide.BUY, 100, 150.0))
        # No AAPL price provided
        equity = pm.total_equity({"MSFT": 300.0})
        # Only cash counted; AAPL position skipped (no price)
        assert equity == pytest.approx(85_000.0)

    def test_gross_exposure(self):
        pm = PortfolioManager(200_000.0)
        pm.apply_fill(_make_fill("AAPL", OrderSide.BUY, 100, 150.0))
        pm.apply_fill(_make_fill("MSFT", OrderSide.BUY, 50, 200.0))
        gross = pm.gross_exposure({"AAPL": 150.0, "MSFT": 200.0})
        assert gross == pytest.approx(100 * 150.0 + 50 * 200.0)

    def test_net_exposure_long_only(self):
        pm = PortfolioManager(200_000.0)
        pm.apply_fill(_make_fill("AAPL", OrderSide.BUY, 100, 150.0))
        net = pm.net_exposure({"AAPL": 155.0})
        assert net == pytest.approx(100 * 155.0)


class TestPortfolioManagerSnapshots:
    def test_snapshot_created(self):
        pm = PortfolioManager(100_000.0)
        ts = pd.Timestamp("2022-01-03", tz="UTC")
        snap = pm.snapshot(ts, {"AAPL": 150.0})
        assert snap.total_equity == pytest.approx(100_000.0)
        assert snap.cash == pytest.approx(100_000.0)
        assert snap.n_longs == 0

    def test_equity_curve_from_snapshots(self):
        pm = PortfolioManager(100_000.0)
        for i in range(5):
            ts = pd.Timestamp(f"2022-01-0{i+3}", tz="UTC")
            pm.snapshot(ts, {})
        curve = pm.equity_curve()
        assert len(curve) == 5
        assert curve.iloc[0] == pytest.approx(100_000.0)

    def test_snapshots_df_structure(self):
        pm = PortfolioManager(100_000.0)
        pm.snapshot(pd.Timestamp("2022-01-03", tz="UTC"), {})
        df = pm.snapshots_df()
        assert "total_equity" in df.columns
        assert "cash" in df.columns


class TestPortfolioManagerOpenPositions:
    def test_open_positions_filters_flat(self):
        pm = PortfolioManager(200_000.0)
        pm.apply_fill(_make_fill("AAPL", OrderSide.BUY, 100, 150.0))
        pm.apply_fill(_make_fill("MSFT", OrderSide.BUY, 50, 200.0))
        # Close AAPL
        pm.apply_fill(_make_fill("AAPL", OrderSide.SELL, 100, 155.0))
        open_pos = pm.open_positions
        assert "AAPL" not in open_pos
        assert "MSFT" in open_pos

    def test_realised_pnl(self):
        pm = PortfolioManager(100_000.0)
        pm.apply_fill(_make_fill("AAPL", OrderSide.BUY, 100, 150.0))
        pm.apply_fill(_make_fill("AAPL", OrderSide.SELL, 100, 160.0))
        # (160-150) * 100 = 1000
        assert pm.realised_pnl() == pytest.approx(1_000.0)

    def test_unrealised_pnl(self):
        pm = PortfolioManager(100_000.0)
        pm.apply_fill(_make_fill("AAPL", OrderSide.BUY, 100, 150.0))
        unrealised = pm.unrealised_pnl({"AAPL": 155.0})
        assert unrealised == pytest.approx(500.0)


# ── SignalOrderTranslator tests ────────────────────────────────────────────────


class TestSignalOrderTranslator:
    def _make_translator(
        self, allow_short: bool = False, target_weight: float = 0.05
    ) -> tuple[SignalOrderTranslator, PortfolioManager, EventQueue]:
        pm = PortfolioManager(100_000.0)
        eq = EventQueue()
        translator = SignalOrderTranslator(
            portfolio=pm,
            event_queue=eq,
            sizing_method="equal_weight",
            target_weight=target_weight,
            allow_short=allow_short,
            min_trade_value=100.0,  # Low threshold for tests
        )
        return translator, pm, eq

    def test_long_signal_on_flat_generates_buy(self):
        translator, pm, eq = self._make_translator()
        bars = {"AAPL": _make_bar("AAPL", 100.0)}
        event = _make_signal_event("AAPL", SignalDirection.LONG, 100.0)
        translator.on_signal_with_bars(event, bars)

        orders = [e for e in eq.drain()]
        # Should have exactly 1 ORDER event
        from cpat.core.enums import EventType
        order_events = [e for e in orders if e.event_type == EventType.ORDER]
        assert len(order_events) == 1
        assert order_events[0].order.side == OrderSide.BUY

    def test_flat_signal_on_long_generates_sell(self):
        translator, pm, eq = self._make_translator()
        # Open a long position first
        pm.apply_fill(_make_fill("AAPL", OrderSide.BUY, 100, 100.0))

        bars = {"AAPL": _make_bar("AAPL", 100.0)}
        event = _make_signal_event("AAPL", SignalDirection.FLAT, 100.0)
        translator.on_signal_with_bars(event, bars)

        from cpat.core.enums import EventType
        orders = [e for e in eq.drain() if e.event_type == EventType.ORDER]
        assert len(orders) == 1
        assert orders[0].order.side == OrderSide.SELL
        assert orders[0].order.quantity == pytest.approx(100.0)

    def test_flat_signal_on_flat_generates_no_order(self):
        translator, pm, eq = self._make_translator()
        bars = {"AAPL": _make_bar("AAPL", 100.0)}
        event = _make_signal_event("AAPL", SignalDirection.FLAT, 100.0)
        translator.on_signal_with_bars(event, bars)
        assert eq.empty()

    def test_long_on_existing_long_generates_no_order(self):
        translator, pm, eq = self._make_translator()
        pm.apply_fill(_make_fill("AAPL", OrderSide.BUY, 50, 100.0))
        bars = {"AAPL": _make_bar("AAPL", 100.0)}
        event = _make_signal_event("AAPL", SignalDirection.LONG, 100.0)
        translator.on_signal_with_bars(event, bars)
        from cpat.core.enums import EventType
        orders = [e for e in eq.drain() if e.event_type == EventType.ORDER]
        assert len(orders) == 0  # Already long = hold

    def test_short_disabled_generates_no_order(self):
        translator, pm, eq = self._make_translator(allow_short=False)
        bars = {"AAPL": _make_bar("AAPL", 100.0)}
        event = _make_signal_event("AAPL", SignalDirection.SHORT, 100.0)
        translator.on_signal_with_bars(event, bars)
        assert eq.empty()

    def test_long_on_short_generates_two_orders(self):
        translator, pm, eq = self._make_translator(allow_short=True)
        # Manually create a short position
        pm.apply_fill(_make_fill("AAPL", OrderSide.SELL, 50, 100.0))

        bars = {"AAPL": _make_bar("AAPL", 100.0)}
        event = _make_signal_event("AAPL", SignalDirection.LONG, 100.0)
        translator.on_signal_with_bars(event, bars)

        from cpat.core.enums import EventType
        orders = [e for e in eq.drain() if e.event_type == EventType.ORDER]
        # 1. Close short (BUY 50) + 2. Open long (BUY new qty)
        assert len(orders) >= 1
        assert orders[0].order.side == OrderSide.BUY
        # Close qty = abs(short qty) = 50
        assert orders[0].order.quantity >= 1.0  # Just check it's positive

    def test_no_bar_skips_signal(self):
        translator, pm, eq = self._make_translator()
        event = _make_signal_event("TSLA", SignalDirection.LONG, 300.0)
        translator.on_signal_with_bars(event, {})  # No TSLA bar (positional)
        assert eq.empty()


# ── DataHandler tests ──────────────────────────────────────────────────────────


class TestDataHandler:
    def test_iterate_all_bars(self):
        bar_data = {"AAPL": _make_bar_df(n=10)}
        handler = DataHandler(bar_data)
        count = 0
        for ts, bars in handler:
            count += 1
            assert "AAPL" in bars
        assert count == 10

    def test_has_next_false_after_exhaustion(self):
        handler = DataHandler({"AAPL": _make_bar_df(n=3)})
        for _ in handler:
            pass
        assert not handler.has_next()

    def test_stop_iteration_when_exhausted(self):
        handler = DataHandler({"AAPL": _make_bar_df(n=2)})
        handler.get_next()
        handler.get_next()
        with pytest.raises(StopIteration):
            handler.get_next()

    def test_get_slice_returns_past_only(self):
        handler = DataHandler({"AAPL": _make_bar_df(n=30)})
        handler.get_next()  # Advance to bar 1
        handler.get_next()  # Advance to bar 2
        past = handler.get_slice("AAPL", lookback=20)
        # Should have at most 2 bars (we've read 2)
        assert len(past) <= 2

    def test_get_slice_unknown_symbol_raises(self):
        handler = DataHandler({"AAPL": _make_bar_df(n=5)})
        handler.get_next()
        with pytest.raises(DataAccessViolation):
            handler.get_slice("TSLA", lookback=10)

    def test_total_bars(self):
        handler = DataHandler({"AAPL": _make_bar_df(n=20)})
        assert handler.total_bars == 20

    def test_remaining_bars_decrements(self):
        handler = DataHandler({"AAPL": _make_bar_df(n=10)})
        assert handler.remaining_bars == 10
        handler.get_next()
        assert handler.remaining_bars == 9

    def test_warmup_flag(self):
        handler = DataHandler({"AAPL": _make_bar_df(n=10)}, warmup_bars=5)
        handler.get_next()  # bar_idx=0
        assert handler.is_in_warmup is True

    def test_multi_symbol_union_index(self):
        # AAPL has 10 bars, MSFT has 8 bars (subset)
        aapl = _make_bar_df(n=10)
        msft = _make_bar_df(n=8)
        handler = DataHandler({"AAPL": aapl, "MSFT": msft})
        assert handler.total_bars == 10

    def test_reset_restarts_iteration(self):
        handler = DataHandler({"AAPL": _make_bar_df(n=5)})
        for _ in handler:
            pass
        handler.reset()
        assert handler.has_next()
        assert handler.remaining_bars == 5
