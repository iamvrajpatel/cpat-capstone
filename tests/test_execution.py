"""
Unit tests for ExecutionEngine v2 and slippage models.

Covers:
    - FixedBpsSlippage.compute()
    - VolumeWeightedSlippage.compute() — scales with order/volume ratio
    - SpreadBasedSlippage.compute() — scales with high-low range
    - build_slippage_model factory
    - ExecutionEngine: BUY fill price > open (slippage added)
    - ExecutionEngine: SELL fill price < open (slippage subtracted)
    - ExecutionEngine: deferred order when symbol not in bars
    - ExecutionEngine: zero slippage model baseline
    - ExecutionEngine: commission computed and attached to fill
    - ExecutionEngine: partial fill ratio
    - ExecutionEngine: clear_pending
    - ExecutionEngine: FillEvent posted to queue
    - Edge cases: zero volume, very large slippage
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from cpat.core.enums import CommissionModel, OrderSide, OrderType, SlippageModel
from cpat.core.models import Bar, CostConfig, Fill, Order
from cpat.backtest.event_queue import EventQueue
from cpat.execution.engine import (
    ExecutionEngine,
    FixedBpsSlippage,
    SpreadBasedSlippage,
    VolumeWeightedSlippage,
    build_slippage_model,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_bar(
    symbol: str = "AAPL",
    price: float = 150.0,
    volume: float = 1_000_000.0,
    spread_pct: float = 0.01,
    ts: str = "2022-01-03",
) -> Bar:
    high = price * (1 + spread_pct / 2)
    low = price * (1 - spread_pct / 2)
    return Bar(
        symbol=symbol, timestamp=pd.Timestamp(ts, tz="UTC"),
        open=price, high=high, low=low,
        close=price, volume=volume, adj_close=price,
    )


def _make_order(
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    qty: float = 100.0,
    ts: str = "2022-01-02",
) -> Order:
    return Order(
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=qty,
        timestamp=pd.Timestamp(ts, tz="UTC"),
        strategy_id="test",
    )


def _make_engine(
    bps: float = 5.0,
    commission_pct: float = 0.001,
    fill_ratio: float = 1.0,
    slippage_model_name: str = "FIXED_BPS",
) -> tuple[ExecutionEngine, EventQueue]:
    eq = EventQueue()
    cost = CostConfig(
        commission_model=CommissionModel.PERCENTAGE,
        commission_value=commission_pct,
        min_commission=0.0,
        slippage_model=SlippageModel.FIXED_BPS,
        slippage_bps=bps,
    )
    engine = ExecutionEngine.from_cost_config(
        cost_config=cost,
        event_queue=eq,
        slippage_model_name=slippage_model_name,
        bps=bps,
        fill_ratio=fill_ratio,
    )
    return engine, eq


# ── Slippage model tests ───────────────────────────────────────────────────────


class TestFixedBpsSlippage:
    def test_basic_computation(self):
        model = FixedBpsSlippage(bps=10.0)
        bar = _make_bar(price=100.0)
        order = _make_order()
        slip = model.compute(order, bar)
        assert slip == pytest.approx(0.10)  # 100 * 10/10000

    def test_zero_bps(self):
        model = FixedBpsSlippage(bps=0.0)
        bar = _make_bar(price=200.0)
        assert model.compute(_make_order(), bar) == pytest.approx(0.0)

    def test_scales_with_price(self):
        model = FixedBpsSlippage(bps=5.0)
        low_bar = _make_bar(price=100.0)
        high_bar = _make_bar(price=200.0)
        order = _make_order()
        assert model.compute(order, high_bar) == pytest.approx(
            model.compute(order, low_bar) * 2, rel=1e-5
        )


class TestVolumeWeightedSlippage:
    def test_larger_order_more_slippage(self):
        model = VolumeWeightedSlippage(base_bps=3.0, impact_factor=0.1)
        bar = _make_bar(price=100.0, volume=100_000.0)
        small_order = _make_order(qty=100)
        large_order = _make_order(qty=10_000)
        slip_small = model.compute(small_order, bar)
        slip_large = model.compute(large_order, bar)
        assert slip_large > slip_small

    def test_zero_volume_uses_base_bps(self):
        model = VolumeWeightedSlippage(base_bps=5.0)
        bar = _make_bar(price=100.0, volume=0.0)
        slip = model.compute(_make_order(), bar)
        assert slip == pytest.approx(100.0 * 5.0 / 10_000)


class TestSpreadBasedSlippage:
    def test_wider_spread_more_slippage(self):
        model = SpreadBasedSlippage(spread_fraction=0.15)
        narrow_bar = _make_bar(price=100.0, spread_pct=0.005)   # 0.5% range
        wide_bar = _make_bar(price=100.0, spread_pct=0.02)     # 2% range
        order = _make_order()
        assert model.compute(order, wide_bar) > model.compute(order, narrow_bar)

    def test_spread_computation(self):
        model = SpreadBasedSlippage(spread_fraction=0.20)
        bar = _make_bar(price=100.0, spread_pct=0.01)  # high=100.5, low=99.5
        # spread estimate = 0.20 * (100.5-99.5) = 0.20
        # slippage = 0.5 * 0.20 = 0.10
        slip = model.compute(_make_order(), bar)
        assert slip == pytest.approx(0.10, rel=0.01)


class TestBuildSlippageModel:
    def test_fixed_bps_from_string(self):
        model = build_slippage_model("FIXED_BPS", bps=5.0)
        assert isinstance(model, FixedBpsSlippage)

    def test_volume_weighted_from_string(self):
        model = build_slippage_model("VOLUME_WEIGHTED")
        assert isinstance(model, VolumeWeightedSlippage)

    def test_spread_based_from_string(self):
        model = build_slippage_model("SPREAD_BASED")
        assert isinstance(model, SpreadBasedSlippage)

    def test_zero_from_string(self):
        model = build_slippage_model("ZERO")
        bar = _make_bar(price=100.0)
        assert model.compute(_make_order(), bar) == pytest.approx(0.0)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            build_slippage_model("MAGIC_MODEL")


# ── ExecutionEngine tests ──────────────────────────────────────────────────────


class TestExecutionEngineBasic:
    def test_buy_fill_price_above_open(self):
        engine, eq = _make_engine(bps=10.0)
        order = _make_order(side=OrderSide.BUY)
        engine.submit(order)
        bars = {"AAPL": _make_bar(price=100.0)}
        result = engine.process_pending(bars, pd.Timestamp("2022-01-03", tz="UTC"))
        assert len(result.fills) == 1
        fill = result.fills[0]
        assert fill.fill_price > 100.0  # Slippage added for buys

    def test_sell_fill_price_below_open(self):
        engine, eq = _make_engine(bps=10.0)
        order = _make_order(side=OrderSide.SELL)
        engine.submit(order)
        bars = {"AAPL": _make_bar(price=100.0)}
        result = engine.process_pending(bars, pd.Timestamp("2022-01-03", tz="UTC"))
        assert result.fills[0].fill_price < 100.0  # Slippage subtracted for sells

    def test_zero_slippage_model(self):
        engine, eq = _make_engine(bps=0.0, slippage_model_name="ZERO")
        order = _make_order(side=OrderSide.BUY)
        engine.submit(order)
        bars = {"AAPL": _make_bar(price=100.0)}
        result = engine.process_pending(bars, pd.Timestamp("2022-01-03", tz="UTC"))
        assert result.fills[0].fill_price == pytest.approx(100.0)

    def test_commission_applied(self):
        engine, eq = _make_engine(bps=0.0, commission_pct=0.001)
        order = _make_order(qty=100.0)
        engine.submit(order)
        bars = {"AAPL": _make_bar(price=150.0)}
        result = engine.process_pending(bars, pd.Timestamp("2022-01-03", tz="UTC"))
        fill = result.fills[0]
        # commission = 100 * 150 * 0.001 = 15.0
        assert fill.commission == pytest.approx(15.0, rel=0.05)

    def test_deferred_order_when_no_bar(self):
        engine, eq = _make_engine()
        engine.submit(_make_order(symbol="TSLA"))
        result = engine.process_pending(
            {"AAPL": _make_bar()},  # No TSLA bar
            pd.Timestamp("2022-01-03", tz="UTC"),
        )
        assert len(result.fills) == 0
        assert len(result.deferred) == 1
        assert engine.pending_order_count == 1

    def test_fill_event_posted_to_queue(self):
        engine, eq = _make_engine()
        engine.submit(_make_order())
        engine.process_pending({"AAPL": _make_bar()}, pd.Timestamp("2022-01-03", tz="UTC"))
        from cpat.core.enums import EventType
        events = list(eq.drain())
        fill_events = [e for e in events if e.event_type == EventType.FILL]
        assert len(fill_events) == 1

    def test_clear_pending(self):
        engine, eq = _make_engine()
        engine.submit(_make_order())
        engine.submit(_make_order())
        assert engine.pending_order_count == 2
        engine.clear_pending()
        assert engine.pending_order_count == 0

    def test_multiple_symbols(self):
        engine, eq = _make_engine()
        engine.submit(_make_order("AAPL", qty=100))
        engine.submit(_make_order("MSFT", qty=50))
        bars = {
            "AAPL": _make_bar("AAPL", 150.0),
            "MSFT": _make_bar("MSFT", 300.0),
        }
        result = engine.process_pending(bars, pd.Timestamp("2022-01-03", tz="UTC"))
        assert len(result.fills) == 2


class TestExecutionEnginePartialFill:
    def test_partial_fill_reduces_quantity(self):
        # fill_ratio=0.5 means ~50% of order is filled
        engine, eq = _make_engine(fill_ratio=0.5)
        engine.submit(_make_order(qty=1000.0))
        bars = {"AAPL": _make_bar(price=100.0)}
        result = engine.process_pending(bars, pd.Timestamp("2022-01-03", tz="UTC"))
        assert len(result.fills) == 1
        fill = result.fills[0]
        # Fill qty should be less than 1000
        assert fill.quantity < 1000.0
        assert fill.quantity >= 1.0

    def test_full_fill_at_ratio_one(self):
        engine, eq = _make_engine(fill_ratio=1.0)
        engine.submit(_make_order(qty=100.0))
        bars = {"AAPL": _make_bar()}
        result = engine.process_pending(bars, pd.Timestamp("2022-01-03", tz="UTC"))
        assert result.fills[0].quantity == pytest.approx(100.0)


class TestExecutionEngineEdgeCases:
    def test_all_fills_accumulated(self):
        engine, eq = _make_engine()
        for i in range(5):
            engine.submit(_make_order())
            engine.process_pending(
                {"AAPL": _make_bar()},
                pd.Timestamp(f"2022-01-0{i+3}", tz="UTC"),
            )
        assert len(engine.all_fills) == 5

    def test_zero_quantity_order_raises(self):
        # Order validation prevents zero quantity at the domain level
        with pytest.raises(ValueError, match="quantity"):
            _make_order(qty=0.0)
