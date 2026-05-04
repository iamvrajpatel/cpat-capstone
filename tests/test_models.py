"""
Unit tests for core domain models.

Covers:
    - Bar construction and invariant validation
    - Signal strength validation
    - Order type validation
    - Fill value computations
    - Position FIFO P&L accounting
    - CostConfig commission and slippage calculations
"""

from __future__ import annotations

import pytest
import pandas as pd

from cpat.core.enums import (
    CommissionModel,
    OrderSide,
    OrderStatus,
    OrderType,
    SignalDirection,
    SlippageModel,
)
from cpat.core.models import Bar, CostConfig, Fill, Order, Position, Signal


# ── Bar tests ──────────────────────────────────────────────────────────────────

class TestBar:
    def test_valid_bar_construction(self) -> None:
        bar = Bar(
            symbol="AAPL",
            timestamp=pd.Timestamp("2022-01-01", tz="UTC"),
            open=100.0, high=105.0, low=98.0, close=102.0,
            volume=1_000_000.0, adj_close=102.0,
        )
        assert bar.symbol == "AAPL"
        assert bar.close == 102.0

    def test_bar_high_lt_low_raises(self) -> None:
        with pytest.raises(ValueError, match="high.*low"):
            Bar(
                symbol="AAPL",
                timestamp=pd.Timestamp("2022-01-01", tz="UTC"),
                open=100.0, high=90.0, low=95.0, close=92.0,
                volume=1_000_000.0, adj_close=92.0,
            )

    def test_bar_negative_price_raises(self) -> None:
        with pytest.raises(ValueError, match="Non-positive"):
            Bar(
                symbol="AAPL",
                timestamp=pd.Timestamp("2022-01-01", tz="UTC"),
                open=-1.0, high=5.0, low=1.0, close=3.0,
                volume=1_000_000.0, adj_close=3.0,
            )

    def test_bar_negative_volume_raises(self) -> None:
        with pytest.raises(ValueError, match="Negative volume"):
            Bar(
                symbol="AAPL",
                timestamp=pd.Timestamp("2022-01-01", tz="UTC"),
                open=100.0, high=105.0, low=98.0, close=102.0,
                volume=-1.0, adj_close=102.0,
            )

    def test_bar_mid_property(self) -> None:
        bar = Bar(
            symbol="X", timestamp=pd.Timestamp("2022-01-01", tz="UTC"),
            open=100.0, high=120.0, low=80.0, close=110.0,
            volume=1.0, adj_close=110.0,
        )
        assert bar.mid == pytest.approx(100.0)

    def test_bar_is_bullish(self) -> None:
        bar = Bar(
            symbol="X", timestamp=pd.Timestamp("2022-01-01", tz="UTC"),
            open=100.0, high=110.0, low=99.0, close=108.0,
            volume=1.0, adj_close=108.0,
        )
        assert bar.is_bullish is True

    def test_bar_is_immutable(self) -> None:
        bar = Bar(
            symbol="X", timestamp=pd.Timestamp("2022-01-01", tz="UTC"),
            open=100.0, high=110.0, low=99.0, close=108.0,
            volume=1.0, adj_close=108.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            bar.close = 999.0  # type: ignore[misc]


# ── Signal tests ───────────────────────────────────────────────────────────────

class TestSignal:
    def test_valid_signal(self) -> None:
        sig = Signal(
            strategy_id="test",
            symbol="AAPL",
            direction=SignalDirection.LONG,
            timestamp=pd.Timestamp("2022-01-01", tz="UTC"),
            strength=0.75,
        )
        assert sig.direction == SignalDirection.LONG
        assert sig.strength == pytest.approx(0.75)

    def test_signal_strength_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="strength"):
            Signal(
                strategy_id="test",
                symbol="AAPL",
                direction=SignalDirection.LONG,
                timestamp=pd.Timestamp("2022-01-01", tz="UTC"),
                strength=1.5,
            )

    def test_signal_negative_strength_valid(self) -> None:
        sig = Signal(
            strategy_id="test", symbol="AAPL",
            direction=SignalDirection.SHORT,
            timestamp=pd.Timestamp("2022-01-01", tz="UTC"),
            strength=-0.5,
        )
        assert sig.strength == pytest.approx(-0.5)


# ── Order tests ────────────────────────────────────────────────────────────────

class TestOrder:
    def test_valid_market_order(self) -> None:
        order = Order(
            symbol="MSFT", side=OrderSide.BUY, order_type=OrderType.MARKET,
            quantity=50.0, timestamp=pd.Timestamp("2022-01-01", tz="UTC"),
            strategy_id="test",
        )
        assert order.quantity == 50.0
        assert order.status == OrderStatus.PENDING

    def test_zero_quantity_raises(self) -> None:
        with pytest.raises(ValueError, match="quantity"):
            Order(
                symbol="MSFT", side=OrderSide.BUY, order_type=OrderType.MARKET,
                quantity=0.0, timestamp=pd.Timestamp("2022-01-01", tz="UTC"),
                strategy_id="test",
            )

    def test_limit_order_without_price_raises(self) -> None:
        with pytest.raises(ValueError, match="LIMIT"):
            Order(
                symbol="MSFT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
                quantity=10.0, timestamp=pd.Timestamp("2022-01-01", tz="UTC"),
                strategy_id="test",
                limit_price=None,  # Missing!
            )

    def test_limit_order_with_price_valid(self) -> None:
        order = Order(
            symbol="MSFT", side=OrderSide.BUY, order_type=OrderType.LIMIT,
            quantity=10.0, timestamp=pd.Timestamp("2022-01-01", tz="UTC"),
            strategy_id="test",
            limit_price=300.0,
        )
        assert order.limit_price == 300.0


# ── Fill tests ─────────────────────────────────────────────────────────────────

class TestFill:
    def test_gross_value(self) -> None:
        from uuid import uuid4
        fill = Fill(
            order_id=uuid4(), symbol="AAPL", side=OrderSide.BUY,
            quantity=100.0, fill_price=150.0, commission=1.5,
            slippage=0.5, timestamp=pd.Timestamp("2022-01-01", tz="UTC"),
        )
        assert fill.gross_value == pytest.approx(15_000.0)

    def test_net_value_includes_commission(self) -> None:
        from uuid import uuid4
        fill = Fill(
            order_id=uuid4(), symbol="AAPL", side=OrderSide.BUY,
            quantity=100.0, fill_price=150.0, commission=1.5,
            slippage=0.5, timestamp=pd.Timestamp("2022-01-01", tz="UTC"),
        )
        assert fill.net_value == pytest.approx(15_001.5)


# ── Position tests ─────────────────────────────────────────────────────────────

class TestPosition:
    def _make_fill(self, side: OrderSide, qty: float, price: float) -> Fill:
        from uuid import uuid4
        return Fill(
            order_id=uuid4(), symbol="AAPL", side=side,
            quantity=qty, fill_price=price, commission=0.0,
            slippage=0.0, timestamp=pd.Timestamp("2022-01-01", tz="UTC"),
        )

    def test_open_long_position(self) -> None:
        pos = Position(symbol="AAPL")
        fill = self._make_fill(OrderSide.BUY, 100.0, 150.0)
        pos.apply_fill(fill)
        assert pos.quantity == 100.0
        assert pos.avg_cost == pytest.approx(150.0)

    def test_add_to_position_updates_vwac(self) -> None:
        pos = Position(symbol="AAPL")
        pos.apply_fill(self._make_fill(OrderSide.BUY, 100.0, 150.0))
        pos.apply_fill(self._make_fill(OrderSide.BUY, 100.0, 160.0))
        assert pos.quantity == 200.0
        assert pos.avg_cost == pytest.approx(155.0)

    def test_close_position_realises_pnl(self) -> None:
        pos = Position(symbol="AAPL")
        pos.apply_fill(self._make_fill(OrderSide.BUY, 100.0, 150.0))
        pos.apply_fill(self._make_fill(OrderSide.SELL, 100.0, 160.0))
        assert pos.is_flat
        assert pos.realised_pnl == pytest.approx(1_000.0)  # (160-150)*100

    def test_unrealised_pnl(self) -> None:
        pos = Position(symbol="AAPL")
        pos.apply_fill(self._make_fill(OrderSide.BUY, 100.0, 150.0))
        assert pos.unrealised_pnl(155.0) == pytest.approx(500.0)

    def test_flat_position_unrealised_pnl_is_zero(self) -> None:
        pos = Position(symbol="AAPL")
        assert pos.unrealised_pnl(150.0) == 0.0


# ── CostConfig tests ───────────────────────────────────────────────────────────

class TestCostConfig:
    def test_zero_commission(self) -> None:
        cfg = CostConfig(commission_model=CommissionModel.ZERO)
        assert cfg.compute_commission(100.0, 150.0) == 0.0

    def test_percentage_commission(self) -> None:
        cfg = CostConfig(
            commission_model=CommissionModel.PERCENTAGE,
            commission_value=0.001,  # 10 bps
            min_commission=0.0,
        )
        # 100 shares * $150 * 0.001 = $15.00
        assert cfg.compute_commission(100.0, 150.0) == pytest.approx(15.0)

    def test_min_commission_applied(self) -> None:
        cfg = CostConfig(
            commission_model=CommissionModel.PERCENTAGE,
            commission_value=0.0001,  # 1 bps — tiny trade
            min_commission=5.0,
        )
        # Would be $0.15 but min is $5.00
        assert cfg.compute_commission(10.0, 150.0) == pytest.approx(5.0)

    def test_fixed_bps_slippage(self) -> None:
        cfg = CostConfig(
            slippage_model=SlippageModel.FIXED_BPS,
            slippage_bps=10.0,  # 10 bps
        )
        # 10 bps on $100 = $0.10
        assert cfg.compute_slippage(100.0) == pytest.approx(0.10)

    def test_zero_slippage(self) -> None:
        cfg = CostConfig(slippage_model=SlippageModel.ZERO)
        assert cfg.compute_slippage(150.0) == 0.0
