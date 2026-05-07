"""Tests — Position Sizing (Week 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cpat.risk.position_sizing import (
    PositionSize,
    FixedFractionalSizer,
    InverseVolatilitySizer,
    KellySizer,
    PositionSizerFactory,
    _compute_atr,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_bars(n: int = 100, base_price: float = 1000.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = base_price * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    highs  = closes * (1 + rng.uniform(0.001, 0.01, n))
    lows   = closes * (1 - rng.uniform(0.001, 0.01, n))
    opens  = closes * (1 + rng.uniform(-0.005, 0.005, n))
    vols   = rng.integers(500_000, 2_000_000, n).astype(float)
    idx    = pd.date_range("2020-01-02", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": vols,
    }, index=idx)


EQUITY = 10_000_000.0   # ₹1 Crore
SYM    = "RELIANCE.NS"


# ── ATR helper ────────────────────────────────────────────────────────────────

class TestComputeATR:
    def test_returns_positive(self):
        df = _make_bars(30)
        atr = _compute_atr(df, period=14)
        assert atr > 0.0

    def test_zero_for_insufficient_data(self):
        df = _make_bars(5)
        assert _compute_atr(df, period=14) == 0.0

    def test_larger_swings_larger_atr(self):
        low_vol  = _make_bars(50, seed=0)
        high_vol = _make_bars(50, seed=99)
        # inflate high_vol swings
        high_vol["high"] *= 1.05
        high_vol["low"]  *= 0.95
        assert _compute_atr(high_vol) > _compute_atr(low_vol)

    def test_reproducible(self):
        df = _make_bars(40)
        assert _compute_atr(df, 14) == _compute_atr(df, 14)


# ── PositionSize dataclass ─────────────────────────────────────────────────────

class TestPositionSize:
    def _make(self, qty=10, entry=1000.0, stop=950.0, tp=1150.0) -> PositionSize:
        return PositionSize(
            symbol=SYM, quantity=qty, entry_price=entry,
            stop_price=stop, take_profit_price=tp,
            risk_per_share=entry-stop, total_risk=qty*(entry-stop),
            atr=25.0, sizing_method="test",
        )

    def test_is_valid_true(self):
        ps = self._make(qty=10)
        assert ps.is_valid is True

    def test_is_valid_false_zero_qty(self):
        ps = self._make(qty=0)
        assert ps.is_valid is False

    def test_is_valid_false_stop_above_entry(self):
        ps = self._make(stop=1100.0)   # stop > entry
        assert ps.is_valid is False

    def test_to_dict_keys(self):
        d = self._make().to_dict()
        for k in ["symbol", "quantity", "entry_price", "stop_price",
                  "risk_per_share", "total_risk", "atr", "sizing_method"]:
            assert k in d


# ── Fixed Fractional Sizer ─────────────────────────────────────────────────────

class TestFixedFractionalSizer:
    def test_returns_position_size(self):
        sizer = FixedFractionalSizer(risk_per_trade_pct=0.01)
        ps = sizer.compute(SYM, 1000.0, EQUITY, _make_bars(50))
        assert isinstance(ps, PositionSize)

    def test_quantity_positive_with_enough_data(self):
        sizer = FixedFractionalSizer(risk_per_trade_pct=0.01, min_trade_value=100.0)
        ps = sizer.compute(SYM, 1000.0, EQUITY, _make_bars(50))
        assert ps.quantity >= 1

    def test_stop_below_entry(self):
        sizer = FixedFractionalSizer(risk_per_trade_pct=0.01)
        ps = sizer.compute(SYM, 1000.0, EQUITY, _make_bars(50))
        if ps.quantity > 0:
            assert ps.stop_price < ps.entry_price

    def test_take_profit_above_entry(self):
        sizer = FixedFractionalSizer(take_profit_mult=3.0)
        ps = sizer.compute(SYM, 1000.0, EQUITY, _make_bars(50))
        if ps.quantity > 0 and ps.take_profit_price:
            assert ps.take_profit_price > ps.entry_price

    def test_no_tp_when_mult_none(self):
        sizer = FixedFractionalSizer(take_profit_mult=None)
        ps = sizer.compute(SYM, 1000.0, EQUITY, _make_bars(50))
        assert ps.take_profit_price is None

    def test_zero_when_insufficient_data(self):
        sizer = FixedFractionalSizer()
        ps = sizer.compute(SYM, 1000.0, EQUITY, _make_bars(5))  # ATR fails
        assert ps.quantity == 0.0

    def test_zero_when_price_zero(self):
        sizer = FixedFractionalSizer()
        ps = sizer.compute(SYM, 0.0, EQUITY, _make_bars(50))
        assert ps.quantity == 0.0

    def test_zero_when_equity_zero(self):
        sizer = FixedFractionalSizer()
        ps = sizer.compute(SYM, 1000.0, 0.0, _make_bars(50))
        assert ps.quantity == 0.0

    def test_max_position_cap_respected(self):
        sizer = FixedFractionalSizer(
            risk_per_trade_pct=0.10,  # aggressive — 10% at risk
            max_position_pct=0.03,    # but capped at 3% of equity
        )
        ps = sizer.compute(SYM, 1000.0, EQUITY, _make_bars(50))
        if ps.quantity > 0:
            assert ps.quantity * 1000.0 <= EQUITY * 0.03 + 1  # +1 for floor rounding

    def test_total_risk_equals_qty_times_risk_per_share(self):
        sizer = FixedFractionalSizer(risk_per_trade_pct=0.01)
        ps = sizer.compute(SYM, 1000.0, EQUITY, _make_bars(50))
        if ps.quantity > 0:
            assert abs(ps.total_risk - ps.quantity * ps.risk_per_share) < 0.01

    def test_sizing_method_label(self):
        sizer = FixedFractionalSizer()
        ps = sizer.compute(SYM, 1000.0, EQUITY, _make_bars(50))
        assert ps.sizing_method == "fixed_fractional"

    def test_higher_risk_pct_larger_quantity(self):
        df = _make_bars(50)
        s1 = FixedFractionalSizer(risk_per_trade_pct=0.005, min_trade_value=100.0)
        s2 = FixedFractionalSizer(risk_per_trade_pct=0.02,  min_trade_value=100.0)
        q1 = s1.compute(SYM, 1000.0, EQUITY, df).quantity
        q2 = s2.compute(SYM, 1000.0, EQUITY, df).quantity
        assert q2 >= q1

    def test_respects_capital_budget(self):
        sizer = FixedFractionalSizer(risk_per_trade_pct=0.05, min_trade_value=100.0)
        ps = sizer.compute(SYM, 1000.0, EQUITY, _make_bars(50), capital_budget=25_000.0)
        assert ps.quantity * 1000.0 <= 25_000.0 + 1.0

    def test_fixed_stop_method_uses_pct(self):
        sizer = FixedFractionalSizer(
            risk_per_trade_pct=0.01,
            stop_method="fixed_pct",
            fixed_stop_pct=0.03,
            min_trade_value=100.0,
        )
        ps = sizer.compute(SYM, 1000.0, EQUITY, _make_bars(5), capital_budget=100_000.0)
        assert ps.stop_price == pytest.approx(970.0)


# ── Inverse Volatility Sizer ───────────────────────────────────────────────────

class TestInverseVolatilitySizer:
    def test_returns_position_size(self):
        sizer = InverseVolatilitySizer(risk_per_trade_pct=0.01)
        ps = sizer.compute(SYM, 1000.0, EQUITY, _make_bars(50))
        assert isinstance(ps, PositionSize)

    def test_quantity_positive_with_enough_data(self):
        sizer = InverseVolatilitySizer(risk_per_trade_pct=0.01, min_trade_value=100.0)
        ps = sizer.compute(SYM, 1000.0, EQUITY, _make_bars(50))
        assert ps.quantity >= 1

    def test_stop_below_entry(self):
        sizer = InverseVolatilitySizer(risk_per_trade_pct=0.01)
        ps = sizer.compute(SYM, 1000.0, EQUITY, _make_bars(50))
        if ps.quantity > 0:
            assert ps.stop_price < ps.entry_price

    def test_high_vol_smaller_position(self):
        """Higher volatility asset should get smaller position."""
        df_low  = _make_bars(50, seed=1)
        df_high = _make_bars(50, seed=1)
        # Inflate high-vol bars
        df_high["high"] *= 1.1
        df_high["low"]  *= 0.9
        s = InverseVolatilitySizer(risk_per_trade_pct=0.01, min_trade_value=100.0)
        qty_low  = s.compute(SYM, 1000.0, EQUITY, df_low).quantity
        qty_high = s.compute(SYM, 1000.0, EQUITY, df_high).quantity
        assert qty_low >= qty_high  # more volatile → smaller quantity

    def test_sizing_method_label(self):
        ps = InverseVolatilitySizer().compute(SYM, 1000.0, EQUITY, _make_bars(50))
        assert ps.sizing_method == "inverse_volatility"

    def test_zero_for_insufficient_data(self):
        ps = InverseVolatilitySizer().compute(SYM, 1000.0, EQUITY, _make_bars(5))
        assert ps.quantity == 0.0

    def test_capital_budget_caps_inverse_volatility_position(self):
        sizer = InverseVolatilitySizer(risk_per_trade_pct=0.03, min_trade_value=100.0)
        ps = sizer.compute(SYM, 1000.0, EQUITY, _make_bars(50), capital_budget=20_000.0)
        assert ps.quantity * 1000.0 <= 20_000.0 + 1.0


# ── Kelly Sizer ────────────────────────────────────────────────────────────────

class TestKellySizer:
    def test_returns_position_size(self):
        sizer = KellySizer(win_rate=0.60, avg_win=1.5, avg_loss=1.0)
        ps = sizer.compute(SYM, 1000.0, EQUITY, _make_bars(50))
        assert isinstance(ps, PositionSize)

    def test_positive_edge_gives_positive_qty(self):
        sizer = KellySizer(win_rate=0.60, avg_win=2.0, avg_loss=1.0,
                            min_trade_value=100.0)
        ps = sizer.compute(SYM, 1000.0, EQUITY, _make_bars(50))
        assert ps.quantity >= 1

    def test_zero_or_negative_edge_gives_zero_qty(self):
        # win_rate=0.3, avg_win=1.0 → negative edge
        sizer = KellySizer(win_rate=0.30, avg_win=1.0, avg_loss=1.0)
        ps = sizer.compute(SYM, 1000.0, EQUITY, _make_bars(50))
        assert ps.quantity == 0.0

    def test_max_kelly_fraction_respected(self):
        # Even with extreme params, position shouldn't exceed max_kelly × equity / price
        sizer = KellySizer(win_rate=0.99, avg_win=100.0, avg_loss=1.0,
                            max_kelly_fraction=0.05, max_position_pct=0.10)
        ps = sizer.compute(SYM, 1000.0, EQUITY, _make_bars(50))
        max_value = EQUITY * 0.10  # max_position_pct cap
        if ps.quantity > 0:
            assert ps.quantity * 1000.0 <= max_value + 1

    def test_sizing_method_label(self):
        ps = KellySizer(win_rate=0.6, avg_win=1.5, avg_loss=1.0).compute(
            SYM, 1000.0, EQUITY, _make_bars(50)
        )
        assert ps.sizing_method == "kelly"


# ── Factory ────────────────────────────────────────────────────────────────────

class TestPositionSizerFactory:
    def test_builds_fixed_fractional(self):
        s = PositionSizerFactory.build("fixed_fractional")
        assert isinstance(s, FixedFractionalSizer)

    def test_builds_inverse_volatility(self):
        s = PositionSizerFactory.build("inverse_volatility")
        assert isinstance(s, InverseVolatilitySizer)

    def test_builds_kelly(self):
        s = PositionSizerFactory.build("kelly")
        assert isinstance(s, KellySizer)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            PositionSizerFactory.build("magic_sizer")

    def test_factory_result_computes(self):
        for method in ["fixed_fractional", "inverse_volatility", "kelly"]:
            s = PositionSizerFactory.build(
                method, risk_per_trade_pct=0.01, min_trade_value=100.0,
                win_rate=0.60, avg_win=1.5, avg_loss=1.0,
            )
            ps = s.compute(SYM, 1000.0, EQUITY, _make_bars(50))
            assert isinstance(ps, PositionSize)
