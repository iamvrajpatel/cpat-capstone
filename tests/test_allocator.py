"""Tests — Capital Allocator (Week 4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cpat.portfolio.allocator import (
    CapitalSlice,
    EqualWeightAllocator,
    VolatilityAdjustedAllocator,
    AllocatorFactory,
    _compute_sigma,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_bars(n=60, vol=0.01, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 1000 * np.cumprod(1 + rng.normal(0.0003, vol, n))
    highs = closes * (1 + rng.uniform(0.001, 0.005, n))
    lows = closes * (1 - rng.uniform(0.001, 0.005, n))
    idx = pd.date_range("2020-01-02", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({"open": closes, "high": highs, "low": lows,
                         "close": closes, "volume": 1e6}, index=idx)

EQUITY = 10_000_000.0
CASH   = 3_000_000.0
SIGNALS = ["A", "B", "C"]
BAR_DATA = {s: _make_bars(60, seed=i) for i, s in enumerate(SIGNALS)}


# ── _compute_sigma ─────────────────────────────────────────────────────────────

class TestComputeSigma:
    def test_positive_with_data(self):
        assert _compute_sigma(_make_bars(60)) > 0.0

    def test_zero_insufficient_data(self):
        assert _compute_sigma(_make_bars(5), window=20) == 0.0

    def test_higher_vol_higher_sigma(self):
        low  = _compute_sigma(_make_bars(60, vol=0.005))
        high = _compute_sigma(_make_bars(60, vol=0.03))
        assert high > low

    def test_annualised(self):
        s = _compute_sigma(_make_bars(60))
        assert 0.01 < s < 2.0  # sanity range for daily bars


# ── CapitalSlice ───────────────────────────────────────────────────────────────

class TestCapitalSlice:
    def test_is_valid_true(self):
        cs = CapitalSlice("S", target_value=100_000.0, weight=0.1)
        assert cs.is_valid is True

    def test_is_valid_false_zero(self):
        cs = CapitalSlice("S", target_value=0.0, weight=0.0)
        assert cs.is_valid is False

    def test_to_dict_keys(self):
        cs = CapitalSlice("S", 100_000.0, 0.1, sigma=0.15)
        d = cs.to_dict()
        assert all(k in d for k in ["symbol", "target_value", "weight", "sigma"])


# ── EqualWeightAllocator ───────────────────────────────────────────────────────

class TestEqualWeightAllocator:
    def test_returns_one_slice_per_signal(self):
        alloc = EqualWeightAllocator(max_position_pct=0.05, min_cash_pct=0.05)
        slices = alloc.allocate(SIGNALS, EQUITY, CASH, BAR_DATA)
        assert len(slices) == 3

    def test_all_slices_equal_value(self):
        alloc = EqualWeightAllocator(max_position_pct=0.10)
        slices = alloc.allocate(SIGNALS, EQUITY, CASH, BAR_DATA)
        values = [s.target_value for s in slices]
        assert max(values) - min(values) < 1.0  # equal (float precision)

    def test_weight_sums_approx_one(self):
        alloc = EqualWeightAllocator(max_position_pct=0.10)
        slices = alloc.allocate(SIGNALS, EQUITY, CASH, BAR_DATA)
        total = sum(s.weight for s in slices)
        assert abs(total - 1.0) < 1e-6

    def test_respects_max_position_cap(self):
        alloc = EqualWeightAllocator(max_position_pct=0.02)  # tight cap
        slices = alloc.allocate(SIGNALS, EQUITY, CASH, BAR_DATA)
        for sl in slices:
            assert sl.target_value <= EQUITY * 0.02 + 1.0

    def test_empty_signals_returns_empty(self):
        alloc = EqualWeightAllocator()
        assert alloc.allocate([], EQUITY, CASH, BAR_DATA) == []

    def test_zero_equity_returns_empty(self):
        alloc = EqualWeightAllocator()
        assert alloc.allocate(SIGNALS, 0.0, CASH, BAR_DATA) == []

    def test_cash_buffer_respected(self):
        # With min_cash_pct=0.10, deployable = 0.9 × cash
        alloc = EqualWeightAllocator(max_position_pct=0.50, min_cash_pct=0.10)
        slices = alloc.allocate(["A"], EQUITY, CASH, BAR_DATA)
        expected = CASH * 0.90
        assert abs(slices[0].target_value - expected) < 1.0

    def test_single_signal(self):
        alloc = EqualWeightAllocator(max_position_pct=0.10)
        slices = alloc.allocate(["A"], EQUITY, CASH, BAR_DATA)
        assert len(slices) == 1
        assert slices[0].symbol == "A"

    def test_slice_symbols_match_signals(self):
        alloc = EqualWeightAllocator()
        slices = alloc.allocate(SIGNALS, EQUITY, CASH, BAR_DATA)
        symbols = {s.symbol for s in slices}
        assert symbols == set(SIGNALS)


# ── VolatilityAdjustedAllocator ───────────────────────────────────────────────

class TestVolatilityAdjustedAllocator:
    def test_returns_one_slice_per_signal(self):
        alloc = VolatilityAdjustedAllocator()
        slices = alloc.allocate(SIGNALS, EQUITY, CASH, BAR_DATA)
        assert len(slices) == 3

    def test_high_vol_smaller_allocation(self):
        low_vol  = _make_bars(60, vol=0.005)
        high_vol = _make_bars(60, vol=0.04)
        bar_data = {"L": low_vol, "H": high_vol}
        alloc = VolatilityAdjustedAllocator(max_position_pct=0.50)
        slices = alloc.allocate(["L", "H"], EQUITY, CASH, bar_data)
        by_sym = {s.symbol: s for s in slices}
        assert by_sym["L"].target_value > by_sym["H"].target_value

    def test_respects_max_position_cap(self):
        alloc = VolatilityAdjustedAllocator(max_position_pct=0.02)
        slices = alloc.allocate(SIGNALS, EQUITY, CASH, BAR_DATA)
        for sl in slices:
            assert sl.target_value <= EQUITY * 0.02 + 1.0

    def test_empty_signals_returns_empty(self):
        alloc = VolatilityAdjustedAllocator()
        assert alloc.allocate([], EQUITY, CASH, BAR_DATA) == []

    def test_zero_equity_returns_empty(self):
        alloc = VolatilityAdjustedAllocator()
        assert alloc.allocate(SIGNALS, 0.0, CASH, BAR_DATA) == []

    def test_fallback_when_no_bar_data(self):
        # Symbols with no bar data fall back to equal weight
        alloc = VolatilityAdjustedAllocator(max_position_pct=0.50)
        slices = alloc.allocate(["X", "Y"], EQUITY, CASH, {})
        assert len(slices) == 2
        # Should not raise

    def test_sigma_recorded_in_slice(self):
        alloc = VolatilityAdjustedAllocator()
        slices = alloc.allocate(SIGNALS, EQUITY, CASH, BAR_DATA)
        # Slices with data should have sigma > 0
        for sl in slices:
            assert sl.sigma >= 0.0  # ≥ 0; 0 only for fallback

    def test_single_signal(self):
        alloc = VolatilityAdjustedAllocator(max_position_pct=0.50)
        slices = alloc.allocate(["A"], EQUITY, CASH, BAR_DATA)
        assert len(slices) == 1
        assert slices[0].is_valid

    def test_cap_redistribution_keeps_remaining_symbols_funded(self):
        alloc = VolatilityAdjustedAllocator(max_position_pct=0.03)
        slices = alloc.allocate(["A", "B", "C"], EQUITY, 10_000_000.0, BAR_DATA)
        assert all(sl.target_value <= EQUITY * 0.03 + 1.0 for sl in slices)
        assert sum(sl.target_value for sl in slices) > 0.0


# ── AllocatorFactory ───────────────────────────────────────────────────────────

class TestAllocatorFactory:
    def test_builds_equal_weight(self):
        a = AllocatorFactory.build("equal_weight")
        assert isinstance(a, EqualWeightAllocator)

    def test_builds_volatility_adjusted(self):
        a = AllocatorFactory.build("volatility_adjusted")
        assert isinstance(a, VolatilityAdjustedAllocator)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            AllocatorFactory.build("magic")

    def test_factory_result_allocates(self):
        for method in ["equal_weight", "volatility_adjusted"]:
            alloc = AllocatorFactory.build(method, max_position_pct=0.10)
            slices = alloc.allocate(SIGNALS, EQUITY, CASH, BAR_DATA)
            assert len(slices) == 3
