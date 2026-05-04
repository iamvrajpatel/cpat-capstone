"""Tests — Analytics Extensions (Week 3): drawdown, distributions, new PerformanceReport fields."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from cpat.analytics.drawdown import (
    DrawdownPeriod, compute_drawdown_series, compute_drawdown_table,
    drawdown_table_to_dataframe, max_drawdown_details, ulcer_index,
)
from cpat.analytics.distributions import (
    ReturnDistribution, compute_distribution, is_normal, monthly_returns_table,
)
from cpat.analytics.performance import PerformanceTracker


def _eq(values, freq="B"):
    idx = pd.date_range("2020-01-02", periods=len(values), freq=freq, tz="UTC")
    return pd.Series(values, index=idx, name="equity")

def _rets(n=500, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n, freq="B", tz="UTC")
    return pd.Series(rng.normal(0.0003, 0.01, n), index=idx, name="returns")

def _tracker(n=300):
    t = PerformanceTracker(initial_capital=1_000_000, risk_free_rate=0.04)
    rng = np.random.default_rng(42)
    equity = 1_000_000.0
    for ts in pd.date_range("2020-01-02", periods=n, freq="B", tz="UTC"):
        equity *= (1 + rng.normal(0.0003, 0.01))
        t.record(ts, equity)
    return t


# ── Drawdown series ────────────────────────────────────────────────────────────
class TestDrawdownSeries:
    def test_flat_zero_drawdown(self):
        assert (compute_drawdown_series(_eq([100, 100, 100])) == 0.0).all()

    def test_rising_zero_drawdown(self):
        assert (compute_drawdown_series(_eq([100, 110, 120])) == 0.0).all()

    def test_known_depth(self):
        dd = compute_drawdown_series(_eq([100, 120, 90, 100]))
        assert dd.iloc[2] == pytest.approx(-0.25, abs=1e-9)

    def test_values_non_positive(self):
        rng = np.random.default_rng(0)
        eq = _eq((100 * np.cumprod(1 + rng.normal(0, 0.01, 100))).tolist())
        assert (compute_drawdown_series(eq) <= 0.0).all()

    def test_empty(self):
        assert compute_drawdown_series(pd.Series(dtype=float)).empty


# ── Drawdown table ─────────────────────────────────────────────────────────────
class TestDrawdownTable:
    def test_no_drawdowns(self):
        assert compute_drawdown_table(_eq([100, 105, 110])) == []

    def test_single_recovered(self):
        ps = compute_drawdown_table(_eq([100, 120, 90, 130]))
        assert len(ps) == 1
        assert ps[0].depth == pytest.approx(-0.25, abs=1e-9)
        assert ps[0].is_recovered

    def test_open_drawdown(self):
        ps = compute_drawdown_table(_eq([100, 120, 90, 100]))
        open_ps = [p for p in ps if not p.is_recovered]
        assert len(open_ps) >= 1

    def test_sorted_worst_first(self):
        ps = compute_drawdown_table(_eq([100, 110, 99, 110, 114, 108, 114]))
        if len(ps) >= 2:
            assert ps[0].depth <= ps[1].depth

    def test_depth_negative(self):
        for p in compute_drawdown_table(_eq([100, 120, 60, 130])):
            assert p.depth < 0

    def test_to_dict(self):
        ps = compute_drawdown_table(_eq([100, 120, 90, 130]))
        if ps:
            d = ps[0].to_dict()
            assert "depth_pct" in d and "is_recovered" in d

    def test_dataframe_empty_input(self):
        df = drawdown_table_to_dataframe([])
        assert df.empty

    def test_dataframe_columns(self):
        ps = compute_drawdown_table(_eq([100, 120, 90, 130]))
        df = drawdown_table_to_dataframe(ps)
        assert "depth_pct" in df.columns


class TestMaxDrawdownDetails:
    def test_worst_period(self):
        result = max_drawdown_details(_eq([100, 120, 90, 130, 80, 130]))
        assert result is not None and result.depth < -0.30

    def test_none_for_rising(self):
        assert max_drawdown_details(_eq([100, 110, 120])) is None


class TestUlcerIndex:
    def test_zero_rising(self):
        assert ulcer_index(_eq([100, 110, 120])) == pytest.approx(0.0, abs=1e-9)

    def test_positive_with_dd(self):
        assert ulcer_index(_eq([100, 120, 90, 100, 120, 90])) > 0.0

    def test_empty_zero(self):
        assert ulcer_index(pd.Series(dtype=float)) == 0.0

    def test_larger_dd_higher_ui(self):
        assert ulcer_index(_eq([100, 120, 60, 120])) > ulcer_index(_eq([100, 105, 102, 105]))


# ── Return Distributions ───────────────────────────────────────────────────────
class TestComputeDistribution:
    def test_structure(self):
        d = compute_distribution(_rets(200))
        assert isinstance(d, ReturnDistribution)
        assert d.n_obs == 200 and d.std > 0

    def test_cvar_leq_var(self):
        d = compute_distribution(_rets(500))
        assert d.cvar_95 <= d.var_95 + 1e-9

    def test_tail_ratio_positive(self):
        assert compute_distribution(_rets(300)).tail_ratio > 0.0

    def test_to_dict_keys(self):
        d = compute_distribution(_rets(100)).to_dict()
        for k in ["mean_daily_pct", "skewness", "kurtosis_excess", "var_95_pct", "is_normal"]:
            assert k in d

    def test_insufficient_data_no_raise(self):
        d = compute_distribution(pd.Series([0.01, -0.01, 0.005]))
        assert d.n_obs == 3

    def test_is_normal_bool(self):
        assert isinstance(compute_distribution(_rets(200)).is_normal, bool)


class TestIsNormal:
    def test_returns_bool(self):
        assert isinstance(is_normal(_rets(200)), bool)

    def test_short_series_true(self):
        assert is_normal(pd.Series([0.01, -0.01])) is True


class TestMonthlyReturnsTable:
    def test_returns_dataframe(self):
        rng = np.random.default_rng(0)
        eq = _eq((100 * np.cumprod(1 + rng.normal(0.0005, 0.01, 500))).tolist())
        assert isinstance(monthly_returns_table(eq), pd.DataFrame)

    def test_empty_input(self):
        assert monthly_returns_table(pd.Series(dtype=float)).empty


# ── PerformanceReport extended fields ─────────────────────────────────────────
class TestPerformanceReportExtended:
    def test_expectancy_exists(self):
        r = _tracker().compute()
        assert hasattr(r, "expectancy") and isinstance(r.expectancy, float)

    def test_skewness_exists(self):
        r = _tracker().compute()
        assert hasattr(r, "skewness") and isinstance(r.skewness, float)

    def test_kurtosis_exists(self):
        r = _tracker().compute()
        assert hasattr(r, "kurtosis") and isinstance(r.kurtosis, float)

    def test_tail_ratio_positive(self):
        assert _tracker().compute().tail_ratio > 0.0

    def test_ulcer_index_nonneg(self):
        assert _tracker().compute().ulcer_index >= 0.0

    def test_recovery_factor_float(self):
        assert isinstance(_tracker().compute().recovery_factor, float)

    def test_to_dict_new_fields(self):
        d = _tracker().compute().to_dict()
        for f in ["expectancy", "skewness", "kurtosis", "tail_ratio", "ulcer_index", "recovery_factor"]:
            assert f in d, f"Missing '{f}' in to_dict()"

    def test_str_contains_ulcer(self):
        assert "Ulcer" in str(_tracker().compute())

    def test_recovery_factor_zero_no_dd(self):
        t = PerformanceTracker(initial_capital=1_000_000)
        equity = 1_000_000.0
        for ts in pd.date_range("2021-01-04", periods=20, freq="B", tz="UTC"):
            equity *= 1.001
            t.record(ts, equity)
        assert t.compute().recovery_factor == 0.0
