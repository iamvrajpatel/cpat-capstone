"""Tests — Validation (Week 3): splitter, walk-forward, overfitting detection."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from cpat.validation.splitter import (
    TimeSeriesSplit, train_test_split, time_series_cv_folds, _union_timestamps,
)
from cpat.validation.walk_forward import (
    WalkForwardConfig, WalkForwardValidator, WalkForwardResult, _stitch_equity_curves,
)
from cpat.validation.overfitting import (
    degradation_report, compute_sensitivity, stability_check, ParameterSensitivity,
)
from cpat.analytics.performance import PerformanceReport, PerformanceTracker
from cpat.config.loader import load_config


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_bar_df(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-02", periods=n, freq="B", tz="UTC")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    open_ = close * (1 + rng.uniform(-0.002, 0.002, n))
    high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.005, n))
    low  = np.minimum(close, open_) * (1 - rng.uniform(0, 0.005, n))
    vol  = rng.integers(500_000, 2_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "adj_close": close, "volume": vol}, index=idx,
    )

def _make_bar_data(n_sym=5, n_bars=800):
    return {f"SYM{i}": _make_bar_df(n_bars, seed=i) for i in range(n_sym)}

def _dummy_report(**overrides) -> PerformanceReport:
    defaults = dict(
        total_return=0.1, ann_return=0.08, ann_volatility=0.15,
        sharpe_ratio=0.9, sortino_ratio=1.2, max_drawdown=-0.1,
        max_drawdown_duration_days=30, calmar_ratio=0.8,
        win_rate=0.55, profit_factor=1.3, n_trades=50, n_profitable=27,
        avg_win=200.0, avg_loss=-100.0, initial_capital=1_000_000,
        final_equity=1_100_000, risk_free_rate=0.04,
    )
    defaults.update(overrides)
    return PerformanceReport(**defaults)

@pytest.fixture(scope="module")
def cfg():
    return load_config()

@pytest.fixture(scope="module")
def bar_data():
    return _make_bar_data()


# ═══════════════════════════════════════════════════════════════════════════════
# Splitter
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnionTimestamps:
    def test_single_symbol(self, bar_data):
        sym = next(iter(bar_data))
        ts = _union_timestamps({sym: bar_data[sym]})
        assert len(ts) == len(bar_data[sym])

    def test_multiple_symbols_same_index(self, bar_data):
        ts = _union_timestamps(bar_data)
        # All symbols have same index so union == single symbol length
        assert len(ts) == len(next(iter(bar_data.values())))


class TestTrainTestSplit:
    def test_sizes_approx_70_30(self, bar_data):
        train, test = train_test_split(bar_data, train_ratio=0.70, warmup_bars=50)
        total = _union_timestamps(bar_data)
        n_train = len(_union_timestamps(train))
        n_test  = len(_union_timestamps(test))
        # Train should be ~70% of total
        assert n_train / len(total) == pytest.approx(0.70, abs=0.05)
        assert n_test > 0

    def test_no_timestamp_overlap(self, bar_data):
        train, test = train_test_split(bar_data, train_ratio=0.70, warmup_bars=50)
        train_ts = set(_union_timestamps(train))
        test_ts  = set(_union_timestamps(test))
        assert train_ts.isdisjoint(test_ts)

    def test_test_after_train(self, bar_data):
        train, test = train_test_split(bar_data, warmup_bars=50)
        train_end = max(_union_timestamps(train))
        test_start = min(_union_timestamps(test))
        assert test_start > train_end

    def test_all_symbols_present(self, bar_data):
        train, test = train_test_split(bar_data, warmup_bars=50)
        assert set(train.keys()) == set(bar_data.keys())
        assert set(test.keys()) == set(bar_data.keys())

    def test_empty_data_raises(self):
        with pytest.raises(ValueError, match="empty"):
            train_test_split({})

    def test_invalid_ratio_raises(self, bar_data):
        with pytest.raises(ValueError):
            train_test_split(bar_data, train_ratio=1.5)

    def test_too_short_warmup_raises(self):
        tiny = {"S": _make_bar_df(n=10)}
        with pytest.raises((ValueError, Exception)):
            train_test_split(tiny, train_ratio=0.7, warmup_bars=200)


class TestTimeSeriesCVFolds:
    def test_returns_list_of_splits(self, bar_data):
        folds = time_series_cv_folds(bar_data, n_folds=3, warmup_bars=50)
        assert isinstance(folds, list)
        assert len(folds) >= 1

    def test_test_windows_non_overlapping(self, bar_data):
        folds = time_series_cv_folds(bar_data, n_folds=4, warmup_bars=50)
        for i in range(1, len(folds)):
            assert folds[i].test_start > folds[i-1].test_end

    def test_expanding_train(self, bar_data):
        folds = time_series_cv_folds(bar_data, n_folds=3, warmup_bars=50)
        if len(folds) >= 2:
            assert folds[0].train_start == folds[1].train_start  # same start
            assert folds[1].n_train > folds[0].n_train  # expanding

    def test_fold_ids_sequential(self, bar_data):
        folds = time_series_cv_folds(bar_data, n_folds=3, warmup_bars=50)
        for i, f in enumerate(folds):
            assert f.fold_id == i

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            time_series_cv_folds({}, n_folds=3)

    def test_too_few_folds_raises(self, bar_data):
        with pytest.raises(ValueError):
            time_series_cv_folds(bar_data, n_folds=1)

    def test_repr(self, bar_data):
        folds = time_series_cv_folds(bar_data, n_folds=2, warmup_bars=50)
        if folds:
            assert "TimeSeriesSplit" in repr(folds[0])


# ═══════════════════════════════════════════════════════════════════════════════
# Walk-Forward
# ═══════════════════════════════════════════════════════════════════════════════

class TestStitchEquityCurves:
    def test_empty_returns_empty(self):
        result = _stitch_equity_curves([])
        assert result.empty

    def test_single_curve_preserved(self):
        idx = pd.date_range("2020-01-02", periods=50, freq="B", tz="UTC")
        curve = pd.Series(range(1_000_000, 1_000_050, 1), index=idx, dtype=float)
        result = _stitch_equity_curves([curve])
        assert len(result) == len(curve)

    def test_two_curves_continuous(self):
        idx1 = pd.date_range("2020-01-02", periods=20, freq="B", tz="UTC")
        idx2 = pd.date_range("2020-02-03", periods=20, freq="B", tz="UTC")
        c1 = pd.Series([1_000_000.0] * 20, index=idx1)
        c2 = pd.Series([900_000.0] * 20, index=idx2)
        result = _stitch_equity_curves([c1, c2])
        assert len(result) == 40
        # After stitching, the transition should be continuous at c1[-1]
        assert result.iloc[20] == pytest.approx(result.iloc[19], rel=0.01)


class TestWalkForwardValidator:
    @pytest.fixture(scope="class")
    def wf_config(self):
        return WalkForwardConfig(
            train_bars=300, test_bars=100, step_bars=100,
            optimize_on_fold=False,
        )

    def test_runs_without_error(self, cfg, bar_data, wf_config):
        wf = WalkForwardValidator(strategy="momentum", wf_config=wf_config)
        result = wf.run(bar_data, cfg)
        assert isinstance(result, WalkForwardResult)

    def test_has_folds(self, cfg, bar_data, wf_config):
        wf = WalkForwardValidator(strategy="momentum", wf_config=wf_config)
        result = wf.run(bar_data, cfg)
        assert len(result.folds) >= 1

    def test_oos_equity_non_empty(self, cfg, bar_data, wf_config):
        wf = WalkForwardValidator(strategy="momentum", wf_config=wf_config)
        result = wf.run(bar_data, cfg)
        assert not result.combined_oos_equity.empty

    def test_degradation_ratio_is_float(self, cfg, bar_data, wf_config):
        wf = WalkForwardValidator(strategy="momentum", wf_config=wf_config)
        result = wf.run(bar_data, cfg)
        assert isinstance(result.degradation_ratio, float)

    def test_consistency_score_in_range(self, cfg, bar_data, wf_config):
        wf = WalkForwardValidator(strategy="momentum", wf_config=wf_config)
        result = wf.run(bar_data, cfg)
        assert 0.0 <= result.consistency_score <= 1.0

    def test_summary_dataframe(self, cfg, bar_data, wf_config):
        wf = WalkForwardValidator(strategy="momentum", wf_config=wf_config)
        result = wf.run(bar_data, cfg)
        df = result.summary_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "is_sharpe" in df.columns
        assert "oos_sharpe" in df.columns

    def test_fold_degradation_is_float(self, cfg, bar_data, wf_config):
        wf = WalkForwardValidator(strategy="momentum", wf_config=wf_config)
        result = wf.run(bar_data, cfg)
        for fold in result.folds:
            assert isinstance(fold.degradation, float)

    def test_str_output(self, cfg, bar_data, wf_config):
        wf = WalkForwardValidator(strategy="momentum", wf_config=wf_config)
        result = wf.run(bar_data, cfg)
        s = str(result)
        assert "Walk-Forward" in s and "Degradation" in s

    def test_insufficient_data_raises(self, cfg):
        tiny = {"S": _make_bar_df(n=50)}
        wf_cfg = WalkForwardConfig(train_bars=300, test_bars=100)
        wf = WalkForwardValidator(strategy="momentum", wf_config=wf_cfg)
        with pytest.raises((ValueError, RuntimeError)):
            wf.run(tiny, cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# Overfitting Detection
# ═══════════════════════════════════════════════════════════════════════════════

class TestDegradationReport:
    def test_structure(self):
        is_r  = _dummy_report(sharpe_ratio=1.5, total_return=0.20)
        oos_r = _dummy_report(sharpe_ratio=0.9, total_return=0.10)
        d = degradation_report(is_r, oos_r)
        assert "assessment" in d
        assert "is" in d and "oos" in d
        assert "degradation" in d

    def test_robust_assessment(self):
        is_r  = _dummy_report(sharpe_ratio=1.0)
        oos_r = _dummy_report(sharpe_ratio=0.8)
        d = degradation_report(is_r, oos_r)
        assert "ROBUST" in d["assessment"]

    def test_failed_assessment(self):
        is_r  = _dummy_report(sharpe_ratio=1.0)
        oos_r = _dummy_report(sharpe_ratio=-0.5)
        d = degradation_report(is_r, oos_r)
        assert "FAILED" in d["assessment"]

    def test_degradation_ratio_formula(self):
        is_r  = _dummy_report(sharpe_ratio=2.0)
        oos_r = _dummy_report(sharpe_ratio=1.0)
        d = degradation_report(is_r, oos_r)
        assert d["degradation"]["sharpe_ratio"] == pytest.approx(0.5, abs=1e-9)

    def test_label_preserved(self):
        is_r = oos_r = _dummy_report()
        d = degradation_report(is_r, oos_r, label="fold_3")
        assert d["label"] == "fold_3"

    def test_is_metrics_present(self):
        is_r  = _dummy_report(n_trades=80)
        oos_r = _dummy_report(n_trades=30)
        d = degradation_report(is_r, oos_r)
        assert d["is"]["n_trades"] == 80
        assert d["oos"]["n_trades"] == 30


class TestComputeSensitivity:
    @pytest.fixture
    def results_df(self):
        rows = [
            {"lookback_long": 126, "top_n_pct": 0.10, "sharpe_ratio": 0.5, "n_trades": 10},
            {"lookback_long": 126, "top_n_pct": 0.20, "sharpe_ratio": 0.8, "n_trades": 12},
            {"lookback_long": 252, "top_n_pct": 0.10, "sharpe_ratio": 1.2, "n_trades": 8},
            {"lookback_long": 252, "top_n_pct": 0.20, "sharpe_ratio": 1.5, "n_trades": 9},
        ]
        return pd.DataFrame(rows)

    def test_returns_sensitivity(self, results_df):
        s = compute_sensitivity(results_df, "lookback_long", "sharpe_ratio")
        assert isinstance(s, ParameterSensitivity)
        assert s.param_name == "lookback_long"
        assert s.metric_name == "sharpe_ratio"

    def test_cv_non_negative(self, results_df):
        s = compute_sensitivity(results_df, "lookback_long", "sharpe_ratio")
        assert s.cv >= 0.0

    def test_stable_flag(self, results_df):
        # Stable = CV < 0.30
        s = compute_sensitivity(results_df, "lookback_long", "sharpe_ratio")
        assert s.is_stable == (s.cv < 0.30)

    def test_to_dict_keys(self, results_df):
        d = compute_sensitivity(results_df, "lookback_long", "sharpe_ratio").to_dict()
        assert "cv" in d and "is_stable" in d and "values" in d

    def test_bad_param_raises(self, results_df):
        with pytest.raises(KeyError):
            compute_sensitivity(results_df, "nonexistent", "sharpe_ratio")

    def test_bad_metric_raises(self, results_df):
        with pytest.raises(KeyError):
            compute_sensitivity(results_df, "lookback_long", "nonexistent")


class TestStabilityCheck:
    @pytest.fixture
    def results_df(self):
        rows = [
            {"lookback_long": 126, "top_n_pct": 0.10, "sharpe_ratio": 0.5},
            {"lookback_long": 189, "top_n_pct": 0.15, "sharpe_ratio": 0.9},
            {"lookback_long": 252, "top_n_pct": 0.20, "sharpe_ratio": 1.2},
        ]
        return pd.DataFrame(rows)

    def test_returns_dataframe(self, results_df):
        df = stability_check(results_df, metric="sharpe_ratio")
        assert isinstance(df, pd.DataFrame)

    def test_sorted_by_cv(self, results_df):
        df = stability_check(results_df, metric="sharpe_ratio")
        if len(df) >= 2:
            assert df["cv"].iloc[0] >= df["cv"].iloc[-1]

    def test_columns_present(self, results_df):
        df = stability_check(results_df)
        if not df.empty:
            assert "param_name" in df.columns
            assert "cv" in df.columns
            assert "is_stable" in df.columns

    def test_empty_df_returns_empty(self):
        df = stability_check(pd.DataFrame())
        assert df.empty
