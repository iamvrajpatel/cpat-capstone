"""Tests — Optimizer (Week 3): GridSearch, RandomSearch, results_to_dataframe."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from cpat.optimization.optimizer import (
    GridSearchOptimizer, RandomSearchOptimizer,
    OptimizationResult, results_to_dataframe,
    MOMENTUM_PARAM_GRID, MEAN_REVERSION_PARAM_GRID,
    _apply_params_to_config,
)
from cpat.config.loader import load_config


# ── Minimal synthetic bar data ─────────────────────────────────────────────────

def _make_bar_df(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-02", periods=n, freq="B", tz="UTC")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, n))
    open_ = close * (1 + rng.uniform(-0.002, 0.002, n))
    high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.005, n))
    low  = np.minimum(close, open_) * (1 - rng.uniform(0, 0.005, n))
    vol  = rng.integers(500_000, 2_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "adj_close": close, "volume": vol},
        index=idx,
    )

def _make_bar_data(n_sym: int = 5, n_bars: int = 600) -> dict:
    syms = [f"SYM{i}" for i in range(n_sym)]
    return {s: _make_bar_df(n_bars, seed=i) for i, s in enumerate(syms)}

@pytest.fixture(scope="module")
def cfg():
    return load_config()

@pytest.fixture(scope="module")
def small_bar_data():
    return _make_bar_data(n_sym=4, n_bars=600)

@pytest.fixture(scope="module")
def tiny_param_grid():
    """2 × 2 grid = 4 combinations max."""
    return {
        "lookback_long": [126, 252],
        "top_n_pct": [0.15, 0.20],
    }

@pytest.fixture(scope="module")
def tiny_mr_grid():
    return {
        "lookback_window": [10, 20],
        "entry_zscore": [1.5, 2.0],
    }


# ── _apply_params_to_config ────────────────────────────────────────────────────
class TestApplyParamsToConfig:
    def test_applies_lookback(self, cfg):
        new_cfg = _apply_params_to_config(cfg, {"lookback_long": 189}, "momentum")
        assert new_cfg is not None
        assert new_cfg.strategies.momentum.lookback_long == 189

    def test_invalid_combo_returns_none(self, cfg):
        # lookback_long must be > lookback_short; set both equal
        result = _apply_params_to_config(cfg, {"lookback_long": 10, "lookback_short": 10}, "momentum")
        assert result is None

    def test_mean_reversion_params(self, cfg):
        new_cfg = _apply_params_to_config(cfg, {"lookback_window": 15}, "mean_reversion")
        assert new_cfg is not None
        assert new_cfg.strategies.mean_reversion.lookback_window == 15


# ── GridSearchOptimizer ────────────────────────────────────────────────────────
class TestGridSearchOptimizer:
    def test_returns_list(self, cfg, small_bar_data, tiny_param_grid):
        opt = GridSearchOptimizer(strategy="momentum", metric="sharpe_ratio")
        results = opt.run(small_bar_data, cfg, tiny_param_grid)
        assert isinstance(results, list)

    def test_results_sorted_best_first(self, cfg, small_bar_data, tiny_param_grid):
        opt = GridSearchOptimizer(strategy="momentum", metric="sharpe_ratio")
        results = opt.run(small_bar_data, cfg, tiny_param_grid)
        if len(results) >= 2:
            sharpes = [r.metric("sharpe_ratio") for r in results]
            assert sharpes == sorted(sharpes, reverse=True)

    def test_each_result_has_params(self, cfg, small_bar_data, tiny_param_grid):
        opt = GridSearchOptimizer(strategy="momentum", metric="sharpe_ratio")
        results = opt.run(small_bar_data, cfg, tiny_param_grid)
        for r in results:
            assert "lookback_long" in r.params
            assert isinstance(r.report.sharpe_ratio, float)

    def test_slice_label_train(self, cfg, small_bar_data, tiny_param_grid):
        opt = GridSearchOptimizer(strategy="momentum")
        results = opt.run(small_bar_data, cfg, tiny_param_grid)
        for r in results:
            assert r.slice_label == "train"

    def test_no_more_than_grid_size(self, cfg, small_bar_data, tiny_param_grid):
        opt = GridSearchOptimizer(strategy="momentum")
        results = opt.run(small_bar_data, cfg, tiny_param_grid)
        # 2×2 grid = 4 combinations maximum
        assert len(results) <= 4

    def test_mean_reversion_strategy(self, cfg, small_bar_data, tiny_mr_grid):
        opt = GridSearchOptimizer(strategy="mean_reversion", metric="sharpe_ratio")
        results = opt.run(small_bar_data, cfg, tiny_mr_grid)
        assert isinstance(results, list)


# ── RandomSearchOptimizer ──────────────────────────────────────────────────────
class TestRandomSearchOptimizer:
    def test_returns_list(self, cfg, small_bar_data, tiny_param_grid):
        opt = RandomSearchOptimizer(strategy="momentum", n_trials=3, random_seed=42)
        results = opt.run(small_bar_data, cfg, tiny_param_grid)
        assert isinstance(results, list)

    def test_n_trials_respected(self, cfg, small_bar_data, tiny_param_grid):
        # 4 total combos, request 3
        opt = RandomSearchOptimizer(strategy="momentum", n_trials=3, random_seed=0)
        results = opt.run(small_bar_data, cfg, tiny_param_grid)
        assert len(results) <= 3

    def test_reproducible_with_same_seed(self, cfg, small_bar_data, tiny_param_grid):
        opt1 = RandomSearchOptimizer(strategy="momentum", n_trials=2, random_seed=7)
        opt2 = RandomSearchOptimizer(strategy="momentum", n_trials=2, random_seed=7)
        r1 = opt1.run(small_bar_data, cfg, tiny_param_grid)
        r2 = opt2.run(small_bar_data, cfg, tiny_param_grid)
        if r1 and r2:
            assert r1[0].params == r2[0].params

    def test_sorted_best_first(self, cfg, small_bar_data, tiny_param_grid):
        opt = RandomSearchOptimizer(strategy="momentum", n_trials=4, random_seed=1)
        results = opt.run(small_bar_data, cfg, tiny_param_grid)
        if len(results) >= 2:
            sharpes = [r.metric("sharpe_ratio") for r in results]
            assert sharpes == sorted(sharpes, reverse=True)

    def test_capped_at_grid_size(self, cfg, small_bar_data, tiny_param_grid):
        # Grid only has 4 combos; requesting 100 should cap at 4
        opt = RandomSearchOptimizer(strategy="momentum", n_trials=100, random_seed=0)
        results = opt.run(small_bar_data, cfg, tiny_param_grid)
        assert len(results) <= 4


# ── OptimizationResult ─────────────────────────────────────────────────────────
class TestOptimizationResult:
    def test_metric_method(self, cfg, small_bar_data, tiny_param_grid):
        opt = GridSearchOptimizer(strategy="momentum")
        results = opt.run(small_bar_data, cfg, tiny_param_grid)
        if results:
            r = results[0]
            assert isinstance(r.metric("sharpe_ratio"), float)
            assert isinstance(r.metric("total_return"), float)

    def test_metric_bad_name_raises(self, cfg, small_bar_data, tiny_param_grid):
        opt = GridSearchOptimizer(strategy="momentum")
        results = opt.run(small_bar_data, cfg, tiny_param_grid)
        if results:
            with pytest.raises(AttributeError):
                results[0].metric("nonexistent_field")


# ── results_to_dataframe ───────────────────────────────────────────────────────
class TestResultsToDataframe:
    def test_empty_input(self):
        df = results_to_dataframe([])
        assert df.empty

    def test_columns_present(self, cfg, small_bar_data, tiny_param_grid):
        opt = GridSearchOptimizer(strategy="momentum")
        results = opt.run(small_bar_data, cfg, tiny_param_grid)
        if results:
            df = results_to_dataframe(results)
            assert "sharpe_ratio" in df.columns
            assert "total_return_pct" in df.columns
            assert "n_trades" in df.columns
            assert "lookback_long" in df.columns

    def test_sorted_best_sharpe_first(self, cfg, small_bar_data, tiny_param_grid):
        opt = GridSearchOptimizer(strategy="momentum")
        results = opt.run(small_bar_data, cfg, tiny_param_grid)
        if results:
            df = results_to_dataframe(results)
            if len(df) >= 2:
                assert df["sharpe_ratio"].iloc[0] >= df["sharpe_ratio"].iloc[-1]

    def test_one_row_per_result(self, cfg, small_bar_data, tiny_param_grid):
        opt = RandomSearchOptimizer(strategy="momentum", n_trials=2)
        results = opt.run(small_bar_data, cfg, tiny_param_grid)
        df = results_to_dataframe(results)
        assert len(df) == len(results)
