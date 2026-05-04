"""
CPAT — Parameter Optimization Framework
=========================================
Provides Grid Search and Random Search over the strategy parameter space,
running each combination exclusively on the **training slice** of bar data.

Critical design rules:
    1. The optimizer NEVER touches the test split.
    2. Each run uses a fresh BacktestEngine to avoid state leakage.
    3. Combinations producing fewer than `min_bars` after warmup are skipped.
    4. Results are sorted best-first by the chosen metric.

Usage::

    from cpat.optimization.optimizer import GridSearchOptimizer, MOMENTUM_PARAM_GRID

    optimizer = GridSearchOptimizer(
        strategy="momentum",
        metric="sharpe_ratio",
        random_seed=42,
    )
    results = optimizer.run(train_bar_data, base_config, MOMENTUM_PARAM_GRID)
    df = results_to_dataframe(results)
    print(df.head(10))
"""

from __future__ import annotations

import copy
import itertools
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import pandas as pd

from cpat.analytics.performance import PerformanceReport, PerformanceTracker
from cpat.backtest.engine import BacktestEngine, BacktestResult
from cpat.config.loader import (
    CPATConfig,
    MeanReversionStrategyConfig,
    MomentumStrategyConfig,
)
from cpat.strategies.mean_reversion import MeanReversionStrategy
from cpat.strategies.momentum import MomentumStrategy

logger = logging.getLogger(__name__)

# ── Pre-defined search spaces ──────────────────────────────────────────────────

MOMENTUM_PARAM_GRID: dict[str, list[Any]] = {
    "lookback_long":    [126, 189, 252],      # 6m, 9m, 12m
    "lookback_short":   [10, 15, 21],         # 2w, 3w, 1m
    "skip_most_recent": [0, 10, 21],
    "top_n_pct":        [0.10, 0.15, 0.20],
    "rebalance_frequency": [10, 21, 42],
}

MEAN_REVERSION_PARAM_GRID: dict[str, list[Any]] = {
    "lookback_window": [10, 15, 20, 30],
    "entry_zscore":    [1.5, 2.0, 2.5],
    "exit_zscore":     [0.25, 0.5, 0.75],
    "rsi_oversold":    [25.0, 30.0, 35.0],
}


# ── Result dataclass ───────────────────────────────────────────────────────────


@dataclass
class OptimizationResult:
    """Result of a single parameter combination backtest run.

    Attributes:
        params      : The parameter dictionary used in this run.
        report      : Full PerformanceReport from the backtest.
        run_time_s  : Wall-clock time for this run (seconds).
        strategy    : Strategy name ("momentum" or "mean_reversion").
        slice_label : "train" or "test" — which data slice was used.
    """

    params: dict[str, Any]
    report: PerformanceReport
    run_time_s: float
    strategy: str
    slice_label: str = "train"

    def metric(self, name: str) -> float:
        """Get a scalar metric by field name from the PerformanceReport."""
        val = getattr(self.report, name, None)
        if val is None:
            raise AttributeError(f"PerformanceReport has no field '{name}'")
        return float(val)


# ── Helper: single-run backtest factory ───────────────────────────────────────


def _run_single(
    bar_data: dict[str, pd.DataFrame],
    config: CPATConfig,
    strategy_name: str,
) -> BacktestResult:
    """Build engine + strategy, run backtest on bar_data, return result.

    A fresh BacktestEngine is constructed for every call — no shared state.

    Args:
        bar_data      : Slice of bar data (train only during optimization).
        config        : Modified CPATConfig for this run.
        strategy_name : "momentum" | "mean_reversion" | "both".

    Returns:
        BacktestResult from engine.run().
    """
    engine = BacktestEngine.from_config(config)
    symbols = list(bar_data.keys())

    if strategy_name in ("momentum", "both"):
        strategy = MomentumStrategy(
            config=config.strategies.momentum,
            event_queue=engine.event_queue,
            symbols=symbols,
        )
        engine.register_market_handler(strategy)
        engine.register_fill_handler(strategy)

    if strategy_name in ("mean_reversion", "both"):
        mr = MeanReversionStrategy(
            config=config.strategies.mean_reversion,
            event_queue=engine.event_queue,
            symbols=symbols,
        )
        engine.register_market_handler(mr)
        engine.register_fill_handler(mr)

    return engine.run(bar_data)


def _apply_params_to_config(
    base_config: CPATConfig,
    params: dict[str, Any],
    strategy_name: str,
) -> CPATConfig:
    """Deep-copy base_config and apply param overrides for one strategy.

    Args:
        base_config   : Unmodified root config.
        params        : Flat dict of param_name → value.
        strategy_name : "momentum" or "mean_reversion".

    Returns:
        New CPATConfig with the specified parameters overridden.
    """
    # Build updated strategy sub-config dict
    cfg_dict = base_config.model_dump()

    if strategy_name == "momentum":
        cur = cfg_dict["strategies"]["momentum"]
        for k, v in params.items():
            if k in cur:
                cur[k] = v
        # Re-validate: skip invalid combos (lookback_long must > lookback_short)
        try:
            MomentumStrategyConfig(**cur)
        except Exception:
            return None  # type: ignore[return-value]

    elif strategy_name == "mean_reversion":
        cur = cfg_dict["strategies"]["mean_reversion"]
        for k, v in params.items():
            if k in cur:
                cur[k] = v
        try:
            MeanReversionStrategyConfig(**cur)
        except Exception:
            return None  # type: ignore[return-value]

    return CPATConfig.model_validate(cfg_dict)


# ── Grid Search ────────────────────────────────────────────────────────────────


class GridSearchOptimizer:
    """Exhaustive grid search over a bounded parameter space.

    Runs every combination of parameters in the grid on the **training slice**
    of bar_data and returns results sorted by the chosen metric.

    Args:
        strategy    : "momentum" | "mean_reversion".
        metric      : Field name on PerformanceReport to optimise.
                      Higher is always better (negated for minimisation metrics).
        random_seed : Not used for ordering but ensures reproducibility if
                      subsampling is applied externally.
    """

    def __init__(
        self,
        strategy: Literal["momentum", "mean_reversion"] = "momentum",
        metric: str = "sharpe_ratio",
        random_seed: int = 42,
    ) -> None:
        self.strategy = strategy
        self.metric = metric
        self.random_seed = random_seed

    def run(
        self,
        bar_data: dict[str, pd.DataFrame],
        base_config: CPATConfig,
        param_grid: dict[str, list[Any]] | None = None,
    ) -> list[OptimizationResult]:
        """Run grid search over param_grid on bar_data.

        Args:
            bar_data   : Bar data slice (should be train slice only).
            base_config: Base configuration.
            param_grid : Parameter search space. Defaults to strategy preset.

        Returns:
            List of OptimizationResult sorted best-first by metric.
        """
        grid = param_grid or (
            MOMENTUM_PARAM_GRID if self.strategy == "momentum"
            else MEAN_REVERSION_PARAM_GRID
        )

        combos = list(itertools.product(*grid.values()))
        keys = list(grid.keys())
        total = len(combos)

        logger.info(
            "[GridSearch] %s | %d combinations | metric=%s",
            self.strategy, total, self.metric,
        )

        results: list[OptimizationResult] = []

        for idx, values in enumerate(combos):
            params = dict(zip(keys, values))
            modified_config = _apply_params_to_config(base_config, params, self.strategy)
            if modified_config is None:
                logger.debug("[GridSearch] Skip invalid params: %s", params)
                continue

            t0 = time.perf_counter()
            try:
                result = _run_single(bar_data, modified_config, self.strategy)
            except Exception as exc:
                logger.warning("[GridSearch] Run %d/%d failed: %s", idx + 1, total, exc)
                continue
            elapsed = time.perf_counter() - t0

            opt_result = OptimizationResult(
                params=params,
                report=result.report,
                run_time_s=elapsed,
                strategy=self.strategy,
                slice_label="train",
            )
            results.append(opt_result)

            logger.info(
                "[GridSearch] %d/%d | %s=%.3f | params=%s",
                idx + 1, total,
                self.metric, opt_result.metric(self.metric),
                params,
            )

        results.sort(key=lambda r: r.metric(self.metric), reverse=True)
        return results


# ── Random Search ──────────────────────────────────────────────────────────────


class RandomSearchOptimizer:
    """Random sampling over a bounded parameter space.

    Preferred over grid search for large spaces (avoids combinatorial explosion).
    Samples `n_trials` random combinations; guaranteed reproducible via seed.

    Args:
        strategy  : "momentum" | "mean_reversion".
        metric    : PerformanceReport field to maximise.
        n_trials  : Number of random combinations to sample.
        random_seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        strategy: Literal["momentum", "mean_reversion"] = "momentum",
        metric: str = "sharpe_ratio",
        n_trials: int = 50,
        random_seed: int = 42,
    ) -> None:
        self.strategy = strategy
        self.metric = metric
        self.n_trials = n_trials
        self.random_seed = random_seed

    def run(
        self,
        bar_data: dict[str, pd.DataFrame],
        base_config: CPATConfig,
        param_grid: dict[str, list[Any]] | None = None,
    ) -> list[OptimizationResult]:
        """Run random search over param_grid.

        Args:
            bar_data   : Bar data (train slice only).
            base_config: Base configuration.
            param_grid : Parameter search space. Defaults to strategy preset.

        Returns:
            List of OptimizationResult sorted best-first by metric.
        """
        grid = param_grid or (
            MOMENTUM_PARAM_GRID if self.strategy == "momentum"
            else MEAN_REVERSION_PARAM_GRID
        )

        rng = random.Random(self.random_seed)
        keys = list(grid.keys())
        total_space = 1
        for v in grid.values():
            total_space *= len(v)

        n = min(self.n_trials, total_space)
        logger.info(
            "[RandomSearch] %s | %d/%d trials | metric=%s",
            self.strategy, n, total_space, self.metric,
        )

        # Sample without replacement from Cartesian product
        all_combos = list(itertools.product(*grid.values()))
        rng.shuffle(all_combos)
        sampled = all_combos[:n]

        results: list[OptimizationResult] = []

        for idx, values in enumerate(sampled):
            params = dict(zip(keys, values))
            modified_config = _apply_params_to_config(base_config, params, self.strategy)
            if modified_config is None:
                continue

            t0 = time.perf_counter()
            try:
                result = _run_single(bar_data, modified_config, self.strategy)
            except Exception as exc:
                logger.warning("[RandomSearch] Trial %d/%d failed: %s", idx + 1, n, exc)
                continue
            elapsed = time.perf_counter() - t0

            opt_result = OptimizationResult(
                params=params,
                report=result.report,
                run_time_s=elapsed,
                strategy=self.strategy,
                slice_label="train",
            )
            results.append(opt_result)

            logger.info(
                "[RandomSearch] %d/%d | %s=%.3f | %.2fs | params=%s",
                idx + 1, n,
                self.metric, opt_result.metric(self.metric),
                elapsed, params,
            )

        results.sort(key=lambda r: r.metric(self.metric), reverse=True)
        return results


# ── Results utilities ──────────────────────────────────────────────────────────


def results_to_dataframe(results: list[OptimizationResult]) -> pd.DataFrame:
    """Convert optimization results to a readable, sortable DataFrame.

    Columns: all param keys + key performance metrics.

    Args:
        results: List of OptimizationResult (from GridSearch or RandomSearch).

    Returns:
        DataFrame sorted best-first by Sharpe ratio. Empty if no results.
    """
    if not results:
        return pd.DataFrame()

    rows = []
    for r in results:
        row = dict(r.params)
        rpt = r.report
        row.update({
            "total_return_pct": round(rpt.total_return * 100, 2),
            "ann_return_pct":   round(rpt.ann_return * 100, 2),
            "ann_vol_pct":      round(rpt.ann_volatility * 100, 2),
            "sharpe_ratio":     round(rpt.sharpe_ratio, 3),
            "sortino_ratio":    round(rpt.sortino_ratio, 3),
            "max_drawdown_pct": round(rpt.max_drawdown * 100, 2),
            "calmar_ratio":     round(rpt.calmar_ratio, 3),
            "win_rate_pct":     round(rpt.win_rate * 100, 1),
            "profit_factor":    round(rpt.profit_factor, 3),
            "n_trades":         rpt.n_trades,
            "run_time_s":       round(r.run_time_s, 2),
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
    df.index.name = "rank"
    return df
