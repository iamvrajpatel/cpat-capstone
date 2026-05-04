"""
CPAT — Walk-Forward Validation
================================
Implements rolling walk-forward validation — the gold standard for evaluating
strategy robustness without look-ahead bias at the evaluation level.

Walk-forward process per fold:
    1. Train on window [t_0 → t_n] (fixed or expanding)
    2. Optionally: optimize parameters on train window
    3. Test on window [t_n+1 → t_n+k] with best params from step 2
    4. Record both IS (in-sample) and OOS (out-of-sample) metrics
    5. Advance window by step_bars
    6. Stitch all OOS equity segments → combined OOS equity curve

The combined OOS curve is the most honest single-number summary of
strategy performance — it has never been seen during parameter selection.

Usage::

    from cpat.validation.walk_forward import WalkForwardValidator, WalkForwardConfig

    wf = WalkForwardValidator(
        strategy="momentum",
        config=WalkForwardConfig(train_bars=756, test_bars=252, step_bars=63),
    )
    result = wf.run(bar_data, base_config)
    print(result.oos_report)
    print(f"Degradation ratio: {result.degradation_ratio:.2f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import pandas as pd

from cpat.analytics.performance import PerformanceReport, PerformanceTracker
from cpat.backtest.engine import BacktestResult
from cpat.config.loader import CPATConfig
from cpat.optimization.optimizer import (
    MEAN_REVERSION_PARAM_GRID,
    MOMENTUM_PARAM_GRID,
    OptimizationResult,
    RandomSearchOptimizer,
    _apply_params_to_config,
    _run_single,
    results_to_dataframe,
)
from cpat.validation.splitter import _slice_bar_data, _union_timestamps

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────────


@dataclass
class WalkForwardConfig:
    """Configuration for the walk-forward validation harness.

    Attributes:
        train_bars         : Bars in each training window (default 756 ≈ 3yr daily).
        test_bars          : Bars in each out-of-sample test window (default 252 ≈ 1yr).
        step_bars          : Bars to advance between folds (default 63 ≈ 1 quarter).
        optimize_on_fold   : If True, run RandomSearch on each fold's train window.
                             Set False for speed — uses base_config params throughout.
        n_opt_trials       : Random search trials per fold (if optimize_on_fold=True).
        opt_metric         : Metric to optimise within each fold.
        random_seed        : RNG seed for reproducibility.
    """

    train_bars: int = 756
    test_bars: int = 252
    step_bars: int = 63
    optimize_on_fold: bool = False       # default False for speed in tests
    n_opt_trials: int = 20
    opt_metric: str = "sharpe_ratio"
    random_seed: int = 42


# ── Result data classes ────────────────────────────────────────────────────────


@dataclass
class WalkForwardFold:
    """Results for a single walk-forward fold.

    Attributes:
        fold_id      : Zero-indexed fold number.
        train_start  : Start timestamp of training window.
        train_end    : End timestamp of training window.
        test_start   : Start timestamp of OOS test window.
        test_end     : End timestamp of OOS test window.
        best_params  : Parameters selected from train optimization (or base).
        is_report    : PerformanceReport on training data (in-sample).
        oos_report   : PerformanceReport on test data (out-of-sample).
        oos_equity   : Equity curve from the OOS test window.
    """

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict[str, Any]
    is_report: PerformanceReport
    oos_report: PerformanceReport
    oos_equity: pd.Series

    @property
    def degradation(self) -> float:
        """IS Sharpe → OOS Sharpe degradation. 1.0 = no loss, <0 = sign flip."""
        if self.is_report.sharpe_ratio == 0:
            return 0.0
        return self.oos_report.sharpe_ratio / self.is_report.sharpe_ratio

    def to_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "train_start": str(self.train_start.date()),
            "train_end": str(self.train_end.date()),
            "test_start": str(self.test_start.date()),
            "test_end": str(self.test_end.date()),
            "is_sharpe": round(self.is_report.sharpe_ratio, 3),
            "oos_sharpe": round(self.oos_report.sharpe_ratio, 3),
            "is_return_pct": round(self.is_report.total_return * 100, 2),
            "oos_return_pct": round(self.oos_report.total_return * 100, 2),
            "is_max_dd_pct": round(self.is_report.max_drawdown * 100, 2),
            "oos_max_dd_pct": round(self.oos_report.max_drawdown * 100, 2),
            "degradation_ratio": round(self.degradation, 3),
            "best_params": self.best_params,
        }


@dataclass
class WalkForwardResult:
    """Aggregated results across all walk-forward folds.

    Attributes:
        folds            : Per-fold results.
        combined_oos_equity : Stitched OOS equity curve across all folds.
        oos_report       : PerformanceReport computed on combined OOS equity.
        mean_is_sharpe   : Average in-sample Sharpe across folds.
        mean_oos_sharpe  : Average out-of-sample Sharpe across folds.
        consistency_score: Fraction of OOS folds with positive total return.
        degradation_ratio: mean_oos_sharpe / mean_is_sharpe.
    """

    folds: list[WalkForwardFold]
    combined_oos_equity: pd.Series
    oos_report: PerformanceReport
    mean_is_sharpe: float
    mean_oos_sharpe: float
    consistency_score: float
    degradation_ratio: float

    def summary_dataframe(self) -> pd.DataFrame:
        """Return per-fold summary as a DataFrame."""
        rows = [f.to_dict() for f in self.folds]
        return pd.DataFrame(rows).set_index("fold_id")

    def __str__(self) -> str:
        lines = [
            f"\n{'='*60}",
            "  Walk-Forward Validation Summary",
            f"{'='*60}",
            f"  Folds:              {len(self.folds)}",
            f"  Avg IS  Sharpe:     {self.mean_is_sharpe:>+.3f}",
            f"  Avg OOS Sharpe:     {self.mean_oos_sharpe:>+.3f}",
            f"  Degradation Ratio:  {self.degradation_ratio:>6.3f}  "
            f"{'✓ Robust' if self.degradation_ratio >= 0.5 else '⚠ Overfitted'}",
            f"  Consistency:        {self.consistency_score:.1%} folds profitable",
            f"  OOS Total Return:   {self.oos_report.total_return:>+.2%}",
            f"  OOS Max Drawdown:   {self.oos_report.max_drawdown:>+.2%}",
            f"{'='*60}",
        ]
        return "\n".join(lines)


# ── Validator ──────────────────────────────────────────────────────────────────


class WalkForwardValidator:
    """Rolls a train/test window across the full timeline.

    At each fold:
        - If `config.optimize_on_fold=True`, RandomSearch is run on train data.
        - Otherwise, base_config parameters are used as-is.
        - IS backtest and OOS backtest are run with the selected params.
        - OOS equity segments are stitched into a single combined curve.

    Args:
        strategy : "momentum" | "mean_reversion".
        config   : Walk-forward configuration.
    """

    def __init__(
        self,
        strategy: Literal["momentum", "mean_reversion"] = "momentum",
        wf_config: Optional[WalkForwardConfig] = None,
    ) -> None:
        self.strategy = strategy
        self.wf_config = wf_config or WalkForwardConfig()

    def run(
        self,
        bar_data: dict[str, pd.DataFrame],
        base_config: CPATConfig,
        param_grid: dict[str, list] | None = None,
    ) -> WalkForwardResult:
        """Execute the full walk-forward validation.

        Args:
            bar_data   : Full bar data {symbol: DataFrame}.
            base_config: Base configuration (used unless optimize_on_fold=True).
            param_grid : Custom parameter grid (optional).

        Returns:
            WalkForwardResult with per-fold breakdown and combined OOS metrics.
        """
        timestamps = _union_timestamps(bar_data)
        n = len(timestamps)

        cfg = self.wf_config
        train_n = cfg.train_bars
        test_n = cfg.test_bars
        step_n = cfg.step_bars

        if n < train_n + test_n:
            raise ValueError(
                f"Insufficient data ({n} bars) for WF with "
                f"train={train_n} + test={test_n}."
            )

        grid = param_grid or (
            MOMENTUM_PARAM_GRID if self.strategy == "momentum"
            else MEAN_REVERSION_PARAM_GRID
        )

        folds: list[WalkForwardFold] = []
        fold_id = 0
        start_idx = 0

        while True:
            train_end_idx = start_idx + train_n - 1
            test_start_idx = train_end_idx + 1
            test_end_idx = test_start_idx + test_n - 1

            if test_end_idx >= n:
                break

            train_end = timestamps[train_end_idx]
            test_start = timestamps[test_start_idx]
            test_end = timestamps[test_end_idx]
            train_start = timestamps[start_idx]

            train_data = _slice_bar_data(bar_data, start=train_start, end=train_end)
            test_data = _slice_bar_data(bar_data, start=test_start, end=test_end)

            if not train_data or not test_data:
                start_idx += step_n
                continue

            logger.info(
                "[WF] Fold %d | train=[%s→%s] | test=[%s→%s]",
                fold_id, train_start.date(), train_end.date(),
                test_start.date(), test_end.date(),
            )

            # ── Step 1: Select parameters ──────────────────────────────────────
            if cfg.optimize_on_fold:
                optimizer = RandomSearchOptimizer(
                    strategy=self.strategy,
                    metric=cfg.opt_metric,
                    n_trials=cfg.n_opt_trials,
                    random_seed=cfg.random_seed + fold_id,
                )
                opt_results = optimizer.run(train_data, base_config, grid)
                best_params = opt_results[0].params if opt_results else {}
                fold_config = _apply_params_to_config(base_config, best_params, self.strategy)
                if fold_config is None:
                    fold_config = base_config
                    best_params = {}
            else:
                fold_config = base_config
                best_params = {}

            # ── Step 2: IS backtest (train window) ─────────────────────────────
            try:
                is_result = _run_single(train_data, fold_config, self.strategy)
                is_report = is_result.report
            except Exception as exc:
                logger.warning("[WF] Fold %d IS failed: %s", fold_id, exc)
                start_idx += step_n
                fold_id += 1
                continue

            # ── Step 3: OOS backtest (test window) ────────────────────────────
            try:
                oos_result = _run_single(test_data, fold_config, self.strategy)
                oos_report = oos_result.report
                oos_equity = oos_result.equity_curve
            except Exception as exc:
                logger.warning("[WF] Fold %d OOS failed: %s", fold_id, exc)
                start_idx += step_n
                fold_id += 1
                continue

            fold = WalkForwardFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best_params,
                is_report=is_report,
                oos_report=oos_report,
                oos_equity=oos_equity,
            )
            folds.append(fold)

            start_idx += step_n
            fold_id += 1

        if not folds:
            raise RuntimeError("Walk-forward produced zero folds. Check data length and config.")

        # ── Step 4: Stitch OOS equity curves ─────────────────────────────────
        combined_oos_equity = _stitch_equity_curves([f.oos_equity for f in folds])

        # ── Step 5: Compute aggregate metrics ────────────────────────────────
        oos_tracker = PerformanceTracker(
            initial_capital=base_config.backtest.initial_capital,
            risk_free_rate=getattr(base_config.risk, "risk_free_rate", 0.04),
        )
        for ts, val in combined_oos_equity.items():
            oos_tracker.record(ts, val)
        oos_report = oos_tracker.compute()

        mean_is = float(sum(f.is_report.sharpe_ratio for f in folds) / len(folds))
        mean_oos = float(sum(f.oos_report.sharpe_ratio for f in folds) / len(folds))
        consistency = float(
            sum(1 for f in folds if f.oos_report.total_return > 0) / len(folds)
        )
        degradation = mean_oos / mean_is if abs(mean_is) > 1e-6 else 0.0

        return WalkForwardResult(
            folds=folds,
            combined_oos_equity=combined_oos_equity,
            oos_report=oos_report,
            mean_is_sharpe=mean_is,
            mean_oos_sharpe=mean_oos,
            consistency_score=consistency,
            degradation_ratio=degradation,
        )


# ── Helper ─────────────────────────────────────────────────────────────────────


def _stitch_equity_curves(curves: list[pd.Series]) -> pd.Series:
    """Concatenate OOS equity curves, normalising each to start where previous ended.

    Each fold's equity is rebased so the combined curve is continuous.
    This correctly represents compounded wealth from stitched OOS periods.

    Args:
        curves: List of equity Series from each OOS fold.

    Returns:
        Combined equity Series with a continuous DatetimeIndex.
    """
    if not curves:
        return pd.Series(dtype=float, name="equity")

    segments = []
    running_equity = curves[0].iloc[0] if not curves[0].empty else 1.0

    for curve in curves:
        if curve.empty:
            continue
        # Rebase: scale so first value = previous ending equity
        first_val = curve.iloc[0]
        if first_val == 0:
            continue
        scale = running_equity / first_val
        rebased = curve * scale
        segments.append(rebased)
        running_equity = float(rebased.iloc[-1])

    if not segments:
        return pd.Series(dtype=float, name="equity")

    combined = pd.concat(segments)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    combined.name = "equity"
    return combined
