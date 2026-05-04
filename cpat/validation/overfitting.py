"""
CPAT — Overfitting Detection & Parameter Sensitivity Analysis
==============================================================
Tools for detecting curve-fitting and evaluating whether strategy
performance is robust to small parameter perturbations.

Key functions:
    degradation_report     : Structured IS vs OOS comparison.
    compute_sensitivity    : CV of a metric across parameter values.
    stability_check        : Flag parameters where small changes cause large performance swings.

Academic context:
    A well-generalising strategy should have:
        degradation_ratio ≥ 0.5  (OOS Sharpe ≥ 50% of IS Sharpe)
        parameter_sensitivity_cv < 0.30  (stable across params)

Usage::

    from cpat.validation.overfitting import degradation_report, compute_sensitivity

    report = degradation_report(is_report, oos_report)
    sensitivity = compute_sensitivity(results_df, "lookback_long", "sharpe_ratio")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cpat.analytics.performance import PerformanceReport


# ── Degradation analysis ───────────────────────────────────────────────────────


def degradation_report(
    is_report: PerformanceReport,
    oos_report: PerformanceReport,
    label: str = "full_period",
) -> dict:
    """Compute a structured comparison between IS and OOS performance.

    Args:
        is_report : In-sample PerformanceReport.
        oos_report: Out-of-sample PerformanceReport.
        label     : Identifier for this comparison (e.g. fold ID or "full").

    Returns:
        Dict with per-metric IS/OOS values, degradation ratios,
        and an overall assessment string.
    """
    def _ratio(oos_val: float, is_val: float) -> float:
        if abs(is_val) < 1e-9:
            return 0.0
        return oos_val / is_val

    sharpe_deg = _ratio(oos_report.sharpe_ratio, is_report.sharpe_ratio)
    return_deg = _ratio(oos_report.total_return, is_report.total_return)
    dd_worsening = oos_report.max_drawdown - is_report.max_drawdown  # negative = OOS worse

    if sharpe_deg >= 0.7:
        assessment = "ROBUST — strong OOS generalisation"
    elif sharpe_deg >= 0.5:
        assessment = "ACCEPTABLE — moderate degradation (common in practice)"
    elif sharpe_deg >= 0.25:
        assessment = "MARGINAL — significant degradation, review parameters"
    elif sharpe_deg >= 0:
        assessment = "WEAK — OOS barely positive, likely over-optimised"
    else:
        assessment = "FAILED — OOS Sharpe negative, strategy does not generalise"

    return {
        "label": label,
        "assessment": assessment,
        "is": {
            "total_return_pct": round(is_report.total_return * 100, 2),
            "ann_return_pct":   round(is_report.ann_return * 100, 2),
            "sharpe_ratio":     round(is_report.sharpe_ratio, 3),
            "sortino_ratio":    round(is_report.sortino_ratio, 3),
            "max_drawdown_pct": round(is_report.max_drawdown * 100, 2),
            "calmar_ratio":     round(is_report.calmar_ratio, 3),
            "win_rate_pct":     round(is_report.win_rate * 100, 1),
            "profit_factor":    round(is_report.profit_factor, 3),
            "n_trades":         is_report.n_trades,
        },
        "oos": {
            "total_return_pct": round(oos_report.total_return * 100, 2),
            "ann_return_pct":   round(oos_report.ann_return * 100, 2),
            "sharpe_ratio":     round(oos_report.sharpe_ratio, 3),
            "sortino_ratio":    round(oos_report.sortino_ratio, 3),
            "max_drawdown_pct": round(oos_report.max_drawdown * 100, 2),
            "calmar_ratio":     round(oos_report.calmar_ratio, 3),
            "win_rate_pct":     round(oos_report.win_rate * 100, 1),
            "profit_factor":    round(oos_report.profit_factor, 3),
            "n_trades":         oos_report.n_trades,
        },
        "degradation": {
            "sharpe_ratio":     round(sharpe_deg, 3),
            "total_return":     round(return_deg, 3),
            "dd_worsening_pct": round(dd_worsening * 100, 2),
        },
    }


# ── Parameter sensitivity ──────────────────────────────────────────────────────


@dataclass
class ParameterSensitivity:
    """Sensitivity of a performance metric to one parameter's values.

    Attributes:
        param_name      : Name of the parameter being tested.
        metric_name     : Name of the PerformanceReport metric.
        values          : Parameter values tested.
        metric_values   : Corresponding metric values.
        mean_metric     : Mean of metric across all param values.
        std_metric      : Std of metric across all param values.
        cv              : Coefficient of variation = std/abs(mean).
                          Low CV (< 0.30) = stable. High CV = fragile.
        is_stable       : True if CV < 0.30.
    """

    param_name: str
    metric_name: str
    values: list
    metric_values: list[float]
    mean_metric: float
    std_metric: float
    cv: float
    is_stable: bool

    def to_dict(self) -> dict:
        return {
            "param_name": self.param_name,
            "metric_name": self.metric_name,
            "mean": round(self.mean_metric, 4),
            "std": round(self.std_metric, 4),
            "cv": round(self.cv, 4),
            "is_stable": self.is_stable,
            "values": self.values,
            "metric_values": [round(v, 4) for v in self.metric_values],
        }


def compute_sensitivity(
    results_df: pd.DataFrame,
    param_name: str,
    metric: str = "sharpe_ratio",
    cv_threshold: float = 0.30,
) -> ParameterSensitivity:
    """Compute metric sensitivity to a single parameter.

    Groups results by param value and aggregates metric; computes CV.

    Args:
        results_df  : DataFrame from results_to_dataframe().
        param_name  : Column name of the parameter to analyse.
        metric      : Column name of the metric to analyse.
        cv_threshold: CV above which the parameter is flagged as unstable.

    Returns:
        ParameterSensitivity dataclass.

    Raises:
        KeyError: If param_name or metric is not in results_df.
    """
    if param_name not in results_df.columns:
        raise KeyError(f"Parameter '{param_name}' not in results DataFrame.")
    if metric not in results_df.columns:
        raise KeyError(f"Metric '{metric}' not in results DataFrame.")

    grouped = results_df.groupby(param_name)[metric].mean()
    param_values = list(grouped.index)
    metric_values = list(grouped.values.astype(float))

    mean_val = float(np.mean(metric_values))
    std_val = float(np.std(metric_values, ddof=1)) if len(metric_values) > 1 else 0.0
    cv = std_val / max(abs(mean_val), 1e-9)

    return ParameterSensitivity(
        param_name=param_name,
        metric_name=metric,
        values=param_values,
        metric_values=metric_values,
        mean_metric=mean_val,
        std_metric=std_val,
        cv=cv,
        is_stable=cv < cv_threshold,
    )


def stability_check(
    results_df: pd.DataFrame,
    metric: str = "sharpe_ratio",
    cv_threshold: float = 0.30,
) -> pd.DataFrame:
    """Run sensitivity analysis on ALL parameters in the results DataFrame.

    Args:
        results_df  : DataFrame from results_to_dataframe().
        metric      : Metric column to analyse.
        cv_threshold: CV threshold for stability flag.

    Returns:
        DataFrame with one row per parameter, sorted by CV (most unstable first).
    """
    # Identify parameter columns (non-metric, non-meta columns)
    metric_cols = {
        "total_return_pct", "ann_return_pct", "ann_vol_pct", "sharpe_ratio",
        "sortino_ratio", "max_drawdown_pct", "calmar_ratio", "win_rate_pct",
        "profit_factor", "n_trades", "run_time_s",
    }
    param_cols = [c for c in results_df.columns if c not in metric_cols]

    rows = []
    for param in param_cols:
        try:
            sens = compute_sensitivity(results_df, param, metric, cv_threshold)
            rows.append(sens.to_dict())
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)[["param_name", "mean", "std", "cv", "is_stable"]]
    return df.sort_values("cv", ascending=False).reset_index(drop=True)
