"""
CPAT — Return Distribution Analysis
=====================================
Characterises the statistical properties of the strategy's return series.
Used to assess whether returns are normal, fat-tailed, or skewed — all of
which affect the validity of Sharpe ratio and similar normal-assumption metrics.

Key metrics:
    mean        : Arithmetic mean of periodic returns
    std         : Standard deviation of returns
    skewness    : Third standardised moment; negative = left-tail dominance
    kurtosis    : Excess kurtosis; > 0 = heavier tails than Normal
    var_95      : Value at Risk at 95% confidence (negative = loss)
    cvar_95     : Conditional VaR / Expected Shortfall (average of worst 5%)
    jarque_bera : p-value for normality; < 0.05 means reject normality

Usage::

    from cpat.analytics.distributions import compute_distribution, monthly_returns_table

    dist = compute_distribution(returns_series)
    print(dist.skewness, dist.kurtosis)

    table = monthly_returns_table(equity_curve)  # months × years
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ── Data class ─────────────────────────────────────────────────────────────────


@dataclass
class ReturnDistribution:
    """Statistical characterisation of a return series.

    Attributes:
        mean            : Arithmetic mean of periodic returns.
        std             : Standard deviation of returns.
        skewness        : Third standardised moment.
                          Negative = left-skewed (bad: large losses).
                          Positive = right-skewed (good: large gains).
        kurtosis        : Excess kurtosis (0 = Normal, >0 = fat tails).
        var_95          : 5th percentile of returns (95% VaR, negative = loss).
        cvar_95         : Mean of all returns below var_95 (CVaR / ES).
        tail_ratio      : abs(95th pct) / abs(5th pct). >1 means right-skewed tails.
        jarque_bera_stat: J-B test statistic.
        jarque_bera_pval: p-value; < 0.05 rejects normality at 5% level.
        n_obs           : Number of observations.
    """

    mean: float
    std: float
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    tail_ratio: float
    jarque_bera_stat: float
    jarque_bera_pval: float
    n_obs: int

    @property
    def is_normal(self) -> bool:
        """True if returns are consistent with normality at 5% significance."""
        return self.jarque_bera_pval >= 0.05

    @property
    def is_negatively_skewed(self) -> bool:
        """True if the left tail dominates (typical for most equity strategies)."""
        return self.skewness < -0.5

    def to_dict(self) -> dict:
        return {
            "mean_daily_pct": round(self.mean * 100, 5),
            "std_daily_pct": round(self.std * 100, 5),
            "skewness": round(self.skewness, 4),
            "kurtosis_excess": round(self.kurtosis, 4),
            "var_95_pct": round(self.var_95 * 100, 4),
            "cvar_95_pct": round(self.cvar_95 * 100, 4),
            "tail_ratio": round(self.tail_ratio, 4),
            "jarque_bera_stat": round(self.jarque_bera_stat, 4),
            "jarque_bera_pval": round(self.jarque_bera_pval, 6),
            "is_normal": self.is_normal,
            "n_obs": self.n_obs,
        }

    def __str__(self) -> str:
        normality = "✓ Normal" if self.is_normal else "✗ Non-Normal"
        return (
            f"\n{'='*50}\n"
            f"  Return Distribution ({self.n_obs} observations)\n"
            f"{'='*50}\n"
            f"  Mean (daily):     {self.mean:+.4%}\n"
            f"  Std  (daily):     {self.std:.4%}\n"
            f"  Skewness:         {self.skewness:+.3f}  "
            f"{'← left tail heavy ⚠' if self.skewness < -0.5 else ''}\n"
            f"  Kurtosis (excess):{self.kurtosis:+.3f}  "
            f"{'← fat tails ⚠' if self.kurtosis > 1 else ''}\n"
            f"  VaR (95%):        {self.var_95:+.3%}\n"
            f"  CVaR (95%):       {self.cvar_95:+.3%}\n"
            f"  Tail Ratio:       {self.tail_ratio:.3f}\n"
            f"  Normality:        {normality} (JB p={self.jarque_bera_pval:.4f})\n"
            f"{'='*50}\n"
        )


# ── Core functions ─────────────────────────────────────────────────────────────


def compute_distribution(returns: pd.Series) -> ReturnDistribution:
    """Compute the full statistical distribution profile of a return series.

    Args:
        returns: Periodic returns (e.g. daily pct_change). Must not contain NaN.

    Returns:
        ReturnDistribution dataclass with all metrics populated.
    """
    r = returns.dropna()
    n = len(r)

    if n < 4:
        # Not enough data for meaningful statistics
        return ReturnDistribution(
            mean=float(r.mean()) if n > 0 else 0.0,
            std=float(r.std()) if n > 1 else 0.0,
            skewness=0.0, kurtosis=0.0,
            var_95=0.0, cvar_95=0.0, tail_ratio=1.0,
            jarque_bera_stat=0.0, jarque_bera_pval=1.0,
            n_obs=n,
        )

    arr = r.to_numpy()

    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    skewness = float(scipy_stats.skew(arr, bias=False))
    kurtosis = float(scipy_stats.kurtosis(arr, fisher=True, bias=False))  # excess

    var_95 = float(np.percentile(arr, 5))   # 5th pct = 95% VaR (worst losses)
    cvar_95 = float(arr[arr <= var_95].mean()) if any(arr <= var_95) else var_95

    p95 = float(np.percentile(arr, 95))
    tail_ratio = abs(p95) / max(abs(var_95), 1e-10)

    jb_stat, jb_pval = scipy_stats.jarque_bera(arr)

    return ReturnDistribution(
        mean=mean,
        std=std,
        skewness=skewness,
        kurtosis=kurtosis,
        var_95=var_95,
        cvar_95=cvar_95,
        tail_ratio=tail_ratio,
        jarque_bera_stat=float(jb_stat),
        jarque_bera_pval=float(jb_pval),
        n_obs=n,
    )


def is_normal(returns: pd.Series, alpha: float = 0.05) -> bool:
    """Test whether returns are consistent with a Normal distribution.

    Uses the Jarque-Bera test (tests both skewness and kurtosis jointly).
    Requires at least 8 observations; returns True (assume normal) otherwise.

    Args:
        returns: Return series.
        alpha  : Significance level (default 5%).

    Returns:
        True if we cannot reject normality at the given significance level.
    """
    r = returns.dropna()
    if len(r) < 8:
        return True
    _, pval = scipy_stats.jarque_bera(r.to_numpy())
    return float(pval) >= alpha


def monthly_returns_table(equity_curve: pd.Series) -> pd.DataFrame:
    """Build a months × years return table from an equity curve.

    Standard in quant research — immediately reveals seasonal patterns
    and consistency of returns across calendar periods.

    Args:
        equity_curve: Equity values with DatetimeIndex (UTC).

    Returns:
        DataFrame indexed by month (1–12), columns are years.
        Values are monthly returns in percent, rounded to 2dp.
        Missing months are NaN.
    """
    if equity_curve.empty:
        return pd.DataFrame()

    # Resample to end-of-month values
    monthly = equity_curve.resample("ME").last().dropna()

    if len(monthly) < 2:
        return pd.DataFrame()

    # Monthly returns
    monthly_returns = monthly.pct_change().dropna()

    df = pd.DataFrame({
        "year": monthly_returns.index.year,
        "month": monthly_returns.index.month,
        "return_pct": (monthly_returns.values * 100).round(2),
    })

    # Pivot: rows = month, columns = year
    table = df.pivot(index="month", columns="year", values="return_pct")
    table.index.name = "Month"
    table.columns.name = "Year"

    # Add annual total row
    annual = (equity_curve.resample("YE").last().pct_change().dropna() * 100).round(2)
    annual_row = pd.Series({yr: ret for yr, ret in zip(annual.index.year, annual.values)},
                            name="Annual")
    table = pd.concat([table, annual_row.to_frame().T])

    return table


def return_heatmap_data(equity_curve: pd.Series) -> dict:
    """Return data suitable for building a return heatmap visualisation.

    Args:
        equity_curve: Equity values with DatetimeIndex.

    Returns:
        Dict with keys 'table' (monthly returns DataFrame), 'distribution'
        (ReturnDistribution), and 'summary' statistics.
    """
    returns = equity_curve.pct_change().dropna()
    dist = compute_distribution(returns)
    table = monthly_returns_table(equity_curve)

    return {
        "monthly_table": table,
        "distribution": dist,
        "positive_months_pct": round(
            100 * (returns > 0).mean(), 1
        ) if len(returns) > 0 else 0.0,
        "best_month_pct": round(float(returns.max() * 100), 2),
        "worst_month_pct": round(float(returns.min() * 100), 2),
    }
