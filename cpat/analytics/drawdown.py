"""
CPAT — Drawdown Analysis
=========================
Detailed drawdown decomposition: identifies every discrete drawdown period,
computes depth / duration / recovery, and stores the full drawdown curve.

A drawdown *period* begins when equity falls below its prior peak and ends
when equity recovers to a new all-time-high (or when the series terminates).

Key definitions:
    depth           : (peak - trough) / peak — always negative
    duration_bars   : number of bars from peak to recovery (or series end)
    underwater_bars : number of bars from peak to trough
    recovery_bars   : duration_bars - underwater_bars

Usage::

    from cpat.analytics.drawdown import compute_drawdown_table, max_drawdown_details

    dd_table = compute_drawdown_table(equity_curve)
    worst    = max_drawdown_details(equity_curve)
    print(worst.depth, worst.duration_bars)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ── Data classes ───────────────────────────────────────────────────────────────


@dataclass
class DrawdownPeriod:
    """One discrete drawdown episode.

    Attributes:
        peak_date     : Date/time when the prior all-time-high was set.
        trough_date   : Date/time of maximum loss within this period.
        recovery_date : Date/time when equity recovers to peak (None if open).
        depth         : Fractional loss from peak to trough (negative, e.g. -0.15).
        underwater_bars : Bars from peak to trough.
        recovery_bars   : Bars from trough to recovery (0 if not yet recovered).
        duration_bars   : Total bars from peak to recovery (or series end).
    """

    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    recovery_date: Optional[pd.Timestamp]
    depth: float
    underwater_bars: int
    recovery_bars: int
    duration_bars: int

    @property
    def is_recovered(self) -> bool:
        """True if the equity curve has returned to the prior peak."""
        return self.recovery_date is not None

    def to_dict(self) -> dict:
        return {
            "peak_date": str(self.peak_date.date()),
            "trough_date": str(self.trough_date.date()),
            "recovery_date": str(self.recovery_date.date()) if self.recovery_date else None,
            "depth_pct": round(self.depth * 100, 3),
            "underwater_bars": self.underwater_bars,
            "recovery_bars": self.recovery_bars,
            "duration_bars": self.duration_bars,
            "is_recovered": self.is_recovered,
        }


# ── Core functions ─────────────────────────────────────────────────────────────


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """Compute the rolling drawdown from peak at every bar.

    Args:
        equity_curve: Equity values with DatetimeIndex (UTC).

    Returns:
        Series of drawdown values in decimal (e.g. -0.05 = -5%).
        Zero when at an all-time-high.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return pd.Series(dtype=float, name="drawdown", index=equity_curve.index)

    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak.replace(0.0, float("nan"))
    return dd.fillna(0.0).rename("drawdown")


def compute_drawdown_table(equity_curve: pd.Series) -> list[DrawdownPeriod]:
    """Decompose the equity curve into a list of discrete drawdown periods.

    Algorithm:
        1. Scan forward, tracking the running peak.
        2. When equity < peak, a drawdown starts at the last peak date.
        3. The trough is the bar of maximum loss within the episode.
        4. The episode ends when equity returns to or exceeds the prior peak.

    Args:
        equity_curve: Equity values with DatetimeIndex (UTC).

    Returns:
        List of DrawdownPeriod objects, sorted by start date.
        Empty list if no drawdowns occurred.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return []

    curve = equity_curve.reset_index(drop=False)
    curve.columns = ["ts", "equity"]

    periods: list[DrawdownPeriod] = []

    peak_idx = 0
    peak_val = curve["equity"].iloc[0]
    in_drawdown = False
    trough_idx = 0
    trough_val = peak_val

    for i in range(1, len(curve)):
        eq = curve["equity"].iloc[i]

        if in_drawdown:
            if eq < trough_val:
                trough_idx = i
                trough_val = eq
            if eq >= peak_val:
                # Recovery — close the period
                depth = (trough_val - peak_val) / peak_val
                uw = trough_idx - peak_idx
                rec = i - trough_idx
                periods.append(
                    DrawdownPeriod(
                        peak_date=curve["ts"].iloc[peak_idx],
                        trough_date=curve["ts"].iloc[trough_idx],
                        recovery_date=curve["ts"].iloc[i],
                        depth=depth,
                        underwater_bars=uw,
                        recovery_bars=rec,
                        duration_bars=uw + rec,
                    )
                )
                peak_idx = i
                peak_val = eq
                in_drawdown = False
                trough_idx = i
                trough_val = eq
        else:
            if eq < peak_val:
                in_drawdown = True
                trough_idx = i
                trough_val = eq
            elif eq > peak_val:
                peak_idx = i
                peak_val = eq

    # Handle open drawdown at end of series
    if in_drawdown:
        depth = (trough_val - peak_val) / peak_val
        uw = trough_idx - peak_idx
        dur = (len(curve) - 1) - peak_idx
        periods.append(
            DrawdownPeriod(
                peak_date=curve["ts"].iloc[peak_idx],
                trough_date=curve["ts"].iloc[trough_idx],
                recovery_date=None,
                depth=depth,
                underwater_bars=uw,
                recovery_bars=0,
                duration_bars=dur,
            )
        )

    return sorted(periods, key=lambda p: p.depth)  # worst first


def max_drawdown_details(equity_curve: pd.Series) -> Optional[DrawdownPeriod]:
    """Return the single worst drawdown period by depth.

    Args:
        equity_curve: Equity values with DatetimeIndex.

    Returns:
        The DrawdownPeriod with the largest depth, or None if no drawdowns.
    """
    periods = compute_drawdown_table(equity_curve)
    return periods[0] if periods else None


def drawdown_table_to_dataframe(periods: list[DrawdownPeriod]) -> pd.DataFrame:
    """Convert a list of DrawdownPeriod objects to a readable DataFrame.

    Args:
        periods: List from compute_drawdown_table().

    Returns:
        DataFrame with one row per drawdown period, sorted worst-first.
    """
    if not periods:
        return pd.DataFrame(
            columns=[
                "peak_date", "trough_date", "recovery_date",
                "depth_pct", "underwater_bars", "recovery_bars",
                "duration_bars", "is_recovered",
            ]
        )
    rows = [p.to_dict() for p in periods]
    df = pd.DataFrame(rows)
    df["depth_pct"] = df["depth_pct"].round(3)
    return df


def ulcer_index(equity_curve: pd.Series) -> float:
    """Compute the Ulcer Index — RMS of all drawdown depths.

    A drawdown-pain metric that penalises both depth and duration.
    Lower values = smoother equity curve.

        UI = sqrt(mean(DD_i^2))   where DD_i = drawdown at bar i (%)

    Args:
        equity_curve: Equity values with DatetimeIndex.

    Returns:
        Ulcer Index (non-negative float). 0.0 if no drawdowns.
    """
    dd = compute_drawdown_series(equity_curve)
    if dd.empty:
        return 0.0
    dd_pct = dd * 100.0  # convert to percentage
    return float(math.sqrt((dd_pct ** 2).mean()))
