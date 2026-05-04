"""
CPAT — Time-Series Splitting Utilities
========================================
Forward-only train/test splitting for time-series data.

Critical rule: test data is ALWAYS chronologically after train data.
No shuffling, no random sampling from the full period — that would be
look-ahead bias at the evaluation level.

Usage::

    from cpat.validation.splitter import train_test_split, time_series_cv_folds

    train, test = train_test_split(bar_data, train_ratio=0.70)

    folds = time_series_cv_folds(bar_data, n_folds=5)
    for fold in folds:
        print(fold.train_end, fold.test_end)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


# ── Data class ─────────────────────────────────────────────────────────────────


@dataclass
class TimeSeriesSplit:
    """One train/test fold for time-series cross-validation.

    Attributes:
        fold_id    : Zero-indexed fold number.
        train_start: First timestamp in training window.
        train_end  : Last timestamp in training window (inclusive).
        test_start : First timestamp in test window.
        test_end   : Last timestamp in test window (inclusive).
        n_train    : Number of timestamps in training window.
        n_test     : Number of timestamps in test window.
    """

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int

    def __repr__(self) -> str:
        return (
            f"TimeSeriesSplit(fold={self.fold_id} | "
            f"train=[{self.train_start.date()} → {self.train_end.date()}] n={self.n_train} | "
            f"test=[{self.test_start.date()} → {self.test_end.date()}] n={self.n_test})"
        )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _union_timestamps(bar_data: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Compute the sorted union of all timestamps across all symbols.

    Args:
        bar_data: {symbol: OHLCV DataFrame}.

    Returns:
        Sorted DatetimeIndex of all unique timestamps.
    """
    all_ts: set[pd.Timestamp] = set()
    for df in bar_data.values():
        if not df.empty:
            all_ts.update(df.index.tolist())
    return pd.DatetimeIndex(sorted(all_ts))


def _slice_bar_data(
    bar_data: dict[str, pd.DataFrame],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> dict[str, pd.DataFrame]:
    """Slice bar_data to the [start, end] timestamp range (inclusive).

    Args:
        bar_data: Full bar data dict.
        start   : Inclusive start (None = no lower bound).
        end     : Inclusive end (None = no upper bound).

    Returns:
        New dict with each DataFrame sliced to [start, end].
        Symbols with empty slices are excluded.
    """
    result: dict[str, pd.DataFrame] = {}
    for sym, df in bar_data.items():
        if df.empty:
            continue
        sliced = df.loc[
            (df.index >= start if start is not None else True) &  # type: ignore[operator]
            (df.index <= end if end is not None else True)         # type: ignore[operator]
        ]
        if not sliced.empty:
            result[sym] = sliced
    return result


# ── Core splitting functions ───────────────────────────────────────────────────


def train_test_split(
    bar_data: dict[str, pd.DataFrame],
    train_ratio: float = 0.70,
    warmup_bars: int = 252,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Split bar_data into train and test slices by timestamp.

    The split point is determined by the union of all timestamps:
        split_idx = int(total_timestamps × train_ratio)

    Both splits always share the full symbol universe (symbols not present
    in a given window are dropped from that window's dict).

    Args:
        bar_data    : Full bar data {symbol: DataFrame}.
        train_ratio : Fraction of timeline allocated to training (default 70%).
        warmup_bars : Minimum bars required in train (raises if not met).

    Returns:
        (train_data, test_data) — both forward-only, non-overlapping.

    Raises:
        ValueError: If bar_data is empty, train_ratio invalid, or
                    train window is too short for warmup.
    """
    if not bar_data:
        raise ValueError("bar_data is empty — cannot split.")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    timestamps = _union_timestamps(bar_data)
    n = len(timestamps)
    if n < 4:
        raise ValueError(f"Too few timestamps to split ({n}).")

    split_idx = int(n * train_ratio)
    split_idx = max(warmup_bars + 1, min(split_idx, n - 1))

    train_end = timestamps[split_idx - 1]
    test_start = timestamps[split_idx]

    train_data = _slice_bar_data(bar_data, start=None, end=train_end)
    test_data = _slice_bar_data(bar_data, start=test_start, end=None)

    if len(train_data) == 0 or len(test_data) == 0:
        raise ValueError("Split produced empty train or test set.")

    n_train = sum(len(df) for df in train_data.values()) // max(len(train_data), 1)
    n_test = sum(len(df) for df in test_data.values()) // max(len(test_data), 1)

    if n_train < warmup_bars:
        raise ValueError(
            f"Train window ({n_train} bars) is shorter than warmup_bars ({warmup_bars}). "
            f"Increase train_ratio or reduce warmup_bars."
        )

    return train_data, test_data


def time_series_cv_folds(
    bar_data: dict[str, pd.DataFrame],
    n_folds: int = 5,
    warmup_bars: int = 252,
) -> list[TimeSeriesSplit]:
    """Generate expanding-window cross-validation folds.

    Expanding window means the training set grows with each fold:
        Fold 0: train=[t0 → t_k],   test=[t_k+1 → t_2k]
        Fold 1: train=[t0 → t_2k],  test=[t_2k+1 → t_3k]
        ...

    This maximises training data in later folds and is preferred over
    rolling windows for low-frequency strategies.

    Args:
        bar_data   : Full bar data {symbol: DataFrame}.
        n_folds    : Number of CV folds.
        warmup_bars: Minimum bars in initial training window.

    Returns:
        List of TimeSeriesSplit objects.

    Raises:
        ValueError: If there are insufficient timestamps for the requested folds.
    """
    if not bar_data:
        raise ValueError("bar_data is empty.")
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2.")

    timestamps = _union_timestamps(bar_data)
    n = len(timestamps)

    # Reserve initial warmup + 1 period for first train window
    min_train = warmup_bars + 20
    available = n - min_train
    if available < n_folds * 2:
        raise ValueError(
            f"Insufficient timestamps ({n}) for {n_folds} folds with warmup={warmup_bars}."
        )

    # Each fold's test window is 1/(n_folds+1) of the available range
    fold_size = available // (n_folds + 1)
    fold_size = max(fold_size, 20)  # at least 20 bars per fold

    folds: list[TimeSeriesSplit] = []
    for i in range(n_folds):
        # Expanding train: always starts at t0
        train_end_idx = min_train + i * fold_size - 1
        test_start_idx = train_end_idx + 1
        test_end_idx = min(test_start_idx + fold_size - 1, n - 1)

        if test_start_idx >= n:
            break

        fold = TimeSeriesSplit(
            fold_id=i,
            train_start=timestamps[0],
            train_end=timestamps[train_end_idx],
            test_start=timestamps[test_start_idx],
            test_end=timestamps[test_end_idx],
            n_train=train_end_idx + 1,
            n_test=test_end_idx - test_start_idx + 1,
        )
        folds.append(fold)

    return folds
