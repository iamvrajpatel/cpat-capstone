"""Result parsing and chart-preparation utilities for the Streamlit UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cpat.analytics.distributions import compute_distribution
from cpat.analytics.performance import PerformanceTracker


@dataclass(frozen=True)
class ResultArtifactBundle:
    """Resolved result-file bundle for one strategy namespace."""

    strategy: str
    equity_curve: Path | None = None
    comparison: Path | None = None
    optimization: Path | None = None
    risk_history: Path | None = None
    portfolio_snapshots: Path | None = None
    trade_risk: Path | None = None
    walk_forward: Path | None = None


class AnalyticsService:
    """Read backend artifacts and convert them into UI-friendly datasets."""

    def __init__(self, results_dir: str | Path = "data/results", logs_dir: str | Path = "logs") -> None:
        self._results_dir = Path(results_dir)
        self._logs_dir = Path(logs_dir)

    @property
    def results_dir(self) -> Path:
        """Return the configured results directory."""
        return self._results_dir

    def discover_artifacts(self, strategy: str) -> ResultArtifactBundle:
        """Find the expected result files for a strategy namespace."""
        return ResultArtifactBundle(
            strategy=strategy,
            equity_curve=self._path_if_exists(f"equity_curve_{strategy}.csv"),
            comparison=self._path_if_exists(f"comparison_{strategy}.csv"),
            optimization=self._path_if_exists(f"optimization_{strategy}.csv"),
            risk_history=self._path_if_exists(f"risk_history_{strategy}.csv"),
            portfolio_snapshots=self._path_if_exists(f"portfolio_snapshots_{strategy}.csv"),
            trade_risk=self._path_if_exists(f"trade_risk_{strategy}.csv"),
            walk_forward=self._path_if_exists(f"walk_forward_oos_{strategy}.csv"),
        )

    def load_dataframe(self, path: str | Path | None, index_col: int | str | None = 0) -> pd.DataFrame:
        """Load a CSV artifact into a DataFrame or return an empty frame."""
        if path is None:
            return pd.DataFrame()
        target = Path(path)
        if not target.exists():
            return pd.DataFrame()
        return pd.read_csv(target, index_col=index_col)

    def load_series(self, path: str | Path | None) -> pd.Series:
        """Load a single-column CSV artifact into a Series."""
        frame = self.load_dataframe(path, index_col=0)
        if frame.empty:
            return pd.Series(dtype=float, name="value")
        if frame.shape[1] == 1:
            series = frame.iloc[:, 0]
        else:
            series = frame.squeeze("columns")
        series.index = pd.to_datetime(series.index, utc=True, errors="coerce")
        return pd.to_numeric(series, errors="coerce").dropna()

    def latest_dashboard_snapshot(self, strategy: str) -> dict[str, Any]:
        """Build a KPI snapshot from the latest saved artifacts."""
        artifacts = self.discover_artifacts(strategy)
        equity = self.load_series(artifacts.equity_curve)
        snapshots = self.load_dataframe(artifacts.portfolio_snapshots)
        risk = self.load_dataframe(artifacts.risk_history)
        trade_risk = self.load_dataframe(artifacts.trade_risk)

        latest_equity = float(equity.iloc[-1]) if not equity.empty else 0.0
        previous_equity = float(equity.iloc[-2]) if len(equity) > 1 else latest_equity
        daily_pnl = latest_equity - previous_equity

        latest_snapshot = snapshots.iloc[-1].to_dict() if not snapshots.empty else {}
        latest_risk = risk.iloc[-1].to_dict() if not risk.empty else {}

        return {
            "equity": latest_equity,
            "daily_pnl": daily_pnl,
            "gross_exposure": float(latest_snapshot.get("gross_exposure", 0.0)),
            "net_exposure": float(latest_snapshot.get("net_exposure", 0.0)),
            "long_exposure": float(latest_snapshot.get("long_exposure", 0.0)),
            "short_exposure": float(latest_snapshot.get("short_exposure", 0.0)),
            "open_positions": int(latest_snapshot.get("open_positions", 0)),
            "active_risk": float(latest_risk.get("planned_risk", 0.0)),
            "tracked_trades": int(len(trade_risk.index)),
        }

    def equity_drawdown_frame(self, equity_curve: pd.Series) -> pd.DataFrame:
        """Return equity and drawdown in one chart-friendly frame."""
        if equity_curve.empty:
            return pd.DataFrame(columns=["equity", "drawdown"])
        equity = equity_curve.copy()
        equity.index = pd.to_datetime(equity.index, utc=True)
        peak = equity.cummax()
        drawdown = ((equity - peak) / peak.replace(0, np.nan)).fillna(0.0)
        return pd.DataFrame({"equity": equity, "drawdown": drawdown})

    def rolling_sharpe_frame(self, equity_curve: pd.Series, window: int = 63) -> pd.DataFrame:
        """Compute rolling returns and rolling Sharpe from an equity curve."""
        if equity_curve.empty:
            return pd.DataFrame(columns=["returns", "rolling_sharpe", "rolling_return"])
        equity = equity_curve.copy()
        equity.index = pd.to_datetime(equity.index, utc=True)
        returns = equity.pct_change().fillna(0.0)
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std().replace(0.0, np.nan)
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        rolling_return = equity.pct_change(window).fillna(0.0)
        return pd.DataFrame(
            {
                "returns": returns,
                "rolling_sharpe": rolling_sharpe.fillna(0.0),
                "rolling_return": rolling_return,
            }
        )

    def performance_report_from_equity(
        self,
        equity_curve: pd.Series,
        initial_capital: float,
    ) -> dict[str, Any]:
        """Recompute a performance report from an equity curve alone."""
        tracker = PerformanceTracker(initial_capital=initial_capital)
        for timestamp, value in equity_curve.items():
            tracker.record(pd.Timestamp(timestamp), float(value))
        report = tracker.compute(fills=[])
        return report.to_dict()

    def distribution_summary(self, equity_curve: pd.Series) -> dict[str, Any]:
        """Compute return-distribution stats from an equity curve."""
        if equity_curve.empty:
            return {}
        returns = equity_curve.pct_change().dropna()
        if returns.empty:
            return {}
        dist = compute_distribution(returns)
        return {
            "mean_daily_return": getattr(dist, "mean", 0.0),
            "std_daily_return": getattr(dist, "std", 0.0),
            "skewness": getattr(dist, "skewness", 0.0),
            "kurtosis": getattr(dist, "kurtosis", 0.0),
            "tail_ratio": getattr(dist, "tail_ratio", 1.0),
            "var_95": getattr(dist, "var_95", 0.0),
            "cvar_95": getattr(dist, "cvar_95", 0.0),
        }

    def load_live_jsonl(self, path: str | Path) -> pd.DataFrame:
        """Parse a live-trading JSONL log into a DataFrame."""
        target = Path(path)
        if not target.exists():
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append({"timestamp": None, "event_type": "RAW", "message": line})
        return pd.DataFrame(rows)

    def _path_if_exists(self, name: str) -> Path | None:
        """Return a result-file path if present."""
        path = self._results_dir / name
        return path if path.exists() else None

