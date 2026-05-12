"""Tests for UI analytics helpers."""

from __future__ import annotations

import json

import pandas as pd

from ui.services.analytics_service import AnalyticsService


def test_latest_dashboard_snapshot_reads_saved_artifacts(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    equity = pd.Series([100.0, 105.0], index=pd.date_range("2024-01-01", periods=2, tz="UTC"), name="equity")
    equity.to_csv(results_dir / "equity_curve_momentum.csv")
    snapshots = pd.DataFrame(
        {
            "gross_exposure": [0.3],
            "net_exposure": [0.25],
            "long_exposure": [0.25],
            "short_exposure": [0.0],
            "open_positions": [4],
        },
        index=[pd.Timestamp("2024-01-02", tz="UTC")],
    )
    snapshots.to_csv(results_dir / "portfolio_snapshots_momentum.csv")
    risk = pd.DataFrame({"planned_risk": [12500.0]}, index=[pd.Timestamp("2024-01-02", tz="UTC")])
    risk.to_csv(results_dir / "risk_history_momentum.csv")
    trade_risk = pd.DataFrame({"planned_risk": [5000.0, 7500.0]}, index=["a", "b"])
    trade_risk.to_csv(results_dir / "trade_risk_momentum.csv")

    service = AnalyticsService(results_dir=results_dir, logs_dir=tmp_path / "logs")
    snapshot = service.latest_dashboard_snapshot("momentum")

    assert snapshot["equity"] == 105.0
    assert snapshot["daily_pnl"] == 5.0
    assert snapshot["gross_exposure"] == 0.3
    assert snapshot["tracked_trades"] == 2


def test_rolling_sharpe_frame_returns_expected_columns():
    series = pd.Series(
        [100, 101, 102, 103, 104],
        index=pd.date_range("2024-01-01", periods=5, tz="UTC"),
        name="equity",
    )
    service = AnalyticsService()
    frame = service.rolling_sharpe_frame(series, window=2)

    assert {"returns", "rolling_sharpe", "rolling_return"}.issubset(frame.columns)
    assert len(frame) == 5


def test_load_live_jsonl_parses_rows(tmp_path):
    log_path = tmp_path / "live.jsonl"
    rows = [
        {"timestamp": "2024-01-01T00:00:00Z", "event_type": "SIGNAL", "symbol": "AAPL"},
        {"timestamp": "2024-01-01T00:01:00Z", "event_type": "FILL", "symbol": "AAPL"},
    ]
    log_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    service = AnalyticsService()
    frame = service.load_live_jsonl(log_path)

    assert list(frame["event_type"]) == ["SIGNAL", "FILL"]

