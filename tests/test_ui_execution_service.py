"""Tests for execution service helpers and live-process controls."""

from __future__ import annotations

import json
from pathlib import Path

from ui.services.execution_service import ExecutionService, LiveRuntimeStatus


class _FakeProcess:
    """Small stand-in for subprocess.Popen in unit tests."""

    def __init__(self, pid: int = 43210) -> None:
        self.pid = pid


def test_start_live_process_writes_runtime_status(tmp_path, monkeypatch):
    service = ExecutionService(
        config_path=tmp_path / "settings.yaml",
        results_dir=tmp_path / "results",
        logs_dir=tmp_path / "logs",
    )
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: _FakeProcess())

    status = service.start_live_process(
        mode="paper",
        strategy="momentum",
        symbols=["RELIANCE.NS", "TCS.NS"],
        config_path=tmp_path / "settings.yaml",
        dry_run=True,
        demo=True,
    )

    assert status.pid == 43210
    assert status.state == "running"
    persisted = json.loads((tmp_path / "logs" / "live" / "ui_runtime_status.json").read_text())
    assert persisted["mode"] == "paper"
    assert persisted["symbols"] == ["RELIANCE.NS", "TCS.NS"]


def test_read_live_status_marks_stale_pid_stopped(tmp_path, monkeypatch):
    service = ExecutionService(
        config_path=tmp_path / "settings.yaml",
        results_dir=tmp_path / "results",
        logs_dir=tmp_path / "logs",
    )
    live_dir = tmp_path / "logs" / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    payload = LiveRuntimeStatus(pid=99999, state="running", mode="paper").to_dict()
    (live_dir / "ui_runtime_status.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(service, "_is_pid_alive", lambda pid: False)

    status = service.read_live_status()

    assert status.pid is None
    assert status.state == "stopped"


def test_missing_credentials_for_dhan(monkeypatch):
    service = ExecutionService()
    monkeypatch.delenv("DHAN_CLIENT_ID", raising=False)
    monkeypatch.delenv("DHAN_ACCESS_TOKEN", raising=False)

    assert service.missing_credentials("dhan") == ("DHAN_CLIENT_ID", "DHAN_ACCESS_TOKEN")

