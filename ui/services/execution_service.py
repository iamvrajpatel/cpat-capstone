"""Backend orchestration and managed subprocess controls for the UI."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd

from cpat.backtest.engine import BacktestEngine, BacktestResult
from cpat.config.loader import CPATConfig
from cpat.data.pipeline import DataPipeline
from cpat.optimization.optimizer import (
    GridSearchOptimizer,
    RandomSearchOptimizer,
    _apply_params_to_config,
    _run_single,
    results_to_dataframe,
)
from cpat.strategies.mean_reversion import MeanReversionStrategy
from cpat.strategies.momentum import MomentumStrategy
from cpat.validation.overfitting import degradation_report, stability_check
from cpat.validation.splitter import train_test_split
from cpat.validation.walk_forward import WalkForwardConfig, WalkForwardValidator


LiveProcessState = Literal["stopped", "starting", "running", "stopping", "errored"]


@dataclass(frozen=True)
class LiveRuntimeStatus:
    """Persistent UI-facing snapshot of the managed live subprocess."""

    pid: int | None = None
    mode: str = "paper"
    symbols: tuple[str, ...] = ()
    started_at: str = ""
    config_path: str = "config/settings.yaml"
    dry_run: bool = False
    state: LiveProcessState = "stopped"
    last_heartbeat_at: str = ""
    last_error: str = ""
    command: tuple[str, ...] = ()
    summary_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly mapping."""
        payload = asdict(self)
        payload["symbols"] = list(self.symbols)
        payload["command"] = list(self.command)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LiveRuntimeStatus":
        """Construct a status object from persisted JSON."""
        return cls(
            pid=payload.get("pid"),
            mode=payload.get("mode", "paper"),
            symbols=tuple(payload.get("symbols", [])),
            started_at=payload.get("started_at", ""),
            config_path=payload.get("config_path", "config/settings.yaml"),
            dry_run=bool(payload.get("dry_run", False)),
            state=payload.get("state", "stopped"),
            last_heartbeat_at=payload.get("last_heartbeat_at", ""),
            last_error=payload.get("last_error", ""),
            command=tuple(payload.get("command", [])),
            summary_snapshot=dict(payload.get("summary_snapshot", {})),
        )


class ExecutionService:
    """Service layer for research runs and live-process lifecycle management."""

    def __init__(
        self,
        config_path: str | Path = "config/settings.yaml",
        results_dir: str | Path = "data/results",
        logs_dir: str | Path = "logs",
    ) -> None:
        self._config_path = Path(config_path)
        self._results_dir = Path(results_dir)
        self._logs_dir = Path(logs_dir)
        self._live_dir = self._logs_dir / "live"
        self._status_path = self._live_dir / "ui_runtime_status.json"
        self._process_log_path = self._live_dir / "ui_process.log"
        self._live_dir.mkdir(parents=True, exist_ok=True)

    def list_result_files(self) -> list[Path]:
        """Return sorted CSV artifacts in the results directory."""
        if not self._results_dir.exists():
            return []
        return sorted(self._results_dir.glob("*.csv"))

    def list_log_files(self) -> list[Path]:
        """Return sorted log artifacts across core and live directories."""
        candidates = list(self._logs_dir.glob("*.log")) + list(self._live_dir.glob("*.jsonl"))
        if (self._live_dir / "ui_process.log").exists():
            candidates.append(self._live_dir / "ui_process.log")
        return sorted(set(candidates))

    def run_backtest(self, config: CPATConfig, strategy: str, managed_risk: bool = True) -> dict[str, Any]:
        """Run a full backtest and return UI-friendly artifacts."""
        bar_data = self._load_bar_data(config)
        engine = BacktestEngine.from_config(config, managed_risk=managed_risk)
        self._register_strategies(engine, strategy, bar_data, config)
        result = engine.run(bar_data)
        self._persist_backtest_artifacts(result, strategy)
        return {
            "result": result,
            "report": result.report.to_dict(),
            "equity_curve": result.equity_curve,
            "drawdown": result.drawdown_series,
            "trade_log_df": result.trade_log_df,
            "portfolio_snapshots_df": result.portfolio_snapshots_df,
            "risk_history_df": result.risk_history_df,
            "trade_risk_df": result.trade_risk_df,
            "stats": result.stats,
        }

    def run_compare(self, config: CPATConfig, strategy: str) -> dict[str, Any]:
        """Run baseline vs managed comparison."""
        bar_data = self._load_bar_data(config)
        baseline = self.run_backtest(config, strategy, managed_risk=False)["result"]
        managed = self.run_backtest(config, strategy, managed_risk=True)["result"]
        comparison = pd.DataFrame(
            [
                self._result_summary("baseline", baseline),
                self._result_summary("managed", managed),
            ]
        )
        comparison_path = self._results_dir / f"comparison_{strategy}.csv"
        comparison.to_csv(comparison_path, index=False)
        return {
            "baseline": baseline,
            "managed": managed,
            "comparison_df": comparison,
            "comparison_path": comparison_path,
        }

    def run_walk_forward(self, config: CPATConfig, strategy: str) -> dict[str, Any]:
        """Run walk-forward validation and persist the stitched OOS curve."""
        bar_data = self._load_bar_data(config)
        wf_cfg = WalkForwardConfig(
            train_bars=config.validation.walk_forward_train_bars,
            test_bars=config.validation.walk_forward_test_bars,
            step_bars=config.validation.walk_forward_step_bars,
            optimize_on_fold=config.validation.optimize_on_fold,
            n_opt_trials=config.validation.n_opt_trials,
            opt_metric=config.optimization.metric,
            random_seed=config.optimization.random_seed,
        )
        validator = WalkForwardValidator(strategy=strategy, wf_config=wf_cfg)
        result = validator.run(bar_data, config)
        output_path = self._results_dir / f"walk_forward_oos_{strategy}.csv"
        result.combined_oos_equity.to_csv(output_path)
        return {
            "result": result,
            "summary_df": result.summary_dataframe(),
            "oos_report": result.oos_report.to_dict(),
            "oos_equity": result.combined_oos_equity,
            "output_path": output_path,
        }

    def run_optimization(
        self,
        config: CPATConfig,
        strategy: str,
        param_grid: dict[str, list[Any]] | None = None,
    ) -> dict[str, Any]:
        """Run train-only optimization plus held-out test evaluation."""
        bar_data = self._load_bar_data(config)
        train_data, test_data = train_test_split(
            bar_data,
            train_ratio=config.optimization.train_ratio,
            warmup_bars=config.backtest.warmup_bars,
        )
        if config.optimization.method == "grid":
            optimizer = GridSearchOptimizer(strategy=strategy, metric=config.optimization.metric)
        else:
            optimizer = RandomSearchOptimizer(
                strategy=strategy,
                metric=config.optimization.metric,
                n_trials=config.optimization.n_trials,
                random_seed=config.optimization.random_seed,
            )
        results = optimizer.run(train_data, config, param_grid=param_grid)
        results_df = results_to_dataframe(results) if results else pd.DataFrame()
        best_params: dict[str, Any] = results[0].params if results else {}
        stability_df = stability_check(results_df, metric=config.optimization.metric) if not results_df.empty else pd.DataFrame()
        test_report: dict[str, Any] | None = None
        degradation: dict[str, Any] | None = None
        if results:
            best_cfg = _apply_params_to_config(config, best_params, strategy)
            if best_cfg is not None:
                oos_result = _run_single(test_data, best_cfg, strategy)
                test_report = oos_result.report.to_dict()
                degradation = degradation_report(results[0].report, oos_result.report, label="train_vs_test")

        output_path = self._results_dir / f"optimization_{strategy}.csv"
        if not results_df.empty:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)

        return {
            "results_df": results_df,
            "best_params": best_params,
            "stability_df": stability_df,
            "test_report": test_report,
            "degradation": degradation,
            "output_path": output_path,
        }

    def start_live_process(
        self,
        *,
        mode: str,
        strategy: str,
        symbols: list[str],
        config_path: str | Path | None = None,
        dry_run: bool = False,
        demo: bool = False,
    ) -> LiveRuntimeStatus:
        """Start the managed `run_live.py` subprocess."""
        current = self.read_live_status()
        if current.pid and self._is_pid_alive(current.pid):
            return current

        target_config = str(config_path or self._config_path)
        command = [
            sys.executable,
            "scripts/run_live.py",
            "--mode",
            mode,
            "--config",
            target_config,
        ]
        if strategy:
            command.extend(["--strategy", strategy])
        if symbols:
            command.extend(["--symbols", *symbols])
        if dry_run:
            command.append("--dry-run")
        if demo:
            command.append("--demo")

        self._live_dir.mkdir(parents=True, exist_ok=True)
        log_handle = self._process_log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        status = LiveRuntimeStatus(
            pid=process.pid,
            mode=mode,
            symbols=tuple(symbols),
            started_at=self._now_iso(),
            config_path=target_config,
            dry_run=dry_run,
            state="running",
            last_heartbeat_at=self._now_iso(),
            command=tuple(command),
        )
        self._write_status(status)
        return status

    def stop_live_process(self) -> LiveRuntimeStatus:
        """Attempt a graceful shutdown of the managed live subprocess."""
        status = self.read_live_status()
        if status.pid is None:
            return status
        if self._is_pid_alive(status.pid):
            os.kill(status.pid, signal.SIGTERM)
            status = LiveRuntimeStatus.from_dict(
                {
                    **status.to_dict(),
                    "state": "stopping",
                    "last_heartbeat_at": self._now_iso(),
                }
            )
            self._write_status(status)
            return status
        return self._write_status(LiveRuntimeStatus())

    def kill_live_process(self) -> LiveRuntimeStatus:
        """Force-kill the managed live subprocess."""
        status = self.read_live_status()
        if status.pid is not None and self._is_pid_alive(status.pid):
            os.kill(status.pid, signal.SIGKILL)
        cleared = LiveRuntimeStatus(last_error="Process terminated by emergency kill.")
        return self._write_status(cleared)

    def read_live_status(self) -> LiveRuntimeStatus:
        """Read and refresh the live runtime metadata contract."""
        if not self._status_path.exists():
            return LiveRuntimeStatus()
        with self._status_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        status = LiveRuntimeStatus.from_dict(payload)
        if status.pid is not None and not self._is_pid_alive(status.pid):
            return self._write_status(
                LiveRuntimeStatus.from_dict(
                    {
                        **status.to_dict(),
                        "state": "stopped",
                        "pid": None,
                        "last_heartbeat_at": self._now_iso(),
                    }
                )
            )
        return status

    def read_text_tail(self, path: str | Path, limit: int = 200) -> str:
        """Read the last `limit` lines of a text log."""
        target = Path(path)
        if not target.exists():
            return ""
        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-limit:])

    def required_credentials(self, mode: str) -> tuple[str, ...]:
        """Return environment variables required for a live mode."""
        if mode == "dhan":
            return ("DHAN_CLIENT_ID", "DHAN_ACCESS_TOKEN")
        return ()

    def missing_credentials(self, mode: str) -> tuple[str, ...]:
        """Return missing environment variables for the selected broker mode."""
        return tuple(name for name in self.required_credentials(mode) if not os.getenv(name))

    def _load_bar_data(self, config: CPATConfig) -> dict[str, pd.DataFrame]:
        """Load stored bar data through the existing data pipeline."""
        pipeline = DataPipeline.from_config(config)
        return pipeline.load_bars()

    def _register_strategies(
        self,
        engine: BacktestEngine,
        strategy: str,
        bar_data: dict[str, pd.DataFrame],
        config: CPATConfig,
    ) -> None:
        """Attach the requested strategies to a backtest engine."""
        symbols = list(bar_data.keys())
        if strategy in {"momentum", "both"}:
            momentum = MomentumStrategy(config.strategies.momentum, engine.event_queue, symbols)
            engine.register_market_handler(momentum)
            engine.register_fill_handler(momentum)
        if strategy in {"mean_reversion", "both"}:
            mean_reversion = MeanReversionStrategy(config.strategies.mean_reversion, engine.event_queue, symbols)
            engine.register_market_handler(mean_reversion)
            engine.register_fill_handler(mean_reversion)

    def _persist_backtest_artifacts(self, result: BacktestResult, strategy: str) -> None:
        """Write core backtest outputs to the results directory."""
        self._results_dir.mkdir(parents=True, exist_ok=True)
        result.equity_curve.to_csv(self._results_dir / f"equity_curve_{strategy}.csv")
        result.portfolio_snapshots_df.to_csv(self._results_dir / f"portfolio_snapshots_{strategy}.csv")
        result.risk_history_df.to_csv(self._results_dir / f"risk_history_{strategy}.csv")
        result.trade_risk_df.to_csv(self._results_dir / f"trade_risk_{strategy}.csv")

    def _result_summary(self, label: str, result: BacktestResult) -> dict[str, Any]:
        """Return compact comparison metrics for backtest modes."""
        return {
            "mode": label,
            "final_equity": round(result.final_equity, 2),
            "total_return": round(result.report.total_return, 6),
            "sharpe": round(result.report.sharpe_ratio, 4),
            "max_drawdown": round(result.report.max_drawdown, 6),
            "ulcer_index": round(result.report.ulcer_index, 4),
            "risk_rejects": int(result.stats.get("risk_rejects", 0)),
            "stop_triggers": int(result.stats.get("stop_triggers", 0)),
            "max_gross_exposure": round(float(result.stats.get("max_gross_exposure", 0.0)), 2),
        }

    def _write_status(self, status: LiveRuntimeStatus) -> LiveRuntimeStatus:
        """Persist runtime status JSON."""
        self._live_dir.mkdir(parents=True, exist_ok=True)
        with self._status_path.open("w", encoding="utf-8") as handle:
            json.dump(status.to_dict(), handle, indent=2)
        return status

    def _is_pid_alive(self, pid: int) -> bool:
        """Check whether a Unix PID is still alive."""
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def _now_iso(self) -> str:
        """Return an ISO-8601 UTC timestamp."""
        return datetime.now(timezone.utc).isoformat()
