"""
CPAT — Backtest Runner (Week 3)
================================
CLI script to run backtests, parameter optimization, and walk-forward validation.

Usage::

    # Standard backtest
    python scripts/run_backtest.py

    # Optimize momentum parameters (train split only)
    python scripts/run_backtest.py --mode optimize --strategy momentum

    # Walk-forward validation (OOS robustness check)
    python scripts/run_backtest.py --mode walk-forward --strategy momentum

    # Full backtest with specific strategy
    python scripts/run_backtest.py --mode backtest --strategy mean_reversion
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import click
import pandas as pd

from cpat.config.loader import load_config
from cpat.backtest.engine import BacktestEngine
from cpat.data.pipeline import DataPipeline
from cpat.infrastructure.logging import setup_logging
from cpat.strategies.momentum import MomentumStrategy
from cpat.strategies.mean_reversion import MeanReversionStrategy
from cpat.analytics.drawdown import compute_drawdown_table, drawdown_table_to_dataframe
from cpat.analytics.distributions import compute_distribution
from cpat.optimization.optimizer import (
    GridSearchOptimizer, RandomSearchOptimizer, results_to_dataframe,
)
from cpat.validation.splitter import train_test_split
from cpat.validation.walk_forward import WalkForwardValidator, WalkForwardConfig
from cpat.validation.overfitting import degradation_report, stability_check


# ── Shared helpers ─────────────────────────────────────────────────────────────


def _register_strategies(engine: BacktestEngine, strategy: str, bar_data: dict, cfg) -> None:
    """Register strategies on the engine based on the --strategy flag."""
    symbols = list(bar_data.keys())
    if strategy in ("momentum", "both"):
        mom = MomentumStrategy(cfg.strategies.momentum, engine.event_queue, symbols)
        engine.register_market_handler(mom)
        engine.register_fill_handler(mom)
    if strategy in ("mean_reversion", "both"):
        mr = MeanReversionStrategy(cfg.strategies.mean_reversion, engine.event_queue, symbols)
        engine.register_market_handler(mr)
        engine.register_fill_handler(mr)


def _print_full_report(result) -> None:
    """Print the full PerformanceReport including Week 3 metrics."""
    click.echo(str(result.report))

    # Drawdown table (top-5 worst)
    dd_table = compute_drawdown_table(result.equity_curve)
    if dd_table:
        click.echo("\n── Top Drawdown Periods ───────────────────────────────────────")
        df = drawdown_table_to_dataframe(dd_table[:5])
        click.echo(df.to_string(index=False))

    # Return distribution
    returns = result.equity_curve.pct_change().dropna()
    dist = compute_distribution(returns)
    click.echo(str(dist))


# ── Standard backtest ──────────────────────────────────────────────────────────


def _run_backtest(bar_data: dict, cfg, strategy: str) -> None:
    """Run a standard full-period backtest."""
    click.echo(f"\nRunning backtest [{cfg.backtest.start_date} → {cfg.backtest.end_date}]...")
    engine = BacktestEngine.from_config(cfg, managed_risk=True)
    _register_strategies(engine, strategy, bar_data, cfg)
    result = engine.run(bar_data)
    _print_full_report(result)

    # Save equity curve
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    equity_path = output_dir / f"equity_curve_{strategy}.csv"
    result.equity_curve.to_csv(equity_path)
    result.portfolio_snapshots_df.to_csv(output_dir / f"portfolio_snapshots_{strategy}.csv")
    result.risk_history_df.to_csv(output_dir / f"risk_history_{strategy}.csv")
    result.trade_risk_df.to_csv(output_dir / f"trade_risk_{strategy}.csv")
    click.echo(f"\nEquity curve saved → {equity_path}")


def _result_summary(label: str, result) -> dict[str, float | int | str]:
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


def _run_compare(bar_data: dict, cfg, strategy: str) -> None:
    """Run baseline vs managed backtests and compare results."""
    click.echo(f"\nRunning baseline vs managed comparison [{cfg.backtest.start_date} → {cfg.backtest.end_date}]...")

    baseline_engine = BacktestEngine.from_config(cfg, managed_risk=False)
    _register_strategies(baseline_engine, strategy, bar_data, cfg)
    baseline = baseline_engine.run(bar_data)

    managed_engine = BacktestEngine.from_config(cfg, managed_risk=True)
    _register_strategies(managed_engine, strategy, bar_data, cfg)
    managed = managed_engine.run(bar_data)

    comparison = pd.DataFrame([
        _result_summary("baseline", baseline),
        _result_summary("managed", managed),
    ])
    click.echo("\n── Baseline vs Managed Comparison ───────────────────────────────")
    click.echo(comparison.to_string(index=False))

    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / f"comparison_{strategy}.csv"
    comparison.to_csv(comparison_path, index=False)
    baseline.equity_curve.to_csv(output_dir / f"equity_curve_{strategy}_baseline.csv")
    managed.equity_curve.to_csv(output_dir / f"equity_curve_{strategy}_managed.csv")
    managed.risk_history_df.to_csv(output_dir / f"risk_history_{strategy}_managed.csv")
    managed.trade_risk_df.to_csv(output_dir / f"trade_risk_{strategy}_managed.csv")
    click.echo(f"\nComparison saved → {comparison_path}")


# ── Optimization ───────────────────────────────────────────────────────────────


def _run_optimize(bar_data: dict, cfg, strategy: str) -> None:
    """Run parameter optimization on the training split."""
    opt_cfg = cfg.optimization
    train_ratio = opt_cfg.train_ratio

    click.echo(f"\n── Parameter Optimization ────────────────────────────────────────")
    click.echo(f"  Strategy:    {strategy}")
    click.echo(f"  Method:      {opt_cfg.method}")
    click.echo(f"  Metric:      {opt_cfg.metric}")
    click.echo(f"  Train ratio: {train_ratio:.0%}  (test split never touched)")
    click.echo(f"  Trials:      {opt_cfg.n_trials}")

    # Split: only pass train data to optimizer
    train_data, test_data = train_test_split(
        bar_data,
        train_ratio=train_ratio,
        warmup_bars=cfg.backtest.warmup_bars,
    )
    click.echo(f"\n  Train: {len(next(iter(train_data.values())))} bars")
    click.echo(f"  Test:  {len(next(iter(test_data.values())))} bars (held out)\n")

    if opt_cfg.method == "grid":
        optimizer = GridSearchOptimizer(strategy=strategy, metric=opt_cfg.metric)
    else:
        optimizer = RandomSearchOptimizer(
            strategy=strategy, metric=opt_cfg.metric,
            n_trials=opt_cfg.n_trials, random_seed=opt_cfg.random_seed,
        )

    results = optimizer.run(train_data, cfg)
    if not results:
        click.echo("No valid results produced.")
        return

    df = results_to_dataframe(results)
    click.echo("── Optimization Results (train split) ────────────────────────────")
    click.echo(df.head(15).to_string())

    # Evaluate best params on test split
    best = results[0]
    click.echo(f"\n── Best Parameters ───────────────────────────────────────────────")
    for k, v in best.params.items():
        click.echo(f"  {k}: {v}")

    # Rebuild engine with best params and evaluate on TEST
    from cpat.optimization.optimizer import _apply_params_to_config, _run_single
    best_cfg = _apply_params_to_config(cfg, best.params, strategy)
    if best_cfg:
        click.echo(f"\n── Test Split Evaluation (OOS) ───────────────────────────────────")
        test_result = _run_single(test_data, best_cfg, strategy)
        dr = degradation_report(best.report, test_result.report, label="train_vs_test")
        click.echo(f"  Assessment:       {dr['assessment']}")
        click.echo(f"  IS  Sharpe:       {dr['is']['sharpe_ratio']:.3f}")
        click.echo(f"  OOS Sharpe:       {dr['oos']['sharpe_ratio']:.3f}")
        click.echo(f"  Degradation:      {dr['degradation']['sharpe_ratio']:.3f}")

    # Sensitivity analysis
    click.echo(f"\n── Parameter Sensitivity ─────────────────────────────────────────")
    stab = stability_check(df, metric="sharpe_ratio")
    if not stab.empty:
        click.echo(stab.to_string(index=False))

    # Save results
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    opt_path = output_dir / f"optimization_{strategy}.csv"
    df.to_csv(opt_path)
    click.echo(f"\nOptimization results saved → {opt_path}")


# ── Walk-Forward ───────────────────────────────────────────────────────────────


def _run_walk_forward(bar_data: dict, cfg, strategy: str) -> None:
    """Run walk-forward validation."""
    val_cfg = cfg.validation
    wf_config = WalkForwardConfig(
        train_bars=val_cfg.walk_forward_train_bars,
        test_bars=val_cfg.walk_forward_test_bars,
        step_bars=val_cfg.walk_forward_step_bars,
        optimize_on_fold=val_cfg.optimize_on_fold,
        n_opt_trials=val_cfg.n_opt_trials,
    )

    click.echo(f"\n── Walk-Forward Validation ───────────────────────────────────────")
    click.echo(f"  Strategy:     {strategy}")
    click.echo(f"  Train bars:   {wf_config.train_bars} (~{wf_config.train_bars//252}yr)")
    click.echo(f"  Test bars:    {wf_config.test_bars} (~{wf_config.test_bars//252}yr)")
    click.echo(f"  Step bars:    {wf_config.step_bars}")
    click.echo(f"  Optimize/fold:{wf_config.optimize_on_fold}\n")

    validator = WalkForwardValidator(strategy=strategy, wf_config=wf_config)
    result = validator.run(bar_data, cfg)

    # Per-fold summary table
    click.echo("── Per-Fold Results ──────────────────────────────────────────────")
    click.echo(result.summary_dataframe().to_string())

    # Aggregate summary
    click.echo(str(result))

    # Overall OOS report
    click.echo("\n── Combined OOS Performance ──────────────────────────────────────")
    click.echo(str(result.oos_report))

    # Save OOS equity
    output_dir = Path("data/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    oos_path = output_dir / f"walk_forward_oos_{strategy}.csv"
    result.combined_oos_equity.to_csv(oos_path)
    click.echo(f"OOS equity curve saved → {oos_path}")


# ── CLI entry point ────────────────────────────────────────────────────────────


@click.command()
@click.option("--config", default="config/settings.yaml", help="Path to settings.yaml")
@click.option(
    "--strategy", default="momentum",
    type=click.Choice(["momentum", "mean_reversion", "both"]),
    help="Strategy to run",
)
@click.option(
    "--mode", default="backtest",
    type=click.Choice(["backtest", "optimize", "walk-forward", "compare"]),
    help="Execution mode",
)
@click.option("--log-level", default="INFO", help="Log level")
def main(config: str, strategy: str, mode: str, log_level: str) -> None:
    """CPAT — Multi-Asset Algorithmic Trading System CLI."""
    cfg = load_config(config)
    setup_logging(log_level=log_level, log_format="console", log_dir=cfg.logging.log_dir)

    click.echo(f"\nCPAT {'='*50}")
    click.echo(f"  Mode:     {mode}")
    click.echo(f"  Strategy: {strategy}")
    click.echo(f"  Universe: {cfg.universe.name} ({len(cfg.universe.all_symbols)} symbols)")
    click.echo(f"{'='*54}")

    # Load data
    pipeline = DataPipeline.from_config(cfg)
    bar_data = pipeline.load_bars()

    if not bar_data:
        click.echo("ERROR: No data available. Run scripts/fetch_universe.py first.")
        sys.exit(1)

    click.echo(f"Loaded {len(bar_data)} symbols × {len(next(iter(bar_data.values())))} bars")

    if mode == "backtest":
        _run_backtest(bar_data, cfg, strategy)
    elif mode == "compare":
        _run_compare(bar_data, cfg, strategy)
    elif mode == "optimize":
        _run_optimize(bar_data, cfg, strategy)
    elif mode == "walk-forward":
        _run_walk_forward(bar_data, cfg, strategy)


if __name__ == "__main__":
    main()
