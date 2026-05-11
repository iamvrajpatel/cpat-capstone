"""
CPAT — Live / Paper Trading Runner (Week 5)
===========================================
CLI entry point for paper/demo/live execution.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

from cpat.backtest.event_queue import EventQueue
from cpat.brokers.paper import PaperBroker
from cpat.config.loader import CPATConfig, load_config
from cpat.core.enums import Interval
from cpat.infrastructure.execution_engine_live import LiveExecutionEngine
from cpat.infrastructure.live_logger import LiveTradeLogger
from cpat.infrastructure.logging import setup_logging
from cpat.infrastructure.order_manager import OrderManager
from cpat.infrastructure.scheduler import TradingScheduler
from cpat.portfolio.allocator import AllocatorFactory
from cpat.portfolio.manager import PortfolioManager
from cpat.risk.engine import RiskEngine
from cpat.risk.position_sizing import PositionSizerFactory
from cpat.risk.risk_manager import PortfolioRiskManager, StopLossTracker
from cpat.strategies.momentum import MomentumStrategy

logger = logging.getLogger(__name__)


def _make_demo_data_provider(symbols: list[str], n_bars: int = 300):
    """Synthetic OHLCV feed for paper/demo mode."""
    rng = np.random.default_rng(42)
    dfs: dict[str, pd.DataFrame] = {}
    end_ts = pd.Timestamp.utcnow().floor("min")

    for sym in symbols:
        closes = 1000.0 * np.cumprod(1 + rng.normal(0.0004, 0.01, n_bars))
        highs = closes * (1 + rng.uniform(0.001, 0.01, n_bars))
        lows = closes * (1 - rng.uniform(0.001, 0.01, n_bars))
        opens = closes * (1 + rng.uniform(-0.005, 0.005, n_bars))
        vols = rng.integers(500_000, 5_000_000, n_bars).astype(float)
        idx = pd.date_range(end=end_ts, periods=n_bars, freq="min", tz="UTC")
        dfs[sym] = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "adj_close": closes,
                "volume": vols,
            },
            index=idx,
        )

    def _provider(requested: list[str]) -> dict[str, pd.DataFrame]:
        result: dict[str, pd.DataFrame] = {}
        now_ts = pd.Timestamp.utcnow().floor("min")
        for sym in requested:
            df = dfs[sym]
            last_close = float(df["close"].iloc[-1])
            new_close = last_close * (1 + rng.normal(0.0004, 0.01))
            new_row = pd.DataFrame(
                {
                    "open": [last_close],
                    "high": [max(last_close, new_close) * 1.003],
                    "low": [min(last_close, new_close) * 0.997],
                    "close": [new_close],
                    "adj_close": [new_close],
                    "volume": [float(rng.integers(500_000, 5_000_000))],
                },
                index=[now_ts],
            )
            dfs[sym] = pd.concat([df, new_row]).sort_index().tail(600)
            result[sym] = dfs[sym].copy()
        return result

    return _provider


def _make_broker_seeded_data_provider(
    cfg: CPATConfig,
    broker,
    symbols: list[str],
):
    """Quote-driven rolling-bar provider seeded from local parquet data."""
    from cpat.data.store import ParquetDataStore

    store = ParquetDataStore(cfg.data.processed_data_path)
    seed_bars = max(
        cfg.backtest.warmup_bars,
        cfg.strategies.momentum.lookback_long + cfg.strategies.momentum.skip_most_recent + 1,
    )
    history: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        try:
            df = store.load(sym, interval=Interval.ONE_DAY).tail(seed_bars).copy()
            if "adj_close" not in df.columns:
                df["adj_close"] = df["close"]
            history[sym] = df
        except Exception as exc:  # noqa: BLE001
            logger.warning("No local seed data for %s: %s", sym, exc)
            history[sym] = pd.DataFrame(
                columns=["open", "high", "low", "close", "adj_close", "volume"]
            )

    def _provider(requested: list[str]) -> dict[str, pd.DataFrame]:
        now_ts = pd.Timestamp.utcnow().floor("min")
        result: dict[str, pd.DataFrame] = {}
        for sym in requested:
            quote = broker.get_market_data(sym)
            last_px = float(quote.last)
            row = pd.DataFrame(
                {
                    "open": [last_px],
                    "high": [max(last_px, float(quote.ask or last_px))],
                    "low": [min(last_px, float(quote.bid or last_px))],
                    "close": [last_px],
                    "adj_close": [last_px],
                    "volume": [float(quote.volume)],
                },
                index=[now_ts],
            )
            history[sym] = pd.concat([history[sym], row]).sort_index()
            history[sym] = history[sym][~history[sym].index.duplicated(keep="last")].tail(seed_bars + 250)
            result[sym] = history[sym].copy()
        return result

    return _provider


def build_live_engine(
    cfg: CPATConfig,
    symbols: list[str],
    broker,
    data_provider,
    dry_run: bool = False,
) -> LiveExecutionEngine:
    """Assemble the live execution engine from current config APIs."""
    event_queue = EventQueue()
    trade_logger = LiveTradeLogger(log_dir=cfg.live.trade_log_dir)
    oms = OrderManager(audit_log_path=cfg.live.oms_audit_log)
    oms.load_audit_log()

    if isinstance(broker, PaperBroker):
        initial_capital = cfg.live.paper_initial_capital
    else:
        initial_capital = broker.get_account_info().portfolio_value

    portfolio = PortfolioManager(initial_capital=initial_capital)
    risk_engine = RiskEngine(
        portfolio=portfolio,
        max_position_pct=cfg.risk.max_position_pct,
        max_sector_pct=cfg.risk.max_sector_pct,
        max_gross_exposure_pct=cfg.risk.max_gross_exposure_pct,
        max_drawdown_pct=cfg.risk.max_drawdown_pct,
        min_cash_pct=cfg.risk.min_cash_pct,
    )
    sizer = PositionSizerFactory.build(
        cfg.position_sizing.method,
        risk_per_trade_pct=cfg.position_sizing.risk_per_trade_pct,
        stop_method=cfg.position_sizing.stop_method,
        fixed_stop_pct=cfg.position_sizing.fixed_stop_pct,
        atr_period=cfg.position_sizing.atr_period,
        atr_multiplier=cfg.position_sizing.atr_multiplier,
        take_profit_mult=cfg.position_sizing.take_profit_mult,
        max_position_pct=cfg.risk.max_position_pct,
        min_trade_value=cfg.position_sizing.min_trade_value,
        win_rate=cfg.position_sizing.kelly_win_rate,
        avg_win=cfg.position_sizing.kelly_avg_win,
        avg_loss=cfg.position_sizing.kelly_avg_loss,
        max_kelly_fraction=cfg.position_sizing.kelly_max_fraction,
    )
    allocator = AllocatorFactory.build(
        cfg.risk.allocation_method,
        max_position_pct=cfg.risk.max_position_pct,
        min_cash_pct=cfg.risk.min_cash_pct,
        vol_window=cfg.position_sizing.vol_window,
    )
    stop_tracker = StopLossTracker(
        event_queue=event_queue,
        trailing_stop=cfg.position_sizing.trailing_stop,
    )
    port_risk = PortfolioRiskManager(
        max_open_positions=cfg.risk.max_open_positions,
        max_daily_loss_pct=cfg.risk.max_daily_loss_pct,
        timezone=cfg.system.timezone,
    )
    strategy = MomentumStrategy(
        config=cfg.strategies.momentum,
        event_queue=event_queue,
        symbols=symbols,
    )

    return LiveExecutionEngine(
        broker=broker,
        strategy=strategy,
        portfolio=portfolio,
        risk_engine=risk_engine,
        position_sizer=sizer,
        allocator=allocator,
        stop_tracker=stop_tracker,
        portfolio_risk_manager=port_risk,
        data_provider=data_provider,
        symbols=symbols,
        trade_logger=trade_logger,
        oms=oms,
        event_queue=event_queue,
        dry_run=dry_run,
        stale_data_seconds=cfg.live.stale_data_seconds,
        order_status_poll_interval_seconds=cfg.live.order_status_poll_interval_seconds,
        order_ttl_seconds=cfg.live.order_ttl_seconds,
        max_order_retries=cfg.live.max_order_retries,
        retry_backoff_seconds=cfg.live.retry_backoff_seconds,
        reconcile_on_startup=cfg.live.reconcile_on_startup,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CPAT Live / Paper Trading Runner")
    parser.add_argument("--mode", choices=["paper", "dhan"], default=None)
    parser.add_argument("--strategy", choices=["momentum"], default="momentum")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--scheduler-mode", choices=["market_hours", "always"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(
        log_level=cfg.system.log_level,
        log_format=cfg.system.log_format,
        log_dir=cfg.logging.log_dir,
        max_bytes=cfg.logging.max_bytes,
        backup_count=cfg.logging.backup_count,
    )

    mode = args.mode or cfg.live.broker
    symbols = args.symbols or cfg.universe.equities[:3]
    dry_run = args.dry_run or cfg.live.dry_run

    print("\n" + "=" * 60)
    print("  CPAT — Live Trading System  (Week 5)")
    print("=" * 60)
    print(f"  Mode       : {mode.upper()}")
    print(f"  Symbols    : {', '.join(symbols)}")
    print(f"  Dry-run    : {dry_run}")
    print(f"  Demo       : {args.demo}")
    print("=" * 60 + "\n")

    if mode == "paper" or args.demo:
        broker = PaperBroker(
            initial_capital=cfg.live.paper_initial_capital,
            currency="INR",
            slip_bps=cfg.live.paper_slip_bps,
            commission_pct=cfg.live.paper_commission_pct,
        )
        broker.connect()
        data_provider = _make_demo_data_provider(symbols)
    else:
        from cpat.brokers.dhan import DhanBroker

        broker = DhanBroker()
        broker.connect()
        data_provider = _make_broker_seeded_data_provider(cfg, broker, symbols)

    engine = build_live_engine(cfg, symbols, broker, data_provider, dry_run=dry_run)

    if args.demo:
        print(f"\n[DEMO] Running {cfg.live.demo_ticks} ticks of paper trading...\n")
        for tick_no in range(1, cfg.live.demo_ticks + 1):
            print(f"─── Tick {tick_no} ──────────────────────────────────────────")
            engine.run_tick()
        summary = engine.summary()
        print("\n" + "=" * 60)
        print("  DEMO COMPLETE — Session Summary")
        print("=" * 60)
        for key, value in summary.items():
            print(f"  {key:<25}: {value}")
        print(f"  logs                   : {cfg.live.trade_log_dir}")
        print("=" * 60)
        broker.disconnect()
        return

    scheduler = TradingScheduler(
        callback=engine.run_tick,
        interval_seconds=cfg.live.scheduler_interval_seconds,
        market_open=cfg.live.market_open,
        market_close=cfg.live.market_close,
        timezone=cfg.live.scheduler_timezone,
        mode=args.scheduler_mode or cfg.live.scheduler_mode,
        max_tick_errors=cfg.live.max_tick_errors,
    )

    def _shutdown(signum, frame):  # noqa: ARG001
        print("\n[SHUTDOWN] Signal received — stopping scheduler...")
        scheduler.stop()
        summary = engine.summary()
        print("\n" + "=" * 60)
        print("  Session Summary")
        print("=" * 60)
        for key, value in summary.items():
            print(f"  {key:<25}: {value}")
        print(f"  overlap_skips             : {scheduler.overlap_skips}")
        print(f"  logs                      : {cfg.live.trade_log_dir}")
        print("=" * 60)
        broker.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[CPAT] Scheduler started. Mode={args.scheduler_mode or cfg.live.scheduler_mode}. Press Ctrl+C to stop.\n")
    scheduler.start()
    while scheduler.is_running:
        time.sleep(1)


if __name__ == "__main__":
    main()
