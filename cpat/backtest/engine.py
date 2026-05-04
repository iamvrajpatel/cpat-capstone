"""
CPAT — Backtest Engine v2
==========================
Fully integrated event-driven backtesting loop.

Week 2 changes vs Week 1:
    - Uses DataHandler (forward-only iterator) instead of raw dict access
    - Uses PortfolioManager (extracted from engine) for all state
    - Uses ExecutionEngine v2 with pluggable slippage models
    - Uses SignalOrderTranslator for signal → order conversion
    - Uses RiskEngine for pre-trade constraint checks
    - Uses PerformanceTracker and TradeLog for analytics
    - BacktestResult now includes a full PerformanceReport

Event flow (strict sequential per bar):
    ┌─────────────────────────────────────────────────────┐
    │  get_next() → (ts, bars)                           │
    │                                                     │
    │  1. execution_engine.process_pending(bars)          │
    │     └─> portfolio.apply_fill(fill)                  │
    │     └─> trade_log.record(fill)                      │
    │     └─> perf_tracker.record(ts, equity)             │
    │                                                     │
    │  2. (if past warmup)                               │
    │     queue.put(MarketEvent(bars))                    │
    │                                                     │
    │  3. drain_queue():                                  │
    │     MARKET  → strategy.on_market()                  │
    │             → (strategy emits SignalEvent)           │
    │     SIGNAL  → translator.on_signal_with_bars()      │
    │             → risk.check() → OrderEvent             │
    │     ORDER   → execution_engine.submit()             │
    │     FILL    → (already handled in step 1)           │
    │                                                     │
    │  4. perf_tracker.record(ts, equity)                 │
    └─────────────────────────────────────────────────────┘

Look-ahead bias prevention:
    • Orders at bar T execute at bar T+1's open.
    • Strategies only see bars[0..T] via the MarketEvent.
    • Rolling indicators read only past data (enforced by strategy impls).
    • Risk checks use current prices (bar T close), not future opens.
"""

from __future__ import annotations

import logging
import queue as stdlib_queue
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

import pandas as pd

from cpat.analytics.performance import PerformanceReport, PerformanceTracker
from cpat.analytics.trade_log import TradeLog
from cpat.backtest.event_queue import EventQueue
from cpat.backtest.handlers import (
    FillEventHandler,
    MarketEventHandler,
    OrderEventHandler,
    RiskEventHandler,
    SignalEventHandler,
)
from cpat.config.loader import CPATConfig
from cpat.core.enums import OrderSide
from cpat.core.events import (
    AnyEvent,
    FillEvent,
    MarketEvent,
    OrderEvent,
    RiskEvent,
    SignalEvent,
    SystemEvent,
)
from cpat.core.models import Bar, CostConfig, Fill, Order
from cpat.data.handler import DataHandler
from cpat.execution.engine import ExecutionEngine, build_slippage_model
from cpat.portfolio.manager import PortfolioManager
from cpat.portfolio.translator import SignalOrderTranslator
from cpat.risk.engine import RiskEngine, RiskVerdict

logger = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────────


@dataclass
class BacktestResult:
    """Complete output of a backtest run.

    Attributes:
        equity_curve: Daily portfolio value series.
        daily_returns: Period returns.
        drawdown_series: Rolling drawdown from peak.
        fills: All fills executed.
        orders: All orders submitted.
        report: Full PerformanceReport.
        start_date: Backtest start date.
        end_date: Backtest end date.
        initial_capital: Starting capital.
        final_equity: Ending portfolio value.
        trade_log_df: DataFrame of all fills.
        stats: Arbitrary key-value summary statistics.
    """

    equity_curve: pd.Series
    daily_returns: pd.Series
    drawdown_series: pd.Series
    fills: list[Fill]
    orders: list[Order]
    report: PerformanceReport
    start_date: date
    end_date: date
    initial_capital: float
    final_equity: float
    trade_log_df: pd.DataFrame
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        return self.report.total_return

    @property
    def total_trades(self) -> int:
        return len(self.fills)

    @property
    def sharpe_ratio(self) -> float:
        return self.report.sharpe_ratio

    @property
    def max_drawdown(self) -> float:
        return self.report.max_drawdown


# ── BacktestEngine v2 ──────────────────────────────────────────────────────────


class BacktestEngine:
    """Event-driven backtesting engine (Week 2 — fully integrated).

    Composes DataHandler, PortfolioManager, ExecutionEngine, RiskEngine,
    SignalOrderTranslator, PerformanceTracker, and TradeLog into a
    single, auditable backtesting pipeline.

    Usage::

        engine = BacktestEngine.from_config(config)
        engine.register_strategy(MomentumStrategy(cfg, engine.event_queue))
        result = engine.run(bar_data)
        print(result.report)

    Args:
        config: System configuration.
        cost_config: Transaction cost parameters.
        slippage_model_name: "FIXED_BPS" | "VOLUME_WEIGHTED" | "SPREAD_BASED".
    """

    def __init__(
        self,
        config: CPATConfig,
        cost_config: Optional[CostConfig] = None,
        slippage_model_name: str = "FIXED_BPS",
    ) -> None:
        self._config = config
        self._cost_config = cost_config or CostConfig()
        self._slippage_model_name = slippage_model_name

        # ── Shared event bus ───────────────────────────────────────────────────
        self._event_queue = EventQueue()

        # ── Core components ────────────────────────────────────────────────────
        self._portfolio = PortfolioManager(
            initial_capital=config.backtest.initial_capital,
        )
        self._execution_engine = ExecutionEngine.from_cost_config(
            cost_config=self._cost_config,
            event_queue=self._event_queue,
            slippage_model_name=slippage_model_name,
            bps=self._cost_config.slippage_bps,
        )
        self._risk_engine = RiskEngine(
            portfolio=self._portfolio,
            max_position_pct=config.risk.max_position_pct,
            max_sector_pct=config.risk.max_sector_pct,
            max_gross_exposure_pct=config.risk.max_gross_exposure_pct,
            max_drawdown_pct=config.risk.max_drawdown_pct,
            min_cash_pct=config.risk.min_cash_pct,
        )
        self._translator = SignalOrderTranslator(
            portfolio=self._portfolio,
            event_queue=self._event_queue,
            sizing_method="equal_weight",
            target_weight=config.risk.target_position_weight,
            allow_short=config.strategies.momentum.allow_short,
        )
        self._perf_tracker = PerformanceTracker(
            initial_capital=config.backtest.initial_capital,
            risk_free_rate=getattr(config.risk, "risk_free_rate", 0.04),
        )
        self._trade_log = TradeLog(output_dir="data/results")

        # ── Handler registries (for user-registered strategies) ────────────────
        self._market_handlers: list[MarketEventHandler] = []
        self._signal_handlers: list[SignalEventHandler] = []
        self._order_handlers: list[OrderEventHandler] = []
        self._fill_handlers: list[FillEventHandler] = []
        self._risk_handlers: list[RiskEventHandler] = []

        # ── Audit trail ────────────────────────────────────────────────────────
        self._all_fills: list[Fill] = []
        self._all_orders: list[Order] = []
        self._risk_rejects: int = 0

    @classmethod
    def from_config(cls, config: CPATConfig) -> "BacktestEngine":
        """Factory: build BacktestEngine from a CPATConfig.

        Args:
            config: CPAT system configuration.

        Returns:
            Configured BacktestEngine.
        """
        from cpat.core.enums import CommissionModel, SlippageModel

        cost = CostConfig(
            commission_model=CommissionModel(config.costs.commission.model.value),
            commission_value=config.costs.commission.value,
            min_commission=config.costs.commission.min_commission,
            slippage_model=SlippageModel(config.costs.slippage.model.value),
            slippage_bps=config.costs.slippage.bps,
        )
        slippage_name = getattr(config.costs.slippage, "slippage_model_name", "FIXED_BPS")
        return cls(config=config, cost_config=cost, slippage_model_name=slippage_name)

    # ── Handler registration ───────────────────────────────────────────────────

    def register_market_handler(self, handler: MarketEventHandler) -> None:
        """Register a market event handler (strategy or indicator)."""
        self._market_handlers.append(handler)

    def register_signal_handler(self, handler: SignalEventHandler) -> None:
        """Register an additional signal handler (beyond the built-in translator)."""
        self._signal_handlers.append(handler)

    def register_fill_handler(self, handler: FillEventHandler) -> None:
        """Register a fill handler (e.g. strategy portfolio callback)."""
        self._fill_handlers.append(handler)

    def register_order_handler(self, handler: OrderEventHandler) -> None:
        """Register an order handler (e.g. order logger)."""
        self._order_handlers.append(handler)

    def register_risk_handler(self, handler: RiskEventHandler) -> None:
        """Register a risk event handler."""
        self._risk_handlers.append(handler)

    @property
    def event_queue(self) -> EventQueue:
        """Shared event bus for strategy registration."""
        return self._event_queue

    @property
    def portfolio(self) -> PortfolioManager:
        """Portfolio manager (read access for tests and scripts)."""
        return self._portfolio

    # ── Main backtest loop ─────────────────────────────────────────────────────

    def run(self, bar_data: dict[str, pd.DataFrame]) -> BacktestResult:
        """Execute the full backtest.

        Anti-look-ahead guarantee:
            1. Fills execute at the NEXT bar's open price.
            2. Strategies only receive bars through MarketEvents.
            3. Risk checks use the CURRENT bar's close price.
            4. Rolling indicators use only bars up to and including T.

        Args:
            bar_data: Mapping of symbol → OHLCV DataFrame (UTC DatetimeIndex).

        Returns:
            BacktestResult with equity curve, trade log, and PerformanceReport.
        """
        warmup_bars = self._config.backtest.warmup_bars
        data_handler = DataHandler(bar_data=bar_data, warmup_bars=warmup_bars)

        logger.info(
            "BacktestEngine v2 starting | bars=%d | symbols=%d | capital=%.0f | warmup=%d",
            data_handler.total_bars,
            len(bar_data),
            self._portfolio.initial_capital,
            warmup_bars,
        )

        self._event_queue.put(
            SystemEvent.start(data_handler.timestamps[0] if data_handler.timestamps else pd.Timestamp.now(tz="UTC"))
        )

        bar_idx = 0
        for ts, current_bars in data_handler:
            # ── Step 1: Execute pending orders at this bar's open ──────────────
            exec_result = self._execution_engine.process_pending(current_bars, ts)
            for fill in exec_result.fills:
                self._apply_fill(fill, current_bars)

            # ── Step 2: Emit MarketEvent (skip during warmup) ──────────────────
            if bar_idx >= warmup_bars:
                market_event = MarketEvent.from_bars(current_bars, ts)
                self._event_queue.put(market_event)

            # ── Step 3: Drain event queue ──────────────────────────────────────
            self._drain_queue(current_bars, ts)

            # ── Step 4: Mark-to-market and record equity ───────────────────────
            prices = {sym: bar.close for sym, bar in current_bars.items()}
            equity = self._portfolio.total_equity(prices)
            self._perf_tracker.record(ts, equity)
            self._portfolio.snapshot(ts, prices)
            self._risk_engine.update_equity(equity)

            # Periodic logging
            if bar_idx % 50 == 0:
                open_pos = len(self._portfolio.open_positions)
                logger.debug(
                    "Bar %d [%s] equity=%.0f cash=%.0f positions=%d rejects=%d",
                    bar_idx, ts.date(), equity,
                    self._portfolio.cash, open_pos, self._risk_rejects,
                )

            bar_idx += 1

        # ── Finalise ───────────────────────────────────────────────────────────
        last_prices = self._get_last_prices(bar_data)
        final_equity = self._portfolio.total_equity(last_prices)

        self._event_queue.put(
            SystemEvent.stop(data_handler.timestamps[-1] if data_handler.timestamps else pd.Timestamp.now(tz="UTC"))
        )

        # Compute analytics
        equity_curve = self._perf_tracker.equity_curve()
        daily_returns = self._perf_tracker.daily_returns()
        drawdown_series = self._perf_tracker.drawdown_series()
        report = self._perf_tracker.compute(fills=self._all_fills)
        trade_log_df = self._trade_log.to_dataframe()

        logger.info(
            "BacktestEngine complete | equity=%.0f | return=%.2f%% | sharpe=%.2f | "
            "fills=%d | rejects=%d",
            final_equity,
            report.total_return * 100,
            report.sharpe_ratio,
            len(self._all_fills),
            self._risk_rejects,
        )

        return BacktestResult(
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            drawdown_series=drawdown_series,
            fills=self._all_fills,
            orders=self._all_orders,
            report=report,
            start_date=self._config.backtest.start_date,
            end_date=self._config.backtest.end_date,
            initial_capital=self._portfolio.initial_capital,
            final_equity=final_equity,
            trade_log_df=trade_log_df,
            stats={
                "risk_rejects": self._risk_rejects,
                "total_bars": bar_idx,
                "total_symbols": len(bar_data),
            },
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _apply_fill(self, fill: Fill, current_bars: dict[str, Bar]) -> None:
        """Apply a fill to portfolio and record in trade log."""
        self._portfolio.apply_fill(fill)
        self._trade_log.record(fill)
        self._all_fills.append(fill)
        # Notify fill handlers (strategies that want to track their own fills)
        fill_event = FillEvent.from_fill(fill)
        for handler in self._fill_handlers:
            handler.on_fill(fill_event)

    def _drain_queue(
        self,
        current_bars: dict[str, Bar],
        ts: pd.Timestamp,
    ) -> None:
        """Process all events in the queue until empty.

        Processing order is enforced by the EventQueue's priority:
            1. MarketEvent → strategy on_market → SignalEvent emitted
            2. SignalEvent → translator → RiskEngine → OrderEvent emitted
            3. OrderEvent  → execution_engine.submit (deferred to next bar)
            4. FillEvent   → already handled; notify fill handlers
        """
        max_iterations = 50_000
        iteration = 0

        while not self._event_queue.empty() and iteration < max_iterations:
            iteration += 1
            try:
                event: AnyEvent = self._event_queue.get_nowait()
            except stdlib_queue.Empty:
                break

            match event.event_type.name:
                case "MARKET":
                    assert isinstance(event, MarketEvent)
                    for handler in self._market_handlers:
                        handler.on_market(event)

                case "SIGNAL":
                    assert isinstance(event, SignalEvent)
                    # Built-in translator
                    self._translator.on_signal_with_bars(event, current_bars)
                    # External handlers
                    for handler in self._signal_handlers:
                        handler.on_signal(event)

                case "ORDER":
                    assert isinstance(event, OrderEvent)
                    order = event.order
                    # Risk check
                    prices = {sym: b.close for sym, b in current_bars.items()}
                    risk_decision = self._risk_engine.check(order, prices)

                    if risk_decision.is_approved:
                        # Use potentially reduced quantity
                        if risk_decision.approved_quantity < order.quantity:
                            from cpat.core.models import Order as O
                            from cpat.core.enums import OrderType
                            order = O(
                                symbol=order.symbol,
                                side=order.side,
                                order_type=order.order_type,
                                quantity=risk_decision.approved_quantity,
                                timestamp=order.timestamp,
                                strategy_id=order.strategy_id,
                                signal_id=order.signal_id,
                                limit_price=order.limit_price,
                                stop_price=order.stop_price,
                            )
                        self._all_orders.append(order)
                        self._execution_engine.submit(order)
                        for handler in self._order_handlers:
                            handler.on_order(event)
                    else:
                        self._risk_rejects += 1
                        logger.debug(
                            "Risk rejected: %s %s — %s",
                            order.side.value, order.symbol, risk_decision.reason,
                        )

                case "FILL":
                    assert isinstance(event, FillEvent)
                    # FillEvents from execution are handled in Step 1 (apply_fill)
                    # This branch handles any out-of-band fills
                    for handler in self._fill_handlers:
                        handler.on_fill(event)

                case "RISK":
                    assert isinstance(event, RiskEvent)
                    for handler in self._risk_handlers:
                        handler.on_risk(event)

                case "SYSTEM":
                    logger.debug("System event: %s", getattr(event, "message", ""))

    @staticmethod
    def _get_last_prices(bar_data: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Extract the last close price for each symbol."""
        return {
            sym: float(df["close"].iloc[-1])
            for sym, df in bar_data.items()
            if not df.empty
        }
