"""
CPAT — Backtest Engine
======================
Event-driven backtest engine with an optional Week 4 managed-risk path.

Managed-risk path:
    Signal batch → portfolio guardrails → capital allocation → position sizing
    → final RiskEngine gate → execution → stop registration → portfolio update

Baseline path:
    Preserves the legacy per-signal translation flow for comparison runs.
"""

from __future__ import annotations

import logging
import queue as stdlib_queue
from dataclasses import dataclass, field, replace
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
from cpat.core.enums import OrderSide, SignalDirection
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
from cpat.execution.engine import ExecutionEngine
from cpat.portfolio.allocator import AllocatorFactory
from cpat.portfolio.manager import PortfolioManager
from cpat.portfolio.translator import SignalOrderTranslator
from cpat.risk.engine import RiskEngine
from cpat.risk.position_sizing import PositionSize, PositionSizerFactory
from cpat.risk.risk_manager import PortfolioRiskManager, StopLossTracker

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Complete output of a backtest run."""

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
    portfolio_snapshots_df: pd.DataFrame
    risk_history_df: pd.DataFrame
    trade_risk_df: pd.DataFrame
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


class BacktestEngine:
    """Event-driven backtest engine with baseline and managed-risk modes."""

    def __init__(
        self,
        config: CPATConfig,
        cost_config: Optional[CostConfig] = None,
        slippage_model_name: str = "FIXED_BPS",
        managed_risk: bool = True,
    ) -> None:
        self._config = config
        self._cost_config = cost_config or CostConfig()
        self._slippage_model_name = slippage_model_name
        self._managed_risk = managed_risk

        self._event_queue = EventQueue()
        self._portfolio = PortfolioManager(initial_capital=config.backtest.initial_capital)
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
            target_weight=min(config.risk.max_position_pct, 0.02),
            allow_short=config.strategies.momentum.allow_short,
        )
        self._allocator = AllocatorFactory.build(
            config.risk.allocation_method,
            max_position_pct=config.risk.max_position_pct,
            min_cash_pct=config.risk.min_cash_pct,
            vol_window=config.position_sizing.vol_window,
        )
        self._position_sizer = PositionSizerFactory.build(
            config.position_sizing.method,
            risk_per_trade_pct=config.position_sizing.risk_per_trade_pct,
            stop_method=config.position_sizing.stop_method,
            fixed_stop_pct=config.position_sizing.fixed_stop_pct,
            atr_period=config.position_sizing.atr_period,
            atr_multiplier=config.position_sizing.atr_multiplier,
            take_profit_mult=config.position_sizing.take_profit_mult,
            max_position_pct=config.risk.max_position_pct,
            min_trade_value=config.position_sizing.min_trade_value,
            win_rate=config.position_sizing.kelly_win_rate,
            avg_win=config.position_sizing.kelly_avg_win,
            avg_loss=config.position_sizing.kelly_avg_loss,
            max_kelly_fraction=config.position_sizing.kelly_max_fraction,
        )
        self._portfolio_risk_manager = PortfolioRiskManager(
            max_open_positions=config.risk.max_open_positions,
            max_daily_loss_pct=config.risk.max_daily_loss_pct,
            timezone=config.system.timezone,
        )
        self._stop_loss_tracker = StopLossTracker(
            event_queue=self._event_queue,
            trailing_stop=config.position_sizing.trailing_stop,
        )
        self._perf_tracker = PerformanceTracker(
            initial_capital=config.backtest.initial_capital,
            risk_free_rate=config.risk.risk_free_rate,
        )
        self._trade_log = TradeLog(output_dir="data/results")

        self._market_handlers: list[MarketEventHandler] = []
        self._signal_handlers: list[SignalEventHandler] = []
        self._order_handlers: list[OrderEventHandler] = []
        self._fill_handlers: list[FillEventHandler] = []
        self._risk_handlers: list[RiskEventHandler] = []

        self._all_fills: list[Fill] = []
        self._all_orders: list[Order] = []
        self._risk_rejects: int = 0
        self._submitted_orders: dict[object, Order] = {}
        self._trade_risk_book: dict[str, dict[str, object]] = {}
        self._risk_history_rows: list[dict[str, object]] = []
        self._active_risk_order_by_symbol: dict[str, str] = {}
        self._stop_triggers: int = 0
        self._signal_count: int = 0
        self._warmup_signal_count: int = 0
        self._submitted_order_count: int = 0

    @classmethod
    def from_config(cls, config: CPATConfig, managed_risk: bool = True) -> "BacktestEngine":
        """Factory: build BacktestEngine from a CPATConfig."""
        from cpat.core.enums import CommissionModel, SlippageModel

        cost = CostConfig(
            commission_model=CommissionModel(config.costs.commission.model.value),
            commission_value=config.costs.commission.value,
            min_commission=config.costs.commission.min_commission,
            slippage_model=SlippageModel(config.costs.slippage.model.value),
            slippage_bps=config.costs.slippage.bps,
        )
        slippage_name = getattr(config.costs.slippage, "slippage_model_name", "FIXED_BPS")
        return cls(
            config=config,
            cost_config=cost,
            slippage_model_name=slippage_name,
            managed_risk=managed_risk,
        )

    def register_market_handler(self, handler: MarketEventHandler) -> None:
        self._market_handlers.append(handler)

    def register_signal_handler(self, handler: SignalEventHandler) -> None:
        self._signal_handlers.append(handler)

    def register_fill_handler(self, handler: FillEventHandler) -> None:
        self._fill_handlers.append(handler)

    def register_order_handler(self, handler: OrderEventHandler) -> None:
        self._order_handlers.append(handler)

    def register_risk_handler(self, handler: RiskEventHandler) -> None:
        self._risk_handlers.append(handler)

    @property
    def event_queue(self) -> EventQueue:
        return self._event_queue

    @property
    def portfolio(self) -> PortfolioManager:
        return self._portfolio

    def run(self, bar_data: dict[str, pd.DataFrame]) -> BacktestResult:
        """Execute the backtest."""
        self._reset_run_state()
        warmup_bars = self._config.backtest.warmup_bars
        data_handler = DataHandler(bar_data=bar_data, warmup_bars=warmup_bars)

        logger.info(
            "BacktestEngine starting | mode=%s | bars=%d | symbols=%d | capital=%.0f",
            "managed" if self._managed_risk else "baseline",
            data_handler.total_bars,
            len(bar_data),
            self._portfolio.initial_capital,
        )

        self._event_queue.put(
            SystemEvent.start(
                data_handler.timestamps[0] if data_handler.timestamps else pd.Timestamp.now(tz="UTC")
            )
        )

        bar_idx = 0
        for ts, current_bars in data_handler:
            exec_result = self._execution_engine.process_pending(current_bars, ts)
            for fill in exec_result.fills:
                self._apply_fill(fill)

            prices = {sym: bar.close for sym, bar in current_bars.items()}
            current_equity = self._portfolio.total_equity(prices)
            self._portfolio_risk_manager.update_daily_equity(ts, current_equity)

            stop_symbols: set[str] = set()
            if self._managed_risk:
                stop_symbols = set(self._stop_loss_tracker.check_stops(current_bars, ts))
                self._stop_triggers += len(stop_symbols)

            in_warmup = bar_idx < warmup_bars
            self._event_queue.put(MarketEvent.from_bars(current_bars, ts))

            self._drain_queue(
                current_bars=current_bars,
                ts=ts,
                data_handler=data_handler,
                stop_symbols=stop_symbols,
                in_warmup=in_warmup,
            )

            prices = {sym: bar.close for sym, bar in current_bars.items()}
            equity = self._portfolio.total_equity(prices)
            self._perf_tracker.record(ts, equity)
            self._portfolio.snapshot(ts, prices)
            self._risk_engine.update_equity(equity)
            self._record_risk_snapshot(ts, equity)

            if bar_idx % 50 == 0:
                logger.debug(
                    "Bar %d [%s] equity=%.0f cash=%.0f positions=%d rejects=%d stops=%d",
                    bar_idx,
                    ts.date(),
                    equity,
                    self._portfolio.cash,
                    len(self._portfolio.open_positions),
                    self._risk_rejects,
                    self._stop_triggers,
                )

            bar_idx += 1

        last_prices = self._get_last_prices(bar_data)
        final_equity = self._portfolio.total_equity(last_prices)
        self._event_queue.put(
            SystemEvent.stop(
                data_handler.timestamps[-1] if data_handler.timestamps else pd.Timestamp.now(tz="UTC")
            )
        )

        equity_curve = self._perf_tracker.equity_curve()
        daily_returns = self._perf_tracker.daily_returns()
        drawdown_series = self._perf_tracker.drawdown_series()
        report = self._perf_tracker.compute(fills=self._all_fills)
        trade_log_df = self._trade_log.to_dataframe()
        portfolio_snapshots_df = self._portfolio.snapshots_df()
        risk_history_df = pd.DataFrame(self._risk_history_rows).set_index("timestamp") if self._risk_history_rows else pd.DataFrame()
        trade_risk_df = pd.DataFrame(self._trade_risk_book.values()).set_index("order_id") if self._trade_risk_book else pd.DataFrame()

        max_gross_exposure = (
            float(portfolio_snapshots_df["gross_exposure"].max())
            if not portfolio_snapshots_df.empty
            else 0.0
        )
        logger.info(
            "Backtest complete | equity=%.0f | sharpe=%.2f | max_dd=%.2f%% | rejects=%d | stops=%d",
            final_equity,
            report.sharpe_ratio,
            report.max_drawdown * 100,
            self._risk_rejects,
            self._stop_triggers,
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
            portfolio_snapshots_df=portfolio_snapshots_df,
            risk_history_df=risk_history_df,
            trade_risk_df=trade_risk_df,
            stats={
                "risk_rejects": self._risk_rejects,
                "stop_triggers": self._stop_triggers,
                "total_bars": bar_idx,
                "total_symbols": len(bar_data),
                "max_gross_exposure": max_gross_exposure,
                "managed_risk": self._managed_risk,
                "warmup_bars": warmup_bars,
                "signal_count": self._signal_count,
                "warmup_signal_count": self._warmup_signal_count,
                "order_count": self._submitted_order_count,
                "fill_count": len(self._all_fills),
            },
        )

    def _reset_run_state(self) -> None:
        self._all_fills.clear()
        self._all_orders.clear()
        self._submitted_orders.clear()
        self._trade_risk_book.clear()
        self._risk_history_rows.clear()
        self._active_risk_order_by_symbol.clear()
        self._risk_rejects = 0
        self._stop_triggers = 0
        self._signal_count = 0
        self._warmup_signal_count = 0
        self._submitted_order_count = 0

    def _apply_fill(self, fill: Fill) -> None:
        self._portfolio.apply_fill(fill)
        self._trade_log.record(fill)
        self._all_fills.append(fill)

        order = self._submitted_orders.get(fill.order_id)
        if order is not None:
            self._mark_trade_risk_filled(fill, order)
            if self._managed_risk:
                self._sync_stop_state(fill, order)

        fill_event = FillEvent.from_fill(fill)
        for handler in self._fill_handlers:
            handler.on_fill(fill_event)

    def _drain_queue(
        self,
        current_bars: dict[str, Bar],
        ts: pd.Timestamp,
        data_handler: DataHandler,
        stop_symbols: set[str],
        in_warmup: bool,
    ) -> None:
        pending_signals: list[SignalEvent] = []
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
                    self._signal_count += 1
                    if in_warmup:
                        self._warmup_signal_count += 1
                    if self._managed_risk and not in_warmup:
                        pending_signals.append(event)
                    elif not self._managed_risk and not in_warmup:
                        self._translator.on_signal_with_bars(event, current_bars)
                    for handler in self._signal_handlers:
                        handler.on_signal(event)

                case "ORDER":
                    assert isinstance(event, OrderEvent)
                    if not in_warmup:
                        self._route_order(event.order, current_bars)

                case "FILL":
                    # Fills from the execution engine are applied at the start of the bar.
                    continue

                case "RISK":
                    assert isinstance(event, RiskEvent)
                    for handler in self._risk_handlers:
                        handler.on_risk(event)

                case "SYSTEM":
                    logger.debug("System event: %s", getattr(event, "message", ""))

        if self._managed_risk and pending_signals and not in_warmup:
            self._process_signal_batch(
                signal_events=pending_signals,
                current_bars=current_bars,
                ts=ts,
                data_handler=data_handler,
                stop_symbols=stop_symbols,
            )

    def _process_signal_batch(
        self,
        signal_events: list[SignalEvent],
        current_bars: dict[str, Bar],
        ts: pd.Timestamp,
        data_handler: DataHandler,
        stop_symbols: set[str],
    ) -> None:
        latest_by_symbol: dict[str, SignalEvent] = {}
        stop_events: dict[str, SignalEvent] = {}

        for event in signal_events:
            symbol = event.signal.symbol
            latest_by_symbol[symbol] = event
            if event.signal.metadata.get("source") == "stop_loss_tracker":
                stop_events[symbol] = event

        for symbol, stop_event in stop_events.items():
            latest_by_symbol[symbol] = stop_event

        close_orders: list[Order] = []
        opening_events: list[SignalEvent] = []
        for event in latest_by_symbol.values():
            signal = event.signal
            symbol = signal.symbol
            position = self._portfolio.get_position(symbol)

            if symbol in stop_symbols or signal.direction == SignalDirection.FLAT:
                close_order = self._translator.build_close_order(
                    symbol=symbol,
                    timestamp=ts,
                    strategy_id=signal.strategy_id,
                    signal_id=signal.signal_id,
                )
                if close_order is not None:
                    close_orders.append(close_order)
                    if signal.metadata.get("reason") and symbol in self._active_risk_order_by_symbol:
                        order_id = self._active_risk_order_by_symbol[symbol]
                        if order_id in self._trade_risk_book:
                            self._trade_risk_book[order_id]["stop_trigger_reason"] = signal.metadata["reason"]
                continue

            if signal.direction == SignalDirection.SHORT:
                if position.is_long:
                    close_order = self._translator.build_close_order(
                        symbol=symbol,
                        timestamp=ts,
                        strategy_id=signal.strategy_id,
                        signal_id=signal.signal_id,
                    )
                    if close_order is not None:
                        close_orders.append(close_order)
                continue

            if position.is_long:
                continue

            if position.is_short:
                close_order = self._translator.build_close_order(
                    symbol=symbol,
                    timestamp=ts,
                    strategy_id=signal.strategy_id,
                    signal_id=signal.signal_id,
                )
                if close_order is not None:
                    close_orders.append(close_order)
                continue

            opening_events.append(event)

        for order in close_orders:
            self._route_order(order, current_bars)

        if not opening_events:
            return

        prices = {sym: bar.close for sym, bar in current_bars.items()}
        equity = self._portfolio.total_equity(prices)
        allowed, reason = self._portfolio_risk_manager.can_open_new_position(self._portfolio, equity)
        if not allowed:
            logger.debug("Portfolio risk gate blocked new positions: %s", reason)
            return

        open_slots = max(0, self._config.risk.max_open_positions - len(self._portfolio.open_positions))
        if open_slots <= 0:
            return

        opening_events = opening_events[:open_slots]
        history_map = {
            event.signal.symbol: self._history_frame_for_symbol(data_handler, event.signal.symbol, ts)
            for event in opening_events
        }
        allocation_slices = self._allocator.allocate(
            signals=[event.signal.symbol for event in opening_events],
            equity=equity,
            cash=self._portfolio.cash,
            bar_data=history_map,
        )
        allocation_map = {capital_slice.symbol: capital_slice for capital_slice in allocation_slices}

        for event in opening_events:
            signal = event.signal
            bar = current_bars.get(signal.symbol)
            capital_slice = allocation_map.get(signal.symbol)
            if bar is None or capital_slice is None or capital_slice.target_value <= 0:
                continue

            size = self._position_sizer.compute(
                symbol=signal.symbol,
                entry_price=float(bar.close),
                equity=equity,
                bar_data=history_map[signal.symbol],
                capital_budget=capital_slice.target_value,
                direction=signal.direction,
            )
            if not size.is_valid:
                continue

            orders = self._translator.build_orders(
                symbol=signal.symbol,
                direction=signal.direction,
                timestamp=ts,
                strategy_id=signal.strategy_id,
                signal_id=signal.signal_id,
                position_size=size,
            )
            for order in orders:
                metadata = dict(order.metadata)
                metadata["allocation_weight"] = capital_slice.weight
                order = replace(order, metadata=metadata)
                self._record_planned_trade(order, size, capital_slice.weight)
                self._route_order(order, current_bars)

    def _route_order(self, order: Order, current_bars: dict[str, Bar]) -> None:
        prices = {sym: bar.close for sym, bar in current_bars.items()}
        risk_decision = self._risk_engine.check(order, prices)

        if not risk_decision.is_approved:
            self._risk_rejects += 1
            self._record_trade_rejection(order, risk_decision.reason)
            logger.debug("Risk rejected: %s %s — %s", order.side.value, order.symbol, risk_decision.reason)
            return

        if risk_decision.approved_quantity < order.quantity:
            scale = risk_decision.approved_quantity / max(order.quantity, 1.0)
            metadata = dict(order.metadata)
            if "planned_risk" in metadata:
                metadata["planned_risk"] = float(metadata["planned_risk"]) * scale
            order = replace(
                order,
                quantity=risk_decision.approved_quantity,
                metadata=metadata,
            )

        self._submitted_orders[order.order_id] = order
        self._all_orders.append(order)
        self._submitted_order_count += 1
        self._execution_engine.submit(order)

        order_event = OrderEvent.from_order(order)
        for handler in self._order_handlers:
            handler.on_order(order_event)

    def _sync_stop_state(self, fill: Fill, order: Order) -> None:
        position = self._portfolio.get_position(fill.symbol)
        stop_price = order.metadata.get("stop_price")
        take_profit = order.metadata.get("take_profit_price")
        is_closing = bool(order.metadata.get("closing_order"))

        if is_closing or position.is_flat or not position.is_long:
            self._stop_loss_tracker.deregister(fill.symbol)
            self._active_risk_order_by_symbol.pop(fill.symbol, None)
            return

        if fill.side == OrderSide.BUY and stop_price:
            self._stop_loss_tracker.register(
                symbol=fill.symbol,
                entry_price=fill.fill_price,
                stop_price=float(stop_price),
                strategy_id=order.strategy_id,
                take_profit_price=float(take_profit) if take_profit is not None else None,
                trailing_stop=self._config.position_sizing.trailing_stop,
            )
            self._active_risk_order_by_symbol[fill.symbol] = str(order.order_id)

    def _record_planned_trade(self, order: Order, size: PositionSize, allocation_weight: float) -> None:
        self._trade_risk_book[str(order.order_id)] = {
            "order_id": str(order.order_id),
            "signal_id": str(order.signal_id) if order.signal_id else "",
            "symbol": order.symbol,
            "side": order.side.value,
            "planned_entry": size.entry_price,
            "stop_price": size.stop_price,
            "take_profit_price": size.take_profit_price,
            "planned_risk": size.total_risk,
            "risk_per_share": size.risk_per_share,
            "atr": size.atr,
            "quantity": order.quantity,
            "sizing_method": size.sizing_method,
            "allocation_weight": allocation_weight,
            "capital_budget": size.capital_budget,
            "status": "planned",
            "strategy_id": order.strategy_id,
            "stop_trigger_reason": "",
        }

    def _record_trade_rejection(self, order: Order, reason: str) -> None:
        order_id = str(order.order_id)
        if order_id not in self._trade_risk_book:
            self._trade_risk_book[order_id] = {
                "order_id": order_id,
                "signal_id": str(order.signal_id) if order.signal_id else "",
                "symbol": order.symbol,
                "side": order.side.value,
                "planned_entry": order.metadata.get("entry_price"),
                "stop_price": order.metadata.get("stop_price"),
                "take_profit_price": order.metadata.get("take_profit_price"),
                "planned_risk": order.metadata.get("planned_risk", 0.0),
                "risk_per_share": order.metadata.get("risk_per_share", 0.0),
                "atr": order.metadata.get("atr", 0.0),
                "quantity": order.quantity,
                "sizing_method": order.metadata.get("sizing_method", ""),
                "allocation_weight": order.metadata.get("allocation_weight", 0.0),
                "capital_budget": order.metadata.get("capital_budget", 0.0),
                "status": "rejected",
                "strategy_id": order.strategy_id,
                "stop_trigger_reason": "",
            }
        self._trade_risk_book[order_id]["status"] = "rejected"
        self._trade_risk_book[order_id]["rejection_reason"] = reason

    def _mark_trade_risk_filled(self, fill: Fill, order: Order) -> None:
        order_id = str(order.order_id)
        if order_id not in self._trade_risk_book:
            return
        self._trade_risk_book[order_id]["status"] = "filled"
        self._trade_risk_book[order_id]["fill_price"] = fill.fill_price
        self._trade_risk_book[order_id]["fill_timestamp"] = fill.timestamp

    def _record_risk_snapshot(self, timestamp: pd.Timestamp, equity: float) -> None:
        snapshot_df = self._portfolio.snapshots_df()
        latest = snapshot_df.iloc[-1] if not snapshot_df.empty else None
        peak_equity = max(getattr(self._risk_engine, "_peak_equity", equity), 1e-9)
        drawdown = (equity - peak_equity) / peak_equity
        self._risk_history_rows.append({
            "timestamp": timestamp,
            "equity": equity,
            "cash": self._portfolio.cash,
            "gross_exposure": float(latest["gross_exposure"]) if latest is not None else 0.0,
            "net_exposure": float(latest["net_exposure"]) if latest is not None else 0.0,
            "long_exposure": float(latest["long_exposure"]) if latest is not None else 0.0,
            "short_exposure": float(latest["short_exposure"]) if latest is not None else 0.0,
            "open_positions": int(latest["n_open_positions"]) if latest is not None else 0,
            "drawdown": drawdown,
            "drawdown_halt_active": self._risk_engine.drawdown_halt_active,
            "daily_halt_active": self._portfolio_risk_manager.daily_halt_active,
            "risk_rejects": self._risk_rejects,
            "stop_triggers": self._stop_triggers,
        })

    def _history_frame_for_symbol(
        self,
        data_handler: DataHandler,
        symbol: str,
        timestamp: pd.Timestamp,
    ) -> pd.DataFrame:
        df = data_handler.bar_data.get(symbol)
        if df is None or df.empty:
            return pd.DataFrame()
        return df.loc[df.index <= timestamp].copy()

    @staticmethod
    def _get_last_prices(bar_data: dict[str, pd.DataFrame]) -> dict[str, float]:
        return {
            sym: float(df["close"].iloc[-1])
            for sym, df in bar_data.items()
            if not df.empty
        }
