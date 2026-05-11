"""
CPAT — Live Execution Engine (Week 5)
=====================================
Runs the Week 4 managed-risk flow in paper/live mode using the existing
event-driven strategy contract.
"""

from __future__ import annotations

import logging
import queue as stdlib_queue
import time
from dataclasses import replace
from typing import Callable, Optional

import pandas as pd

from cpat.backtest.event_queue import EventQueue
from cpat.brokers.paper import PaperBroker
from cpat.core.enums import EventType, OrderSide, OrderStatus, OrderType, SignalDirection
from cpat.core.events import FillEvent, MarketEvent, SignalEvent
from cpat.core.models import Bar, Fill, Order
from cpat.infrastructure.broker_interface import (
    BrokerConnectionError,
    BrokerError,
    BrokerInterface,
    BrokerOrderError,
)
from cpat.infrastructure.live_logger import LiveTradeLogger
from cpat.infrastructure.order_manager import ManagedOrder, OrderManager
from cpat.portfolio.allocator import CapitalAllocator
from cpat.portfolio.manager import PortfolioManager
from cpat.portfolio.translator import SignalOrderTranslator
from cpat.risk.engine import RiskEngine
from cpat.risk.position_sizing import PositionSizer
from cpat.risk.risk_manager import PortfolioRiskManager, StopLossTracker
from cpat.strategies.base import AbstractStrategy

logger = logging.getLogger(__name__)


class LiveExecutionEngine:
    """Candle-driven paper/live engine that reuses the Week 4 risk path."""

    def __init__(
        self,
        broker: BrokerInterface,
        strategy: AbstractStrategy,
        portfolio: PortfolioManager,
        risk_engine: RiskEngine,
        position_sizer: PositionSizer,
        allocator: CapitalAllocator,
        stop_tracker: StopLossTracker,
        portfolio_risk_manager: PortfolioRiskManager,
        data_provider: Callable[[list[str]], dict[str, pd.DataFrame]],
        symbols: list[str],
        trade_logger: LiveTradeLogger,
        oms: OrderManager,
        event_queue: Optional[EventQueue] = None,
        dry_run: bool = False,
        stale_data_seconds: int = 300,
        order_status_poll_interval_seconds: int = 5,
        order_ttl_seconds: int = 300,
        max_order_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        reconcile_on_startup: bool = True,
    ) -> None:
        self._broker = broker
        self._strategy = strategy
        self._portfolio = portfolio
        self._risk = risk_engine
        self._sizer = position_sizer
        self._allocator = allocator
        self._stop_tracker = stop_tracker
        self._port_risk = portfolio_risk_manager
        self._data_fn = data_provider
        self._symbols = symbols
        self._tlog = trade_logger
        self._oms = oms
        self._dry_run = dry_run
        self._stale_data_seconds = stale_data_seconds
        self._order_status_poll_interval_seconds = order_status_poll_interval_seconds
        self._order_ttl_seconds = order_ttl_seconds
        self._max_order_retries = max_order_retries
        self._retry_backoff_seconds = retry_backoff_seconds

        self._event_queue = event_queue or getattr(strategy, "_event_queue", None) or EventQueue()
        self._translator = SignalOrderTranslator(
            portfolio=portfolio,
            event_queue=self._event_queue,
            allow_short=False,
        )

        self._tick_count = 0
        self._orders_placed = 0
        self._orders_rejected = 0
        self._fills_received = 0
        self._api_errors = 0
        self._retry_count = 0
        self._stale_data_skips = 0
        self._cancelled_orders = 0
        self._expired_orders = 0
        self._last_status_poll_at: Optional[pd.Timestamp] = None
        self._latest_prices: dict[str, float] = {}
        self._reconciliation_ok = True
        self._reconciliation_reason = "not_run"
        self._session_start = pd.Timestamp.utcnow()

        if reconcile_on_startup:
            self._reconcile_broker_state()

        self._tlog.system(
            "LiveExecutionEngine initialised",
            dry_run=dry_run,
            symbols=len(symbols),
            stale_data_seconds=stale_data_seconds,
        )

    def run_tick(self) -> None:
        """Execute one full scheduler tick without raising."""
        self._tick_count += 1
        tick_ts = pd.Timestamp.utcnow()

        try:
            self._ensure_connected()
            raw_bar_data = self._fetch_prices()
            bar_data = self._filter_fresh_bar_data(raw_bar_data, tick_ts)
            if not bar_data:
                self._emit_heartbeat(0.0, {}, tick_ts)
                return

            current_bars = self._latest_bars(bar_data)
            if not current_bars:
                self._emit_heartbeat(0.0, {}, tick_ts)
                return

            current_prices = {
                symbol: float(bar.close)
                for symbol, bar in current_bars.items()
                if bar.close > 0
            }
            self._latest_prices = current_prices
            self._sync_paper_prices(current_prices)

            self._poll_order_updates(current_prices, tick_ts, force=True)
            self._expire_stale_orders(tick_ts)

            equity = self._portfolio.total_equity(current_prices)
            self._risk.update_equity(equity)
            self._port_risk.update_daily_equity(tick_ts, equity)

            stop_symbols = set(self._stop_tracker.check_stops(current_bars, tick_ts))
            signal_events = self._collect_signal_events(current_bars, tick_ts)
            self._reconcile_broker_state(current_prices)
            self._process_signal_batch(signal_events, current_bars, bar_data, tick_ts, stop_symbols)
            self._poll_order_updates(current_prices, tick_ts)

            final_equity = self._portfolio.total_equity(current_prices)
            self._risk.update_equity(final_equity)
            self._emit_heartbeat(final_equity, current_prices, tick_ts)

        except Exception as exc:  # noqa: BLE001
            self._api_errors += 1
            self._tlog.error("Unhandled tick error", exc=exc)
            logger.error("[live_engine] Tick #%d error: %s", self._tick_count, exc, exc_info=True)

    def summary(self) -> dict:
        """Return session-level live execution statistics."""
        elapsed = (pd.Timestamp.utcnow() - self._session_start).total_seconds()
        return {
            "tick_count": self._tick_count,
            "orders_placed": self._orders_placed,
            "orders_rejected": self._orders_rejected,
            "fills_received": self._fills_received,
            "api_errors": self._api_errors,
            "retry_count": self._retry_count,
            "stale_data_skips": self._stale_data_skips,
            "cancelled_orders": self._cancelled_orders,
            "expired_orders": self._expired_orders,
            "broker_connected": self._broker.is_connected,
            "reconciliation_ok": self._reconciliation_ok,
            "reconciliation_reason": self._reconciliation_reason,
            "elapsed_seconds": round(elapsed, 1),
            "session_start": str(self._session_start),
            "oms_summary": self._oms.summary(),
        }

    def _ensure_connected(self) -> None:
        if not self._broker.is_connected:
            self._broker.connect()
            self._tlog.system("Broker reconnected")

    def _fetch_prices(self) -> dict[str, pd.DataFrame]:
        try:
            return self._data_fn(self._symbols)
        except Exception as exc:  # noqa: BLE001
            self._api_errors += 1
            self._tlog.error("Data fetch failed", exc=exc)
            return {}

    def _filter_fresh_bar_data(
        self,
        bar_data: dict[str, pd.DataFrame],
        tick_ts: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        valid: dict[str, pd.DataFrame] = {}
        for symbol in self._symbols:
            df = bar_data.get(symbol)
            if df is None or df.empty:
                self._stale_data_skips += 1
                self._tlog.data_issue(symbol, "missing_data")
                continue

            latest_ts = df.index[-1]
            if latest_ts.tzinfo is None:
                latest_ts = latest_ts.tz_localize("UTC")
                df = df.copy()
                df.index = pd.DatetimeIndex(
                    [idx.tz_localize("UTC") if idx.tzinfo is None else idx for idx in df.index]
                )

            if self._stale_data_seconds > 0:
                age = abs((tick_ts - latest_ts).total_seconds())
                if age > self._stale_data_seconds:
                    self._stale_data_skips += 1
                    self._tlog.data_issue(symbol, "stale_data", age_seconds=round(age, 1))
                    continue

            if "adj_close" not in df.columns:
                df = df.copy()
                df["adj_close"] = df["close"]
            valid[symbol] = df
        return valid

    def _latest_bars(self, bar_data: dict[str, pd.DataFrame]) -> dict[str, Bar]:
        bars: dict[str, Bar] = {}
        for symbol, df in bar_data.items():
            row = df.iloc[-1]
            ts = df.index[-1]
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            bars[symbol] = Bar(
                symbol=symbol,
                timestamp=ts,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0)),
                adj_close=float(row.get("adj_close", row["close"])),
            )
        return bars

    def _sync_paper_prices(self, current_prices: dict[str, float]) -> None:
        if isinstance(self._broker, PaperBroker):
            self._broker.update_prices(current_prices)

    def _collect_signal_events(
        self,
        current_bars: dict[str, Bar],
        tick_ts: pd.Timestamp,
    ) -> list[SignalEvent]:
        self._event_queue.put(MarketEvent.from_bars(current_bars, tick_ts))
        pending: list[SignalEvent] = []
        max_iterations = 10_000
        iterations = 0

        while not self._event_queue.empty() and iterations < max_iterations:
            iterations += 1
            try:
                event = self._event_queue.get_nowait()
            except stdlib_queue.Empty:
                break

            if event.event_type == EventType.MARKET:
                self._strategy.on_market(event)
            elif event.event_type == EventType.SIGNAL:
                pending.append(event)

        return pending

    def _process_signal_batch(
        self,
        signal_events: list[SignalEvent],
        current_bars: dict[str, Bar],
        bar_data: dict[str, pd.DataFrame],
        tick_ts: pd.Timestamp,
        stop_symbols: set[str],
    ) -> None:
        if not signal_events:
            return

        latest_by_symbol: dict[str, SignalEvent] = {}
        stop_events: dict[str, SignalEvent] = {}
        for event in signal_events:
            latest_by_symbol[event.signal.symbol] = event
            if event.signal.metadata.get("source") == "stop_loss_tracker":
                stop_events[event.signal.symbol] = event
        latest_by_symbol.update(stop_events)

        close_orders: list[Order] = []
        opening_events: list[SignalEvent] = []

        for event in latest_by_symbol.values():
            signal = event.signal
            position = self._portfolio.get_position(signal.symbol)

            if signal.symbol in stop_symbols or signal.direction == SignalDirection.FLAT:
                close_order = self._translator.build_close_order(
                    symbol=signal.symbol,
                    timestamp=tick_ts,
                    strategy_id=signal.strategy_id,
                    signal_id=signal.signal_id,
                )
                if close_order is not None:
                    close_orders.append(close_order)
                continue

            if signal.direction == SignalDirection.SHORT:
                if position.is_long:
                    close_order = self._translator.build_close_order(
                        symbol=signal.symbol,
                        timestamp=tick_ts,
                        strategy_id=signal.strategy_id,
                        signal_id=signal.signal_id,
                    )
                    if close_order is not None:
                        close_orders.append(close_order)
                continue

            if position.is_long:
                continue

            opening_events.append(event)

        for order in close_orders:
            self._route_order(order, current_bars)

        if not opening_events:
            return

        prices = {symbol: bar.close for symbol, bar in current_bars.items()}
        equity = self._portfolio.total_equity(prices)

        if not self._reconciliation_ok:
            for event in opening_events:
                self._tlog.risk_block(
                    event.signal.symbol,
                    "reconciliation",
                    self._reconciliation_reason,
                )
            return

        allowed, reason = self._port_risk.can_open_new_position(self._portfolio, equity)
        if not allowed:
            for event in opening_events:
                self._tlog.risk_block(event.signal.symbol, "portfolio_risk", reason)
            return

        open_slots = max(
            0,
            self._port_risk.max_open_positions - len(self._portfolio.open_positions),
        )
        if open_slots <= 0:
            return

        selected_events = opening_events[:open_slots]
        allocation_slices = self._allocator.allocate(
            signals=[event.signal.symbol for event in selected_events],
            equity=equity,
            cash=self._portfolio.cash,
            bar_data=bar_data,
        )
        allocation_map = {capital_slice.symbol: capital_slice for capital_slice in allocation_slices}

        for event in selected_events:
            signal = event.signal
            bar = current_bars.get(signal.symbol)
            capital_slice = allocation_map.get(signal.symbol)
            if bar is None or capital_slice is None or capital_slice.target_value <= 0:
                continue

            size = self._sizer.compute(
                symbol=signal.symbol,
                entry_price=float(bar.close),
                equity=equity,
                bar_data=bar_data.get(signal.symbol, pd.DataFrame()),
                capital_budget=capital_slice.target_value,
                direction=signal.direction,
            )
            if not size.is_valid:
                continue

            orders = self._translator.build_orders(
                symbol=signal.symbol,
                direction=signal.direction,
                timestamp=tick_ts,
                strategy_id=signal.strategy_id,
                signal_id=signal.signal_id,
                position_size=size,
            )
            for order in orders:
                metadata = dict(order.metadata)
                metadata["allocation_weight"] = capital_slice.weight
                order = replace(order, metadata=metadata)
                self._route_order(order, current_bars)

    def _route_order(self, order: Order, current_bars: dict[str, Bar]) -> None:
        prices = {symbol: bar.close for symbol, bar in current_bars.items()}
        risk_decision = self._risk.check(order, prices)
        if not risk_decision.is_approved:
            self._orders_rejected += 1
            self._tlog.risk_block(order.symbol, risk_decision.constraint_name or "risk_engine", risk_decision.reason)
            return

        if risk_decision.approved_quantity < order.quantity:
            scale = risk_decision.approved_quantity / max(order.quantity, 1.0)
            metadata = dict(order.metadata)
            if "planned_risk" in metadata:
                metadata["planned_risk"] = float(metadata["planned_risk"]) * scale
            order = replace(order, quantity=risk_decision.approved_quantity, metadata=metadata)

        self._submit_order(order)

    def _submit_order(self, order: Order) -> None:
        managed = self._oms.create(order)

        for attempt in range(self._max_order_retries + 1):
            try:
                if self._dry_run:
                    dry_broker_id = f"DRY-RUN-{order.order_id.hex[:8].upper()}"
                    self._oms.mark_submitted(order.order_id, broker_order_id=dry_broker_id)
                    self._orders_placed += 1
                    self._tlog.order_placed(
                        order.symbol,
                        qty=order.quantity,
                        side=order.side.value,
                        broker_order_id=dry_broker_id,
                        dry_run=True,
                    )
                    return

                broker_id = self._broker.place_order(order)
                self._oms.mark_submitted(order.order_id, broker_order_id=broker_id)
                self._orders_placed += 1
                self._tlog.order_placed(
                    order.symbol,
                    qty=order.quantity,
                    side=order.side.value,
                    broker_order_id=broker_id,
                    order_type=order.order_type.value,
                )
                return

            except BrokerOrderError as exc:
                self._oms.mark_rejected(order.order_id, reason=str(exc))
                self._orders_rejected += 1
                self._tlog.order_rejected(order.symbol, reason=str(exc))
                return

            except BrokerConnectionError as exc:
                self._retry_count += 1
                retry_no = self._oms.increment_retry(order.order_id)
                self._tlog.order_retry(
                    order.symbol,
                    attempt=retry_no,
                    max_attempts=self._max_order_retries + 1,
                    reason=str(exc),
                )
                if attempt < self._max_order_retries:
                    time.sleep(self._retry_backoff_seconds * (2 ** attempt))
                else:
                    self._oms.mark_rejected(order.order_id, reason=f"connection_error: {exc}")
                    self._orders_rejected += 1
                continue

            except Exception as exc:  # noqa: BLE001
                self._oms.mark_rejected(order.order_id, reason=str(exc))
                self._orders_rejected += 1
                self._tlog.error("Unexpected order submission error", exc=exc, symbol=order.symbol)
                return

    def _poll_order_updates(
        self,
        current_prices: dict[str, float],
        tick_ts: pd.Timestamp,
        force: bool = False,
    ) -> None:
        if isinstance(self._broker, PaperBroker):
            self._drain_paper_fills()
            return

        if (
            not force
            and self._last_status_poll_at is not None
            and (tick_ts - self._last_status_poll_at).total_seconds()
            < self._order_status_poll_interval_seconds
        ):
            return
        self._last_status_poll_at = tick_ts

        for managed in self._oms.pending_orders:
            if not managed.broker_order_id:
                continue
            try:
                status = self._broker.get_order_status(managed.broker_order_id)
                prev_filled = managed.filled_qty
                updated = self._oms.apply_broker_status(status)
                if updated is None:
                    continue

                if status.filled_qty > prev_filled:
                    fill = Fill(
                        order_id=updated.order_id,
                        symbol=updated.order.symbol,
                        side=updated.order.side,
                        quantity=status.filled_qty - prev_filled,
                        fill_price=status.avg_fill_price,
                        commission=0.0,
                        slippage=0.0,
                        timestamp=status.timestamp,
                        exchange="DHAN",
                    )
                    self._apply_fill(fill, updated.order, managed.broker_order_id)

                if updated.status == OrderStatus.CANCELLED:
                    self._cancelled_orders += 1
                    self._tlog.order_cancelled(updated.order.symbol, broker_order_id=updated.broker_order_id)
                elif updated.status == OrderStatus.REJECTED:
                    self._orders_rejected += 1
                    self._tlog.order_rejected(updated.order.symbol, updated.rejection_reason, broker_order_id=updated.broker_order_id)
                elif updated.status == OrderStatus.EXPIRED:
                    self._expired_orders += 1

            except BrokerError as exc:
                self._api_errors += 1
                self._tlog.error("Order status poll failed", exc=exc, symbol=managed.order.symbol)

    def _drain_paper_fills(self) -> None:
        for fill in self._broker.get_fill_log():
            managed = self._oms.get(fill.order_id)
            if managed is None:
                continue
            if managed.filled_qty >= fill.quantity and managed.status == OrderStatus.FILLED:
                continue
            self._oms.apply_fill(
                fill.order_id,
                filled_qty=max(fill.quantity - managed.filled_qty, 0.0),
                avg_price=fill.fill_price,
                fill=fill,
            )
            self._apply_fill(fill, managed.order, managed.broker_order_id)

    def _apply_fill(self, fill: Fill, order: Order, broker_order_id: str = "") -> None:
        self._portfolio.apply_fill(fill)
        self._fills_received += 1
        self._tlog.fill(
            fill.symbol,
            qty=fill.quantity,
            price=fill.fill_price,
            side=fill.side.value,
            commission=fill.commission,
            broker_order_id=broker_order_id,
        )
        try:
            self._strategy.on_fill(FillEvent.from_fill(fill))
        except Exception as exc:  # noqa: BLE001
            self._tlog.error("Strategy fill callback failed", exc=exc, symbol=fill.symbol)
        self._sync_stop_state(fill, order)

    def _sync_stop_state(self, fill: Fill, order: Order) -> None:
        position = self._portfolio.get_position(fill.symbol)
        stop_price = order.metadata.get("stop_price")
        take_profit = order.metadata.get("take_profit_price")
        is_closing = bool(order.metadata.get("closing_order"))

        if is_closing or position.is_flat or not position.is_long:
            self._stop_tracker.deregister(fill.symbol)
            return

        if fill.side == OrderSide.BUY and stop_price:
            self._stop_tracker.register(
                symbol=fill.symbol,
                entry_price=fill.fill_price,
                stop_price=float(stop_price),
                strategy_id=order.strategy_id,
                take_profit_price=float(take_profit) if take_profit is not None else None,
            )

    def _expire_stale_orders(self, tick_ts: pd.Timestamp) -> None:
        for managed in self._oms.stale_open_orders(self._order_ttl_seconds, now=tick_ts):
            if managed.is_terminal:
                continue
            try:
                cancelled = False
                if managed.broker_order_id and not self._dry_run:
                    cancelled = self._broker.cancel_order(managed.broker_order_id)
                if cancelled:
                    self._oms.mark_cancelled(managed.order_id)
                    self._cancelled_orders += 1
                    self._tlog.order_cancelled(managed.order.symbol, broker_order_id=managed.broker_order_id)
                else:
                    self._oms.mark_expired(managed.order_id)
                    self._expired_orders += 1
                    self._tlog.system(
                        "Order expired",
                        symbol=managed.order.symbol,
                        broker_order_id=managed.broker_order_id,
                    )
            except Exception as exc:  # noqa: BLE001
                self._api_errors += 1
                self._tlog.error("Order expiry handling failed", exc=exc, symbol=managed.order.symbol)

    def _reconcile_broker_state(self, current_prices: Optional[dict[str, float]] = None) -> None:
        try:
            broker_account = self._broker.get_account_info()
            broker_positions = {pos.symbol: pos for pos in self._broker.get_positions()}
            local_positions = self._portfolio.open_positions

            mismatches: list[str] = []
            price_map = current_prices or self._latest_prices
            local_equity = self._portfolio.total_equity(price_map) if price_map else self._portfolio.cash

            if abs(broker_account.cash_balance - self._portfolio.cash) > 1.0:
                mismatches.append(
                    f"cash broker={broker_account.cash_balance:.2f} local={self._portfolio.cash:.2f}"
                )

            for symbol, local_pos in local_positions.items():
                broker_pos = broker_positions.get(symbol)
                if broker_pos is None:
                    mismatches.append(f"missing broker position {symbol}")
                    continue
                if abs(broker_pos.quantity - local_pos.quantity) > 1e-6:
                    mismatches.append(
                        f"qty mismatch {symbol}: broker={broker_pos.quantity:.4f} local={local_pos.quantity:.4f}"
                    )

            for symbol in broker_positions:
                if symbol not in local_positions and abs(broker_positions[symbol].quantity) > 1e-6:
                    mismatches.append(f"unexpected broker position {symbol}")

            if price_map and abs(broker_account.portfolio_value - local_equity) > max(10.0, abs(local_equity) * 0.001):
                mismatches.append(
                    f"equity broker={broker_account.portfolio_value:.2f} local={local_equity:.2f}"
                )

            if mismatches:
                self._reconciliation_ok = False
                self._reconciliation_reason = "; ".join(mismatches[:3])
                self._tlog.reconciliation("blocked", self._reconciliation_reason)
            else:
                self._reconciliation_ok = True
                self._reconciliation_reason = "ok"
                self._tlog.reconciliation("ok", "broker and local state aligned")

        except Exception as exc:  # noqa: BLE001
            self._api_errors += 1
            self._reconciliation_ok = False
            self._reconciliation_reason = f"reconciliation error: {exc}"
            self._tlog.reconciliation("blocked", self._reconciliation_reason)

    def _emit_heartbeat(
        self,
        equity: float,
        prices: dict[str, float],
        tick_ts: pd.Timestamp,
    ) -> None:
        n_positions = len(self._portfolio.open_positions)
        n_orders = len(self._oms.open_orders)
        snapshot = self._portfolio.snapshot(tick_ts, prices) if prices else None
        peak_equity = max(getattr(self._risk, "_peak_equity", equity), 1e-9)
        drawdown = max(0.0, (peak_equity - equity) / peak_equity) if equity > 0 else 0.0
        self._tlog.heartbeat(
            equity=equity,
            n_positions=n_positions,
            n_open_orders=n_orders,
            drawdown_pct=drawdown,
            gross_exposure=float(snapshot.gross_exposure) if snapshot is not None else 0.0,
            reconciliation_ok=self._reconciliation_ok,
        )
