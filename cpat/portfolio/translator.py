"""
CPAT — Signal → Order Translator
==================================
Builds concrete market orders from already-sized trade intents.

Week 4 design:
    - Capital allocation and position sizing happen BEFORE the translator.
    - The translator is now responsible only for turning portfolio state and
      a desired direction into the correct close/open market orders.
    - A legacy compatibility wrapper is kept for the baseline comparison mode.
"""

from __future__ import annotations

import logging
import math
from typing import Literal, Optional

import pandas as pd

from cpat.backtest.event_queue import EventQueue
from cpat.core.enums import OrderSide, OrderType, SignalDirection
from cpat.core.events import OrderEvent, SignalEvent
from cpat.core.models import Bar, Order
from cpat.portfolio.manager import PortfolioManager
from cpat.risk.position_sizing import PositionSize

logger = logging.getLogger(__name__)

SizingMethod = Literal["equal_weight", "fixed_fraction", "inverse_volatility"]


class SignalOrderTranslator:
    """Translate a signal plus portfolio state into executable orders."""

    def __init__(
        self,
        portfolio: PortfolioManager,
        event_queue: EventQueue,
        sizing_method: SizingMethod = "equal_weight",
        target_weight: float = 0.02,
        allow_short: bool = False,
        min_trade_value: float = 500.0,
        max_position_weight: float = 0.05,
    ) -> None:
        self._portfolio = portfolio
        self._event_queue = event_queue
        self._sizing_method = sizing_method
        self._target_weight = target_weight
        self._allow_short = allow_short
        self._min_trade_value = min_trade_value
        self._max_position_weight = max_position_weight

    def on_signal(self, event: SignalEvent) -> None:
        """Compatibility path for legacy baseline flows."""
        signal = event.signal
        last_price = float(signal.metadata.get("last_price", 0.0))
        if last_price <= 0:
            logger.debug("[translator] No price metadata for %s; skipping.", signal.symbol)
            return

        legacy_size = self._build_legacy_position_size(signal.symbol, last_price)
        orders = self.build_orders(
            symbol=signal.symbol,
            direction=signal.direction,
            timestamp=signal.timestamp,
            strategy_id=signal.strategy_id,
            signal_id=signal.signal_id,
            position_size=legacy_size,
        )
        for order in orders:
            self._event_queue.put(OrderEvent.from_order(order))

    def on_signal_with_bars(
        self,
        event: SignalEvent,
        current_bars: dict[str, Bar],
    ) -> None:
        """Compatibility path for legacy baseline flows."""
        bar = current_bars.get(event.signal.symbol)
        if bar is None:
            logger.debug("[translator] No bar for %s; skipping signal.", event.signal.symbol)
            return

        legacy_size = self._build_legacy_position_size(event.signal.symbol, float(bar.close))
        orders = self.build_orders(
            symbol=event.signal.symbol,
            direction=event.signal.direction,
            timestamp=event.signal.timestamp,
            strategy_id=event.signal.strategy_id,
            signal_id=event.signal.signal_id,
            position_size=legacy_size,
        )
        for order in orders:
            self._event_queue.put(OrderEvent.from_order(order))

    def build_orders(
        self,
        symbol: str,
        direction: SignalDirection,
        timestamp: pd.Timestamp,
        strategy_id: str,
        signal_id: Optional["UUID"] = None,  # type: ignore[name-defined]
        position_size: Optional[PositionSize] = None,
    ) -> list[Order]:
        """Build the required close/open orders for a desired portfolio state."""
        current_position = self._portfolio.get_position(symbol)
        orders: list[Order] = []

        if direction == SignalDirection.FLAT:
            close_order = self.build_close_order(symbol, timestamp, strategy_id, signal_id)
            return [close_order] if close_order else []

        if direction == SignalDirection.SHORT and not self._allow_short:
            # In long-first mode, a short signal only closes an existing long.
            close_order = self.build_close_order(symbol, timestamp, strategy_id, signal_id)
            return [close_order] if close_order else []

        if current_position.is_long and direction == SignalDirection.LONG:
            return []

        if current_position.is_short and direction == SignalDirection.SHORT:
            return []

        if current_position.is_short and direction == SignalDirection.LONG:
            close_order = self.build_close_order(symbol, timestamp, strategy_id, signal_id)
            return [close_order] if close_order else []

        if current_position.is_long and direction == SignalDirection.SHORT:
            close_order = self.build_close_order(symbol, timestamp, strategy_id, signal_id)
            return [close_order] if close_order else []

        if position_size is None or not position_size.is_valid:
            return []

        side = OrderSide.BUY if direction == SignalDirection.LONG else OrderSide.SELL
        orders.append(
            self._make_order(
                symbol=symbol,
                side=side,
                quantity=position_size.quantity,
                timestamp=timestamp,
                strategy_id=strategy_id,
                signal_id=signal_id,
                metadata={
                    "stop_price": position_size.stop_price,
                    "take_profit_price": position_size.take_profit_price,
                    "risk_per_share": position_size.risk_per_share,
                    "planned_risk": position_size.total_risk,
                    "sizing_method": position_size.sizing_method,
                    "allocation_weight": 0.0,
                    "entry_price": position_size.entry_price,
                    "atr": position_size.atr,
                    "capital_budget": position_size.capital_budget,
                },
            )
        )
        return orders

    def build_close_order(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        strategy_id: str,
        signal_id: Optional["UUID"] = None,  # type: ignore[name-defined]
    ) -> Optional[Order]:
        """Build a closing order for the current position, if one exists."""
        position = self._portfolio.get_position(symbol)
        if position.is_flat:
            return None

        side = OrderSide.SELL if position.is_long else OrderSide.BUY
        return self._make_order(
            symbol=symbol,
            side=side,
            quantity=abs(position.quantity),
            timestamp=timestamp,
            strategy_id=strategy_id,
            signal_id=signal_id,
            metadata={"closing_order": True},
        )

    def _build_legacy_position_size(self, symbol: str, price: float) -> PositionSize:
        """Legacy equal-weight sizing used only for baseline comparison mode."""
        if price <= 0:
            return PositionSize(
                symbol=symbol,
                quantity=0.0,
                entry_price=price,
                stop_price=0.0,
                take_profit_price=None,
                risk_per_share=0.0,
                total_risk=0.0,
                atr=0.0,
                sizing_method="legacy_equal_weight",
                capital_budget=0.0,
            )

        target_value = max(
            self._min_trade_value,
            self._portfolio.cash * min(self._target_weight, self._max_position_weight),
        )
        quantity = max(0.0, float(math.floor(target_value / price)))
        stop_price = price * 0.98
        return PositionSize(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            stop_price=stop_price,
            take_profit_price=None,
            risk_per_share=max(price - stop_price, 0.0),
            total_risk=max(quantity * (price - stop_price), 0.0),
            atr=0.0,
            sizing_method="legacy_equal_weight",
            capital_budget=target_value,
        )

    @staticmethod
    def _make_order(
        symbol: str,
        side: OrderSide,
        quantity: float,
        timestamp: pd.Timestamp,
        strategy_id: str,
        signal_id: Optional["UUID"] = None,  # type: ignore[name-defined]
        metadata: Optional[dict[str, object]] = None,
    ) -> Order:
        return Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            timestamp=timestamp,
            strategy_id=strategy_id,
            signal_id=signal_id,
            metadata=metadata or {},
        )
