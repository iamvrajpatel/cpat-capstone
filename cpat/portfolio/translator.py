"""
CPAT — Signal → Order Translator
==================================
Converts SignalEvents from strategies into sized OrderEvents, using
current portfolio state to determine the correct action.

Signal Semantics:
    LONG  (+1) → establish or maintain a long position
    SHORT (-1) → establish or maintain a short position (if allowed)
    FLAT  ( 0) → close any existing position in this symbol

Position Flipping Logic:
    - LONG  + currently short  → close short first, then open long (2 orders)
    - SHORT + currently long   → close long first, then open short (2 orders)
    - LONG  + currently flat   → open long
    - LONG  + currently long   → no-op (already positioned)
    - FLAT  + any position     → close position (1 order)
    - FLAT  + flat             → no-op

Position Sizing (Week 2 implementations):
    - ``equal_weight``: allocate ``1 / n_signals`` of portfolio to each signal
    - ``fixed_fraction``: allocate a fixed % of equity per signal
    - ``inverse_volatility``: (stub, implemented in Week 3)

Key design principle:
    The Translator is STATELESS regarding time — it makes decisions
    based purely on the current portfolio snapshot and the signal.
    It never reads price history or future data.
"""

from __future__ import annotations

import logging
import math
from typing import Literal, Optional

import pandas as pd

from cpat.core.enums import OrderSide, OrderType, SignalDirection
from cpat.core.events import OrderEvent, SignalEvent
from cpat.core.models import Bar, Order
from cpat.backtest.event_queue import EventQueue
from cpat.portfolio.manager import PortfolioManager

logger = logging.getLogger(__name__)

SizingMethod = Literal["equal_weight", "fixed_fraction", "inverse_volatility"]


class SignalOrderTranslator:
    """Converts SignalEvents into OrderEvents using portfolio state.

    This component implements the **portfolio construction** layer:
    it receives raw signals (direction only) and produces concrete
    orders with specific quantities and sides.

    Args:
        portfolio: The PortfolioManager to query for current positions.
        event_queue: Event bus to post OrderEvents to.
        sizing_method: Position sizing algorithm.
        target_weight: Target weight per position for equal_weight/fixed_fraction.
        allow_short: Whether to generate SELL orders for SHORT signals.
        min_trade_value: Minimum trade value (USD) — avoids tiny round lots.
        max_position_weight: Hard cap on any single position (fraction of equity).
    """

    def __init__(
        self,
        portfolio: PortfolioManager,
        event_queue: EventQueue,
        sizing_method: SizingMethod = "equal_weight",
        target_weight: float = 0.02,  # 2% per position by default
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
        """Process a SignalEvent and emit zero or more OrderEvents.

        This is the Protocol-compatible handler for the signal event bus.

        Args:
            event: SignalEvent wrapping the Signal domain object.
        """
        signal = event.signal
        symbol = signal.symbol
        direction = signal.direction
        strategy_id = signal.strategy_id

        # Get current position and bar price from last snapshot
        current_position = self._portfolio.get_position(symbol)

        # We need the current price to size the order.
        # It is passed via signal.metadata["last_price"] (set by engine)
        last_price = signal.metadata.get("last_price", 0.0)
        if last_price <= 0:
            logger.debug(
                "[translator] No price for %s in signal metadata; skipping.", symbol
            )
            return

        orders = self._build_orders(
            symbol=symbol,
            direction=direction,
            current_position=current_position,
            last_price=last_price,
            timestamp=signal.timestamp,
            strategy_id=strategy_id,
            signal_id=signal.signal_id,
        )

        for order in orders:
            order_event = OrderEvent.from_order(order)
            self._event_queue.put(order_event)
            logger.info(
                "[translator] %s %s %.0f shares (signal=%s)",
                order.side.value, order.symbol, order.quantity, direction.name,
            )

    def on_signal_with_bars(
        self,
        event: SignalEvent,
        current_bars: dict[str, Bar],
    ) -> None:
        """Process a SignalEvent with access to current bar prices.

        Preferred over ``on_signal`` when the engine passes bars explicitly,
        avoiding the metadata dependency.

        Args:
            event: SignalEvent to process.
            current_bars: Current bar data for price lookup.
        """
        signal = event.signal
        symbol = signal.symbol

        bar = current_bars.get(symbol)
        if bar is None:
            logger.debug("[translator] No bar for %s; skipping signal.", symbol)
            return

        # Use close price for sizing (next bar's open will be the actual fill price)
        last_price = bar.close
        current_position = self._portfolio.get_position(symbol)

        orders = self._build_orders(
            symbol=symbol,
            direction=signal.direction,
            current_position=current_position,
            last_price=last_price,
            timestamp=signal.timestamp,
            strategy_id=signal.strategy_id,
            signal_id=signal.signal_id,
        )

        for order in orders:
            order_event = OrderEvent.from_order(order)
            self._event_queue.put(order_event)

    # ── Internal order-building logic ──────────────────────────────────────────

    def _build_orders(
        self,
        symbol: str,
        direction: SignalDirection,
        current_position: "Position",  # noqa: F821 # type: ignore[name-defined]
        last_price: float,
        timestamp: pd.Timestamp,
        strategy_id: str,
        signal_id: Optional["UUID"] = None,  # type: ignore[name-defined]
    ) -> list[Order]:
        """Core translation logic: signal + position state → orders.

        Returns:
            List of Orders (may be empty, 1 order, or 2 for position flips).
        """
        from cpat.core.models import Position as Pos
        orders: list[Order] = []

        # ── Compute target quantity ────────────────────────────────────────────
        target_qty = self._compute_target_quantity(symbol, last_price)

        match direction:
            case SignalDirection.LONG:
                if not self._allow_short and current_position.is_short:
                    # Close short first
                    close_order = self._make_order(
                        symbol=symbol, side=OrderSide.BUY,
                        quantity=abs(current_position.quantity),
                        timestamp=timestamp, strategy_id=strategy_id,
                        signal_id=signal_id,
                    )
                    orders.append(close_order)
                    logger.debug("[translator] Close short before going long: %s", symbol)

                if current_position.is_flat or current_position.is_short:
                    # Open long
                    if target_qty >= 1.0:
                        buy_order = self._make_order(
                            symbol=symbol, side=OrderSide.BUY,
                            quantity=target_qty,
                            timestamp=timestamp, strategy_id=strategy_id,
                            signal_id=signal_id,
                        )
                        orders.append(buy_order)
                # If already long: no-op (hold)

            case SignalDirection.SHORT:
                if not self._allow_short:
                    logger.debug(
                        "[translator] SHORT signal for %s ignored (short selling disabled).", symbol
                    )
                    return []

                if current_position.is_long:
                    # Close long first
                    close_order = self._make_order(
                        symbol=symbol, side=OrderSide.SELL,
                        quantity=abs(current_position.quantity),
                        timestamp=timestamp, strategy_id=strategy_id,
                        signal_id=signal_id,
                    )
                    orders.append(close_order)

                if current_position.is_flat or current_position.is_long:
                    if target_qty >= 1.0:
                        sell_order = self._make_order(
                            symbol=symbol, side=OrderSide.SELL,
                            quantity=target_qty,
                            timestamp=timestamp, strategy_id=strategy_id,
                            signal_id=signal_id,
                        )
                        orders.append(sell_order)

            case SignalDirection.FLAT:
                if current_position.is_long:
                    close_order = self._make_order(
                        symbol=symbol, side=OrderSide.SELL,
                        quantity=abs(current_position.quantity),
                        timestamp=timestamp, strategy_id=strategy_id,
                        signal_id=signal_id,
                    )
                    orders.append(close_order)
                elif current_position.is_short:
                    close_order = self._make_order(
                        symbol=symbol, side=OrderSide.BUY,
                        quantity=abs(current_position.quantity),
                        timestamp=timestamp, strategy_id=strategy_id,
                        signal_id=signal_id,
                    )
                    orders.append(close_order)
                # If flat: no-op

        return orders

    def _compute_target_quantity(self, symbol: str, price: float) -> float:
        """Compute the target number of shares for a new position.

        Args:
            symbol: Symbol being traded.
            price: Reference price for sizing.

        Returns:
            Integer share quantity (floor to whole shares).
        """
        if price <= 0:
            return 0.0

        # Use a surrogate price dict for equity lookup
        # We approximate with cash as the base for sizing
        equity = max(self._portfolio.cash, 1.0)
        # Also count existing positions' approximate value
        # (simplified: use cash for sizing to avoid circular dependency)

        target_value = equity * min(self._target_weight, self._max_position_weight)

        # Minimum trade filter
        if target_value < self._min_trade_value:
            target_value = self._min_trade_value

        # Floor to whole shares
        qty = math.floor(target_value / price)
        return max(0.0, float(qty))

    @staticmethod
    def _make_order(
        symbol: str,
        side: OrderSide,
        quantity: float,
        timestamp: pd.Timestamp,
        strategy_id: str,
        signal_id: Optional["UUID"] = None,  # type: ignore[name-defined]
    ) -> Order:
        return Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            timestamp=timestamp,
            strategy_id=strategy_id,
            signal_id=signal_id,
        )
