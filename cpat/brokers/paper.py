"""
CPAT — Paper Broker (Week 5)
============================
A fully in-memory broker adapter that simulates order execution without
any network calls.  Implements the ``BrokerInterface`` protocol exactly.

Purpose
-------
* Safe pre-live validation: run the complete live execution stack with
  no API keys, no internet, and no real capital.
* Regression tests: deterministic fill simulation for unit/integration tests.
* Demo runs: produce realistic log output to validate the pipeline.

Simulation Model
----------------
Order Fills
    MARKET orders  → filled immediately at ``last_price + slip_bps × last``
    LIMIT orders   → filled immediately at the limit price (optimistic paper fill)
    All fills are instant (no queue, no latency simulation).

Price Feed
    Prices are seeded via ``set_price()`` or ``update_prices()``.
    If no price is set for a symbol, ``get_quote()`` raises ``BrokerError``.

Account
    Starts with ``initial_capital``.
    Cash is reduced on BUY fills and increased on SELL fills.
    No margin / leverage — positions cannot exceed available cash.

Usage::

    from cpat.brokers.paper import PaperBroker

    broker = PaperBroker(initial_capital=10_000_000.0)
    broker.connect()
    broker.set_price("RELIANCE.NS", bid=2498.0, ask=2502.0, last=2500.0)

    order_id = broker.place_order(my_order)
    status   = broker.get_order_status(order_id)
    account  = broker.get_account_info()
"""

from __future__ import annotations

import logging
import threading
from typing import Optional
from uuid import UUID, uuid4

import pandas as pd

from cpat.core.enums import OrderSide, OrderStatus, OrderType
from cpat.core.models import Fill, Order
from cpat.infrastructure.broker_interface import (
    BrokerAccountInfo,
    BrokerError,
    BrokerOrderError,
    BrokerOrderStatus,
    BrokerPosition,
    BrokerQuote,
)

logger = logging.getLogger(__name__)


class PaperBroker:
    """In-memory paper trading broker — implements ``BrokerInterface``.

    Args:
        initial_capital: Starting cash balance in ``currency``.
        currency       : ISO 4217 currency code.
        slip_bps       : Slippage applied to MARKET order fills (basis points).
        commission_pct : Commission as fraction of trade value (e.g. 0.0003 = 3 bps).
    """

    def __init__(
        self,
        initial_capital: float = 10_000_000.0,
        currency: str = "INR",
        slip_bps: float = 3.0,
        commission_pct: float = 0.0003,
    ) -> None:
        self._initial_capital = initial_capital
        self._currency        = currency
        self._slip_bps        = slip_bps
        self._commission_pct  = commission_pct

        # State
        self._connected  = False
        self._cash       = initial_capital
        self._lock       = threading.Lock()

        # symbol → {bid, ask, last}
        self._quotes: dict[str, dict[str, float]] = {}

        # broker_order_id → BrokerOrderStatus
        self._orders: dict[str, BrokerOrderStatus] = {}

        # symbol → net signed quantity
        self._positions: dict[str, float] = {}
        self._avg_costs:  dict[str, float] = {}

        # Filled orders → Fill objects (for portfolio reconciliation)
        self.fill_log: list[Fill] = []

    # ── BrokerInterface: Connection ────────────────────────────────────────────

    def connect(self) -> None:
        """Activate the paper broker session."""
        self._connected = True
        logger.info("[paper_broker] Connected | capital=%.2f %s",
                    self._initial_capital, self._currency)

    def disconnect(self) -> None:
        """Deactivate the paper broker session."""
        self._connected = False
        logger.info("[paper_broker] Disconnected.")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── BrokerInterface: Orders ────────────────────────────────────────────────

    def place_order(self, order: Order) -> str:
        """Simulate order placement with immediate fill for MARKET orders.

        Args:
            order: The ``Order`` to place.

        Returns:
            Broker-assigned order ID string.

        Raises:
            BrokerError: If not connected or price unavailable.
            BrokerOrderError: If insufficient cash.
        """
        self._require_connected()
        broker_id = f"PAPER-{uuid4().hex[:8].upper()}"

        with self._lock:
            quote = self._quotes.get(order.symbol)
            if quote is None:
                raise BrokerError(
                    f"No price set for {order.symbol} in PaperBroker.",
                    code="NO_PRICE",
                )

            last = quote["last"]
            fill_price = self._compute_fill_price(order, last)
            commission = fill_price * order.quantity * self._commission_pct

            # Cash check
            cost = fill_price * order.quantity + commission
            if order.side == OrderSide.BUY and cost > self._cash:
                raise BrokerOrderError(
                    f"Insufficient cash: need ₹{cost:.0f}, have ₹{self._cash:.0f}",
                    code="INSUFFICIENT_FUNDS",
                )

            # Create terminal status immediately (paper = instant fill)
            bos = BrokerOrderStatus(
                broker_order_id=broker_id,
                internal_order_id=order.order_id,
                status=OrderStatus.FILLED,
                filled_qty=order.quantity,
                avg_fill_price=fill_price,
                remaining_qty=0.0,
            )
            self._orders[broker_id] = bos

            # Apply fill to account
            self._apply_fill_to_account(order, fill_price, commission)

            # Record Fill domain object
            fill = Fill(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                fill_price=fill_price,
                commission=commission,
                slippage=abs(fill_price - last),
                timestamp=pd.Timestamp.utcnow(),
                exchange="PAPER",
            )
            self.fill_log.append(fill)

            logger.info(
                "[paper_broker] FILLED %s %s %.0f @ %.4f (commission=%.2f)",
                order.side.value, order.symbol, order.quantity, fill_price, commission,
            )

        return broker_id

    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel a pending paper order.

        Args:
            broker_order_id: Broker-side ID.

        Returns:
            True if cancelled; False if already terminal.
        """
        self._require_connected()
        with self._lock:
            bos = self._orders.get(broker_order_id)
            if bos is None:
                return False
            if bos.is_terminal:
                return False
            bos.status = OrderStatus.CANCELLED
            logger.info("[paper_broker] CANCELLED %s", broker_order_id)
            return True

    def get_order_status(self, broker_order_id: str) -> BrokerOrderStatus:
        """Return current order status.

        Args:
            broker_order_id: Broker-side ID.

        Raises:
            BrokerError: If order not found.
        """
        self._require_connected()
        with self._lock:
            bos = self._orders.get(broker_order_id)
            if bos is None:
                raise BrokerError(f"Order {broker_order_id} not found.", code="ORDER_NOT_FOUND")
            return bos

    # ── BrokerInterface: Market Data ───────────────────────────────────────────

    def get_market_data(self, symbol: str) -> BrokerQuote:
        """Return latest quote for a symbol.

        Args:
            symbol: Ticker symbol.

        Raises:
            BrokerError: If no price has been set for this symbol.
        """
        self._require_connected()
        with self._lock:
            q = self._quotes.get(symbol)
            if q is None:
                raise BrokerError(f"No quote for {symbol}.", code="NO_PRICE")
            return BrokerQuote(
                symbol=symbol,
                bid=q["bid"],
                ask=q["ask"],
                last=q["last"],
                timestamp=pd.Timestamp.utcnow(),
            )

    def get_quote(self, symbol: str) -> BrokerQuote:
        """Backward-compatible alias for ``get_market_data()``."""
        return self.get_market_data(symbol)

    # ── BrokerInterface: Account ───────────────────────────────────────────────

    def get_account_info(self) -> BrokerAccountInfo:
        """Return current account balances."""
        self._require_connected()
        with self._lock:
            portfolio_value = self._cash + self._positions_market_value()
            return BrokerAccountInfo(
                cash_balance=self._cash,
                portfolio_value=portfolio_value,
                buying_power=self._cash,
                currency=self._currency,
            )

    def get_positions(self) -> list[BrokerPosition]:
        """Return all open positions.

        Returns:
            List of ``BrokerPosition`` objects.
        """
        self._require_connected()
        with self._lock:
            positions = []
            for sym, qty in self._positions.items():
                if abs(qty) > 0:
                    q = self._quotes.get(sym, {})
                    last = q.get("last", self._avg_costs.get(sym, 0.0))
                    avg  = self._avg_costs.get(sym, last)
                    positions.append(BrokerPosition(
                        symbol=sym,
                        quantity=qty,
                        avg_cost=avg,
                        current_price=last,
                        unrealised_pnl=(last - avg) * qty,
                    ))
            return positions

    # ── PaperBroker-specific: Price Feed ──────────────────────────────────────

    def set_price(
        self,
        symbol: str,
        last: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
    ) -> None:
        """Set or update the current price for a symbol.

        Args:
            symbol: Ticker.
            last  : Last traded price.
            bid   : Best bid (defaults to last - 0.5 spread).
            ask   : Best ask (defaults to last + 0.5 spread).
        """
        spread_half = last * (self._slip_bps / 10_000) / 2
        with self._lock:
            self._quotes[symbol] = {
                "bid": bid if bid is not None else round(last - spread_half, 4),
                "ask": ask if ask is not None else round(last + spread_half, 4),
                "last": last,
            }

    def update_prices(self, prices: dict[str, float]) -> None:
        """Bulk-update last prices for multiple symbols.

        Args:
            prices: Dict of symbol → last price.
        """
        for sym, price in prices.items():
            self.set_price(sym, price)

    def get_fill_log(self) -> list[Fill]:
        """Return all fills generated by this session."""
        with self._lock:
            return list(self.fill_log)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _require_connected(self) -> None:
        if not self._connected:
            from cpat.infrastructure.broker_interface import BrokerConnectionError
            raise BrokerConnectionError("PaperBroker is not connected.")

    def _compute_fill_price(self, order: Order, last: float) -> float:
        """Compute simulated fill price with slippage.

        Args:
            order: The order being filled.
            last : Current last price.

        Returns:
            Fill price including simulated slippage.
        """
        slip = last * (self._slip_bps / 10_000)
        if order.order_type == OrderType.MARKET:
            return last + slip if order.side == OrderSide.BUY else last - slip
        elif order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            return order.limit_price  # type: ignore[return-value]
        elif order.order_type == OrderType.STOP:
            return order.stop_price   # type: ignore[return-value]
        return last

    def _apply_fill_to_account(
        self, order: Order, fill_price: float, commission: float
    ) -> None:
        """Update cash and position state after a fill.

        Must be called under ``self._lock``.

        Args:
            order      : Original order.
            fill_price : Achieved fill price.
            commission : Commission charged.
        """
        trade_value = fill_price * order.quantity
        sym = order.symbol
        prev_qty  = self._positions.get(sym, 0.0)
        prev_cost = self._avg_costs.get(sym, 0.0)

        if order.side == OrderSide.BUY:
            self._cash -= trade_value + commission
            new_qty = prev_qty + order.quantity
            if new_qty != 0:
                self._avg_costs[sym] = (
                    (prev_cost * prev_qty + fill_price * order.quantity) / new_qty
                )
            self._positions[sym] = new_qty
        else:  # SELL
            self._cash += trade_value - commission
            new_qty = prev_qty - order.quantity
            self._positions[sym] = new_qty
            if abs(new_qty) < 1e-9:
                self._positions.pop(sym, None)
                self._avg_costs.pop(sym, None)

    def _positions_market_value(self) -> float:
        """Compute total mark-to-market value of all positions.

        Returns:
            Sum of (last_price × quantity) for all open positions.
        """
        total = 0.0
        for sym, qty in self._positions.items():
            q = self._quotes.get(sym, {})
            last = q.get("last", self._avg_costs.get(sym, 0.0))
            total += last * qty
        return total
