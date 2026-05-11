"""Tests — Broker Interface + PaperBroker (Week 5)."""
from __future__ import annotations

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from uuid import uuid4

from cpat.brokers.paper import PaperBroker
from cpat.core.enums import OrderSide, OrderStatus, OrderType
from cpat.core.models import Order
from cpat.infrastructure.broker_interface import (
    BrokerError, BrokerConnectionError, BrokerOrderError,
    BrokerOrderStatus, BrokerAccountInfo, BrokerPosition, BrokerQuote,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_order(symbol="RELIANCE.NS", qty=10.0, side=OrderSide.BUY,
                order_type=OrderType.MARKET) -> Order:
    return Order(
        symbol=symbol, side=side, order_type=order_type,
        quantity=qty, timestamp=pd.Timestamp.utcnow(),
        strategy_id="test",
    )

def _connected_broker(**kwargs) -> PaperBroker:
    b = PaperBroker(**kwargs)
    b.connect()
    b.set_price("RELIANCE.NS", last=2500.0)
    return b


# ── BrokerError hierarchy ─────────────────────────────────────────────────────

class TestBrokerErrors:
    def test_broker_error_retryable_false(self):
        e = BrokerError("msg")
        assert e.retryable is False

    def test_connection_error_retryable(self):
        e = BrokerConnectionError("network down")
        assert e.retryable is True

    def test_auth_error_not_retryable(self):
        from cpat.infrastructure.broker_interface import BrokerAuthError
        e = BrokerAuthError("bad token")
        assert e.retryable is False

    def test_order_error_not_retryable_by_default(self):
        e = BrokerOrderError("insufficient funds")
        assert e.retryable is False


# ── BrokerOrderStatus ─────────────────────────────────────────────────────────

class TestBrokerOrderStatus:
    def _make_status(self, status=OrderStatus.SUBMITTED, filled=0.0) -> BrokerOrderStatus:
        return BrokerOrderStatus(
            broker_order_id="BRK001",
            internal_order_id=uuid4(),
            status=status,
            filled_qty=filled,
            avg_fill_price=2500.0,
            remaining_qty=10.0 - filled,
        )

    def test_is_terminal_filled(self):
        assert self._make_status(OrderStatus.FILLED).is_terminal is True

    def test_is_terminal_cancelled(self):
        assert self._make_status(OrderStatus.CANCELLED).is_terminal is True

    def test_is_terminal_rejected(self):
        assert self._make_status(OrderStatus.REJECTED).is_terminal is True

    def test_not_terminal_submitted(self):
        assert self._make_status(OrderStatus.SUBMITTED).is_terminal is False

    def test_not_terminal_partially_filled(self):
        assert self._make_status(OrderStatus.PARTIALLY_FILLED, filled=5.0).is_terminal is False


# ── BrokerQuote ───────────────────────────────────────────────────────────────

class TestBrokerQuote:
    def test_mid(self):
        q = BrokerQuote("S", bid=100.0, ask=102.0, last=101.0)
        assert q.mid == 101.0

    def test_spread(self):
        q = BrokerQuote("S", bid=100.0, ask=102.0, last=101.0)
        assert q.spread == 2.0


# ── PaperBroker ───────────────────────────────────────────────────────────────

class TestPaperBrokerConnection:
    def test_starts_disconnected(self):
        b = PaperBroker()
        assert b.is_connected is False

    def test_connect(self):
        b = PaperBroker()
        b.connect()
        assert b.is_connected is True

    def test_disconnect(self):
        b = PaperBroker()
        b.connect()
        b.disconnect()
        assert b.is_connected is False

    def test_place_order_requires_connection(self):
        b = PaperBroker()
        with pytest.raises(BrokerConnectionError):
            b.place_order(_make_order())


class TestPaperBrokerOrders:
    def test_place_market_order_returns_id(self):
        b = _connected_broker()
        broker_id = b.place_order(_make_order(qty=5.0))
        assert broker_id.startswith("PAPER-")

    def test_market_order_fills_immediately(self):
        b = _connected_broker()
        order = _make_order(qty=5.0)
        broker_id = b.place_order(order)
        status = b.get_order_status(broker_id)
        assert status.status == OrderStatus.FILLED
        assert status.filled_qty == 5.0

    def test_buy_reduces_cash(self):
        b = _connected_broker(initial_capital=1_000_000.0)
        b.place_order(_make_order(qty=10.0))
        info = b.get_account_info()
        assert info.cash_balance < 1_000_000.0

    def test_sell_increases_cash(self):
        b = _connected_broker(initial_capital=1_000_000.0)
        b.place_order(_make_order(qty=10.0, side=OrderSide.BUY))
        cash_after_buy = b.get_account_info().cash_balance
        b.place_order(_make_order(qty=10.0, side=OrderSide.SELL))
        cash_after_sell = b.get_account_info().cash_balance
        assert cash_after_sell > cash_after_buy

    def test_insufficient_cash_raises(self):
        b = _connected_broker(initial_capital=100.0)  # only ₹100
        with pytest.raises(BrokerOrderError, match="Insufficient"):
            b.place_order(_make_order(qty=1000.0))  # ₹2.5M order

    def test_no_price_raises_error(self):
        b = PaperBroker()
        b.connect()
        # No price set for TCS.NS
        with pytest.raises(BrokerError):
            b.place_order(_make_order(symbol="TCS.NS"))

    def test_cancel_order_not_found_returns_false(self):
        b = _connected_broker()
        result = b.cancel_order("NONEXISTENT")
        assert result is False

    def test_get_order_status_not_found_raises(self):
        b = _connected_broker()
        with pytest.raises(BrokerError):
            b.get_order_status("NONEXISTENT")

    def test_limit_order_fills_at_limit_price(self):
        b = _connected_broker()
        order = Order(
            symbol="RELIANCE.NS", side=OrderSide.BUY,
            order_type=OrderType.LIMIT, quantity=5.0,
            limit_price=2490.0,
            timestamp=pd.Timestamp.utcnow(), strategy_id="test",
        )
        broker_id = b.place_order(order)
        status = b.get_order_status(broker_id)
        assert abs(status.avg_fill_price - 2490.0) < 0.01

    def test_slippage_applied_on_buy(self):
        b = _connected_broker(slip_bps=10.0)
        order = _make_order(qty=1.0)
        broker_id = b.place_order(order)
        status = b.get_order_status(broker_id)
        # BUY fill price should be above last (2500) due to slippage
        assert status.avg_fill_price > 2500.0


class TestPaperBrokerAccount:
    def test_initial_account_info(self):
        b = _connected_broker(initial_capital=5_000_000.0)
        info = b.get_account_info()
        assert info.cash_balance == pytest.approx(5_000_000.0, rel=0.01)
        assert info.currency == "INR"

    def test_no_positions_initially(self):
        b = _connected_broker()
        assert b.get_positions() == []

    def test_position_after_buy(self):
        b = _connected_broker()
        b.place_order(_make_order(qty=5.0))
        positions = b.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "RELIANCE.NS"
        assert positions[0].quantity == 5.0

    def test_flat_after_sell(self):
        b = _connected_broker()
        b.place_order(_make_order(qty=5.0, side=OrderSide.BUY))
        b.place_order(_make_order(qty=5.0, side=OrderSide.SELL))
        assert b.get_positions() == []

    def test_fill_log_recorded(self):
        b = _connected_broker()
        b.place_order(_make_order(qty=3.0))
        fills = b.get_fill_log()
        assert len(fills) == 1
        assert fills[0].quantity == 3.0


class TestPaperBrokerPriceFeed:
    def test_get_market_data_alias(self):
        b = _connected_broker()
        q = b.get_market_data("RELIANCE.NS")
        assert q.last == 2500.0

    def test_set_price_and_get_quote(self):
        b = _connected_broker()
        b.set_price("TCS.NS", last=3500.0, bid=3498.0, ask=3502.0)
        q = b.get_quote("TCS.NS")
        assert q.last == 3500.0
        assert q.bid == 3498.0
        assert q.ask == 3502.0

    def test_update_prices_bulk(self):
        b = _connected_broker()
        b.update_prices({"TCS.NS": 3500.0, "HDFCBANK.NS": 1600.0})
        assert b.get_quote("TCS.NS").last == 3500.0
        assert b.get_quote("HDFCBANK.NS").last == 1600.0

    def test_get_quote_no_price_raises(self):
        b = _connected_broker()
        with pytest.raises(BrokerError):
            b.get_quote("UNKNOWN.NS")

    def test_bid_ask_auto_set_from_last(self):
        b = _connected_broker(slip_bps=10.0)
        b.set_price("TEST.NS", last=1000.0)
        q = b.get_quote("TEST.NS")
        assert q.bid < 1000.0 < q.ask
