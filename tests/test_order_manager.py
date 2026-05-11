"""Tests — Order Manager (OMS) (Week 5)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from cpat.core.enums import OrderSide, OrderStatus, OrderType
from cpat.core.models import Fill, Order
from cpat.infrastructure.broker_interface import BrokerOrderStatus
from cpat.infrastructure.order_manager import ManagedOrder, OrderManager


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_order(symbol="RELIANCE.NS", qty=10.0, side=OrderSide.BUY) -> Order:
    return Order(
        symbol=symbol, side=side, order_type=OrderType.MARKET,
        quantity=qty, timestamp=pd.Timestamp.utcnow(), strategy_id="test",
    )

def _make_fill(order: Order, qty=10.0, price=2500.0) -> Fill:
    return Fill(
        order_id=order.order_id, symbol=order.symbol,
        side=order.side, quantity=qty, fill_price=price,
        commission=7.50, slippage=0.50, timestamp=pd.Timestamp.utcnow(),
    )

def _oms(audit=False) -> OrderManager:
    if audit:
        tmp = tempfile.mktemp(suffix=".jsonl")
        return OrderManager(audit_log_path=tmp)
    return OrderManager()


# ── ManagedOrder ───────────────────────────────────────────────────────────────

class TestManagedOrder:
    def _mo(self, qty=10.0) -> ManagedOrder:
        o = _make_order(qty=qty)
        return ManagedOrder(order=o)

    def test_order_id_shortcut(self):
        mo = self._mo()
        assert mo.order_id == mo.order.order_id

    def test_remaining_qty_initial(self):
        mo = self._mo(qty=10.0)
        assert mo.remaining_qty == 10.0

    def test_remaining_qty_after_fill(self):
        mo = self._mo(qty=10.0)
        mo.filled_qty = 4.0
        assert mo.remaining_qty == 6.0

    def test_fill_pct(self):
        mo = self._mo(qty=10.0)
        mo.filled_qty = 5.0
        assert mo.fill_pct == pytest.approx(50.0)

    def test_is_terminal_pending(self):
        mo = self._mo()
        assert mo.is_terminal is False

    def test_is_terminal_filled(self):
        mo = self._mo()
        mo.status = OrderStatus.FILLED
        assert mo.is_terminal is True

    def test_is_terminal_rejected(self):
        mo = self._mo()
        mo.status = OrderStatus.REJECTED
        assert mo.is_terminal is True

    def test_is_terminal_cancelled(self):
        mo = self._mo()
        mo.status = OrderStatus.CANCELLED
        assert mo.is_terminal is True

    def test_to_dict_keys(self):
        mo = self._mo()
        d = mo.to_dict()
        for k in ["order_id", "symbol", "side", "status", "filled_qty", "remaining_qty"]:
            assert k in d

    def test_lifecycle_stage_mapping(self):
        mo = self._mo()
        assert mo.lifecycle_stage == "CREATED"
        mo.status = OrderStatus.SUBMITTED
        assert mo.lifecycle_stage == "SENT"


# ── OrderManager: Create ──────────────────────────────────────────────────────

class TestOrderManagerCreate:
    def test_create_registers_order(self):
        oms = _oms()
        order = _make_order()
        mo = oms.create(order)
        assert oms.get(order.order_id) is not None

    def test_create_returns_managed_order(self):
        oms = _oms()
        mo = oms.create(_make_order())
        assert isinstance(mo, ManagedOrder)

    def test_create_duplicate_raises(self):
        oms = _oms()
        order = _make_order()
        oms.create(order)
        with pytest.raises(ValueError, match="already registered"):
            oms.create(order)

    def test_initial_status_pending(self):
        oms = _oms()
        mo = oms.create(_make_order())
        assert mo.status == OrderStatus.PENDING


# ── OrderManager: Transitions ─────────────────────────────────────────────────

class TestOrderManagerTransitions:
    def setup_method(self):
        self.oms = _oms()
        self.order = _make_order()
        self.mo = self.oms.create(self.order)

    def test_mark_submitted(self):
        self.oms.mark_submitted(self.order.order_id, "BRK001")
        mo = self.oms.get(self.order.order_id)
        assert mo.status == OrderStatus.SUBMITTED
        assert mo.broker_order_id == "BRK001"

    def test_broker_id_index_populated(self):
        self.oms.mark_submitted(self.order.order_id, "BRK002")
        mo = self.oms.get_by_broker_id("BRK002")
        assert mo is not None
        assert mo.order_id == self.order.order_id

    def test_apply_fill_partial(self):
        self.oms.mark_submitted(self.order.order_id, "BRK001")
        self.oms.apply_fill(self.order.order_id, filled_qty=5.0, avg_price=2500.0)
        mo = self.oms.get(self.order.order_id)
        assert mo.status == OrderStatus.PARTIALLY_FILLED
        assert mo.filled_qty == 5.0

    def test_apply_fill_complete(self):
        self.oms.apply_fill(self.order.order_id, filled_qty=10.0, avg_price=2500.0)
        mo = self.oms.get(self.order.order_id)
        assert mo.status == OrderStatus.FILLED
        assert mo.filled_qty == 10.0

    def test_apply_fill_vwap(self):
        self.oms.apply_fill(self.order.order_id, filled_qty=5.0, avg_price=2500.0)
        self.oms.apply_fill(self.order.order_id, filled_qty=5.0, avg_price=2600.0)
        mo = self.oms.get(self.order.order_id)
        assert mo.avg_fill_price == pytest.approx(2550.0, rel=0.01)

    def test_fill_on_terminal_ignored(self):
        self.oms.mark_filled(self.order.order_id)
        # Second fill on already-filled order should be ignored silently
        self.oms.apply_fill(self.order.order_id, filled_qty=5.0, avg_price=2500.0)
        mo = self.oms.get(self.order.order_id)
        assert mo.filled_qty == 0.0  # unchanged

    def test_mark_cancelled(self):
        self.oms.mark_cancelled(self.order.order_id)
        assert self.oms.get(self.order.order_id).status == OrderStatus.CANCELLED

    def test_mark_rejected(self):
        self.oms.mark_rejected(self.order.order_id, "Insufficient funds")
        mo = self.oms.get(self.order.order_id)
        assert mo.status == OrderStatus.REJECTED
        assert "Insufficient" in mo.rejection_reason

    def test_mark_expired(self):
        self.oms.mark_expired(self.order.order_id)
        assert self.oms.get(self.order.order_id).status == OrderStatus.EXPIRED

    def test_increment_retry(self):
        count = self.oms.increment_retry(self.order.order_id)
        assert count == 1
        count2 = self.oms.increment_retry(self.order.order_id)
        assert count2 == 2

    def test_apply_broker_status_idempotent(self):
        self.oms.mark_submitted(self.order.order_id, "BRK001")
        status = BrokerOrderStatus(
            broker_order_id="BRK001",
            internal_order_id=self.order.order_id,
            status=OrderStatus.PARTIALLY_FILLED,
            filled_qty=4.0,
            avg_fill_price=2500.0,
            remaining_qty=6.0,
        )
        self.oms.apply_broker_status(status)
        self.oms.apply_broker_status(status)
        mo = self.oms.get(self.order.order_id)
        assert mo.filled_qty == pytest.approx(4.0)
        assert mo.status == OrderStatus.PARTIALLY_FILLED


# ── OrderManager: Queries ─────────────────────────────────────────────────────

class TestOrderManagerQueries:
    def test_pending_orders(self):
        oms = _oms()
        o1, o2 = _make_order("A"), _make_order("B")
        oms.create(o1); oms.create(o2)
        oms.mark_submitted(o1.order_id, "B1")
        oms.mark_filled(o2.order_id)
        pending = oms.pending_orders
        assert len(pending) == 1
        assert pending[0].order.symbol == "A"

    def test_open_orders_includes_pending(self):
        oms = _oms()
        o = _make_order()
        oms.create(o)
        assert len(oms.open_orders) == 1

    def test_all_orders(self):
        oms = _oms()
        for _ in range(5):
            oms.create(_make_order())
        assert len(oms.all_orders) == 5

    def test_get_by_broker_id(self):
        oms = _oms()
        o = _make_order()
        oms.create(o)
        oms.mark_submitted(o.order_id, "MYID")
        result = oms.get_by_broker_id("MYID")
        assert result is not None
        assert result.order_id == o.order_id

    def test_get_by_broker_id_none_if_not_found(self):
        oms = _oms()
        assert oms.get_by_broker_id("NOTFOUND") is None

    def test_summary_counts(self):
        oms = _oms()
        o1, o2 = _make_order("A"), _make_order("B")
        oms.create(o1); oms.create(o2)
        oms.mark_submitted(o1.order_id, "B1")
        oms.mark_filled(o1.order_id)
        oms.mark_rejected(o2.order_id, "bad")
        summary = oms.summary()
        assert summary.get("FILLED", 0) >= 1
        assert summary.get("REJECTED", 0) >= 1

    def test_get_returns_none_for_unknown(self):
        oms = _oms()
        assert oms.get(uuid4()) is None


# ── OrderManager: Audit Log ───────────────────────────────────────────────────

class TestOrderManagerAuditLog:
    def test_audit_log_written(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        oms = OrderManager(audit_log_path=path)
        o = _make_order()
        oms.create(o)
        oms.mark_submitted(o.order_id, "B1")
        oms.mark_filled(o.order_id)

        lines = Path(path).read_text().strip().splitlines()
        assert len(lines) >= 3  # create + submit + fill

        # Each line must be valid JSON
        for line in lines:
            record = json.loads(line)
            assert "order_id" in record
            assert "status" in record

    def test_audit_log_none_disabled(self):
        oms = OrderManager(audit_log_path=None)
        o = _make_order()
        oms.create(o)   # Should not raise

    def test_load_audit_log_reconstructs_orders(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        oms = OrderManager(audit_log_path=path)
        order = _make_order()
        oms.create(order)
        oms.mark_submitted(order.order_id, "B1")
        oms.mark_rejected(order.order_id, "bad")

        restored = OrderManager(audit_log_path=path)
        count = restored.load_audit_log()
        assert count == 1
        restored_order = restored.get(order.order_id)
        assert restored_order is not None
        assert restored_order.status == OrderStatus.REJECTED
