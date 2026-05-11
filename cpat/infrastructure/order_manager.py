"""
CPAT — Order Management System (Week 5)
========================================
The OMS is the single source of truth for all live orders.

It tracks every order from CREATED → SENT → FILLED/REJECTED/CANCELLED
and provides atomic lookups so the live execution engine never loses an order.

Design
------
* In-memory primary store (dict keyed by internal UUID).
* Optional JSON-line audit log written to disk for post-trade reconciliation.
* Thread-safe via ``threading.Lock`` — the live engine may call OMS from
  multiple threads (signal thread + fill-listener thread).
* NO business logic — OMS is a bookkeeping layer. Risk and portfolio
  logic remain in their own modules.

Order Lifecycle
---------------
    CREATED  → order object stored, not yet sent to broker
    SUBMITTED→ ``place_order()`` called, broker_order_id received
    PARTIALLY_FILLED → some qty filled, remainder still live
    FILLED   → fully executed (terminal)
    CANCELLED→ cancellation confirmed (terminal)
    REJECTED → broker refused the order (terminal)
    EXPIRED  → order timed out without fill (terminal)

Usage::

    from cpat.infrastructure.order_manager import OrderManager, ManagedOrder

    oms = OrderManager(audit_log_path="logs/oms_audit.jsonl")
    managed = oms.create(order)          # → ManagedOrder
    oms.mark_submitted(order.order_id, broker_id="BRK123")
    oms.apply_fill(order.order_id, filled_qty=50, avg_price=2500.0)
    oms.mark_filled(order.order_id)
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from uuid import UUID

import pandas as pd

from cpat.core.enums import OrderSide, OrderStatus, OrderType
from cpat.core.models import Fill, Order
from cpat.infrastructure.broker_interface import BrokerOrderStatus

logger = logging.getLogger(__name__)


# ── Managed Order ──────────────────────────────────────────────────────────────


@dataclass
class ManagedOrder:
    """OMS wrapper around a core ``Order`` with live-tracking fields.

    Attributes:
        order          : Immutable original Order domain object.
        status         : Current lifecycle state.
        broker_order_id: Broker-assigned ID (populated after submission).
        filled_qty     : Cumulative quantity filled.
        avg_fill_price : Volume-weighted average fill price.
        fills          : All Fill objects applied to this order.
        rejection_reason: Populated on REJECTED status.
        created_at     : OMS creation timestamp.
        submitted_at   : When ``place_order()`` was called.
        terminal_at    : When the order reached a terminal state.
        retry_count    : Number of submission retries attempted.
    """

    order: Order
    status: OrderStatus = OrderStatus.PENDING
    broker_order_id: str = ""
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    rejection_reason: str = ""
    created_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.utcnow())
    submitted_at: Optional[pd.Timestamp] = None
    terminal_at: Optional[pd.Timestamp] = None
    retry_count: int = 0
    last_update_at: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.utcnow())
    terminal_reason: str = ""
    last_broker_payload: Optional[dict] = None

    @property
    def order_id(self) -> UUID:
        """Shortcut to the internal order UUID."""
        return self.order.order_id

    @property
    def remaining_qty(self) -> float:
        """Quantity not yet filled."""
        return max(0.0, self.order.quantity - self.filled_qty)

    @property
    def is_terminal(self) -> bool:
        """True if no further state transitions are possible."""
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )

    @property
    def fill_pct(self) -> float:
        """Percentage of original quantity that has been filled."""
        if self.order.quantity <= 0:
            return 0.0
        return self.filled_qty / self.order.quantity * 100.0

    @property
    def lifecycle_stage(self) -> str:
        """Prompt-facing OMS lifecycle label."""
        mapping = {
            OrderStatus.PENDING: "CREATED",
            OrderStatus.SUBMITTED: "SENT",
            OrderStatus.PARTIALLY_FILLED: "PARTIALLY_FILLED",
            OrderStatus.FILLED: "FILLED",
            OrderStatus.REJECTED: "REJECTED",
            OrderStatus.CANCELLED: "CANCELLED",
            OrderStatus.EXPIRED: "EXPIRED",
        }
        return mapping.get(self.status, self.status.value)

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON audit logging."""
        return {
            "order_id": str(self.order_id),
            "broker_order_id": self.broker_order_id,
            "symbol": self.order.symbol,
            "side": self.order.side.value,
            "order_type": self.order.order_type.value,
            "quantity": self.order.quantity,
            "status": self.status.value,
            "lifecycle_stage": self.lifecycle_stage,
            "filled_qty": self.filled_qty,
            "avg_fill_price": round(self.avg_fill_price, 4),
            "remaining_qty": self.remaining_qty,
            "fill_pct": round(self.fill_pct, 2),
            "rejection_reason": self.rejection_reason,
            "created_at": str(self.created_at),
            "submitted_at": str(self.submitted_at) if self.submitted_at else None,
            "terminal_at": str(self.terminal_at) if self.terminal_at else None,
            "last_update_at": str(self.last_update_at),
            "retry_count": self.retry_count,
            "strategy_id": self.order.strategy_id,
            "terminal_reason": self.terminal_reason,
            "last_broker_payload": self.last_broker_payload,
            "signal_id": str(self.order.signal_id) if self.order.signal_id else None,
            "timestamp": str(self.order.timestamp),
            "metadata": _json_safe(self.order.metadata),
        }


# ── Order Manager ──────────────────────────────────────────────────────────────


class OrderManager:
    """Thread-safe in-memory order book with optional JSON-line audit trail.

    Args:
        audit_log_path: Path to the JSONL audit file.  Set to ``None``
                        to disable file logging (useful in tests).
    """

    def __init__(self, audit_log_path: Optional[str | Path] = None) -> None:
        self._orders: dict[UUID, ManagedOrder] = {}
        self._broker_id_index: dict[str, UUID] = {}  # broker_id → internal UUID
        self._lock = threading.RLock()

        self._audit_path: Optional[Path] = None
        if audit_log_path:
            self._audit_path = Path(audit_log_path)
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("OrderManager initialised | audit_log=%s", self._audit_path)

    # ── Create ─────────────────────────────────────────────────────────────────

    def create(self, order: Order) -> ManagedOrder:
        """Register a new order in the OMS.

        Args:
            order: The ``Order`` to track.

        Returns:
            The newly created ``ManagedOrder``.

        Raises:
            ValueError: If an order with the same ID already exists.
        """
        with self._lock:
            if order.order_id in self._orders:
                raise ValueError(f"Order {order.order_id} already registered.")
            managed = ManagedOrder(order=order)
            self._orders[order.order_id] = managed
            logger.debug("[OMS] CREATED %s %s %.0f @ %s",
                         order.side.value, order.symbol, order.quantity, order.order_type.value)
            self._audit(managed)
            return managed

    # ── State Transitions ──────────────────────────────────────────────────────

    def mark_submitted(self, order_id: UUID, broker_order_id: str) -> ManagedOrder:
        """Record that the order has been sent to the broker.

        Args:
            order_id       : Internal UUID.
            broker_order_id: Broker-assigned ID string.

        Returns:
            Updated ``ManagedOrder``.
        """
        with self._lock:
            mo = self._get_or_raise(order_id)
            if mo.broker_order_id == broker_order_id and mo.status in (
                OrderStatus.SUBMITTED,
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.FILLED,
            ):
                return mo
            mo.status = OrderStatus.SUBMITTED
            mo.broker_order_id = broker_order_id
            mo.submitted_at = pd.Timestamp.utcnow()
            mo.last_update_at = pd.Timestamp.utcnow()
            self._broker_id_index[broker_order_id] = order_id
            logger.info("[OMS] SUBMITTED %s → broker_id=%s", order_id, broker_order_id)
            self._audit(mo)
            return mo

    def apply_fill(
        self,
        order_id: UUID,
        filled_qty: float,
        avg_price: float,
        fill: Optional[Fill] = None,
    ) -> ManagedOrder:
        """Apply a (partial) fill to a managed order.

        Updates cumulative fill quantities and VWAP.  Transitions to
        PARTIALLY_FILLED or stays SUBMITTED depending on remaining qty.

        Args:
            order_id  : Internal UUID.
            filled_qty: Quantity filled in this fill event.
            avg_price : Execution price for this fill.
            fill      : Optional ``Fill`` domain object for audit trail.

        Returns:
            Updated ``ManagedOrder``.
        """
        with self._lock:
            mo = self._get_or_raise(order_id)
            if mo.is_terminal:
                logger.warning("[OMS] Fill received for terminal order %s — ignoring.", order_id)
                return mo

            if filled_qty <= 0 or mo.filled_qty + 1e-9 >= mo.order.quantity:
                return mo

            # Update VWAP
            prev_value = mo.avg_fill_price * mo.filled_qty
            new_value  = avg_price * filled_qty
            mo.filled_qty += filled_qty
            mo.avg_fill_price = (prev_value + new_value) / mo.filled_qty if mo.filled_qty > 0 else 0.0
            mo.last_update_at = pd.Timestamp.utcnow()

            if fill is not None:
                if not any(existing.fill_id == fill.fill_id for existing in mo.fills):
                    mo.fills.append(fill)

            # Transition state
            if mo.remaining_qty <= 0.01:   # float tolerance
                mo.status = OrderStatus.FILLED
                mo.terminal_at = pd.Timestamp.utcnow()
                mo.terminal_reason = mo.terminal_reason or "filled"
                logger.info("[OMS] FILLED %s | qty=%.0f | avg_px=%.4f",
                            mo.order.symbol, mo.filled_qty, mo.avg_fill_price)
            else:
                mo.status = OrderStatus.PARTIALLY_FILLED
                logger.info("[OMS] PARTIAL_FILL %s | filled=%.0f/%.0f | avg_px=%.4f",
                            mo.order.symbol, mo.filled_qty, mo.order.quantity, mo.avg_fill_price)

            self._audit(mo)
            return mo

    def mark_filled(self, order_id: UUID) -> ManagedOrder:
        """Force-transition to FILLED (for brokers that confirm after fills).

        Args:
            order_id: Internal UUID.

        Returns:
            Updated ``ManagedOrder``.
        """
        with self._lock:
            mo = self._get_or_raise(order_id)
            if mo.status == OrderStatus.FILLED:
                return mo
            mo.status = OrderStatus.FILLED
            mo.terminal_at = pd.Timestamp.utcnow()
            mo.last_update_at = pd.Timestamp.utcnow()
            mo.terminal_reason = mo.terminal_reason or "filled"
            self._audit(mo)
            return mo

    def mark_cancelled(self, order_id: UUID) -> ManagedOrder:
        """Transition to CANCELLED.

        Args:
            order_id: Internal UUID.

        Returns:
            Updated ``ManagedOrder``.
        """
        with self._lock:
            mo = self._get_or_raise(order_id)
            if not mo.is_terminal:
                mo.status = OrderStatus.CANCELLED
                mo.terminal_at = pd.Timestamp.utcnow()
                mo.last_update_at = pd.Timestamp.utcnow()
                mo.terminal_reason = "cancelled"
                logger.info("[OMS] CANCELLED %s", order_id)
                self._audit(mo)
            return mo

    def mark_rejected(self, order_id: UUID, reason: str) -> ManagedOrder:
        """Transition to REJECTED with a reason string.

        Args:
            order_id: Internal UUID.
            reason  : Human-readable rejection reason.

        Returns:
            Updated ``ManagedOrder``.
        """
        with self._lock:
            mo = self._get_or_raise(order_id)
            if mo.status == OrderStatus.REJECTED and mo.rejection_reason == reason:
                return mo
            mo.status = OrderStatus.REJECTED
            mo.rejection_reason = reason
            mo.terminal_at = pd.Timestamp.utcnow()
            mo.last_update_at = pd.Timestamp.utcnow()
            mo.terminal_reason = reason or "rejected"
            logger.warning("[OMS] REJECTED %s | reason: %s", mo.order.symbol, reason)
            self._audit(mo)
            return mo

    def mark_expired(self, order_id: UUID) -> ManagedOrder:
        """Transition to EXPIRED (GTC or GTD order that ran out of time).

        Args:
            order_id: Internal UUID.

        Returns:
            Updated ``ManagedOrder``.
        """
        with self._lock:
            mo = self._get_or_raise(order_id)
            if not mo.is_terminal:
                mo.status = OrderStatus.EXPIRED
                mo.terminal_at = pd.Timestamp.utcnow()
                mo.last_update_at = pd.Timestamp.utcnow()
                mo.terminal_reason = "expired"
                logger.info("[OMS] EXPIRED %s", order_id)
                self._audit(mo)
            return mo

    def increment_retry(self, order_id: UUID) -> int:
        """Increment retry counter and return the new count.

        Args:
            order_id: Internal UUID.

        Returns:
            New retry count.
        """
        with self._lock:
            mo = self._get_or_raise(order_id)
            mo.retry_count += 1
            mo.last_update_at = pd.Timestamp.utcnow()
            return mo.retry_count

    def apply_broker_status(
        self,
        status: BrokerOrderStatus,
        payload: Optional[dict] = None,
    ) -> Optional[ManagedOrder]:
        """Apply a broker-side status update idempotently."""
        with self._lock:
            mo = self._orders.get(status.internal_order_id)
            if mo is None and status.broker_order_id:
                uid = self._broker_id_index.get(status.broker_order_id)
                mo = self._orders.get(uid) if uid else None
            if mo is None:
                return None

            if status.broker_order_id and not mo.broker_order_id:
                mo.broker_order_id = status.broker_order_id
                self._broker_id_index[status.broker_order_id] = mo.order_id

            mo.last_broker_payload = payload
            mo.last_update_at = pd.Timestamp.utcnow()

            if status.filled_qty > mo.filled_qty:
                incremental_fill = status.filled_qty - mo.filled_qty
                self.apply_fill(
                    mo.order_id,
                    filled_qty=incremental_fill,
                    avg_price=status.avg_fill_price,
                )
                mo = self._get_or_raise(mo.order_id)

            if mo.is_terminal:
                return mo

            if status.status == OrderStatus.REJECTED:
                mo.status = OrderStatus.REJECTED
                mo.rejection_reason = status.rejected_reason
                mo.terminal_reason = status.rejected_reason or "rejected"
                mo.terminal_at = pd.Timestamp.utcnow()
            elif status.status == OrderStatus.CANCELLED:
                mo.status = OrderStatus.CANCELLED
                mo.terminal_reason = "cancelled"
                mo.terminal_at = pd.Timestamp.utcnow()
            elif status.status == OrderStatus.EXPIRED:
                mo.status = OrderStatus.EXPIRED
                mo.terminal_reason = "expired"
                mo.terminal_at = pd.Timestamp.utcnow()
            elif status.status == OrderStatus.FILLED:
                mo.status = OrderStatus.FILLED
                mo.filled_qty = max(mo.filled_qty, status.filled_qty)
                mo.avg_fill_price = status.avg_fill_price or mo.avg_fill_price
                mo.terminal_reason = mo.terminal_reason or "filled"
                mo.terminal_at = pd.Timestamp.utcnow()
            elif status.status == OrderStatus.PARTIALLY_FILLED:
                mo.status = OrderStatus.PARTIALLY_FILLED
            else:
                mo.status = OrderStatus.SUBMITTED

            self._audit(mo)
            return mo

    # ── Queries ────────────────────────────────────────────────────────────────

    def get(self, order_id: UUID) -> Optional[ManagedOrder]:
        """Return a ManagedOrder by internal UUID, or None.

        Args:
            order_id: Internal UUID.
        """
        with self._lock:
            return self._orders.get(order_id)

    def get_by_broker_id(self, broker_order_id: str) -> Optional[ManagedOrder]:
        """Look up by broker-assigned ID.

        Args:
            broker_order_id: Broker-side order ID.
        """
        with self._lock:
            uid = self._broker_id_index.get(broker_order_id)
            return self._orders.get(uid) if uid else None

    @property
    def pending_orders(self) -> list[ManagedOrder]:
        """All orders in SUBMITTED or PARTIALLY_FILLED state."""
        with self._lock:
            return [
                mo for mo in self._orders.values()
                if mo.status in (OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED)
            ]

    @property
    def open_orders(self) -> list[ManagedOrder]:
        """Pending + PENDING (created but not yet submitted)."""
        with self._lock:
            return [mo for mo in self._orders.values() if not mo.is_terminal]

    @property
    def all_orders(self) -> list[ManagedOrder]:
        """All tracked orders (any state)."""
        with self._lock:
            return list(self._orders.values())

    def summary(self) -> dict:
        """Return a count summary of orders by status.

        Returns:
            Dict mapping status name → count.
        """
        with self._lock:
            counts: dict[str, int] = {}
            for mo in self._orders.values():
                key = mo.status.value
                counts[key] = counts.get(key, 0) + 1
            return counts

    def stale_open_orders(
        self,
        older_than_seconds: float,
        now: Optional[pd.Timestamp] = None,
    ) -> list[ManagedOrder]:
        """Return open orders older than the configured TTL."""
        if older_than_seconds <= 0:
            return []
        now_ts = now or pd.Timestamp.utcnow()
        with self._lock:
            stale: list[ManagedOrder] = []
            for mo in self._orders.values():
                if mo.is_terminal:
                    continue
                reference_ts = mo.submitted_at or mo.created_at
                if (now_ts - reference_ts).total_seconds() >= older_than_seconds:
                    stale.append(mo)
            return stale

    def load_audit_log(self) -> int:
        """Replay the JSONL audit file into the in-memory OMS."""
        if self._audit_path is None or not self._audit_path.exists():
            return 0

        with self._lock:
            latest_records: dict[str, dict] = {}
            with self._audit_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    latest_records[record["order_id"]] = record

            self._orders.clear()
            self._broker_id_index.clear()

            for record in latest_records.values():
                order = Order(
                    symbol=record["symbol"],
                    side=OrderSide(record["side"]),
                    order_type=OrderType(record["order_type"]),
                    quantity=float(record["quantity"]),
                    timestamp=pd.Timestamp(record["timestamp"]),
                    strategy_id=record.get("strategy_id", ""),
                    order_id=UUID(record["order_id"]),
                    metadata=record.get("metadata", {}) or {},
                )
                managed = ManagedOrder(
                    order=order,
                    status=OrderStatus(record["status"]),
                    broker_order_id=record.get("broker_order_id", "") or "",
                    filled_qty=float(record.get("filled_qty", 0.0)),
                    avg_fill_price=float(record.get("avg_fill_price", 0.0)),
                    rejection_reason=record.get("rejection_reason", ""),
                    created_at=pd.Timestamp(record["created_at"]),
                    submitted_at=pd.Timestamp(record["submitted_at"]) if record.get("submitted_at") else None,
                    terminal_at=pd.Timestamp(record["terminal_at"]) if record.get("terminal_at") else None,
                    retry_count=int(record.get("retry_count", 0)),
                    last_update_at=pd.Timestamp(record.get("last_update_at") or record["created_at"]),
                    terminal_reason=record.get("terminal_reason", ""),
                    last_broker_payload=record.get("last_broker_payload"),
                )
                self._orders[managed.order_id] = managed
                if managed.broker_order_id:
                    self._broker_id_index[managed.broker_order_id] = managed.order_id

            return len(self._orders)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _get_or_raise(self, order_id: UUID) -> ManagedOrder:
        mo = self._orders.get(order_id)
        if mo is None:
            raise KeyError(f"Order {order_id} not found in OMS.")
        return mo

    def _audit(self, mo: ManagedOrder) -> None:
        """Append a JSON-line entry to the audit log."""
        if self._audit_path is None:
            return
        try:
            with self._audit_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(mo.to_dict()) + "\n")
        except OSError as exc:
            logger.error("[OMS] Audit log write failed: %s", exc)


def _json_safe(value):
    """Recursively coerce values into JSON-serialisable types."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    return value
