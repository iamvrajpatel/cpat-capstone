"""
CPAT — Live Trading Structured Logger (Week 5)
===============================================
Dedicated logger for live trading events: signals, orders, fills, errors,
and system heartbeats.

Writes two streams simultaneously:
    1. Console (human-readable, coloured)
    2. Rotating JSONL file (machine-readable, one event per line)

JSONL format is parseable by any downstream monitoring tool (Grafana,
ELK, Datadog, etc.) without preprocessing.

All records include a mandatory ``event_type`` field so dashboards can
filter by category without regex.

Usage::

    from cpat.infrastructure.live_logger import LiveTradeLogger

    log = LiveTradeLogger(log_dir="logs/live")
    log.signal("RELIANCE.NS", direction="LONG", strength=0.85)
    log.order_placed("RELIANCE.NS", qty=10, side="BUY", broker_id="BRK001")
    log.fill("RELIANCE.NS", qty=10, price=2500.50, commission=7.50)
    log.error("API timeout", symbol="TCS.NS", retry=1)
    log.heartbeat(equity=10_235_000.0, n_positions=5, n_open_orders=2)
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class LiveTradeLogger:
    """Structured logger for live trading events.

    Every method writes to both a rotating JSONL file and the console.
    Methods never raise — failures are silently logged to stderr to avoid
    disrupting the trading loop.

    Args:
        log_dir    : Directory for log files (created if missing).
        log_prefix : Filename prefix for log files.
        max_bytes  : Max bytes per JSONL file before rotation (default 20 MB).
        backup_count: Number of rotated files to keep.
        console    : If True, also write coloured output to stdout.
    """

    def __init__(
        self,
        log_dir: str | Path = "logs/live",
        log_prefix: str = "trading",
        max_bytes: int = 20_971_520,  # 20 MB
        backup_count: int = 10,
        console: bool = True,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._console = console

        # ── JSONL rotating file handler ────────────────────────────────────────
        log_path = self._log_dir / f"{log_prefix}.jsonl"
        self._file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        self._file_handler.setFormatter(logging.Formatter("%(message)s"))

        # Internal Python logger (file only)
        self._log = logging.getLogger(f"cpat.live.{log_prefix}")
        self._log.setLevel(logging.DEBUG)
        self._log.propagate = False
        self._log.addHandler(self._file_handler)

    # ── Public event methods ────────────────────────────────────────────────────

    def signal(
        self,
        symbol: str,
        direction: str,
        strategy_id: str = "",
        strength: float = 0.0,
        **extra: Any,
    ) -> None:
        """Log a signal generation event.

        Args:
            symbol     : Instrument ticker.
            direction  : "LONG", "SHORT", or "FLAT".
            strategy_id: Strategy that produced the signal.
            strength   : Signal strength in [-1, 1].
        """
        self._write("SIGNAL", symbol=symbol, direction=direction,
                    strategy_id=strategy_id, strength=round(strength, 4), **extra)

    def order_placed(
        self,
        symbol: str,
        qty: float,
        side: str,
        broker_order_id: str = "",
        order_type: str = "MARKET",
        price: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """Log an order submission.

        Args:
            symbol         : Ticker.
            qty            : Quantity.
            side           : "BUY" or "SELL".
            broker_order_id: Broker-assigned ID.
            order_type     : "MARKET", "LIMIT", etc.
            price          : Limit/stop price (if applicable).
        """
        self._write("ORDER_PLACED", symbol=symbol, qty=qty, side=side,
                    broker_order_id=broker_order_id, order_type=order_type,
                    price=price, **extra)

    def fill(
        self,
        symbol: str,
        qty: float,
        price: float,
        side: str,
        commission: float = 0.0,
        broker_order_id: str = "",
        **extra: Any,
    ) -> None:
        """Log a fill/execution confirmation.

        Args:
            symbol         : Ticker.
            qty            : Filled quantity.
            price          : Fill price.
            side           : "BUY" or "SELL".
            commission     : Commission charged.
            broker_order_id: Broker order reference.
        """
        self._write("FILL", symbol=symbol, qty=qty, price=round(price, 4),
                    side=side, commission=round(commission, 4),
                    broker_order_id=broker_order_id, **extra)

    def order_rejected(
        self,
        symbol: str,
        reason: str,
        broker_order_id: str = "",
        **extra: Any,
    ) -> None:
        """Log an order rejection.

        Args:
            symbol         : Ticker.
            reason         : Rejection message.
            broker_order_id: Broker order reference.
        """
        self._write("ORDER_REJECTED", symbol=symbol, reason=reason,
                    broker_order_id=broker_order_id, level="WARNING", **extra)

    def order_cancelled(self, symbol: str, broker_order_id: str = "", **extra: Any) -> None:
        """Log an order cancellation."""
        self._write("ORDER_CANCELLED", symbol=symbol,
                    broker_order_id=broker_order_id, **extra)

    def risk_block(self, symbol: str, constraint: str, reason: str, **extra: Any) -> None:
        """Log a risk constraint block.

        Args:
            symbol    : Blocked ticker.
            constraint: Which constraint blocked the order.
            reason    : Human-readable explanation.
        """
        self._write("RISK_BLOCK", symbol=symbol, constraint=constraint,
                    reason=reason, level="WARNING", **extra)

    def order_retry(
        self,
        symbol: str,
        attempt: int,
        max_attempts: int,
        reason: str,
        **extra: Any,
    ) -> None:
        """Log an order placement retry/backoff event."""
        self._write(
            "ORDER_RETRY",
            symbol=symbol,
            attempt=attempt,
            max_attempts=max_attempts,
            reason=reason,
            level="WARNING",
            **extra,
        )

    def data_issue(
        self,
        symbol: Optional[str],
        issue: str,
        **extra: Any,
    ) -> None:
        """Log stale or missing market-data issues."""
        self._write("DATA_ISSUE", symbol=symbol, issue=issue, level="WARNING", **extra)

    def reconciliation(
        self,
        status: str,
        message: str,
        **extra: Any,
    ) -> None:
        """Log broker/account reconciliation outcome."""
        level = "INFO" if status == "ok" else "ERROR"
        self._write("RECONCILIATION", status=status, message=message, level=level, **extra)

    def stop_triggered(self, symbol: str, stop_price: float, current_price: float, **extra: Any) -> None:
        """Log a stop-loss or take-profit trigger.

        Args:
            symbol       : Ticker.
            stop_price   : The stop level that was hit.
            current_price: The price that triggered the stop.
        """
        self._write("STOP_TRIGGERED", symbol=symbol,
                    stop_price=round(stop_price, 4),
                    current_price=round(current_price, 4), **extra)

    def error(
        self,
        message: str,
        exc: Optional[Exception] = None,
        symbol: Optional[str] = None,
        retry: int = 0,
        **extra: Any,
    ) -> None:
        """Log an error event.

        Args:
            message: Error description.
            exc    : Exception object (traceback included if present).
            symbol : Affected symbol (if any).
            retry  : Retry attempt number.
        """
        exc_str = f"{type(exc).__name__}: {exc}" if exc else ""
        self._write("ERROR", message=message, exc=exc_str, symbol=symbol,
                    retry=retry, level="ERROR", **extra)

    def heartbeat(
        self,
        equity: float,
        n_positions: int,
        n_open_orders: int,
        drawdown_pct: float = 0.0,
        **extra: Any,
    ) -> None:
        """Log a periodic system heartbeat.

        Args:
            equity        : Current portfolio equity.
            n_positions   : Number of open positions.
            n_open_orders : Number of live orders in OMS.
            drawdown_pct  : Current drawdown from peak (fraction).
        """
        self._write("HEARTBEAT", equity=round(equity, 2),
                    n_positions=n_positions, n_open_orders=n_open_orders,
                    drawdown_pct=round(drawdown_pct, 4), **extra)

    def system(self, message: str, **extra: Any) -> None:
        """Log a system lifecycle event (START, STOP, RECONNECT, etc.).

        Args:
            message: Lifecycle message.
        """
        self._write("SYSTEM", message=message, level="INFO", **extra)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _write(self, event_type: str, level: str = "INFO", **fields: Any) -> None:
        """Compose and emit a structured log record.

        Args:
            event_type: Category string (e.g. "SIGNAL", "FILL").
            level     : Log level string.
            fields    : Arbitrary key-value payload.
        """
        try:
            record = {
                "ts": pd.Timestamp.utcnow().isoformat(),
                "event": event_type,
                **{k: v for k, v in fields.items() if v is not None},
            }
            line = json.dumps(record, default=str)

            # File
            log_fn = getattr(self._log, level.lower(), self._log.info)
            log_fn(line)

            # Console
            if self._console:
                colour = _LEVEL_COLORS.get(level, "")
                reset  = "\033[0m"
                symbol = fields.get("symbol", "")
                tag    = f"[{event_type}]" + (f" {symbol}" if symbol else "")
                detail = {k: v for k, v in fields.items()
                          if k not in ("symbol",) and v is not None}
                detail_str = "  ".join(f"{k}={v}" for k, v in detail.items())
                ts_short = record["ts"][11:19]  # HH:MM:SS
                print(f"{colour}{ts_short} {tag:<28} {detail_str}{reset}", flush=True)

        except Exception as exc:  # noqa: BLE001
            print(f"[LiveTradeLogger._write FAILED] {exc}", file=sys.stderr)


_LEVEL_COLORS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[35m",   # Magenta
}
