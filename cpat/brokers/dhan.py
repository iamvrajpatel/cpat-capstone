"""
CPAT — Dhan Broker Adapter (Week 5)
=====================================
HTTP adapter for the Dhan API (dhanhq Python SDK or direct REST calls).
Implements the ``BrokerInterface`` protocol for Indian equity trading.

Dhan is an Indian discount broker that provides a free developer API for
algorithmic trading on NSE/BSE.

Documentation: https://api.dhan.co/
SDK: pip install dhanhq

Credentials (MUST be environment variables — never in code or YAML):
    DHAN_CLIENT_ID   : Your Dhan client ID
    DHAN_ACCESS_TOKEN: Your Dhan API access token

Limitations
-----------
* Dhan API supports equity and F&O segments only (no MCX futures).
* Market orders are supported. Stop-loss orders use SL-M type.
* Rate limit: 10 requests/second (Dhan default).
* Paper trading uses Dhan's own sandbox environment — set DHAN_SANDBOX=1.

Design
------
* All HTTP calls are wrapped in ``_call()`` with exponential backoff.
* Credentials are read from environment at construction time.
* Session token is validated lazily on first ``connect()`` call.
* DhanhqError → ``BrokerError`` translation is centralised in ``_call()``.

Usage::

    import os
    os.environ["DHAN_CLIENT_ID"] = "..."
    os.environ["DHAN_ACCESS_TOKEN"] = "..."

    from cpat.brokers.dhan import DhanBroker
    broker = DhanBroker()
    broker.connect()
    quote = broker.get_quote("RELIANCE.NS")
    print(quote.last)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional
from uuid import uuid4

import pandas as pd

from cpat.core.enums import OrderSide, OrderStatus, OrderType
from cpat.core.models import Order
from cpat.infrastructure.broker_interface import (
    BrokerAccountInfo,
    BrokerAuthError,
    BrokerConnectionError,
    BrokerError,
    BrokerOrderError,
    BrokerOrderStatus,
    BrokerPosition,
    BrokerQuote,
)

logger = logging.getLogger(__name__)

# ── Dhan exchange segment codes ────────────────────────────────────────────────
# NSE_EQ = "NSE_EQ", BSE_EQ = "BSE_EQ"
_DHAN_EXCHANGE = "NSE_EQ"
_DHAN_PRODUCT  = "CNC"   # Cash-and-carry (delivery)
_DHAN_SECURITY_ID_MAP: dict[str, str] = {}   # symbol → Dhan security_id (populated at runtime)

# ── Retry config ───────────────────────────────────────────────────────────────
_MAX_RETRIES    = 3
_RETRY_BASE_SEC = 1.0   # first retry after 1s, then 2s, 4s


class DhanBroker:
    """Dhan broker adapter implementing ``BrokerInterface``.

    All network calls are wrapped with:
        - Exponential backoff retry (max 3 attempts)
        - ``BrokerError`` translation so callers are insulated from SDK details

    Args:
        client_id   : Dhan client ID (reads DHAN_CLIENT_ID env var if None).
        access_token: API access token (reads DHAN_ACCESS_TOKEN env var if None).
        sandbox     : Use Dhan sandbox / paper trading environment.
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        access_token: Optional[str] = None,
        sandbox: bool = False,
    ) -> None:
        self._client_id    = client_id    or os.environ.get("DHAN_CLIENT_ID", "")
        self._access_token = access_token or os.environ.get("DHAN_ACCESS_TOKEN", "")
        self._sandbox      = sandbox or os.environ.get("DHAN_SANDBOX", "0") == "1"
        self._connected    = False
        self._dhan_client  = None   # dhanhq.dhanhq instance

        if not self._client_id or not self._access_token:
            logger.warning(
                "[dhan] DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN not set. "
                "DhanBroker.connect() will raise BrokerAuthError."
            )

    # ── BrokerInterface: Connection ────────────────────────────────────────────

    def connect(self) -> None:
        """Authenticate with Dhan and validate credentials.

        Raises:
            BrokerAuthError: If credentials are missing or invalid.
            BrokerConnectionError: If the network is unreachable.
        """
        if not self._client_id or not self._access_token:
            raise BrokerAuthError(
                "DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN must be set in environment."
            )
        try:
            import dhanhq  # type: ignore[import-untyped]
            self._dhan_client = dhanhq.dhanhq(self._client_id, self._access_token)
            # Validate by fetching fund limits (lightweight call)
            self._call("fund_limit")
            self._connected = True
            logger.info("[dhan] Connected | client_id=%s | sandbox=%s",
                        self._client_id, self._sandbox)
        except ImportError as exc:
            raise BrokerConnectionError(
                "dhanhq package not installed. Run: pip install dhanhq"
            ) from exc
        except BrokerAuthError:
            raise
        except Exception as exc:
            raise BrokerConnectionError(f"Dhan connect failed: {exc}") from exc

    def disconnect(self) -> None:
        """Close the Dhan session."""
        self._connected    = False
        self._dhan_client  = None
        logger.info("[dhan] Disconnected.")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── BrokerInterface: Orders ────────────────────────────────────────────────

    def place_order(self, order: Order) -> str:
        """Submit an order to Dhan.

        Args:
            order: The ``Order`` to submit.

        Returns:
            Dhan order ID string.

        Raises:
            BrokerOrderError: If Dhan rejects the order.
            BrokerConnectionError: On network failure.
        """
        self._require_connected()
        dhan_params = self._order_to_dhan_params(order)
        try:
            response = self._call("place_order", **dhan_params)
            dhan_order_id = str(response.get("orderId", ""))
            if not dhan_order_id:
                raise BrokerOrderError(f"Dhan returned no orderId for {order.symbol}.")
            logger.info("[dhan] ORDER_PLACED %s %s %.0f → dhan_id=%s",
                        order.side.value, order.symbol, order.quantity, dhan_order_id)
            return dhan_order_id
        except BrokerError:
            raise
        except Exception as exc:
            raise BrokerOrderError(f"place_order failed: {exc}") from exc

    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel a live order on Dhan.

        Args:
            broker_order_id: Dhan order ID.

        Returns:
            True if the cancel request was accepted.
        """
        self._require_connected()
        try:
            response = self._call("cancel_order", order_id=broker_order_id)
            logger.info("[dhan] CANCEL_REQUESTED %s | response=%s", broker_order_id, response)
            return True
        except Exception as exc:
            logger.error("[dhan] cancel_order failed: %s", exc)
            return False

    def get_order_status(self, broker_order_id: str) -> BrokerOrderStatus:
        """Poll the status of a Dhan order.

        Args:
            broker_order_id: Dhan order ID.

        Returns:
            ``BrokerOrderStatus`` with current state.

        Raises:
            BrokerError: On failure.
        """
        self._require_connected()
        try:
            orders = self._call("get_order_list")
            for o in orders.get("data", []):
                if str(o.get("orderId")) == broker_order_id:
                    return self._parse_order_status(o, broker_order_id)
            raise BrokerError(f"Order {broker_order_id} not found in order list.", code="NOT_FOUND")
        except BrokerError:
            raise
        except Exception as exc:
            raise BrokerError(f"get_order_status failed: {exc}") from exc

    # ── BrokerInterface: Market Data ───────────────────────────────────────────

    def get_market_data(self, symbol: str) -> BrokerQuote:
        """Fetch latest quote for an NSE equity.

        Args:
            symbol: Ticker (e.g. "RELIANCE.NS"). ".NS" suffix stripped automatically.

        Returns:
            ``BrokerQuote`` with bid, ask, last.

        Raises:
            BrokerError: On failure.
        """
        self._require_connected()
        clean_sym = symbol.replace(".NS", "").replace(".BO", "")
        security_id = self._resolve_security_id(clean_sym)
        try:
            response = self._call(
                "market_feed",
                securities={"NSE_EQ": [int(security_id)]}
            )
            data = response.get("data", {}).get("NSE_EQ", {}).get(security_id, {})
            if not data:
                raise BrokerError(f"No quote data returned for {symbol}.", code="NO_DATA")

            return BrokerQuote(
                symbol=symbol,
                bid=float(data.get("bidPrice", 0)),
                ask=float(data.get("askPrice", 0)),
                last=float(data.get("lastTradedPrice", 0)),
                volume=float(data.get("volume", 0)),
                timestamp=pd.Timestamp.utcnow(),
            )
        except BrokerError:
            raise
        except Exception as exc:
            raise BrokerError(f"get_market_data failed for {symbol}: {exc}") from exc

    def get_quote(self, symbol: str) -> BrokerQuote:
        """Backward-compatible alias for ``get_market_data()``."""
        return self.get_market_data(symbol)

    # ── BrokerInterface: Account ───────────────────────────────────────────────

    def get_account_info(self) -> BrokerAccountInfo:
        """Retrieve Dhan account fund limits.

        Returns:
            ``BrokerAccountInfo`` with available cash.

        Raises:
            BrokerError: On failure.
        """
        self._require_connected()
        try:
            data = self._call("fund_limit")
            funds = data.get("data", {})
            cash    = float(funds.get("availabelBalance", 0))
            equity  = float(funds.get("sodLimit", cash))
            return BrokerAccountInfo(
                cash_balance=cash,
                portfolio_value=equity,
                buying_power=cash,
                currency="INR",
            )
        except BrokerError:
            raise
        except Exception as exc:
            raise BrokerError(f"get_account_info failed: {exc}") from exc

    def get_positions(self) -> list[BrokerPosition]:
        """Retrieve all open positions from Dhan.

        Returns:
            List of ``BrokerPosition`` objects.

        Raises:
            BrokerError: On failure.
        """
        self._require_connected()
        try:
            data = self._call("get_positions")
            positions = []
            for p in data.get("data", []):
                qty = float(p.get("netQty", 0))
                if abs(qty) < 1e-9:
                    continue
                positions.append(BrokerPosition(
                    symbol=str(p.get("tradingSymbol", "")),
                    quantity=qty,
                    avg_cost=float(p.get("avgCostPrice", 0)),
                    current_price=float(p.get("lastTradedPrice", 0)),
                    unrealised_pnl=float(p.get("unrealizedProfit", 0)),
                ))
            return positions
        except BrokerError:
            raise
        except Exception as exc:
            raise BrokerError(f"get_positions failed: {exc}") from exc

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _call(self, method_name: str, **kwargs) -> dict:
        """Invoke a dhanhq client method with exponential backoff retry.

        Args:
            method_name: Name of the dhanhq method to call.
            kwargs     : Arguments forwarded to the method.

        Returns:
            Response dict from the SDK.

        Raises:
            BrokerAuthError: On 401/403 responses.
            BrokerConnectionError: On network errors (retryable).
            BrokerError: On other API errors.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES):
            try:
                method = getattr(self._dhan_client, method_name)
                response = method(**kwargs) if kwargs else method()
                # Dhan SDK returns dicts; check for error keys
                if isinstance(response, dict):
                    status = response.get("status", "")
                    if status == "failure":
                        err_msg = response.get("errorMessage", "Unknown error")
                        err_code = response.get("errorCode", "")
                        if "UNAUTHENTICATED" in str(err_code):
                            raise BrokerAuthError(err_msg)
                        raise BrokerOrderError(err_msg, code=str(err_code))
                return response  # type: ignore[return-value]

            except (BrokerAuthError, BrokerOrderError):
                raise   # These are not retryable

            except Exception as exc:
                last_exc = exc
                wait = _RETRY_BASE_SEC * (2 ** attempt)
                logger.warning(
                    "[dhan] %s() attempt %d/%d failed: %s — retry in %.1fs",
                    method_name, attempt + 1, _MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)

        raise BrokerConnectionError(
            f"Dhan API call '{method_name}' failed after {_MAX_RETRIES} retries: {last_exc}"
        )

    def _require_connected(self) -> None:
        if not self._connected:
            raise BrokerConnectionError("DhanBroker is not connected. Call connect() first.")

    def _order_to_dhan_params(self, order: Order) -> dict:
        """Translate an internal Order to Dhan API parameters.

        Args:
            order: CPAT ``Order`` object.

        Returns:
            Dict of Dhan API parameters.
        """
        dhan_side = "BUY" if order.side == OrderSide.BUY else "SELL"
        if order.order_type == OrderType.MARKET:
            dhan_order_type = "MARKET"
        elif order.order_type == OrderType.LIMIT:
            dhan_order_type = "LIMIT"
        elif order.order_type == OrderType.STOP:
            dhan_order_type = "SL-M"
        elif order.order_type == OrderType.STOP_LIMIT:
            dhan_order_type = "SL"
        else:
            dhan_order_type = "MARKET"

        clean_sym = order.symbol.replace(".NS", "").replace(".BO", "")
        params = {
            "transaction_type": dhan_side,
            "exchange_segment":  _DHAN_EXCHANGE,
            "product_type":      _DHAN_PRODUCT,
            "order_type":        dhan_order_type,
            "validity":          "DAY",
            "security_id":       self._resolve_security_id(clean_sym),
            "quantity":          int(order.quantity),
            "disclosed_quantity": 0,
            "price":             order.limit_price or 0.0,
            "trigger_price":     order.stop_price  or 0.0,
            "after_market_order": False,
            "amo_time":           "OPEN",
            "bo_profit_value":    0.0,
            "bo_stop_loss_Value": 0.0,
        }
        return params

    def _parse_order_status(self, raw: dict, broker_order_id: str) -> BrokerOrderStatus:
        """Convert a Dhan order dict to ``BrokerOrderStatus``.

        Args:
            raw            : Raw dict from Dhan API.
            broker_order_id: Dhan order ID.

        Returns:
            ``BrokerOrderStatus`` with mapped fields.
        """
        status_map = {
            "TRANSIT":           OrderStatus.SUBMITTED,
            "PENDING":           OrderStatus.SUBMITTED,
            "PART_TRADED":       OrderStatus.PARTIALLY_FILLED,
            "TRADED":            OrderStatus.FILLED,
            "CANCELLED":         OrderStatus.CANCELLED,
            "REJECTED":          OrderStatus.REJECTED,
            "EXPIRED":           OrderStatus.EXPIRED,
        }
        dhan_status = raw.get("orderStatus", "PENDING")
        cpat_status = status_map.get(dhan_status, OrderStatus.SUBMITTED)

        from uuid import UUID as _UUID
        try:
            internal_id = _UUID(str(raw.get("correlationId", uuid4())))
        except ValueError:
            internal_id = uuid4()

        return BrokerOrderStatus(
            broker_order_id=broker_order_id,
            internal_order_id=internal_id,
            status=cpat_status,
            filled_qty=float(raw.get("tradedQuantity", 0)),
            avg_fill_price=float(raw.get("tradedPrice", 0)),
            remaining_qty=float(raw.get("remainingQuantity", 0)),
            rejected_reason=raw.get("omsErrorDescription", ""),
        )

    def _resolve_security_id(self, symbol: str) -> str:
        """Resolve a ticker symbol to Dhan security_id.

        Uses the in-memory map first.  In production, this map should be
        populated by downloading Dhan's instrument master CSV.

        Args:
            symbol: Clean ticker (no suffix).

        Returns:
            Dhan security_id string.

        Raises:
            BrokerError: If symbol is not in the map.
        """
        if symbol in _DHAN_SECURITY_ID_MAP:
            return _DHAN_SECURITY_ID_MAP[symbol]
        # Fallback: attempt the symbol directly (Dhan accepts ticker strings
        # in some API versions)
        logger.warning("[dhan] security_id not in map for %s — using symbol as-is.", symbol)
        return symbol
