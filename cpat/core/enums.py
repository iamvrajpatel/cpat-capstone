"""
CPAT — Core Enumerations
========================
All system-wide enumerations are defined here. Keeping enums in a single
module prevents circular imports and provides a canonical reference for
every other module in the system.
"""

from enum import Enum, auto


class OrderSide(Enum):
    """Direction of a trade order."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Execution type of an order."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    """Lifecycle state of an order."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class SignalDirection(Enum):
    """Raw directional bias produced by a strategy."""

    LONG = 1
    FLAT = 0
    SHORT = -1


class AssetClass(Enum):
    """Broad asset classification."""

    EQUITY = "EQUITY"
    ETF = "ETF"
    INDEX = "INDEX"
    FUTURES = "FUTURES"
    CRYPTO = "CRYPTO"


class Interval(Enum):
    """Bar frequency identifier — maps to data-provider string literals."""

    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1wk"
    ONE_MONTH = "1mo"


class EventType(Enum):
    """All event types flowing through the backtest event bus."""

    MARKET = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()
    RISK = auto()
    SYSTEM = auto()


class ExecutionMode(Enum):
    """Operational mode of the trading engine."""

    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"


class SlippageModel(Enum):
    """Slippage estimation strategy."""

    ZERO = "ZERO"
    FIXED_BPS = "FIXED_BPS"
    VOLUME_WEIGHTED = "VOLUME_WEIGHTED"
    SPREAD_BASED = "SPREAD_BASED"


class CommissionModel(Enum):
    """Commission calculation method."""

    ZERO = "ZERO"
    PER_SHARE = "PER_SHARE"
    PERCENTAGE = "PERCENTAGE"
    TIERED = "TIERED"
