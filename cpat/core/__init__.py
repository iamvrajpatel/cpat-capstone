"""CPAT core package."""

from cpat.core.enums import (
    AssetClass,
    CommissionModel,
    EventType,
    ExecutionMode,
    Interval,
    OrderSide,
    OrderStatus,
    OrderType,
    SignalDirection,
    SlippageModel,
)
from cpat.core.events import (
    AnyEvent,
    FillEvent,
    MarketEvent,
    OrderEvent,
    RiskEvent,
    SignalEvent,
    SystemEvent,
)
from cpat.core.models import (
    Bar,
    CostConfig,
    Fill,
    Instrument,
    Order,
    Position,
    Signal,
    Tick,
)

__all__ = [
    # enums
    "AssetClass",
    "CommissionModel",
    "EventType",
    "ExecutionMode",
    "Interval",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "SignalDirection",
    "SlippageModel",
    # models
    "Bar",
    "CostConfig",
    "Fill",
    "Instrument",
    "Order",
    "Position",
    "Signal",
    "Tick",
    # events
    "AnyEvent",
    "FillEvent",
    "MarketEvent",
    "OrderEvent",
    "RiskEvent",
    "SignalEvent",
    "SystemEvent",
]
