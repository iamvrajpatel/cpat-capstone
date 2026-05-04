"""CPAT backtest package."""

from cpat.backtest.engine import BacktestEngine, BacktestResult
from cpat.backtest.event_queue import EventQueue
from cpat.backtest.handlers import (
    FillEventHandler,
    MarketEventHandler,
    OrderEventHandler,
    RiskEventHandler,
    SignalEventHandler,
)

# Backwards compatibility alias — SimulatedExecutionEngine moved to cpat.execution
from cpat.execution.engine import ExecutionEngine as SimulatedExecutionEngine

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "SimulatedExecutionEngine",
    "EventQueue",
    "FillEventHandler",
    "MarketEventHandler",
    "OrderEventHandler",
    "RiskEventHandler",
    "SignalEventHandler",
]
