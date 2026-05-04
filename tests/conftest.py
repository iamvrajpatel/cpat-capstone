"""
Pytest configuration and shared fixtures for the CPAT test suite.

All fixtures follow the Dependency Injection pattern used in the main codebase.
No network requests are made in any unit test (integration tests are marked
with @pytest.mark.integration and excluded by default).
"""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from cpat.config.loader import (
    BacktestConfig,
    BrokerConfig,
    CommissionConfig,
    CostsConfig,
    CPATConfig,
    DataConfig,
    LoggingConfig,
    MeanReversionStrategyConfig,
    MomentumStrategyConfig,
    RiskConfig,
    SlippageConfig,
    StrategiesConfig,
    SystemConfig,
    UniverseConfig,
)
from cpat.core.enums import (
    CommissionModel,
    ExecutionMode,
    Interval,
    OrderSide,
    OrderType,
    SlippageModel,
)
from cpat.core.models import Bar, CostConfig, Fill, Order, Position, Signal
from cpat.core.enums import SignalDirection
from cpat.backtest.event_queue import EventQueue


# ── Config fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def minimal_config() -> CPATConfig:
    """A minimal valid CPATConfig for testing (no network calls)."""
    return CPATConfig(
        system=SystemConfig(execution_mode=ExecutionMode.BACKTEST, log_level="DEBUG"),
        backtest=BacktestConfig(
            start_date=date(2022, 1, 1),
            end_date=date(2023, 12, 31),
            initial_capital=100_000.0,
            warmup_bars=0,
        ),
        universe=UniverseConfig(
            name="test_universe",
            equities=["AAPL", "MSFT", "GOOGL"],
        ),
        data=DataConfig(provider="csv"),
        costs=CostsConfig(
            commission=CommissionConfig(model=CommissionModel.PERCENTAGE, value=0.001),
            slippage=SlippageConfig(model=SlippageModel.FIXED_BPS, bps=5.0),
        ),
        strategies=StrategiesConfig(
            momentum=MomentumStrategyConfig(lookback_short=5, lookback_long=20),
            mean_reversion=MeanReversionStrategyConfig(lookback_window=10),
        ),
        risk=RiskConfig(),
        broker=BrokerConfig(),
        logging=LoggingConfig(),
    )


@pytest.fixture
def temp_data_dir() -> Path:
    """Temporary directory for data store tests (cleaned up automatically)."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """50 days of synthetic OHLCV data for a single symbol."""
    import numpy as np
    n = 50
    dates = pd.date_range("2022-01-01", periods=n, freq="B", tz="UTC")
    p = np.array([100.0 + i * 0.5 for i in range(n)], dtype=float)
    df = pd.DataFrame(
        {
            "open": p,
            "high": p * 1.01,
            "low": p * 0.99,
            "close": p,
            "adj_close": p,
            "volume": np.full(n, 1_000_000.0),
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df


@pytest.fixture
def sample_bar() -> Bar:
    """A single valid Bar for use in unit tests."""
    return Bar(
        symbol="AAPL",
        timestamp=pd.Timestamp("2022-06-01", tz="UTC"),
        open=150.0,
        high=155.0,
        low=148.0,
        close=153.0,
        volume=50_000_000.0,
        adj_close=153.0,
    )


@pytest.fixture
def sample_order() -> Order:
    """A market BUY order for testing."""
    return Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100.0,
        timestamp=pd.Timestamp("2022-06-01", tz="UTC"),
        strategy_id="test_strategy",
    )


@pytest.fixture
def sample_fill(sample_order: Order) -> Fill:
    """A fill corresponding to sample_order."""
    return Fill(
        order_id=sample_order.order_id,
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100.0,
        fill_price=150.25,
        commission=1.50,
        slippage=0.25,
        timestamp=pd.Timestamp("2022-06-02", tz="UTC"),
    )


@pytest.fixture
def event_queue() -> EventQueue:
    """Fresh EventQueue for testing."""
    return EventQueue()


@pytest.fixture
def cost_config_zero() -> CostConfig:
    """Zero-cost CostConfig for clean P&L tests."""
    return CostConfig(
        commission_model=CommissionModel.ZERO,
        slippage_model=SlippageModel.ZERO,
    )
