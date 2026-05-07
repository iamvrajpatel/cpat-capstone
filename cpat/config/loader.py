"""
CPAT — Configuration Loader
============================
Loads and validates ``config/settings.yaml`` into typed Pydantic v2 models.
All components receive a ``CPATConfig`` instance via dependency injection;
no component reads the YAML file directly.

Usage::

    from cpat.config.loader import load_config

    cfg = load_config()                          # uses default path
    cfg = load_config("config/my_settings.yaml") # custom path

    # Access typed fields:
    print(cfg.backtest.initial_capital)   # float
    print(cfg.universe.equities)          # list[str]
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from cpat.core.enums import (
    CommissionModel,
    ExecutionMode,
    Interval,
    SlippageModel,
)

_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "config" / "settings.yaml"


# ── Sub-models ─────────────────────────────────────────────────────────────────


class SystemConfig(BaseModel):
    """System-level operational settings."""

    execution_mode: ExecutionMode = ExecutionMode.BACKTEST
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    random_seed: int = 42
    timezone: str = "UTC"


class BacktestConfig(BaseModel):
    """Backtest simulation parameters."""

    start_date: date
    end_date: date
    initial_capital: Annotated[float, Field(gt=0)] = 1_000_000.0
    benchmark_symbol: str = "SPY"
    warmup_bars: Annotated[int, Field(ge=0)] = 252

    @model_validator(mode="after")
    def validate_date_range(self) -> "BacktestConfig":
        if self.start_date >= self.end_date:
            raise ValueError(
                f"start_date ({self.start_date}) must be before end_date ({self.end_date})"
            )
        return self


class UniverseConfig(BaseModel):
    """Instrument universe definition."""

    name: str = "default"
    interval: str = "1d"
    etfs: list[str] = Field(default_factory=list)
    indices: list[str] = Field(default_factory=list)
    equities: list[str] = Field(default_factory=list)

    @property
    def all_symbols(self) -> list[str]:
        """Return deduplicated list of all symbols across all categories."""
        seen: set[str] = set()
        result: list[str] = []
        for sym in self.etfs + self.indices + self.equities:
            sym_upper = sym.upper()
            if sym_upper not in seen:
                seen.add(sym_upper)
                result.append(sym_upper)
        return result

    @property
    def interval_enum(self) -> Interval:
        """Return the Interval enum corresponding to the configured string."""
        return Interval(self.interval)


class DataConfig(BaseModel):
    """Data pipeline settings."""

    provider: Literal["yahoo", "csv", "polygon"] = "yahoo"
    store_type: Literal["parquet"] = "parquet"
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    cache_path: str = "data/cache"
    download_retry_limit: Annotated[int, Field(ge=1)] = 3
    download_retry_delay_seconds: Annotated[float, Field(ge=0)] = 5.0
    max_missing_bar_pct: Annotated[float, Field(ge=0, le=1)] = 0.02


class CommissionConfig(BaseModel):
    """Commission model parameters."""

    model: CommissionModel = CommissionModel.PERCENTAGE
    value: Annotated[float, Field(ge=0)] = 0.0005
    min_commission: Annotated[float, Field(ge=0)] = 1.0


class SlippageConfig(BaseModel):
    """Slippage model parameters."""

    model: SlippageModel = SlippageModel.FIXED_BPS
    bps: Annotated[float, Field(ge=0)] = 5.0
    slippage_model_name: str = "FIXED_BPS"  # FIXED_BPS | VOLUME_WEIGHTED | SPREAD_BASED


class CostsConfig(BaseModel):
    """Combined transaction cost configuration."""

    commission: CommissionConfig = Field(default_factory=CommissionConfig)
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)


class MomentumStrategyConfig(BaseModel):
    """Cross-sectional momentum strategy parameters."""

    enabled: bool = True
    lookback_short: Annotated[int, Field(gt=0)] = 21
    lookback_long: Annotated[int, Field(gt=0)] = 252
    skip_most_recent: Annotated[int, Field(ge=0)] = 21
    rebalance_frequency: Annotated[int, Field(gt=0)] = 21
    top_n_pct: Annotated[float, Field(gt=0, le=1)] = 0.2
    bottom_n_pct: Annotated[float, Field(ge=0, le=1)] = 0.2
    allow_short: bool = False

    @model_validator(mode="after")
    def validate_lookback(self) -> "MomentumStrategyConfig":
        if self.lookback_long <= self.lookback_short:
            raise ValueError("lookback_long must be greater than lookback_short")
        return self


class MeanReversionStrategyConfig(BaseModel):
    """Mean reversion strategy parameters."""

    enabled: bool = True
    lookback_window: Annotated[int, Field(gt=0)] = 20
    entry_zscore: Annotated[float, Field(gt=0)] = 2.0
    exit_zscore: Annotated[float, Field(ge=0)] = 0.5
    rsi_period: Annotated[int, Field(gt=0)] = 14
    rsi_oversold: Annotated[float, Field(ge=0, le=100)] = 30.0
    rsi_overbought: Annotated[float, Field(ge=0, le=100)] = 70.0
    allow_short: bool = False

    @model_validator(mode="after")
    def validate_zscore_order(self) -> "MeanReversionStrategyConfig":
        if self.exit_zscore >= self.entry_zscore:
            raise ValueError("exit_zscore must be less than entry_zscore")
        return self


class StrategiesConfig(BaseModel):
    """Container for all strategy configurations."""

    momentum: MomentumStrategyConfig = Field(default_factory=MomentumStrategyConfig)
    mean_reversion: MeanReversionStrategyConfig = Field(
        default_factory=MeanReversionStrategyConfig
    )


class RiskConfig(BaseModel):
    """Portfolio risk management constraints."""

    max_position_pct: Annotated[float, Field(gt=0, le=1)] = 0.05
    max_sector_pct: Annotated[float, Field(gt=0, le=1)] = 0.25
    max_gross_exposure_pct: Annotated[float, Field(gt=0, le=3.0)] = 1.0
    max_drawdown_pct: Annotated[float, Field(gt=0, le=1)] = 0.15
    min_cash_pct: Annotated[float, Field(ge=0, le=0.5)] = 0.03
    risk_free_rate: Annotated[float, Field(ge=0, le=0.2)] = 0.04
    allocation_method: Literal["equal_weight", "volatility_adjusted"] = "equal_weight"
    max_open_positions: Annotated[int, Field(ge=1)] = 20
    max_daily_loss_pct: Annotated[float, Field(gt=0, le=0.5)] = 0.02


class BrokerConfig(BaseModel):
    """Live/paper broker configuration."""

    provider: Literal["alpaca", "ibkr"] = "alpaca"
    paper_trading: bool = True


class LoggingConfig(BaseModel):
    """Logging infrastructure settings."""

    log_dir: str = "logs"
    max_bytes: int = 10_485_760   # 10 MB
    backup_count: int = 5


class OptimizationConfig(BaseModel):
    """Parameter optimization settings (Week 3)."""

    method: Literal["grid", "random"] = "random"
    n_trials: Annotated[int, Field(ge=1)] = 50
    metric: str = "sharpe_ratio"       # field name on PerformanceReport
    train_ratio: Annotated[float, Field(gt=0, lt=1)] = 0.70
    random_seed: int = 42


class ValidationConfig(BaseModel):
    """Walk-forward validation settings (Week 3)."""

    walk_forward_train_bars: Annotated[int, Field(gt=0)] = 756   # ~3yr daily
    walk_forward_test_bars: Annotated[int, Field(gt=0)] = 252    # ~1yr daily
    walk_forward_step_bars: Annotated[int, Field(gt=0)] = 63     # ~1 quarter
    optimize_on_fold: bool = False
    n_opt_trials: Annotated[int, Field(ge=1)] = 20


class PositionSizingConfig(BaseModel):
    """Position sizing configuration (Week 4)."""

    method: Literal["fixed_fractional", "inverse_volatility", "kelly"] = "inverse_volatility"
    stop_method: Literal["atr", "fixed_pct"] = "atr"
    risk_per_trade_pct: Annotated[float, Field(gt=0, le=0.1)] = 0.01   # 1% equity at risk
    fixed_stop_pct: Annotated[float, Field(gt=0, le=0.5)] = 0.02
    atr_period: Annotated[int, Field(ge=5, le=100)] = 14
    atr_multiplier: Annotated[float, Field(gt=0, le=10)] = 2.0         # stop = 2×ATR
    take_profit_mult: Optional[float] = 3.0                            # TP = entry + 3×ATR
    trailing_stop: bool = False
    vol_window: Annotated[int, Field(ge=5, le=252)] = 20
    min_trade_value: Annotated[float, Field(ge=0)] = 5_000.0
    # Kelly-specific
    kelly_win_rate: Annotated[float, Field(gt=0, lt=1)] = 0.55
    kelly_avg_win: Annotated[float, Field(gt=0)] = 1.5
    kelly_avg_loss: Annotated[float, Field(gt=0)] = 1.0
    kelly_max_fraction: Annotated[float, Field(gt=0, le=0.5)] = 0.25


# ── Root Config ────────────────────────────────────────────────────────────────


class CPATConfig(BaseModel):
    """Root configuration model for the entire CPAT system.

    Attributes:
        system: Operational mode and environment settings.
        backtest: Backtest simulation date range and capital.
        universe: Instrument universe composition.
        data: Data pipeline settings.
        costs: Transaction cost parameters.
        strategies: Per-strategy parameter namespaces.
        risk: Risk management constraints.
        broker: Broker connectivity settings.
        logging: Log file settings.
    """

    system: SystemConfig = Field(default_factory=SystemConfig)
    backtest: BacktestConfig
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    position_sizing: PositionSizingConfig = Field(default_factory=PositionSizingConfig)


# ── Loader ─────────────────────────────────────────────────────────────────────


def load_config(path: str | Path | None = None) -> CPATConfig:
    """Load and validate the CPAT configuration from a YAML file.

    Environment variable overrides are applied after YAML loading:
      - ``CPAT_EXECUTION_MODE`` → ``system.execution_mode``
      - ``CPAT_LOG_LEVEL`` → ``system.log_level``

    Args:
        path: Path to the YAML settings file.  Defaults to
              ``<project_root>/config/settings.yaml``.

    Returns:
        A fully validated ``CPATConfig`` instance.

    Raises:
        FileNotFoundError: If the settings file does not exist.
        pydantic.ValidationError: If the configuration is invalid.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            f"Copy config/settings.yaml.example and adjust."
        )

    with config_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    # Apply environment variable overrides
    if mode := os.getenv("CPAT_EXECUTION_MODE"):
        raw.setdefault("system", {})["execution_mode"] = mode
    if level := os.getenv("CPAT_LOG_LEVEL"):
        raw.setdefault("system", {})["log_level"] = level

    return CPATConfig.model_validate(raw)
