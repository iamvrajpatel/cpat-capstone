"""Service bundle definitions for the Streamlit control console."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ui.services.analytics_service import AnalyticsService
from ui.services.config_service import ConfigService
from ui.services.execution_service import ExecutionService
from ui.services.strategy_service import StrategyService
from ui.state.session_manager import SessionManager


@dataclass(frozen=True)
class UIServiceBundle:
    """Typed service container shared across Streamlit pages."""

    config_path: Path
    session: SessionManager
    config_service: ConfigService
    execution_service: ExecutionService
    analytics_service: AnalyticsService
    strategy_service: StrategyService

