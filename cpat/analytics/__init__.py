"""
CPAT Analytics Package
========================
Performance metrics, drawdown analysis, return distributions, and trade log.
"""

from cpat.analytics.performance import PerformanceReport, PerformanceTracker
from cpat.analytics.trade_log import TradeLog
from cpat.analytics.drawdown import (
    DrawdownPeriod,
    compute_drawdown_series,
    compute_drawdown_table,
    drawdown_table_to_dataframe,
    max_drawdown_details,
    ulcer_index,
)
from cpat.analytics.distributions import (
    ReturnDistribution,
    compute_distribution,
    is_normal,
    monthly_returns_table,
    return_heatmap_data,
)

__all__ = [
    "PerformanceReport",
    "PerformanceTracker",
    "TradeLog",
    "DrawdownPeriod",
    "compute_drawdown_series",
    "compute_drawdown_table",
    "drawdown_table_to_dataframe",
    "max_drawdown_details",
    "ulcer_index",
    "ReturnDistribution",
    "compute_distribution",
    "is_normal",
    "monthly_returns_table",
    "return_heatmap_data",
]
