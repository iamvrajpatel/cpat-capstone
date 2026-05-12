"""Standalone analytics and charting page."""

from __future__ import annotations

import streamlit as st

from ui.components.charts.performance_charts import drawdown_chart, equity_curve_chart, rolling_metric_chart
from ui.components.metrics.kpi_cards import show_kpis
from ui.services import UIServiceBundle


def render(services: UIServiceBundle) -> None:
    """Render saved-result analytics."""
    st.title("Analytics")
    descriptors = services.strategy_service.list_strategies(
        services.config_service.load_config(services.config_path)
    )
    strategy = st.selectbox(
        "Result Namespace",
        options=[descriptor.key for descriptor in descriptors],
        format_func=lambda key: next(item.label for item in descriptors if item.key == key),
    )
    artifacts = services.analytics_service.discover_artifacts(strategy)
    equity = services.analytics_service.load_series(artifacts.equity_curve or artifacts.walk_forward)
    if equity.empty:
        st.warning("No saved equity curve available for the selected namespace.")
        return

    frame = services.analytics_service.equity_drawdown_frame(equity)
    rolling = services.analytics_service.rolling_sharpe_frame(equity)
    distribution = services.analytics_service.distribution_summary(equity)
    report = services.analytics_service.performance_report_from_equity(equity, float(equity.iloc[0]))

    show_kpis(
        {
            "Sharpe": f"{report['sharpe_ratio']:.3f}",
            "Sortino": f"{report['sortino_ratio']:.3f}",
            "Max Drawdown": f"{report['max_drawdown_pct']:.2f}%",
            "Tail Ratio": f"{distribution.get('tail_ratio', 1.0):.3f}",
            "VaR 95": f"{distribution.get('var_95', 0.0):.2%}",
            "CVaR 95": f"{distribution.get('cvar_95', 0.0):.2%}",
        },
        columns=3,
    )

    col1, col2 = st.columns(2)
    col1.plotly_chart(equity_curve_chart(frame), use_container_width=True)
    col2.plotly_chart(drawdown_chart(frame), use_container_width=True)
    col3, col4 = st.columns(2)
    col3.plotly_chart(rolling_metric_chart(rolling, "rolling_sharpe", "Rolling Sharpe"), use_container_width=True)
    col4.plotly_chart(rolling_metric_chart(rolling, "rolling_return", "Rolling Return"), use_container_width=True)
