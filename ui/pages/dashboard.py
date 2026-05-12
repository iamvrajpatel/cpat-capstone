"""Portfolio and system dashboard page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.components.charts.performance_charts import drawdown_chart, equity_curve_chart
from ui.components.metrics.kpi_cards import show_kpis
from ui.components.tables.data_tables import show_latest_rows
from ui.services import UIServiceBundle


def render(services: UIServiceBundle) -> None:
    """Render the research and operations dashboard."""
    st.title("CPAT Control Console")
    st.caption("Institutional-grade orchestration for research, validation, and execution.")

    descriptors = services.strategy_service.list_strategies(
        services.config_service.load_config(services.config_path)
    )
    options = [descriptor.key for descriptor in descriptors]
    strategy = st.selectbox(
        "Result Namespace",
        options=options,
        index=options.index(services.session.get_selected_result())
        if services.session.get_selected_result() in options
        else 0,
        format_func=lambda key: next(item.label for item in descriptors if item.key == key),
    )
    services.session.set_selected_result(strategy)

    snapshot = services.analytics_service.latest_dashboard_snapshot(strategy)
    show_kpis(
        {
            "Total Equity": f"{snapshot.get('equity', 0.0):,.0f}",
            "Daily PnL": f"{snapshot.get('daily_pnl', 0.0):+,.0f}",
            "Gross Exposure": f"{snapshot.get('gross_exposure', 0.0):.2%}",
            "Net Exposure": f"{snapshot.get('net_exposure', 0.0):.2%}",
            "Open Positions": snapshot.get("open_positions", 0),
            "Tracked Trades": snapshot.get("tracked_trades", 0),
        },
        columns=3,
    )

    artifacts = services.analytics_service.discover_artifacts(strategy)
    equity = services.analytics_service.load_series(artifacts.equity_curve)
    frame = services.analytics_service.equity_drawdown_frame(equity)
    col1, col2 = st.columns(2)
    col1.plotly_chart(equity_curve_chart(frame, "Portfolio Equity"), use_container_width=True)
    col2.plotly_chart(drawdown_chart(frame, "Portfolio Drawdown"), use_container_width=True)

    snapshots = services.analytics_service.load_dataframe(artifacts.portfolio_snapshots)
    if not snapshots.empty:
        snapshots.index = pd.to_datetime(snapshots.index, utc=True, errors="coerce")
        exposure_frame = snapshots[["gross_exposure", "net_exposure"]].dropna().tail(200)
        st.subheader("Exposure History")
        st.line_chart(exposure_frame, use_container_width=True)

    st.subheader("Recent Portfolio Snapshots")
    show_latest_rows(snapshots.reset_index(), rows=15)

    live_status = services.execution_service.read_live_status()
    st.subheader("Live Runtime State")
    st.json(live_status.to_dict())
