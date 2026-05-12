"""Risk, sizing, and portfolio state page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.components.charts.performance_charts import allocation_pie_chart
from ui.components.tables.data_tables import show_dataframe
from ui.services import UIServiceBundle


def render(services: UIServiceBundle) -> None:
    """Render portfolio controls and risk telemetry."""
    st.title("Portfolio / Risk")
    raw_config = services.session.get_config_draft() or services.config_service.load_raw_config()
    validation = services.config_service.validate_config(raw_config)
    if validation.is_valid:
        st.success("Configuration validates cleanly.")
    else:
        st.error("Configuration has validation issues.")
        for error in validation.errors:
            st.write(f"- {error}")

    risk_cfg = raw_config.get("risk", {})
    sizing_cfg = raw_config.get("position_sizing", {})
    col1, col2 = st.columns(2)
    col1.json(risk_cfg)
    col2.json(sizing_cfg)

    strategy = services.session.get_selected_result()
    artifacts = services.analytics_service.discover_artifacts(strategy)
    snapshots = services.analytics_service.load_dataframe(artifacts.portfolio_snapshots)
    risk_history = services.analytics_service.load_dataframe(artifacts.risk_history)
    trade_risk = services.analytics_service.load_dataframe(artifacts.trade_risk)

    if not snapshots.empty:
        snapshots.index = pd.to_datetime(snapshots.index, utc=True, errors="coerce")
        latest = snapshots.tail(1).reset_index()
        st.subheader("Latest Exposure Snapshot")
        show_dataframe(latest)
        if "cash" in snapshots.columns and "equity" in snapshots.columns:
            alloc = pd.DataFrame(
                {"bucket": ["Cash", "Invested"], "value": [float(snapshots["cash"].iloc[-1]), float(max(snapshots["equity"].iloc[-1] - snapshots["cash"].iloc[-1], 0.0))]}
            )
            st.plotly_chart(allocation_pie_chart(alloc, value_col="value", title="Capital Allocation"), use_container_width=True)

    st.subheader("Risk History")
    show_dataframe(risk_history.reset_index())
    st.subheader("Trade Risk Ledger")
    show_dataframe(trade_risk.reset_index())

