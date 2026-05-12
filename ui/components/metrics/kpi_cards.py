"""KPI card rendering helpers."""

from __future__ import annotations

from typing import Any

import streamlit as st


def show_kpis(metrics: dict[str, Any], columns: int = 4) -> None:
    """Render a metric dictionary as a responsive KPI grid."""
    items = list(metrics.items())
    if not items:
        st.info("No KPI data available.")
        return
    for start in range(0, len(items), columns):
        row = items[start:start + columns]
        cols = st.columns(len(row))
        for col, (label, value) in zip(cols, row, strict=False):
            col.metric(label=label, value=value)

