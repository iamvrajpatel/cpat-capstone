"""Logs and monitoring page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ui.components.tables.data_tables import show_dataframe
from ui.services import UIServiceBundle


def render(services: UIServiceBundle) -> None:
    """Render searchable logs across core and live subsystems."""
    st.title("Logs & Monitoring")
    log_files = services.execution_service.list_log_files()
    if not log_files:
        st.info("No log files available yet.")
        return

    selected = st.selectbox("Log Source", options=log_files, format_func=lambda path: str(path))
    filters = services.session.get_log_filters()
    filters["search"] = st.text_input("Search", value=str(filters.get("search", "")))
    services.session.set_log_filters(filters)

    if str(selected).endswith(".jsonl"):
        frame = services.analytics_service.load_live_jsonl(selected)
        if filters["search"]:
            search = str(filters["search"]).lower()
            mask = frame.astype(str).apply(lambda col: col.str.lower().str.contains(search, na=False))
            frame = frame[mask.any(axis=1)]
        show_dataframe(frame)
    else:
        text = services.execution_service.read_text_tail(selected, limit=300)
        if filters["search"]:
            lines = [line for line in text.splitlines() if filters["search"].lower() in line.lower()]
            text = "\n".join(lines)
        st.code(text or "No matching log lines.", language="text")

