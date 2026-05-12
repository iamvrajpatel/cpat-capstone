"""DataFrame presentation helpers for Streamlit."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def show_dataframe(frame: pd.DataFrame, *, use_container_width: bool = True, height: int = 360) -> None:
    """Render a DataFrame with a consistent table style."""
    if frame.empty:
        st.info("No data available for this view yet.")
        return
    st.dataframe(frame, use_container_width=use_container_width, height=height)


def show_latest_rows(frame: pd.DataFrame, rows: int = 20) -> None:
    """Render the most recent rows of a table-like DataFrame."""
    show_dataframe(frame.tail(rows))

