"""Live-process control widgets."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_live_control_panel(
    status: dict[str, Any],
    *,
    allow_live: bool,
    confirmation_checked: bool,
) -> dict[str, bool]:
    """Render live process controls and return button intents."""
    st.subheader("Execution Controls")
    col1, col2, col3 = st.columns(3)
    start_clicked = col1.button("Start Trading", type="primary", disabled=not allow_live)
    stop_clicked = col2.button("Stop Trading")
    kill_clicked = col3.button("Emergency Kill")

    if not confirmation_checked:
        st.warning("Live controls are disarmed until the confirmation checkbox is enabled.")
    if status.get("state") == "running":
        st.success(f"Process running (PID {status.get('pid')})")
    elif status.get("state") == "stopped":
        st.info("No live process is currently running.")
    else:
        st.warning(f"Process state: {status.get('state', 'unknown')}")

    return {
        "start": start_clicked,
        "stop": stop_clicked,
        "kill": kill_clicked,
    }
