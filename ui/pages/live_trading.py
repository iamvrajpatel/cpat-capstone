"""Managed live/paper trading controls."""

from __future__ import annotations

import streamlit as st

from ui.components.controls.live_controls import render_live_control_panel
from ui.components.tables.data_tables import show_dataframe
from ui.services import UIServiceBundle


def render(services: UIServiceBundle) -> None:
    """Render paper/live trading controls and runtime telemetry."""
    st.title("Live / Paper Trading")
    raw_config = services.session.get_config_draft() or services.config_service.load_raw_config()
    live_cfg = raw_config.get("live", {})
    cfg = services.config_service.load_config(services.config_path)
    live_strategies = [
        descriptor
        for descriptor in services.strategy_service.list_strategies(cfg)
        if descriptor.supports_live and descriptor.enabled
    ]
    mode = st.selectbox("Broker Mode", options=["paper", "dhan"], index=0 if live_cfg.get("broker", "paper") == "paper" else 1)
    selected_strategy = st.selectbox(
        "Strategy",
        options=[descriptor.key for descriptor in live_strategies] or ["momentum"],
        format_func=lambda key: next((item.label for item in live_strategies if item.key == key), key),
    )
    services.session.set_live_process_mode(mode)
    demo_mode = st.checkbox("Demo Mode", value=True, help="Runs a bounded paper session rather than unattended trading.")
    dry_run = st.checkbox("Dry Run", value=bool(live_cfg.get("dry_run", False)))
    confirmation = st.checkbox(
        "I understand live trading can place real orders.",
        value=services.session.is_live_confirmed(),
    )
    services.session.set_live_confirmed(confirmation)

    missing = services.execution_service.missing_credentials(mode)
    if missing:
        st.warning(f"Missing credentials for {mode}: {', '.join(missing)}")

    status = services.execution_service.read_live_status().to_dict()
    intents = render_live_control_panel(
        status,
        allow_live=(mode == "paper" or (not missing and confirmation)),
        confirmation_checked=confirmation,
    )

    symbols = raw_config.get("universe", {}).get("equities", [])[:5]
    if intents["start"]:
        services.execution_service.start_live_process(
            mode=mode,
            strategy=selected_strategy,
            symbols=symbols,
            config_path=services.config_path,
            dry_run=dry_run,
            demo=demo_mode,
        )
        st.success("Live process started.")
    if intents["stop"]:
        services.execution_service.stop_live_process()
        st.info("Stop signal sent.")
    if intents["kill"]:
        services.execution_service.kill_live_process()
        st.error("Emergency kill issued.")

    st.subheader("Runtime Status")
    st.json(services.execution_service.read_live_status().to_dict())

    oms_path = raw_config.get("live", {}).get("oms_audit_log", "logs/live/oms_audit.jsonl")
    oms_df = services.analytics_service.load_live_jsonl(oms_path)
    st.subheader("OMS Audit Trail")
    show_dataframe(oms_df)

    process_log = services.execution_service.read_text_tail("logs/live/ui_process.log", limit=120)
    st.subheader("Managed Process Log Tail")
    st.code(process_log or "No live process log available.", language="text")
