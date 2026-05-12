"""Dynamic YAML settings editor page."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import streamlit as st
import yaml

from ui.components.forms.config_forms import render_section_form
from ui.services import UIServiceBundle


def render(services: UIServiceBundle) -> None:
    """Render the full dynamic config editor."""
    st.title("Settings")
    raw_config = services.session.get_config_draft() or services.config_service.load_raw_config()
    schema = services.config_service.build_ui_schema(raw_config)
    grouped: dict[str, list] = defaultdict(list)
    for field in schema:
        grouped[field.section].append(field)

    updated_raw = raw_config
    with st.form("settings-editor"):
        collected: dict[str, Any] = {}
        for section, fields in grouped.items():
            with st.expander(section.replace("_", " ").title(), expanded=False):
                collected.update(render_section_form(section, fields, updated_raw, services.config_service, key_prefix=f"settings.{section}"))
        submitted = st.form_submit_button("Apply Draft Changes", type="primary")
        if submitted:
            updated_raw = services.config_service.apply_overrides(raw_config, collected)
            services.session.set_config_draft(updated_raw)
            st.success("Draft configuration updated in session.")

    current_draft = services.session.get_config_draft() or updated_raw
    validation = services.config_service.validate_config(current_draft)
    if validation.is_valid:
        st.success("Draft configuration is valid.")
    else:
        st.error("Draft configuration failed validation.")
        for error in validation.errors:
            st.write(f"- {error}")

    col1, col2, col3 = st.columns(3)
    if col1.button("Save to settings.yaml", disabled=not validation.is_valid):
        services.config_service.save_config(current_draft, services.config_path)
        st.success(f"Saved {services.config_path}")
    if col2.button("Reset Draft"):
        services.session.set_config_draft(services.config_service.load_raw_config())
        st.info("Draft reset from disk.")
    col3.download_button(
        "Export Draft",
        data=services.config_service.export_preset(current_draft),
        file_name="cpat_ui_preset.yaml",
        mime="application/x-yaml",
    )

    uploaded = st.file_uploader("Import YAML Preset", type=["yaml", "yml"])
    if uploaded is not None:
        imported = yaml.safe_load(uploaded.getvalue().decode("utf-8"))
        services.session.set_config_draft(imported)
        st.success("Imported preset into session draft.")

    st.subheader("Current Draft Preview")
    st.code(services.config_service.export_preset(current_draft), language="yaml")

