"""Dynamic config form rendering helpers."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ui.services.config_service import ConfigService, UIFieldSpec


def _coerce_numeric_widget_args(field: UIFieldSpec, current_value: Any) -> dict[str, Any]:
    """Normalize numeric widget arguments so Streamlit sees one numeric type."""
    numeric_type = float if field.python_type == "float" else int
    value = numeric_type(current_value)
    payload: dict[str, Any] = {"value": value}
    if field.minimum is not None:
        payload["min_value"] = numeric_type(field.minimum)
    if field.maximum is not None:
        payload["max_value"] = numeric_type(field.maximum)
    if field.step is not None:
        payload["step"] = numeric_type(field.step)
    return payload


def render_field(
    field: UIFieldSpec,
    current_value: Any,
    config_service: ConfigService,
    key_prefix: str = "cfg",
) -> Any:
    """Render one config field and return the edited value."""
    widget_key = f"{key_prefix}.{field.path}"
    help_text = field.description or field.path
    if field.widget_type == "checkbox":
        return st.checkbox(field.label, value=bool(current_value), help=help_text, key=widget_key)
    if field.widget_type == "selectbox":
        options = list(field.options)
        index = options.index(current_value) if current_value in options else 0
        return st.selectbox(field.label, options=options, index=index, help=help_text, key=widget_key)
    if field.widget_type == "number_input":
        kwargs: dict[str, Any] = {
            "label": field.label,
            "help": help_text,
            "key": widget_key,
        }
        kwargs.update(_coerce_numeric_widget_args(field, current_value))
        return st.number_input(**kwargs)
    if field.widget_type == "list_editor":
        initial = "\n".join(str(item) for item in (current_value or []))
        raw_value = st.text_area(field.label, value=initial, help=help_text, key=widget_key, height=120)
        return config_service.parse_list_value(raw_value, field.python_type)
    return st.text_input(field.label, value="" if current_value is None else str(current_value), help=help_text, key=widget_key)


def render_section_form(
    section_name: str,
    fields: list[UIFieldSpec],
    raw_config: dict[str, Any],
    config_service: ConfigService,
    key_prefix: str = "cfg",
) -> dict[str, Any]:
    """Render a section of config fields and return flat overrides for that section."""
    flat = config_service.flatten_config(raw_config)
    updates: dict[str, Any] = {}
    for field in fields:
        current_value = flat.get(field.path, field.value)
        updates[field.path] = render_field(field, current_value, config_service, key_prefix=key_prefix)
    return updates
