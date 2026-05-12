"""Centralised Streamlit session-state accessors.

The rest of the UI should treat this class as the only write path to
``st.session_state`` so rerun behaviour remains predictable.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import streamlit as st


class SessionManager:
    """Wrapper around ``st.session_state`` with typed convenience methods."""

    _DEFAULTS: dict[str, Any] = {
        "ui.active_page": "Dashboard",
        "ui.config_draft": {},
        "ui.sweep_specs": {},
        "ui.selected_result": "momentum",
        "ui.log_filters": {
            "level": "ALL",
            "event_type": "ALL",
            "symbol": "",
            "search": "",
        },
        "ui.live_confirmed": False,
        "ui.live_process_mode": "paper",
    }

    def __init__(self, state: MutableMapping[str, Any] | None = None) -> None:
        self._state = state if state is not None else st.session_state
        for key, value in self._DEFAULTS.items():
            self._state.setdefault(key, value.copy() if isinstance(value, dict) else value)

    def get_active_page(self) -> str:
        """Return the current page name."""
        return str(self._state["ui.active_page"])

    def set_active_page(self, page: str) -> None:
        """Persist the current page name."""
        self._state["ui.active_page"] = page

    def get_config_draft(self) -> dict[str, Any]:
        """Return the in-memory editable config draft."""
        return dict(self._state["ui.config_draft"])

    def set_config_draft(self, draft: dict[str, Any]) -> None:
        """Persist a config draft."""
        self._state["ui.config_draft"] = draft

    def clear_config_draft(self) -> None:
        """Reset the config draft to an empty override map."""
        self._state["ui.config_draft"] = {}

    def get_sweep_specs(self) -> dict[str, dict[str, Any]]:
        """Return parameter-sweep overlays keyed by dotted field path."""
        return dict(self._state["ui.sweep_specs"])

    def set_sweep_specs(self, specs: dict[str, dict[str, Any]]) -> None:
        """Persist sweep overlay definitions."""
        self._state["ui.sweep_specs"] = specs

    def get_selected_result(self) -> str:
        """Return the currently selected strategy/result namespace."""
        return str(self._state["ui.selected_result"])

    def set_selected_result(self, name: str) -> None:
        """Persist the selected result namespace."""
        self._state["ui.selected_result"] = name

    def get_log_filters(self) -> dict[str, Any]:
        """Return log viewer filters."""
        return dict(self._state["ui.log_filters"])

    def set_log_filters(self, filters: dict[str, Any]) -> None:
        """Persist log viewer filters."""
        self._state["ui.log_filters"] = filters

    def is_live_confirmed(self) -> bool:
        """Return whether the operator has armed live execution controls."""
        return bool(self._state["ui.live_confirmed"])

    def set_live_confirmed(self, confirmed: bool) -> None:
        """Persist live-mode confirmation state."""
        self._state["ui.live_confirmed"] = confirmed

    def get_live_process_mode(self) -> str:
        """Return the last-selected live process mode."""
        return str(self._state["ui.live_process_mode"])

    def set_live_process_mode(self, mode: str) -> None:
        """Persist the live process mode."""
        self._state["ui.live_process_mode"] = mode

