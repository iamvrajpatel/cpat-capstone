"""Tests for the Streamlit session manager abstraction."""

from __future__ import annotations

from ui.state.session_manager import SessionManager


def test_session_manager_persists_values_in_mapping():
    state: dict[str, object] = {}
    session = SessionManager(state=state)

    session.set_active_page("Optimization")
    session.set_selected_result("momentum")
    session.set_live_confirmed(True)
    session.set_sweep_specs({"strategies.momentum.lookback_long": {"mode": "list", "values": [126, 252]}})

    assert session.get_active_page() == "Optimization"
    assert session.get_selected_result() == "momentum"
    assert session.is_live_confirmed() is True
    assert "strategies.momentum.lookback_long" in session.get_sweep_specs()
