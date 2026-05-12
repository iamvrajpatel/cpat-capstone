"""Tests for the Streamlit config service."""

from __future__ import annotations

from datetime import date

from ui.services.config_service import ConfigService, SweepFieldSpec


def test_flatten_and_unflatten_round_trip():
    service = ConfigService()
    nested = {
        "risk": {"max_position_pct": 0.05, "min_cash_pct": 0.03},
        "strategies": {"momentum": {"lookback_long": 252}},
    }
    flat = service.flatten_config(nested)

    assert flat["risk.max_position_pct"] == 0.05
    assert flat["strategies.momentum.lookback_long"] == 252
    assert service.unflatten_config(flat) == nested


def test_build_ui_schema_contains_expected_fields(minimal_config):
    service = ConfigService()
    schema = service.build_ui_schema(minimal_config.model_dump(mode="python"))
    paths = {field.path: field for field in schema}

    assert "risk.max_position_pct" in paths
    assert "strategies.momentum.enabled" in paths
    assert paths["strategies.momentum.enabled"].widget_type == "checkbox"
    assert paths["risk.max_position_pct"].widget_type == "number_input"


def test_generate_parameter_grid_uses_cartesian_product():
    service = ConfigService()
    base = {
        "strategies": {
            "momentum": {
                "lookback_short": 10,
                "lookback_long": 100,
            }
        }
    }
    specs = [
        SweepFieldSpec(path="strategies.momentum.lookback_short", mode="list", values=(10, 20)),
        SweepFieldSpec(path="strategies.momentum.lookback_long", mode="range", start=100, stop=200, step=100),
    ]

    grid = service.generate_parameter_grid(base, specs)

    assert len(grid) == 4
    combos = {
        (item["strategies"]["momentum"]["lookback_short"], item["strategies"]["momentum"]["lookback_long"])
        for item in grid
    }
    assert combos == {(10, 100), (10, 200), (20, 100), (20, 200)}


def test_validate_config_returns_errors_for_invalid_dates(minimal_config):
    service = ConfigService()
    raw = minimal_config.model_dump(mode="python")
    raw["backtest"]["start_date"] = date(2024, 1, 1)
    raw["backtest"]["end_date"] = date(2023, 1, 1)

    result = service.validate_config(raw)

    assert not result.is_valid
    assert result.errors

