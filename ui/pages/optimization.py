"""Parameter-sweep and optimization page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.components.charts.performance_charts import optimization_heatmap
from ui.components.metrics.kpi_cards import show_kpis
from ui.components.tables.data_tables import show_dataframe
from ui.services import UIServiceBundle
from ui.services.config_service import SweepFieldSpec


def render(services: UIServiceBundle) -> None:
    """Render the optimization workbench."""
    st.title("Optimization")
    raw_config = services.session.get_config_draft() or services.config_service.load_raw_config()
    validation = services.config_service.validate_config(raw_config)
    if not validation.is_valid:
        st.error("Fix configuration issues before running optimization.")
        for error in validation.errors:
            st.write(f"- {error}")
        return

    descriptors = [
        descriptor
        for descriptor in services.strategy_service.list_strategies(validation.config)
        if descriptor.key != "both"
    ]
    strategy = st.selectbox(
        "Strategy Namespace",
        options=[descriptor.key for descriptor in descriptors],
        format_func=lambda key: next(item.label for item in descriptors if item.key == key),
    )
    strategy_prefix = f"strategies.{strategy}."
    schema = [
        field
        for field in services.config_service.build_ui_schema(raw_config)
        if field.path.startswith(strategy_prefix) and field.python_type in {"int", "float"}
    ]
    selected_paths = st.multiselect(
        "Sweep Parameters",
        options=[field.path for field in schema],
        format_func=lambda path: path.split(".")[-1].replace("_", " ").title(),
    )

    base_flat = services.config_service.flatten_config(raw_config)
    stored_specs = services.session.get_sweep_specs()
    specs: dict[str, dict[str, object]] = dict(stored_specs)
    for path in selected_paths:
        current = specs.get(path, {"mode": "fixed"})
        with st.expander(path, expanded=False):
            mode = st.selectbox(
                "Mode",
                options=["fixed", "list", "range", "scenario"],
                index=["fixed", "list", "range", "scenario"].index(str(current.get("mode", "fixed"))),
                key=f"opt.mode.{path}",
            )
            base_value = base_flat[path]
            spec_payload: dict[str, object] = {"mode": mode}
            if mode in {"list", "scenario"}:
                default_value = ", ".join(str(item) for item in current.get("values", [base_value]))
                raw_value = st.text_input("Values", value=default_value, key=f"opt.values.{path}")
                spec_payload["values"] = services.config_service.parse_list_value(raw_value, next(field.python_type for field in schema if field.path == path))
            elif mode == "range":
                spec_payload["start"] = st.number_input("Start", value=float(current.get("start", base_value)), key=f"opt.start.{path}")
                spec_payload["stop"] = st.number_input("Stop", value=float(current.get("stop", base_value)), key=f"opt.stop.{path}")
                spec_payload["step"] = st.number_input("Step", value=float(current.get("step", 1.0)), min_value=0.0001, key=f"opt.step.{path}")
            specs[path] = spec_payload

    services.session.set_sweep_specs(specs)
    active_specs = [
        SweepFieldSpec(
            path=path,
            mode=str(spec.get("mode", "fixed")),
            values=tuple(spec.get("values", [])),
            start=spec.get("start"),
            stop=spec.get("stop"),
            step=spec.get("step"),
        )
        for path, spec in specs.items()
        if path in selected_paths
    ]
    combinations = services.config_service.generate_parameter_grid(raw_config, active_specs)
    st.info(f"Generated {len(combinations)} parameter combinations from the current overlay set.")

    if st.button("Run Optimization", type="primary"):
        optimizer_grid = {
            spec.path.split(".")[-1]: spec.resolved_values(base_flat.get(spec.path))
            for spec in active_specs
        } or None
        payload = services.execution_service.run_optimization(validation.config, strategy, param_grid=optimizer_grid)

        results_df = payload["results_df"]
        if results_df.empty:
            st.warning("No optimization results were produced.")
            return

        best_params = payload["best_params"]
        show_kpis({f"Best {key}": value for key, value in best_params.items()}, columns=3)
        st.subheader("Optimization Results")
        show_dataframe(results_df)
        st.subheader("Robustness / Sensitivity")
        show_dataframe(payload["stability_df"])

        if len(optimizer_grid or {}) >= 2:
            x, y = list((optimizer_grid or {}).keys())[:2]
            st.plotly_chart(optimization_heatmap(results_df, x=x, y=y, z="sharpe_ratio"), use_container_width=True)

        if payload["degradation"]:
            st.subheader("Held-Out Test Evaluation")
            st.json(payload["degradation"])
