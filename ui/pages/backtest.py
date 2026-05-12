"""Backtesting and walk-forward controls."""

from __future__ import annotations

import streamlit as st

from ui.components.charts.performance_charts import drawdown_chart, equity_curve_chart
from ui.components.metrics.kpi_cards import show_kpis
from ui.components.tables.data_tables import show_dataframe
from ui.services import UIServiceBundle
from ui.services.config_service import UIFieldSpec


def _coerce_numeric_widget_args(field: UIFieldSpec, current_value: object) -> dict[str, object]:
    """Normalize numeric widget arguments so Streamlit sees one numeric type."""
    numeric_type = float if field.python_type == "float" else int
    value = numeric_type(current_value)
    payload: dict[str, object] = {"value": value}
    if field.minimum is not None:
        payload["min_value"] = numeric_type(field.minimum)
    if field.maximum is not None:
        payload["max_value"] = numeric_type(field.maximum)
    if field.step is not None:
        payload["step"] = numeric_type(field.step)
    return payload


def render(services: UIServiceBundle) -> None:
    """Render the backtest orchestration page."""
    st.title("Backtesting")
    raw_config = services.session.get_config_draft() or services.config_service.load_raw_config()
    base_validation = services.config_service.validate_config(raw_config)
    schema = services.config_service.build_ui_schema(raw_config)
    descriptors = services.strategy_service.list_strategies(
        base_validation.config if base_validation.is_valid else services.config_service.load_config(services.config_path)
    )
    strategy_keys = [descriptor.key for descriptor in descriptors]
    strategy = st.selectbox(
        "Strategy",
        options=strategy_keys,
        format_func=lambda key: next(item.label for item in descriptors if item.key == key),
    )
    universe = raw_config.get("universe", {}).get("equities", [])
    selected_universe = st.multiselect(
        "Active Equity Universe",
        options=universe,
        default=universe,
        help="Temporary research subset for this run.",
    )

    editable_prefixes = {
        "backtest",
        "risk",
        "position_sizing",
        "strategies.momentum" if strategy in {"momentum", "both"} else "strategies.mean_reversion",
    }
    filtered_fields = [
        field for field in schema
        if any(field.path.startswith(prefix) for prefix in editable_prefixes)
    ]
    overrides: dict[str, object] = {}
    with st.expander("Run-Time Overrides", expanded=False):
        for field in filtered_fields:
            current_value = services.config_service.flatten_config(raw_config).get(field.path, field.value)
            overrides[field.path] = _render_compact_field(services, field, current_value)

    updated_raw = services.config_service.apply_overrides(raw_config, overrides)
    updated_raw.setdefault("universe", {})["equities"] = selected_universe
    validation = services.config_service.validate_config(updated_raw)
    if not validation.is_valid:
        st.error("Current overrides are invalid.")
        for error in validation.errors:
            st.write(f"- {error}")
        return

    col1, col2, col3 = st.columns(3)
    run_backtest = col1.button("Run Backtest", type="primary")
    run_compare = col2.button("Run Compare")
    run_walk_forward = col3.button("Run Walk-Forward")

    payload: dict[str, object] | None = None
    if run_backtest:
        payload = services.execution_service.run_backtest(validation.config, strategy)
    elif run_compare:
        payload = services.execution_service.run_compare(validation.config, strategy)
    elif run_walk_forward:
        payload = services.execution_service.run_walk_forward(validation.config, strategy)

    if not payload:
        return

    if "comparison_df" in payload:
        st.subheader("Baseline vs Managed")
        show_dataframe(payload["comparison_df"])
        return

    if "summary_df" in payload:
        st.subheader("Per-Fold Walk-Forward Summary")
        show_dataframe(payload["summary_df"])
        report = payload["oos_report"]
        show_kpis(
            {
                "OOS Total Return": f"{report.get('total_return_pct', 0.0):.2f}%",
                "OOS Sharpe": f"{report.get('sharpe_ratio', 0.0):.3f}",
                "OOS Max DD": f"{report.get('max_drawdown_pct', 0.0):.2f}%",
                "Trades": report.get("n_trades", 0),
            }
        )
        frame = services.analytics_service.equity_drawdown_frame(payload["oos_equity"])
        st.plotly_chart(equity_curve_chart(frame, "Combined OOS Equity"), use_container_width=True)
        return

    report = payload["report"]
    show_kpis(
        {
            "Total Return": f"{report['total_return_pct']:.2f}%",
            "Sharpe": f"{report['sharpe_ratio']:.3f}",
            "Max Drawdown": f"{report['max_drawdown_pct']:.2f}%",
            "Win Rate": f"{report['win_rate_pct']:.1f}%",
            "Trades": report["n_trades"],
            "Final Equity": f"{report['final_equity']:,.0f}",
        },
        columns=3,
    )
    frame = services.analytics_service.equity_drawdown_frame(payload["equity_curve"])
    col1, col2 = st.columns(2)
    col1.plotly_chart(equity_curve_chart(frame, "Equity Curve"), use_container_width=True)
    col2.plotly_chart(drawdown_chart(frame, "Drawdown"), use_container_width=True)
    st.subheader("Trade Log")
    show_dataframe(payload["trade_log_df"])


def _render_compact_field(services: UIServiceBundle, field: UIFieldSpec, current_value: object) -> object:
    """Render a compact field editor for run-time overrides."""
    if field.widget_type == "checkbox":
        return st.checkbox(field.label, value=bool(current_value), key=f"bt.{field.path}")
    if field.widget_type == "selectbox":
        options = list(field.options)
        index = options.index(current_value) if current_value in options else 0
        return st.selectbox(field.label, options=options, index=index, key=f"bt.{field.path}")
    if field.widget_type == "number_input":
        kwargs: dict[str, object] = {"label": field.label, "key": f"bt.{field.path}"}
        kwargs.update(_coerce_numeric_widget_args(field, current_value))
        return st.number_input(**kwargs)
    return st.text_input(field.label, value=str(current_value), key=f"bt.{field.path}")
