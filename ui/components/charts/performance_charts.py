"""Plotly chart builders for analytics and monitoring pages."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def equity_curve_chart(frame: pd.DataFrame, title: str = "Equity Curve") -> go.Figure:
    """Build an equity curve line chart."""
    figure = go.Figure()
    if not frame.empty and "equity" in frame.columns:
        figure.add_trace(go.Scatter(x=frame.index, y=frame["equity"], mode="lines", name="Equity"))
    figure.update_layout(title=title, template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))
    return figure


def drawdown_chart(frame: pd.DataFrame, title: str = "Drawdown") -> go.Figure:
    """Build a filled drawdown area chart."""
    figure = go.Figure()
    if not frame.empty and "drawdown" in frame.columns:
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["drawdown"],
                mode="lines",
                fill="tozeroy",
                name="Drawdown",
            )
        )
    figure.update_layout(title=title, template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))
    return figure


def rolling_metric_chart(frame: pd.DataFrame, column: str, title: str) -> go.Figure:
    """Build a rolling metric line chart."""
    figure = go.Figure()
    if not frame.empty and column in frame.columns:
        figure.add_trace(go.Scatter(x=frame.index, y=frame[column], mode="lines", name=column))
    figure.update_layout(title=title, template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))
    return figure


def allocation_pie_chart(frame: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    """Build a portfolio-allocation pie chart."""
    if frame.empty or value_col not in frame.columns:
        return go.Figure()
    label_col = "symbol" if "symbol" in frame.columns else frame.columns[0]
    figure = px.pie(frame, names=label_col, values=value_col, title=title)
    figure.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))
    return figure


def optimization_heatmap(frame: pd.DataFrame, x: str, y: str, z: str) -> go.Figure:
    """Build a heatmap for optimization surfaces."""
    if frame.empty or any(column not in frame.columns for column in (x, y, z)):
        return go.Figure()
    pivot = frame.pivot_table(index=y, columns=x, values=z, aggfunc="mean")
    figure = px.imshow(
        pivot,
        labels={"x": x, "y": y, "color": z},
        aspect="auto",
        title=f"{z} Heatmap",
    )
    figure.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20))
    return figure

