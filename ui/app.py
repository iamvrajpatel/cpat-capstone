"""Entry point for the CPAT Streamlit control console."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ui.pages import analytics, backtest, dashboard, live_trading, logs, optimization, portfolio, settings
from ui.services import UIServiceBundle
from ui.services.analytics_service import AnalyticsService
from ui.services.config_service import ConfigService
from ui.services.execution_service import ExecutionService
from ui.services.strategy_service import StrategyService
from ui.state.session_manager import SessionManager


PAGE_REGISTRY = {
    "Dashboard": dashboard.render,
    "Backtesting": backtest.render,
    "Optimization": optimization.render,
    "Portfolio / Risk": portfolio.render,
    "Analytics": analytics.render,
    "Live Trading": live_trading.render,
    "Logs": logs.render,
    "Settings": settings.render,
}


@st.cache_resource
def build_backend_services(config_path: str = "config/settings.yaml") -> tuple[Path, ConfigService, ExecutionService, AnalyticsService, StrategyService]:
    """Construct long-lived backend-facing services once per Streamlit process."""
    cfg_path = Path(config_path)
    config_service = ConfigService(cfg_path)
    execution_service = ExecutionService(cfg_path)
    analytics_service = AnalyticsService()
    strategy_service = StrategyService()
    return cfg_path, config_service, execution_service, analytics_service, strategy_service


def main() -> None:
    """Render the CPAT multipage control console."""
    st.set_page_config(
        page_title="CPAT Control Console",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_theme()
    cfg_path, config_service, execution_service, analytics_service, strategy_service = build_backend_services()
    services = UIServiceBundle(
        config_path=cfg_path,
        session=SessionManager(),
        config_service=config_service,
        execution_service=execution_service,
        analytics_service=analytics_service,
        strategy_service=strategy_service,
    )

    if not services.session.get_config_draft():
        services.session.set_config_draft(services.config_service.load_raw_config())

    with st.sidebar:
        st.markdown("## CPAT")
        st.caption("Quant research terminal + trading operations console")
        page = st.radio("Navigation", options=list(PAGE_REGISTRY.keys()), index=list(PAGE_REGISTRY.keys()).index(services.session.get_active_page()) if services.session.get_active_page() in PAGE_REGISTRY else 0)
        services.session.set_active_page(page)
        st.divider()
        st.caption(f"Config: `{services.config_path}`")
        st.caption(f"Results: `{len(services.execution_service.list_result_files())}` CSV artifacts")

    PAGE_REGISTRY[page](services)


def _inject_theme() -> None:
    """Apply a denser, operations-oriented visual style."""
    st.markdown(
        """
        <style>
        :root {
            --cpat-ink: #112031;
            --cpat-muted: #566579;
            --cpat-panel: rgba(255, 251, 245, 0.90);
            --cpat-panel-strong: #fffdf8;
            --cpat-border: rgba(17, 32, 49, 0.12);
            --cpat-accent: #0b6e69;
            --cpat-accent-strong: #084c61;
            --cpat-warm: #c86b3c;
            --cpat-danger: #a63446;
            --cpat-sidebar-top: #13293d;
            --cpat-sidebar-bottom: #1b3a4b;
            --cpat-grid: rgba(11, 110, 105, 0.06);
        }
        .stApp {
            background:
                linear-gradient(90deg, transparent 0 47px, var(--cpat-grid) 47px 48px),
                linear-gradient(transparent 0 47px, var(--cpat-grid) 47px 48px),
                radial-gradient(circle at top left, rgba(200, 107, 60, 0.10), transparent 24%),
                radial-gradient(circle at top right, rgba(11, 110, 105, 0.10), transparent 22%),
                linear-gradient(180deg, #f8f4ec 0%, #eef3f2 100%);
            background-size: 48px 48px, 48px 48px, auto, auto, auto;
            color: var(--cpat-ink);
        }
        .stApp,
        .stApp p,
        .stApp li,
        .stApp label,
        .stApp span,
        .stApp div,
        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp h5,
        .stApp h6 {
            color: var(--cpat-ink);
        }
        .stApp a {
            color: var(--cpat-accent-strong);
        }
        [data-testid="stHeader"] {
            background: rgba(248, 244, 236, 0.85);
            backdrop-filter: blur(10px);
        }
        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top, rgba(200, 107, 60, 0.18), transparent 26%),
                linear-gradient(180deg, var(--cpat-sidebar-top) 0%, var(--cpat-sidebar-bottom) 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        [data-testid="stSidebar"] * {
            color: #f7f4ef !important;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: rgba(247, 244, 239, 0.76) !important;
        }
        [data-testid="stSidebar"] .stRadio label,
        [data-testid="stSidebar"] .stSelectbox label {
            color: #f7f4ef !important;
        }
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2rem;
        }
        [data-testid="stMetric"],
        .stExpander,
        .stAlert,
        [data-testid="stForm"],
        [data-testid="stVerticalBlockBorderWrapper"] > div:has(> [data-testid="stDataFrame"]),
        [data-testid="stDataFrame"],
        .stCodeBlock,
        [data-testid="stCode"],
        [data-testid="stJson"],
        .stTabs [data-baseweb="tab-panel"] {
            background: var(--cpat-panel);
            border: 1px solid var(--cpat-border);
            border-radius: 18px;
            box-shadow: 0 10px 24px rgba(17, 32, 49, 0.05);
        }
        [data-testid="stMetric"] {
            padding: 0.8rem 1rem;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.72) 0%, rgba(255,255,255,0.92) 100%);
        }
        [data-testid="stMetricLabel"] {
            color: var(--cpat-muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.45rem;
            font-weight: 700;
            color: var(--cpat-ink);
        }
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 999px;
            border: 1px solid var(--cpat-border);
            color: var(--cpat-ink);
            background: rgba(255,255,255,0.72);
            transition: all 0.18s ease;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--cpat-accent);
            color: var(--cpat-accent-strong);
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(11, 110, 105, 0.12);
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--cpat-accent) 0%, var(--cpat-accent-strong) 100%);
            border: 1px solid var(--cpat-accent-strong);
            color: #f8fbfd;
        }
        .stButton > button[kind="primary"]:hover {
            color: #f8fbfd;
            box-shadow: 0 12px 24px rgba(8, 76, 97, 0.18);
        }
        .stCheckbox label,
        .stSelectbox label,
        .stMultiSelect label,
        .stTextInput label,
        .stNumberInput label,
        .stTextArea label {
            color: var(--cpat-ink) !important;
            font-weight: 600;
        }
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea,
        [data-baseweb="select"] > div,
        [data-baseweb="base-input"] {
            background: rgba(255,255,255,0.86) !important;
            color: var(--cpat-ink) !important;
            border-radius: 14px !important;
            border-color: rgba(17, 32, 49, 0.16) !important;
        }
        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {
            color: #7a8898 !important;
        }
        .stCodeBlock code,
        .stCodeBlock pre,
        [data-testid="stCode"] code,
        [data-testid="stCode"] pre,
        [data-testid="stJson"] pre,
        [data-testid="stJson"] code,
        [data-testid="stJson"] span,
        [data-testid="stJson"] div,
        [data-testid="stJson"] p {
            color: var(--cpat-ink) !important;
            background: transparent !important;
            text-shadow: none !important;
        }
        [data-testid="stCode"] > div,
        [data-testid="stCodeBlock"] > div,
        [data-testid="stJson"] > div {
            background: var(--cpat-panel-strong) !important;
            border-radius: 16px !important;
        }
        [data-testid="stCode"] {
            background: var(--cpat-panel-strong) !important;
        }
        [data-testid="stCode"] button,
        [data-testid="stCodeBlock"] button,
        [data-testid="stJson"] button {
            background: rgba(255,255,255,0.78) !important;
            color: var(--cpat-ink) !important;
            border: 1px solid rgba(17, 32, 49, 0.12) !important;
        }
        [data-testid="stCode"] button svg,
        [data-testid="stCodeBlock"] button svg,
        [data-testid="stJson"] button svg {
            fill: var(--cpat-ink) !important;
        }
        textarea[readonly] {
            background: var(--cpat-panel-strong) !important;
            color: var(--cpat-ink) !important;
            -webkit-text-fill-color: var(--cpat-ink) !important;
            border: 1px solid rgba(17, 32, 49, 0.16) !important;
        }
        textarea[readonly]::selection,
        [data-testid="stCode"] pre::selection,
        [data-testid="stJson"] pre::selection {
            background: rgba(11, 110, 105, 0.18) !important;
        }
        [data-testid="stDataFrame"] * {
            color: var(--cpat-ink) !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            background: rgba(255,255,255,0.62);
            border: 1px solid rgba(17, 32, 49, 0.08);
            color: var(--cpat-muted);
            padding-inline: 1rem;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(11, 110, 105, 0.10) !important;
            color: var(--cpat-accent-strong) !important;
            border-color: rgba(11, 110, 105, 0.30) !important;
        }
        .stAlert [data-testid="stMarkdownContainer"] p {
            color: inherit !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
