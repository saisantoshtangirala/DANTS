"""
Streamlit dashboard for Astra-Trade QML.

Renders the metrics listed in config.yaml's `dashboard.metrics`:
pnl_chart, confidence_distribution, regime_timeline,
quantum_circuit_metrics, trade_journal, model_performance.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils.config import load_config
from src.utils.database import DatabaseManager

st.set_page_config(page_title="Astra-Trade QML", layout="wide")


@st.cache_resource
def get_db(database_url: str) -> DatabaseManager:
    return DatabaseManager(database_url)


def render_pnl_chart(db: DatabaseManager) -> None:
    st.subheader("P&L Curve")
    trades = db.get_trades(status="CLOSED")
    if trades.empty:
        st.info("No closed trades yet.")
        return
    trades = trades.sort_values("timestamp")
    trades["cumulative_pnl"] = trades["pnl"].cumsum()
    fig = px.line(trades, x="timestamp", y="cumulative_pnl", title="Cumulative P&L")
    st.plotly_chart(fig, use_container_width=True)


def render_confidence_distribution(db: DatabaseManager) -> None:
    st.subheader("Signal Confidence Distribution")
    trades = db.get_trades()
    if trades.empty:
        st.info("No trades logged yet.")
        return
    fig = px.histogram(trades, x="confidence", nbins=20, title="Confidence Distribution")
    st.plotly_chart(fig, use_container_width=True)


def render_regime_timeline(db: DatabaseManager) -> None:
    st.subheader("Regime Timeline")
    trades = db.get_trades()
    if trades.empty:
        st.info("No trades logged yet.")
        return
    fig = px.scatter(trades, x="timestamp", y="regime", color="action", title="Regime at Trade Time")
    st.plotly_chart(fig, use_container_width=True)


def render_quantum_circuit_metrics(db: DatabaseManager) -> None:
    st.subheader("Quantum Circuit Metrics")
    query = (
        "SELECT * FROM model_metrics WHERE quantum_circuit_depth > 0 "
        "ORDER BY timestamp DESC LIMIT 50"
    )
    df = pd.read_sql_query(query, db.engine)
    if df.empty:
        st.info("No quantum model metrics logged yet.")
        return
    fig = px.line(
        df, x="timestamp", y="quantum_circuit_depth", color="model_type",
        title="Quantum Circuit Depth Over Time",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df)


def render_trade_journal(db: DatabaseManager) -> None:
    st.subheader("Trade Journal")
    st.dataframe(db.get_trades())


def render_model_performance(db: DatabaseManager) -> None:
    st.subheader("Model Performance Summary (last 30 days)")
    summary = db.get_performance_summary(days=30)
    cols = st.columns(len(summary))
    for col, (key, value) in zip(cols, summary.items()):
        label = key.replace("_", " ").title()
        col.metric(label, f"{value:.4f}" if isinstance(value, float) else value)


RENDERERS = {
    "pnl_chart": render_pnl_chart,
    "confidence_distribution": render_confidence_distribution,
    "regime_timeline": render_regime_timeline,
    "quantum_circuit_metrics": render_quantum_circuit_metrics,
    "trade_journal": render_trade_journal,
    "model_performance": render_model_performance,
}


def main() -> None:
    config = load_config()
    dashboard_cfg = config.get("dashboard", {})
    db = get_db(config.get("logging", {}).get("database", "sqlite:///logs/astra_trade.db"))

    st.title("Astra-Trade QML Dashboard")
    project = config.get("project", {}).get("name", "Astra-Trade")
    mode = config.get("trading", {}).get("mode", "paper").upper()
    st.caption(f"{project} — {mode} mode")

    render_model_performance(db)

    for metric_name in dashboard_cfg.get("metrics", []):
        renderer = RENDERERS.get(metric_name)
        if renderer and renderer is not render_model_performance:
            renderer(db)


if __name__ == "__main__":
    main()
