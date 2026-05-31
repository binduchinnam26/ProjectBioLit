"""KPI metric display components for BioLitAI-X."""

from typing import Any, Dict, Optional
import streamlit as st
import config


def kpi_row(stats: Dict[str, Any], papers_df=None):
    """
    Render a row of 5 KPI metric cards from pipeline statistics.
    """
    import pandas as pd

    n_papers   = stats.get("paper_count", 0)
    n_authors  = stats.get("author_count", 0)
    n_keywords = stats.get("keyword_count", 0)
    n_clusters = 0
    year_range = "—"

    if papers_df is not None and not papers_df.empty and "pub_year" in papers_df.columns:
        years = papers_df["pub_year"].dropna().astype(int)
        if not years.empty:
            year_range = f"{years.min()} – {years.max()}"

    # Try to get cluster count from network stats stored in session
    network_stats = st.session_state.get("coauth_stats", {})
    n_clusters = network_stats.get("num_communities", stats.get("entity_count", 0))

    cols = st.columns(5)
    items = [
        ("Total Papers", f"{n_papers:,}",  config.PRIMARY_ACCENT),
        ("Unique Authors", f"{n_authors:,}", config.SECONDARY_ACCENT),
        ("Keywords Indexed", f"{n_keywords:,}", config.SUCCESS_COLOR),
        ("Research Clusters", f"{n_clusters:,}", config.WARNING_COLOR),
        ("Year Range", year_range, config.TEXT_PRIMARY),
    ]
    for col, (label, value, color) in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="bx-card" style="text-align:center;padding:1rem">
                  <div class="bx-card-title">{label}</div>
                  <div class="bx-card-value" style="color:{color}">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def pipeline_status_indicator(step: int, total: int = 6, message: str = ""):
    """Render an animated pipeline step indicator (steps 1-6)."""
    step_labels = [
        "Fetching Papers",
        "Parsing XML",
        "Cleaning Data",
        "NLP Processing",
        "Building Embeddings",
        "Building Graph",
    ]

    st.markdown(
        f"<div style='margin:1rem 0'>",
        unsafe_allow_html=True,
    )
    for i, label in enumerate(step_labels, start=1):
        if i < step:
            css = "done"
            icon = "✅"
        elif i == step:
            css = "active"
            icon = "⏳"
        else:
            css = "pending"
            icon = "○"
        st.markdown(
            f"<div class='pipeline-step {css}'>{icon} &nbsp; {label}</div>",
            unsafe_allow_html=True,
        )

    if message:
        st.markdown(
            f"<p style='font-size:12px;color:{config.TEXT_SECONDARY};"
            f"margin-top:8px'>{message}</p>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
