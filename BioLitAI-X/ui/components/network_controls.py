"""Reusable network visualization control components for the Bibliometric Network Explorer."""

import networkx as nx
import streamlit as st

import config


def viz_mode_selector(key: str) -> str:
    """
    Render the three visualization-mode radio buttons (Network / Overlay / Density).
    Returns one of 'network', 'overlay', 'density'. Defaults to 'network' (index 0).
    """
    st.markdown(
        "<div style='display:flex;align-items:center;gap:8px;"
        "margin-bottom:4px;font-size:12px;color:#6B7280'>"
        "<span style='font-weight:600'>Visualization:</span></div>",
        unsafe_allow_html=True,
    )
    mode = st.radio(
        "viz_mode",
        options=["🔵  Network", "🎨  Overlay", "🌡️  Density"],
        horizontal=True,
        index=0,
        key=f"viz_mode_{key}",
        label_visibility="collapsed",
    )
    st.markdown(
        "<div style='font-size:11px;color:#9CA3AF;margin-bottom:8px'>"
        "Network&nbsp;= cluster colors &nbsp;·&nbsp; "
        "Overlay&nbsp;= avg publication year (Viridis) &nbsp;·&nbsp; "
        "Density&nbsp;= Gaussian KDE heatmap (Viridis)"
        "</div>",
        unsafe_allow_html=True,
    )
    if "Network" in mode:
        return "network"
    if "Overlay" in mode:
        return "overlay"
    return "density"


def controls_hint(show_bandwidth: bool = False) -> None:
    """Render a small label row describing the sidebar controls."""
    cols = ["🔍 Search", "Min link strength", "Min node weight", "Labels", "Physics"]
    if show_bandwidth:
        cols.append("Kernel width (h)")
    st.markdown(
        "<div style='display:flex;gap:16px;align-items:center;"
        "margin-bottom:2px;font-size:11px;color:#9CA3AF'>"
        + "".join(f"<span>{c}</span>" for c in cols)
        + "</div>",
        unsafe_allow_html=True,
    )


def cluster_legend(G: nx.Graph) -> None:
    """Render a color-coded cluster legend for a NetworkX graph."""
    clusters = sorted({d.get("community", 0) for _, d in G.nodes(data=True)})
    st.markdown(
        f"<p style='font-size:11px;font-weight:600;color:{config.TEXT_SECONDARY};"
        f"margin-top:12px;margin-bottom:4px'>CLUSTERS</p>",
        unsafe_allow_html=True,
    )
    for cid in clusters[:18]:
        color = config.COMMUNITY_COLORS[cid % len(config.COMMUNITY_COLORS)]
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:3px'>"
            f"<div style='width:9px;height:9px;border-radius:50%;background:{color};"
            f"flex-shrink:0'></div>"
            f"<span style='font-size:10px;color:{config.TEXT_PRIMARY}'>Cluster&nbsp;{cid + 1}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
