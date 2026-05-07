"""
network_viz.py — VOSviewer-style force-directed network visualizations
for co-authorship, keyword co-occurrence, and topic landscape networks.

All graphs use PyVis with Barnes-Hut physics, community coloring,
size-proportional nodes, and dark canvas matching the app theme.
"""

import json
import logging
import math
import os
import tempfile
from typing import Any, Dict, List, Optional

import networkx as nx
import streamlit as st

import config

logger = logging.getLogger(__name__)

# ── Shared PyVis HTML template injected into every graph ──────────────────────
_SHARED_JS = """
<style>
  #mynetwork {
    background-color: """ + config.CANVAS_BACKGROUND + """;
    border: 1px solid """ + config.BORDER_COLOR + """;
    border-radius: 8px;
  }
  .vis-tooltip {
    background-color: """ + config.SURFACE_ELEVATED + """ !important;
    color: """ + config.TEXT_PRIMARY + """ !important;
    border: 1px solid """ + config.BORDER_COLOR + """ !important;
    border-radius: 6px !important;
    font-family: 'Open Sans', sans-serif !important;
    font-size: 12px !important;
    padding: 8px !important;
    max-width: 280px !important;
  }
</style>
"""


def _get_pyvis_net(height: str = "700px", directed: bool = False):
    """Create a configured PyVis Network instance."""
    from pyvis.network import Network
    net = Network(
        height=height,
        width="100%",
        bgcolor=config.CANVAS_BACKGROUND,
        font_color=config.TEXT_PRIMARY,
        directed=directed,
        notebook=False,
    )
    net.set_options(json.dumps({
        "physics": config.BARNES_HUT_PHYSICS,
        "nodes": {
            "font": {
                "face": "Open Sans",
                "color": config.TEXT_PRIMARY,
            },
            "borderWidth": 1.5,
            "borderWidthSelected": 3,
        },
        "edges": {
            "smooth": {"type": "continuous"},
            "scaling": {"min": config.EDGE_WIDTH_MIN, "max": config.EDGE_WIDTH_MAX},
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 100,
            "navigationButtons": True,
            "keyboard": True,
            "zoomView": True,
        },
        "layout": {"improvedLayout": True},
    }))
    return net


def _color_with_opacity(hex_color: str, opacity: float) -> str:
    """Convert hex color to rgba string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"


def _community_color(community_id: int) -> str:
    return config.COMMUNITY_COLORS[community_id % len(config.COMMUNITY_COLORS)]


def _render_pyvis_in_streamlit(net, key: str, height: int = 700):
    """Write PyVis HTML to a temp file and render in Streamlit."""
    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as f:
        net.save_graph(f.name)
        html_path = f.name

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    os.unlink(html_path)

    # Inject shared styles
    html = html.replace("</head>", _SHARED_JS + "</head>", 1)
    st.components.v1.html(html, height=height + 30, scrolling=False)


def _build_controls_panel(network_type: str) -> Dict[str, Any]:
    """
    Render the controls panel above a network graph.
    Returns dict of selected filter values.
    """
    cols = st.columns([3, 2, 2, 2, 1])
    with cols[0]:
        search = st.text_input(
            "Search nodes",
            placeholder="Type to highlight...",
            key=f"search_{network_type}",
            label_visibility="collapsed",
        )
    with cols[1]:
        min_edge = st.slider(
            "Min link strength",
            min_value=1, max_value=20, value=1,
            key=f"min_edge_{network_type}",
            label_visibility="collapsed",
        )
    with cols[2]:
        min_node = st.slider(
            "Min node weight",
            min_value=1, max_value=20, value=1,
            key=f"min_node_{network_type}",
            label_visibility="collapsed",
        )
    with cols[3]:
        freeze = st.toggle("Freeze layout", value=False, key=f"freeze_{network_type}")
    with cols[4]:
        density = st.toggle("Density", value=False, key=f"density_{network_type}")

    return {
        "search": search,
        "min_edge": min_edge,
        "min_node": min_node,
        "freeze": freeze,
        "density": density,
    }


def _apply_filters(
    G: nx.Graph,
    min_node_weight: int,
    min_edge_weight: int,
    weight_attr: str = "paper_count",
    search: str = "",
) -> nx.Graph:
    """Filter graph by node weight and edge weight thresholds."""
    G2 = G.copy()

    # Remove nodes below weight threshold
    remove_nodes = [
        n for n, d in G2.nodes(data=True)
        if d.get(weight_attr, d.get("frequency", d.get("paper_count", 1))) < min_node_weight
    ]
    G2.remove_nodes_from(remove_nodes)

    # Remove edges below weight threshold
    remove_edges = [
        (u, v) for u, v, d in G2.edges(data=True)
        if d.get("weight", 1) < min_edge_weight
    ]
    G2.remove_edges_from(remove_edges)

    return G2


def _add_nodes_to_net(
    net,
    G: nx.Graph,
    search: str,
    label_attr: str = None,
    node_shape: str = "dot",
    shape_by_type: bool = False,
):
    """Add filtered/styled nodes to PyVis network."""
    search_lower = search.lower().strip() if search else ""

    # Compute median weight for label visibility
    weights = [
        d.get("paper_count", d.get("frequency", d.get("size", 1)))
        for _, d in G.nodes(data=True)
    ]
    median_w = sorted(weights)[len(weights) // 2] if weights else 0

    for node, data in G.nodes(data=True):
        node_str = str(node)
        label_text = str(data.get(label_attr, node)) if label_attr else node_str
        weight = data.get("paper_count", data.get("frequency", 1))
        size = float(data.get("size", config.NODE_SIZE_MIN))
        color = data.get("color", config.PRIMARY_ACCENT)
        community = data.get("community", 0)

        # Label visibility: only above-median nodes get labels
        show_label = weight >= median_w
        display_label = label_text[:30] if show_label else ""

        # Search highlight
        is_match = search_lower and search_lower in node_str.lower()
        border_color = "#FFD700" if is_match else _color_with_opacity(color, 0.7)
        border_width = 3 if is_match else 1.5

        # Shape by keyword type
        if shape_by_type:
            ktype = data.get("keyword_type", "author")
            shape = {
                "author": "dot",
                "mesh_descriptor": "square",
                "chemical": "diamond",
                "mesh_qualifier": "triangle",
                "publication_type": "star",
            }.get(ktype, "dot")
        else:
            shape = node_shape

        # Build tooltip
        top_neighbors = sorted(
            [(nb, G[node][nb].get("weight", 1)) for nb in G.neighbors(node)],
            key=lambda x: x[1], reverse=True,
        )[:5]
        top_nb_str = "<br>".join(
            f"  • {nb}: {w}" for nb, w in top_neighbors
        )

        tooltip = (
            f"<b>{node_str}</b><br>"
            f"Type: {data.get('keyword_type', data.get('entity_type', '—'))}<br>"
            f"Weight: {weight}<br>"
            f"Community: {community}<br>"
            f"Top connections:<br>{top_nb_str}"
        )

        net.add_node(
            node_str,
            label=display_label,
            size=size,
            color={
                "background": color,
                "border": border_color,
                "highlight": {"background": "#FFD700", "border": "#FFA500"},
                "hover": {"background": color, "border": "#FFFFFF"},
            },
            borderWidth=border_width,
            title=tooltip,
            shape=shape,
            font={"size": max(8, int(size * 0.35)), "color": config.TEXT_PRIMARY},
        )


def _add_edges_to_net(net, G: nx.Graph, directed: bool = False):
    """Add styled edges to PyVis network."""
    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1)
        width = float(data.get("width", config.EDGE_WIDTH_MIN))
        src_color = G.nodes[u].get("color", config.PRIMARY_ACCENT)
        edge_color = _color_with_opacity(src_color, 0.30)

        pmids = data.get("evidence_pmids", [])
        pmid_str = ", ".join(str(p) for p in pmids[:3])
        tooltip = (
            f"<b>{u}</b> → <b>{v}</b><br>"
            f"Weight: {weight}<br>"
            + (f"PMIDs: {pmid_str}" if pmid_str else "")
        )

        net.add_edge(
            str(u), str(v),
            weight=weight,
            width=width,
            color={"color": edge_color, "highlight": "#FFD700", "hover": "#FFFFFF"},
            title=tooltip,
            arrows={"to": {"enabled": directed, "scaleFactor": 0.5}},
        )


# ── A: Co-authorship network ──────────────────────────────────────────────────

def render_coauthorship_network(
    G: nx.Graph,
    controls: Optional[Dict] = None,
    height: int = 700,
):
    """
    Render co-authorship network in VOSviewer style.
    Node size = publication count | Color = research cluster |
    Edge thickness = collaboration strength.
    """
    if G is None or G.number_of_nodes() == 0:
        st.info("No co-authorship data available. Run the pipeline first.")
        return

    if controls is None:
        controls = _build_controls_panel("coauth")

    G2 = _apply_filters(
        G,
        min_node_weight=controls.get("min_node", 1),
        min_edge_weight=controls.get("min_edge", 1),
        weight_attr="paper_count",
        search=controls.get("search", ""),
    )

    if G2.number_of_nodes() == 0:
        st.warning("All nodes filtered out. Lower the minimum weight thresholds.")
        return

    net = _get_pyvis_net(height=f"{height}px", directed=False)

    if controls.get("freeze"):
        net.toggle_physics(False)

    _add_nodes_to_net(net, G2, search=controls.get("search", ""), node_shape="dot")
    _add_edges_to_net(net, G2)
    _render_pyvis_in_streamlit(net, key="coauth", height=height)


# ── B: Keyword co-occurrence network ─────────────────────────────────────────

def render_keyword_network(
    G: nx.Graph,
    controls: Optional[Dict] = None,
    height: int = 700,
):
    """
    Render keyword co-occurrence map in VOSviewer style.
    Node size = frequency | Color = thematic cluster |
    Shape = keyword type (dot/square/diamond/triangle).
    """
    if G is None or G.number_of_nodes() == 0:
        st.info("No keyword co-occurrence data available. Run the pipeline first.")
        return

    if controls is None:
        controls = _build_controls_panel("keyword")

    G2 = _apply_filters(
        G,
        min_node_weight=controls.get("min_node", 1),
        min_edge_weight=controls.get("min_edge", 1),
        weight_attr="frequency",
        search=controls.get("search", ""),
    )

    if G2.number_of_nodes() == 0:
        st.warning("All nodes filtered out. Lower the minimum weight thresholds.")
        return

    net = _get_pyvis_net(height=f"{height}px", directed=False)

    if controls.get("freeze"):
        net.toggle_physics(False)

    _add_nodes_to_net(net, G2, search=controls.get("search", ""), shape_by_type=True)
    _add_edges_to_net(net, G2)
    _render_pyvis_in_streamlit(net, key="keyword", height=height)


# ── C: Topic landscape network ────────────────────────────────────────────────

def render_topic_network(
    G: nx.Graph,
    controls: Optional[Dict] = None,
    height: int = 700,
    year_range: Optional[tuple] = None,
):
    """
    Render topic landscape in VOSviewer style.
    Node size = paper count in topic | Edge = shared papers between topics.
    """
    if G is None or G.number_of_nodes() == 0:
        st.info("No topic data available. Run the pipeline first.")
        return

    if controls is None:
        controls = _build_controls_panel("topic")

    G2 = _apply_filters(
        G,
        min_node_weight=controls.get("min_node", 1),
        min_edge_weight=controls.get("min_edge", 1),
        weight_attr="paper_count",
        search=controls.get("search", ""),
    )

    if G2.number_of_nodes() == 0:
        st.warning("All nodes filtered out. Lower the minimum weight thresholds.")
        return

    net = _get_pyvis_net(height=f"{height}px", directed=False)

    if controls.get("freeze"):
        net.toggle_physics(False)

    # For topics: label is always shown (there are few nodes)
    for node, data in G2.nodes(data=True):
        node_str = str(node)
        label = str(data.get("label", f"Topic {node}"))[:40]
        top_words = data.get("top_words", [])
        size = float(data.get("size", config.NODE_SIZE_MIN))
        color = data.get("color", config.PRIMARY_ACCENT)
        paper_count = data.get("paper_count", 0)

        tooltip = (
            f"<b>{label}</b><br>"
            f"Papers: {paper_count}<br>"
            f"Top words: {', '.join(top_words[:5])}"
        )
        search_lower = controls.get("search", "").lower()
        border_color = "#FFD700" if search_lower and search_lower in label.lower() else _color_with_opacity(color, 0.7)

        net.add_node(
            node_str,
            label=label,
            size=size,
            color={"background": color, "border": border_color,
                   "highlight": {"background": "#FFD700", "border": "#FFA500"}},
            title=tooltip,
            shape="dot",
            font={"size": max(9, int(size * 0.3)), "color": config.TEXT_PRIMARY},
        )

    _add_edges_to_net(net, G2)
    _render_pyvis_in_streamlit(net, key="topic", height=height)


# ── Network statistics panel ──────────────────────────────────────────────────

def render_network_stats(stats: Dict[str, Any], title: str = "Network Statistics"):
    """Render a collapsible statistics panel alongside a network graph."""
    with st.expander(title, expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nodes", stats.get("node_count", 0))
            st.metric("Edges", stats.get("edge_count", 0))
            st.metric("Communities", stats.get("num_communities", 0))
        with col2:
            st.metric("Density", f"{stats.get('density', 0):.4f}")
            st.metric("Modularity", f"{stats.get('modularity', 0):.4f}")
            st.metric("Avg Clustering", f"{stats.get('avg_clustering_coefficient', 0):.4f}")

        top_deg = stats.get("top10_degree_centrality", [])
        if top_deg:
            st.markdown("**Top nodes by degree centrality:**")
            for node, score in top_deg[:5]:
                st.markdown(
                    f"<span style='color:{config.PRIMARY_ACCENT}'>{node}</span> — {score:.3f}",
                    unsafe_allow_html=True,
                )
