"""
network_viz.py — VOSviewer-accurate bibliometric network visualizations.

White canvas, cluster-colored nodes, colored semi-transparent curved edges,
always-visible labels, and Barnes-Hut physics with strong cluster separation.
Matches the look of real VOSviewer co-authorship / keyword maps.
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

# ── VOSviewer-style constants ─────────────────────────────────────────────────
_VOS_BG        = "#FFFFFF"
_VOS_FONT      = "#1F2937"
_VOS_HIGHLIGHT = "#FFD700"

# White-canvas tooltip + nav-button overrides
_VOS_CSS = """
<style>
  #mynetwork {
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.08);
  }
  .vis-tooltip {
    background-color: #FFFFFF !important;
    color: #111827 !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 6px !important;
    font-family: 'Open Sans', sans-serif !important;
    font-size: 12px !important;
    padding: 8px 10px !important;
    max-width: 300px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12) !important;
  }
  .vis-navigation .vis-button {
    background-color: #F9FAFB !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 4px !important;
  }
  .vis-navigation .vis-button:hover {
    background-color: #F3F4F6 !important;
  }
</style>
"""

# Physics that produces tight cluster separation matching VOSviewer
_VOS_PHYSICS = {
    "barnesHut": {
        "gravitationalConstant": -8000,
        "centralGravity": 0.15,
        "springLength": 130,
        "springConstant": 0.04,
        "damping": 0.10,
        "avoidOverlap": 0.8,
    },
    "minVelocity": 0.5,
    "stabilization": {
        "enabled": True,
        "iterations": 1500,
        "updateInterval": 50,
    },
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_vosviewer_net(height: str = "700px"):
    """Return a PyVis Network pre-configured for VOSviewer style."""
    from pyvis.network import Network
    net = Network(
        height=height,
        width="100%",
        bgcolor=_VOS_BG,
        font_color=_VOS_FONT,
        directed=False,
        notebook=False,
    )
    net.set_options(json.dumps({
        "physics": _VOS_PHYSICS,
        "nodes": {
            "font": {
                "face": "Open Sans",
                "color": _VOS_FONT,
                "size": 11,
                "strokeWidth": 4,
                "strokeColor": "#FFFFFF",
            },
            "borderWidth": 1.5,
            "borderWidthSelected": 3,
            "shape": "dot",
        },
        "edges": {
            "smooth": {"type": "continuous", "roundness": 0.15},
            "selectionWidth": 2,
            "scaling": {
                "min": config.EDGE_WIDTH_MIN,
                "max": config.EDGE_WIDTH_MAX,
            },
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 100,
            "navigationButtons": True,
            "keyboard": True,
            "zoomView": True,
            "hideEdgesOnDrag": True,
        },
        "layout": {"improvedLayout": False},
    }))
    return net


def _render_vosviewer_html(net, height: int):
    """Inject VOSviewer CSS into PyVis HTML and render in Streamlit."""
    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as f:
        net.save_graph(f.name)
        html_path = f.name

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(html_path)

    html = html.replace("</head>", _VOS_CSS + "</head>", 1)
    st.components.v1.html(html, height=height + 30, scrolling=False)


def _rgba(hex_color: str, opacity: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"


def _darken(hex_color: str, factor: float = 0.75) -> str:
    h = hex_color.lstrip("#")
    r = max(0, int(int(h[0:2], 16) * factor))
    g = max(0, int(int(h[2:4], 16) * factor))
    b = max(0, int(int(h[4:6], 16) * factor))
    return f"#{r:02x}{g:02x}{b:02x}"


def _community_color(community_id: int) -> str:
    return config.COMMUNITY_COLORS[community_id % len(config.COMMUNITY_COLORS)]


def _apply_filters(
    G: nx.Graph,
    min_node_weight: int,
    min_edge_weight: int,
    weight_attr: str = "paper_count",
) -> nx.Graph:
    G2 = G.copy()
    remove_nodes = [
        n for n, d in G2.nodes(data=True)
        if d.get(weight_attr, d.get("frequency", d.get("paper_count", 1))) < min_node_weight
    ]
    G2.remove_nodes_from(remove_nodes)
    remove_edges = [
        (u, v) for u, v, d in G2.edges(data=True)
        if d.get("weight", 1) < min_edge_weight
    ]
    G2.remove_edges_from(remove_edges)
    # Drop isolates after edge filtering
    G2.remove_nodes_from(list(nx.isolates(G2)))
    return G2


# ── Controls panel ────────────────────────────────────────────────────────────

def _build_controls_panel(network_type: str) -> Dict[str, Any]:
    """Render the filter / search bar above a network graph."""
    c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 1, 1])
    with c1:
        search = st.text_input(
            "Search",
            placeholder="Highlight node...",
            key=f"search_{network_type}",
            label_visibility="collapsed",
        )
    with c2:
        min_edge = st.slider(
            "Min link strength",
            min_value=1, max_value=20, value=1,
            key=f"min_edge_{network_type}",
            label_visibility="collapsed",
        )
    with c3:
        min_node = st.slider(
            "Min node weight",
            min_value=1, max_value=20, value=1,
            key=f"min_node_{network_type}",
            label_visibility="collapsed",
        )
    with c4:
        freeze = st.toggle("Freeze", value=False, key=f"freeze_{network_type}")
    with c5:
        labels_all = st.toggle("All labels", value=True, key=f"labels_{network_type}")

    return {
        "search": search,
        "min_edge": min_edge,
        "min_node": min_node,
        "freeze": freeze,
        "labels_all": labels_all,
    }


# ── Node / edge adders ────────────────────────────────────────────────────────

def _add_coauth_nodes(net, G: nx.Graph, search: str, labels_all: bool = True):
    """Author nodes: circle, cluster color, always-visible name label."""
    search_lower = search.lower().strip() if search else ""

    # Compute median for optional label threshold
    weights = [d.get("paper_count", 1) for _, d in G.nodes(data=True)]
    median_w = sorted(weights)[len(weights) // 2] if weights else 1

    for node, data in G.nodes(data=True):
        node_str  = str(node)
        color     = data.get("color", config.COMMUNITY_COLORS[0])
        size      = float(data.get("size", config.NODE_SIZE_MIN))
        weight    = data.get("paper_count", 1)
        community = data.get("community", 0)

        is_match     = search_lower and search_lower in node_str.lower()
        border_color = _VOS_HIGHLIGHT if is_match else _darken(color, 0.78)
        border_width = 3 if is_match else 1.5

        show_label = labels_all or is_match or weight >= median_w
        label_text = node_str if show_label else ""

        top_nb = sorted(
            [(nb, G[node][nb].get("weight", 1)) for nb in G.neighbors(node)],
            key=lambda x: x[1], reverse=True,
        )[:5]
        nb_html = "<br>".join(f"&nbsp;&nbsp;• {nb} ({w})" for nb, w in top_nb)
        tooltip = (
            f"<b>{node_str}</b><br>"
            f"Publications:&nbsp;{weight}<br>"
            f"Cluster:&nbsp;{community}<br>"
            f"Top collaborators:<br>{nb_html or '—'}"
        )

        net.add_node(
            node_str,
            label=label_text,
            size=size,
            color={
                "background": color,
                "border": border_color,
                "highlight": {"background": _VOS_HIGHLIGHT, "border": "#FFA500"},
                "hover": {"background": color, "border": _darken(color, 0.65)},
            },
            borderWidth=border_width,
            title=tooltip,
            shape="dot",
            font={
                "size": max(9, int(size * 0.30)),
                "color": _VOS_FONT,
                "face": "Open Sans",
                "strokeWidth": 4,
                "strokeColor": "#FFFFFF",
            },
        )


def _add_coauth_edges(net, G: nx.Graph):
    """Co-authorship edges: colored to match source cluster, semi-transparent."""
    for u, v, data in G.edges(data=True):
        weight    = data.get("weight", 1)
        width     = float(data.get("width", config.EDGE_WIDTH_MIN))
        src_color = G.nodes[u].get("color", config.COMMUNITY_COLORS[0])

        net.add_edge(
            str(u), str(v),
            weight=weight,
            width=width,
            color={
                "color": _rgba(src_color, 0.45),
                "highlight": _VOS_HIGHLIGHT,
                "hover": _rgba(src_color, 0.80),
            },
            title=f"<b>{u}</b> — <b>{v}</b><br>Shared publications: {weight}",
            arrows={"to": {"enabled": False}},
        )


def _add_keyword_nodes(net, G: nx.Graph, search: str, labels_all: bool = True):
    """Keyword nodes: shape varies by keyword type, cluster color."""
    search_lower = search.lower().strip() if search else ""

    weights = [d.get("frequency", 1) for _, d in G.nodes(data=True)]
    median_w = sorted(weights)[len(weights) // 2] if weights else 1

    _shape_map = {
        "author": "dot",
        "mesh_descriptor": "square",
        "chemical": "diamond",
        "mesh_qualifier": "triangle",
        "publication_type": "star",
    }

    for node, data in G.nodes(data=True):
        node_str = str(node)
        color    = data.get("color", config.COMMUNITY_COLORS[0])
        size     = float(data.get("size", config.NODE_SIZE_MIN))
        freq     = data.get("frequency", 1)
        ktype    = data.get("keyword_type", "author")

        is_match     = search_lower and search_lower in node_str.lower()
        border_color = _VOS_HIGHLIGHT if is_match else _darken(color, 0.78)
        border_width = 3 if is_match else 1.5

        show_label = labels_all or is_match or freq >= median_w
        label_text = node_str if show_label else ""

        tooltip = (
            f"<b>{node_str}</b><br>"
            f"Type:&nbsp;{ktype}<br>"
            f"Occurrences:&nbsp;{freq}"
        )

        net.add_node(
            node_str,
            label=label_text,
            size=size,
            color={
                "background": color,
                "border": border_color,
                "highlight": {"background": _VOS_HIGHLIGHT, "border": "#FFA500"},
                "hover": {"background": color, "border": _darken(color, 0.65)},
            },
            borderWidth=border_width,
            title=tooltip,
            shape=_shape_map.get(ktype, "dot"),
            font={
                "size": max(9, int(size * 0.30)),
                "color": _VOS_FONT,
                "face": "Open Sans",
                "strokeWidth": 4,
                "strokeColor": "#FFFFFF",
            },
        )


def _add_keyword_edges(net, G: nx.Graph):
    for u, v, data in G.edges(data=True):
        weight    = data.get("weight", 1)
        width     = float(data.get("width", config.EDGE_WIDTH_MIN))
        src_color = G.nodes[u].get("color", config.COMMUNITY_COLORS[0])

        net.add_edge(
            str(u), str(v),
            weight=weight,
            width=width,
            color={
                "color": _rgba(src_color, 0.40),
                "highlight": _VOS_HIGHLIGHT,
                "hover": _rgba(src_color, 0.75),
            },
            title=f"<b>{u}</b> — <b>{v}</b><br>Co-occurrences: {weight}",
            arrows={"to": {"enabled": False}},
        )


# ── Public render functions ────────────────────────────────────────────────────

def render_coauthorship_network(
    G: nx.Graph,
    controls: Optional[Dict] = None,
    height: int = 750,
):
    """
    VOSviewer-style co-authorship network.
    Node size ∝ publication count · Color = research cluster ·
    Edge thickness ∝ collaboration strength.
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
    )

    if G2.number_of_nodes() == 0:
        st.warning("All nodes filtered out — lower the minimum weight thresholds.")
        return

    # Stats bar
    communities = len({d.get("community", 0) for _, d in G2.nodes(data=True)})
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Authors", G2.number_of_nodes())
    s2.metric("Links", G2.number_of_edges())
    s3.metric("Clusters", communities)
    s4.metric("Total link strength",
              sum(d.get("weight", 1) for _, _, d in G2.edges(data=True)))

    net = _get_vosviewer_net(height=f"{height}px")
    if controls.get("freeze"):
        net.toggle_physics(False)

    _add_coauth_nodes(
        net, G2,
        search=controls.get("search", ""),
        labels_all=controls.get("labels_all", True),
    )
    _add_coauth_edges(net, G2)
    _render_vosviewer_html(net, height=height)


def render_keyword_network(
    G: nx.Graph,
    controls: Optional[Dict] = None,
    height: int = 750,
):
    """
    VOSviewer-style keyword co-occurrence map.
    Node size ∝ frequency · Color = thematic cluster ·
    Shape = keyword type (●=author  ■=MeSH  ◆=chemical  ▲=qualifier).
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
    )

    if G2.number_of_nodes() == 0:
        st.warning("All nodes filtered out — lower the minimum weight thresholds.")
        return

    communities = len({d.get("community", 0) for _, d in G2.nodes(data=True)})
    s1, s2, s3 = st.columns(3)
    s1.metric("Keywords", G2.number_of_nodes())
    s2.metric("Links", G2.number_of_edges())
    s3.metric("Clusters", communities)

    net = _get_vosviewer_net(height=f"{height}px")
    if controls.get("freeze"):
        net.toggle_physics(False)

    _add_keyword_nodes(
        net, G2,
        search=controls.get("search", ""),
        labels_all=controls.get("labels_all", True),
    )
    _add_keyword_edges(net, G2)
    _render_vosviewer_html(net, height=height)


def render_topic_network(
    G: nx.Graph,
    controls: Optional[Dict] = None,
    height: int = 750,
    year_range: Optional[tuple] = None,
):
    """
    VOSviewer-style research topic landscape.
    Node size ∝ paper count · Edge = shared papers between topics.
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
    )

    if G2.number_of_nodes() == 0:
        st.warning("All nodes filtered out — lower the minimum weight thresholds.")
        return

    net = _get_vosviewer_net(height=f"{height}px")
    if controls.get("freeze"):
        net.toggle_physics(False)

    search_lower = controls.get("search", "").lower()

    for node, data in G2.nodes(data=True):
        node_str    = str(node)
        label       = str(data.get("label", f"Topic {node}"))[:40]
        top_words   = data.get("top_words", [])
        size        = float(data.get("size", config.NODE_SIZE_MIN))
        color       = data.get("color", config.COMMUNITY_COLORS[0])
        paper_count = data.get("paper_count", 0)

        is_match     = search_lower and search_lower in label.lower()
        border_color = _VOS_HIGHLIGHT if is_match else _darken(color, 0.78)

        net.add_node(
            node_str,
            label=label,
            size=size,
            color={
                "background": color,
                "border": border_color,
                "highlight": {"background": _VOS_HIGHLIGHT, "border": "#FFA500"},
                "hover": {"background": color, "border": _darken(color, 0.65)},
            },
            title=(
                f"<b>{label}</b><br>"
                f"Papers: {paper_count}<br>"
                f"Top words: {', '.join(top_words[:5])}"
            ),
            shape="dot",
            font={
                "size": max(10, int(size * 0.30)),
                "color": _VOS_FONT,
                "face": "Open Sans",
                "strokeWidth": 4,
                "strokeColor": "#FFFFFF",
            },
        )

    for u, v, data in G2.edges(data=True):
        weight    = data.get("weight", 1)
        width     = float(data.get("width", config.EDGE_WIDTH_MIN))
        src_color = G2.nodes[u].get("color", config.COMMUNITY_COLORS[0])
        net.add_edge(
            str(u), str(v),
            weight=weight,
            width=width,
            color={
                "color": _rgba(src_color, 0.40),
                "highlight": _VOS_HIGHLIGHT,
                "hover": _rgba(src_color, 0.75),
            },
            arrows={"to": {"enabled": False}},
        )

    _render_vosviewer_html(net, height=height)


# ── Network statistics panel ──────────────────────────────────────────────────

def render_network_stats(stats: Dict[str, Any], title: str = "Network Statistics"):
    with st.expander(title, expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nodes",       stats.get("node_count", 0))
            st.metric("Edges",       stats.get("edge_count", 0))
            st.metric("Communities", stats.get("num_communities", 0))
        with col2:
            st.metric("Density",      f"{stats.get('density', 0):.4f}")
            st.metric("Modularity",   f"{stats.get('modularity', 0):.4f}")
            st.metric("Avg Clust.",   f"{stats.get('avg_clustering_coefficient', 0):.4f}")

        top_deg = stats.get("top10_degree_centrality", [])
        if top_deg:
            st.markdown("**Top nodes by degree:**")
            for node, score in top_deg[:5]:
                st.markdown(
                    f"<span style='color:{config.PRIMARY_ACCENT}'>{node}</span>"
                    f" — {score:.3f}",
                    unsafe_allow_html=True,
                )
