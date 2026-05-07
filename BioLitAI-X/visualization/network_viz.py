"""
network_viz.py — VOSviewer-accurate bibliometric network visualizations.

Three visualization modes matching VOSviewer 1.6.18+:

  Network Visualization  — cluster-colored nodes, fixed VOS-like layout,
                           interactive (drag/zoom), colored edges.

  Overlay Visualization  — same positions, Viridis color gradient mapped
                           to each node's average publication year score.

  Density Visualization  — Gaussian KDE heatmap using the exact VOSviewer
                           formula: D(x) = Σ wᵢ·K(‖x−xᵢ‖ / (d̄·h))
                           with Viridis colorscale, white node circles on top.

All three modes share the same pre-computed kamada-kawai layout so positions
are consistent when switching between modes.
"""

import json
import logging
import math
import os
import random
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import streamlit as st

import config

logger = logging.getLogger(__name__)


# ── VOSviewer visual constants ────────────────────────────────────────────────
_VOS_BG        = "#FFFFFF"
_VOS_FONT      = "#1F2937"
_VOS_HIGHLIGHT = "#FFD700"
_LAYOUT_SCALE  = 1400.0          # NetworkX [-1,1] → PyVis pixel space

# VOSviewer v1.6.7+ uses Viridis for both overlay and density
_OVERLAY_COLORSCALE = "Viridis"  # dark-blue (low/old) → bright-yellow (high/new)
_DENSITY_COLORSCALE = "Viridis"  # dark-blue (sparse)  → bright-yellow (dense)

# White-canvas CSS injected into every PyVis HTML frame
_VOS_CSS = """
<style>
  #mynetwork {
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.07);
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
</style>
"""

# Barnes-Hut physics used when the physics toggle is ON
_VOS_PHYSICS_ON = {
    "barnesHut": {
        "gravitationalConstant": -8000,
        "centralGravity": 0.15,
        "springLength": 130,
        "springConstant": 0.04,
        "damping": 0.10,
        "avoidOverlap": 0.8,
    },
    "minVelocity": 0.5,
    "stabilization": {"enabled": True, "iterations": 1500, "updateInterval": 50},
}


# ── Layout computation (shared across all three modes) ────────────────────────

def _compute_layout(G: nx.Graph, layout_key: str = "") -> Dict[str, Tuple[float, float]]:
    """
    Compute and cache the VOS-like layout for graph G.

    Uses kamada_kawai_layout (≤300 nodes) or spring_layout for larger graphs —
    both approximate the multidimensional-scaling approach that VOSviewer uses.
    Results are cached in st.session_state so all three viz modes share the
    same positions.

    Returns {str(node): (x, y)} with values in approximately [-1, 1].
    """
    cache_key = f"__pos_{layout_key}_{G.number_of_nodes()}_{G.number_of_edges()}"
    cached = st.session_state.get(cache_key)
    if cached is not None:
        return cached

    n = G.number_of_nodes()
    if n == 0:
        return {}

    try:
        if n <= 300:
            pos = nx.kamada_kawai_layout(G, weight="weight")
        else:
            k = 2.5 / math.sqrt(n)
            pos = nx.spring_layout(G, weight="weight", seed=42, k=k, iterations=100)
    except Exception:
        pos = nx.random_layout(G, seed=42)

    result: Dict[str, Tuple[float, float]] = {
        str(node): (float(x), float(y)) for node, (x, y) in pos.items()
    }
    st.session_state[cache_key] = result
    return result


def _compute_d_bar(pos: Dict[str, Tuple[float, float]]) -> float:
    """
    Average pairwise distance between all items — VOSviewer's d̄.

    Formula: d̄ = (2 / (n(n−1))) · Σ_{i<j} ‖xᵢ − xⱼ‖

    Sampled (≤3000 pairs) for efficiency on large graphs.
    """
    positions = list(pos.values())
    n = len(positions)
    if n < 2:
        return 1.0

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(all_pairs) > 3000:
        all_pairs = random.sample(all_pairs, 3000)

    total = sum(
        math.sqrt((positions[i][0] - positions[j][0]) ** 2 +
                  (positions[i][1] - positions[j][1]) ** 2)
        for i, j in all_pairs
    )
    return total / len(all_pairs) if all_pairs else 1.0


# ── Internal PyVis helpers ────────────────────────────────────────────────────

def _get_vosviewer_net(height: str = "700px", physics: bool = False):
    """Create a PyVis Network; physics off by default (positions pre-computed)."""
    from pyvis.network import Network

    net = Network(
        height=height, width="100%",
        bgcolor=_VOS_BG, font_color=_VOS_FONT,
        directed=False, notebook=False,
    )
    opts: Dict[str, Any] = {
        "nodes": {
            "font": {
                "face": "Open Sans", "color": _VOS_FONT,
                "size": 11, "strokeWidth": 4, "strokeColor": "#FFFFFF",
            },
            "borderWidth": 1.5,
            "borderWidthSelected": 3,
            "shape": "dot",
        },
        "edges": {
            "smooth": {"type": "continuous", "roundness": 0.15},
            "selectionWidth": 2,
            "scaling": {"min": config.EDGE_WIDTH_MIN, "max": config.EDGE_WIDTH_MAX},
        },
        "interaction": {
            "hover": True, "tooltipDelay": 100,
            "navigationButtons": True, "keyboard": True,
            "zoomView": True, "dragNodes": True, "dragView": True,
        },
        "layout": {"improvedLayout": False},
    }
    if physics:
        opts["physics"] = _VOS_PHYSICS_ON
    else:
        opts["physics"] = {"enabled": False}

    net.set_options(json.dumps(opts))
    return net


def _render_vosviewer_html(net, height: int):
    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as f:
        net.save_graph(f.name)
        path = f.name

    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    os.unlink(path)

    html = html.replace("</head>", _VOS_CSS + "</head>", 1)
    st.components.v1.html(html, height=height + 30, scrolling=False)


# ── Shared helpers ────────────────────────────────────────────────────────────

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


def _apply_filters(
    G: nx.Graph,
    min_node_weight: int,
    min_edge_weight: int,
    weight_attr: str = "paper_count",
) -> nx.Graph:
    G2 = G.copy()
    G2.remove_nodes_from([
        n for n, d in G2.nodes(data=True)
        if d.get(weight_attr, d.get("frequency", d.get("paper_count", 1))) < min_node_weight
    ])
    G2.remove_edges_from([
        (u, v) for u, v, d in G2.edges(data=True) if d.get("weight", 1) < min_edge_weight
    ])
    G2.remove_nodes_from(list(nx.isolates(G2)))
    return G2


# ── Controls panel ────────────────────────────────────────────────────────────

def _build_controls_panel(network_type: str, show_bandwidth: bool = False) -> Dict[str, Any]:
    """Controls row above any network graph."""
    cols = st.columns([3, 2, 2, 1, 1] + ([2] if show_bandwidth else []))
    with cols[0]:
        search = st.text_input(
            "Search", placeholder="Highlight node...",
            key=f"search_{network_type}", label_visibility="collapsed",
        )
    with cols[1]:
        min_edge = st.slider(
            "Min link strength", min_value=1, max_value=20, value=1,
            key=f"min_edge_{network_type}", label_visibility="collapsed",
        )
    with cols[2]:
        min_node = st.slider(
            "Min node weight", min_value=1, max_value=20, value=1,
            key=f"min_node_{network_type}", label_visibility="collapsed",
        )
    with cols[3]:
        labels_all = st.toggle("Labels", value=True, key=f"labels_{network_type}")
    with cols[4]:
        physics = st.toggle("Physics", value=False, key=f"physics_{network_type}")

    bandwidth = 1.0
    if show_bandwidth and len(cols) > 5:
        with cols[5]:
            bandwidth = st.slider(
                "Kernel width (h)", min_value=0.2, max_value=3.0, value=1.0, step=0.1,
                key=f"bw_{network_type}", label_visibility="collapsed",
            )

    return {
        "search": search, "min_edge": min_edge, "min_node": min_node,
        "labels_all": labels_all, "physics": physics, "bandwidth": bandwidth,
    }


# ── Node / edge adders for Network view ──────────────────────────────────────

def _add_coauth_nodes(
    net, G: nx.Graph, pos: Dict[str, Tuple[float, float]],
    search: str, labels_all: bool,
):
    search_lower = search.lower().strip() if search else ""
    weights = [d.get("paper_count", 1) for _, d in G.nodes(data=True)]
    median_w = sorted(weights)[len(weights) // 2] if weights else 1

    for node, data in G.nodes(data=True):
        ns       = str(node)
        color    = data.get("color", config.COMMUNITY_COLORS[0])
        size     = float(data.get("size", config.NODE_SIZE_MIN))
        weight   = data.get("paper_count", 1)
        comm     = data.get("community", 0)
        x, y     = pos.get(ns, (0.0, 0.0))

        is_match      = search_lower and search_lower in ns.lower()
        border_color  = _VOS_HIGHLIGHT if is_match else _darken(color, 0.78)
        border_width  = 3 if is_match else 1.5
        show_label    = labels_all or is_match or weight >= median_w
        label_text    = ns if show_label else ""

        top_nb = sorted(
            [(nb, G[node][nb].get("weight", 1)) for nb in G.neighbors(node)],
            key=lambda x: x[1], reverse=True,
        )[:5]
        nb_html = "<br>".join(f"&nbsp;&nbsp;• {nb} ({w})" for nb, w in top_nb)
        tooltip = (
            f"<b>{ns}</b><br>Publications:&nbsp;{weight}<br>"
            f"Cluster:&nbsp;{comm}<br>Top collaborators:<br>{nb_html or '—'}"
        )

        net.add_node(
            ns, label=label_text, size=size,
            x=x * _LAYOUT_SCALE, y=y * _LAYOUT_SCALE,
            color={
                "background": color, "border": border_color,
                "highlight": {"background": _VOS_HIGHLIGHT, "border": "#FFA500"},
                "hover": {"background": color, "border": _darken(color, 0.65)},
            },
            borderWidth=border_width, title=tooltip, shape="dot",
            font={
                "size": max(9, int(size * 0.30)), "color": _VOS_FONT,
                "face": "Open Sans", "strokeWidth": 4, "strokeColor": "#FFFFFF",
            },
        )


def _add_keyword_nodes(
    net, G: nx.Graph, pos: Dict[str, Tuple[float, float]],
    search: str, labels_all: bool,
):
    search_lower = search.lower().strip() if search else ""
    weights = [d.get("frequency", 1) for _, d in G.nodes(data=True)]
    median_w = sorted(weights)[len(weights) // 2] if weights else 1

    _shape_map = {
        "author": "dot", "mesh_descriptor": "square",
        "chemical": "diamond", "mesh_qualifier": "triangle",
        "publication_type": "star",
    }

    for node, data in G.nodes(data=True):
        ns     = str(node)
        color  = data.get("color", config.COMMUNITY_COLORS[0])
        size   = float(data.get("size", config.NODE_SIZE_MIN))
        freq   = data.get("frequency", 1)
        ktype  = data.get("keyword_type", "author")
        x, y   = pos.get(ns, (0.0, 0.0))

        is_match     = search_lower and search_lower in ns.lower()
        border_color = _VOS_HIGHLIGHT if is_match else _darken(color, 0.78)
        border_width = 3 if is_match else 1.5
        show_label   = labels_all or is_match or freq >= median_w
        label_text   = ns if show_label else ""

        net.add_node(
            ns, label=label_text, size=size,
            x=x * _LAYOUT_SCALE, y=y * _LAYOUT_SCALE,
            color={
                "background": color, "border": border_color,
                "highlight": {"background": _VOS_HIGHLIGHT, "border": "#FFA500"},
                "hover": {"background": color, "border": _darken(color, 0.65)},
            },
            borderWidth=border_width,
            title=f"<b>{ns}</b><br>Type:&nbsp;{ktype}<br>Occurrences:&nbsp;{freq}",
            shape=_shape_map.get(ktype, "dot"),
            font={
                "size": max(9, int(size * 0.30)), "color": _VOS_FONT,
                "face": "Open Sans", "strokeWidth": 4, "strokeColor": "#FFFFFF",
            },
        )


def _add_edges_to_net(net, G: nx.Graph):
    for u, v, data in G.edges(data=True):
        weight    = data.get("weight", 1)
        width     = float(data.get("width", config.EDGE_WIDTH_MIN))
        src_color = G.nodes[u].get("color", config.COMMUNITY_COLORS[0])
        net.add_edge(
            str(u), str(v), weight=weight, width=width,
            color={
                "color": _rgba(src_color, 0.45),
                "highlight": _VOS_HIGHLIGHT,
                "hover": _rgba(src_color, 0.80),
            },
            title=f"<b>{u}</b> — <b>{v}</b><br>Shared publications: {weight}",
            arrows={"to": {"enabled": False}},
        )


# ── Overlay helper — average year per node ───────────────────────────────────

def _overlay_year_per_node(
    G2: nx.Graph, papers_df, network_type: str
) -> Dict[str, float]:
    """
    Compute average publication year for each node.
    Co-authorship → year averaged over each author's papers.
    Keyword      → year averaged over papers containing the keyword.
    """
    if papers_df is None or papers_df.empty:
        return {}

    node_strs = {str(n) for n in G2.nodes()}
    node_years: Dict[str, List[float]] = {ns: [] for ns in node_strs}

    def _year(row) -> Optional[int]:
        pd_val = row.get("pub_date", "")
        if not pd_val:
            return None
        try:
            yr = int(str(pd_val)[:4])
            return yr if 1900 <= yr <= 2030 else None
        except Exception:
            return None

    for _, row in papers_df.iterrows():
        yr = _year(row)
        if not yr:
            continue

        if network_type == "coauth":
            for author in (row.get("authors") or []):
                name = author["name"] if isinstance(author, dict) else str(author)
                if name in node_years:
                    node_years[name].append(yr)
        else:
            for kw in (row.get("author_keywords") or []):
                if kw and kw in node_years:
                    node_years[kw].append(yr)
            for mesh in (row.get("mesh_terms") or []):
                if isinstance(mesh, dict):
                    for field in ("descriptor", "qualifier"):
                        val = mesh.get(field, "")
                        if val and val in node_years:
                            node_years[val].append(yr)

    return {ns: sum(yrs) / len(yrs) for ns, yrs in node_years.items() if yrs}


# ── Plotly edge + node builders ───────────────────────────────────────────────

def _plotly_edge_trace(G2: nx.Graph, pos: Dict[str, Tuple[float, float]]):
    """Build a single Plotly Scatter trace for all edges."""
    import plotly.graph_objects as go

    edge_x: List[float] = []
    edge_y: List[float] = []
    for u, v in G2.edges():
        xu, yu = pos.get(str(u), (0.0, 0.0))
        xv, yv = pos.get(str(v), (0.0, 0.0))
        edge_x += [xu, xv, None]
        edge_y += [yu, yv, None]

    return go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.7, color="rgba(180,180,180,0.45)"),
        hoverinfo="none", showlegend=False,
    )


def _plotly_layout(height: int) -> "plotly.graph_objects.Layout":
    import plotly.graph_objects as go

    return go.Layout(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=110, t=10, b=10),
        height=height, hovermode="closest",
        legend=dict(x=0.01, y=0.01, bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#E5E7EB", borderwidth=1, font=dict(size=10)),
    )


# ── Public render functions ────────────────────────────────────────────────────

# A — Network Visualization ───────────────────────────────────────────────────

def render_coauthorship_network(
    G: nx.Graph,
    controls: Optional[Dict] = None,
    height: int = 750,
):
    """
    VOSviewer Network Visualization — co-authorship.
    Positions come from pre-computed kamada-kawai layout (shared with overlay/density).
    """
    if G is None or G.number_of_nodes() == 0:
        st.info("No co-authorship data available. Run the pipeline first.")
        return

    if controls is None:
        controls = _build_controls_panel("coauth")

    G2 = _apply_filters(G, controls.get("min_node", 1), controls.get("min_edge", 1), "paper_count")

    if G2.number_of_nodes() == 0:
        st.warning("All nodes filtered out — lower the minimum weight thresholds.")
        return

    clusters = len({d.get("community", 0) for _, d in G2.nodes(data=True)})
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Items", G2.number_of_nodes())
    s2.metric("Links", G2.number_of_edges())
    s3.metric("Clusters", clusters)
    s4.metric("Total link strength", sum(d.get("weight", 1) for _, _, d in G2.edges(data=True)))

    pos = _compute_layout(G2, layout_key="coauth")
    net = _get_vosviewer_net(height=f"{height}px", physics=controls.get("physics", False))
    _add_coauth_nodes(net, G2, pos, controls.get("search", ""), controls.get("labels_all", True))
    _add_edges_to_net(net, G2)
    _render_vosviewer_html(net, height=height)


def render_keyword_network(
    G: nx.Graph,
    controls: Optional[Dict] = None,
    height: int = 750,
):
    """VOSviewer Network Visualization — keyword co-occurrence."""
    if G is None or G.number_of_nodes() == 0:
        st.info("No keyword co-occurrence data available. Run the pipeline first.")
        return

    if controls is None:
        controls = _build_controls_panel("keyword")

    G2 = _apply_filters(G, controls.get("min_node", 1), controls.get("min_edge", 1), "frequency")

    if G2.number_of_nodes() == 0:
        st.warning("All nodes filtered out — lower the minimum weight thresholds.")
        return

    clusters = len({d.get("community", 0) for _, d in G2.nodes(data=True)})
    s1, s2, s3 = st.columns(3)
    s1.metric("Keywords", G2.number_of_nodes())
    s2.metric("Links", G2.number_of_edges())
    s3.metric("Clusters", clusters)

    pos = _compute_layout(G2, layout_key="keyword")
    net = _get_vosviewer_net(height=f"{height}px", physics=controls.get("physics", False))
    _add_keyword_nodes(net, G2, pos, controls.get("search", ""), controls.get("labels_all", True))
    _add_edges_to_net(net, G2)
    _render_vosviewer_html(net, height=height)


def render_topic_network(
    G: nx.Graph,
    controls: Optional[Dict] = None,
    height: int = 750,
    year_range: Optional[tuple] = None,
):
    """VOSviewer Network Visualization — topic landscape."""
    if G is None or G.number_of_nodes() == 0:
        st.info("No topic data available. Run the pipeline first.")
        return

    if controls is None:
        controls = _build_controls_panel("topic")

    G2 = _apply_filters(G, controls.get("min_node", 1), controls.get("min_edge", 1), "paper_count")

    if G2.number_of_nodes() == 0:
        st.warning("All nodes filtered out — lower the minimum weight thresholds.")
        return

    pos = _compute_layout(G2, layout_key="topic")
    net = _get_vosviewer_net(height=f"{height}px", physics=controls.get("physics", False))
    search_lower = controls.get("search", "").lower()

    for node, data in G2.nodes(data=True):
        ns          = str(node)
        label       = str(data.get("label", f"Topic {node}"))[:40]
        top_words   = data.get("top_words", [])
        size        = float(data.get("size", config.NODE_SIZE_MIN))
        color       = data.get("color", config.COMMUNITY_COLORS[0])
        paper_count = data.get("paper_count", 0)
        x, y        = pos.get(ns, (0.0, 0.0))

        is_match     = search_lower and search_lower in label.lower()
        border_color = _VOS_HIGHLIGHT if is_match else _darken(color, 0.78)

        net.add_node(
            ns, label=label, size=size,
            x=x * _LAYOUT_SCALE, y=y * _LAYOUT_SCALE,
            color={
                "background": color, "border": border_color,
                "highlight": {"background": _VOS_HIGHLIGHT, "border": "#FFA500"},
                "hover": {"background": color, "border": _darken(color, 0.65)},
            },
            title=(f"<b>{label}</b><br>Papers: {paper_count}<br>"
                   f"Top words: {', '.join(top_words[:5])}"),
            shape="dot",
            font={"size": max(10, int(size * 0.30)), "color": _VOS_FONT,
                  "face": "Open Sans", "strokeWidth": 4, "strokeColor": "#FFFFFF"},
        )

    _add_edges_to_net(net, G2)
    _render_vosviewer_html(net, height=height)


# B — Overlay Visualization ───────────────────────────────────────────────────

def render_overlay_visualization(
    G: nx.Graph,
    papers_df,
    network_type: str = "coauth",
    controls: Optional[Dict] = None,
    height: int = 750,
):
    """
    VOSviewer Overlay Visualization.

    Same positions as Network view (shared layout cache).
    Node color = average publication year mapped to Viridis:
      dark blue (oldest research) → bright yellow (most recent).
    Nodes without year data are shown in gray.
    """
    import plotly.graph_objects as go

    if G is None or G.number_of_nodes() == 0:
        st.info("No data available.")
        return

    if controls is None:
        controls = _build_controls_panel(f"ov_{network_type}")

    weight_attr = "paper_count" if network_type == "coauth" else "frequency"
    G2 = _apply_filters(G, controls.get("min_node", 1), controls.get("min_edge", 1), weight_attr)

    if G2.number_of_nodes() == 0:
        st.warning("All nodes filtered out — lower the minimum weight thresholds.")
        return

    # Compute overlay score
    overlay_vals = _overlay_year_per_node(G2, papers_df, network_type)

    # Shared layout positions
    pos = _compute_layout(G2, layout_key=network_type)

    node_strs = [str(n) for n in G2.nodes()]
    weights   = [G2.nodes[n].get(weight_attr, 1) for n in G2.nodes()]
    w_max     = max(weights) if weights else 1
    node_sizes = [max(8, 44 * (w / w_max) ** 0.5) for w in weights]

    valid_vals = [overlay_vals[ns] for ns in node_strs if ns in overlay_vals]
    v_min = min(valid_vals) if valid_vals else 2000
    v_max = max(valid_vals) if valid_vals else 2024

    # Stats bar
    no_yr = sum(1 for ns in node_strs if ns not in overlay_vals)
    s1, s2, s3 = st.columns(3)
    s1.metric("Items", G2.number_of_nodes())
    s2.metric("Year range", f"{int(v_min)} – {int(v_max)}")
    s3.metric("No year data", no_yr)

    # Edge trace (thin gray, same as network view)
    edge_trace = _plotly_edge_trace(G2, pos)

    # Gray trace — nodes without year data
    ns_no = [ns for ns in node_strs if ns not in overlay_vals]
    gray_trace = go.Scatter(
        x=[pos.get(ns, (0, 0))[0] for ns in ns_no],
        y=[pos.get(ns, (0, 0))[1] for ns in ns_no],
        mode="markers+text",
        marker=dict(
            size=[node_sizes[i] for i, ns in enumerate(node_strs) if ns not in overlay_vals],
            color="#D1D5DB",
            line=dict(width=1, color="rgba(100,100,100,0.35)"),
        ),
        text=[ns[:22] for ns in ns_no],
        textposition="bottom center",
        textfont=dict(size=8, color="#6B7280", family="Open Sans"),
        hovertext=[f"<b>{ns}</b><br>Avg year: N/A" for ns in ns_no],
        hoverinfo="text",
        name="No year data",
        showlegend=bool(ns_no),
    )

    # Colored trace — Viridis gradient by avg year
    ns_with = [ns for ns in node_strs if ns in overlay_vals]
    color_trace = go.Scatter(
        x=[pos.get(ns, (0, 0))[0] for ns in ns_with],
        y=[pos.get(ns, (0, 0))[1] for ns in ns_with],
        mode="markers+text",
        marker=dict(
            size=[node_sizes[i] for i, ns in enumerate(node_strs) if ns in overlay_vals],
            color=[overlay_vals[ns] for ns in ns_with],
            colorscale=_OVERLAY_COLORSCALE,
            cmin=v_min, cmax=v_max,
            showscale=True,
            colorbar=dict(
                title=dict(text="Avg. Publication Year", font=dict(size=11), side="right"),
                tickvals=[v_min, (v_min + v_max) / 2, v_max],
                ticktext=[str(int(v_min)), str(int((v_min + v_max) / 2)), str(int(v_max))],
                len=0.65, thickness=14, x=1.01,
                ticks="outside", tickfont=dict(size=10),
            ),
            line=dict(width=1, color="rgba(80,80,80,0.35)"),
        ),
        text=[ns[:22] for ns in ns_with],
        textposition="bottom center",
        textfont=dict(size=8, color=_VOS_FONT, family="Open Sans"),
        hovertext=[f"<b>{ns}</b><br>Avg year: {overlay_vals[ns]:.1f}" for ns in ns_with],
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(
        data=[edge_trace, gray_trace, color_trace],
        layout=_plotly_layout(height),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div style='font-size:11px;color:#6B7280;text-align:center;margin-top:-8px'>"
        "Color encodes average publication year per item (Viridis scale). &nbsp;"
        "<b style='color:#440154'>Dark blue</b> = oldest research &nbsp;·&nbsp; "
        "<b style='color:#fde725'>Bright yellow</b> = most recent research."
        "</div>",
        unsafe_allow_html=True,
    )


# C — Density Visualization ───────────────────────────────────────────────────

def render_density_visualization(
    G: nx.Graph,
    network_type: str = "coauth",
    controls: Optional[Dict] = None,
    height: int = 750,
):
    """
    VOSviewer Density Visualization.

    Gaussian KDE heatmap using the exact VOSviewer formula:
        D(x) = Σᵢ wᵢ · exp(−‖x − xᵢ‖² / 2(d̄·h)²)

    where d̄ is the average pairwise distance between items and h is the
    user-adjustable kernel width parameter.

    Color: Viridis — dark (sparse) → bright yellow (dense).
    Nodes are shown as white circles on top of the heatmap.
    Edges are hidden (matching VOSviewer density view).
    """
    import numpy as np
    import plotly.graph_objects as go

    if G is None or G.number_of_nodes() == 0:
        st.info("No data available.")
        return

    if controls is None:
        controls = _build_controls_panel(f"dn_{network_type}", show_bandwidth=True)

    weight_attr = "paper_count" if network_type == "coauth" else "frequency"
    G2 = _apply_filters(G, controls.get("min_node", 1), controls.get("min_edge", 1), weight_attr)

    if G2.number_of_nodes() == 0:
        st.warning("All nodes filtered out.")
        return

    # Shared layout
    pos = _compute_layout(G2, layout_key=network_type)

    node_strs = [str(n) for n in G2.nodes()]
    weights   = np.array([G2.nodes[n].get(weight_attr, 1) for n in G2.nodes()], dtype=float)
    w_max     = float(weights.max()) if weights.size > 0 else 1.0
    norm_w    = weights / (w_max + 1e-9)

    xs = np.array([pos.get(ns, (0.0, 0.0))[0] for ns in node_strs])
    ys = np.array([pos.get(ns, (0.0, 0.0))[1] for ns in node_strs])

    s1, s2, s3 = st.columns(3)
    s1.metric("Items", G2.number_of_nodes())
    s2.metric("Links", G2.number_of_edges())
    s3.metric("Kernel width (h)", f"{controls.get('bandwidth', 1.0):.1f}")

    if xs.size == 0:
        st.info("No layout positions computed.")
        return

    # ── KDE grid ──────────────────────────────────────────────────────────────
    grid_n = 220
    pad    = max(0.10, (xs.ptp() + ys.ptp()) * 0.10)
    x_min, x_max = float(xs.min()) - pad, float(xs.max()) + pad
    y_min, y_max = float(ys.min()) - pad, float(ys.max()) + pad

    xi = np.linspace(x_min, x_max, grid_n)
    yi = np.linspace(y_min, y_max, grid_n)
    XX, YY = np.meshgrid(xi, yi)
    ZZ = np.zeros_like(XX, dtype=float)

    # VOSviewer bandwidth: sigma = d̄ · h
    d_bar = _compute_d_bar(pos)
    h     = float(controls.get("bandwidth", 1.0))
    sigma = d_bar * h

    # D(x) = Σᵢ wᵢ · exp(−‖x−xᵢ‖² / 2σ²)
    for i, ns in enumerate(node_strs):
        px, py = pos.get(ns, (0.0, 0.0))
        ZZ += norm_w[i] * np.exp(
            -((XX - px) ** 2 + (YY - py) ** 2) / (2.0 * sigma ** 2 + 1e-12)
        )

    ZZ_norm = (ZZ - ZZ.min()) / (ZZ.max() - ZZ.min() + 1e-9)

    # ── Heatmap trace ─────────────────────────────────────────────────────────
    heatmap = go.Heatmap(
        x=xi.tolist(), y=yi.tolist(), z=ZZ_norm.tolist(),
        colorscale=_DENSITY_COLORSCALE,
        showscale=True,
        colorbar=dict(
            title=dict(text="Density", font=dict(size=11), side="right"),
            tickvals=[0, 0.5, 1],
            ticktext=["Low", "Medium", "High"],
            len=0.65, thickness=14, x=1.01,
            ticks="outside", tickfont=dict(size=10),
        ),
        hoverinfo="skip",
        zsmooth="best",
    )

    # ── Node overlay — white circles (VOSviewer style) ────────────────────────
    node_sizes_plt = [max(8, 38 * (G2.nodes[n].get(weight_attr, 1) / w_max) ** 0.5) for n in G2.nodes()]
    label_attr     = "Publications" if network_type == "coauth" else "Occurrences"

    node_overlay = go.Scatter(
        x=xs.tolist(), y=ys.tolist(),
        mode="markers+text",
        marker=dict(
            size=node_sizes_plt,
            color="rgba(255,255,255,0.85)",
            line=dict(width=1.5, color="rgba(40,40,40,0.65)"),
        ),
        text=[ns[:20] for ns in node_strs],
        textposition="bottom center",
        textfont=dict(size=8, color="#111827", family="Open Sans"),
        hovertext=[
            f"<b>{ns}</b><br>{label_attr}: {G2.nodes[n].get(weight_attr, 1)}"
            for ns, n in zip(node_strs, G2.nodes())
        ],
        hoverinfo="text",
        showlegend=False,
    )

    layout = _plotly_layout(height)
    layout.xaxis.update(range=[x_min, x_max])
    layout.yaxis.update(range=[y_min, y_max])

    fig = go.Figure(data=[heatmap, node_overlay], layout=layout)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div style='font-size:11px;color:#6B7280;text-align:center;margin-top:-8px'>"
        "Research activity density — Gaussian KDE weighted by publication count "
        f"(σ = d̄·h = {sigma:.3f}). &nbsp;"
        "<b style='color:#440154'>Dark</b> = sparse &nbsp;·&nbsp; "
        "<b style='color:#fde725'>Bright yellow</b> = high concentration."
        "</div>",
        unsafe_allow_html=True,
    )


# ── Network statistics panel ──────────────────────────────────────────────────

def render_network_stats(stats: Dict[str, Any], title: str = "Network Statistics"):
    with st.expander(title, expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nodes",       stats.get("node_count", 0))
            st.metric("Edges",       stats.get("edge_count", 0))
            st.metric("Communities", stats.get("num_communities", 0))
        with col2:
            st.metric("Density",    f"{stats.get('density', 0):.4f}")
            st.metric("Modularity", f"{stats.get('modularity', 0):.4f}")
            st.metric("Avg Clust.", f"{stats.get('avg_clustering_coefficient', 0):.4f}")

        top_deg = stats.get("top10_degree_centrality", [])
        if top_deg:
            st.markdown("**Top nodes by degree:**")
            for node, score in top_deg[:5]:
                st.markdown(
                    f"<span style='color:{config.PRIMARY_ACCENT}'>{node}</span>"
                    f" — {score:.3f}",
                    unsafe_allow_html=True,
                )
