"""
network_viz.py — VOSviewer-accurate bibliometric network visualizations.

Three visualization modes matching VOSviewer 1.6.18+:

  Network Visualization  — cluster-colored nodes, UMAP+ForceAtlas2 layout,
                           interactive (drag/zoom), colored edges.

  Overlay Visualization  — same positions, Viridis color gradient mapped
                           to each node's average publication year score.

  Density Visualization  — Gaussian KDE heatmap using the exact VOSviewer
                           formula: D(x) = Σ wᵢ·K(‖x−xᵢ‖ / (d̄·h))
                           with Viridis colorscale, white node circles on top.

All three modes share the same pre-computed UMAP+ForceAtlas2 layout so
positions are consistent when switching between modes — identical to the
way VOSviewer keeps positions stable across its Network/Overlay/Density tabs.
"""

import colorsys
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
from visualization.layout_engine import compute_vos_layout

logger = logging.getLogger(__name__)


# ── VOSviewer visual constants ────────────────────────────────────────────────
_VOS_BG        = "#0A0F1E"   # dark canvas matching app theme
_VOS_FONT      = "#F9FAFB"   # near-white text for dark background
_VOS_HIGHLIGHT = "#FFD700"   # gold highlight on selection / search
_LAYOUT_SCALE  = 1400.0      # NetworkX [-1,1] → PyVis pixel space

# Hard render caps — beyond these numbers vis.js freezes the browser
_MAX_RENDER_NODES = 300
_MAX_RENDER_EDGES = 1500

# VOSviewer blue→teal→green→yellow overlay colorscale (publication year)
_OVERLAY_COLORSCALE = [
    [0.00, "#1a0050"],
    [0.25, "#0d4f8b"],
    [0.50, "#00897b"],
    [0.75, "#66bb6a"],
    [1.00, "#ffee58"],
]

# VOSviewer dark density colorscale (sparse→dense research areas)
_DENSITY_COLORSCALE = [
    [0.00, "#000033"],
    [0.14, "#000066"],
    [0.28, "#003366"],
    [0.43, "#006666"],
    [0.57, "#009966"],
    [0.71, "#66cc00"],
    [0.85, "#cccc00"],
    [1.00, "#ffff00"],
]

# Dark-canvas CSS injected into every PyVis HTML frame
_VOS_CSS = """
<style>
  body { background: #0A0F1E !important; margin: 0; }
  #mynetwork {
    background-color: #0A0F1E !important;
    border: 1px solid #1F2937;
    border-radius: 8px;
    box-shadow: 0 2px 24px rgba(0,0,0,0.6);
  }
  canvas { background: #0A0F1E !important; }
  .vis-tooltip {
    background-color: #1C2539 !important;
    color: #F9FAFB !important;
    border: 1px solid #374151 !important;
    border-radius: 6px !important;
    font-family: 'Open Sans', sans-serif !important;
    font-size: 12px !important;
    padding: 8px 10px !important;
    max-width: 300px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5) !important;
  }
  .vis-navigation .vis-button {
    background-color: #1C2539 !important;
    border: 1px solid #374151 !important;
    border-radius: 4px !important;
  }
  .vis-navigation .vis-button:hover {
    background-color: #374151 !important;
  }
</style>
"""

# Barnes-Hut physics used only when the user enables the Physics toggle.
# Low centralGravity (0.1) prevents the spherical collapse; high springLength
# (200) and low springConstant (0.02) let clusters spread organically.
_VOS_PHYSICS_ON = {
    "barnesHut": {
        "gravitationalConstant": -12000,
        "centralGravity": 0.1,
        "springLength": 200,
        "springConstant": 0.02,
        "damping": 0.15,
        "avoidOverlap": 0.8,
    },
    "minVelocity": 0.5,
    "solver": "barnesHut",
    "stabilization": {"enabled": True, "iterations": 2000, "updateInterval": 50, "fit": True},
}


# ── Community color palette ───────────────────────────────────────────────────

def _community_color_palette(n: int) -> List[str]:
    """
    Return n visually distinct colors.

    Up to 30 communities: use the hand-curated config.COMMUNITY_COLORS list.
    Beyond 30: generate via golden-ratio hue spacing so adjacent community
    IDs are never similar hues — identical to how VOSviewer handles 38+ clusters.
    """
    if n <= 0:
        return [config.COMMUNITY_COLORS[0]]
    if n <= len(config.COMMUNITY_COLORS):
        return list(config.COMMUNITY_COLORS[:n])
    golden = 0.618033988749895
    result: List[str] = []
    h = 0.0
    for i in range(n):
        h = (h + golden) % 1.0
        lightness = 0.55 if i % 2 == 0 else 0.65
        r, g, b = colorsys.hls_to_rgb(h, lightness, 0.70)
        result.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return result


# ── Layout computation (shared across all three modes) ────────────────────────

def _compute_layout(
    G: nx.Graph,
    layout_key: str = "",
    network_type: str = "",
    physics: bool = False,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute and cache the VOSviewer-style layout for graph G.

    physics=False (default): UMAP+ForceAtlas2 pre-computed layout — stable,
      consistent positions across Network/Overlay/Density tabs.
    physics=True: NetworkX spring layout — organic, force-directed positions;
      re-computed each toggle so the graph "settles" visually differently.

    Returns {str(node): (x, y)} with values in approximately [-1, 1].
    """
    n = G.number_of_nodes()
    if n == 0:
        return {}

    # Cache key includes physics state so toggling physics invalidates cache
    physics_tag = "phy1" if physics else "phy0"
    ss_key = f"__pos_{layout_key}_{network_type}_{n}_{G.number_of_edges()}_{physics_tag}"
    cached = st.session_state.get(ss_key)
    if cached is not None:
        return cached

    if physics:
        # Spring layout: organic force-directed, fast on graphs ≤ 300 nodes
        raw = nx.spring_layout(G, weight="weight", seed=42, iterations=80, k=1.5 / max(n ** 0.5, 1))
        pos = {str(node): (float(xy[0]), float(xy[1])) for node, xy in raw.items()}
    else:
        # Pull embeddings + metadata from session state (set by the pipeline)
        embeddings_array = st.session_state.get("embeddings_array")
        papers_df        = st.session_state.get("papers_df")
        doc_topics_df    = st.session_state.get("doc_topics_df")
        query            = st.session_state.get("current_query", "")

        pos = compute_vos_layout(
            G,
            network_type=network_type or layout_key,
            papers_df=papers_df,
            embeddings_array=embeddings_array,
            doc_topics_df=doc_topics_df,
            query=query,
        )

    st.session_state[ss_key] = pos
    return pos


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
            "zoomView": True, "dragNodes": False, "dragView": True,
        },
        "layout": {"improvedLayout": False},
    }
    if physics:
        opts["physics"] = _VOS_PHYSICS_ON
    else:
        opts["physics"] = {"enabled": False}

    net.set_options(json.dumps(opts))
    return net


def _render_vosviewer_html(net, height: int, cache_key: str = ""):
    """Render PyVis HTML, caching the generated string in session_state."""
    html: Optional[str] = None
    if cache_key:
        html = st.session_state.get(f"__html_{cache_key}")

    if html is None:
        with tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8"
        ) as f:
            net.save_graph(f.name)
            path = f.name
        with open(path, "r", encoding="utf-8") as f:
            html = f.read()
        os.unlink(path)
        html = html.replace("</head>", _VOS_CSS + "</head>", 1)
        if cache_key:
            st.session_state[f"__html_{cache_key}"] = html

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
    keep_isolates: bool = False,
) -> nx.Graph:
    G2 = G.copy()
    G2.remove_nodes_from([
        n for n, d in G2.nodes(data=True)
        if d.get(weight_attr, d.get("frequency", d.get("paper_count", 1))) < min_node_weight
    ])
    G2.remove_edges_from([
        (u, v) for u, v, d in G2.edges(data=True) if d.get("weight", 1) < min_edge_weight
    ])
    # Only remove isolates when the user explicitly raises the minimum edge threshold;
    # with the default (min_edge_weight=1) we keep isolated nodes so that topic
    # landscape nodes remain visible even when they share no papers.
    if not keep_isolates and min_edge_weight > 1:
        G2.remove_nodes_from(list(nx.isolates(G2)))
    return G2


def _trim_for_render(
    G: nx.Graph, weight_attr: str = "paper_count"
) -> Tuple[nx.Graph, bool]:
    """
    Return a subgraph capped at _MAX_RENDER_NODES (top nodes by weight).
    Second return value is True if the graph was trimmed.
    """
    n = G.number_of_nodes()
    if n <= _MAX_RENDER_NODES:
        return G, False
    top_nodes = sorted(
        G.nodes(data=True),
        key=lambda nd: nd[1].get(weight_attr, nd[1].get("frequency", 1)),
        reverse=True,
    )[:_MAX_RENDER_NODES]
    keep = {nd[0] for nd in top_nodes}
    return G.subgraph(keep).copy(), True


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


def _add_edges_to_net(net, G: nx.Graph) -> int:
    """Add edges to PyVis network, capped at _MAX_RENDER_EDGES (top by weight)."""
    all_edges = sorted(
        G.edges(data=True),
        key=lambda e: e[2].get("weight", 1),
        reverse=True,
    )[:_MAX_RENDER_EDGES]
    for u, v, data in all_edges:
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
    return len(all_edges)


# ── Overlay helper — average year per node ───────────────────────────────────

def _overlay_year_per_node(
    G2: nx.Graph, papers_df, network_type: str
) -> Dict[str, float]:
    """
    Compute average publication year for each node.
    coauth  → year averaged over each author's papers.
    keyword → year averaged over papers containing the keyword.
    topic   → year averaged over papers assigned to each topic (via doc_topics_df).
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

    if network_type == "topic":
        # Build pmid→year mapping first, then join through doc_topics_df
        pmid_year: Dict[str, int] = {}
        for _, row in papers_df.iterrows():
            yr = _year(row)
            if yr:
                pmid_year[str(row.get("pmid", ""))] = yr
        doc_topics_df = st.session_state.get("doc_topics_df")
        if doc_topics_df is not None and not doc_topics_df.empty:
            for _, tr in doc_topics_df.iterrows():
                tid = str(int(tr.get("topic_id", -1)))
                if tid not in node_years:
                    continue
                yr2 = pmid_year.get(str(tr.get("pmid", "")))
                if yr2:
                    node_years[tid].append(yr2)
        return {ns: sum(yrs) / len(yrs) for ns, yrs in node_years.items() if yrs}

    for _, row in papers_df.iterrows():
        yr = _year(row)
        if not yr:
            continue

        if network_type == "coauth":
            for author in (row.get("authors") or []):
                name = author["name"] if isinstance(author, dict) else str(author)
                if name in node_years:
                    node_years[name].append(yr)
        else:  # keyword
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
    """Build a single Plotly Scatter trace for edges (capped at _MAX_RENDER_EDGES)."""
    import plotly.graph_objects as go

    edges = sorted(G2.edges(data=True), key=lambda e: e[2].get("weight", 1), reverse=True)[:_MAX_RENDER_EDGES]
    edge_x: List[float] = []
    edge_y: List[float] = []
    for u, v, _ in edges:
        xu, yu = pos.get(str(u), (0.0, 0.0))
        xv, yv = pos.get(str(v), (0.0, 0.0))
        edge_x += [xu, xv, None]
        edge_y += [yu, yv, None]

    return go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=0.9, color="rgba(100,130,180,0.38)"),
        hoverinfo="none", showlegend=False,
    )


def _plotly_layout(height: int) -> "plotly.graph_objects.Layout":
    import plotly.graph_objects as go

    return go.Layout(
        paper_bgcolor=_VOS_BG, plot_bgcolor=_VOS_BG,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=120, t=10, b=10),
        height=height, hovermode="closest",
        legend=dict(
            x=1.01, y=1,
            bgcolor="rgba(10,15,30,0.92)",
            bordercolor="#374151", borderwidth=1,
            font=dict(size=10, color=_VOS_FONT),
        ),
    )


def _render_network_plotly(
    G: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    weight_attr: str,
    controls: Dict[str, Any],
    height: int,
) -> None:
    """
    Render the Network view using Plotly Canvas (never freezes browser).

    One Scatter trace per community for the cluster legend.
    A separate text-only trace carries per-node label sizes and colors,
    showing labels only for nodes above the 25th-percentile weight
    — matching VOSviewer's label density behaviour.
    """
    import plotly.graph_objects as go

    search_lower = controls.get("search", "").lower().strip()
    labels_all   = controls.get("labels_all", True)

    edge_trace = _plotly_edge_trace(G, pos)

    all_weights = [d.get(weight_attr, d.get("frequency", 1)) for _, d in G.nodes(data=True)]
    if not all_weights:
        return
    w_min = min(all_weights)
    w_max = max(all_weights) if max(all_weights) > w_min else w_min + 1
    sorted_w = sorted(all_weights)
    p25 = sorted_w[max(0, len(sorted_w) // 4)]   # 25th-percentile threshold

    communities = sorted({d.get("community", 0) for _, d in G.nodes(data=True)})
    palette = _community_color_palette(len(communities))
    # Map community id → color using the sorted position (not raw id) so
    # the first community always gets the first palette colour regardless of
    # what integer Louvain happened to assign.
    comm_to_color = {cid: palette[i % len(palette)] for i, cid in enumerate(communities)}

    node_traces = []
    for cid in communities:
        comm_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("community", 0) == cid]
        if not comm_nodes:
            continue

        color = comm_to_color[cid]
        xs, ys, sizes, hovers, borders_c, borders_w = [], [], [], [], [], []

        for n, d in comm_nodes:
            ns  = str(n)
            x, y = pos.get(ns, (0.0, 0.0))
            w    = d.get(weight_attr, d.get("frequency", 1))
            # Re-apply sqrt scaling formula with updated NODE_SIZE_MAX (80)
            norm = (w - w_min) / (w_max - w_min + 1e-9)
            size = config.NODE_SIZE_MIN + (norm ** 0.5) * (config.NODE_SIZE_MAX - config.NODE_SIZE_MIN)
            is_match = bool(search_lower and search_lower in ns.lower())

            xs.append(x); ys.append(y); sizes.append(size)
            borders_c.append(_VOS_HIGHLIGHT if is_match else _darken(color, 0.78))
            borders_w.append(3 if is_match else 1)

            neighbors = sorted(
                [(str(nb), G[n][nb].get("weight", 1)) for nb in G.neighbors(n)],
                key=lambda e: e[1], reverse=True,
            )[:5]
            nb_html = "<br>".join(f"  • {nb} ({w2})" for nb, w2 in neighbors)
            hovers.append(
                f"<b>{ns}</b><br>Papers:&nbsp;{w}<br>Cluster:&nbsp;{cid + 1}"
                f"<br>Top connections:<br>{nb_html or '—'}"
            )

        node_traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers",
            marker=dict(
                size=sizes, color=color, opacity=0.92,
                line=dict(width=borders_w, color=borders_c),
            ),
            hovertext=hovers,
            hoverinfo="text",
            name=f"Cluster {cid + 1}",
        ))

    # ── Per-node label trace ──────────────────────────────────────────────────
    # Labels have per-node font sizes and colors (same as node color) — this
    # cannot be done inside a community trace, so a separate text-only trace
    # carries them.  Sizes range 8–24px via sqrt scaling; only nodes above the
    # 25th-percentile weight get a label (prevents overcrowding).
    if labels_all or search_lower:
        lx, ly, ltexts, lcolors, lsizes = [], [], [], [], []
        for n, d in G.nodes(data=True):
            ns       = str(n)
            w        = d.get(weight_attr, d.get("frequency", 1))
            is_match = bool(search_lower and search_lower in ns.lower())
            # Only label: search hits always; otherwise only above p25 when labels on
            if not (is_match or (labels_all and w >= p25)):
                continue
            norm      = (w - w_min) / (w_max - w_min + 1e-9)
            lsize     = max(8, int(8 + (norm ** 0.5) * 16))   # 8–24 px
            cid       = d.get("community", 0)
            lcolor    = comm_to_color.get(cid, palette[0])
            x, y      = pos.get(ns, (0.0, 0.0))
            lx.append(x); ly.append(y)
            ltexts.append(ns[:24])
            lcolors.append(lcolor)
            lsizes.append(lsize)

        if lx:
            node_traces.append(go.Scatter(
                x=lx, y=ly,
                mode="text",
                text=ltexts,
                textfont=dict(size=lsizes, color=lcolors, family="Open Sans"),
                textposition="bottom center",
                hoverinfo="none",
                showlegend=False,
            ))

    layout = _plotly_layout(height)
    layout.update(
        showlegend=len(communities) <= 25,
        legend=dict(
            x=1.01, y=1,
            bgcolor="rgba(10,15,30,0.92)",
            bordercolor="#374151", borderwidth=1,
            font=dict(size=9, color=_VOS_FONT),
            itemsizing="constant",
        ),
    )
    fig = go.Figure(data=[edge_trace] + node_traces, layout=layout)
    st.plotly_chart(fig, width="stretch")


# ── Public render functions ────────────────────────────────────────────────────

# A — Network Visualization ───────────────────────────────────────────────────

def render_coauthorship_network(
    G: nx.Graph,
    controls: Optional[Dict] = None,
    height: int = 750,
):
    """
    Co-authorship Network view — rendered with Plotly (not vis.js).
    Plotly uses Canvas rendering so it never freezes the browser.
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

    total_nodes = G2.number_of_nodes()
    total_edges = G2.number_of_edges()
    G2, trimmed = _trim_for_render(G2, "paper_count")
    if trimmed:
        st.info(
            f"Showing top {_MAX_RENDER_NODES} authors of {total_nodes:,}. "
            "Raise Min node weight to focus on more-published authors."
        )

    clusters = len({d.get("community", 0) for _, d in G2.nodes(data=True)})
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Items", G2.number_of_nodes())
    s2.metric("Links", f"{min(total_edges, _MAX_RENDER_EDGES):,}" + (f" of {total_edges:,}" if total_edges > _MAX_RENDER_EDGES else ""))
    s3.metric("Clusters", clusters)
    s4.metric("Total link strength", sum(d.get("weight", 1) for _, _, d in G2.edges(data=True)))

    pos = _compute_layout(G2, layout_key="coauth", network_type="coauth",
                          physics=controls.get("physics", False))
    _render_network_plotly(G2, pos, "paper_count", controls, height)


def render_keyword_network(
    G: nx.Graph,
    controls: Optional[Dict] = None,
    height: int = 750,
):
    """
    Keyword co-occurrence Network view — rendered with Plotly (not vis.js).
    Plotly uses Canvas rendering so it never freezes the browser.
    """
    if G is None or G.number_of_nodes() == 0:
        st.info("No keyword co-occurrence data available. Run the pipeline first.")
        return

    if controls is None:
        controls = _build_controls_panel("keyword")

    G2 = _apply_filters(G, controls.get("min_node", 1), controls.get("min_edge", 1), "frequency")

    if G2.number_of_nodes() == 0:
        st.warning("All nodes filtered out — lower the minimum weight thresholds.")
        return

    total_nodes = G2.number_of_nodes()
    total_edges = G2.number_of_edges()
    G2, trimmed = _trim_for_render(G2, "frequency")
    if trimmed:
        st.info(
            f"Showing top {_MAX_RENDER_NODES} keywords of {total_nodes:,}. "
            "Raise Min node weight to narrow the graph."
        )

    clusters = len({d.get("community", 0) for _, d in G2.nodes(data=True)})
    s1, s2, s3 = st.columns(3)
    s1.metric("Keywords", G2.number_of_nodes())
    s2.metric("Links", f"{min(total_edges, _MAX_RENDER_EDGES):,}" + (f" of {total_edges:,}" if total_edges > _MAX_RENDER_EDGES else ""))
    s3.metric("Clusters", clusters)

    pos = _compute_layout(G2, layout_key="keyword", network_type="keyword",
                          physics=controls.get("physics", False))
    _render_network_plotly(G2, pos, "frequency", controls, height)


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

    pos = _compute_layout(G2, layout_key="topic", network_type="topic",
                          physics=controls.get("physics", False))
    net = _get_vosviewer_net(height=f"{height}px", physics=controls.get("physics", False))
    search_lower = controls.get("search", "").lower()
    labels_all   = controls.get("labels_all", True)

    for node, data in G2.nodes(data=True):
        ns          = str(node)
        label       = str(data.get("label", f"Topic {node}"))[:40]
        top_words   = data.get("top_words", [])
        size        = float(data.get("size", config.NODE_SIZE_MIN))
        color       = data.get("color", config.COMMUNITY_COLORS[0])
        paper_count = data.get("paper_count", 0)
        x, y        = pos.get(ns, (0.0, 0.0))

        is_match     = bool(search_lower and search_lower in label.lower())
        border_color = _VOS_HIGHLIGHT if is_match else _darken(color, 0.78)
        # Respect Labels toggle: OFF hides all labels except search hits
        label_text   = label if (labels_all or is_match) else ""

        net.add_node(
            ns, label=label_text, size=size,
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
    # Cache key includes labels_all and physics so toggling either regenerates HTML
    html_key = (f"topic_{G2.number_of_nodes()}_{G2.number_of_edges()}_"
                f"{controls.get('search','')}_phy{controls.get('physics')}_"
                f"lbl{controls.get('labels_all', True)}")
    _render_vosviewer_html(net, height=height, cache_key=html_key)


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

    # "keyword" nodes use "frequency"; coauth and topic nodes use "paper_count"
    weight_attr = "frequency" if network_type == "keyword" else "paper_count"
    G2 = _apply_filters(G, controls.get("min_node", 1), controls.get("min_edge", 1), weight_attr)

    if G2.number_of_nodes() == 0:
        st.warning("All nodes filtered out — lower the minimum weight thresholds.")
        return

    # Compute overlay score
    overlay_vals = _overlay_year_per_node(G2, papers_df, network_type)

    # Shared layout positions (same physics state as Network view for this network_type)
    pos = _compute_layout(G2, layout_key=network_type, network_type=network_type,
                          physics=controls.get("physics", False))

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

    # Edge trace (same positions as Network view)
    edge_trace = _plotly_edge_trace(G2, pos)

    # Gray trace — nodes without year data shown as dim dots
    ns_no = [ns for ns in node_strs if ns not in overlay_vals]
    gray_trace = go.Scatter(
        x=[pos.get(ns, (0, 0))[0] for ns in ns_no],
        y=[pos.get(ns, (0, 0))[1] for ns in ns_no],
        mode="markers+text",
        marker=dict(
            size=[node_sizes[i] for i, ns in enumerate(node_strs) if ns not in overlay_vals],
            color="#374151",
            line=dict(width=1, color="rgba(100,100,100,0.30)"),
        ),
        text=[ns[:22] for ns in ns_no],
        textposition="bottom center",
        textfont=dict(size=8, color="#6B7280", family="Open Sans"),
        hovertext=[f"<b>{ns}</b><br>Avg year: N/A" for ns in ns_no],
        hoverinfo="text",
        name="No year data",
        showlegend=bool(ns_no),
    )

    # Colored trace — blue→teal→green→yellow gradient by avg publication year
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
                title=dict(text="Avg. Publication Year", font=dict(size=11, color=_VOS_FONT), side="right"),
                tickvals=[v_min, (v_min + v_max) / 2, v_max],
                ticktext=[str(int(v_min)), str(int((v_min + v_max) / 2)), str(int(v_max))],
                len=0.65, thickness=14, x=1.01,
                ticks="outside",
                tickfont=dict(size=10, color=_VOS_FONT),
                bgcolor="rgba(10,15,30,0.8)",
                bordercolor="#374151",
            ),
            line=dict(width=1, color="rgba(255,255,255,0.20)"),
            opacity=0.92,
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
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        f"<div style='font-size:11px;color:#9CA3AF;text-align:center;margin-top:-8px'>"
        "Color encodes average publication year per item. &nbsp;"
        "<b style='color:#1a0050'>Deep purple</b> = oldest &nbsp;·&nbsp;"
        "<b style='color:#00897b'>Teal</b> = mid-period &nbsp;·&nbsp;"
        "<b style='color:#ffee58'>Yellow</b> = most recent research."
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

    # "keyword" nodes use "frequency"; coauth and topic nodes use "paper_count"
    weight_attr = "frequency" if network_type == "keyword" else "paper_count"
    G2 = _apply_filters(G, controls.get("min_node", 1), controls.get("min_edge", 1), weight_attr)

    if G2.number_of_nodes() == 0:
        st.warning("All nodes filtered out.")
        return

    # Shared layout (same physics state as Network view for this network_type)
    pos = _compute_layout(G2, layout_key=network_type, network_type=network_type,
                          physics=controls.get("physics", False))

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

    # ── Heatmap trace — VOSviewer dark density colorscale ─────────────────────
    heatmap = go.Heatmap(
        x=xi.tolist(), y=yi.tolist(), z=ZZ_norm.tolist(),
        colorscale=_DENSITY_COLORSCALE,
        showscale=True,
        colorbar=dict(
            title=dict(text="Research Density", font=dict(size=11, color=_VOS_FONT), side="right"),
            tickvals=[0, 0.5, 1],
            ticktext=["Sparse", "Medium", "Dense"],
            len=0.65, thickness=14, x=1.01,
            ticks="outside",
            tickfont=dict(size=10, color=_VOS_FONT),
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="#374151",
        ),
        hoverinfo="skip",
        zsmooth="best",
    )

    # ── Node overlay — near-transparent dots + white floating labels ──────────
    # VOSviewer density view: nodes become nearly invisible so the heatmap
    # dominates; only labels float above the density landscape.
    node_sizes_plt = [
        max(6, int(config.NODE_SIZE_MIN + ((G2.nodes[n].get(weight_attr, 1) / w_max) ** 0.5)
                   * (config.NODE_SIZE_MAX - config.NODE_SIZE_MIN) * 0.60))
        for n in G2.nodes()
    ]
    label_attr = "Occurrences" if network_type == "keyword" else "Publications"

    node_overlay = go.Scatter(
        x=xs.tolist(), y=ys.tolist(),
        mode="markers",
        marker=dict(
            size=node_sizes_plt,
            color="rgba(255,255,255,0.12)",   # nearly transparent — heatmap dominates
            line=dict(width=1.0, color="rgba(255,255,255,0.25)"),
        ),
        hovertext=[
            f"<b>{ns}</b><br>{label_attr}: {G2.nodes[n].get(weight_attr, 1)}"
            for ns, n in zip(node_strs, G2.nodes())
        ],
        hoverinfo="text",
        showlegend=False,
    )

    # White floating label trace — scaled by weight, always visible
    lx2, ly2, ltexts2, lsizes2 = [], [], [], []
    for ns, n in zip(node_strs, G2.nodes()):
        w_n   = G2.nodes[n].get(weight_attr, 1)
        norm  = (w_n / w_max) ** 0.5
        lsize = max(7, int(7 + norm * 15))   # 7–22 px
        x_n, y_n = pos.get(ns, (0.0, 0.0))
        lx2.append(x_n); ly2.append(y_n)
        ltexts2.append(ns[:22])
        lsizes2.append(lsize)

    label_trace = go.Scatter(
        x=lx2, y=ly2,
        mode="text",
        text=ltexts2,
        textfont=dict(size=lsizes2, color="#FFFFFF", family="Open Sans"),
        textposition="bottom center",
        hoverinfo="none",
        showlegend=False,
    )

    layout = _plotly_layout(height)
    # Override to pure black for density view — matches VOSviewer Image 6
    layout.paper_bgcolor = "#000000"
    layout.plot_bgcolor  = "#000000"
    layout.xaxis.update(range=[x_min, x_max])
    layout.yaxis.update(range=[y_min, y_max])

    fig = go.Figure(data=[heatmap, node_overlay, label_trace], layout=layout)
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        f"<div style='font-size:11px;color:#9CA3AF;text-align:center;margin-top:-8px'>"
        "Research activity density — Gaussian KDE weighted by publication count "
        f"(σ = d̄·h = {sigma:.3f}). &nbsp;"
        "<b style='color:#003366'>Dark blue</b> = sparse &nbsp;·&nbsp; "
        "<b style='color:#66cc00'>Green</b> = moderate &nbsp;·&nbsp; "
        "<b style='color:#ffff00'>Bright yellow</b> = high concentration."
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
