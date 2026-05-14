"""
graph_viz.py — VOSviewer-style knowledge graph visualization.

Renders a directed biomedical entity-relationship graph with:
- Node color by entity type
- Edge color by relationship type
- Gap nodes highlighted with pulsing yellow border
- Full interactive controls (hover, click, search, filter)
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import streamlit as st

import config
from visualization.layout_engine import compute_vos_layout

logger = logging.getLogger(__name__)

# Entity type → node color (Problem 2)
_ENTITY_COLOR_MAP: Dict[str, str] = {
    "DISEASE":              "#FF5252",
    "CANCER":               "#FF5252",
    "CHEMICAL":             "#00D4FF",
    "SIMPLE_CHEMICAL":      "#00D4FF",
    "GENE_OR_GENE_PRODUCT": "#00E676",
    "GENE_OR_GENOME":       "#00E676",   # backward compat
    "PROTEIN":              "#00E676",
    "DNA":                  "#00E676",
    "RNA":                  "#00E676",
    "ORGANISM":             "#FFD600",
    "TAXON":                "#FFD600",
    "CELL_TYPE":            "#7B61FF",
    "CELL_LINE":            "#7B61FF",
    "CELL":                 "#7B61FF",   # backward compat
    "ANATOMY":              "#FF8B94",
    "PATHWAY":              "#A8E6CF",
    "BIOLOGICAL_PROCESS":   "#A8E6CF",   # backward compat
    "LABORATORY_PROCEDURE": "#8899AA",
}
_DEFAULT_KG_NODE_COLOR = "#8899AA"

# ── Entity type → display shape ───────────────────────────────────────────────
_ENTITY_SHAPE = {
    "DISEASE":              "dot",
    "CANCER":               "dot",
    "GENE_OR_GENOME":       "diamond",
    "DNA":                  "diamond",
    "RNA":                  "diamond",
    "PROTEIN":              "diamond",
    "CHEMICAL":             "square",
    "BIOLOGICAL_PROCESS":   "ellipse",
    "PATHWAY":              "ellipse",
    "CELL":                 "triangle",
    "CELL_TYPE":            "triangle",
    "CELL_LINE":            "triangle",
    "ORGANISM":             "star",
    "ANATOMY":              "triangleDown",
    "LABORATORY_PROCEDURE": "triangleDown",
}

_EDGE_COLORS = {
    "inhibit":             "#EF4444",
    "activate":            "#10B981",
    "regulate":            "#F59E0B",
    "express":             "#3B82F6",
    "bind":                "#8B5CF6",
    "treat":               "#06B6D4",
    "cause":               "#F97316",
    "associate":           "#EC4899",
    "semantic_similarity": "#9CA3AF",
    "co-occurs with":      "#374151",
}
_DEFAULT_EDGE_COLOR = "#4B5563"

_SHARED_STYLE = """
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
    max-width: 300px !important;
  }
</style>
"""

# Pulsing border animation injected for gap nodes
_GAP_NODE_CSS = """
<style>
@keyframes pulse-border {
  0%   { box-shadow: 0 0 0 0   rgba(255, 215, 0, 0.7); }
  70%  { box-shadow: 0 0 0 10px rgba(255, 215, 0, 0.0); }
  100% { box-shadow: 0 0 0 0   rgba(255, 215, 0, 0.0); }
}
.gap-node { animation: pulse-border 1.5s infinite; }
</style>
"""


def _color_opacity(hex_color: str, opacity: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{opacity})"


_DEFAULT_MAX_KG_NODES = 150  # default node cap; overrideable via render param
_MAX_KG_EDGES = 500          # cap entity KG edges sent to vis.js

# Scale factor applied to normalised [-1,1] positions when injecting into PyVis
_KG_LAYOUT_SCALE = 2000


def _get_pyvis_net(height: str = "800px"):
    from pyvis.network import Network
    net = Network(
        height=height,
        width="100%",
        bgcolor=config.CANVAS_BACKGROUND,
        font_color=config.TEXT_PRIMARY,
        directed=True,
        notebook=False,
    )
    net.set_options(json.dumps({
        # Physics fully disabled — positions are pre-computed by UMAP + ForceAtlas2
        "physics": {"enabled": False},
        "nodes": {
            "font": {"face": "Open Sans", "color": config.TEXT_PRIMARY},
            "borderWidth": 1.5,
            "borderWidthSelected": 4,
        },
        "edges": {
            "smooth": {"type": "curvedCW", "roundness": 0.15},
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 80,
            "navigationButtons": True,
            "keyboard": True,
            "zoomView": True,
            "dragView": True,
            "dragNodes": False,
            "multiselect": False,
            "selectConnectedEdges": True,
        },
    }))
    return net


def _compute_kg_layout(G_sub: nx.MultiDiGraph) -> Dict[str, Tuple[float, float]]:
    """
    Compute and cache UMAP + ForceAtlas2 layout for the knowledge graph subgraph.
    Positions returned in ~[-1, 1] range.
    """
    n = G_sub.number_of_nodes()
    ss_key = f"__kgpos_{n}_{G_sub.number_of_edges()}"
    cached = st.session_state.get(ss_key)
    if cached is not None:
        return cached

    papers_df     = st.session_state.get("papers_df")
    embeddings    = st.session_state.get("embeddings_array")
    entities_df   = st.session_state.get("entities_df")

    pos = compute_vos_layout(
        G_sub,
        network_type="kg",
        papers_df=papers_df,
        embeddings_array=embeddings,
        entities_df=entities_df,
    )
    st.session_state[ss_key] = pos
    return pos


def _render_html(net, height: int, extra_css: str = "", cache_key: str = ""):
    """Render PyVis HTML, caching in session_state to speed up page navigation."""
    from typing import Optional
    html: Optional[str] = None
    if cache_key:
        html = st.session_state.get(f"__kghtml_{cache_key}")

    if html is None:
        with tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, mode="w", encoding="utf-8"
        ) as f:
            net.save_graph(f.name)
            html_path = f.name
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        os.unlink(html_path)
        html = html.replace("</head>", _SHARED_STYLE + extra_css + "</head>", 1)
        if cache_key:
            st.session_state[f"__kghtml_{cache_key}"] = html

    st.components.v1.html(html, height=height + 30, scrolling=False)


def render_knowledge_graph(
    G: nx.MultiDiGraph,
    highlight_entity: Optional[str] = None,
    entity_type_filter: Optional[Set[str]] = None,
    relationship_type_filter: Optional[Set[str]] = None,
    min_evidence: int = 1,
    show_gap_nodes: bool = True,
    search_term: str = "",
    height: int = 800,
    max_nodes: int = _DEFAULT_MAX_KG_NODES,
    min_edge_cooccurrence: int = 2,
):
    """
    Render the full biomedical knowledge graph in VOSviewer style.

    Node color = entity type | Edge direction = relationship direction
    Gap nodes highlighted with pulsing yellow border.
    """
    if G is None or G.number_of_nodes() == 0:
        st.info(
            "Knowledge graph is empty. Run the full pipeline (including NLP processing) first."
        )
        return

    # ── Filter graph ──────────────────────────────────────────────────────────
    allowed_entities = entity_type_filter or set(config.ENTITY_TYPE_COLORS.keys())
    nodes_to_keep = [
        n for n, d in G.nodes(data=True)
        if d.get("entity_type", "UNKNOWN") in allowed_entities
        and d.get("paper_count", 0) >= min_evidence
    ]
    G_sub = G.subgraph(nodes_to_keep).copy()

    # Filter edges by relationship type
    if relationship_type_filter:
        edges_to_remove = [
            (u, v, k)
            for u, v, k, d in G_sub.edges(data=True, keys=True)
            if d.get("relationship_type", "") not in relationship_type_filter
        ]
        G_sub.remove_edges_from(edges_to_remove)

    if G_sub.number_of_nodes() == 0:
        st.warning("No nodes match the current filters.")
        return

    # ── Node cap: keep top-N by paper_count to prevent vis.js freeze ─────────
    total_kg_nodes = G_sub.number_of_nodes()
    if total_kg_nodes > max_nodes:
        top_nodes = sorted(
            G_sub.nodes(data=True),
            key=lambda nd: nd[1].get("paper_count", 0),
            reverse=True,
        )[:max_nodes]
        G_sub = G_sub.subgraph({nd[0] for nd in top_nodes}).copy()
        st.info(
            f"Showing top {max_nodes} entities of {total_kg_nodes:,} by evidence strength. "
            "Adjust the Max nodes slider or filter by entity type to change the selection."
        )

    # Problem 4 — apply min_edge_cooccurrence filter directly on G_sub
    edges_to_remove = [
        (u, v, k) for u, v, k, d in G_sub.edges(data=True, keys=True)
        if d.get("weight", len(d.get("evidence_pmids", []))) < min_edge_cooccurrence
    ]
    G_sub.remove_edges_from(edges_to_remove)
    isolated = list(nx.isolates(G_sub))
    G_sub.remove_nodes_from(isolated)

    if G_sub.number_of_nodes() == 0:
        st.warning("No nodes remain after applying the edge co-occurrence filter. Lower the Min edge co-occurrence slider.")
        return

    # ── Lazy-load: only build vis.js HTML when user clicks Render ─────────────
    render_key = f"kg_render_{total_kg_nodes}_{G_sub.number_of_edges()}_{min_evidence}_{min_edge_cooccurrence}_{max_nodes}_{search_term}"
    if not st.session_state.get(render_key):
        st.info(
            f"Entity graph ready: {G_sub.number_of_nodes()} nodes, "
            f"{min(G_sub.number_of_edges(), _MAX_KG_EDGES):,} edges. "
            "Click **Render Graph** to visualize."
        )
        if st.button("🔬 Render Graph", key=f"btn_{render_key}"):
            st.session_state[render_key] = True
            st.rerun()
        return

    net = _get_pyvis_net(height=f"{height}px")
    search_lower = search_term.lower().strip()

    # ── Pre-compute UMAP + ForceAtlas2 layout ─────────────────────────────────
    positions = _compute_kg_layout(G_sub)

    # ── Add nodes with pre-computed positions (physics=False per node) ────────
    counts = [d.get("paper_count", 1) for _, d in G_sub.nodes(data=True)]
    median_count = sorted(counts)[len(counts) // 2] if counts else 1

    has_gap_nodes = False
    for node, data in G_sub.nodes(data=True):
        node_str = str(node)
        etype = data.get("entity_type", "UNKNOWN")
        color = _ENTITY_COLOR_MAP.get(etype, _DEFAULT_KG_NODE_COLOR)
        size = float(max(10, min(60, data.get("paper_count", 1) * 3)))
        paper_count = data.get("paper_count", 0)
        umls_id = data.get("umls_id", "—")
        is_gap = data.get("is_gap_node", False) and show_gap_nodes
        is_highlight = highlight_entity and node_str == highlight_entity
        is_search_match = search_lower and search_lower in node_str.lower()

        if is_gap:
            has_gap_nodes = True
            border_color = "#FFD700"
            border_width = 4
            final_color = color
        elif is_highlight:
            border_color = "#FFD700"
            border_width = 4
            final_color = "#FFD700"
        elif is_search_match:
            border_color = "#FFD700"
            border_width = 3
            final_color = color
        else:
            border_color = _color_opacity(color, 0.6)
            border_width = 1.5
            final_color = color

        # Neighbor summary for tooltip
        out_edges = list(G_sub.out_edges(node, data=True))[:5]
        edge_summary = "<br>".join(
            f"  → {v}: {d.get('relationship_type', '?')}"
            for _, v, d in out_edges
        )

        # Always show label — white, 14px
        display_label = node_str[:30]

        node_pmids = data.get("evidence_pmids", [])[:3]
        pmid_str = ", ".join(str(p) for p in node_pmids)

        tooltip = (
            f"{node_str}\n"
            f"Type: {etype}\n"
            f"Papers: {paper_count}"
            + (f"\nPMIDs: {pmid_str}" if pmid_str else "")
        )

        px, py = positions.get(node_str, (0.0, 0.0))
        net.add_node(
            node_str,
            label=display_label,
            size=size,
            x=px * _KG_LAYOUT_SCALE,
            y=py * _KG_LAYOUT_SCALE,
            physics=False,
            color={
                "background": final_color,
                "border": border_color,
                "highlight": {"background": "#FFD700", "border": "#FFA500"},
                "hover": {"background": final_color, "border": "#FFFFFF"},
            },
            borderWidth=border_width,
            title=tooltip,
            shape=_ENTITY_SHAPE.get(etype, "dot"),
            font={"size": 14, "color": "#FFFFFF", "face": "Open Sans"},
        )

    # ── Add edges (capped at _MAX_KG_EDGES, verb-based preferred) ────────────
    total_edges = G_sub.number_of_edges()
    # Sort: verb-based relationships first (not co-occurrence), then by confidence
    all_edges = sorted(
        G_sub.edges(data=True),
        key=lambda e: (
            0 if e[2].get("relationship_type", "") != "co-occurs with" else 1,
            -e[2].get("confidence_score", 0.5),
        ),
    )
    seen_edges: Set[tuple] = set()
    for u, v, data in all_edges:
        if len(seen_edges) >= _MAX_KG_EDGES:
            break
        edge_key = (str(u), str(v))
        if edge_key in seen_edges:
            continue
        # Enforce minimum co-occurrence evidence for edges
        evidence_pmids = data.get("evidence_pmids", [])
        if len(evidence_pmids) < min_edge_cooccurrence:
            continue
        seen_edges.add(edge_key)

        rel_type = data.get("relationship_type", "unknown")
        evidence_pmids = data.get("evidence_pmids", [])
        confidence = data.get("confidence_score", 0.5)
        edge_color = _EDGE_COLORS.get(rel_type, _DEFAULT_EDGE_COLOR)
        edge_color_alpha = _color_opacity(edge_color, 0.5)

        pmid_str = ", ".join(str(p) for p in evidence_pmids[:3])
        tooltip = (
            f"<b>{u}</b> → <b>{v}</b><br>"
            f"Relationship: {rel_type}<br>"
            f"Confidence: {confidence:.2f}<br>"
            + (f"PMIDs: {pmid_str}" if pmid_str else "")
        )

        net.add_edge(
            str(u), str(v),
            color={"color": edge_color_alpha, "highlight": "#FFD700", "hover": "#FFFFFF"},
            title=tooltip,
            label=rel_type if len(seen_edges) < 200 else "",
            font={"size": 7, "color": config.TEXT_SECONDARY, "align": "middle"},
        )

    if total_edges > _MAX_KG_EDGES:
        st.caption(
            f"Showing {_MAX_KG_EDGES:,} of {total_edges:,} edges (highest confidence). "
            "Use filters to narrow the graph."
        )

    extra_css = _GAP_NODE_CSS if has_gap_nodes else ""
    cache_key = f"kg_{G_sub.number_of_nodes()}_{min(total_edges, _MAX_KG_EDGES)}_{search_lower}_{min_evidence}_{min_edge_cooccurrence}_{max_nodes}"
    _render_html(net, height=height, extra_css=extra_css, cache_key=cache_key)


def render_entity_legend():
    """Render a grouped entity type legend with node size indicator."""
    st.markdown(
        f"<p style='color:{config.TEXT_SECONDARY};font-size:13px;"
        f"font-weight:600;margin-bottom:6px'>Entity Types</p>",
        unsafe_allow_html=True,
    )
    # Canonical display groups: (label, color, shape_hint)
    _LEGEND_GROUPS = [
        ("Disease / Cancer",            "#FF5252"),
        ("Gene / DNA / RNA / Protein",  "#00E676"),
        ("Chemical / Drug",             "#00D4FF"),
        ("Biological Process / Pathway","#A8E6CF"),
        ("Cell / Cell Type / Cell Line","#7B61FF"),
        ("Organism",                    "#FFD600"),
        ("Anatomy",                     "#FF8B94"),
        ("Lab Procedure",               "#8899AA"),
    ]
    for label, color in _LEGEND_GROUPS:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>"
            f"<div style='width:12px;height:12px;border-radius:50%;"
            f"background:{color};flex-shrink:0'></div>"
            f"<span style='color:{config.TEXT_PRIMARY};font-size:11px'>{label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    # Node size legend
    st.markdown(
        f"<p style='color:{config.TEXT_SECONDARY};font-size:12px;"
        f"font-weight:600;margin-top:10px;margin-bottom:4px'>Node Size</p>",
        unsafe_allow_html=True,
    )
    for label, px in [("Low frequency", 6), ("Medium", 11), ("High frequency", 18)]:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>"
            f"<div style='width:{px}px;height:{px}px;border-radius:50%;"
            f"background:#9CA3AF;flex-shrink:0'></div>"
            f"<span style='color:{config.TEXT_PRIMARY};font-size:11px'>{label}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_relationship_evidence_table(relationships_df, papers_df=None):
    """
    Render the relationship evidence table with sortable columns,
    expandable rows for sentence context, and clickable PMID links.
    """
    if relationships_df is None or relationships_df.empty:
        st.info("No relationships extracted yet.")
        return

    import pandas as pd

    display_df = relationships_df.copy()

    # Add clickable PMID links
    def pmid_link(pmid):
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

    cols_to_show = [
        c for c in [
            "subject_entity", "relationship_verb", "object_entity",
            "pmid", "sentence_context",
        ]
        if c in display_df.columns
    ]
    display_df = display_df[cols_to_show]

    # Pagination
    page_size = 25
    total_rows = len(display_df)
    total_pages = max(1, (total_rows + page_size - 1) // page_size)
    page = st.number_input(
        f"Page (of {total_pages})", min_value=1, max_value=total_pages,
        value=1, key="rel_table_page",
    )
    start = (page - 1) * page_size
    page_df = display_df.iloc[start : start + page_size]

    st.dataframe(
        page_df,
        width="stretch",
        column_config={
            "pmid": st.column_config.LinkColumn(
                "PMID",
                display_text=r"(\d+)",
            ),
        } if "pmid" in page_df.columns else None,
    )

    # Export
    csv = display_df.to_csv(index=False)
    st.download_button(
        "Export relationships as CSV",
        data=csv,
        file_name="relationships.csv",
        mime="text/csv",
        key="rel_export_btn",
    )
