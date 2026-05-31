"""
Knowledge Graph page — Bibliometric Network Explorer.

Tab selection is tracked in session state so exactly ONE visualization
block executes per Streamlit rerun:
  st.session_state["bne_active_tab"]  → "coauth" | "keyword" | "topic" | "entity"
  viz_mode_selector()                 → "network" | "overlay" | "density"

Defaults: Co-authorship tab · Network mode.
"""

import logging

import pandas as pd
import streamlit as st

import config
from pipeline.gap_detector import GapDetector
from pipeline.lazy_builders import extract_entities_lazy, extract_relationships_lazy
from ui.components.cards import empty_state
from ui.components.network_controls import cluster_legend, controls_hint, viz_mode_selector
from visualization.graph_viz import (
    render_entity_legend,
    render_knowledge_graph,
    render_relationship_evidence_table,
)
from visualization.network_viz import (
    _build_controls_panel,
    render_coauthorship_network,
    render_density_visualization,
    render_keyword_network,
    render_network_stats,
    render_overlay_visualization,
    render_topic_network,
)

logger = logging.getLogger(__name__)

# ── Tab registry (key → display label) ───────────────────────────────────────
_TABS = [
    ("coauth",  "👥  Co-authorship"),
    ("keyword", "🔑  Keyword Co-occurrence"),
    ("topic",   "📚  Topic Landscape"),
    ("entity",  "🔬  Entity Knowledge Graph"),
]


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    if not st.session_state.get("pipeline_complete"):
        empty_state("🕸️", "No data loaded yet",
                    "Complete a search on the Home page first.")
        return

    papers_df     = st.session_state.get("papers_df")
    coauth_graph  = st.session_state.get("coauth_graph")
    keyword_graph = st.session_state.get("keyword_graph")
    topic_graph   = st.session_state.get("topic_graph")
    coauth_stats  = st.session_state.get("coauth_stats", {})
    keyword_stats = st.session_state.get("keyword_stats", {})
    kg_graph      = st.session_state.get("kg_graph")
    kg            = st.session_state.get("knowledge_graph")
    rels_df       = st.session_state.get("relationships_df")
    topic_labels  = st.session_state.get("topic_labels", {})
    query         = st.session_state.get("current_query", "")

    # ── Page header ───────────────────────────────────────────────────────────
    n_papers = len(papers_df) if papers_df is not None else 0
    coauth_n = coauth_graph.number_of_nodes() if coauth_graph else 0
    kw_n     = keyword_graph.number_of_nodes() if keyword_graph else 0
    st.markdown(
        f"<h2 style='margin-bottom:2px'>Bibliometric Network Explorer</h2>"
        f"<p style='color:{config.TEXT_SECONDARY};font-size:13px;margin-bottom:0.8rem'>"
        f"Query: <b>{query}</b> &nbsp;·&nbsp; "
        f"{n_papers:,} papers &nbsp;·&nbsp; "
        f"{coauth_n:,} authors &nbsp;·&nbsp; {kw_n:,} keywords &nbsp;·&nbsp; "
        f"Network / Overlay / Density views</p>",
        unsafe_allow_html=True,
    )

    # ── Session-state tab tracking (default: "coauth") ────────────────────────
    if "bne_active_tab" not in st.session_state:
        st.session_state["bne_active_tab"] = "coauth"
    active_tab = st.session_state["bne_active_tab"]

    # ── Tab selector row — styled buttons that look like tabs ─────────────────
    tab_cols = st.columns(len(_TABS))
    for (tab_key, tab_label), col in zip(_TABS, tab_cols):
        with col:
            is_active = active_tab == tab_key
            if st.button(
                tab_label,
                key=f"bne_tab_{tab_key}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                st.session_state["bne_active_tab"] = tab_key
                st.rerun()

    st.markdown(
        f"<hr style='border-color:{config.BORDER_COLOR};margin:0.5rem 0 1rem'>",
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # Single dispatch — only one block executes per rerun
    # ══════════════════════════════════════════════════════════════════════════

    if active_tab == "coauth":
        _render_coauth(coauth_graph, coauth_stats, papers_df)

    elif active_tab == "keyword":
        _render_keyword(keyword_graph, keyword_stats, papers_df)

    elif active_tab == "topic":
        _render_topic(topic_graph, topic_labels, papers_df)

    else:  # entity
        _render_entity(kg_graph, kg, rels_df, papers_df)


# ── Per-tab render helpers ────────────────────────────────────────────────────

def _render_coauth(coauth_graph, coauth_stats, papers_df):
    if not coauth_graph or coauth_graph.number_of_nodes() == 0:
        st.info("Co-authorship network not available. Run the pipeline first.")
        return

    mode = viz_mode_selector("coauth")

    if mode == "network":
        controls_hint()
        ctrl = _build_controls_panel("coauth")
        graph_col, stats_col = st.columns([5, 1])
        with graph_col:
            render_coauthorship_network(coauth_graph, controls=ctrl, height=780)
        with stats_col:
            render_network_stats(coauth_stats, "Stats")
            cluster_legend(coauth_graph)

    elif mode == "overlay":
        st.markdown(
            "<div style='font-size:12px;color:#6B7280;margin-bottom:4px'>"
            "Color encodes the <b>average publication year</b> per author. "
            "Positions are identical to the Network view.</div>",
            unsafe_allow_html=True,
        )
        controls_hint()
        ctrl = _build_controls_panel("ov_coauth")
        render_overlay_visualization(coauth_graph, papers_df, "coauth", ctrl, height=780)

    else:  # density
        st.markdown(
            "<div style='font-size:12px;color:#6B7280;margin-bottom:4px'>"
            "Heatmap computed with the VOSviewer kernel formula "
            "<i>D(x) = Σ wᵢ·exp(−‖x−xᵢ‖²/2(d̄·h)²)</i>. "
            "Adjust <b>Kernel width (h)</b> to broaden or sharpen clusters.</div>",
            unsafe_allow_html=True,
        )
        controls_hint(show_bandwidth=True)
        ctrl = _build_controls_panel("dn_coauth", show_bandwidth=True)
        render_density_visualization(coauth_graph, "coauth", ctrl, height=780)


def _render_keyword(keyword_graph, keyword_stats, papers_df):
    if not keyword_graph or keyword_graph.number_of_nodes() == 0:
        st.info("Keyword network not available. Run the pipeline first.")
        return

    kw_types = sorted({d.get("keyword_type", "author")
                       for _, d in keyword_graph.nodes(data=True)})
    selected_types = st.multiselect(
        "Keyword types", options=kw_types, default=kw_types, key="kg_kw_types",
    )
    kw_g = (
        keyword_graph.subgraph(
            [n for n, d in keyword_graph.nodes(data=True)
             if d.get("keyword_type") in selected_types]
        )
        if selected_types and len(selected_types) < len(kw_types)
        else keyword_graph
    )

    st.markdown(
        "<div style='font-size:12px;color:#6B7280;margin-bottom:4px'>"
        "●&nbsp;Author&nbsp; ■&nbsp;MeSH&nbsp; ◆&nbsp;Chemical&nbsp; ▲&nbsp;Qualifier</div>",
        unsafe_allow_html=True,
    )

    mode = viz_mode_selector("kw")

    if mode == "network":
        controls_hint()
        ctrl = _build_controls_panel("keyword")
        graph_col, stats_col = st.columns([5, 1])
        with graph_col:
            render_keyword_network(kw_g, controls=ctrl, height=780)
        with stats_col:
            render_network_stats(keyword_stats, "Stats")
            cluster_legend(kw_g)

    elif mode == "overlay":
        st.markdown(
            "<div style='font-size:12px;color:#6B7280;margin-bottom:4px'>"
            "Color encodes the <b>average publication year</b> "
            "in which each keyword appears.</div>",
            unsafe_allow_html=True,
        )
        controls_hint()
        ctrl = _build_controls_panel("ov_keyword")
        render_overlay_visualization(kw_g, papers_df, "keyword", ctrl, height=780)

    else:  # density
        st.markdown(
            "<div style='font-size:12px;color:#6B7280;margin-bottom:4px'>"
            "Keyword density — shows thematic hotspots in the research landscape.</div>",
            unsafe_allow_html=True,
        )
        controls_hint(show_bandwidth=True)
        ctrl = _build_controls_panel("dn_keyword", show_bandwidth=True)
        render_density_visualization(kw_g, "keyword", ctrl, height=780)


def _render_topic(topic_graph, topic_labels, papers_df):
    if not topic_graph or topic_graph.number_of_nodes() == 0:
        st.info(
            "Topic landscape not available. "
            "NLP and embedding steps must complete first."
        )
        return

    yr_min = yr_max = None
    if papers_df is not None and "pub_year" in papers_df.columns:
        years = sorted(papers_df["pub_year"].dropna().astype(int).unique())
        if len(years) >= 2:
            yr_min, yr_max = st.slider(
                "Publication year range",
                min_value=int(min(years)), max_value=int(max(years)),
                value=(int(min(years)), int(max(years))),
                key="kg_topic_year",
            )

    mode = viz_mode_selector("topic")

    if mode == "network":
        st.markdown(
            "<p style='font-size:13px;color:#6B7280;margin-bottom:6px'>"
            "Node&nbsp;size&nbsp;=&nbsp;paper count &nbsp;·&nbsp; "
            "Edge&nbsp;=&nbsp;shared papers between topics.</p>",
            unsafe_allow_html=True,
        )
        controls_hint()
        ctrl = _build_controls_panel("topic_kg")
        render_topic_network(
            topic_graph, controls=ctrl, height=780,
            year_range=(yr_min, yr_max) if yr_min else None,
        )

    elif mode == "overlay":
        st.markdown(
            "<div style='font-size:12px;color:#6B7280;margin-bottom:4px'>"
            "Color encodes the <b>average publication year</b> of papers "
            "assigned to each topic.</div>",
            unsafe_allow_html=True,
        )
        controls_hint()
        ctrl = _build_controls_panel("ov_topic")
        render_overlay_visualization(topic_graph, papers_df, "topic", ctrl, height=780)

    else:  # density
        st.markdown(
            "<div style='font-size:12px;color:#6B7280;margin-bottom:4px'>"
            "Topic density — shows research concentration across the topic landscape.</div>",
            unsafe_allow_html=True,
        )
        controls_hint(show_bandwidth=True)
        ctrl = _build_controls_panel("dn_topic", show_bandwidth=True)
        render_density_visualization(topic_graph, "topic", ctrl, height=780)

    if topic_labels:
        chips = " ".join(
            f"<span style='display:inline-block;"
            f"background:{config.COMMUNITY_COLORS[i % len(config.COMMUNITY_COLORS)]}22;"
            f"border:1px solid {config.COMMUNITY_COLORS[i % len(config.COMMUNITY_COLORS)]}66;"
            f"color:{config.COMMUNITY_COLORS[i % len(config.COMMUNITY_COLORS)]};"
            f"border-radius:20px;padding:2px 10px;font-size:11px;margin:3px'>"
            f"{v['label']} ({v['count']})</span>"
            for i, (_, v) in enumerate(
                sorted(topic_labels.items(), key=lambda x: x[1]["count"], reverse=True)[:16]
            )
        )
        st.markdown(f"<div style='margin-top:10px'>{chips}</div>", unsafe_allow_html=True)


def _render_entity(kg_graph, kg, rels_df, papers_df):
    papers_df_local   = st.session_state.get("papers_df")
    entities_df_local = st.session_state.get("entities_df")
    _has_entities = entities_df_local is not None and not entities_df_local.empty
    _has_rels     = rels_df is not None and not rels_df.empty

    if not _has_entities and papers_df_local is not None and not papers_df_local.empty:
        st.warning(
            "Entity extraction did not complete during the main pipeline "
            "(this can happen if the spaCy model is not installed). "
            "Click below to re-run it now (~30-90 sec)."
        )
        if st.button("Extract Entities", type="primary", key="extract_ents_btn"):
            extract_entities_lazy(papers_df_local)
            st.rerun()

    elif _has_entities and not _has_rels:
        st.info(
            "Relationship edges (dependency-parse) not yet extracted — optional, ~2-5 min. "
            "The entity graph above is fully interactive without them."
        )
        if st.button("Extract Relationships", type="primary", key="extract_rels_btn"):
            extract_relationships_lazy(papers_df_local, entities_df_local)
            st.rerun()

    if kg_graph is None or kg_graph.number_of_nodes() == 0:
        if not _has_entities:
            pass  # recovery button already shown above
        else:
            st.info("Entity knowledge graph is empty — no entities were found for this corpus.")
        return

    kg_nodes = kg_graph.number_of_nodes()
    kg_edges = kg_graph.number_of_edges()
    st.markdown(
        f"<p style='font-size:13px;color:#6B7280;margin-bottom:6px'>"
        f"Node&nbsp;size&nbsp;=&nbsp;evidence strength &nbsp;·&nbsp; "
        f"Color&nbsp;=&nbsp;entity type &nbsp;·&nbsp; "
        f"Arrows&nbsp;=&nbsp;relationship direction &nbsp;·&nbsp; "
        f"{kg_nodes:,} nodes, {kg_edges:,} edges</p>",
        unsafe_allow_html=True,
    )
    if kg_nodes > config.GRAPH_MAX_DISPLAY_NODES:
        st.info(
            f"Entity graph has {kg_nodes:,} nodes. "
            "Use the filters on the left to focus on specific entity types or relationships."
        )

    ctrl_col, graph_col = st.columns([1, 4])

    with ctrl_col:
        st.markdown(
            f"<div style='font-size:11px;font-weight:600;"
            f"color:{config.TEXT_SECONDARY};text-transform:uppercase;"
            f"letter-spacing:.06em;margin-bottom:10px'>Graph Controls</div>",
            unsafe_allow_html=True,
        )
        all_types = list(config.ENTITY_TYPE_COLORS.keys())
        selected_types = st.multiselect(
            "Entity Types", options=all_types, default=all_types,
            key="kg_entity_types",
        )

        if rels_df is not None and not rels_df.empty and "relationship_verb" in rels_df.columns:
            all_rels = sorted(rels_df["relationship_verb"].dropna().unique().tolist())
        else:
            all_rels = []
        selected_rels = (
            st.multiselect("Relationship Types", options=all_rels,
                           default=all_rels[:10], key="kg_rel_types")
            if all_rels else None
        )

        min_evidence  = st.slider("Min evidence (papers)", 1, 20, 1, key="kg_min_ev")
        search_entity = st.text_input("Search entity", placeholder="Entity name...",
                                      key="kg_search")
        show_gaps = st.toggle("Highlight Research Gaps", value=True, key="kg_show_gaps")

        if show_gaps and kg and not st.session_state.get("gap_report"):
            with st.spinner("Detecting research gaps..."):
                try:
                    gd = GapDetector(kg)
                    gd.find_structural_gaps()
                    gd.find_cross_domain_gaps()
                    report = gd.compile_gap_report()
                    st.session_state["gap_report"] = report
                    pairs = [(r["concept_a"], r["concept_b"])
                             for r in report if r.get("concept_b")]
                    if pairs:
                        kg.mark_gap_nodes(pairs)
                except Exception as exc:
                    st.warning(f"Gap detection failed: {exc}")

        st.markdown("<hr>", unsafe_allow_html=True)
        render_entity_legend()

        st.markdown("<hr>", unsafe_allow_html=True)
        if kg_graph:
            st.download_button(
                "⬇ Export JSON",
                data=kg.export_to_json() if kg else "{}",
                file_name="knowledge_graph.json",
                mime="application/json",
                key="kg_export_json",
            )
            if rels_df is not None and not rels_df.empty:
                st.download_button(
                    "⬇ Export CSV",
                    data=rels_df.to_csv(index=False),
                    file_name="relationships.csv",
                    mime="text/csv",
                    key="kg_export_csv",
                )

    with graph_col:
        render_knowledge_graph(
            G=kg_graph,
            highlight_entity=search_entity or None,
            entity_type_filter=set(selected_types),
            relationship_type_filter=set(selected_rels) if selected_rels else None,
            min_evidence=min_evidence,
            show_gap_nodes=show_gaps,
            search_term=search_entity,
            height=800,
        )

    st.markdown(
        f"<hr style='border-color:{config.BORDER_COLOR};margin:2rem 0'>",
        unsafe_allow_html=True,
    )
    st.markdown("### Relationship Evidence")
    if rels_df is not None and not rels_df.empty:
        fc1, fc2 = st.columns(2)
        with fc1:
            filter_src = st.text_input("Filter source entity", key="rel_filter_src")
        with fc2:
            filter_rel = st.text_input("Filter relationship type", key="rel_filter_rel")

        disp = rels_df.copy()
        if filter_src:
            disp = disp[
                disp.get("subject_entity", pd.Series(dtype=str))
                .str.contains(filter_src, case=False, na=False)
            ]
        if filter_rel:
            disp = disp[
                disp.get("relationship_verb", pd.Series(dtype=str))
                .str.contains(filter_rel, case=False, na=False)
            ]
        render_relationship_evidence_table(disp, papers_df)
    else:
        st.info("No relationship data extracted yet.")
