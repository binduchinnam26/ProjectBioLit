"""
Knowledge Graph page — Bibliometric Network Explorer.

State-driven rendering: exactly ONE visualization is shown at all times,
determined by the combination of:
  active_tab  ∈ {coauthorship, keyword, topic, entity}
  active_view ∈ {network, overlay, density}

Defaults: Co-authorship tab · Network mode.
"""

import logging
import pickle

import streamlit as st
import config
from ui.components.cards import empty_state

logger = logging.getLogger(__name__)


# ── Lazy pipeline helpers (unchanged) ────────────────────────────────────────

def _extract_entities_lazy(papers_df):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from pipeline.nlp_processor import NLPProcessor
    from pipeline.knowledge_graph import KnowledgeGraph
    from pathlib import Path
    from utils.helpers import query_hash
    import pandas as pd

    query = st.session_state.get("current_query", "")
    with st.spinner(
        f"Extracting entities from {len(papers_df):,} papers (NER-only — ~30-90 sec)…"
    ):
        try:
            nlp_p = NLPProcessor(db_manager=None)
            nlp_p.setup()
            entities_df, _ = nlp_p.process_corpus(papers_df)
            kg = KnowledgeGraph()
            kg_graph = kg.build_from_entities(entities_df, pd.DataFrame())
            st.session_state["entities_df"] = entities_df
            st.session_state["kg_graph"] = kg_graph
            st.session_state["knowledge_graph"] = kg
            qh = query_hash(query) if query else "default"
            derived_cache = Path(config.PROCESSED_DIR) / f"{qh}_derived.pkl"
            if derived_cache.exists():
                with open(derived_cache, "rb") as _f:
                    _d = pickle.load(_f)
                _d.update({"entities_df": entities_df, "kg_graph": kg_graph})
                with open(derived_cache, "wb") as _f:
                    pickle.dump(_d, _f, protocol=pickle.HIGHEST_PROTOCOL)
            st.success(
                f"Entity extraction complete: {len(entities_df):,} entities "
                f"from {len(papers_df):,} papers."
            )
        except Exception as exc:
            logger.error("Lazy entity extraction failed: %s", exc)
            st.error(f"Entity extraction failed: {exc}")


def _extract_relationships_lazy(papers_df, entities_df):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from pipeline.nlp_processor import NLPProcessor
    from pipeline.knowledge_graph import KnowledgeGraph
    from pathlib import Path
    from utils.helpers import query_hash

    query = st.session_state.get("current_query", "")
    with st.spinner(
        f"Extracting relationships from {len(papers_df):,} papers (dep parser — ~2-5 min)…"
    ):
        try:
            nlp_p = NLPProcessor(db_manager=None)
            nlp_p.setup()
            rels_df = nlp_p.extract_relationships_corpus(papers_df, entities_df)
            kg = KnowledgeGraph()
            kg_graph = kg.build_from_entities(entities_df, rels_df)
            st.session_state["relationships_df"] = rels_df
            st.session_state["kg_graph"] = kg_graph
            st.session_state["knowledge_graph"] = kg
            qh = query_hash(query) if query else "default"
            derived_cache = Path(config.PROCESSED_DIR) / f"{qh}_derived.pkl"
            if derived_cache.exists():
                with open(derived_cache, "rb") as _f:
                    _d = pickle.load(_f)
                _d.update({"relationships_df": rels_df, "kg_graph": kg_graph})
                with open(derived_cache, "wb") as _f:
                    pickle.dump(_d, _f, protocol=pickle.HIGHEST_PROTOCOL)
            st.success(
                f"Relationships extracted: {len(rels_df):,} edges added to knowledge graph."
            )
        except Exception as exc:
            logger.error("Lazy relationship extraction failed: %s", exc)
            st.error(f"Relationship extraction failed: {exc}")


# ── UI helpers ────────────────────────────────────────────────────────────────

def _controls_hint(show_bandwidth: bool = False):
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


def _cluster_legend(G):
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


def _topic_chips(topic_labels: dict):
    if not topic_labels:
        return
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


def _mode_description(active_view: str, active_tab: str):
    desc = {
        "network": "Cluster-colored nodes · Edge thickness = link strength · "
                   "Positions driven by semantic similarity (UMAP + ForceAtlas2)",
        "overlay": "Node color = average publication year (Viridis: "
                   "dark blue → oldest · yellow → most recent) · "
                   "Positions identical to Network view",
        "density": "Gaussian KDE heatmap · "
                   "D(x) = Σ wᵢ·exp(−‖x−xᵢ‖²/2(d̄·h)²) · "
                   "Adjust Kernel width (h) to broaden or sharpen clusters",
    }
    st.markdown(
        f"<div style='font-size:12px;color:#6B7280;margin-bottom:6px'>{desc[active_view]}</div>",
        unsafe_allow_html=True,
    )


# ── Tab content renderers ─────────────────────────────────────────────────────

def _render_coauth(active_view: str, coauth_graph, coauth_stats, papers_df):
    from visualization.network_viz import (
        render_coauthorship_network,
        render_overlay_visualization,
        render_density_visualization,
        render_network_stats,
        _build_controls_panel,
    )

    if not coauth_graph or coauth_graph.number_of_nodes() == 0:
        st.info("Co-authorship network not available. Run the pipeline first.")
        return

    _mode_description(active_view, "coauthorship")

    if active_view == "network":
        _controls_hint()
        ctrl = _build_controls_panel("coauth")
        graph_col, stats_col = st.columns([5, 1])
        with graph_col:
            render_coauthorship_network(coauth_graph, controls=ctrl, height=780)
        with stats_col:
            render_network_stats(coauth_stats, "Stats")
            _cluster_legend(coauth_graph)

    elif active_view == "overlay":
        _controls_hint()
        ctrl = _build_controls_panel("ov_coauth")
        render_overlay_visualization(coauth_graph, papers_df, "coauth", ctrl, height=780)

    else:  # density
        _controls_hint(show_bandwidth=True)
        ctrl = _build_controls_panel("dn_coauth", show_bandwidth=True)
        render_density_visualization(coauth_graph, "coauth", ctrl, height=780)


def _render_keyword(active_view: str, keyword_graph, keyword_stats, papers_df):
    from visualization.network_viz import (
        render_keyword_network,
        render_overlay_visualization,
        render_density_visualization,
        render_network_stats,
        _build_controls_panel,
    )

    if not keyword_graph or keyword_graph.number_of_nodes() == 0:
        st.info("Keyword network not available. Run the pipeline first.")
        return

    # Keyword type filter (always visible, independent of mode)
    kw_types = sorted({d.get("keyword_type", "author")
                       for _, d in keyword_graph.nodes(data=True)})
    selected_types = st.multiselect(
        "Keyword types", options=kw_types, default=kw_types, key="bne_kw_types",
    )
    if selected_types and len(selected_types) < len(kw_types):
        keep = [n for n, d in keyword_graph.nodes(data=True)
                if d.get("keyword_type") in selected_types]
        kw_g = keyword_graph.subgraph(keep)
    else:
        kw_g = keyword_graph

    st.markdown(
        "<div style='font-size:12px;color:#6B7280;margin-bottom:4px'>"
        "●&nbsp;Author&nbsp; ■&nbsp;MeSH&nbsp; ◆&nbsp;Chemical&nbsp; ▲&nbsp;Qualifier</div>",
        unsafe_allow_html=True,
    )
    _mode_description(active_view, "keyword")

    if active_view == "network":
        _controls_hint()
        ctrl = _build_controls_panel("keyword")
        graph_col, stats_col = st.columns([5, 1])
        with graph_col:
            render_keyword_network(kw_g, controls=ctrl, height=780)
        with stats_col:
            render_network_stats(keyword_stats, "Stats")
            _cluster_legend(kw_g)

    elif active_view == "overlay":
        _controls_hint()
        ctrl = _build_controls_panel("ov_keyword")
        render_overlay_visualization(kw_g, papers_df, "keyword", ctrl, height=780)

    else:  # density
        _controls_hint(show_bandwidth=True)
        ctrl = _build_controls_panel("dn_keyword", show_bandwidth=True)
        render_density_visualization(kw_g, "keyword", ctrl, height=780)


def _render_topic(active_view: str, topic_graph, topic_labels, papers_df):
    from visualization.network_viz import (
        render_topic_network,
        render_overlay_visualization,
        render_density_visualization,
        _build_controls_panel,
    )

    if not topic_graph or topic_graph.number_of_nodes() == 0:
        st.info(
            "Topic landscape not available. "
            "Topic modeling requires NLP and embedding steps to complete."
        )
        return

    # Optional year range filter (only meaningful for network mode with year_range param)
    yr_min = yr_max = None
    if papers_df is not None and "pub_year" in papers_df.columns:
        years = sorted(papers_df["pub_year"].dropna().astype(int).unique())
        if len(years) >= 2:
            yr_min, yr_max = st.slider(
                "Publication year range",
                min_value=int(min(years)), max_value=int(max(years)),
                value=(int(min(years)), int(max(years))),
                key="bne_topic_year",
            )

    _mode_description(active_view, "topic")

    if active_view == "network":
        _controls_hint()
        ctrl = _build_controls_panel("topic_kg")
        render_topic_network(
            topic_graph, controls=ctrl, height=780,
            year_range=(yr_min, yr_max) if yr_min else None,
        )

    elif active_view == "overlay":
        _controls_hint()
        ctrl = _build_controls_panel("ov_topic")
        render_overlay_visualization(topic_graph, papers_df, "topic", ctrl, height=780)

    else:  # density
        _controls_hint(show_bandwidth=True)
        ctrl = _build_controls_panel("dn_topic", show_bandwidth=True)
        render_density_visualization(topic_graph, "topic", ctrl, height=780)

    _topic_chips(topic_labels)


def _render_entity(kg_graph, kg, rels_df, papers_df):
    """Entity Knowledge Graph tab — preserved behaviour, no mode selector."""
    from visualization.graph_viz import (
        render_knowledge_graph,
        render_entity_legend,
        render_relationship_evidence_table,
    )
    from pipeline.gap_detector import GapDetector

    entities_df_local = st.session_state.get("entities_df")
    _has_entities = entities_df_local is not None and not entities_df_local.empty
    _has_rels = rels_df is not None and not rels_df.empty

    if not _has_entities and papers_df is not None and not papers_df.empty:
        st.warning(
            "Entity extraction did not complete during the main pipeline. "
            "Click below to re-run it now (~30-90 sec)."
        )
        if st.button("Extract Entities", type="primary", key="extract_ents_btn"):
            _extract_entities_lazy(papers_df)
            st.rerun()
        return

    elif _has_entities and not _has_rels:
        st.info(
            "Relationship edges (dependency-parse) not yet extracted — optional, ~2-5 min."
        )
        if st.button("Extract Relationships", type="primary", key="extract_rels_btn"):
            _extract_relationships_lazy(papers_df, entities_df_local)
            st.rerun()

    if kg_graph is None or kg_graph.number_of_nodes() == 0:
        if not _has_entities:
            pass  # recovery button shown above
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
            "Use the filters on the left to focus on specific entity types."
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
            "Entity Types", options=all_types, default=all_types, key="kg_entity_types",
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
        import pandas as pd
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


# ── Main render ───────────────────────────────────────────────────────────────

# Tab and view labels — order determines default (index 0)
_TABS = [
    ("coauthorship", "👥  Co-authorship"),
    ("keyword",      "🔑  Keyword Co-occurrence"),
    ("topic",        "📚  Topic Landscape"),
    ("entity",       "🔬  Entity Knowledge Graph"),
]
_VIEWS = [
    ("network", "🔵  Network"),
    ("overlay", "🎨  Overlay"),
    ("density", "🌡️  Density"),
]


def render():
    if not st.session_state.get("pipeline_complete"):
        empty_state("🕸️", "No data loaded yet",
                    "Complete a search on the Home page first.")
        return

    # ── Session state: pull all data once ────────────────────────────────────
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
    n_papers  = len(papers_df) if papers_df is not None else 0
    coauth_n  = coauth_graph.number_of_nodes() if coauth_graph else 0
    kw_n      = keyword_graph.number_of_nodes() if keyword_graph else 0
    st.markdown(
        f"<h2 style='margin-bottom:2px'>Bibliometric Network Explorer</h2>"
        f"<p style='color:{config.TEXT_SECONDARY};font-size:13px;margin-bottom:1rem'>"
        f"Query: <b>{query}</b> &nbsp;·&nbsp; "
        f"{n_papers:,} papers &nbsp;·&nbsp; "
        f"{coauth_n:,} authors &nbsp;·&nbsp; {kw_n:,} keywords</p>",
        unsafe_allow_html=True,
    )

    # ── State: initialise defaults ────────────────────────────────────────────
    # Defaults: first tab (Co-authorship) · first mode (Network)
    if "bne_active_tab" not in st.session_state:
        st.session_state["bne_active_tab"] = _TABS[0][0]
    if "bne_active_view" not in st.session_state:
        st.session_state["bne_active_view"] = _VIEWS[0][0]

    active_tab  = st.session_state["bne_active_tab"]
    active_view = st.session_state["bne_active_view"]

    # ── Tab selector row ──────────────────────────────────────────────────────
    tab_cols = st.columns(len(_TABS))
    for col, (tab_key, tab_label) in zip(tab_cols, _TABS):
        with col:
            is_active = active_tab == tab_key
            # Highlight active tab with primary style; use CSS border trick
            label = f"**{tab_label}**" if is_active else tab_label
            if st.button(label, key=f"bne_tab_{tab_key}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state["bne_active_tab"] = tab_key
                st.rerun()

    st.markdown(
        f"<div style='border-bottom:2px solid {config.BORDER_COLOR};"
        f"margin-bottom:0.75rem'></div>",
        unsafe_allow_html=True,
    )

    # ── Visualization mode selector (hidden for Entity KG) ───────────────────
    if active_tab != "entity":
        st.markdown(
            "<div style='display:flex;align-items:center;gap:8px;"
            "margin-bottom:4px;font-size:12px;color:#6B7280'>"
            "<span style='font-weight:600'>Visualization mode:</span></div>",
            unsafe_allow_html=True,
        )
        mode_cols = st.columns(len(_VIEWS))
        for col, (view_key, view_label) in zip(mode_cols, _VIEWS):
            with col:
                is_active = active_view == view_key
                label = f"**{view_label}**" if is_active else view_label
                if st.button(label, key=f"bne_view_{view_key}", use_container_width=True,
                             type="primary" if is_active else "secondary"):
                    st.session_state["bne_active_view"] = view_key
                    st.rerun()

        st.markdown(
            "<div style='font-size:11px;color:#9CA3AF;margin-bottom:10px'>"
            "Network&nbsp;= cluster colors &nbsp;·&nbsp; "
            "Overlay&nbsp;= avg publication year (Viridis) &nbsp;·&nbsp; "
            "Density&nbsp;= Gaussian KDE heatmap (Viridis)"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Single conditional dispatch — ONE visualization rendered ─────────────
    if active_tab == "coauthorship":
        _render_coauth(active_view, coauth_graph, coauth_stats, papers_df)

    elif active_tab == "keyword":
        _render_keyword(active_view, keyword_graph, keyword_stats, papers_df)

    elif active_tab == "topic":
        _render_topic(active_view, topic_graph, topic_labels, papers_df)

    elif active_tab == "entity":
        _render_entity(kg_graph, kg, rels_df, papers_df)
