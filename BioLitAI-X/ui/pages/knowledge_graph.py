"""
Knowledge Graph page — VOSviewer-style bibliometric networks.

Three visualization modes (matching VOSviewer):
  Network Visualization  — cluster-colored nodes, fixed kamada-kawai layout.
  Overlay Visualization  — same positions, Viridis color by avg publication year.
  Density Visualization  — Gaussian KDE heatmap (D = Σ wᵢ·K(‖x−xᵢ‖/d̄h)),
                           Viridis, white node circles on top.

Four network types: Co-authorship · Keyword Co-occurrence · Topic · Entity KG.
"""

import streamlit as st
import config
from ui.components.cards import empty_state


# ── Mode selector widget ──────────────────────────────────────────────────────

def _viz_mode_selector(key: str) -> str:
    """
    Render the three VOSviewer visualization-mode buttons.
    Returns one of 'network', 'overlay', 'density'.
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


# ── Controls hint row ─────────────────────────────────────────────────────────

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


# ── Cluster legend widget ─────────────────────────────────────────────────────

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
    st.markdown(
        f"<h2 style='margin-bottom:2px'>Bibliometric Network Explorer</h2>"
        f"<p style='color:{config.TEXT_SECONDARY};font-size:13px;margin-bottom:0.8rem'>"
        f"Query: <b>{query}</b> &nbsp;·&nbsp; "
        f"Three visualization modes — Network / Overlay / Density — "
        f"matching VOSviewer 1.6.x</p>",
        unsafe_allow_html=True,
    )

    # ── Four network-type tabs ────────────────────────────────────────────────
    tab_coauth, tab_kw, tab_topic, tab_entity = st.tabs([
        "👥  Co-authorship",
        "🔑  Keyword Co-occurrence",
        "📚  Topic Landscape",
        "🔬  Entity Knowledge Graph",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Co-authorship
    # ══════════════════════════════════════════════════════════════════════════
    with tab_coauth:
        if not coauth_graph or coauth_graph.number_of_nodes() == 0:
            st.info("Co-authorship network not available. Run the pipeline first.")
        else:
            from visualization.network_viz import (
                render_coauthorship_network,
                render_overlay_visualization,
                render_density_visualization,
                render_network_stats,
                _build_controls_panel,
            )

            mode = _viz_mode_selector("coauth")

            if mode == "network":
                _controls_hint()
                ctrl = _build_controls_panel("coauth")
                graph_col, stats_col = st.columns([5, 1])
                with graph_col:
                    render_coauthorship_network(coauth_graph, controls=ctrl, height=780)
                with stats_col:
                    render_network_stats(coauth_stats, "Stats")
                    _cluster_legend(coauth_graph)

            elif mode == "overlay":
                st.markdown(
                    "<div style='font-size:12px;color:#6B7280;margin-bottom:4px'>"
                    "Color encodes the <b>average publication year</b> per author. "
                    "Positions are identical to the Network view.</div>",
                    unsafe_allow_html=True,
                )
                _controls_hint()
                ctrl = _build_controls_panel("ov_coauth")
                render_overlay_visualization(
                    coauth_graph, papers_df, "coauth", ctrl, height=780,
                )

            else:  # density
                st.markdown(
                    "<div style='font-size:12px;color:#6B7280;margin-bottom:4px'>"
                    "Heatmap computed with the VOSviewer kernel formula "
                    "<i>D(x) = Σ wᵢ·exp(−‖x−xᵢ‖²/2(d̄·h)²)</i>. "
                    "Adjust <b>Kernel width (h)</b> to broaden or sharpen clusters.</div>",
                    unsafe_allow_html=True,
                )
                _controls_hint(show_bandwidth=True)
                ctrl = _build_controls_panel("dn_coauth", show_bandwidth=True)
                render_density_visualization(
                    coauth_graph, "coauth", ctrl, height=780,
                )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Keyword Co-occurrence
    # ══════════════════════════════════════════════════════════════════════════
    with tab_kw:
        if not keyword_graph or keyword_graph.number_of_nodes() == 0:
            st.info("Keyword network not available. Run the pipeline first.")
        else:
            from visualization.network_viz import (
                render_keyword_network,
                render_overlay_visualization,
                render_density_visualization,
                render_network_stats,
                _build_controls_panel,
            )
            import networkx as nx

            # Keyword type filter (only relevant for network view, applied before all modes)
            kw_types = sorted({d.get("keyword_type", "author")
                               for _, d in keyword_graph.nodes(data=True)})
            selected_types = st.multiselect(
                "Keyword types", options=kw_types, default=kw_types, key="kg_kw_types",
            )
            kw_g = keyword_graph.copy()
            if selected_types:
                kw_g.remove_nodes_from([
                    n for n, d in kw_g.nodes(data=True)
                    if d.get("keyword_type") not in selected_types
                ])

            st.markdown(
                "<div style='font-size:12px;color:#6B7280;margin-bottom:4px'>"
                "●&nbsp;Author&nbsp; ■&nbsp;MeSH&nbsp; ◆&nbsp;Chemical&nbsp; ▲&nbsp;Qualifier</div>",
                unsafe_allow_html=True,
            )

            mode = _viz_mode_selector("kw")

            if mode == "network":
                _controls_hint()
                ctrl = _build_controls_panel("keyword")
                graph_col, stats_col = st.columns([5, 1])
                with graph_col:
                    render_keyword_network(kw_g, controls=ctrl, height=780)
                with stats_col:
                    render_network_stats(keyword_stats, "Stats")
                    _cluster_legend(kw_g)

            elif mode == "overlay":
                st.markdown(
                    "<div style='font-size:12px;color:#6B7280;margin-bottom:4px'>"
                    "Color encodes the <b>average publication year</b> in which each keyword appears.</div>",
                    unsafe_allow_html=True,
                )
                _controls_hint()
                ctrl = _build_controls_panel("ov_keyword")
                render_overlay_visualization(
                    kw_g, papers_df, "keyword", ctrl, height=780,
                )

            else:  # density
                st.markdown(
                    "<div style='font-size:12px;color:#6B7280;margin-bottom:4px'>"
                    "Keyword density — shows thematic hotspots in the research landscape.</div>",
                    unsafe_allow_html=True,
                )
                _controls_hint(show_bandwidth=True)
                ctrl = _build_controls_panel("dn_keyword", show_bandwidth=True)
                render_density_visualization(
                    kw_g, "keyword", ctrl, height=780,
                )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Topic Landscape
    # ══════════════════════════════════════════════════════════════════════════
    with tab_topic:
        if not topic_graph or topic_graph.number_of_nodes() == 0:
            st.info("Topic landscape not available. NLP and embedding steps must complete first.")
        else:
            from visualization.network_viz import render_topic_network, _build_controls_panel

            st.markdown(
                "<p style='font-size:13px;color:#6B7280;margin-bottom:6px'>"
                "Node&nbsp;size&nbsp;=&nbsp;paper count &nbsp;·&nbsp; "
                "Edge&nbsp;=&nbsp;shared papers between topics &nbsp;·&nbsp; "
                "Network visualization mode only.</p>",
                unsafe_allow_html=True,
            )

            # Year filter
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

            _controls_hint()
            ctrl = _build_controls_panel("topic_kg")
            render_topic_network(
                topic_graph, controls=ctrl, height=780,
                year_range=(yr_min, yr_max) if yr_min else None,
            )

            if topic_labels:
                chips = " ".join(
                    f"<span style='display:inline-block;"
                    f"background:{config.COMMUNITY_COLORS[i % len(config.COMMUNITY_COLORS)]}22;"
                    f"border:1px solid {config.COMMUNITY_COLORS[i % len(config.COMMUNITY_COLORS)]}66;"
                    f"color:{config.COMMUNITY_COLORS[i % len(config.COMMUNITY_COLORS)]};"
                    f"border-radius:20px;padding:2px 10px;font-size:11px;margin:3px'>"
                    f"{v['label']} ({v['count']})</span>"
                    for i, (_, v) in enumerate(
                        sorted(topic_labels.items(),
                               key=lambda x: x[1]["count"], reverse=True)[:16]
                    )
                )
                st.markdown(f"<div style='margin-top:10px'>{chips}</div>",
                            unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — Entity Knowledge Graph (NLP-derived)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_entity:
        if kg_graph is None or kg_graph.number_of_nodes() == 0:
            st.info(
                "Entity knowledge graph is empty. "
                "NLP processing must complete successfully to extract entities."
            )
        else:
            from visualization.graph_viz import (
                render_knowledge_graph,
                render_entity_legend,
                render_relationship_evidence_table,
            )
            from pipeline.gap_detector import GapDetector

            st.markdown(
                "<p style='font-size:13px;color:#6B7280;margin-bottom:6px'>"
                "Node&nbsp;size&nbsp;=&nbsp;evidence strength &nbsp;·&nbsp; "
                "Color&nbsp;=&nbsp;entity type &nbsp;·&nbsp; "
                "Arrows&nbsp;=&nbsp;relationship direction</p>",
                unsafe_allow_html=True,
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

            # ── Relationship evidence table ────────────────────────────────
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
