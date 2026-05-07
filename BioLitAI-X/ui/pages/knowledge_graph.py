"""Knowledge Graph page — VOSviewer-style bibliometric networks + entity KG."""

import streamlit as st
import config
from ui.components.cards import empty_state


def render():
    if not st.session_state.get("pipeline_complete"):
        empty_state(
            "🕸️",
            "No data loaded yet",
            "Complete a search on the Home page first.",
        )
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

    # ── Page header ───────────────────────────────────────────────────────────
    query = st.session_state.get("current_query", "")
    st.markdown(
        f"<h2 style='margin-bottom:2px'>Bibliometric Network Explorer</h2>"
        f"<p style='color:{config.TEXT_SECONDARY};font-size:13px;margin-bottom:1rem'>"
        f"Query: <b>{query}</b> &nbsp;·&nbsp; "
        f"VOSviewer-style cluster maps — node size ∝ weight · color = cluster</p>",
        unsafe_allow_html=True,
    )

    # ── Four tabs ─────────────────────────────────────────────────────────────
    tab_coauth, tab_kw, tab_topic, tab_entity = st.tabs([
        "👥  Co-authorship Network",
        "🔑  Keyword Co-occurrence",
        "📚  Topic Landscape",
        "🔬  Entity Knowledge Graph",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Co-authorship (main VOSviewer view)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_coauth:
        st.markdown(
            "<p style='font-size:13px;color:#6B7280;margin-bottom:6px'>"
            "Node&nbsp;size&nbsp;=&nbsp;publication count &nbsp;·&nbsp; "
            "Color&nbsp;=&nbsp;research cluster &nbsp;·&nbsp; "
            "Edge&nbsp;thickness&nbsp;=&nbsp;collaboration strength</p>",
            unsafe_allow_html=True,
        )

        if coauth_graph and coauth_graph.number_of_nodes() > 0:
            from visualization.network_viz import (
                render_coauthorship_network,
                render_network_stats,
                _build_controls_panel,
            )

            # Controls row
            st.markdown(
                "<div style='display:flex;gap:12px;align-items:center;"
                "margin-bottom:4px;font-size:12px;color:#6B7280'>"
                "<span>🔍 Search</span>"
                "<span>Min link strength</span>"
                "<span>Min publications</span>"
                "<span>Freeze</span>"
                "<span>All labels</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            controls = _build_controls_panel("coauth")

            graph_col, stats_col = st.columns([5, 1])
            with graph_col:
                render_coauthorship_network(coauth_graph, controls=controls, height=780)
            with stats_col:
                render_network_stats(coauth_stats, "Stats")

                # Cluster legend
                st.markdown(
                    f"<p style='font-size:11px;font-weight:600;"
                    f"color:{config.TEXT_SECONDARY};margin-top:12px;margin-bottom:4px'>"
                    f"CLUSTERS</p>",
                    unsafe_allow_html=True,
                )
                clusters_seen = sorted({
                    d.get("community", 0)
                    for _, d in coauth_graph.nodes(data=True)
                })
                for cid in clusters_seen[:15]:
                    color = config.COMMUNITY_COLORS[cid % len(config.COMMUNITY_COLORS)]
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:6px;"
                        f"margin-bottom:3px'>"
                        f"<div style='width:10px;height:10px;border-radius:50%;"
                        f"background:{color};flex-shrink:0'></div>"
                        f"<span style='font-size:11px;color:{config.TEXT_PRIMARY}'>"
                        f"Cluster {cid + 1}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info(
                "Co-authorship network not available. "
                "Run the pipeline on the Home page first."
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Keyword Co-occurrence
    # ══════════════════════════════════════════════════════════════════════════
    with tab_kw:
        st.markdown(
            "<p style='font-size:13px;color:#6B7280;margin-bottom:6px'>"
            "Node&nbsp;size&nbsp;=&nbsp;occurrence frequency &nbsp;·&nbsp; "
            "Color&nbsp;=&nbsp;thematic cluster &nbsp;·&nbsp; "
            "●&nbsp;Author &nbsp;■&nbsp;MeSH &nbsp;◆&nbsp;Chemical &nbsp;▲&nbsp;Qualifier</p>",
            unsafe_allow_html=True,
        )

        if keyword_graph and keyword_graph.number_of_nodes() > 0:
            from visualization.network_viz import (
                render_keyword_network,
                render_network_stats,
                _build_controls_panel,
            )
            import networkx as nx

            # Keyword type filter
            kw_types = sorted({
                d.get("keyword_type", "author")
                for _, d in keyword_graph.nodes(data=True)
            })
            selected_types = st.multiselect(
                "Keyword types to show",
                options=kw_types,
                default=kw_types,
                key="kg_kw_type_filter",
            )
            kw_filtered = keyword_graph.copy()
            if selected_types:
                remove = [
                    n for n, d in kw_filtered.nodes(data=True)
                    if d.get("keyword_type") not in selected_types
                ]
                kw_filtered.remove_nodes_from(remove)

            st.markdown(
                "<div style='display:flex;gap:12px;align-items:center;"
                "margin-bottom:4px;font-size:12px;color:#6B7280'>"
                "<span>🔍 Search</span>"
                "<span>Min co-occurrences</span>"
                "<span>Min frequency</span>"
                "<span>Freeze</span>"
                "<span>All labels</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            controls2 = _build_controls_panel("keyword")

            graph_col2, stats_col2 = st.columns([5, 1])
            with graph_col2:
                render_keyword_network(kw_filtered, controls=controls2, height=780)
            with stats_col2:
                render_network_stats(keyword_stats, "Stats")
        else:
            st.info("Keyword network not available. Run the pipeline first.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Topic Landscape
    # ══════════════════════════════════════════════════════════════════════════
    with tab_topic:
        st.markdown(
            "<p style='font-size:13px;color:#6B7280;margin-bottom:6px'>"
            "Node&nbsp;size&nbsp;=&nbsp;paper count &nbsp;·&nbsp; "
            "Edge&nbsp;=&nbsp;shared papers between topics</p>",
            unsafe_allow_html=True,
        )

        if topic_graph and topic_graph.number_of_nodes() > 0:
            from visualization.network_viz import render_topic_network, _build_controls_panel

            # Year filter if available
            yr_min = yr_max = None
            if papers_df is not None and "pub_year" in papers_df.columns:
                years = sorted(papers_df["pub_year"].dropna().astype(int).unique())
                if len(years) >= 2:
                    yr_min, yr_max = st.slider(
                        "Filter by publication year",
                        min_value=int(min(years)),
                        max_value=int(max(years)),
                        value=(int(min(years)), int(max(years))),
                        key="kg_topic_year",
                    )

            controls3 = _build_controls_panel("topic_kg")
            render_topic_network(
                topic_graph, controls=controls3, height=780,
                year_range=(yr_min, yr_max) if yr_min else None,
            )

            # Topic chips
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
                st.markdown(
                    f"<div style='margin-top:10px'>{chips}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info(
                "Topic landscape not available. "
                "NLP and embedding steps must complete first."
            )

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
                    f"<div style='font-size:12px;font-weight:600;"
                    f"color:{config.TEXT_SECONDARY};text-transform:uppercase;"
                    f"letter-spacing:.05em;margin-bottom:10px'>Graph Controls</div>",
                    unsafe_allow_html=True,
                )

                all_types = list(config.ENTITY_TYPE_COLORS.keys())
                selected_types = st.multiselect(
                    "Entity Types",
                    options=all_types,
                    default=all_types,
                    key="kg_entity_types",
                )

                if rels_df is not None and not rels_df.empty and "relationship_verb" in rels_df.columns:
                    all_rels = sorted(rels_df["relationship_verb"].dropna().unique().tolist())
                else:
                    all_rels = []
                selected_rels = (
                    st.multiselect(
                        "Relationship Types",
                        options=all_rels,
                        default=all_rels[:10],
                        key="kg_rel_types",
                    )
                    if all_rels else None
                )

                min_evidence = st.slider(
                    "Min evidence (papers)",
                    min_value=1, max_value=20, value=1, key="kg_min_ev",
                )
                search_entity = st.text_input(
                    "Search entity",
                    placeholder="Type entity name...",
                    key="kg_search",
                )
                show_gaps = st.toggle("Highlight Research Gaps", value=True, key="kg_show_gaps")

                if show_gaps and kg:
                    if not st.session_state.get("gap_report"):
                        with st.spinner("Detecting research gaps..."):
                            try:
                                gd = GapDetector(kg)
                                gd.find_structural_gaps()
                                gd.find_cross_domain_gaps()
                                report = gd.compile_gap_report()
                                st.session_state["gap_report"] = report
                                pairs = [
                                    (r["concept_a"], r["concept_b"])
                                    for r in report if r.get("concept_b")
                                ]
                                if pairs:
                                    kg.mark_gap_nodes(pairs)
                            except Exception as exc:
                                st.warning(f"Gap detection failed: {exc}")

                st.markdown("<hr>", unsafe_allow_html=True)
                render_entity_legend()

                st.markdown("<hr>", unsafe_allow_html=True)
                if kg_graph:
                    json_str = kg.export_to_json() if kg else "{}"
                    st.download_button(
                        "⬇ Export JSON",
                        data=json_str,
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

                display_rels = rels_df.copy()
                if filter_src:
                    display_rels = display_rels[
                        display_rels.get("subject_entity", pd.Series(dtype=str))
                        .str.contains(filter_src, case=False, na=False)
                    ]
                if filter_rel:
                    display_rels = display_rels[
                        display_rels.get("relationship_verb", pd.Series(dtype=str))
                        .str.contains(filter_rel, case=False, na=False)
                    ]
                render_relationship_evidence_table(display_rels, papers_df)
            else:
                st.info("No relationship data extracted yet.")
