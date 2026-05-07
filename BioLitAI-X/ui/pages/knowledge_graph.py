"""Knowledge Graph explorer page."""

import streamlit as st
import config
from ui.components.cards import empty_state
from utils.helpers import pubmed_url


def render():
    if not st.session_state.get("pipeline_complete"):
        empty_state("🕸️", "Knowledge graph not built yet",
                    "Complete a search on the Home page first.")
        return

    kg      = st.session_state.get("knowledge_graph")
    kg_graph = st.session_state.get("kg_graph")
    papers_df = st.session_state.get("papers_df")
    rels_df   = st.session_state.get("relationships_df")

    if kg_graph is None or kg_graph.number_of_nodes() == 0:
        empty_state(
            "🔬",
            "Knowledge graph is empty",
            "NLP processing must complete successfully to extract entities and relationships.",
        )
        return

    from visualization.graph_viz import (
        render_knowledge_graph,
        render_entity_legend,
        render_relationship_evidence_table,
    )
    from pipeline.gap_detector import GapDetector

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f"<h2 style='margin-bottom:4px'>Biomedical Knowledge Graph</h2>"
        f"<p style='color:{config.TEXT_SECONDARY};font-size:13px;margin-bottom:1.5rem'>"
        f"Node size = evidence strength · Color = entity type · Arrows = relationship direction</p>",
        unsafe_allow_html=True,
    )

    # ── Layout: left control panel + main canvas ──────────────────────────────
    ctrl_col, graph_col = st.columns([1, 4])

    with ctrl_col:
        st.markdown(
            f"<div style='font-size:12px;font-weight:600;color:{config.TEXT_SECONDARY};"
            f"text-transform:uppercase;letter-spacing:.05em;margin-bottom:10px'>"
            f"Graph Controls</div>",
            unsafe_allow_html=True,
        )

        # Entity type filter
        all_types = list(config.ENTITY_TYPE_COLORS.keys())
        selected_types = st.multiselect(
            "Entity Types",
            options=all_types,
            default=all_types,
            key="kg_entity_types",
        )

        # Relationship type filter
        if rels_df is not None and not rels_df.empty and "relationship_verb" in rels_df.columns:
            all_rels = sorted(rels_df["relationship_verb"].dropna().unique().tolist())
        else:
            all_rels = []
        if all_rels:
            selected_rels = st.multiselect(
                "Relationship Types",
                options=all_rels,
                default=all_rels[:10],
                key="kg_rel_types",
            )
        else:
            selected_rels = None

        depth = st.slider("Exploration Depth", min_value=1, max_value=3, value=2, key="kg_depth")

        search_entity = st.text_input(
            "Search entity",
            placeholder="Type entity name...",
            key="kg_search",
        )

        min_evidence = st.slider(
            "Min evidence (papers)",
            min_value=1, max_value=20, value=1, key="kg_min_ev",
        )

        show_gaps = st.toggle("Highlight Research Gaps", value=True, key="kg_show_gaps")

        # Run gap detection if requested
        if show_gaps and kg:
            if "gap_report" not in st.session_state or not st.session_state.get("gap_report"):
                with st.spinner("Detecting research gaps..."):
                    try:
                        gd = GapDetector(kg)
                        structural = gd.find_structural_gaps()
                        crossdomain = gd.find_cross_domain_gaps()
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
        # Export buttons
        if kg_graph:
            json_str = kg.export_to_json() if kg else "{}"
            st.download_button(
                "⬇ Export as JSON",
                data=json_str,
                file_name="knowledge_graph.json",
                mime="application/json",
                key="kg_export_json",
            )
            if rels_df is not None and not rels_df.empty:
                st.download_button(
                    "⬇ Export relationships CSV",
                    data=rels_df.to_csv(index=False),
                    file_name="relationships.csv",
                    mime="text/csv",
                    key="kg_export_csv",
                )

    with graph_col:
        rel_type_set = set(selected_rels) if selected_rels else None
        render_knowledge_graph(
            G=kg_graph,
            highlight_entity=search_entity or None,
            entity_type_filter=set(selected_types),
            relationship_type_filter=rel_type_set,
            min_evidence=min_evidence,
            show_gap_nodes=show_gaps,
            search_term=search_entity,
            height=800,
        )

    # ── Relationship evidence table ────────────────────────────────────────────
    st.markdown(
        f"<hr style='border-color:{config.BORDER_COLOR};margin:2rem 0'>",
        unsafe_allow_html=True,
    )
    st.markdown("### Relationship Evidence", unsafe_allow_html=False)

    if rels_df is not None and not rels_df.empty:
        # Filters above table
        fc1, fc2 = st.columns(2)
        with fc1:
            filter_src = st.text_input("Filter source entity", key="rel_filter_src")
        with fc2:
            filter_rel = st.text_input("Filter relationship type", key="rel_filter_rel")

        display_rels = rels_df.copy()
        if filter_src:
            display_rels = display_rels[
                display_rels.get("subject_entity", pd.Series()).str.contains(
                    filter_src, case=False, na=False
                )
            ]
        if filter_rel:
            display_rels = display_rels[
                display_rels.get("relationship_verb", pd.Series()).str.contains(
                    filter_rel, case=False, na=False
                )
            ]
        render_relationship_evidence_table(display_rels, papers_df)
    else:
        st.info("No relationship data extracted yet.")


try:
    import pandas as pd
except ImportError:
    pass
