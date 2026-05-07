"""AI Hypotheses panel page."""

import streamlit as st
import config
from ui.components.cards import empty_state, hypothesis_card


def render():
    if not st.session_state.get("pipeline_complete"):
        empty_state("💡", "No data loaded yet",
                    "Run a search on the Home page first.")
        return

    papers_df   = st.session_state.get("papers_df")
    entities_df = st.session_state.get("entities_df")
    embedder    = st.session_state.get("embedder")
    db          = st.session_state.get("db")
    query       = st.session_state.get("current_query", "")
    gap_report  = st.session_state.get("gap_report", [])
    hypotheses  = st.session_state.get("hypotheses", [])

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f"<h2 style='margin-bottom:4px'>AI Research Hypotheses</h2>"
        f"<p style='color:{config.TEXT_SECONDARY};font-size:13px;margin-bottom:1rem'>"
        f"Generated from knowledge graph gaps, grounded in retrieved literature evidence.</p>",
        unsafe_allow_html=True,
    )

    # ── Generate button ───────────────────────────────────────────────────────
    gen_col, filt_col = st.columns([2, 3])

    with gen_col:
        if st.button(
            "⚡ Generate New Hypotheses",
            type="primary",
            use_container_width=True,
            key="gen_hyp_btn",
        ):
            _generate_hypotheses(papers_df, entities_df, embedder, db, query, gap_report)

    with filt_col:
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            conf_filter = st.multiselect(
                "Confidence",
                ["High", "Medium", "Low"],
                default=["High", "Medium", "Low"],
                key="hyp_conf_filter",
            )
        with fc2:
            sort_by = st.selectbox(
                "Sort by",
                ["Confidence", "Novelty", "Date"],
                key="hyp_sort",
            )
        with fc3:
            gap_type_filter = st.multiselect(
                "Gap type",
                ["structural", "cross_domain", "temporal"],
                default=["structural", "cross_domain", "temporal"],
                key="hyp_gap_type",
            )

    st.markdown(
        f"<hr style='border-color:{config.BORDER_COLOR};margin:1rem 0'>",
        unsafe_allow_html=True,
    )

    # ── If no gaps detected yet, run gap detection ────────────────────────────
    if not gap_report:
        kg = st.session_state.get("knowledge_graph")
        if kg and kg.graph.number_of_nodes() > 0:
            with st.spinner("Detecting research gaps..."):
                try:
                    from pipeline.gap_detector import GapDetector
                    gd = GapDetector(kg)
                    gd.find_structural_gaps()
                    gd.find_cross_domain_gaps()
                    gd.find_temporal_gaps(papers_df)
                    gap_report = gd.compile_gap_report()
                    st.session_state["gap_report"] = gap_report
                except Exception as exc:
                    st.warning(f"Gap detection failed: {exc}")
        else:
            st.info(
                "Knowledge graph is empty. NLP processing must complete to detect gaps. "
                "You can still generate hypotheses once the graph is available."
            )

    # ── Display hypotheses ────────────────────────────────────────────────────
    if not hypotheses:
        # Try loading from database
        if db and query:
            try:
                hypotheses = db.get_hypotheses_by_query(query)
                st.session_state["hypotheses"] = hypotheses
            except Exception:
                pass

    if not hypotheses:
        st.markdown(
            f"""
            <div style="text-align:center;padding:3rem 2rem;background:{config.SURFACE_ELEVATED};
                        border:1px solid {config.BORDER_COLOR};border-radius:8px;margin:1rem 0">
              <div style="font-size:2.5rem;margin-bottom:1rem">💡</div>
              <div style="font-size:1.1rem;font-weight:600;color:{config.TEXT_PRIMARY};margin-bottom:0.5rem">
                No hypotheses generated yet
              </div>
              <div style="font-size:13px;color:{config.TEXT_SECONDARY}">
                Click "Generate New Hypotheses" above to have Gemini analyse the research gaps.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Apply filters
    filtered = [
        h for h in hypotheses
        if h.get("confidence_label", "Low") in conf_filter
        and h.get("gap_type", "structural") in gap_type_filter
    ]

    # Sort
    if sort_by == "Confidence":
        filtered.sort(key=lambda h: h.get("confidence_score", 0), reverse=True)
    elif sort_by == "Date":
        filtered.sort(key=lambda h: h.get("created_at", ""), reverse=True)

    st.markdown(
        f"<p style='font-size:13px;color:{config.TEXT_SECONDARY};"
        f"margin-bottom:1rem'>{len(filtered)} hypothesis{'es' if len(filtered)!=1 else ''} shown</p>",
        unsafe_allow_html=True,
    )

    # Two-column grid
    if filtered:
        col_a, col_b = st.columns(2)
        for i, hyp in enumerate(filtered):
            with (col_a if i % 2 == 0 else col_b):
                hypothesis_card(hyp)
    else:
        st.info("No hypotheses match the current filters.")


def _generate_hypotheses(papers_df, entities_df, embedder, db, query, gap_report):
    """Run batch hypothesis generation with progress tracking."""
    if papers_df is None or papers_df.empty:
        st.error("No papers loaded.")
        return

    if not gap_report:
        st.error("No research gaps detected. Ensure NLP and knowledge graph steps completed.")
        return

    progress = st.progress(0.0)
    status = st.empty()

    def _cb(done: int, total: int, msg: str):
        if total > 0:
            progress.progress(done / total)
        status.markdown(
            f"<p style='font-size:12px;color:{config.TEXT_SECONDARY}'>{msg}</p>",
            unsafe_allow_html=True,
        )

    try:
        from pipeline.hypothesis_generator import HypothesisGenerator
        gen = HypothesisGenerator(db_manager=db, embedding_engine=embedder)
        gen.setup()

        hypotheses = gen.generate_batch_hypotheses(
            gap_report=gap_report,
            papers_df=papers_df,
            entities_df=entities_df,
            query_used=query,
            progress_callback=_cb,
        )
        st.session_state["hypotheses"] = hypotheses
        progress.progress(1.0)
        status.success(f"✅ {len(hypotheses)} hypotheses generated.")
        st.rerun()

    except Exception as exc:
        st.error(f"Hypothesis generation failed: {exc}")
