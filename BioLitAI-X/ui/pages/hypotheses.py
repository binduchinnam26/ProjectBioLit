"""AI Hypotheses panel page."""

import logging
import os

import streamlit as st

logger = logging.getLogger(__name__)
import config
from ui.components.cards import empty_state, hypothesis_card


def render():
    try:
        _render_inner()
    except Exception as exc:
        logger.exception("Hypotheses page crashed: %s", exc)
        st.error(f"⚠️ Page error: {exc}")
        st.info("Try refreshing the page. If the issue persists, return to Home and re-run the search.")


def _render_inner():
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

    # ── API key status banner ────────────────────────────────────────────────
    if not os.getenv("GEMINI_API_KEY"):
        st.info(
            "ℹ️ **GEMINI_API_KEY not configured** — running in offline mode. "
            "Hypotheses will be generated using template-based reasoning from "
            "the knowledge graph gaps. For full AI-powered hypotheses, add "
            "`GEMINI_API_KEY=your_key` to your `.env` file."
        )

    # ── Generate button ───────────────────────────────────────────────────────
    gen_col, filt_col = st.columns([2, 3])

    with gen_col:
        if st.button(
            "⚡ Generate New Hypotheses",
            type="primary",
            width="stretch",
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

    # ── Gap status (shown once, never blocks page load) ───────────────────────
    if gap_report:
        st.caption(f"✅ {len(gap_report)} research gaps ready as hypothesis input.")
    else:
        kg = st.session_state.get("knowledge_graph")
        if kg and kg.graph.number_of_nodes() > 0:
            st.caption("Research gaps will be detected automatically when you generate hypotheses.")
        else:
            st.caption(
                "No NLP knowledge graph available — keyword-based gaps will be used automatically."
            )

    # ── Display hypotheses ────────────────────────────────────────────────────
    # NOTE: No DB auto-load here — avoids SQLite lock wait when background
    # paper-write thread is still running. Session-state is the source of truth.

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


def _keyword_gap_fallback(papers_df) -> list:
    """
    Derive simple concept-pair gaps from top keyword co-occurrence when the
    NLP knowledge graph was not built.  Returns a gap_report-compatible list.
    """
    from collections import Counter, defaultdict
    from itertools import combinations

    keyword_counts: Counter = Counter()
    pair_counts: defaultdict = defaultdict(int)

    for _, row in papers_df.iterrows():
        kws = []
        for kw in (row.get("author_keywords") or []):
            if kw and len(kw) > 3:
                kws.append(kw.lower().strip())
        for mesh in (row.get("mesh_terms") or []):
            if isinstance(mesh, dict):
                desc = mesh.get("descriptor", "")
                if desc and len(desc) > 3:
                    kws.append(desc.lower().strip())
        kws = list(set(kws))[:15]
        for kw in kws:
            keyword_counts[kw] += 1
        for a, b in combinations(sorted(kws), 2):
            pair_counts[(a, b)] += 1

    # Keep only well-attested keywords
    min_freq = max(2, len(papers_df) // 50)
    top_kws = {kw for kw, c in keyword_counts.items() if c >= min_freq}

    # Candidate pairs: co-occur rarely relative to their individual frequency
    # → they represent a gap (both present but not often together)
    gap_pairs = []
    for (a, b), co_count in pair_counts.items():
        if a not in top_kws or b not in top_kws:
            continue
        freq_a = keyword_counts[a]
        freq_b = keyword_counts[b]
        expected = (freq_a * freq_b) / max(len(papers_df), 1)
        # Under-connected: co-occurrence far below expectation
        if co_count < max(2, expected * 0.3):
            gap_pairs.append({
                "concept_a":        a,
                "concept_b":        b,
                "gap_type":         "structural",
                "shared_neighbors": [],
                "score":            1.0 / max(co_count, 1),
            })

    gap_pairs.sort(key=lambda g: g["score"], reverse=True)
    return gap_pairs[:20]


def _generate_hypotheses(papers_df, entities_df, embedder, db, query, gap_report):
    """Run batch hypothesis generation with progress tracking."""
    if papers_df is None or papers_df.empty:
        st.error("No papers loaded.")
        return

    if not gap_report:
        # 1) Try KG-based gap detection first
        kg = st.session_state.get("knowledge_graph")
        if kg and kg.graph.number_of_nodes() > 0:
            try:
                from pipeline.gap_detector import GapDetector
                with st.spinner("Detecting research gaps from knowledge graph..."):
                    detector = GapDetector(knowledge_graph=kg)
                    gap_report = detector.compile_gap_report()
                if gap_report:
                    st.session_state["gap_report"] = gap_report
                    st.info(f"Detected {len(gap_report)} research gaps from knowledge graph.")
            except Exception as exc:
                logger.debug("KG gap detection failed: %s", exc)

        # 2) Fall back to keyword co-occurrence if KG gave nothing
        if not gap_report:
            gap_report = _keyword_gap_fallback(papers_df) if papers_df is not None else []
            if gap_report:
                st.session_state["gap_report"] = gap_report
                st.info(f"Using keyword-derived gaps ({len(gap_report)} pairs) as input.")
            else:
                st.error("No research gaps detected. Run the pipeline so entities can be extracted.")
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
