"""Home page — search, pipeline execution, session management."""

import gc
import logging
import time
from datetime import datetime
from typing import Callable, Optional

import pandas as pd
import streamlit as st

import config
from ui.components.cards import empty_state
from ui.components.loaders import live_stats_bar, progress_pipeline
from ui.components.metrics import pipeline_status_indicator

logger = logging.getLogger(__name__)



_PIPELINE_STEPS = [
    "Fetching Papers",
    "Parsing XML",
    "Cleaning Data",
    "Entity Extraction",
    "Building Embeddings",
    "Building Graphs",
]


@st.cache_data(ttl=300, show_spinner=False)
def _get_pubmed_count(query: str, year_start: int, year_end: int) -> Optional[int]:
    """
    Query PubMed with retmax=0 to retrieve the total hit count without
    fetching any records. Returns None on failure. Cached for 5 minutes.
    """
    try:
        from Bio import Entrez
        Entrez.email = config.ENTREZ_EMAIL
        if config.ENTREZ_API_KEY:
            Entrez.api_key = config.ENTREZ_API_KEY
        date_filter = f" AND {year_start}:{year_end}[dp]" if year_start and year_end else ""
        handle = Entrez.esearch(db="pubmed", term=query + date_filter, retmax=0, usehistory="n")
        record = Entrez.read(handle)
        handle.close()
        return int(record.get("Count", 0))
    except Exception as exc:
        logger.debug("PubMed count preview failed: %s", exc)
        return None


def render():
    """Render the Home / search page."""
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="bx-hero">
          <div class="bx-hero-title">BioLitAI-X</div>
          <div class="bx-hero-subtitle">
            From Literature to Discovery - 
            AI-Powered Biomedical Intelligence
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Search bar ────────────────────────────────────────────────────────────
    with st.container():
        query = st.text_input(
            "Search",
            placeholder="Enter any biomedical research query (e.g. CRISPR cancer therapy, "
                        "antibiotic resistance, Alzheimer's tau protein...)",
            label_visibility="collapsed",
            key="home_query_input",
        )

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            max_results = st.slider(
                "Max Results",
                min_value=config.MAX_RESULTS_MIN,
                max_value=config.MAX_RESULTS_MAX,
                value=config.MAX_RESULTS_DEFAULT,
                step=100,
                help="Papers to fetch from PubMed (up to 500).",
            )
        with col2:
            _cur_year = datetime.now().year
            year_min, year_max = st.slider(
                "Year Range",
                min_value=1990, max_value=_cur_year,
                value=(2000, _cur_year),
                help="Filter papers by publication year",
            )
        with col3:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            search_clicked = st.button(
                "🔍  Search & Analyse",
                width="stretch",
                type="primary",
            )

    # ── PubMed count preview + cache status ──────────────────────────────────
    if query and query.strip():
        q_stripped = query.strip()
        # Count preview (non-blocking — uses @st.cache_data with 5-min TTL)
        pubmed_count = _get_pubmed_count(q_stripped, year_min, year_max)
        if pubmed_count is not None:
            capped = min(pubmed_count, max_results)
            if pubmed_count > max_results:
                st.caption(
                    f"PubMed has **{pubmed_count:,}** papers matching this query "
                    f"({year_min}–{year_max}). Pipeline will fetch the most recent **{capped:,}**."
                )
            else:
                st.caption(
                    f"PubMed has **{pubmed_count:,}** papers matching this query "
                    f"({year_min}–{year_max}). All will be fetched."
                )


    # ── Past sessions chips ───────────────────────────────────────────────────
    past = st.session_state.get("past_sessions", [])
    if past:
        st.markdown(
            f"<div style='font-size:12px;color:{config.TEXT_SECONDARY};"
            f"margin:0.5rem 0 0.3rem;font-weight:600'>Past Queries</div>",
            unsafe_allow_html=True,
        )
        chip_cols = st.columns(min(len(past), 5))
        for i, sess in enumerate(past[-5:][::-1]):
            with chip_cols[i % len(chip_cols)]:
                q_short = sess.get("query_text", "")[:35]
                if st.button(
                    f"🕐 {q_short}",
                    key=f"chip_{sess.get('id')}",
                    help=sess.get("query_text", ""),
                ):
                    _restore_session(sess)
                    st.rerun()

    st.markdown(
        f"<hr style='border-color:{config.BORDER_COLOR};margin:1.5rem 0'>",
        unsafe_allow_html=True,
    )

    # ── Run pipeline on search ────────────────────────────────────────────────
    if search_clicked:
        q = (query or "").strip()
        if not q:
            st.error("Please enter a biomedical research query.")
            return
        _run_pipeline(q, max_results, year_min, year_max)

    # ── Show results summary if pipeline is complete ──────────────────────────
    elif st.session_state.get("pipeline_complete"):
        _show_results_summary()


# ── Pipeline orchestrator ─────────────────────────────────────────────────────


def _run_pipeline(
    query: str,
    max_results: int,
    year_min: int,
    year_max: int,
):
    """Execute the full pipeline: fetch → parse → clean → NLP → embed → graphs."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    from database.db_manager import DatabaseManager
    from pipeline.network_builder import NetworkBuilder
    from pipeline.knowledge_graph import KnowledgeGraph

    db = DatabaseManager(config.DB_PATH)
    session_id = db.save_query_session(query, max_results)
    db.update_query_session(session_id, pipeline_status="running")

    step_container = st.empty()
    stats_container = st.empty()
    status_msg      = st.empty()
    pipeline_start  = time.monotonic()
    _live: dict     = {"papers": 0, "ents": 0, "rels": 0}

    def _elapsed() -> str:
        secs = int(time.monotonic() - pipeline_start)
        return f"{secs // 60}m {secs % 60}s" if secs >= 60 else f"{secs}s"

    def _update(step: int, msg: str, papers: int = None, ents: int = None, rels: int = None):
        if papers is not None: _live["papers"] = papers
        if ents   is not None: _live["ents"]   = ents
        if rels   is not None: _live["rels"]   = rels
        with step_container.container():
            pipeline_status_indicator(step, 6, msg)
        with stats_container.container():
            live_stats_bar(_live["papers"], _live["ents"], _live["rels"])
        status_msg.markdown(
            f"<p style='font-size:12px;color:{config.TEXT_SECONDARY}'>"
            f"{msg} <span style='opacity:0.5'>({_elapsed()})</span></p>",
            unsafe_allow_html=True,
        )

    try:
        from pipeline.retrieval import PubMedRetriever
        from pipeline.parser import XMLParser
        from pipeline.cleaner import DataCleaner

        # Step 1: Fetch
        _update(1, "Fetching papers from PubMed...")
        retriever = PubMedRetriever()
        pmids, xml_batches = retriever.fetch_with_progress(
            query, max_results,
            progress_callback=lambda d, t, m: _update(1, m, papers=d),
        )

        # Step 2: Parse
        _update(2, "Parsing XML records...", papers=len(pmids))
        parser = XMLParser()
        raw_papers = []
        for xml_batch in xml_batches:
            raw_papers.extend(parser.parse_batch(xml_batch, query_used=query))
        del xml_batches
        _update(2, f"Parsed {len(raw_papers):,} records.", papers=len(raw_papers))

        # Step 3: Clean + year filter
        _update(3, f"Cleaning {len(raw_papers):,} papers...")
        cleaner = DataCleaner()
        papers_df = cleaner.run_full_pipeline(raw_papers)
        del raw_papers
        if "pub_year" in papers_df.columns:
            papers_df = papers_df[
                papers_df["pub_year"].fillna(0).astype(int).between(year_min, year_max)
                | papers_df["pub_year"].isna()
            ].reset_index(drop=True)
        _update(3, f"Cleaning complete: {len(papers_df):,} papers.", papers=len(papers_df))

        import concurrent.futures as _cf

        _n_papers = len(papers_df)

        # ── Step 4: Entity extraction ─────────────────────────────────────────
        # Runs in the main thread so progress callbacks update the UI live.
        # Running NLP and embeddings as threads caused PyTorch MKL/BLAS
        # initialisation conflicts (both use PyTorch internally) which made
        # the step appear stuck with no progress shown.
        _update(4, f"Extracting entities from {_n_papers:,} abstracts…",
                papers=_n_papers)
        entities_df      = pd.DataFrame()
        relationships_df = pd.DataFrame()
        try:
            from pipeline.nlp_processor import NLPProcessor
            _nlp_p = NLPProcessor(db_manager=None)
            _nlp_p.setup()
            entities_df, relationships_df = _nlp_p.process_corpus(
                papers_df,
                progress_callback=lambda e, r, m: _update(
                    4, m, papers=_n_papers, ents=e, rels=r),
            )
            del _nlp_p
            gc.collect()
        except Exception as exc:
            logger.warning("NLP skipped: %s", exc)
            st.warning(f"Entity extraction issue: {exc}")
        _update(4, f"Entities done: {len(entities_df):,} extracted.",
                papers=_n_papers, ents=len(entities_df),
                rels=len(relationships_df))

        # ── Step 5: Embeddings (runs after NLP to avoid PyTorch conflicts) ────
        _update(5, f"Building semantic embeddings for {_n_papers:,} papers…",
                papers=_n_papers, ents=len(entities_df),
                rels=len(relationships_df))
        embedder         = None
        embeddings_array = None
        try:
            from pipeline.embedder import EmbeddingEngine
            _emb = EmbeddingEngine()
            _emb.setup()
            embeddings_array = _emb.embed_corpus(papers_df, query=query)
            embedder = _emb
        except Exception as exc:
            logger.warning("Embeddings skipped: %s", exc)
            st.warning(f"Embedding issue: {exc}")
        _update(5, f"Embeddings ready ({len(embeddings_array):,} vectors)."
                   if embeddings_array is not None else "Embeddings skipped.",
                papers=_n_papers, ents=len(entities_df),
                rels=len(relationships_df))

        # ── Step 6: Bibliometric networks + Knowledge Graph ───────────────────
        _update(6, "Building networks and knowledge graph...",
                papers=_n_papers, ents=len(entities_df),
                rels=len(relationships_df))
        _nb1 = NetworkBuilder()
        _nb2 = NetworkBuilder()
        try:
            with _cf.ThreadPoolExecutor(max_workers=2) as _pool:
                _fc = _pool.submit(_nb1.build_coauthorship_network, papers_df)
                _fk = _pool.submit(_nb2.build_keyword_cooccurrence_network, papers_df)
                cog_full = _fc.result()
                kwd_full = _fk.result()
            with _cf.ThreadPoolExecutor(max_workers=4) as _pool:
                _fcs = _pool.submit(_nb1.calculate_network_statistics, cog_full)
                _fks = _pool.submit(_nb2.calculate_network_statistics, kwd_full)
                _fcd = _pool.submit(_nb1.prepare_graph_for_display, cog_full)
                _fkd = _pool.submit(_nb2.prepare_graph_for_display, kwd_full)
                coauth_stats  = _fcs.result()
                keyword_stats = _fks.result()
                coauth_graph  = _fcd.result()
                keyword_graph = _fkd.result()
        except Exception as exc:
            logger.warning("Parallel network build failed, retrying sequentially: %s", exc)
            try:
                cog_full      = _nb1.build_coauthorship_network(papers_df)
                kwd_full      = _nb1.build_keyword_cooccurrence_network(papers_df)
                coauth_stats  = _nb1.calculate_network_statistics(cog_full)
                keyword_stats = _nb1.calculate_network_statistics(kwd_full)
                coauth_graph  = _nb1.prepare_graph_for_display(cog_full)
                keyword_graph = _nb1.prepare_graph_for_display(kwd_full)
            except Exception as exc2:
                logger.warning("Network build failed: %s", exc2)
                coauth_graph = keyword_graph = None
                coauth_stats = keyword_stats = {}

        kg = KnowledgeGraph()
        kg_graph = kg.build_from_entities(entities_df, relationships_df)

        # Topics and relationship dep-parse are deferred to their respective pages.
        topic_graph      = None
        topic_labels     = {}
        topics_over_time = pd.DataFrame()
        doc_topics_df    = pd.DataFrame()
        gc.collect()

        db.update_query_session(session_id, papers_fetched=_n_papers,
                                pipeline_status="complete")
        total_elapsed = _elapsed()
        _update(7, f"Pipeline complete! ({total_elapsed})",
                papers=_n_papers, ents=len(entities_df),
                rels=len(relationships_df))
        st.success(
            f"✅ Pipeline complete in {total_elapsed}! "
            f"{_n_papers:,} papers · {len(entities_df):,} entities · embeddings ready. "
            f"Topics & relationships available on their pages."
        )

        st.session_state.update({
            "pipeline_complete":   True,
            "current_query":       query,
            "active_session_id":   session_id,
            "papers_df":           papers_df,
            "entities_df":         entities_df,
            "relationships_df":    relationships_df,
            "coauth_graph":        coauth_graph,
            "keyword_graph":       keyword_graph,
            "topic_graph":         topic_graph,
            "knowledge_graph":     kg,
            "kg_graph":            kg_graph,
            "coauth_stats":        coauth_stats,
            "keyword_stats":       keyword_stats,
            "topic_labels":        topic_labels,
            "topics_over_time":    topics_over_time,
            "doc_topics_df":       doc_topics_df,
            "embedder":            embedder,
            "embeddings_array":    embeddings_array,
            "db":                  db,
        })
        past = st.session_state.get("past_sessions", [])
        past.append({
            "id": session_id, "query_text": query,
            "pipeline_status": "complete", "papers_fetched": len(papers_df),
        })
        st.session_state["past_sessions"] = past
        # Force sidebar + page to reflect new pipeline state immediately
        st.rerun()

    except Exception as exc:
        db.update_query_session(session_id, pipeline_status="error")
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        st.error(f"Pipeline failed: {exc}")


def _restore_session(sess: dict):
    """Reload a past session's results from database."""
    from database.db_manager import DatabaseManager
    try:
        db = DatabaseManager(config.DB_PATH)
        query = sess.get("query_text", "")
        papers = db.get_papers_by_query(query)
        if papers:
            papers_df = pd.DataFrame(papers)
            st.session_state.update({
                "pipeline_complete": True,
                "current_query": query,
                "active_session_id": sess.get("id"),
                "papers_df": papers_df,
                "db": db,
            })
    except Exception as exc:
        st.error(f"Failed to restore session: {exc}")


def _show_results_summary():
    """Show a summary when pipeline has already run."""
    papers_df = st.session_state.get("papers_df")
    query = st.session_state.get("current_query", "")
    if papers_df is None or papers_df.empty:
        empty_state("📭", "No results loaded", "Run a search above to get started.")
        return

    st.success(
        f"✅ **{len(papers_df):,} papers** loaded for query: **{query}**. "
        f"Navigate using the sidebar to explore analysis, graphs, and hypotheses."
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Papers", f"{len(papers_df):,}")
    with col2:
        ents = st.session_state.get("entities_df")
        st.metric("Entities", f"{len(ents):,}" if ents is not None and not ents.empty else "—")
    with col3:
        rels = st.session_state.get("relationships_df")
        st.metric("Relationships Mapped",
                  f"{len(rels):,}" if rels is not None and not rels.empty else "—")
    with col4:
        hyps = st.session_state.get("hypotheses", [])
        st.metric("Hypotheses", str(len(hyps)))


# Avoid circular import
try:
    from database.db_manager import DatabaseManager
except ImportError:
    pass
