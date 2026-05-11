"""Home page — search, pipeline execution, session management."""

import gc
import logging
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import streamlit as st

import config
from ui.components.cards import empty_state
from ui.components.loaders import live_stats_bar, progress_pipeline
from ui.components.metrics import pipeline_status_indicator
from utils.helpers import query_hash

logger = logging.getLogger(__name__)

_DISK_WARN_MB = 500    # warn when free space < 500 MB
_DISK_MIN_MB  = 100    # disable all disk writes below 100 MB


def _free_mb() -> int:
    """Return free disk space in MB on the data directory's partition."""
    try:
        return shutil.disk_usage(config.DATA_DIR).free // (1024 * 1024)
    except Exception:
        return 9999  # can't check → assume OK


def _auto_cleanup_disk() -> int:
    """
    Delete old parquet caches, FAISS indexes, derived pickles and compact
    the SQLite WAL file to reclaim disk space. Returns MB freed (approximate).
    Safe to call at any time — only removes re-generatable cache files.
    """
    import sqlite3 as _sqlite3
    freed = 0
    for pattern in ("*.parquet", "*_derived.pkl"):
        for p in Path(config.PROCESSED_DIR).glob(pattern):
            try:
                freed += p.stat().st_size
                p.unlink()
            except Exception:
                pass
    for pattern in ("*.faiss", "*.json", "*.npy"):
        for p in Path(config.EMBEDDINGS_DIR).glob(pattern):
            try:
                freed += p.stat().st_size
                p.unlink()
            except Exception:
                pass
    # Checkpoint + shrink the SQLite WAL file
    try:
        conn = _sqlite3.connect(config.DB_PATH, timeout=10)
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("VACUUM")
        conn.close()
    except Exception:
        pass
    return freed // (1024 * 1024)

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

        # Cache status banner
        qhash = query_hash(q_stripped)
        _cache_path = Path(config.PROCESSED_DIR) / f"{qhash}.parquet"
        if _cache_path.exists():
            _age_days = (time.time() - _cache_path.stat().st_mtime) / 86400
            try:
                import pyarrow.parquet as _pq
                _cached_n = _pq.read_metadata(str(_cache_path)).num_rows
                _n_str = f"{_cached_n:,} papers"
            except Exception:
                _n_str = "unknown size"
            if _age_days < 1:
                _age_str = "today"
            elif _age_days < 2:
                _age_str = "yesterday"
            else:
                _age_str = f"{int(_age_days)} days ago"
            st.info(
                f"✅ **Cached results available** — {_n_str}, saved {_age_str}. "
                f"Click **Search & Analyse** to load from cache."
            )
        else:
            st.caption("🔄 No cache found — full pipeline will run.")

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

def _load_papers_parquet(cache_path: Path) -> pd.DataFrame:
    """Load papers from parquet cache and restore list columns."""
    import json as _json
    df = pd.read_parquet(cache_path)
    for col in ("authors", "author_keywords", "mesh_terms",
                "chemicals", "publication_types", "grants"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: _json.loads(v) if isinstance(v, str) else (v if isinstance(v, list) else [])
            )
    return df


def _save_papers_parquet(papers_df: pd.DataFrame, cache_path: Path):
    """Save papers_df to parquet, serialising list columns as JSON strings."""
    import json as _json
    cache_df = papers_df.copy()
    for col in ("authors", "author_keywords", "mesh_terms",
                "chemicals", "publication_types", "grants"):
        if col in cache_df.columns:
            cache_df[col] = cache_df[col].apply(
                lambda v: _json.dumps(v) if not isinstance(v, str) else v
            )
    cache_df.to_parquet(cache_path, index=False)


def _run_pipeline(
    query: str,
    max_results: int,
    year_min: int,
    year_max: int,
):
    """
    Execute the full pipeline with two modes:

    FAST PATH  — all caches present:
        Loads papers (parquet) + derived results (pickle) + embeddings (npy/faiss).
        Skips all heavy compute. Typically completes in 10–30 seconds.

    SLOW PATH  — first run:
        Runs full pipeline (fetch → NLP → embed → topics → graphs).
        Saves all results to cache at the end so next run is instant.
    """
    import pickle
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    from database.db_manager import DatabaseManager
    from pipeline.network_builder import NetworkBuilder
    from pipeline.knowledge_graph import KnowledgeGraph

    # ── Disk space pre-check ──────────────────────────────────────────────────
    free_mb = _free_mb()
    if free_mb < _DISK_WARN_MB:
        freed_mb = _auto_cleanup_disk()
        free_mb  = _free_mb()
        if freed_mb > 0:
            st.info(f"🧹 Auto-cleaned {freed_mb} MB of old cache files. {free_mb} MB now free.")
    low_disk = free_mb < _DISK_WARN_MB
    no_disk  = free_mb < _DISK_MIN_MB
    if no_disk:
        st.error(f"❌ Critical: only {free_mb} MB free. Pipeline runs in memory-only mode.")
    elif low_disk:
        st.warning(f"⚠️ Low disk: {free_mb} MB. Cache writes skipped.")

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

    qhash           = query_hash(query)
    papers_cache    = Path(config.PROCESSED_DIR) / f"{qhash}.parquet"
    derived_cache   = Path(config.PROCESSED_DIR) / f"{qhash}_derived.pkl"
    use_full_cache  = papers_cache.exists() and derived_cache.exists()

    try:
        # ══════════════════════════════════════════════════════════════════════
        # FAST PATH — load everything from disk, skip all heavy compute
        # ══════════════════════════════════════════════════════════════════════
        if use_full_cache:
            _update(1, "Loading papers from cache...")
            papers_df = _load_papers_parquet(papers_cache)
            _update(2, f"Loading derived results for {len(papers_df):,} papers...",
                    papers=len(papers_df))

            with open(derived_cache, "rb") as _f:
                _d = pickle.load(_f)

            entities_df      = _d.get("entities_df",      pd.DataFrame())
            relationships_df = _d.get("relationships_df", pd.DataFrame())
            coauth_graph     = _d.get("coauth_graph")
            keyword_graph    = _d.get("keyword_graph")
            kg_graph         = _d.get("kg_graph")
            topic_graph      = _d.get("topic_graph")
            coauth_stats     = _d.get("coauth_stats",     {})
            keyword_stats    = _d.get("keyword_stats",    {})
            doc_topics_df    = _d.get("doc_topics_df",    pd.DataFrame())
            topic_labels     = _d.get("topic_labels",     {})
            topics_over_time = _d.get("topics_over_time", pd.DataFrame())
            del _d

            _update(3, "Loading embeddings from cache...",
                    papers=len(papers_df), ents=len(entities_df), rels=len(relationships_df))

            embedder = None
            embeddings_array = None
            try:
                from pipeline.embedder import EmbeddingEngine
                embedder = EmbeddingEngine(persist_index=not low_disk)
                embedder.setup()
                if embedder.index_exists(query):
                    embedder.load_index(query)
                    embeddings_array = embedder.load_embeddings(query)
            except Exception as exc:
                logger.warning("Embedding load skipped: %s", exc)

            # Rebuild KnowledgeGraph wrapper around the cached NetworkX graph
            kg = KnowledgeGraph()
            if kg_graph is not None:
                kg.graph = kg_graph

            db.update_query_session(session_id, papers_fetched=len(papers_df),
                                    pipeline_status="complete")
            total_elapsed = _elapsed()
            # step=7 marks all 6 steps as done (✅)
            _update(7, f"Loaded from cache in {total_elapsed}!",
                    papers=len(papers_df), ents=len(entities_df), rels=len(relationships_df))
            st.success(
                f"⚡ Loaded from cache in {total_elapsed}! "
                f"{len(papers_df):,} papers, {len(entities_df):,} entities. "
                f"Use the sidebar to explore results."
            )

        # ══════════════════════════════════════════════════════════════════════
        # SLOW PATH — full pipeline; caches saved at end for instant future loads
        # ══════════════════════════════════════════════════════════════════════
        else:
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

            # Save papers to parquet
            if not low_disk:
                try:
                    _save_papers_parquet(papers_df, papers_cache)
                except Exception as exc:
                    logger.warning("Parquet cache save failed: %s", exc)

            import concurrent.futures as _cf

            _n_papers = len(papers_df)

            # ── Step 4: Entity extraction (NER, dep parser disabled) ──────────
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

            # ── Step 5: Embeddings (runs after NLP to avoid PyTorch conflicts) ─
            _update(5, f"Building semantic embeddings for {_n_papers:,} papers…",
                    papers=_n_papers, ents=len(entities_df),
                    rels=len(relationships_df))
            embedder         = None
            embeddings_array = None
            try:
                from pipeline.embedder import EmbeddingEngine
                _emb = EmbeddingEngine(persist_index=not low_disk)
                _emb.setup()
                if _emb.index_exists(query):
                    _emb.load_index(query)
                    embeddings_array = _emb.load_embeddings(query)
                    if embeddings_array is None:
                        embeddings_array = _emb.embed_corpus(papers_df, query=query)
                else:
                    embeddings_array = _emb.embed_corpus(papers_df, query=query)
                embedder = _emb
            except Exception as exc:
                logger.warning("Embeddings skipped: %s", exc)
                st.warning(f"Embedding issue: {exc}")
            _update(5, f"Embeddings ready ({len(embeddings_array):,} vectors)."
                       if embeddings_array is not None else "Embeddings skipped.",
                    papers=_n_papers, ents=len(entities_df),
                    rels=len(relationships_df))

            # ── Step 6: Bibliometric networks + basic Knowledge Graph ─────────
            _update(6, "Building networks and knowledge graph...",
                    papers=_n_papers, ents=len(entities_df),
                    rels=len(relationships_df))
            _nb1 = NetworkBuilder()
            _nb2 = NetworkBuilder()
            try:
                # Build both networks in parallel — separate instances, no shared state
                with _cf.ThreadPoolExecutor(max_workers=2) as _pool:
                    _fc = _pool.submit(_nb1.build_coauthorship_network, papers_df)
                    _fk = _pool.submit(_nb2.build_keyword_cooccurrence_network, papers_df)
                    cog_full = _fc.result()
                    kwd_full = _fk.result()
                # Stats + display-prep are read-only on finished graphs → run 4 in parallel
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

            # Topics and relationship dep-parse are still deferred (too slow for main pipeline):
            # • Topics        → Analysis page → Topic Evolution tab
            # • Relationships → Knowledge Graph page → Entity KG tab
            topic_graph      = None
            topic_labels     = {}
            topics_over_time = pd.DataFrame()
            doc_topics_df    = pd.DataFrame()
            gc.collect()

            # ── Step 6: Save derived cache in background (non-blocking) ─────────
            # Runs in a daemon thread so the pipeline completes immediately while
            # the cache is written in parallel. Next run will use the FAST PATH.
            if not no_disk and not low_disk:
                _derived = {
                    "entities_df":      entities_df,
                    "relationships_df": relationships_df,
                    "coauth_graph":     coauth_graph,
                    "keyword_graph":    keyword_graph,
                    "kg_graph":         kg_graph,
                    "topic_graph":      topic_graph,
                    "coauth_stats":     coauth_stats,
                    "keyword_stats":    keyword_stats,
                    "doc_topics_df":    doc_topics_df,
                    "topic_labels":     topic_labels,
                    "topics_over_time": topics_over_time,
                }
                def _bg_save_derived(_d=_derived, _p=derived_cache):
                    try:
                        with open(_p, "wb") as _f:
                            pickle.dump(_d, _f, protocol=pickle.HIGHEST_PROTOCOL)
                        logger.info("Derived cache saved: %s", _p)
                    except Exception as _e:
                        logger.warning("Derived cache save failed: %s", _e)
                threading.Thread(target=_bg_save_derived, daemon=True).start()

            # DB paper writes run in background — only needed for session history
            if not no_disk:
                def _bg_db_write(_df=papers_df, _db=db):
                    try:
                        _db.insert_papers_batch(_df.to_dict("records"))
                        _db.insert_paper_metadata_batch(_df.to_dict("records"))
                    except Exception as _e:
                        logger.warning("Background DB write failed: %s", _e)
                threading.Thread(target=_bg_db_write, daemon=True).start()

            db.update_query_session(session_id, papers_fetched=_n_papers,
                                    pipeline_status="complete")
            total_elapsed = _elapsed()
            # step=7 marks all 6 steps as done (✅)
            _update(7, f"Pipeline complete! ({total_elapsed})",
                    papers=_n_papers, ents=len(entities_df),
                    rels=len(relationships_df))
            st.success(
                f"✅ Pipeline complete in {total_elapsed}! "
                f"{_n_papers:,} papers · {len(entities_df):,} entities · embeddings ready. "
                f"Topics & relationships available on their pages. "
                f"Next run loads from cache instantly."
            )

        # ── Save all results to session state (both paths) ────────────────────
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
