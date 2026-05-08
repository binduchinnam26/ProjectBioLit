"""Home page — search, pipeline execution, session management."""

import gc
import logging
import shutil
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
    Delete old parquet caches, FAISS indexes, and compact the SQLite WAL file
    to reclaim disk space.  Returns MB freed (approximate).
    Safe to call at any time — only removes re-generatable cache files.
    """
    import sqlite3 as _sqlite3
    freed = 0
    for pattern in ("*.parquet",):
        for p in Path(config.PROCESSED_DIR).glob(pattern):
            try:
                freed += p.stat().st_size
                p.unlink()
            except Exception:
                pass
    for pattern in ("*.faiss", "*.json"):
        for p in Path(config.EMBEDDINGS_DIR).glob(pattern):
            try:
                freed += p.stat().st_size
                p.unlink()
            except Exception:
                pass
    # Checkpoint + shrink the SQLite WAL file
    try:
        conn = _sqlite3.connect(config.DB_PATH)
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
    "NLP Processing",
    "Building Embeddings",
    "Building Graph",
]


def render():
    """Render the Home / search page."""
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="bx-hero">
          <div class="bx-hero-title">BioLitAI-X</div>
          <div class="bx-hero-subtitle">
            From Literature to Discovery —
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

        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            max_results = st.slider(
                "Max Results",
                min_value=config.MAX_RESULTS_MIN,
                max_value=config.MAX_RESULTS_MAX,
                value=config.MAX_RESULTS_DEFAULT,
                step=100,
                help="Number of papers to retrieve from PubMed (100–3,000). "
                     "2,000 is recommended for best speed and stability.",
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
                use_container_width=True,
                type="primary",
            )
        with col4:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            force_rerun = st.button(
                "↺  Force Rerun",
                use_container_width=True,
                help="Ignore cached results and re-fetch from PubMed",
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
    if search_clicked or force_rerun:
        q = (query or "").strip()
        if not q:
            st.error("Please enter a biomedical research query.")
            return
        _run_pipeline(q, max_results, year_min, year_max, force_rerun=force_rerun)

    # ── Show results summary if pipeline is complete ──────────────────────────
    elif st.session_state.get("pipeline_complete"):
        _show_results_summary()


# ── Pipeline orchestrator ─────────────────────────────────────────────────────

def _run_pipeline(
    query: str,
    max_results: int,
    year_min: int,
    year_max: int,
    force_rerun: bool = False,
):
    """Execute the full 6-step pipeline with live progress and parquet caching."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    from database.db_manager import DatabaseManager
    from pipeline.retrieval import PubMedRetriever
    from pipeline.parser import XMLParser
    from pipeline.cleaner import DataCleaner
    from pipeline.network_builder import NetworkBuilder
    from pipeline.knowledge_graph import KnowledgeGraph

    # ── Disk space pre-check + auto-cleanup ──────────────────────────────────
    free_mb = _free_mb()
    if free_mb < _DISK_WARN_MB:
        freed_mb = _auto_cleanup_disk()
        free_mb  = _free_mb()   # re-measure after cleanup
        if freed_mb > 0:
            st.info(f"🧹 Auto-cleaned {freed_mb} MB of old cache files. {free_mb} MB now free.")

    low_disk = free_mb < _DISK_WARN_MB
    no_disk  = free_mb < _DISK_MIN_MB

    if no_disk:
        st.error(
            f"❌ Critical: only {free_mb} MB free on disk after cleanup. "
            "The pipeline will run in memory-only mode — results will NOT be saved. "
            "Please free at least 500 MB on your drive before running again."
        )
    elif low_disk:
        st.warning(
            f"⚠️ Low disk space: {free_mb} MB free. "
            "Parquet cache and FAISS index will be skipped. "
            "Free more drive space for full persistence between sessions."
        )

    db = DatabaseManager(config.DB_PATH)
    session_id = db.save_query_session(query, max_results)
    db.update_query_session(session_id, pipeline_status="running")

    step_container = st.empty()
    stats_container = st.empty()
    status_msg = st.empty()
    pipeline_start = time.monotonic()

    def _elapsed() -> str:
        secs = int(time.monotonic() - pipeline_start)
        return f"{secs // 60}m {secs % 60}s" if secs >= 60 else f"{secs}s"

    def _update(step: int, msg: str, papers: int = 0, ents: int = 0, rels: int = 0):
        with step_container.container():
            pipeline_status_indicator(step, 6, msg)
        with stats_container.container():
            live_stats_bar(papers, ents, rels)
        status_msg.markdown(
            f"<p style='font-size:12px;color:{config.TEXT_SECONDARY}'>"
            f"{msg} <span style='opacity:0.5'>({_elapsed()})</span></p>",
            unsafe_allow_html=True,
        )

    # ── Parquet cache ─────────────────────────────────────────────────────────
    qhash = query_hash(query)
    cache_path = Path(config.PROCESSED_DIR) / f"{qhash}.parquet"
    papers_df = pd.DataFrame()

    if cache_path.exists() and not force_rerun:
        try:
            papers_df = pd.read_parquet(cache_path)
            _update(3, f"Loaded {len(papers_df):,} papers from cache.", papers=len(papers_df))
            logger.info("Loaded %d papers from parquet cache: %s", len(papers_df), cache_path)
        except Exception as exc:
            logger.warning("Cache load failed (%s); re-running pipeline.", exc)
            papers_df = pd.DataFrame()

    try:
        if papers_df.empty:
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
            del xml_batches  # free XML strings — no longer needed
            _update(2, f"Parsed {len(raw_papers):,} records.", papers=len(raw_papers))

            # Step 3: Clean
            n_raw = len(raw_papers)
            _update(3, f"Cleaning and normalising {n_raw:,} papers — please wait...")
            cleaner = DataCleaner()
            papers_df = cleaner.run_full_pipeline(raw_papers)
            del raw_papers   # free raw dicts — DataFrame is the working copy now

            # Apply year filter
            if "pub_year" in papers_df.columns:
                before = len(papers_df)
                papers_df = papers_df[
                    papers_df["pub_year"].fillna(0).astype(int).between(year_min, year_max)
                    | papers_df["pub_year"].isna()
                ].reset_index(drop=True)
                filtered_out = before - len(papers_df)
                if filtered_out:
                    logger.info(
                        "Year filter removed %d papers outside %d–%d",
                        filtered_out, year_min, year_max,
                    )

            # Persist to parquet cache (skipped when disk is low)
            if not low_disk:
                try:
                    import json as _json
                    cache_df = papers_df.copy()
                    for col in ("authors", "author_keywords", "mesh_terms",
                                "chemicals", "publication_types", "grants"):
                        if col in cache_df.columns:
                            cache_df[col] = cache_df[col].apply(
                                lambda v: _json.dumps(v) if not isinstance(v, str) else v
                            )
                    cache_df.to_parquet(cache_path, index=False)
                    logger.info("Parquet cache saved: %s", cache_path)
                except OSError as exc:
                    logger.warning("Parquet cache save failed (disk full?): %s", exc)
                    st.warning(f"Cache not saved: {exc}")
                except Exception as exc:
                    logger.warning("Parquet cache save failed: %s", exc)

            _update(3, f"Cleaning complete. {len(papers_df):,} papers retained.",
                    papers=len(papers_df))
        else:
            # Restore list columns serialised as JSON strings in the cache
            import json as _json
            for col in ("authors", "author_keywords", "mesh_terms",
                        "chemicals", "publication_types", "grants"):
                if col in papers_df.columns:
                    def _try_parse(v):
                        if isinstance(v, str):
                            try:
                                return _json.loads(v)
                            except Exception:
                                return []
                        return v if isinstance(v, list) else []
                    papers_df[col] = papers_df[col].apply(_try_parse)

        # Store papers + all relational metadata to database (single transaction each)
        if no_disk:
            st.warning("Disk full — skipping database writes. Results are in memory only.")
        else:
            _update(3, f"Storing {len(papers_df):,} papers to database...", papers=len(papers_df))
            try:
                db.insert_papers_batch(papers_df.to_dict("records"))
            except OSError as exc:
                logger.error("Paper batch insert failed (disk full): %s", exc)
                st.error(f"Database write failed — disk is full ({_free_mb()} MB free). Free space and retry.")
            except Exception as exc:
                logger.warning("Batch paper insert failed: %s", exc)

            _update(3, "Indexing authors, keywords, MeSH terms...", papers=len(papers_df))
            try:
                db.insert_paper_metadata_batch(papers_df.to_dict("records"))
            except OSError as exc:
                logger.error("Metadata batch insert failed (disk full): %s", exc)
                st.error(f"Metadata write failed — disk is full ({_free_mb()} MB free).")
            except Exception as exc:
                logger.warning("Batch metadata insert failed: %s", exc)

        # Step 4: NLP (optional — skip if scispacy not installed)
        _update(4, "Running NLP entity extraction...")
        entities_df = pd.DataFrame()
        relationships_df = pd.DataFrame()
        try:
            from pipeline.nlp_processor import NLPProcessor
            nlp = NLPProcessor(db_manager=db)
            nlp.setup()
            entities_df, relationships_df = nlp.process_corpus(
                papers_df,
                progress_callback=lambda ents, rels, m: _update(4, m, papers=len(papers_df),
                                                               ents=ents, rels=rels),
            )
        except Exception as exc:
            logger.warning("NLP step skipped (%s). Knowledge graph will be limited.", exc)
            st.warning(f"NLP processing skipped: {exc}. Knowledge graph will have limited data.")

        _update(4, f"NLP done: {len(entities_df):,} entities, {len(relationships_df):,} relationships.",
                papers=len(papers_df), ents=len(entities_df), rels=len(relationships_df))

        # Step 5: Embeddings (optional)
        _update(5, "Building semantic embeddings...")
        embeddings_array = None
        embedder = None
        try:
            from pipeline.embedder import EmbeddingEngine
            embedder = EmbeddingEngine(persist_index=not low_disk)
            embedder.setup()
            embeddings_array = embedder.embed_corpus(
                papers_df, query=query,
                progress_callback=lambda d, t, m: _update(5, m, papers=len(papers_df),
                                                           ents=len(entities_df)),
            )
        except Exception as exc:
            logger.warning("Embedding step skipped: %s", exc)
            st.warning(f"Embedding step skipped: {exc}. Semantic search will be unavailable.")

        # Step 6: Networks and knowledge graph
        _update(6, "Building bibliometric networks and knowledge graph...")
        nb = NetworkBuilder()
        coauth_graph_full  = nb.build_coauthorship_network(papers_df)
        keyword_graph_full = nb.build_keyword_cooccurrence_network(papers_df)
        coauth_stats  = nb.calculate_network_statistics(coauth_graph_full)
        keyword_stats = nb.calculate_network_statistics(keyword_graph_full)
        # Thin graphs to render limit before handing to browser
        coauth_graph  = nb.prepare_graph_for_display(coauth_graph_full)
        keyword_graph = nb.prepare_graph_for_display(keyword_graph_full)
        del coauth_graph_full, keyword_graph_full
        gc.collect()

        kg = KnowledgeGraph()
        kg_graph = kg.build_from_entities(entities_df, relationships_df)

        topic_graph = None
        topic_labels = {}
        topics_over_time = pd.DataFrame()
        doc_topics_df = pd.DataFrame()
        try:
            from pipeline.topic_modeler import TopicModeler
            tm = TopicModeler()
            tm.setup()
            abstracts = papers_df["abstract"].fillna("").tolist()
            doc_topics, probs = tm.fit_transform(abstracts, embeddings_array)
            topic_labels = tm.get_topic_labels()
            topics_over_time = tm.get_topic_over_time(papers_df)
            doc_topics_df = tm.get_document_topics(papers_df)
            topic_graph = nb.build_topic_network({
                "doc_topics": doc_topics,
                "topic_labels": topic_labels,
            })
        except Exception as exc:
            logger.warning("Topic modeling skipped: %s", exc)

        db.update_query_session(
            session_id,
            papers_fetched=len(papers_df),
            pipeline_status="complete",
        )

        # ── Save all results to session state ─────────────────────────────
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

        # Track session in past_sessions list
        past = st.session_state.get("past_sessions", [])
        past.append({
            "id": session_id,
            "query_text": query,
            "pipeline_status": "complete",
            "papers_fetched": len(papers_df),
        })
        st.session_state["past_sessions"] = past

        total_elapsed = _elapsed()
        _update(7, f"Pipeline complete! ({total_elapsed})", papers=len(papers_df),
                ents=len(entities_df), rels=len(relationships_df))
        st.success(
            f"✅ Pipeline complete in {total_elapsed}! {len(papers_df):,} papers analysed. "
            f"Use the sidebar to explore results."
        )

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
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Papers", f"{len(papers_df):,}")
    with col2:
        ents = st.session_state.get("entities_df")
        st.metric("Entities", f"{len(ents):,}" if ents is not None and not ents.empty else "—")
    with col3:
        hyps = st.session_state.get("hypotheses", [])
        st.metric("Hypotheses", str(len(hyps)))


# Avoid circular import
try:
    from database.db_manager import DatabaseManager
except ImportError:
    pass
