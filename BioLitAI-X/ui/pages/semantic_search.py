"""Semantic similarity search page."""

import logging

import streamlit as st
import config
from ui.components.cards import empty_state, paper_card

logger = logging.getLogger(__name__)


def _build_embeddings_lazy(papers_df):
    """Build and cache embeddings on first visit; updates session state."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from pipeline.embedder import EmbeddingEngine
    from utils.helpers import query_hash
    query = st.session_state.get("current_query", "")
    low_disk = st.session_state.get("_low_disk", False)
    with st.spinner(f"Building semantic embeddings for {len(papers_df):,} papers…"):
        try:
            emb = EmbeddingEngine(persist_index=not low_disk)
            emb.setup()
            if emb.index_exists(query):
                emb.load_index(query)
                arr = emb.load_embeddings(query)
                if arr is None:
                    arr = emb.embed_corpus(papers_df, query=query)
            else:
                arr = emb.embed_corpus(papers_df, query=query)
            st.session_state["embedder"] = emb
            st.session_state["embeddings_array"] = arr
            st.session_state["embeddings_ready"] = True
            # Patch derived cache with embeddings so FAST PATH gets them next time
            _patch_derived_cache(query, embedder=emb, embeddings_array=arr)
            st.success(f"Embeddings ready: {len(arr):,} vectors.")
        except Exception as exc:
            logger.error("Lazy embedding build failed: %s", exc)
            st.error(f"Embedding failed: {exc}")


def _patch_derived_cache(query: str, **kwargs):
    """Merge new keys into the existing derived pickle without rewriting everything."""
    import pickle
    from pathlib import Path
    import config as _cfg
    from utils.helpers import query_hash as _qh
    qh = _qh(query) if query else "default"
    derived_cache = Path(_cfg.PROCESSED_DIR) / f"{qh}_derived.pkl"
    if not derived_cache.exists():
        return
    try:
        with open(derived_cache, "rb") as _f:
            _d = pickle.load(_f)
        _d.update(kwargs)
        with open(derived_cache, "wb") as _f:
            pickle.dump(_d, _f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        logger.warning("_patch_derived_cache failed: %s", exc)


def render():
    if not st.session_state.get("pipeline_complete"):
        empty_state("🔍", "No data loaded yet",
                    "Run a search on the Home page first.")
        return

    papers_df = st.session_state.get("papers_df")
    embedder  = st.session_state.get("embedder")

    st.markdown(
        f"<h2 style='margin-bottom:4px'>Semantic Search</h2>"
        f"<p style='color:{config.TEXT_SECONDARY};font-size:13px;margin-bottom:1.5rem'>"
        f"Search within the currently loaded paper set using meaning, not just keywords.</p>",
        unsafe_allow_html=True,
    )

    # ── Embeddings are built in the main pipeline. Recovery button if they failed.
    if embedder is None and papers_df is not None and not papers_df.empty:
        st.warning(
            "Semantic embeddings are not ready (they may have failed during pipeline, "
            "e.g. model download required on first run). Click below to build them now."
        )
        if st.button("Build Embeddings", type="primary", key="build_emb_btn"):
            _build_embeddings_lazy(papers_df)
            st.rerun()
        return

    embedder = st.session_state.get("embedder")

    # ── Search controls ───────────────────────────────────────────────────────
    search_q = st.text_input(
        "Semantic query",
        placeholder="Describe a concept or paste a sentence from a paper...",
        label_visibility="collapsed",
        key="sem_search_input",
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        top_k = st.select_slider(
            "Results to return",
            options=[5, 10, 20, 50],
            value=10,
            key="sem_top_k",
        )
    with c2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run = st.button("Search", type="primary", width="stretch", key="sem_run")

    st.markdown(
        f"<hr style='border-color:{config.BORDER_COLOR};margin:1rem 0'>",
        unsafe_allow_html=True,
    )

    if run and search_q.strip():
        _run_search(search_q.strip(), top_k, papers_df, embedder)
    elif not run:
        # Show starter suggestions
        suggestions = [
            "Role of oxidative stress in neurodegeneration",
            "Mechanisms of drug resistance in bacteria",
            "Gene expression regulation by non-coding RNA",
            "Inflammatory pathways in metabolic disease",
        ]
        st.markdown(
            f"<p style='font-size:13px;color:{config.TEXT_SECONDARY};margin-bottom:8px'>"
            f"Try searching for:</p>",
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(
                    f"💬 {s}",
                    key=f"sug_{i}",
                    width="stretch",
                ):
                    st.session_state["sem_prefill"] = s
                    st.rerun()

        if "sem_prefill" in st.session_state:
            _run_search(
                st.session_state["sem_prefill"],
                10, papers_df, embedder,
            )


def _run_search(query: str, top_k: int, papers_df, embedder):
    """Run semantic search and display ranked results."""
    if papers_df is None or papers_df.empty:
        st.error("No papers loaded.")
        return

    results = []

    # FAISS semantic search
    if embedder and hasattr(embedder, "index") and \
       embedder.index is not None and embedder.index.ntotal > 0:
        with st.spinner("Searching embeddings..."):
            try:
                results = embedder.semantic_search(query, top_k=top_k)
            except Exception as exc:
                st.warning(f"FAISS search failed: {exc}. Falling back to text search.")
                results = []

    # Text fallback
    if not results:
        with st.spinner("Text searching..."):
            matched = papers_df[
                papers_df["abstract"].fillna("").str.lower().str.contains(
                    query.lower(), regex=False, na=False
                )
            ].head(top_k)
            results = [
                {"pmid": row["pmid"], "score": 0.9 - i * 0.02, "rank": i + 1}
                for i, (_, row) in enumerate(matched.iterrows())
            ]

    if not results:
        st.info(f"No results found for: **{query}**")
        return

    st.markdown(
        f"<p style='font-size:13px;color:{config.TEXT_SECONDARY};"
        f"margin-bottom:1rem'>Found <b style='color:{config.TEXT_PRIMARY}'>"
        f"{len(results)}</b> results for: "
        f"<b style='color:{config.PRIMARY_ACCENT}'>{query}</b></p>",
        unsafe_allow_html=True,
    )

    for res in results:
        pmid = res["pmid"]
        score = res.get("score", 0.0)
        row = papers_df[papers_df["pmid"] == pmid]
        if row.empty:
            continue
        row = row.iloc[0]
        authors_list = row.get("authors") or []
        authors_str = ", ".join(
            a.get("name", "") if isinstance(a, dict) else str(a)
            for a in authors_list[:3]
        )
        paper_card(
            title=str(row.get("title", "Unknown title")),
            pmid=pmid,
            authors=authors_str,
            journal=str(row.get("journal", "")),
            year=str(row.get("pub_year", "")),
            abstract=str(row.get("abstract", "")),
            score=score,
        )
