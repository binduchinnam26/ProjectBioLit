"""Analysis page — KPI metrics, publication trend, and topic charts."""

import logging

import streamlit as st
import config
from ui.components.cards import empty_state
from ui.components.metrics import kpi_row

logger = logging.getLogger(__name__)


def _build_topics_lazy(papers_df, embedder):
    """Run BERTopic on demand and persist results into session state + derived cache."""
    import sys, os, pickle
    import pandas as pd
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from pipeline.topic_modeler import TopicModeler
    from pipeline.network_builder import NetworkBuilder
    from pathlib import Path
    from utils.helpers import query_hash

    query = st.session_state.get("current_query", "")
    embeddings_array = st.session_state.get("embeddings_array")

    with st.spinner("Running topic modelling (UMAP + HDBSCAN)… this may take 1-3 minutes."):
        try:
            tm = TopicModeler()
            tm.setup(n_papers=len(papers_df))
            abstracts = papers_df["abstract"].fillna("").tolist()
            doc_topics, _ = tm.fit_transform(abstracts, embeddings_array)
            topic_labels     = tm.get_topic_labels()
            topics_over_time = tm.get_topic_over_time(papers_df)
            doc_topics_df    = tm.get_document_topics(papers_df)
            nb = NetworkBuilder()
            topic_graph = nb.build_topic_network(
                {"doc_topics": doc_topics, "topic_labels": topic_labels}
            )
            st.session_state.update({
                "topic_labels":     topic_labels,
                "topics_over_time": topics_over_time,
                "doc_topics_df":    doc_topics_df,
                "topic_graph":      topic_graph,
            })
            # Patch derived cache so next FAST PATH load includes topics
            qh = query_hash(query) if query else "default"
            derived_cache = Path(config.PROCESSED_DIR) / f"{qh}_derived.pkl"
            if derived_cache.exists():
                with open(derived_cache, "rb") as _f:
                    _d = pickle.load(_f)
                _d.update({
                    "topic_labels": topic_labels, "topics_over_time": topics_over_time,
                    "doc_topics_df": doc_topics_df, "topic_graph": topic_graph,
                })
                with open(derived_cache, "wb") as _f:
                    pickle.dump(_d, _f, protocol=pickle.HIGHEST_PROTOCOL)
            st.success(f"Topic model complete: {len(topic_labels)} topics discovered.")
        except Exception as exc:
            logger.error("Lazy topic build failed: %s", exc)
            st.error(f"Topic modelling failed: {exc}")


def render():
    if not st.session_state.get("pipeline_complete"):
        empty_state("📊", "No data loaded yet",
                    "Run a search on the Home page to populate the analysis dashboard.")
        return

    papers_df        = st.session_state.get("papers_df")
    topics_over_time = st.session_state.get("topics_over_time")

    query = st.session_state.get("current_query", "")
    db_stats = {}
    db = st.session_state.get("db")
    if db:
        try:
            db_stats = db.get_statistics(query_used=query)
        except Exception:
            pass

    # ── Active query tag ──────────────────────────────────────────────────────
    st.markdown(
        f"<div style='margin-bottom:1rem'>"
        f"<span style='font-size:11px;color:{config.TEXT_SECONDARY};font-weight:600;"
        f"text-transform:uppercase'>Active Query</span> &nbsp;"
        f"<span style='background:{config.PRIMARY_ACCENT}20;color:{config.PRIMARY_ACCENT};"
        f"border:1px solid {config.PRIMARY_ACCENT}40;border-radius:20px;padding:2px 12px;"
        f"font-size:12px;font-weight:600'>{query}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── KPI row ───────────────────────────────────────────────────────────────
    kpi_row(db_stats, papers_df)

    st.markdown(
        f"<hr style='border-color:{config.BORDER_COLOR};margin:1.5rem 0'>",
        unsafe_allow_html=True,
    )

    # ── Publication trend (the only chart-style viz on this page) ────────────
    from visualization.trend_charts import (
        render_publication_trend,
        render_top_keywords,
        render_author_productivity,
        render_topic_evolution,
    )

    st.markdown("### Publication Trend", unsafe_allow_html=False)
    fig_trend = render_publication_trend(papers_df)
    st.plotly_chart(fig_trend, width="stretch")

    tab1, tab2, tab3 = st.tabs(["Top Keywords", "Author Productivity", "Topic Evolution"])
    with tab1:
        st.plotly_chart(render_top_keywords(papers_df), width="stretch")
    with tab2:
        st.plotly_chart(render_author_productivity(papers_df), width="stretch")
    with tab3:
        if topics_over_time is not None and not topics_over_time.empty:
            st.plotly_chart(render_topic_evolution(topics_over_time), width="stretch")
        else:
            st.info(
                "Topic evolution not yet computed — embeddings and topic modelling are "
                "deferred for faster initial pipeline runs."
            )
            embedder = st.session_state.get("embedder")
            if embedder is None:
                st.caption(
                    "Build embeddings first: go to **Semantic Search** and click "
                    "**Build Embeddings**, then return here."
                )
            else:
                if st.button("Compute Topic Model", type="primary", key="build_topics_btn"):
                    _build_topics_lazy(papers_df, embedder)
                    st.rerun()

    # ── Network Explorer callout ──────────────────────────────────────────────
    st.markdown(
        f"<hr style='border-color:{config.BORDER_COLOR};margin:1.5rem 0'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='background:{config.PRIMARY_ACCENT}10;border:1px solid "
        f"{config.PRIMARY_ACCENT}30;border-radius:8px;padding:14px 18px;"
        f"margin-bottom:0.5rem'>"
        f"<span style='font-size:14px;font-weight:600;color:{config.PRIMARY_ACCENT}'>"
        f"🕸️ Bibliometric Network Explorer</span><br>"
        f"<span style='font-size:13px;color:{config.TEXT_SECONDARY}'>"
        f"Interactive co-authorship, keyword, topic, and entity networks "
        f"with Network · Overlay · Density visualization modes are available "
        f"on the <b>Knowledge Graph</b> page.</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
