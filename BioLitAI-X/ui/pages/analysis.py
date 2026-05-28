"""Analysis page — KPI metrics, publication trend, and topic charts."""

import logging

import streamlit as st

import config
from pipeline.lazy_builders import build_topics_lazy
from ui.components.cards import empty_state
from ui.components.metrics import kpi_row
from visualization.trend_charts import (
    render_author_productivity,
    render_publication_trend,
    render_topic_evolution,
    render_top_keywords,
)

logger = logging.getLogger(__name__)


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

    # ── Publication trend ─────────────────────────────────────────────────────
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
                    build_topics_lazy(papers_df)
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
