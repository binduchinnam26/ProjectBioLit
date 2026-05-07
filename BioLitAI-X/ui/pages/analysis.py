"""Analysis page — KPI metrics, publication trend, and three VOSviewer network sections."""

import streamlit as st
import config
from ui.components.cards import empty_state, section_header
from ui.components.metrics import kpi_row


def render():
    if not st.session_state.get("pipeline_complete"):
        empty_state("📊", "No data loaded yet",
                    "Run a search on the Home page to populate the analysis dashboard.")
        return

    papers_df    = st.session_state.get("papers_df")
    coauth_graph = st.session_state.get("coauth_graph")
    keyword_graph = st.session_state.get("keyword_graph")
    topic_graph  = st.session_state.get("topic_graph")
    coauth_stats = st.session_state.get("coauth_stats", {})
    keyword_stats = st.session_state.get("keyword_stats", {})
    topics_over_time = st.session_state.get("topics_over_time")
    topic_labels = st.session_state.get("topic_labels", {})

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
    st.plotly_chart(fig_trend, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Top Keywords", "Author Productivity", "Topic Evolution"])
    with tab1:
        st.plotly_chart(render_top_keywords(papers_df), use_container_width=True)
    with tab2:
        st.plotly_chart(render_author_productivity(papers_df), use_container_width=True)
    with tab3:
        if topics_over_time is not None and not topics_over_time.empty:
            st.plotly_chart(render_topic_evolution(topics_over_time), use_container_width=True)
        else:
            st.info("Topic evolution data not available. NLP and topic modeling must complete first.")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 01 — Author Collaboration Network
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        f"<hr style='border:2px solid {config.BORDER_COLOR};margin:2rem 0'>",
        unsafe_allow_html=True,
    )
    section_header(
        "01",
        "Author Collaboration Network",
        "Node size = publication count · Color = research cluster · "
        "Edge thickness = collaboration strength",
    )

    if coauth_graph and coauth_graph.number_of_nodes() > 0:
        from visualization.network_viz import (
            render_coauthorship_network,
            render_network_stats,
            _build_controls_panel,
        )
        graph_col, stats_col = st.columns([4, 1])
        with graph_col:
            controls = _build_controls_panel("coauth")
            render_coauthorship_network(coauth_graph, controls=controls, height=700)
        with stats_col:
            render_network_stats(coauth_stats, "Collaboration Stats")
    else:
        st.info("Co-authorship network not available. Run the pipeline first.")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 02 — Keyword Co-occurrence Map
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        f"<hr style='border:2px solid {config.BORDER_COLOR};margin:2rem 0'>",
        unsafe_allow_html=True,
    )
    section_header(
        "02",
        "Keyword Co-occurrence Map",
        "Node size = frequency · Color = thematic cluster · "
        "Shape = keyword type (●=author  ■=MeSH  ◆=chemical)",
    )

    # Keyword type shape legend
    _kw_legend()

    if keyword_graph and keyword_graph.number_of_nodes() > 0:
        from visualization.network_viz import (
            render_keyword_network,
            render_network_stats,
            _build_controls_panel,
        )

        # Extra keyword type filter
        kw_types = list({
            d.get("keyword_type", "author")
            for _, d in keyword_graph.nodes(data=True)
        })
        selected_types = st.multiselect(
            "Filter by keyword type",
            options=kw_types,
            default=kw_types,
            key="kw_type_filter",
        )

        import networkx as nx
        kw_filtered = keyword_graph.copy()
        if selected_types:
            remove = [
                n for n, d in kw_filtered.nodes(data=True)
                if d.get("keyword_type") not in selected_types
            ]
            kw_filtered.remove_nodes_from(remove)

        graph_col2, stats_col2 = st.columns([4, 1])
        with graph_col2:
            controls2 = _build_controls_panel("keyword")
            render_keyword_network(kw_filtered, controls=controls2, height=700)
        with stats_col2:
            render_network_stats(keyword_stats, "Keyword Stats")
    else:
        st.info("Keyword co-occurrence network not available.")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 03 — Research Topic Landscape
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        f"<hr style='border:2px solid {config.BORDER_COLOR};margin:2rem 0'>",
        unsafe_allow_html=True,
    )
    section_header(
        "03",
        "Research Topic Landscape",
        "Node size = paper count · Edge = shared papers between topics",
    )

    if topic_graph and topic_graph.number_of_nodes() > 0:
        from visualization.network_viz import render_topic_network, _build_controls_panel

        if papers_df is not None and "pub_year" in papers_df.columns:
            years = sorted(papers_df["pub_year"].dropna().astype(int).unique())
            if len(years) >= 2:
                yr_min, yr_max = st.slider(
                    "Filter by year",
                    min_value=int(min(years)),
                    max_value=int(max(years)),
                    value=(int(min(years)), int(max(years))),
                    key="topic_year_slider",
                )
            else:
                yr_min, yr_max = None, None
        else:
            yr_min, yr_max = None, None

        controls3 = _build_controls_panel("topic")
        render_topic_network(topic_graph, controls=controls3, height=700,
                             year_range=(yr_min, yr_max))

        # Topic summary chips
        if topic_labels:
            st.markdown(
                f"<div style='font-size:12px;color:{config.TEXT_SECONDARY};"
                f"font-weight:600;margin-top:10px'>Topics discovered:</div>",
                unsafe_allow_html=True,
            )
            chips = " ".join(
                f"<span style='display:inline-block;background:{config.COMMUNITY_COLORS[i%10]}22;"
                f"border:1px solid {config.COMMUNITY_COLORS[i%10]}55;"
                f"color:{config.COMMUNITY_COLORS[i%10]};border-radius:20px;padding:2px 10px;"
                f"font-size:11px;margin:2px'>{v['label']} ({v['count']})</span>"
                for i, (tid, v) in enumerate(sorted(topic_labels.items(), key=lambda x: x[1]['count'], reverse=True)[:12])
            )
            st.markdown(chips, unsafe_allow_html=True)
    else:
        st.info(
            "Topic landscape not available. Topic modeling requires NLP and embedding steps to complete."
        )


def _kw_legend():
    st.markdown(
        f"""
        <div style='display:flex;gap:16px;font-size:12px;color:{config.TEXT_SECONDARY};
                    margin-bottom:8px;flex-wrap:wrap'>
          <span>● Author keyword</span>
          <span>■ MeSH Descriptor</span>
          <span>◆ Chemical term</span>
          <span>▲ MeSH Qualifier</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
