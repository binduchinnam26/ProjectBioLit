"""
trend_charts.py — Plotly trend and topic visualizations for BioLitAI-X.
All charts use the dark design system palette.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config

logger = logging.getLogger(__name__)

_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Open Sans", color=config.TEXT_PRIMARY),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(
        bgcolor="rgba(17,24,39,0.8)",
        bordercolor=config.BORDER_COLOR,
        borderwidth=1,
        font=dict(color=config.TEXT_PRIMARY, size=11),
    ),
    xaxis=dict(
        gridcolor=config.BORDER_COLOR,
        linecolor=config.BORDER_COLOR,
        tickfont=dict(color=config.TEXT_SECONDARY),
        title_font=dict(color=config.TEXT_SECONDARY),
    ),
    yaxis=dict(
        gridcolor=config.BORDER_COLOR,
        linecolor=config.BORDER_COLOR,
        tickfont=dict(color=config.TEXT_SECONDARY),
        title_font=dict(color=config.TEXT_SECONDARY),
    ),
)


def _apply_base(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    layout = dict(_LAYOUT_BASE)
    layout["height"] = height
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(color=config.TEXT_PRIMARY, size=15, family="Open Sans"),
            x=0.0,
            xanchor="left",
        )
    fig.update_layout(**layout)
    return fig


# ── Publication trend ─────────────────────────────────────────────────────────

def render_publication_trend(papers_df: pd.DataFrame) -> go.Figure:
    """
    Bar + cumulative line chart of papers published per year.
    """
    if papers_df.empty or "pub_year" not in papers_df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No publication data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color=config.TEXT_SECONDARY, size=14),
        )
        return _apply_base(fig, "Publications per Year")

    yearly = (
        papers_df.dropna(subset=["pub_year"])
        .assign(pub_year=lambda d: d["pub_year"].astype(int))
        .groupby("pub_year")
        .size()
        .reset_index(name="count")
        .sort_values("pub_year")
    )

    if yearly.empty:
        return go.Figure()

    yearly["cumulative"] = yearly["count"].cumsum()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=yearly["pub_year"],
            y=yearly["count"],
            name="Papers per Year",
            marker_color=config.PRIMARY_ACCENT,
            marker_line_color="rgba(0,0,0,0)",
            opacity=0.85,
            hovertemplate="<b>%{x}</b><br>Papers: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=yearly["pub_year"],
            y=yearly["cumulative"],
            name="Cumulative",
            mode="lines+markers",
            line=dict(color=config.SECONDARY_ACCENT, width=2.5),
            marker=dict(size=5, color=config.SECONDARY_ACCENT),
            hovertemplate="<b>%{x}</b><br>Cumulative: %{y}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(**{**_LAYOUT_BASE, "height": 380})
    fig.update_layout(
        title=dict(
            text="Publication Trend",
            font=dict(color=config.TEXT_PRIMARY, size=15), x=0.0,
        ),
        barmode="overlay",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Papers per Year", secondary_y=False,
                     gridcolor=config.BORDER_COLOR, tickfont=dict(color=config.TEXT_SECONDARY))
    fig.update_yaxes(title_text="Cumulative Papers", secondary_y=True,
                     gridcolor="rgba(0,0,0,0)", tickfont=dict(color=config.TEXT_SECONDARY))
    fig.update_xaxes(title_text="Year", tickfont=dict(color=config.TEXT_SECONDARY),
                     gridcolor=config.BORDER_COLOR)
    return fig


# ── Topic evolution ───────────────────────────────────────────────────────────

def render_topic_evolution(topics_over_time_df: pd.DataFrame) -> go.Figure:
    """
    Stacked area chart of topic frequency over years.
    Each topic is a different color; legend is interactive.
    """
    if topics_over_time_df is None or topics_over_time_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No topic timeline data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color=config.TEXT_SECONDARY, size=14),
        )
        return _apply_base(fig, "Topic Evolution Over Time")

    required_cols = {"Topic", "Frequency", "Timestamp"}
    if not required_cols.issubset(set(topics_over_time_df.columns)):
        logger.warning("topics_over_time_df missing required columns")
        return go.Figure()

    df = topics_over_time_df.copy()
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df["Timestamp"] = df["Timestamp"].astype(int)

    # Limit to top 10 topics by total frequency
    top_topics = (
        df.groupby("Topic")["Frequency"].sum()
        .nlargest(10).index.tolist()
    )
    df = df[df["Topic"].isin(top_topics)]

    palette = config.COMMUNITY_COLORS
    fig = go.Figure()

    for i, topic_id in enumerate(sorted(top_topics)):
        topic_df = df[df["Topic"] == topic_id].sort_values("Timestamp")
        words = topic_df["Words"].iloc[0] if "Words" in topic_df.columns and len(topic_df) else f"Topic {topic_id}"
        label = str(words)[:30]
        color = palette[i % len(palette)]

        fig.add_trace(go.Scatter(
            x=topic_df["Timestamp"],
            y=topic_df["Frequency"],
            name=label,
            mode="lines",
            stackgroup="one",
            line=dict(color=color, width=1),
            fillcolor=color.replace("#", "rgba(") + ",0.6)" if color.startswith("#") else color,
            hovertemplate=f"<b>{label}</b><br>Year: %{{x}}<br>Freq: %{{y:.2f}}<extra></extra>",
        ))

    fig = _apply_base(fig, "Topic Evolution Over Time", height=400)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Relative Frequency")
    return fig


# ── Top keywords bar chart ────────────────────────────────────────────────────

def render_top_keywords(papers_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """
    Horizontal bar chart of top keywords colored by type:
    author keyword / MeSH descriptor / MeSH qualifier / chemical / publication type.
    """
    if papers_df.empty:
        return _apply_base(go.Figure(), "Top Keywords")

    keyword_counts: Dict[str, Dict[str, Any]] = {}

    def _add(kw: str, ktype: str):
        if not kw or not kw.strip():
            return
        k = kw.strip().lower()
        if k not in keyword_counts:
            keyword_counts[k] = {"count": 0, "type": ktype, "display": kw.strip()}
        keyword_counts[k]["count"] += 1

    for _, row in papers_df.iterrows():
        for kw in (row.get("author_keywords") or []):
            _add(kw, "Author Keyword")
        for mesh in (row.get("mesh_terms") or []):
            if isinstance(mesh, dict):
                if mesh.get("descriptor"):
                    _add(mesh["descriptor"], "MeSH Descriptor")
                if mesh.get("qualifier"):
                    _add(mesh["qualifier"], "MeSH Qualifier")
        for chem in (row.get("chemicals") or []):
            name = chem.get("name") if isinstance(chem, dict) else chem
            if name:
                _add(name, "Chemical")
        for pt in (row.get("publication_types") or []):
            _add(pt, "Publication Type")

    if not keyword_counts:
        return _apply_base(go.Figure(), "Top Keywords")

    sorted_kws = sorted(keyword_counts.items(), key=lambda x: x[1]["count"], reverse=True)[:top_n]

    type_colors = {
        "Author Keyword":   config.PRIMARY_ACCENT,
        "MeSH Descriptor":  config.SUCCESS_COLOR,
        "MeSH Qualifier":   config.WARNING_COLOR,
        "Chemical":         config.SECONDARY_ACCENT,
        "Publication Type": config.DANGER_COLOR,
    }

    labels = [v["display"] for _, v in sorted_kws]
    counts = [v["count"] for _, v in sorted_kws]
    colors = [type_colors.get(v["type"], config.TEXT_SECONDARY) for _, v in sorted_kws]
    types  = [v["type"] for _, v in sorted_kws]

    fig = go.Figure(go.Bar(
        x=counts,
        y=labels,
        orientation="h",
        marker_color=colors,
        customdata=types,
        hovertemplate="<b>%{y}</b><br>Count: %{x}<br>Type: %{customdata}<extra></extra>",
    ))

    # Add dummy traces for legend
    for tname, tcolor in type_colors.items():
        fig.add_trace(go.Bar(
            x=[None], y=[None],
            name=tname,
            marker_color=tcolor,
            showlegend=True,
        ))

    fig = _apply_base(fig, f"Top {top_n} Keywords by Type", height=max(400, top_n * 22))
    fig.update_layout(
        yaxis=dict(autorange="reversed", tickfont=dict(color=config.TEXT_PRIMARY, size=11)),
        barmode="overlay",
        showlegend=True,
    )
    fig.update_xaxes(title_text="Frequency")
    return fig


# ── Author productivity ───────────────────────────────────────────────────────

def render_author_productivity(papers_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of top authors by paper count."""
    if papers_df.empty:
        return _apply_base(go.Figure(), "Top Authors by Paper Count")

    from collections import Counter
    author_counts: Counter = Counter()

    for _, row in papers_df.iterrows():
        for author in (row.get("authors") or []):
            name = author.get("name") if isinstance(author, dict) else str(author)
            if name and name.strip():
                author_counts[name.strip()] += 1

    if not author_counts:
        return _apply_base(go.Figure(), "Top Authors by Paper Count")

    top = author_counts.most_common(top_n)
    names = [t[0] for t in top]
    counts = [t[1] for t in top]

    fig = go.Figure(go.Bar(
        x=counts,
        y=names,
        orientation="h",
        marker_color=config.PRIMARY_ACCENT,
        marker_line_color="rgba(0,0,0,0)",
        hovertemplate="<b>%{y}</b><br>Papers: %{x}<extra></extra>",
    ))

    fig = _apply_base(fig, f"Top {top_n} Authors by Paper Count", height=max(380, top_n * 24))
    fig.update_layout(yaxis=dict(autorange="reversed", tickfont=dict(color=config.TEXT_PRIMARY, size=11)))
    fig.update_xaxes(title_text="Papers")
    return fig
