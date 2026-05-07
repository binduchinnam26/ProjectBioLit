"""Reusable UI card components for BioLitAI-X."""

from typing import List, Optional
import streamlit as st
import config


def paper_card(
    title: str,
    pmid: str,
    authors: str = "",
    journal: str = "",
    year: str = "",
    abstract: str = "",
    score: Optional[float] = None,
    expanded: bool = False,
):
    """Render a single paper result card."""
    from utils.helpers import pubmed_url, truncate
    url = pubmed_url(pmid)
    score_html = ""
    if score is not None:
        pct = int(score * 100)
        bar_color = config.SUCCESS_COLOR if pct > 75 else (config.WARNING_COLOR if pct > 50 else config.DANGER_COLOR)
        score_html = f"""
        <div style="margin-bottom:8px">
          <div style="display:flex;justify-content:space-between;font-size:11px;
                      color:{config.TEXT_SECONDARY};margin-bottom:3px">
            <span>Similarity</span><span style="color:{bar_color};font-weight:700">{pct}%</span>
          </div>
          <div style="background:{config.BORDER_COLOR};border-radius:4px;height:4px">
            <div style="background:{bar_color};width:{pct}%;height:4px;border-radius:4px"></div>
          </div>
        </div>"""

    meta = " · ".join(filter(None, [authors[:60] + "…" if len(authors) > 60 else authors, journal, str(year)]))
    excerpt = truncate(abstract, 280) if abstract else ""
    excerpt_html = f"<p style='font-size:13px;color:{config.TEXT_SECONDARY};margin:6px 0 0'>{excerpt}</p>" if excerpt else ""

    st.markdown(
        f"""
        <div class="bx-card" style="margin-bottom:10px">
          {score_html}
          <a href="{url}" target="_blank" style="text-decoration:none">
            <div style="font-size:14px;font-weight:600;color:{config.TEXT_PRIMARY};
                        line-height:1.4;margin-bottom:4px">{title}</div>
          </a>
          <div style="font-size:11px;color:{config.TEXT_SECONDARY};margin-bottom:4px">{meta}</div>
          <span class="pmid-tag">PMID:{pmid}</span>
          {excerpt_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def hypothesis_card(hyp: dict):
    """Render a hypothesis result card with expandable sections."""
    ca = hyp.get("concept_a", "")
    cb = hyp.get("concept_b", "")
    conf = hyp.get("confidence_label", "Low")
    conf_class = f"badge-{conf.lower()}"
    score = hyp.get("confidence_score", 1)
    hypothesis_text = hyp.get("hypothesis_text", "")
    pmids = hyp.get("evidence_pmids", [])
    gap_type = hyp.get("gap_type", "structural").replace("_", " ").title()

    pmid_links = " ".join(
        f'<a href="https://pubmed.ncbi.nlm.nih.gov/{p}/" target="_blank" '
        f'class="pmid-tag" style="text-decoration:none">PMID:{p}</a>'
        for p in pmids[:6]
    )

    with st.container():
        st.markdown(
            f"""
            <div class="bx-card">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
                <div>
                  <span style="font-size:15px;font-weight:700;color:{config.TEXT_PRIMARY}">{ca}</span>
                  <span style="color:{config.TEXT_SECONDARY};margin:0 8px">↔</span>
                  <span style="font-size:15px;font-weight:700;color:{config.TEXT_PRIMARY}">{cb}</span>
                </div>
                <div style="display:flex;gap:6px;flex-shrink:0">
                  <span class="{conf_class}">{conf}</span>
                  <span style="font-size:11px;color:{config.TEXT_SECONDARY};
                               background:{config.SURFACE_COLOR};border:1px solid {config.BORDER_COLOR};
                               padding:2px 8px;border-radius:12px">{gap_type}</span>
                </div>
              </div>
              <p style="font-size:13px;color:{config.TEXT_PRIMARY};line-height:1.6;margin-bottom:10px">
                {hypothesis_text}
              </p>
              <div style="margin-top:6px">{pmid_links}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("View details", expanded=False):
            if hyp.get("biological_rationale"):
                st.markdown(f"**Biological Rationale**\n\n{hyp['biological_rationale']}")
            if hyp.get("suggested_experiment"):
                st.markdown(f"**Suggested Experiment**\n\n{hyp['suggested_experiment']}")
            if hyp.get("novelty"):
                st.markdown(f"**Novelty**\n\n{hyp['novelty']}")
            if hyp.get("supporting_evidence"):
                st.markdown(f"**Supporting Evidence**\n\n{hyp['supporting_evidence']}")


def metric_chip(label: str, value: str, color: str = None):
    """Render a small inline metric chip."""
    color = color or config.PRIMARY_ACCENT
    st.markdown(
        f"<div style='display:inline-block;background:{config.SURFACE_ELEVATED};"
        f"border:1px solid {config.BORDER_COLOR};border-radius:6px;"
        f"padding:4px 12px;margin:3px'>"
        f"<span style='font-size:11px;color:{config.TEXT_SECONDARY}'>{label}: </span>"
        f"<span style='font-size:13px;font-weight:700;color:{color}'>{value}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def section_header(number: str, title: str, subtitle: str = ""):
    """Render a numbered section header with optional subtitle."""
    st.markdown(
        f"""
        <div style="margin:2rem 0 1rem">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
            <span class="bx-section-badge">{number}</span>
            <h3 style="margin:0;font-size:1.1rem;font-weight:700;color:{config.TEXT_PRIMARY}">{title}</h3>
          </div>
          {"" if not subtitle else f'<p style="font-size:12px;color:{config.TEXT_SECONDARY};margin:0 0 0 36px">{subtitle}</p>'}
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(icon: str, message: str, hint: str = ""):
    """Render an instructional empty state screen."""
    st.markdown(
        f"""
        <div style="text-align:center;padding:4rem 2rem">
          <div style="font-size:3rem;margin-bottom:1rem">{icon}</div>
          <div style="font-size:1.1rem;font-weight:600;color:{config.TEXT_PRIMARY};
                      margin-bottom:0.5rem">{message}</div>
          {"" if not hint else f'<div style="font-size:13px;color:{config.TEXT_SECONDARY}">{hint}</div>'}
        </div>
        """,
        unsafe_allow_html=True,
    )
