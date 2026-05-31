"""Loading spinners, progress bars, and skeleton screens for BioLitAI-X."""

import streamlit as st
import config


def skeleton_card(lines: int = 3):
    """Animated skeleton placeholder while content loads."""
    widths = ["100%", "75%", "50%"]
    lines_html = "".join(
        f'<div class="skeleton" style="width:{widths[i % len(widths)]};'
        f'height:{"18px" if i > 0 else "22px"}"></div>'
        for i in range(lines)
    )
    st.markdown(
        f'<div class="bx-card">{lines_html}</div>',
        unsafe_allow_html=True,
    )


def skeleton_row(n: int = 3):
    """Render n skeleton cards side by side."""
    cols = st.columns(n)
    for col in cols:
        with col:
            skeleton_card(lines=2)


def progress_pipeline(
    step: int,
    total_steps: int,
    label: str,
    detail: str = "",
):
    """Render a labeled Streamlit progress bar for a multi-step pipeline."""
    pct = max(0.0, min(1.0, (step - 1) / max(total_steps, 1)))
    st.markdown(
        f"<div style='font-size:13px;font-weight:600;color:{config.TEXT_PRIMARY};"
        f"margin-bottom:4px'>{label}</div>",
        unsafe_allow_html=True,
    )
    st.progress(pct)
    if detail:
        st.markdown(
            f"<p style='font-size:12px;color:{config.TEXT_SECONDARY};margin-top:2px'>{detail}</p>",
            unsafe_allow_html=True,
        )


def live_stats_bar(papers_found: int = 0, entities: int = 0, relationships: int = 0):
    """Show live-updating stats during pipeline execution."""
    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:1.5rem;font-weight:700;color:{config.PRIMARY_ACCENT}'>"
            f"{papers_found:,}</div>"
            f"<div style='font-size:11px;color:{config.TEXT_SECONDARY}'>Papers Found</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:1.5rem;font-weight:700;color:{config.SUCCESS_COLOR}'>"
            f"{entities:,}</div>"
            f"<div style='font-size:11px;color:{config.TEXT_SECONDARY}'>Entities Extracted</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:1.5rem;font-weight:700;color:{config.SECONDARY_ACCENT}'>"
            f"{relationships:,}</div>"
            f"<div style='font-size:11px;color:{config.TEXT_SECONDARY}'>Relationships Mapped</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
