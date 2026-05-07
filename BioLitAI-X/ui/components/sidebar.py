"""Sidebar navigation for BioLitAI-X."""

import streamlit as st
import config


_NAV_ITEMS = [
    ("🏠", "Home",                 "home"),
    ("📊", "Analysis",             "analysis"),
    ("🕸️", "Knowledge Graph",      "knowledge_graph"),
    ("💡", "Hypotheses",           "hypotheses"),
    ("🔍", "Semantic Search",      "semantic_search"),
    ("💬", "Chat with Literature", "chat"),
]

_STATUS_COLOR = {
    "complete": config.SUCCESS_COLOR,
    "running":  config.WARNING_COLOR,
    "error":    config.DANGER_COLOR,
    "pending":  config.TEXT_SECONDARY,
}


def render_sidebar() -> str:
    """
    Render the sidebar navigation.
    Returns the selected page key (e.g. 'home', 'analysis', ...).
    """
    with st.sidebar:
        # ── Logo / title ──────────────────────────────────────────────────
        st.markdown(
            f"""
            <div style="padding:0.5rem 0 1.25rem 0; text-align:center;">
              <div style="font-size:2rem; margin-bottom:4px;">🧬</div>
              <div style="font-size:1.25rem; font-weight:700;
                          background:linear-gradient(135deg,{config.PRIMARY_ACCENT},{config.SECONDARY_ACCENT});
                          -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                          background-clip:text;">
                BioLitAI-X
              </div>
              <div style="font-size:11px; color:{config.TEXT_SECONDARY}; font-weight:300;">
                Biomedical Literature Intelligence
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"<hr style='border-color:{config.BORDER_COLOR};margin:0 0 1rem 0'>",
            unsafe_allow_html=True,
        )

        # ── Navigation ────────────────────────────────────────────────────
        if "active_page" not in st.session_state:
            st.session_state.active_page = "home"

        for icon, label, key in _NAV_ITEMS:
            is_active = st.session_state.active_page == key
            btn_style = (
                f"background-color:{config.PRIMARY_ACCENT}; color:#fff;"
                if is_active
                else f"background-color:{config.SURFACE_ELEVATED}; color:{config.TEXT_PRIMARY};"
            )
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{key}",
                use_container_width=True,
                help=f"Go to {label}",
            ):
                st.session_state.active_page = key
                st.rerun()

        st.markdown(
            f"<hr style='border-color:{config.BORDER_COLOR};margin:1rem 0'>",
            unsafe_allow_html=True,
        )

        # ── Pipeline status ───────────────────────────────────────────────
        pipeline_complete = st.session_state.get("pipeline_complete", False)
        current_query    = st.session_state.get("current_query", "")
        papers_df        = st.session_state.get("papers_df")

        if pipeline_complete and current_query:
            st.markdown(
                f"<div style='font-size:11px;color:{config.TEXT_SECONDARY};"
                f"text-transform:uppercase;letter-spacing:.05em;font-weight:600;"
                f"margin-bottom:6px'>Active Query</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-size:12px;color:{config.TEXT_PRIMARY};"
                f"background:{config.SURFACE_ELEVATED};border:1px solid {config.BORDER_COLOR};"
                f"border-radius:6px;padding:6px 10px;word-break:break-word;"
                f"margin-bottom:10px'>{current_query}</div>",
                unsafe_allow_html=True,
            )

            if papers_df is not None and not papers_df.empty:
                n_papers = len(papers_df)
                n_authors = sum(
                    len(r) for r in papers_df.get("authors", [pd.Series()]) if isinstance(r, list)
                ) if "authors" in papers_df.columns else 0
                st.markdown(
                    f"<div style='font-size:12px;color:{config.TEXT_SECONDARY};"
                    f"margin-bottom:4px'>📄 {n_papers:,} papers loaded</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"<div style='display:flex;align-items:center;gap:6px;font-size:12px;"
                f"color:{config.SUCCESS_COLOR}'>"
                f"<div style='width:8px;height:8px;border-radius:50%;"
                f"background:{config.SUCCESS_COLOR}'></div>"
                f"Pipeline complete</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='font-size:12px;color:{config.TEXT_SECONDARY};"
                f"text-align:center;padding:8px 0'>"
                f"No active query.<br>Go to Home to search.</div>",
                unsafe_allow_html=True,
            )

        # ── Past sessions ─────────────────────────────────────────────────
        sessions = st.session_state.get("past_sessions", [])
        if sessions:
            st.markdown(
                f"<div style='font-size:11px;color:{config.TEXT_SECONDARY};"
                f"text-transform:uppercase;letter-spacing:.05em;font-weight:600;"
                f"margin:10px 0 6px'>Past Sessions</div>",
                unsafe_allow_html=True,
            )
            for sess in sessions[-5:][::-1]:
                q_text = sess.get("query_text", "")[:40]
                status = sess.get("pipeline_status", "pending")
                dot_color = _STATUS_COLOR.get(status, config.TEXT_SECONDARY)
                if st.button(
                    f"● {q_text}",
                    key=f"sess_{sess.get('id')}",
                    use_container_width=True,
                    help=f"Load: {sess.get('query_text', '')}",
                ):
                    st.session_state["restore_session_id"] = sess.get("id")
                    st.rerun()

    return st.session_state.get("active_page", "home")


# pandas import needed for author count
try:
    import pandas as pd
except ImportError:
    pass
