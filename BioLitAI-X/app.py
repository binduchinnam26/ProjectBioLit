"""
BioLitAI-X — Main Streamlit application entry point.
Configures page, loads CSS, initialises session state, routes pages.
"""

import sys
import os

# Ensure the project root is on the path regardless of working directory
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st

# ── Page configuration (must be the very first Streamlit call) ────────────────
st.set_page_config(
    page_title="BioLitAI-X",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load custom CSS ───────────────────────────────────────────────────────────
_CSS_PATH = os.path.join(_ROOT, "ui", "styles", "theme.css")
if os.path.exists(_CSS_PATH):
    with open(_CSS_PATH, "r", encoding="utf-8") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "pipeline_complete":  False,
    "papers_df":          None,
    "entities_df":        None,
    "relationships_df":   None,
    "knowledge_graph":    None,
    "kg_graph":           None,
    "coauth_graph":       None,
    "keyword_graph":      None,
    "topic_graph":        None,
    "hypotheses":         [],
    "current_query":      "",
    "embeddings_ready":   False,
    "active_session_id":  None,
    "embedder":           None,
    "embeddings_array":   None,
    "coauth_stats":       {},
    "keyword_stats":      {},
    "topic_labels":       {},
    "topics_over_time":   None,
    "doc_topics_df":      None,
    "gap_report":         [],
    "chat_history":       [],
    "past_sessions":      [],
    "db":                 None,
    "active_page":        "home",
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Sidebar navigation ────────────────────────────────────────────────────────
from ui.components.sidebar import render_sidebar

active_page = render_sidebar()

# ── Page routing ──────────────────────────────────────────────────────────────
if active_page == "home":
    from ui.pages.home import render as _render
    _render()

elif active_page == "analysis":
    from ui.pages.analysis import render as _render
    _render()

elif active_page == "knowledge_graph":
    from ui.pages.knowledge_graph import render as _render
    _render()

elif active_page == "hypotheses":
    from ui.pages.hypotheses import render as _render
    _render()

elif active_page == "semantic_search":
    from ui.pages.semantic_search import render as _render
    _render()

elif active_page == "chat":
    from ui.pages.chat import render as _render
    _render()

else:
    from ui.pages.home import render as _render
    _render()
