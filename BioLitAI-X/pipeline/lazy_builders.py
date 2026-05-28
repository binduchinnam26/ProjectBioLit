"""
Lazy pipeline builders — on-demand computation of topics, entities, and relationships.

Called from UI pages when optional heavy steps were skipped during the initial pipeline run.
All session-state writes and cache updates are handled here so UI pages stay free of
pipeline logic.
"""

import logging
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from pipeline.knowledge_graph import KnowledgeGraph
from pipeline.network_builder import NetworkBuilder
from pipeline.nlp_processor import NLPProcessor
from pipeline.topic_modeler import TopicModeler
from utils.helpers import query_hash

logger = logging.getLogger(__name__)


def build_topics_lazy(papers_df: pd.DataFrame) -> None:
    """Run BERTopic on demand and persist results into session state + derived cache."""
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
            qh = query_hash(query) if query else "default"
            derived_cache = Path(config.PROCESSED_DIR) / f"{qh}_derived.pkl"
            if derived_cache.exists():
                with open(derived_cache, "rb") as _f:
                    _d = pickle.load(_f)
                _d.update({
                    "topic_labels":     topic_labels,
                    "topics_over_time": topics_over_time,
                    "doc_topics_df":    doc_topics_df,
                    "topic_graph":      topic_graph,
                })
                with open(derived_cache, "wb") as _f:
                    pickle.dump(_d, _f, protocol=pickle.HIGHEST_PROTOCOL)
            st.success(f"Topic model complete: {len(topic_labels)} topics discovered.")
        except Exception as exc:
            logger.error("Lazy topic build failed: %s", exc)
            st.error(f"Topic modelling failed: {exc}")


def extract_entities_lazy(papers_df: pd.DataFrame) -> None:
    """Extract biomedical entities on demand and persist results into session state + derived cache."""
    query = st.session_state.get("current_query", "")

    with st.spinner(
        f"Extracting entities from {len(papers_df):,} papers "
        f"(NER-only — typically 30-90 sec)…"
    ):
        try:
            nlp_p = NLPProcessor(db_manager=None)
            nlp_p.setup()
            entities_df, _ = nlp_p.process_corpus(papers_df)
            kg = KnowledgeGraph()
            kg_graph = kg.build_from_entities(entities_df, pd.DataFrame())
            st.session_state["entities_df"]    = entities_df
            st.session_state["kg_graph"]       = kg_graph
            st.session_state["knowledge_graph"] = kg
            qh = query_hash(query) if query else "default"
            derived_cache = Path(config.PROCESSED_DIR) / f"{qh}_derived.pkl"
            if derived_cache.exists():
                with open(derived_cache, "rb") as _f:
                    _d = pickle.load(_f)
                _d.update({"entities_df": entities_df, "kg_graph": kg_graph})
                with open(derived_cache, "wb") as _f:
                    pickle.dump(_d, _f, protocol=pickle.HIGHEST_PROTOCOL)
            st.success(
                f"Entity extraction complete: {len(entities_df):,} entities from "
                f"{len(papers_df):,} papers."
            )
        except Exception as exc:
            logger.error("Lazy entity extraction failed: %s", exc)
            st.error(f"Entity extraction failed: {exc}")


def extract_relationships_lazy(papers_df: pd.DataFrame, entities_df: pd.DataFrame) -> None:
    """Extract relationships on demand and persist results into session state + derived cache."""
    query = st.session_state.get("current_query", "")

    with st.spinner(
        f"Extracting relationships from {len(papers_df):,} papers "
        f"(dep parser — may take 3-6 min)…"
    ):
        try:
            nlp_p = NLPProcessor(db_manager=None)
            nlp_p.setup()
            rels_df = nlp_p.extract_relationships_corpus(papers_df, entities_df)
            kg = KnowledgeGraph()
            kg_graph = kg.build_from_entities(entities_df, rels_df)
            st.session_state["relationships_df"] = rels_df
            st.session_state["kg_graph"]         = kg_graph
            st.session_state["knowledge_graph"]  = kg
            qh = query_hash(query) if query else "default"
            derived_cache = Path(config.PROCESSED_DIR) / f"{qh}_derived.pkl"
            if derived_cache.exists():
                with open(derived_cache, "rb") as _f:
                    _d = pickle.load(_f)
                _d.update({"relationships_df": rels_df, "kg_graph": kg_graph})
                with open(derived_cache, "wb") as _f:
                    pickle.dump(_d, _f, protocol=pickle.HIGHEST_PROTOCOL)
            st.success(
                f"Relationships extracted: {len(rels_df):,} edges added to knowledge graph."
            )
        except Exception as exc:
            logger.error("Lazy relationship extraction failed: %s", exc)
            st.error(f"Relationship extraction failed: {exc}")
