"""
TopicModeler — dynamic topic discovery using BERTopic.
Topics are discovered entirely from the retrieved corpus; no predefined
categories. Works for any biomedical domain.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


class TopicModeler:
    """
    Wraps BERTopic for dynamic topic discovery and temporal analysis.
    Topics are data-driven: labels and count are generated from the corpus,
    never hardcoded.
    """

    def __init__(self):
        self.model = None
        self._topics: Optional[List[int]] = None
        self._probs: Optional[np.ndarray] = None
        self._ready = False

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self):
        """
        Initialise BERTopic with the biomedical embedding model,
        auto topic count, and minimum topic size from config.
        """
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer

        logger.info("Initialising BERTopic (min_topic_size=%d)", config.BERTOPIC_MIN_TOPIC_SIZE)

        embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

        self.model = BERTopic(
            embedding_model=embedding_model,
            min_topic_size=config.BERTOPIC_MIN_TOPIC_SIZE,
            nr_topics="auto",
            verbose=True,
            calculate_probabilities=True,
        )

        self._ready = True
        logger.info("TopicModeler ready")

    def _check_ready(self):
        if not self._ready or self.model is None:
            raise RuntimeError("TopicModeler.setup() must be called first.")

    # ── Fit & transform ───────────────────────────────────────────────────────

    def fit_transform(
        self,
        abstracts_list: List[str],
        embeddings_array: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Fit BERTopic on the corpus and return (topic_assignments, probabilities).

        If *embeddings_array* is provided (pre-computed from EmbeddingEngine),
        BERTopic skips its own embedding step — faster and consistent.

        topic_assignments: list of int topic IDs per document (-1 = outlier)
        probabilities:     array of shape [n_docs, n_topics]
        """
        self._check_ready()

        def _cb(done: int, total: int, msg: str):
            if progress_callback:
                progress_callback(done, total, msg)

        valid_abstracts = [a if isinstance(a, str) and a.strip() else "" for a in abstracts_list]

        if not any(valid_abstracts):
            logger.warning("fit_transform: no non-empty abstracts")
            self._topics = []
            self._probs = np.array([])
            return [], np.array([])

        _cb(0, len(valid_abstracts), "Fitting topic model on corpus...")

        try:
            if embeddings_array is not None and len(embeddings_array) == len(valid_abstracts):
                topics, probs = self.model.fit_transform(valid_abstracts, embeddings_array)
            else:
                topics, probs = self.model.fit_transform(valid_abstracts)
        except Exception as exc:
            logger.error("BERTopic fit_transform failed: %s", exc)
            raise

        self._topics = topics
        self._probs = probs if probs is not None else np.array([])

        topic_info = self.model.get_topic_info()
        n_topics = len(topic_info[topic_info["Topic"] >= 0])
        _cb(len(valid_abstracts), len(valid_abstracts), f"Topic modeling complete: {n_topics} topics discovered")
        logger.info("fit_transform done: %d topics, %d documents", n_topics, len(valid_abstracts))

        return topics, self._probs

    # ── Topic-over-time ───────────────────────────────────────────────────────

    def get_topic_over_time(self, papers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute topic prevalence over publication years using BERTopic's
        topics_over_time method.

        Returns a DataFrame with columns: Topic, Words, Frequency, Timestamp.
        """
        self._check_ready()

        if self._topics is None:
            raise RuntimeError("fit_transform must be called before get_topic_over_time.")

        abstracts = papers_df["abstract"].fillna("").tolist()
        timestamps = papers_df["pub_year"].fillna(0).astype(int).tolist()

        if not any(timestamps):
            logger.warning("get_topic_over_time: no publication years available")
            return pd.DataFrame()

        try:
            tot_df = self.model.topics_over_time(
                abstracts,
                timestamps,
                global_tuning=True,
                evolution_tuning=True,
            )
            return tot_df
        except Exception as exc:
            logger.error("topics_over_time failed: %s", exc)
            return pd.DataFrame()

    # ── Topic labels ──────────────────────────────────────────────────────────

    def get_topic_labels(self) -> Dict[int, Dict]:
        """
        Return human-readable topic labels with top keywords.
        Labels are generated from the data — never hardcoded.

        Returns dict: {topic_id: {label, top_words, count}}
        """
        self._check_ready()

        topic_info = self.model.get_topic_info()
        labels: Dict[int, Dict] = {}

        for _, row in topic_info.iterrows():
            tid = int(row["Topic"])
            if tid == -1:
                continue  # outlier bucket

            top_words_data = self.model.get_topic(tid)
            top_words = (
                [w for w, _ in top_words_data[:10]]
                if top_words_data
                else []
            )

            # Build a concise label from top 3 keywords
            label = " | ".join(top_words[:3]) if top_words else f"Topic {tid}"

            labels[tid] = {
                "label": label,
                "top_words": top_words,
                "count": int(row.get("Count", 0)),
            }

        return labels

    def get_document_topics(self, papers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame mapping each paper (PMID) to its topic assignment
        and probability.

        Columns: pmid, topic_id, probability, topic_label
        """
        if self._topics is None or papers_df.empty:
            return pd.DataFrame(
                columns=["pmid", "topic_id", "probability", "topic_label"]
            )

        labels = self.get_topic_labels()
        pmids = papers_df["pmid"].tolist()
        rows = []

        for i, (pmid, tid) in enumerate(zip(pmids, self._topics)):
            prob = float(self._probs[i].max()) if self._probs is not None and len(self._probs) > i else 0.0
            topic_label = labels.get(tid, {}).get("label", f"Topic {tid}") if tid >= 0 else "Outlier"
            rows.append(
                {"pmid": pmid, "topic_id": int(tid), "probability": prob, "topic_label": topic_label}
            )

        return pd.DataFrame(rows)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_model(self, path: str):
        """Save the fitted BERTopic model to *path*."""
        self._check_ready()
        try:
            save_dir = Path(path)
            save_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(save_dir), serialization="pickle", save_ctfidf=True)
            logger.info("BERTopic model saved to %s", save_dir)
        except Exception as exc:
            logger.error("save_model failed: %s", exc)
            raise

    def load_model(self, path: str):
        """Load a previously saved BERTopic model from *path*."""
        from bertopic import BERTopic
        try:
            self.model = BERTopic.load(path)
            self._ready = True
            logger.info("BERTopic model loaded from %s", path)
        except Exception as exc:
            logger.error("load_model failed: %s", exc)
            raise
