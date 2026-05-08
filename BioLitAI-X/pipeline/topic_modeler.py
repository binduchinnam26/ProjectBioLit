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

    def setup(self, n_papers: int = 1000):
        """
        Initialise BERTopic with UMAP/HDBSCAN parameters scaled to *n_papers*.
        embedding_model=None because pre-computed embeddings are passed in
        fit_transform(); this avoids a second full-corpus encoding pass.
        calculate_probabilities=False cuts memory and time significantly.
        """
        import warnings
        # Suppress transformers internal path access noise
        warnings.filterwarnings("ignore", message=".*Accessing.*__path__.*")
        warnings.filterwarnings("ignore", message=".*image_processing.*")

        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer

        # Scale UMAP/HDBSCAN to dataset size for balanced granularity.
        # n_neighbors capped at 15: UMAP complexity is O(N * n_neighbors) so
        # 15 is ~3x faster than 50 with negligible quality loss for NLP tasks.
        # n_components=3 keeps enough structure for HDBSCAN while being faster.
        n_neighbors = max(10, min(15, n_papers // 100))
        min_cluster_size = max(config.BERTOPIC_MIN_TOPIC_SIZE, min(30, n_papers // 100))
        min_samples = max(5, min_cluster_size // 3)

        logger.info(
            "Initialising BERTopic (n_papers=%d, n_neighbors=%d, min_cluster_size=%d)",
            n_papers, n_neighbors, min_cluster_size,
        )

        umap_model = UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            metric="cosine",
            low_memory=True,
            verbose=False,
            random_state=42,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="leaf",
            prediction_data=True,
            core_dist_n_jobs=-1,
        )
        vectorizer_model = CountVectorizer(
            stop_words="english",
            min_df=2,
            ngram_range=(1, 2),
        )

        self.model = BERTopic(
            embedding_model=None,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            min_topic_size=min_cluster_size,
            nr_topics=None,
            verbose=False,
            calculate_probabilities=False,
        )

        self._ready = True
        logger.info("TopicModeler ready (n_papers=%d)", n_papers)

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

        Empty abstracts are excluded from BERTopic (which crashes on blank strings)
        but the returned topic list is padded back to the original length with -1
        so callers always get one topic ID per input document.

        If *embeddings_array* is provided (pre-computed from EmbeddingEngine),
        only the rows for valid documents are passed, avoiding a full re-encode.
        """
        self._check_ready()

        def _cb(done: int, total: int, msg: str):
            if progress_callback:
                progress_callback(done, total, msg)

        n_total = len(abstracts_list)

        # Build valid-document mask
        valid_indices = [
            i for i, a in enumerate(abstracts_list)
            if isinstance(a, str) and a.strip()
        ]

        if not valid_indices:
            logger.warning("fit_transform: no non-empty abstracts")
            self._topics = [-1] * n_total
            self._probs = np.array([])
            return self._topics, np.array([])

        valid_docs = [abstracts_list[i] for i in valid_indices]
        valid_embeddings = (
            embeddings_array[valid_indices]
            if embeddings_array is not None and len(embeddings_array) == n_total
            else None
        )

        _cb(0, len(valid_docs), "Fitting topic model on corpus...")

        try:
            if valid_embeddings is not None:
                topics_valid, probs = self.model.fit_transform(valid_docs, valid_embeddings)
            else:
                topics_valid, probs = self.model.fit_transform(valid_docs)
        except Exception as exc:
            logger.error("BERTopic fit_transform failed: %s", exc)
            raise

        # Reconstruct full-length topic list; empty docs → outlier (-1)
        full_topics = [-1] * n_total
        for idx, topic in zip(valid_indices, topics_valid):
            full_topics[idx] = int(topic)

        self._topics = full_topics
        self._probs = probs if probs is not None else np.array([])

        topic_info = self.model.get_topic_info()
        n_topics = len(topic_info[topic_info["Topic"] >= 0])
        _cb(n_total, n_total, f"Topic modeling complete: {n_topics} topics discovered")
        logger.info("fit_transform done: %d topics, %d/%d documents used", n_topics, len(valid_docs), n_total)

        return full_topics, self._probs

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

        n_papers = len(papers_df)
        abstracts = papers_df["abstract"].fillna("").tolist()
        timestamps = papers_df["pub_year"].fillna(0).astype(int).tolist()

        if not any(timestamps):
            logger.warning("get_topic_over_time: no publication years available")
            return pd.DataFrame()

        # BERTopic.topics_over_time requires all three arrays to be the same length.
        # self._topics is padded to n_papers in fit_transform, so we must pass it
        # explicitly rather than letting BERTopic use its internal self.topics_
        # (which has length n_valid_docs, not n_papers).
        topics = self._topics[:n_papers] if len(self._topics) >= n_papers else self._topics
        if len(topics) != n_papers:
            logger.warning(
                "get_topic_over_time: topics length %d != papers %d, skipping",
                len(topics), n_papers,
            )
            return pd.DataFrame()

        try:
            tot_df = self.model.topics_over_time(
                abstracts,
                timestamps,
                topics=topics,
                global_tuning=False,    # retuning adds 1-2 min; skip for speed
                evolution_tuning=False,
            )
            return tot_df
        except TypeError:
            # Older BERTopic versions don't accept the topics kwarg — fall back
            try:
                tot_df = self.model.topics_over_time(abstracts, timestamps)
                return tot_df
            except Exception as exc:
                logger.error("topics_over_time failed: %s", exc)
                return pd.DataFrame()
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
