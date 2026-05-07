"""
EmbeddingEngine — biomedical sentence embeddings using PubMedBERT +
FAISS index for fast semantic search. Query-agnostic: works for any
topic retrieved from PubMed.
"""

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from utils.helpers import query_hash

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Embeds paper abstracts with a biomedical sentence-transformer and
    stores / retrieves them via FAISS for cosine-similarity search.
    """

    def __init__(self):
        self.model = None
        self.index = None
        self._pmid_list: List[str] = []      # ordered — position i → FAISS vector i
        self._current_query_hash: Optional[str] = None
        self._ready = False

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self):
        """Load sentence-transformer model and initialise FAISS index."""
        from sentence_transformers import SentenceTransformer
        import faiss

        logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)

        self._init_index()
        self._ready = True
        logger.info("EmbeddingEngine ready (dim=%d)", config.EMBEDDING_DIMENSION)

    def _init_index(self):
        import faiss
        self.index = faiss.IndexFlatIP(config.EMBEDDING_DIMENSION)
        self._pmid_list = []

    def _check_ready(self):
        if not self._ready or self.model is None:
            raise RuntimeError("EmbeddingEngine.setup() must be called first.")

    # ── Single-abstract embedding ─────────────────────────────────────────────

    def embed_abstract(self, text: str) -> np.ndarray:
        """
        Embed a single abstract and return an L2-normalised vector
        suitable for cosine similarity via inner-product FAISS search.
        """
        self._check_ready()
        vec = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0].astype("float32")

    # ── Corpus embedding ──────────────────────────────────────────────────────

    def embed_corpus(
        self,
        papers_df: pd.DataFrame,
        query: str = "",
        batch_size: int = 64,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> np.ndarray:
        """
        Embed all abstracts in *papers_df*, add them to the FAISS index,
        and persist both index and PMID mapping to disk.

        Index file name is derived from a hash of *query* so different
        queries each get a separate stored index.

        Returns the full embeddings array (shape: [n_papers, embedding_dim]).
        """
        self._check_ready()

        def _cb(done: int, total: int, msg: str):
            if progress_callback:
                progress_callback(done, total, msg)

        if papers_df.empty:
            logger.warning("embed_corpus: empty DataFrame")
            return np.empty((0, config.EMBEDDING_DIMENSION), dtype="float32")

        pmids = papers_df["pmid"].tolist()
        abstracts = papers_df["abstract"].fillna("").tolist()
        total = len(abstracts)

        _cb(0, total, "Initialising embedding index...")
        self._init_index()
        self._current_query_hash = query_hash(query) if query else "default"

        all_embeddings: List[np.ndarray] = []

        for start in range(0, total, batch_size):
            batch_texts = abstracts[start : start + batch_size]
            batch_pmids = pmids[start : start + batch_size]

            try:
                vecs = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=batch_size,
                )
                vecs = vecs.astype("float32")
            except Exception as exc:
                logger.error("Embedding batch %d failed: %s", start, exc)
                vecs = np.zeros(
                    (len(batch_texts), config.EMBEDDING_DIMENSION), dtype="float32"
                )

            self.index.add(vecs)
            self._pmid_list.extend(batch_pmids)
            all_embeddings.append(vecs)

            done = min(start + batch_size, total)
            _cb(done, total, f"Embedded {done}/{total} abstracts...")

        full_embeddings = np.vstack(all_embeddings)
        self._persist_index(query)
        _cb(total, total, f"Embeddings complete: {total} vectors stored.")
        logger.info("embed_corpus done: %d vectors, index saved", total)
        return full_embeddings

    # ── Persistence ───────────────────────────────────────────────────────────

    def _index_paths(self, query: str) -> Tuple[Path, Path]:
        """Return (faiss_path, mapping_path) for a given query string."""
        qh = query_hash(query) if query else "default"
        base = Path(config.EMBEDDINGS_DIR) / qh
        return base.with_suffix(".faiss"), base.with_suffix(".json")

    def _persist_index(self, query: str):
        import faiss
        faiss_path, map_path = self._index_paths(query)
        try:
            faiss.write_index(self.index, str(faiss_path))
            map_path.write_text(json.dumps(self._pmid_list), encoding="utf-8")
            logger.info("FAISS index saved to %s", faiss_path)
        except Exception as exc:
            logger.error("Failed to save FAISS index: %s", exc)

    def load_index(self, query: str) -> bool:
        """
        Load a previously saved FAISS index for *query* from disk.
        Returns True on success, False if no saved index exists.
        """
        import faiss
        faiss_path, map_path = self._index_paths(query)
        if not faiss_path.exists() or not map_path.exists():
            return False
        try:
            self._check_ready()
            self.index = faiss.read_index(str(faiss_path))
            self._pmid_list = json.loads(map_path.read_text(encoding="utf-8"))
            self._current_query_hash = query_hash(query)
            logger.info(
                "Loaded FAISS index (%d vectors) from %s", len(self._pmid_list), faiss_path
            )
            return True
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc)
            return False

    def index_exists(self, query: str) -> bool:
        faiss_path, map_path = self._index_paths(query)
        return faiss_path.exists() and map_path.exists()

    # ── Semantic search ───────────────────────────────────────────────────────

    def semantic_search(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Embed *query_text* at runtime and search the loaded FAISS index.

        Returns list of dicts with keys: pmid, score, rank.
        Score is cosine similarity in [0, 1] (inner product of L2-normalised vecs).
        """
        self._check_ready()

        if self.index is None or self.index.ntotal == 0:
            logger.warning("semantic_search: FAISS index is empty")
            return []

        k = min(top_k, self.index.ntotal)
        query_vec = self.embed_abstract(query_text).reshape(1, -1)

        try:
            scores, indices = self.index.search(query_vec, k)
        except Exception as exc:
            logger.error("FAISS search failed: %s", exc)
            return []

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0 or idx >= len(self._pmid_list):
                continue
            results.append(
                {
                    "pmid": self._pmid_list[idx],
                    "score": float(score),
                    "rank": rank + 1,
                }
            )

        return results
