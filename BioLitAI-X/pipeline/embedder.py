"""
EmbeddingEngine — biomedical sentence embeddings using a sentence-transformer +
FAISS index for fast semantic search. Query-agnostic: works for any
topic retrieved from PubMed.

All embeddings and indexes are held in memory only; nothing is written to disk.
Abstract-missing papers use the title as fallback; if both are absent
a minimal stub is used so every paper gets a valid embedding.
"""

import gc
import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def _safe_text(text: Any) -> str:
    """Return a clean string. Converts NaN/None/non-str to empty string."""
    if text is None:
        return ""
    if not isinstance(text, str):
        return ""
    return text.strip()


class EmbeddingEngine:
    """
    Embeds paper abstracts with a biomedical sentence-transformer and
    holds the FAISS index in memory for cosine-similarity search.
    """

    def __init__(self):
        self.model = None
        self.index = None
        self._pmid_list: List[str] = []      # ordered — position i → FAISS vector i
        self._ready = False

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self):
        """Load sentence-transformer model and initialise FAISS index."""
        import os
        import warnings
        import torch

        # Pin CPU thread counts. set_num_interop_threads() must be called before
        # any inter-op parallelism starts; guard with a try so it's a no-op if
        # another library (spaCy, NLP pipeline) already started the PyTorch runtime.
        num_threads = os.cpu_count() or 4
        try:
            torch.set_num_threads(num_threads)
        except RuntimeError:
            pass
        try:
            torch.set_num_interop_threads(num_threads)
        except RuntimeError:
            pass
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)

        warnings.filterwarnings("ignore", message=".*Accessing.*__path__.*", category=UserWarning)
        import logging as _logging
        _logging.getLogger("transformers").setLevel(_logging.ERROR)
        _logging.getLogger("transformers.modeling_utils").setLevel(_logging.ERROR)
        _logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)

        from sentence_transformers import SentenceTransformer
        import faiss

        logger.info("Loading embedding model: %s (threads=%d)", config.EMBEDDING_MODEL, num_threads)
        self.model = SentenceTransformer(config.EMBEDDING_MODEL, token=config.HF_TOKEN)

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
        batch_size: int = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> np.ndarray:
        """
        Embed all papers in *papers_df* and add them to the in-memory FAISS index.

        Text priority per paper: abstract → title → stub.
        All papers receive an embedding; no data is dropped.

        Returns the full embeddings array (shape: [n_papers, embedding_dim]).
        """
        if batch_size is None:
            batch_size = config.EMBEDDING_BATCH_SIZE
        self._check_ready()

        def _cb(done: int, total: int, msg: str):
            if progress_callback:
                progress_callback(done, total, msg)

        if papers_df.empty:
            logger.warning("embed_corpus: empty DataFrame")
            return np.empty((0, config.EMBEDDING_DIMENSION), dtype="float32")

        pmids = papers_df["pmid"].tolist()
        _abs   = (papers_df["abstract"].fillna("").astype(str).str.strip()
                  if "abstract" in papers_df.columns
                  else pd.Series("", index=papers_df.index))
        _title = (papers_df["title"].fillna("").astype(str).str.strip()
                  if "title" in papers_df.columns
                  else pd.Series("", index=papers_df.index))
        _pmid  = (papers_df["pmid"].astype(str)
                  if "pmid" in papers_df.columns
                  else pd.Series("", index=papers_df.index))
        texts = [a or t or f"biomedical paper {p}"
                 for a, t, p in zip(_abs, _title, _pmid)]
        total = len(texts)

        self._init_index()

        _chunk_size = max(100, (total + 4) // 5)
        _chunks: List[np.ndarray] = []
        _cb(0, total, f"Embedding {total} papers…")
        try:
            import torch
            with torch.inference_mode():
                for _start in range(0, total, _chunk_size):
                    _batch_texts = texts[_start:_start + _chunk_size]
                    _emb = self.model.encode(
                        _batch_texts,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        device="cpu",
                    ).astype("float32")
                    _chunks.append(_emb)
                    _done = min(_start + _chunk_size, total)
                    _cb(_done, total,
                        f"Embedding: {_done}/{total} papers ({_done * 100 // total}%)…")
            embeddings = (np.vstack(_chunks) if _chunks
                          else np.empty((0, config.EMBEDDING_DIMENSION), dtype="float32"))
        except Exception as exc:
            logger.error("model.encode() failed: %s", exc)
            embeddings = np.zeros((total, config.EMBEDDING_DIMENSION), dtype="float32")

        self.index.add(embeddings)
        gc.collect()
        self._pmid_list.extend(pmids)
        _cb(total, total, f"Embeddings complete: {total} vectors stored.")
        logger.info("embed_corpus done: %d vectors", total)
        return embeddings

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
