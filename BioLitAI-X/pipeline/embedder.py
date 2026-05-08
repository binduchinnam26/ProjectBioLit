"""
EmbeddingEngine — biomedical sentence embeddings using PubMedBERT +
FAISS index for fast semantic search. Query-agnostic: works for any
topic retrieved from PubMed.

Tuned for 2k–3k papers: batch_size=64 keeps RAM usage flat while
still being much faster than encoding one abstract at a time.
Abstract-missing papers use the title as fallback; if both are absent
a minimal stub is used so every paper gets a valid embedding.
"""

import gc
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from utils.helpers import query_hash

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
    stores / retrieves them via FAISS for cosine-similarity search.
    """

    def __init__(self, persist_index: bool = True):
        self.model = None
        self.index = None
        self._pmid_list: List[str] = []      # ordered — position i → FAISS vector i
        self._current_query_hash: Optional[str] = None
        self._persist_index = persist_index  # False when disk space is low
        self._ready = False

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self):
        """Load sentence-transformer model and initialise FAISS index."""
        import os
        import warnings
        import torch

        # Pin CPU threads before model load so they apply to all downstream compute
        num_threads = os.cpu_count() or 4
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)

        # Suppress HF Hub authentication advisory (noise, not an error)
        warnings.filterwarnings("ignore", message=".*unauthenticated requests.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
        warnings.filterwarnings("ignore", message=".*Accessing.*__path__.*", category=UserWarning)
        # Silence transformers library's own logger (__path__ access messages)
        import logging as _logging
        _logging.getLogger("transformers").setLevel(_logging.ERROR)
        _logging.getLogger("transformers.modeling_utils").setLevel(_logging.ERROR)

        from sentence_transformers import SentenceTransformer
        import faiss

        logger.info("Loading embedding model: %s (threads=%d)", config.EMBEDDING_MODEL, num_threads)
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
        batch_size: int = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> np.ndarray:
        """
        Embed all papers in *papers_df*, add them to the FAISS index,
        and persist both index and PMID mapping to disk.

        Text priority per paper: abstract → title → stub.
        All papers receive an embedding; no data is dropped.
        batch_size defaults to config.EMBEDDING_BATCH_SIZE (64) to keep
        RAM usage low for 2k-3k paper corpora.

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
        texts = [
            _safe_text(row.get("abstract"))
            or _safe_text(row.get("title"))
            or f"biomedical paper {row.get('pmid', '')}"
            for _, row in papers_df.iterrows()
        ]
        total = len(texts)

        _cb(0, total, "Embedding corpus — this may take a few minutes...")
        self._init_index()
        self._current_query_hash = query_hash(query) if query else "default"

        try:
            import torch
            with torch.inference_mode():
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    device="cpu",
                ).astype("float32")
        except Exception as exc:
            logger.error("model.encode() failed: %s", exc)
            embeddings = np.zeros((total, config.EMBEDDING_DIMENSION), dtype="float32")

        self.index.add(embeddings)
        gc.collect()
        self._pmid_list.extend(pmids)

        if self._persist_index:
            self._save_index(query, embeddings)
        else:
            logger.info("embed_corpus: skipping index/embedding save (low disk mode)")
        _cb(total, total, f"Embeddings complete: {total} vectors stored.")
        logger.info("embed_corpus done: %d vectors", total)
        return embeddings

    # ── Persistence ───────────────────────────────────────────────────────────

    def _index_paths(self, query: str) -> Tuple[Path, Path]:
        """Return (faiss_path, mapping_path) for a given query string."""
        qh = query_hash(query) if query else "default"
        base = Path(config.EMBEDDINGS_DIR) / qh
        return base.with_suffix(".faiss"), base.with_suffix(".json")

    def _embeddings_path(self, query: str) -> Path:
        """Return path for the numpy embeddings cache file."""
        qh = query_hash(query) if query else "default"
        return Path(config.EMBEDDINGS_DIR) / f"{qh}.npy"

    def _save_index(self, query: str, embeddings: np.ndarray):
        """Save FAISS index, PMID map, and raw embeddings array to disk."""
        import faiss
        faiss_path, map_path = self._index_paths(query)
        emb_path = self._embeddings_path(query)
        try:
            faiss.write_index(self.index, str(faiss_path))
            map_path.write_text(json.dumps(self._pmid_list), encoding="utf-8")
            np.save(str(emb_path), embeddings)
            logger.info("FAISS index + embeddings saved to %s", faiss_path)
        except OSError as exc:
            logger.error("Failed to save FAISS index (disk full?): %s", exc)
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
            loaded = faiss.read_index(str(faiss_path))
            if loaded.d != config.EMBEDDING_DIMENSION:
                logger.warning(
                    "Stale FAISS index (dim=%d) does not match current model (dim=%d) — ignoring cache",
                    loaded.d, config.EMBEDDING_DIMENSION,
                )
                return False
            self.index = loaded
            self._pmid_list = json.loads(map_path.read_text(encoding="utf-8"))
            self._current_query_hash = query_hash(query)
            logger.info(
                "Loaded FAISS index (%d vectors) from %s", len(self._pmid_list), faiss_path
            )
            return True
        except Exception as exc:
            logger.error("Failed to load FAISS index: %s", exc)
            return False

    def load_embeddings(self, query: str) -> Optional[np.ndarray]:
        """Load the cached numpy embeddings array for *query*. Returns None on miss."""
        emb_path = self._embeddings_path(query)
        if not emb_path.exists():
            return None
        try:
            arr = np.load(str(emb_path))
            if arr.ndim == 2 and arr.shape[1] != config.EMBEDDING_DIMENSION:
                logger.warning(
                    "Stale embeddings cache (dim=%d) does not match current model (dim=%d) — ignoring",
                    arr.shape[1], config.EMBEDDING_DIMENSION,
                )
                return None
            logger.info("Loaded embeddings array (%d vectors) from %s", len(arr), emb_path)
            return arr
        except Exception as exc:
            logger.error("Failed to load embeddings array: %s", exc)
            return None

    def index_exists(self, query: str) -> bool:
        """True only if FAISS index, PMID map, AND numpy array are all present."""
        faiss_path, map_path = self._index_paths(query)
        emb_path = self._embeddings_path(query)
        return faiss_path.exists() and map_path.exists() and emb_path.exists()

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
