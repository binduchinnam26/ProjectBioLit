"""
PubMedRetriever — fetch PMIDs and raw XML records from PubMed via Biopython Entrez.

For large result sets (up to 10,000 papers) the retriever uses PubMed's server-side
history (WebEnv / query_key) so efetch never passes thousands of PMIDs as URL
parameters.  Batches of 500 are fetched in parallel (ThreadPoolExecutor, 10 workers)
while a threading.Lock serialises rate-limit accounting.  Results are re-ordered to
match the original PMID list so the rest of the pipeline sees a stable order.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Tuple

from Bio import Entrez

import config

logger = logging.getLogger(__name__)


def _configure_entrez():
    Entrez.email = config.ENTREZ_EMAIL
    if config.ENTREZ_API_KEY:
        Entrez.api_key = config.ENTREZ_API_KEY


class PubMedRetriever:
    """Fetches PMIDs and XML records from PubMed with rate limiting and retries."""

    def __init__(self):
        _configure_entrez()
        self._rate_limit = (
            config.RATE_LIMIT_WITH_KEY
            if config.ENTREZ_API_KEY
            else config.RATE_LIMIT_WITHOUT_KEY
        )
        self._min_interval = 1.0 / self._rate_limit
        self._last_call_time: float = 0.0
        self._rate_lock = threading.Lock()

    # ── Rate limiting (thread-safe) ───────────────────────────────────────────

    def _wait(self):
        with self._rate_lock:
            elapsed = time.monotonic() - self._last_call_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call_time = time.monotonic()

    # ── Retry wrapper ─────────────────────────────────────────────────────────

    def _call_with_retry(self, fn, *args, **kwargs):
        delay = config.FETCH_RETRY_DELAY
        for attempt in range(1, config.FETCH_RETRIES + 1):
            try:
                self._wait()
                return fn(*args, **kwargs)
            except Exception as exc:
                logger.warning(
                    "Entrez call failed (attempt %d/%d): %s",
                    attempt, config.FETCH_RETRIES, exc,
                )
                if attempt == config.FETCH_RETRIES:
                    raise
                time.sleep(delay)
                delay *= 2

    # ── Search (returns PMIDs + server-side history handles) ─────────────────

    def search(self, query: str, max_results: int) -> Tuple[List[str], str, str]:
        """
        Search PubMed and return (pmids, WebEnv, query_key).

        WebEnv / query_key are server-side history tokens used by fetch_batch_via_history()
        so efetch never needs to embed thousands of PMIDs in the request URL.
        max_results is clamped to [MAX_RESULTS_MIN, MAX_RESULTS_MAX].
        """
        if not query or not query.strip():
            raise ValueError("Query string must not be empty.")

        max_results = max(
            min(max_results, config.MAX_RESULTS_MAX),
            config.MAX_RESULTS_MIN,
        )

        logger.info("Searching PubMed: query=%r max=%d", query, max_results)

        handle = self._call_with_retry(
            Entrez.esearch,
            db="pubmed",
            term=query,
            retmax=max_results,
            usehistory="y",
        )
        record = Entrez.read(handle)
        handle.close()

        pmids: List[str] = record.get("IdList", [])
        web_env: str = record.get("WebEnv", "")
        query_key: str = record.get("QueryKey", "")
        count = int(record.get("Count", 0))

        logger.info(
            "PubMed: %d total hits, %d PMIDs retrieved, WebEnv=%s",
            count, len(pmids), bool(web_env),
        )

        if not pmids:
            raise ValueError(
                f"No results found for query '{query}'. "
                "Try broadening your search terms or check for typos."
            )

        return pmids, web_env, query_key

    # ── Batch fetch via server-side history ───────────────────────────────────

    def fetch_batch_via_history(
        self, web_env: str, query_key: str, ret_start: int, ret_max: int
    ) -> str:
        """
        Fetch one batch of XML using PubMed server-side history.
        Uses retstart/retmax instead of passing PMIDs in the URL — essential
        for large result sets where URL-embedded PMID lists become unwieldy.
        """
        handle = self._call_with_retry(
            Entrez.efetch,
            db="pubmed",
            webenv=web_env,
            query_key=query_key,
            retstart=ret_start,
            retmax=ret_max,
            rettype="xml",
            retmode="xml",
        )
        xml_data = handle.read()
        handle.close()
        return xml_data if isinstance(xml_data, str) else xml_data.decode("utf-8", errors="replace")

    def fetch_records(self, pmids: List[str]) -> str:
        """Fetch PubMed XML for a list of PMIDs directly (fallback / small batches)."""
        handle = self._call_with_retry(
            Entrez.efetch,
            db="pubmed",
            id=",".join(pmids),
            rettype="xml",
            retmode="xml",
        )
        xml_data = handle.read()
        handle.close()
        return xml_data if isinstance(xml_data, str) else xml_data.decode("utf-8", errors="replace")

    # ── Main public API ───────────────────────────────────────────────────────

    def fetch_with_progress(
        self,
        query: str,
        max_results: int,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Search PubMed and fetch all XML records using server-side history for
        efficiency.  Up to 10 batches fetched in parallel; thread-safe rate
        limiter ensures NCBI quota compliance.

        Returns (pmids, xml_batches) in stable order matching pmids.
        Each xml_batch contains FETCH_BATCH_SIZE (≤500) records.
        """
        def _cb(done: int, total: int, msg: str):
            if progress_callback:
                progress_callback(done, total, msg)

        _cb(0, 1, "Searching PubMed...")
        pmids, web_env, query_key = self.search(query, max_results)
        total = len(pmids)
        _cb(0, total, f"Found {total} papers. Fetching full records...")

        batch_size = config.FETCH_BATCH_SIZE
        # Build list of (batch_index, ret_start) for every batch
        batch_offsets = [
            (i, ret_start)
            for i, ret_start in enumerate(range(0, total, batch_size))
        ]
        n_batches = len(batch_offsets)
        xml_batches_map: dict = {}
        fetched_lock = threading.Lock()
        fetched_count = [0]

        def _fetch(idx_offset):
            idx, ret_start = idx_offset
            actual_ret_max = min(batch_size, total - ret_start)
            if web_env and query_key:
                xml = self.fetch_batch_via_history(
                    web_env, query_key, ret_start, actual_ret_max
                )
            else:
                # Fallback: use explicit PMIDs for this slice
                batch_pmids = pmids[ret_start: ret_start + actual_ret_max]
                xml = self.fetch_records(batch_pmids)
            return idx, xml, actual_ret_max

        max_workers = min(10, n_batches)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_fetch, offset): offset[0]
                for offset in batch_offsets
            }
            for future in as_completed(futures):
                try:
                    idx, xml, n = future.result()
                    xml_batches_map[idx] = xml
                    with fetched_lock:
                        fetched_count[0] += n
                        done = fetched_count[0]
                    _cb(done, total, f"Fetched {done:,}/{total:,} records...")
                except Exception as exc:
                    batch_idx = futures[future]
                    logger.error("fetch batch %d failed: %s", batch_idx, exc)
                    _cb(fetched_count[0], total,
                        f"Warning: batch {batch_idx} failed — continuing...")

        # Reconstruct in original order, skipping failed batches
        xml_batches = [
            xml_batches_map[i]
            for i in range(n_batches)
            if i in xml_batches_map
        ]

        retrieved = fetched_count[0]
        _cb(total, total, f"Fetch complete: {retrieved:,}/{total:,} records retrieved.")
        logger.info("fetch_with_progress done: %d/%d records in %d batches", retrieved, total, n_batches)
        return pmids, xml_batches
