"""
PubMedRetriever — fetch PMIDs and raw XML records from PubMed via Biopython Entrez.

fetch_with_progress() uses a ThreadPoolExecutor (5 workers) to fetch XML batches
in parallel while a threading.Lock serialises rate-limit accounting so we never
exceed the NCBI API quota.  Results are re-ordered to match the original PMID list.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Generator, List, Optional, Tuple

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
                    attempt,
                    config.FETCH_RETRIES,
                    exc,
                )
                if attempt == config.FETCH_RETRIES:
                    raise
                time.sleep(delay)
                delay *= 2

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query: str, max_results: int) -> List[str]:
        """Return list of PMIDs matching *query* (up to *max_results*)."""
        if not query or not query.strip():
            raise ValueError("Query string must not be empty.")

        max_results = min(
            max(max_results, config.MAX_RESULTS_MIN), config.MAX_RESULTS_MAX
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
        count = int(record.get("Count", 0))

        logger.info("PubMed returned %d total hits; retrieved %d PMIDs", count, len(pmids))

        if not pmids:
            raise ValueError(
                f"No results found for query '{query}'. "
                "Try broadening your search terms or check for typos."
            )

        return pmids

    def fetch_records(self, pmids: List[str]) -> str:
        """Fetch PubMed XML for a list of PMIDs in one batch call."""
        ids = ",".join(pmids)
        handle = self._call_with_retry(
            Entrez.efetch,
            db="pubmed",
            id=ids,
            rettype="xml",
            retmode="xml",
        )
        xml_data = handle.read()
        handle.close()
        return xml_data if isinstance(xml_data, str) else xml_data.decode("utf-8", errors="replace")

    def _batch_pmids(self, pmids: List[str]) -> Generator[List[str], None, None]:
        """Yield FETCH_BATCH_SIZE-sized chunks of pmids."""
        size = config.FETCH_BATCH_SIZE
        for i in range(0, len(pmids), size):
            yield pmids[i : i + size]

    def fetch_with_progress(
        self,
        query: str,
        max_results: int,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Search and fetch all records in parallel, reporting progress via
        *progress_callback(completed, total, message)*.

        Up to 5 batches are fetched concurrently while the thread-safe rate
        limiter ensures NCBI quota compliance.  Results are returned in the
        same order as the original PMID list.

        Returns (pmids, xml_batches) where xml_batches is a list of raw XML
        strings, one per batch of FETCH_BATCH_SIZE.
        """
        def _cb(done: int, total: int, msg: str):
            if progress_callback:
                progress_callback(done, total, msg)

        _cb(0, 1, "Searching PubMed...")
        pmids = self.search(query, max_results)
        total = len(pmids)
        _cb(0, total, f"Found {total} papers. Fetching records...")

        batches = list(self._batch_pmids(pmids))
        xml_batches_map: dict = {}
        fetched_lock = threading.Lock()
        fetched_count = [0]

        def _fetch_batch(idx_batch):
            idx, batch = idx_batch
            xml = self.fetch_records(batch)
            return idx, xml, len(batch)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(_fetch_batch, (i, batch)): i
                for i, batch in enumerate(batches)
            }
            for future in as_completed(futures):
                try:
                    idx, xml, n = future.result()
                    xml_batches_map[idx] = xml
                    with fetched_lock:
                        fetched_count[0] += n
                        done = fetched_count[0]
                    _cb(done, total, f"Fetched {done}/{total} records...")
                except Exception as exc:
                    batch_idx = futures[future]
                    logger.error("fetch_records batch %d failed: %s", batch_idx, exc)
                    _cb(fetched_count[0], total, f"Warning: batch {batch_idx} failed — {exc}")

        # Reconstruct in original order, skipping failed batches
        xml_batches = [
            xml_batches_map[i]
            for i in range(len(batches))
            if i in xml_batches_map
        ]

        _cb(total, total, f"Fetch complete. {fetched_count[0]} records retrieved.")
        return pmids, xml_batches
