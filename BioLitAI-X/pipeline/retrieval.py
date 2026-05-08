"""
PubMedRetriever — fetch PMIDs and XML records from PubMed via Biopython Entrez.

Uses PubMed server-side history (WebEnv / query_key) so efetch never embeds
thousands of PMIDs in a URL.  Batches are fetched sequentially with exponential-
backoff retry.  For the 2 000–3 000 paper target (<=10 batches of 300) sequential
fetching is reliable, avoids thread race conditions, and stays within NCBI quota.
"""

import logging
import time
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

    # ── Rate limiting ─────────────────────────────────────────────────────────

    def _wait(self):
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
        """Fetch one batch of XML using PubMed server-side history (no PMID list in URL)."""
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
        Search PubMed and fetch all XML records using server-side history.
        Batches are fetched sequentially — simple, stable, and quota-safe
        for the 2 000–3 000 paper target (<=10 batches of 300 records each).

        Returns (pmids, xml_batches) where each xml_batch is one efetch XML string.
        """
        def _cb(done: int, total: int, msg: str):
            if progress_callback:
                progress_callback(done, total, msg)

        _cb(0, 1, "Searching PubMed...")
        pmids, web_env, query_key = self.search(query, max_results)
        total = len(pmids)
        _cb(0, total, f"Found {total:,} papers. Fetching records...")

        batch_size = config.FETCH_BATCH_SIZE
        xml_batches: List[str] = []
        fetched = 0
        n_batches = (total + batch_size - 1) // batch_size

        for batch_num, ret_start in enumerate(range(0, total, batch_size), start=1):
            actual_ret_max = min(batch_size, total - ret_start)
            try:
                if web_env and query_key:
                    xml = self.fetch_batch_via_history(web_env, query_key, ret_start, actual_ret_max)
                else:
                    xml = self.fetch_records(pmids[ret_start: ret_start + actual_ret_max])
                xml_batches.append(xml)
                fetched += actual_ret_max
                _cb(fetched, total,
                    f"Fetched batch {batch_num}/{n_batches} ({fetched:,}/{total:,} records)...")
            except Exception as exc:
                logger.error("Fetch batch %d failed: %s", batch_num, exc)
                _cb(fetched, total, f"Warning: batch {batch_num} failed — continuing...")

        _cb(total, total, f"Fetch complete: {fetched:,}/{total:,} records retrieved.")
        logger.info(
            "fetch_with_progress done: %d/%d records in %d batches",
            fetched, total, len(xml_batches),
        )
        return pmids, xml_batches
