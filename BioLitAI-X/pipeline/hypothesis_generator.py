"""
HypothesisGenerator — LLM-powered hypothesis generation and literature
chat using the Gemini REST API directly (no gRPC, no SDK transport layer).

All hypotheses are strictly grounded in retrieved paper evidence
(PMIDs cited in every response). Works for any biomedical domain.
"""

import logging
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

import config

logger = logging.getLogger(__name__)

# ── Prompt constants ──────────────────────────────────────────────────────────
_SYSTEM_INSTRUCTION = (
    "You are an expert biomedical research scientist specializing in hypothesis "
    "generation from literature evidence. You only generate hypotheses grounded "
    "in the provided evidence. Never speculate beyond what the evidence supports. "
    "Always cite specific PMIDs from the provided context in your response. "
    "Be precise, concise, and scientifically rigorous."
)

_CHAT_SYSTEM = (
    "You are a biomedical research assistant. Answer the following question using "
    "ONLY the evidence from the papers provided below. If the answer cannot be "
    "found in these papers, say so clearly. Always cite PMIDs."
)

_CONFIDENCE_MAP = {"low": 1, "medium": 2, "high": 3}

_GEMINI_REST_BASE = "https://generativelanguage.googleapis.com/v1beta"


class HypothesisGenerator:
    """
    Generates research hypotheses from knowledge graph gaps using Gemini,
    and supports multi-turn literature chat grounded in FAISS semantic search.

    Uses the Gemini REST API directly via requests (verify=False) to avoid
    gRPC SSL certificate failures in restricted network environments.
    """

    def __init__(self, db_manager=None, embedding_engine=None):
        self.db = db_manager
        self.embedder = embedding_engine
        self._api_key: Optional[str] = None
        self._session = None          # requests.Session
        self._ready = False

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self):
        """
        Initialise a requests.Session pointed at the Gemini REST API.
        SSL verification is disabled to work in environments with a
        self-signed corporate proxy certificate in the chain.
        Raises EnvironmentError if GEMINI_API_KEY is missing.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not found in .env file. Please add your "
                "Google Gemini API key as GEMINI_API_KEY=your_key_here "
                "in the .env file before running."
            )

        import requests
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self._api_key = api_key
        self._session = requests.Session()
        self._session.verify = False   # bypass self-signed cert chain
        self._session.headers.update({"x-goog-api-key": api_key})

        # Smoke-test connectivity with a minimal prompt
        try:
            test = self._call_gemini_with_retry("Reply with the single word: ready", max_retries=1)
            if test:
                logger.info("Gemini REST API reachable (model=%s)", config.GEMINI_MODEL)
            else:
                logger.warning("Gemini REST API test call returned no text — quota or model issue")
        except Exception as exc:
            logger.warning("Gemini REST API smoke test failed: %s", exc)

        self._ready = True
        logger.info("HypothesisGenerator ready (transport=REST, model=%s)", config.GEMINI_MODEL)

    def _check_ready(self):
        if not self._ready or self._session is None:
            raise RuntimeError(
                "HypothesisGenerator.setup() must be called before use."
            )

    # ── Gemini REST call ──────────────────────────────────────────────────────

    def _call_gemini_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
    ) -> Optional[str]:
        """
        POST to the Gemini generateContent REST endpoint with exponential
        backoff.  Returns the text string on success, None on total failure.
        """
        url = f"{_GEMINI_REST_BASE}/models/{config.GEMINI_MODEL}:generateContent"
        payload = {
            "system_instruction": {"parts": [{"text": _SYSTEM_INSTRUCTION}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": config.GEMINI_TEMPERATURE,
                "topP":        config.GEMINI_TOP_P,
                "topK":        config.GEMINI_TOP_K,
                "maxOutputTokens": config.GEMINI_MAX_OUTPUT_TOKENS,
            },
        }

        delay = 2.0
        last_exc: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                resp = self._session.post(url, json=payload, timeout=60)
                if resp.status_code == 429:
                    raise RuntimeError(f"429 RESOURCE_EXHAUSTED: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]

            except Exception as exc:
                last_exc = exc
                exc_str = str(exc).lower()
                retryable = any(k in exc_str for k in (
                    "429", "quota", "resource_exhausted",
                    "timeout", "connection", "network", "ssl", "503",
                ))
                if retryable and attempt < max_retries:
                    logger.warning(
                        "Gemini call failed (attempt %d/%d): %s — retrying in %.0fs",
                        attempt, max_retries, exc, delay,
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    break

        logger.error("Gemini call failed after %d retries: %s", max_retries, last_exc)
        return None

    # kept for interface compatibility — routes to REST now
    def _call_langchain(self, prompt: str) -> str:
        result = self._call_gemini_with_retry(prompt)
        return result if result else "Unable to generate a response. Please try again."

    # ── Evidence context builder ───────────────────────────────────────────────

    def build_evidence_context(
        self,
        gap_pair: Dict[str, Any],
        papers_df: pd.DataFrame,
        entities_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, str]:
        """
        For a gap pair (concept_a, concept_b), retrieve:
          - Top 5 most relevant abstracts for concept_a via FAISS
          - Top 5 most relevant abstracts for concept_b via FAISS
          - Connecting evidence from shared neighbors in the gap pair

        Returns dict with keys: evidence_a, evidence_b, connecting_evidence.
        """
        concept_a = gap_pair.get("concept_a", "")
        concept_b = gap_pair.get("concept_b", "") or ""

        def _retrieve_evidence(concept: str, top_k: int = 5) -> str:
            if not concept:
                return "(No concept provided)"

            retrieval_pmids: List[str] = []
            if self.embedder and hasattr(self.embedder, "index") and \
               self.embedder.index is not None and self.embedder.index.ntotal > 0:
                try:
                    results = self.embedder.semantic_search(concept, top_k=top_k)
                    retrieval_pmids = [r["pmid"] for r in results]
                except Exception as exc:
                    logger.warning("FAISS search failed for '%s': %s", concept, exc)

            if not retrieval_pmids and not papers_df.empty:
                matched = papers_df[
                    papers_df["abstract"].fillna("").str.lower().str.contains(
                        concept.lower(), regex=False, na=False
                    )
                ].head(top_k)
                retrieval_pmids = matched["pmid"].tolist()

            if not retrieval_pmids:
                return f"(No papers found mentioning '{concept}')"

            snippets: List[str] = []
            for pmid in retrieval_pmids[:top_k]:
                row = papers_df[papers_df["pmid"] == pmid]
                if row.empty:
                    continue
                row = row.iloc[0]
                abstract = str(row.get("abstract", ""))[:600]
                title = str(row.get("title", "Unknown title"))
                year = row.get("pub_year", "")
                snippets.append(f"[PMID:{pmid}] ({year}) {title}\n{abstract}")

            return "\n\n".join(snippets) if snippets else f"(No abstracts retrieved for '{concept}')"

        evidence_a = _retrieve_evidence(concept_a)
        evidence_b = _retrieve_evidence(concept_b) if concept_b else "(Single-concept gap)"

        shared_neighbors = gap_pair.get("shared_neighbors", [])
        connecting_parts: List[str] = []
        for neighbor in shared_neighbors[:3]:
            nbr_evidence = _retrieve_evidence(str(neighbor), top_k=2)
            connecting_parts.append(f"Via '{neighbor}':\n{nbr_evidence}")
        connecting_evidence = (
            "\n\n".join(connecting_parts)
            if connecting_parts
            else "(No direct connecting evidence found)"
        )

        return {
            "evidence_a": evidence_a,
            "evidence_b": evidence_b,
            "connecting_evidence": connecting_evidence,
        }

    # ── Hypothesis generation ─────────────────────────────────────────────────

    def generate_hypothesis(
        self,
        gap_pair: Dict[str, Any],
        evidence_context: Dict[str, str],
        query_used: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Send structured prompt to Gemini and parse the 6-section response.
        Stores result to database if db_manager is set.
        Returns hypothesis dict or None on failure.
        """
        self._check_ready()

        concept_a = gap_pair.get("concept_a", "")
        concept_b = gap_pair.get("concept_b", "") or ""

        prompt = (
            f"{_SYSTEM_INSTRUCTION}\n\n"
            f"Based on the following evidence from biomedical literature, analyze "
            f"the relationship between [{concept_a}] and [{concept_b}].\n\n"
            f"Evidence for [{concept_a}]:\n{evidence_context.get('evidence_a','')}\n\n"
            f"Evidence for [{concept_b}]:\n{evidence_context.get('evidence_b','')}\n\n"
            f"Connecting evidence:\n{evidence_context.get('connecting_evidence','')}\n\n"
            f"These two concepts have no direct connection in current literature "
            f"but share significant indirect relationships.\n\n"
            f"Please provide your response in exactly this structure:\n\n"
            f"1. HYPOTHESIS: A single clear, testable scientific hypothesis "
            f"connecting these concepts (2-3 sentences)\n"
            f"2. BIOLOGICAL RATIONALE: Why this connection is biologically plausible "
            f"based on the evidence (3-4 sentences)\n"
            f"3. SUPPORTING EVIDENCE: Which specific PMIDs most strongly support "
            f"this hypothesis\n"
            f"4. SUGGESTED EXPERIMENT: One concrete experimental approach to test "
            f"this hypothesis\n"
            f"5. CONFIDENCE: Rate your confidence as Low/Medium/High with one sentence "
            f"justification\n"
            f"6. NOVELTY: Explain what makes this hypothesis novel compared to "
            f"existing work"
        )

        raw_response = self._call_gemini_with_retry(prompt)
        if raw_response is None:
            return None

        parsed = self._parse_structured_response(raw_response)
        confidence_score = _CONFIDENCE_MAP.get(
            parsed.get("confidence_level", "").lower(), 1
        )

        cited_pmids = list(set(re.findall(r"PMID[:\s]*(\d{6,9})", raw_response)))

        hypothesis_obj = {
            "concept_a": concept_a,
            "concept_b": concept_b,
            "hypothesis_text": parsed.get("hypothesis", ""),
            "biological_rationale": parsed.get("biological_rationale", ""),
            "supporting_evidence": parsed.get("supporting_evidence", ""),
            "suggested_experiment": parsed.get("suggested_experiment", ""),
            "confidence_label": parsed.get("confidence_level", "Low"),
            "confidence_score": confidence_score,
            "novelty": parsed.get("novelty", ""),
            "evidence_pmids": cited_pmids,
            "raw_response": raw_response,
            "query_used": query_used,
            "gap_type": gap_pair.get("gap_type", "structural"),
        }

        if self.db:
            try:
                self.db.insert_hypothesis(
                    concept_a=concept_a,
                    concept_b=concept_b,
                    hypothesis_text=parsed.get("hypothesis", ""),
                    evidence_pmids=cited_pmids,
                    confidence_score=confidence_score,
                    raw_response=raw_response,
                    query_used=query_used,
                )
            except Exception as exc:
                logger.error("Failed to persist hypothesis to DB: %s", exc)

        logger.info(
            "Hypothesis generated: %s ↔ %s (confidence=%s)",
            concept_a, concept_b, parsed.get("confidence_level", "?"),
        )
        return hypothesis_obj

    # ── Batch hypothesis generation ───────────────────────────────────────────

    def generate_batch_hypotheses(
        self,
        gap_report: List[Dict[str, Any]],
        papers_df: pd.DataFrame,
        entities_df: Optional[pd.DataFrame] = None,
        query_used: str = "",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process the top HYPOTHESIS_BATCH_SIZE gaps, generating a hypothesis for
        each with a 2-second delay between API calls.
        Returns list of hypothesis dicts ranked by confidence_score descending.
        """
        self._check_ready()

        def _cb(done: int, total: int, msg: str):
            if progress_callback:
                progress_callback(done, total, msg)

        valid_gaps = [
            g for g in gap_report
            if g.get("concept_a") and g.get("concept_b")
        ][:config.HYPOTHESIS_BATCH_SIZE]

        total = len(valid_gaps)
        if total == 0:
            logger.warning("generate_batch_hypotheses: no valid gap pairs")
            return []

        _cb(0, total, f"Generating hypotheses for {total} research gaps...")
        hypotheses: List[Dict[str, Any]] = []

        for i, gap in enumerate(valid_gaps):
            ca, cb = gap.get("concept_a", ""), gap.get("concept_b", "")
            _cb(i, total, f"[{i+1}/{total}] Analysing: {ca} ↔ {cb}...")
            try:
                evidence = self.build_evidence_context(gap, papers_df, entities_df)
                hyp = self.generate_hypothesis(gap, evidence, query_used=query_used)
                if hyp:
                    hypotheses.append(hyp)
            except Exception as exc:
                logger.error("Hypothesis failed for %s ↔ %s: %s", ca, cb, exc)

            if i < total - 1:
                time.sleep(config.HYPOTHESIS_API_DELAY)

        hypotheses.sort(key=lambda h: h.get("confidence_score", 0), reverse=True)
        _cb(total, total, f"Done: {len(hypotheses)} hypotheses generated.")
        logger.info("Batch complete: %d/%d hypotheses", len(hypotheses), total)
        return hypotheses

    # ── Literature chat ───────────────────────────────────────────────────────

    def chat_about_literature(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        papers_df: pd.DataFrame,
        knowledge_graph=None,
    ) -> Tuple[str, List[str]]:
        """
        Answer a user question grounded strictly in the retrieved paper set.
        Returns (response_text, source_pmids).
        """
        self._check_ready()

        source_pmids: List[str] = []
        paper_context_parts: List[str] = []

        # FAISS retrieval
        retrieval_pmids: List[str] = []
        if self.embedder and hasattr(self.embedder, "index") and \
           self.embedder.index is not None and self.embedder.index.ntotal > 0:
            try:
                results = self.embedder.semantic_search(user_message, top_k=5)
                retrieval_pmids = [r["pmid"] for r in results]
            except Exception:
                pass

        # Text-match fallback
        if not retrieval_pmids and not papers_df.empty:
            keywords = user_message.lower().split()[:5]
            pattern = "|".join(re.escape(k) for k in keywords)
            matched = papers_df[
                papers_df["abstract"].fillna("").str.lower().str.contains(
                    pattern, regex=True, na=False
                )
            ].head(5)
            retrieval_pmids = matched["pmid"].tolist()

        for pmid in retrieval_pmids[:5]:
            row = papers_df[papers_df["pmid"] == pmid]
            if row.empty:
                continue
            row = row.iloc[0]
            abstract = str(row.get("abstract", ""))[:700]
            title = str(row.get("title", "Unknown"))
            year = row.get("pub_year", "")
            paper_context_parts.append(f"[PMID:{pmid}] ({year}) {title}\n{abstract}")
            source_pmids.append(pmid)

        paper_context = (
            "\n\n---\n\n".join(paper_context_parts)
            if paper_context_parts
            else "(No relevant papers found for this question in the current dataset.)"
        )

        history_str = ""
        for turn in conversation_history[-6:]:
            role = turn.get("role", "user").upper()
            content = turn.get("content", "")
            history_str += f"{role}: {content}\n"

        prompt = (
            f"{_CHAT_SYSTEM}\n\n"
            f"Retrieved papers:\n{paper_context}\n\n"
            + (f"Conversation history:\n{history_str}\n\n" if history_str else "")
            + f"User question: {user_message}"
        )

        response_text = self._call_gemini_with_retry(prompt) or \
            "Unable to generate a response. Please try again."

        return response_text, source_pmids

    # ── Response parser ───────────────────────────────────────────────────────

    def _parse_structured_response(self, raw: str) -> Dict[str, str]:
        """Parse the 6-section numbered Gemini response into a structured dict."""
        if not raw:
            return {}

        result: Dict[str, str] = {}

        sections = re.split(
            r"\n?\s*\d+\.\s+(?:HYPOTHESIS|BIOLOGICAL RATIONALE|SUPPORTING EVIDENCE|"
            r"SUGGESTED EXPERIMENT|CONFIDENCE|NOVELTY)\s*:?\s*",
            raw,
            flags=re.IGNORECASE,
        )

        key_order = [
            "hypothesis",
            "biological_rationale",
            "supporting_evidence",
            "suggested_experiment",
            "confidence_raw",
            "novelty",
        ]

        for i, key in enumerate(key_order):
            idx = i + 1
            if idx < len(sections):
                result[key] = sections[idx].strip()

        conf_raw = result.get("confidence_raw", "")
        conf_match = re.search(r"\b(low|medium|high)\b", conf_raw, re.IGNORECASE)
        result["confidence_level"] = conf_match.group(1).capitalize() if conf_match else "Low"

        if not result.get("hypothesis"):
            result["hypothesis"] = raw[:1000]
            result["confidence_level"] = "Low"

        return result
