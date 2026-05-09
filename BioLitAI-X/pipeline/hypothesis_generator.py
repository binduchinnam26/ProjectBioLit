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
        self._api_working = False     # True only after successful smoke-test

    @property
    def has_api_key(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY", "").strip())

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self):
        """
        Initialise a requests.Session pointed at the Gemini REST API.
        Calls load_dotenv(override=True) so keys added to .env after server
        start are picked up without needing a server restart.
        SSL verification is disabled for environments with a self-signed proxy.
        When GEMINI_API_KEY is absent or unreachable the generator falls back
        to offline template-mode automatically.
        """
        # Re-read .env on every setup() call — picks up keys added after startup
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except ImportError:
            pass

        # Strip whitespace/CRLF that Windows editors sometimes inject
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            logger.warning(
                "GEMINI_API_KEY not set — HypothesisGenerator running in offline "
                "template mode. Add GEMINI_API_KEY to .env for AI-generated hypotheses."
            )
            self._ready = True
            return

        import requests
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self._api_key = api_key
        self._session = requests.Session()
        self._session.verify = False   # bypass self-signed cert chain
        # Send key both as header and as query-param fallback
        self._session.headers.update({
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        })

        # Smoke-test connectivity with a minimal prompt
        self._api_working = False
        try:
            test = self._call_gemini_with_retry("Reply OK", max_retries=1)
            if test:
                self._api_working = True
                logger.info("Gemini REST API reachable (model=%s)", config.GEMINI_MODEL)
            else:
                logger.warning(
                    "Gemini REST API smoke test returned no text — "
                    "check quota, model name (%s), and network access",
                    config.GEMINI_MODEL,
                )
        except Exception as exc:
            logger.warning("Gemini REST API smoke test failed: %s", exc)

        self._ready = True
        logger.info(
            "HypothesisGenerator ready (api_working=%s, model=%s)",
            self._api_working, config.GEMINI_MODEL,
        )

    def _check_ready(self):
        if not self._ready:
            raise RuntimeError(
                "HypothesisGenerator.setup() must be called before use."
            )
        # In online mode a session is required; offline mode has no session
        if self.has_api_key and self._session is None:
            raise RuntimeError(
                "HypothesisGenerator session not initialised. Call setup() first."
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
        url = (
            f"{_GEMINI_REST_BASE}/models/{config.GEMINI_MODEL}:generateContent"
            f"?key={self._api_key}"
        )
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
        self._last_api_error: str = ""

        for attempt in range(1, max_retries + 1):
            try:
                resp = self._session.post(url, json=payload, timeout=60)
                if resp.status_code == 429:
                    raise RuntimeError(f"429 RESOURCE_EXHAUSTED: {resp.text[:200]}")
                if resp.status_code in (400, 401, 403):
                    # Auth / key errors — no point retrying
                    try:
                        err_msg = resp.json().get("error", {}).get("message", resp.text[:200])
                    except Exception:
                        err_msg = resp.text[:200]
                    self._last_api_error = f"HTTP {resp.status_code}: {err_msg}"
                    logger.error("Gemini API key error: %s", self._last_api_error)
                    return None
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

        # Offline fallback when no API key is configured
        if not self.has_api_key:
            return self._generate_hypothesis_offline(gap_pair, evidence_context, query_used)

        raw_response = self._call_gemini_with_retry(prompt)
        if raw_response is None:
            # API call failed — fall back to template-based offline generation
            logger.warning(
                "Gemini API call failed for %s ↔ %s; using offline fallback",
                concept_a, concept_b,
            )
            return self._generate_hypothesis_offline(gap_pair, evidence_context, query_used)

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

            # Only sleep between API calls; skip delay entirely in offline mode
            if i < total - 1 and self.has_api_key:
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

        # Offline fallback when no API key is configured
        if not self.has_api_key:
            return self._chat_offline(user_message, paper_context, source_pmids)

        response_text = self._call_gemini_with_retry(prompt)
        if response_text is None:
            # API call failed — fall back to keyword-match offline response
            logger.warning("Gemini API call failed for chat query; using offline fallback")
            return self._chat_offline(user_message, paper_context, source_pmids)

        return response_text, source_pmids

    # ── Response parser ───────────────────────────────────────────────────────

    # ── Offline / template-based hypothesis generation ────────────────────────

    def _generate_hypothesis_offline(
        self,
        gap_pair: Dict[str, Any],
        evidence_context: Dict[str, str],
        query_used: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Template-based hypothesis builder that works without any API key.
        Produces a structured hypothesis dict grounded in retrieved paper snippets.
        """
        concept_a = gap_pair.get("concept_a", "")
        concept_b = gap_pair.get("concept_b", "") or ""
        gap_type  = gap_pair.get("gap_type", "structural")

        # Extract PMID citations from evidence text
        all_evidence = (
            evidence_context.get("evidence_a", "")
            + evidence_context.get("evidence_b", "")
            + evidence_context.get("connecting_evidence", "")
        )
        cited_pmids = list(set(re.findall(r"PMID[:\s]*(\d{6,9})", all_evidence)))[:5]

        hyp_text = (
            f"We hypothesize that {concept_a} and {concept_b} share a "
            f"previously undescribed functional relationship. "
            f"Investigating this connection may reveal novel therapeutic or "
            f"mechanistic insights in the context of {query_used or 'biomedical research'}."
        )
        rationale = (
            f"Both {concept_a} and {concept_b} appear in the retrieved literature "
            f"but no direct co-occurrence edge exists between them in the current "
            f"knowledge graph (gap type: {gap_type}). "
            f"The connecting evidence suggests shared biological pathways or co-regulatory "
            f"mechanisms that have not been explicitly tested."
        )
        experiment = (
            f"Design a controlled in vitro assay measuring the effect of modulating "
            f"{concept_a} activity on {concept_b}-associated endpoints, using the "
            f"experimental models described in the supporting papers as reference controls."
        )
        supporting = (
            f"Papers {', '.join(['PMID:' + p for p in cited_pmids]) or 'from the corpus'} "
            f"provide indirect evidence supporting this connection."
        )
        novelty = (
            f"No direct study linking {concept_a} and {concept_b} was identified "
            f"in the current corpus, making this a structurally novel research gap."
        )

        hypothesis_obj = {
            "concept_a":            concept_a,
            "concept_b":            concept_b,
            "hypothesis_text":      hyp_text,
            "biological_rationale": rationale,
            "supporting_evidence":  supporting,
            "suggested_experiment": experiment,
            "confidence_label":     "Medium",
            "confidence_score":     2,
            "novelty":              novelty,
            "evidence_pmids":       cited_pmids,
            "raw_response":         hyp_text,
            "query_used":           query_used,
            "gap_type":             gap_type,
        }

        if self.db:
            try:
                self.db.insert_hypothesis(
                    concept_a=concept_a,
                    concept_b=concept_b,
                    hypothesis_text=hyp_text,
                    evidence_pmids=cited_pmids,
                    confidence_score=2,
                    raw_response=hyp_text,
                    query_used=query_used,
                )
            except Exception as exc:
                logger.warning("Failed to persist offline hypothesis to DB: %s", exc)

        return hypothesis_obj

    def _chat_offline(
        self,
        user_message: str,
        paper_context: str,
        source_pmids: List[str],
    ) -> Tuple[str, List[str]]:
        """
        Keyword-match based chat response when Gemini API is unavailable.
        Extracts relevant sentences and paper metadata from retrieved abstracts.
        """
        keywords = [w for w in user_message.lower().split() if len(w) > 3]

        # Parse the paper_context into (header, body) pairs
        blocks: List[Dict[str, str]] = []
        current: Dict[str, str] = {}
        for line in paper_context.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("[PMID"):
                if current:
                    blocks.append(current)
                current = {"header": line, "text": ""}
            elif current:
                current["text"] += " " + line

        if current:
            blocks.append(current)

        # Score each block by keyword matches in its text
        scored: List[tuple] = []
        for blk in blocks:
            text_lower = blk["text"].lower()
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scored.append((score, blk))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_blocks = scored[:4]

        # Choose the right note based on why we're offline
        if self.has_api_key:
            err_detail = getattr(self, "_last_api_error", "")
            if err_detail:
                api_note = (
                    f"*Note: Gemini API error — **{err_detail}**. "
                    f"Your key may be invalid or expired. "
                    f"Regenerate it at https://aistudio.google.com/app/apikey then "
                    f"update your `.env` file.*"
                )
            else:
                api_note = (
                    "*Note: Your GEMINI_API_KEY is set but the Gemini API is currently "
                    "unreachable (network/quota/model issue). Showing keyword-matched excerpts instead. "
                    "Check your API key at https://aistudio.google.com/app/apikey*"
                )
        else:
            api_note = (
                "*Note: Add `GEMINI_API_KEY=your_key` to your `.env` file for "
                "full AI-powered responses.*"
            )

        if top_blocks:
            bullets: List[str] = []
            for _, blk in top_blocks:
                header = blk["header"]
                sentences = [s.strip() for s in re.split(r"[.!?]", blk["text"]) if len(s.strip()) > 30]
                matching = [s for s in sentences if any(kw in s.lower() for kw in keywords)][:2]
                snippet = ". ".join(matching) + "." if matching else sentences[0] + "." if sentences else ""
                if snippet:
                    bullets.append(f"**{header}**\n{snippet}")

            bullet_text = "\n\n".join(bullets)
            response = (
                f"Based on {len(source_pmids)} retrieved papers, here are the most relevant findings "
                f"for *\"{user_message}\"*:\n\n"
                f"{bullet_text}\n\n"
                f"---\n{api_note}"
            )
        else:
            response = (
                f"The {len(source_pmids)} retrieved papers were searched for terms related to "
                f"*\"{user_message}\"*, but no directly matching excerpts were found in the "
                f"top results. Try rephrasing using specific biomedical terms from the dataset.\n\n"
                f"---\n{api_note}"
            )
        return response, source_pmids

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
