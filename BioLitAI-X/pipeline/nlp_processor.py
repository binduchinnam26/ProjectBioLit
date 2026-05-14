"""
NLPProcessor — SciSpaCy NER, UMLS entity linking, and dependency-based
relationship extraction for any biomedical domain.

Performance design
──────────────────
process_corpus() feeds all abstracts through nlp.pipe() in one batched call
with the dep parser DISABLED (_DISABLE_PIPES) for a fast NER-only pass.
This cuts per-doc time 5-10x; relationship extraction is deferred and runs
lazily via extract_relationships_corpus() which re-enables the parser.

All text fields are sanitized via _safe_text() before being passed to spaCy
to prevent 'float has no attribute strip' crashes on NaN abstracts.

Config knobs (config.py / .env):
  NLP_BATCH_SIZE  — abstracts per pipe batch (default 256)
  NLP_N_PROCESS   — worker processes (default 1; keep 1 on Windows)
"""

import gc
import logging
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

import config
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


# ── Safe text helper ──────────────────────────────────────────────────────────

def _safe_text(text: Any) -> str:
    """Return a clean string. Converts NaN/None/non-str to empty string."""
    if text is None:
        return ""
    if not isinstance(text, str):
        return ""
    return text.strip()


# ── UMLS semantic type → entity type mapping ──────────────────────────────────

_UMLS_SEMTYPE_MAP: Dict[str, str] = {
    "T047": "DISEASE", "T048": "DISEASE", "T191": "DISEASE",
    "T019": "DISEASE", "T033": "DISEASE", "T037": "DISEASE",
    "T046": "DISEASE", "T049": "DISEASE", "T184": "DISEASE",
    "T028": "GENE_OR_GENOME", "T045": "GENE_OR_GENOME",
    "T085": "GENE_OR_GENOME", "T087": "GENE_OR_GENOME",
    "T088": "GENE_OR_GENOME", "T086": "GENE_OR_GENOME",
    "T109": "CHEMICAL", "T110": "CHEMICAL", "T114": "CHEMICAL",
    "T116": "CHEMICAL", "T117": "CHEMICAL", "T118": "CHEMICAL",
    "T119": "CHEMICAL", "T120": "CHEMICAL", "T121": "CHEMICAL",
    "T122": "CHEMICAL", "T123": "CHEMICAL", "T125": "CHEMICAL",
    "T126": "CHEMICAL", "T127": "CHEMICAL", "T129": "CHEMICAL",
    "T130": "CHEMICAL", "T131": "CHEMICAL", "T192": "CHEMICAL",
    "T195": "CHEMICAL",
    "T043": "BIOLOGICAL_PROCESS", "T044": "BIOLOGICAL_PROCESS",
    "T169": "BIOLOGICAL_PROCESS", "T038": "BIOLOGICAL_PROCESS",
    "T042": "BIOLOGICAL_PROCESS",
    "T025": "CELL", "T026": "CELL",
    "T001": "ORGANISM", "T002": "ORGANISM", "T004": "ORGANISM",
    "T005": "ORGANISM", "T007": "ORGANISM", "T008": "ORGANISM",
    "T010": "ORGANISM",
    "T059": "LABORATORY_PROCEDURE", "T060": "LABORATORY_PROCEDURE",
    "T061": "LABORATORY_PROCEDURE", "T063": "LABORATORY_PROCEDURE",
}

# Only these labels are meaningful biomedical entities.
# "ENTITY" is the only label en_core_sci_lg emits; specific labels
# (DISEASE, CHEMICAL, etc.) come from UMLS reclassification or other models.
# We keep ENTITY here so the model produces output, but filter its content
# heavily via _BIO_STOPWORDS and minimum length below.
_BIOMEDICAL_ENTITY_TYPES: frozenset = frozenset({
    # Generic label from en_core_sci_lg (reclassified to specific type if UMLS is on)
    "ENTITY",
    # Specific labels from UMLS reclassification or domain-specific SciSpaCy models
    "DISEASE", "GENE_OR_GENOME", "CHEMICAL",
    "BIOLOGICAL_PROCESS", "CELL", "ORGANISM", "LABORATORY_PROCEDURE",
    # Extended labels from UMLS or future models
    "CANCER", "DNA", "RNA", "PROTEIN",
    "CELL_TYPE", "CELL_LINE", "PATHWAY", "ANATOMY",
})

# Generic words that appear as NER spans but carry no biomedical meaning.
# Checked against entity TEXT (not the label) after lowercasing.
_BIO_STOPWORDS: frozenset = frozenset({
    "study", "studies", "data", "results", "result", "level", "levels",
    "finding", "findings", "evidence", "week", "weeks", "day", "days",
    "review", "tissue", "tissues", "induced", "diverse", "production",
    "increased", "decreased", "evaluated", "analyzed", "investigated",
    "potential", "complex", "model", "models", "mechanism", "mechanisms",
    "efficacy", "effect", "effects", "species", "cells", "cell",
    "function", "functions", "skin", "healthy", "unclear", "expressed",
    "expression", "consistent", "background", "objective", "objectives",
    "method", "methods", "conclusion", "conclusions", "introduction",
    "abstract", "purpose", "aims", "aim", "scope", "context",
    "analysis", "analyses", "information", "literature", "research",
    "report", "reports", "case", "cases", "subjects", "subject",
    "patient", "patients", "sample", "samples", "group", "groups",
    "control", "controls", "number", "numbers", "rate", "rates",
    "factor", "factors", "role", "roles", "type", "types",
    "significant", "significantly", "associated", "association",
    "compared", "comparison", "including", "related", "different",
    "differences", "common", "similar", "various", "multiple",
    "large", "small", "high", "low", "dose", "doses",
    "response", "responses", "activity", "activities", "protein",
    "proteins", "gene", "genes", "pathway", "pathways",
    "line", "lines", "organ", "organs", "marker", "markers",
    "target", "targets", "site", "sites", "process", "processes",
    "condition", "conditions", "status", "outcome", "outcomes",
    "endpoint", "endpoints", "concentration", "concentrations",
    "treatment", "treatments", "therapy", "therapies",
})

_NLP_BATCH_SIZE = config.NLP_BATCH_SIZE   # abstracts per nlp.pipe() batch

# Components disabled for the fast NER-only pass.
# parser is the slowest component (dep-parse is O(n) but with high constant);
# disabling it drops per-doc time by 5-10x while keeping full entity extraction.
# Relationship extraction (which needs dep arcs) is performed lazily on demand.
_DISABLE_PIPES = ["tagger", "attribute_ruler", "lemmatizer", "parser"]


class NLPProcessor:
    """
    Extracts biomedical entities and verb-based relationships from paper abstracts
    using SciSpaCy. Covers all biomedical domains universally.

    process_corpus() uses nlp.pipe() with the dep parser disabled for ~4x
    throughput vs per-abstract calls. Relationship extraction runs lazily via
    extract_relationships_corpus().
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.nlp     = None
        self.db      = db_manager
        self._linker = None
        self._ready  = False

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self):
        """Load SciSpaCy model and optionally the UMLS entity linker."""
        import spacy
        # spaCy 3.x emits a FutureWarning about tokenizer set union during model load;
        # it's an upstream issue in spaCy, not actionable from user code.
        warnings.filterwarnings("ignore", category=FutureWarning, module="spacy")
        warnings.filterwarnings("ignore", message=".*Possible set union.*", category=FutureWarning)

        logger.info("Loading SciSpaCy model: %s", config.SCISPACY_MODEL)
        try:
            self.nlp = spacy.load(config.SCISPACY_MODEL)
        except OSError:
            raise OSError(
                f"SciSpaCy model '{config.SCISPACY_MODEL}' not found. "
                f"Install with: pip install https://s3-us-west-2.amazonaws.com/"
                f"ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
            )

        # When the dep parser is disabled for speed, doc.sents raises E030
        # (sentence boundaries unset). The sentencizer provides rule-based
        # sentence splitting so _extract_from_doc() can iterate over sentences.
        if "parser" in _DISABLE_PIPES and "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer", first=True)
            logger.info("Added sentencizer (parser disabled for fast NER pass).")

        if config.USE_UMLS_LINKER:
            logger.info("Adding UMLS entity linker (USE_UMLS_LINKER=true)")
            try:
                from scispacy.linking import EntityLinker  # noqa: F401
                self.nlp.add_pipe(
                    "scispacy_linker",
                    config={
                        "resolve_abbreviations": True,
                        "linker_name": "umls",
                        "threshold": 0.85,
                        "filter_for_definitions": False,
                        "no_definition_threshold": 0.95,
                    },
                )
            except Exception as exc:
                logger.warning(
                    "UMLS linker could not be loaded (%s). "
                    "Entity extraction will proceed without UMLS CUI linking.", exc,
                )
            try:
                self._linker = self.nlp.get_pipe("scispacy_linker")
            except Exception:
                self._linker = None
        else:
            logger.info(
                "UMLS linker skipped (USE_UMLS_LINKER=false). "
                "Set USE_UMLS_LINKER=true in .env to enable entity reclassification."
            )

        self._ready = True
        logger.info(
            "NLPProcessor ready — pipes=%s  batch_size=%d  n_process=%d",
            self.nlp.pipe_names, config.NLP_BATCH_SIZE, config.NLP_N_PROCESS,
        )

    def _check_ready(self):
        if not self._ready or self.nlp is None:
            raise RuntimeError("NLPProcessor.setup() must be called before extraction.")

    # ── Doc-level extraction (operate on an already-processed Doc) ────────────

    def _extract_from_doc(
        self, doc
    ) -> List[Tuple[str, str, Optional[str], str]]:
        """
        Extract entities from a pre-computed spaCy Doc object.
        Returns list of (entity_text, entity_type, umls_id, sentence_context).
        Called by extract_entities() and process_corpus() via nlp.pipe().
        """
        results: List[Tuple[str, str, Optional[str], str]] = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            for ent in sent.ents:
                ent_type = ent.label_

                # Try UMLS reclassification for ENTITY catch-all before type filter
                umls_id: Optional[str] = None
                try:
                    if ent._.kb_ents:
                        umls_id = ent._.kb_ents[0][0]
                        if ent_type == "ENTITY" and self._linker and umls_id:
                            kb_ent = self._linker.kb.cui_to_entity.get(umls_id)
                            if kb_ent and kb_ent.types:
                                for sem_type in kb_ent.types:
                                    mapped = _UMLS_SEMTYPE_MAP.get(sem_type)
                                    if mapped:
                                        ent_type = mapped
                                        break
                except AttributeError:
                    pass

                # Reject types not in biomedical set (incl. unresolved ENTITY)
                if ent_type not in _BIOMEDICAL_ENTITY_TYPES:
                    continue

                entity_text = ent.text.strip()
                # Reject single chars, very short tokens, and generic stopwords
                if len(entity_text) < 4:
                    continue
                if entity_text.lower() in _BIO_STOPWORDS:
                    continue

                results.append((entity_text, ent_type, umls_id, sent_text))

        return results

    def _extract_relationships_from_doc(
        self,
        doc,
        entities: List[Tuple[str, str, Optional[str], str]],
    ) -> List[Tuple[str, str, str, str]]:
        """
        Extract verb-based relationships from a pre-computed spaCy Doc.
        Requires the dep parser to be enabled (not used in the fast NER pass).
        Returns list of (subject_entity, relationship_verb, object_entity,
        sentence_context).
        """
        if not entities:
            return []

        entity_texts = {e[0].lower() for e in entities}
        results: List[Tuple[str, str, str, str]] = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            for token in sent:
                if token.pos_ not in ("VERB", "AUX"):
                    continue

                subj_toks = [c for c in token.children
                             if c.dep_ in ("nsubj", "nsubjpass", "csubj")]
                obj_toks  = [c for c in token.children
                             if c.dep_ in ("dobj", "pobj", "attr", "iobj", "nmod")]

                for subj in subj_toks:
                    for obj in obj_toks:
                        subj_text = subj.text.strip()
                        obj_text  = obj.text.strip()

                        subj_lower = subj_text.lower()
                        obj_lower  = obj_text.lower()

                        subj_is_ent = any(
                            subj_lower in e or e in subj_lower for e in entity_texts
                        )
                        obj_is_ent = any(
                            obj_lower in e or e in obj_lower for e in entity_texts
                        )

                        if not (subj_is_ent and obj_is_ent):
                            continue

                        verb_lemma = token.lemma_.lower()
                        if not verb_lemma or len(verb_lemma) < 2:
                            continue

                        results.append((subj_text, verb_lemma, obj_text, sent_text))

        return results

    # ── Public single-text API (backward compatible) ──────────────────────────

    def extract_entities(
        self, abstract_text: Any
    ) -> List[Tuple[str, str, Optional[str], str]]:
        """Extract entities from a single abstract string. NaN-safe."""
        self._check_ready()
        text = _safe_text(abstract_text)
        if not text:
            return []
        try:
            doc = self.nlp(text[:100_000])
        except Exception as exc:
            logger.error("spaCy processing failed: %s", exc)
            return []
        return self._extract_from_doc(doc)

    def extract_relationships(
        self,
        abstract_text: Any,
        entities: List[Tuple[str, str, Optional[str], str]],
    ) -> List[Tuple[str, str, str, str]]:
        """Extract relationships from a single abstract string. NaN-safe."""
        self._check_ready()
        text = _safe_text(abstract_text)
        if not text or not entities:
            return []
        try:
            doc = self.nlp(text[:100_000])
        except Exception as exc:
            logger.error("spaCy dependency parse failed: %s", exc)
            return []
        return self._extract_relationships_from_doc(doc, entities)

    # ── Corpus processing (batch via nlp.pipe) ────────────────────────────────

    def extract_relationships_corpus(
        self,
        papers_df: pd.DataFrame,
        entities_df: pd.DataFrame,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> pd.DataFrame:
        """
        Lazy relationship extraction — runs the full pipeline WITH the dep parser.
        Call only after process_corpus() has already extracted entities.
        Returns relationships_df.
        """
        self._check_ready()
        if papers_df.empty or entities_df.empty:
            return pd.DataFrame(
                columns=["pmid", "subject_entity", "relationship_verb",
                         "object_entity", "sentence_context"]
            )

        pmids = papers_df["pmid"].tolist()
        texts = [
            _safe_text(row.get("abstract")) or _safe_text(row.get("title")) or ""
            for _, row in papers_df.iterrows()
        ]
        # Build per-pmid entity lookup
        ent_by_pmid: dict = {}
        for _, row in entities_df.iterrows():
            ent_by_pmid.setdefault(row["pmid"], []).append(
                (row["entity_text"], row["entity_type"], row.get("umls_id"), row.get("sentence_context", ""))
            )

        all_rels: List[dict] = []
        t0 = time.monotonic()
        # Run WITH parser (don't disable it here)
        for i, (pmid, doc) in enumerate(
            zip(pmids, self.nlp.pipe(texts, batch_size=_NLP_BATCH_SIZE, n_process=1))
        ):
            entities = ent_by_pmid.get(pmid, [])
            try:
                rels = self._extract_relationships_from_doc(doc, entities)
            except Exception:
                rels = []
            for subj, verb, obj, ctx in rels:
                all_rels.append({
                    "pmid": pmid, "subject_entity": subj,
                    "relationship_verb": verb, "object_entity": obj,
                    "sentence_context": ctx,
                })
            if progress_callback and (i + 1) % 200 == 0:
                progress_callback(len(all_rels), i + 1,
                    f"Relationships: {i + 1}/{len(pmids)} papers | {len(all_rels):,} rels "
                    f"({time.monotonic() - t0:.0f}s)")
            if (i + 1) % 500 == 0:
                gc.collect()

        gc.collect()
        return pd.DataFrame(all_rels) if all_rels else pd.DataFrame(
            columns=["pmid", "subject_entity", "relationship_verb",
                     "object_entity", "sentence_context"]
        )

    def process_corpus(
        self,
        papers_df: pd.DataFrame,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fast entity extraction using nlp.pipe() with dep parser DISABLED.
        The parser is the slowest spaCy component (5-10x overhead); disabling
        it cuts per-doc time dramatically while keeping full NER accuracy.

        Supports NLP_N_PROCESS workers (config.py / .env). On Windows or when
        the UMLS linker is loaded, n_process > 1 may fail; the fallback
        automatically retries with n_process=1.

        Papers with missing abstracts use the title as fallback.
        Returns (entities_df, relationships_df) where relationships_df is always
        empty — call extract_relationships_corpus() lazily for dep-parse rels.
        """
        self._check_ready()

        def _cb(entities: int, rels: int, msg: str):
            if progress_callback:
                progress_callback(entities, rels, msg)

        if papers_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        pmids = papers_df["pmid"].tolist()
        # Abstract with title fallback — vectorized pandas is 10-50x faster than iterrows
        abs_series = papers_df["abstract"].fillna("").str.strip()
        title_series = papers_df.get(
            "title", pd.Series("", index=papers_df.index)
        ).fillna("").str.strip()
        texts = abs_series.where(abs_series != "", title_series).tolist()

        total = len(texts)
        all_entities: List[Dict[str, Any]] = []

        # Disable parser + other unused components for NER-only pass.
        disable = [p for p in _DISABLE_PIPES if p in self.nlp.pipe_names]

        _cb(0, total,
            f"Starting NLP pipeline on {total} abstracts "
            f"(batch={config.NLP_BATCH_SIZE}, workers={config.NLP_N_PROCESS})…")

        t0 = time.monotonic()

        def _run_pipe(n_proc: int):
            """Run nlp.pipe() and collect entities; returns list of entity dicts."""
            collected: List[Dict[str, Any]] = []
            pipe_iter = self.nlp.pipe(
                (t[:100_000] for t in texts),
                batch_size=_NLP_BATCH_SIZE,
                n_process=n_proc,
                disable=disable,
            )
            for i, (pmid, doc) in enumerate(zip(pmids, pipe_iter)):
                if not doc.text.strip():
                    continue
                try:
                    entities = self._extract_from_doc(doc)
                except Exception as exc:
                    logger.error("Entity extraction failed PMID %s: %s", pmid, exc)
                    entities = []
                for ent_text, ent_type, umls_id, sent_ctx in entities:
                    collected.append({
                        "pmid": pmid,
                        "entity_text": ent_text,
                        "entity_type": ent_type,
                        "umls_id": umls_id,
                        "sentence_context": sent_ctx,
                    })
                if (i + 1) % 100 == 0 or (i + 1) == total:
                    elapsed = time.monotonic() - t0
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    _cb(
                        len(collected), 0,
                        f"NLP: {i + 1}/{total} papers | "
                        f"{len(collected):,} entities | "
                        f"{rate:.0f} papers/s ({elapsed:.0f}s elapsed)",
                    )
                if (i + 1) % 500 == 0:
                    gc.collect()
            return collected

        try:
            all_entities = _run_pipe(config.NLP_N_PROCESS)
        except Exception as exc:
            # n_process > 1 fails on Windows / with UMLS linker (not picklable).
            if config.NLP_N_PROCESS != 1:
                logger.warning(
                    "nlp.pipe() with n_process=%d failed (%s). Retrying with n_process=1.",
                    config.NLP_N_PROCESS, exc,
                )
                try:
                    all_entities = _run_pipe(1)
                except Exception as exc2:
                    logger.error("nlp.pipe() fallback also failed: %s — using single-doc mode", exc2)
                    for pmid, text in zip(pmids, texts):
                        try:
                            with self.nlp.select_pipes(disable=disable):
                                doc = self.nlp(_safe_text(text)[:100_000])
                            for ent_text, ent_type, umls_id, sent_ctx in self._extract_from_doc(doc):
                                all_entities.append({
                                    "pmid": pmid, "entity_text": ent_text,
                                    "entity_type": ent_type, "umls_id": umls_id,
                                    "sentence_context": sent_ctx,
                                })
                        except Exception:
                            pass
            else:
                logger.error("nlp.pipe() failed: %s — using single-doc fallback", exc)
                for pmid, text in zip(pmids, texts):
                    try:
                        with self.nlp.select_pipes(disable=disable):
                            doc = self.nlp(_safe_text(text)[:100_000])
                        for ent_text, ent_type, umls_id, sent_ctx in self._extract_from_doc(doc):
                            all_entities.append({
                                "pmid": pmid, "entity_text": ent_text,
                                "entity_type": ent_type, "umls_id": umls_id,
                                "sentence_context": sent_ctx,
                            })
                    except Exception:
                        pass

        gc.collect()

        _cb(
            len(all_entities), 0,
            f"NLP complete: {len(all_entities):,} entities extracted "
            f"(relationships extracted lazily on demand)",
        )
        logger.info(
            "process_corpus done: %d entities across %d papers (dep-parse deferred)",
            len(all_entities), total,
        )

        entities_df = (
            pd.DataFrame(all_entities) if all_entities
            else pd.DataFrame(columns=[
                "pmid", "entity_text", "entity_type", "umls_id", "sentence_context"
            ])
        )
        # Relationships are always empty here — use extract_relationships_corpus() lazily
        relationships_df = pd.DataFrame(
            columns=["pmid", "subject_entity", "relationship_verb",
                     "object_entity", "sentence_context"]
        )
        return entities_df, relationships_df

    # ── Database persistence ──────────────────────────────────────────────────

    def _persist_paper_results(
        self,
        pmid: str,
        entities:      List[Tuple[str, str, Optional[str], str]],
        relationships: List[Tuple[str, str, str, str]],
    ):
        """Write entities and relationships for one paper to the database."""
        entity_id_cache: Dict[Tuple[str, str], int] = {}

        for ent_text, ent_type, umls_id, sent_ctx in entities:
            try:
                key = (ent_text, ent_type)
                if key not in entity_id_cache:
                    eid = self.db.insert_entity(ent_text, ent_type, umls_id)
                    entity_id_cache[key] = eid
                self.db.link_paper_entity(pmid, entity_id_cache[key], sent_ctx)
            except Exception as exc:
                logger.error("DB persist entity failed (PMID %s): %s", pmid, exc)

        for subj, verb, obj, sent_ctx in relationships:
            try:
                src_id = entity_id_cache.get((subj, None))
                tgt_id = entity_id_cache.get((obj, None))
                if src_id is None:
                    for (txt, _), eid in entity_id_cache.items():
                        if txt.lower() == subj.lower():
                            src_id = eid
                            break
                if tgt_id is None:
                    for (txt, _), eid in entity_id_cache.items():
                        if txt.lower() == obj.lower():
                            tgt_id = eid
                            break
                if src_id and tgt_id:
                    self.db.insert_relationship(
                        src_id, tgt_id, verb, pmid, confidence_score=0.5
                    )
            except Exception as exc:
                logger.error("DB persist relationship failed (PMID %s): %s", pmid, exc)
