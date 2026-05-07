"""
NLPProcessor — SciSpaCy NER, UMLS entity linking, and dependency-based
relationship extraction for any biomedical domain.

Uses nlp.pipe() for batch processing (3-8x faster than per-abstract calls).
All text fields are sanitized via _safe_text() before being passed to spaCy
to prevent 'float object has no attribute strip' crashes on NaN abstracts.
"""

import logging
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

_NLP_BATCH_SIZE = 64   # abstracts per nlp.pipe() batch


class NLPProcessor:
    """
    Extracts biomedical entities and verb-based relationships from paper abstracts
    using SciSpaCy. Covers all biomedical domains universally.

    process_corpus() uses nlp.pipe() for ~4x throughput vs per-abstract calls.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.nlp = None
        self.db = db_manager
        self._linker = None
        self._ready = False

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self):
        """Load SciSpaCy model and UMLS entity linker."""
        import spacy

        logger.info("Loading SciSpaCy model: %s", config.SCISPACY_MODEL)
        try:
            self.nlp = spacy.load(config.SCISPACY_MODEL)
        except OSError:
            raise OSError(
                f"SciSpaCy model '{config.SCISPACY_MODEL}' not found. "
                f"Install with: pip install https://s3-us-west-2.amazonaws.com/"
                f"ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
            )

        if config.USE_UMLS_LINKER:
            logger.info("Adding UMLS entity linker (USE_UMLS_LINKER=true)")
            try:
                from scispacy.linking import EntityLinker  # noqa: F401 — registers the factory
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
                    "UMLS linker could not be loaded (%s). Proceeding without UMLS linking.", exc
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
        logger.info("NLPProcessor ready. Pipes: %s", self.nlp.pipe_names)

    def _check_ready(self):
        if not self._ready or self.nlp is None:
            raise RuntimeError("NLPProcessor.setup() must be called before extraction.")

    # ── Doc-level extraction (work on pre-computed spaCy Doc) ─────────────────

    def _extract_from_doc(
        self, doc
    ) -> List[Tuple[str, str, Optional[str], str]]:
        """
        Extract entities from a pre-computed spaCy Doc object.
        Returns list of (entity_text, entity_type, umls_id, sentence_context).
        Called by extract_entities() and process_corpus() via nlp.pipe().
        """
        results: List[Tuple[str, str, Optional[str], str]] = []
        allowed = set(config.NER_ENTITY_TYPES)

        for sent in doc.sents:
            sent_text = sent.text.strip()
            for ent in sent.ents:
                ent_type = ent.label_
                if ent_type not in allowed:
                    continue

                entity_text = ent.text.strip()
                if not entity_text or len(entity_text) < 2:
                    continue

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

                results.append((entity_text, ent_type, umls_id, sent_text))

        return results

    def _extract_relationships_from_doc(
        self,
        doc,
        entities: List[Tuple[str, str, Optional[str], str]],
    ) -> List[Tuple[str, str, str, str]]:
        """
        Extract verb-based relationships from a pre-computed spaCy Doc.
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

                subj_toks = [
                    c for c in token.children
                    if c.dep_ in ("nsubj", "nsubjpass", "csubj")
                ]
                obj_toks = [
                    c for c in token.children
                    if c.dep_ in ("dobj", "pobj", "attr", "iobj", "nmod")
                ]

                for subj in subj_toks:
                    for obj in obj_toks:
                        subj_text = subj.text.strip()
                        obj_text = obj.text.strip()

                        subj_is_ent = any(
                            subj_text.lower() in e.lower() or e.lower() in subj_text.lower()
                            for e in entity_texts
                        )
                        obj_is_ent = any(
                            obj_text.lower() in e.lower() or e.lower() in obj_text.lower()
                            for e in entity_texts
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

    def process_corpus(
        self,
        papers_df: pd.DataFrame,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply entity and relationship extraction to all paper abstracts using
        nlp.pipe() for batch throughput (~4x faster than per-abstract calls).

        Papers with missing abstracts use the title as fallback; papers with
        neither are skipped for NLP but NOT removed from the database.

        Returns (entities_df, relationships_df).
        """
        self._check_ready()

        def _cb(entities: int, rels: int, msg: str):
            if progress_callback:
                progress_callback(entities, rels, msg)

        if papers_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        pmids = papers_df["pmid"].tolist()
        # Abstract with title fallback — ensures every paper gets text for NLP
        texts = [
            _safe_text(row.get("abstract")) or _safe_text(row.get("title")) or ""
            for _, row in papers_df.iterrows()
        ]

        total = len(texts)
        all_entities: List[Dict[str, Any]] = []
        all_relationships: List[Dict[str, Any]] = []

        # Process in batches via nlp.pipe() — much faster than nlp() per doc
        processed = 0
        for batch_start in range(0, total, _NLP_BATCH_SIZE):
            batch_end = min(batch_start + _NLP_BATCH_SIZE, total)
            batch_texts = texts[batch_start:batch_end]
            batch_pmids = pmids[batch_start:batch_end]

            try:
                docs = list(
                    self.nlp.pipe(batch_texts, batch_size=_NLP_BATCH_SIZE, n_process=1)
                )
            except Exception as exc:
                logger.error("nlp.pipe() batch %d failed: %s", batch_start, exc)
                docs = []
                for t in batch_texts:
                    try:
                        docs.append(self.nlp(_safe_text(t)[:100_000]))
                    except Exception:
                        docs.append(self.nlp(""))

            for pmid, doc in zip(batch_pmids, docs):
                if not doc.text.strip():
                    # No text — skip NLP but paper record is already in DB
                    continue

                try:
                    entities = self._extract_from_doc(doc)
                except Exception as exc:
                    logger.error("Entity extraction failed PMID %s: %s", pmid, exc)
                    entities = []

                try:
                    rels = self._extract_relationships_from_doc(doc, entities)
                except Exception as exc:
                    logger.error("Rel extraction failed PMID %s: %s", pmid, exc)
                    rels = []

                for ent_text, ent_type, umls_id, sent_ctx in entities:
                    all_entities.append({
                        "pmid": pmid,
                        "entity_text": ent_text,
                        "entity_type": ent_type,
                        "umls_id": umls_id,
                        "sentence_context": sent_ctx,
                    })

                for subj, verb, obj, sent_ctx in rels:
                    all_relationships.append({
                        "pmid": pmid,
                        "subject_entity": subj,
                        "relationship_verb": verb,
                        "object_entity": obj,
                        "sentence_context": sent_ctx,
                    })

            processed = batch_end
            _cb(
                len(all_entities),
                len(all_relationships),
                f"NLP: {processed}/{total} papers | "
                f"{len(all_entities):,} entities | {len(all_relationships):,} rels",
            )

        # Batch-persist to database (single transaction — much faster)
        entity_id_map: Dict = {}
        if self.db and all_entities:
            try:
                entity_id_map = self.db.insert_entities_batch(all_entities)
            except Exception as exc:
                logger.error("Batch entity insert failed: %s", exc)

        if self.db and all_relationships and entity_id_map:
            try:
                self.db.insert_relationships_batch(all_relationships, entity_id_map)
            except Exception as exc:
                logger.error("Batch relationship insert failed: %s", exc)

        _cb(
            len(all_entities),
            len(all_relationships),
            f"NLP complete: {len(all_entities):,} entities, {len(all_relationships):,} relationships",
        )
        logger.info(
            "process_corpus done: %d entities, %d relationships across %d papers",
            len(all_entities), len(all_relationships), total,
        )

        entities_df = pd.DataFrame(all_entities) if all_entities else pd.DataFrame(
            columns=["pmid", "entity_text", "entity_type", "umls_id", "sentence_context"]
        )
        relationships_df = pd.DataFrame(all_relationships) if all_relationships else pd.DataFrame(
            columns=["pmid", "subject_entity", "relationship_verb", "object_entity", "sentence_context"]
        )

        return entities_df, relationships_df
