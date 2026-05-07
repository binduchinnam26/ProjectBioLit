"""
NLPProcessor — SciSpaCy NER, UMLS entity linking, and dependency-based
relationship extraction for any biomedical domain.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

import config
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

# Maps UMLS semantic type IDs → our entity type labels.
# en_core_sci_lg labels all entities as generic "ENTITY"; we use this map
# to reclassify them into specific categories when UMLS info is available.
_UMLS_SEMTYPE_MAP: Dict[str, str] = {
    # Diseases / disorders
    "T047": "DISEASE", "T048": "DISEASE", "T191": "DISEASE",
    "T019": "DISEASE", "T033": "DISEASE", "T037": "DISEASE",
    "T046": "DISEASE", "T049": "DISEASE", "T184": "DISEASE",
    # Genes / genomes
    "T028": "GENE_OR_GENOME", "T045": "GENE_OR_GENOME",
    "T085": "GENE_OR_GENOME", "T087": "GENE_OR_GENOME",
    "T088": "GENE_OR_GENOME", "T086": "GENE_OR_GENOME",
    # Chemicals / drugs
    "T109": "CHEMICAL", "T110": "CHEMICAL", "T114": "CHEMICAL",
    "T116": "CHEMICAL", "T117": "CHEMICAL", "T118": "CHEMICAL",
    "T119": "CHEMICAL", "T120": "CHEMICAL", "T121": "CHEMICAL",
    "T122": "CHEMICAL", "T123": "CHEMICAL", "T125": "CHEMICAL",
    "T126": "CHEMICAL", "T127": "CHEMICAL", "T129": "CHEMICAL",
    "T130": "CHEMICAL", "T131": "CHEMICAL", "T192": "CHEMICAL",
    "T195": "CHEMICAL",
    # Biological processes
    "T043": "BIOLOGICAL_PROCESS", "T044": "BIOLOGICAL_PROCESS",
    "T169": "BIOLOGICAL_PROCESS", "T038": "BIOLOGICAL_PROCESS",
    "T042": "BIOLOGICAL_PROCESS",
    # Cells / cell components
    "T025": "CELL", "T026": "CELL",
    # Organisms
    "T001": "ORGANISM", "T002": "ORGANISM", "T004": "ORGANISM",
    "T005": "ORGANISM", "T007": "ORGANISM", "T008": "ORGANISM",
    "T010": "ORGANISM",
    # Laboratory procedures
    "T059": "LABORATORY_PROCEDURE", "T060": "LABORATORY_PROCEDURE",
    "T061": "LABORATORY_PROCEDURE", "T063": "LABORATORY_PROCEDURE",
}


class NLPProcessor:
    """
    Extracts biomedical entities and verb-based relationships from paper abstracts
    using SciSpaCy. Covers all biomedical domains universally — no topic-specific
    logic anywhere in this class.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.nlp = None
        self.db = db_manager
        self._linker = None    # reference to the scispacy EntityLinker pipe
        self._ready = False

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self):
        """
        Load SciSpaCy model and UMLS entity linker.
        Must be called once before any extraction methods.
        """
        import spacy

        logger.info("Loading SciSpaCy model: %s", config.SCISPACY_MODEL)
        try:
            self.nlp = spacy.load(config.SCISPACY_MODEL)
        except OSError:
            raise OSError(
                f"SciSpaCy model '{config.SCISPACY_MODEL}' not found. "
                f"Install it with:\n"
                f"  pip install https://s3-us-west-2.amazonaws.com/ai2-s3-scispacy/"
                f"releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz"
            )

        logger.info("Adding UMLS entity linker to pipeline")
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
                "UMLS linker could not be loaded (%s). "
                "Entity extraction will proceed without UMLS CUI linking.",
                exc,
            )

        # Store linker reference for semantic type lookup
        try:
            self._linker = self.nlp.get_pipe("scispacy_linker")
        except Exception:
            self._linker = None

        self._ready = True
        logger.info("NLPProcessor ready. Entity types: %s", config.NER_ENTITY_TYPES)

    def _check_ready(self):
        if not self._ready or self.nlp is None:
            raise RuntimeError("NLPProcessor.setup() must be called before extraction.")

    # ── Entity extraction ─────────────────────────────────────────────────────

    def extract_entities(
        self, abstract_text: str
    ) -> List[Tuple[str, str, Optional[str], str]]:
        """
        Run SciSpaCy NER on *abstract_text*.

        Returns list of (entity_text, entity_type, umls_id, sentence_context).
        Only returns entities whose type is in config.NER_ENTITY_TYPES.
        Entities without a UMLS link have umls_id=None.
        """
        self._check_ready()

        if not abstract_text or not abstract_text.strip():
            return []

        results: List[Tuple[str, str, Optional[str], str]] = []
        try:
            doc = self.nlp(abstract_text[:100_000])  # hard cap for safety
        except Exception as exc:
            logger.error("spaCy processing failed: %s", exc)
            return results

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
                        umls_id = ent._.kb_ents[0][0]  # top UMLS CUI

                        # Reclassify generic "ENTITY" label using UMLS semantic types
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

    # ── Relationship extraction ───────────────────────────────────────────────

    def extract_relationships(
        self,
        abstract_text: str,
        entities: List[Tuple[str, str, Optional[str], str]],
    ) -> List[Tuple[str, str, str, str]]:
        """
        Use spaCy dependency parsing to find verb-based relationships between
        entity pairs that appear in the same sentence.

        Returns list of (subject_entity, relationship_verb, object_entity,
        sentence_context).
        """
        self._check_ready()

        if not abstract_text or not entities:
            return []

        entity_texts = {e[0].lower() for e in entities}
        results: List[Tuple[str, str, str, str]] = []

        try:
            doc = self.nlp(abstract_text[:100_000])
        except Exception as exc:
            logger.error("spaCy dependency parse failed: %s", exc)
            return results

        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_entities = [
                tok for tok in sent
                if tok.text.lower() in entity_texts
                or any(tok.text.lower() in e.lower() for e in entity_texts)
            ]

            # Find verbs in the sentence that connect entity pairs
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

                        # At least one side must be a recognized entity
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

    # ── Corpus processing ─────────────────────────────────────────────────────

    def process_corpus(
        self,
        papers_df: pd.DataFrame,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply entity and relationship extraction to all paper abstracts.
        Stores results in the database if db_manager was provided.

        progress_callback(completed, total, message)

        Returns (entities_df, relationships_df).
        Works on any biomedical domain without modification.
        """
        self._check_ready()

        def _cb(done: int, total: int, msg: str):
            if progress_callback:
                progress_callback(done, total, msg)

        if papers_df.empty:
            logger.warning("process_corpus: empty DataFrame, nothing to process")
            return pd.DataFrame(), pd.DataFrame()

        total = len(papers_df)
        all_entities: List[Dict[str, Any]] = []
        all_relationships: List[Dict[str, Any]] = []

        for i, row in papers_df.iterrows():
            pmid = row.get("pmid", "")
            abstract = row.get("abstract", "")

            _cb(i, total, f"NLP processing {i}/{total} papers...")

            if not abstract:
                continue

            try:
                entities = self.extract_entities(abstract)
            except Exception as exc:
                logger.error("Entity extraction failed for PMID %s: %s", pmid, exc)
                entities = []

            try:
                relationships = self.extract_relationships(abstract, entities)
            except Exception as exc:
                logger.error("Relationship extraction failed for PMID %s: %s", pmid, exc)
                relationships = []

            # Collect entity rows
            for ent_text, ent_type, umls_id, sent_ctx in entities:
                all_entities.append(
                    {
                        "pmid": pmid,
                        "entity_text": ent_text,
                        "entity_type": ent_type,
                        "umls_id": umls_id,
                        "sentence_context": sent_ctx,
                    }
                )

            # Collect relationship rows
            for subj, verb, obj, sent_ctx in relationships:
                all_relationships.append(
                    {
                        "pmid": pmid,
                        "subject_entity": subj,
                        "relationship_verb": verb,
                        "object_entity": obj,
                        "sentence_context": sent_ctx,
                    }
                )

            # Persist to database
            if self.db:
                self._persist_paper_results(pmid, entities, relationships)

        _cb(total, total, f"NLP complete: {len(all_entities)} entities, {len(all_relationships)} relationships")
        logger.info(
            "process_corpus done: %d entities, %d relationships across %d papers",
            len(all_entities),
            len(all_relationships),
            total,
        )

        entities_df = pd.DataFrame(all_entities) if all_entities else pd.DataFrame(
            columns=["pmid", "entity_text", "entity_type", "umls_id", "sentence_context"]
        )
        relationships_df = pd.DataFrame(all_relationships) if all_relationships else pd.DataFrame(
            columns=["pmid", "subject_entity", "relationship_verb", "object_entity", "sentence_context"]
        )

        return entities_df, relationships_df

    def _persist_paper_results(
        self,
        pmid: str,
        entities: List[Tuple[str, str, Optional[str], str]],
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
                    # Fallback: search by text prefix
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
