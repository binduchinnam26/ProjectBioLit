"""
DatabaseManager — thread-safe SQLite CRUD layer for BioLitAI-X.
Supports multiple concurrent query sessions.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

from database.schema import SCHEMA_SQL

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Thread-safe SQLite wrapper with connection pooling via a per-thread lock."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = Lock()
        self._initialize_schema()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=100000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def _conn(self):
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_schema(self):
        with self._lock, self._conn() as conn:
            conn.executescript(SCHEMA_SQL)
        logger.info("Database schema initialized at %s", self.db_path)

    # ── Query sessions ────────────────────────────────────────────────────────

    def save_query_session(self, query_text: str, max_results: int) -> int:
        try:
            with self._lock, self._conn() as conn:
                cur = conn.execute(
                    """INSERT INTO query_sessions (query_text, max_results, pipeline_status)
                       VALUES (?, ?, 'pending')""",
                    (query_text, max_results),
                )
                return cur.lastrowid
        except Exception as exc:
            logger.error("save_query_session failed: %s", exc)
            raise

    def update_query_session(
        self,
        session_id: int,
        papers_fetched: Optional[int] = None,
        pipeline_status: Optional[str] = None,
    ):
        try:
            fields, values = [], []
            if papers_fetched is not None:
                fields.append("papers_fetched = ?")
                values.append(papers_fetched)
            if pipeline_status is not None:
                fields.append("pipeline_status = ?")
                values.append(pipeline_status)
            if not fields:
                return
            values.append(session_id)
            with self._lock, self._conn() as conn:
                conn.execute(
                    f"UPDATE query_sessions SET {', '.join(fields)} WHERE id = ?",
                    values,
                )
        except Exception as exc:
            logger.error("update_query_session failed: %s", exc)
            raise

    def get_all_sessions(self) -> List[Dict]:
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM query_sessions ORDER BY created_at DESC"
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("get_all_sessions failed: %s", exc)
            return []

    # ── Papers ────────────────────────────────────────────────────────────────

    def insert_paper(self, paper: Dict[str, Any]):
        try:
            with self._lock, self._conn() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO papers
                       (pmid, title, abstract, journal, volume, issue, pages,
                        pub_date, doi, language, query_used, fetched_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        paper.get("pmid"),
                        paper.get("title"),
                        paper.get("abstract"),
                        paper.get("journal"),
                        paper.get("volume"),
                        paper.get("issue"),
                        paper.get("pages"),
                        paper.get("pub_date"),
                        paper.get("doi"),
                        paper.get("language"),
                        paper.get("query_used"),
                        datetime.utcnow().isoformat(),
                    ),
                )
        except Exception as exc:
            logger.error("insert_paper(%s) failed: %s", paper.get("pmid"), exc)
            raise

    def get_paper(self, pmid: str) -> Optional[Dict]:
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM papers WHERE pmid = ?", (pmid,)
                ).fetchone()
                return dict(row) if row else None
        except Exception as exc:
            logger.error("get_paper(%s) failed: %s", pmid, exc)
            return None

    def get_all_papers(self) -> List[Dict]:
        try:
            with self._conn() as conn:
                rows = conn.execute("SELECT * FROM papers").fetchall()
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("get_all_papers failed: %s", exc)
            return []

    def get_papers_by_query(self, query_used: str) -> List[Dict]:
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM papers WHERE query_used = ?", (query_used,)
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("get_papers_by_query failed: %s", exc)
            return []

    def insert_papers_batch(self, papers_list: List[Dict[str, Any]]):
        """Insert or replace a list of paper dicts in a single transaction."""
        now = datetime.utcnow().isoformat()
        rows = [
            (
                p.get("pmid"), p.get("title"), p.get("abstract"), p.get("journal"),
                p.get("volume"), p.get("issue"), p.get("pages"), p.get("pub_date"),
                p.get("doi"), p.get("language"), p.get("query_used"), now,
            )
            for p in papers_list
        ]
        try:
            with self._lock, self._conn() as conn:
                conn.executemany(
                    """INSERT OR REPLACE INTO papers
                       (pmid, title, abstract, journal, volume, issue, pages,
                        pub_date, doi, language, query_used, fetched_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    rows,
                )
        except Exception as exc:
            logger.error("insert_papers_batch failed: %s", exc)
            raise

    def search_papers(self, search_term: str, limit: int = 50) -> List[Dict]:
        try:
            pattern = f"%{search_term}%"
            with self._conn() as conn:
                rows = conn.execute(
                    """SELECT * FROM papers
                       WHERE title LIKE ? OR abstract LIKE ?
                       LIMIT ?""",
                    (pattern, pattern, limit),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("search_papers failed: %s", exc)
            return []

    # ── Authors ───────────────────────────────────────────────────────────────

    def insert_author(self, name: str, normalized_name: str, affiliation: str) -> int:
        try:
            with self._lock, self._conn() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO authors (name, normalized_name, affiliation)
                       VALUES (?, ?, ?)""",
                    (name, normalized_name, affiliation),
                )
                row = conn.execute(
                    "SELECT id FROM authors WHERE name = ? AND affiliation = ?",
                    (name, affiliation),
                ).fetchone()
                return row["id"]
        except Exception as exc:
            logger.error("insert_author failed: %s", exc)
            raise

    def link_paper_author(self, pmid: str, author_id: int, order: int):
        try:
            with self._lock, self._conn() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO paper_authors (paper_pmid, author_id, author_order)
                       VALUES (?, ?, ?)""",
                    (pmid, author_id, order),
                )
        except Exception as exc:
            logger.error("link_paper_author failed: %s", exc)
            raise

    # ── Keywords ──────────────────────────────────────────────────────────────

    def insert_keyword(
        self, keyword: str, normalized: str, keyword_type: str
    ) -> int:
        try:
            with self._lock, self._conn() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO keywords (keyword, normalized_keyword, keyword_type)
                       VALUES (?, ?, ?)""",
                    (keyword, normalized, keyword_type),
                )
                row = conn.execute(
                    "SELECT id FROM keywords WHERE keyword = ? AND keyword_type = ?",
                    (keyword, keyword_type),
                ).fetchone()
                return row["id"]
        except Exception as exc:
            logger.error("insert_keyword failed: %s", exc)
            raise

    def link_paper_keyword(self, pmid: str, keyword_id: int):
        try:
            with self._lock, self._conn() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO paper_keywords (paper_pmid, keyword_id)
                       VALUES (?, ?)""",
                    (pmid, keyword_id),
                )
        except Exception as exc:
            logger.error("link_paper_keyword failed: %s", exc)
            raise

    # ── MeSH terms ───────────────────────────────────────────────────────────

    def insert_mesh_term(
        self, descriptor: str, qualifier: Optional[str], is_major: bool
    ) -> int:
        try:
            with self._lock, self._conn() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO mesh_terms (descriptor, qualifier, is_major_topic)
                       VALUES (?, ?, ?)""",
                    (descriptor, qualifier, int(is_major)),
                )
                row = conn.execute(
                    """SELECT id FROM mesh_terms
                       WHERE descriptor = ? AND qualifier IS ? AND is_major_topic = ?""",
                    (descriptor, qualifier, int(is_major)),
                ).fetchone()
                return row["id"]
        except Exception as exc:
            logger.error("insert_mesh_term failed: %s", exc)
            raise

    def link_paper_mesh(self, pmid: str, mesh_id: int):
        try:
            with self._lock, self._conn() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO paper_mesh (paper_pmid, mesh_term_id)
                       VALUES (?, ?)""",
                    (pmid, mesh_id),
                )
        except Exception as exc:
            logger.error("link_paper_mesh failed: %s", exc)
            raise

    def get_mesh_by_paper(self, pmid: str) -> List[Dict]:
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    """SELECT m.descriptor, m.qualifier, m.is_major_topic
                       FROM mesh_terms m
                       JOIN paper_mesh pm ON pm.mesh_term_id = m.id
                       WHERE pm.paper_pmid = ?""",
                    (pmid,),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("get_mesh_by_paper failed: %s", exc)
            return []

    # ── Chemical terms ────────────────────────────────────────────────────────

    def insert_chemical_term(self, name: str, registry_number: Optional[str]) -> int:
        try:
            with self._lock, self._conn() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO chemical_terms (name, registry_number)
                       VALUES (?, ?)""",
                    (name, registry_number),
                )
                row = conn.execute(
                    "SELECT id FROM chemical_terms WHERE name = ?", (name,)
                ).fetchone()
                return row["id"]
        except Exception as exc:
            logger.error("insert_chemical_term failed: %s", exc)
            raise

    def link_paper_chemical(self, pmid: str, chemical_id: int):
        try:
            with self._lock, self._conn() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO paper_chemicals (paper_pmid, chemical_term_id)
                       VALUES (?, ?)""",
                    (pmid, chemical_id),
                )
        except Exception as exc:
            logger.error("link_paper_chemical failed: %s", exc)
            raise

    def get_chemicals_by_paper(self, pmid: str) -> List[Dict]:
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    """SELECT c.name, c.registry_number
                       FROM chemical_terms c
                       JOIN paper_chemicals pc ON pc.chemical_term_id = c.id
                       WHERE pc.paper_pmid = ?""",
                    (pmid,),
                ).fetchall()
                return [dict(r) for r in rows]
        except Exception as exc:
            logger.error("get_chemicals_by_paper failed: %s", exc)
            return []

    # ── Publication types ─────────────────────────────────────────────────────

    def insert_publication_type(self, pub_type: str) -> int:
        try:
            with self._lock, self._conn() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO publication_types (publication_type) VALUES (?)",
                    (pub_type,),
                )
                row = conn.execute(
                    "SELECT id FROM publication_types WHERE publication_type = ?",
                    (pub_type,),
                ).fetchone()
                return row["id"]
        except Exception as exc:
            logger.error("insert_publication_type failed: %s", exc)
            raise

    def link_paper_publication_type(self, pmid: str, pub_type_id: int):
        try:
            with self._lock, self._conn() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO paper_publication_types
                       (paper_pmid, publication_type_id) VALUES (?, ?)""",
                    (pmid, pub_type_id),
                )
        except Exception as exc:
            logger.error("link_paper_publication_type failed: %s", exc)
            raise

    def get_publication_types_by_paper(self, pmid: str) -> List[str]:
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    """SELECT pt.publication_type
                       FROM publication_types pt
                       JOIN paper_publication_types ppt ON ppt.publication_type_id = pt.id
                       WHERE ppt.paper_pmid = ?""",
                    (pmid,),
                ).fetchall()
                return [r["publication_type"] for r in rows]
        except Exception as exc:
            logger.error("get_publication_types_by_paper failed: %s", exc)
            return []

    # ── Grant info ────────────────────────────────────────────────────────────

    def insert_grant(
        self,
        pmid: str,
        grant_id: Optional[str],
        acronym: Optional[str],
        agency: Optional[str],
        country: Optional[str],
    ):
        try:
            with self._lock, self._conn() as conn:
                conn.execute(
                    """INSERT INTO grant_info (paper_pmid, grant_id, acronym, agency, country)
                       VALUES (?, ?, ?, ?, ?)""",
                    (pmid, grant_id, acronym, agency, country),
                )
        except Exception as exc:
            logger.error("insert_grant failed: %s", exc)
            raise

    # ── Entities & Relationships ──────────────────────────────────────────────

    def insert_entity(
        self, name: str, entity_type: str, umls_id: Optional[str]
    ) -> int:
        try:
            with self._lock, self._conn() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO entities (name, entity_type, umls_id)
                       VALUES (?, ?, ?)""",
                    (name, entity_type, umls_id),
                )
                row = conn.execute(
                    "SELECT id FROM entities WHERE name = ? AND entity_type = ?",
                    (name, entity_type),
                ).fetchone()
                return row["id"]
        except Exception as exc:
            logger.error("insert_entity failed: %s", exc)
            raise

    def link_paper_entity(self, pmid: str, entity_id: int, sentence_context: str):
        try:
            with self._lock, self._conn() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO paper_entities
                       (paper_pmid, entity_id, sentence_context)
                       VALUES (?, ?, ?)""",
                    (pmid, entity_id, sentence_context),
                )
        except Exception as exc:
            logger.error("link_paper_entity failed: %s", exc)
            raise

    def insert_relationship(
        self,
        source_entity_id: int,
        target_entity_id: int,
        relationship_type: str,
        evidence_pmid: Optional[str],
        confidence_score: float,
    ) -> int:
        try:
            with self._lock, self._conn() as conn:
                cur = conn.execute(
                    """INSERT INTO relationships
                       (source_entity_id, target_entity_id, relationship_type,
                        evidence_pmid, confidence_score)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        source_entity_id,
                        target_entity_id,
                        relationship_type,
                        evidence_pmid,
                        confidence_score,
                    ),
                )
                return cur.lastrowid
        except Exception as exc:
            logger.error("insert_relationship failed: %s", exc)
            raise

    def insert_entities_batch(self, entity_rows: List[Dict]) -> Dict:
        """
        Batch-insert entities and paper-entity links in a single transaction.

        Returns {(entity_text, entity_type): entity_id} for use by
        insert_relationships_batch().
        """
        # Collect unique (name, type) pairs
        unique: Dict[tuple, Optional[str]] = {}
        for row in entity_rows:
            key = (row["entity_text"], row["entity_type"])
            if key not in unique:
                unique[key] = row.get("umls_id")

        entity_id_map: Dict = {}
        try:
            with self._lock, self._conn() as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO entities (name, entity_type, umls_id) VALUES (?, ?, ?)",
                    [(name, etype, umls_id) for (name, etype), umls_id in unique.items()],
                )
                for name, etype in unique:
                    row = conn.execute(
                        "SELECT id FROM entities WHERE name = ? AND entity_type = ?",
                        (name, etype),
                    ).fetchone()
                    if row:
                        entity_id_map[(name, etype)] = row["id"]

                pe_rows = []
                for row in entity_rows:
                    key = (row["entity_text"], row["entity_type"])
                    eid = entity_id_map.get(key)
                    if eid:
                        pe_rows.append((row["pmid"], eid, row.get("sentence_context", "")))
                if pe_rows:
                    conn.executemany(
                        """INSERT OR IGNORE INTO paper_entities
                           (paper_pmid, entity_id, sentence_context) VALUES (?, ?, ?)""",
                        pe_rows,
                    )
        except Exception as exc:
            logger.error("insert_entities_batch failed: %s", exc)
            raise
        return entity_id_map

    def insert_relationships_batch(
        self, rel_rows: List[Dict], entity_id_map: Dict
    ):
        """
        Batch-insert relationships using the entity_id_map returned by
        insert_entities_batch().  Rows where either endpoint cannot be
        resolved are silently skipped (entity not extracted for that paper).
        """
        # Build text-lower → entity_id lookup (first entity type wins)
        text_to_id: Dict[str, int] = {}
        for (name, _etype), eid in entity_id_map.items():
            key = name.lower()
            if key not in text_to_id:
                text_to_id[key] = eid

        rows = []
        for rel in rel_rows:
            src_id = text_to_id.get(rel.get("subject_entity", "").lower())
            tgt_id = text_to_id.get(rel.get("object_entity", "").lower())
            if src_id and tgt_id and src_id != tgt_id:
                rows.append((
                    src_id, tgt_id,
                    rel.get("relationship_verb", ""),
                    rel.get("pmid"),
                    1.0,
                ))

        if not rows:
            return
        try:
            with self._lock, self._conn() as conn:
                conn.executemany(
                    """INSERT INTO relationships
                       (source_entity_id, target_entity_id, relationship_type,
                        evidence_pmid, confidence_score)
                       VALUES (?, ?, ?, ?, ?)""",
                    rows,
                )
        except Exception as exc:
            logger.error("insert_relationships_batch failed: %s", exc)
            raise

    # ── Hypotheses ────────────────────────────────────────────────────────────

    def insert_hypothesis(
        self,
        concept_a: str,
        concept_b: str,
        hypothesis_text: str,
        evidence_pmids: List[str],
        confidence_score: int,
        raw_response: str,
        query_used: str,
    ) -> int:
        try:
            with self._lock, self._conn() as conn:
                cur = conn.execute(
                    """INSERT INTO hypotheses
                       (concept_a, concept_b, hypothesis_text, evidence_pmids,
                        confidence_score, raw_response, query_used)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        concept_a,
                        concept_b,
                        hypothesis_text,
                        json.dumps(evidence_pmids),
                        confidence_score,
                        raw_response,
                        query_used,
                    ),
                )
                return cur.lastrowid
        except Exception as exc:
            logger.error("insert_hypothesis failed: %s", exc)
            raise

    def get_hypotheses_by_query(self, query_used: str) -> List[Dict]:
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM hypotheses WHERE query_used = ? ORDER BY confidence_score DESC",
                    (query_used,),
                ).fetchall()
                result = []
                for r in rows:
                    d = dict(r)
                    d["evidence_pmids"] = json.loads(d.get("evidence_pmids") or "[]")
                    result.append(d)
                return result
        except Exception as exc:
            logger.error("get_hypotheses_by_query failed: %s", exc)
            return []

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_statistics(self, query_used: Optional[str] = None) -> Dict[str, Any]:
        try:
            with self._conn() as conn:
                if query_used:
                    paper_count = conn.execute(
                        "SELECT COUNT(*) FROM papers WHERE query_used = ?",
                        (query_used,),
                    ).fetchone()[0]
                else:
                    paper_count = conn.execute(
                        "SELECT COUNT(*) FROM papers"
                    ).fetchone()[0]

                author_count = conn.execute(
                    "SELECT COUNT(*) FROM authors"
                ).fetchone()[0]
                entity_count = conn.execute(
                    "SELECT COUNT(*) FROM entities"
                ).fetchone()[0]
                relationship_count = conn.execute(
                    "SELECT COUNT(*) FROM relationships"
                ).fetchone()[0]
                hypothesis_count = conn.execute(
                    "SELECT COUNT(*) FROM hypotheses"
                ).fetchone()[0]
                keyword_count = conn.execute(
                    "SELECT COUNT(*) FROM keywords"
                ).fetchone()[0]

                return {
                    "paper_count": paper_count,
                    "author_count": author_count,
                    "entity_count": entity_count,
                    "relationship_count": relationship_count,
                    "hypothesis_count": hypothesis_count,
                    "keyword_count": keyword_count,
                }
        except Exception as exc:
            logger.error("get_statistics failed: %s", exc)
            return {}

    # ── Embeddings meta ───────────────────────────────────────────────────────

    def insert_embedding_meta(self, pmid: str, embedding_path: str, model_used: str):
        try:
            with self._lock, self._conn() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO embeddings_meta
                       (pmid, embedding_path, model_used) VALUES (?, ?, ?)""",
                    (pmid, embedding_path, model_used),
                )
        except Exception as exc:
            logger.error("insert_embedding_meta failed: %s", exc)
            raise
