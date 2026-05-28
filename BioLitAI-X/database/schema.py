"""
SQLite schema definitions for BioLitAI-X.
All tables are created here via CREATE TABLE IF NOT EXISTS statements.
"""

SCHEMA_SQL = """
PRAGMA foreign_keys=ON;

-- ── Core paper record ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS papers (
    pmid                TEXT PRIMARY KEY,
    title               TEXT,
    abstract            TEXT,
    journal             TEXT,
    volume              TEXT,
    issue               TEXT,
    pages               TEXT,
    pub_date            TEXT,
    doi                 TEXT,
    language            TEXT,
    query_used          TEXT,
    fetched_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── Authors ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS authors (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL,
    normalized_name     TEXT,
    affiliation         TEXT,
    UNIQUE(name, affiliation)
);

CREATE TABLE IF NOT EXISTS paper_authors (
    paper_pmid          TEXT NOT NULL REFERENCES papers(pmid) ON DELETE CASCADE,
    author_id           INTEGER NOT NULL REFERENCES authors(id) ON DELETE CASCADE,
    author_order        INTEGER,
    PRIMARY KEY (paper_pmid, author_id)
);

-- ── Author-assigned keywords ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS keywords (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    keyword             TEXT NOT NULL,
    normalized_keyword  TEXT,
    keyword_type        TEXT NOT NULL DEFAULT 'author',  -- 'author' | 'mesh_descriptor' | 'mesh_qualifier' | 'chemical' | 'publication_type'
    UNIQUE(keyword, keyword_type)
);

CREATE TABLE IF NOT EXISTS paper_keywords (
    paper_pmid          TEXT NOT NULL REFERENCES papers(pmid) ON DELETE CASCADE,
    keyword_id          INTEGER NOT NULL REFERENCES keywords(id) ON DELETE CASCADE,
    PRIMARY KEY (paper_pmid, keyword_id)
);

-- ── MeSH descriptor terms ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS mesh_terms (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    descriptor          TEXT NOT NULL,
    qualifier           TEXT,
    is_major_topic      INTEGER NOT NULL DEFAULT 0,  -- 1 = major, 0 = minor
    UNIQUE(descriptor, qualifier, is_major_topic)
);

CREATE TABLE IF NOT EXISTS paper_mesh (
    paper_pmid          TEXT NOT NULL REFERENCES papers(pmid) ON DELETE CASCADE,
    mesh_term_id        INTEGER NOT NULL REFERENCES mesh_terms(id) ON DELETE CASCADE,
    PRIMARY KEY (paper_pmid, mesh_term_id)
);

-- ── Chemical / substance terms ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chemical_terms (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL,
    registry_number     TEXT,
    UNIQUE(name)
);

CREATE TABLE IF NOT EXISTS paper_chemicals (
    paper_pmid          TEXT NOT NULL REFERENCES papers(pmid) ON DELETE CASCADE,
    chemical_term_id    INTEGER NOT NULL REFERENCES chemical_terms(id) ON DELETE CASCADE,
    PRIMARY KEY (paper_pmid, chemical_term_id)
);

-- ── Publication types ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS publication_types (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    publication_type    TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS paper_publication_types (
    paper_pmid          TEXT NOT NULL REFERENCES papers(pmid) ON DELETE CASCADE,
    publication_type_id INTEGER NOT NULL REFERENCES publication_types(id) ON DELETE CASCADE,
    PRIMARY KEY (paper_pmid, publication_type_id)
);

-- ── Grant / funding information ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS grant_info (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_pmid          TEXT NOT NULL REFERENCES papers(pmid) ON DELETE CASCADE,
    grant_id            TEXT,
    acronym             TEXT,
    agency              TEXT,
    country             TEXT
);

-- ── Topics (BERTopic output) ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS topics (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_label         TEXT NOT NULL,
    top_words           TEXT,           -- JSON array of top keywords
    paper_count         INTEGER DEFAULT 0,
    query_used          TEXT
);

CREATE TABLE IF NOT EXISTS paper_topics (
    paper_pmid          TEXT NOT NULL REFERENCES papers(pmid) ON DELETE CASCADE,
    topic_id            INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    probability         REAL,
    PRIMARY KEY (paper_pmid, topic_id)
);

-- ── NLP entities (SciSpaCy NER) ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS entities (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL,
    entity_type         TEXT NOT NULL,
    umls_id             TEXT,
    UNIQUE(name, entity_type)
);

CREATE TABLE IF NOT EXISTS paper_entities (
    paper_pmid          TEXT NOT NULL REFERENCES papers(pmid) ON DELETE CASCADE,
    entity_id           INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    sentence_context    TEXT,
    PRIMARY KEY (paper_pmid, entity_id)
);

-- ── Relationships between entities ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS relationships (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source_entity_id    INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_entity_id    INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relationship_type   TEXT NOT NULL,
    evidence_pmid       TEXT REFERENCES papers(pmid) ON DELETE SET NULL,
    confidence_score    REAL DEFAULT 0.0
);

-- ── AI-generated hypotheses ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS hypotheses (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_a           TEXT NOT NULL,
    concept_b           TEXT NOT NULL,
    hypothesis_text     TEXT,
    evidence_pmids      TEXT,           -- JSON array of PMIDs
    confidence_score    INTEGER DEFAULT 0,  -- 1=Low 2=Medium 3=High
    raw_response        TEXT,           -- full Gemini response for audit
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    query_used          TEXT
);

-- ── Embedding metadata ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS embeddings_meta (
    pmid                TEXT PRIMARY KEY REFERENCES papers(pmid) ON DELETE CASCADE,
    embedding_path      TEXT,
    model_used          TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── Query sessions ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS query_sessions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text          TEXT NOT NULL,
    max_results         INTEGER,
    papers_fetched      INTEGER DEFAULT 0,
    pipeline_status     TEXT DEFAULT 'pending',  -- pending|running|complete|error
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── Indices for common look-ups ───────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_papers_query       ON papers(query_used);
CREATE INDEX IF NOT EXISTS idx_paper_authors_pmid ON paper_authors(paper_pmid);
CREATE INDEX IF NOT EXISTS idx_paper_mesh_pmid    ON paper_mesh(paper_pmid);
CREATE INDEX IF NOT EXISTS idx_paper_chem_pmid    ON paper_chemicals(paper_pmid);
CREATE INDEX IF NOT EXISTS idx_paper_kw_pmid      ON paper_keywords(paper_pmid);
CREATE INDEX IF NOT EXISTS idx_paper_topics_pmid  ON paper_topics(paper_pmid);
CREATE INDEX IF NOT EXISTS idx_entities_type      ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_relationships_src  ON relationships(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_tgt  ON relationships(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_hypotheses_query   ON hypotheses(query_used);
CREATE INDEX IF NOT EXISTS idx_sessions_created   ON query_sessions(created_at DESC);
"""
