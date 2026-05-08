import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
DATABASE_DIR = DATA_DIR / "database"

# Ensure directories exist at import time
for _d in (RAW_DIR, PROCESSED_DIR, EMBEDDINGS_DIR, DATABASE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = str(DATABASE_DIR / "biolita.db")

# ── Entrez / PubMed ───────────────────────────────────────────────────────────
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")
ENTREZ_API_KEY = os.getenv("ENTREZ_API_KEY")

if not ENTREZ_EMAIL:
    raise EnvironmentError(
        "ENTREZ_EMAIL not found in .env file. "
        "Please add ENTREZ_EMAIL=your_email@example.com to .env."
    )

# Rate limits (requests per second)
RATE_LIMIT_WITHOUT_KEY = 3
RATE_LIMIT_WITH_KEY = 10
FETCH_BATCH_SIZE = 500       # records per efetch call; 500 halves round-trips vs 300
FETCH_RETRIES = 3
FETCH_RETRY_DELAY = 2        # seconds, doubles on each retry

# ── Query / result sizing ─────────────────────────────────────────────────────
MAX_RESULTS_DEFAULT = 1000   # 1000 papers completes in ~60-90s; raise slider for deeper searches
MAX_RESULTS_MIN = 100
MAX_RESULTS_MAX = 3000       # hard cap; >2000 may exceed 2-min target on slow hardware

# ── NLP / Embeddings ──────────────────────────────────────────────────────────
SCISPACY_MODEL = "en_core_sci_lg"

# Primary model: all-MiniLM-L6-v2 — 5x faster than PubMedBERT on CPU,
# 384-dim vectors, excellent quality for scientific text.
# Override via .env: EMBEDDING_MODEL=pritamdeka/S-PubMedBert-MS-MARCO
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
# Dimension must match the model output size.
# MiniLM=384, PubMedBERT/scibert=768. Override via .env if changing model.
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

NLP_BATCH_SIZE = 128         # abstracts per nlp.pipe() call; larger = fewer overhead calls
EMBEDDING_BATCH_SIZE = 256   # MiniLM is small (384-dim); 256 fills CPU cache efficiently

# Set to True to enable UMLS entity linking (downloads ~724 MB index on first run).
# When False, NER still runs but entities keep the generic "ENTITY" label.
USE_UMLS_LINKER = os.getenv("USE_UMLS_LINKER", "false").lower() == "true"

# Entity types extracted by SciSpaCy — covers all biomedical domains.
# "ENTITY" is the generic label used by en_core_sci_lg; the NLP processor
# reclassifies it using UMLS semantic types where available.
NER_ENTITY_TYPES = [
    "DISEASE",
    "GENE_OR_GENOME",
    "CHEMICAL",
    "BIOLOGICAL_PROCESS",
    "CELL",
    "ORGANISM",
    "LABORATORY_PROCEDURE",
    "ENTITY",          # fallback for en_core_sci_lg generic labels
]

# ── Topic Modeling ────────────────────────────────────────────────────────────
BERTOPIC_MIN_TOPIC_SIZE = 10

# ── Network / Graph ───────────────────────────────────────────────────────────
KEYWORD_MIN_FREQUENCY = 3    # minimum papers a keyword must appear in
SEMANTIC_SIMILARITY_THRESHOLD = 0.85
GRAPH_MAX_DISPLAY_NODES = 500    # max nodes sent to browser for interactive rendering

# ── Gap Detection ─────────────────────────────────────────────────────────────
GAP_SHARED_NEIGHBORS_MIN = 3
GAP_TOP_N = 20

# ── Hypothesis Generation ─────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_TEMPERATURE = 0.3
GEMINI_TOP_P = 0.85
GEMINI_TOP_K = 40
GEMINI_MAX_OUTPUT_TOKENS = 2048
HYPOTHESIS_BATCH_SIZE = 10
HYPOTHESIS_API_DELAY = 2     # seconds between Gemini calls

# ── Visualization ─────────────────────────────────────────────────────────────
COMMUNITY_COLORS = [
    "#E8534A",  # red
    "#3B82F6",  # blue
    "#10B981",  # emerald
    "#F59E0B",  # amber
    "#8B5CF6",  # violet
    "#EC4899",  # pink
    "#06B6D4",  # cyan
    "#F97316",  # orange
    "#84CC16",  # lime
    "#14B8A6",  # teal
    "#A855F7",  # purple
    "#22C55E",  # green
    "#EAB308",  # yellow
    "#6366F1",  # indigo
    "#D946EF",  # fuchsia
    "#0EA5E9",  # sky
    "#FB923C",  # orange-400
    "#4ADE80",  # green-400
    "#C084FC",  # purple-400
    "#F472B6",  # pink-400
    "#38BDF8",  # sky-400
    "#FBBF24",  # amber-400
    "#34D399",  # emerald-400
    "#60A5FA",  # blue-400
    "#FB7185",  # rose-400
    "#A3E635",  # lime-400
    "#2DD4BF",  # teal-400
    "#818CF8",  # indigo-400
    "#FCA5A5",  # red-300
    "#93C5FD",  # blue-300
]

ENTITY_TYPE_COLORS = {
    "DISEASE": "#EF4444",
    "GENE_OR_GENOME": "#3B82F6",
    "CHEMICAL": "#10B981",
    "LABORATORY_PROCEDURE": "#F59E0B",
    "CELL": "#8B5CF6",
    "ORGANISM": "#06B6D4",
    "BIOLOGICAL_PROCESS": "#F97316",
    "ENTITY": "#9CA3AF",    # neutral grey for unclassified en_core_sci_lg entities
}

CANVAS_BACKGROUND = "#0A0F1E"
SURFACE_COLOR = "#111827"
SURFACE_ELEVATED = "#1C2539"
PRIMARY_ACCENT = "#3B82F6"
SECONDARY_ACCENT = "#8B5CF6"
SUCCESS_COLOR = "#10B981"
WARNING_COLOR = "#F59E0B"
DANGER_COLOR = "#EF4444"
TEXT_PRIMARY = "#F9FAFB"
TEXT_SECONDARY = "#9CA3AF"
BORDER_COLOR = "#1F2937"

# Node sizing bounds for VOSviewer-style graphs
NODE_SIZE_MIN = 10
NODE_SIZE_MAX = 60

# Edge sizing bounds
EDGE_WIDTH_MIN = 0.5
EDGE_WIDTH_MAX = 8.0

# Physics settings for Barnes-Hut layout
BARNES_HUT_PHYSICS = {
    "barnesHut": {
        "gravitationalConstant": -8000,
        "centralGravity": 0.3,
        "springLength": 150,
        "springConstant": 0.04,
        "damping": 0.09,
        "avoidOverlap": 0.5,
    },
    "minVelocity": 0.75,
    "stabilization": {
        "enabled": True,
        "iterations": 200,
        "updateInterval": 25,
    },
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE = str(BASE_DIR / "bioliteai.log")
