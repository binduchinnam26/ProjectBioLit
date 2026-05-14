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

# ── Hugging Face ─────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")  # read-only token is sufficient


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
MAX_RESULTS_DEFAULT = 500
MAX_RESULTS_MIN = 100
MAX_RESULTS_MAX = 500

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

NLP_BATCH_SIZE = 256         # abstracts per nlp.pipe() call; larger batches amortize tok2vec overhead
EMBEDDING_BATCH_SIZE = 256   # MiniLM is small (384-dim); 256 fills CPU cache efficiently

# Set to True to enable UMLS entity linking (downloads ~724 MB index on first run).
# When False, NER still runs but entities keep the generic "ENTITY" label.
USE_UMLS_LINKER = os.getenv("USE_UMLS_LINKER", "false").lower() == "true"

# Number of abstracts passed to nlp.pipe() in one batch.
# 256 is the sweet spot: large enough to amortise Python↔C overhead,
# small enough to keep memory predictable.
NLP_BATCH_SIZE = int(os.getenv("NLP_BATCH_SIZE", "256"))

# Worker processes for nlp.pipe().
# KEEP AT 1 ON WINDOWS — spaCy uses 'spawn' on Windows, which requires
# pickling the model; the UMLS linker is not picklable and will raise errors.
# On Linux / macOS (fork semantics) default is 2 workers for ~2x NER speed.
# Override via NLP_N_PROCESS env var. The NLPProcessor falls back to
# n_process=1 automatically if multiprocessing fails (e.g. UMLS enabled).
import platform as _platform
NLP_N_PROCESS = int(os.getenv(
    "NLP_N_PROCESS",
    "1" if _platform.system() == "Windows" else "2",
))

# Entity types extracted by SciSpaCy — covers all biomedical domains.
# ENTITY catch-all is excluded; the NLP processor reclassifies it via UMLS
# or drops it if no specific type can be assigned.
NER_ENTITY_TYPES = [
    "DISEASE",
    "CANCER",
    "GENE_OR_GENOME",
    "DNA",
    "RNA",
    "PROTEIN",
    "CHEMICAL",
    "BIOLOGICAL_PROCESS",
    "PATHWAY",
    "CELL",
    "CELL_TYPE",
    "CELL_LINE",
    "ORGANISM",
    "ANATOMY",
    "LABORATORY_PROCEDURE",
]

# ── Topic Modeling ────────────────────────────────────────────────────────────
BERTOPIC_MIN_TOPIC_SIZE = 5

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
HYPOTHESIS_API_DELAY = 5     # seconds between Gemini calls; free tier = 15 RPM → need ≥4s

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
    # Disease group — red
    "DISEASE":              "#FF5252",
    "CANCER":               "#FF5252",
    # Gene / molecular biology group — green
    "GENE_OR_GENOME":       "#00E676",
    "DNA":                  "#00E676",
    "RNA":                  "#00E676",
    "PROTEIN":              "#00E676",
    # Chemical / drug group — cyan
    "CHEMICAL":             "#00D4FF",
    # Biological process / pathway group — mint
    "BIOLOGICAL_PROCESS":   "#A8E6CF",
    "PATHWAY":              "#A8E6CF",
    # Cell group — purple
    "CELL":                 "#7B61FF",
    "CELL_TYPE":            "#7B61FF",
    "CELL_LINE":            "#7B61FF",
    # Organism group — yellow
    "ORGANISM":             "#FFD600",
    # Anatomy group — pink
    "ANATOMY":              "#FF8B94",
    # Procedure / other — grey
    "LABORATORY_PROCEDURE": "#8899AA",
    "ENTITY":               "#8899AA",
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
