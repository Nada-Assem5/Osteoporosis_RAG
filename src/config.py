"""
Centralized Configuration for Clinical Guidelines RAG Pipeline.
"""

from pathlib import Path

# Directory & Storage Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_GUIDELINES_DIR = DATA_DIR / "guidelines"
DEFAULT_CLEANED_DIR = DATA_DIR / "cleaned"
DEFAULT_VECTOR_STORE_DIR = DATA_DIR / "vector_store"
DEFAULT_INDEX_PATH = DEFAULT_VECTOR_STORE_DIR / "index.json"

# Ingestion & Cleaning Thresholds
SCANNED_DOC_MIN_CHARS = 50
DEFAULT_PARTITION_STRATEGY = "fast"

# Chunking Configuration
DEFAULT_CHUNK_SIZE_CHARS = 600
DEFAULT_CHUNK_OVERLAP_CHARS = 100

# Retrieval Configuration
DEFAULT_RETRIEVAL_TOP_K = 3
CLINICAL_KEYWORDS = {
    "osteoporosis", "bone", "density", "bmd", "dxa", "vfa", "fracture", "fragility",
    "t-score", "z-score", "frax", "qfracture", "ost", "orai", "glucocorticoid",
    "prednisolone", "anabolic", "antiresorptive", "bisphosphonate", "denosumab",
    "calcium", "vitamin", "fall", "screening", "treatment", "menopause", "postmenopausal",
    "risk", "hip", "vertebral", "spine", "wrist", "score", "hormone", "hrt", "guideline"
}
