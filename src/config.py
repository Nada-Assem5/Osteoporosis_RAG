"""
Centralized Configuration for Clinical Guidelines RAG Pipeline.
Owns paths, thresholds, retrieval constants, clinical keywords, and patterns.
"""

from pathlib import Path

# Directory & Storage Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_GUIDELINES_DIR = DATA_DIR / "guidelines"
DEFAULT_CLEANED_DIR = DATA_DIR / "cleaned"
DEFAULT_VECTOR_STORE_DIR = DATA_DIR / "vector_store"
DEFAULT_INDEX_PATH = DEFAULT_VECTOR_STORE_DIR / "index.json"
DEFAULT_EVAL_DIR = DATA_DIR / "eval_results"
DEFAULT_EVAL_QUESTIONS_PATH = DATA_DIR / "eval_questions.json"

# Ingestion & Cleaning Thresholds
SCANNED_DOC_MIN_CHARS = 50
DEFAULT_PARTITION_STRATEGY = "fast"
SHORT_TITLE_THRESHOLD = 20
MIN_CONTENT_LENGTH = 50

# Chunking Configuration
DEFAULT_CHUNK_SIZE_CHARS = 600
DEFAULT_CHUNK_OVERLAP_CHARS = 100

# Retrieval & Hybrid Search Configuration
DEFAULT_RETRIEVAL_TOP_K = 3
DEFAULT_RETRIEVAL_MODE = "hybrid"  # "keyword", "semantic", or "hybrid"
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
DEFAULT_HYBRID_ALPHA = 0.5

# Clinical Synthesis LLM Configuration (Google Gemini)
DEFAULT_LLM_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
MIN_SYNTHESIS_SCORE_THRESHOLD = 0.015

# Canonical 4-Tier Confidence Standard (Hackathon Agenda)
from enum import Enum

class ConfidenceTier(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INSUFFICIENT_EVIDENCE = "Insufficient Evidence"

# Guideline Source Registry
GUIDELINE_SOURCE_URLS = {
    "osteoporosis-risk-assessment-pdf-66144025463749": "https://www.nice.org.uk/guidance/ng259",
    "osteoporosis-screening-final-recommendation": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/osteoporosis-screening",
    "nice_ng259": "https://www.nice.org.uk/guidance/ng259",
    "uspstf_osteoporosis": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/osteoporosis-screening"
}

# Clinical Safety & Guardrail Workflow Rules (3-Tier Input Classification)
UNSUPPORTED_CLAIM_OVERLAP_THRESHOLD = 0.20

EMERGENCY_KEYWORDS = {
    "chest pain", "shortness of breath", "can't breathe", "cannot breathe",
    "severe bleeding", "heavy bleeding", "suicidal", "suicide", "stroke",
    "unconscious", "loss of consciousness", "loss of bowel", "loss of bladder",
    "cauda equina", "paralysis", "hemorrhage", "overdose", "heart attack",
    "emergency", "acute trauma"
}

NEEDS_CAUTION_PATTERNS = [
    r"\b(?:i\s+have|i'm|i\s+am|my\s+mother|my\s+father|my\s+patient|my\s+t-score|my\s+score|my\s+dxa|my\s+bone|should\s+i\s+take|prescribe\s+me|diagnose\s+me|my\s+age\s+is|\d{1,2}\s+year\s+old\s+(?:male|female|man|woman))\b"
]

PATIENT_SPECIFIC_PATTERNS = NEEDS_CAUTION_PATTERNS

# Clinical Keywords Whitelist for Guardrails
CLINICAL_KEYWORDS = {
    "osteoporosis", "bone", "density", "bmd", "dxa", "vfa", "fracture", "fragility",
    "t-score", "z-score", "frax", "qfracture", "ost", "orai", "glucocorticoid",
    "prednisolone", "anabolic", "antiresorptive", "bisphosphonate", "denosumab",
    "calcium", "vitamin", "fall", "screening", "treatment", "menopause", "postmenopausal",
    "risk", "hip", "vertebral", "spine", "wrist", "score", "hormone", "hrt", "guideline",
    "alendronate", "zoledronic", "raloxifene", "teriparatide", "romosozumab", "calcidiol"
}

# Clinical Target Population Recognition Rules
POPULATION_PATTERNS = {
    "postmenopausal_women": r"(?:postmenopausal\s+women|women\s+aged\s+65|women\s+over\s+65)",
    "older_men": r"(?:men\s+aged\s+70|men\s+over\s+70|older\s+men)",
    "high_risk_adults": r"(?:fragility\s+fracture|previous\s+fracture|glucocorticoid|high\s+risk)",
    "general_adults": r"(?:adults|people\s+aged|patients)"
}

# Clinical Topic Categories
TOPIC_KEYWORDS = {
    "Screening & Diagnosis": ["screening", "dxa", "bmd", "bone density", "t-score", "z-score", "vfa"],
    "Risk Assessment Tools": ["frax", "qfracture", "risk assessment", "risk tool", "prediction tool"],
    "Pharmacological Therapy": ["bisphosphonate", "alendronate", "zoledronic", "denosumab", "teriparatide", "romosozumab", "treatment"],
    "Lifestyle & Supplementation": ["calcium", "vitamin d", "exercise", "fall prevention", "diet", "nutrition"],
    "Secondary Osteoporosis": ["glucocorticoid", "prednisolone", "steroid", "hyperparathyroidism", "malabsorption"]
}

# Academic Boilerplate Noise Patterns
ACADEMIC_NOISE_PATTERNS = [
    r"(?i)author\s+affiliations?:.*?(?=\n\n|\Z)",
    r"(?i)conflict\s+of\s+interest\s+disclosures?:.*?(?=\n\n|\Z)",
    r"(?i)funding\s*/\s*support:.*?(?=\n\n|\Z)",
    r"(?i)article\s+information:.*?(?=\n\n|\Z)",
    r"(?i)corresponding\s+author:.*?(?=\n\n|\Z)",
    r"(?i)downloaded\s+from:.*?(?=\n|\Z)",
    r"(?i)jama\s+published\s+online.*?(?=\n|\Z)",
    r"(?i)copyright\s+20\d\d\s+american\s+medical\s+association.*?(?=\n|\Z)",
    r"(?i)all\s+rights\s+reserved\.?",
    r"(?i)doi:10\.\d{4,9}/[-._;()/:A-Z0-9]+",
    r"(?i)supplemental\s+content\s+at\s+jama\.com"
]
