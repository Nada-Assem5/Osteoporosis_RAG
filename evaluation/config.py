"""
Evaluation System Configuration (evaluation/config.py).

Central configuration file to customize grid search hyperparameters,
caching directories, and evaluation datasets.
"""

import os
from pathlib import Path

# Base Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EVALUATION_DIR = ROOT_DIR / "evaluation"
RESULTS_DIR = EVALUATION_DIR
PLOTS_DIR = EVALUATION_DIR / "plots"
CACHE_DIR = EVALUATION_DIR / "cache"

# Input Data Files
ELEMENTS_JSON_PATH = PROCESSED_DATA_DIR / "elements.json"
CHUNKS_BASELINE_PATH = PROCESSED_DATA_DIR / "chunks.json"
EVAL_QUESTIONS_PATH = DATA_DIR / "eval_questions.json"
if not EVAL_QUESTIONS_PATH.exists():
    EVAL_QUESTIONS_PATH = ROOT_DIR / "scripts/data/eval_questions.json"

# =====================================================================
# Hyperparameter Grid Definition
# =====================================================================

# 1. Chunk sizes in tokens (tiktoken cl100k_base)
CHUNK_SIZES = [128, 256, 400, 512]

# 2. Chunk overlaps in tokens
CHUNK_OVERLAPS = [0, 20, 50, 100]

# 3. Retrieval / Search modes to evaluate
SEARCH_TYPES = [
    "keyword",   # Okapi BM25 inverted index
    "semantic",  # Dense vector cosine similarity
    "hybrid"     # Reciprocal Rank Fusion (RRF)
]

# 4. Dense embedding models
EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",          # Lightweight sentence-transformers standard
    "BAAI/bge-small-en-v1.5"     # High-accuracy small general embedding
]

# 5. Hybrid search alpha weights (0.0 = keyword, 1.0 = semantic)
HYBRID_ALPHAS = [0.3, 0.5, 0.7]

# 6. Top-K retrieval depths to evaluate
TOP_K_VALUES = [1, 3, 5, 10]

# =====================================================================
# Quick Evaluation Subset (for rapid debugging / smoke tests)
# =====================================================================
QUICK_CHUNK_SIZES = [256, 400]
QUICK_CHUNK_OVERLAPS = [20, 50]
QUICK_SEARCH_TYPES = ["keyword", "semantic", "hybrid"]
QUICK_EMBEDDING_MODELS = ["all-MiniLM-L6-v2"]
QUICK_HYBRID_ALPHAS = [0.5]
QUICK_TOP_K_VALUES = [1, 3, 5, 10]
