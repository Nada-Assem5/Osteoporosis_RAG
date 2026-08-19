"""
Stage 3: Dense Semantic Embedding Generation (scripts/Embeddings.py).

Responsibilities:
- Ingests data/processed/chunks.json
- Generates 384-dimensional dense semantic vectors using sentence-transformers (all-MiniLM-L6-v2)
- Provides a deterministic fallback pseudo-embedding generator for offline/low-resource
  environments, CLEARLY FLAGGED as non-semantic since it will degrade retrieval quality
- Records the embedding method/model used per chunk for downstream auditability
  (System Architecture spec: "each output auditable")
- Validates vector dimensionality consistency and rejects NaN/Inf vectors
- Preserves full metadata schema across every embedded chunk
- Saves data/processed/embeddings.json
"""

import os
import sys
import json
import math
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = ROOT_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed")
CHUNKS_JSON_PATH = PROCESSED_DATA_DIR / "chunks.json"
EMBEDDINGS_JSON_PATH = PROCESSED_DATA_DIR / "embeddings.json"

DEFAULT_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EXPECTED_EMBEDDING_DIM = int(os.getenv("EXPECTED_EMBEDDING_DIM", "384"))

FALLBACK_METHOD_LABEL = "deterministic_fallback_NON_SEMANTIC"
REAL_METHOD_LABEL = "sentence-transformers"


def _deterministic_hash_embedding(text: str, dim: int = 384) -> List[float]:
    """
    Fallback deterministic pseudo-embedding generator using SHA-512 expansions.

    WARNING: This is NOT a semantic embedding. Cosine similarity between these
    vectors carries no meaning relative to text similarity. It exists only so
    the pipeline doesn't crash offline - it should never be used to power
    retrieval that judges/users will evaluate. Every chunk embedded this way
    is tagged with FALLBACK_METHOD_LABEL and surfaced loudly in the summary.
    """
    vector = []
    seed_bytes = text.encode("utf-8")
    for i in range(dim):
        h = hashlib.sha512(seed_bytes + str(i).encode("utf-8")).digest()
        val = (int.from_bytes(h[:4], "big", signed=True) / (2**31 - 1))
        vector.append(round(val, 6))

    # L2 normalize
    norm = sum(x * x for x in vector) ** 0.5
    if norm > 0:
        vector = [round(x / norm, 6) for x in vector]
    return vector


def _validate_vector(vector: List[float], expected_dim: int) -> Tuple[bool, Optional[str]]:
    """Reject vectors with wrong dimensionality or non-finite values (NaN/Inf)."""
    if len(vector) != expected_dim:
        return False, f"dimension mismatch: got {len(vector)}, expected {expected_dim}"
    for x in vector:
        if math.isnan(x) or math.isinf(x):
            return False, "vector contains NaN/Inf values"
    return True, None


def generate_embeddings(
    chunks: List[Dict[str, Any]],
    model_name: str = DEFAULT_MODEL_NAME
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generates dense embeddings for each chunk in the dataset.
    Returns (enriched_chunks, run_stats) where run_stats reports how many
    chunks used the real model vs. the non-semantic fallback, and how many
    vectors failed validation.
    """
    if not chunks:
        return [], {"real_count": 0, "fallback_count": 0, "rejected_count": 0, "used_fallback": False}

    texts = [c.get("text", "") for c in chunks]
    embedded_vectors: List[List[float]] = []
    method_used = REAL_METHOD_LABEL
    used_fallback = False
    actual_dim = EXPECTED_EMBEDDING_DIM

    # Attempt to load sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: '{model_name}'...")
        model = SentenceTransformer(model_name)
        embeddings_raw = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embedded_vectors = [v.tolist() for v in embeddings_raw]
        if embedded_vectors:
            actual_dim = len(embedded_vectors[0])
    except Exception as exc:
        logger.error(
            f"[CRITICAL] Could not initialize sentence-transformers ('{exc}'). "
            f"Falling back to NON-SEMANTIC deterministic embeddings — retrieval quality "
            f"will be effectively random until this is fixed (install sentence-transformers "
            f"and/or its model weights)."
        )
        embedded_vectors = [_deterministic_hash_embedding(t, dim=EXPECTED_EMBEDDING_DIM) for t in texts]
        method_used = FALLBACK_METHOD_LABEL
        used_fallback = True
        actual_dim = EXPECTED_EMBEDDING_DIM

    if actual_dim != EXPECTED_EMBEDDING_DIM:
        logger.warning(
            f"[WARN] Model '{model_name}' produced {actual_dim}-dim vectors, "
            f"but EXPECTED_EMBEDDING_DIM is {EXPECTED_EMBEDDING_DIM}. Downstream Vector_db "
            f"consumers assuming a fixed dimension may break."
        )

    enriched_chunks: List[Dict[str, Any]] = []
    real_count = 0
    fallback_count = 0
    rejected_count = 0

    for chunk, vector in zip(chunks, embedded_vectors):
        c_dict = dict(chunk)

        is_valid, err = _validate_vector(vector, actual_dim)
        if not is_valid:
            logger.error(
                f"[ERROR] Rejecting embedding for chunk_id='{c_dict.get('chunk_id', '?')}': {err}. "
                f"This chunk will be excluded from embeddings.json and thus from retrieval."
            )
            rejected_count += 1
            continue

        c_dict["embedding"] = [round(float(x), 6) for x in vector]
        c_dict["embedding_method"] = method_used
        c_dict["embedding_model"] = model_name if method_used == REAL_METHOD_LABEL else "sha512_hash_fallback"
        c_dict["embedding_dim"] = len(vector)

        # Ensure schema completeness
        c_dict.setdefault("section_title", "Unknown Section")
        c_dict.setdefault("source_url", None)

        enriched_chunks.append(c_dict)

        if method_used == FALLBACK_METHOD_LABEL:
            fallback_count += 1
        else:
            real_count += 1

    run_stats = {
        "real_count": real_count,
        "fallback_count": fallback_count,
        "rejected_count": rejected_count,
        "used_fallback": used_fallback,
        "embedding_dim": actual_dim
    }

    return enriched_chunks, run_stats


def run_embeddings(
    chunks_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    model_name: str = DEFAULT_MODEL_NAME
) -> List[Dict[str, Any]]:
    """Stage 3 execution runner."""
    in_path = Path(chunks_path) if chunks_path else CHUNKS_JSON_PATH
    out_path = Path(output_path) if output_path else EMBEDDINGS_JSON_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        logger.error(f"[ERROR] chunks.json not found at '{in_path}'. Run Stage 2 (Chunk.py) first.")
        return []

    with open(in_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embedded_chunks, run_stats = generate_embeddings(chunks, model_name=model_name)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, indent=2, ensure_ascii=False)

    print("\n" + "_" * 90)
    print("  STAGE 3: DENSE SEMANTIC EMBEDDING SUMMARY")
    print("=" * 90)
    print(f"  Input Chunks                : {len(chunks)}")
    print(f"  Successfully Embedded       : {len(embedded_chunks)}")
    print(f"  Rejected (invalid vectors)  : {run_stats['rejected_count']}")
    print(f"  Embedding Dimension         : {run_stats['embedding_dim']}")
    print(f"  Embedding Model             : {model_name}")
    print(f"  Real Semantic Embeddings    : {run_stats['real_count']}")
    print(f"  Fallback (NON-SEMANTIC)     : {run_stats['fallback_count']}")
    if run_stats["used_fallback"]:
        print("  " + "!" * 86)
        print("  WARNING: NON-SEMANTIC FALLBACK EMBEDDINGS WERE USED FOR THIS RUN.")
        print("  Retrieval quality will be effectively random. Install/fix sentence-transformers")
        print("  and re-run this stage before relying on any Retrieval or Evaluation results.")
        print("  " + "!" * 86)
    print(f"  Saved Artifact Path         : {out_path}")
    print("=" * 90 + "\n")

    return embedded_chunks


run = run_embeddings
main = run_embeddings


if __name__ == "__main__":
    run_embeddings()