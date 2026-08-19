"""
Package initialization for scripts/ directory.

Exposes clean, importable module aliases via STANDARD package imports (no
manual importlib exec). Because this directory already has this __init__.py,
it IS a real Python package - submodules are cached exactly once in
sys.modules under their real dotted name ("scripts.Chunk", "scripts.Vector_db",
"scripts.Retrieval", etc). Every sibling script file cross-imports the others
using that exact capitalized form, e.g.:

    from scripts.Chunk import chunk_extracted_elements
    from scripts.Vector_db import VectorStore, _tokenize, _cosine_similarity
    from scripts.Retrieval import classify_query_risk, load_eval_questions

so the aliases below MUST resolve to those same real module names.

FIX: the previous version of this file used importlib.util to manually
read+exec each script file a second time under a separate LOWERCASE module
name (sys.modules["scripts.ingest"] instead of sys.modules["scripts.Ingest"]).
That meant every script got imported and executed TWICE - once here under
the lowercase alias, and once again under its real capitalized name whenever
another script did `from scripts.Chunk import ...` (which none of them could
find under the lowercase alias). Besides doubling import-time cost, this
produced two independent copies of the same classes (e.g. two distinct
`VectorStore` classes, two distinct `Chunk` handling functions), which can
silently break isinstance checks, caches, and shared state across modules -
including run_pipeline_checks.py, which imports every module by its real
capitalized name and would end up testing yet a third copy of each.

Plain package-relative imports avoid all of that: each file is loaded
exactly once, and the aliases below (ingest, chunk, embeddings, ...) simply
point at those single canonical module objects.

Exposes:
- ingest               -> scripts/Ingest.py
- chunk                -> scripts/Chunk.py
- embeddings           -> scripts/Embeddings.py
- vector_db            -> scripts/Vector_db.py
- retrieval            -> scripts/Retrieval.py
- grounded_generation  -> scripts/Grounded_Generation.py
- evaluate_experiments -> scripts/evaluate_experiments.py
"""

import logging

logger = logging.getLogger(__name__)


def _safe_import(module_path: str):
    """
    Import a scripts submodule by its real (capitalized) dotted path.
    Returns None with a logged warning instead of raising, so `import scripts`
    doesn't hard-crash the whole pipeline just because one stage's file is
    still missing a dependency (e.g. sentence-transformers, chromadb) during
    early setup.
    """
    try:
        return __import__(module_path, fromlist=["_"])
    except Exception as exc:
        logger.warning(f"[scripts/__init__] Could not import '{module_path}': {exc}")
        return None


ingest = _safe_import("scripts.Ingest")
chunk = _safe_import("scripts.Chunk")
embeddings = _safe_import("scripts.Embeddings")
vector_db = _safe_import("scripts.Vector_db")
retrieval = _safe_import("scripts.Retrieval")
grounded_generation = _safe_import("scripts.Grounded_Generation")
evaluate_experiments = _safe_import("scripts.evaluate_experiments")

__all__ = [
    "ingest",
    "chunk",
    "embeddings",
    "vector_db",
    "retrieval",
    "grounded_generation",
    "evaluate_experiments",
]