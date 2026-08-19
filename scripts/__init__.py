"""
Package initialization for scripts/ directory.

Exposes clean, importable module aliases:
- ingest               -> scripts/Ingest.py
- chunk                -> scripts/Chunk.py
- embeddings           -> scripts/Embeddings.py
- vector_db            -> scripts/Vector_db.py
- retrieval            -> scripts/Retrieval.py
- grounded_generation  -> scripts/Grounded_Generation.py
- evaluate_experiments -> scripts/evaluate_experiments.py
"""

import sys
import importlib.util
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def _import_script(filename: str, module_name: str):
    script_path = SCRIPTS_DIR / filename
    if not script_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"scripts.{module_name}"] = mod
    spec.loader.exec_module(mod)
    return mod


# Load stage modules
ingest = _import_script("Ingest.py", "ingest")
chunk = _import_script("Chunk.py", "chunk")
embeddings = _import_script("Embeddings.py", "embeddings")
vector_db = _import_script("Vector_db.py", "vector_db")
retrieval = _import_script("Retrieval.py", "retrieval")
grounded_generation = _import_script("Grounded_Generation.py", "grounded_generation")
evaluate_experiments = _import_script("evaluate_experiments.py", "evaluate_experiments")

__all__ = [
    "ingest",
    "chunk",
    "embeddings",
    "vector_db",
    "retrieval",
    "grounded_generation",
    "evaluate_experiments"
]
