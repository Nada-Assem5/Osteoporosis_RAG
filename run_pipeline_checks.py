"""
Forwarding runner for backward compatibility.
Canonical verification logic lives in scripts/validate_pipeline.py.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.validate_pipeline import run_all_checks

if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)
