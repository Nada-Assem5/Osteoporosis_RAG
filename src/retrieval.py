"""
Compatibility shim: Vector store and retrieval logic live in src.embedded.
Clinical synthesis lives strictly in src.synthesis.
"""

from src.embedded import (
    VectorStore,
    build_vector_index,
    check_scope_guardrail,
    classify_query_risk
)

__all__ = [
    "VectorStore",
    "build_vector_index",
    "check_scope_guardrail",
    "classify_query_risk"
]
