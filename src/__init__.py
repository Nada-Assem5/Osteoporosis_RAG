"""
Clinical Practice Guidelines Preprocessing & Retrieval Engine.
"""

from src.config import (
    DEFAULT_GUIDELINES_DIR,
    DEFAULT_CLEANED_DIR,
    DEFAULT_INDEX_PATH,
    CLINICAL_KEYWORDS
)
from src.ingestion import (
    Page,
    partition_and_filter_pdf,
    extract_and_clean_pdf,
    clean_all_guidelines,
    discover_and_sync_guidelines,
    save_cleaned_text,
    format_summary_table
)
from src.chunking import Chunk, chunk_document
from src.vector_store import VectorStore

__all__ = [
    "Page",
    "partition_and_filter_pdf",
    "extract_and_clean_pdf",
    "clean_all_guidelines",
    "discover_and_sync_guidelines",
    "save_cleaned_text",
    "format_summary_table",
    "Chunk",
    "chunk_document",
    "VectorStore",
    "DEFAULT_GUIDELINES_DIR",
    "DEFAULT_CLEANED_DIR",
    "DEFAULT_INDEX_PATH",
    "CLINICAL_KEYWORDS"
]

__version__ = "1.0.0"
