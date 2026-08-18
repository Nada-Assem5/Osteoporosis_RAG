"""
Clinical Practice Guidelines Preprocessing & Retrieval Engine.

Modular Architecture:
- src.parsing    : Step 1 (PDF layout extraction & structural filtering ONLY)
- src.clean      : Step 2 (Text cleaning, noise-title drop, wordfreq repair)
- src.chunking   : Step 3 (Sentence-aware semantic chunking & clinical metadata)
- src.embedded   : Step 4 (Multi-mode vector store, hybrid retrieval & synthesis)
- src.evaluation : Step 5 (Precision@K benchmark & chunk size ablation)
"""

from src.config import (
    DEFAULT_GUIDELINES_DIR,
    DEFAULT_CLEANED_DIR,
    DEFAULT_INDEX_PATH,
    DEFAULT_EVAL_DIR,
    DEFAULT_EVAL_QUESTIONS_PATH,
    CLINICAL_KEYWORDS,
    TOPIC_KEYWORDS,
    POPULATION_PATTERNS,
    GUIDELINE_SOURCE_URLS,
    ConfidenceTier,
    UNSUPPORTED_CLAIM_OVERLAP_THRESHOLD
)
from src.parsing import (
    Page,
    partition_pdf_pages,
    filter_structural_noise,
    discover_and_sync_guidelines
)
from src.clean import (
    clean_pages,
    clean_all_guidelines,
    save_cleaned_text,
    format_summary_table,
    is_noise_title,
    filter_elements,
    strip_punctuation,
    fix_concatenated_word,
    fix_concatenated_text,
    clean_academic_boilerplate,
    count_concatenated_words,
    is_valid_word
)
from src.chunking import (
    Chunk,
    chunk_document,
    extract_clinical_metadata
)
from src.embedded import (
    VectorStore,
    build_vector_index,
    check_scope_guardrail,
    classify_query_risk
)
from src.synthesis import (
    ClinicalSynthesizer,
    ClinicalSynthesisResponse,
    detect_unsupported_claims
)
from src.evaluation import (
    RAGEvaluator,
    EvalQuestion,
    load_eval_questions,
    run_full_evaluation
)

__all__ = [
    # Data Contracts
    "Page",
    "Chunk",
    "ClinicalSynthesisResponse",
    "EvalQuestion",
    # Step 1: Parsing
    "partition_pdf_pages",
    "filter_structural_noise",
    "discover_and_sync_guidelines",
    # Step 2: Cleaning
    "clean_pages",
    "clean_all_guidelines",
    "save_cleaned_text",
    "format_summary_table",
    "is_noise_title",
    "filter_elements",
    "strip_punctuation",
    "fix_concatenated_word",
    "fix_concatenated_text",
    "clean_academic_boilerplate",
    "count_concatenated_words",
    "is_valid_word",
    # Step 3: Chunking
    "chunk_document",
    "extract_clinical_metadata",
    # Step 4: Embedded (Retrieval & Guardrails)
    "VectorStore",
    "build_vector_index",
    "check_scope_guardrail",
    "classify_query_risk",
    # Clinical Synthesis & Guardrails
    "ClinicalSynthesizer",
    "detect_unsupported_claims",
    # Step 5: Evaluation
    "RAGEvaluator",
    "load_eval_questions",
    "run_full_evaluation",
    # Config & Constants
    "DEFAULT_GUIDELINES_DIR",
    "DEFAULT_CLEANED_DIR",
    "DEFAULT_INDEX_PATH",
    "DEFAULT_EVAL_DIR",
    "DEFAULT_EVAL_QUESTIONS_PATH",
    "CLINICAL_KEYWORDS",
    "TOPIC_KEYWORDS",
    "POPULATION_PATTERNS",
    "GUIDELINE_SOURCE_URLS",
    "ConfidenceTier",
    "UNSUPPORTED_CLAIM_OVERLAP_THRESHOLD"
]

__version__ = "2.1.0"
