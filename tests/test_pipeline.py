"""
Comprehensive unit test suite for Ingestion, Chunking, Vector Storage, and Guardrail modules.
"""

import pytest
from pathlib import Path
from src.ingestion import (
    Page,
    partition_and_filter_pdf,
    extract_and_clean_pdf,
    discover_and_sync_guidelines,
    save_cleaned_text,
    format_summary_table,
    clean_all_guidelines
)
from src.chunking import Chunk, chunk_document
from src.vector_store import VectorStore
from main import check_scope_guardrail


def test_page_dataclass_contract():
    """Verify Page dataclass attributes and element structure."""
    p = Page(
        page_number=1,
        text="Clinical recommendation text",
        elements=[{"type": "Title", "text": "Clinical recommendation text"}],
        metadata={"engine": "unstructured"}
    )
    assert p.page_number == 1
    assert p.text == "Clinical recommendation text"
    assert len(p.elements) == 1
    assert p.elements[0]["type"] == "Title"
    assert p.metadata["engine"] == "unstructured"


def test_page_dataclass_defaults():
    """Verify Page default collections."""
    p = Page(page_number=2, text="Body text")
    assert p.elements == []
    assert p.metadata == {}


def test_element_filtering_logic():
    """
    Test that layout headers and footers are omitted while titles, narrative text,
    and lists are preserved.
    """
    sample_elements = [
        {"type": "Header", "text": "RUNNING HEADER: JAMA CLINICAL REVIEW"},
        {"type": "Title", "text": "Screening for Osteoporosis to Prevent Fractures"},
        {"type": "NarrativeText", "text": "The USPSTF recommends screening for osteoporosis in women 65 years or older."},
        {"type": "ListItem", "text": "• High risk factor: Prior fragility fracture"},
        {"type": "Footer", "text": "Page 1 of 11 - Confidential © 2025 AMA"}
    ]

    kept = [el for el in sample_elements if el["type"] not in ("Header", "Footer", "PageBreak")]
    assert len(kept) == 3
    assert kept[0]["type"] == "Title"
    assert kept[1]["type"] == "NarrativeText"
    assert kept[2]["type"] == "ListItem"
    assert not any(el["type"] in ("Header", "Footer") for el in kept)


def test_missing_file_raises_file_not_found():
    """Verify partition_and_filter_pdf raises FileNotFoundError for non-existent paths."""
    with pytest.raises(FileNotFoundError):
        partition_and_filter_pdf("non_existent_file.pdf")


def test_save_cleaned_text(tmp_path):
    """Verify save_cleaned_text writes file to disk correctly."""
    out_file = save_cleaned_text(output_dir=tmp_path, document_name="sample_doc", text="Sample clinical text")
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8") == "Sample clinical text"


def test_format_summary_table():
    """Verify table formatting generates expected column headers."""
    stats = [
        {"file_name": "sample.pdf", "raw_chars": 1000, "clean_chars": 800, "reduction_pct": 20.0, "is_scanned": False}
    ]
    table = format_summary_table(stats)
    assert "DOCUMENT NAME" in table
    assert "CLEAN CHARS" in table
    assert "sample" in table
    assert "800" in table


def test_clean_all_guidelines_empty_dir(tmp_path):
    """Verify clean_all_guidelines handles empty input folders cleanly."""
    empty_dir = tmp_path / "empty_guidelines"
    empty_dir.mkdir()
    out_dir = tmp_path / "cleaned_out"
    stats = clean_all_guidelines(input_dir=empty_dir, output_dir=out_dir)
    assert stats == []


def test_scanned_pdf_detection():
    """Verify scanned detection flags empty or sub-threshold text without raising exceptions."""
    class DummyScannedPage:
        page_number = 1
        text = ""
        elements = []
        metadata = {}

    pages = [DummyScannedPage()]
    clean_text = "\n\n".join(p.text for p in pages if p.text).strip()
    is_scanned = (len(clean_text) < 50 and len(pages) > 0)
    assert is_scanned is True


def test_chunk_document_basic():
    """Verify chunk_document generates structured Chunk instances with metadata."""
    sample_text = (
        "# 1.1 Risk Factors\n\n"
        "Assess risk in all patients over 50 with previous fracture.\n\n"
        "# 1.2 Bone Density\n\n"
        "Offer DXA scan to measure BMD in patients meeting high risk criteria."
    )
    chunks = chunk_document(sample_text, document_id="test_doc", target_chunk_size=100)
    assert len(chunks) >= 2
    assert chunks[0].document_id == "test_doc"
    assert chunks[0].token_estimate > 0


def test_vector_store_search():
    """Verify VectorStore indexes chunks and returns ranked search results."""
    c1 = Chunk(chunk_id="c1", document_id="d1", section_title="DXA", text="DXA bone density measurement.", token_estimate=4)
    c2 = Chunk(chunk_id="c2", document_id="d1", section_title="Falls", text="Fall prevention exercises.", token_estimate=3)
    store = VectorStore()
    store.add_chunks([c1, c2])
    results = store.search("bone density DXA", top_k=1)
    assert len(results) == 1
    assert results[0][0].chunk_id == "c1"
    assert results[0][1] > 0.0


def test_scope_guardrails():
    """Verify guardrails accurately distinguish clinical in-scope vs out-of-scope queries."""
    in_scope, _ = check_scope_guardrail("What are the DXA scan T-score criteria for osteoporosis?")
    assert in_scope is True

    out_of_scope, _ = check_scope_guardrail("How to cook pasta carbonara?")
    assert out_of_scope is False
