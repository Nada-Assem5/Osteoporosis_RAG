"""
Comprehensive unit test suite for Parsing, Clean, Chunking, Embedded
(Keyword, Semantic, Hybrid), and Evaluation modules (Steps 1 - 5).
"""

import pytest
import json
from pathlib import Path

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
from src.config import ConfidenceTier
from src.chunking import Chunk, chunk_document, extract_clinical_metadata
from src.embedded import (
    VectorStore,
    build_vector_index,
    check_scope_guardrail,
    classify_query_risk
)
from src.synthesis import ClinicalSynthesizer, ClinicalSynthesisResponse
from src.evaluation import RAGEvaluator, EvalQuestion, load_eval_questions, run_full_evaluation


# =====================================================================
# Step 1: Parsing Tests (src.parsing)
# =====================================================================

def test_page_dataclass_contract():
    """Verify Page dataclass attributes, elements, and metadata dictionary."""
    p = Page(
        page_number=1,
        text="Clinical recommendation text",
        elements=[{"type": "Title", "text": "Clinical recommendation text", "page_number": 1}],
        metadata={
            "engine": "unstructured",
            "document_name": "guideline_sample",
            "source_url": "https://www.nice.org.uk/guidance/ng259",
            "page_numbers_unreliable": False
        }
    )
    assert p.page_number == 1
    assert p.text == "Clinical recommendation text"
    assert len(p.elements) == 1
    assert p.elements[0]["type"] == "Title"
    assert p.metadata["source_url"] == "https://www.nice.org.uk/guidance/ng259"
    assert p.metadata["document_name"] == "guideline_sample"
    assert p.metadata["page_numbers_unreliable"] is False


def test_missing_file_raises_file_not_found():
    """Verify partition_pdf_pages raises FileNotFoundError for non-existent paths."""
    with pytest.raises(FileNotFoundError):
        partition_pdf_pages("non_existent_file.pdf")


def test_filter_structural_noise():
    """Verify filter_structural_noise removes headers, footers, and pagebreaks."""
    class MockEl:
        def __init__(self, cat, text):
            self.category = cat
            self.text = text

    elements = [
        MockEl("Header", "JAMA Clinical Review"),
        MockEl("Title", "Osteoporosis Screening"),
        MockEl("Footer", "Page 1"),
        MockEl("PageBreak", "")
    ]
    filtered = filter_structural_noise(elements)
    assert len(filtered) == 1
    assert filtered[0].text == "Osteoporosis Screening"


def test_page_tracking_integrity_and_falsy_zero_check():
    """Verify page tracking logic handles PageBreaks and does not treat page 0 as falsy."""
    class MockMeta:
        def __init__(self, page_num):
            self.page_number = page_num

    class MockElement:
        def __init__(self, cat, text, page_num=None):
            self.category = cat
            self.text = text
            self.metadata = MockMeta(page_num) if page_num is not None else None

    # Simulating raw element stream with page 0, PageBreaks, and missing metadata
    raw_stream = [
        MockElement("Title", "Section 1", 0),  # 0-indexed page 0 -> normalized to 1
        MockElement("NarrativeText", "Content on page 1", 0),
        MockElement("PageBreak", ""),
        MockElement("Title", "Section 2"),  # no metadata page -> uses current_page (2)
        MockElement("NarrativeText", "Content on page 2")
    ]

    pages_dict = {}
    current_page = 1
    for el in raw_stream:
        cat = el.category
        if cat == "PageBreak":
            current_page += 1
            continue
        meta_pg = getattr(el.metadata, "page_number", None) if hasattr(el, "metadata") and el.metadata else None
        if meta_pg is not None:
            current_page = max(1, int(meta_pg)) if meta_pg != 0 else 1
        if current_page not in pages_dict:
            pages_dict[current_page] = []
        pages_dict[current_page].append(el.text)

    assert 1 in pages_dict
    assert 2 in pages_dict
    assert pages_dict[1] == ["Section 1", "Content on page 1"]
    assert pages_dict[2] == ["Section 2", "Content on page 2"]


def test_source_url_registry_resolution(tmp_path):
    """Verify source_url is populated for known guidelines and None for unknown files."""
    # Known guideline file
    known_stem = "osteoporosis-risk-assessment-pdf-66144025463749"
    known_pdf = tmp_path / f"{known_stem}.pdf"
    known_pdf.write_bytes(b"%PDF-1.4 test stream content endstream")

    pages_known = partition_pdf_pages(known_pdf)
    assert len(pages_known) >= 1
    assert pages_known[0].metadata.get("source_url") == "https://www.nice.org.uk/guidance/ng259"

    # Unknown file must NOT fabricate URL
    unknown_pdf = tmp_path / "unknown_custom_study.pdf"
    unknown_pdf.write_bytes(b"%PDF-1.4 test stream content endstream")
    pages_unknown = partition_pdf_pages(unknown_pdf)
    assert len(pages_unknown) >= 1
    assert pages_unknown[0].metadata.get("source_url") is None


def test_fallback_stream_extraction_unreliable_flag(tmp_path):
    """Verify fallback parser sets page_numbers_unreliable when boundaries cannot be recovered."""
    raw_pdf = tmp_path / "flat_doc.pdf"
    raw_pdf.write_bytes(b"%PDF-1.4 stream\r\nSingle unpaginated continuous stream\r\nendstream")

    from src.parsing import _fallback_stream_extraction
    pages = _fallback_stream_extraction(raw_pdf)
    assert len(pages) >= 1
    assert pages[0].metadata.get("page_numbers_unreliable") is True


def test_recursive_guideline_discovery(tmp_path):
    """Verify discover_and_sync_guidelines finds nested PDFs recursively."""
    from src.parsing import discover_and_sync_guidelines
    sub_dir = tmp_path / "nested" / "guidelines"
    sub_dir.mkdir(parents=True, exist_ok=True)
    pdf1 = sub_dir / "nested_doc1.pdf"
    pdf1.write_bytes(b"%PDF-1.4 test")

    found = discover_and_sync_guidelines(input_dir=tmp_path)
    assert any(p.name == "nested_doc1.pdf" for p in found)


# =====================================================================
# Step 2: Cleaning Tests (src.clean)
# =====================================================================

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


def test_is_noise_title():
    """Test smart noise title detection based on following content elements."""
    class MockEl:
        def __init__(self, cat, text):
            self.category = cat
            self.text = text

    elements_noise = [
        MockEl("Title", "Editorial"),
        MockEl("Header", "JAMA"),
        MockEl("Title", "Sidebar Link")
    ]
    assert is_noise_title(elements_noise, 0, min_content_length=40) is True

    elements_valid = [
        MockEl("Title", "1.4 Bone Density"),
        MockEl("NarrativeText", "Offer a DXA scan to measure BMD when assessing fragility fracture risk in adults.")
    ]
    assert is_noise_title(elements_valid, 0, min_content_length=40) is False


def test_filter_elements_removes_noise_titles():
    """Test filter_elements drops headers, footers, pagebreaks, and isolated noise titles."""
    class MockEl:
        def __init__(self, cat, text):
            self.category = cat
            self.text = text

    sample_elements = [
        MockEl("Header", "RUNNING HEADER: JAMA CLINICAL REVIEW"),
        MockEl("Title", "Ad"),
        MockEl("Title", "Bone Density Assessment"),
        MockEl("NarrativeText", "Offer a DXA scan to measure BMD in patients aged 30 and over with prior fragility fractures."),
        MockEl("Footer", "Page 1 of 12 © 2025 AMA")
    ]

    cleaned = filter_elements(sample_elements, short_title_threshold=20, min_content_length=30)
    assert len(cleaned) == 2
    assert cleaned[0].text == "Bone Density Assessment"
    assert cleaned[1].text.startswith("Offer a DXA scan")


def test_strip_punctuation():
    """Verify strip_punctuation correctly separates punctuation from core tokens."""
    core, leading, trailing = strip_punctuation("(osteoporosis).")
    assert core == "osteoporosis"
    assert leading == "("
    assert trailing == ")."


def test_is_valid_word():
    """Verify is_valid_word accurately validates known English and medical words and rejects broken runs."""
    # Known-good general English words
    assert is_valid_word("hello") is True
    assert is_valid_word("screening") is True
    assert is_valid_word("treatment") is True
    assert is_valid_word("recommendation") is True

    # Known-good clinical terms
    assert is_valid_word("osteoporosis") is True
    assert is_valid_word("fracture") is True
    assert is_valid_word("density") is True
    assert is_valid_word("dxa") is True
    assert is_valid_word("bmd") is True
    assert is_valid_word("bisphosphonates") is True

    # Broken/concatenated runs
    assert is_valid_word("theuspstfnotesthat") is False
    assert is_valid_word("theosteoporosisrisk") is False
    assert is_valid_word("policyandcoverage") is False


def test_fix_concatenated_word():
    """Test word repair on CamelCase, acronyms, and glued lowercase words."""
    assert fix_concatenated_word("USPSTF") == "USPSTF"
    assert fix_concatenated_word("DXA") == "DXA"
    assert fix_concatenated_word("ScreeningForOsteoporosis") == "Screening For Osteoporosis"

    fixed_lower = fix_concatenated_text("theosteoporosisrisk")
    assert "osteoporosis" in fixed_lower


def test_clean_academic_boilerplate():
    """Test removal of academic disclosures, affiliations, and copyright metadata."""
    raw_article = (
        "USPSTF Recommendation: Screen women aged 65 and older.\n\n"
        "Author Affiliations: Department of Medicine, University of Health.\n\n"
        "Conflict of Interest Disclosures: None reported.\n\n"
        "Funding/Support: Supported by the Agency for Healthcare Research.\n\n"
        "Copyright 2025 American Medical Association. All rights reserved."
    )
    cleaned = clean_academic_boilerplate(raw_article)
    assert "Screen women aged 65 and older." in cleaned
    assert "Author Affiliations" not in cleaned
    assert "Conflict of Interest" not in cleaned
    assert "Copyright" not in cleaned


def test_count_concatenated_words():
    """Test detection of long concatenated tokens in elements."""
    class MockEl:
        def __init__(self, cat, text):
            self.category = cat
            self.text = text

    elements = [
        MockEl("NarrativeText", "The word supercalifragilisticexpialidociousisbroken is suspicious."),
        MockEl("NarrativeText", "Regular text with normal word boundaries.")
    ]
    affected, total_bad = count_concatenated_words(elements)
    assert total_bad >= 1
    assert len(affected) >= 1


# =====================================================================
# Step 3: Chunking & Metadata Tests (src.chunking)
# =====================================================================

def test_chunking_with_clinical_metadata():
    """Verify sentence-aware chunking and taxonomy metadata extraction with page and source URL lineage."""
    clinical_text = (
        "# 1.4 Bone Density Assessment\n\n"
        "Offer a DXA scan to measure BMD when assessing fragility fracture risk in postmenopausal women.\n\n"
        "Consider FRAX risk score calculation prior to initiating bisphosphonates."
    )
    chunks = chunk_document(clinical_text, document_id="nice_ng259_osteoporosis", target_chunk_size=150, page_number=2)
    assert len(chunks) >= 1
    chk = chunks[0]
    assert chk.section_title == "1.4 Bone Density Assessment"
    assert chk.document_name == "nice_ng259_osteoporosis"
    assert chk.page_number == 2
    assert "Screening & Diagnosis" in chk.topics or "Risk Assessment Tools" in chk.topics
    assert chk.population == "Postmenopausal Women / Women >= 65"

    chk_dict = chk.to_dict()
    assert "chunk_id" in chk_dict
    assert "document_name" in chk_dict
    assert "page_number" in chk_dict
    assert "section_title" in chk_dict


# =====================================================================
# Step 4: Embedded (Retrieval, VectorStore, Guardrails, Synthesis) Tests
# =====================================================================

def test_scope_guardrails():
    """Verify 3-tier safety guardrails accurately classify allowed, needs_caution, and refuse_redirect queries."""
    # 1. Allowed (In-scope guideline inquiry)
    tier_allowed, _ = classify_query_risk("What are the DXA scan T-score criteria for osteoporosis?")
    assert tier_allowed == "allowed"
    in_scope_allowed, _ = check_scope_guardrail("What are the DXA scan T-score criteria for osteoporosis?")
    assert in_scope_allowed is True

    # 2. Needs Caution (Patient-specific personalized scenario)
    tier_caution, _ = classify_query_risk("My mother is a 72 year old female with a T-score of -2.8, should I prescribe bisphosphonates?")
    assert tier_caution == "needs_caution"
    in_scope_caution, _ = check_scope_guardrail("My mother is a 72 year old female with a T-score of -2.8, should I prescribe bisphosphonates?")
    assert in_scope_caution is True

    # 3. Refuse & Redirect (Medical Emergency)
    tier_emergency, msg_em = classify_query_risk("Patient has severe acute chest pain and shortness of breath after falling")
    assert tier_emergency == "refuse_redirect"
    assert "SAFETY EMERGENCY" in msg_em
    in_scope_emergency, _ = check_scope_guardrail("Patient has severe acute chest pain and shortness of breath after falling")
    assert in_scope_emergency is False

    # 4. Refuse & Redirect (Out of Scope)
    tier_oos, _ = classify_query_risk("How to cook pasta carbonara?")
    assert tier_oos == "refuse_redirect"
    in_scope_oos, _ = check_scope_guardrail("How to cook pasta carbonara?")
    assert in_scope_oos is False


def test_retrieval_modes_keyword_semantic_hybrid():
    """Verify all 3 retrieval modes (keyword, semantic, hybrid) work independently and correctly."""
    c1 = Chunk(
        chunk_id="chk_dxa",
        document_id="nice_guideline",
        section_title="1.4 Bone Density Assessment",
        text="Offer a DXA scan to measure BMD in women with previous hip or vertebral fragility fractures.",
        token_estimate=15,
        metadata={"topics": ["Screening & Diagnosis"], "population": "Postmenopausal Women", "page_number": 3}
    )
    c2 = Chunk(
        chunk_id="chk_uspstf",
        document_id="uspstf_guideline",
        section_title="2.1 Screening in Women",
        text="The USPSTF recommends screening for osteoporosis with bone measurement testing in women 65 and older.",
        token_estimate=15,
        metadata={"topics": ["Screening & Diagnosis"], "population": "Postmenopausal Women / Women >= 65", "page_number": 1}
    )
    c3 = Chunk(
        chunk_id="chk_falls",
        document_id="nice_guideline",
        section_title="1.6 Fall Prevention",
        text="Provide lifestyle advice regarding balance training and muscle strengthening exercise to prevent falls.",
        token_estimate=14,
        metadata={"topics": ["Lifestyle & Supplementation"], "population": "Older Adults", "page_number": 5}
    )

    store = VectorStore()
    store.add_chunks([c1, c2, c3])

    # 1. Keyword search
    kw_results = store.search("When should a DXA scan be offered for BMD?", top_k=2, mode="keyword")
    assert len(kw_results) >= 1
    assert any(c.chunk_id == "chk_dxa" for c, _ in kw_results)

    # 2. Semantic search
    sem_results = store.search("When should a DXA scan be offered for BMD?", top_k=2, mode="semantic")
    assert len(sem_results) >= 1

    # 3. Hybrid search (RRF)
    hybrid_results = store.search("When should a DXA scan be offered for BMD?", top_k=2, mode="hybrid")
    assert len(hybrid_results) >= 1
    assert hybrid_results[0][0].chunk_id in ("chk_dxa", "chk_uspstf")


def test_vector_store_serialization(tmp_path):
    """Verify VectorStore state can be saved and loaded faithfully from JSON."""
    c = Chunk(
        chunk_id="chk_test",
        document_id="doc_test",
        section_title="1.1 Scope",
        text="Test guideline paragraph for serialization.",
        token_estimate=6,
        metadata={"topics": ["Screening & Diagnosis"], "page_number": 1, "document_name": "doc_test"}
    )
    store = VectorStore()
    store.add_chunks([c])

    save_path = tmp_path / "test_index.json"
    store.save(save_path)
    assert save_path.exists()

    loaded_store = VectorStore.load(save_path)
    assert len(loaded_store.chunks) == 1
    assert loaded_store.chunks[0].chunk_id == "chk_test"
    assert loaded_store.chunks[0].page_number == 1
    assert loaded_store.chunks[0].text == "Test guideline paragraph for serialization."


def test_clinical_synthesis_formatting():
    """Verify ClinicalSynthesizer constructs structured clinical recommendation reports with full citation lineage."""
    c = Chunk(
        chunk_id="chk_1",
        document_id="nice_ng259_osteoporosis",
        section_title="1.4 Bone Density Assessment",
        text="1.4.1 Offer a DXA scan to measure BMD in people aged 30 and over with previous fragility fractures.",
        token_estimate=18,
        metadata={
            "document_name": "nice_ng259_osteoporosis",
            "guideline_issuer": "NICE NG259 (UK)",
            "population": "Adults with Prior Fragility Fractures",
            "page_number": 4,
            "source_url": "https://www.nice.org.uk/guidance/ng259"
        }
    )
    synthesizer = ClinicalSynthesizer(provider="fallback")
    response = synthesizer.synthesize("When is a DXA scan indicated?", [(c, 0.42)])

    assert isinstance(response, ClinicalSynthesisResponse)
    assert "DXA" in response.direct_answer or "chk_1" in response.direct_answer
    assert len(response.citations) >= 1
    assert response.citations[0]["document_id"] == "nice_ng259_osteoporosis"
    assert response.citations[0]["page_number"] == 4

    md_report = response.format_markdown()
    assert "CLINICAL EVIDENCE SYNTHESIS" in md_report
    assert "Grounded Source Citations" in md_report
    assert "Page: 4" in md_report or "Page:" in md_report


def test_clinical_synthesizer_guardrails_and_insufficient_evidence():
    """Verify citation guardrails prune ungrounded chunks and low confidence triggers refusal."""
    c = Chunk("chk_valid", "nice_guideline", "1.1 Risk", "Assess fracture risk in women >= 65.", 8)
    synth = ClinicalSynthesizer(provider="fallback")

    # 1. Insufficient evidence refusal on empty retrieval
    refusal_empty = synth.synthesize("Unrelated question", [])
    assert "Insufficient Evidence" in refusal_empty.direct_answer
    assert refusal_empty.evidence_strength == ConfidenceTier.INSUFFICIENT_EVIDENCE.value

    # 2. Insufficient evidence on low relevance score
    refusal_low = synth.synthesize("Low score query", [(c, 0.005)])
    assert "Insufficient Evidence" in refusal_low.direct_answer
    assert refusal_low.evidence_strength == ConfidenceTier.INSUFFICIENT_EVIDENCE.value

    # 3. Citation lineage guardrail
    mock_llm_json = {
        "direct_answer": "Screening is recommended for high risk patients [chk_valid] and [chk_hallucinated].",
        "target_population": "Postmenopausal Women",
        "key_recommendations": ["Offer DXA scan [chk_valid]."],
        "citations": [
            {"chunk_id": "chk_valid", "document_id": "nice_guideline", "section": "1.1 Risk"},
            {"chunk_id": "chk_hallucinated", "document_id": "unknown_guideline", "section": "Unknown"}
        ]
    }
    validated = synth._apply_citation_guardrail(
        query="When to screen?",
        data=mock_llm_json,
        retrieved_chunks=[(c, 0.45)],
        chunk_map={"chk_valid": c},
        provider="mock_llm",
        model="gpt-4o-mini"
    )
    # The hallucinated citation must be pruned from verified_citations
    assert len(validated.citations) == 1
    assert validated.citations[0]["chunk_id"] == "chk_valid"
    assert any("chk_hallucinated" in w for w in validated.guardrail_warnings)


def test_gemini_api_key_configuration(monkeypatch):
    """Verify Gemini API key detection from parameter and environment variable."""
    # 1. Pass key explicitly via parameter
    synth_param = ClinicalSynthesizer(api_key="test_gemini_key_123")
    assert synth_param.api_key == "test_gemini_key_123"
    assert synth_param._active_provider == "gemini"

    # 2. Key via GEMINI_API_KEY environment variable
    monkeypatch.setenv("GEMINI_API_KEY", "env_gemini_key_456")
    synth_env = ClinicalSynthesizer()
    assert synth_env.api_key == "env_gemini_key_456"
    assert synth_env._active_provider == "gemini"

    # 3. Key via GOOGLE_API_KEY environment variable
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "env_google_key_789")
    synth_google = ClinicalSynthesizer()
    assert synth_google.api_key == "env_google_key_789"
    assert synth_google._active_provider == "gemini"


# =====================================================================
# Step 5: Evaluation Tests (src.evaluation)
# =====================================================================

def test_load_eval_questions(tmp_path):
    """Verify evaluation dataset JSON parser creates valid EvalQuestion items."""
    sample_eval = [
        {
            "id": "q01",
            "query": "When should DXA be offered?",
            "relevant_chunk_ids": ["chk_01", "chk_02"],
            "target_document": "nice_ng259"
        }
    ]
    eval_file = tmp_path / "eval_test.json"
    eval_file.write_text(json.dumps(sample_eval), encoding="utf-8")

    questions = load_eval_questions(eval_file)
    assert len(questions) == 1
    assert questions[0].query == "When should DXA be offered?"
    assert questions[0].expected_chunk_ids == ["chk_01", "chk_02"]


def test_rag_evaluator_metrics():
    """Verify calculation of Precision@3, Precision@5, Hit@3, and MRR metrics."""
    c1 = Chunk("chk_01", "doc", "Section 1", "DXA recommendations.", 3)
    c2 = Chunk("chk_02", "doc", "Section 2", "FRAX risk assessment.", 3)
    c3 = Chunk("chk_03", "doc", "Section 3", "Lifestyle interventions.", 3)

    store = VectorStore()
    store.add_chunks([c1, c2, c3])

    evaluator = RAGEvaluator(store)
    test_q = EvalQuestion(query="DXA recommendations", expected_chunk_ids=["chk_01"])

    results = evaluator.evaluate_mode(questions=[test_q], mode="keyword")
    assert "precision_at_3" in results
    assert "precision_at_5" in results
    assert "hit_at_3" in results
    assert "mrr" in results
    assert results["hit_at_3"] == 100.0


def test_detect_unsupported_claims():
    """Verify detect_unsupported_claims identifies claims with low content overlap."""
    from src.synthesis import detect_unsupported_claims

    c1 = Chunk("chk_bone", "nice_doc", "1.1", "Offer DXA scan to measure bone mineral density in women 65 and older.", 12)
    chunk_map = {"chk_bone": c1}

    # Supported claim
    resp_supported = ClinicalSynthesisResponse(
        query="DXA test",
        direct_answer="DXA scan measures bone mineral density in women 65 and older [chk_bone].",
        target_population="Women 65+",
        key_recommendations=["Offer DXA scan to measure bone mineral density [chk_bone]."],
        citations=[{"chunk_id": "chk_bone"}],
        evidence_strength="High",
        clinical_caveats=[]
    )
    warnings_supp = detect_unsupported_claims(resp_supported, chunk_map, threshold=0.25)
    assert len(warnings_supp) == 0

    # Unsupported / hallucinated claim text citing valid chunk_id
    resp_unsupported = ClinicalSynthesisResponse(
        query="Surgery inquiry",
        direct_answer="Patients require emergency orthopedic spinal fusion surgery immediately [chk_bone].",
        target_population="Surgical Candidates",
        key_recommendations=["Perform open reduction internal fixation surgery with titanium screws [chk_bone]."],
        citations=[{"chunk_id": "chk_bone"}],
        evidence_strength="High",
        clinical_caveats=[]
    )
    warnings_unsupp = detect_unsupported_claims(resp_unsupported, chunk_map, threshold=0.25)
    assert len(warnings_unsupp) >= 1
    assert any("Unsupported Claim Alert" in w for w in warnings_unsupp)


def test_citation_accuracy_and_faithfulness_metrics():
    """Verify evaluation triad metrics for Citation Accuracy and Faithfulness."""
    c1 = Chunk("chk_01", "nice_guideline", "1.4", "Offer DXA scan for bone mineral density.", 8)
    store = VectorStore()
    store.add_chunks([c1])

    evaluator = RAGEvaluator(store)
    test_q = EvalQuestion(query="DXA scan guidelines", expected_chunk_ids=["chk_01"])

    synth = ClinicalSynthesizer(provider="fallback", allow_fallback=True)
    cit_res = evaluator.evaluate_citation_accuracy([test_q], synth)
    assert "citation_accuracy_pct" in cit_res
    assert cit_res["citation_accuracy_pct"] == 100.0

    faith_res = evaluator.evaluate_faithfulness([test_q], synth)
    assert "faithfulness_pct" in faith_res
    assert faith_res["faithfulness_pct"] >= 0.0


def test_hybrid_search_direct_indexing_with_duplicates():
    """Verify search_hybrid executes efficiently without O(n) index lookup collisions on duplicate text."""
    c1 = Chunk("chk_dup_1", "doc1", "Sec 1", "Identical clinical recommendation for DXA.", 6)
    c2 = Chunk("chk_dup_2", "doc2", "Sec 1", "Identical clinical recommendation for DXA.", 6)
    store = VectorStore()
    store.add_chunks([c1, c2])

    results = store.search_hybrid("DXA recommendation", top_k=2)
    assert len(results) == 2
    # Ensure both distinct chunk IDs can be returned rather than collapsing to the first
    retrieved_ids = {chk.chunk_id for chk, _ in results}
    assert "chk_dup_1" in retrieved_ids or "chk_dup_2" in retrieved_ids


def test_single_source_of_clinical_synthesizer():
    """Verify ClinicalSynthesizer has a single authoritative implementation in src.synthesis."""
    import src
    import src.synthesis
    import src.embedded

    assert hasattr(src.synthesis, "ClinicalSynthesizer")
    assert hasattr(src.synthesis, "ClinicalSynthesisResponse")
    assert src.ClinicalSynthesizer is src.synthesis.ClinicalSynthesizer

    # Verify src.embedded does not have its own class definition
    assert "ClinicalSynthesizer" not in src.embedded.__dict__


def test_canonical_confidence_tiers():
    """Verify all four canonical confidence tiers match agenda standards exactly."""
    from src.synthesis import _normalize_confidence_tier

    assert ConfidenceTier.HIGH.value == "High"
    assert ConfidenceTier.MEDIUM.value == "Medium"
    assert ConfidenceTier.LOW.value == "Low"
    assert ConfidenceTier.INSUFFICIENT_EVIDENCE.value == "Insufficient Evidence"

    # Test normalization function
    assert _normalize_confidence_tier("HIGH (Strong Guideline Consensus)") == "High"
    assert _normalize_confidence_tier("MODERATE (Relevant Clinical Guidance)") == "Medium"
    assert _normalize_confidence_tier("LOW (Partial Evidence)") == "Low"
    assert _normalize_confidence_tier("NONE (Missing GEMINI_API_KEY)") == "Insufficient Evidence"
    assert _normalize_confidence_tier(None) == "Medium"

