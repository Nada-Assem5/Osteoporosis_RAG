"""
Comprehensive Pytest Test Suite for Clinical Practice Guidelines RAG Pipeline.
Imports exclusively from scripts/ (single source of truth).
"""

import os
import sys
import json
import pytest
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts import (
    ingest,
    chunk,
    embeddings,
    vector_db,
    retrieval,
    grounded_generation
)


# =====================================================================
# STAGE 1: INGESTION & TEXT NORMALIZATION TESTS
# =====================================================================

def test_filter_structural_noise_drops_headers_footers():
    """Verify filter_structural_noise filters Header, Footer, and PageBreak elements."""
    class MockEl:
        def __init__(self, cat, text):
            self.category = cat
            self.text = text

    raw_elements = [
        MockEl("Header", "RUNNING HEADER: CLINICAL REVIEW"),
        MockEl("Title", "1.4 Bone Density Assessment"),
        MockEl("NarrativeText", "Offer a central DXA bone mineral density scan to confirm osteoporosis."),
        MockEl("Footer", "Page 1 of 12 © 2025 AMA"),
        MockEl("PageBreak", "")
    ]

    filtered = ingest.filter_structural_noise(raw_elements)
    assert len(filtered) == 2
    assert filtered[0].text == "1.4 Bone Density Assessment"
    assert filtered[1].text.startswith("Offer a central DXA")


def test_normalize_unicode():
    """Verify normalize_unicode strips zero-width artifacts and standardizes NFKC characters."""
    raw = "Clinical\xa0guideline\u200b with\ufeff unicode."
    norm = ingest.normalize_unicode(raw)
    assert "Clinical guideline with unicode." == norm


def test_dehyphenate_text():
    """Verify de-hyphenation merges line-split words and normalizes soft hyphens."""
    raw = "osteo-\nporosis screening and post-\nmenopausal fracture risk."
    cleaned = ingest.dehyphenate_text(raw)
    assert "osteoporosis screening" in cleaned
    assert "postmenopausal fracture" in cleaned


def test_strip_punctuation():
    """Verify strip_punctuation strips ASCII punctuation using string.punctuation."""
    assert ingest.strip_punctuation("[osteoporosis]...") == "osteoporosis"
    assert ingest.strip_punctuation("  (dxa)  ").strip() == "dxa"


def test_clean_text_unified():
    """Verify clean_text performs end-to-end Unicode normalization and de-hyphenation."""
    raw = "Offer osteo-\nporosis screening\xa0to women aged 65 and older."
    cleaned = ingest.clean_text(raw)
    assert cleaned == "Offer osteoporosis screening to women aged 65 and older."


# =====================================================================
# STAGE 2: TOKEN CHUNKING & METADATA TESTS
# =====================================================================

def test_token_based_chunking_and_sizing():
    """Verify chunk_document adheres to target token bounds and generates chunk IDs."""
    sample_text = (
        "NICE guideline NG259 recommends assessing fracture risk in adults. "
        "Offer a central DXA scan to assess bone mineral density in women aged 65 and older. "
        "For younger postmenopausal women, calculate 10-year fracture risk using FRAX or QFracture. "
    ) * 15

    chunks = chunk.chunk_document(sample_text, document_id="test_guideline", target_chunk_tokens=100, chunk_overlap_tokens=20)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.chunk_id.startswith("test_guideline_chk_")
        assert c.token_estimate > 0


def test_extract_clinical_metadata():
    """Verify extraction of target population, topics, and guideline issuer."""
    text = "USPSTF recommends screening for osteoporosis with DXA in women aged 65 and older to prevent fractures."
    meta = chunk.extract_clinical_metadata(text, "osteoporosis-screening-final-recommendation")

    assert "Screening & Diagnosis" in meta["topics"]
    assert "Women Aged >= 65" in meta["population"]
    assert meta["guideline_issuer"] == "USPSTF"


# =====================================================================
# STAGE 3 & 4: VECTOR STORE & HYBRID SEARCH TESTS
# =====================================================================

def test_vector_store_keyword_semantic_hybrid():
    """Verify Okapi BM25, Dense Semantic, and Hybrid RRF searches."""
    c1 = vector_db.Chunk("chk_01", "nice_guideline", "1.4", "Offer DXA scan for bone mineral density in women 65 and older.", 1, 14)
    c2 = vector_db.Chunk("chk_02", "uspstf_rec", "Recommendation", "Screen all women aged 65 and older with dual-energy X-ray absorptiometry.", 1, 14)

    store = vector_db.VectorStore()
    store.add_chunks([c1, c2])

    res_kw = store.search("DXA scan", mode="keyword", top_k=2)
    res_sem = store.search("DXA scan", mode="semantic", top_k=2)
    res_hyb = store.search("DXA scan", mode="hybrid", top_k=2)

    assert len(res_kw) > 0
    assert len(res_sem) > 0
    assert len(res_hyb) > 0


# =====================================================================
# STAGE 5: SAFETY GUARDRAILS & CONFIDENCE BANDS
# =====================================================================

def test_scope_guardrails():
    """Verify 3-tier clinical risk triage and emergency deflection."""
    tier1, _ = retrieval.classify_query_risk("When should DXA scan be ordered according to NICE?")
    assert tier1 == retrieval.QueryRiskCategory.ALLOWED

    tier2, _ = retrieval.classify_query_risk("My mother is 72 with hip fracture, what should I give her?")
    assert tier2 == retrieval.QueryRiskCategory.NEEDS_CAUTION

    tier3, _ = retrieval.classify_query_risk("Patient collapsed with severe chest pain and acute cardiac arrest.")
    assert tier3 == retrieval.QueryRiskCategory.REFUSE_REDIRECT

    tier4, _ = retrieval.classify_query_risk("How do I replace a car transmission?")
    assert tier4 == retrieval.QueryRiskCategory.REFUSE_REDIRECT


def test_confidence_tier_score_bands():
    """Verify canonical confidence tier cutoff boundaries."""
    assert retrieval.compute_confidence_tier(0.85) == retrieval.ConfidenceTier.HIGH
    assert retrieval.compute_confidence_tier(0.60) == retrieval.ConfidenceTier.HIGH
    assert retrieval.compute_confidence_tier(0.50) == retrieval.ConfidenceTier.MEDIUM
    assert retrieval.compute_confidence_tier(0.30) == retrieval.ConfidenceTier.MEDIUM
    assert retrieval.compute_confidence_tier(0.15) == retrieval.ConfidenceTier.LOW
    assert retrieval.compute_confidence_tier(0.015) == retrieval.ConfidenceTier.LOW
    assert retrieval.compute_confidence_tier(0.005) == retrieval.ConfidenceTier.INSUFFICIENT_EVIDENCE


# =====================================================================
# STAGE 6: DAY 3 GROUNDED SYNTHESIS & CLAIM VERIFICATION
# =====================================================================

def test_filter_unsupported_claims_runtime_stripping():
    """Verify unsupported claim filtering strips hallucinated sentences."""
    c1 = grounded_generation.Chunk(
        "nice_chk_01",
        "nice_guideline",
        "1.4 Bone density",
        "Offer a central DXA scan to assess bone mineral density in women aged 65 and older.",
        1,
        15
    )
    chunk_map = {"nice_chk_01": c1}

    resp = grounded_generation.ClinicalSynthesisResponse(
        query="When to offer DXA scan?",
        direct_answer="Offer DXA scan for bone mineral density [nice_chk_01]. Perform spine surgery immediately [nice_chk_01].",
        target_population="Women 65+",
        key_recommendations=[
            "Offer central DXA scan to women 65 and older. [nice_chk_01]",
            "Perform immediate spine surgery on all patients. [nice_chk_01]"
        ],
        citations=[{"chunk_id": "nice_chk_01"}],
        evidence_strength="High",
        clinical_caveats=[]
    )

    filtered_resp, logs = grounded_generation.filter_unsupported_claims(resp, chunk_map, threshold=0.25)
    assert len(logs) >= 1
    assert not any("surgery" in rec.lower() for rec in filtered_resp.key_recommendations)
    assert any("dxa" in rec.lower() for rec in filtered_resp.key_recommendations)


def test_grounded_generation_with_custom_prompt():
    """Verify custom user prompt integration into clinical synthesis prompt."""
    c1 = grounded_generation.Chunk(
        "nice_chk_01",
        "nice_guideline",
        "1.4 Bone density",
        "Offer a central DXA scan to assess bone mineral density in women aged 65 and older.",
        1,
        15
    )
    synth = grounded_generation.ClinicalSynthesizer(provider="deterministic_fallback")
    prompt_str = synth._build_synthesis_prompt(
        query="When to offer DXA scan?",
        retrieved_passages=[(c1, 0.85)],
        custom_prompt="Summarize in simple bullet points for general practitioners."
    )
    assert "Summarize in simple bullet points for general practitioners." in prompt_str
    assert "When to offer DXA scan?" in prompt_str
    assert "nice_chk_01" in prompt_str

    resp = synth.synthesize(
        query="When to offer DXA scan?",
        retrieved_passages=[(c1, 0.85)],
        custom_prompt="Summarize in simple bullet points for general practitioners."
    )
    assert resp.query == "When to offer DXA scan?"
    assert len(resp.key_recommendations) > 0


def test_categorized_evaluation_benchmark():
    """Verify 24 categorized benchmark test questions in scripts/data/eval_questions.json."""
    q_path = ROOT_DIR / "scripts" / "data" / "eval_questions.json"
    questions = retrieval.load_eval_questions(q_path)
    assert len(questions) == 24

    categories = {q.category for q in questions}
    assert "direct" in categories
    assert "multi_chunk" in categories
    assert "ambiguous" in categories
    assert "out_of_scope" in categories

    direct_qs = [q for q in questions if q.category == "direct"]
    multi_qs = [q for q in questions if q.category == "multi_chunk"]
    ambig_qs = [q for q in questions if q.category == "ambiguous"]
    out_qs = [q for q in questions if q.category == "out_of_scope"]

    assert len(direct_qs) == 6
    assert len(multi_qs) == 6
    assert len(ambig_qs) == 6
    assert len(out_qs) == 6


def test_ranking_metrics_calculation():
    """Verify Precision@K, Recall@K, Hit@K, MRR, MAP@K, NDCG@K, and Latency calculations."""
    c1 = vector_db.Chunk("chk_01", "nice_guideline", "1.4", "Offer DXA scan for bone mineral density.", 1, 8)
    c2 = vector_db.Chunk("chk_02", "uspstf_rec", "Recommendation", "Screen women 65 and older with DXA.", 1, 8)

    questions = [
        retrieval.EvalQuestion("When should DXA be offered?", ["chk_01"], "direct"),
        retrieval.EvalQuestion("How do I fix a car alternator?", [], "out_of_scope")
    ]

    evaluator = retrieval.RAGEvaluator(questions)
    evaluator.store = vector_db.VectorStore()
    evaluator.store.add_chunks([c1, c2])

    res = evaluator.evaluate(top_k=2, mode="keyword")
    m = res["metrics"]

    assert m["precision_at_2"] > 0
    assert m["recall_at_2"] > 0
    assert m["hit_at_2"] == 1.0
    assert m["mrr"] > 0
    assert m["ndcg_at_2"] > 0
    assert m["guardrail_deflection_rate"] == 1.0
    assert "direct" in res["category_breakdown"]
    assert "out_of_scope" in res["category_breakdown"]


def test_retrieval_comparison_engine_execution(tmp_path):
    """Verify RetrievalComparisonEngine generates JSON and Markdown comparison reports."""
    c1 = vector_db.Chunk("chk_01", "nice_guideline", "1.4", "Offer DXA scan for bone mineral density.", 1, 8)
    engine = retrieval.RetrievalComparisonEngine(output_dir=tmp_path)
    engine.evaluator.store = vector_db.VectorStore()
    engine.evaluator.store.add_chunks([c1])

    test_configs = [
        {"name": "Test BM25", "mode": "keyword", "top_k": 1, "alpha": 0.5},
        {"name": "Test Hybrid", "mode": "hybrid", "top_k": 1, "alpha": 0.5}
    ]

    summary = engine.run_comparison(configurations=test_configs)
    assert summary["num_configurations"] == 2
    assert "winner" in summary
    assert (tmp_path / "retrieval_comparison_report.json").exists()
    assert (tmp_path / "retrieval_comparison_report.md").exists()

