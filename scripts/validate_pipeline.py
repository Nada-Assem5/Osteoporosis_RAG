"""
Verification runner: executes architectural checks and runs the end-to-end pipeline stages.
Located in scripts/ to validate the self-contained pipeline stages.
"""

import sys
import traceback
from pathlib import Path

# Add project root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# FIX: the actual module files are Ingest.py, Chunk.py, Embeddings.py,
# Vector_db.py, Retrieval.py, Grounded_Generation.py (capitalized) — the
# original lowercase import names don't exist and raised ImportError before
# a single check could run. Import the real module names and alias them to
# the lowercase names used throughout this file.
from scripts import (
    Ingest as ingest,
    Chunk as chunk,
    Embeddings as embeddings,
    Vector_db as vector_db,
    Retrieval as retrieval,
    Grounded_Generation as grounded_generation
)


def run_all_checks():
    print("=" * 88)
    print("  RUNNING PIPELINE ARCHITECTURAL & VALIDATION TESTS (scripts/ SOURCE OF TRUTH)")
    print("=" * 88)

    test_count = 0
    passed_count = 0

    def check(name, fn):
        nonlocal test_count, passed_count
        test_count += 1
        try:
            fn()
            print(f"  [PASS] {name}")
            passed_count += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()

    # 1. Structural Filter (Ingest.py)
    def test_filter():
        class MockEl:
            def __init__(self, cat, text):
                self.category = cat
                self.text = text
        els = [
            MockEl("Header", "RUNNING HEADER"),
            MockEl("Title", "1.4 Bone density"),
            MockEl("NarrativeText", "Offer central DXA scan to assess bone mineral density."),
            MockEl("Footer", "Page 1 of 12 © 2025 AMA")
        ]
        filtered = ingest.filter_structural_noise(els)
        assert len(filtered) == 2

    check("1. Structural Layout Noise Filtering (Ingest.py)", test_filter)

    # 2. Unicode & Hyphenation Normalization (Ingest.py)
    def test_norm():
        raw = "Offer osteo-\nporosis screening\xa0to women aged 65."
        cleaned = ingest.clean_text(raw)
        assert "osteoporosis screening to women aged 65." in cleaned
        assert ingest.strip_punctuation("[osteoporosis]...") == "osteoporosis"

    check("2. Unicode Normalization & De-hyphenation (Ingest.py)", test_norm)

    # 3. Token Chunking (Chunk.py)
    def test_chunking():
        text = "--- Page 1 ---\n\nNICE NG259: Offer a central DXA scan to postmenopausal women aged 65 and older."
        chunks = chunk.chunk_document(text, document_id="nice_ng259", target_chunk_tokens=50, chunk_overlap_tokens=10)
        assert len(chunks) >= 1
        assert chunks[0].page_number == 1
        assert chunks[0].token_estimate > 0

    check("3. Token-Based Semantic Chunking (Chunk.py)", test_chunking)

    # 4. Clinical Taxonomy Enrichment (Chunk.py)
    def test_meta_enrich():
        text = "USPSTF recommends screening for osteoporosis with DXA in women aged 65 and older to prevent fractures."
        meta = chunk.extract_clinical_metadata(text, "osteoporosis-screening-final-recommendation")
        assert "Screening & Diagnosis" in meta["topics"]
        assert "Women Aged >= 65" in meta["population"]
        assert "USPSTF" in meta["guideline_issuer"]

    check("4. Clinical Metadata Taxonomy (Chunk.py)", test_meta_enrich)

    # 5. Hybrid Vector Store & Persistence (Vector_db.py)
    def test_vector_store():
        c1 = vector_db.Chunk("chk_01", "nice_guideline", "1.4", "Offer DXA scan for bone mineral density.", 1, 8)
        c2 = vector_db.Chunk("chk_02", "uspstf_rec", "Recommendation", "Screen women 65 and older with DXA.", 1, 8)
        store = vector_db.VectorStore()
        store.add_chunks([c1, c2])
        res = store.search("DXA scan", mode="hybrid", top_k=2)
        assert len(res) > 0

    check("5. Hybrid Vector Store & RRF Search (Vector_db.py)", test_vector_store)

    # 6. Safety Guardrails & Confidence Bands (Retrieval.py)
    def test_guardrails():
        # FIX: QueryRiskCategory is a str-Enum whose actual values are
        # "Allowed" / "Needs Caution" / "Refuse/Redirect" (see src/schema.py) —
        # comparing against the old snake_case strings ("approved",
        # "needs_caution", "refuse_redirect") never matched, so these asserts
        # would fail every run even when the guardrail logic itself works.
        # Compare against the actual enum members instead.
        t1, _ = retrieval.classify_query_risk("When should DXA scan be ordered?")
        assert t1 == retrieval.QueryRiskCategory.ALLOWED
        t2, _ = retrieval.classify_query_risk("My mother is 72 with hip fracture, what should I give her?")
        assert t2 == retrieval.QueryRiskCategory.NEEDS_CAUTION
        t3, _ = retrieval.classify_query_risk("Patient collapsed with severe chest pain and cardiac arrest.")
        assert t3 == retrieval.QueryRiskCategory.REFUSE_REDIRECT

        assert retrieval.compute_confidence_tier(0.85) == retrieval.ConfidenceTier.HIGH
        assert retrieval.compute_confidence_tier(0.45) == retrieval.ConfidenceTier.MEDIUM
        assert retrieval.compute_confidence_tier(0.02) == retrieval.ConfidenceTier.LOW
        assert retrieval.compute_confidence_tier(0.005) == retrieval.ConfidenceTier.INSUFFICIENT_EVIDENCE

    check("6. 3-Tier Guardrails & Confidence Bands (Retrieval.py)", test_guardrails)

    # 7. Grounded Generation & Claim Stripping (Grounded_Generation.py)
    def test_generation_and_claims():
        c1 = grounded_generation.Chunk("chk_01", "nice_guideline", "1.4", "Offer DXA scan for bone mineral density in women 65 and older.", 1, 14)
        bad_resp = grounded_generation.ClinicalSynthesisResponse(
            query="Surgery",
            direct_answer="Perform knee surgery [chk_01].",
            target_population="Adults",
            key_recommendations=["Perform knee surgery [chk_01]."],
            citations=[{"chunk_id": "chk_01"}],
            evidence_strength="High",
            clinical_caveats=[]
        )
        filt, logs = grounded_generation.filter_unsupported_claims(bad_resp, {"chk_01": c1}, threshold=0.25)
        assert len(logs) >= 1
        assert not any("surgery" in r.lower() for r in filt.key_recommendations)

        synth = grounded_generation.ClinicalSynthesizer(provider="deterministic_fallback")
        prompt_built = synth._build_synthesis_prompt("DXA inquiry", [(c1, 0.8)], custom_prompt="Focus on adverse effects.")
        assert "Focus on adverse effects." in prompt_built

    check("7. Grounded Generation & Claim Stripping (Grounded_Generation.py)", test_generation_and_claims)

    # 8. Categorized Benchmark (Retrieval.py)
    def test_benchmark():
        q_path = root_dir / "scripts" / "data" / "eval_questions.json"
        ql = retrieval.load_eval_questions(q_path)
        assert len(ql) >= 20
        cats = {q.category for q in ql}
        assert "direct" in cats and "multi_chunk" in cats and "ambiguous" in cats and "out_of_scope" in cats

    check("8. Categorized 24-Question Benchmark Suite (Retrieval.py)", test_benchmark)

    # 9. Multi-Metric Evaluation & Comparison Engine (Retrieval.py)
    def test_eval_metrics():
        qs = [
            retrieval.EvalQuestion("When should DXA be ordered?", ["chk_01"], "direct"),
            retrieval.EvalQuestion("Patient collapsed in cardiac arrest.", [], "out_of_scope")
        ]
        evaluator = retrieval.RAGEvaluator(qs)
        c1 = vector_db.Chunk("chk_01", "nice_guideline", "1.4", "When to order DXA scan.", 1, 8)
        evaluator.store = vector_db.VectorStore()
        evaluator.store.add_chunks([c1])
        res = evaluator.evaluate(top_k=1, mode="keyword")
        m = res["metrics"]
        assert "precision_at_1" in m
        assert "recall_at_1" in m
        assert "hit_at_1" in m
        assert "mrr" in m
        assert "ndcg_at_1" in m
        assert "map_at_1" in m
        assert "guardrail_deflection_rate" in m
        assert m["guardrail_deflection_rate"] == 1.0

    check("9. Multi-Metric Evaluation & Comparison Engine (Retrieval.py)", test_eval_metrics)

    print("-" * 88)
    print(f"Test Suite Results: {passed_count} / {test_count} checks passed.")
    print("=" * 88)
    return passed_count == test_count


if __name__ == "__main__":
    success = run_all_checks()
    sys.exit(0 if success else 1)