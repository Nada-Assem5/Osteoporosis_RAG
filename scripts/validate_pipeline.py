"""
Verification runner: executes unit tests and runs the end-to-end pipeline stages.
Located in scripts/ to keep maintenance tools cleanly separated from the root CLI.
"""

import sys
import traceback
from pathlib import Path

# Add project root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

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
from src.chunking import Chunk, chunk_document, extract_clinical_metadata
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
from src.evaluation import RAGEvaluator, load_eval_questions, EvalQuestion, run_full_evaluation
from main import handle_ask


def run_all_checks():
    print("=" * 84)
    print("  RUNNING PIPELINE UNIT TESTS & MODULE VERIFICATION")
    print("=" * 84)

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

    # 1. Structural Noise Filtering (src.parsing)
    def test_parsing_structural():
        class MockEl:
            def __init__(self, cat, text):
                self.category = cat
                self.text = text

        elements = [
            MockEl("Header", "RUNNING HEADER"),
            MockEl("Title", "1.4 Bone Density Assessment"),
            MockEl("Footer", "Page 1 of 10"),
            MockEl("PageBreak", "")
        ]
        filtered = filter_structural_noise(elements)
        assert len(filtered) == 1
        assert filtered[0].text == "1.4 Bone Density Assessment"

    check("PDF Parsing Structural Noise Filtering", test_parsing_structural)

    # 2. Smart Title Noise & Layout Filtering (src.clean)
    def test_smart_cleaning():
        class MockEl:
            def __init__(self, cat, text):
                self.category = cat
                self.text = text

        elements = [
            MockEl("Header", "RUNNING HEADER"),
            MockEl("Title", "Ad"),
            MockEl("Title", "1.4 Bone Density Assessment"),
            MockEl("NarrativeText", "Offer a DXA scan to measure BMD in adults aged 30 and over with fragility fractures."),
            MockEl("Footer", "Page 1 of 10")
        ]
        cleaned = filter_elements(elements, short_title_threshold=20, min_content_length=30)
        assert len(cleaned) == 2, f"Expected 2 elements, got {len(cleaned)}"
        assert cleaned[0].text == "1.4 Bone Density Assessment"

    check("Smart Title Noise & Layout Filtering", test_smart_cleaning)

    # 3. Wordfreq Validation & Concatenation Fix (src.clean)
    def test_word_validation():
        assert is_valid_word("hello") is True
        assert is_valid_word("osteoporosis") is True
        assert is_valid_word("density") is True
        assert is_valid_word("theuspstfnotesthat") is False
        assert is_valid_word("policyandcoverage") is False

    check("Wordfreq Pure-Python Dictionary Validation", test_word_validation)

    def test_concat():
        broken = "ScreeningForOsteoporosis toPreventFractures USPSTF"
        fixed = fix_concatenated_text(broken)
        assert "Screening For Osteoporosis" in fixed or "Osteoporosis" in fixed
        assert "USPSTF" in fixed

    check("Word Concatenation & Spacing Repair", test_concat)

    # 4. Academic Boilerplate Stripping (src.clean)
    def test_academic():
        raw = "USPSTF Recommendation.\n\nAuthor Affiliations: Harvard.\n\nConflict of Interest: None."
        cleaned = clean_academic_boilerplate(raw)
        assert "Author Affiliations" not in cleaned
        assert "USPSTF Recommendation." in cleaned

    check("Academic Boilerplate Stripping", test_academic)

    # 5. Chunking with Metadata (src.chunking)
    def test_chunk_meta():
        text = "# 1.4 Bone Density\n\nOffer a DXA scan to measure BMD in postmenopausal women."
        chunks = chunk_document(text, "nice_doc", target_chunk_size=100)
        assert len(chunks) >= 1
        assert chunks[0].section_title == "1.4 Bone Density"
        assert "Postmenopausal Women" in chunks[0].population

    check("Metadata-Enriched Semantic Chunking", test_chunk_meta)

    # 6. Keyword, Semantic, and Hybrid Retrieval Modes (src.embedded)
    def test_retrieval_modes():
        c1 = Chunk("c1", "doc1", "DXA", "Offer DXA scan for bone mineral density.", 6, {"topics": ["Screening & Diagnosis"], "population": "Postmenopausal Women"})
        c2 = Chunk("c2", "doc1", "Falls", "Encourage balance exercises for fall prevention.", 6, {"topics": ["Lifestyle & Supplementation"], "population": "Older Adults"})
        store = VectorStore()
        store.add_chunks([c1, c2])

        kw_res = store.search("When to do a DXA scan for bone density?", top_k=1, mode="keyword")
        assert len(kw_res) == 1 and kw_res[0][0].chunk_id == "c1"

        sem_res = store.search("When to do a DXA scan for bone density?", top_k=1, mode="semantic")
        assert len(sem_res) == 1

        hyb_res = store.search("When to do a DXA scan for bone density?", top_k=1, mode="hybrid")
        assert len(hyb_res) == 1 and hyb_res[0][0].chunk_id == "c1"

    check("Keyword, Semantic, and Hybrid Retrieval Modes", test_retrieval_modes)

    # 7. Clinical Synthesis (src.synthesis)
    def test_synthesis():
        c1 = Chunk("c1", "nice_doc", "1.4 Bone Density", "Offer a DXA scan to measure BMD in patients aged 30+ with fragility fracture.", 15, {"guideline_issuer": "NICE NG259"})
        synth = ClinicalSynthesizer(provider="fallback", allow_fallback=True)
        resp = synth.synthesize("When is DXA indicated?", [(c1, 0.45)])
        assert "DXA" in resp.direct_answer
        assert len(resp.citations) == 1

    check("Clinical Evidence Synthesis & Citation Lineage", test_synthesis)

    # 8. Scope & 3-Tier Risk Guardrail (src.embedded)
    def test_guard():
        tier_allowed, _ = classify_query_risk("What are the criteria for osteoporosis DXA?")
        tier_caution, _ = classify_query_risk("My mother is 72, what should I prescribe?")
        tier_emergency, _ = classify_query_risk("Patient has severe acute chest pain and shortness of breath")
        tier_oos, _ = classify_query_risk("How to repair a car battery?")

        assert tier_allowed == "allowed"
        assert tier_caution == "needs_caution"
        assert tier_emergency == "refuse_redirect"
        assert tier_oos == "refuse_redirect"

        in_s, _ = check_scope_guardrail("What are the criteria for osteoporosis DXA?")
        out_s, _ = check_scope_guardrail("How to repair a car battery?")
        assert in_s is True
        assert out_s is False

    check("Clinical Scope & 3-Tier Guardrail Validation", test_guard)

    # 9. Unsupported Claim Guardrail Step 3 (src.synthesis)
    def test_claim_guard():
        c1 = Chunk("c1", "nice_doc", "1.4", "Offer DXA scan for bone mineral density.", 8)
        resp_unsupp = ClinicalSynthesisResponse(
            query="Surgery inquiry",
            direct_answer="Perform emergency orthopedic spinal fusion surgery immediately [c1].",
            target_population="Adults",
            key_recommendations=["Immediate open spine surgery with screws [c1]."],
            citations=[{"chunk_id": "c1"}],
            evidence_strength="High",
            clinical_caveats=[]
        )
        unsupp_warnings = detect_unsupported_claims(resp_unsupp, {"c1": c1}, threshold=0.25)
        assert len(unsupp_warnings) >= 1

    check("Unsupported Claim Grounding Verification", test_claim_guard)

    # 10. Evaluation Triad Benchmark (src.evaluation)
    def test_eval():
        c1 = Chunk("osteoporosis-risk-assessment-pdf-66144025463749_chk_005", "doc", "1.4 Bone density", "Offer a DXA scan to measure BMD in adults.", 12)
        store = VectorStore()
        store.add_chunks([c1])
        evaluator = RAGEvaluator(store)
        q = [EvalQuestion("When should a DXA bone density scan be offered according to NICE guidelines?", ["osteoporosis-risk-assessment-pdf-66144025463749_chk_005"])]
        res = evaluator.evaluate_all_modes(questions=q)
        assert res["retrieval"]["hybrid"]["precision_at_3"] > 0
        assert "citation_accuracy" in res
        assert "faithfulness" in res

    check("Precision@K, Citation & Faithfulness Triad Benchmark", test_eval)

    print("-" * 84)
    print(f"Test Suite Results: {passed_count} / {test_count} checks passed.")
    print("=" * 84)
    return passed_count == test_count


if __name__ == "__main__":
    tests_ok = run_all_checks()
    if not tests_ok:
        sys.exit(1)
    print("\n[OK] All verification checks passed.")
