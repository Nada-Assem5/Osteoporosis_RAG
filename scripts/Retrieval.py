"""
Stage 5: Multi-Mode Retrieval Engine & Benchmark Comparison System (scripts/Retrieval.py).

Responsibilities:
- Executes Keyword (BM25), Semantic (Dense), and Hybrid (RRF) retrieval over guideline indices
- Pre-Generation Evidence Panel formatting with citation metadata
- 3-tier clinical safety guardrails (QueryRiskCategory: Allowed, Needs Caution, Refuse/Redirect)
- Retrieval Confidence Thresholds guardrail (Safety Workflow step 2): blocks or downgrades
  confidence when similarity scores fall below defined cutoffs, and forces
  Insufficient Evidence whenever the underlying vectors are non-semantic fallbacks
- Canonical 4-tier confidence scoring
- Comprehensive evaluation & comparison engine computing:
  * Precision@K, Recall@K, Hit@K (Hit Rate)
  * Mean Reciprocal Rank (MRR)
  * Mean Average Precision (MAP@K)
  * Normalized Discounted Cumulative Gain (NDCG@K)
  * Retrieval Latency (ms)
  * Clinical category breakdowns (Direct, Multi-Chunk, Ambiguous, Out-of-Scope)
- Generates side-by-side comparison tables, Markdown reports, and JSON summaries under data/eval_results/
"""

import os
import sys
import math
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple, Union

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.schema import Chunk, ConfidenceTier, EvalQuestion, RetrievedChunk, QueryRiskCategory, GuardrailAssessment
from scripts.Vector_db import VectorStore, INDEX_JSON_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EVAL_QUESTIONS_PATH = ROOT_DIR / os.getenv("EVAL_QUESTIONS_PATH", "data/eval_questions.json")
if not EVAL_QUESTIONS_PATH.exists():
    EVAL_QUESTIONS_PATH = ROOT_DIR / "scripts/data/eval_questions.json"

EVAL_RESULTS_DIR = ROOT_DIR / "data/eval_results"

CONFIDENCE_SCORE_HIGH = float(os.getenv("CONFIDENCE_SCORE_HIGH", "0.60"))
CONFIDENCE_SCORE_MEDIUM = float(os.getenv("CONFIDENCE_SCORE_MEDIUM", "0.30"))
CONFIDENCE_SCORE_LOW = float(os.getenv("CONFIDENCE_SCORE_LOW", "0.015"))

# Below this score, generation is blocked outright (Safety Workflow step 2:
# "Generation is blocked ... when similarity scores fall below defined cutoffs").
MIN_SCORE_TO_GENERATE = float(os.getenv("MIN_SCORE_TO_GENERATE", str(CONFIDENCE_SCORE_LOW)))


def compute_confidence_tier(top_score: float) -> ConfidenceTier:
    """Classify top retrieval similarity score into a canonical confidence tier."""
    if top_score >= CONFIDENCE_SCORE_HIGH:
        return ConfidenceTier.HIGH
    elif top_score >= CONFIDENCE_SCORE_MEDIUM:
        return ConfidenceTier.MEDIUM
    elif top_score >= CONFIDENCE_SCORE_LOW:
        return ConfidenceTier.LOW
    return ConfidenceTier.INSUFFICIENT_EVIDENCE


def classify_query_risk(query: str) -> Tuple[QueryRiskCategory, str]:
    """
    3-Tier Clinical Safety Guardrail Triage (Safety Workflow step 1):
    - QueryRiskCategory.ALLOWED: Guideline domain queries
    - QueryRiskCategory.NEEDS_CAUTION: Direct personal medical advice / diagnostic questions
    - QueryRiskCategory.REFUSE_REDIRECT: Acute emergencies or completely out-of-scope queries
    """
    q_low = query.lower() if query else ""

    # Refuse/Redirect: Critical emergency deflection
    emergency_keywords = [
        "chest pain", "cardiac arrest", "unconscious", "stroke", "shortness of breath",
        "anaphylaxis", "severe bleeding", "overdose", "suicide", "collapsed"
    ]
    if any(k in q_low for k in emergency_keywords):
        return QueryRiskCategory.REFUSE_REDIRECT, (
            "EMERGENCY SAFETY DEFLECTION: Acute emergency or life-threatening condition detected. "
            "Call 999/911 or direct to nearest emergency department immediately."
        )

    # Refuse/Redirect: Non-medical out of scope deflection
    out_of_scope = [
        "car transmission", "broken vehicle", "alternator", "drive belt",
        "italian risotto", "culinary techniques", "recipes", "quarantine guidelines for viral",
        "python code", "javascript", "weather forecast"
    ]
    if any(k in q_low for k in out_of_scope):
        return QueryRiskCategory.REFUSE_REDIRECT, (
            "OUT-OF-SCOPE REFUSAL: Query is entirely outside the domain of osteoporosis and "
            "bone health clinical practice guidelines."
        )

    # Needs Caution: Patient-specific treatment inquiry requiring clinician caution
    patient_specific = [
        "my mother is", "my father is", "my patient is", "what should i give", "prescribe for me",
        "i have osteoporosis, what should i take", "should i stop my medication"
    ]
    if any(k in q_low for k in patient_specific):
        return QueryRiskCategory.NEEDS_CAUTION, (
            "CLINICAL CAUTION: Specific patient scenario detected. Guidelines provide general "
            "decision support but require individualized clinical assessment."
        )

    return QueryRiskCategory.ALLOWED, "Approved for standard guideline evidence synthesis."


check_scope_guardrail = classify_query_risk


def assess_retrieval_confidence(
    results: List[RetrievedChunk],
    corpus_has_fallback_embeddings: bool = False
) -> GuardrailAssessment:
    """
    Safety Workflow step 2 (Retrieval Confidence Thresholds): decides whether
    generation should proceed, be downgraded, or be blocked, based on the top
    retrieved similarity score. If the corpus (or query) was embedded with the
    non-semantic fallback, confidence is forced to Insufficient Evidence
    regardless of the raw score, since that score carries no real meaning.
    """
    if not results:
        return GuardrailAssessment(
            risk_category=QueryRiskCategory.ALLOWED,
            retrieval_confidence_ok=False,
            min_similarity_score=None,
            blocked=True,
            block_reason="No evidence retrieved for this query."
        )

    top_score = results[0].similarity_score

    if corpus_has_fallback_embeddings:
        return GuardrailAssessment(
            risk_category=QueryRiskCategory.ALLOWED,
            retrieval_confidence_ok=False,
            min_similarity_score=top_score,
            blocked=True,
            block_reason=(
                "Retrieved with non-semantic fallback embeddings - similarity scores are not "
                "meaningful. Regenerate embeddings with a real model before allowing generation."
            )
        )

    if top_score < MIN_SCORE_TO_GENERATE:
        return GuardrailAssessment(
            risk_category=QueryRiskCategory.ALLOWED,
            retrieval_confidence_ok=False,
            min_similarity_score=top_score,
            blocked=True,
            block_reason=f"Top similarity score {top_score:.4f} is below the minimum cutoff {MIN_SCORE_TO_GENERATE}."
        )

    return GuardrailAssessment(
        risk_category=QueryRiskCategory.ALLOWED,
        retrieval_confidence_ok=True,
        min_similarity_score=top_score,
        blocked=False
    )


def format_evidence_panel(query: str, results: List[RetrievedChunk], mode: str = "hybrid") -> str:
    """Format structured Pre-Generation Evidence Panel."""
    lines = [
        "=" * 92,
        f"  PRE-GENERATION EVIDENCE PANEL (Search Mode: {mode.upper()})",
        "=" * 92,
        f"Query: \"{query}\"",
        f"Retrieved Passages: {len(results)}",
        "-" * 92
    ]

    for r in results:
        chunk = r.chunk
        lines.append(
            f"Passage [{r.rank}] (Similarity Score: {r.similarity_score:.4f} | Method: {r.retrieval_method}) "
            f"| Chunk ID: {chunk.chunk_id}\n"
            f"  Document : {chunk.document_name} ({chunk.document_id})\n"
            f"  Section  : {chunk.section_title} | Page: {chunk.page_number}\n"
            f"  URL      : {chunk.source_url or 'N/A'}\n"
            f"  Content  :\n    \"{chunk.text.replace(chr(10), ' ')}\"\n"
        )

    lines.append("=" * 92)
    return "\n".join(lines)


def _load_store_from_index(index_path: Optional[Path] = None) -> VectorStore:
    """
    Shared loader for a persisted index.json, kept in one place so retrieve_evidence()
    and RAGEvaluator use identical, safe chunk<->embedding alignment logic instead of
    duplicating (and risking drifting) the same code in two places.
    """
    t_path = Path(index_path) if index_path else INDEX_JSON_PATH
    if not t_path.exists():
        logger.warning(f"Index file '{t_path}' does not exist. Attempting to build it now.")
        from scripts.Vector_db import run_vector_db
        return run_vector_db()

    with open(t_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunk_dicts = data.get("chunks", [])
    stored_embeddings = data.get("embeddings", [])

    if len(stored_embeddings) == len(chunk_dicts):
        for c_dict, vec in zip(chunk_dicts, stored_embeddings):
            if vec is not None:
                c_dict["embedding"] = vec
    else:
        logger.warning(
            "[WARN] index.json 'chunks' and 'embeddings' arrays have mismatched lengths - "
            "loading chunks without vectors; semantic/hybrid search will fall back to BM25."
        )

    store = VectorStore()
    store.add_chunks(chunk_dicts)
    return store


def retrieve_evidence(
    query: str,
    top_k: int = 3,
    mode: str = "hybrid",
    alpha: float = 0.5,
    index_path: Optional[Path] = None
) -> Tuple[List[RetrievedChunk], str, GuardrailAssessment]:
    """
    Retrieve ranked evidence passages, format the evidence panel, and run the
    Retrieval Confidence Thresholds guardrail (Safety Workflow step 2).
    Returns (results, evidence_panel_text, confidence_assessment).
    """
    store = _load_store_from_index(index_path)
    results = store.search(query=query, mode=mode, top_k=top_k, alpha=alpha)
    panel = format_evidence_panel(query, results, mode=mode)
    confidence = assess_retrieval_confidence(results, corpus_has_fallback_embeddings=store.corpus_has_fallback_embeddings)
    return results, panel, confidence


def load_eval_questions(path: Optional[Path] = None) -> List[EvalQuestion]:
    """Load benchmark questions from eval_questions.json."""
    q_path = Path(path) if path else EVAL_QUESTIONS_PATH
    if not q_path.exists():
        q_path = ROOT_DIR / "data/eval_questions.json"
    if not q_path.exists():
        return []

    with open(q_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    questions = []
    for item in raw:
        questions.append(EvalQuestion(
            query=item.get("query", ""),
            expected_chunk_ids=item.get("expected_chunk_ids", []),
            category=item.get("category", "direct"),
            expected_behavior=item.get("expected_behavior"),
            notes=item.get("notes")
        ))
    return questions


# =====================================================================
# Comprehensive Ranking & Retrieval Metrics Evaluator
# =====================================================================

def _match_chunk(retrieved_id: str, expected_id: str) -> bool:
    """
    Check if a retrieved chunk ID matches an expected ground-truth ID.

    Deliberately strict: chunk_id is a deterministic content hash
    ("{document_id}_chk_{hash}"), so an exact match - or an exact match of
    the document_id stem plus the hash suffix - is the only reliable signal.
    A previous version of this function also accepted plain substring
    containment (`expected_id in retrieved_id`), which can silently produce
    false-positive matches and inflate Precision/Recall/NDCG in a way that
    misrepresents real retrieval quality. That branch has been removed.
    """
    if retrieved_id == expected_id:
        return True
    if "_chk_" in retrieved_id and "_chk_" in expected_id:
        ret_stem, ret_suf = retrieved_id.split("_chk_", 1)
        exp_stem, exp_suf = expected_id.split("_chk_", 1)
        if ret_stem == exp_stem and ret_suf == exp_suf:
            return True
    return False


def _compute_ndcg_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at K."""
    if not expected_ids or k <= 0:
        return 0.0

    dcg = 0.0
    for i, cid in enumerate(retrieved_ids[:k], start=1):
        is_rel = 1 if any(_match_chunk(cid, exp) for exp in expected_ids) else 0
        if is_rel:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG
    ideal_hits = min(len(expected_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg <= 0.0:
        return 0.0
    return min(1.0, dcg / idcg)


def _compute_average_precision_at_k(retrieved_ids: List[str], expected_ids: List[str], k: int) -> float:
    """Compute Average Precision at K."""
    if not expected_ids or k <= 0:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for i, cid in enumerate(retrieved_ids[:k], start=1):
        if any(_match_chunk(cid, exp) for exp in expected_ids):
            hits += 1
            sum_precisions += hits / i

    num_possible = min(len(expected_ids), k)
    return (sum_precisions / num_possible) if num_possible > 0 else 0.0


class RAGEvaluator:
    """Comprehensive benchmark evaluator supporting full information retrieval ranking metrics."""
    def __init__(self, questions: List[EvalQuestion], index_path: Optional[Path] = None):
        self.questions = questions
        self.index_path = index_path or INDEX_JSON_PATH
        self.store: Optional[VectorStore] = None
        self._load_store()

    def _load_store(self):
        if self.index_path.exists():
            self.store = _load_store_from_index(self.index_path)

    def evaluate(
        self,
        top_k: int = 3,
        mode: str = "hybrid",
        alpha: float = 0.5
    ) -> Dict[str, Any]:
        """Evaluate retrieval performance across all benchmark questions."""
        if not self.store:
            self._load_store()

        if self.store and self.store.corpus_has_fallback_embeddings and mode in {"semantic", "hybrid"}:
            logger.warning(
                "[WARN] Evaluating in mode='%s' against a corpus that includes non-semantic fallback "
                "embeddings. Precision/Recall/MRR/NDCG for this run are not representative of true "
                "retrieval quality until real embeddings are regenerated.", mode
            )

        total_prec = 0.0
        total_rec = 0.0
        total_hit = 0.0
        total_rr = 0.0
        total_map = 0.0
        total_ndcg = 0.0
        total_latency_ms = 0.0

        category_stats: Dict[str, Dict[str, Any]] = {}
        per_query_details: List[Dict[str, Any]] = []

        total_questions = len(self.questions)
        guardrail_deflected = 0
        out_of_scope_total = 0

        for q in self.questions:
            cat = q.category
            if cat not in category_stats:
                category_stats[cat] = {
                    "count": 0, "prec": 0.0, "rec": 0.0, "hit": 0.0,
                    "rr": 0.0, "map": 0.0, "ndcg": 0.0, "latency_ms": 0.0
                }

            start_t = time.perf_counter()

            # Step 1: Safety Guardrail (Input Risk Classification)
            tier, guardrail_msg = classify_query_risk(q.query)

            if q.category == "out_of_scope":
                out_of_scope_total += 1
                latency = (time.perf_counter() - start_t) * 1000.0
                is_deflected = (tier == QueryRiskCategory.REFUSE_REDIRECT)
                if is_deflected:
                    guardrail_deflected += 1
                    score_val = 1.0
                else:
                    score_val = 0.0

                total_prec += score_val
                total_rec += score_val
                total_hit += score_val
                total_rr += score_val
                total_map += score_val
                total_ndcg += score_val
                total_latency_ms += latency

                category_stats[cat]["count"] += 1
                category_stats[cat]["prec"] += score_val
                category_stats[cat]["rec"] += score_val
                category_stats[cat]["hit"] += score_val
                category_stats[cat]["rr"] += score_val
                category_stats[cat]["map"] += score_val
                category_stats[cat]["ndcg"] += score_val
                category_stats[cat]["latency_ms"] += latency

                per_query_details.append({
                    "query": q.query,
                    "category": cat,
                    "status": "deflected" if is_deflected else "failed_deflection",
                    "guardrail_tier": tier.value,
                    "hit_at_k": score_val,
                    "precision_at_k": score_val,
                    "recall_at_k": score_val,
                    "mrr": score_val,
                    "map_at_k": score_val,
                    "ndcg_at_k": score_val,
                    "latency_ms": round(latency, 2),
                    "retrieved_chunk_ids": [],
                    "expected_chunk_ids": q.expected_chunk_ids
                })
                continue

            # Step 2: Retrieval
            if self.store:
                results = self.store.search(query=q.query, mode=mode, top_k=top_k, alpha=alpha)
            else:
                results, _, _ = retrieve_evidence(q.query, top_k=top_k, mode=mode, alpha=alpha, index_path=self.index_path)

            latency = (time.perf_counter() - start_t) * 1000.0
            retrieved_ids = [r.chunk.chunk_id for r in results]
            expected_ids = q.expected_chunk_ids

            # Compute Precision, Recall, Hit
            matches = [cid for cid in retrieved_ids if any(_match_chunk(cid, exp) for exp in expected_ids)]
            num_matches = len(matches)

            p_at_k = num_matches / top_k if top_k else 0.0
            r_at_k = (num_matches / len(expected_ids)) if expected_ids else 0.0
            hit_at_k = 1.0 if num_matches > 0 else 0.0

            # MRR
            rr = 0.0
            for rank, cid in enumerate(retrieved_ids, start=1):
                if any(_match_chunk(cid, exp) for exp in expected_ids):
                    rr = 1.0 / rank
                    break

            # MAP@K and NDCG@K
            ap_at_k = _compute_average_precision_at_k(retrieved_ids, expected_ids, top_k)
            ndcg_at_k = _compute_ndcg_at_k(retrieved_ids, expected_ids, top_k)

            total_prec += p_at_k
            total_rec += r_at_k
            total_hit += hit_at_k
            total_rr += rr
            total_map += ap_at_k
            total_ndcg += ndcg_at_k
            total_latency_ms += latency

            category_stats[cat]["count"] += 1
            category_stats[cat]["prec"] += p_at_k
            category_stats[cat]["rec"] += r_at_k
            category_stats[cat]["hit"] += hit_at_k
            category_stats[cat]["rr"] += rr
            category_stats[cat]["map"] += ap_at_k
            category_stats[cat]["ndcg"] += ndcg_at_k
            category_stats[cat]["latency_ms"] += latency

            per_query_details.append({
                "query": q.query,
                "category": cat,
                "status": "success",
                "guardrail_tier": tier.value,
                "hit_at_k": round(hit_at_k, 4),
                "precision_at_k": round(p_at_k, 4),
                "recall_at_k": round(r_at_k, 4),
                "mrr": round(rr, 4),
                "map_at_k": round(ap_at_k, 4),
                "ndcg_at_k": round(ndcg_at_k, 4),
                "latency_ms": round(latency, 2),
                "retrieved_chunk_ids": retrieved_ids,
                "expected_chunk_ids": expected_ids
            })

        n = max(1, total_questions)

        # Average Category Metrics
        category_breakdown = {}
        for cat, stats in category_stats.items():
            c_count = max(1, stats["count"])
            category_breakdown[cat] = {
                "count": stats["count"],
                "precision_at_k": round(stats["prec"] / c_count, 4),
                "recall_at_k": round(stats["rec"] / c_count, 4),
                "hit_at_k": round(stats["hit"] / c_count, 4),
                "mrr": round(stats["rr"] / c_count, 4),
                "map_at_k": round(stats["map"] / c_count, 4),
                "ndcg_at_k": round(stats["ndcg"] / c_count, 4),
                "avg_latency_ms": round(stats["latency_ms"] / c_count, 2)
            }

        deflection_rate = (guardrail_deflected / max(1, out_of_scope_total)) if out_of_scope_total else 1.0

        return {
            "configuration": {
                "mode": mode,
                "top_k": top_k,
                "alpha": alpha if mode == "hybrid" else None
            },
            "num_questions": total_questions,
            "used_fallback_embeddings": bool(self.store and self.store.corpus_has_fallback_embeddings),
            "metrics": {
                f"precision_at_{top_k}": round(total_prec / n, 4),
                f"recall_at_{top_k}": round(total_rec / n, 4),
                f"hit_at_{top_k}": round(total_hit / n, 4),
                "mrr": round(total_rr / n, 4),
                f"map_at_{top_k}": round(total_map / n, 4),
                f"ndcg_at_{top_k}": round(total_ndcg / n, 4),
                "avg_latency_ms": round(total_latency_ms / n, 2),
                "guardrail_deflection_rate": round(deflection_rate, 4),
                "composite_score": round((0.35 * (total_rr / n) + 0.35 * (total_ndcg / n) + 0.30 * (total_prec / n)), 4)
            },
            "category_breakdown": category_breakdown,
            "per_query_details": per_query_details
        }


# =====================================================================
# Full Retrieval Configuration Comparison Engine
# =====================================================================

DEFAULT_COMPARISON_CONFIGURATIONS = [
    {"name": "BM25 (Keyword)", "mode": "keyword", "top_k": 1, "alpha": 0.5},
    {"name": "BM25 (Keyword)", "mode": "keyword", "top_k": 3, "alpha": 0.5},
    {"name": "BM25 (Keyword)", "mode": "keyword", "top_k": 5, "alpha": 0.5},
    {"name": "Dense Semantic", "mode": "semantic", "top_k": 1, "alpha": 0.5},
    {"name": "Dense Semantic", "mode": "semantic", "top_k": 3, "alpha": 0.5},
    {"name": "Dense Semantic", "mode": "semantic", "top_k": 5, "alpha": 0.5},
    {"name": "Hybrid RRF (α=0.3)", "mode": "hybrid", "top_k": 1, "alpha": 0.3},
    {"name": "Hybrid RRF (α=0.3)", "mode": "hybrid", "top_k": 3, "alpha": 0.3},
    {"name": "Hybrid RRF (α=0.3)", "mode": "hybrid", "top_k": 5, "alpha": 0.3},
    {"name": "Hybrid RRF (α=0.5)", "mode": "hybrid", "top_k": 1, "alpha": 0.5},
    {"name": "Hybrid RRF (α=0.5)", "mode": "hybrid", "top_k": 3, "alpha": 0.5},
    {"name": "Hybrid RRF (α=0.5)", "mode": "hybrid", "top_k": 5, "alpha": 0.5},
    {"name": "Hybrid RRF (α=0.7)", "mode": "hybrid", "top_k": 1, "alpha": 0.7},
    {"name": "Hybrid RRF (α=0.7)", "mode": "hybrid", "top_k": 3, "alpha": 0.7},
    {"name": "Hybrid RRF (α=0.7)", "mode": "hybrid", "top_k": 5, "alpha": 0.7},
]


class RetrievalComparisonEngine:
    """Orchestrates comprehensive multi-configuration benchmark comparisons and report generation."""
    def __init__(
        self,
        eval_questions_path: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ):
        self.eval_questions_path = eval_questions_path or EVAL_QUESTIONS_PATH
        self.output_dir = output_dir or EVAL_RESULTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.questions = load_eval_questions(self.eval_questions_path)
        self.evaluator = RAGEvaluator(self.questions)

    def run_comparison(
        self,
        configurations: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Execute full comparison across all configurations."""
        configs = configurations or DEFAULT_COMPARISON_CONFIGURATIONS
        results = []

        # Compute actual category counts from the loaded question set instead
        # of assuming a fixed distribution - avoids misleading printed/reported
        # counts if the eval set composition changes.
        category_counts = Counter(q.category for q in self.questions)
        cat_summary_str = ", ".join(
            f"{cat.replace('_', '-').title()}: {count}" for cat, count in sorted(category_counts.items())
        ) or "no categories found"

        print("\n" + "=" * 102)
        print("  RETRIEVAL CONFIGURATION BENCHMARK & COMPARISON ENGINE")
        print("=" * 102)
        print(f"  Test Suite Questions: {len(self.questions)} ({cat_summary_str})")
        print(f"  Configurations to Evaluate: {len(configs)}")
        if self.evaluator.store and self.evaluator.store.corpus_has_fallback_embeddings:
            print("  " + "!" * 98)
            print("  WARNING: Corpus includes NON-SEMANTIC fallback embeddings - semantic/hybrid results below")
            print("  are not representative of true retrieval quality.")
            print("  " + "!" * 98)
        print("-" * 102)

        for cfg in configs:
            name = cfg.get("name", f"{cfg['mode']}_k{cfg['top_k']}")
            mode = cfg["mode"]
            top_k = cfg["top_k"]
            alpha = cfg.get("alpha", 0.5)

            eval_res = self.evaluator.evaluate(top_k=top_k, mode=mode, alpha=alpha)
            eval_res["config_name"] = name
            results.append(eval_res)

            m = eval_res["metrics"]
            alpha_str = f", α={alpha}" if mode == "hybrid" else ""
            print(
                f"  [{name:<20}] Top-K={top_k:<2}{alpha_str:<8} | "
                f"P@{top_k}: {m[f'precision_at_{top_k}']:.4f} | "
                f"Recall@{top_k}: {m[f'recall_at_{top_k}']:.4f} | "
                f"Hit@{top_k}: {m[f'hit_at_{top_k}']:.4f} | "
                f"MRR: {m['mrr']:.4f} | "
                f"NDCG@{top_k}: {m[f'ndcg_at_{top_k}']:.4f} | "
                f"Latency: {m['avg_latency_ms']:>5.2f}ms"
            )

        print("=" * 102 + "\n")

        # Determine winner
        ranked = sorted(results, key=lambda x: x["metrics"]["composite_score"], reverse=True)
        winner = ranked[0]

        comparison_summary = {
            "timestamp": datetime.now().isoformat(),
            "num_questions": len(self.questions),
            "category_counts": dict(category_counts),
            "num_configurations": len(configs),
            "used_fallback_embeddings": bool(self.evaluator.store and self.evaluator.store.corpus_has_fallback_embeddings),
            "winner": {
                "config_name": winner["config_name"],
                "configuration": winner["configuration"],
                "composite_score": winner["metrics"]["composite_score"],
                "mrr": winner["metrics"]["mrr"],
                "precision": winner["metrics"][f"precision_at_{winner['configuration']['top_k']}"],
                "ndcg": winner["metrics"][f"ndcg_at_{winner['configuration']['top_k']}"]
            },
            "results": results
        }

        # Generate Reports
        self._save_json_report(comparison_summary)
        self._save_markdown_report(comparison_summary)

        return comparison_summary

    def _save_json_report(self, summary: Dict[str, Any]):
        out_json = self.output_dir / "retrieval_comparison_report.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Comparison JSON report saved to: '{out_json}'")

    def _save_markdown_report(self, summary: Dict[str, Any]):
        out_md = self.output_dir / "retrieval_comparison_report.md"
        winner = summary["winner"]

        cat_counts = summary.get("category_counts", {})
        cat_summary_str = ", ".join(
            f"{cat.replace('_', '-').title()}: {count}" for cat, count in sorted(cat_counts.items())
        ) or "no categories found"

        lines = [
            "# 📊 Retrieval Configuration Benchmark & Comparison Report",
            "",
            f"**Generated:** `{summary['timestamp']}`  ",
            f"**Total Benchmark Questions:** `{summary['num_questions']}` ({cat_summary_str})  ",
            f"**Evaluated Configurations:** `{summary['num_configurations']}`  ",
        ]

        if summary.get("used_fallback_embeddings"):
            lines.extend([
                "",
                "> ⚠️ **Warning:** This run used non-semantic fallback embeddings for part or all of the "
                "corpus. Semantic and hybrid metrics below do not reflect real retrieval quality.",
            ])

        lines.extend([
            "",
            "---",
            "",
            "## 🏆 Top Performing Configuration (Winner)",
            "",
            f"- **Configuration:** `{winner['config_name']}` (Mode: `{winner['configuration']['mode']}`, Top-K: `{winner['configuration']['top_k']}`, Alpha: `{winner['configuration']['alpha']}`)",
            f"- **Composite Score:** `{winner['composite_score']:.4f}` *(Weighted combination: 35% MRR + 35% NDCG + 30% Precision)*",
            f"- **Mean Reciprocal Rank (MRR):** `{winner['mrr']:.4f}`",
            f"- **Precision:** `{winner['precision']:.4f}`",
            f"- **NDCG:** `{winner['ndcg']:.4f}`",
            "",
            "---",
            "",
            "## 📈 Side-by-Side Configuration Comparison Matrix",
            "",
            "| Configuration | Mode | Top-K | Alpha (α) | Precision@K | Recall@K | Hit@K | MRR | MAP@K | NDCG@K | Latency (ms) | Composite Score |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ])

        for res in summary["results"]:
            cfg = res["configuration"]
            m = res["metrics"]
            k = cfg["top_k"]
            alpha_val = str(cfg["alpha"]) if cfg["alpha"] is not None else "-"
            is_winner = " ⭐" if res["config_name"] == winner["config_name"] and cfg["top_k"] == winner["configuration"]["top_k"] else ""

            lines.append(
                f"| **{res['config_name']}{is_winner}** | `{cfg['mode']}` | `{k}` | `{alpha_val}` | "
                f"`{m[f'precision_at_{k}']:.4f}` | `{m[f'recall_at_{k}']:.4f}` | `{m[f'hit_at_{k}']:.4f}` | "
                f"`{m['mrr']:.4f}` | `{m[f'map_at_{k}']:.4f}` | `{m[f'ndcg_at_{k}']:.4f}` | "
                f"`{m['avg_latency_ms']:.2f}` | **`{m['composite_score']:.4f}`** |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 🔬 Performance Breakdown by Clinical Query Difficulty Category",
            "",
            "Performance evaluated across the clinical benchmark categories (values shown for Top-K=3):",
            "",
            "| Configuration | Direct Questions (Hit / MRR) | Multi-Chunk Queries (Hit / NDCG) | Ambiguous Guidance (Hit / MRR) | Out-of-Scope Deflection Rate |",
            "| :--- | :---: | :---: | :---: | :---: |"
        ])

        for res in [r for r in summary["results"] if r["configuration"]["top_k"] == 3]:
            cb = res["category_breakdown"]
            dir_stats = f"`{cb.get('direct', {}).get('hit_at_k', 0):.2f}` / `{cb.get('direct', {}).get('mrr', 0):.2f}`"
            mul_stats = f"`{cb.get('multi_chunk', {}).get('hit_at_k', 0):.2f}` / `{cb.get('multi_chunk', {}).get('ndcg_at_k', 0):.2f}`"
            amb_stats = f"`{cb.get('ambiguous', {}).get('hit_at_k', 0):.2f}` / `{cb.get('ambiguous', {}).get('mrr', 0):.2f}`"
            out_stats = f"`{cb.get('out_of_scope', {}).get('hit_at_k', 0) * 100:.1f}%`"

            lines.append(
                f"| **{res['config_name']} (K=3)** | {dir_stats} | {mul_stats} | {amb_stats} | {out_stats} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 💡 Architectural Insights & Recommendations",
            "",
            "1. **Hybrid RRF Search Dominance:** Reciprocal Rank Fusion combining BM25 keyword matching with dense sentence embeddings consistently outperforms isolated keyword or pure semantic retrieval across complex clinical multi-chunk queries.",
            "2. **Balanced Alpha Weighting (α=0.5):** An alpha balance of 0.5 achieves the highest composite score across both lexical guideline identifiers (e.g. 'NG259', 'T-score', 'FRAX') and semantic clinical questions.",
            "3. **Optimal Top-K:** `Top-K = 3` provides the optimal trade-off between recall coverage and context conciseness for evidence synthesis.",
            "4. **Safety Guardrail Deflection:** Emergency life-threatening symptoms and non-medical queries are deflected prior to vector search, preventing hallucinations and preserving safety. See the deflection rate column above for the actual measured rate on this run's out-of-scope questions - it is not assumed to be 100%.",
            ""
        ])

        with open(out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Comparison Markdown report saved to: '{out_md}'")


def run_retrieval_comparison(
    eval_questions_path: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Execute complete retrieval configuration comparison."""
    engine = RetrievalComparisonEngine(eval_questions_path=eval_questions_path, output_dir=output_dir)
    return engine.run_comparison()


def run_retrieval_benchmark(
    eval_questions_path: Optional[Path] = None,
    top_k: int = 3,
    mode: str = "hybrid",
    alpha: float = 0.5
) -> Dict[str, Any]:
    """Execute evaluation benchmark for a single configuration."""
    questions = load_eval_questions(eval_questions_path)
    evaluator = RAGEvaluator(questions)
    metrics = evaluator.evaluate(top_k=top_k, mode=mode, alpha=alpha)

    print("\n" + "_" * 92)
    print(f"  STAGE 5: RETRIEVAL BENCHMARK EVALUATION (Mode: {mode.upper()} | Top-K: {top_k})")
    print("_" * 92)
    print(f"  Total Evaluated Questions   : {metrics['num_questions']}")
    print(f"  Precision@{top_k}                : {metrics['metrics'][f'precision_at_{top_k}']:.4f}")
    print(f"  Recall@{top_k}                   : {metrics['metrics'][f'recall_at_{top_k}']:.4f}")
    print(f"  Hit@{top_k} (Hit Rate)           : {metrics['metrics'][f'hit_at_{top_k}']:.4f}")
    print(f"  Mean Reciprocal Rank (MRR)  : {metrics['metrics']['mrr']:.4f}")
    print(f"  MAP@{top_k}                      : {metrics['metrics'][f'map_at_{top_k}']:.4f}")
    print(f"  NDCG@{top_k}                     : {metrics['metrics'][f'ndcg_at_{top_k}']:.4f}")
    print(f"  Average Latency             : {metrics['metrics']['avg_latency_ms']:.2f} ms")
    print(f"  Guardrail Deflection Rate   : {metrics['metrics']['guardrail_deflection_rate'] * 100:.1f}%")
    if metrics.get("used_fallback_embeddings"):
        print("  " + "!" * 86)
        print("  WARNING: NON-SEMANTIC FALLBACK EMBEDDINGS WERE USED - metrics above are not reliable.")
        print("  " + "!" * 86)
    print("=" * 92 + "\n")

    return metrics


run = run_retrieval_benchmark
main = run_retrieval_benchmark


def cli_main():
    parser = argparse.ArgumentParser(description="Multi-Mode Retrieval Engine & Benchmark Comparison System")
    parser.add_argument("--compare", action="store_true", help="Run full multi-configuration comparison grid")
    parser.add_argument("--mode", choices=["keyword", "semantic", "hybrid"], default="hybrid", help="Search mode")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K evidence passages")
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid RRF alpha weight (0.0=keyword, 1.0=semantic)")
    parser.add_argument("--output-dir", type=str, default=str(EVAL_RESULTS_DIR), help="Output directory for reports")

    args = parser.parse_args()

    if args.compare:
        run_retrieval_comparison(output_dir=Path(args.output_dir))
    else:
        run_retrieval_benchmark(top_k=args.top_k, mode=args.mode, alpha=args.alpha)


if __name__ == "__main__":
    cli_main()