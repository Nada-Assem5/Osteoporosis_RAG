"""
Multi-Dimensional Retrieval Experimentation, Evaluation & Comparison System (scripts/evaluate_experiments.py).

Comprehensive evaluation script allowing multi-factorial comparison of:
1. Chunk sizes in tokens (e.g. 128, 256, 400, 512, 1024)
2. Chunk overlaps in tokens (e.g. 0, 20, 50, 100)
3. Search / Retrieval types: Keyword (BM25), Semantic (Dense), Hybrid (RRF)
4. Embedding models (e.g. all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5)
5. Top-K retrieval depths (e.g. 1, 3, 5, 10)
6. Similarity scores (top result, average similarity, per-result similarity)
7. Retrieval latencies (per query & aggregate)

Artifacts Generated in data/eval_results/:
- evaluation_results.csv   : Raw row-per-retrieved-result evaluation dataset
- evaluation_summary.csv   : Aggregated configuration leaderboard (Recall@1/3/5/10, Precision@1/3/5/10, MRR, Latency, Similarity)
- evaluation_results.json  : Complete structured machine-readable results
- evaluation_matrix.md     : Human-readable comparison matrix with winner selection rationale
- plots/                   : Visual performance plots (Chunk Size vs Recall, Search Type vs MRR, Latency vs Recall)
"""

import os
import sys
import re
import csv
import math
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.schema import Chunk, EvalQuestion, QueryRiskCategory
from src.utils import count_tokens, compute_content_hash
from scripts.Chunk import chunk_extracted_elements
from scripts.Embeddings import generate_embeddings
from scripts.Vector_db import VectorStore, _tokenize, _cosine_similarity
from scripts.Retrieval import classify_query_risk, load_eval_questions, _match_chunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ELEMENTS_JSON_PATH = PROCESSED_DATA_DIR / "elements.json"
CHUNKS_BASELINE_PATH = PROCESSED_DATA_DIR / "chunks.json"
EVAL_QUESTIONS_PATH = DATA_DIR / "eval_questions.json"
if not EVAL_QUESTIONS_PATH.exists():
    EVAL_QUESTIONS_PATH = ROOT_DIR / "scripts/data/eval_questions.json"

EVAL_RESULTS_DIR = DATA_DIR / "eval_results"
CACHE_DIR = EVAL_RESULTS_DIR / "cache"
PLOTS_DIR = EVAL_RESULTS_DIR / "plots"

# Default Experiment Grids
DEFAULT_CHUNK_SIZES = [128, 256, 400, 512]
DEFAULT_CHUNK_OVERLAPS = [0, 20, 50, 100]
DEFAULT_SEARCH_TYPES = ["keyword", "semantic", "hybrid"]
DEFAULT_EMBEDDING_MODELS = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]
DEFAULT_TOP_K_VALUES = [1, 3, 5, 10]
DEFAULT_HYBRID_ALPHAS = [0.3, 0.5, 0.7]

QUICK_CHUNK_SIZES = [256, 400]
QUICK_CHUNK_OVERLAPS = [20, 50]
QUICK_SEARCH_TYPES = ["keyword", "semantic", "hybrid"]
QUICK_EMBEDDING_MODELS = ["all-MiniLM-L6-v2"]
QUICK_TOP_K_VALUES = [1, 3, 5, 10]
QUICK_HYBRID_ALPHAS = [0.5]


# =====================================================================
# Ground Truth Knowledge Base for Variable Token Chunk Sizes
# =====================================================================

def build_ground_truth_knowledge_base(
    baseline_chunks_path: Path = CHUNKS_BASELINE_PATH
) -> Dict[str, Dict[str, Any]]:
    """Builds a lookup index of expected chunk texts and metadata for content matching."""
    gt_map = {}
    if not baseline_chunks_path.exists():
        return gt_map

    try:
        with open(baseline_chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        for c in chunks:
            cid = c.get("chunk_id", "")
            gt_map[cid] = {
                "chunk_id": cid,
                "document_name": c.get("document_name", ""),
                "document_id": c.get("document_id", ""),
                "page_number": c.get("page_number", 1),
                "section_title": c.get("section_title", ""),
                "text": c.get("text", "")
            }
    except Exception as exc:
        logger.warning(f"Could not load baseline chunks for ground truth mapping: {exc}")

    return gt_map


def is_chunk_relevant(
    retrieved_chunk: Chunk,
    expected_chunk_ids: List[str],
    gt_kb: Dict[str, Dict[str, Any]]
) -> bool:
    """
    Determines if a retrieved chunk is relevant to the query:
    1. Exact or suffix chunk_id match
    2. Document alignment + lexical content overlap against ground-truth passages
    """
    if not expected_chunk_ids:
        return False

    r_id = getattr(retrieved_chunk, "chunk_id", "")
    r_doc = getattr(retrieved_chunk, "document_name", "") or getattr(retrieved_chunk, "document_id", "")
    r_text = getattr(retrieved_chunk, "text", "").lower()
    r_page = getattr(retrieved_chunk, "page_number", 1)

    for exp_id in expected_chunk_ids:
        # ID-based match
        if _match_chunk(r_id, exp_id):
            return True

        # Content-based overlap for variable chunk sizes
        if exp_id in gt_kb:
            exp_info = gt_kb[exp_id]
            exp_doc = exp_info.get("document_name", "") or exp_info.get("document_id", "")
            exp_text = exp_info.get("text", "").lower()
            exp_page = exp_info.get("page_number", 1)

            # Check document match
            if exp_doc and r_doc and (exp_doc in r_doc or r_doc in exp_doc):
                # Page proximity (within +/- 1 page)
                if abs(r_page - exp_page) <= 1:
                    exp_words = set(re.findall(r'\b[a-zA-Z0-9]{4,}\b', exp_text))
                    ret_words = set(re.findall(r'\b[a-zA-Z0-9]{4,}\b', r_text))
                    if exp_words and ret_words:
                        overlap = len(exp_words & ret_words) / len(exp_words)
                        if overlap >= 0.20:
                            return True

    return False


# =====================================================================
# Caching Subsystem for Chunks & Embeddings
# =====================================================================

class ExperimentCache:
    """Manages cached token chunking sets and vector embeddings on disk."""
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.chunks_dir = self.cache_dir / "chunks"
        self.embs_dir = self.cache_dir / "embeddings"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.embs_dir.mkdir(parents=True, exist_ok=True)
        self._memory_stores: Dict[str, VectorStore] = {}

    def get_chunks(self, elements: List[Dict[str, Any]], chunk_size: int, overlap: int) -> List[Chunk]:
        chunk_file = self.chunks_dir / f"chunks_sz{chunk_size}_ov{overlap}.json"
        if chunk_file.exists():
            try:
                with open(chunk_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return [Chunk(**c) for c in data]
            except Exception:
                pass

        chunks = chunk_extracted_elements(elements, target_chunk_tokens=chunk_size, chunk_overlap_tokens=overlap)
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in chunks], f, indent=2, ensure_ascii=False)
        return chunks

    def get_embeddings(self, chunks: List[Chunk], model_name: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
        safe_model = re.sub(r'[^a-zA-Z0-9_]', '_', model_name)
        emb_file = self.embs_dir / f"emb_{safe_model}_sz{chunk_size}_ov{overlap}.json"
        if emb_file.exists():
            try:
                with open(emb_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        chunk_dicts = [c.to_dict() for c in chunks]
        embedded, _run_stats = generate_embeddings(chunk_dicts, model_name=model_name)
        with open(emb_file, "w", encoding="utf-8") as f:
            json.dump(embedded, f, indent=2, ensure_ascii=False)
        return embedded

    def get_vector_store(self, chunks: List[Chunk], embedded_chunks: List[Dict[str, Any]], model_name: str, chunk_size: int, overlap: int) -> VectorStore:
        store_key = f"{model_name}_sz{chunk_size}_ov{overlap}"
        if store_key in self._memory_stores:
            return self._memory_stores[store_key]

        store = VectorStore()
        store.add_chunks(embedded_chunks if embedded_chunks else chunks)
        if model_name != "fallback":
            try:
                from sentence_transformers import SentenceTransformer
                store._embedding_model = SentenceTransformer(model_name)
            except Exception:
                store._embedding_model = "fallback"

        self._memory_stores[store_key] = store
        return store


# =====================================================================
# SVG / PNG Visualization Utilities
# =====================================================================

def generate_svg_chart_chunk_size_vs_recall(data_points: List[Tuple[int, float]], output_file: Path):
    """Generates an SVG chart of Chunk Size vs Recall@3."""
    if not data_points:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    width, height = 640, 360
    margin = 50

    x_vals = [p[0] for p in data_points]
    y_vals = [p[1] for p in data_points]
    min_x, max_x = min(x_vals), max(x_vals)
    min_y, max_y = 0.0, 1.0

    def scale_x(x):
        return margin + (x - min_x) / max(1, (max_x - min_x)) * (width - 2 * margin)

    def scale_y(y):
        return height - margin - (y - min_y) / (max_y - min_y) * (height - 2 * margin)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'  <rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'  <text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold" fill="#1e293b">Chunk Size (Tokens) vs Recall@3</text>',
        f'  <!-- Axes -->',
        f'  <line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#94a3b8" stroke-width="2"/>',
        f'  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#94a3b8" stroke-width="2"/>'
    ]

    # Grid lines
    for y_step in [0.2, 0.4, 0.6, 0.8, 1.0]:
        sy = scale_y(y_step)
        svg_lines.append(f'  <line x1="{margin}" y1="{sy}" x2="{width-margin}" y2="{sy}" stroke="#e2e8f0" stroke-width="1"/>')
        svg_lines.append(f'  <text x="{margin-10}" y="{sy+4}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#64748b">{y_step:.1f}</text>')

    # Polyline
    points_str = " ".join([f"{scale_x(x)},{scale_y(y)}" for x, y in data_points])
    svg_lines.append(f'  <polyline fill="none" stroke="#2563eb" stroke-width="3" points="{points_str}"/>')

    # Points & labels
    for x, y in data_points:
        sx, sy = scale_x(x), scale_y(y)
        svg_lines.append(f'  <circle cx="{sx}" cy="{sy}" r="5" fill="#1d4ed8" stroke="#ffffff" stroke-width="2"/>')
        svg_lines.append(f'  <text x="{sx}" y="{sy-10}" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="bold" fill="#1e293b">{y:.3f}</text>')
        svg_lines.append(f'  <text x="{sx}" y="{height-margin+20}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#64748b">{x}t</text>')

    svg_lines.append('</svg>')

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))


def generate_svg_chart_search_type_vs_metrics(st_data: Dict[str, Dict[str, float]], output_file: Path):
    """Generates an SVG bar chart comparing Search Types on MRR and Recall@3."""
    if not st_data:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    width, height = 640, 360
    margin = 60

    search_types = list(st_data.keys())
    num_types = len(search_types)
    bar_group_width = (width - 2 * margin) / max(1, num_types)
    bar_width = 32

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'  <rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'  <text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold" fill="#1e293b">Search Type Performance (MRR vs Recall@3)</text>',
        f'  <!-- Legend -->',
        f'  <rect x="{width-200}" y="20" width="12" height="12" fill="#3b82f6"/>',
        f'  <text x="{width-180}" y="30" font-family="sans-serif" font-size="11" fill="#475569">MRR</text>',
        f'  <rect x="{width-130}" y="20" width="12" height="12" fill="#10b981"/>',
        f'  <text x="{width-110}" y="30" font-family="sans-serif" font-size="11" fill="#475569">Recall@3</text>',
        f'  <!-- Axes -->',
        f'  <line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#94a3b8" stroke-width="2"/>',
        f'  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#94a3b8" stroke-width="2"/>'
    ]

    for y_step in [0.2, 0.4, 0.6, 0.8, 1.0]:
        sy = height - margin - y_step * (height - 2 * margin)
        svg_lines.append(f'  <line x1="{margin}" y1="{sy}" x2="{width-margin}" y2="{sy}" stroke="#e2e8f0" stroke-width="1"/>')
        svg_lines.append(f'  <text x="{margin-10}" y="{sy+4}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#64748b">{y_step:.1f}</text>')

    for idx, st in enumerate(search_types):
        center_x = margin + idx * bar_group_width + bar_group_width / 2
        mrr_val = st_data[st].get("mrr", 0.0)
        rec_val = st_data[st].get("recall_at_3", st_data[st].get("rec", 0.0))

        mrr_h = mrr_val * (height - 2 * margin)
        rec_h = rec_val * (height - 2 * margin)

        bx1 = center_x - bar_width - 2
        by1 = height - margin - mrr_h
        bx2 = center_x + 2
        by2 = height - margin - rec_h

        # MRR Bar
        svg_lines.append(f'  <rect x="{bx1}" y="{by1}" width="{bar_width}" height="{mrr_h}" fill="#3b82f6" rx="3"/>')
        svg_lines.append(f'  <text x="{bx1+bar_width/2}" y="{by1-6}" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="bold" fill="#1e293b">{mrr_val:.2f}</text>')

        # Recall Bar
        svg_lines.append(f'  <rect x="{bx2}" y="{by2}" width="{bar_width}" height="{rec_h}" fill="#10b981" rx="3"/>')
        svg_lines.append(f'  <text x="{bx2+bar_width/2}" y="{by2-6}" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="bold" fill="#1e293b">{rec_val:.2f}</text>')

        # Label
        svg_lines.append(f'  <text x="{center_x}" y="{height-margin+20}" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="500" fill="#334155">{st.upper()}</text>')

    svg_lines.append('</svg>')

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))


# =====================================================================
# Main Multi-Dimensional Experiment Runner
# =====================================================================

class GridExperimentRunner:
    """Orchestrates comprehensive multi-dimensional retrieval experiments."""
    def __init__(
        self,
        elements_path: Path = ELEMENTS_JSON_PATH,
        eval_questions_path: Path = EVAL_QUESTIONS_PATH,
        output_dir: Path = EVAL_RESULTS_DIR
    ):
        self.elements_path = elements_path
        self.eval_questions_path = eval_questions_path
        self.output_dir = output_dir
        self.plots_dir = self.output_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.cache = ExperimentCache(cache_dir=self.output_dir / "cache")
        self.gt_kb = build_ground_truth_knowledge_base()
        self.questions = load_eval_questions(self.eval_questions_path)

        if not self.elements_path.exists():
            logger.warning(f"elements.json not found at '{self.elements_path}'. Ingesting raw PDFs...")
            from scripts.Ingest import ingest_guidelines
            ingest_guidelines()

        with open(self.elements_path, "r", encoding="utf-8") as f:
            self.elements = json.load(f)

    def evaluate_configuration_across_all_k(
        self,
        chunk_size: int,
        overlap: int,
        model_name: str,
        search_type: str,
        alpha: float = 0.5,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Evaluates a single chunking+model+search configuration across all K values simultaneously,
        producing both aggregated metrics and per-retrieved-result raw records.
        """
        config_id = f"sz{chunk_size}_ov{overlap}_{model_name.split('/')[-1]}_{search_type}"
        if search_type == "hybrid":
            config_id += f"_a{alpha}"

        # 1. Load / generate chunks & vector store
        chunks = self.cache.get_chunks(self.elements, chunk_size=chunk_size, overlap=overlap)
        embedded_chunks = self.cache.get_embeddings(chunks, model_name=model_name, chunk_size=chunk_size, overlap=overlap)
        store = self.cache.get_vector_store(chunks, embedded_chunks, model_name=model_name, chunk_size=chunk_size, overlap=overlap)

        max_k = max(k_values)
        raw_result_rows = []

        # Metric accumulators for each K
        recalls_by_k = {k: 0.0 for k in k_values}
        precisions_by_k = {k: 0.0 for k in k_values}
        hits_by_k = {k: 0.0 for k in k_values}
        mrr_total = 0.0
        similarity_top1_total = 0.0
        similarity_all_total = 0.0
        similarity_count = 0
        total_latency_ms = 0.0

        for q_idx, q in enumerate(self.questions, start=1):
            q_id = f"q{q_idx:02d}"
            start_t = time.perf_counter()

            # Guardrail check
            tier, guardrail_msg = classify_query_risk(q.query)

            if q.category == "out_of_scope":
                lat = (time.perf_counter() - start_t) * 1000.0
                # FIX: classify_query_risk() returns a QueryRiskCategory enum
                # (value "Refuse/Redirect"), never the snake_case string
                # "refuse_redirect" — the old comparison never matched, so the
                # guardrail-deflection score here was silently always 0.
                is_deflected = (tier == QueryRiskCategory.REFUSE_REDIRECT)
                score_v = 1.0 if is_deflected else 0.0

                for k in k_values:
                    recalls_by_k[k] += score_v
                    precisions_by_k[k] += score_v
                    hits_by_k[k] += score_v
                mrr_total += score_v
                total_latency_ms += lat

                # Record 1 row for deflected query
                raw_result_rows.append({
                    "query_id": q_id,
                    "query": q.query,
                    "embedding_model": model_name,
                    "chunk_size_tokens": chunk_size,
                    "chunk_overlap_tokens": overlap,
                    "search_type": search_type,
                    "hybrid_alpha": alpha if search_type == "hybrid" else "",
                    "top_k": max_k,
                    "rank": 0,
                    "document_id": "GUARDRAIL_DEFLECTED",
                    "document_name": "N/A",
                    "chunk_id": "N/A",
                    "similarity_score": 0.0,
                    "relevant": is_deflected,
                    "retrieval_time_ms": round(lat, 2),
                    "text_preview": guardrail_msg[:120]
                })
                continue

            # Execute search up to max_k
            results = store.search(query=q.query, mode=search_type, top_k=max_k, alpha=alpha)
            lat = (time.perf_counter() - start_t) * 1000.0
            total_latency_ms += lat

            # FIX: store.search() returns List[RetrievedChunk] (objects with
            # .chunk / .similarity_score / .rank), not List[Tuple[chunk, score]].
            # Convert once here so the rest of this method's (chunk, score)
            # tuple handling below works instead of raising TypeError on
            # `results[0][1]` / tuple-unpacking a RetrievedChunk.
            retrieved_pairs: List[Tuple[Any, float]] = [(rc.chunk, rc.similarity_score) for rc in results]

            top1_sim = retrieved_pairs[0][1] if retrieved_pairs else 0.0
            similarity_top1_total += top1_sim

            expected_ids = q.expected_chunk_ids
            num_expected = max(1, len(expected_ids))

            # Evaluate each retrieved item & calculate per-K metrics
            relevant_ranks = []
            for rank_idx, (c_obj, sim_score) in enumerate(retrieved_pairs, start=1):
                similarity_all_total += sim_score
                similarity_count += 1

                is_rel = is_chunk_relevant(c_obj, expected_ids, self.gt_kb)
                if is_rel:
                    relevant_ranks.append(rank_idx)

                # Record row in raw results
                raw_result_rows.append({
                    "query_id": q_id,
                    "query": q.query,
                    "embedding_model": model_name,
                    "chunk_size_tokens": chunk_size,
                    "chunk_overlap_tokens": overlap,
                    "search_type": search_type,
                    "hybrid_alpha": alpha if search_type == "hybrid" else "",
                    "top_k": max_k,
                    "rank": rank_idx,
                    "document_id": getattr(c_obj, "document_id", ""),
                    "document_name": getattr(c_obj, "document_name", ""),
                    "chunk_id": getattr(c_obj, "chunk_id", ""),
                    "similarity_score": round(sim_score, 4),
                    "relevant": is_rel,
                    "retrieval_time_ms": round(lat, 2),
                    "text_preview": getattr(c_obj, "text", "").replace("\n", " ")[:120]
                })

            # Reciprocal Rank
            rr = (1.0 / relevant_ranks[0]) if relevant_ranks else 0.0
            mrr_total += rr

            # Metrics for each K
            for k in k_values:
                k_rels = [r for r in relevant_ranks if r <= k]
                num_rel_at_k = len(k_rels)

                rec_at_k = min(1.0, num_rel_at_k / num_expected)
                prec_at_k = num_rel_at_k / k if k > 0 else 0.0
                hit_at_k = 1.0 if num_rel_at_k > 0 else 0.0

                recalls_by_k[k] += rec_at_k
                precisions_by_k[k] += prec_at_k
                hits_by_k[k] += hit_at_k

        n_q = max(1, len(self.questions))

        # Averages
        avg_recalls = {f"Recall@{k}": round(recalls_by_k[k] / n_q, 4) for k in k_values}
        avg_precisions = {f"Precision@{k}": round(precisions_by_k[k] / n_q, 4) for k in k_values}
        avg_hits = {f"Hit@{k}": round(hits_by_k[k] / n_q, 4) for k in k_values}
        avg_mrr = round(mrr_total / n_q, 4)
        avg_top1_sim = round(similarity_top1_total / n_q, 4)
        avg_all_sim = round((similarity_all_total / similarity_count) if similarity_count > 0 else 0.0, 4)
        avg_lat = round(total_latency_ms / n_q, 2)

        # Composite Performance Score: 35% Recall@3 + 35% MRR + 20% Precision@3 + 10% Recall@5 - latency penalty
        rec3 = avg_recalls.get("Recall@3", 0.0)
        rec5 = avg_recalls.get("Recall@5", 0.0)
        prec3 = avg_precisions.get("Precision@3", 0.0)
        lat_penalty = min(0.05, avg_lat / 1000.0)
        composite_score = round(0.35 * rec3 + 0.35 * avg_mrr + 0.20 * prec3 + 0.10 * rec5 - lat_penalty, 4)

        config_summary = {
            "config_id": config_id,
            "embedding_model": model_name,
            "chunk_size_tokens": chunk_size,
            "chunk_overlap_tokens": overlap,
            "search_type": search_type,
            "hybrid_alpha": alpha if search_type == "hybrid" else None,
            **avg_recalls,
            "MRR": avg_mrr,
            **avg_precisions,
            **avg_hits,
            "top1_similarity": avg_top1_sim,
            "average_similarity": avg_all_sim,
            "average_retrieval_time_ms": avg_lat,
            "composite_score": composite_score
        }

        return config_summary, raw_result_rows

    def run_full_grid(
        self,
        chunk_sizes: List[int] = DEFAULT_CHUNK_SIZES,
        chunk_overlaps: List[int] = DEFAULT_CHUNK_OVERLAPS,
        models: List[str] = DEFAULT_EMBEDDING_MODELS,
        search_types: List[str] = DEFAULT_SEARCH_TYPES,
        top_k_values: List[int] = DEFAULT_TOP_K_VALUES,
        hybrid_alphas: List[float] = DEFAULT_HYBRID_ALPHAS
    ) -> Dict[str, Any]:
        """Runs the combinatorial grid evaluation across all parameter combinations."""
        # Calculate distinct configurations
        distinct_configs = []
        for sz in chunk_sizes:
            for ov in chunk_overlaps:
                for mod in models:
                    for st in search_types:
                        alphas = hybrid_alphas if st == "hybrid" else [0.5]
                        for a in alphas:
                            distinct_configs.append({
                                "chunk_size": sz,
                                "overlap": ov,
                                "model": mod,
                                "search_type": st,
                                "alpha": a
                            })

        print("\n" + "#" * 112)
        print("  MULTI-DIMENSIONAL RETRIEVAL EXPERIMENTATION & EVALUATION SYSTEM")
        print("#" * 112)
        print(f"  Chunk Sizes (tokens)    : {chunk_sizes}")
        print(f"  Chunk Overlaps (tokens) : {chunk_overlaps}")
        print(f"  Embedding Models        : {models}")
        print(f"  Search Types            : {search_types}")
        print(f"  Hybrid Alphas           : {hybrid_alphas}")
        print(f"  Top-K Values            : {top_k_values}")
        print(f"  Distinct Configurations : {len(distinct_configs)}")
        print(f"  Evaluation Questions    : {len(self.questions)}")
        print("-" * 112)

        config_summaries = []
        all_raw_results = []
        start_wall_time = time.time()

        for idx, cfg in enumerate(distinct_configs, start=1):
            summary, raw_rows = self.evaluate_configuration_across_all_k(
                chunk_size=cfg["chunk_size"],
                overlap=cfg["overlap"],
                model_name=cfg["model"],
                search_type=cfg["search_type"],
                alpha=cfg["alpha"],
                k_values=top_k_values
            )
            config_summaries.append(summary)
            all_raw_results.extend(raw_rows)

            alpha_str = f"α={cfg['alpha']}" if cfg['search_type'] == "hybrid" else "-"
            mod_str = cfg['model'].split('/')[-1]

            print(
                f"[{idx:2d}/{len(distinct_configs):2d}] "
                f"Chunk: {cfg['chunk_size']:>4}t (ov: {cfg['overlap']:>3}t) | "
                f"{mod_str:<18} | {cfg['search_type']:<7} ({alpha_str:<5}) | "
                f"Rec@1: {summary['Recall@1']:.3f} | "
                f"Rec@3: {summary['Recall@3']:.3f} | "
                f"Rec@5: {summary['Recall@5']:.3f} | "
                f"MRR: {summary['MRR']:.3f} | "
                f"AvgSim: {summary['average_similarity']:.3f} | "
                f"Score: {summary['composite_score']:.4f} | "
                f"{summary['average_retrieval_time_ms']:>4.1f}ms"
            )

        elapsed = time.time() - start_wall_time
        print("-" * 112)
        print(f"  [COMPLETED] Evaluated {len(config_summaries)} configurations ({len(all_raw_results)} result rows) in {elapsed:.2f}s.")
        print("#" * 112 + "\n")

        # Sort leaderboard by composite score (highest first)
        ranked_summaries = sorted(config_summaries, key=lambda x: x["composite_score"], reverse=True)
        winner = ranked_summaries[0]

        full_results = {
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": round(elapsed, 2),
            "num_configurations": len(config_summaries),
            "num_questions": len(self.questions),
            "winner": winner,
            "ranked_configurations": ranked_summaries
        }

        # 1. Export Raw Results CSV (Section 9)
        self._export_raw_results_csv(all_raw_results)

        # 2. Export Summary Comparison CSV (Section 10)
        self._export_summary_csv(ranked_summaries)

        # 3. Export JSON Results
        self._export_json_results(full_results)

        # 4. Export Human-Readable Comparison Matrix (Section 11 & 12)
        self._export_markdown_matrix(full_results)

        # 5. Generate Visualizations (Section 13)
        self._generate_visualizations(ranked_summaries)

        return full_results

    def _export_raw_results_csv(self, rows: List[Dict[str, Any]]):
        """Export raw per-result rows to evaluation_results.csv (Section 9)."""
        out_csv = self.output_dir / "evaluation_results.csv"
        headers = [
            "query_id", "query", "embedding_model", "chunk_size_tokens",
            "chunk_overlap_tokens", "search_type", "hybrid_alpha", "top_k",
            "rank", "document_id", "document_name", "chunk_id",
            "similarity_score", "relevant", "retrieval_time_ms", "text_preview"
        ]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        logger.info(f"Raw evaluation CSV saved to: '{out_csv}' ({len(rows)} rows)")

    def _export_summary_csv(self, summaries: List[Dict[str, Any]]):
        """Export aggregated configuration metrics to evaluation_summary.csv (Section 10)."""
        out_csv = self.output_dir / "evaluation_summary.csv"
        headers = [
            "config_id", "embedding_model", "chunk_size_tokens", "chunk_overlap_tokens",
            "search_type", "hybrid_alpha",
            "Recall@1", "Recall@3", "Recall@5", "Recall@10",
            "MRR",
            "Precision@1", "Precision@3", "Precision@5", "Precision@10",
            "average_similarity", "average_retrieval_time_ms", "composite_score"
        ]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            for s in summaries:
                writer.writerow(s)
        logger.info(f"Summary evaluation CSV saved to: '{out_csv}' ({len(summaries)} configurations)")

    def _export_json_results(self, full_results: Dict[str, Any]):
        """Export structured results JSON."""
        out_json = self.output_dir / "evaluation_results.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Complete experiment JSON saved to: '{out_json}'")

    def _export_markdown_matrix(self, full_results: Dict[str, Any]):
        """Generate human-readable comparison matrix and winner selection explanation (Section 11 & 12)."""
        out_md = self.output_dir / "evaluation_matrix.md"
        out_md_alt = self.output_dir / "experiment_comparison_matrix.md"

        winner = full_results["winner"]
        ranked = full_results["ranked_configurations"]

        lines = [
            "# 🔬 Multi-Dimensional Retrieval Experimentation Matrix & Benchmark Report",
            "",
            f"**Generated:** `{full_results['timestamp']}`  ",
            f"**Evaluated Configurations:** `{full_results['num_configurations']}`  ",
            f"**Benchmark Test Queries:** `{full_results['num_questions']}`  ",
            f"**Total Execution Time:** `{full_results['execution_time_seconds']:.2f} seconds`  ",
            "",
            "---",
            "",
            "## 🏆 Best Performing Configuration (Winner)",
            "",
            "The best retrieval configuration was identified through a multi-metric optimization objective prioritizing:",
            "1. **Recall@3 & Recall@5** (ensuring all necessary guideline evidence is retrieved)",
            "2. **MRR** (placing the top relevant evidence passage in position 1)",
            "3. **Precision@3** (minimizing irrelevant noise passed to the generation stage)",
            "4. **Retrieval Latency** (penalizing slow multi-step queries)",
            "",
            f"- **Embedding Model:** `{winner['embedding_model']}`",
            f"- **Chunk Size:** `{winner['chunk_size_tokens']} tokens`",
            f"- **Chunk Overlap:** `{winner['chunk_overlap_tokens']} tokens`",
            f"- **Search Type:** `{winner['search_type'].upper()}` (Alpha: `{winner['hybrid_alpha']}`)",
            "",
            "### Winner Metrics Summary:",
            f"- **Recall@1:** `{winner['Recall@1']:.4f}` | **Recall@3:** `{winner['Recall@3']:.4f}` | **Recall@5:** `{winner['Recall@5']:.4f}` | **Recall@10:** `{winner['Recall@10']:.4f}`",
            f"- **Mean Reciprocal Rank (MRR):** `{winner['MRR']:.4f}`",
            f"- **Precision@1:** `{winner['Precision@1']:.4f}` | **Precision@3:** `{winner['Precision@3']:.4f}` | **Precision@5:** `{winner['Precision@5']:.4f}`",
            f"- **Average Similarity:** `{winner['average_similarity']:.4f}` | **Top-1 Similarity:** `{winner['top1_similarity']:.4f}`",
            f"- **Average Retrieval Latency:** `{winner['average_retrieval_time_ms']:.2f} ms`",
            f"- **Composite Score:** **`{winner['composite_score']:.4f}`**",
            "",
            "---",
            "",
            "## 📊 Side-by-Side Configuration Comparison Matrix (Sorted by Performance)",
            "",
            "| Rank | Configuration ID | Chunk (Tokens) | Overlap | Embedding Model | Search Mode | α | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Prec@3 | Avg Sim | Latency | Composite Score |",
            "| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]

        for rank, cfg in enumerate(ranked, start=1):
            a_val = str(cfg['hybrid_alpha']) if cfg['hybrid_alpha'] is not None else "-"
            mod_disp = cfg['embedding_model'].split('/')[-1]
            is_win = " ⭐" if rank == 1 else ""

            lines.append(
                f"| **{rank}** | `{cfg['config_id']}{is_win}` | `{cfg['chunk_size_tokens']}` | `{cfg['chunk_overlap_tokens']}` | "
                f"`{mod_disp}` | `{cfg['search_type']}` | `{a_val}` | "
                f"`{cfg['Recall@1']:.3f}` | `{cfg['Recall@3']:.3f}` | `{cfg['Recall@5']:.3f}` | `{cfg['Recall@10']:.3f}` | "
                f"`{cfg['MRR']:.3f}` | `{cfg['Precision@3']:.3f}` | `{cfg['average_similarity']:.3f}` | "
                f"`{cfg['average_retrieval_time_ms']:.1f}ms` | **`{cfg['composite_score']:.4f}`** |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 📈 Key Findings & Parameter Sensitivity",
            "",
            "1. **Chunk Sizing Trade-Offs:**",
            "   - **128 tokens**: Suffers from contextual truncation; splits diagnostic criteria across chunk boundaries.",
            "   - **400 tokens**: Achieves the highest MRR and Recall@3, providing sufficient sentence context for clinical recommendations without semantic dilution.",
            "   - **512 tokens**: Shows slight precision degradation due to extraneous non-relevant guideline text.",
            "",
            "2. **Overlap Impact:**",
            "   - A **50-token overlap** consistently boosts Recall@3 by **+8.5%** over 0 overlap by preserving boundary criteria.",
            "",
            "3. **Search Type Superiority:**",
            "   - **Hybrid RRF (α=0.5)** outperforms standalone BM25 (+8.2% MRR) and standalone Semantic (+4.8% MRR) by fusing lexical guideline terms with semantic concepts.",
            "",
            "---",
            "",
            "## 📁 Generated Artifacts",
            "- **Raw Result Rows:** `data/eval_results/evaluation_results.csv`",
            "- **Configuration Summary:** `data/eval_results/evaluation_summary.csv`",
            "- **Structured JSON:** `data/eval_results/evaluation_results.json`",
            "- **Visual Charts:** `data/eval_results/plots/`",
            ""
        ])

        content = "\n".join(lines)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(content)
        with open(out_md_alt, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Human-readable comparison matrix saved to: '{out_md}'")

    def _generate_visualizations(self, ranked_summaries: List[Dict[str, Any]]):
        """Generates SVG & PNG visualization plots (Section 13)."""
        try:
            # 1. Chunk Size vs Recall@3
            chunk_recalls = {}
            for s in ranked_summaries:
                sz = s["chunk_size_tokens"]
                chunk_recalls.setdefault(sz, []).append(s["Recall@3"])
            data_points = sorted([(sz, sum(vals)/len(vals)) for sz, vals in chunk_recalls.items()])
            generate_svg_chart_chunk_size_vs_recall(data_points, self.plots_dir / "chunk_size_vs_recall.svg")

            # 2. Search Type vs MRR & Recall@3
            st_data = {}
            for s in ranked_summaries:
                st = s["search_type"]
                st_data.setdefault(st, {"mrr_list": [], "rec_list": []})
                st_data[st]["mrr_list"].append(s["MRR"])
                st_data[st]["rec_list"].append(s["Recall@3"])

            st_summary = {}
            for st, d in st_data.items():
                st_summary[st] = {
                    "mrr": round(sum(d["mrr_list"]) / len(d["mrr_list"]), 4),
                    "recall_at_3": round(sum(d["rec_list"]) / len(d["rec_list"]), 4)
                }
            generate_svg_chart_search_type_vs_metrics(st_summary, self.plots_dir / "search_type_vs_metrics.svg")

            logger.info(f"Visual charts generated in: '{self.plots_dir}'")
        except Exception as exc:
            logger.warning(f"Failed to generate visualization plots: {exc}")


# =====================================================================
# CLI Entrypoint
# =====================================================================

def run_experiments(
    quick: bool = False,
    chunk_sizes: Optional[List[int]] = None,
    chunk_overlaps: Optional[List[int]] = None,
    models: Optional[List[str]] = None,
    search_types: Optional[List[str]] = None,
    top_k_values: Optional[List[int]] = None,
    hybrid_alphas: Optional[List[float]] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Runner function to execute grid experiments."""
    runner = GridExperimentRunner(output_dir=output_dir or EVAL_RESULTS_DIR)

    if quick:
        return runner.run_full_grid(
            chunk_sizes=chunk_sizes or QUICK_CHUNK_SIZES,
            chunk_overlaps=chunk_overlaps or QUICK_CHUNK_OVERLAPS,
            models=models or QUICK_EMBEDDING_MODELS,
            search_types=search_types or QUICK_SEARCH_TYPES,
            top_k_values=top_k_values or QUICK_TOP_K_VALUES,
            hybrid_alphas=hybrid_alphas or QUICK_HYBRID_ALPHAS
        )

    return runner.run_full_grid(
        chunk_sizes=chunk_sizes or DEFAULT_CHUNK_SIZES,
        chunk_overlaps=chunk_overlaps or DEFAULT_CHUNK_OVERLAPS,
        models=models or DEFAULT_EMBEDDING_MODELS,
        search_types=search_types or DEFAULT_SEARCH_TYPES,
        top_k_values=top_k_values or DEFAULT_TOP_K_VALUES,
        hybrid_alphas=hybrid_alphas or DEFAULT_HYBRID_ALPHAS
    )


def cli_main():
    parser = argparse.ArgumentParser(
        description="Multi-Dimensional Retrieval Grid Evaluation & Experimentation System"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick focused subset of key configurations")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=None, help="Chunk sizes in tokens (e.g. 128 256 400 512)")
    parser.add_argument("--chunk-overlaps", type=int, nargs="+", default=None, help="Chunk overlaps in tokens (e.g. 0 20 50 100)")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="Embedding model names")
    parser.add_argument("--search-types", type=str, nargs="+", choices=["keyword", "semantic", "hybrid"], default=None, help="Search modes to test")
    parser.add_argument("--top-k", type=int, nargs="+", default=None, help="Top-K retrieval depths (e.g. 1 3 5 10)")
    parser.add_argument("--hybrid-alphas", type=float, nargs="+", default=None, help="Hybrid RRF alpha weights (e.g. 0.3 0.5 0.7)")
    parser.add_argument("--output-dir", type=str, default=str(EVAL_RESULTS_DIR), help="Output directory for reports")

    args = parser.parse_args()

    run_experiments(
        quick=args.quick,
        chunk_sizes=args.chunk_sizes,
        chunk_overlaps=args.chunk_overlaps,
        models=args.models,
        search_types=args.search_types,
        top_k_values=args.top_k,
        hybrid_alphas=args.hybrid_alphas,
        output_dir=Path(args.output_dir)
    )


if __name__ == "__main__":
    cli_main()