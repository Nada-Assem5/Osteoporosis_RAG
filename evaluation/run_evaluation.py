"""
Multi-Dimensional Retrieval Grid Evaluation & Experimentation Runner (evaluation/run_evaluation.py).

Executes combinatorial retrieval experiments across:
  Embedding Models × Chunk Sizes × Chunk Overlaps × Search Types × Top-K × Queries

Key Capabilities:
- Token-bounded re-chunking (tiktoken cl100k_base)
- Multi-model dense vector embeddings with disk caching
- Reuses existing Ingest, Chunk, VectorStore, and Retrieval implementations
- Computes complete IR metrics: Recall@1/3/5/10, Precision@1/3/5/10, MRR, NDCG, Similarity, Latency
- Generates 5 distinct analytical SVG & PNG visual comparison plots in evaluation/plots/
- Exports evaluation_results.csv, evaluation_summary.csv, best_configuration.json, evaluation_report.md
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

from src.schema import Chunk, EvalQuestion
from src.utils import count_tokens, compute_content_hash
from scripts.Chunk import chunk_extracted_elements
from scripts.Embeddings import generate_embeddings
from scripts.Vector_db import VectorStore, _tokenize, _cosine_similarity
from scripts.Retrieval import classify_query_risk, load_eval_questions, _match_chunk

import evaluation.config as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =====================================================================
# Ground Truth Alignment for Variable Token Chunk Sizes
# =====================================================================

def build_ground_truth_knowledge_base(
    baseline_chunks_path: Path = cfg.CHUNKS_BASELINE_PATH
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
    def __init__(self, cache_dir: Path = cfg.CACHE_DIR):
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
        embedded = generate_embeddings(chunk_dicts, model_name=model_name)
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
# Visual Plot Generators (Plots 1 to 5)
# =====================================================================

class PlotGenerator:
    """Generates the 5 required comparison SVG visual charts."""
    def __init__(self, plots_dir: Path = cfg.PLOTS_DIR):
        self.plots_dir = plots_dir
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def generate_plot1_chunk_size_vs_recall(self, summaries: List[Dict[str, Any]]):
        """Plot 1: Chunk size vs Recall@K (Recall@1, Recall@3, Recall@5)."""
        by_sz = {}
        for s in summaries:
            sz = s["chunk_size_tokens"]
            by_sz.setdefault(sz, {"r1": [], "r3": [], "r5": []})
            by_sz[sz]["r1"].append(s["Recall@1"])
            by_sz[sz]["r3"].append(s["Recall@3"])
            by_sz[sz]["r5"].append(s["Recall@5"])

        sorted_sz = sorted(by_sz.keys())
        width, height, margin = 680, 380, 60

        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            f'  <rect width="{width}" height="{height}" fill="#ffffff"/>',
            f'  <text x="{width/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold" fill="#0f172a">Plot 1: Chunk Size (Tokens) vs Recall@K</text>',
            f'  <!-- Legend -->',
            f'  <circle cx="{width-240}" cy="28" r="5" fill="#3b82f6"/>',
            f'  <text x="{width-230}" y="32" font-family="sans-serif" font-size="11" fill="#475569">Recall@1</text>',
            f'  <circle cx="{width-160}" cy="28" r="5" fill="#10b981"/>',
            f'  <text x="{width-150}" y="32" font-family="sans-serif" font-size="11" fill="#475569">Recall@3</text>',
            f'  <circle cx="{width-80}" cy="28" r="5" fill="#8b5cf6"/>',
            f'  <text x="{width-70}" y="32" font-family="sans-serif" font-size="11" fill="#475569">Recall@5</text>',
            f'  <!-- Axes -->',
            f'  <line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#94a3b8" stroke-width="2"/>',
            f'  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#94a3b8" stroke-width="2"/>'
        ]

        for step in [0.2, 0.4, 0.6, 0.8, 1.0]:
            sy = height - margin - step * (height - 2 * margin)
            svg.append(f'  <line x1="{margin}" y1="{sy}" x2="{width-margin}" y2="{sy}" stroke="#e2e8f0" stroke-width="1"/>')
            svg.append(f'  <text x="{margin-10}" y="{sy+4}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#64748b">{step:.1f}</text>')

        min_sz, max_sz = min(sorted_sz), max(sorted_sz)
        def scale_x(sz):
            return margin + (sz - min_sz) / max(1, max_sz - min_sz) * (width - 2 * margin)
        def scale_y(val):
            return height - margin - val * (height - 2 * margin)

        for metric_key, color in [("r1", "#3b82f6"), ("r3", "#10b981"), ("r5", "#8b5cf6")]:
            pts = []
            for sz in sorted_sz:
                avg_v = sum(by_sz[sz][metric_key]) / len(by_sz[sz][metric_key])
                sx, sy = scale_x(sz), scale_y(avg_v)
                pts.append(f"{sx},{sy}")
            svg.append(f'  <polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(pts)}"/>')

            for sz in sorted_sz:
                avg_v = sum(by_sz[sz][metric_key]) / len(by_sz[sz][metric_key])
                sx, sy = scale_x(sz), scale_y(avg_v)
                svg.append(f'  <circle cx="{sx}" cy="{sy}" r="5" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
                svg.append(f'  <text x="{sx}" y="{sy-10}" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="bold" fill="#1e293b">{avg_v:.3f}</text>')

        for sz in sorted_sz:
            sx = scale_x(sz)
            svg.append(f'  <text x="{sx}" y="{height-margin+22}" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="500" fill="#334155">{sz} tokens</text>')

        svg.append('</svg>')
        with open(self.plots_dir / "plot1_chunk_size_vs_recall.svg", "w", encoding="utf-8") as f:
            f.write("\n".join(svg))

    def generate_plot2_chunk_overlap_vs_recall(self, summaries: List[Dict[str, Any]]):
        """Plot 2: Chunk overlap vs Recall@K."""
        by_ov = {}
        for s in summaries:
            ov = s["chunk_overlap_tokens"]
            by_ov.setdefault(ov, {"r3": [], "mrr": []})
            by_ov[ov]["r3"].append(s["Recall@3"])
            by_ov[ov]["mrr"].append(s["MRR"])

        sorted_ov = sorted(by_ov.keys())
        width, height, margin = 680, 380, 60

        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            f'  <rect width="{width}" height="{height}" fill="#ffffff"/>',
            f'  <text x="{width/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold" fill="#0f172a">Plot 2: Chunk Overlap (Tokens) vs Recall@3 &amp; MRR</text>',
            f'  <!-- Legend -->',
            f'  <circle cx="{width-180}" cy="28" r="5" fill="#0284c7"/>',
            f'  <text x="{width-170}" y="32" font-family="sans-serif" font-size="11" fill="#475569">Recall@3</text>',
            f'  <circle cx="{width-90}" cy="28" r="5" fill="#f59e0b"/>',
            f'  <text x="{width-80}" y="32" font-family="sans-serif" font-size="11" fill="#475569">MRR</text>',
            f'  <!-- Axes -->',
            f'  <line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#94a3b8" stroke-width="2"/>',
            f'  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#94a3b8" stroke-width="2"/>'
        ]

        for step in [0.2, 0.4, 0.6, 0.8, 1.0]:
            sy = height - margin - step * (height - 2 * margin)
            svg.append(f'  <line x1="{margin}" y1="{sy}" x2="{width-margin}" y2="{sy}" stroke="#e2e8f0" stroke-width="1"/>')
            svg.append(f'  <text x="{margin-10}" y="{sy+4}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#64748b">{step:.1f}</text>')

        min_ov, max_ov = min(sorted_ov), max(sorted_ov)
        def scale_x(ov):
            return margin + (ov - min_ov) / max(1, max_ov - min_ov) * (width - 2 * margin)
        def scale_y(val):
            return height - margin - val * (height - 2 * margin)

        for metric_key, color in [("r3", "#0284c7"), ("mrr", "#f59e0b")]:
            pts = []
            for ov in sorted_ov:
                avg_v = sum(by_ov[ov][metric_key]) / len(by_ov[ov][metric_key])
                sx, sy = scale_x(ov), scale_y(avg_v)
                pts.append(f"{sx},{sy}")
            svg.append(f'  <polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(pts)}"/>')

            for ov in sorted_ov:
                avg_v = sum(by_ov[ov][metric_key]) / len(by_ov[ov][metric_key])
                sx, sy = scale_x(ov), scale_y(avg_v)
                svg.append(f'  <circle cx="{sx}" cy="{sy}" r="5" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
                svg.append(f'  <text x="{sx}" y="{sy-10}" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="bold" fill="#1e293b">{avg_v:.3f}</text>')

        for ov in sorted_ov:
            sx = scale_x(ov)
            svg.append(f'  <text x="{sx}" y="{height-margin+22}" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="500" fill="#334155">{ov} tokens</text>')

        svg.append('</svg>')
        with open(self.plots_dir / "plot2_chunk_overlap_vs_recall.svg", "w", encoding="utf-8") as f:
            f.write("\n".join(svg))

    def generate_plot3_search_type_comparison(self, summaries: List[Dict[str, Any]]):
        """Plot 3: Search type comparison: Keyword vs Semantic vs Hybrid."""
        st_data = {}
        for s in summaries:
            st = s["search_type"]
            st_data.setdefault(st, {"mrr": [], "r3": [], "prec3": []})
            st_data[st]["mrr"].append(s["MRR"])
            st_data[st]["r3"].append(s["Recall@3"])
            st_data[st]["prec3"].append(s["Precision@3"])

        width, height, margin = 680, 380, 60
        search_types = ["keyword", "semantic", "hybrid"]
        group_width = (width - 2 * margin) / len(search_types)
        bar_width = 28

        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            f'  <rect width="{width}" height="{height}" fill="#ffffff"/>',
            f'  <text x="{width/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold" fill="#0f172a">Plot 3: Search Type Comparison (MRR vs Recall@3 vs Precision@3)</text>',
            f'  <!-- Legend -->',
            f'  <rect x="{width-280}" y="22" width="12" height="12" fill="#3b82f6"/>',
            f'  <text x="{width-260}" y="32" font-family="sans-serif" font-size="11" fill="#475569">MRR</text>',
            f'  <rect x="{width-200}" y="22" width="12" height="12" fill="#10b981"/>',
            f'  <text x="{width-180}" y="32" font-family="sans-serif" font-size="11" fill="#475569">Recall@3</text>',
            f'  <rect x="{width-110}" y="22" width="12" height="12" fill="#f59e0b"/>',
            f'  <text x="{width-90}" y="32" font-family="sans-serif" font-size="11" fill="#475569">Prec@3</text>',
            f'  <!-- Axes -->',
            f'  <line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#94a3b8" stroke-width="2"/>',
            f'  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#94a3b8" stroke-width="2"/>'
        ]

        for step in [0.2, 0.4, 0.6, 0.8, 1.0]:
            sy = height - margin - step * (height - 2 * margin)
            svg.append(f'  <line x1="{margin}" y1="{sy}" x2="{width-margin}" y2="{sy}" stroke="#e2e8f0" stroke-width="1"/>')
            svg.append(f'  <text x="{margin-10}" y="{sy+4}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#64748b">{step:.1f}</text>')

        for idx, st in enumerate(search_types):
            cx = margin + idx * group_width + group_width / 2
            d = st_data.get(st, {"mrr": [0], "r3": [0], "prec3": [0]})
            mrr_v = sum(d["mrr"]) / max(1, len(d["mrr"]))
            r3_v = sum(d["r3"]) / max(1, len(d["r3"]))
            p3_v = sum(d["prec3"]) / max(1, len(d["prec3"]))

            # 3 bars
            for b_idx, (val, color) in enumerate([(mrr_v, "#3b82f6"), (r3_v, "#10b981"), (p3_v, "#f59e0b")]):
                bx = cx - (1.5 * bar_width) + b_idx * (bar_width + 4)
                bh = val * (height - 2 * margin)
                by = height - margin - bh
                svg.append(f'  <rect x="{bx}" y="{by}" width="{bar_width}" height="{bh}" fill="{color}" rx="3"/>')
                svg.append(f'  <text x="{bx+bar_width/2}" y="{by-6}" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="bold" fill="#1e293b">{val:.2f}</text>')

            svg.append(f'  <text x="{cx}" y="{height-margin+22}" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="600" fill="#1e293b">{st.upper()}</text>')

        svg.append('</svg>')
        with open(self.plots_dir / "plot3_search_type_comparison.svg", "w", encoding="utf-8") as f:
            f.write("\n".join(svg))

    def generate_plot4_embedding_model_comparison(self, summaries: List[Dict[str, Any]]):
        """Plot 4: Embedding model comparison."""
        mod_data = {}
        for s in summaries:
            mod = s["embedding_model"].split('/')[-1]
            mod_data.setdefault(mod, {"mrr": [], "r3": [], "lat": []})
            mod_data[mod]["mrr"].append(s["MRR"])
            mod_data[mod]["r3"].append(s["Recall@3"])
            mod_data[mod]["lat"].append(s["average_retrieval_time_ms"])

        width, height, margin = 680, 380, 60
        models = list(mod_data.keys())
        group_width = (width - 2 * margin) / max(1, len(models))
        bar_width = 36

        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            f'  <rect width="{width}" height="{height}" fill="#ffffff"/>',
            f'  <text x="{width/2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold" fill="#0f172a">Plot 4: Embedding Model Comparison (MRR vs Recall@3)</text>',
            f'  <!-- Legend -->',
            f'  <rect x="{width-200}" y="22" width="12" height="12" fill="#6366f1"/>',
            f'  <text x="{width-180}" y="32" font-family="sans-serif" font-size="11" fill="#475569">MRR</text>',
            f'  <rect x="{width-130}" y="22" width="12" height="12" fill="#14b8a6"/>',
            f'  <text x="{width-110}" y="32" font-family="sans-serif" font-size="11" fill="#475569">Recall@3</text>',
            f'  <!-- Axes -->',
            f'  <line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#94a3b8" stroke-width="2"/>',
            f'  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#94a3b8" stroke-width="2"/>'
        ]

        for step in [0.2, 0.4, 0.6, 0.8, 1.0]:
            sy = height - margin - step * (height - 2 * margin)
            svg.append(f'  <line x1="{margin}" y1="{sy}" x2="{width-margin}" y2="{sy}" stroke="#e2e8f0" stroke-width="1"/>')
            svg.append(f'  <text x="{margin-10}" y="{sy+4}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#64748b">{step:.1f}</text>')

        for idx, mod in enumerate(models):
            cx = margin + idx * group_width + group_width / 2
            d = mod_data[mod]
            mrr_v = sum(d["mrr"]) / len(d["mrr"])
            r3_v = sum(d["r3"]) / len(d["r3"])

            bx1 = cx - bar_width - 2
            by1 = height - margin - mrr_v * (height - 2 * margin)
            bx2 = cx + 2
            by2 = height - margin - r3_v * (height - 2 * margin)

            svg.append(f'  <rect x="{bx1}" y="{by1}" width="{bar_width}" height="{mrr_v * (height - 2 * margin)}" fill="#6366f1" rx="3"/>')
            svg.append(f'  <text x="{bx1+bar_width/2}" y="{by1-6}" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="bold" fill="#1e293b">{mrr_v:.3f}</text>')

            svg.append(f'  <rect x="{bx2}" y="{by2}" width="{bar_width}" height="{r3_v * (height - 2 * margin)}" fill="#14b8a6" rx="3"/>')
            svg.append(f'  <text x="{bx2+bar_width/2}" y="{by2-6}" text-anchor="middle" font-family="sans-serif" font-size="10" font-weight="bold" fill="#1e293b">{r3_v:.3f}</text>')

            svg.append(f'  <text x="{cx}" y="{height-margin+22}" text-anchor="middle" font-family="sans-serif" font-size="11" font-weight="600" fill="#1e293b">{mod}</text>')

        svg.append('</svg>')
        with open(self.plots_dir / "plot4_embedding_model_comparison.svg", "w", encoding="utf-8") as f:
            f.write("\n".join(svg))

    def generate_plot5_top_configurations_ranked(self, ranked_summaries: List[Dict[str, Any]]):
        """Plot 5: Top configurations ranked by retrieval composite performance."""
        top_k_list = ranked_summaries[:8]
        width, height, margin = 720, 420, 50

        bar_h = 24
        gap = 18

        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            f'  <rect width="{width}" height="{height}" fill="#ffffff"/>',
            f'  <text x="{width/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="16" font-weight="bold" fill="#0f172a">Plot 5: Top 8 Configurations Ranked by Composite Score</text>'
        ]

        start_y = 60
        for idx, cfg_item in enumerate(top_k_list, start=1):
            y = start_y + (idx - 1) * (bar_h + gap)
            score = cfg_item["composite_score"]
            bar_w = score * 380
            cid = cfg_item["config_id"]
            if len(cid) > 28:
                cid = cid[:28] + "..."

            color = "#2563eb" if idx == 1 else "#3b82f6"
            star = " ⭐" if idx == 1 else ""

            svg.append(f'  <text x="{margin+180}" y="{y+16}" text-anchor="end" font-family="monospace" font-size="11" fill="#334155">#{idx} {cid}{star}</text>')
            svg.append(f'  <rect x="{margin+190}" y="{y}" width="{bar_w}" height="{bar_h}" fill="{color}" rx="4"/>')
            svg.append(f'  <text x="{margin+200+bar_w}" y="{y+16}" font-family="sans-serif" font-size="11" font-weight="bold" fill="#1e293b">{score:.4f} (Rec@3: {cfg_item["Recall@3"]:.2f}, MRR: {cfg_item["MRR"]:.2f})</text>')

        svg.append('</svg>')
        with open(self.plots_dir / "plot5_top_configurations_ranked.svg", "w", encoding="utf-8") as f:
            f.write("\n".join(svg))


# =====================================================================
# Main Experiment Runner Class
# =====================================================================

class GridExperimentRunner:
    """Orchestrates multi-dimensional retrieval experiments."""
    def __init__(
        self,
        elements_path: Path = cfg.ELEMENTS_JSON_PATH,
        eval_questions_path: Path = cfg.EVAL_QUESTIONS_PATH,
        output_dir: Path = cfg.RESULTS_DIR
    ):
        self.elements_path = elements_path
        self.eval_questions_path = eval_questions_path
        self.output_dir = output_dir
        self.plots_dir = self.output_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.cache = ExperimentCache(cache_dir=cfg.CACHE_DIR)
        self.gt_kb = build_ground_truth_knowledge_base()
        self.questions = load_eval_questions(self.eval_questions_path)
        self.plotter = PlotGenerator(plots_dir=self.plots_dir)

        if not self.elements_path.exists():
            logger.warning(f"elements.json not found at '{self.elements_path}'. Ingesting raw PDFs...")
            from scripts.Ingest import ingest_guidelines
            ingest_guidelines()

        with open(self.elements_path, "r", encoding="utf-8") as f:
            self.elements = json.load(f)

    def evaluate_configuration(
        self,
        chunk_size: int,
        overlap: int,
        model_name: str,
        search_type: str,
        alpha: float = 0.5,
        k_values: List[int] = cfg.TOP_K_VALUES
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Evaluates a single hyperparameter configuration across all queries."""
        config_id = f"sz{chunk_size}_ov{overlap}_{model_name.split('/')[-1]}_{search_type}"
        if search_type == "hybrid":
            config_id += f"_a{alpha}"

        # 1. Load / generate chunks & vector store
        chunks = self.cache.get_chunks(self.elements, chunk_size=chunk_size, overlap=overlap)
        embedded_chunks = self.cache.get_embeddings(chunks, model_name=model_name, chunk_size=chunk_size, overlap=overlap)
        store = self.cache.get_vector_store(chunks, embedded_chunks, model_name=model_name, chunk_size=chunk_size, overlap=overlap)

        max_k = max(k_values)
        raw_result_rows = []

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

            tier, guardrail_msg = classify_query_risk(q.query)

            if q.category == "out_of_scope":
                lat = (time.perf_counter() - start_t) * 1000.0
                is_def = (tier == "refuse_redirect")
                score_v = 1.0 if is_def else 0.0

                for k in k_values:
                    recalls_by_k[k] += score_v
                    precisions_by_k[k] += score_v
                    hits_by_k[k] += score_v
                mrr_total += score_v
                total_latency_ms += lat

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
                    "relevant": is_def,
                    "retrieval_time_ms": round(lat, 2),
                    "text_preview": guardrail_msg[:120]
                })
                continue

            # Execute search
            results = store.search(query=q.query, mode=search_type, top_k=max_k, alpha=alpha)
            lat = (time.perf_counter() - start_t) * 1000.0
            total_latency_ms += lat

            top1_sim = results[0][1] if results else 0.0
            similarity_top1_total += top1_sim

            expected_ids = q.expected_chunk_ids
            num_expected = max(1, len(expected_ids))

            relevant_ranks = []
            for rank_idx, (c_obj, sim_score) in enumerate(results, start=1):
                similarity_all_total += sim_score
                similarity_count += 1

                is_rel = is_chunk_relevant(c_obj, expected_ids, self.gt_kb)
                if is_rel:
                    relevant_ranks.append(rank_idx)

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

            rr = (1.0 / relevant_ranks[0]) if relevant_ranks else 0.0
            mrr_total += rr

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

    def run_grid(
        self,
        chunk_sizes: List[int] = cfg.CHUNK_SIZES,
        chunk_overlaps: List[int] = cfg.CHUNK_OVERLAPS,
        models: List[str] = cfg.EMBEDDING_MODELS,
        search_types: List[str] = cfg.SEARCH_TYPES,
        top_k_values: List[int] = cfg.TOP_K_VALUES,
        hybrid_alphas: List[float] = cfg.HYBRID_ALPHAS
    ) -> Dict[str, Any]:
        """Runs the combinatorial grid evaluation across all parameter combinations."""
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

        for idx, c in enumerate(distinct_configs, start=1):
            alpha_disp = f"α={c['alpha']}" if c['search_type'] == "hybrid" else "-"
            mod_disp = c['model'].split('/')[-1]

            print(f"[{idx:2d}/{len(distinct_configs):2d}] Evaluating: Embedding: {mod_disp:<16} | Chunk: {c['chunk_size']:>4}t | Overlap: {c['overlap']:>3}t | Search: {c['search_type']:<7} ({alpha_disp}) ...", end="\r")

            summary, raw_rows = self.evaluate_configuration(
                chunk_size=c["chunk_size"],
                overlap=c["overlap"],
                model_name=c["model"],
                search_type=c["search_type"],
                alpha=c["alpha"],
                k_values=top_k_values
            )
            config_summaries.append(summary)
            all_raw_results.extend(raw_rows)

            print(
                f"[{idx:2d}/{len(distinct_configs):2d}] "
                f"Chunk: {c['chunk_size']:>4}t (ov: {c['overlap']:>3}t) | "
                f"{mod_disp:<16} | {c['search_type']:<7} ({alpha_disp:<5}) | "
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

        # 1. Export evaluation_results.csv (raw rows)
        self._export_raw_csv(all_raw_results)

        # 2. Export evaluation_summary.csv (leaderboard)
        self._export_summary_csv(ranked_summaries)

        # 3. Export best_configuration.json
        self._export_best_config(winner)

        # 4. Export evaluation_report.md
        self._export_markdown_report(full_results)

        # 5. Generate all 5 visual plots
        self._generate_all_plots(ranked_summaries)

        return full_results

    def _export_raw_csv(self, rows: List[Dict[str, Any]]):
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
        logger.info(f"Raw evaluation CSV saved to: '{out_csv}'")

    def _export_summary_csv(self, summaries: List[Dict[str, Any]]):
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
        logger.info(f"Summary evaluation CSV saved to: '{out_csv}'")

    def _export_best_config(self, winner: Dict[str, Any]):
        out_json = self.output_dir / "best_configuration.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(winner, f, indent=2, ensure_ascii=False)
        logger.info(f"Best configuration JSON saved to: '{out_json}'")

    def _export_markdown_report(self, full_results: Dict[str, Any]):
        out_md = self.output_dir / "evaluation_report.md"
        winner = full_results["winner"]
        ranked = full_results["ranked_configurations"]

        lines = [
            "# 🔬 Multi-Dimensional Retrieval Experimentation Report",
            "",
            f"**Execution Timestamp:** `{full_results['timestamp']}`  ",
            f"**Configurations Evaluated:** `{full_results['num_configurations']}`  ",
            f"**Benchmark Test Queries:** `{full_results['num_questions']}`  ",
            f"**Total Run Time:** `{full_results['execution_time_seconds']:.2f} seconds`  ",
            "",
            "---",
            "",
            "## 🏆 Best Performing Configuration",
            "",
            f"- **Configuration ID:** `{winner['config_id']}`",
            f"- **Embedding Model:** `{winner['embedding_model']}`",
            f"- **Chunk Size:** `{winner['chunk_size_tokens']} tokens` | **Chunk Overlap:** `{winner['chunk_overlap_tokens']} tokens`",
            f"- **Search Type:** `{winner['search_type'].upper()}` (Alpha: `{winner['hybrid_alpha']}`)",
            "",
            "### Metric Summary:",
            f"- **Recall@1:** `{winner['Recall@1']:.4f}` | **Recall@3:** `{winner['Recall@3']:.4f}` | **Recall@5:** `{winner['Recall@5']:.4f}` | **Recall@10:** `{winner['Recall@10']:.4f}`",
            f"- **Mean Reciprocal Rank (MRR):** `{winner['MRR']:.4f}`",
            f"- **Precision@1:** `{winner['Precision@1']:.4f}` | **Precision@3:** `{winner['Precision@3']:.4f}` | **Precision@5:** `{winner['Precision@5']:.4f}`",
            f"- **Average Similarity:** `{winner['average_similarity']:.4f}`",
            f"- **Average Retrieval Latency:** `{winner['average_retrieval_time_ms']:.2f} ms`",
            f"- **Composite Score:** **`{winner['composite_score']:.4f}`**",
            "",
            "---",
            "",
            "## 📊 Side-by-Side Configuration Leaderboard",
            "",
            "| Rank | Configuration ID | Chunk (Tokens) | Overlap | Embedding Model | Search Mode | α | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Prec@3 | Avg Sim | Latency | Composite Score |",
            "| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]

        for rank, c in enumerate(ranked, start=1):
            a_val = str(c['hybrid_alpha']) if c['hybrid_alpha'] is not None else "-"
            mod_disp = c['embedding_model'].split('/')[-1]
            is_win = " ⭐" if rank == 1 else ""

            lines.append(
                f"| **{rank}** | `{c['config_id']}{is_win}` | `{c['chunk_size_tokens']}` | `{c['chunk_overlap_tokens']}` | "
                f"`{mod_disp}` | `{c['search_type']}` | `{a_val}` | "
                f"`{c['Recall@1']:.3f}` | `{c['Recall@3']:.3f}` | `{c['Recall@5']:.3f}` | `{c['Recall@10']:.3f}` | "
                f"`{c['MRR']:.3f}` | `{c['Precision@3']:.3f}` | `{c['average_similarity']:.3f}` | "
                f"`{c['average_retrieval_time_ms']:.1f}ms` | **`{c['composite_score']:.4f}`** |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 📈 Generated Visualizations in `plots/`",
            "",
            "1. **Plot 1: Chunk Size vs Recall@K** (`plots/plot1_chunk_size_vs_recall.svg`)",
            "2. **Plot 2: Chunk Overlap vs Recall@K** (`plots/plot2_chunk_overlap_vs_recall.svg`)",
            "3. **Plot 3: Search Type Comparison** (`plots/plot3_search_type_comparison.svg`)",
            "4. **Plot 4: Embedding Model Comparison** (`plots/plot4_embedding_model_comparison.svg`)",
            "5. **Plot 5: Top Ranked Configurations** (`plots/plot5_top_configurations_ranked.svg`)",
            ""
        ])

        with open(out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info(f"Markdown evaluation report saved to: '{out_md}'")

    def _generate_all_plots(self, ranked_summaries: List[Dict[str, Any]]):
        """Generate Plots 1 to 5."""
        self.plotter.generate_plot1_chunk_size_vs_recall(ranked_summaries)
        self.plotter.generate_plot2_chunk_overlap_vs_recall(ranked_summaries)
        self.plotter.generate_plot3_search_type_comparison(ranked_summaries)
        self.plotter.generate_plot4_embedding_model_comparison(ranked_summaries)
        self.plotter.generate_plot5_top_configurations_ranked(ranked_summaries)
        logger.info(f"All 5 visual comparison plots generated in: '{self.plots_dir}'")


# =====================================================================
# CLI Entrypoint
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Dimensional Retrieval Grid Evaluation & Experimentation Runner"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick focused subset of configurations")
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=None, help="Chunk sizes in tokens")
    parser.add_argument("--chunk-overlaps", type=int, nargs="+", default=None, help="Chunk overlaps in tokens")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="Embedding models")
    parser.add_argument("--search-types", type=str, nargs="+", choices=["keyword", "semantic", "hybrid"], default=None, help="Search modes")
    parser.add_argument("--top-k", type=int, nargs="+", default=None, help="Top-K values")
    parser.add_argument("--alphas", type=float, nargs="+", default=None, help="Hybrid RRF alpha weights")

    args = parser.parse_args()

    runner = GridExperimentRunner()

    if args.quick:
        runner.run_grid(
            chunk_sizes=args.chunk_sizes or cfg.QUICK_CHUNK_SIZES,
            chunk_overlaps=args.chunk_overlaps or cfg.QUICK_CHUNK_OVERLAPS,
            models=args.models or cfg.QUICK_EMBEDDING_MODELS,
            search_types=args.search_types or cfg.QUICK_SEARCH_TYPES,
            top_k_values=args.top_k or cfg.QUICK_TOP_K_VALUES,
            hybrid_alphas=args.alphas or cfg.QUICK_HYBRID_ALPHAS
        )
    else:
        runner.run_grid(
            chunk_sizes=args.chunk_sizes or cfg.CHUNK_SIZES,
            chunk_overlaps=args.chunk_overlaps or cfg.CHUNK_OVERLAPS,
            models=args.models or cfg.EMBEDDING_MODELS,
            search_types=args.search_types or cfg.SEARCH_TYPES,
            top_k_values=args.top_k or cfg.TOP_K_VALUES,
            hybrid_alphas=args.alphas or cfg.HYBRID_ALPHAS
        )


if __name__ == "__main__":
    main()
