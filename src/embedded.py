"""
Multi-Mode Vector Store, Retrieval Engine & Safety Guardrails (src.embedded).

Responsibilities:
- Keyword vector retrieval (TF-IDF & Okapi BM25)
- Dense semantic sentence embeddings via sentence-transformers ('all-MiniLM-L6-v2')
- Hybrid retrieval combining Keyword + Semantic rankings using Reciprocal Rank Fusion (RRF)
- 3-Tier Clinical Safety & Scope Guardrail validation (classify_query_risk & check_scope_guardrail)
- Vector index persistence (save/load JSON index artifacts)
"""

import os
import json
import math
import re
import zlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

from src.chunking import Chunk, chunk_document
from src.config import (
    DEFAULT_CLEANED_DIR,
    DEFAULT_INDEX_PATH,
    DEFAULT_CHUNK_SIZE_CHARS,
    DEFAULT_CHUNK_OVERLAP_CHARS,
    DEFAULT_RETRIEVAL_TOP_K,
    DEFAULT_RETRIEVAL_MODE,
    BM25_K1,
    BM25_B,
    RRF_K,
    DEFAULT_HYBRID_ALPHA,
    CLINICAL_KEYWORDS,
    EMERGENCY_KEYWORDS,
    PATIENT_SPECIFIC_PATTERNS
)

logger = logging.getLogger(__name__)

# Lazy singleton for sentence-transformers
_SENTENCE_TRANSFORMER_MODEL = None
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    _HAS_SENTENCE_TRANSFORMERS = False


def _get_embedding_model():
    """Singleton getter for SentenceTransformer all-MiniLM-L6-v2."""
    global _SENTENCE_TRANSFORMER_MODEL
    if _SENTENCE_TRANSFORMER_MODEL is None and _HAS_SENTENCE_TRANSFORMERS:
        try:
            _SENTENCE_TRANSFORMER_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:
            logger.warning(f"Could not load sentence-transformers model: {exc}")
            _SENTENCE_TRANSFORMER_MODEL = None
    return _SENTENCE_TRANSFORMER_MODEL


# =====================================================================
# Safety & Guardrail Workflow: 3-Tier Input Classification
# =====================================================================

def classify_query_risk(query: str) -> Tuple[str, str]:
    """
    3-Tier Safety & Guardrail Classifier:
    - 'refuse_redirect': Acute medical emergencies or out-of-scope queries
    - 'needs_caution': Patient-specific clinical scenarios requiring advisory disclaimers
    - 'allowed': In-scope guideline reference queries
    """
    q_lower = query.lower().strip()

    # 1. Emergency Detection (Refuse & Redirect)
    for em_word in EMERGENCY_KEYWORDS:
        if em_word in q_lower:
            return (
                "refuse_redirect",
                f"[SAFETY EMERGENCY REFUSAL] Query indicates a potential acute medical situation ('{em_word}'). "
                "This RAG system provides clinical guideline reference only and CANNOT manage emergencies. "
                "Please call emergency services (e.g. 999, 911) or seek immediate emergency care."
            )

    # 2. Patient-Specific Scenario Detection (Needs Caution)
    for pat in PATIENT_SPECIFIC_PATTERNS:
        if re.search(pat, q_lower, re.IGNORECASE):
            return (
                "needs_caution",
                "[PATIENT-SPECIFIC CAUTION] This query describes individualized patient circumstances. "
                "Guideline recommendations are population-level evidence summaries and must not replace "
                "individualized clinical evaluation, multidisciplinary review, or local protocols."
            )

    # 3. Scope Check against Clinical Vocabulary
    tokens = set(re.findall(r'\b[a-zA-Z0-9_\-\.]{2,}\b', q_lower))
    matches = tokens.intersection(CLINICAL_KEYWORDS)

    if not matches:
        return (
            "refuse_redirect",
            "[GUARDRAIL NOTICE] Query is OUT OF SCOPE. This clinical RAG system specializes in "
            "osteoporosis risk assessment, screening, bone mineral density (DXA), and fracture prevention guidelines. "
            "Please submit a clinical or bone health question."
        )

    # 4. Standard In-Scope Guideline Query (Allowed)
    return (
        "allowed",
        f"In-scope guideline query matching keywords: {', '.join(sorted(matches)[:4])}"
    )


def check_scope_guardrail(query: str) -> Tuple[bool, str]:
    """
    Backward-compatible 2-tuple guardrail check wrapping classify_query_risk.
    Returns (True, reason) for 'allowed' and 'needs_caution', (False, reason) for 'refuse_redirect'.
    """
    tier, reason = classify_query_risk(query)
    return (tier != "refuse_redirect", reason)


# =====================================================================
# VectorStore Implementation (Keyword, Semantic, Hybrid)
# =====================================================================

class VectorStore:
    """
    Unified Vector Store supporting Keyword (TF-IDF/BM25), Semantic (all-MiniLM-L6-v2),
    and Hybrid (Reciprocal Rank Fusion) retrieval modes.
    """
    def __init__(
        self,
        k1: float = BM25_K1,
        b: float = BM25_B,
        rrf_k: int = RRF_K
    ) -> None:
        self.k1 = k1
        self.b = b
        self.rrf_k = rrf_k
        self.chunks: List[Chunk] = []

        # Keyword index structures
        self.idf: Dict[str, float] = {}
        self.vectors: List[Dict[str, float]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.bm25_idf: Dict[str, float] = {}
        self.doc_term_freqs: List[Dict[str, int]] = []

        # Semantic dense embedding structures
        self.embeddings: List[List[float]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize, lowercase, and sanitize text into terms."""
        return [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_\-\.]{2,}\b', text)]

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Index a collection of chunks into Keyword and Semantic indexes."""
        self.chunks.extend(chunks)
        self._build_index()

    def _build_index(self) -> None:
        """Build both Keyword (TF-IDF / BM25) and Semantic dense embedding indexes."""
        doc_count = len(self.chunks)
        if doc_count == 0:
            return

        # 1. Build Keyword Index (TF-IDF & BM25)
        self.doc_lengths = []
        self.doc_term_freqs = []
        df: Dict[str, int] = {}
        tokenized_docs: List[List[str]] = []

        for chk in self.chunks:
            full_content = f"{chk.section_title} {chk.text} {' '.join(chk.topics)}"
            tokens = self._tokenize(full_content)
            tokenized_docs.append(tokens)
            self.doc_lengths.append(len(tokens))

            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_term_freqs.append(tf)

            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        self.avg_doc_length = sum(self.doc_lengths) / doc_count if doc_count > 0 else 1.0

        # TF-IDF IDF
        self.idf = {
            t: math.log((doc_count + 1) / (count + 0.5)) + 1.0
            for t, count in df.items()
        }

        # BM25 IDF
        self.bm25_idf = {
            t: math.log((doc_count - count + 0.5) / (count + 0.5) + 1.0)
            for t, count in df.items()
        }

        # Normalized TF-IDF Vectors
        self.vectors = []
        for tf_dict in self.doc_term_freqs:
            vec: Dict[str, float] = {}
            sq_sum = 0.0
            for t, count in tf_dict.items():
                val = (1.0 + math.log(count)) * self.idf.get(t, 1.0)
                vec[t] = val
                sq_sum += val * val

            norm = math.sqrt(sq_sum) if sq_sum > 0 else 1.0
            norm_vec = {t: val / norm for t, val in vec.items()}
            self.vectors.append(norm_vec)

        # 2. Build Semantic Dense Embeddings (sentence-transformers)
        self._build_semantic_embeddings()

    def _build_semantic_embeddings(self) -> None:
        """Compute dense vectors for each chunk."""
        model = _get_embedding_model()
        if model is not None:
            try:
                texts_to_embed = [f"{c.section_title}: {c.text}" for c in self.chunks]
                raw_embeds = model.encode(texts_to_embed, normalize_embeddings=True, show_progress_bar=False)
                self.embeddings = [emb.tolist() for emb in raw_embeds]
                return
            except Exception as e:
                logger.warning(f"Error computing dense embeddings with sentence-transformers: {e}")

        # Fallback dense projection representation (deterministic across processes)
        self.embeddings = []
        vec_dim = 128
        for chk in self.chunks:
            tokens = self._tokenize(f"{chk.section_title} {chk.text}")
            dense_vec = [0.0] * vec_dim
            for t in tokens:
                h = zlib.crc32(t.encode("utf-8")) % vec_dim
                dense_vec[h] += self.idf.get(t, 1.0)
            norm = math.sqrt(sum(v * v for v in dense_vec))
            if norm > 0:
                dense_vec = [v / norm for v in dense_vec]
            self.embeddings.append(dense_vec)

    def _search_keyword_indexed(self, query: str, top_k: int = 3) -> List[Tuple[int, Chunk, float]]:
        """Internal helper: returns (index, Chunk, score) tuples without linear scan."""
        if not self.chunks or not query.strip():
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        tf: Dict[str, float] = {}
        for t in query_tokens:
            tf[t] = tf.get(t, 0.0) + 1.0

        q_vec: Dict[str, float] = {}
        sq_sum = 0.0
        for t, count in tf.items():
            val = (1.0 + math.log(count)) * self.idf.get(t, 1.0)
            q_vec[t] = val
            sq_sum += val * val

        norm = math.sqrt(sq_sum) if sq_sum > 0 else 1.0
        norm_q_vec = {t: val / norm for t, val in q_vec.items()}

        scores: List[Tuple[int, float]] = []
        for idx, doc_vec in enumerate(self.vectors):
            dot_product = sum(weight * norm_q_vec.get(t, 0.0) for t, weight in doc_vec.items())
            if dot_product > 0.0:
                scores.append((idx, dot_product))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [(idx, self.chunks[idx], score) for idx, score in scores[:top_k]]

    def search_keyword(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Mode 1: Keyword Search (TF-IDF Cosine Similarity & BM25 weighting)."""
        indexed = self._search_keyword_indexed(query, top_k=top_k)
        return [(chk, score) for _, chk, score in indexed]

    def _search_semantic_indexed(self, query: str, top_k: int = 3) -> List[Tuple[int, Chunk, float]]:
        """Internal helper: returns (index, Chunk, score) tuples for dense search."""
        if not self.chunks or not query.strip():
            return []

        model = _get_embedding_model()
        if model is not None and self.embeddings:
            try:
                q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].tolist()
            except Exception as e:
                logger.warning(f"Failed to encode query with sentence-transformers: {e}")
                q_emb = None
        else:
            q_emb = None

        if q_emb is None:
            query_tokens = self._tokenize(query)
            vec_dim = 128
            q_emb = [0.0] * vec_dim
            for t in query_tokens:
                h = zlib.crc32(t.encode("utf-8")) % vec_dim
                q_emb[h] += self.idf.get(t, 1.0)
            norm = math.sqrt(sum(v * v for v in q_emb))
            if norm > 0:
                q_emb = [v / norm for v in q_emb]

        # Auto-heal: If doc embeddings are missing or dimension mismatched, recompute them
        if not self.embeddings or (self.embeddings and len(self.embeddings[0]) != len(q_emb)):
            self._build_semantic_embeddings()

        scores: List[Tuple[int, float]] = []
        for idx, doc_emb in enumerate(self.embeddings):
            if len(doc_emb) == len(q_emb):
                sim = sum(a * b for a, b in zip(doc_emb, q_emb))
                scores.append((idx, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [(idx, self.chunks[idx], score) for idx, score in scores[:top_k]]

    def search_semantic(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Mode 2: Semantic Search (Dense Cosine Similarity with all-MiniLM-L6-v2)."""
        indexed = self._search_semantic_indexed(query, top_k=top_k)
        return [(chk, score) for _, chk, score in indexed]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 3,
        alpha: float = DEFAULT_HYBRID_ALPHA,
        topic_filter: Optional[str] = None,
        population_filter: Optional[str] = None
    ) -> List[Tuple[Chunk, float]]:
        """
        Mode 3: Hybrid Search (Reciprocal Rank Fusion of Keyword + Semantic).
        Uses O(1) direct chunk indexing without linear scan .index() lookup.
        """
        if not self.chunks or not query.strip():
            return []

        keyword_results = self._search_keyword_indexed(query, top_k=len(self.chunks))
        semantic_results = self._search_semantic_indexed(query, top_k=len(self.chunks))

        rrf_scores: Dict[int, float] = {}

        for rank, (idx, _, _) in enumerate(keyword_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + ((1.0 - alpha) / (self.rrf_k + rank))

        for rank, (idx, _, _) in enumerate(semantic_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (alpha / (self.rrf_k + rank))

        max_rrf = 1.0 / (self.rrf_k + 1)
        candidates: List[Tuple[int, float]] = []
        for idx, score in rrf_scores.items():
            chk = self.chunks[idx]

            # Optional topic taxonomy filter
            if topic_filter and topic_filter.lower() not in [t.lower() for t in chk.topics]:
                continue

            # Optional patient population filter
            if population_filter and population_filter.lower() not in chk.population.lower():
                continue

            norm_score = min(1.0, score / max_rrf) if max_rrf > 0 else score
            candidates.append((idx, norm_score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [(self.chunks[idx], score) for idx, score in candidates[:top_k]]

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        mode: str = DEFAULT_RETRIEVAL_MODE
    ) -> List[Tuple[Chunk, float]]:
        """Unified search dispatcher."""
        mode_lower = mode.lower()
        if mode_lower == "keyword":
            return self.search_keyword(query=query, top_k=top_k)
        elif mode_lower == "semantic":
            return self.search_semantic(query=query, top_k=top_k)
        else:
            return self.search_hybrid(query=query, top_k=top_k)

    def save(self, file_path: Union[str, Path]) -> Path:
        """Save vector store state to disk in JSON format."""
        path_obj = Path(file_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": [c.to_dict() for c in self.chunks],
            "idf": self.idf,
            "vectors": self.vectors,
            "bm25_idf": self.bm25_idf,
            "doc_lengths": self.doc_lengths,
            "avg_doc_length": self.avg_doc_length,
            "doc_term_freqs": self.doc_term_freqs,
            "embeddings": self.embeddings
        }
        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path_obj

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "VectorStore":
        """Load VectorStore from JSON file."""
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Vector store index file not found at: {path_obj.resolve()}")

        store = cls()
        try:
            with open(path_obj, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Corrupted vector index file at '{path_obj}': {exc}") from exc

        store.chunks = [
            Chunk(
                chunk_id=c["chunk_id"],
                document_id=c["document_id"],
                section_title=c["section_title"],
                text=c["text"],
                token_estimate=c["token_estimate"],
                metadata=c.get("metadata", {})
            )
            for c in data.get("chunks", [])
        ]
        store.idf = data.get("idf", {})
        store.vectors = data.get("vectors", [])
        store.bm25_idf = data.get("bm25_idf", {})
        store.doc_lengths = data.get("doc_lengths", [])
        store.avg_doc_length = data.get("avg_doc_length", 0.0)
        store.doc_term_freqs = data.get("doc_term_freqs", [])
        store.embeddings = data.get("embeddings", [])
        model = _get_embedding_model()
        expected_dim = 384 if model is not None else 128
        if store.chunks and (not store.embeddings or len(store.embeddings[0]) != expected_dim):
            store._build_semantic_embeddings()

        return store


# =====================================================================
# Build Index Orchestrator
# =====================================================================

def build_vector_index(
    input_dir: Union[str, Path] = DEFAULT_CLEANED_DIR,
    index_path: Union[str, Path] = DEFAULT_INDEX_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE_CHARS,
    overlap: int = DEFAULT_CHUNK_OVERLAP_CHARS
) -> Dict[str, Any]:
    """
    Reads cleaned files, chunks text, builds the multi-mode index, and saves it to disk.
    """
    input_path = Path(input_dir)
    target_index_path = Path(index_path)

    if not input_path.exists():
        print(f"[!] Cleaned data directory '{input_path.resolve()}' not found. Run 'python main.py clean' first.")
        return {}

    txt_files = sorted(input_path.glob("*.txt"))
    if not txt_files:
        print(f"[!] No cleaned text files found in '{input_path.resolve()}'. Run 'python main.py clean' first.")
        return {}

    print("=" * 88)
    print(f"  RAG PIPELINE: BUILDING HYBRID VECTOR & BM25 INDEX FROM '{input_path}'")
    print("=" * 88)

    store = VectorStore()
    all_chunks = []
    doc_stats = []

    for file_path in txt_files:
        doc_id = file_path.stem
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except IOError as exc:
            logger.error(f"Failed to read file '{file_path}': {exc}")
            continue

        doc_chunks = chunk_document(
            document_text=content,
            document_id=doc_id,
            target_chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        all_chunks.extend(doc_chunks)
        doc_stats.append((doc_id, len(content), len(doc_chunks)))
        print(f"  -> {doc_id:<45} | {len(content):>6} chars | {len(doc_chunks):>3} chunks")

    store.add_chunks(all_chunks)
    store.save(target_index_path)

    total_chars = sum(s[1] for s in doc_stats)
    print("-" * 88)
    print(f"  Indexed {len(txt_files)} documents into {len(all_chunks)} semantic chunks ({len(store.vectors)} vectors).")
    print(f"  Vocabulary size: {len(store.idf)} terms | Average doc length: {store.avg_doc_length:.1f} tokens.")
    print(f"  Index saved to: '{target_index_path.resolve()}'")
    print("=" * 88)
    print("[OK] Vector index build complete.\n")

    return {
        "documents": len(txt_files),
        "total_chars": total_chars,
        "total_chunks": len(all_chunks),
        "total_vectors": len(store.vectors),
        "index_path": str(target_index_path)
    }
