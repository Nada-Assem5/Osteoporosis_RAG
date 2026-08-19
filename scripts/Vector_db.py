"""
Stage 4: Vector Database Store & Hybrid Indexer (scripts/Vector_db.py).

Responsibilities:
- Ingests data/processed/chunks.json and data/processed/embeddings.json
- Builds persistent ChromaDB collection under data/processed/chroma_db/
- Generates data/processed/index.json for high-performance Okapi BM25 and Semantic search
- Provides unified VectorStore with multi-mode search (keyword, semantic, hybrid RRF)
- Aligns chunks <-> embeddings strictly by chunk_id (never by list position), since
  Stage 3 may reject invalid vectors and produce a shorter embeddings list
- Preserves embedding_method/embedding_model provenance through to index.json and
  the Evidence Panel, per the "each output auditable" architecture requirement
- Returns RetrievedChunk records (chunk + score + rank + method) matching the
  schema used by the Evidence Panel UI and Precision@K evaluation
"""

import os
import sys
import math
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.schema import Chunk, RetrievedChunk
from src.utils import compute_content_hash, count_tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = ROOT_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed")
CHUNKS_JSON_PATH = PROCESSED_DATA_DIR / "chunks.json"
EMBEDDINGS_JSON_PATH = PROCESSED_DATA_DIR / "embeddings.json"
INDEX_JSON_PATH = PROCESSED_DATA_DIR / "index.json"
CHROMA_DB_DIR = ROOT_DIR / os.getenv("CHROMA_DB_DIR", "data/processed/chroma_db")

FALLBACK_METHOD_LABEL = "deterministic_fallback_NON_SEMANTIC"


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words."""
    import re
    return re.findall(r'\b[a-zA-Z0-9_]{2,}\b', text.lower()) if text else []


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not vec_a or not vec_b:
        return 0.0
    if len(vec_a) != len(vec_b):
        logger.warning(
            f"[WARN] Cosine similarity requested between vectors of mismatched dimension "
            f"({len(vec_a)} vs {len(vec_b)}). Returning 0.0 - check embedding model consistency."
        )
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


class BM25Index:
    """Lightweight pure-Python Okapi BM25 index."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: List[int] = []
        self.avg_len: float = 0.0
        self.doc_freq: Dict[str, int] = {}
        self.num_docs: int = 0
        self.doc_tokens: List[List[str]] = []

    def fit(self, corpus_tokens: List[List[str]]):
        self.doc_tokens = corpus_tokens
        self.num_docs = len(corpus_tokens)
        if self.num_docs == 0:
            return
        self.doc_len = [len(doc) for doc in corpus_tokens]
        self.avg_len = sum(self.doc_len) / self.num_docs if self.num_docs else 1.0

        self.doc_freq = {}
        for doc in corpus_tokens:
            for term in set(doc):
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

    def score(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.num_docs
        for term in query_tokens:
            if term not in self.doc_freq:
                continue
            df = self.doc_freq[term]
            idf = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))
            for i, doc in enumerate(self.doc_tokens):
                tf = doc.count(term)
                if tf == 0:
                    continue
                num = tf * (self.k1 + 1)
                denom = tf + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / (self.avg_len or 1.0)))
                scores[i] += idf * (num / denom)
        return scores


class VectorStore:
    """
    Hybrid Vector Store providing Keyword (BM25), Semantic (Dense), and Hybrid (RRF) search.
    """
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.embeddings: List[Optional[List[float]]] = []  # parallel to self.chunks; None = no vector
        self.bm25 = BM25Index()
        self.chroma_collection = None
        self._embedding_model = None
        self._query_embedding_method: Optional[str] = None
        self.corpus_has_fallback_embeddings: bool = False

    def add_chunks(self, chunks: List[Union[Chunk, Dict[str, Any]]], embeddings: Optional[List[List[float]]] = None):
        """
        Add chunks and optional precomputed embeddings to the store.

        Alignment contract: if a per-chunk dict already carries an "embedding" key
        (as produced by Stage 3), that vector is used and kept strictly parallel
        (same index) to self.chunks - a chunk with no "embedding" key gets a
        None placeholder rather than silently shifting later vectors out of
        position. The optional `embeddings` positional list (used by
        query_vector_store when reading index.json) is only applied when it is
        exactly as long as `chunks`.
        """
        start_idx = len(self.chunks)

        for c in chunks:
            if isinstance(c, dict):
                meta = dict(c.get("metadata", {}))
                # Preserve embedding provenance for auditability even though
                # Chunk itself has no dedicated field for it.
                if "embedding_method" in c:
                    meta["embedding_method"] = c["embedding_method"]
                if "embedding_model" in c:
                    meta["embedding_model"] = c["embedding_model"]

                obj = Chunk(
                    chunk_id=c.get("chunk_id", ""),
                    document_name=c.get("document_name", ""),
                    section_title=c.get("section_title", "Unknown Section"),
                    text=c.get("text", ""),
                    page_number=c.get("page_number", 1),
                    token_estimate=c.get("token_estimate", 0),
                    source_url=c.get("source_url"),
                    document_id=c.get("document_id"),
                    metadata=meta
                )
                self.chunks.append(obj)
                self.embeddings.append(c.get("embedding"))  # None if absent, not silently skipped

                if meta.get("embedding_method") == FALLBACK_METHOD_LABEL:
                    self.corpus_has_fallback_embeddings = True
            else:
                self.chunks.append(c)
                self.embeddings.append(None)
                if c.metadata.get("embedding_method") == FALLBACK_METHOD_LABEL:
                    self.corpus_has_fallback_embeddings = True

        # If a separate positional embeddings list was passed (e.g. from
        # index.json's "embeddings" array), apply it only when lengths match
        # exactly the batch just added - otherwise we cannot trust positional
        # alignment and we keep whatever was embedded inline per-chunk instead.
        if embeddings is not None:
            batch_len = len(self.chunks) - start_idx
            if len(embeddings) == batch_len:
                for offset, vec in enumerate(embeddings):
                    if self.embeddings[start_idx + offset] is None:
                        self.embeddings[start_idx + offset] = vec
            else:
                logger.warning(
                    f"[WARN] Positional 'embeddings' list length ({len(embeddings)}) does not match "
                    f"the number of chunks just added ({batch_len}). Ignoring positional list to avoid "
                    f"misaligning vectors with the wrong chunks; relying on inline per-chunk 'embedding' "
                    f"fields only."
                )

        missing_vectors = sum(1 for v in self.embeddings[start_idx:] if v is None)
        if missing_vectors:
            logger.warning(
                f"[WARN] {missing_vectors} of {len(self.chunks) - start_idx} newly added chunk(s) have no "
                f"embedding vector. Semantic/hybrid search will treat these as zero-similarity."
            )

        # Index BM25 over the full current corpus
        corpus_tokens = [_tokenize(c.text) for c in self.chunks]
        self.bm25.fit(corpus_tokens)

    def _get_query_embedding(self, query: str) -> Tuple[List[float], str]:
        """Encode query text into a dense vector. Returns (vector, method_used)."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
                self._embedding_model = SentenceTransformer(model_name)
            except Exception:
                self._embedding_model = "fallback"

        if self._embedding_model != "fallback":
            emb = self._embedding_model.encode([query], normalize_embeddings=True)[0]
            return [float(x) for x in emb], "sentence-transformers"
        else:
            from scripts.Embeddings import _deterministic_hash_embedding
            return _deterministic_hash_embedding(query), FALLBACK_METHOD_LABEL

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 3,
        alpha: float = 0.5
    ) -> List[RetrievedChunk]:
        """
        Search chunks using 'keyword', 'semantic', or 'hybrid' (Reciprocal Rank Fusion) mode.
        Returns RetrievedChunk records (chunk + similarity_score + rank + retrieval_method)
        ready for the Evidence Panel UI and Precision@K evaluation.
        """
        if not self.chunks:
            return []

        q_tokens = _tokenize(query)
        bm25_scores = self.bm25.score(q_tokens) if q_tokens else [0.0] * len(self.chunks)

        vectors_available = sum(1 for v in self.embeddings if v is not None) == len(self.chunks) and len(self.chunks) > 0

        # Semantic scores
        sem_scores = [0.0] * len(self.chunks)
        query_embedding_method = None
        if mode in {"semantic", "hybrid"}:
            if vectors_available:
                q_vec, query_embedding_method = self._get_query_embedding(query)
                if self.corpus_has_fallback_embeddings or query_embedding_method == FALLBACK_METHOD_LABEL:
                    logger.warning(
                        "[WARN] Semantic search is using NON-SEMANTIC fallback vectors for the query and/or "
                        "corpus. Similarity scores are not meaningful until real embeddings are regenerated."
                    )
                sem_scores = [
                    _cosine_similarity(q_vec, emb) if emb is not None else 0.0
                    for emb in self.embeddings
                ]
            else:
                logger.warning(
                    "[WARN] Corpus is missing embeddings for one or more chunks - falling back to BM25 "
                    "scores in place of semantic scores for this query."
                )
                sem_scores = [float(s) for s in bm25_scores]

        if mode == "keyword":
            max_s = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1.0
            norm_scores = [s / max_s for s in bm25_scores]
            ranked = sorted(enumerate(norm_scores), key=lambda x: x[1], reverse=True)
            return [
                RetrievedChunk(chunk=self.chunks[i], similarity_score=score, rank=rank + 1, retrieval_method="keyword")
                for rank, (i, score) in enumerate(ranked[:top_k])
            ]

        elif mode == "semantic":
            ranked = sorted(enumerate(sem_scores), key=lambda x: x[1], reverse=True)
            return [
                RetrievedChunk(chunk=self.chunks[i], similarity_score=max(0.0, score), rank=rank + 1, retrieval_method="semantic")
                for rank, (i, score) in enumerate(ranked[:top_k])
            ]

        else:  # hybrid RRF
            kw_ranked = [i for i, _ in sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)]
            sem_ranked = [i for i, _ in sorted(enumerate(sem_scores), key=lambda x: x[1], reverse=True)]

            rrf_scores = [0.0] * len(self.chunks)
            k_const = 60
            for rank, idx in enumerate(kw_ranked):
                if bm25_scores[idx] > 0:
                    rrf_scores[idx] += (1.0 - alpha) * (1.0 / (k_const + rank + 1))
            for rank, idx in enumerate(sem_ranked):
                if sem_scores[idx] > 0:
                    rrf_scores[idx] += alpha * (1.0 / (k_const + rank + 1))

            max_rrf = max(rrf_scores) if rrf_scores and max(rrf_scores) > 0 else 1.0
            norm_rrf = [s / max_rrf for s in rrf_scores]
            ranked = sorted(enumerate(norm_rrf), key=lambda x: x[1], reverse=True)
            return [
                RetrievedChunk(chunk=self.chunks[i], similarity_score=score, rank=rank + 1, retrieval_method="hybrid")
                for rank, (i, score) in enumerate(ranked[:top_k])
            ]


def build_chroma_database(
    chunks: List[Dict[str, Any]],
    embeddings: List[Dict[str, Any]],
    persist_dir: Optional[Path] = None
) -> VectorStore:
    """
    Builds index.json and syncs ChromaDB persistent collection.

    Chunks and embeddings are joined strictly by chunk_id (never by list
    position), because Stage 3 (Embeddings.py) may reject invalid vectors and
    return a shorter list than the original chunks - a positional join would
    silently attach the wrong vector to the wrong chunk.
    """
    p_dir = Path(persist_dir) if persist_dir else CHROMA_DB_DIR
    p_dir.mkdir(parents=True, exist_ok=True)

    # Build a chunk_id -> embedded_record lookup for safe joining
    embeddings_by_id: Dict[str, Dict[str, Any]] = {
        e["chunk_id"]: e for e in embeddings if e.get("chunk_id")
    }

    joined_records: List[Dict[str, Any]] = []
    missing_embedding_ids: List[str] = []
    for c in chunks:
        cid = c.get("chunk_id", "")
        emb_record = embeddings_by_id.get(cid)
        merged = dict(c)
        if emb_record:
            merged["embedding"] = emb_record.get("embedding")
            merged["embedding_method"] = emb_record.get("embedding_method")
            merged["embedding_model"] = emb_record.get("embedding_model")
        else:
            missing_embedding_ids.append(cid)
        joined_records.append(merged)

    if missing_embedding_ids:
        logger.warning(
            f"[WARN] {len(missing_embedding_ids)} chunk(s) have no matching embedding "
            f"(e.g. rejected in Stage 3) and will be indexed for keyword search only: "
            f"{missing_embedding_ids[:5]}{'...' if len(missing_embedding_ids) > 5 else ''}"
        )

    store = VectorStore()
    store.add_chunks(joined_records)

    # Persist ChromaDB - only chunks with a valid embedding go into the ANN index,
    # since ChromaDB requires embeddings for every upserted record here.
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(p_dir))
        collection = client.get_or_create_collection(
            name="clinical_guidelines",
            metadata={"hnsw:space": "cosine"}
        )
        embeddable_records = [r for r in joined_records if r.get("embedding") is not None]
        if embeddable_records:
            ids = [r["chunk_id"] for r in embeddable_records]
            texts = [r["text"] for r in embeddable_records]
            metadatas = [{
                "document_id": r.get("document_id", "") or "",
                "document_name": r.get("document_name", ""),
                "section_title": r.get("section_title", "Unknown Section"),
                "page_number": r.get("page_number", 1),
                "source_url": r.get("source_url") or "",
                "embedding_method": r.get("embedding_method") or "",
                "embedding_model": r.get("embedding_model") or ""
            } for r in embeddable_records]
            embs = [r["embedding"] for r in embeddable_records]

            collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embs
            )
            logger.info(f"ChromaDB persistent store updated with {len(ids)} chunks at '{p_dir}'.")
    except Exception as exc:
        logger.warning(f"ChromaDB store initialization skipped ('{exc}'). Local memory store active.")

    # Save index.json - embeddings kept strictly parallel to store.chunks (None where absent)
    index_data = {
        "chunks": [c.to_dict() for c in store.chunks],
        "embeddings": store.embeddings
    }
    with open(INDEX_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2, ensure_ascii=False)

    return store


def query_vector_store(query: str, top_k: int = 3, mode: str = "hybrid") -> List[RetrievedChunk]:
    """Helper to query the persisted index.json store."""
    if not INDEX_JSON_PATH.exists():
        logger.error(f"[ERROR] Index file not found at '{INDEX_JSON_PATH}'. Run Stage 4 (Vector_db.py) first.")
        return []
    with open(INDEX_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    store = VectorStore()
    chunk_dicts = data.get("chunks", [])
    stored_embeddings = data.get("embeddings", [])

    # Re-attach embeddings positionally since index.json guarantees this
    # parallel ordering (both arrays are written from the same store.chunks
    # iteration in build_chroma_database).
    if len(stored_embeddings) == len(chunk_dicts):
        for c_dict, vec in zip(chunk_dicts, stored_embeddings):
            if vec is not None:
                c_dict["embedding"] = vec
    else:
        logger.warning(
            "[WARN] index.json 'chunks' and 'embeddings' arrays have mismatched lengths - "
            "loading chunks without vectors; semantic/hybrid search will fall back to BM25."
        )

    store.add_chunks(chunk_dicts)
    return store.search(query=query, mode=mode, top_k=top_k)


def run_vector_db(
    chunks_path: Optional[Path] = None,
    embeddings_path: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> VectorStore:
    """Stage 4 execution runner."""
    c_path = Path(chunks_path) if chunks_path else CHUNKS_JSON_PATH
    e_path = Path(embeddings_path) if embeddings_path else EMBEDDINGS_JSON_PATH
    out_dir = Path(output_dir) if output_dir else CHROMA_DB_DIR

    if not c_path.exists():
        logger.error(f"[ERROR] chunks.json not found at '{c_path}'. Run Stage 2 (Chunk.py) first.")
        return VectorStore()

    with open(c_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = []
    if e_path.exists():
        with open(e_path, "r", encoding="utf-8") as f:
            embeddings = json.load(f)
    else:
        logger.warning(f"[WARN] embeddings.json not found at '{e_path}'. Index will support keyword search only.")

    store = build_chroma_database(chunks=chunks, embeddings=embeddings, persist_dir=out_dir)

    print("\n" + "_" * 90)
    print("  STAGE 4: VECTOR DATABASE & HYBRID INDEXING SUMMARY")
    print("_" * 90)
    print(f"  Indexed Chunks Count       : {len(store.chunks)}")
    vector_count = sum(1 for v in store.embeddings if v is not None)
    print(f"  Chunks With Embeddings     : {vector_count} / {len(store.chunks)}")
    if store.corpus_has_fallback_embeddings:
        print("  " + "!" * 86)
        print("  WARNING: SOME INDEXED CHUNKS USE NON-SEMANTIC FALLBACK EMBEDDINGS.")
        print("  Semantic/hybrid search quality for those chunks will be effectively random.")
        print("  " + "!" * 86)
    print(f"  Persistent ChromaDB Path   : {out_dir}")
    print(f"  Unified Index JSON Path    : {INDEX_JSON_PATH}")
    print("=" * 90 + "\n")

    return store


run = run_vector_db
main = run_vector_db


if __name__ == "__main__":
    run_vector_db()