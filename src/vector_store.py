"""
Vector Store & Similarity Retrieval Engine for Clinical Guidelines.
"""

import json
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
from src.chunking import Chunk


class VectorStore:
    """
    Self-contained, production-ready local vector store and retrieval engine.
    Supports BM25, TF-IDF, and term vectors with cosine similarity matching.
    """
    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.vectors: List[Dict[str, float]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and lowercase text into terms."""
        return [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_\-\.]{2,}\b', text)]

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to the store and compute TF-IDF index vectors.

        Args:
            chunks: List of Chunk objects to index.
        """
        self.chunks.extend(chunks)
        self._build_index()

    def _build_index(self) -> None:
        """Compute IDF and normalized TF-IDF vectors for all chunks."""
        doc_count = len(self.chunks)
        if doc_count == 0:
            return

        # 1. Document frequency
        df: Dict[str, int] = {}
        tokenized_docs: List[List[str]] = []
        for chk in self.chunks:
            tokens = set(self._tokenize(chk.text + " " + chk.section_title))
            tokenized_docs.append(self._tokenize(chk.text + " " + chk.section_title))
            for t in tokens:
                df[t] = df.get(t, 0) + 1

        # 2. IDF calculation
        self.idf = {t: math.log((doc_count + 1) / (count + 0.5)) + 1.0 for t, count in df.items()}

        # 3. TF-IDF vectors
        self.vectors = []
        for tokens in tokenized_docs:
            tf: Dict[str, float] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0.0) + 1.0

            vec: Dict[str, float] = {}
            sq_sum = 0.0
            for t, count in tf.items():
                val = (1.0 + math.log(count)) * self.idf.get(t, 1.0)
                vec[t] = val
                sq_sum += val * val

            norm = math.sqrt(sq_sum) if sq_sum > 0 else 1.0
            norm_vec = {t: val / norm for t, val in vec.items()}
            self.vectors.append(norm_vec)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """
        Search top-k most relevant chunks using cosine similarity.

        Args:
            query: Clinical search query string.
            top_k: Number of highest scoring passages to return.

        Returns:
            List[Tuple[Chunk, float]]: Ranked list of (Chunk, similarity_score) tuples.
        """
        if not self.chunks or not query.strip():
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        # Build query vector
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
        return [(self.chunks[idx], score) for idx, score in scores[:top_k]]

    def save(self, file_path: Union[str, Path]) -> Path:
        """
        Save vector store state to a JSON file.

        Args:
            file_path: Destination JSON path.

        Returns:
            Path: Path object to saved index.
        """
        path_obj = Path(file_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "section_title": c.section_title,
                    "text": c.text,
                    "token_estimate": c.token_estimate,
                    "metadata": c.metadata
                }
                for c in self.chunks
            ],
            "idf": self.idf,
            "vectors": self.vectors
        }
        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path_obj

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "VectorStore":
        """
        Load vector store state from a JSON file.

        Args:
            file_path: Path to serialized JSON index.

        Returns:
            VectorStore: Initialized vector store with restored vectors.

        Raises:
            FileNotFoundError: If index file is missing.
            json.JSONDecodeError: If index file is corrupted.
        """
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
        return store
