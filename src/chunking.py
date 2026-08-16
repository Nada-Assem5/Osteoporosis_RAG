"""
Semantic & Structural Chunking Module for Clinical Guidelines RAG.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.config import DEFAULT_CHUNK_SIZE_CHARS, DEFAULT_CHUNK_OVERLAP_CHARS


@dataclass
class Chunk:
    """Represents a text chunk with clinical metadata and lineage."""
    chunk_id: str
    document_id: str
    section_title: str
    text: str
    token_estimate: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def chunk_document(
    document_text: str,
    document_id: str,
    target_chunk_size: int = DEFAULT_CHUNK_SIZE_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_CHARS
) -> List[Chunk]:
    """
    Split a clinical guideline document into structured chunks based on section headers.

    Args:
        document_text: Raw or cleaned document text string.
        document_id: Identifier of the source document.
        target_chunk_size: Soft maximum character count per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List[Chunk]: Structured Chunk dataclass instances with metadata.
    """
    if not document_text.strip():
        return []

    # Section header pattern (Markdown headers, numbered sections, category titles)
    section_pattern = re.compile(r'(?m)^(?:#+\s+[^\n]+|\d+\.\d+\s+[^\n]+|[A-Z][A-Za-z\s]{3,30}:)')
    raw_sections = re.split(r'\n\s*\n', document_text)
    
    chunks: List[Chunk] = []
    current_section = "General Overview"
    chunk_buffer: List[str] = []
    current_len = 0
    chunk_idx = 1

    for para in raw_sections:
        para = para.strip()
        if not para:
            continue

        if section_pattern.match(para) and len(para) < 100:
            current_section = para.replace("#", "").strip()

        para_len = len(para)
        if current_len + para_len > target_chunk_size and chunk_buffer:
            chunk_content = "\n\n".join(chunk_buffer)
            chunks.append(Chunk(
                chunk_id=f"{document_id}_chk_{chunk_idx:03d}",
                document_id=document_id,
                section_title=current_section,
                text=chunk_content,
                token_estimate=len(chunk_content.split()),
                metadata={
                    "char_count": len(chunk_content),
                    "document_id": document_id,
                    "section": current_section
                }
            ))
            chunk_idx += 1
            chunk_buffer = [chunk_buffer[-1]] if chunk_overlap > 0 and len(chunk_buffer) > 1 else []
            current_len = sum(len(p) for p in chunk_buffer)

        chunk_buffer.append(para)
        current_len += para_len

    if chunk_buffer:
        chunk_content = "\n\n".join(chunk_buffer)
        chunks.append(Chunk(
            chunk_id=f"{document_id}_chk_{chunk_idx:03d}",
            document_id=document_id,
            section_title=current_section,
            text=chunk_content,
            token_estimate=len(chunk_content.split()),
            metadata={
                "char_count": len(chunk_content),
                "document_id": document_id,
                "section": current_section
            }
        ))

    return chunks
