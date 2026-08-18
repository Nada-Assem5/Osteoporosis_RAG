"""
Step 2: Semantic & Structural Chunking Module with Clinical Metadata Enrichment.

Responsibilities:
- Sentence-aware semantic boundary segmentation
- Multi-tiered clinical metadata enrichment (target population, clinical topic taxonomy, recommendation grade)
- Source provenance and page lineage attribution (document_name, page_number, section_title, chunk_id, source_url)
- Section hierarchy preservation for retrieval grounding
- Chunk dataclass contract representation
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.config import (
    DEFAULT_CHUNK_SIZE_CHARS,
    DEFAULT_CHUNK_OVERLAP_CHARS,
    POPULATION_PATTERNS,
    TOPIC_KEYWORDS,
    GUIDELINE_SOURCE_URLS
)


@dataclass
class Chunk:
    """Represents a text chunk with clinical metadata, taxonomy tags, and lineage."""
    chunk_id: str
    document_id: str
    section_title: str
    text: str
    token_estimate: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def document_name(self) -> str:
        return self.metadata.get("document_name", self.document_id)

    @property
    def page_number(self) -> int:
        return self.metadata.get("page_number", 1)

    @property
    def source_url(self) -> Optional[str]:
        return self.metadata.get("source_url")

    @property
    def topics(self) -> List[str]:
        return self.metadata.get("topics", [])

    @property
    def population(self) -> str:
        return self.metadata.get("population", "General Population")

    @property
    def recommendation_grade(self) -> Optional[str]:
        return self.metadata.get("recommendation_grade")

    @property
    def guideline_issuer(self) -> str:
        return self.metadata.get("guideline_issuer", "Clinical Practice Guideline")

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk into a serializable dictionary matching the required metadata schema."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "section_title": self.section_title,
            "page_number": self.page_number,
            "source_url": self.source_url,
            "text": self.text,
            "token_estimate": self.token_estimate,
            "metadata": self.metadata
        }


def extract_clinical_metadata(
    text: str,
    section_title: str,
    doc_id: str,
    page_number: int = 1,
    source_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Extracts clinical taxonomy tags, population targets, and guideline attributes from text.
    """
    combined_content = f"{section_title} {text}".lower()

    # 1. Detect Clinical Topics
    detected_topics: List[str] = []
    for topic_name, keywords in TOPIC_KEYWORDS.items():
        if any(kw in combined_content for kw in keywords):
            detected_topics.append(topic_name)
    if not detected_topics:
        detected_topics.append("General Osteoporosis Care")

    # 2. Detect Target Population
    target_pop = "General Adult Population"
    for pop_key, pattern in POPULATION_PATTERNS.items():
        if re.search(pattern, combined_content, re.IGNORECASE):
            if pop_key == "postmenopausal_women":
                target_pop = "Postmenopausal Women / Women >= 65"
            elif pop_key == "older_men":
                target_pop = "Older Men (>= 70)"
            elif pop_key == "high_risk_adults":
                target_pop = "Adults with Prior Fragility Fractures / High Risk"
            break

    # 3. Detect Recommendation Grade or Number (e.g. 'Grade B', '1.4.1')
    grade_match = re.search(r'\b(Grade\s+[A-DI]|Recommendation\s+\d+(\.\d+)*|\d+\.\d+\.\d+)\b', text, re.IGNORECASE)
    rec_grade = grade_match.group(1) if grade_match else None

    # 4. Identify Source Guideline Issuer
    guideline_issuer = "Official Clinical Practice Guideline"
    if "nice" in doc_id.lower() or "66144025463749" in doc_id:
        guideline_issuer = "NICE Guideline NG259 (UK National Institute for Health and Care Excellence)"
    elif "screening" in doc_id.lower() or "uspstf" in doc_id.lower():
        guideline_issuer = "USPSTF (US Preventive Services Task Force)"

    # 5. Resolve Source URL
    resolved_url = source_url or GUIDELINE_SOURCE_URLS.get(doc_id)

    return {
        "char_count": len(text),
        "document_id": doc_id,
        "document_name": doc_id,
        "section": section_title,
        "section_title": section_title,
        "page_number": page_number,
        "source_url": resolved_url,
        "topics": detected_topics,
        "population": target_pop,
        "recommendation_grade": rec_grade,
        "guideline_issuer": guideline_issuer
    }


def _split_into_sentences(paragraph: str) -> List[str]:
    """Splits a paragraph into clean sentences, preserving list item integrity."""
    # Split on period/question/exclamation followed by space and uppercase letter/digit
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', paragraph)
    res = [s.strip() for s in sentences if s.strip()]
    return res if res else [paragraph.strip()]


def chunk_document(
    document_text: str,
    document_id: str,
    target_chunk_size: int = DEFAULT_CHUNK_SIZE_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_CHARS,
    page_number: int = 1,
    source_url: Optional[str] = None
) -> List[Chunk]:
    """
    Step 2 Main Orchestrator: Splits a clinical guideline document into
    structured, sentence-bounded chunks with rich clinical metadata.
    """
    if not document_text or not document_text.strip():
        return []

    section_pattern = re.compile(r'(?m)^(?:#+\s+[^\n]+|\d+\.\d+\s+[^\n]+|[A-Z][A-Za-z\s]{3,35}:)')
    raw_paragraphs = re.split(r'\n\s*\n', document_text)

    chunks: List[Chunk] = []
    current_section = "General Overview"
    chunk_buffer: List[str] = []
    current_len = 0
    chunk_idx = 1
    current_pg = page_number

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        if section_pattern.match(para) and len(para) < 120:
            current_section = para.replace("#", "").strip()

        sentences = _split_into_sentences(para)
        for sent in sentences:
            sent_len = len(sent)

            # If a single sentence exceeds target_chunk_size, break it into smaller sub-parts
            if sent_len > target_chunk_size:
                if chunk_buffer:
                    chunk_content = "\n\n".join(chunk_buffer)
                    meta = extract_clinical_metadata(
                        text=chunk_content,
                        section_title=current_section,
                        doc_id=document_id,
                        page_number=current_pg,
                        source_url=source_url
                    )
                    chunks.append(Chunk(
                        chunk_id=f"{document_id}_chk_{chunk_idx:03d}",
                        document_id=document_id,
                        section_title=current_section,
                        text=chunk_content,
                        token_estimate=len(chunk_content.split()),
                        metadata=meta
                    ))
                    chunk_idx += 1
                    chunk_buffer = []
                    current_len = 0

                sub_start = 0
                step = max(100, target_chunk_size - chunk_overlap)
                while sub_start < sent_len:
                    sub_end = min(sent_len, sub_start + target_chunk_size)
                    sub_text = sent[sub_start:sub_end].strip()
                    if sub_text:
                        meta = extract_clinical_metadata(
                            text=sub_text,
                            section_title=current_section,
                            doc_id=document_id,
                            page_number=current_pg,
                            source_url=source_url
                        )
                        chunks.append(Chunk(
                            chunk_id=f"{document_id}_chk_{chunk_idx:03d}",
                            document_id=document_id,
                            section_title=current_section,
                            text=sub_text,
                            token_estimate=len(sub_text.split()),
                            metadata=meta
                        ))
                        chunk_idx += 1
                    sub_start += step
                continue

            if current_len + sent_len > target_chunk_size and chunk_buffer:
                chunk_content = "\n\n".join(chunk_buffer)
                meta = extract_clinical_metadata(
                    text=chunk_content,
                    section_title=current_section,
                    doc_id=document_id,
                    page_number=current_pg,
                    source_url=source_url
                )
                chunks.append(Chunk(
                    chunk_id=f"{document_id}_chk_{chunk_idx:03d}",
                    document_id=document_id,
                    section_title=current_section,
                    text=chunk_content,
                    token_estimate=len(chunk_content.split()),
                    metadata=meta
                ))
                chunk_idx += 1

                if chunk_overlap > 0 and len(chunk_buffer) > 1:
                    chunk_buffer = [chunk_buffer[-1]]
                    current_len = len(chunk_buffer[0])
                else:
                    chunk_buffer = []
                    current_len = 0

            chunk_buffer.append(sent)
            current_len += sent_len

    if chunk_buffer:
        chunk_content = "\n\n".join(chunk_buffer)
        meta = extract_clinical_metadata(
            text=chunk_content,
            section_title=current_section,
            doc_id=document_id,
            page_number=current_pg,
            source_url=source_url
        )
        chunks.append(Chunk(
            chunk_id=f"{document_id}_chk_{chunk_idx:03d}",
            document_id=document_id,
            section_title=current_section,
            text=chunk_content,
            token_estimate=len(chunk_content.split()),
            metadata=meta
        ))

    return chunks
