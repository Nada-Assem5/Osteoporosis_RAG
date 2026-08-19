"""
Shared Data Contracts & Schemas (src/schema.py).

Defines core pipeline dataclasses and canonical confidence tiers.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


class ConfidenceTier(str, Enum):
    """Canonical 4-Tier Confidence Rating standard for clinical synthesis."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INSUFFICIENT_EVIDENCE = "Insufficient Evidence"


class QueryRiskCategory(str, Enum):
    """
    Input Risk Classification tiers per the Safety & Guardrail Workflow spec:
    queries are categorized as Allowed, Needs Caution (patient scenarios),
    or Refuse/Redirect (emergencies, out-of-scope).
    """
    ALLOWED = "Allowed"
    NEEDS_CAUTION = "Needs Caution"
    REFUSE_REDIRECT = "Refuse/Redirect"


@dataclass
class Page:
    """Represents an extracted document page with structural elements and metadata."""
    page_number: int
    text: str
    elements: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Represents a section-aware semantic chunk with complete metadata schema."""
    chunk_id: str
    document_name: str
    section_title: str
    text: str
    page_number: int = 1
    token_estimate: int = 0
    source_url: Optional[str] = None
    document_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id or self.document_name,
            "document_name": self.document_name,
            "section_title": self.section_title,
            "page_number": self.page_number,
            "source_url": self.source_url,
            "text": self.text,
            "token_estimate": self.token_estimate,
            "metadata": self.metadata
        }


@dataclass
class RetrievedChunk:
    """
    A Chunk paired with its query-time retrieval score and rank, for the
    Evidence Panel UI ("shows retrieved chunks, scores & metadata before
    generation") and for Precision@K evaluation.
    """
    chunk: Chunk
    similarity_score: float
    rank: int
    retrieval_method: str = "semantic"  # "semantic", "keyword", "hybrid", "reranked"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "similarity_score": round(self.similarity_score, 4),
            "rank": self.rank,
            "retrieval_method": self.retrieval_method
        }


@dataclass
class Citation:
    """
    Enforces the citation schema required by the agenda's Citation Mechanics
    spec: Document + Section + Page Number + Chunk ID, plus a short quoted
    excerpt used as supporting evidence.
    """
    document_name: str
    section_title: str
    page_number: int
    chunk_id: str
    quoted_excerpt: str
    source_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_name": self.document_name,
            "section_title": self.section_title,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "quoted_excerpt": self.quoted_excerpt,
            "source_url": self.source_url
        }

    @classmethod
    def from_chunk(cls, chunk: Chunk, quoted_excerpt: str) -> "Citation":
        """Build a schema-compliant Citation directly from a source Chunk."""
        return cls(
            document_name=chunk.document_name,
            section_title=chunk.section_title,
            page_number=chunk.page_number,
            chunk_id=chunk.chunk_id,
            quoted_excerpt=quoted_excerpt,
            source_url=chunk.source_url
        )


@dataclass
class GuardrailAssessment:
    """
    Captures the outcome of the 3-step Safety & Guardrail Workflow:
    (1) Input Risk Classification, (2) Retrieval Confidence Thresholds,
    (3) Unsupported Claim Detection.
    """
    risk_category: QueryRiskCategory
    retrieval_confidence_ok: bool
    min_similarity_score: Optional[float] = None
    unsupported_claims: List[str] = field(default_factory=list)
    blocked: bool = False
    block_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_category": self.risk_category.value,
            "retrieval_confidence_ok": self.retrieval_confidence_ok,
            "min_similarity_score": self.min_similarity_score,
            "unsupported_claims": self.unsupported_claims,
            "blocked": self.blocked,
            "block_reason": self.block_reason
        }


@dataclass
class ClinicalSynthesisResponse:
    """Structured clinical response schema with canonical confidence tiers and full lineage."""
    query: str
    direct_answer: str
    target_population: str
    key_recommendations: List[str]
    citations: List[Citation]
    evidence_strength: ConfidenceTier
    clinical_caveats: List[str]
    provider: str = "gemini"
    model: str = "gemini-1.5-flash"
    guardrail_warnings: List[str] = field(default_factory=list)
    disclaimer: str = (
        "CLINICAL DISCLAIMER: This evidence synthesis is generated from official clinical practice "
        "guidelines for informational and clinical decision support purposes only. "
        "It does not replace individualized clinical judgment, multidisciplinary review, or local protocols."
    )

    def __post_init__(self):
        # Accept a plain string for evidence_strength for caller convenience,
        # but always normalize to the canonical ConfidenceTier enum.
        if isinstance(self.evidence_strength, str) and not isinstance(self.evidence_strength, ConfidenceTier):
            try:
                self.evidence_strength = ConfidenceTier(self.evidence_strength)
            except ValueError:
                self.evidence_strength = ConfidenceTier.INSUFFICIENT_EVIDENCE

        # Accept raw dicts for citations for caller convenience, but enforce
        # the Document + Section + Page Number + Chunk ID schema.
        normalized_citations: List[Citation] = []
        for c in self.citations:
            if isinstance(c, Citation):
                normalized_citations.append(c)
            elif isinstance(c, dict):
                normalized_citations.append(Citation(
                    document_name=c.get("document_name", ""),
                    section_title=c.get("section_title", ""),
                    page_number=c.get("page_number", 0),
                    chunk_id=c.get("chunk_id", ""),
                    quoted_excerpt=c.get("quoted_excerpt", ""),
                    source_url=c.get("source_url")
                ))
        self.citations = normalized_citations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "direct_answer": self.direct_answer,
            "target_population": self.target_population,
            "key_recommendations": self.key_recommendations,
            "citations": [c.to_dict() for c in self.citations],
            "evidence_strength": self.evidence_strength.value,
            "clinical_caveats": self.clinical_caveats,
            "provider": self.provider,
            "model": self.model,
            "guardrail_warnings": self.guardrail_warnings,
            "disclaimer": self.disclaimer
        }


@dataclass
class EvalQuestion:
    """Categorized clinical benchmark question with target ground-truth chunk IDs."""
    query: str
    expected_chunk_ids: List[str]
    category: str = "direct"  # "direct", "multi_chunk", "ambiguous", "out_of_scope"
    expected_behavior: Optional[str] = None
    notes: Optional[str] = None