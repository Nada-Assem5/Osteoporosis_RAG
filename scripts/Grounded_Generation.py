"""
Stage 6: Grounded Clinical Generation & Claim Grounding Verification (scripts/Grounded_Generation.py).

Responsibilities:
- Takes user query and optional custom generation prompt (positional, keyword, or interactive)
- Takes pre-generation Evidence Panel as strict source-of-truth context
- Swappable LLM execution: Google Gemini (gemini-1.5-flash) via call_llm() with robust structured JSON synthesis and deterministic fallback
- Strict Evidence Synthesizer persona (never an autonomous diagnostician)
- Produces 4 canonical output sections:
  1. RECOMMENDATION (actionable guideline guidance with inline citations)
  2. SUPPORTING EVIDENCE (direct quoted excerpts from retrieved passages)
  3. CITATIONS (document_name + section_title + page_number + chunk_id + source_url)
  4. CONFIDENCE LEVEL (High / Medium / Low / Insufficient Evidence + safety disclaimer)
- Retrieval Confidence Score Gating: delegates to Retrieval.py's assess_retrieval_confidence()
  (Safety Workflow step 2) as the single source of truth, so this stage never disagrees
  with the Retrieval stage about whether generation should proceed
- Post-processing Claim Grounding: strips unsupported claims failing lexical overlap
  verification, and records every stripped claim in guardrail_warnings for auditability
"""

import os
import sys
import re
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Load environment configuration
load_dotenv(ROOT_DIR / ".env")
load_dotenv()

from src.schema import Chunk, ConfidenceTier, ClinicalSynthesisResponse, QueryRiskCategory
from src.utils import compute_content_hash, count_tokens

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Configurable Paths & Parameters
PROCESSED_DATA_DIR = ROOT_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed")
INDEX_JSON_PATH = PROCESSED_DATA_DIR / "index.json"

DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "gemini")
DEFAULT_GEMINI_MODEL = os.getenv("DEFAULT_GEMINI_MODEL", "gemini-1.5-flash")
UNSUPPORTED_CLAIM_OVERLAP_THRESHOLD = float(os.getenv("UNSUPPORTED_CLAIM_OVERLAP_THRESHOLD", "0.25"))

CONFIDENCE_SCORE_HIGH = float(os.getenv("CONFIDENCE_SCORE_HIGH", "0.60"))
CONFIDENCE_SCORE_MEDIUM = float(os.getenv("CONFIDENCE_SCORE_MEDIUM", "0.30"))
CONFIDENCE_SCORE_LOW = float(os.getenv("CONFIDENCE_SCORE_LOW", "0.015"))
# NOTE: kept for display/tier purposes only. The actual block/proceed decision
# is now delegated entirely to Retrieval.assess_retrieval_confidence() (see
# the FIX note in generate_grounded_clinical_response below), so this constant
# no longer drives an independent gating check here.
MIN_SYNTHESIS_SCORE_THRESHOLD = CONFIDENCE_SCORE_LOW

DEFAULT_GENERATION_PROMPT = (
    "Synthesize the official clinical practice guideline recommendations to directly answer the clinical query, "
    "providing clear, actionable guidance and contraindications with inline [chunk_id] citations."
)


# =====================================================================
# Domain Scope Guard
# =====================================================================

# This RAG is intentionally limited to the osteoporosis / bone-health
# guideline corpus. Retrieval similarity alone must NOT be used to decide
# whether a question belongs to this domain, because generic PDF text
# (tables of contents, headings, boilerplate, etc.) can receive a high
# semantic similarity score for completely unrelated questions.
DOMAIN_TERMS = {
    "osteoporosis", "osteoporotic", "osteopenia",
    "bone", "bone health", "bone density", "bone mass",
    "fracture", "fractures", "fragility fracture", "vertebral fracture",
    "dxa", "dexascan", "bmd", "bone mineral density",
    "qct", "vfa", "vertebral fracture assessment",
    "falls", "fall risk",
    "bisphosphonate", "denosumab", "teriparatide",
    "romosozumab", "alendronate", "risedronate", "zoledronate",
    "calcium", "vitamin d",
    "osteoporosis screening", "fracture risk",
    "nice ng259", "ng259",
    "uspstf osteoporosis", "osteoporosis screening"
}

# Terms that strongly indicate a generic non-clinical question.
# They are not enough by themselves to reject a question if it also contains
# a domain term; the domain check below always takes precedence.
GENERIC_OFF_TOPIC_PATTERNS = [
    r"\bcapital of\b",
    r"\bprime minister of\b",
    r"\bpresident of\b",
    r"\bweather in\b",
    r"\bpopulation of\b",
    r"\bwho won\b",
    r"\bfootball\b",
    r"\bsoccer\b",
    r"\brecipe\b",
    r"\bmovie\b",
    r"\bsong\b",
    r"\bprogramming\b",
    r"\bpython\b",
    r"\bjavascript\b",
    r"\bjava\b"
]


def _normalize_query_for_scope(query: str) -> str:
    """Normalize a query for deterministic domain-scope matching."""
    return re.sub(r"\s+", " ", (query or "").strip().lower())


def is_query_in_domain(query: str) -> bool:
    """
    Return True only when the query contains clear evidence that it belongs
    to this osteoporosis/bone-health guideline domain.

    IMPORTANT:
    This runs BEFORE retrieval. A retrieval score must never be allowed to
    turn an unrelated question into a clinical answer.
    """
    q = _normalize_query_for_scope(query)

    if not q:
        return False

    # A clear domain term is the strongest signal.
    for term in DOMAIN_TERMS:
        if re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", q):
            return True

    # Explicitly generic/off-topic questions are rejected.
    for pattern in GENERIC_OFF_TOPIC_PATTERNS:
        if re.search(pattern, q):
            return False

    return False


def build_out_of_scope_response(query: str) -> Dict[str, Any]:
    """Build a clean user-facing response without performing retrieval."""
    message = (
        "Sorry, but this question is outside my scope.\n\n"
        "I can only answer questions related to osteoporosis, bone health, "
        "fracture risk, bone-density assessment, and the clinical guidelines "
        "available in my knowledge base."
    )
    return {
        "query": query,
        "prompt": None,
        "custom_prompt": None,
        "status": "out_of_scope",
        "risk_tier": "out_of_scope",
        "message": message,
        "output_text": message
    }


def compute_confidence_tier(top_score: float) -> ConfidenceTier:
    """Classify top retrieval similarity score into a canonical confidence tier."""
    if top_score >= CONFIDENCE_SCORE_HIGH:
        return ConfidenceTier.HIGH
    elif top_score >= CONFIDENCE_SCORE_MEDIUM:
        return ConfidenceTier.MEDIUM
    elif top_score >= CONFIDENCE_SCORE_LOW:
        return ConfidenceTier.LOW
    return ConfidenceTier.INSUFFICIENT_EVIDENCE


# =====================================================================
# Small utility: safe attribute/key access for citation entries
# =====================================================================

def _cit_get(cit: Any, key: str, default: Any = None) -> Any:
    """
    Safely read a field off a citation entry, whether it's a plain dict
    (as built by ClinicalSynthesizer._build_citations) or a Citation
    object/pydantic model (if src.schema.ClinicalSynthesisResponse coerces
    the citations field into typed Citation instances on assignment).

    FIX: previously the report-formatting loop in
    generate_grounded_clinical_response() called `cit.get(...)` directly,
    which crashed with `'Citation' object has no attribute 'get'` whenever
    the schema layer converted the citation dicts into Citation objects.
    This helper works for both shapes so that conversion (or lack of it)
    never breaks report generation.
    """
    if isinstance(cit, dict):
        return cit.get(key, default)
    return getattr(cit, key, default)


# =====================================================================
# Runtime Unsupported-Claim Detection & Stripping
# =====================================================================

def detect_unsupported_claims(
    claim_text: str,
    source_chunks: Dict[str, Any],
    threshold: float = UNSUPPORTED_CLAIM_OVERLAP_THRESHOLD
) -> Tuple[bool, float, List[str]]:
    """
    Verify claim grounding against cited chunk texts via lexical token overlap.
    Returns: (is_unsupported, max_overlap_score, cited_chunk_ids)
    """
    if not claim_text or not claim_text.strip():
        return False, 1.0, []

    # Extract cited chunk IDs in format [chunk_id]
    cited_ids = re.findall(r'\[([a-zA-Z0-9_\-]+(?:_chk_[a-zA-Z0-9_\-]+)?)\]', claim_text)

    # Filter to cited IDs that actually exist in source_chunks if available
    valid_cited_ids = [cid for cid in cited_ids if cid in source_chunks]
    target_ids = valid_cited_ids if valid_cited_ids else list(source_chunks.keys())

    # Extract informative tokens (>= 3 chars, excluding common stop words)
    claim_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', claim_text.lower()))
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "should", "must",
        "can", "are", "were", "been", "have", "has", "will", "what", "when",
        "where", "which", "who", "whom", "into", "onto", "upon", "about",
        "above", "below", "guideline", "recommendation", "recommendations",
        "offer", "consider", "assess", "patient", "patients", "clinical"
    }
    informative_words = claim_words - stopwords

    if not informative_words:
        return False, 1.0, cited_ids

    total_overlap = 0.0
    for cid in target_ids:
        chunk = source_chunks.get(cid)
        if chunk:
            chunk_text = getattr(chunk, "text", str(chunk))
            chunk_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', chunk_text.lower()))
            overlap = len(informative_words & chunk_words) / len(informative_words)
            if overlap > total_overlap:
                total_overlap = overlap

    is_unsupported = total_overlap < threshold
    return is_unsupported, total_overlap, cited_ids


def filter_unsupported_claims(
    response: ClinicalSynthesisResponse,
    source_chunks: Dict[str, Any],
    threshold: float = UNSUPPORTED_CLAIM_OVERLAP_THRESHOLD
) -> Tuple[ClinicalSynthesisResponse, List[str]]:
    """
    Filter out ungrounded recommendations and flag unverified statements.
    Ensures that any generated recommendation has verified lexical overlap with source evidence.
    """
    verified_recs = []
    dropped_logs = []

    for rec in response.key_recommendations:
        unsupported, score, cids = detect_unsupported_claims(rec, source_chunks, threshold=threshold)
        if unsupported:
            log_msg = f"Dropped ungrounded claim (overlap={score:.2f} < {threshold}, cited={cids}): '{rec}'"
            dropped_logs.append(log_msg)
            logger.warning(f"[UNSUPPORTED CLAIM DROPPED] '{rec}' (overlap {score:.2f} < {threshold})")
        else:
            verified_recs.append(rec)

    if not verified_recs:
        response.direct_answer = (
            "Clinical guidance withheld: generated claims could not be verified against the source guideline text."
        )
        response.key_recommendations = [
            "Generated claims failed grounded verification against retrieved guideline chunks."
        ]
        # FIX: assign the enum member itself (not .value) so to_dict()'s
        # self.evidence_strength.value access keeps working.
        response.evidence_strength = ConfidenceTier.INSUFFICIENT_EVIDENCE
    else:
        response.key_recommendations = verified_recs

    return response, dropped_logs


# =====================================================================
# LLM Execution Wrapper
# =====================================================================

def call_llm(
    prompt: str,
    provider: str = DEFAULT_LLM_PROVIDER,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.1
) -> str:
    """
    Swappable LLM execution wrapper.
    Calls Google Gemini (or configured provider) and returns raw generated response text.
    """
    resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("APIKEY") or os.getenv("GOOGLE_API_KEY")
    if resolved_key:
        resolved_key = resolved_key.strip().strip('"').strip("'")
    target_model = model_name or DEFAULT_GEMINI_MODEL

    if provider == "gemini" and resolved_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=resolved_key)
            client = genai.GenerativeModel(
                model_name=target_model,
                generation_config={
                    "temperature": temperature,
                    "top_p": 0.95,
                    "response_mime_type": "application/json"
                }
            )
            resp = client.generate_content(prompt)
            if resp and hasattr(resp, "text") and resp.text:
                return resp.text
        except Exception as exc:
            logger.error(f"call_llm execution failed on model '{target_model}': {exc}")
    return ""


# =====================================================================
# Clinical Synthesizer Engine
# =====================================================================

class ClinicalSynthesizer:
    """
    Evidence Synthesizer producing structured, grounded clinical recommendations.
    Accepts user clinical query and user custom prompt instructions.
    Uses Google Gemini via call_llm() with full structured JSON prompting,
    and falls back to deterministic extraction when offline or unconfigured.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = DEFAULT_LLM_PROVIDER,
        model_name: Optional[str] = None,
        allow_fallback: bool = True,
        temperature: float = 0.1
    ):
        self.provider = provider
        self.model_name = model_name or DEFAULT_GEMINI_MODEL
        self.allow_fallback = allow_fallback
        self.temperature = temperature

        # Resolve API Key from arguments or environment variables (.env)
        resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("APIKEY") or os.getenv("GOOGLE_API_KEY")
        if resolved_key:
            resolved_key = resolved_key.strip().strip('"').strip("'")
        self.api_key = resolved_key

    def _build_synthesis_prompt(
        self,
        query: str,
        retrieved_passages: List[Tuple[Any, float]],
        custom_prompt: Optional[str] = None,
        evidence_panel: Optional[str] = None
    ) -> str:
        """
        Construct a strict Evidence Synthesizer prompt incorporating:
        1. User's custom generation prompt / instruction (or safe default)
        2. User's clinical query
        3. Retrieved guideline passages / Evidence Panel (strict source of truth)
        4. Mandatory grounding, safety rules, and structured JSON schema
        """
        if evidence_panel:
            evidence_text = evidence_panel
        else:
            context_blocks = []
            for i, (chunk, score) in enumerate(retrieved_passages, 1):
                cid = getattr(chunk, "chunk_id", f"chk_{i}")
                doc = getattr(chunk, "document_name", "Guideline")
                sec = getattr(chunk, "section_title", "General Guidance")
                page = getattr(chunk, "page_number", 1)
                url = getattr(chunk, "source_url", "N/A")
                text = getattr(chunk, "text", str(chunk))

                context_blocks.append(
                    f"--- EVIDENCE PASSAGE [{i}] ---\n"
                    f"Chunk ID: {cid}\n"
                    f"Document: {doc}\n"
                    f"Section: {sec} (Page {page})\n"
                    f"URL: {url}\n"
                    f"Retrieval Score: {score:.4f}\n"
                    f"Content:\n{text}\n"
                )
            evidence_text = "\n".join(context_blocks)

        instruction_text = custom_prompt.strip() if (custom_prompt and custom_prompt.strip()) else DEFAULT_GENERATION_PROMPT

        prompt = f"""You are a clinical evidence assistant.
Answer the user's question naturally, clearly, and directly, as if you were explaining the result to them in a helpful conversation.

USER GENERATION INSTRUCTION:
{instruction_text}

USER QUESTION:
"{query}"

RETRIEVED GUIDELINE EVIDENCE (YOUR ONLY SOURCE OF TRUTH):
{evidence_text}

IMPORTANT RULES:
1. Use ONLY the evidence provided above. Do not add medical facts from your own knowledge.
2. Answer the actual question first. Do not start with technical details about retrieval, similarity scores, chunks, or the RAG system.
3. Write in natural, easy-to-read language.
4. If the question asks for a comparison, explicitly compare the sources using simple wording such as "NICE..." and "USPSTF...".
5. If evidence for one source is missing, say: "The retrieved evidence does not provide a clear recommendation from [source]." Do not guess.
6. Do not merge recommendations from different guidelines.
7. Every clinical claim must have an inline [chunk_id] citation.
8. Mention important limitations only when they are supported by the retrieved evidence.
9. Keep the answer concise. Avoid repeating the same recommendation in multiple sections.
10. Do not mention internal implementation details such as similarity scores, confidence bands, chunk IDs (except citations), retrieval mode, or guardrails in the natural-language answer.

Return valid JSON only:

{{
  "direct_answer": "A natural 2-4 sentence answer that directly answers the user's question. Use source names when comparing guidelines and include [chunk_id] citations.",
  "target_population": "Only if relevant to understanding the answer.",
  "key_recommendations": [
    "Only the most important guideline findings, written naturally and briefly, with [chunk_id] citations."
  ],
  "source_conflicts": [
    "Only if the retrieved guidelines actually disagree. Otherwise return an empty list."
  ],
  "clinical_caveats": [
    "Only important evidence limitations supported by the retrieved text. Otherwise return an empty list."
  ]
}}

Return JSON only."""
        return prompt

    def _parse_llm_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Safely parse JSON from LLM response with code fence stripping."""
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except Exception:
            match = re.search(r'(\{.*\})', cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    pass
        return None

    def _synthesize_deterministic(
        self,
        query: str,
        retrieved_passages: List[Tuple[Any, float]],
        evidence_tier: str,
        top_score: float,
        custom_prompt: Optional[str] = None
    ) -> ClinicalSynthesisResponse:
        """Deterministic extraction fallback when LLM is offline or unconfigured."""
        top_chunk, _ = retrieved_passages[0]

        # Extract target population from metadata
        pop_candidates = set()
        for chunk, _ in retrieved_passages:
            pops = chunk.metadata.get("population", []) if hasattr(chunk, "metadata") and chunk.metadata else []
            pop_candidates.update(pops)
        target_pop = ", ".join(sorted(pop_candidates)) if pop_candidates else "Adults undergoing osteoporosis risk assessment"

        # Structured deterministic synthesis with inline citations
        sentences = []
        for chunk, _ in retrieved_passages[:2]:
            clean_s = chunk.text.replace("\n", " ").strip()
            first_sent = clean_s.split(". ")[0].strip()
            if not first_sent.endswith("."):
                first_sent += "."
            sentences.append(f"{first_sent} [{chunk.chunk_id}]")

        direct_answer = " ".join(sentences)
        key_recs = []
        for chunk, _ in retrieved_passages[:3]:
            sec = getattr(chunk, "section_title", "Guideline Recommendation")
            clean_s = chunk.text.replace("\n", " ").strip()
            first_sent = clean_s.split(". ")[0].strip()
            key_recs.append(f"{sec}: {first_sent} [{chunk.chunk_id}]")

        citations = self._build_citations(retrieved_passages)
        caveat_msg = f"Evidence synthesized deterministically from top guideline passage (similarity score: {top_score:.4f})."
        if custom_prompt and custom_prompt != DEFAULT_GENERATION_PROMPT:
            caveat_msg += " (Deterministic mode active; custom prompt recorded in request context)."

        return ClinicalSynthesisResponse(
            query=query,
            direct_answer=direct_answer,
            target_population=target_pop,
            key_recommendations=key_recs,
            citations=citations,
            evidence_strength=evidence_tier,
            clinical_caveats=[caveat_msg],
            provider="deterministic_fallback",
            model="heuristic_extractor"
        )

    def _build_citations(self, retrieved_passages: List[Tuple[Any, float]]) -> List[Dict[str, Any]]:
        """
        Construct full metadata citation objects preserving:
        - document_name
        - section_title
        - page_number
        - chunk_id
        - source_url
        - similarity_score
        """
        return [
            {
                "chunk_id": chunk.chunk_id,
                "document_id": getattr(chunk, "document_id", ""),
                "document_name": getattr(chunk, "document_name", ""),
                "section_title": getattr(chunk, "section_title", "General Clinical Guidance"),
                "page_number": getattr(chunk, "page_number", 1),
                "source_url": getattr(chunk, "source_url", None),
                "similarity_score": round(score, 4)
            }
            for chunk, score in retrieved_passages
        ]

    def synthesize(
        self,
        query: str,
        retrieved_passages: List[Tuple[Any, float]],
        prompt: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        evidence_panel: Optional[str] = None
    ) -> ClinicalSynthesisResponse:
        """
        Synthesize clinical query and custom prompt against retrieved guideline passages.
        Executes Gemini LLM synthesis via call_llm() if configured, falling back to deterministic extraction.
        """
        if not retrieved_passages:
            return ClinicalSynthesisResponse(
                query=query,
                direct_answer="No relevant clinical guidelines found in the knowledge base.",
                target_population="Not Specified",
                key_recommendations=[],
                citations=[],
                evidence_strength=ConfidenceTier.INSUFFICIENT_EVIDENCE.value,
                clinical_caveats=["No matching guideline chunks found."]
            )

        active_prompt = prompt if prompt is not None else custom_prompt
        effective_prompt = active_prompt.strip() if (active_prompt and active_prompt.strip()) else DEFAULT_GENERATION_PROMPT

        top_chunk, top_score = retrieved_passages[0]
        evidence_tier = compute_confidence_tier(top_score).value
        citations = self._build_citations(retrieved_passages)

        # 1. Attempt LLM Generation via call_llm()
        if self.provider == "gemini" and self.api_key:
            try:
                prompt_str = self._build_synthesis_prompt(
                    query=query,
                    retrieved_passages=retrieved_passages,
                    custom_prompt=effective_prompt,
                    evidence_panel=evidence_panel
                )
                raw_llm_text = call_llm(
                    prompt=prompt_str,
                    provider=self.provider,
                    model_name=self.model_name,
                    api_key=self.api_key,
                    temperature=self.temperature
                )

                if raw_llm_text:
                    parsed_json = self._parse_llm_json(raw_llm_text)
                    if parsed_json and isinstance(parsed_json, dict):
                        direct_ans = str(parsed_json.get("direct_answer", "")).strip()
                        target_pop = str(parsed_json.get("target_population", "")).strip() or "Adults evaluated for osteoporosis"
                        key_recs = parsed_json.get("key_recommendations", [])
                        caveats = parsed_json.get("clinical_caveats", [])

                        if isinstance(key_recs, str):
                            key_recs = [key_recs]
                        if isinstance(caveats, str):
                            caveats = [caveats]

                        if direct_ans and key_recs:
                            return ClinicalSynthesisResponse(
                                query=query,
                                direct_answer=direct_ans,
                                target_population=target_pop,
                                key_recommendations=[str(r) for r in key_recs],
                                citations=citations,
                                evidence_strength=evidence_tier,
                                clinical_caveats=[str(c) for c in caveats] if caveats else [f"Evidence synthesized from top retrieved passage (Score: {top_score:.4f})."],
                                provider="gemini",
                                model=self.model_name
                            )
                logger.warning("Gemini response did not contain expected JSON structure. Using deterministic fallback.")
            except Exception as exc:
                logger.warning(f"Gemini generation call failed ('{exc}'). Using deterministic fallback.")

        # 2. Fallback to deterministic synthesis
        return self._synthesize_deterministic(
            query=query,
            retrieved_passages=retrieved_passages,
            evidence_tier=evidence_tier,
            top_score=top_score,
            custom_prompt=effective_prompt
        )


# =====================================================================
# Main Generation Orchestrator
# =====================================================================

def generate_grounded_clinical_response(
    query: str,
    prompt: Optional[str] = None,
    top_k: int = 3,
    mode: str = "hybrid",
    provider: str = DEFAULT_LLM_PROVIDER,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    custom_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main Generation Orchestrator:
    1. Rejects questions outside the osteoporosis/bone-health domain BEFORE retrieval
    2. Evaluates safety/emergency guardrails on in-domain queries
    3. Retrieves ranked evidence passages (Evidence Panel as source of truth)
    3. Enforces retrieval confidence gating using Retrieval.py's own assessment
       (single source of truth - see FIX note below)
    4. Generates grounded recommendations using user query and custom prompt (LLM via call_llm or deterministic fallback)
    5. Post-processes output to strip unsupported claims and records them for audit
    6. Formats into the 4 canonical clinical report sections
    """
    from scripts.Retrieval import classify_query_risk, retrieve_evidence

    # Resolve active prompt (supporting both prompt and custom_prompt keyword arguments)
    active_prompt = prompt if prompt is not None else custom_prompt
    effective_prompt = active_prompt.strip() if (active_prompt and active_prompt.strip()) else DEFAULT_GENERATION_PROMPT

    # 1. DOMAIN SCOPE CHECK — MUST happen BEFORE retrieval.
    # Retrieval similarity is not a domain classifier. Unrelated questions can
    # match generic guideline/PDF text (for example a table of contents) with a
    # very high score. Stop those questions before they ever reach Retrieval.py.
    if not is_query_in_domain(query):
        out_of_scope = build_out_of_scope_response(query)
        out_of_scope["prompt"] = active_prompt
        out_of_scope["custom_prompt"] = active_prompt
        return out_of_scope

    # 2. Guardrail Safety Check
    # FIX: compare against the QueryRiskCategory enum values, not mismatched
    # snake_case strings — the old comparison never matched, silently
    # bypassing the emergency/out-of-scope refusal guardrail.
    tier, guardrail_msg = classify_query_risk(query)
    if tier == QueryRiskCategory.REFUSE_REDIRECT:
        return {
            "query": query,
            "prompt": active_prompt,
            "custom_prompt": active_prompt,
            "status": "refused",
            "risk_tier": tier.value,
            "message": guardrail_msg,
            "output_text": f"\n[SAFETY & GUARDRAIL REFUSAL]\n  {guardrail_msg}\n"
        }

    # 3. Retrieval & Evidence Panel Construction
    # FIX: retrieve_evidence() returns a 3-tuple (results, evidence_panel, confidence),
    # not 2 — and `results` is a List[RetrievedChunk], not List[Tuple[chunk, score]].
    # Convert once here so the rest of this module's (chunk, score) tuple interface works.
    results, evidence_panel, confidence_assessment = retrieve_evidence(
        query, top_k=top_k, mode=mode, index_path=INDEX_JSON_PATH
    )
    retrieved_passages: List[Tuple[Any, float]] = [(rc.chunk, rc.similarity_score) for rc in results]
    top_score = retrieved_passages[0][1] if retrieved_passages else 0.0

    # 4. Retrieval Confidence Threshold Gating
    # FIX: this used to recompute its own independent threshold check here
    # (`top_score < MIN_SYNTHESIS_SCORE_THRESHOLD`), duplicating the exact
    # same decision Retrieval.assess_retrieval_confidence() already makes in
    # retrieve_evidence(). Because the two checks read from different env
    # vars (MIN_SCORE_TO_GENERATE in Retrieval.py vs CONFIDENCE_SCORE_LOW
    # here), they could silently disagree if only one was reconfigured — the
    # Retrieval stage's guardrail could say "blocked" while this stage still
    # generated an answer, or vice versa. Retrieval.py's confidence_assessment
    # is now the single source of truth for the block/proceed decision; it
    # also already accounts for the corpus-has-fallback-embeddings case,
    # which this independent check never did.
    if confidence_assessment.blocked:
        block_reason = confidence_assessment.block_reason or (
            f"Top passage similarity score ({top_score:.4f}) did not meet the retrieval confidence threshold."
        )
        synthesis_resp = ClinicalSynthesisResponse(
            query=query,
            direct_answer=(
                "Insufficient clinical evidence found in guideline index to answer this query confidently. "
                f"{block_reason}"
            ),
            target_population="Unspecified / Insufficient Guideline Grounding",
            key_recommendations=[
                "No guideline recommendations meet the minimum similarity threshold for reliable clinical decision support."
            ],
            citations=[
                {
                    "chunk_id": c.chunk_id,
                    "document_id": getattr(c, "document_id", ""),
                    "document_name": getattr(c, "document_name", ""),
                    "section_title": getattr(c, "section_title", "General Clinical Guidance"),
                    "page_number": getattr(c, "page_number", 1),
                    "source_url": getattr(c, "source_url", None)
                }
                for c, _ in retrieved_passages
            ],
            evidence_strength=ConfidenceTier.INSUFFICIENT_EVIDENCE.value,
            clinical_caveats=[
                f"{block_reason} Generation withheld to prevent ungrounded clinical guidance."
            ],
            provider="system_guardrail",
            model="threshold_gating"
        )
    else:
        synthesizer = ClinicalSynthesizer(api_key=api_key, provider=provider, model_name=model_name)
        synthesis_resp = synthesizer.synthesize(
            query=query,
            retrieved_passages=retrieved_passages,
            prompt=effective_prompt,
            evidence_panel=evidence_panel
        )
        # FIX: assign the enum member (not .value) so ClinicalSynthesisResponse.to_dict()'s
        # `self.evidence_strength.value` lookup doesn't crash on a plain string later.
        synthesis_resp.evidence_strength = compute_confidence_tier(top_score)

        # Post-Process: Strip Unsupported Claims (applied to both LLM and deterministic output)
        chunk_map = {c.chunk_id: c for c, _ in retrieved_passages}
        synthesis_resp, stripped_logs = filter_unsupported_claims(synthesis_resp, chunk_map)
        # FIX: stripped_logs (the Unsupported Claim Detection guardrail's
        # audit trail — Safety Workflow step 3) was computed and then
        # silently discarded, so the report never showed evidence that this
        # guardrail actually ran or removed anything. Surface it via the
        # response's own guardrail_warnings field, which schema.py already
        # defines for exactly this purpose.
        if stripped_logs:
            synthesis_resp.guardrail_warnings.extend(stripped_logs)

    if tier == QueryRiskCategory.NEEDS_CAUTION and guardrail_msg not in synthesis_resp.clinical_caveats:
        synthesis_resp.clinical_caveats.insert(0, guardrail_msg)

    # 5. Format a conversational, assistant-like user-facing answer
    report_lines = [
        "Clinical Guideline Answer",
        "",
        synthesis_resp.direct_answer,
    ]

    if synthesis_resp.key_recommendations:
        report_lines.extend(["", "In short:"])
        for rec in synthesis_resp.key_recommendations:
            report_lines.append(f"• {rec}")

    source_conflicts = getattr(synthesis_resp, "source_conflicts", None)
    if source_conflicts:
        report_lines.extend(["", "One important difference:"])
        for conflict in source_conflicts:
            report_lines.append(f"• {conflict}")

    if synthesis_resp.clinical_caveats:
        report_lines.extend(["", "A note on the evidence:"])
        for cav in synthesis_resp.clinical_caveats:
            report_lines.append(f"• {cav}")

    # Keep only the most relevant evidence snippets.
    report_lines.extend(["", "Evidence from the guidelines:"])
    seen_chunks = set()
    evidence_count = 0

    for chunk, score in retrieved_passages:
        cid = getattr(chunk, "chunk_id", "")
        if cid in seen_chunks:
            continue
        seen_chunks.add(cid)

        excerpt = " ".join(chunk.text.split())
        if len(excerpt) > 300:
            excerpt = excerpt[:300].rsplit(" ", 1)[0] + "..."

        page = getattr(chunk, "page_number", 1)
        report_lines.append(f'• "{excerpt}" [{cid}] — p. {page}')
        evidence_count += 1

        if evidence_count >= 2:
            break

    # Present sources in a compact, human-readable way.
    report_lines.extend(["", "Sources:"])
    seen_sources = set()

    for cit in synthesis_resp.citations:
        doc = _cit_get(cit, "document_name", "") or _cit_get(cit, "document_id", "")
        sec = _cit_get(cit, "section_title", "General Clinical Guidance")
        page = _cit_get(cit, "page_number", 1)
        url = _cit_get(cit, "source_url", "")

        source_key = (doc, page, url)
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)

        source_line = f"• {doc}"
        if sec:
            source_line += f" — {sec}"
        if page:
            source_line += f", p. {page}"
        report_lines.append(source_line)

        if url:
            report_lines.append(f"  {url}")

    # Keep confidence as a small, unobtrusive footer.
    confidence = (
        synthesis_resp.evidence_strength.value
        if hasattr(synthesis_resp.evidence_strength, "value")
        else synthesis_resp.evidence_strength
    )
    report_lines.extend(["", f"Evidence confidence: {confidence.capitalize()}"])

    # Audit warnings are shown only when something was actually stripped.
    if synthesis_resp.guardrail_warnings:
        report_lines.extend([
            "",
            "Evidence check:",
            "• Some claims were removed because they could not be verified against the retrieved guideline text."
        ])

    report_lines.extend(["", synthesis_resp.disclaimer])

    formatted_output = "\n".join(report_lines)
    return {
        "query": query,
        "prompt": active_prompt,
        "custom_prompt": active_prompt,
        "status": "success",
        "evidence_strength": synthesis_resp.evidence_strength.value if hasattr(synthesis_resp.evidence_strength, "value") else synthesis_resp.evidence_strength,
        "top_score": top_score,
        "synthesis": synthesis_resp.to_dict(),
        "output_text": formatted_output
    }


run = generate_grounded_clinical_response
main = generate_grounded_clinical_response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 6: Grounded Clinical Generation with Custom Prompt & Query")
    parser.add_argument("query", nargs="?", default=None, help="Clinical query / question")
    parser.add_argument("prompt", nargs="?", default=None, help="Custom generation prompt / instructions")
    parser.add_argument("--query", "-q", dest="query_opt", default=None, help="Clinical query / question")
    parser.add_argument("--prompt", "-p", "--custom-prompt", dest="prompt_opt", default=None, help="Custom generation prompt / instructions")
    parser.add_argument("--top-k", "-k", type=int, default=3, help="Top-K evidence passages to retrieve")
    parser.add_argument("--mode", "-m", choices=["keyword", "semantic", "hybrid"], default="hybrid", help="Search mode")
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER, help="LLM Provider (default: gemini)")
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL, help="Model name (default: gemini-1.5-flash)")

    parsed_args = parser.parse_args()

    active_query = parsed_args.query_opt or parsed_args.query
    active_prompt = parsed_args.prompt_opt or parsed_args.prompt

    if not active_query:
        # Prompt user dynamically if no CLI argument is supplied
        print("_" * 80)
        print("  CLINICAL PRACTICE GUIDELINE RAG: GROUNDED GENERATION")
        print("_" * 80)
        user_input_query = input("\nEnter clinical query (or press Enter for default): ").strip()
        if not user_input_query:
            active_query = "When should a central DXA bone density scan be offered according to NICE guidelines?"
            print(f"Using default query: {active_query}")
        else:
            active_query = user_input_query

        user_input_prompt = input("\nEnter custom generation prompt (optional, press Enter to skip): ").strip()
        active_prompt = user_input_prompt if user_input_prompt else None

    res = run(
        query=active_query,
        prompt=active_prompt,
        top_k=parsed_args.top_k,
        mode=parsed_args.mode,
        provider=parsed_args.provider,
        model_name=parsed_args.model
    )
    print(res.get("output_text", ""))