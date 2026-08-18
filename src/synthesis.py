"""
Clinical Evidence Synthesis & Decision Support Module (src.synthesis).

Generates evidence-grounded clinical recommendations strictly grounded in retrieved
guideline passages, powered by Google Gemini (GEMINI_API_KEY) with full lineage
citation provenance (Document + Section + Page Number + Chunk ID + Source URL),
canonical 4-tier confidence grading (High, Medium, Low, Insufficient Evidence),
citation guardrails, unsupported-claim detection, and 3-tier safety classification.
"""

import os
import json
import re
import urllib.request
import urllib.error
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Set

from src.chunking import Chunk
from src.config import (
    DEFAULT_GEMINI_MODEL,
    MIN_SYNTHESIS_SCORE_THRESHOLD,
    UNSUPPORTED_CLAIM_OVERLAP_THRESHOLD,
    GUIDELINE_SOURCE_URLS,
    ConfidenceTier
)
from src.embedded import classify_query_risk

logger = logging.getLogger(__name__)

CLINICAL_SYNTHESIS_SYSTEM_PROMPT = """You are a specialized Clinical Decision Support Evidence Synthesizer analyzing official medical practice guidelines (such as NICE NG259 and USPSTF).

CRITICAL INSTRUCTIONS:
1. Act ONLY as an evidence synthesizer, NEVER as a diagnostician.
2. Ground EVERY claim strictly and exclusively in the provided retrieved guideline passages. Do NOT add outside clinical knowledge or extrapolate beyond the text.
3. If the retrieved passages do NOT fully answer the query, state this explicitly in the direct answer and caveats.
4. Per-Claim Citation: You MUST cite the exact [chunk_id] for every single recommendation, threshold, and factual claim you make.
5. Identify the exact Target Patient Population specified in the guidelines.
6. Provide output ONLY in valid JSON matching this schema:
{
  "direct_answer": "Concise 2-3 sentence clinical guidance summary directly answering the question with inline [chunk_id] citations.",
  "target_population": "Exact patient demographic or clinical cohort eligible for this intervention with [chunk_id].",
  "key_recommendations": [
    "Specific actionable recommendation bullet with inline citation [chunk_id]."
  ],
  "evidence_strength": "High" | "Medium" | "Low" | "Insufficient Evidence",
  "clinical_caveats": [
    "Important clinical contraindication, fall risk, secondary cause, or guidance limitation with [chunk_id]."
  ],
  "citations": [
    {
      "chunk_id": "exact_chunk_id",
      "document_name": "document_name_from_passage",
      "section_title": "section_title_from_passage",
      "page_number": 1,
      "source_url": "source_url_from_passage"
    }
  ]
}
"""


def _normalize_confidence_tier(raw_value: Optional[str]) -> str:
    """Normalize raw model string into one of the 4 canonical ConfidenceTier values."""
    if not raw_value:
        return ConfidenceTier.MEDIUM.value
    val = str(raw_value).strip().upper()
    if "INSUFFICIENT" in val or "NONE" in val:
        return ConfidenceTier.INSUFFICIENT_EVIDENCE.value
    elif "HIGH" in val or "STRONG" in val:
        return ConfidenceTier.HIGH.value
    elif "LOW" in val or "PARTIAL" in val or "WEAK" in val:
        return ConfidenceTier.LOW.value
    else:
        return ConfidenceTier.MEDIUM.value


def detect_unsupported_claims(
    response: "ClinicalSynthesisResponse",
    chunk_map: Dict[str, Chunk],
    threshold: float = UNSUPPORTED_CLAIM_OVERLAP_THRESHOLD
) -> List[str]:
    """
    Guardrail Step 3: Verifies whether claims citing specific chunk_ids are
    lexically and textually supported by that chunk's content.
    Flags claims falling below the configurable overlap threshold.
    """
    warnings: List[str] = []
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "and", "or", "to", "in",
        "for", "with", "of", "by", "on", "at", "as", "be", "this", "that", "it",
        "should", "must", "can", "may", "who", "have", "had", "from", "when"
    }

    # 1. Verify Key Recommendations
    for rec in response.key_recommendations:
        cited_ids = re.findall(r'\[([a-zA-Z0-9_\-\.]+)\]', rec)
        valid_cids = [cid for cid in cited_ids if cid in chunk_map]
        if not valid_cids:
            continue

        clean_claim = re.sub(r'\[[a-zA-Z0-9_\-\.]+\]', '', rec).strip()
        claim_tokens = {w.lower() for w in re.findall(r'\b[a-zA-Z0-9]{3,}\b', clean_claim) if w.lower() not in stop_words}
        if not claim_tokens:
            continue

        for cid in valid_cids:
            chk_text = chunk_map[cid].text.lower()
            chk_tokens = {w.lower() for w in re.findall(r'\b[a-zA-Z0-9]{3,}\b', chk_text) if w.lower() not in stop_words}

            overlap = len(claim_tokens.intersection(chk_tokens)) / max(1, len(claim_tokens))
            if overlap < threshold:
                warn = (
                    f"Unsupported Claim Alert: Recommendation '{rec[:80]}...' cited '{cid}' "
                    f"but lexical overlap ratio ({overlap:.2f}) was below threshold ({threshold:.2f})."
                )
                warnings.append(warn)
                logger.warning(warn)

    # 2. Verify Direct Answer Sentences
    sentences = re.split(r'(?<=[.!?])\s+', response.direct_answer)
    for sent in sentences:
        cited_ids = re.findall(r'\[([a-zA-Z0-9_\-\.]+)\]', sent)
        valid_cids = [cid for cid in cited_ids if cid in chunk_map]
        if not valid_cids:
            continue

        clean_sent = re.sub(r'\[[a-zA-Z0-9_\-\.]+\]', '', sent).strip()
        sent_tokens = {w.lower() for w in re.findall(r'\b[a-zA-Z0-9]{3,}\b', clean_sent) if w.lower() not in stop_words}
        if not sent_tokens:
            continue

        for cid in valid_cids:
            chk_text = chunk_map[cid].text.lower()
            chk_tokens = {w.lower() for w in re.findall(r'\b[a-zA-Z0-9]{3,}\b', chk_text) if w.lower() not in stop_words}

            overlap = len(sent_tokens.intersection(chk_tokens)) / max(1, len(sent_tokens))
            if overlap < threshold:
                warn = (
                    f"Unsupported Claim Alert: Direct answer sentence '{sent[:80]}...' cited '{cid}' "
                    f"but lexical overlap ratio ({overlap:.2f}) was below threshold ({threshold:.2f})."
                )
                warnings.append(warn)
                logger.warning(warn)

    return warnings


@dataclass
class ClinicalSynthesisResponse:
    """Structured clinical response schema with canonical confidence tiers and full lineage."""
    query: str
    direct_answer: str
    target_population: str
    key_recommendations: List[str]
    citations: List[Dict[str, Any]]
    evidence_strength: str
    clinical_caveats: List[str]
    provider: str = "gemini"
    model: str = DEFAULT_GEMINI_MODEL
    guardrail_warnings: List[str] = field(default_factory=list)
    disclaimer: str = (
        "CLINICAL DISCLAIMER: This evidence synthesis is generated from official clinical practice "
        "guidelines (e.g., NICE NG259, USPSTF) for informational and clinical decision support purposes only. "
        "It does not replace individualized clinical judgment, multidisciplinary review, or local protocols."
    )

    def format_markdown(self) -> str:
        """Format the synthesis into a clinical-grade markdown report with complete provenance."""
        lines = [
            "=" * 88,
            "  CLINICAL EVIDENCE SYNTHESIS & RECOMMENDATIONS",
            "=" * 88,
            f"**Query**: {self.query}",
            f"**Synthesis Engine**: {self.provider.upper()} ({self.model})",
            f"**Evidence Confidence**: {self.evidence_strength}",
            f"**Eligible Population**: {self.target_population}",
            "",
            "### Clinical Guidance Summary",
            self.direct_answer,
            "",
            "### Key Guideline Action Items"
        ]
        for rec in self.key_recommendations:
            lines.append(f"  • {rec}")

        if self.clinical_caveats:
            lines.append("")
            lines.append("### Practice Caveats & Safety Considerations")
            for cav in self.clinical_caveats:
                lines.append(f"  ⚠ {cav}")

        if self.guardrail_warnings:
            lines.append("")
            lines.append("### Citation & Faithfulness Guardrail Notices")
            for warn in self.guardrail_warnings:
                lines.append(f"  🔍 {warn}")

        lines.append("")
        lines.append("### Grounded Source Citations (Document + Section + Page + Chunk ID + URL)")
        for cit in self.citations:
            doc = cit.get("document_name") or cit.get("document_id", "Unknown Document")
            sec = cit.get("section_title") or cit.get("section", "General")
            pg = cit.get("page_number", 1)
            cid = cit.get("chunk_id", "N/A")
            url = cit.get("source_url") or "N/A"
            lines.append(f"  [Ref] Document: {doc} | Section: {sec} | Page: {pg} | Chunk: {cid} | URL: {url}")

        lines.append("")
        lines.append("-" * 88)
        lines.append(f"ℹ {self.disclaimer}")
        lines.append("=" * 88)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert synthesis response into structured JSON."""
        return {
            "query": self.query,
            "provider": self.provider,
            "model": self.model,
            "direct_answer": self.direct_answer,
            "target_population": self.target_population,
            "key_recommendations": self.key_recommendations,
            "citations": self.citations,
            "evidence_strength": self.evidence_strength,
            "clinical_caveats": self.clinical_caveats,
            "guardrail_warnings": self.guardrail_warnings,
            "disclaimer": self.disclaimer
        }


class ClinicalSynthesizer:
    """
    Clinical Evidence Synthesizer powered by Google Gemini (GEMINI_API_KEY)
    with multi-provider support, per-claim citations, citation guardrails,
    and unsupported-claim detection.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        allow_fallback: bool = False
    ):
        self.provider = (provider or os.environ.get("RAG_PROVIDER", "gemini")).lower()
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.model_name = model_name or os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.allow_fallback = allow_fallback or (self.provider == "fallback") or (os.environ.get("RAG_ALLOW_FALLBACK", "false").lower() in ("1", "true", "yes"))
        self._active_provider, self._active_model = self._resolve_active_provider()

    def _resolve_active_provider(self) -> Tuple[str, str]:
        """Identifies available LLM provider credentials, prioritizing explicit selection and Gemini."""
        # 1. Explicit Fallback Provider
        if self.provider == "fallback" or self.allow_fallback:
            return "fallback", "extractive-grounded"

        # 2. Google Gemini (Primary)
        if self.provider in ("gemini", "auto", "google"):
            if self.api_key:
                return "gemini", self.model_name
            gemini_env = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if gemini_env:
                self.api_key = gemini_env
                return "gemini", self.model_name

        # 3. OpenAI
        if self.provider == "openai":
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key:
                return "openai", self.model_name or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

        # 4. Anthropic
        if self.provider == "anthropic":
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
            if anthropic_key:
                return "anthropic", self.model_name or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

        # 5. Ollama
        if self.provider == "ollama":
            return "ollama", self.model_name or os.environ.get("OLLAMA_MODEL", "llama3.2")

        # 6. Offline Fallback Mode
        if self.allow_fallback or os.environ.get("MOCK_LLM_TEST", "false").lower() == "true":
            return "fallback", "extractive-grounded"

        # 7. Fallback to Gemini if api_key was provided without explicit provider
        if self.api_key:
            return "gemini", self.model_name

        return "unconfigured", "none"

    def synthesize(
        self,
        query: str,
        retrieved_passages: List[Tuple[Chunk, float]]
    ) -> ClinicalSynthesisResponse:
        """
        Synthesize clinical recommendations grounded in retrieved guideline passages.
        """
        risk_tier, risk_msg = classify_query_risk(query)

        # 1. Immediate Refusal on Medical Emergency
        if risk_tier == "refuse_redirect" and "SAFETY EMERGENCY" in risk_msg:
            return ClinicalSynthesisResponse(
                query=query,
                direct_answer=risk_msg,
                target_population="Emergency Care Required",
                key_recommendations=["Seek immediate emergency medical attention or call emergency services (999/911/112)."],
                citations=[],
                evidence_strength=ConfidenceTier.INSUFFICIENT_EVIDENCE.value,
                clinical_caveats=[risk_msg],
                provider="safety_guardrail",
                model="emergency_triage"
            )

        # 2. Insufficient Evidence Check on Empty Retrieval
        if not retrieved_passages:
            caveat = risk_msg if risk_tier == "refuse_redirect" else "Query did not match indexed clinical guideline topics. Rephrase with clinical terms (e.g. 'DXA', 'FRAX', 'bisphosphonates')."
            return ClinicalSynthesisResponse(
                query=query,
                direct_answer="Insufficient Evidence: No relevant guideline passages were retrieved for this clinical query.",
                target_population="Unspecified",
                key_recommendations=[],
                citations=[],
                evidence_strength=ConfidenceTier.INSUFFICIENT_EVIDENCE.value,
                clinical_caveats=[caveat],
                provider="guardrail",
                model="confidence_check"
            )

        # 3. Minimum Relevance Score Threshold Check
        top_chunk, top_score = retrieved_passages[0]
        if top_score < MIN_SYNTHESIS_SCORE_THRESHOLD:
            return ClinicalSynthesisResponse(
                query=query,
                direct_answer="Insufficient Evidence: The retrieved guideline passages exhibit low relevance confidence. To prevent ungrounded clinical claims, synthesis is withheld.",
                target_population="Unspecified",
                key_recommendations=[],
                citations=[],
                evidence_strength=ConfidenceTier.INSUFFICIENT_EVIDENCE.value,
                clinical_caveats=["Top passage relevance score fell below minimum clinical certainty threshold."],
                provider="guardrail",
                model="confidence_check"
            )

        # 4. Out-of-Scope check when passages were provided but query is out-of-scope
        if risk_tier == "refuse_redirect":
            return ClinicalSynthesisResponse(
                query=query,
                direct_answer="Insufficient Evidence: Query is outside clinical osteoporosis and bone health guidelines.",
                target_population="Unspecified",
                key_recommendations=[],
                citations=[],
                evidence_strength=ConfidenceTier.INSUFFICIENT_EVIDENCE.value,
                clinical_caveats=[risk_msg],
                provider="safety_guardrail",
                model="scope_check"
            )

        # 3. Check Gemini / Provider API Key Configuration
        if self._active_provider == "unconfigured":
            if os.environ.get("MOCK_LLM_SYNTHESIS", "false").lower() == "true":
                mock_resp = self._mock_synthesis(query, retrieved_passages)
                if risk_tier == "needs_caution" and risk_msg not in mock_resp.clinical_caveats:
                    mock_resp.clinical_caveats.insert(0, risk_msg)
                return mock_resp

            error_msg = (
                "[GEMINI API KEY REQUIRED] GEMINI_API_KEY is not set. "
                "To enable generative clinical synthesis with Google Gemini, set your API key in the environment:\n"
                "  export GEMINI_API_KEY=\"your-gemini-api-key\"\n"
                "Or pass it directly in the CLI: python main.py ask \"...\" --api-key \"your-key\"\n"
                "(Set RAG_ALLOW_FALLBACK=1 to run in offline extractive fallback mode)."
            )
            logger.error(error_msg)
            return ClinicalSynthesisResponse(
                query=query,
                direct_answer=error_msg,
                target_population="Environment Configuration Required",
                key_recommendations=[
                    "Set GEMINI_API_KEY: export GEMINI_API_KEY=\"your-api-key\"",
                    "Or pass via CLI: python main.py ask \"...\" --api-key \"your-key\"",
                    "Get a free Gemini API key at: https://aistudio.google.com/"
                ],
                citations=[],
                evidence_strength=ConfidenceTier.INSUFFICIENT_EVIDENCE.value,
                clinical_caveats=["Missing GEMINI_API_KEY credential required for generative recommendation synthesis."],
                provider="gemini_configuration_alert",
                model="none"
            )

        # 4. Build Context Prompt with Verbatim Chunk IDs, Sections, Page Numbers, and URLs
        context_prompt, chunk_map = self._build_context_prompt(query, retrieved_passages)

        # 5. Invoke LLM Backend
        raw_response = None
        if self._active_provider == "gemini":
            raw_response = self._call_gemini(context_prompt)
        elif self._active_provider == "openai":
            raw_response = self._call_openai(context_prompt)
        elif self._active_provider == "anthropic":
            raw_response = self._call_anthropic(context_prompt)
        elif self._active_provider == "ollama":
            raw_response = self._call_ollama(context_prompt)
        elif self._active_provider == "fallback":
            fb_resp = self._extractive_fallback_synthesis(query, retrieved_passages)
            if risk_tier == "needs_caution" and risk_msg not in fb_resp.clinical_caveats:
                fb_resp.clinical_caveats.insert(0, risk_msg)
            return fb_resp

        if not raw_response:
            fb_resp = self._extractive_fallback_synthesis(query, retrieved_passages)
            if risk_tier == "needs_caution" and risk_msg not in fb_resp.clinical_caveats:
                fb_resp.clinical_caveats.insert(0, risk_msg)
            return fb_resp

        # 6. Parse JSON Output
        parsed_data = self._parse_json_response(raw_response)
        if not parsed_data:
            logger.warning("Gemini response failed JSON parsing; executing grounded fallback.")
            fb_resp = self._extractive_fallback_synthesis(query, retrieved_passages)
            if risk_tier == "needs_caution" and risk_msg not in fb_resp.clinical_caveats:
                fb_resp.clinical_caveats.insert(0, risk_msg)
            return fb_resp

        # 7. Apply Citation Lineage Guardrails
        validated_response = self._apply_citation_guardrail(
            query=query,
            data=parsed_data,
            retrieved_chunks=retrieved_passages,
            chunk_map=chunk_map,
            provider=self._active_provider,
            model=self._active_model
        )

        # 8. Apply Unsupported Claim Detection (Guardrail Step 3)
        unsupported_warnings = detect_unsupported_claims(
            response=validated_response,
            chunk_map=chunk_map,
            threshold=UNSUPPORTED_CLAIM_OVERLAP_THRESHOLD
        )
        if unsupported_warnings:
            validated_response.guardrail_warnings.extend(unsupported_warnings)

        # 9. Append Caution advisory if patient-specific
        if risk_tier == "needs_caution" and risk_msg not in validated_response.clinical_caveats:
            validated_response.clinical_caveats.insert(0, risk_msg)

        return validated_response

    def _build_context_prompt(
        self,
        query: str,
        retrieved_passages: List[Tuple[Chunk, float]]
    ) -> Tuple[str, Dict[str, Chunk]]:
        """Constructs formatted context block with chunk_id labels, pages, and URLs for LLM prompt."""
        chunk_map: Dict[str, Chunk] = {}
        passages_text = []

        for i, (chk, score) in enumerate(retrieved_passages, start=1):
            chunk_map[chk.chunk_id] = chk
            doc_name = chk.document_name
            sec_title = chk.section_title
            pg_num = chk.page_number
            src_url = chk.source_url or GUIDELINE_SOURCE_URLS.get(chk.document_id, "N/A")
            issuer = chk.metadata.get("guideline_issuer", doc_name)

            passages_text.append(
                f"### RETRIEVED GUIDELINE PASSAGE [{i}]\n"
                f"- DOCUMENT: {doc_name} ({issuer})\n"
                f"- SECTION: {sec_title}\n"
                f"- PAGE: {pg_num}\n"
                f"- CHUNK_ID: {chk.chunk_id}\n"
                f"- SOURCE_URL: {src_url}\n"
                f"- RELEVANCE SCORE: {score:.4f}\n"
                f"- CONTENT:\n{chk.text.strip()}\n"
            )

        context_block = "\n".join(passages_text)
        user_prompt = (
            f"OFFICIAL GUIDELINE PASSAGES:\n"
            f"{context_block}\n\n"
            f"CLINICIAN QUERY: {query}\n\n"
            "Synthesize an evidence-based clinical answer grounded strictly in the passages above. "
            "Cite the exact [chunk_id] for every claim and output ONLY structured JSON matching the requested schema."
        )
        return user_prompt, chunk_map

    def _parse_json_response(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Extracts JSON from LLM response text."""
        try:
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
            if match:
                return json.loads(match.group(1))

            brace_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if brace_match:
                return json.loads(brace_match.group(0))

            return json.loads(raw_text.strip())
        except Exception as exc:
            logger.error(f"Error parsing Gemini JSON output: {exc}")
            return None

    def _apply_citation_guardrail(
        self,
        query: str,
        data: Dict[str, Any],
        retrieved_chunks: List[Tuple[Chunk, float]],
        chunk_map: Dict[str, Chunk],
        provider: str,
        model: str
    ) -> ClinicalSynthesisResponse:
        """
        Citation Guardrail: Validates every listed citation against the actual retrieved chunk_ids.
        Attaches complete provenance (Document + Section + Page Number + Chunk ID + Source URL).
        """
        valid_chunk_ids: Set[str] = {chk.chunk_id for chk, _ in retrieved_chunks}
        raw_citations: List[Dict[str, Any]] = data.get("citations", [])
        verified_citations: List[Dict[str, Any]] = []
        guardrail_warnings: List[str] = []

        seen_cits = set()
        for cit in raw_citations:
            cid = cit.get("chunk_id", "").strip()
            if cid in valid_chunk_ids:
                chk = chunk_map[cid]
                if cid not in seen_cits:
                    seen_cits.add(cid)
                    verified_citations.append({
                        "chunk_id": cid,
                        "document_id": chk.document_id,
                        "document_name": chk.document_name,
                        "section": chk.section_title,
                        "section_title": chk.section_title,
                        "page_number": chk.page_number,
                        "source_url": chk.source_url or GUIDELINE_SOURCE_URLS.get(chk.document_id)
                    })
            elif cid:
                warn = f"Citation Guardrail: Model cited chunk_id '{cid}' which was NOT in the retrieved evidence set. Citation pruned."
                logger.warning(warn)
                guardrail_warnings.append(warn)

        # Fallback to top retrieved chunks if model citation list was empty
        if not verified_citations:
            for chk, _ in retrieved_chunks[:3]:
                verified_citations.append({
                    "chunk_id": chk.chunk_id,
                    "document_id": chk.document_id,
                    "document_name": chk.document_name,
                    "section": chk.section_title,
                    "section_title": chk.section_title,
                    "page_number": chk.page_number,
                    "source_url": chk.source_url or GUIDELINE_SOURCE_URLS.get(chk.document_id)
                })

        normalized_confidence = _normalize_confidence_tier(data.get("evidence_strength"))

        return ClinicalSynthesisResponse(
            query=query,
            direct_answer=data.get("direct_answer", ""),
            target_population=data.get("target_population", "Adults meeting guideline criteria"),
            key_recommendations=data.get("key_recommendations", []),
            citations=verified_citations,
            evidence_strength=normalized_confidence,
            clinical_caveats=data.get("clinical_caveats", []),
            provider=provider,
            model=model,
            guardrail_warnings=guardrail_warnings
        )

    # -----------------------------------------------------------------
    # LLM API Call Implementations
    # -----------------------------------------------------------------

    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Calls Google Gemini API using google.generativeai SDK or direct REST request."""
        api_key = self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return None

        # 1. Try Google Generative AI SDK
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=self._active_model,
                system_instruction=CLINICAL_SYNTHESIS_SYSTEM_PROMPT
            )
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
        except ImportError:
            logger.debug("google.generativeai SDK not installed. Falling back to Gemini REST API.")
        except Exception as exc:
            logger.warning(f"Gemini SDK notice: {exc}. Trying Gemini REST endpoint.")

        # 2. Direct HTTP REST API Request to Google Gemini Endpoint
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._active_model}:generateContent?key={api_key}"
            payload = {
                "system_instruction": {"parts": [{"text": CLINICAL_SYNTHESIS_SYSTEM_PROMPT}]},
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "response_mime_type": "application/json"
                }
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as exc:
            logger.error(f"Gemini REST API call failed: {exc}")
            return None

    def _call_openai(self, prompt: str) -> Optional[str]:
        """Calls OpenAI API."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        try:
            url = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": self._active_model,
                "messages": [
                    {"role": "system", "content": CLINICAL_SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.error(f"OpenAI API call failed: {exc}")
            return None

    def _call_anthropic(self, prompt: str) -> Optional[str]:
        """Calls Anthropic API."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        try:
            url = "https://api.anthropic.com/v1/messages"
            payload = {
                "model": self._active_model,
                "max_tokens": 1024,
                "system": CLINICAL_SYNTHESIS_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"}
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["content"][0]["text"]
        except Exception as exc:
            logger.error(f"Anthropic API call failed: {exc}")
            return None

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Calls local Ollama REST API."""
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            url = f"{base_url}/api/generate"
            payload = {
                "model": self._active_model,
                "system": CLINICAL_SYNTHESIS_SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response")
        except Exception as exc:
            logger.debug(f"Ollama local call failed: {exc}")
            return None

    def _extractive_fallback_synthesis(
        self,
        query: str,
        retrieved_passages: List[Tuple[Chunk, float]]
    ) -> ClinicalSynthesisResponse:
        """Heuristic fallback for offline execution with canonical confidence tier."""
        top_chunk, _ = retrieved_passages[0]
        citations = [
            {
                "chunk_id": chk.chunk_id,
                "document_id": chk.document_id,
                "document_name": chk.document_name,
                "section": chk.section_title,
                "section_title": chk.section_title,
                "page_number": chk.page_number,
                "source_url": chk.source_url or GUIDELINE_SOURCE_URLS.get(chk.document_id)
            }
            for chk, _ in retrieved_passages[:3]
        ]
        return ClinicalSynthesisResponse(
            query=query,
            direct_answer=f"According to {top_chunk.document_name} ({top_chunk.section_title}, Page {top_chunk.page_number}) [{top_chunk.chunk_id}], clinical guideline evidence indicates appropriate assessment and screening based on validated risk factors and DXA measurement.",
            target_population=top_chunk.population,
            key_recommendations=[f"{chk.text[:180].strip()}... [{chk.chunk_id}]" for chk, _ in retrieved_passages[:3]],
            citations=citations,
            evidence_strength=ConfidenceTier.MEDIUM.value,
            clinical_caveats=["Ensure complete clinical evaluation and review of secondary osteoporosis causes."],
            provider="extractive_fallback",
            model="heuristic"
        )

    def _mock_synthesis(
        self,
        query: str,
        retrieved_passages: List[Tuple[Chunk, float]]
    ) -> ClinicalSynthesisResponse:
        """Deterministic mock synthesizer for unit testing."""
        top_chunk, _ = retrieved_passages[0]
        return ClinicalSynthesisResponse(
            query=query,
            direct_answer=f"According to guideline recommendations in [{top_chunk.chunk_id}], patients meeting formal risk criteria should receive DXA bone measurement testing.",
            target_population=top_chunk.population,
            key_recommendations=[f"Offer DXA scan to measure bone mineral density in eligible patients [{top_chunk.chunk_id}]."],
            citations=[{
                "chunk_id": top_chunk.chunk_id,
                "document_id": top_chunk.document_id,
                "document_name": top_chunk.document_name,
                "section": top_chunk.section_title,
                "section_title": top_chunk.section_title,
                "page_number": top_chunk.page_number,
                "source_url": top_chunk.source_url or GUIDELINE_SOURCE_URLS.get(top_chunk.document_id)
            }],
            evidence_strength=ConfidenceTier.HIGH.value,
            clinical_caveats=["Recalculate 10-year fracture risk after central DXA completion."],
            provider="gemini_mock",
            model="gemini-1.5-flash-stub"
        )
