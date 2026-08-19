"""
Stage 2: Section-Aware Semantic Chunking & Metadata Enrichment (scripts/Chunk.py).

Responsibilities:
- Ingests data/processed/elements.json (or raw guideline text)
- Performs configurable section-aware chunking, target 400-800 tokens
  (default 400) with sliding overlap, per the event's chunking-strategy spec
- Generates deterministic content-based chunk_id hashes from (document_id + text + page_number)
- Prefers the section_title already computed in Stage 1 (running Title-based
  context); falls back to Title/Header element detection or regex heading
  matching, with "Unknown Section" as a last resort
- Enriches chunks with clinical taxonomy metadata (population, topics, issuer),
  preferring the authoritative source_org from data/sources.json over guesswork
- Emits standardized Chunk records and saves data/processed/chunks.json
"""

import os
import re
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.schema import Chunk
from src.utils import count_tokens, compute_content_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = ROOT_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed")
ELEMENTS_JSON_PATH = PROCESSED_DATA_DIR / "elements.json"
CHUNKS_JSON_PATH = PROCESSED_DATA_DIR / "chunks.json"

DEFAULT_CHUNK_SIZE_TOKENS = int(os.getenv("DEFAULT_CHUNK_SIZE_TOKENS", "400"))
DEFAULT_CHUNK_OVERLAP_TOKENS = int(os.getenv("DEFAULT_CHUNK_OVERLAP_TOKENS", "50"))
# Upper bound from the event's chunking-strategy spec ("400-800 token chunks").
# A finished chunk larger than this is flagged in the summary for review.
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "800"))

# Heading regex pattern for numbered sections, tables, and standard guideline headers
HEADING_REGEX = re.compile(
    r'^(?:(?:\d+\.[\d\.]*\s+[A-Z][^\n]{3,80})|(?:Table\s+\d+[^\n]{0,80})|(?:Figure\s+\d+[^\n]{0,80})|(?:Recommendation[s]?\s+[A-Z0-9\.\:\s]{2,80})|(?:Section\s+\d+[^\n]{0,80}))',
    re.MULTILINE | re.IGNORECASE
)


def extract_clinical_metadata(
    text: str,
    document_name: Optional[str] = None,
    issuer_hint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Enrich chunk with standardized clinical taxonomy:
    - Guideline issuer (NICE, USPSTF, etc.) - uses the authoritative source_org
      from data/sources.json (issuer_hint) when available, falling back to
      filename/text heuristics only when no configured source exists.
    - Clinical topics (Screening & Diagnosis, Risk Assessment, Vertebral Fracture, etc.)
    - Target populations (Women >= 65, Men >= 70, Younger Postmenopausal, etc.)
    """
    t_lower = text.lower() if text else ""
    doc_lower = document_name.lower() if document_name else ""

    # Guideline Issuer - prefer the authoritative, configured source_org
    if issuer_hint and issuer_hint.strip():
        issuer = issuer_hint.strip().upper()
    else:
        issuer = "Other"
        if "nice" in doc_lower or "ng259" in doc_lower or "national institute for health" in t_lower:
            issuer = "NICE"
        elif "uspstf" in doc_lower or "osteoporosis-screening" in doc_lower or "u.s. preventive services" in t_lower:
            issuer = "USPSTF"

    # Clinical Topics
    topics = []
    if any(k in t_lower for k in ["screen", "screening", "diagnos", "dxa", "bmd", "bone mineral density", "t-score", "osteopenia", "osteoporosis"]):
        topics.append("Screening & Diagnosis")
    if any(k in t_lower for k in ["frax", "qfracture", "ost", "orai", "fracture risk calculator", "risk prediction", "10-year probability", "risk assessment tool"]):
        topics.append("Risk Assessment Tools")
    if any(k in t_lower for k in ["vfa", "vertebral fracture assessment", "vertebral fracture", "lateral spine", "spine imaging", "morphometry"]):
        topics.append("Vertebral Fracture Assessment")
    if any(k in t_lower for k in ["bisphosphonate", "alendronate", "risedronate", "zoledronic", "zoledronate", "denosumab", "teriparatide", "romosozumab", "raloxifene", "pharmacological", "hormone replacement", "anabolic"]):
        topics.append("Pharmacological Interventions")
    if any(k in t_lower for k in ["calcium", "vitamin d", "exercise", "fall prevention", "lifestyle", "dietary", "nutrition", "smoking cessation"]):
        topics.append("Non-Pharmacological & Lifestyle")
    if any(k in t_lower for k in ["rescreen", "monitoring", "repeat dxa", "follow-up", "interval", "repeat risk"]):
        topics.append("Monitoring & Rescreening")

    # Target Populations
    population = []
    if any(k in t_lower for k in ["women aged 65", "women 65 and older", "women 65 years", "women ≥ 65", "women >= 65", "women older than 65"]):
        population.append("Women Aged >= 65")
    if any(k in t_lower for k in ["men aged 70", "men 70 and older", "men aged 75", "men 75 and older", "men older than 70"]):
        population.append("Men Aged >= 70")
    if any(k in t_lower for k in ["younger postmenopausal", "postmenopausal women younger than 65", "women aged under 65", "women younger than 65"]):
        population.append("Younger Postmenopausal Women (< 65)")
    if any(k in t_lower for k in ["glucocorticoid", "steroid", "prednisolone", "aromatase inhibitor", "androgen deprivation", "secondary risk", "hypogonadism", "hyperthyroidism", "rheumatoid arthritis"]):
        population.append("Adults with Secondary Risk Factors")
    if any(k in t_lower for k in ["previous fragility fracture", "prior fracture", "history of fracture", "prior fragility"]):
        population.append("Adults with Prior Fragility Fracture")

    return {
        "topics": sorted(list(set(topics))),
        "population": sorted(list(set(population))),
        "guideline_issuer": issuer
    }


def _split_oversized_text(text: str, max_tokens: int) -> List[str]:
    """
    Hard-split a single oversized text block (e.g. a large table or unbroken
    paragraph) into sub-pieces that each fit within max_tokens, so no single
    chunk can silently blow past the 400-800 token spec. Splits on sentence
    boundaries first, falling back to whitespace if a single "sentence" is
    still too large.
    """
    if count_tokens(text) <= max_tokens:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) <= 1:
        sentences = text.split()  # last-resort whitespace split

    pieces: List[str] = []
    buffer: List[str] = []
    buffer_tokens = 0

    for unit in sentences:
        u_tokens = count_tokens(unit)
        if buffer_tokens + u_tokens > max_tokens and buffer:
            pieces.append(" ".join(buffer).strip())
            buffer, buffer_tokens = [], 0
        buffer.append(unit)
        buffer_tokens += u_tokens

    if buffer:
        pieces.append(" ".join(buffer).strip())

    return [p for p in pieces if p.strip()]


def chunk_document(
    text: str,
    document_id: str = "guideline",
    target_chunk_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    section_title: Optional[str] = None,
    page_number: int = 1,
    document_name: Optional[str] = None,
    source_url: Optional[str] = None,
    issuer_hint: Optional[str] = None
) -> List[Chunk]:
    """
    Splits text into token-bounded Chunk objects with deterministic content-based chunk_ids.
    """
    if not text or not text.strip():
        return []

    doc_name = document_name or document_id
    sec_title = section_title if section_title and section_title.strip() else "Unknown Section"

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: List[Chunk] = []
    current_tokens: List[str] = []
    current_token_count = 0
    current_section = sec_title
    current_page = page_number

    def _finalize_chunk():
        chunk_text = "\n\n".join(current_tokens).strip()
        t_count = count_tokens(chunk_text)
        h_val = compute_content_hash(document_id, chunk_text, str(current_page), length=8)
        cid = f"{document_id}_chk_{h_val}"
        meta = extract_clinical_metadata(chunk_text, doc_name, issuer_hint=issuer_hint)
        chunks.append(Chunk(
            chunk_id=cid,
            document_id=document_id,
            document_name=doc_name,
            section_title=current_section,
            page_number=current_page,
            text=chunk_text,
            token_estimate=t_count,
            source_url=source_url,
            metadata=meta
        ))

    for para in paragraphs:
        # Check for page indicator
        page_match = re.search(r'---\s*Page\s+(\d+)\s*---', para, re.IGNORECASE)
        if page_match:
            current_page = int(page_match.group(1))

        # Check for inline section heading
        h_match = HEADING_REGEX.search(para)
        if h_match:
            detected_sec = h_match.group(0).strip()
            if len(detected_sec) > 3:
                current_section = detected_sec

        # Guard against a single oversized paragraph blowing past the token budget
        sub_paras = _split_oversized_text(para, target_chunk_tokens)
        if len(sub_paras) > 1:
            logger.warning(
                f"[WARN] Oversized paragraph in '{doc_name}' (page {current_page}) split into "
                f"{len(sub_paras)} sub-pieces to respect the {target_chunk_tokens}-token target."
            )

        for sub_para in sub_paras:
            para_token_len = count_tokens(sub_para)

            if current_token_count + para_token_len <= target_chunk_tokens:
                current_tokens.append(sub_para)
                current_token_count += para_token_len
            else:
                if current_tokens:
                    _finalize_chunk()

                    # Sliding overlap calculation
                    overlap_tokens: List[str] = []
                    overlap_count = 0
                    for prev_para in reversed(current_tokens):
                        p_len = count_tokens(prev_para)
                        if overlap_count + p_len <= chunk_overlap_tokens:
                            overlap_tokens.insert(0, prev_para)
                            overlap_count += p_len
                        else:
                            break
                    current_tokens = overlap_tokens
                    current_token_count = overlap_count

                current_tokens.append(sub_para)
                current_token_count += para_token_len

    if current_tokens:
        _finalize_chunk()

    return chunks


def chunk_extracted_elements(
    elements: List[Dict[str, Any]],
    target_chunk_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS
) -> List[Chunk]:
    """
    Converts structured layout elements into section-aware semantic Chunks.
    Preserves heading hierarchy, page lineage, and deterministic IDs.

    Section context resolution order (most to least authoritative):
      1. el["section_title"] as computed by Stage 1 (Ingest.py)
      2. This element's own type (Title/Header) or a regex heading match
      3. "Unknown Section" fallback
    """
    if not elements:
        return []

    # Group elements by document_id
    doc_elements: Dict[str, List[Dict[str, Any]]] = {}
    for el in elements:
        d_id = el.get("document_id", "guideline")
        doc_elements.setdefault(d_id, []).append(el)

    all_chunks: List[Chunk] = []
    oversized_chunk_count = 0
    unknown_section_chunk_count = 0

    for d_id, elem_list in doc_elements.items():
        doc_name = elem_list[0].get("document_name", d_id)
        source_url = elem_list[0].get("source_url")
        # Authoritative issuer from data/sources.json (set in Stage 1), preferred
        # over the text/filename heuristics in extract_clinical_metadata.
        issuer_hint = elem_list[0].get("metadata", {}).get("source_org", "")

        current_section = "Unknown Section"
        current_page = elem_list[0].get("page_number", 1)
        current_texts: List[str] = []
        current_token_count = 0

        def _finalize_and_append():
            nonlocal oversized_chunk_count, unknown_section_chunk_count
            chunk_text = "\n\n".join(current_texts).strip()
            t_est = count_tokens(chunk_text)
            h_val = compute_content_hash(d_id, chunk_text, str(current_page), length=8)
            cid = f"{d_id}_chk_{h_val}"
            meta = extract_clinical_metadata(chunk_text, doc_name, issuer_hint=issuer_hint)
            final_section = current_section or "Unknown Section"

            if t_est > MAX_CHUNK_TOKENS:
                oversized_chunk_count += 1
                logger.warning(
                    f"[WARN] Chunk {cid} for '{doc_name}' is {t_est} tokens, exceeding the "
                    f"{MAX_CHUNK_TOKENS}-token spec ceiling."
                )
            if final_section == "Unknown Section":
                unknown_section_chunk_count += 1

            all_chunks.append(Chunk(
                chunk_id=cid,
                document_id=d_id,
                document_name=doc_name,
                section_title=final_section,
                page_number=current_page,
                text=chunk_text,
                token_estimate=t_est,
                source_url=source_url,
                metadata=meta
            ))

        for el in elem_list:
            el_type = el.get("type", "NarrativeText")
            el_text = el.get("text", "").strip()
            if not el_text:
                continue

            page_num = el.get("page_number", current_page)

            # Section resolution: trust Stage 1's running section_title first
            stage1_section = el.get("section_title", "").strip() if el.get("section_title") else ""
            if stage1_section:
                current_section = stage1_section
            elif el_type in {"Title", "Header"}:
                current_section = el_text
            else:
                h_match = HEADING_REGEX.search(el_text)
                if h_match and len(h_match.group(0).strip()) > 3:
                    current_section = h_match.group(0).strip()

            # Guard against a single oversized element (e.g. a large table)
            sub_texts = _split_oversized_text(el_text, target_chunk_tokens)
            if len(sub_texts) > 1:
                logger.warning(
                    f"[WARN] Oversized element in '{doc_name}' (page {page_num}) split into "
                    f"{len(sub_texts)} sub-pieces to respect the {target_chunk_tokens}-token target."
                )

            for sub_text in sub_texts:
                el_token_count = count_tokens(sub_text)

                if current_token_count + el_token_count <= target_chunk_tokens:
                    current_texts.append(sub_text)
                    current_token_count += el_token_count
                    current_page = page_num
                else:
                    if current_texts:
                        _finalize_and_append()

                        # Sliding overlap
                        overlap_texts = []
                        overlap_tok = 0
                        for prev_t in reversed(current_texts):
                            pt_len = count_tokens(prev_t)
                            if overlap_tok + pt_len <= chunk_overlap_tokens:
                                overlap_texts.insert(0, prev_t)
                                overlap_tok += pt_len
                            else:
                                break
                        current_texts = overlap_texts
                        current_token_count = overlap_tok

                    current_texts.append(sub_text)
                    current_token_count += el_token_count
                    current_page = page_num

        if current_texts:
            _finalize_and_append()

    if oversized_chunk_count or unknown_section_chunk_count:
        logger.info(
            f"[INFO] Chunking audit: {oversized_chunk_count} chunk(s) over {MAX_CHUNK_TOKENS} tokens, "
            f"{unknown_section_chunk_count} chunk(s) with no resolvable section_title."
        )

    return all_chunks


def run_chunking(
    elements_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    target_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS
) -> List[Chunk]:
    """Execute Stage 2 chunking pipeline and serialize to chunks.json."""
    in_path = Path(elements_path) if elements_path else ELEMENTS_JSON_PATH
    out_path = Path(output_path) if output_path else CHUNKS_JSON_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        logger.error(f"[ERROR] elements.json not found at '{in_path}'. Run Stage 1 (Ingest.py) first.")
        return []

    with open(in_path, "r", encoding="utf-8") as f:
        elements = json.load(f)

    chunks = chunk_extracted_elements(elements, target_chunk_tokens=target_tokens, chunk_overlap_tokens=overlap_tokens)
    serialized = [c.to_dict() for c in chunks]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)

    oversized = sum(1 for c in chunks if c.token_estimate > MAX_CHUNK_TOKENS)
    unknown_section = sum(1 for c in chunks if c.section_title == "Unknown Section")

    print("\n" + "_" * 90)
    print("  STAGE 2: SECTION-AWARE SEMANTIC CHUNKING SUMMARY")
    print("=" * 90)
    print(f"  Input Elements Processed   : {len(elements)}")
    print(f"  Total Semantic Chunks      : {len(chunks)}")
    print(f"  Target Chunk Tokens        : {target_tokens} (Overlap: {overlap_tokens}, Ceiling: {MAX_CHUNK_TOKENS})")
    print(f"  Chunks Over Token Ceiling  : {oversized}")
    print(f"  Chunks with Unknown Section: {unknown_section}")
    print(f"  Saved Artifact Path        : {out_path}")
    print("=" * 90 + "\n")

    return chunks


run = run_chunking
main = run_chunking


if __name__ == "__main__":
    run_chunking()