"""
Stage 1: PDF Document Ingestion & Layout Element Extraction (scripts/Ingest.py).

Responsibilities:
- Inspects data/raw/*.pdf documents
- Computes content-based document_id (SHA-256 hex digest of file BYTES, not filename)
- Retains human-readable document_name from filename
- Integrates with data/sources.json for guideline metadata and source URLs
- Filters layout noise (Headers, Footers, PageBreaks) and standardizes text
- Tracks running section_title context per element (required for Stage 2
  section-aware chunking and for the document_name/page_number/section_title/
  chunk_id/source_url metadata schema)
- Verifies that guideline sources belong to an approved public/official list
  (WHO, CDC, NICE, USPSTF, etc.) per the event's data-sourcing scope
- Emits element_id hashes and saves data/processed/elements.json
- Displays a unified Ingestion Summary block
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.schema import Page
from src.utils import (
    compute_content_hash,
    clean_text,
    normalize_unicode,
    dehyphenate_text,
    normalize_whitespace,
    strip_punctuation
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = ROOT_DIR / os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = ROOT_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed")
SOURCES_JSON_PATH = ROOT_DIR / os.getenv("SOURCES_JSON_PATH", "data/sources.json")
ELEMENTS_JSON_PATH = PROCESSED_DATA_DIR / "elements.json"

# Approved public / official guideline source organizations per the event's
# "Data Sourcing" scope requirement (official, public guideline PDFs only —
# no private or credential-gated data). Extend as needed for your event.
ALLOWED_SOURCE_ORGS = {"WHO", "CDC", "NICE", "USPSTF"}


def load_sources_config(sources_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load guideline source metadata from data/sources.json without hardcoded dicts."""
    target_path = Path(sources_path) if sources_path else SOURCES_JSON_PATH
    if not target_path.exists():
        logger.warning(f"[WARN] sources.json not found at '{target_path}'. Using empty sources config.")
        return {}
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error(f"[ERROR] Failed to load sources config from '{target_path}': {exc}")
        return {}


def verify_pdf_integrity(pdf_path: Path) -> bool:
    """Verify that a PDF file exists, is non-empty, and has a valid PDF magic header."""
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        return False
    try:
        with open(pdf_path, "rb") as f:
            header = f.read(5)
            return header.startswith(b"%PDF")
    except Exception:
        return False


def filter_structural_noise(elements: List[Any]) -> List[Any]:
    """
    Drop structural layout artifacts: Header, Footer, and PageBreak elements,
    and remove empty text blocks.
    """
    filtered = []
    noise_categories = {"Header", "Footer", "PageBreak"}

    for el in elements:
        category = getattr(el, "category", getattr(el, "type", None))
        if not category and isinstance(el, dict):
            category = el.get("type", el.get("category", ""))

        if category in noise_categories:
            continue

        text = getattr(el, "text", "") if not isinstance(el, dict) else el.get("text", "")
        if not text or not str(text).strip():
            continue

        filtered.append(el)

    return filtered


def partition_pdf_pages(pdf_path: Path) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extract layout elements page by page.
    Attempts unstructured partition_pdf first, falling back to pypdf / stream parser.
    Returns (raw_element_dicts, method_used).
    """
    method = "unstructured"
    elements_raw = []

    # Attempt 1: Unstructured layout parser
    try:
        from unstructured.partition.pdf import partition_pdf
        unstructured_elements = partition_pdf(
            filename=str(pdf_path),
            strategy="fast",
            include_page_breaks=True
        )
        for el in unstructured_elements:
            page_num = getattr(el.metadata, "page_number", 1) if hasattr(el, "metadata") and el.metadata else 1
            cat = getattr(el, "category", "NarrativeText")
            elements_raw.append({
                "type": cat,
                "text": str(el.text),
                "page_number": int(page_num) if page_num else 1,
                "metadata": getattr(el, "metadata", {}).to_dict() if hasattr(getattr(el, "metadata", None), "to_dict") else {}
            })
        if elements_raw:
            return elements_raw, "unstructured"
    except Exception as exc:
        logger.debug(f"Unstructured parsing unavailable or failed for {pdf_path.name} ('{exc}'). Falling back to stream parser.")

    # Attempt 2: pypdf fallback stream parser
    method = "stream_fallback"
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
            for line in lines:
                # Basic layout classification heuristic
                if len(line) < 80 and (line.startswith("Table") or line.startswith("Figure") or (line[0].isdigit() and "." in line[:5])):
                    cat = "Title"
                else:
                    cat = "NarrativeText"
                elements_raw.append({
                    "type": cat,
                    "text": line,
                    "page_number": page_idx,
                    "metadata": {"page_number": page_idx, "filename": pdf_path.name}
                })
    except Exception as exc:
        logger.error(f"[ERROR] Stream parser also failed for {pdf_path.name}: {exc}")
        method = "failed"

    return elements_raw, method


def ingest_guidelines(
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    sources_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Main Stage 1 Orchestrator:
    - Reads raw PDFs
    - Computes byte-based SHA-256 document_id
    - Maps metadata from sources.json
    - Verifies source_org is on the approved public-guideline list
    - Partitions and filters noise
    - Tracks a running section_title per element for downstream chunking
    - Saves processed/elements.json
    - Outputs a unified summary block
    """
    raw_path = Path(raw_dir) if raw_dir else RAW_DATA_DIR
    proc_path = Path(processed_dir) if processed_dir else PROCESSED_DATA_DIR
    proc_path.mkdir(parents=True, exist_ok=True)

    sources_map = load_sources_config(sources_path)
    pdf_files = sorted(list(raw_path.glob("*.pdf")))

    total_pdfs = len(pdf_files)
    unstructured_count = 0
    stream_fallback_count = 0
    failed_files = []
    missing_source_urls = []
    docs_without_sections: List[str] = []
    non_standard_sources: List[str] = []

    all_cleaned_elements: List[Dict[str, Any]] = []

    for pdf_file in pdf_files:
        doc_name = pdf_file.stem
        filename = pdf_file.name

        if not verify_pdf_integrity(pdf_file):
            logger.error(f"[ERROR] Corrupt or invalid PDF file: {filename}")
            failed_files.append(filename)
            continue

        # Compute document_id from file BYTES
        try:
            with open(pdf_file, "rb") as f:
                file_bytes = f.read()
            doc_id = compute_content_hash(file_bytes, length=12)
        except Exception as exc:
            logger.error(f"[ERROR] Failed reading bytes for {filename}: {exc}")
            failed_files.append(filename)
            continue

        # Metadata lookup from sources.json
        source_meta = sources_map.get(doc_name) or sources_map.get(filename) or sources_map.get(doc_id) or {}
        source_url = source_meta.get("source_url")
        if not source_url:
            logger.warning(
                f"[WARN] No source_url configured for {filename} in data/sources.json - citations for this document will omit URL"
            )
            missing_source_urls.append(filename)

        # Verify official / public source scope
        source_org = source_meta.get("source_org", "").strip()
        if source_org and source_org.upper() not in ALLOWED_SOURCE_ORGS:
            logger.warning(
                f"[WARN] '{filename}' has source_org='{source_org}' which is not in the approved public-guideline "
                f"list {sorted(ALLOWED_SOURCE_ORGS)}. Verify this is an official, non-credential-gated source before use."
            )
            non_standard_sources.append(filename)
        elif not source_org:
            logger.warning(f"[WARN] '{filename}' has no source_org set in sources.json - cannot verify official status")
            non_standard_sources.append(filename)

        # Partition PDF
        raw_elements, method = partition_pdf_pages(pdf_file)
        if method == "unstructured":
            unstructured_count += 1
        elif method == "stream_fallback":
            stream_fallback_count += 1
        else:
            failed_files.append(filename)
            continue

        # Filter noise
        filtered_elements = filter_structural_noise(raw_elements)

        # Clean text, track section context, and assign element_id hashes
        current_section_title = ""
        doc_has_section_titles = False

        for idx, el in enumerate(filtered_elements):
            raw_text = el.get("text", "")
            cleaned = clean_text(raw_text)
            if not cleaned:
                continue

            page_num = el.get("page_number", 1)
            elem_type = el.get("type", "NarrativeText")

            # Track running section context for downstream Stage-2 chunking.
            # Any "Title" element updates the active section until the next one.
            if elem_type in ("Title",):
                current_section_title = cleaned
                doc_has_section_titles = True

            elem_id = compute_content_hash(doc_id, str(page_num), str(idx), cleaned, length=12)

            elem_dict = {
                "element_id": elem_id,
                "document_id": doc_id,
                "document_name": doc_name,
                "type": elem_type,
                "text": cleaned,
                "page_number": page_num,
                "section_title": current_section_title,
                "source_url": source_url,
                "metadata": {
                    "page_number": page_num,
                    "section_title": current_section_title,
                    "filename": filename,
                    "file_directory": str(raw_path),
                    "source_org": source_meta.get("source_org", ""),
                    "title": source_meta.get("title", "")
                }
            }
            all_cleaned_elements.append(elem_dict)

        if not doc_has_section_titles:
            logger.warning(f"[WARN] No 'Title' elements detected in {filename} - chunking will lack section boundaries")
            docs_without_sections.append(filename)

    # Save to elements.json
    out_file = proc_path / "elements.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_cleaned_elements, f, indent=2, ensure_ascii=False)

    # Final Ingestion Summary Block
    print("\n" + "=" * 90)
    print("  STAGE 1: PDF INGESTION & STRUCTURAL FILTERING SUMMARY")
    print("=" * 90)
    print(f"  Total PDFs Processed              : {total_pdfs}")
    print(f"  Extracted via Unstructured        : {unstructured_count}")
    print(f"  Extracted via Stream Fallback     : {stream_fallback_count}")
    print(f"  Failed Files                      : {len(failed_files)} {('(' + ', '.join(failed_files) + ')') if failed_files else ''}")
    print(f"  Missing source_url in sources.json: {len(missing_source_urls)} {('(' + ', '.join(missing_source_urls) + ')') if missing_source_urls else ''}")
    print(f"  Documents Missing section_title   : {len(docs_without_sections)} {('(' + ', '.join(docs_without_sections) + ')') if docs_without_sections else ''}")
    print(f"  Non-Standard / Unverified Sources : {len(non_standard_sources)} {('(' + ', '.join(non_standard_sources) + ')') if non_standard_sources else ''}")
    print(f"  Total Layout Elements Extracted   : {len(all_cleaned_elements)}")
    print(f"  Saved Artifact Path               : {out_file}")
    print("=" * 90 + "\n")

    return all_cleaned_elements


run = ingest_guidelines
main = ingest_guidelines


if __name__ == "__main__":
    ingest_guidelines()