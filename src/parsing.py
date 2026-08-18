"""
PDF Document Extraction Module (Step 1: Parsing ONLY).

Responsibilities:
- PDF layout extraction via unstructured.partition.pdf with robust page tracking
- Structural element filtering (drops Header, Footer, PageBreak from content)
- Accurate 1-indexed monotonic page tracking across element streams
- Guideline source_url and document metadata tracking on Page contracts
- Built-in multi-page PDF Flate stream fallback extraction with page boundary recovery
- Recursive discovery and syncing of PDF guideline files
"""

import os
import zlib
import re
import shutil
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from src.config import (
    DEFAULT_GUIDELINES_DIR,
    DATA_DIR,
    SCANNED_DOC_MIN_CHARS,
    DEFAULT_PARTITION_STRATEGY,
    GUIDELINE_SOURCE_URLS
)

logger = logging.getLogger(__name__)

# Registry mapping known guideline file stems to official public URLs
SOURCE_URL_REGISTRY: Dict[str, str] = {
    "osteoporosis-risk-assessment-pdf-66144025463749": "https://www.nice.org.uk/guidance/ng259",
    "osteoporosis-screening-final-recommendation": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/osteoporosis-screening",
    "nice_ng259": "https://www.nice.org.uk/guidance/ng259",
    "uspstf_osteoporosis": "https://www.uspreventiveservicestaskforce.org/uspstf/recommendation/osteoporosis-screening",
    **GUIDELINE_SOURCE_URLS
}


@dataclass
class Page:
    """Represents an extracted page with structural elements, lineage, and source provenance."""
    page_number: int
    text: str
    elements: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def filter_structural_noise(elements: List[Any]) -> List[Any]:
    """
    Drops structural layout noise elements (Header, Footer, PageBreak)
    tied to extraction from content.
    """
    cleaned: List[Any] = []
    for el in elements:
        cat = getattr(el, "category", None) or (el.get("type") if isinstance(el, dict) else getattr(el, "type", type(el).__name__))
        if cat in ("Header", "Footer", "PageBreak"):
            continue
        cleaned.append(el)
    return cleaned


def partition_pdf_pages(
    pdf_path: Union[str, Path],
    strategy: str = DEFAULT_PARTITION_STRATEGY,
    source_url: Optional[str] = None
) -> List[Page]:
    """
    Extract raw pages from a PDF document using Unstructured with robust
    page-boundary tracking and source URL lineage attribution.

    Args:
        pdf_path: Path to the target PDF file.
        strategy: Unstructured partition strategy ('fast' or 'hi_res').
        source_url: Optional web URL of the official guideline.

    Returns:
        List[Page]: Monotonically sequenced, structurally filtered Page objects.
    """
    path_obj = Path(pdf_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"PDF document not found at: {path_obj.resolve()}")

    # Resolve source URL from parameter or central registry (None/empty if unknown, no fabrication)
    resolved_source_url = source_url or SOURCE_URL_REGISTRY.get(path_obj.stem)

    # 1. Narrow exception handling for Unstructured call only
    raw_elements = None
    try:
        from unstructured.partition.pdf import partition_pdf

        raw_elements = partition_pdf(
            filename=str(path_obj),
            strategy=strategy,
            include_page_breaks=True
        )
    except ImportError:
        logger.warning("Unstructured library not installed. Falling back to built-in stream extraction.")
        return _fallback_stream_extraction(path_obj, source_url=resolved_source_url)
    except Exception as exc:
        logger.error(f"Unstructured extraction failed for '{path_obj}': {exc}. Using fallback stream parser.")
        return _fallback_stream_extraction(path_obj, source_url=resolved_source_url)

    if not raw_elements:
        return _fallback_stream_extraction(path_obj, source_url=resolved_source_url)

    # 2. Assembly logic separated from library call
    pages_dict: Dict[int, List[str]] = {}
    page_elements: Dict[int, List[Dict[str, Any]]] = {}

    current_page = 1
    for el in raw_elements:
        cat = getattr(el, "category", None) or (el.get("type") if isinstance(el, dict) else getattr(el, "type", type(el).__name__))

        # PageBreak handling: advance page tracker and skip content inclusion
        if cat == "PageBreak":
            current_page += 1
            continue

        # Fix 3: Explicit None check for page_number (avoids treating page 0 as falsy)
        el_metadata = getattr(el, "metadata", None)
        if el_metadata is not None:
            meta_pg = getattr(el_metadata, "page_number", None)
            if meta_pg is not None:
                try:
                    pg_val = int(meta_pg)
                    # Normalize 0-indexed to 1-indexed
                    current_page = pg_val if pg_val >= 1 else pg_val + 1
                except (ValueError, TypeError):
                    pass

        # Filter out structural layout noise from page content
        if cat in ("Header", "Footer"):
            continue

        el_text = (el.text if hasattr(el, "text") else (el.get("text") if isinstance(el, dict) else str(el))).strip()
        if not el_text:
            continue

        if current_page not in pages_dict:
            pages_dict[current_page] = []
            page_elements[current_page] = []

        pages_dict[current_page].append(el_text)
        page_elements[current_page].append({
            "type": cat,
            "text": el_text,
            "page_number": current_page
        })

    pages: List[Page] = []
    for pg_num in sorted(pages_dict.keys()):
        combined_text = "\n\n".join(pages_dict[pg_num])
        if combined_text.strip():
            pages.append(Page(
                page_number=pg_num,
                text=combined_text,
                elements=page_elements[pg_num],
                metadata={
                    "engine": "unstructured",
                    "pdf_path": str(path_obj),
                    "document_name": path_obj.stem,
                    "source_url": resolved_source_url,
                    "page_numbers_unreliable": False,
                    "element_count": len(page_elements[pg_num])
                }
            ))

    if pages and sum(len(p.text) for p in pages) >= SCANNED_DOC_MIN_CHARS:
        return pages

    return _fallback_stream_extraction(path_obj, source_url=resolved_source_url)


def _fallback_stream_extraction(
    path_obj: Path,
    source_url: Optional[str] = None
) -> List[Page]:
    """
    Fallback parser extracting text streams directly from PDF with
    page-boundary recovery and source URL attachment.
    """
    try:
        with open(path_obj, "rb") as f:
            content = f.read()
    except IOError as exc:
        logger.error(f"Failed to read file '{path_obj}': {exc}")
        return []

    resolved_source_url = source_url or SOURCE_URL_REGISTRY.get(path_obj.stem)

    # 1. Attempt page boundary detection via /Type /Page markers
    # Split content by /Type /Page objects in PDF stream
    page_chunks: List[str] = []
    page_splits = re.split(rb'/Type\s*/Page\b', content)

    if len(page_splits) > 1:
        # Detected discrete page objects
        for page_bytes in page_splits[1:]:
            page_text_parts: List[str] = []
            stream_matches = re.findall(rb'stream\r?\n(.*?)\r?\nendstream', page_bytes, re.DOTALL)
            for raw_stream in stream_matches:
                try:
                    decomp = zlib.decompress(raw_stream)
                    for tm in re.findall(rb'\((.*?)\)\s*Tj', decomp):
                        page_text_parts.append(tm.decode("latin1", errors="ignore"))
                    for am in re.findall(rb'\[(.*?)\]\s*TJ', decomp):
                        for inner in re.findall(rb'\((.*?)\)', am):
                            page_text_parts.append(inner.decode("latin1", errors="ignore"))
                except Exception:
                    continue

            # Check for direct text matches in page block
            direct_matches = re.findall(rb'\((.*?)\)\s*Tj', page_bytes)
            for dm in direct_matches:
                page_text_parts.append(dm.decode("latin1", errors="ignore"))

            page_content = " ".join(page_text_parts).strip()
            if page_content:
                page_chunks.append(page_content)

    # 2. Fallback to global stream extraction if page objects were not separated
    if not page_chunks:
        global_text_parts: List[str] = []
        stream_matches = re.findall(rb'stream\r?\n(.*?)\r?\nendstream', content, re.DOTALL)
        for raw_stream in stream_matches:
            try:
                decomp = zlib.decompress(raw_stream)
                for tm in re.findall(rb'\((.*?)\)\s*Tj', decomp):
                    global_text_parts.append(tm.decode("latin1", errors="ignore"))
                for am in re.findall(rb'\[(.*?)\]\s*TJ', decomp):
                    for inner in re.findall(rb'\((.*?)\)', am):
                        global_text_parts.append(inner.decode("latin1", errors="ignore"))
            except Exception:
                continue

        full_global_text = " ".join(global_text_parts).strip()

        # Check for form feed delimiters (\f)
        if "\x0c" in full_global_text or "\f" in full_global_text:
            form_feed_pages = re.split(r'[\x0c\f]', full_global_text)
            page_chunks = [p.strip() for p in form_feed_pages if p.strip()]
        elif full_global_text:
            # Single stream without recoverable page boundaries -> mark unreliable
            return [Page(
                page_number=1,
                text=full_global_text,
                elements=[{"type": "NarrativeText", "text": full_global_text, "page_number": 1}],
                metadata={
                    "engine": "stream_parser",
                    "pdf_path": str(path_obj),
                    "document_name": path_obj.stem,
                    "source_url": resolved_source_url,
                    "page_numbers_unreliable": True
                }
            )]

    if not page_chunks:
        return [Page(
            page_number=1,
            text="",
            elements=[],
            metadata={
                "engine": "stream_parser",
                "pdf_path": str(path_obj),
                "document_name": path_obj.stem,
                "source_url": resolved_source_url,
                "page_numbers_unreliable": True
            }
        )]

    # Map recovered page chunks into sequentially tracked Page objects
    pages: List[Page] = []
    for idx, page_txt in enumerate(page_chunks, start=1):
        clean_txt = page_txt.strip()
        if clean_txt:
            pages.append(Page(
                page_number=idx,
                text=clean_txt,
                elements=[{"type": "NarrativeText", "text": clean_txt, "page_number": idx}],
                metadata={
                    "engine": "stream_parser",
                    "pdf_path": str(path_obj),
                    "document_name": path_obj.stem,
                    "source_url": resolved_source_url,
                    "page_numbers_unreliable": False,
                    "stream_page_index": idx
                }
            ))

    return pages


def discover_and_sync_guidelines(
    input_dir: Union[str, Path] = DEFAULT_GUIDELINES_DIR,
    fallback_dir: Union[str, Path] = DATA_DIR
) -> List[Path]:
    """
    Discover all PDF guideline files in input_dir recursively (**/*.pdf),
    syncing from fallback_dir recursively (**/*.pdf) if needed.
    """
    input_path = Path(input_dir)
    fallback_path = Path(fallback_dir)
    input_path.mkdir(parents=True, exist_ok=True)

    # Recursive sync from fallback_path to input_path
    if input_path.resolve() == DEFAULT_GUIDELINES_DIR.resolve() and fallback_path.exists() and fallback_path.resolve() != input_path.resolve():
        for root_pdf in fallback_path.glob("**/*.pdf"):
            dest = input_path / root_pdf.name
            if not dest.exists():
                try:
                    shutil.copy2(root_pdf, dest)
                except Exception as e:
                    logger.debug(f"Sync copy notice: {e}")

    # Recursive search in input_path
    found_pdfs = sorted(input_path.glob("**/*.pdf"))
    if not found_pdfs and fallback_path.exists():
        found_pdfs = sorted(fallback_path.glob("**/*.pdf"))
    return found_pdfs
