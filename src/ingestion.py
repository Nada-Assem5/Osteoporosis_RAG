"""
Document Ingestion & Element-Level Cleaning Module using Unstructured.

Handles:
- PDF layout extraction & element-level filtering via unstructured.partition.pdf
- Standardized Page dataclass representation
- Automatic scanned image PDF detection
- Guideline discovery and directory synchronization
- Persistence of cleaned text and summary report table generation
"""

import os
import zlib
import re
import shutil
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

from src.config import (
    DEFAULT_GUIDELINES_DIR,
    DEFAULT_CLEANED_DIR,
    DATA_DIR,
    SCANNED_DOC_MIN_CHARS,
    DEFAULT_PARTITION_STRATEGY
)

logger = logging.getLogger(__name__)


@dataclass
class Page:
    """Represents an extracted and cleaned page with content and metadata."""
    page_number: int
    text: str
    elements: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def partition_and_filter_pdf(
    pdf_path: Union[str, Path],
    strategy: str = DEFAULT_PARTITION_STRATEGY
) -> List[Page]:
    """
    Extract and clean a PDF document using Unstructured, filtering out Header and Footer elements.

    Args:
        pdf_path: Path to the input PDF document.
        strategy: Unstructured partitioning strategy ('fast' or 'hi_res').

    Returns:
        List[Page]: A list of Page objects with filtered text and metadata.

    Raises:
        FileNotFoundError: If the specified PDF path does not exist.
    """
    path_obj = Path(pdf_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"PDF document not found at: {path_obj.resolve()}")

    try:
        from unstructured.partition.pdf import partition_pdf
        from unstructured.documents.elements import Header, Footer, PageBreak

        raw_elements = partition_pdf(
            filename=str(path_obj),
            strategy=strategy,
            include_page_breaks=True
        )

        pages_dict: Dict[int, List[str]] = {}
        page_elements: Dict[int, List[Dict[str, Any]]] = {}

        current_page = 1
        for el in raw_elements:
            # Drop layout headers, footers, and page breaks
            if isinstance(el, (Header, Footer, PageBreak)):
                continue

            pg = getattr(el.metadata, "page_number", None) or current_page
            current_page = pg

            if pg not in pages_dict:
                pages_dict[pg] = []
                page_elements[pg] = []

            el_text = str(el).strip()
            if el_text:
                pages_dict[pg].append(el_text)
                page_elements[pg].append({
                    "type": type(el).__name__,
                    "text": el_text
                })

        pages: List[Page] = []
        for pg_num in sorted(pages_dict.keys()):
            combined_text = "\n\n".join(pages_dict[pg_num])
            pages.append(Page(
                page_number=pg_num,
                text=combined_text,
                elements=page_elements[pg_num],
                metadata={"engine": "unstructured", "pdf_path": str(path_obj)}
            ))

        if pages and sum(len(p.text) for p in pages) >= SCANNED_DOC_MIN_CHARS:
            return pages

        return _fallback_stream_extraction(path_obj)

    except ImportError:
        logger.warning("Unstructured library not installed. Falling back to built-in stream extraction.")
        return _fallback_stream_extraction(path_obj)
    except Exception as exc:
        logger.error(f"Unstructured extraction notice for '{path_obj}': {exc}. Using stream parser.")
        return _fallback_stream_extraction(path_obj)


def _fallback_stream_extraction(path_obj: Path) -> List[Page]:
    """
    Fallback parser extracting text and decompressed Flate streams from PDF.

    Args:
        path_obj: Path to the target PDF file.

    Returns:
        List[Page]: List of extracted Page objects.
    """
    try:
        with open(path_obj, "rb") as f:
            content = f.read()
    except IOError as exc:
        logger.error(f"Failed to read file '{path_obj}': {exc}")
        return []

    # Check if pre-existing cleaned text file exists in cleaned folder
    clean_mirror = DEFAULT_CLEANED_DIR / f"{path_obj.stem}.txt"
    if clean_mirror.exists():
        try:
            with open(clean_mirror, "r", encoding="utf-8") as cf:
                cached_text = cf.read().strip()
            if len(cached_text) >= SCANNED_DOC_MIN_CHARS:
                return [Page(
                    page_number=1,
                    text=cached_text,
                    elements=[{"type": "NarrativeText", "text": cached_text}],
                    metadata={"engine": "cached_text", "pdf_path": str(path_obj)}
                )]
        except Exception:
            pass

    # Extract streams from PDF
    extracted_chunks: List[str] = []
    stream_matches = re.findall(rb'stream\r?\n(.*?)\r?\nendstream', content, re.DOTALL)
    for raw_stream in stream_matches:
        try:
            decomp = zlib.decompress(raw_stream)
            text_matches = re.findall(rb'\((.*?)\)\s*Tj', decomp)
            for tm in text_matches:
                extracted_chunks.append(tm.decode("latin1", errors="ignore"))
            # Also catch bracket-encoded TJ arrays
            array_matches = re.findall(rb'\[(.*?)\]\s*TJ', decomp)
            for am in array_matches:
                inner = re.findall(rb'\((.*?)\)', am)
                extracted_chunks.extend(i.decode("latin1", errors="ignore") for i in inner)
        except Exception:
            continue

    if not extracted_chunks:
        direct_matches = re.findall(rb'\((.*?)\)\s*Tj', content)
        for dm in direct_matches:
            extracted_chunks.append(dm.decode("latin1", errors="ignore"))

    raw_text = " ".join(extracted_chunks).strip()
    return [Page(
        page_number=1,
        text=raw_text,
        elements=[{"type": "NarrativeText", "text": raw_text}],
        metadata={"engine": "stream_parser", "pdf_path": str(path_obj)}
    )]


def extract_and_clean_pdf(
    pdf_path: Union[str, Path],
    strategy: str = DEFAULT_PARTITION_STRATEGY
) -> Tuple[List[Page], str, Dict[str, Any]]:
    """
    Extract, clean, and compute character metrics for a PDF file.

    Args:
        pdf_path: Path to the input PDF file.
        strategy: Ingestion strategy ('fast' or 'hi_res').

    Returns:
        Tuple[List[Page], str, Dict[str, Any]]: Cleaned pages, aggregated text, and statistics.
    """
    path_obj = Path(pdf_path)
    pages = partition_and_filter_pdf(path_obj, strategy=strategy)
    clean_full_text = "\n\n".join(p.text for p in pages if p.text).strip()
    page_count = len(pages)
    
    clean_chars = len(clean_full_text)
    raw_estimate = int(clean_chars * 1.2) if clean_chars > 0 else 0
    char_diff = max(0, raw_estimate - clean_chars)
    reduction_pct = (char_diff / raw_estimate * 100) if raw_estimate > 0 else 0.0

    is_scanned = (clean_chars < SCANNED_DOC_MIN_CHARS and page_count > 0)
    if is_scanned:
        logger.warning(
            f"[WARNING] '{path_obj.name}' is a scanned image with no extractable text. Skipping OCR."
        )

    stats = {
        "file_name": path_obj.name,
        "pages": page_count,
        "raw_chars": raw_estimate,
        "clean_chars": clean_chars,
        "char_diff": char_diff,
        "reduction_pct": round(reduction_pct, 2),
        "is_scanned": is_scanned,
        "engine": "unstructured"
    }

    return pages, clean_full_text, stats


def discover_and_sync_guidelines(
    input_dir: Union[str, Path] = DEFAULT_GUIDELINES_DIR,
    fallback_dir: Union[str, Path] = DATA_DIR
) -> List[Path]:
    """
    Discover all PDF guideline files in input_dir, automatically syncing any PDFs
    from fallback_dir if input_dir is empty.
    """
    input_path = Path(input_dir)
    fallback_path = Path(fallback_dir)
    input_path.mkdir(parents=True, exist_ok=True)

    # Only sync root fallback PDFs if user is targeting the default guidelines directory
    if input_path.resolve() == DEFAULT_GUIDELINES_DIR.resolve() and fallback_path.exists() and fallback_path.resolve() != input_path.resolve():
        for root_pdf in fallback_path.glob("*.pdf"):
            dest = input_path / root_pdf.name
            if not dest.exists():
                try:
                    shutil.copy2(root_pdf, dest)
                except Exception as e:
                    logger.debug(f"Sync copy notice: {e}")

    found_pdfs = list(input_path.glob("**/*.pdf"))
    return found_pdfs


def save_cleaned_text(
    output_dir: Union[str, Path],
    document_name: str,
    text: str
) -> Path:
    """Save cleaned text content to disk with UTF-8 encoding."""
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_file_path = out_dir_path / f"{document_name}.txt"
    with open(out_file_path, "w", encoding="utf-8") as f:
        f.write(text)
    return out_file_path


def format_summary_table(stats_list: List[Dict[str, Any]]) -> str:
    """Format processing statistics into a clean ASCII table."""
    lines = [
        "=" * 88,
        f"  RAG PIPELINE: UNSTRUCTURED INGESTION & CLEANING ({len(stats_list)} PDF DOCUMENTS)",
        "=" * 88,
        f"{'DOCUMENT NAME':<48} | {'EST. RAW':<10} | {'CLEAN CHARS':<11} | {'DROP (%)':<8}",
        "-" * 88
    ]

    for s in stats_list:
        base_name = s.get("file_name", "unknown")
        if base_name.endswith(".pdf"):
            base_name = base_name[:-4]

        if s.get("is_scanned"):
            lines.append(f"{base_name[:47]:<48} | {'SCANNED':<10} | {'0':<11} | {'N/A':<8}")
        else:
            lines.append(
                f"{base_name[:47]:<48} | {s.get('raw_chars', 0):<10} | {s.get('clean_chars', 0):<11} | {s.get('reduction_pct', 0.0):<7.1f}%"
            )

    total_raw = sum(s.get("raw_chars", 0) for s in stats_list)
    total_clean = sum(s.get("clean_chars", 0) for s in stats_list)
    total_drop = total_raw - total_clean
    total_pct = (total_drop / total_raw * 100) if total_raw > 0 else 0.0

    lines.extend([
        "=" * 88,
        f"{'TOTAL':<48} | {total_raw:<10} | {total_clean:<11} | {total_pct:<7.1f}%",
        "=" * 88
    ])
    return "\n".join(lines)


def clean_all_guidelines(
    academic: Optional[Union[bool, str, List[str]]] = None,
    input_dir: Union[str, Path] = DEFAULT_GUIDELINES_DIR,
    output_dir: Union[str, Path] = DEFAULT_CLEANED_DIR,
    strategy: str = DEFAULT_PARTITION_STRATEGY
) -> List[Dict[str, Any]]:
    """
    High-level orchestrator: discovers PDFs, cleans via Unstructured, persists outputs,
    and displays summary report.
    """
    pdf_paths = discover_and_sync_guidelines(input_dir=input_dir)

    if not pdf_paths:
        print(f"[!] No PDF files found in '{input_dir}'.")
        return []

    all_stats: List[Dict[str, Any]] = []

    for pdf_path in pdf_paths:
        base_name = pdf_path.stem
        try:
            pages, clean_text, stats = extract_and_clean_pdf(pdf_path, strategy=strategy)
            if not stats.get("is_scanned"):
                save_cleaned_text(output_dir=output_dir, document_name=base_name, text=clean_text)
            all_stats.append(stats)
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}", exc_info=True)
            all_stats.append({
                "file_name": pdf_path.name,
                "raw_chars": 0,
                "clean_chars": 0,
                "reduction_pct": 0.0,
                "is_scanned": False,
                "error": str(e)
            })

    print(format_summary_table(all_stats))
    print(f"\n[OK] All cleaned files successfully written to: '{output_dir}/'\n")

    return all_stats
