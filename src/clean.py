"""
Text Cleaning & Normalization Module (Step 2: Cleaning ONLY).

Responsibilities:
- Short-title isolated noise detection (is_noise_title & filter_elements)
- Word-concatenation repair & spacing normalization via wordfreq (fix_concatenated_text)
- De-hyphenation across line breaks
- Academic article boilerplate & disclosure stripping (clean_academic_boilerplate)
- Operates purely on text or Page elements without requiring PDF/Unstructured knowledge
- Cleaned text persistence (*.txt) and character reduction reporting
"""

import re
import string
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union, Set

from src.config import (
    DEFAULT_GUIDELINES_DIR,
    DEFAULT_CLEANED_DIR,
    SCANNED_DOC_MIN_CHARS,
    DEFAULT_PARTITION_STRATEGY,
    SHORT_TITLE_THRESHOLD,
    MIN_CONTENT_LENGTH,
    ACADEMIC_NOISE_PATTERNS
)
from src.parsing import Page, partition_pdf_pages, discover_and_sync_guidelines

logger = logging.getLogger(__name__)

# Wordfreq initialization with fallback
try:
    from wordfreq import zipf_frequency
    _HAS_WORDFREQ = True
except Exception:
    _HAS_WORDFREQ = False

# Wordninja initialization with fallback
try:
    import wordninja
    _HAS_WORDNINJA = True
except Exception:
    _HAS_WORDNINJA = False

KNOWN_MEDICAL_VOCABULARY = {
    "osteoporosis", "dxa", "bmd", "vfa", "frax", "qfracture",
    "glucocorticoid", "prednisolone", "alendronate", "zoledronic",
    "denosumab", "teriparatide", "romosozumab", "raloxifene",
    "postmenopausal", "fragility", "calcidiol", "cholecalciferol",
    "uspstf", "nice", "who", "nhanes", "nof", "iscd", "iof", "bics",
    "bisphosphonates", "bisphosphonate", "vertebral", "calcium", "vitamin"
}

COMMON_ENGLISH_WORDS = {
    "hello", "screening", "treatment", "recommendation", "fracture", "density",
    "the", "of", "and", "a", "to", "in", "is", "you", "that", "it", "he", "was", "for",
    "on", "are", "as", "with", "his", "they", "i", "at", "be", "this", "have", "from",
    "or", "one", "had", "by", "word", "but", "not", "what", "all", "were", "we", "when",
    "your", "can", "said", "there", "use", "an", "each", "which", "she", "do", "how",
    "their", "if", "will", "up", "other", "about", "out", "many", "then", "them", "these",
    "so", "some", "her", "would", "make", "like", "him", "into", "time", "has", "look",
    "two", "more", "write", "go", "see", "number", "no", "way", "could", "people", "my",
    "than", "first", "water", "been", "call", "who", "oil", "its", "now", "find", "long",
    "down", "day", "did", "get", "come", "made", "may", "part", "over", "new", "sound",
    "take", "only", "little", "work", "know", "place", "year", "live", "me", "back", "give",
    "most", "very", "after", "thing", "our", "just", "name", "good", "sentence", "man",
    "think", "say", "great", "where", "help", "through", "much", "before", "line", "right",
    "too", "mean", "old", "any", "same", "tell", "boy", "follow", "came", "want", "show",
    "also", "around", "form", "three", "small", "set", "put", "end", "does", "another",
    "well", "large", "must", "big", "even", "such", "because", "turn", "here", "why",
    "ask", "went", "men", "read", "need", "land", "different", "home", "us", "move",
    "try", "kind", "hand", "picture", "again", "change", "off", "play", "spell", "air",
    "away", "animal", "house", "point", "page", "letter", "mother", "answer", "found",
    "study", "still", "learn", "should", "america", "world", "high", "every", "near",
    "add", "food", "between", "own", "below", "country", "plant", "last", "school",
    "father", "keep", "tree", "never", "start", "city", "earth", "eye", "light", "thought",
    "head", "under", "story", "saw", "left", "don't", "few", "while", "along", "might",
    "close", "something", "seem", "next", "hard", "open", "example", "begin", "life",
    "always", "those", "both", "paper", "together", "got", "group", "often", "run",
    "important", "until", "children", "side", "feet", "car", "mile", "night", "walk",
    "white", "sea", "began", "grow", "took", "river", "four", "carry", "state", "once",
    "book", "hear", "stop", "without", "second", "later", "miss", "idea", "enough", "eat",
    "face", "watch", "far", "indian", "really", "almost", "let", "above", "girl",
    "sometimes", "mountains", "cut", "young", "talk", "soon", "list", "song", "being",
    "leave", "family", "assess", "assessment", "clinical", "guideline", "guidelines",
    "patient", "patients", "evidence", "health", "management", "diagnosis", "therapy",
    "adult", "adults", "women", "age", "aged", "score", "risk", "scan", "scans", "table",
    "measure", "measurement", "testing", "prevent", "prevention", "fall", "falls", "intake"
}


def is_noise_title(
    elements_or_text: Union[List[Any], str],
    index: int = 0,
    min_content_length: int = MIN_CONTENT_LENGTH,
    threshold: int = SHORT_TITLE_THRESHOLD
) -> bool:
    """
    Detect isolated short noise titles.
    Supports either passing a List[Element] with an index, or raw title text string.
    """
    if isinstance(elements_or_text, list):
        if not elements_or_text or index >= len(elements_or_text):
            return True
        el = elements_or_text[index]
        s = (el.text if hasattr(el, "text") else (el.get("text") if isinstance(el, dict) else str(el))).strip()
        if not s:
            return True
        # Check if title itself contains section number or clinical keyword
        if re.match(r'^(?:\d+\.|\d+\.\d+|\d+\.\d+\.\d+|[A-Z]\.)\s+[A-Za-z]', s):
            return False
        if any(term in s.lower() for term in ["dxa", "bmd", "frax", "risk", "score", "scan", "bone", "osteoporosis", "screening"]):
            return False
        # Look ahead at the next non-header/footer element
        next_content = ""
        for nxt in elements_or_text[index + 1:]:
            nxt_cat = getattr(nxt, "category", None) or (nxt.get("type") if isinstance(nxt, dict) else getattr(nxt, "type", type(nxt).__name__))
            if nxt_cat in ("Header", "Footer", "PageBreak"):
                continue
            nxt_text = (nxt.text if hasattr(nxt, "text") else (nxt.get("text") if isinstance(nxt, dict) else str(nxt))).strip()
            if nxt_text:
                next_content = nxt_text
                break
        if not next_content or len(next_content) < min_content_length:
            return True
        return False
    else:
        s = str(elements_or_text).strip()
        if not s:
            return True
        if len(s) >= threshold:
            return False
        if re.match(r'^(?:\d+\.|\d+\.\d+|\d+\.\d+\.\d+|[A-Z]\.)\s+[A-Za-z]', s):
            return False
        if any(term in s.lower() for term in ["dxa", "bmd", "frax", "risk", "score", "scan", "bone", "osteoporosis"]):
            return False
        if len(s.split()) <= 2 and len(s) < 15:
            return True
        return False


def filter_elements(
    elements: List[Any],
    short_title_threshold: int = SHORT_TITLE_THRESHOLD,
    min_content_length: int = MIN_CONTENT_LENGTH
) -> List[Any]:
    """Filter out headers, footers, pagebreaks, noise titles and low-information layout elements."""
    filtered: List[Any] = []
    for i, el in enumerate(elements):
        cat = getattr(el, "category", None) or (el.get("type") if isinstance(el, dict) else getattr(el, "type", type(el).__name__))
        if cat in ("Header", "Footer", "PageBreak"):
            continue
        el_text = (el.text if hasattr(el, "text") else (el.get("text") if isinstance(el, dict) else str(el))).strip()
        if not el_text:
            continue
        if cat == "Title" and is_noise_title(elements, i, min_content_length=min_content_length, threshold=short_title_threshold):
            continue
        if len(el_text) < 3 and not re.match(r'^\d+$', el_text):
            continue
        filtered.append(el)
    return filtered


def strip_punctuation(w: str) -> Tuple[str, str, str]:
    """Strip leading and trailing punctuation from a token, returning (core, leading, trailing)."""
    m = re.match(r'^(\W*)(.*?)(\W*)$', w)
    if m:
        leading, core, trailing = m.group(1), m.group(2), m.group(3)
        return core, leading, trailing
    return w, "", ""


def is_valid_word(word: str) -> bool:
    """Pure-Python word frequency and dictionary validation."""
    clean, _, _ = strip_punctuation(word)
    clean = clean.lower()
    if not clean:
        return False
    if clean in KNOWN_MEDICAL_VOCABULARY or clean in COMMON_ENGLISH_WORDS:
        return True
    if clean.isdigit():
        return True
    if len(clean) == 1 and clean in ("a", "i"):
        return True
    if len(clean) <= 2:
        return False
    if _HAS_WORDFREQ:
        try:
            freq = zipf_frequency(clean, 'en')
            if freq > 1.8:
                return True
            return False
        except Exception:
            pass
    # Fallback heuristic: reject long unhyphenated concatenated tokens
    if len(clean) > 13:
        return False
    # Check for known glued prefix patterns
    for pfx in ["the", "policy", "screening", "recommendation", "patient", "guideline"]:
        if clean.startswith(pfx) and len(clean) > len(pfx) + 2:
            rem = clean[len(pfx):]
            if rem in COMMON_ENGLISH_WORDS or rem in KNOWN_MEDICAL_VOCABULARY or rem.startswith(("and", "for", "with", "risk", "notes")):
                return False
    return True


def _dp_word_split(s: str) -> List[str]:
    """Dynamic programming word segmentation fallback."""
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    parent = [-1] * (n + 1)
    for i in range(1, n + 1):
        for j in range(max(0, i - 20), i):
            if dp[j] and is_valid_word(s[j:i]):
                dp[i] = True
                parent[i] = j
                break
    if not dp[n]:
        return [s]
    tokens = []
    curr = n
    while curr > 0:
        p = parent[curr]
        tokens.append(s[p:curr])
        curr = p
    tokens.reverse()
    return tokens


def fix_concatenated_word(word: str) -> str:
    """Fix glued words, preserving acronyms and capitalizations."""
    clean, lead, trail = strip_punctuation(word)
    if not clean:
        return word
    if clean.isupper() and len(clean) <= 6:
        return word
    camel = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', clean)
    if camel != clean:
        parts = camel.split()
        if all(is_valid_word(p) or p.isupper() for p in parts):
            return f"{lead}{camel}{trail}"
    if is_valid_word(clean):
        return word
    if _HAS_WORDNINJA:
        try:
            split_words = wordninja.split(clean)
            if len(split_words) > 1 and all(is_valid_word(w) for w in split_words):
                return f"{lead}{' '.join(split_words)}{trail}"
        except Exception:
            pass
    dp_tokens = _dp_word_split(clean)
    if len(dp_tokens) > 1:
        return f"{lead}{' '.join(dp_tokens)}{trail}"
    return word


def fix_concatenated_text(text: str) -> str:
    """Repair concatenated words across full text while preserving formatting."""
    tokens = text.split(" ")
    repaired: List[str] = []
    for token in tokens:
        if "\n" in token:
            sublines = token.split("\n")
            repaired_sub = [fix_concatenated_word(st) for st in sublines]
            repaired.append("\n".join(repaired_sub))
        else:
            repaired.append(fix_concatenated_word(token))
    return " ".join(repaired)


def count_concatenated_words(
    elements: Union[List[Any], str]
) -> Tuple[List[Any], int]:
    """
    Count concatenated and dictionary-invalid tokens across layout elements or raw text.

    Args:
        elements: List of Unstructured element objects, dicts, or a raw text string.

    Returns:
        Tuple[List[Any], int]: (affected_elements, total_bad_words_count)
    """
    if isinstance(elements, str):
        elements_list = [elements]
    else:
        elements_list = list(elements)

    affected: List[Any] = []
    total_bad = 0

    for el in elements_list:
        t = (el.text if hasattr(el, "text") else (el.get("text") if isinstance(el, dict) else str(el))).strip()
        bad_in_el = 0
        for tok in t.split():
            clean, _, _ = strip_punctuation(tok)
            if len(clean) > 8 and not is_valid_word(clean):
                bad_in_el += 1
        if bad_in_el > 0:
            affected.append(el)
            total_bad += bad_in_el

    return affected, total_bad


def clean_academic_boilerplate(text: str) -> str:
    """Strip academic disclosures, author affiliations, and copyright blocks."""
    cleaned = text
    for pat in ACADEMIC_NOISE_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'(\w+)-\n(\w+)', r'\1\2', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def clean_pages(
    pages: List[Page],
    academic: bool = True
) -> str:
    """Orchestrate cleaning across Page objects."""
    cleaned_page_texts: List[str] = []
    for page in pages:
        page_text = page.text
        if page.elements:
            filtered_els = filter_elements(page.elements)
            el_texts = []
            for el in filtered_els:
                t = (el.text if hasattr(el, "text") else (el.get("text") if isinstance(el, dict) else str(el))).strip()
                if t:
                    el_texts.append(t)
            if el_texts:
                page_text = "\n\n".join(el_texts)
        if academic:
            page_text = clean_academic_boilerplate(page_text)
        repaired_text = fix_concatenated_text(page_text)
        cleaned_page_texts.append(repaired_text.strip())
    full_text = "\n\n".join(t for t in cleaned_page_texts if t)
    return full_text


def save_cleaned_text(
    clean_text: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    document_name: Optional[str] = None,
    text: Optional[str] = None
) -> Path:
    """Persist cleaned text to file, supporting both path or dir/name signatures."""
    target_text = text if text is not None else (clean_text or "")
    if output_path is not None:
        target_path = Path(output_path)
    elif output_dir is not None and document_name is not None:
        doc_filename = f"{document_name}.txt" if not document_name.endswith(".txt") else document_name
        target_path = Path(output_dir) / doc_filename
    else:
        raise ValueError("Must provide either output_path or (output_dir, document_name).")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(target_text)
    return target_path


def format_summary_table(doc_summaries: List[Dict[str, Any]]) -> str:
    """Format tabular summary of cleaning stats."""
    lines = [
        "=" * 92,
        f"  RAG PIPELINE: SMART INGESTION & CLEANING ({len(doc_summaries)} PDF DOCUMENTS)",
        "=" * 92,
        f"{'DOCUMENT NAME':<50} | {'EST. RAW':<10} | {'CLEAN CHARS':<11} | {'DROP (%)':<8}",
        "-" * 92
    ]
    tot_raw = 0
    tot_clean = 0
    for s in doc_summaries:
        raw_c = s.get("raw_chars", 0)
        clean_c = s.get("clean_chars", 0)
        drop_p = s.get("reduction_pct", 0.0)
        tot_raw += raw_c
        tot_clean += clean_c
        doc_name = str(s.get("document") or s.get("file_name") or s.get("document_name") or "Unknown")[:48]
        lines.append(f"{doc_name:<50} | {raw_c:<10} | {clean_c:<11} | {drop_p:<6.1f} %")
    tot_drop = ((tot_raw - tot_clean) / tot_raw * 100) if tot_raw > 0 else 0.0
    lines.append("=" * 92)
    lines.append(f"{'TOTAL':<50} | {tot_raw:<10} | {tot_clean:<11} | {tot_drop:<6.1f} %")
    lines.append("=" * 92)
    return "\n".join(lines)


def clean_all_guidelines(
    academic: bool = True,
    input_dir: Union[str, Path] = DEFAULT_GUIDELINES_DIR,
    output_dir: Union[str, Path] = DEFAULT_CLEANED_DIR,
    strategy: str = DEFAULT_PARTITION_STRATEGY
) -> List[Dict[str, Any]]:
    """End-to-end cleaning orchestrator for all discovered guideline PDFs."""
    pdf_files = discover_and_sync_guidelines(input_dir=input_dir)
    if not pdf_files:
        logger.warning(f"No PDF guideline files found in '{input_dir}'.")
        return []
    summaries: List[Dict[str, Any]] = []
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    for pdf_path in pdf_files:
        doc_id = pdf_path.stem
        pages = partition_pdf_pages(pdf_path, strategy=strategy)
        raw_char_count = sum(len(p.text) for p in pages)
        cleaned_text = clean_pages(pages, academic=academic)
        clean_char_count = len(cleaned_text)
        reduction_pct = ((raw_char_count - clean_char_count) / raw_char_count * 100) if raw_char_count > 0 else 0.0
        out_file = out_dir_path / f"{doc_id}.txt"
        save_cleaned_text(cleaned_text, out_file)
        summaries.append({
            "document": doc_id,
            "document_name": doc_id,
            "file_name": pdf_path.name,
            "raw_chars": raw_char_count,
            "clean_chars": clean_char_count,
            "reduction_pct": reduction_pct,
            "output_path": str(out_file)
        })
    print(format_summary_table(summaries))
    print(f"\n[OK] All cleaned files successfully written to: '{output_dir}/'")
    return summaries
