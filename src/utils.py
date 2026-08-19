"""
Shared Text Processing & Tokenization Utilities (src/utils.py).
"""

import re
import string
import hashlib
import unicodedata
from typing import Optional, Union

# Lazy initialization of tiktoken cl100k_base standard encoding
_TIKTOKEN_ENCODING = None
try:
    import tiktoken
    _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENCODING = None


def compute_content_hash(*parts: Union[str, bytes], length: int = 12) -> str:
    """
    Compute a deterministic SHA-256 content hash across one or more input parts.
    Handles str and bytes inputs cleanly, returning a truncated hex digest.
    """
    hasher = hashlib.sha256()
    for part in parts:
        if isinstance(part, bytes):
            hasher.update(part)
        elif isinstance(part, str):
            hasher.update(part.encode("utf-8"))
        elif part is not None:
            hasher.update(str(part).encode("utf-8"))
    return hasher.hexdigest()[:length]


def normalize_unicode(text: str) -> str:
    """
    Standardize Unicode representation using NFKC normalization,
    stripping soft hyphens and zero-width artifacts.
    """
    if not text:
        return ""
    norm = unicodedata.normalize("NFKC", text)
    return norm.replace("\xad", "").replace("\u200b", "").replace("\ufeff", "")


def dehyphenate_text(text: str) -> str:
    """
    Remove soft hyphens, unify Unicode hyphens, and merge hyphenated words across line breaks.
    """
    if not text:
        return ""
    cleaned = normalize_unicode(text)
    cleaned = cleaned.replace("\u2010", "-").replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
    cleaned = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', cleaned)
    cleaned = re.sub(r'(\w+)-\s+([a-z]+)', r'\1\2', cleaned)
    return cleaned


def normalize_whitespace(text: str) -> str:
    """
    Normalize non-breaking spaces, excess tabs, and collapse 3+ consecutive newlines into 2.
    """
    if not text:
        return ""
    cleaned = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r'[ \t]+$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def clean_text(text: str) -> str:
    """
    End-to-end text normalization: Unicode NFKC normalization, dehyphenation,
    and whitespace normalization.
    """
    if not text:
        return ""
    cleaned = dehyphenate_text(text)
    cleaned = normalize_whitespace(cleaned)
    return cleaned


def strip_punctuation(text: str) -> str:
    """Strip leading and trailing ASCII punctuation using string.punctuation."""
    if not text:
        return ""
    return text.strip(string.punctuation)


def count_tokens(text: str) -> int:
    """
    Count tokens using tiktoken cl100k_base standard encoding with fallback to word count.
    """
    if not text:
        return 0
    if _TIKTOKEN_ENCODING is not None:
        try:
            return len(_TIKTOKEN_ENCODING.encode(text))
        except Exception:
            pass
    # Standard 4-character per token heuristic fallback
    return max(1, len(text.split()))
