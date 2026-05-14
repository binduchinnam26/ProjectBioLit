"""Shared utility functions for BioLitAI-X."""

import re
from typing import Any, Dict, List


def safe_filename(text: str, max_len: int = 64) -> str:
    """Convert arbitrary text into a filesystem-safe filename."""
    text = re.sub(r"[^\w\s-]", "", text).strip()
    text = re.sub(r"[\s]+", "_", text)
    return text[:max_len]


def chunk_list(lst: List[Any], size: int) -> List[List[Any]]:
    """Split a list into chunks of *size*."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def flatten(nested: List[List[Any]]) -> List[Any]:
    return [item for sublist in nested for item in sublist]


def pubmed_url(pmid: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def truncate(text: str, max_chars: int = 300) -> str:
    if not text:
        return ""
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"
