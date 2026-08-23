"""Text utility functions for caption processing and keyword extraction."""

from __future__ import annotations

import re
import json
from typing import Any


def extract_keywords(caption: str) -> list[str]:
    """Extract meaningful keywords from a caption string.

    Removes common stop words and returns unique lowercased tokens.

    Args:
        caption: Input caption text.

    Returns:
        List of extracted keywords.
    """
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same", "so",
        "than", "too", "very", "and", "but", "or", "yet", "it", "its",
        "this", "that", "these", "those", "i", "you", "he", "she", "we",
        "they", "me", "him", "her", "us", "them", "my", "your", "his",
        "our", "their", "what", "which", "who", "whom",
    }

    # Tokenize: split on non-alphanumeric (keep CJK characters)
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", caption.lower())
    keywords = [t for t in tokens if t not in stop_words and len(t) > 1]

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            result.append(kw)
    return result


def has_keyword_overlap(text: str, keywords: list[str]) -> bool:
    """Check if text contains any of the given keywords.

    Args:
        text: Input text to search.
        keywords: List of keywords to match.

    Returns:
        True if any keyword is found in the text.
    """
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def count_keyword_overlap(text: str, keywords: list[str]) -> int:
    """Count how many keywords appear in the text.

    Args:
        text: Input text to search.
        keywords: List of keywords to match.

    Returns:
        Number of keywords found.
    """
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def safe_json_parse(text: str) -> dict[str, Any] | None:
    """Safely parse JSON from LLM output, handling code fences.

    Strips markdown code fences (```json ... ```) and attempts JSON parse.
    Also tries to extract JSON from within surrounding text.

    Args:
        text: Raw text that may contain JSON.

    Returns:
        Parsed dict or None if parsing fails.
    """
    if not text or not text.strip():
        return None

    cleaned = text.strip()

    # Strip code fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Try direct parse first
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting JSON object from within text (find first { to last })
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = cleaned[first_brace:last_brace + 1]
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to a maximum length.

    Args:
        text: Input text.
        max_length: Maximum allowed length.
        suffix: Suffix to append when truncated.

    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix
