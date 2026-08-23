"""Negation Patterns — Reference data for scoring negation detection.

Used by Eq.9 (Healthy Score) to detect negated disease mentions.
𝒩 = {"非","不是","排除","並非","誤報","未見","not","unlikely","ruled out"}
"""

from __future__ import annotations

# Negation patterns by language
NEGATION_PATTERNS: dict[str, list[str]] = {
    "zh": [
        "非", "不是", "排除", "並非", "誤報", "未見", "沒有",
        "無", "不像", "不符合", "可能性低",
    ],
    "en": [
        "not", "unlikely", "ruled out", "no evidence", "absent",
        "negative", "no sign", "without", "excluded", "improbable",
    ],
}


def get_all_negation_patterns() -> list[str]:
    """Get all negation patterns across all languages."""
    patterns: list[str] = []
    for lang_patterns in NEGATION_PATTERNS.values():
        patterns.extend(lang_patterns)
    return patterns
