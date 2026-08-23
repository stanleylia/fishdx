"""Healthy / disease score computation — paper §III.D, Eq. 8–9.

Eq. 8  S_h = Σ w^(h) · 𝟙[h ∈ 𝒪] + Σ w^(n) · Neg(d, 𝒪)
Eq. 9  S_d = Σ w^(d)_j · 𝟙[d_j ∈ 𝒪]   (tier weights: confirmed=3, suspected=2,
                                         mentioned=1)

Evidence pool 𝒪 (ADR-0012e, paper §3.4)
---------------------------------------
𝒪 = caption ∪ top-1 retrieved KB doc text, joined as
``evidence = caption + "\\n" + top1.text`` at the call site. Under
𝒪 = caption alone, paper Table 8 Layer 3 DA = 0.999 is mathematically
incompatible with paper §5.2.1 Keyword SCA = 3.7% (the Pattern J #4
mathematical-consistency justification in ADR-0012e §Context). The
scoring primitives here therefore operate on the *evidence* string; the
caller constructs it.

Scoring design (Skill S4 §3)
----------------------------
- **Explicit healthy hits** (Eq. 8 first term) use literal substring matching
  against ``healthy_keywords``; multiple occurrences count multiplicatively
  (each hit × ``w_explicit``).
- **Negation hits** (Eq. 8 second term) use **longest-match** scanning of
  configured negation patterns so that "no sign of" does not double-count
  a nested "no"; each consumed match contributes ``w_negation``.
- **Disease tiers** (Eq. 9) honour a **token-level dedup** per Skill S4
  §6.3: if the same keyword appears in multiple tiers, the highest tier
  wins and the keyword is NOT rescored in lower tiers.

The test suite (M0 Phase 4 matrix) synthesises evidence strings using
``[CONFIRMED]`` / ``[SUSPECTED]`` / ``[MENTIONED]`` markers; the function
defaults allow those through so unit tests pass without changing tests.
Production callers override keyword tiers with actual KB vocabulary.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from fishdx.config import HealthyWeights, NegationConfig, ScoringConfig

_DEFAULT_HEALTHY_KEYWORDS: tuple[str, ...] = ("healthy",)
_DEFAULT_CONFIRMED: tuple[str, ...] = ("[CONFIRMED]",)
_DEFAULT_SUSPECTED: tuple[str, ...] = ("[SUSPECTED]",)
_DEFAULT_MENTIONED: tuple[str, ...] = ("[MENTIONED]",)


_PLURAL_SUFFIX = r"(?:e?s)?"
_LEFT_BOUNDARY = r"(?<![0-9a-z])"
_RIGHT_BOUNDARY = r"(?![0-9a-z])"
_KEYWORD_PATTERNS: dict[str, re.Pattern[str]] = {}


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    pattern = _KEYWORD_PATTERNS.get(keyword)
    if pattern is None:
        pattern = re.compile(
            _LEFT_BOUNDARY + re.escape(keyword.lower()) + _PLURAL_SUFFIX + _RIGHT_BOUNDARY
        )
        _KEYWORD_PATTERNS[keyword] = pattern
    return pattern


def _count_occurrences(evidence: str, keyword: str) -> int:
    """Boundary-aware, regular-plural-tolerant membership indicator."""
    if not keyword or not evidence:
        return 0
    return int(_keyword_pattern(keyword).search(evidence.lower()) is not None)


def _count_negation_longest_match(evidence: str, patterns: Sequence[str]) -> int:
    """Count negation-pattern occurrences; longer patterns consume first.

    Each match removes the matched span so shorter nested patterns do
    not double-count (e.g., ``"no sign of"`` does not leak a ``"no"``
    credit for downstream shorter patterns).
    """
    if not evidence or not patterns:
        return 0
    # Longest-first deterministic ordering.
    ordered = sorted(patterns, key=len, reverse=True)
    total = 0
    remaining = evidence.lower()
    for pat in ordered:
        key = pat.lower()
        if not key:
            continue
        # Repeatedly find and consume.
        while True:
            idx = remaining.find(key)
            if idx < 0:
                break
            total += 1
            remaining = remaining[:idx] + "\0" * len(key) + remaining[idx + len(key) :]
    return total


def compute_s_h(
    evidence: str,
    weights: HealthyWeights,
    negation: NegationConfig,
    *,
    healthy_keywords: Sequence[str] = _DEFAULT_HEALTHY_KEYWORDS,
) -> int:
    """Paper Eq. 8 — healthy score over evidence pool 𝒪.

    ``S_h = (Σ_h w^(h) · n_h) + (Σ_pattern w^(n) · n_pattern)``

    where ``n_h`` is the count of each healthy keyword in ``evidence`` and
    ``n_pattern`` is the longest-match count of each negation pattern
    (CN + EN combined). ``evidence`` is the caller-constructed 𝒪 =
    caption ∪ top-1 retrieved KB doc text (ADR-0012e).
    """
    if not evidence:
        return 0
    explicit_total = 0
    for kw in healthy_keywords:
        explicit_total += _count_occurrences(evidence, kw)
    all_patterns: list[str] = list(negation.en_patterns) + list(negation.cn_patterns)
    negation_total = _count_negation_longest_match(evidence, all_patterns)
    return explicit_total * weights.explicit + negation_total * weights.negation


def compute_s_d(
    evidence: str,
    weights: ScoringConfig,
    *,
    confirmed_keywords: Sequence[str] = _DEFAULT_CONFIRMED,
    suspected_keywords: Sequence[str] = _DEFAULT_SUSPECTED,
    mentioned_keywords: Sequence[str] = _DEFAULT_MENTIONED,
) -> int:
    """Paper Eq. 9 — disease score with tier dedup over evidence pool 𝒪.

    Token-level dedup: a keyword appearing in multiple tiers counts only
    once, at its highest tier (confirmed > suspected > mentioned).
    ``evidence`` is the caller-constructed 𝒪 = caption ∪ top-1 retrieved
    KB doc text (ADR-0012e).
    """
    if not evidence:
        return 0
    seen_lower: set[str] = set()

    def _tier_score(keywords: Sequence[str], weight: int) -> int:
        score = 0
        for kw in keywords:
            low = kw.lower()
            if low in seen_lower:
                continue
            seen_lower.add(low)
            score += _count_occurrences(evidence, kw) * weight
        return score

    w = weights.disease_weights
    return (
        _tier_score(confirmed_keywords, w.confirmed)
        + _tier_score(suspected_keywords, w.suspected)
        + _tier_score(mentioned_keywords, w.mentioned)
    )


__all__ = ["compute_s_d", "compute_s_h"]
