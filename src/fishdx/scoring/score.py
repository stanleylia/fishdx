"""Healthy / disease score computation — paper §III.D, Eq. 8–9.

    Eq. 8  S_h = Σ w^(h) · 𝟙[h ∈ 𝒪] + Σ w^(n) · Neg_k(𝒪)
    Eq. 9  S_d = Σ w^(d)_j · 𝟙[d_j ∈ 𝒪]   (tier weights: confirmed=3,
                                            suspected=2, mentioned=1)

Evidence pool 𝒪 (paper Methods §Stage 3)
----------------------------------------
``𝒪 = tokens(G') ∪ tokens(r₁.text)`` is the union of the lower-cased
word-token sets of (i) the Stage-1 corrected caption G' and (ii) the
top-1 retrieval's knowledge-base document text, **after stop-word removal
using the NLTK English stop-word list**. Membership tests ``h_i ∈ 𝒪`` in
Eqs. 8–9 are evaluated against this token set, matching the SCA
tokenisation contract used in Eq. 12.

Scoring design
--------------
- **Explicit healthy hits** (Eq. 8 first term) use binary indicator-set
  membership (``𝟙[h ∈ 𝒪]``); a token's presence contributes ``w_explicit``
  exactly once regardless of multi-occurrence.
- **Negation hits** (Eq. 8 second term) match negation patterns directly
  against the joined evidence STRING (not the token set) using longest-
  match scanning, because patterns like "no sign of" span multiple tokens
  and would not survive tokenisation.
- **Disease tiers** (Eq. 9) honour token-level dedup: a keyword in
  multiple tiers is scored only at its highest tier.
"""

from __future__ import annotations

from collections.abc import Sequence

from fishdx.config import HealthyWeights, NegationConfig, ScoringConfig

_DEFAULT_HEALTHY_KEYWORDS: tuple[str, ...] = ("healthy",)
_DEFAULT_CONFIRMED: tuple[str, ...] = ("[CONFIRMED]",)
_DEFAULT_SUSPECTED: tuple[str, ...] = ("[SUSPECTED]",)
_DEFAULT_MENTIONED: tuple[str, ...] = ("[MENTIONED]",)

# NLTK English stop-word list (snapshot, frozen for deterministic
# reproduction). If nltk is installed at runtime, we prefer the live list
# (sourced from the same corpus); otherwise we fall back to this snapshot.
_NLTK_STOP_WORDS_FALLBACK: frozenset[str] = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s",
    "t", "can", "will", "just", "don", "should", "now",
})


def _stop_words() -> frozenset[str]:
    """Return the active stop-word set (NLTK if importable, else snapshot)."""
    try:
        from nltk.corpus import stopwords  # type: ignore
        return frozenset(w.lower() for w in stopwords.words("english"))
    except (ImportError, LookupError):
        return _NLTK_STOP_WORDS_FALLBACK


def tokenize_evidence(evidence: str) -> frozenset[str]:
    """Return ``tokens(evidence) − stopwords`` as a frozen set.

    Implements the paper's Methods §Stage 3 tokenisation contract:
    lower-case, whitespace-split, drop NLTK English stop-words.
    """
    if not evidence:
        return frozenset()
    raw = (tok.strip(".,;:()[]{}\"'") for tok in evidence.lower().split())
    stops = _stop_words()
    return frozenset(t for t in raw if t and t not in stops)


def _count_occurrences(evidence: str, keyword: str) -> int:
    """Substring count for negation patterns that span multiple tokens.

    Negation patterns like "no sign of" cannot be matched after token-set
    construction; this helper retains the substring-match fallback used
    by Eq. 8's negation term only.
    """
    if not keyword:
        return 0
    return evidence.lower().count(keyword.lower())


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

    ``S_h = Σ_i w^(h) · 𝟙[h_i ∈ 𝒪]  +  Σ_k w^(n) · Neg_k(𝒪)``

    The first term is a binary indicator over the token-set 𝒪 (NLTK
    stop-word-filtered, lower-cased): each unique healthy keyword present
    contributes ``w^(h) = w_explicit`` exactly once. The second term is
    the longest-match count of multi-token negation patterns in the joined
    evidence string (negation patterns span tokens and cannot be expressed
    as set-membership tests).
    """
    if not evidence:
        return 0
    token_set = tokenize_evidence(evidence)
    explicit_present = sum(1 for kw in healthy_keywords if kw.lower() in token_set)
    all_patterns: list[str] = list(negation.en_patterns) + list(negation.cn_patterns)
    negation_total = _count_negation_longest_match(evidence, all_patterns)
    return explicit_present * weights.explicit + negation_total * weights.negation


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
    token_set = tokenize_evidence(evidence)
    seen_lower: set[str] = set()

    def _tier_score(keywords: Sequence[str], weight: int) -> int:
        score = 0
        for kw in keywords:
            low = kw.lower()
            if low in seen_lower:
                continue
            seen_lower.add(low)
            # Eq. 9 indicator: keyword present (1) or absent (0) in 𝒪.
            if low in token_set:
                score += weight
        return score

    w = weights.disease_weights
    return (
        _tier_score(confirmed_keywords, w.confirmed)
        + _tier_score(suspected_keywords, w.suspected)
        + _tier_score(mentioned_keywords, w.mentioned)
    )


__all__ = ["compute_s_d", "compute_s_h"]
