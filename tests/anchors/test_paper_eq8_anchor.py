"""Anchor for Eq. 8 — Healthy score uses token-set indicator, not substring count.

Locks the paper-mandated invariant: the first term of S_h is a binary
indicator over the NLTK-stopword-filtered token set 𝒪, NOT a multi-
occurrence substring count. Token-level cross-tier deduplication is
implicit because set membership is binary.

Manuscript Eq. 8:
    S_h = Σ_i w^(h)_i · 𝟙[h_i ∈ 𝒪] + Σ_k w^(n)_k · Neg_k(𝒪)

where 𝒪 = tokens(G') ∪ tokens(r_1.text) after NLTK stop-word removal.

If this test breaks, do NOT update the expected values.
The indicator-set semantic is paper-mandated.

Cross-reference: Methods §Stage 3, Eq. 8 of the manuscript.
"""

from __future__ import annotations

from fishdx.config import HealthyWeights, NegationConfig
from fishdx.scoring.score import compute_s_h

_HW = HealthyWeights(explicit=2, negation=1)
_NEG = NegationConfig(
    window_chars=50,
    cn_patterns=("無",),
    en_patterns=("no sign of",),
)


def test_anchor_eq8_indicator_not_count() -> None:
    """S_h must be IDENTICAL for 1, 2, 3 occurrences of the same keyword.

    If S_h grew linearly with occurrence count, the contract would be
    substring-count rather than token-set indicator — paper non-faithful.
    """
    s1 = compute_s_h("healthy", _HW, _NEG, healthy_keywords=("healthy",))
    s2 = compute_s_h("healthy healthy", _HW, _NEG, healthy_keywords=("healthy",))
    s3 = compute_s_h(
        "healthy healthy healthy", _HW, _NEG, healthy_keywords=("healthy",)
    )
    assert s1 == s2 == s3
    # And the value is exactly w_explicit (no spurious negation hits).
    assert s1 == _HW.explicit


def test_anchor_eq8_distinct_keywords_add() -> None:
    """Two DIFFERENT healthy keywords add exactly w_explicit each.

    Distinct tokens are independent; only same-token duplication is
    deduplicated by the indicator semantic.
    """
    s_one = compute_s_h(
        "healthy fish", _HW, _NEG, healthy_keywords=("healthy",)
    )
    s_two = compute_s_h(
        "healthy vibrant fish", _HW, _NEG,
        healthy_keywords=("healthy", "vibrant"),
    )
    # Adding a NEW keyword that's present in evidence adds exactly w_explicit.
    assert s_two == s_one + _HW.explicit
