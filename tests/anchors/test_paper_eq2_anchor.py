"""Anchor for Eq. 2 — Soft pareidolia path's second-sigmoid negation.

Locks the paper-mandated invariant: the second factor in P_soft must be
σ(τ - cos_l) — equivalently σ(-(cos_l - τ)) — i.e. it MUST contain a
negation of the (cos_l - τ) gap. Without the negation, the soft path
fires on biological labels (the wrong polarity), inverting the entire
pareidolia-correction logic.

Manuscript Eq. 2:
    P_soft(l_i) = σ(cos(CLIP_T(G), C_s) - τ)
                · σ(-(cos(CLIP_T(l_i), C_b) - τ))

If this test breaks, do NOT update the expected values.
The negation in the second sigmoid is paper-mandated.

Cross-reference: Methods §Stage 1, Eq. 2 of the manuscript.
"""

from __future__ import annotations

from fishdx.perception.pareidolia import compute_p_soft

_TAU: float = 0.75


def test_anchor_eq2_at_tau_yields_quarter() -> None:
    """When both cos_G == τ AND cos_l == τ, both sigmoids are σ(0) = 0.5,
    so the product is exactly 0.25 — closed-form upper bound."""
    result = compute_p_soft(_TAU, _TAU, _TAU)
    assert abs(result - 0.25) < 1e-6


def test_anchor_eq2_label_more_biological_decreases_soft() -> None:
    """When cos_l > τ (label is MORE biological), the negated sigmoid
    σ(τ - cos_l) drops below 0.5, so the product drops below 0.25.

    If the second sigmoid lacked the negation, it would INCREASE here —
    the polarity reversal is exactly what this anchor catches.
    """
    base = compute_p_soft(_TAU, _TAU, _TAU)  # = 0.25
    more_bio = compute_p_soft(_TAU, _TAU + 0.10, _TAU)
    assert more_bio < base


def test_anchor_eq2_label_less_biological_increases_soft() -> None:
    """When cos_l < τ (label is LESS biological), the negated sigmoid
    σ(τ - cos_l) rises above 0.5, so the product rises above the σ(0)
    baseline.

    If the second sigmoid lacked the negation, it would DECREASE here.
    """
    base = compute_p_soft(_TAU, _TAU, _TAU)  # = 0.25
    less_bio = compute_p_soft(_TAU, _TAU - 0.10, _TAU)
    assert less_bio > base
