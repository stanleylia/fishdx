"""Anchor for Eq. 9 — Disease score with strict three-tier priority ordering.

Locks the strict priority ordering: confirmed > suspected > mentioned.
Implementation uses 'highest-tier-wins' semantic via seen_lower
accumulator (src/fishdx/scoring/score.py).

Manuscript Eq. 9 form (post-typo-fix):
    S_d = 3·|K_d^conf ∩ 𝒪|
        + 2·|(K_d^susp ∖ K_d^conf) ∩ 𝒪|
        + 1·|(K_d^ment ∖ K_d^conf ∖ K_d^susp) ∩ 𝒪|

Manuscript uses uniform weights (3/2/1) across all keywords within
each tier. The release code's per-keyword w_d^ment(k) is an
implementation generalization that defaults to 1, reproducing the
paper's behavior. Anchor here locks the default-weight equivalence.

If this test breaks, do NOT update the expected values.
The strict priority ordering is paper-mandated.

Cross-reference: Methods §Stage 3, Eq. 9 of the manuscript (post-typo-fix
form with ∖ K_d^susp added to the mentioned term).
"""

from __future__ import annotations

from fishdx.config import DiseaseWeights, HealthyWeights, ScoringConfig
from fishdx.scoring.score import compute_s_d

_SCORING = ScoringConfig(
    disease_weights=DiseaseWeights(confirmed=3, suspected=2, mentioned=1),
    healthy_weights=HealthyWeights(explicit=2, negation=1),
)


def test_anchor_eq9_condition_1_conf_intersect_ment() -> None:
    """條件 1: kw ∈ confirmed ∩ mentioned → only confirmed=3 counts.

    If lattice were broken (mentioned not deduplicated against confirmed),
    S_d would be 3+1=4 instead of 3.
    """
    s = compute_s_d(
        "ulcer", _SCORING,
        confirmed_keywords=("ulcer",),
        suspected_keywords=(),
        mentioned_keywords=("ulcer",),
    )
    assert s == 3


def test_anchor_eq9_condition_2_conf_intersect_susp() -> None:
    """條件 2: kw ∈ confirmed ∩ suspected → only confirmed=3 counts.

    If lattice were broken, S_d would be 3+2=5 instead of 3.
    """
    s = compute_s_d(
        "ulcer", _SCORING,
        confirmed_keywords=("ulcer",),
        suspected_keywords=("ulcer",),
        mentioned_keywords=(),
    )
    assert s == 3


def test_anchor_eq9_condition_3_mentioned_only() -> None:
    """條件 3: kw ∈ mentioned only → mentioned=1 counts."""
    s = compute_s_d(
        "redness", _SCORING,
        confirmed_keywords=(),
        suspected_keywords=(),
        mentioned_keywords=("redness",),
    )
    assert s == 1


def test_anchor_eq9_condition_4_suspected_only() -> None:
    """條件 4: kw ∈ suspected only → suspected=2 counts."""
    s = compute_s_d(
        "inflammation", _SCORING,
        confirmed_keywords=(),
        suspected_keywords=("inflammation",),
        mentioned_keywords=(),
    )
    assert s == 2


def test_anchor_eq9_condition_5_susp_intersect_ment_no_conf() -> None:
    """條件 5 (NEW — Phase 3 upgrade): kw ∈ suspected ∩ mentioned but ∉ confirmed
    → only suspected=2 counts (NOT 2+1=3).

    This is the case the typo-fix protects against: if Eq. 9 lacked the
    ∖ K_d^susp in the mentioned term, S_d would be 3 here. The complete
    two-tier set-difference (highest-tier-wins) ensures suspected
    suppresses mentioned in this overlap.
    """
    s = compute_s_d(
        "lesion", _SCORING,
        confirmed_keywords=(),
        suspected_keywords=("lesion",),
        mentioned_keywords=("lesion",),
    )
    assert s == 2
