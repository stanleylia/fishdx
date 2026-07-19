"""Pareidolia correction tests — paper §III.B, Eq. 1–4.

Covers docs/m0/test-matrix.md case IDs:
    Eq. 1 Hard Path (5):   HP1–HP5
    Eq. 2 Soft Path (6):   SP1–SP6
    Eq. 3 Merge    (4):    MG1–MG4
    Eq. 4 Remap    (4):    RM1–RM4

All 19 cases are RED: target functions raise ``NotImplementedError``
(M2 implementation). pytest.param ids match matrix case IDs exactly.

References
----------
Paper §III.B "Visual Perception and Pareidolia Correction".
"""

from __future__ import annotations

import pytest

from fishdx.perception.pareidolia import (
    compute_p_hard,
    compute_p_soft,
    merge_pareidolia,
    remap_label,
)

_KS = ["net", "fence", "mesh", "cage", "grid", "wire", "pen", "enclosure", "lattice", "netting"]
_KB = ["face", "person", "animal", "human", "head", "eye", "body"]
_TAU = 0.75


# ────────────────────────── Eq. 1 Hard Path (HP1–HP5) ──────────────────────────
@pytest.mark.parametrize(
    ("caption", "label", "expected"),
    [
        pytest.param("a cage near water", "face", 1, id="HP1-cage-K_s-hit-no-K_b-in-G"),
        pytest.param("fish swimming", "head", 0, id="HP2-miss-K_s"),
        # HP3 — caption MUST contain a K_b token to fail the "G ∩ K_b = ∅" check
        # under paper-faithful Eq. 1 (caption-only indicator). The original
        # caption "net and mesh" had K_s but no K_b, which under paper-faithful
        # Eq. 1 returns 1 — so the test was patched to use a caption that
        # contains both K_s and K_b, preserving the "miss-K_b" semantic
        # (the indicator misses the no-K_b condition because K_b is present).
        pytest.param("a person near mesh", "fin", 0, id="HP3-K_b-in-G-blocks-trigger"),
        pytest.param("wire enclosure lattice", "body", 1, id="HP4-multi-K_s-hit"),
        pytest.param("", "face", 0, id="HP5-empty-caption"),
    ],
)
def test_eq1_hard_path(caption: str, label: str, expected: int) -> None:
    """Paper Eq. 1 | Case HP{1-5}.

    Expected: ``P_hard = 𝟙[G ∩ K_s ≠ ∅] · 𝟙[G ∩ K_b = ∅]`` returns matrix
    value. Operator precision: intersection uses token-level membership;
    BOTH indicators test the caption G (label is consumed only by Eq. 4
    remap downstream).
    """
    assert compute_p_hard(caption, label, _KS, _KB) == expected


# ────────────────────────── Eq. 2 Soft Path (SP1–SP6) ──────────────────────────
# Bounds reflect paper-literal Eq. 2: σ(cos(G,C_s) − τ) · σ(−(cos(l,C_b) − τ))
# with standard sigmoid σ(x) = 1/(1+e^{-x}). Note the negation in the second
# sigmoid: σ(τ − cos_l), not σ(cos_l − τ). Each fixture's expected range is
# centered on the analytically-computed value with ±0.01 tolerance window
# (allows for cross-platform sigmoid implementation drift).
#
# SP5 has its own dedicated test function (test_eq2_soft_path_sp5) with
# a full analytical-derivation docstring + DO-NOT-WIDEN-TOLERANCE invariant.
@pytest.mark.parametrize(
    ("cos_g", "cos_l", "expected_low", "expected_high"),
    [
        # σ(0.15)·σ(−0.15) ≈ 0.5374·0.4626 = 0.2486
        pytest.param(0.90, 0.90, 0.24, 0.26, id="SP1-both-positive-gaps"),
        # σ(−0.01)·σ(−0.15) ≈ 0.4975·0.4626 = 0.2301
        pytest.param(0.74, 0.90, 0.22, 0.24, id="SP2-G-negative-gap"),
        # σ(0.01)·σ(−0.15) ≈ 0.5025·0.4626 = 0.2325
        pytest.param(0.76, 0.90, 0.22, 0.24, id="SP3-G-small-positive-gap"),
        # σ(0.15)·σ(0.01) ≈ 0.5374·0.5025 = 0.2700
        pytest.param(0.90, 0.74, 0.26, 0.28, id="SP4-l-less-biological"),
        # σ(0.01)·σ(−0.01) ≈ 0.5025·0.4975 = 0.2500
        pytest.param(0.76, 0.76, 0.24, 0.26, id="SP6-both-small-positive"),
    ],
)
def test_eq2_soft_path(
    cos_g: float, cos_l: float, expected_low: float, expected_high: float
) -> None:
    """Paper Eq. 2 | Case SP{1-4, 6} — paper-literal vanilla sigmoid with
    negation in second factor.

    Expected: ``P_soft = σ(cos(G,C_s) − τ) · σ(−(cos(l,C_b) − τ))`` with
    standard ``σ(x) = 1/(1+e^{-x})`` falls inside [expected_low,
    expected_high]. Under vanilla σ with the paper-literal negation, the
    product lies in ``[0.22, 0.25]`` across the typical CLIP cosine range
    cos ∈ [0.20, 0.35] (closed-form upper bound 0.25 at cos == τ). The
    soft path alone never crosses Eq. 4's strict ``> 0.5`` remap trigger;
    it acts as an informational signal that contributes through Eq. 3's
    max-merge when the hard path fires.

    See test_paper_eq2_anchor.py for the negation-invariant lock.
    """
    p_soft = compute_p_soft(cos_g, cos_l, _TAU)
    assert expected_low <= p_soft <= expected_high


def test_eq2_soft_path_sp5() -> None:
    """Paper Eq. 2 | Case SP5 — both inputs deeply below τ.

    Analytical derivation (cos_g = cos_l = 0.50, τ = 0.75):

        σ(cos_g − τ) = σ(0.50 − 0.75) = σ(−0.25)
                     = 1 / (1 + e^{0.25})
                     ≈ 1 / 2.2840
                     ≈ 0.4378
        σ(−(cos_l − τ)) = σ(τ − cos_l) = σ(0.75 − 0.50) = σ(0.25)
                        = 1 / (1 + e^{−0.25})
                        ≈ 1 / 1.7788
                        ≈ 0.5622
        P_soft = 0.4378 × 0.5622 ≈ 0.24613

    Tolerance basis: ±0.01 around the analytical value 0.2461. The window
    [0.235, 0.255] accommodates cross-platform sigmoid implementation
    drift (numpy / scipy / pure-Python math.exp differences in the last
    1-2 ulps) without admitting any qualitatively-wrong implementation.

    DO NOT widen this tolerance window. If a sigmoid implementation produces
    output outside [0.235, 0.255] for these inputs, the implementation is
    paper non-faithful and must be fixed — the expected range MUST NOT be
    relaxed to accommodate buggy behaviour.
    """
    p_soft = compute_p_soft(0.50, 0.50, _TAU)
    assert 0.235 <= p_soft <= 0.255


# ────────────────────────── Eq. 3 Two-Tier Merge (MG1–MG4) ──────────────────────────
@pytest.mark.parametrize(
    ("p_hard", "p_soft", "expected"),
    [
        pytest.param(1.0, 0.0, 1.0, id="MG1-hard-only"),
        pytest.param(0.0, 0.9, 0.9, id="MG2-soft-only"),
        pytest.param(1.0, 0.9, 1.0, id="MG3-both"),
        pytest.param(0.0, 0.0, 0.0, id="MG4-neither"),
    ],
)
def test_eq3_merge(p_hard: float, p_soft: float, expected: float) -> None:
    """Paper Eq. 3 | Case MG{1-4}.

    Expected: ``P = max(P_hard, P_soft)`` — four-quadrant truth table.

    RED: ``merge_pareidolia`` raises ``NotImplementedError`` until M2.
    """
    assert merge_pareidolia(p_hard, p_soft) == pytest.approx(expected)


# ────────────────────────── Eq. 4 Label Remap (RM1–RM4) ──────────────────────────
@pytest.mark.parametrize(
    ("p", "expected"),
    [
        pytest.param(0.49, "face", id="RM1-below-threshold"),
        pytest.param(0.50, "face", id="RM2-equal-boundary-strict-gt"),
        pytest.param(0.51, "net_damage", id="RM3-above-threshold"),
        pytest.param(1.00, "net_damage", id="RM4-max"),
    ],
)
def test_eq4_remap(p: float, expected: str) -> None:
    """Paper Eq. 4 | Case RM{1-4}.

    Operator precision: remap triggers on **strict** ``P > 0.5``; equality
    (P=0.5) leaves label unchanged.

    RED: ``remap_label`` raises ``NotImplementedError`` until M2.
    """
    assert remap_label("face", p, "net_damage") == expected
