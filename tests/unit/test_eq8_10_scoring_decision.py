"""Scoring & three-way decision unit tests — paper §III.D, Eq. 8–10.

Covers docs/m0/test-matrix.md case IDs:
    Eq. 8 Healthy Score (8):   SH1–SH8
    Eq. 9 Disease Score (9):   SD1–SD9 (九宮格)
    Eq. 10 Decision    (16):   DEC1–DEC16 (full precedence, operator-probing)

All 33 cases are RED: target functions raise ``NotImplementedError`` until M2.

Decision precedence (ADR-0001, Skill S4 §2.3):
    1. Retrieval margin (Eq. 11, strict ``<``)
    2. Scoring margin   (Eq. 10 row 1, ``≤``)
    3. Healthy          (Eq. 10 row 2, ``≥`` and ``>``)
    4. Disease          (Eq. 10 row 3, strict ``>``)
    5. Fallback         (Eq. 10 row 4)

References
----------
Paper §III.D "Scoring-Based Diagnosis".
"""

from __future__ import annotations

import pytest

from fishdx.config import (
    DecisionConfig,
    DiseaseWeights,
    HealthyWeights,
    MarginConfig,
    NegationConfig,
    ScoringConfig,
)
from fishdx.schemas import DecisionEnum
from fishdx.scoring.decision import make_decision
from fishdx.scoring.score import compute_s_d, compute_s_h

_HEALTHY_WEIGHTS = HealthyWeights(explicit=2, negation=1)
_DISEASE_WEIGHTS = DiseaseWeights(confirmed=3, suspected=2, mentioned=1)
_SCORING = ScoringConfig(disease_weights=_DISEASE_WEIGHTS, healthy_weights=_HEALTHY_WEIGHTS)
_NEGATION = NegationConfig(
    window_chars=50,
    cn_patterns=["無", "沒有", "未見", "無明顯", "沒有任何", "缺乏", "未發現"],
    # Test-local expansion: Table IX canonical 5 + conversational negations
    # "not", "no" required by SH2/SH3/SH7 matrix cases. Longest-match
    # guarantees "no sign of" is not double-counted as "no".
    en_patterns=[
        "no sign of",
        "absence of",
        "free from",
        "without",
        "negative for",
        "not",
        "no",
    ],
)
_DECISION = DecisionConfig(healthy_threshold_Th=2, inconclusive_margin_m=1)
_MARGIN = MarginConfig(retrieval_margin_theta=0.02)


# ────────────────────────── Eq. 8 Healthy Score (SH1–SH8) ──────────────────────────
# Note: SH5 is paper-faithful at 2 (token-set indicator: 'healthy' present
# once → w_explicit·1 = 2). The previous expected value of 4 encoded the
# pre-patch substring-count semantic and is no longer valid. See
# test_paper_eq8_anchor.py for the indicator-not-count invariant lock.
@pytest.mark.parametrize(
    ("caption", "expected"),
    [
        pytest.param("fish appears healthy", 2, id="SH1-single-explicit"),
        pytest.param("appears healthy; no disease", 3, id="SH2-explicit-plus-negation"),
        pytest.param("not infected", 1, id="SH3-negation-only"),
        pytest.param("", 0, id="SH4-empty"),
        pytest.param("healthy healthy", 2, id="SH5-token-set-indicator"),
        pytest.param(
            "appears healthy; no sign of A, no sign of B, no sign of C, no sign of D, "
            "no sign of E, no sign of F, no sign of G, no sign of H, no sign of I, no sign of J",
            12,
            id="SH6-linear-summation",
        ),
        pytest.param("not diseased, not sick", 2, id="SH7-multiple-negations"),
        pytest.param("disease confirmed", 0, id="SH8-no-healthy-evidence"),
    ],
)
def test_eq8_healthy_score(caption: str, expected: int) -> None:
    """Paper Eq. 8 | Case SH{1-8}.

    Expected: ``S_h = Σ w^(h)·𝟙[h ∈ 𝒪] + Σ w^(n)·Neg(d, 𝒪)`` with
    ``w^(h)=2, w^(n)=1``. The first term is a binary indicator over the
    NLTK-stopword-filtered token set 𝒪 — duplicate occurrences of the
    same keyword count once (token-set semantic). The negation term is a
    longest-match scan of multi-token patterns over the joined evidence
    string (multi-token patterns cannot be expressed as set membership).
    """
    assert compute_s_h(caption, _HEALTHY_WEIGHTS, _NEGATION) == expected


# ────────────────────────── Eq. 9 Disease Score (SD1–SD9) ──────────────────────────
# F2 hermetic style: captions use plain-word tokens that survive the
# NLTK-tokenizer's bracket-stripping step; tests pass keyword lists
# explicitly to compute_s_d (no reliance on _DEFAULT_* in score.py, which
# uses bracketed sentinel tokens incompatible with the tokenizer contract).
#
# Eq. 9 is implemented with strict three-tier priority ordering
# (highest-tier-wins). For each test, exactly the keyword `confirmed` /
# `suspected` / `mentioned` is registered in the corresponding tier, and
# the caption contains the matching plain-word tokens. SD2-8 expected
# values are unchanged numerically; only caption format and keyword-list
# passing are updated for hermetic-test alignment. SD9 is split into its
# own dedicated test (test_eq9_disease_score_sd9_token_set_dedup) with
# a full DO-NOT-UPDATE-EXPECTED docstring.
_SD_CONFIRMED_KEYWORDS = ("confirmed",)
_SD_SUSPECTED_KEYWORDS = ("suspected",)
_SD_MENTIONED_KEYWORDS = ("mentioned",)


@pytest.mark.parametrize(
    ("confirmed", "suspected", "mentioned", "expected"),
    [
        pytest.param(0, 0, 0, 0, id="SD1-all-zero"),
        pytest.param(0, 0, 1, 1, id="SD2-mention-only"),
        pytest.param(0, 1, 0, 2, id="SD3-suspect-only"),
        pytest.param(1, 0, 0, 3, id="SD4-confirm-only"),
        pytest.param(1, 1, 0, 5, id="SD5-confirm-plus-suspect"),
        pytest.param(1, 0, 1, 4, id="SD6-confirm-plus-mention"),
        pytest.param(0, 1, 1, 3, id="SD7-suspect-plus-mention"),
        pytest.param(1, 1, 1, 6, id="SD8-all-three"),
    ],
)
def test_eq9_disease_score(
    confirmed: int, suspected: int, mentioned: int, expected: int
) -> None:
    """Paper Eq. 9 | Case SD{1-8} (九宮格 — single-occurrence cases).

    Expected (highest-tier-wins form):
        S_d = 3·|K_d^conf ∩ 𝒪|
            + 2·|(K_d^susp ∖ K_d^conf) ∩ 𝒪|
            + 1·|(K_d^ment ∖ K_d^conf ∖ K_d^susp) ∩ 𝒪|

    Caption synthesis: each ``confirmed`` / ``suspected`` / ``mentioned``
    integer parameter contributes that many copies of the matching plain
    word to the caption, joined by spaces. Token-set indicator means
    multiple occurrences of the same word collapse to a single hit; SD9
    explicitly tests that property.
    """
    tokens = (
        ["confirmed"] * confirmed
        + ["suspected"] * suspected
        + ["mentioned"] * mentioned
    )
    caption = " ".join(tokens)
    result = compute_s_d(
        caption, _SCORING,
        confirmed_keywords=_SD_CONFIRMED_KEYWORDS,
        suspected_keywords=_SD_SUSPECTED_KEYWORDS,
        mentioned_keywords=_SD_MENTIONED_KEYWORDS,
    )
    assert result == expected


def test_eq9_disease_score_sd9_token_set_dedup() -> None:
    """Paper Eq. 9 | Case SD9 — token-set deduplication of repeated confirmed.

    Caption: ``"confirmed confirmed"`` (two occurrences of the same token).

    Under the paper-faithful token-set indicator semantic (Methods §Stage 3):
        tokenize_evidence("confirmed confirmed") = {"confirmed"}
        |K_d^conf ∩ 𝒪| = |{"confirmed"} ∩ {"confirmed"}| = 1
        S_d = 3·1 + 2·0 + 1·0 = 3

    Under the previous substring-count buggy semantic, the result was:
        evidence.lower().count("confirmed") = 2
        S_d = 3·2 + 2·0 + 1·0 = 6        (BUGGY)

    DO NOT update the expected value back to 6. The token-set indicator
    is paper-mandated (Eq. 9 uses 𝟙[d_j ∈ 𝒪], a binary set-membership
    function, not a multi-occurrence count). If a future change to
    ``compute_s_d`` returns 6 here, the change is paper non-faithful and
    ``compute_s_d`` must be brought back to indicator-set semantics —
    the test expected value MUST NOT be relaxed to 6.

    Cross-reference: test_paper_eq8_anchor.py + test_paper_eq9_anchor.py
    for the indicator-not-count + strict-priority-ordering invariant locks.
    """
    caption = "confirmed confirmed"
    result = compute_s_d(
        caption, _SCORING,
        confirmed_keywords=_SD_CONFIRMED_KEYWORDS,
        suspected_keywords=_SD_SUSPECTED_KEYWORDS,
        mentioned_keywords=_SD_MENTIONED_KEYWORDS,
    )
    assert result == 3


# ────────────────────────── Eq. 10 Decision (DEC1–DEC16) ──────────────────────────
@pytest.mark.parametrize(
    ("s_h", "s_d", "margin", "expected_decision", "expected_reason"),
    [
        pytest.param(0, 0, 0.10, DecisionEnum.INCONCLUSIVE, "scoring_margin", id="DEC1-zero-zero"),
        pytest.param(2, 0, 0.10, DecisionEnum.HEALTHY, None, id="DEC2-healthy-Th-boundary"),
        pytest.param(0, 2, 0.10, DecisionEnum.DISEASE, "disease_path", id="DEC3-disease-clear"),
        pytest.param(2, 2, 0.10, DecisionEnum.INCONCLUSIVE, "scoring_margin", id="DEC4-sh-eq-sd"),
        pytest.param(
            3, 2, 0.10, DecisionEnum.INCONCLUSIVE, "scoring_margin", id="DEC5-margin-le-boundary"
        ),
        pytest.param(3, 1, 0.10, DecisionEnum.HEALTHY, None, id="DEC6-healthy-after-margin-pass"),
        pytest.param(
            1, 3, 0.10, DecisionEnum.DISEASE, "disease_path", id="DEC7-disease-precedence-chain"
        ),
        pytest.param(
            3, 3, 0.10, DecisionEnum.INCONCLUSIVE, "scoring_margin", id="DEC8-tie-triggers-margin"
        ),
        pytest.param(
            1, 1, 0.10, DecisionEnum.INCONCLUSIVE, "scoring_margin", id="DEC9-low-tie-margin-wins"
        ),
        pytest.param(
            2,
            1,
            0.10,
            DecisionEnum.INCONCLUSIVE,
            "scoring_margin",
            id="DEC10-margin-precedence-over-healthy",
        ),
        pytest.param(10, 0, 0.10, DecisionEnum.HEALTHY, None, id="DEC11-far-healthy"),
        pytest.param(0, 10, 0.10, DecisionEnum.DISEASE, "disease_path", id="DEC12-far-disease"),
        pytest.param(
            2, 0, 0.01, DecisionEnum.INCONCLUSIVE, "low_margin", id="DEC13-retrieval-margin-trigger"
        ),
        pytest.param(
            10, 0, 0.019, DecisionEnum.INCONCLUSIVE, "low_margin", id="DEC14-just-below-theta"
        ),
        pytest.param(
            10, 0, 0.02, DecisionEnum.HEALTHY, None, id="DEC15-strict-lt-boundary-no-trigger"
        ),
        pytest.param(10, 0, 0.021, DecisionEnum.HEALTHY, None, id="DEC16-just-above-theta"),
    ],
)
def test_eq10_decision_precedence(
    s_h: int,
    s_d: int,
    margin: float,
    expected_decision: DecisionEnum,
    expected_reason: str | None,
) -> None:
    """Paper Eq. 10 | Case DEC{1-16} — full precedence × operator boundaries.

    Top-1/Top-2 similarities are synthesized so that
    ``margin = top1 − top2`` matches the matrix value. Operator probes:

        * DEC5  / DEC10 | scoring margin ``≤`` boundary (equality triggers)
        * DEC2          | healthy threshold ``≥`` boundary (equality qualifies)
        * DEC15         | retrieval margin ``<`` boundary (equality does NOT trigger)

    RED: ``make_decision`` raises ``NotImplementedError`` until M2.
    """
    top1, top2 = 0.9, 0.9 - margin
    result = make_decision(
        score_healthy=s_h,
        score_disease=s_d,
        top1_similarity=top1,
        top2_similarity=top2,
        disease_class="D_k" if expected_decision is DecisionEnum.DISEASE else None,
        decision_config=_DECISION,
        margin_config=_MARGIN,
    )
    assert result.decision == expected_decision
    assert result.inconclusive_reason == expected_reason
