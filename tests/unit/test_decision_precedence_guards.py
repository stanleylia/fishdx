"""Precedence-swap guard tests — Skill S4 §2.3 risk-zone bug-class coverage.

These tests intentionally construct cases where multiple Eq. 10 branches
COULD fire, to protect against silent precedence inversion — a bug class
that individual unit tests on each branch would not catch.

Guard 1  — Scoring margin must fire over Healthy.
Guard 2  — Retrieval margin must fire over all others.
"""

from __future__ import annotations

from fishdx.config import DecisionConfig, MarginConfig
from fishdx.schemas import DecisionEnum
from fishdx.scoring.decision import make_decision

_DECISION = DecisionConfig(healthy_threshold_Th=2, inconclusive_margin_m=1)
_MARGIN = MarginConfig(retrieval_margin_theta=0.02)


def test_guard1_scoring_margin_precedence_over_healthy() -> None:
    """GUARD1 | S_h ≥ T_h AND S_h > S_d yet |S_h − S_d| ≤ m → Inconclusive.

    Both Healthy conditions (S_h=2 ≥ T_h=2, S_h=2 > S_d=1) and scoring-
    margin conditions (|2 − 1| = 1 ≤ m=1) hold simultaneously. The
    precedence chain REQUIRES scoring_margin to fire first. If an
    implementation checked Healthy before scoring_margin, this test
    would fail.
    """
    result = make_decision(
        score_healthy=2,
        score_disease=1,
        top1_similarity=0.9,
        top2_similarity=0.5,
        disease_class=None,
        decision_config=_DECISION,
        margin_config=_MARGIN,
    )
    assert result.decision is DecisionEnum.INCONCLUSIVE
    assert result.inconclusive_reason == "scoring_margin"


def test_guard2_retrieval_margin_precedence_over_all() -> None:
    """GUARD2 | margin < θ AND clear Healthy AND clear scoring-margin pass.

    Retrieval margin MUST fire even when downstream signals look clean.
    Constructed: S_h=10 S_d=0 (clearly Healthy), |Δ|=10 > m (passes
    scoring margin), margin 0.01 < θ=0.02 (triggers retrieval safety
    valve). If precedence were scoring_margin → retrieval_margin, this
    would return Healthy and the test would fail.
    """
    result = make_decision(
        score_healthy=10,
        score_disease=0,
        top1_similarity=0.90,
        top2_similarity=0.89,  # margin = 0.01 < θ = 0.02
        disease_class=None,
        decision_config=_DECISION,
        margin_config=_MARGIN,
    )
    assert result.decision is DecisionEnum.INCONCLUSIVE
    assert result.inconclusive_reason == "low_margin"
