"""Three-way decision + retrieval-margin safety valve — paper §III.D, Eq. 10–11.

Precedence chain (Skill S4 §2.3, ADR-0001, paper §III.D)
--------------------------------------------------------
    [1]  Retrieval margin  (Eq. 11, strict ``<``)  → Inconclusive(low_margin)
    [2]  Scoring margin    (Eq. 10 row 1, ``≤``)    → Inconclusive(scoring_margin)
    [3]  Healthy           (Eq. 10 row 2, ``≥`` ∧ ``>``) → Healthy
    [4]  Disease           (Eq. 10 row 3, strict ``>``)  → Disease(D_k, disease_path)
    [5]  Fallback                                   → Inconclusive(fallback)

The precedence is **not** symmetric with score ordering — the scoring
margin fires even when Healthy conditions hold (see DEC10 in the test
matrix), because confidence must exceed ``m`` before a decisive path is
taken.
"""

from __future__ import annotations

from fishdx.config import DecisionConfig, MarginConfig
from fishdx.schemas import DecisionEnum, DiagnosisResult


def apply_retrieval_margin(
    top1_similarity: float,
    top2_similarity: float,
    margin_config: MarginConfig,
) -> bool:
    """Eq. 11 — return ``True`` iff ``sim₁ − sim₂ < θ_margin`` (strict).

    R3 operator precision: equality ``margin == θ`` does **not** trigger.
    """
    margin = top1_similarity - top2_similarity
    return margin < margin_config.retrieval_margin_theta


def make_decision(
    score_healthy: int,
    score_disease: int,
    top1_similarity: float,
    top2_similarity: float,
    disease_class: str | None,
    decision_config: DecisionConfig,
    margin_config: MarginConfig,
) -> DiagnosisResult:
    """Eq. 10 three-way decision with Eq. 11 retrieval-margin safety valve."""
    retrieval_margin = top1_similarity - top2_similarity

    # Step [1] — retrieval margin safety valve (highest precedence).
    if apply_retrieval_margin(top1_similarity, top2_similarity, margin_config):
        return DiagnosisResult(
            decision=DecisionEnum.INCONCLUSIVE,
            disease_class=None,
            score_healthy=float(score_healthy),
            score_disease=float(score_disease),
            retrieval_margin=retrieval_margin,
            inconclusive_reason="low_margin",
        )

    # Step [2] — scoring margin precedence (fires over Healthy/Disease).
    if abs(score_healthy - score_disease) <= decision_config.inconclusive_margin_m:
        low_health = (
            score_healthy > score_disease and score_healthy < decision_config.healthy_threshold_Th
        )
        return DiagnosisResult(
            decision=DecisionEnum.INCONCLUSIVE,
            disease_class=None,
            score_healthy=float(score_healthy),
            score_disease=float(score_disease),
            retrieval_margin=retrieval_margin,
            inconclusive_reason="low_health_score" if low_health else "scoring_margin",
        )

    # Step [3] — Healthy path (S_h ≥ T_h AND S_h > S_d, both strict-tight).
    if score_healthy >= decision_config.healthy_threshold_Th and score_healthy > score_disease:
        return DiagnosisResult(
            decision=DecisionEnum.HEALTHY,
            disease_class=None,
            score_healthy=float(score_healthy),
            score_disease=float(score_disease),
            retrieval_margin=retrieval_margin,
            inconclusive_reason=None,
        )

    # Step [4] — Disease path (strict S_d > S_h).
    if score_disease > score_healthy:
        return DiagnosisResult(
            decision=DecisionEnum.DISEASE,
            disease_class=disease_class,
            score_healthy=float(score_healthy),
            score_disease=float(score_disease),
            retrieval_margin=retrieval_margin,
            inconclusive_reason=None,
        )

    # Step [5] — Fallback (tie after step 2 / healthy threshold miss).
    return DiagnosisResult(
        decision=DecisionEnum.INCONCLUSIVE,
        disease_class=None,
        score_healthy=float(score_healthy),
        score_disease=float(score_disease),
        retrieval_margin=retrieval_margin,
        inconclusive_reason="fallback",
    )


__all__ = ["apply_retrieval_margin", "make_decision"]
