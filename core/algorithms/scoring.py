"""Scoring-Based Diagnosis Integration — Eq.8, 9, 10, 11 (v20 §3.4).

Three-way decision system: Healthy / Disease / Inconclusive.
Includes negation detection (Eq.8-9), three-way decision (Eq.10),
and margin-based retrieval confidence override (Eq.11).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from core.config import ScoringConfig

logger = logging.getLogger(__name__)


@dataclass
class ScoringDetail:
    """Detailed scoring breakdown for a single dimension (healthy or disease)."""

    total_score: int = 0
    matched_patterns: list[str] = field(default_factory=list)
    weights_applied: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class DiagnosisDecisionResult:
    """Result of the scoring-based diagnosis decision."""

    status: Literal["Healthy", "Disease", "Inconclusive"]
    healthy_score: int  # S_h
    disease_score: int  # S_d
    confidence: float
    healthy_detail: ScoringDetail
    disease_detail: ScoringDetail
    disease_id: str | None = None
    disease_name: str | None = None
    margin_triggered: bool = False  # True if Eq.11 overrode to Inconclusive
    retrieval_margin: float | None = None  # sim(q,r1) - sim(q,r2)


def detect_negation(text: str, keyword: str, config: ScoringConfig) -> bool:
    """Detect if a disease keyword is negated in the text.

    Neg(dₖ, 𝒪) = 𝟙[∃ p ∈ 𝒩 : p + dₖ ⊆ 𝒪]

    Checks if any negation pattern immediately precedes the keyword.

    Args:
        text: Evidence-pool text 𝒪 (caption ∪ top-1 KB document).
        keyword: Disease keyword dₖ.
        config: Scoring configuration with negation patterns.

    Returns:
        True if the keyword is negated.
    """
    text_lower = text.lower()
    keyword_lower = keyword.lower()

    # Collect all negation patterns from both languages
    all_negation_patterns: list[str] = []
    for patterns in config.negation_patterns.values():
        all_negation_patterns.extend(patterns)

    for neg_pattern in all_negation_patterns:
        # Check if negation pattern appears near the keyword
        # Pattern: negation_word ... keyword (within ~50 chars)
        pattern = re.escape(neg_pattern.lower()) + r".{0,50}" + re.escape(keyword_lower)
        if re.search(pattern, text_lower):
            return True

    return False


def compute_healthy_score(text: str, config: ScoringConfig) -> ScoringDetail:
    """Compute Healthy Score S_h (Eq.8, v20 §3.4).

    S_h = Σᵢ wᵢ⁽ʰ⁾ · 𝟙[hᵢ ∈ 𝒪] + Σₖ wₖ⁽ⁿ⁾ · Neg(dₖ, 𝒪)

    Two components:
    1. Direct healthy indicators (explicit + normal condition)
    2. Negated disease mentions (e.g., "ruled out X disease")

    Args:
        text: evidence-pool text 𝒪 (caption ∪ top-1 KB document).
        config: Scoring configuration.

    Returns:
        ScoringDetail with total score and matched patterns.
    """
    detail = ScoringDetail()
    text_lower = text.lower()

    # Component 1: Direct healthy indicators
    # Explicit healthy (weight: healthy_explicit)
    for lang_indicators in config.healthy_indicators.values():
        for indicator in lang_indicators:
            if indicator.lower() in text_lower:
                weight = config.weights.healthy_explicit
                detail.total_score += weight
                detail.matched_patterns.append(indicator)
                detail.weights_applied.append((f"healthy_explicit: {indicator}", weight))

    # Component 2: Negated disease mentions (weight: healthy_negation)
    # Check each disease indicator for negation
    all_disease_words: list[str] = []
    for key, val in config.disease_indicators.items():
        all_disease_words.extend(val)

    for disease_word in all_disease_words:
        if disease_word.lower() in text_lower:
            if detect_negation(text, disease_word, config):
                weight = config.weights.healthy_negation
                detail.total_score += weight
                detail.matched_patterns.append(f"negated:{disease_word}")
                detail.weights_applied.append((f"healthy_negation: {disease_word}", weight))

    return detail


def compute_disease_score(text: str, config: ScoringConfig) -> ScoringDetail:
    """Compute Disease Score S_d (Eq.9, v20 §3.4).

    S_d = Σⱼ wⱼ⁽ᵈ⁾ · 𝟙[dⱼ ∈ 𝒪]

    Matches disease patterns at three levels:
    - Confirmed (+3): 確診, diagnosed
    - Suspected (+2): 感染, suspected
    - Mentioned (+1): disease name directly appeared

    Args:
        text: evidence-pool text 𝒪 (caption ∪ top-1 KB document).
        config: Scoring configuration.

    Returns:
        ScoringDetail with total score and matched patterns.
    """
    detail = ScoringDetail()
    text_lower = text.lower()

    # Level 1: Confirmed diseases (weight: disease_confirmed = 3)
    confirmed_patterns: list[str] = []
    for key in ("confirmed_zh", "confirmed_en"):
        if key in config.disease_indicators:
            confirmed_patterns.extend(config.disease_indicators[key])

    for pattern in confirmed_patterns:
        if pattern.lower() in text_lower:
            # Skip if negated
            if detect_negation(text, pattern, config):
                continue
            weight = config.weights.disease_confirmed
            detail.total_score += weight
            detail.matched_patterns.append(pattern)
            detail.weights_applied.append((f"disease_confirmed: {pattern}", weight))

    # Level 2: Suspected diseases (weight: disease_suspected = 2)
    suspected_patterns: list[str] = []
    for key in ("suspected_zh", "suspected_en"):
        if key in config.disease_indicators:
            suspected_patterns.extend(config.disease_indicators[key])

    for pattern in suspected_patterns:
        if pattern.lower() in text_lower:
            if detect_negation(text, pattern, config):
                continue
            weight = config.weights.disease_suspected
            detail.total_score += weight
            detail.matched_patterns.append(pattern)
            detail.weights_applied.append((f"disease_suspected: {pattern}", weight))

    # Level 3: Mentioned symptom keywords (weight: disease_mentioned = 1)
    mentioned_patterns: list[str] = []
    for key in ("mentioned_zh", "mentioned_en"):
        if key in config.disease_indicators:
            mentioned_patterns.extend(config.disease_indicators[key])

    for pattern in mentioned_patterns:
        if pattern.lower() in text_lower:
            if detect_negation(text, pattern, config):
                continue
            weight = config.weights.disease_mentioned
            detail.total_score += weight
            detail.matched_patterns.append(pattern)
            detail.weights_applied.append((f"disease_mentioned: {pattern}", weight))

    return detail


def compute_retrieval_margin(rag_results: list[dict]) -> float | None:
    """Compute retrieval margin (Eq.11): margin = sim(q,r₁) − sim(q,r₂).

    When margin < θ_margin, the system lacks confident retrieval evidence
    and should output Inconclusive. This has highest decision priority.

    Args:
        rag_results: Sorted list of RAG results with 'similarity' key.

    Returns:
        Margin value, or None if fewer than 2 results.
    """
    if not rag_results or len(rag_results) < 2:
        return None
    sims = sorted(
        (r.get("similarity", 0) for r in rag_results),
        reverse=True,
    )
    return sims[0] - sims[1]


def make_decision(
    s_h: int,
    s_d: int,
    config: ScoringConfig,
) -> Literal["Healthy", "Disease", "Inconclusive"]:
    """Apply the three-way decision rule (Eq.10).

    Decision priority (v20 §3.4):
    1. Retrieval Margin (Eq.11) — handled externally via compute_retrieval_margin
    2. Scoring Margin: Inconclusive if |S_h − S_d| ≤ m
    3. Healthy if S_h ≥ T_h AND S_h > S_d
    4. Disease if S_d > S_h
    5. Inconclusive otherwise

    Args:
        s_h: Healthy score S_h.
        s_d: Disease score S_d.
        config: Scoring configuration with thresholds.

    Returns:
        Decision string.
    """
    margin = abs(s_h - s_d)

    if margin <= config.inconclusive_margin:
        return "Inconclusive"
    elif s_h >= config.healthy_threshold and s_h > s_d:
        return "Healthy"
    elif s_d > s_h:
        return "Disease"
    else:
        return "Inconclusive"


def full_scoring_pipeline(
    evidence_text: str,
    config: ScoringConfig,
    rag_results: list[dict] | None = None,
    theta_margin: float = 0.02,
) -> DiagnosisDecisionResult:
    """Run the complete scoring pipeline: S_h + S_d → Decision (Eq.8-11).

    Decision priority:
    1. Eq.11 Retrieval Margin: if margin < θ_margin → Inconclusive
    2. Eq.10 Three-Way Decision: scoring-based classification

    Args:
        evidence_text: Raw evidence-pool text 𝒪 (caption ∪ top-1 KB document).
        config: Scoring configuration.
        rag_results: Optional RAG results for margin computation (Eq.11).
        theta_margin: Margin threshold (default 0.02 per v20 §3.4).

    Returns:
        DiagnosisDecisionResult with full scoring breakdown.
    """
    healthy_detail = compute_healthy_score(evidence_text, config)
    disease_detail = compute_disease_score(evidence_text, config)

    s_h = healthy_detail.total_score
    s_d = disease_detail.total_score

    status = make_decision(s_h, s_d, config)

    # Eq.11: Retrieval margin override (highest decision priority)
    margin_triggered = False
    retrieval_margin = None
    if rag_results is not None:
        retrieval_margin = compute_retrieval_margin(rag_results)
        if retrieval_margin is not None and retrieval_margin < theta_margin:
            if status != "Inconclusive":
                logger.info(
                    f"Eq.11 margin override: margin={retrieval_margin:.4f} "
                    f"< θ={theta_margin} → Inconclusive (was {status})"
                )
                status = "Inconclusive"
            margin_triggered = True

    # Compute confidence as normalized score difference
    total = s_h + s_d
    if total > 0:
        confidence = max(s_h, s_d) / total
    else:
        confidence = 0.0

    logger.info(f"Scoring: S_h={s_h}, S_d={s_d} → {status} (confidence={confidence:.2f})")

    return DiagnosisDecisionResult(
        status=status,
        healthy_score=s_h,
        disease_score=s_d,
        confidence=confidence,
        healthy_detail=healthy_detail,
        disease_detail=disease_detail,
        margin_triggered=margin_triggered,
        retrieval_margin=retrieval_margin,
    )
