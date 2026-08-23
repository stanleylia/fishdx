"""Two-Tier Pareidolia Detection — Eq.1, 2, 3, 4 (v20 §3.2).

Problem: Net/cage structures are misidentified as biological features (face, person)
by VLMs. This module implements a two-tier correction mechanism.

Eq.1 Hard Path: Fast keyword matching O(1)
Eq.2 Soft Path: CLIP semantic similarity with centroids
Eq.3 Combined: max(P_hard, P_soft)
Eq.4 Label remap: if P(li) > 0.5 → remap to net_damage
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from core.config import PareidoliaConfig

logger = logging.getLogger(__name__)


@dataclass
class PareidoliaResult:
    """Result of pareidolia detection for a single label."""

    label: str
    is_pareidolia: bool
    hard_score: float = 0.0  # P_hard
    soft_score: float = 0.0  # P_soft
    combined_score: float = 0.0  # P = max(P_hard, P_soft)
    remapped_label: str = ""


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    exp_x = np.exp(x)
    return exp_x / (1.0 + exp_x)


def detect_hard(
    caption: str,
    object_labels: list[str],
    config: PareidoliaConfig,
) -> list[PareidoliaResult]:
    """Tier 1 — Fast Path keyword matching (Eq.1a).

    P_hard(lᵢ) = 𝟙[G ∩ 𝒦_s ≠ ∅] · 𝟙[lᵢ ∩ 𝒦_b ≠ ∅]

    Checks if the global caption contains structural keywords and contains no
    biological keyword. The per-label return shape is retained for compatibility.

    Args:
        caption: Global caption G from Florence-2.
        object_labels: Object labels L from Florence-2.
        config: Pareidolia configuration.

    Returns:
        List of PareidoliaResult for each label.
    """
    caption_lower = caption.lower()
    structural_kw = config.structural_keywords
    biological_kw = config.biological_keywords

    # 𝟙[G ∩ 𝒦_s ≠ ∅] — check if caption contains structural keywords
    caption_has_structural = any(kw in caption_lower for kw in structural_kw)
    caption_has_biological = any(kw in caption_lower for kw in biological_kw)

    results: list[PareidoliaResult] = []
    for label in object_labels:
        label_lower = label.lower()  # retained for backward-compatible local tracing
        # 𝟙[lᵢ ∩ 𝒦_b ≠ ∅] — check if label contains biological keywords
        label_has_biological = any(kw in label_lower for kw in biological_kw)

        hard_score = 1.0 if (caption_has_structural and not caption_has_biological) else 0.0

        results.append(PareidoliaResult(
            label=label,
            is_pareidolia=hard_score > 0,
            hard_score=hard_score,
        ))

    return results


def detect_soft(
    caption: str,
    object_labels: list[str],
    clip_text_fn: object,
    config: PareidoliaConfig,
    structural_centroid: NDArray[np.float32] | None = None,
    biological_centroid: NDArray[np.float32] | None = None,
) -> list[PareidoliaResult]:
    """Tier 2 — Soft Path CLIP semantic detection (Eq.1b).

    P_soft(lᵢ) = σ(cos(CLIP_T(G), 𝒞_s) − τ) · σ(−(cos(CLIP_T(lᵢ), 𝒞_b) − τ))

    Uses CLIP text embeddings to compute semantic similarity with
    structural and biological centroids. The second sigmoid's argument is
    negated, per paper Eq. 2.

    Args:
        caption: Global caption G.
        object_labels: Object labels L.
        clip_text_fn: CLIP text encoding callable (str -> NDArray).
        config: Pareidolia configuration.
        structural_centroid: Pre-computed 𝒞_s (optional, computed if None).
        biological_centroid: Pre-computed 𝒞_b (optional, computed if None).

    Returns:
        List of PareidoliaResult with soft scores.
    """
    tau = config.clip_threshold

    # Compute centroids if not provided
    if structural_centroid is None:
        structural_centroid = _compute_centroid(config.structural_keywords, clip_text_fn)
    if biological_centroid is None:
        biological_centroid = _compute_centroid(config.biological_keywords, clip_text_fn)

    # Encode caption: CLIP_T(G)
    caption_emb = clip_text_fn(caption)
    caption_emb = caption_emb / (np.linalg.norm(caption_emb) + 1e-10)

    # cos(CLIP_T(G), 𝒞_s)
    cos_caption_structural = float(np.dot(caption_emb.flatten(), structural_centroid.flatten()))

    results: list[PareidoliaResult] = []
    for label in object_labels:
        # Encode label: CLIP_T(lᵢ)
        label_emb = clip_text_fn(label)
        label_emb = label_emb / (np.linalg.norm(label_emb) + 1e-10)

        # cos(CLIP_T(lᵢ), 𝒞_b)
        cos_label_biological = float(np.dot(label_emb.flatten(), biological_centroid.flatten()))

        # P_soft = σ(cos_s − τ) · σ(−(cos_b − τ))  — negated 2nd sigmoid per Eq. 2
        soft_score = _sigmoid(cos_caption_structural - tau) * _sigmoid(-(cos_label_biological - tau))

        results.append(PareidoliaResult(
            label=label,
            is_pareidolia=soft_score > 0.5,
            soft_score=soft_score,
        ))

    return results


def _compute_centroid(
    keywords: list[str],
    clip_text_fn: object,
) -> NDArray[np.float32]:
    """Compute the centroid of CLIP text embeddings for a set of keywords.

    𝒞 = (1/|𝒦|) Σ_{k∈𝒦} CLIP_T(k)

    Args:
        keywords: List of keywords.
        clip_text_fn: CLIP text encoding callable.

    Returns:
        Normalized centroid vector.
    """
    embeddings = []
    for kw in keywords:
        emb = clip_text_fn(kw)
        embeddings.append(emb.flatten())

    centroid = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
    norm = np.linalg.norm(centroid)
    if norm > 1e-10:
        centroid = centroid / norm
    return centroid


def detect_combined(
    caption: str,
    object_labels: list[str],
    config: PareidoliaConfig,
    clip_text_fn: object | None = None,
    structural_centroid: NDArray[np.float32] | None = None,
    biological_centroid: NDArray[np.float32] | None = None,
) -> list[PareidoliaResult]:
    """Combined Two-Tier Pareidolia Detection (Eq.1).

    P(lᵢ) = max(P_hard(lᵢ), P_soft(lᵢ))

    Falls back to hard-only if CLIP is not available.

    Args:
        caption: Global caption G.
        object_labels: Object labels L.
        config: Pareidolia configuration.
        clip_text_fn: Optional CLIP text encoding callable.
        structural_centroid: Optional pre-computed structural centroid.
        biological_centroid: Optional pre-computed biological centroid.

    Returns:
        List of PareidoliaResult with combined scores.
    """
    hard_results = detect_hard(caption, object_labels, config)

    # If CLIP not available, use hard-only
    if clip_text_fn is None:
        logger.warning("CLIP not available, using hard detection only")
        return hard_results

    soft_results = detect_soft(
        caption, object_labels, clip_text_fn, config,
        structural_centroid, biological_centroid,
    )

    # Combine: P = max(P_hard, P_soft)
    combined: list[PareidoliaResult] = []
    for hard, soft in zip(hard_results, soft_results):
        combined_score = max(hard.hard_score, soft.soft_score)
        combined.append(PareidoliaResult(
            label=hard.label,
            is_pareidolia=combined_score > 0.5,
            hard_score=hard.hard_score,
            soft_score=soft.soft_score,
            combined_score=combined_score,
        ))

    return combined


def remap_labels(
    object_labels: list[str],
    results: list[PareidoliaResult],
    remap_target: str = "net_damage",
) -> list[str]:
    """Remap pareidolia-flagged labels to target label (Eq.2).

    lᵢ' = Remap(lᵢ, "net_damage") if P(lᵢ) > 0

    Args:
        object_labels: Original object labels.
        results: Pareidolia detection results.
        remap_target: Target label for remapping.

    Returns:
        List of remapped labels.
    """
    remapped: list[str] = []
    for label, result in zip(object_labels, results):
        if result.is_pareidolia:
            remapped.append(remap_target)
            logger.info(f"Pareidolia remap: '{label}' → '{remap_target}'")
        else:
            remapped.append(label)
    return remapped
