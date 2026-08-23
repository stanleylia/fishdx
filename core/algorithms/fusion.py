"""λ-Weighted Fusion Embedding — Eq.5, 6, 7 (v20 §3.3).

Eq.5 L2 Pre-normalization
Eq.6 λ-Weighted Fusion (λ* = 0.7)
Eq.7 Post-normalization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.config import FusionConfig

logger = logging.getLogger(__name__)


@dataclass
class FusionResult:
    """Result of the fusion embedding pipeline."""

    fused_embedding: NDArray[np.float32]  # 512-dim Ê_final
    visual_embedding: NDArray[np.float32]  # Ê_v (normalized)
    caption_embedding: NDArray[np.float32]  # Ê_c (normalized)
    lambda_weight: float
    norm_check: float  # ‖Ê_final‖₂ (should be ~1.0)


def normalize_l2(embedding: NDArray[np.float32]) -> NDArray[np.float32]:
    """L2-normalize an embedding vector (Eq.3).

    Ê_c = E_c / ‖E_c‖₂ ,  Ê_v = E_v / ‖E_v‖₂

    Args:
        embedding: Raw embedding vector.

    Returns:
        L2-normalized embedding.
    """
    norm = np.linalg.norm(embedding)
    if norm < 1e-10:
        logger.warning("Near-zero norm embedding detected")
        return embedding.astype(np.float32)
    return (embedding / norm).astype(np.float32)


def fuse_embeddings(
    e_visual: NDArray[np.float32],
    e_caption: NDArray[np.float32],
    lambda_weight: float = 0.7,
) -> NDArray[np.float32]:
    """λ-Weighted Fusion of pre-normalized embeddings (Eq.4, 5).

    E_fused = λ · Ê_v + (1-λ) · Ê_c    (Eq.4)
    Ê_final = E_fused / ‖E_fused‖₂      (Eq.5)

    Args:
        e_visual: Pre-normalized visual embedding Ê_v.
        e_caption: Pre-normalized caption embedding Ê_c.
        lambda_weight: Visual weight λ, default 0.7 (λ*).

    Returns:
        L2-normalized fused embedding Ê_final.
    """
    if not 0.0 <= lambda_weight <= 1.0:
        raise ValueError(f"lambda_weight must be in [0, 1], got {lambda_weight}")

    fused = lambda_weight * e_visual + (1.0 - lambda_weight) * e_caption
    return normalize_l2(fused)


def aggregate_multi_element(
    embeddings: list[NDArray[np.float32]],
) -> NDArray[np.float32]:
    """Multi-element aggregation via mean (Eq.7).

    E_agg = (1/N) Σᵢ Ê_final(i)

    Used when multiple objects are detected in a single image and their
    individual fused embeddings need to be combined.

    Args:
        embeddings: List of normalized fused embeddings.

    Returns:
        Aggregated and re-normalized embedding.

    Raises:
        ValueError: If embeddings list is empty.
    """
    if not embeddings:
        raise ValueError("Cannot aggregate empty embeddings list")

    if len(embeddings) == 1:
        return normalize_l2(embeddings[0])

    stacked = np.stack(embeddings)
    mean_emb = np.mean(stacked, axis=0).astype(np.float32)
    return normalize_l2(mean_emb)


def create_fusion_embedding(
    visual_embedding: NDArray[np.float32],
    caption_embedding: NDArray[np.float32],
    config: FusionConfig,
) -> FusionResult:
    """Complete fusion pipeline: normalize → fuse → re-normalize (Eq.3-5).

    Args:
        visual_embedding: Raw CLIP image embedding E_v.
        caption_embedding: Raw CLIP text embedding E_c.
        config: Fusion configuration (contains λ*).

    Returns:
        FusionResult with all intermediate and final embeddings.
    """
    # Eq.3: Pre-normalization
    e_v_norm = normalize_l2(visual_embedding)
    e_c_norm = normalize_l2(caption_embedding)

    # Eq.4 + Eq.5: Fuse and post-normalize
    fused = fuse_embeddings(e_v_norm, e_c_norm, config.lambda_weight)

    # Verify unit norm
    norm_check = float(np.linalg.norm(fused))

    return FusionResult(
        fused_embedding=fused,
        visual_embedding=e_v_norm,
        caption_embedding=e_c_norm,
        lambda_weight=config.lambda_weight,
        norm_check=norm_check,
    )
