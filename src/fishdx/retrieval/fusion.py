"""λ-weighted fusion — paper §III.C, Eq. 5–7.

Eq. 5  L2 pre-normalization      Ê = E / ‖E‖_2
Eq. 6  λ-weighted sum             E_fused = λ · E_visual + (1-λ) · E_caption
Eq. 7  L2 post-normalization      E_final = E_fused / ‖E_fused‖_2

All fusion math runs in fp32 for numerical stability (Skill S3 §2.2) even
when upstream encoders emit fp16.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from fishdx.errors import FusionNormalizationError

_ZERO_NORM_EPS = 1e-12


def _to_fp32(embedding: Sequence[float]) -> list[float]:
    """Upcast sequence to Python float (fp64) for numerical stability."""
    return [float(x) for x in embedding]


def l2_normalize(embedding: Sequence[float]) -> list[float]:
    """Eq. 5 / Eq. 7 — L2-normalise ``embedding`` to unit sphere.

    Raises
    ------
    FusionNormalizationError
        If ``‖embedding‖_2 < 1e-12`` (zero or near-zero vector cannot be
        normalised without producing NaN / Inf).

    References
    ----------
    Paper Eq. 5, Eq. 7.
    """
    values = _to_fp32(embedding)
    norm = math.sqrt(sum(v * v for v in values))
    if norm < _ZERO_NORM_EPS:
        raise FusionNormalizationError(
            "zero-norm vector cannot be L2-normalised",
            context={"norm": norm, "dim": len(values)},
        )
    return [v / norm for v in values]


def fuse_embeddings(
    e_visual: Sequence[float],
    e_caption: Sequence[float],
    lambda_weight: float,
) -> list[float]:
    """Eq. 6 — ``E_fused = λ·E_visual + (1-λ)·E_caption``.

    ``e_visual`` and ``e_caption`` must share the same dimensionality;
    otherwise a ``ValueError`` is raised (programmer error, not a user
    input error).
    """
    if len(e_visual) != len(e_caption):
        raise ValueError(f"dimension mismatch: visual={len(e_visual)}, caption={len(e_caption)}")
    v = _to_fp32(e_visual)
    c = _to_fp32(e_caption)
    lam = float(lambda_weight)
    complement = 1.0 - lam
    return [lam * vi + complement * ci for vi, ci in zip(v, c)]


def fuse_and_normalize(
    e_visual: Sequence[float],
    e_caption: Sequence[float],
    lambda_weight: float,
    normalize_pre: bool = True,
    normalize_post: bool = True,
) -> list[float]:
    """Paper Eq. 5 → Eq. 6 → Eq. 7 end-to-end fusion pipeline.

    Returns the final L2-normalised 512-d fused embedding suitable for
    ChromaDB cosine retrieval.
    """
    ev = l2_normalize(e_visual) if normalize_pre else _to_fp32(e_visual)
    ec = l2_normalize(e_caption) if normalize_pre else _to_fp32(e_caption)
    fused = fuse_embeddings(ev, ec, lambda_weight)
    return l2_normalize(fused) if normalize_post else fused


__all__ = ["fuse_and_normalize", "fuse_embeddings", "l2_normalize"]
