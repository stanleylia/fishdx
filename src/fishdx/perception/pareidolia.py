"""Pareidolia correction — paper §III.B, Eq. 1–4 (ADR-0003 dual-path).

Design notes (Stage 1 implementation)
-------------------------------------
The paper's Eq. 2 is implemented verbatim, including the negated second
sigmoid::

    P_soft = σ(cos(G, C_s) − τ) · σ(−(cos(l, C_b) − τ))

For the archived caption/label similarities the soft score remained below
Eq. 4's strict ``> 0.5`` remap threshold.  No universal 0.25 upper bound is
claimed: the value depends on both cosine inputs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from fishdx.config import PareidoliaConfig

_REMAP_THRESHOLD = 0.5  # Eq. 4 strict ``P > 0.5`` remap trigger


def _tokens(text: str) -> set[str]:
    """Lowercase-tokenise on whitespace; Skill S2 §5.2 token-level matching."""
    return set(text.lower().split())


def compute_p_hard(
    caption: str,
    label: str,
    structural_keywords: Sequence[str],
    biological_keywords: Sequence[str],
) -> int:
    """Eq. 1 — Hard keyword path.

    Returns ``1`` iff ``G ∩ K_s ≠ ∅ ∧ G ∩ K_b = ∅``, else ``0``.

    Token-level intersection per Skill S2 §5.2 (case-insensitive); both the
    caption is tokenised on whitespace before set intersection.  ``label`` is
    retained in the signature for API compatibility but is not part of Eq. 1.

    References
    ----------
    Paper Eq. 1.
    """
    caption_tokens = _tokens(caption)
    ks = {k.lower() for k in structural_keywords}
    kb = {k.lower() for k in biological_keywords}
    g_hits = bool(caption_tokens & ks)
    g_biological_hits = bool(caption_tokens & kb)
    return 1 if (g_hits and not g_biological_hits) else 0


def compute_p_soft(
    cos_caption_to_structural: float,
    cos_label_to_biological: float,
    tau: float,
) -> float:
    """Eq. 2 — Soft CLIP-similarity path.

    ``P_soft = σ(cos(G, C_s) − τ) · σ(−(cos(l, C_b) − τ))`` with vanilla
    sigmoid (the second sigmoid's argument is negated, per paper Eq. 2).

    Parameters
    ----------
    cos_caption_to_structural : float
        Cosine similarity between caption CLIP-text embedding and the
        structural concept centroid C_s (mean of CLIP_T(K_s)).
    cos_label_to_biological : float
        Cosine similarity between label CLIP-text embedding and the
        biological concept centroid C_b (mean of CLIP_T(K_b)).
    tau : float
        Pareidolia threshold (Paper Table IX τ = 0.75).

    References
    ----------
    Paper Eq. 2.
    """
    gap_g = cos_caption_to_structural - tau
    gap_l = cos_label_to_biological - tau
    sig_g = 1.0 / (1.0 + math.exp(-gap_g))
    # Second sigmoid takes the NEGATED gap, i.e. σ(−(cos−τ)), per Eq. 2.
    sig_l = 1.0 / (1.0 + math.exp(gap_l))
    return sig_g * sig_l


def merge_pareidolia(p_hard: float, p_soft: float) -> float:
    """Eq. 3 — Two-tier merge: ``P = max(P_hard, P_soft)``.

    References
    ----------
    Paper Eq. 3.
    """
    return max(p_hard, p_soft)


def remap_label(label: str, p: float, remap_target: str) -> str:
    """Eq. 4 — Relabel to ``remap_target`` iff ``P > 0.5`` (strict).

    Equality ``P = 0.5`` does **not** trigger remap (per Skill S4 §2.3
    operator-precision resolution R3).

    References
    ----------
    Paper Eq. 4.
    """
    return remap_target if p > _REMAP_THRESHOLD else label


def correct_pareidolia(
    caption: str,
    label: str,
    config: PareidoliaConfig,
    cos_caption_to_structural: float,
    cos_label_to_biological: float,
) -> tuple[str, float, float, float]:
    """End-to-end Eq. 1–4 pipeline → ``(corrected_label, p_hard, p_soft, merged_p)``.

    Orchestrates the dual-path correction using config-provided ``K_s``,
    ``K_b``, ``τ``, and ``remap_label`` (ADR-0003).
    """
    p_hard = float(
        compute_p_hard(caption, label, config.structural_keywords, config.biological_keywords)
    )
    p_soft = compute_p_soft(
        cos_caption_to_structural, cos_label_to_biological, config.clip_threshold_tau
    )
    merged = merge_pareidolia(p_hard, p_soft)
    corrected = remap_label(label, merged, config.remap_label)
    return corrected, p_hard, p_soft, merged


__all__ = [
    "compute_p_hard",
    "compute_p_soft",
    "correct_pareidolia",
    "merge_pareidolia",
    "remap_label",
]
