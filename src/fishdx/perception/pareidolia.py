"""Pareidolia correction — paper §III.B, Eq. 1–4.

This module implements the paper's Eq. 1–4 verbatim:

    Eq. 1  P_hard(l_i) = 𝟙[G ∩ K_s ≠ ∅] · 𝟙[G ∩ K_b = ∅]
    Eq. 2  P_soft(l_i) = σ(cos(CLIP_T(G), C_s) − τ) · σ(−(cos(CLIP_T(l_i), C_b) − τ))
    Eq. 3  P(l_i) = max(P_hard(l_i), P_soft(l_i))
    Eq. 4  l_i' = Remap(l_i, "net_damage")  if  P(l_i) > 0.5

Under the paper-stated cosine range cos ∈ [0.20, 0.35] with τ = 0.75, the
σ-product output sits in ``[0.22, 0.25]`` (closed-form upper bound 0.25 at
cos = τ). The soft path alone therefore never crosses Eq. 4's strict
``> 0.5`` remap trigger; it acts as an informational signal that contributes
through Eq. 3's max-merge when the hard path fires.
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

    Both indicator functions test the caption G — the trigger fires when
    the caption mentions a structural keyword (K_s) AND mentions no
    biological keyword (K_b). The label parameter is accepted for
    interface compatibility with downstream Eq. 4 remapping but is not
    used in the Eq. 1 indicator computation.

    Token-level intersection (case-insensitive, whitespace-split) so
    "cage-enclosure" does not falsely match "cage".

    References
    ----------
    Paper Eq. 1.
    """
    del label  # Eq. 1 indicator is caption-only; label used only by Eq. 4 remap
    caption_tokens = _tokens(caption)
    ks = {k.lower() for k in structural_keywords}
    kb = {k.lower() for k in biological_keywords}
    has_structural = bool(caption_tokens & ks)
    has_biological = bool(caption_tokens & kb)
    return 1 if (has_structural and not has_biological) else 0


def compute_p_soft(
    cos_caption_to_structural: float,
    cos_label_to_biological: float,
    tau: float,
) -> float:
    """Eq. 2 — Soft CLIP-similarity path.

    ``P_soft = σ(cos(G, C_s) − τ) · σ(−(cos(l, C_b) − τ))`` with vanilla sigmoid.

    The first sigmoid is high when the caption is structural-looking;
    the second sigmoid is high when the label is NOT biological-looking.
    Note the negation in the second sigmoid argument: ``σ(τ − cos)``.

    Parameters
    ----------
    cos_caption_to_structural : float
        Cosine similarity between caption CLIP-text embedding and the
        structural concept centroid C_s (mean of CLIP_T(K_s)).
    cos_label_to_biological : float
        Cosine similarity between label CLIP-text embedding and the
        biological concept centroid C_b (mean of CLIP_T(K_b)).
    tau : float
        Pareidolia threshold (Paper Table 2, τ = 0.75).

    References
    ----------
    Paper Eq. 2.
    """
    gap_g = cos_caption_to_structural - tau
    gap_l_negated = -(cos_label_to_biological - tau)  # paper Eq. 2: σ(−(cos − τ))
    sig_g = 1.0 / (1.0 + math.exp(-gap_g))
    sig_l = 1.0 / (1.0 + math.exp(-gap_l_negated))
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
