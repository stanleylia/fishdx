"""Retrieval-confidence reweighting — paper Algorithm 1 (single pass). See ADR-0016.

A single-pass rule that down-weights a low-confidence retrieval::

    s1     ← top-1 gallery cosine similarity
    margin ← sim(c1) − sim(c2)
    penalise once (× penalty_factor) iff  s1 < θ_sim  OR  margin < θ_margin
    drop any candidate whose reweighted score < min_score, then re-rank

No external grounding score is consumed; the confidence signal is derived
entirely from the fused-retrieval similarities. ADR-0016 supersedes the earlier
grounding-based iterative Bidirectional Verification Loop (which ran with a
constant placeholder grounding score and produced no reported number).
"""

from __future__ import annotations

from collections.abc import Sequence

from fishdx.config import ReweightingConfig
from fishdx.schemas import RetrievedDoc


def reweight_candidates(
    candidates: Sequence[RetrievedDoc],
    config: ReweightingConfig,
) -> list[RetrievedDoc]:
    """Apply Algorithm 1 — single-pass retrieval-confidence reweighting.

    Parameters
    ----------
    candidates : Sequence[RetrievedDoc]
        Fused-retrieval candidates (any order; ranked internally by similarity).
    config : ReweightingConfig
        ``similarity_threshold`` (θ_sim), ``margin_threshold`` (θ_margin),
        ``penalty_factor``, ``min_score``.

    Returns
    -------
    list[RetrievedDoc]
        Reweighted candidates with ``similarity_penalized`` populated, filtered
        below ``min_score`` and sorted by reweighted similarity, descending.

    References
    ----------
    Paper Algorithm 1 (Retrieval-confidence reweighting); ADR-0016.
    """
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda c: c.similarity, reverse=True)
    s1 = ranked[0].similarity
    s2 = ranked[1].similarity if len(ranked) > 1 else s1
    margin = s1 - s2
    low_confidence = s1 < config.similarity_threshold or margin < config.margin_threshold
    factor = config.penalty_factor if low_confidence else 1.0
    out: list[RetrievedDoc] = []
    for cand in ranked:
        score = cand.similarity * factor
        if score >= config.min_score:
            out.append(cand.model_copy(update={"similarity_penalized": score}))
    out.sort(key=lambda c: c.similarity_penalized or 0.0, reverse=True)
    return out


__all__ = ["reweight_candidates"]
