"""Algorithm 1 — retrieval-confidence reweighting tests (single pass). See ADR-0016.

Supersedes the grounding-based iterative Bidirectional Verification Loop tests.
Penalise once iff top-1 similarity < θ_sim OR top-1/top-2 margin < θ_margin,
then drop candidates whose reweighted score < min_score.

References
----------
Paper Algorithm 1 (Retrieval-confidence reweighting); ADR-0016.
"""

from __future__ import annotations

from fishdx.config import ReweightingConfig
from fishdx.retrieval.reweighting import reweight_candidates
from fishdx.schemas import RetrievedDoc

_CFG = ReweightingConfig(
    similarity_threshold=0.85,
    margin_threshold=0.02,
    penalty_factor=0.5,
    min_score=0.3,
)


def _make_doc(doc_id: str, similarity: float) -> RetrievedDoc:
    return RetrievedDoc(
        doc_id=doc_id,
        text=f"canned doc {doc_id}",
        disease_class="D_k",
        similarity=similarity,
    )


def test_high_confidence_no_penalty() -> None:
    """s1 ≥ θ_sim and margin ≥ θ_margin → no penalty; scores unchanged."""
    out = reweight_candidates([_make_doc("d1", 0.9), _make_doc("d2", 0.85)], _CFG)
    assert [c.doc_id for c in out] == ["d1", "d2"]
    assert out[0].similarity_penalized == 0.9
    assert out[1].similarity_penalized == 0.85


def test_low_similarity_penalised_and_filtered() -> None:
    """s1 < θ_sim → penalty ×0.5; candidates below min_score are dropped."""
    out = reweight_candidates([_make_doc("d1", 0.8), _make_doc("d2", 0.5)], _CFG)
    # 0.8×0.5 = 0.40 survives; 0.5×0.5 = 0.25 < 0.3 dropped
    assert len(out) == 1
    assert out[0].doc_id == "d1"
    assert out[0].similarity_penalized == 0.8 * 0.5


def test_low_margin_penalised() -> None:
    """s1 ≥ θ_sim but margin < θ_margin → penalty ×0.5 applied to all."""
    out = reweight_candidates([_make_doc("d1", 0.9), _make_doc("d2", 0.89)], _CFG)
    assert len(out) == 2
    assert out[0].similarity_penalized == 0.9 * 0.5
    assert out[1].similarity_penalized == 0.89 * 0.5


def test_empty_candidates() -> None:
    """Empty input → empty output."""
    assert reweight_candidates([], _CFG) == []


def test_penalty_can_drop_all() -> None:
    """Single low-similarity candidate: margin=0 → penalty → below min_score → dropped."""
    out = reweight_candidates([_make_doc("d1", 0.5)], _CFG)
    assert out == []


def test_output_sorted_by_reweighted_similarity() -> None:
    """Survivors are returned sorted by reweighted similarity, descending."""
    out = reweight_candidates(
        [_make_doc("d2", 0.86), _make_doc("d1", 0.90), _make_doc("d3", 0.70)], _CFG
    )
    scores = [c.similarity_penalized for c in out]
    assert scores == sorted(scores, reverse=True)
