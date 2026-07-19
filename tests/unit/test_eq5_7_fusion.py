"""λ-weighted fusion unit tests — paper §III.C, Eq. 5–7.

Covers docs/m0/test-matrix.md case IDs:
    Eq. 5 L2 Pre-norm (3):  PRE1–PRE3
    Eq. 6 λ-Fusion    (5):  FUS1–FUS5
    Eq. 7 L2 Post-norm (2): POST1–POST2

All 10 cases are RED: target functions raise ``NotImplementedError``.
End-to-end integration cases live in tests/integration/test_fusion_e2e.py.

References
----------
Paper §III.C "Knowledge Retrieval via CLIP Fusion Embedding".
"""

from __future__ import annotations

import pytest

from fishdx.errors import FusionNormalizationError
from fishdx.retrieval.fusion import fuse_embeddings, l2_normalize


# ────────────────────────── Eq. 5 L2 Pre-norm (PRE1–PRE3) ──────────────────────────
def test_eq5_pre1_triangle() -> None:
    """Paper Eq. 5 | Case PRE1 | 3-4-5 triangle unit-norm check.

    RED: ``l2_normalize`` raises ``NotImplementedError`` until M2.
    """
    result = l2_normalize([3.0, 4.0])
    assert result == pytest.approx([0.6, 0.8])


def test_eq5_pre2_zero_vector() -> None:
    """Paper Eq. 5 | Case PRE2 | Zero vector raises FusionNormalizationError.

    Operator precision: division by zero MUST surface as a typed error, not
    silent NaN propagation (architecture.md §2.3 R2).
    """
    with pytest.raises(FusionNormalizationError):
        l2_normalize([0.0, 0.0])


def test_eq5_pre3_idempotent() -> None:
    """Paper Eq. 5 | Case PRE3 | Already unit-norm is idempotent."""
    result = l2_normalize([1.0, 0.0])
    assert result == pytest.approx([1.0, 0.0])


# ────────────────────────── Eq. 6 λ-Fusion (FUS1–FUS5) ──────────────────────────
@pytest.mark.parametrize(
    ("lam", "e_visual", "e_caption", "expected"),
    [
        pytest.param(0.0, [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], id="FUS1-lambda-0-caption-only"),
        pytest.param(1.0, [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], id="FUS2-lambda-1-visual-only"),
        pytest.param(0.7, [1.0, 0.0], [0.0, 1.0], [0.7, 0.3], id="FUS3-paper-lambda-star"),
        pytest.param(0.5, [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], id="FUS4-balanced"),
        pytest.param(
            0.7,
            [0.6, 0.8],
            [0.8, 0.6],
            [0.66, 0.74],
            id="FUS5-numerical-stability",
        ),
    ],
)
def test_eq6_fusion(
    lam: float, e_visual: list[float], e_caption: list[float], expected: list[float]
) -> None:
    """Paper Eq. 6 | Case FUS{1-5}.

    Expected: ``E_fused = λ·E_visual + (1−λ)·E_caption``. Tolerance 1e-6.

    RED: ``fuse_embeddings`` raises ``NotImplementedError`` until M2.
    """
    result = fuse_embeddings(e_visual, e_caption, lam)
    assert result == pytest.approx(expected, abs=1e-6)


# ────────────────────────── Eq. 7 L2 Post-norm (POST1–POST2) ──────────────────────────
def test_eq7_post1_unit_norm() -> None:
    """Paper Eq. 7 | Case POST1 | Result has ‖·‖₂ = 1 ± 1e-6."""
    result = l2_normalize([3.0, 4.0])
    norm = sum(x * x for x in result) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_eq7_post2_zero_raises() -> None:
    """Paper Eq. 7 | Case POST2 | Zero input raises typed error."""
    with pytest.raises(FusionNormalizationError):
        l2_normalize([0.0, 0.0])
