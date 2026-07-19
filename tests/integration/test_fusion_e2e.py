"""Fusion end-to-end integration tests — paper §III.C, Eq. 5–7.

Covers docs/m0/test-matrix.md case IDs: E2E1–E2E3 (3 cases).

These tests exercise the full ``normalize → fuse → normalize`` pipeline and
check invariants (unit-norm, degeneracy, snapshot exactness). All cases are
RED: ``fuse_and_normalize`` raises ``NotImplementedError`` until M2.

References
----------
Paper §III.C; Skill S3 §2.1.
"""

from __future__ import annotations

import random

import pytest

from fishdx.errors import FusionNormalizationError
from fishdx.retrieval.fusion import fuse_and_normalize

_EMBEDDING_DIM = 512


def _seeded_vector(seed: int, dim: int = _EMBEDDING_DIM) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


@pytest.mark.integration
def test_e2e1_unit_norm_invariant() -> None:
    """Paper Eq. 5–7 | Case E2E1 | Random seeded inputs.

    Expected: ``‖E_final‖₂ == 1.0 ± 1e-6`` regardless of input magnitude
    (unit-norm invariant under the normalize-fuse-normalize pipeline).

    RED: ``fuse_and_normalize`` raises ``NotImplementedError`` until M2.
    """
    e_visual = _seeded_vector(42)
    e_caption = _seeded_vector(43)
    result = fuse_and_normalize(e_visual, e_caption, lambda_weight=0.7)
    norm = sum(x * x for x in result) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)


@pytest.mark.integration
def test_e2e2_lambda_1_zero_image() -> None:
    """Paper Eq. 5–7 | Case E2E2 | λ=1 with non-zero text and zero image.

    At λ=1, only the (empty) visual path contributes after normalization.
    With zero image, Eq. 5 must raise ``FusionNormalizationError`` —
    contract: invariant violations surface typed, not silent NaN.

    RED: ``fuse_and_normalize`` raises ``NotImplementedError`` until M2.
    """
    e_visual = [0.0] * _EMBEDDING_DIM
    e_caption = _seeded_vector(44)
    with pytest.raises(FusionNormalizationError):
        fuse_and_normalize(e_visual, e_caption, lambda_weight=1.0)


@pytest.mark.integration
def test_e2e3_snapshot_exact() -> None:
    """Paper Eq. 5–7 | Case E2E3 | λ=0.7 fixed-input snapshot.

    Deterministic reproducibility: identical input bytes → identical output
    bytes across runs. Snapshot is a 3-d projection for readability; M2
    replaces with full 512-d fixture.

    RED: ``fuse_and_normalize`` raises ``NotImplementedError`` until M2.
    """
    e_visual = [1.0, 0.0, 0.0]
    e_caption = [0.0, 1.0, 0.0]
    result = fuse_and_normalize(e_visual, e_caption, lambda_weight=0.7)
    # Exact closed-form: [0.7, 0.3, 0] / sqrt(0.58) = [0.919145, 0.393919, 0]
    # (original M0 Phase 4 snapshot had a rounded numerator; corrected here.)
    expected = [0.919145, 0.393919, 0.0]
    assert result == pytest.approx(expected, abs=1e-5)
