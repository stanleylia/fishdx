"""Retrieval-margin safety valve tests — paper §III.D Eq. 11.

Covers docs/m0/test-matrix.md case IDs: MAR1–MAR4 (4 cases).

All cases are RED: ``apply_retrieval_margin`` raises ``NotImplementedError``
until M2. Case MAR3 is the highest-value operator-precision test in the
entire suite — strict ``<`` at the ``θ_margin`` boundary protects the 0.2 %
of D2 samples whose margin equals θ exactly.

References
----------
Paper §III.D, ADR-0001, Skill S4 §2.4.
"""

from __future__ import annotations

import pytest

from fishdx.config import MarginConfig
from fishdx.scoring.decision import apply_retrieval_margin

_MARGIN = MarginConfig(retrieval_margin_theta=0.02)


@pytest.mark.parametrize(
    ("sim1", "sim2", "expected"),
    [
        pytest.param(0.90, 0.50, False, id="MAR1-far-above-theta"),
        pytest.param(0.90, 0.89, True, id="MAR2-below-theta-triggers"),
        pytest.param(0.90, 0.88, False, id="MAR3-strict-lt-boundary-no-trigger"),
        pytest.param(0.90, 0.879, False, id="MAR4-just-above-theta"),
    ],
)
def test_eq11_retrieval_margin(sim1: float, sim2: float, expected: bool) -> None:
    """Paper Eq. 11 | Case MAR{1-4}.

    Expected: ``apply_retrieval_margin`` returns ``True`` iff
    ``sim1 − sim2 < θ_margin`` (strict). Case MAR3 probes the critical
    equality boundary (``0.02 < 0.02`` is FALSE → no trigger).

    RED: ``apply_retrieval_margin`` raises ``NotImplementedError`` until M2.
    """
    assert apply_retrieval_margin(sim1, sim2, _MARGIN) is expected
