"""DatasetAccessGuard runtime tests — paper §IV.A A8, Skill S6 Pillar 2.

4 cases verifying TRAIN / SELECT / EVAL / INFERENCE phase behaviour.
"""

from __future__ import annotations

import pytest

from fishdx.data.isolation import DatasetAccessGuard, Phase, get_guard
from fishdx.errors import DataIsolationViolation


@pytest.fixture(autouse=True)
def _reset_guard() -> None:
    DatasetAccessGuard()._reset_for_tests()


def test_train_phase_blocks_held_out() -> None:
    """ACC1 | TRAIN phase + HELD_OUT access raises."""
    g = get_guard()
    g.register_held_out("D2")
    g.set_phase(Phase.TRAIN)
    with pytest.raises(DataIsolationViolation):
        g.check("D2")


def test_select_phase_blocks_held_out() -> None:
    """ACC2 | SELECT phase + HELD_OUT access raises (paper λ-grid context)."""
    g = get_guard()
    g.register_held_out("D2")
    g.set_phase(Phase.SELECT)
    with pytest.raises(DataIsolationViolation):
        g.check("D2")


def test_eval_phase_allows_held_out() -> None:
    """ACC3 | EVAL phase allows HELD_OUT (cross-dataset benchmarking)."""
    g = get_guard()
    g.register_held_out("D2")
    g.set_phase(Phase.EVAL)
    g.check("D2")  # must not raise


def test_inference_phase_allows_held_out() -> None:
    """ACC4 | INFERENCE phase allows HELD_OUT (production serving)."""
    g = get_guard()
    g.register_held_out("D2")
    g.set_phase(Phase.INFERENCE)
    g.check("D2")  # must not raise
