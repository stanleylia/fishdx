"""Runtime data-access guard — paper §IV.A A8; Skill S6 Pillar 2.

``DatasetAccessGuard`` is a module-level singleton with a global phase
(TRAIN / SELECT / EVAL / INFERENCE). Every ``DatasetLoader.__iter__`` MUST
call :func:`DatasetAccessGuard.check` before yielding records, so that a
HELD_OUT dataset accessed in a TRAIN or SELECT phase raises
:class:`DataIsolationViolation` at runtime — not at static-analysis time.

Rationale
---------
Typed invariants catch static misuse only. Runtime misuse (experiment
script bug, accidental loader swap) requires runtime enforcement. Paper
reproducibility — notably the cross-dataset generalization claim on D2 —
depends on never touching D2 during parameter selection.
"""

from __future__ import annotations

import threading
from enum import Enum

from fishdx.errors import DataIsolationViolation


class Phase(str, Enum):
    """Execution phase — Skill S6 Pillar 2."""

    TRAIN = "train"
    SELECT = "select"
    EVAL = "eval"
    INFERENCE = "inference"


_HELD_OUT_ALLOWED: frozenset[Phase] = frozenset({Phase.EVAL, Phase.INFERENCE})


class DatasetAccessGuard:
    """Thread-safe singleton gating access to HELD_OUT datasets."""

    _instance: DatasetAccessGuard | None = None
    _lock = threading.Lock()

    def __new__(cls) -> DatasetAccessGuard:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._phase = Phase.TRAIN
                    inst._held_out: set[str] = set()
                    cls._instance = inst
        return cls._instance

    _phase: Phase
    _held_out: set[str]

    # ── phase control ─────────────────────────────────────────────
    def set_phase(self, phase: Phase) -> None:
        """Advance / reset the global phase. Call from experiment harness only."""
        with self._lock:
            self._phase = phase

    @property
    def phase(self) -> Phase:
        return self._phase

    # ── registration ─────────────────────────────────────────────
    def register_held_out(self, dataset_id: str) -> None:
        """Register a dataset as HELD_OUT (paper §IV.A A8)."""
        with self._lock:
            self._held_out.add(dataset_id)

    def is_held_out(self, dataset_id: str) -> bool:
        return dataset_id in self._held_out

    # ── runtime check ─────────────────────────────────────────────
    def check(self, dataset_id: str) -> None:
        """Raise :class:`DataIsolationViolation` if held-out access is illegal now."""
        if dataset_id in self._held_out and self._phase not in _HELD_OUT_ALLOWED:
            raise DataIsolationViolation(
                f"Access to HELD_OUT dataset '{dataset_id}' is forbidden in phase "
                f"'{self._phase.value}' (paper §IV.A A8; Skill S6 Pillar 2)",
                context={"dataset_id": dataset_id, "phase": self._phase.value},
            )

    # ── testing utility ──────────────────────────────────────────
    def _reset_for_tests(self) -> None:
        """Clear internal state; used by unit tests to avoid cross-test leakage."""
        with self._lock:
            self._phase = Phase.TRAIN
            self._held_out = set()


def get_guard() -> DatasetAccessGuard:
    """Return the process-wide :class:`DatasetAccessGuard` singleton."""
    return DatasetAccessGuard()


__all__ = ["DatasetAccessGuard", "Phase", "get_guard"]
