"""Visual perception layer — architecture.md §2.1, §4.1."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from fishdx.schemas import HealthStatus, PerceptionResult


class PerceptionStage(Protocol):
    """Pure deterministic Stage 1.

    Determinism: bit-identical output for fixed (config, seed, image_bytes).
    Thread safety: read-only inference is thread-safe after ``warmup()``.
    Failure modes: ``PerceptionError`` (hard), ``InputValidationError`` (pre-Stage 1),
    soft failures surface on ``PerceptionResult.warnings``.
    """

    def process(self, image: Path) -> PerceptionResult: ...

    def warmup(self) -> None: ...

    def health_check(self) -> HealthStatus: ...


__all__ = ["PerceptionStage"]
