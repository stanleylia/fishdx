"""Stage 1 (Visual Perception) skeleton — paper §III.B.

Implementation deferred to M2 per architecture.md §5.
"""

from __future__ import annotations

from pathlib import Path

from fishdx.config import AppConfig
from fishdx.schemas import HealthStatus, PerceptionResult


class PerceptionStageImpl:
    """Florence-2 caption/detection + Eq. 1–4 pareidolia correction."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def process(self, image: Path) -> PerceptionResult:
        raise NotImplementedError("PerceptionStageImpl.process deferred to M2")

    def warmup(self) -> None:
        raise NotImplementedError("PerceptionStageImpl.warmup deferred to M2")

    def health_check(self) -> HealthStatus:
        raise NotImplementedError("PerceptionStageImpl.health_check deferred to M2")


__all__ = ["PerceptionStageImpl"]
