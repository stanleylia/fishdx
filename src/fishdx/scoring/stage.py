"""Stage 3 (Scoring Decision) skeleton — paper §III.D (Eq. 8–11).

Implementation deferred to M2 per architecture.md §5.
"""

from __future__ import annotations

from fishdx.config import AppConfig
from fishdx.schemas import DiagnosisResult, RetrievalResult


class ScoringStageImpl:
    """Rule-based three-way decision with retrieval-margin safety valve."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def decide(self, retrieval: RetrievalResult) -> DiagnosisResult:
        raise NotImplementedError("ScoringStageImpl.decide deferred to M2")


__all__ = ["ScoringStageImpl"]
