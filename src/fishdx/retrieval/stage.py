"""Stage 2 (Knowledge Retrieval) skeleton — paper §III.C.

Implementation deferred to M2 per architecture.md §5.
"""

from __future__ import annotations

from pathlib import Path

from fishdx.config import AppConfig
from fishdx.schemas import HealthStatus, PerceptionResult, RetrievalResult


class RetrievalStageImpl:
    """CLIP fusion (Eq. 5–7) + ChromaDB HNSW query + Algorithm 1 verification."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def retrieve(self, percept: PerceptionResult, image: Path) -> RetrievalResult:
        raise NotImplementedError("RetrievalStageImpl.retrieve deferred to M2")

    def warmup(self) -> None:
        raise NotImplementedError("RetrievalStageImpl.warmup deferred to M2")

    def health_check(self) -> HealthStatus:
        raise NotImplementedError("RetrievalStageImpl.health_check deferred to M2")


__all__ = ["RetrievalStageImpl"]
