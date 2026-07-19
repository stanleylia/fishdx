"""Knowledge retrieval layer — architecture.md §2.2, §4.1."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from fishdx.schemas import HealthStatus, PerceptionResult, RetrievalResult


class RetrievalStage(Protocol):
    """Pure deterministic Stage 2. Deterministic conditional on
    (config, seed, PerceptionResult, image)."""

    def retrieve(self, percept: PerceptionResult, image: Path) -> RetrievalResult: ...

    def warmup(self) -> None: ...

    def health_check(self) -> HealthStatus: ...


__all__ = ["RetrievalStage"]
