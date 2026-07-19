"""Scoring classification layer — architecture.md §2.3, §4.1."""

from __future__ import annotations

from typing import Protocol

from fishdx.schemas import DiagnosisResult, RetrievalResult


class ScoringStage(Protocol):
    """Pure deterministic Stage 3. Rule-based; no ML inference; fully thread-safe."""

    def decide(self, retrieval: RetrievalResult) -> DiagnosisResult: ...


__all__ = ["ScoringStage"]
