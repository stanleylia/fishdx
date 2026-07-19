"""Frozen Pydantic DTOs crossing layer boundaries (architecture.md §2.3 R1).

All DTOs are ``frozen=True, strict=True, extra="forbid"`` per S6 §3 and
architecture.md §4.2.
"""

from __future__ import annotations

from fishdx.schemas.dataset import (
    DatasetRole,
    DatasetSpec,
    ImageRecord,
    IngestReceipt,
    KBDocument,
    SplitManifest,
)
from fishdx.schemas.diagnosis import DecisionEnum, DiagnosisResult, InconclusiveReason
from fishdx.schemas.health import HealthStatus
from fishdx.schemas.perception import DetectedObject, PareidoliaEvent, PerceptionResult
from fishdx.schemas.retrieval import RetrievalResult, RetrievedDoc, VerificationTelemetry

__all__ = [
    "DatasetRole",
    "DatasetSpec",
    "DecisionEnum",
    "DetectedObject",
    "DiagnosisResult",
    "HealthStatus",
    "ImageRecord",
    "InconclusiveReason",
    "IngestReceipt",
    "KBDocument",
    "PareidoliaEvent",
    "PerceptionResult",
    "RetrievalResult",
    "RetrievedDoc",
    "SplitManifest",
    "VerificationTelemetry",
]
