"""Stage 3 (Scoring Decision) DTOs — architecture.md §4.2, paper §III.D."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DecisionEnum(str, Enum):
    """Three-way decision outcome (paper §III.D Table rows)."""

    HEALTHY = "Healthy"
    DISEASE = "Disease"
    INCONCLUSIVE = "Inconclusive"


# Decision-path reason codes. The field is historically named
# ``inconclusive_reason`` but is populated for the Disease path as well
# (``"disease_path"``) to preserve end-to-end traceability of which
# Eq. 10 branch produced the outcome.
InconclusiveReason = Literal[
    "low_margin",
    "scoring_margin",
    "no_evidence",
    "disease_path",
    "fallback",
]


class DiagnosisResult(BaseModel):
    """Output of Stage 3 — frozen."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: DecisionEnum
    disease_class: str | None = None
    score_healthy: float
    score_disease: float
    retrieval_margin: float
    inconclusive_reason: InconclusiveReason | None = None
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    compute_time_ms: float = Field(ge=0.0, default=0.0)
    # M2.4 end-to-end orchestrator telemetry. All optional; unused fields
    # stay empty / zero so unit tests that construct DiagnosisResult with
    # only the mandatory fields continue to pass.
    stage_latencies_ms: dict[str, float] = Field(default_factory=dict)
    retrieval_count: int = Field(ge=0, default=0)
    verification_iterations: int = Field(ge=0, default=0)
    caption_empty: bool = False
    pareidolia_correction_count: int = Field(ge=0, default=0)
