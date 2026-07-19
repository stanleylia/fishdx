"""Stage 2 (Knowledge Retrieval) DTOs — architecture.md §4.2, paper §III.C."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RetrievedDoc(BaseModel):
    """One ChromaDB candidate document, optionally verified via Algorithm 1."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    doc_id: str
    text: str
    disease_class: str | None
    similarity: float = Field(ge=-1.0, le=1.0)
    similarity_penalized: float | None = None
    grounded: bool | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class VerificationTelemetry(BaseModel):
    """Per-iteration trace of the Algorithm 1 verification loop."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    iteration: int = Field(ge=0)
    delta: float = Field(ge=0.0)
    converged: bool
    grounded_count: int = Field(ge=0)
    penalized_count: int = Field(ge=0)


class RetrievalResult(BaseModel):
    """Output of Stage 2 — frozen."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    e_final: list[float]
    candidates_raw: list[RetrievedDoc]
    candidates_verified: list[RetrievedDoc]
    verification_converged: bool
    verification_iterations: int = Field(ge=0)
    verification_trace: list[VerificationTelemetry] = Field(default_factory=list)
    margin_top1_top2: float
    warnings: list[str] = Field(default_factory=list)
    compute_time_ms: float = Field(ge=0.0)
