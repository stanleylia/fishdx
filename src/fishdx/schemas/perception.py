"""Stage 1 (Visual Perception) DTOs — architecture.md §4.2, paper §III.B."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DetectedObject(BaseModel):
    """Florence-2 open-vocabulary detection item — an element of L."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    label: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float = Field(ge=0.0, le=1.0)


class PareidoliaEvent(BaseModel):
    """A single Eq. 1–4 correction event (ADR-0003 dual-path).

    Records hard-path keyword membership, soft-path CLIP similarity, and
    the merged probability that drove the remap decision.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    label_raw: str
    label_corrected: str
    p_hard: float = Field(ge=0.0, le=1.0)
    p_soft: float = Field(ge=0.0, le=1.0)
    merged_p: float = Field(ge=0.0, le=1.0)
    path: str


class PerceptionResult(BaseModel):
    """Output of Stage 1 — frozen, passed downstream by value."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    image_path: Path
    caption_raw: str
    caption_corrected: str
    objects_raw: list[DetectedObject]
    objects_corrected: list[DetectedObject]
    pareidolia_corrections: list[PareidoliaEvent]
    warnings: list[str] = Field(default_factory=list)
    compute_time_ms: float = Field(ge=0.0)
