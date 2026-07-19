"""Cross-cutting health / readiness DTOs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthStatus(BaseModel):
    """Component-level readiness returned by ``*.health_check()``."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    component: str
    ready: bool
    status: Literal["ok", "degraded", "failed"]
    details: dict[str, str] = Field(default_factory=dict)
