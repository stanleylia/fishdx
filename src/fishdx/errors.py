"""fishdx exception hierarchy — architecture.md §2.3 R2.

Every FishdxError carries a structured ``.context: dict[str, Any]`` for
observability (stage, cause, trace_id, config_hash) per architecture.md §2.3.
"""

from __future__ import annotations

from typing import Any


class FishdxError(Exception):
    """Abstract base for all fishdx-raised errors."""

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = dict(context) if context else {}


class InputValidationError(FishdxError):
    """Pre-Stage 1 rejection; invalid image file / missing path / corrupt bytes."""


class PipelineError(FishdxError):
    """Orchestrator-level wrapper for hard stage failures."""


# ───────────────────────── ConfigError family ─────────────────────────
class ConfigError(FishdxError):
    """Config validation / loading / freezing failure."""


class ConfigSchemaError(ConfigError):
    """Pydantic ``ValidationError`` wrapper — schema invariants violated."""


class ConfigFrozenMutationError(ConfigError):
    """Attempted attribute assignment on a frozen ``AppConfig``."""


class ConfigMissingFieldError(ConfigError):
    """Required field absent from ``configs/default.yaml``."""


class ConfigFileError(ConfigError):
    """YAML parse failure / file not found / unsafe loader invoked."""


# ─────────────────────── ReproducibilityError family ───────────────────────
class ReproducibilityError(FishdxError):
    """Seed / determinism invariant violated at runtime."""


class SeedNotSetError(ReproducibilityError):
    """Global seeds not configured before first ``Pipeline.diagnose()`` call."""


class NonDeterministicAlgorithmError(ReproducibilityError):
    """CUDNN determinism flag regression or non-deterministic op in hot path."""


class RevisionPinError(ReproducibilityError):
    """Model ``revision`` is not a 40-char SHA (ADR-0004 M1 enforcement)."""


class ConfigHashDriftError(ReproducibilityError):
    """Frozen config hash differs from previous run without explicit re-freeze."""


# ───────────────────────── StageError family ─────────────────────────
class StageError(FishdxError):
    """Hard failure within a pipeline stage."""


class PerceptionError(StageError):
    """Stage 1 (Visual Perception) hard failure."""


class Florence2InferenceError(PerceptionError):
    """Florence-2 OOM, NaN logits, timeout, or decode failure."""


class PareidoliaCorrectionError(PerceptionError):
    """Eq. 1–4 correction pipeline failure."""


class RetrievalError(StageError):
    """Stage 2 (Knowledge Retrieval) hard failure."""


class ClipEncodingError(RetrievalError):
    """CLIP text/image encoding failure — NaN, OOM, shape mismatch."""


class FusionNormalizationError(RetrievalError):
    """Eq. 5 / Eq. 7 L2-normalization failure — zero norm, NaN, dtype error."""


class ChromaStoreError(RetrievalError):
    """ChromaDB index missing, corrupt, or query failure."""


class VerificationError(RetrievalError):
    """Algorithm 1 grounding loop failure."""


class ScoringError(StageError):
    """Stage 3 (Scoring Decision) invariant violation — Eq. 8–11."""


# ───────────────────────── Data-layer errors (M1) ─────────────────────────
class DataError(FishdxError):
    """Data-layer failure (M1 — dataset, split, KB ingestion)."""


class DatasetNotFoundError(DataError):
    """Dataset root path missing or unreadable (precondition failure)."""


class DatasetSizeMismatchError(ReproducibilityError):
    """Observed dataset size differs from ``DatasetSpec.expected_sample_count``.

    Supply-chain drift detector per Skill S6 Pillar 1.
    """


class DataIsolationViolation(ReproducibilityError):  # noqa: N818  # user-mandated name per M1 Q2
    """Attempted to access a HELD_OUT dataset during TRAIN / SELECT phase.

    Paper §IV.A A8; Skill S6 Pillar 2. Runtime-enforced by
    ``fishdx.data.isolation.DatasetAccessGuard``.
    """


class KBIngestError(DataError):
    """Knowledge-base ingestion failure (parse, embed, or persist)."""


__all__ = [
    "FishdxError",
    "InputValidationError",
    "PipelineError",
    "ConfigError",
    "ConfigSchemaError",
    "ConfigFrozenMutationError",
    "ConfigMissingFieldError",
    "ConfigFileError",
    "ReproducibilityError",
    "SeedNotSetError",
    "NonDeterministicAlgorithmError",
    "RevisionPinError",
    "ConfigHashDriftError",
    "StageError",
    "PerceptionError",
    "Florence2InferenceError",
    "PareidoliaCorrectionError",
    "RetrievalError",
    "ClipEncodingError",
    "FusionNormalizationError",
    "ChromaStoreError",
    "VerificationError",
    "ScoringError",
    "DataError",
    "DatasetNotFoundError",
    "DatasetSizeMismatchError",
    "DataIsolationViolation",
    "KBIngestError",
]
