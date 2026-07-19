"""fishdx exception hierarchy — architecture.md §2.3 R2.

Every FishdxError carries a structured ``.context: dict[str, Any]`` for
observability (stage, cause, trace_id, config_hash) per architecture.md §2.3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final


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


class ConfigNotFoundError(ConfigFileError):
    """A YAML config file (main or any ``extends:`` parent) does not exist.

    Subclass of ``ConfigFileError`` for backward-compatible exception
    handling — code that catches ``ConfigFileError`` will continue to
    catch ``ConfigNotFoundError``.

    Attributes
    ----------
    missing_path : Path
        The specific file that was not found.
    extends_chain : list[Path]
        Full inheritance chain ending at the missing file. Length 1
        means the main config itself is missing; length ≥ 2 means a
        parent in the ``extends:`` chain is missing.
    """

    def __init__(
        self,
        message: str = "",
        *,
        missing_path: Path,
        extends_chain: list[Path],
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message or f"Config file not found: {missing_path}",
                         context=context)
        # Final = immutable post-construction (typing-level convention).
        self.missing_path: Final[Path] = missing_path
        self.extends_chain: Final[list[Path]] = list(extends_chain)

    def __str__(self) -> str:
        if len(self.extends_chain) <= 1:
            return (
                f"Config file not found: {self.missing_path}\n"
                f"\n"
                f"Hint: Check the file path and ensure the YAML file exists."
            )
        chain_repr = " → ".join(str(p) for p in self.extends_chain[:-1])
        chain_repr += f" → {self.extends_chain[-1]} ← not found"
        parent_in_chain = self.extends_chain[-2]
        return (
            f"Config inheritance chain broken:\n"
            f"  {chain_repr}\n"
            f"\n"
            f"Hint: The 'extends:' line in {parent_in_chain} points to a "
            f"non-existent file.\n"
            f"Verify the path or remove the 'extends:' directive."
        )


class ConfigCircularExtendsError(ConfigError):
    """Cyclic inheritance detected in the ``extends:`` chain.

    Each YAML file's ``extends:`` must form a non-cyclic chain. If file
    A extends B and B extends A (or any longer cycle), this exception is
    raised the moment the cycle is detected by the visited-paths set.

    Attributes
    ----------
    cycle_chain : list[Path]
        Full chain leading into the cycle. The last element is the path
        that was already visited (the cycle re-entry point).
    """

    def __init__(
        self,
        message: str = "",
        *,
        cycle_chain: list[Path],
        context: dict[str, Any] | None = None,
    ) -> None:
        chain_repr = " → ".join(str(p) for p in cycle_chain)
        super().__init__(
            message or f"Circular extends detected: {chain_repr}",
            context=context,
        )
        self.cycle_chain: Final[list[Path]] = list(cycle_chain)

    def __str__(self) -> str:
        chain_repr = " → ".join(str(p) for p in self.cycle_chain)
        return (
            f"Circular extends detected in config inheritance chain:\n"
            f"  {chain_repr}\n"
            f"\n"
            f"Hint: Each YAML file's 'extends:' must form a non-cyclic "
            f"chain.\n"
            f"Review the 'extends:' line in each file in the cycle and "
            f"break the loop."
        )


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
    "ConfigNotFoundError",
    "ConfigCircularExtendsError",
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
