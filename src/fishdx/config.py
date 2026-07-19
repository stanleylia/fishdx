"""fishdx configuration schema — architecture.md

``AppConfig`` is the frozen single source of truth loaded from
``configs/default.yaml``. Schema invariants (e.g. λ ∈ [0, 1], keyword set
cardinalities, monotonicity of disease weights) are enforced via Pydantic
validators. SHA256 freeze hash is pinned against
``configs/schema/frozen_config.lock``.

Validation failures are wrapped into ``ConfigSchemaError`` by
``load_config()``; see ``errors.py``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from fishdx.errors import (
    ConfigCircularExtendsError,
    ConfigFileError,
    ConfigMissingFieldError,
    ConfigNotFoundError,
    ConfigSchemaError,
    RevisionPinError,
)

# Pattern A: strict=True blocks legitimate YAML→tuple / YAML→Enum coercion.
# frozen + extra=forbid provide the invariant safety; strict mode is
# over-constraining for config loading.
_FROZEN = ConfigDict(frozen=True, extra="forbid")
_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")

# Paper-mandated cardinalities (Eq. 1 — structural & biological keyword sets).
_KS_CARDINALITY = 10
_KB_CARDINALITY = 7
# Floating-point tolerance for train/test ratio sum invariant.
_RATIO_SUM_TOL = 1e-6


# ─────────────────────────── Subconfigs ───────────────────────────
class MetaConfig(BaseModel):
    model_config = _FROZEN

    project: str
    version: str
    seed: int = Field(ge=0)
    paper_ref: str


class Florence2Config(BaseModel):
    model_config = _FROZEN

    model_id: str
    revision: str
    parameters_billion: float = Field(gt=0.0)
    num_beams: int = Field(ge=1)
    max_new_tokens: int = Field(ge=1)
    do_sample: Literal[False]
    trust_remote_code: bool
    precision: Literal["fp16", "fp32", "bf16"]
    device: str

    @field_validator("revision")
    @classmethod
    def _revision_must_be_sha(cls, v: str) -> str:
        if not _SHA40_RE.fullmatch(v):
            raise RevisionPinError(
                "florence2.revision must be a 40-char commit SHA (ADR-0004 M1)",
                context={"value": v},
            )
        return v


class ClipConfig(BaseModel):
    model_config = _FROZEN

    architecture: str
    pretrained: str
    embedding_dim: int = Field(ge=1)
    text_normalize: Literal["l2"]
    image_normalize: Literal["l2"]
    precision: Literal["fp16", "fp32", "bf16"]
    zero_shot_templates: list[str] = Field(min_length=1)


class PareidoliaConfig(BaseModel):
    model_config = _FROZEN

    structural_keywords: list[str]
    biological_keywords: list[str]
    clip_threshold_tau: float = Field(ge=0.0, le=1.0)
    remap_label: str

    @field_validator("structural_keywords")
    @classmethod
    def _ks_cardinality(cls, v: list[str]) -> list[str]:
        if len(v) != _KS_CARDINALITY:
            raise ValueError(f"|K_s| must equal {_KS_CARDINALITY} (got {len(v)})")
        return v

    @field_validator("biological_keywords")
    @classmethod
    def _kb_cardinality(cls, v: list[str]) -> list[str]:
        if len(v) != _KB_CARDINALITY:
            raise ValueError(f"|K_b| must equal {_KB_CARDINALITY} (got {len(v)})")
        return v


class SceneClassificationConfig(BaseModel):
    model_config = _FROZEN

    rag_gate_threshold: float = Field(ge=0.0, le=1.0)
    clip_override: float = Field(ge=0.0, le=1.0)


class FusionConfig(BaseModel):
    model_config = _FROZEN

    lambda_star: float = Field(ge=0.0, le=1.0)
    normalize_pre: bool
    normalize_post: bool
    compute_dtype: Literal["fp16", "fp32", "bf16"]


class RetrievalConfig(BaseModel):
    model_config = _FROZEN

    index_type: Literal["hnsw"]
    metric: Literal["cosine"]
    top_k: int = Field(ge=1)
    similarity_cutoff: float = Field(ge=-1.0, le=1.0)
    hnsw_ef_construction: int = Field(ge=1)
    hnsw_ef_search: int = Field(ge=1)
    hnsw_M: int = Field(ge=1)  # noqa: N815  # HNSW paper notation (Malkov & Yashunin 2018)
    hnsw_num_threads: int = Field(ge=1)
    # ADR-0012d — Stage 2 retrieval target ChromaCollection name
    # (image gallery per paper §III-E, not text KB). Paired with
    # scoring.keyword_dict which routes Stage 3 to the preserved text KB.
    collection: str = Field(default="image_gallery")


class ReweightingConfig(BaseModel):
    """Retrieval-confidence reweighting (Algorithm 1) — single pass. See ADR-0016."""

    model_config = _FROZEN

    similarity_threshold: float = Field(ge=0.0, le=1.0)  # θ_sim
    margin_threshold: float = Field(ge=0.0, le=1.0)  # θ_margin
    penalty_factor: float = Field(ge=0.0, le=1.0)
    min_score: float = Field(ge=0.0, le=1.0)


class NegationConfig(BaseModel):
    model_config = _FROZEN

    window_chars: int = Field(ge=1)
    cn_patterns: list[str] = Field(min_length=1)
    en_patterns: list[str] = Field(min_length=1)


class DiseaseWeights(BaseModel):
    model_config = _FROZEN

    confirmed: int = Field(ge=0)
    suspected: int = Field(ge=0)
    mentioned: int = Field(ge=0)

    @model_validator(mode="after")
    def _monotonic(self) -> DiseaseWeights:
        if not (self.confirmed >= self.suspected >= self.mentioned):
            raise ValueError("disease_weights must satisfy confirmed >= suspected >= mentioned")
        return self


class HealthyWeights(BaseModel):
    model_config = _FROZEN

    explicit: int = Field(ge=0)
    negation: int = Field(ge=0)

    @model_validator(mode="after")
    def _monotonic(self) -> HealthyWeights:
        if self.explicit < self.negation:
            raise ValueError("healthy_weights must satisfy explicit >= negation")
        return self


class ScoringConfig(BaseModel):
    model_config = _FROZEN

    disease_weights: DiseaseWeights
    healthy_weights: HealthyWeights
    # ADR-0012d — Stage 3 keyword-dictionary ChromaCollection name.
    # Preserves ADR-0007 text KB (8 docs) exclusively for Eq. 8-9
    # keyword scoring; stage 2 retrieval now targets `retrieval.collection`.
    keyword_dict: str = Field(default="fishdx_kb")


class DecisionConfig(BaseModel):
    model_config = _FROZEN

    healthy_threshold_Th: int = Field(ge=0)  # noqa: N815  # T_h per paper Eq. 10
    inconclusive_margin_m: int = Field(ge=0)


class MarginConfig(BaseModel):
    model_config = _FROZEN

    retrieval_margin_theta: float = Field(ge=0.0, le=1.0)


class KBConfig(BaseModel):
    model_config = _FROZEN

    documents_count: int = Field(ge=1)
    root_path: str
    chromadb_persist: str
    collection_name: str
    source_reference: str


class DataSplitConfig(BaseModel):
    model_config = _FROZEN

    train_test_ratio: tuple[float, float]
    stratify: bool
    seed: int = Field(ge=0)

    @field_validator("train_test_ratio")
    @classmethod
    def _ratio_sums_to_one(cls, v: tuple[float, float]) -> tuple[float, float]:
        if abs(sum(v) - 1.0) > _RATIO_SUM_TOL:
            raise ValueError(f"train_test_ratio must sum to 1.0 (got {sum(v)})")
        return v


class StatisticsConfig(BaseModel):
    model_config = _FROZEN

    bootstrap_resamples: int = Field(ge=1)
    alpha: float = Field(gt=0.0, lt=1.0)
    ci_method: Literal["wilson", "normal", "exact"]
    semantic_sca_model: str
    mcnemar_exact_threshold: int = Field(ge=1)


class PerformanceConfig(BaseModel):
    model_config = _FROZEN

    single_image_p95_seconds: float = Field(gt=0.0)
    vram_peak_gb: float = Field(gt=0.0)
    chromadb_build_p95_seconds: float = Field(gt=0.0)
    unit_test_p95_seconds: float = Field(gt=0.0)


class LoggingConfig(BaseModel):
    model_config = _FROZEN

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    format: Literal["structured-json", "plain"]
    output_dir: str
    include_git_commit: bool


# ─────────────────────────── Root ───────────────────────────
class AppConfig(BaseModel):
    """Frozen root config — architecture.md §4.4."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    meta: MetaConfig
    florence2: Florence2Config
    pareidolia: PareidoliaConfig
    scene_classification: SceneClassificationConfig
    clip: ClipConfig
    fusion: FusionConfig
    retrieval: RetrievalConfig
    reweighting: ReweightingConfig
    negation: NegationConfig
    scoring: ScoringConfig
    decision: DecisionConfig
    margin: MarginConfig
    kb: KBConfig
    data_split: DataSplitConfig
    statistics: StatisticsConfig
    performance: PerformanceConfig
    logging: LoggingConfig

    def freeze_hash(self) -> str:
        """SHA256 of the canonical-JSON config dump."""
        payload = self.model_dump(mode="json")
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


# ─────────────────────────── Loader ───────────────────────────
def deep_merge(parent: Any, child: Any) -> Any:
    """Merge two raw config trees (dict / list / scalar).

    Operates on raw dict / list / scalar only. ``AppConfig`` validation
    MUST happen AFTER merge completion — never on intermediate partial
    dicts (踩雷點 1: ``extra='forbid'`` would reject partial sub-configs).

    Merge rules
    -----------
    - ``dict + dict``       : recursive merge (key union; child wins on conflict).
    - ``list + list``       : ``child`` REPLACES ``parent`` (no append; prevents
                              accidental ``zero_shot_templates`` accumulation).
    - ``scalar + scalar``   : ``child`` wins.
    - type mismatch         : ``child`` wins, EXCEPT when ``child is None`` →
                              ``parent`` is preserved (E4 reversed).

    Notes
    -----
    To explicitly clear a parent value, use an empty container (``[]``,
    ``{}``) or the appropriate empty/zero scalar (``0``, ``""``,
    ``false``), **not** ``None``. Future versions may introduce a
    ``__null__`` sentinel string for explicit clearing — this is a
    deliberate non-feature in Phase 2.

    Properties
    ----------
    Associativity holds: ``deep_merge(deep_merge(A, B), C) ==
    deep_merge(A, deep_merge(B, C))``. Proof by structural induction
    over tree depth (per Phase 2 design brief §1.2).
    """
    if isinstance(parent, dict) and isinstance(child, dict):
        return _merge_dicts(parent, child)
    if isinstance(parent, list) and isinstance(child, list):
        # Child fully replaces parent — no append, by design.
        return copy.deepcopy(child)
    # Scalar-on-scalar OR type mismatch.
    if type(parent) is not type(child):  # noqa: E721 — exact-type check intentional
        if child is None:
            # E4 reversed: child=None means "unspecified", keep parent.
            return copy.deepcopy(parent)
        return copy.deepcopy(child)
    return child


def _merge_dicts(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """dict-specific recursive merge — see ``deep_merge`` docstring."""
    result: dict[str, Any] = {}
    for key in set(parent) | set(child):
        if key in parent and key in child:
            result[key] = deep_merge(parent[key], child[key])
        elif key in parent:
            result[key] = copy.deepcopy(parent[key])
        else:
            result[key] = copy.deepcopy(child[key])
    return result


def _validate_root_mapping(raw: Any, p: Path) -> dict[str, Any]:
    """Step 1.5 — coerce empty file to ``{}``; reject non-mapping roots."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigFileError(
            f"Config root must be a YAML mapping, got {type(raw).__name__}",
            context={"path": str(p), "actual_type": type(raw).__name__},
        )
    return raw


def _validate_extends_value(ext: Any, p: Path) -> str:
    """Step 3.5 — extends must be a relative-path string (rejects list / dict / abs path)."""
    if not isinstance(ext, str):
        raise ConfigFileError(
            f"'extends:' must be a string path, got {type(ext).__name__}",
            context={"path": str(p), "extends_value": repr(ext)},
        )
    if Path(ext).is_absolute():
        raise ConfigFileError(
            "'extends:' must be a relative path",
            context={
                "path": str(p),
                "extends_value": ext,
                "rationale": "Absolute paths break portability across hosts.",
            },
        )
    return ext


def _read_yaml(p: Path, chain: list[Path]) -> Any:
    """Step 1 — read YAML or raise ``ConfigNotFoundError`` / ``ConfigFileError``."""
    try:
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ConfigNotFoundError(
            missing_path=p,
            extends_chain=chain + [p],
            context={"path": str(p)},
        ) from e
    except yaml.YAMLError as e:
        raise ConfigFileError(
            f"YAML parse error: {e}", context={"path": str(p)}
        ) from e


def _load_raw_with_extends(
    p: Path, visited: set[Path], chain: list[Path]
) -> dict[str, Any]:
    """Recursively resolve ``extends:`` and return a fully-merged raw dict.

    Does NOT call ``AppConfig.model_validate`` — validation is deferred to
    the outer ``load_config`` so that partial child dicts cannot trigger
    early ``extra='forbid'`` failures (踩雷點 1).
    """
    p_resolved = p.resolve()

    # Step 2 — circular detection (visited entry on the way IN, not OUT).
    if p_resolved in visited:
        raise ConfigCircularExtendsError(
            cycle_chain=chain + [p_resolved],
            context={"path": str(p_resolved)},
        )

    # Step 1 — read YAML
    raw_any = _read_yaml(p_resolved, chain)

    # Step 1.5 — root mapping check
    raw = _validate_root_mapping(raw_any, p_resolved)

    # Mark visited AFTER successful read to keep the chain clean for errors.
    visited.add(p_resolved)
    new_chain = chain + [p_resolved]

    # Step 3 — short-circuit if no extends.
    if "extends" not in raw:
        return raw

    # Step 3.5 — validate extends value type + relativity.
    ext = _validate_extends_value(raw["extends"], p_resolved)

    # Step 4 — resolve parent path relative to child file's directory.
    parent_path = (p_resolved.parent / ext).resolve()

    # Step 5 — recursively load parent as raw dict (no validation yet).
    parent_raw = _load_raw_with_extends(parent_path, visited, new_chain)

    # Step 6 — strip 'extends' key from child before merge (avoid leaking
    # into AppConfig.model_validate where extra='forbid' would reject it).
    child_raw = {k: v for k, v in raw.items() if k != "extends"}

    # Step 7 — deep_merge(parent, child).
    return deep_merge(parent_raw, child_raw)


def load_config(path: Path | str) -> AppConfig:
    """Load + validate a YAML config into a frozen ``AppConfig``.

    Supports single-file configs (no ``extends:``, original behaviour) and
    layered configs via the ``extends:`` directive (Phase 2). The ``extends:``
    value must be a single relative path string; multiple inheritance
    (``extends: [a.yaml, b.yaml]``) is intentionally not supported in this
    implementation. If multi-inheritance is needed in the future, a
    deterministic linearisation order (e.g. C3 MRO) must be defined first.

    Flow
    ----
    Step 0 — Initialise visited-paths set; resolve input path.
    Step 1 — Read YAML.
    Step 1.5 — Coerce empty file to {}; reject non-mapping root.
    Step 2 — Cycle detection via visited set.
    Step 3 — Short-circuit if no ``extends:`` key.
    Step 3.5 — Validate ``extends:`` value (string + relative).
    Step 4 — Resolve parent path relative to child directory.
    Step 5 — Recursively load parent as raw dict (no validation).
    Step 6 — Pop ``extends`` key from child.
    Step 7 — ``deep_merge(parent_raw, child_raw)``.
    Step 8 — One-shot ``AppConfig.model_validate`` on merged dict.

    Raises
    ------
    ConfigNotFoundError
        Main config or any ``extends:`` parent does not exist.
    ConfigCircularExtendsError
        Cyclic ``extends:`` chain detected.
    ConfigFileError
        YAML parse error / non-mapping root / malformed ``extends:`` value.
    ConfigMissingFieldError
        Required field absent from the merged config.
    ConfigSchemaError
        Pydantic validation failure on the merged config.
    RevisionPinError
        Florence-2 revision is not a 40-char SHA (re-raised, never wrapped).
    """
    p = Path(path)
    visited: set[Path] = set()
    chain: list[Path] = []

    merged = _load_raw_with_extends(p, visited, chain)

    # Step 8 — one-shot validation on the fully-merged dict.
    try:
        return AppConfig.model_validate(merged)
    except RevisionPinError:
        raise
    except ValidationError as e:
        missing = [
            ".".join(str(x) for x in err["loc"]) for err in e.errors() if err["type"] == "missing"
        ]
        if missing:
            raise ConfigMissingFieldError(
                f"required fields missing: {missing}",
                context={"path": str(p), "fields": missing},
            ) from e
        raise ConfigSchemaError(str(e), context={"path": str(p)}) from e


__all__ = [
    "AppConfig",
    "ClipConfig",
    "DataSplitConfig",
    "DecisionConfig",
    "DiseaseWeights",
    "Florence2Config",
    "FusionConfig",
    "HealthyWeights",
    "KBConfig",
    "LoggingConfig",
    "MarginConfig",
    "MetaConfig",
    "NegationConfig",
    "PareidoliaConfig",
    "PerformanceConfig",
    "RetrievalConfig",
    "SceneClassificationConfig",
    "ScoringConfig",
    "StatisticsConfig",
    "ReweightingConfig",
    "deep_merge",
    "load_config",
]
