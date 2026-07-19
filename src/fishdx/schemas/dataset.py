"""Dataset / split / KB DTOs — M1 Phase 1 architecture §3-§6."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DatasetRole(str, Enum):
    """Usage-constraint enforcement — paper §IV.A, Skill S6 Pillar 2."""

    TRAIN_ELIGIBLE = "train_eligible"
    TEST_ONLY = "test_only"
    HELD_OUT = "held_out"
    KB_SOURCE = "kb_source"


class DatasetSpec(BaseModel):
    """Frozen registry entry for one dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    human_name: str
    root_path: Path
    role: DatasetRole
    expected_sample_count: int = Field(ge=0)
    expected_classes: int = Field(ge=0)
    license: str
    paper_citation: str
    available: bool = True  # flipped to False when precondition sweep detects missing root


class ImageRecord(BaseModel):
    """One labelled image from any dataset — frozen."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: str
    image_path: Path
    label: str
    dataset_id: str
    role: DatasetRole
    split: Literal["train", "test", "val"] | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class SplitManifest(BaseModel):
    """Persisted split artefact — pins fold-id SHA for cross-run comparison."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    seed: int = Field(ge=0)
    train_ratio: float = Field(ge=0.0, le=1.0)
    stratify: bool
    sha256_ids_train: str
    sha256_ids_test: str
    n_train: int = Field(ge=0)
    n_test: int = Field(ge=0)
    per_class_counts: dict[str, tuple[int, int]]


# ── KB DTOs ────────────────────────────────────────────────────
class KBDocument(BaseModel):
    """One knowledge-base disease document — frozen.

    Under ADR-0007 (KB source substitution), each doc carries a
    keyword-tier vocabulary plus negation patterns, extracted by
    :func:`fishdx.kb.builder.parse_markdown_docs` from the 5/6-section
    Markdown body.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str
    disease_class: str
    title: str
    text: str
    source_reference: str
    source_url: str = ""
    substitution_date: str | None = None
    rewrite_date: str | None = None
    # ADR-0007: int (single ADR); ADR-0012a re-write: list[str] (multiple
    # ADRs in chronological order). Schema accepts either; parser
    # normalises to list[str].
    substituted_under_adr: list[str] = Field(default_factory=list)
    # Tier → keyword list; tier names: "confirmed" / "suspected" / "mentioned".
    clinical_keywords: dict[str, list[str]] = Field(default_factory=dict)
    # Populated on HLT only (Eq. 8 w^(h) channel).
    healthy_keywords: list[str] = Field(default_factory=list)
    # Negation patterns (50-char window per Skill S4 §3.1).
    negation_patterns_cn: list[str] = Field(default_factory=list)
    negation_patterns_en: list[str] = Field(default_factory=list)
    structural_keywords: list[str] = Field(default_factory=list)
    severity_tags: list[str] = Field(default_factory=list)


class IngestReceipt(BaseModel):
    """Deterministic receipt produced by :meth:`KBBuilder.ingest`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    collection_name: str
    n_documents: int = Field(ge=0)
    ingestion_seed: int = Field(ge=0)
    sha256_content: str
    duration_seconds: float = Field(ge=0.0)
    fishdx_version: str


__all__ = [
    "DatasetRole",
    "DatasetSpec",
    "ImageRecord",
    "IngestReceipt",
    "KBDocument",
    "SplitManifest",
]
