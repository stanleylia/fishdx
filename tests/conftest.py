"""Test factories for fishdx config models — Phase 4 reviewer-response.

Provides three importable factories for hermetic test fixtures:

    make_test_config(**overrides)          -> AppConfig
    make_test_negation_config(**overrides) -> NegationConfig
    make_test_scoring_config(**overrides)  -> ScoringConfig

All three reuse ``fishdx.config.deep_merge`` (Phase 2 SSOT) for nested
override semantics — there is exactly **one** merge implementation
across production (``load_config`` / ``extends:``) and tests. If
``deep_merge`` ever needs to change, every consumer (production layered
configs, future A/B overlay configs, and these test factories) inherits
the change atomically.

Importable usage (factories are plain functions, not pytest fixtures):

    from tests.conftest import make_test_config
    cfg = make_test_config(retrieval={"similarity_cutoff": 0.3})

DEFAULTS_REGISTRY classification
================================

Each placeholder value in ``_DEFAULTS`` is tagged in its inline comment
with one of three categories — the categories drive maintenance policy:

    [PAPER]
        Value mirrors paper Methods Table 2 / Eq. 1–11. Test fixtures
        inherit paper-faithful defaults; modifying these silently
        breaks the paper-↔-implementation correspondence. Coordinate
        any change with manuscript-side review and update the paper-
        Table-2 alignment document accordingly.

    [VALIDATOR]
        Value chosen specifically to satisfy a Pydantic validator
        (e.g. 40-char SHA regex on Florence2Config.revision,
        min_length=1 lists on NegationConfig.{cn,en}_patterns,
        cardinality invariants |K_s|=10 / |K_b|=7 on PareidoliaConfig).
        NOT a real production value; only present so the factory's
        no-args call yields a fully-validated AppConfig.

    [PLACEHOLDER]
        Arbitrary marker text irrelevant to test logic (e.g. project
        name "fishdx-test", Pydantic-irrelevant string fields).
        Override freely without paper-side coordination.

Maintenance rule
----------------
Adding a new field to AppConfig (or any sub-config) MUST be reflected
in ``_DEFAULTS``; otherwise ``make_test_config()`` with no overrides
will start raising ``ConfigMissingFieldError`` and break test_F1.
"""

from __future__ import annotations

import copy
from typing import Any

from fishdx.config import (
    AppConfig,
    NegationConfig,
    ScoringConfig,
    deep_merge,
)

# ─── DEFAULTS_REGISTRY ──────────────────────────────────────────────────
# Complete default raw-dict mirror of configs/default.yaml. Sub-config
# fields use list[str] / dict for direct deep_merge compatibility;
# Pydantic auto-converts to tuple where the schema demands.
_DEFAULTS: dict[str, Any] = {
    "meta": {
        "project": "fishdx-test",                  # [PLACEHOLDER]
        "version": "0.0.0-test",                   # [PLACEHOLDER]
        "seed": 42,                                # [PAPER] Table 2
        "paper_ref": "test-fixture",               # [PLACEHOLDER]
    },
    "florence2": {
        "model_id": "microsoft/Florence-2-base",   # [PAPER] Table 2
        "revision": "0" * 40,                      # [VALIDATOR] 40-char SHA regex
        "parameters_billion": 0.23,                # [PAPER] Table 2
        "num_beams": 3,                            # [PAPER] Table 2
        "max_new_tokens": 1024,                    # [PAPER] Table 2
        "do_sample": False,                        # [PAPER] hard-rule deterministic
        "trust_remote_code": True,                 # [PAPER] ADR-0004
        "precision": "fp16",                       # [PLACEHOLDER]
        "device": "cuda:0",                        # [PLACEHOLDER]
    },
    "pareidolia": {
        # [PAPER] Eq. 1: |K_s| = 10 — VALIDATOR enforces cardinality
        "structural_keywords": [
            "net", "fence", "mesh", "cage", "grid",
            "wire", "pen", "enclosure", "lattice", "netting",
        ],
        # [PAPER] Eq. 1: |K_b| = 7 — VALIDATOR enforces cardinality
        "biological_keywords": [
            "face", "person", "animal", "human", "head", "eye", "body",
        ],
        "clip_threshold_tau": 0.75,                # [PAPER] Table 2 (τ)
        "remap_label": "net_damage",               # [PAPER] Eq. 4
    },
    "scene_classification": {
        "rag_gate_threshold": 0.10,                # [PLACEHOLDER]
        "clip_override": 0.24,                     # [PLACEHOLDER]
    },
    "clip": {
        "architecture": "ViT-B-32",                # [PAPER] Table 2
        "pretrained": "laion2b_s34b_b79k",         # [PAPER] Table 2
        "embedding_dim": 512,                      # [PAPER] Table 2
        "text_normalize": "l2",                    # [PAPER] Eq. 5
        "image_normalize": "l2",                   # [PAPER] Eq. 5
        "precision": "fp16",                       # [PLACEHOLDER]
        # [PAPER] §V.B Methods (7 zero-shot templates)
        "zero_shot_templates": [
            "a photo of a {} fish",
            "an image showing {}",
            "a fish with {}",
            "aquaculture photo of {}",
            "underwater image of a fish with {}",
            "a diseased fish suffering from {}",
            "a {} infected fish",
        ],
    },
    "fusion": {
        "lambda_star": 0.7,                        # [PAPER] Table 2 (λ*)
        "normalize_pre": True,                     # [PAPER] Eq. 5
        "normalize_post": True,                    # [PAPER] Eq. 7
        "compute_dtype": "fp32",                   # [PLACEHOLDER]
    },
    "retrieval": {
        "index_type": "hnsw",                      # [PAPER] Table 2
        "metric": "cosine",                        # [PAPER] Table 2
        "top_k": 5,                                # [PAPER] Table 2
        "similarity_cutoff": 0.5,                  # [PAPER] Table 2 (paper-primary)
        "hnsw_ef_construction": 200,               # [PLACEHOLDER]
        "hnsw_ef_search": 100,                     # [PLACEHOLDER]
        "hnsw_M": 16,                              # [PLACEHOLDER]
        "hnsw_num_threads": 1,                     # [PAPER] ADR-0002 (determinism)
        "collection": "image_gallery",             # [PAPER] ADR-0012d
    },
    "verification": {
        "grounding_threshold": 0.5,                # [PAPER] Algorithm 1 (θ)
        "penalty_factor": 0.5,                     # [PAPER] Algorithm 1
        "min_score": 0.3,                          # [PAPER] Algorithm 1
        "epsilon": 0.01,                           # [PAPER] Algorithm 1 (ε)
        "max_iterations": 2,                       # [PAPER] Algorithm 1
    },
    "negation": {
        "window_chars": 50,                        # [PAPER] Table 2
        "cn_patterns": ["不"],                     # [VALIDATOR] min_length=1
        "en_patterns": ["not"],                    # [VALIDATOR] min_length=1
    },
    "scoring": {
        "disease_weights": {
            "confirmed": 3,                        # [PAPER] Eq. 9 (w_d)
            "suspected": 2,                        # [PAPER] Eq. 9 (w_d)
            "mentioned": 1,                        # [PAPER] Eq. 9 (w_d)
        },
        "healthy_weights": {
            "explicit": 2,                         # [PAPER] Eq. 8 (w_h)
            "negation": 1,                         # [PAPER] Eq. 8 (w_n)
        },
        "keyword_dict": "fishdx_kb",               # [PAPER] ADR-0012d
    },
    "decision": {
        "healthy_threshold_Th": 2,                 # [PAPER] Table 2 (T_h)
        "inconclusive_margin_m": 1,                # [PAPER] Table 2 (m)
    },
    "margin": {
        "retrieval_margin_theta": 0.02,            # [PAPER] Eq. 11 (θ_margin)
    },
    "kb": {
        "documents_count": 8,                      # [PAPER] §III.E (7 D1 + 1 EUS)
        "root_path": "./data/kb/",                 # [PLACEHOLDER]
        "chromadb_persist": "./data/kb/chromadb",  # [PLACEHOLDER]
        "collection_name": "fishdx_kb",            # [PLACEHOLDER]
        "source_reference": "test-fixture",        # [PLACEHOLDER]
    },
    "data_split": {
        "train_test_ratio": [0.715, 0.285],        # [PAPER] Table 2
        "stratify": True,                          # [PAPER] Methods §IV.A
        "seed": 42,                                # [PAPER] Table 2
    },
    "statistics": {
        "bootstrap_resamples": 1000,               # [PAPER] Table 2
        "alpha": 0.05,                             # [PAPER] Table 2
        "ci_method": "wilson",                     # [PAPER] §Statistical analysis
        "semantic_sca_model": "sentence-transformers/all-MiniLM-L6-v2",  # [PAPER] Eq. 12
        "mcnemar_exact_threshold": 25,             # [PLACEHOLDER]
    },
    "performance": {
        "single_image_p95_seconds": 2.0,           # [PAPER] §IV (engineering SLA)
        "vram_peak_gb": 18.0,                      # [PLACEHOLDER]
        "chromadb_build_p95_seconds": 90.0,        # [PLACEHOLDER]
        "unit_test_p95_seconds": 60.0,             # [PLACEHOLDER]
    },
    "logging": {
        "level": "INFO",                           # [PLACEHOLDER]
        "format": "structured-json",               # [PLACEHOLDER]
        "output_dir": "./logs/",                   # [PLACEHOLDER]
        "include_git_commit": True,                # [PLACEHOLDER]
    },
}


def make_test_config(**overrides: Any) -> AppConfig:
    """Build a complete, valid ``AppConfig`` with optional deep overrides.

    Reuses ``fishdx.config.deep_merge`` for nested-key override
    semantics — the same algorithm used by ``load_config`` for
    ``extends:`` chains. Validation happens once, after the merge,
    matching the production loader's invariant.

    Examples
    --------
    >>> cfg = make_test_config()
    >>> cfg.fusion.lambda_star
    0.7
    >>> cfg = make_test_config(retrieval={"similarity_cutoff": 0.3})
    >>> cfg.retrieval.similarity_cutoff
    0.3
    >>> cfg.retrieval.top_k        # sibling preserved by deep_merge
    5
    """
    defaults_dict = copy.deepcopy(_DEFAULTS)
    merged = deep_merge(defaults_dict, dict(overrides))
    return AppConfig.model_validate(merged)


def make_test_negation_config(**overrides: Any) -> NegationConfig:
    """Build a valid ``NegationConfig`` with optional deep overrides.

    Defaults (``cn_patterns=["不"]``, ``en_patterns=["not"]``) satisfy
    the Pydantic ``min_length=1`` validators on both fields. Override
    either field to inject domain-specific patterns for a given test.
    """
    defaults_dict = copy.deepcopy(_DEFAULTS["negation"])
    merged = deep_merge(defaults_dict, dict(overrides))
    return NegationConfig.model_validate(merged)


def make_test_scoring_config(**overrides: Any) -> ScoringConfig:
    """Build a valid ``ScoringConfig`` with optional deep overrides.

    Defaults pre-fill all required nested fields (``disease_weights``,
    ``healthy_weights``, ``keyword_dict``) so callers can override only
    the leaves they care about; e.g.
    ``make_test_scoring_config(disease_weights={"confirmed": 5})``
    overrides one leaf and preserves the other two tier weights plus
    the entire ``healthy_weights`` block.
    """
    defaults_dict = copy.deepcopy(_DEFAULTS["scoring"])
    merged = deep_merge(defaults_dict, dict(overrides))
    return ScoringConfig.model_validate(merged)


__all__ = [
    "make_test_config",
    "make_test_negation_config",
    "make_test_scoring_config",
]
