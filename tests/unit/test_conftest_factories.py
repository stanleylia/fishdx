"""Phase 4 — self-tests for tests/conftest.py factories.

Test matrix (per Phase 4 design brief §2):
  T_F1  — make_test_config() with no overrides yields valid AppConfig
  T_F2  — single-leaf override; sibling keys preserved
  T_F3  — deep multi-leaf override (retrieval + fusion); both correct,
          other top-level keys untouched
  T_F4a — make_test_negation_config() with no overrides yields valid object
  T_F4b — make_test_negation_config partial override (en_patterns only)
          preserves cn_patterns sibling
  T_F4c — make_test_scoring_config() with no overrides yields valid object
  T_F4d — make_test_scoring_config partial override (disease_weights leaf)
          preserves healthy_weights sibling
  T_F5  — factory product is acceptable to Pipeline.__init__ (no warmup);
          identity preservation via `is` (lazy import + skip-on-fail)
"""

from __future__ import annotations

from typing import Any

import pytest

from fishdx.config import AppConfig, NegationConfig, ScoringConfig
from tests.conftest import (
    make_test_config,
    make_test_negation_config,
    make_test_scoring_config,
)


# ───────────────────────── T_F1 ─────────────────────────
def test_f1_make_test_config_no_overrides() -> None:
    """T_F1 — factory with no overrides yields valid AppConfig.

    DEFAULTS_REGISTRY completeness invariant: every required AppConfig
    field is present in _DEFAULTS, and all Pydantic validators pass.
    """
    cfg = make_test_config()
    assert isinstance(cfg, AppConfig)
    # Spot-check [PAPER] defaults survive the round-trip:
    assert cfg.meta.seed == 42
    assert cfg.fusion.lambda_star == 0.7
    assert cfg.kb.documents_count == 8
    assert cfg.margin.retrieval_margin_theta == 0.02


# ───────────────────────── T_F2 ─────────────────────────
def test_f2_single_leaf_override_sibling_preserved() -> None:
    """T_F2 — overriding retrieval.similarity_cutoff preserves all sibling keys.

    Locks the deep_merge invariant: nested-dict overrides only mutate the
    specified leaf; other leaves under the same sub-config remain at
    DEFAULTS_REGISTRY values.
    """
    cfg = make_test_config(retrieval={"similarity_cutoff": 0.3})
    # Modified leaf
    assert cfg.retrieval.similarity_cutoff == 0.3
    # Sibling keys under retrieval — must remain at defaults
    assert cfg.retrieval.top_k == 5
    assert cfg.retrieval.metric == "cosine"
    assert cfg.retrieval.collection == "image_gallery"
    assert cfg.retrieval.hnsw_M == 16
    # Other top-level sub-configs untouched
    assert cfg.fusion.lambda_star == 0.7
    assert cfg.meta.seed == 42


# ───────────────────────── T_F3 ─────────────────────────
def test_f3_deep_multi_leaf_override() -> None:
    """T_F3 — multi-sub-config override: retrieval + fusion, each precise.

    Locks: deep_merge handles multiple sub-config overrides independently;
    no cross-contamination between the two override blocks.
    """
    cfg = make_test_config(
        retrieval={"similarity_cutoff": 0.3},
        fusion={"lambda_star": 0.5},
    )
    # Both overrides applied
    assert cfg.retrieval.similarity_cutoff == 0.3
    assert cfg.fusion.lambda_star == 0.5
    # Siblings under each modified sub-config preserved
    assert cfg.retrieval.top_k == 5
    assert cfg.fusion.normalize_pre is True
    assert cfg.fusion.normalize_post is True
    # Top-level keys outside the override path preserved
    assert cfg.meta.seed == 42
    assert cfg.kb.documents_count == 8


# ───────────────────────── T_F4a ─────────────────────────
def test_f4a_make_test_negation_config_no_overrides() -> None:
    """T_F4a — make_test_negation_config() with no overrides yields valid object.

    Defaults satisfy Pydantic min_length=1 validators on both
    cn_patterns and en_patterns.
    """
    neg = make_test_negation_config()
    assert isinstance(neg, NegationConfig)
    assert neg.window_chars == 50
    assert len(neg.cn_patterns) >= 1
    assert len(neg.en_patterns) >= 1
    assert neg.en_patterns == ["not"]


# ───────────────────────── T_F4b ─────────────────────────
def test_f4b_make_test_negation_config_partial_override() -> None:
    """T_F4b — overriding en_patterns preserves cn_patterns sibling."""
    neg_default = make_test_negation_config()
    neg_custom = make_test_negation_config(
        en_patterns=["custom_en_pattern"],
    )
    # en_patterns overridden
    assert neg_custom.en_patterns == ["custom_en_pattern"]
    # cn_patterns sibling preserved (deep_merge invariant)
    assert neg_custom.cn_patterns == neg_default.cn_patterns
    # window_chars preserved
    assert neg_custom.window_chars == neg_default.window_chars


# ───────────────────────── T_F4c ─────────────────────────
def test_f4c_make_test_scoring_config_no_overrides() -> None:
    """T_F4c — make_test_scoring_config() with no overrides yields valid object.

    Defaults pre-fill DiseaseWeights (3/2/1 monotonic) + HealthyWeights
    (2/1 monotonic); both nested-config validators pass.
    """
    sc = make_test_scoring_config()
    assert isinstance(sc, ScoringConfig)
    assert sc.disease_weights.confirmed == 3
    assert sc.disease_weights.suspected == 2
    assert sc.disease_weights.mentioned == 1
    assert sc.healthy_weights.explicit == 2
    assert sc.healthy_weights.negation == 1
    assert sc.keyword_dict == "fishdx_kb"


# ───────────────────────── T_F4d ─────────────────────────
def test_f4d_make_test_scoring_config_partial_override() -> None:
    """T_F4d — overriding disease_weights.confirmed preserves siblings.

    Verifies deep_merge through nested dict (scoring → disease_weights →
    confirmed leaf) without affecting the parallel healthy_weights
    sub-config.
    """
    sc = make_test_scoring_config(
        disease_weights={"confirmed": 5},
    )
    # Modified leaf
    assert sc.disease_weights.confirmed == 5
    # Sibling leaves under disease_weights preserved
    assert sc.disease_weights.suspected == 2
    assert sc.disease_weights.mentioned == 1
    # Parallel sub-config (healthy_weights) entirely preserved
    assert sc.healthy_weights.explicit == 2
    assert sc.healthy_weights.negation == 1
    # Top-level scalar preserved
    assert sc.keyword_dict == "fishdx_kb"


# ───────────────────────── T_F5 ─────────────────────────
def test_f5_factory_product_acceptable_to_pipeline() -> None:
    """T_F5 — factory output is accepted by Pipeline.__init__ (no warmup).

    Verifies that the factory-built AppConfig satisfies every contract
    that Pipeline expects of an AppConfig at construction time. We
    instantiate Pipeline but do NOT call ``warmup()`` — that would
    trigger Florence-2 / OpenCLIP weight loading and require GPU + a
    multi-GB Hugging Face download not available in this test
    environment.

    The assertion uses ``is`` (identity) because Pipeline currently
    stores the config by reference (``self._config = config``). If a
    future refactor moves Pipeline to a value-copy storage strategy,
    update this assertion to ``==`` and document the change in
    CHANGELOG.

    Lazy import + ``pytest.skip`` guard: if Pipeline-side dependencies
    (transformers / open_clip / chromadb) become unavailable due to
    environment drift, this self-test gracefully skips rather than
    cascading into a brittle failure across the whole conftest test
    suite.
    """
    try:
        from fishdx.pipeline import Pipeline
    except ImportError as e:  # pragma: no cover — env-dependent
        pytest.skip(f"Pipeline import failed (env-dependent): {e}")

    cfg = make_test_config()
    pipeline = Pipeline(cfg)
    assert isinstance(pipeline, Pipeline)
    # Identity check (by reference) — locks Pipeline's by-reference
    # storage contract. Pipeline stores config as `self._config`.
    assert pipeline._config is cfg  # noqa: SLF001 — intentional private access
