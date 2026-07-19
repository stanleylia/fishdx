"""Phase 2 — config.py ``extends:`` inheritance + deep_merge tests.

Test matrix (per Phase 2 design brief §4):
  T1  — no extends → backward-compat with single-file behaviour
  T2  — single-level extends → child overrides parent
  T3  — multi-level extends (A→B→default) → deep-to-shallow merge
  T4  — extends target missing → ConfigNotFoundError (strict type check)
  T5  — circular extends → ConfigCircularExtendsError (strict type check)
  T6  — child overrides single nested leaf → siblings unaffected
  T7a — partial child dict loads (no premature validation, black-box)
  T7b — model_validate called exactly once (white-box, monkeypatch spy)
  T7c — deep partial override (nested leaf only)
  T8  — root not a mapping (YAML list) → ConfigFileError
  T9  — empty YAML file with extends: → loads to parent-equivalent
  T10 — extends value not a string → ConfigFileError
  T11 — extends value absolute path → ConfigFileError
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from fishdx import config as config_module
from fishdx.config import AppConfig, deep_merge, load_config
from fishdx.errors import (
    ConfigCircularExtendsError,
    ConfigFileError,
    ConfigMissingFieldError,
    ConfigNotFoundError,
)

# ────────────────────────── helpers ──────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_YAML = _REPO_ROOT / "configs" / "default.yaml"


def _load_default_dict() -> dict[str, Any]:
    """Read configs/default.yaml as raw dict (no extends, no validation)."""
    return yaml.safe_load(_DEFAULT_YAML.read_text(encoding="utf-8"))


def _write_yaml(path: Path, data: Any) -> None:
    """Write ``data`` to ``path`` as YAML."""
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


# ───────────────────────── T1 ─────────────────────────
def test_t1_no_extends_backward_compat(tmp_path: Path) -> None:
    """No extends key → behaviour identical to pre-Phase-2 load."""
    full = _load_default_dict()
    yaml_path = tmp_path / "test.yaml"
    _write_yaml(yaml_path, full)
    cfg = load_config(yaml_path)
    assert cfg.retrieval.similarity_cutoff == 0.5
    assert cfg.fusion.lambda_star == 0.7
    assert cfg.meta.seed == 42
    assert cfg.kb.documents_count == 8


# ───────────────────────── T2 ─────────────────────────
def test_t2_single_level_extends(tmp_path: Path) -> None:
    """Single-level extends → child overrides parent."""
    parent = tmp_path / "parent.yaml"
    _write_yaml(parent, _load_default_dict())
    child = tmp_path / "child.yaml"
    _write_yaml(
        child,
        {"extends": "parent.yaml", "retrieval": {"similarity_cutoff": 0.25}},
    )
    cfg = load_config(child)
    assert cfg.retrieval.similarity_cutoff == 0.25
    assert cfg.fusion.lambda_star == 0.7  # inherited
    assert cfg.retrieval.top_k == 5  # inherited from parent retrieval block


# ───────────────────────── T3 ─────────────────────────
def test_t3_multi_level_extends(tmp_path: Path) -> None:
    """A → B → default — verify three-level merge order."""
    grandparent = tmp_path / "grandparent.yaml"
    _write_yaml(grandparent, _load_default_dict())
    parent = tmp_path / "parent.yaml"
    _write_yaml(
        parent,
        {"extends": "grandparent.yaml", "fusion": {"lambda_star": 0.5}},
    )
    child = tmp_path / "child.yaml"
    _write_yaml(
        child,
        {"extends": "parent.yaml", "retrieval": {"similarity_cutoff": 0.25}},
    )
    cfg = load_config(child)
    assert cfg.retrieval.similarity_cutoff == 0.25  # from child
    assert cfg.fusion.lambda_star == 0.5  # from parent
    assert cfg.meta.seed == 42  # from grandparent


# ───────────────────────── T4 ─────────────────────────
def test_t4_extends_target_missing(tmp_path: Path) -> None:
    """extends pointing to non-existent file → ConfigNotFoundError (strict type)."""
    child = tmp_path / "child.yaml"
    _write_yaml(child, {"extends": "nonexistent.yaml"})
    with pytest.raises(ConfigNotFoundError) as exc_info:
        load_config(child)
    # Strict type assertion (踩雷點 3): not just isinstance.
    assert type(exc_info.value) is ConfigNotFoundError
    assert exc_info.value.missing_path.name == "nonexistent.yaml"
    assert len(exc_info.value.extends_chain) == 2
    # First link in the chain is the child; last is the missing parent.
    assert exc_info.value.extends_chain[0].resolve() == child.resolve()


# ───────────────────────── T5 ─────────────────────────
def test_t5_circular_extends(tmp_path: Path) -> None:
    """A → B → A → ConfigCircularExtendsError (strict type)."""
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    _write_yaml(a, {"extends": "b.yaml"})
    _write_yaml(b, {"extends": "a.yaml"})
    with pytest.raises(ConfigCircularExtendsError) as exc_info:
        load_config(a)
    # Strict type assertion (踩雷點 3): not just isinstance.
    assert type(exc_info.value) is ConfigCircularExtendsError
    chain = exc_info.value.cycle_chain
    assert len(chain) >= 3
    # First and last elements of the cycle chain refer to the same path.
    assert chain[0].resolve() == chain[-1].resolve()


# ───────────────────────── T6 ─────────────────────────
def test_t6_partial_nested_override(tmp_path: Path) -> None:
    """Override a single leaf → siblings under the same key unaffected."""
    parent = tmp_path / "parent.yaml"
    _write_yaml(parent, _load_default_dict())
    child = tmp_path / "child.yaml"
    _write_yaml(
        child,
        {"extends": "parent.yaml", "retrieval": {"similarity_cutoff": 0.25}},
    )
    cfg = load_config(child)
    # Modified leaf
    assert cfg.retrieval.similarity_cutoff == 0.25
    # Sibling keys under retrieval — must remain at parent values
    assert cfg.retrieval.top_k == 5
    assert cfg.retrieval.metric == "cosine"
    assert cfg.retrieval.collection == "image_gallery"
    assert cfg.retrieval.hnsw_M == 16


# ───────────────────────── T7a ─────────────────────────
def test_t7a_partial_dict_loads(tmp_path: Path) -> None:
    """Partial child dict (only 1 leaf) loads OK — no premature validation."""
    parent = tmp_path / "parent.yaml"
    _write_yaml(parent, _load_default_dict())
    child = tmp_path / "child.yaml"
    _write_yaml(
        child,
        {"extends": "parent.yaml", "retrieval": {"similarity_cutoff": 0.25}},
    )
    # If validation fired prematurely on the partial child, this would
    # raise ConfigMissingFieldError. The fact that it succeeds is the
    # regression guard for 踩雷點 1.
    cfg = load_config(child)
    assert cfg.retrieval.similarity_cutoff == 0.25


# ───────────────────────── T7b ─────────────────────────
def test_t7b_validation_called_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """White-box: AppConfig.model_validate is called exactly once (post-merge)."""
    parent = tmp_path / "parent.yaml"
    _write_yaml(parent, _load_default_dict())
    child = tmp_path / "child.yaml"
    _write_yaml(
        child,
        {"extends": "parent.yaml", "retrieval": {"similarity_cutoff": 0.25}},
    )

    captured: dict[str, Any] = {"call_count": 0, "last_payload": None}
    real_validate = AppConfig.model_validate

    @classmethod  # type: ignore[misc]
    def spy_validate(cls: type[AppConfig], obj: Any) -> AppConfig:
        captured["call_count"] += 1
        captured["last_payload"] = obj
        return real_validate(obj)

    monkeypatch.setattr(AppConfig, "model_validate", spy_validate)

    load_config(child)

    assert captured["call_count"] == 1, (
        f"Expected exactly 1 model_validate call, got {captured['call_count']} "
        "— validation MUST happen only after merge completion."
    )
    payload = captured["last_payload"]
    assert isinstance(payload, dict)
    # The merged dict must contain ALL top-level keys (parent + child merged).
    assert "meta" in payload
    assert "florence2" in payload
    assert "pareidolia" in payload
    assert "retrieval" in payload
    # And the override must have landed in the merged result.
    assert payload["retrieval"]["similarity_cutoff"] == 0.25


# ───────────────────────── T7c ─────────────────────────
def test_t7c_deep_partial_override(tmp_path: Path) -> None:
    """Deep partial: override fusion.lambda_star (a nested leaf) only."""
    parent = tmp_path / "parent.yaml"
    _write_yaml(parent, _load_default_dict())
    child = tmp_path / "child.yaml"
    _write_yaml(
        child,
        {"extends": "parent.yaml", "fusion": {"lambda_star": 0.5}},
    )
    cfg = load_config(child)
    assert cfg.fusion.lambda_star == 0.5
    # Other leaves under fusion must remain at parent values.
    assert cfg.fusion.normalize_pre is True
    assert cfg.fusion.normalize_post is True
    assert cfg.fusion.compute_dtype == "fp32"
    # Other top-level keys untouched.
    assert cfg.retrieval.similarity_cutoff == 0.5


# ───────────────────────── T8 ─────────────────────────
def test_t8_root_not_mapping(tmp_path: Path) -> None:
    """Root YAML must be a mapping, not a list."""
    bad = tmp_path / "bad.yaml"
    _write_text(bad, "- item1\n- item2\n")
    with pytest.raises(ConfigFileError) as exc_info:
        load_config(bad)
    assert "must be a YAML mapping" in str(exc_info.value)


# ───────────────────────── T9 ─────────────────────────
def test_t9_empty_yaml_file(tmp_path: Path) -> None:
    """Empty YAML file with extends-only path inherits parent fully."""
    parent = tmp_path / "parent.yaml"
    _write_yaml(parent, _load_default_dict())
    # Workaround: empty raw YAML (None) cannot carry extends, so this T9
    # variant uses an extends-only child that adds nothing else — equivalent
    # to "no override" + parent inheritance.
    child = tmp_path / "child.yaml"
    _write_text(child, "extends: parent.yaml\n")
    cfg = load_config(child)
    # Must equal the parent (no overrides).
    assert cfg.retrieval.similarity_cutoff == 0.5
    assert cfg.fusion.lambda_star == 0.7
    assert cfg.meta.seed == 42

    # And: a fully-empty file (no extends) must fail with missing-field error
    # (since AppConfig requires all 17 top-level keys).
    truly_empty = tmp_path / "truly_empty.yaml"
    _write_text(truly_empty, "")
    with pytest.raises(ConfigMissingFieldError):
        load_config(truly_empty)


# ───────────────────────── T10 ─────────────────────────
def test_t10_extends_not_string(tmp_path: Path) -> None:
    """extends: must be a single string (no multi-inheritance)."""
    child = tmp_path / "child.yaml"
    _write_text(child, "extends:\n  - a.yaml\n  - b.yaml\n")
    with pytest.raises(ConfigFileError) as exc_info:
        load_config(child)
    assert "must be a string path" in str(exc_info.value)


# ───────────────────────── T11 ─────────────────────────
def test_t11_extends_absolute_path(tmp_path: Path) -> None:
    """extends: must be a relative path."""
    child = tmp_path / "child.yaml"
    _write_text(child, "extends: /tmp/parent.yaml\n")
    with pytest.raises(ConfigFileError) as exc_info:
        load_config(child)
    assert "must be a relative path" in str(exc_info.value)


# ───────────────────────── deep_merge unit tests ─────────────────────────
class TestDeepMerge:
    """Unit-level coverage of deep_merge to support T1–T11."""

    def test_dict_dict_recursive(self) -> None:
        a = {"x": {"y": 1, "z": 2}}
        b = {"x": {"y": 99}}
        assert deep_merge(a, b) == {"x": {"y": 99, "z": 2}}

    def test_list_replaced_not_appended(self) -> None:
        assert deep_merge([1, 2, 3], [4, 5]) == [4, 5]

    def test_scalar_child_wins(self) -> None:
        assert deep_merge(1, 2) == 2

    def test_child_none_preserves_parent(self) -> None:
        # E4 reversed: child=None means "unspecified", keep parent.
        assert deep_merge({"k": 1}, {"k": None}) == {"k": 1}
        assert deep_merge([1, 2], None) == [1, 2]

    def test_type_mismatch_child_wins(self) -> None:
        assert deep_merge(1, {"k": 2}) == {"k": 2}
        assert deep_merge({"k": 2}, 1) == 1

    def test_associativity_smoke(self) -> None:
        a = {"x": {"y": 1}, "z": [1, 2]}
        b = {"x": {"y": 2, "w": 3}}
        c = {"z": [9]}
        left = deep_merge(deep_merge(a, b), c)
        right = deep_merge(a, deep_merge(b, c))
        assert left == right
