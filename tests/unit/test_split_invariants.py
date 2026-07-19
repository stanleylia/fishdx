"""M1 split invariants I-Seed…I-D2 — docs/m1/architecture-draft.md §5.2."""

from __future__ import annotations

from pathlib import Path

import pytest

from fishdx.data.splits import StratifiedSplitter
from fishdx.errors import DataIsolationViolation
from fishdx.schemas import DatasetRole, ImageRecord


def _make_records(
    n: int, classes: int = 4, role: DatasetRole = DatasetRole.TRAIN_ELIGIBLE
) -> list[ImageRecord]:
    return [
        ImageRecord(
            record_id=f"DS/c{i % classes}/img_{i:04d}.jpg",
            image_path=Path(f"/tmp/img_{i:04d}.jpg"),
            label=f"c{i % classes}",
            dataset_id="DS",
            role=role,
        )
        for i in range(n)
    ]


def test_i_seed_same_seed_identical() -> None:
    """I-Seed | Identical output for identical (records, ratio, seed, stratify)."""
    records = _make_records(200)
    splitter = StratifiedSplitter()
    a = splitter.split(records, train_ratio=0.715, seed=42)
    b = splitter.split(records, train_ratio=0.715, seed=42)
    assert [r.record_id for r in a[0]] == [r.record_id for r in b[0]]
    assert [r.record_id for r in a[1]] == [r.record_id for r in b[1]]


def test_i_ratio_within_rounding() -> None:
    """I-Ratio | global train ratio == train_ratio ± 1/N."""
    records = _make_records(1000)
    splitter = StratifiedSplitter()
    train, test = splitter.split(records, train_ratio=0.715)
    ratio = len(train) / (len(train) + len(test))
    # per-class rounding budget: n_classes/(2·total) = 4/(2·1000) = 2e-3
    assert abs(ratio - 0.715) < 3e-3


def test_i_strat_per_class_balance() -> None:
    """I-Strat | per-class ratio divergence ≤ 0.01 when stratify=True."""
    records = _make_records(4000, classes=8)
    splitter = StratifiedSplitter()
    train, test = splitter.split(records, train_ratio=0.715, stratify=True)
    per_cls_train: dict[str, int] = {}
    per_cls_total: dict[str, int] = {}
    for r in train:
        per_cls_train[r.label] = per_cls_train.get(r.label, 0) + 1
        per_cls_total[r.label] = per_cls_total.get(r.label, 0) + 1
    for r in test:
        per_cls_total[r.label] = per_cls_total.get(r.label, 0) + 1
    for cls, total in per_cls_total.items():
        assert abs(per_cls_train.get(cls, 0) / total - 0.715) < 0.02


def test_i_disjoint_no_overlap() -> None:
    """I-Disjoint | train ∩ test == ∅ on record_id."""
    records = _make_records(500)
    splitter = StratifiedSplitter()
    train, test = splitter.split(records, train_ratio=0.715)
    assert {r.record_id for r in train}.isdisjoint({r.record_id for r in test})


def test_i_d2_held_out_raises() -> None:
    """I-D2 | HELD_OUT role at split entrypoint raises DataIsolationViolation."""
    records = _make_records(100, role=DatasetRole.HELD_OUT)
    splitter = StratifiedSplitter()
    with pytest.raises(DataIsolationViolation):
        splitter.split(records, train_ratio=0.715)
