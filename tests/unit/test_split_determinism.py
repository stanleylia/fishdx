"""M1 split determinism — docs/m1/architecture-draft.md §9.

3 cases: same seed → bit-identical output across reruns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fishdx.data.splits import StratifiedSplitter
from fishdx.schemas import DatasetRole, ImageRecord


def _make_records(n: int, classes: int = 4) -> list[ImageRecord]:
    return [
        ImageRecord(
            record_id=f"DS/c{i % classes}/img_{i:04d}.jpg",
            image_path=Path(f"/tmp/img_{i:04d}.jpg"),
            label=f"c{i % classes}",
            dataset_id="DS",
            role=DatasetRole.TRAIN_ELIGIBLE,
        )
        for i in range(n)
    ]


@pytest.mark.parametrize("seed", [42, 7, 2026])
def test_split_rerun_is_identical(seed: int) -> None:
    """I-Seed | same seed × same records → bit-identical split across 3 reruns."""
    records = _make_records(400)
    splitter = StratifiedSplitter()
    results = [splitter.split(records, train_ratio=0.715, seed=seed) for _ in range(3)]
    train_ids = [tuple(r.record_id for r in tr) for tr, _ in results]
    test_ids = [tuple(r.record_id for r in te) for _, te in results]
    assert train_ids[0] == train_ids[1] == train_ids[2]
    assert test_ids[0] == test_ids[1] == test_ids[2]
