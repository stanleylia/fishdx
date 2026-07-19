"""Dataset supply-chain drift — M1 architecture §4.3 + registry §3.

3 cases covering missing dataset, size mismatch, and registry integrity.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fishdx.data.loaders import ImageFolderLoader, assert_size
from fishdx.data.registry import load_registry
from fishdx.errors import DatasetNotFoundError, DatasetSizeMismatchError
from fishdx.schemas import DatasetRole, DatasetSpec

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_registry_loads_nine_entries() -> None:
    """SC1 | configs/dataset.yaml parses to exactly 9 D1-D9 specs."""
    registry = load_registry(_REPO_ROOT / "configs" / "dataset.yaml")
    assert len(registry.datasets) == 9
    assert registry.ids() == [f"D{i}" for i in range(1, 10)]


def test_size_mismatch_raises() -> None:
    """SC2 | observed size ≠ expected raises DatasetSizeMismatchError."""
    spec = DatasetSpec(
        dataset_id="TEST",
        human_name="synthetic",
        root_path=Path("/tmp"),
        role=DatasetRole.TRAIN_ELIGIBLE,
        expected_sample_count=1000,
        expected_classes=1,
        license="synthetic",
        paper_citation="n/a",
    )
    with pytest.raises(DatasetSizeMismatchError):
        assert_size(spec, 999)


def test_missing_root_raises_dataset_not_found() -> None:
    """SC3 | loader on absent path raises DatasetNotFoundError."""
    spec = DatasetSpec(
        dataset_id="MISSING",
        human_name="does_not_exist",
        root_path=Path("/nonexistent/path/does/not/exist"),
        role=DatasetRole.TRAIN_ELIGIBLE,
        expected_sample_count=0,
        expected_classes=0,
        license="n/a",
        paper_citation="n/a",
    )
    loader = ImageFolderLoader()
    with pytest.raises(DatasetNotFoundError):
        list(loader.load(spec))
