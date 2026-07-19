"""Dataset loaders — M1 architecture §4.

Three layout strategies observed in D1-D9:

* **ImageFolderLoader** — per-class subdirectory (D1 Train/Test, D2 train_split,
  D3, D4, D6, D8). Most common layout.
* **FlatPrefixLoader** — flat directory with filename prefix = class name
  separated by underscore (D2 test_split: ``EUS_EUS_134.jpg``).
* **BinaryFreshInfectedLoader** — two-class Fresh/Infected layout (D5).

All loaders deterministically sort records by ``record_id`` and call
``DatasetAccessGuard.check()`` before yielding (paper §IV.A A8).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from fishdx.data.isolation import get_guard
from fishdx.errors import DatasetNotFoundError, DatasetSizeMismatchError
from fishdx.schemas import DatasetSpec, HealthStatus, ImageRecord

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


class DatasetLoader(Protocol):
    """Deterministic read-only loader protocol."""

    def load(self, spec: DatasetSpec) -> Iterator[ImageRecord]: ...

    def count(self, spec: DatasetSpec) -> int: ...

    def health_check(self, spec: DatasetSpec) -> HealthStatus: ...


def _assert_root(spec: DatasetSpec) -> Path:
    root = spec.root_path
    if not root.exists():
        raise DatasetNotFoundError(
            f"dataset root missing: {root} (dataset_id={spec.dataset_id})",
            context={"dataset_id": spec.dataset_id, "root": str(root)},
        )
    return root


def _image_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix in _IMAGE_EXTS)


class ImageFolderLoader:
    """Layout: ``<root>/<class_name>/<image>.jpg``."""

    def load(self, spec: DatasetSpec) -> Iterator[ImageRecord]:
        get_guard().check(spec.dataset_id)
        root = _assert_root(spec)
        records: list[ImageRecord] = []
        for cls_dir in sorted(d for d in root.iterdir() if d.is_dir()):
            for img in _image_files(cls_dir):
                rid = f"{spec.dataset_id}/{cls_dir.name}/{img.name}"
                records.append(
                    ImageRecord(
                        record_id=rid,
                        image_path=img,
                        label=cls_dir.name,
                        dataset_id=spec.dataset_id,
                        role=spec.role,
                    )
                )
        records.sort(key=lambda r: r.record_id)
        yield from records

    def count(self, spec: DatasetSpec) -> int:
        root = _assert_root(spec)
        return sum(len(_image_files(d)) for d in root.iterdir() if d.is_dir())

    def health_check(self, spec: DatasetSpec) -> HealthStatus:
        try:
            n = self.count(spec)
        except DatasetNotFoundError:
            return HealthStatus(
                component=f"loader:{spec.dataset_id}",
                ready=False,
                status="failed",
                details={"reason": "root_missing"},
            )
        drift = n != spec.expected_sample_count
        return HealthStatus(
            component=f"loader:{spec.dataset_id}",
            ready=not drift,
            status="degraded" if drift else "ok",
            details={"observed": str(n), "expected": str(spec.expected_sample_count)},
        )


class FlatPrefixLoader:
    """Layout: ``<root>/<label>_<rest>.jpg`` — filename prefix encodes label.

    Paper §IV.A footnote: D2 ``test_split`` uses this layout with EUS files
    named ``EUS_EUS_NNN.jpg``.
    """

    def load(self, spec: DatasetSpec) -> Iterator[ImageRecord]:
        get_guard().check(spec.dataset_id)
        root = _assert_root(spec)
        records: list[ImageRecord] = []
        for img in _image_files(root):
            label = img.stem.split("_")[0]
            records.append(
                ImageRecord(
                    record_id=f"{spec.dataset_id}/flat/{img.name}",
                    image_path=img,
                    label=label,
                    dataset_id=spec.dataset_id,
                    role=spec.role,
                )
            )
        records.sort(key=lambda r: r.record_id)
        yield from records

    def count(self, spec: DatasetSpec) -> int:
        return len(_image_files(_assert_root(spec)))

    def health_check(self, spec: DatasetSpec) -> HealthStatus:
        try:
            n = self.count(spec)
        except DatasetNotFoundError:
            return HealthStatus(
                component=f"loader:{spec.dataset_id}",
                ready=False,
                status="failed",
                details={"reason": "root_missing"},
            )
        return HealthStatus(
            component=f"loader:{spec.dataset_id}",
            ready=True,
            status="ok",
            details={"observed": str(n)},
        )


def assert_size(spec: DatasetSpec, observed: int) -> None:
    """Raise :class:`DatasetSizeMismatchError` if observed != expected."""
    if observed != spec.expected_sample_count:
        raise DatasetSizeMismatchError(
            f"{spec.dataset_id} size drift: observed={observed}, "
            f"expected={spec.expected_sample_count}",
            context={
                "dataset_id": spec.dataset_id,
                "observed": observed,
                "expected": spec.expected_sample_count,
            },
        )


__all__ = ["DatasetLoader", "FlatPrefixLoader", "ImageFolderLoader", "assert_size"]
