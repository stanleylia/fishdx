"""Dataset registry — M1 Q1 (YAML + Pydantic hybrid).

Source-of-truth: ``configs/dataset.yaml``. Loaded + validated into
frozen :class:`DatasetSpec` instances keyed by ``dataset_id``. HELD_OUT
datasets are auto-registered with :class:`DatasetAccessGuard`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from fishdx.data.isolation import get_guard
from fishdx.errors import ConfigFileError, ConfigSchemaError
from fishdx.schemas import DatasetRole, DatasetSpec


class DatasetRegistry(BaseModel):
    """Frozen collection of all :class:`DatasetSpec` entries."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    datasets: list[DatasetSpec] = Field(min_length=1)

    def by_id(self, dataset_id: str) -> DatasetSpec:
        for spec in self.datasets:
            if spec.dataset_id == dataset_id:
                return spec
        raise KeyError(f"dataset_id not in registry: {dataset_id}")

    def ids(self) -> list[str]:
        return [s.dataset_id for s in self.datasets]


def load_registry(path: Path | str, *, resolve_roots: Path | None = None) -> DatasetRegistry:
    """Parse ``configs/dataset.yaml`` into a :class:`DatasetRegistry`.

    If ``resolve_roots`` is given, relative ``root_path`` entries are
    resolved against it and ``available`` is populated from filesystem
    existence. HELD_OUT datasets are registered with the global
    :class:`DatasetAccessGuard` at load time.
    """
    p = Path(path)
    try:
        with p.open("r", encoding="utf-8") as f:
            raw: Any = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ConfigFileError(f"registry file not found: {p}", context={"path": str(p)}) from e
    except yaml.YAMLError as e:
        raise ConfigFileError(f"YAML parse error: {e}", context={"path": str(p)}) from e

    if not isinstance(raw, dict) or "datasets" not in raw:
        raise ConfigFileError(
            "registry root must be a mapping with 'datasets' key", context={"path": str(p)}
        )

    if resolve_roots is not None:
        resolved_entries: list[dict[str, Any]] = []
        for original in raw["datasets"]:
            resolved = dict(original)
            root = Path(resolved["root_path"])
            if not root.is_absolute():
                root = (resolve_roots / root).resolve()
            resolved["root_path"] = str(root)
            resolved.setdefault("available", root.exists())
            resolved_entries.append(resolved)
        raw = {"datasets": resolved_entries}

    try:
        registry = DatasetRegistry.model_validate(raw)
    except ValidationError as e:
        raise ConfigSchemaError(str(e), context={"path": str(p)}) from e

    guard = get_guard()
    for spec in registry.datasets:
        if spec.role is DatasetRole.HELD_OUT:
            guard.register_held_out(spec.dataset_id)

    return registry


__all__ = ["DatasetRegistry", "load_registry"]
