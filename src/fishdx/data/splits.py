"""Stratified split — M1 architecture §5.

Deterministic seed=42 algorithm: sort records by ``record_id`` → per-class
shuffle using ``numpy.random.default_rng`` → take prefix
``train_ratio·|class|`` → remainder to test. Invariants I-Seed … I-D2 are
enforced at the boundary (see ``_validate``).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from fishdx.errors import DataIsolationViolation, ReproducibilityError
from fishdx.schemas import DatasetRole, ImageRecord, SplitManifest

_RATIO_TOLERANCE = 0.01  # per-class stratification drift tolerance
# Global tolerance accounts for per-class int(round(·)) rounding (each class
# contributes up to ±0.5 samples). Total worst-case drift ≈ n_classes / 2, so
# tolerance scales as n_classes / (2·total) with a 5e-3 absolute floor.
_GLOBAL_RATIO_FLOOR = 5e-3


class StratifiedSplitter:
    """Seeded stratified random splitter (paper Table IX seed=42)."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def split(
        self,
        records: Sequence[ImageRecord],
        train_ratio: float,
        seed: int | None = None,
        stratify: bool = True,
    ) -> tuple[list[ImageRecord], list[ImageRecord]]:
        if any(r.role is DatasetRole.HELD_OUT for r in records):
            raise DataIsolationViolation(
                "StratifiedSplitter received records with role=HELD_OUT (§IV.A A8)",
                context={"reason": "split_receives_held_out"},
            )
        effective_seed = seed if seed is not None else self._seed
        ordered = sorted(records, key=lambda r: r.record_id)

        if not stratify:
            rng = np.random.default_rng(effective_seed)
            idx = rng.permutation(len(ordered))
            cut = int(round(train_ratio * len(ordered)))
            train = [ordered[i] for i in idx[:cut]]
            test = [ordered[i] for i in idx[cut:]]
        else:
            by_cls: dict[str, list[ImageRecord]] = defaultdict(list)
            for r in ordered:
                by_cls[r.label].append(r)
            train, test = [], []
            for cls, items in sorted(by_cls.items()):
                rng = np.random.default_rng(
                    int(
                        hashlib.sha256(f"{effective_seed}:{cls}".encode()).hexdigest()[:8],
                        16,
                    )
                )
                perm = rng.permutation(len(items))
                cut = int(round(train_ratio * len(items)))
                train.extend(items[i] for i in perm[:cut])
                test.extend(items[i] for i in perm[cut:])
            train.sort(key=lambda r: r.record_id)
            test.sort(key=lambda r: r.record_id)

        self._validate(train, test, train_ratio, stratify)
        return train, test

    def manifest(
        self,
        train: Sequence[ImageRecord],
        test: Sequence[ImageRecord],
        dataset_id: str,
        train_ratio: float,
        seed: int | None = None,
        stratify: bool = True,
    ) -> SplitManifest:
        effective_seed = seed if seed is not None else self._seed
        per_class: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
        for r in train:
            t, e = per_class[r.label]
            per_class[r.label] = (t + 1, e)
        for r in test:
            t, e = per_class[r.label]
            per_class[r.label] = (t, e + 1)
        return SplitManifest(
            dataset_id=dataset_id,
            seed=effective_seed,
            train_ratio=train_ratio,
            stratify=stratify,
            sha256_ids_train=self._sha(train),
            sha256_ids_test=self._sha(test),
            n_train=len(train),
            n_test=len(test),
            per_class_counts=dict(per_class),
        )

    @staticmethod
    def _sha(records: Sequence[ImageRecord]) -> str:
        h = hashlib.sha256()
        for r in sorted(records, key=lambda x: x.record_id):
            h.update(r.record_id.encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    @staticmethod
    def _validate(
        train: Sequence[ImageRecord],
        test: Sequence[ImageRecord],
        train_ratio: float,
        stratify: bool,
    ) -> None:
        total = len(train) + len(test)
        if total == 0:
            return
        # I-Disjoint
        train_ids = {r.record_id for r in train}
        if train_ids & {r.record_id for r in test}:
            raise ReproducibilityError(
                "StratifiedSplitter produced overlapping train/test ids",
                context={"invariant": "I-Disjoint"},
            )
        # I-Ratio
        observed = len(train) / total
        n_classes = len({r.label for r in train} | {r.label for r in test})
        rounding_budget = n_classes / (2 * total) if total > 0 else 0
        tol = max(_GLOBAL_RATIO_FLOOR, rounding_budget)
        if abs(observed - train_ratio) > tol + 1e-9:
            raise ReproducibilityError(
                f"global ratio drift: observed={observed:.4f}, expected={train_ratio}",
                context={"invariant": "I-Ratio", "tolerance": tol},
            )
        # I-Strat
        if stratify:
            by_cls_total: dict[str, int] = defaultdict(int)
            by_cls_train: dict[str, int] = defaultdict(int)
            for r in train:
                by_cls_total[r.label] += 1
                by_cls_train[r.label] += 1
            for r in test:
                by_cls_total[r.label] += 1
            for cls, n_total in by_cls_total.items():
                if n_total == 0:
                    continue
                cls_ratio = by_cls_train[cls] / n_total
                if abs(cls_ratio - train_ratio) > _RATIO_TOLERANCE + 1 / n_total:
                    raise ReproducibilityError(
                        f"stratification drift for class {cls}: {cls_ratio:.3f}",
                        context={"invariant": "I-Strat", "class": cls},
                    )


__all__ = ["StratifiedSplitter"]
