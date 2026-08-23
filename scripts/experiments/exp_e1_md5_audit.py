"""D1 train/test byte-identity (MD5) audit — the evidence artifact behind
Supplementary Note S1's leakage claim.

The original dedup audit (results/e1_dedup_report.json) recorded only perceptual
hash / CLIP cosine of the nearest pHash neighbour, never MD5. This script records
the definitive byte-level evidence: for every D1 test image, whether a
same-filename twin exists in D1 train and whether the two files are MD5-identical.

Emits results/d1_md5_audit.json.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "data/datasets/D1/test"
TRAIN = ROOT / "data/datasets/D1/train"
OUT = ROOT / "results/d1_md5_audit.json"
EXTS = {".jpg", ".jpeg", ".png"}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> None:
    train_by_name: dict[str, list[Path]] = {}
    for p in TRAIN.rglob("*"):
        if p.suffix.lower() in EXTS:
            train_by_name.setdefault(p.name, []).append(p)

    test_files = sorted(p for p in TEST.rglob("*") if p.suffix.lower() in EXTS)
    n = len(test_files)
    name_match = md5_identical = 0
    pairs = []
    for p in test_files:
        cands = train_by_name.get(p.name, [])
        if not cands:
            continue
        name_match += 1
        tm = md5(p)
        twin = next((c for c in cands if md5(c) == tm), None)
        if twin is not None:
            md5_identical += 1
            pairs.append({"test": p.name, "train": twin.name, "md5": tm, "identical": True})
        else:
            pairs.append({"test": p.name, "train": cands[0].name,
                          "md5_test": tm, "md5_train": md5(cands[0]), "identical": False})

    result = {
        "experiment": "d1_md5_audit",
        "purpose": "Byte-level (MD5) evidence for D1 train/test leakage (Supplementary Note S1).",
        "test_dir": str(TEST.relative_to(ROOT)),
        "train_dir": str(TRAIN.relative_to(ROOT)),
        "n_test": n,
        "n_test_with_same_filename_in_train": name_match,
        "n_md5_identical": md5_identical,
        "md5_identical_fraction": round(md5_identical / n, 4) if n else 0.0,
        "conclusion": ("D1 test is a byte-identical subset of D1 train"
                       if md5_identical == n else "partial overlap"),
        "pairs": pairs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"n_test={n}  same-filename={name_match}  MD5-identical={md5_identical} "
          f"({result['md5_identical_fraction']:.1%})")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
