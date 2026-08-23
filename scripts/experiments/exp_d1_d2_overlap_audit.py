"""D1 <-> D2 cross-dataset overlap audit (addresses the leakage-defence gap).

The D1 internal leakage (test == train, 350/350) was caught by a hash audit; the
paper's claim that D2 "shares no images with D1" was, until now, asserted from dataset
provenance only. This script verifies it directly with two independent instruments:

  1. BYTE-LEVEL MD5  -> exact (re-uploaded) duplicates.
  2. PERCEPTUAL HASH (pHash, 64-bit) -> re-encoded / resized near-duplicates that MD5
     would miss (Hamming distance over the two galleries; a pair with distance <= T is a
     candidate near-duplicate).

Writes results/d1_d2_overlap_audit.json. Read-only over the datasets; no models loaded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
D1_DIR = ROOT / "data/datasets/D1"     # train + test (2,018)
D2_DIR = ROOT / "data/datasets/D2"     # 3,473
RES = ROOT / "results"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PHASH_NEAR_THRESHOLD = 5               # <= 5 bits of 64 -> flag as near-duplicate candidate


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)


def md5_of(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def phash_bits(path: Path) -> np.ndarray:
    """Return the 64-bit pHash as a length-64 uint8 array (0/1)."""
    with Image.open(path) as im:
        h = imagehash.phash(im.convert("RGB"))  # 8x8 boolean
    return h.hash.flatten().astype(np.uint8)


def main() -> None:
    d1 = list_images(D1_DIR)
    d2 = list_images(D2_DIR)
    print(f"D1={len(d1)}  D2={len(d2)}")

    # ---- 1) byte-level MD5 cross-overlap ----
    d1_md5 = {}
    for p in tqdm(d1, desc="D1 md5"):
        d1_md5.setdefault(md5_of(p), []).append(p.name)
    d2_md5 = {}
    for p in tqdm(d2, desc="D2 md5"):
        d2_md5.setdefault(md5_of(p), []).append(p.name)
    shared_md5 = sorted(set(d1_md5) & set(d2_md5))
    md5_examples = [
        {"md5": h, "d1_files": d1_md5[h][:3], "d2_files": d2_md5[h][:3]}
        for h in shared_md5[:20]
    ]

    # ---- 2) perceptual-hash near-duplicate cross-overlap ----
    d1_bits = np.stack([phash_bits(p) for p in tqdm(d1, desc="D1 pHash")]).astype(np.float32)
    d2_bits = np.stack([phash_bits(p) for p in tqdm(d2, desc="D2 pHash")]).astype(np.float32)
    # Hamming(i,j) = sum(d1_i != d2_j) = d1@(1-d2)^T + (1-d1)@d2^T
    ham = d1_bits @ (1.0 - d2_bits).T + (1.0 - d1_bits) @ d2_bits.T  # (N1, N2)
    min_dist = ham.min(axis=1)                      # nearest D2 image for each D1 image
    argmin = ham.argmin(axis=1)
    n_near = int((min_dist <= PHASH_NEAR_THRESHOLD).sum())
    near_examples = []
    for i in np.argsort(min_dist)[:20]:
        near_examples.append({
            "d1_file": d1[i].name,
            "nearest_d2_file": d2[int(argmin[i])].name,
            "phash_hamming": int(min_dist[i]),
        })

    out = {
        "experiment": "d1_d2_cross_overlap_audit",
        "purpose": "Verify D1 and D2 are disjoint (the leakage-defence assumption).",
        "n_d1": len(d1),
        "n_d2": len(d2),
        "byte_level_md5": {
            "n_shared_md5": len(shared_md5),
            "clean": len(shared_md5) == 0,
            "examples": md5_examples,
        },
        "perceptual_phash": {
            "near_threshold_bits": PHASH_NEAR_THRESHOLD,
            "n_d1_with_d2_near_duplicate": n_near,
            "clean": n_near == 0,
            "min_hamming_observed": int(min_dist.min()),
            "median_min_hamming": float(np.median(min_dist)),
            "closest_pairs": near_examples,
        },
        "overall_disjoint": len(shared_md5) == 0 and n_near == 0,
    }
    RES.mkdir(exist_ok=True)
    (RES / "d1_d2_overlap_audit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("byte_level_md5", "perceptual_phash", "overall_disjoint")},
                     indent=2, default=str)[:1400])


if __name__ == "__main__":
    main()
