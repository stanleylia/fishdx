#!/usr/bin/env python3
"""Full D1 (South-Asia) Test->Train byte-identity (MD5) audit for Supplementary Note S1.

Confirms every D1 Test image is byte-identical to a D1 Train image (train/test leakage),
on the FULL 1,747 Train / 697(698) Test development split — not only the 1,668/350 audit
snapshot. Output: results/d1_full_md5_audit.json.
"""
import hashlib, json
from pathlib import Path

BASE = Path("data/datasets/D1_full")  # Kaggle 'Freshwater Fish Disease Aquaculture in South Asia'
EXT = (".jpg", ".jpeg", ".png")


def md5(p): return hashlib.md5(p.read_bytes()).hexdigest()


def files(sub): return [p for p in (BASE / sub).rglob("*") if p.is_file() and p.suffix.lower() in EXT]


def main():
    train = {}
    for p in files("Train"):
        train.setdefault(md5(p), p.name)
    test = files("Test")
    pairs, ident = [], 0
    for p in test:
        h = md5(p); hit = h in train
        ident += hit
        pairs.append({"test": p.name, "md5": h, "train_match": train.get(h), "byte_identical": hit})
    out = {"experiment": "d1_full_md5_audit", "n_train_unique_md5": len(train),
           "n_test": len(test), "n_test_byte_identical_to_train": ident,
           "conclusion": f"{ident}/{len(test)} D1 Test images are byte-identical to a D1 Train image",
           "pairs": pairs}
    Path("results/d1_full_md5_audit.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(out["conclusion"])


if __name__ == "__main__":
    main()
