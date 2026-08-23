"""Gradient-free onboarding of two novel diseases (Argulus, Oodinium/Velvet).

Demonstrates the paper's "open architecture" claim on real, previously-unseen diseases:
adding a handful of reference images to the gallery (no gradient step, no retraining)
lets the pipeline recognise a new class. Mirrors the EUS onboarding curve (exp15) but with
a clean matched design that avoids the confound flagged in the 2026-07-05 integrity audit:

  * Split by SOURCE CLIP, not by frame — near-duplicate fps-sampled frames of the same
    video never straddle the reference/test boundary (no leakage).
  * The held-out TEST set is FIXED as the reference count k grows — every k is scored on
    the identical test images, the same fused modality (lambda=0.7) and the same top-1
    retrieval. Only k varies. (The earlier 60.1% number varied both modality and test set.)

Reuses cached embeddings from exp_newdx_ood_interception.py (results/_cache_newdx_gallery.npz)
so Florence-2 is not re-run. Gallery base = results/_cache_d1_gallery.npz (D1, 7 classes).
Recognition DA = fraction of held-out frames whose top-1 fused neighbour carries the newly
onboarded class label. k=0 (no onboarding) is 0 by construction (class absent from gallery).
Seed 42. No simulation, no hard-coded outputs.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LAMBDA = 0.7
SEED = 42
K_GRID = [0, 1, 3, 5, 10]
N_TEST_SOURCES = 8          # fixed held-out sources per class (rest form the ref pool)
D1_CACHE = ROOT / "results/_cache_d1_gallery.npz"
NEW_CACHE = ROOT / "results/_cache_newdx_gallery.npz"
OUT = ROOT / "results/newdx_onboarding.json"


def l2(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def fused(cache: dict, idx: np.ndarray | None = None) -> np.ndarray:
    v, c = cache["visual"], cache["caption"]
    if idx is not None:
        v, c = v[idx], c[idx]
    return l2(LAMBDA * v + (1 - LAMBDA) * c)


def onboard_curve(cls: str, new: dict, g_fus: np.ndarray, g_lab: np.ndarray, rng) -> dict:
    mask = new["labels"] == cls
    idx = np.where(mask)[0]
    srcs = new["srcs"][idx]
    uniq = sorted(set(srcs.tolist()))
    rng.shuffle(uniq)
    if len(uniq) <= N_TEST_SOURCES:
        raise ValueError(f"{cls}: only {len(uniq)} sources, need > {N_TEST_SOURCES}")
    test_srcs = set(uniq[:N_TEST_SOURCES])
    pool_srcs = uniq[N_TEST_SOURCES:]                      # ref pool grows from here

    test_idx = idx[np.isin(srcs, list(test_srcs))]
    new_fus_all = fused(new)
    q = new_fus_all[test_idx]                              # FIXED test set for all k
    curve = []
    for k in K_GRID:
        ref_srcs = pool_srcs[:k]
        if k == 0:
            gf, gl = g_fus, g_lab
        else:
            ref_idx = idx[np.isin(srcs, ref_srcs)]
            gf = np.concatenate([g_fus, new_fus_all[ref_idx]], axis=0)
            gl = np.concatenate([g_lab, np.array([cls] * len(ref_idx))], axis=0)
        sims = q @ gf.T
        top1 = gl[sims.argmax(axis=1)]
        n_ok = int((top1 == cls).sum())
        curve.append({
            "k_ref_sources": k,
            "n_ref_frames": 0 if k == 0 else int(len(ref_idx)),
            "recognition_DA": round(n_ok / len(q), 4),
            "n_correct": n_ok, "n_test": int(len(q)),
            "wilson_ci95": list(wilson_ci(n_ok, len(q))),
        })
    return {
        "n_total_sources": len(uniq),
        "n_test_sources": N_TEST_SOURCES,
        "n_test_frames": int(len(test_idx)),
        "test_sources_are_fixed_across_k": True,
        "curve": curve,
    }


def main() -> None:
    print("=" * 72)
    print("Gradient-free onboarding of novel diseases (source-split, fixed test set)")
    print("=" * 72)
    d1 = np.load(D1_CACHE, allow_pickle=True)
    g_fus = fused({"visual": d1["visual"], "caption": d1["caption"]})
    g_lab = d1["labels"].astype("<U16")
    new = {k: np.load(NEW_CACHE, allow_pickle=True)[k] for k in
           ["visual", "caption", "labels", "hosts", "srcs", "files"]}

    result = {
        "experiment": "newdx_onboarding",
        "purpose": "Gradient-free onboarding curve for two novel diseases; clean matched "
                   "design (source-level split, fixed held-out test set as k grows).",
        "base_gallery": "results/_cache_d1_gallery.npz (D1, 7 classes)",
        "lambda": LAMBDA, "seed": SEED, "k_grid": K_GRID,
        "metric": "recognition_DA = fraction of held-out frames whose top-1 fused neighbour "
                  "carries the onboarded class label",
        "by_class": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    for cls in ["argulus", "oodinium"]:
        rng = np.random.default_rng(SEED)
        res = onboard_curve(cls, new, g_fus, g_lab, rng)
        result["by_class"][cls] = res
        pts = "  ".join(f"k={c['k_ref_sources']}:{c['recognition_DA']:.3f}" for c in res["curve"])
        print(f"  {cls:9s} (test={res['n_test_frames']} frames / {res['n_test_sources']} src): {pts}")

    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"  saved -> {OUT}")
    print("=" * 72)


if __name__ == "__main__":
    main()
