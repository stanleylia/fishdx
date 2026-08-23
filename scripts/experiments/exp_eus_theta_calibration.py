"""EUS theta_margin calibration/test split (addresses the theta-selection circularity).

theta_margin was previously selected on the same EUS OOD images whose
interception is reported. Here the clean EUS set (D1-disjoint, n=452) is
split 50/50 (seed=42) into a calibration half (used to select theta_margin
under the >=65% interception criterion) and a disjoint test half (on which
interception is reported). Faithful to exp_d2_clean_tables.py: fused
lambda=0.7 retrieval against the D1 gallery, margin = top1_sim - top2_sim,
interception = fraction of EUS with margin < theta.

Writes results/eus_theta_calibration.json.
"""
from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
SEED = 42
DEDUP_COS = 0.95
THETA_SWEEP = [0.01, 0.02, 0.05]
INTERCEPTION_TARGET = 0.65


def l2(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


def main() -> None:
    g = np.load(RES / "_cache_d1_gallery.npz", allow_pickle=True)
    q = np.load(RES / "_cache_d2_query.npz", allow_pickle=True)
    gv, gc, gl = g["visual"], g["caption"], g["labels"]
    qv, qc, ql = q["visual"], q["caption"], q["labels"]

    clean = (qv @ gv.T).max(axis=1) < DEDUP_COS
    qv, qc, ql = qv[clean], qc[clean], ql[clean]
    iseus = ql == "EUS"

    gf = l2(0.7 * gv + 0.3 * gc)
    qf = l2(0.7 * qv + 0.3 * qc)
    sims = qf @ gf.T
    order = np.sort(sims, 1)
    margin = order[:, -1] - order[:, -2]
    eus_margin = margin[iseus]

    def interc(m: np.ndarray, th: float) -> float:
        return float((m < th).mean())

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(eus_margin))
    half = len(idx) // 2
    cal, tst = eus_margin[idx[:half]], eus_margin[idx[half:]]

    theta_star = next((th for th in THETA_SWEEP if interc(cal, th) >= INTERCEPTION_TARGET), None)
    n_int = int((tst < theta_star).sum())
    out = {
        "seed": SEED,
        "n_eus_clean": int(iseus.sum()),
        "full_set_interception": {f"theta_{th}": round(interc(eus_margin, th), 4) for th in THETA_SWEEP},
        "calibration": {
            "n": int(half),
            "interception": {f"theta_{th}": round(interc(cal, th), 4) for th in THETA_SWEEP},
            "criterion": f"smallest theta with interception >= {INTERCEPTION_TARGET}",
            "theta_star": theta_star,
        },
        "test": {
            "n": int(len(tst)),
            "theta_star": theta_star,
            "interception": round(n_int / len(tst), 4),
            "intercepted": n_int,
            "wilson_ci95": wilson(n_int, len(tst)),
        },
    }
    (RES / "eus_theta_calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
