"""Reviewer 5 comment 2(2) — abstention rule for the visual k-NN comparator at matched coverage.

Implements exactly the three variants frozen in ``docs/reviewer5_analysis_protocol.md``
(written and amended before any result for this analysis existed):

  R1  primary   — theta_v is the quantile of the D2-final visual-margin distribution that
                  reproduces the pipeline's coverage. Uses margins only; no labels. This is
                  the "matched coverage" the reviewer asks for. Transductive, and disclosed.
  R2  secondary — split conformal on a seed-42 half of D2-final (calibration), evaluated on
                  the complementary half; the pipeline is scored on the same half.
  R3  secondary — theta_v estimated only on the D1 gallery leave-one-out margin distribution
                  (development-only, disjoint from D2-final) and then locked. Its realised
                  coverage on D2-final is reported as measured and never retuned.

Everything is computed from the per-sample archive written under comment 2(1), plus the D1
gallery cache for R3's calibration. No model is re-run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
R5 = RES / "reviewer5"
OUT = R5 / "matched_coverage_knn.json"

SEED = 42
THETA_MARGIN = 0.02      # pipeline operating point (Table 2)
N_BOOT = 1000            # protocol §4
COVERAGE_TOLERANCE_PP = 1.0   # protocol §5 rule 1


def l2(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 4), round(float(c + h), 4)]


def block(decided: np.ndarray, correct: np.ndarray, n_total: int,
          threshold: float, provenance: str) -> dict:
    nd = int(decided.sum())
    nc = int(correct[decided].sum())
    acc = nc / nd if nd else 0.0
    return {
        "threshold": round(float(threshold), 6),
        "threshold_provenance": provenance,
        "coverage": round(nd / n_total, 4),
        "n_decided": nd,
        "n_correct": nc,
        "selective_accuracy": round(acc, 4),
        "selective_accuracy_wilson95": wilson(nc, nd),
        "selective_risk": round(1 - acc, 4),
        "selective_risk_wilson95": [round(1 - w, 4) for w in reversed(wilson(nc, nd))],
    }


def main() -> None:
    print("=" * 96)
    print("Reviewer 5 / 2(2) — matched-coverage visual k-NN comparator")
    print("  protocol: docs/reviewer5_analysis_protocol.md (frozen before this run)")
    print("=" * 96)

    # ---- per-sample archive from 2(1) ------------------------------------------
    recs = [json.loads(l) for l in (R5 / "selective_predictions_d2final.jsonl").open()]
    pipe = {r["sample_id"]: r for r in recs if r["method"] == "pipeline"}
    vknn = {r["sample_id"]: r for r in recs if r["method"] == "visual_knn_k1"}
    assert pipe.keys() == vknn.keys(), "sample-ID sets differ"
    ids = sorted(pipe)
    n = len(ids)

    p_margin = np.array([pipe[i]["retrieval_margin"] for i in ids])
    p_correct = np.array([pipe[i]["correct_forced"] for i in ids])
    v_margin = np.array([vknn[i]["retrieval_margin"] for i in ids])
    v_correct = np.array([vknn[i]["correct_forced"] for i in ids])

    p_decided = p_margin >= THETA_MARGIN
    cov_target = float(p_decided.mean())
    print(f"  n = {n};  pipeline coverage at theta = {THETA_MARGIN}: {cov_target:.4f}")

    pipeline_block = block(p_decided, p_correct, n, THETA_MARGIN,
                           "fixed a priori on the EUS calibration half (Limitation 6)")

    # ---- R1 (primary): label-free coverage matching on D2-final ----------------
    thr_r1 = float(np.quantile(v_margin, 1 - cov_target))
    r1 = block(v_margin >= thr_r1, v_correct, n, thr_r1,
               "quantile of the D2-final visual-margin distribution reproducing the "
               "pipeline coverage; margins only, no labels (transductive)")
    cov_gap_pp = abs(r1["coverage"] - cov_target) * 100
    r1["coverage_gap_vs_pipeline_pp"] = round(cov_gap_pp, 3)
    r1["within_tolerance"] = bool(cov_gap_pp <= COVERAGE_TOLERANCE_PP)

    # ---- R2: split conformal on an exchangeable seed-42 half --------------------
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    cal_idx, test_idx = perm[: n // 2], perm[n // 2:]
    thr_r2 = float(np.quantile(v_margin[cal_idx], 1 - cov_target))
    v_dec_t = v_margin[test_idx] >= thr_r2
    p_dec_t = p_margin[test_idx] >= THETA_MARGIN
    r2 = block(v_dec_t, v_correct[test_idx], len(test_idx), thr_r2,
               "split conformal: quantile on a seed-42 calibration half of D2-final, "
               "evaluated on the disjoint complementary half")
    r2_pipeline = block(p_dec_t, p_correct[test_idx], len(test_idx), THETA_MARGIN,
                        "pipeline evaluated on the same conformal test half")
    r2["n_calibration"] = int(len(cal_idx))
    r2["n_test_half"] = int(len(test_idx))

    # ---- R3: locked threshold from the D1 gallery LOO margin distribution -------
    g = np.load(RES / "_cache_d1_gallery.npz", allow_pickle=True)
    gvn = l2(g["visual"])
    sg = gvn @ gvn.T
    np.fill_diagonal(sg, -np.inf)          # leave-one-out: exclude self
    srt = np.sort(sg, 1)
    loo_margin = srt[:, -1] - srt[:, -2]
    thr_r3 = float(np.quantile(loo_margin, 1 - cov_target))
    r3 = block(v_margin >= thr_r3, v_correct, n, thr_r3,
               "locked: quantile of the D1 gallery leave-one-out visual-margin "
               "distribution (development-only, disjoint from D2-final); not retuned")
    r3["n_calibration_d1_loo"] = int(len(loo_margin))
    r3["coverage_gap_vs_pipeline_pp"] = round((r3["coverage"] - cov_target) * 100, 3)

    # ---- paired bootstrap of Delta risk (R1 vs pipeline) -----------------------
    # Protocol §4: within each resample the whole procedure is repeated.
    boot_rng = np.random.default_rng(SEED)
    deltas, r1_accs, p_accs = [], [], []
    for _ in range(N_BOOT):
        b = boot_rng.integers(0, n, size=n)
        pm, pc, vm, vc = p_margin[b], p_correct[b], v_margin[b], v_correct[b]
        pd = pm >= THETA_MARGIN
        if pd.sum() == 0:
            continue
        cov_b = pd.mean()
        thr_b = np.quantile(vm, 1 - cov_b)
        vd = vm >= thr_b
        if vd.sum() == 0:
            continue
        pa = pc[pd].mean()
        va = vc[vd].mean()
        p_accs.append(pa)
        r1_accs.append(va)
        deltas.append((1 - va) - (1 - pa))     # Delta risk = comparator - pipeline
    deltas = np.array(deltas)
    boot = {
        "n_resamples": int(len(deltas)),
        "seed": SEED,
        "resampling_unit": "sample (with replacement); threshold re-derived within each resample",
        "delta_risk_mean": round(float(deltas.mean()), 4),
        "delta_risk_ci95": [round(float(np.percentile(deltas, 2.5)), 4),
                            round(float(np.percentile(deltas, 97.5)), 4)],
        "pipeline_selective_accuracy_mean": round(float(np.mean(p_accs)), 4),
        "knn_selective_accuracy_mean": round(float(np.mean(r1_accs)), 4),
        "crosses_zero": bool(np.percentile(deltas, 2.5) <= 0 <= np.percentile(deltas, 97.5)),
        # the percentile interval is discrete near zero; report the mass directly
        "p_delta_risk_ge_0": round(float((deltas >= 0).mean()), 4),
        "p_delta_risk_lt_0": round(float((deltas < 0).mean()), 4),
        "interpretation": "Delta risk < 0 means the deferring visual k-NN has LOWER "
                          "selective risk than the pipeline at matched coverage",
    }

    out = {
        "experiment": "r5_matched_coverage_knn",
        "reviewer_comment": "5-2(2): abstention rule for visual k-NN; risk, accuracy and CIs "
                            "at matched coverage",
        "protocol": "docs/reviewer5_analysis_protocol.md",
        "protocol_status": "time-stamped prospectively specified analysis plan, fixed after "
                           "the reviewer's request and before this re-analysis; NOT a "
                           "pre-registration filed before the study began",
        "source_archive": "results/reviewer5/selective_predictions_d2final.jsonl",
        "models_rerun": False,
        "n_d2final": n,
        "pipeline": pipeline_block,
        "R1_primary_matched_coverage": r1,
        "R2_split_conformal": {"comparator": r2, "pipeline_same_half": r2_pipeline},
        "R3_locked_threshold_d1_loo": r3,
        "paired_bootstrap_R1_vs_pipeline": boot,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    def show(name: str, b: dict) -> None:
        print(f"  {name:34s} cov={b['coverage']:.4f}  n_dec={b['n_decided']:>5}  "
              f"acc={b['selective_accuracy']:.4f} {b['selective_accuracy_wilson95']}  "
              f"risk={b['selective_risk']:.4f}")

    print()
    show("pipeline (theta=0.02)", pipeline_block)
    show("R1 visual k-NN, matched cov", r1)
    print(f"      coverage gap = {r1['coverage_gap_vs_pipeline_pp']} pp "
          f"({'within' if r1['within_tolerance'] else 'OUTSIDE'} the +/-1 pp tolerance)")
    show("R2 conformal (test half)", r2)
    show("R2 pipeline (same half)", r2_pipeline)
    show("R3 locked from D1 LOO", r3)
    print(f"      coverage gap = {r3['coverage_gap_vs_pipeline_pp']} pp "
          f"(reported as measured; threshold not retuned)")
    print(f"\n  paired bootstrap Delta risk (k-NN - pipeline): "
          f"{boot['delta_risk_mean']:+.4f}  95% CI {boot['delta_risk_ci95']}  "
          f"{'crosses zero' if boot['crosses_zero'] else 'excludes zero'}")
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
