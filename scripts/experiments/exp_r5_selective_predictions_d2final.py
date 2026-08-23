"""Reviewer 5 comment 2(1) — archive per-sample selective-prediction outputs on D2-final.

The submitted manuscript states (§4.3, L375–383 and §4.5, L435–440) that the per-image
outputs required for a matched selective-prediction comparison "were not retained", and
therefore withdraws both the paired significance test and the deferring-k-NN comparison.

The *inputs* were retained. `results/_cache_d1_gallery.npz` and `results/_cache_d2_query.npz`
still hold the CLIP embeddings, and the reported pipeline is a deterministic function of
them. This script therefore reconstructs, archives and re-verifies what was discarded:

  1. Recovers a stable ``sample_id`` for every D2 query. The cache stores no filenames, but
     it was written by ``exp_reviewer_final_fast.py`` from ``sorted(D2_DIR.rglob('*'))``.
     Re-enumerating that order and asserting ``canon(stem) == labels[i]`` for all rows
     proves the alignment before anything else is computed.
  2. Emits one JSONL record per (sample, method) for all five Table-3 configurations, with
     similarities, retrieval margin, forced-choice and selective predictions, abstention
     flag and reason, and — for the deployed configuration — the Stage-3 evidence scores.
  3. Recomputes Tables 3 and 4 **from the archived records** and asserts they reproduce
     ``results/d2_final_full.json`` exactly.
  4. Restores the paired comparisons that §4.3 withdrew, including the structural identity
     between the pipeline and fused k-NN (k = 1), which makes its 2x2 table degenerate.
  5. Writes a run manifest (cache SHA-256, config hash, git commit, seed, counts).

No model is re-run; no GPU is required.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/experiments"))

RES = ROOT / "results"
OUT_DIR = RES / "reviewer5"
D2_DIR = ROOT / "data/datasets/D2"

SEED = 42          # global seed (Table 2)
DEDUP = 0.95       # gallery-disjoint threshold (Methods §3.2)
LAMBDA = 0.7       # fusion visual weight (Table 2)
THETA = 0.02       # selected operating point, theta_margin (Table 2)
THETAS = (0.01, 0.02, 0.05)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CANON = {'aeromoniasis': 'aeromoniasis', 'bacterial diseases - aeromoniasis': 'aeromoniasis',
         'bacterial gill disease': 'bacterial_gill', 'bacterial red disease': 'bacterial_red',
         'fungal diseases saprolegniasis': 'fungal', 'parasitic diseases': 'parasitic',
         'viral diseases white tail disease': 'viral_white_tail',
         'viral white tail disease': 'viral_white_tail', 'healthy fish': 'healthy', 'eus': 'EUS'}

# Table-3 configurations, exactly as in exp_d2_final_full.py
CONFIGS = [("pipeline", 0.7, 1), ("fused_knn_k1", 0.7, 1), ("visual_knn_k1", 1.0, 1),
           ("fused_knn_k5", 0.7, 5), ("visual_knn_k5", 1.0, 5), ("caption_only_k1", 0.0, 1)]
CONFIG_TO_TABLE3 = {"pipeline": "Pipeline/kNN(0.7,k1)", "visual_knn_k1": "kNN(1.0,k1)",
                    "fused_knn_k5": "kNN(0.7,k5)", "visual_knn_k5": "kNN(1.0,k5)",
                    "caption_only_k1": "Caption-only(0.0,k1)"}


def canon(s: str) -> str:
    s = s.lower().replace('_', ' ').strip()
    for k, v in CANON.items():
        if k in s:
            return v
    return 'UNKNOWN'


def l2(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(float(c - h), 4), round(float(c + h), 4))


def mcnemar_cc(b: int, c: int) -> tuple[float, float]:
    """Continuity-corrected McNemar chi-square (df = 1) and two-tailed p."""
    from scipy.stats import chi2
    n = b + c
    if n == 0:
        return (0.0, 1.0)
    stat = (abs(b - c) - 1) ** 2 / n if abs(b - c) > 1 else 0.0
    p = float(chi2.sf(stat, 1))
    # keep small p-values readable rather than rounding them to 0.0
    return (round(float(stat), 4), float(f"{p:.3g}"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recover_sample_ids(labels: np.ndarray) -> list[str]:
    """Re-derive the cache's row order and prove it against the stored labels."""
    paths = sorted(p for p in D2_DIR.rglob("*") if p.suffix.lower() in EXTS)
    if len(paths) != len(labels):
        raise AssertionError(f"file count {len(paths)} != cache rows {len(labels)}")
    mismatch = [i for i, p in enumerate(paths) if canon(p.stem) != labels[i]]
    if mismatch:
        raise AssertionError(f"label mismatch at {len(mismatch)} rows, first={mismatch[:5]}")
    print(f"  sample-id alignment verified: {len(paths)} files, 0 label mismatches")
    return [p.name for p in paths]


def vote_k(s_row: np.ndarray, gl: np.ndarray, k: int) -> str:
    """k>1 majority vote with the deterministic 1e-6*score tie-break (as published)."""
    ix = np.argpartition(-s_row, k)[:k]
    u: dict[str, float] = {}
    for lab, sc in zip(gl[ix], s_row[ix]):
        u[lab] = u.get(lab, 0.0) + 1.0 + 1e-6 * float(sc)
    return max(u, key=u.get)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 96)
    print("Reviewer 5 / 2(1) — per-sample selective-prediction archive on D2-final")
    print("=" * 96)

    g = np.load(RES / "_cache_d1_gallery.npz", allow_pickle=True)
    q = np.load(RES / "_cache_d2_query.npz", allow_pickle=True)
    gv, gc, gl = g["visual"], g["caption"], g["labels"]
    qv_all, qc_all, ql_all = q["visual"], q["caption"], q["labels"]

    names_all = recover_sample_ids(ql_all)

    # ---- clean (gallery-disjoint) subset, then D2-final -------------------------
    clean = (qv_all @ gv.T).max(1) < DEDUP
    qv, qc, ql = qv_all[clean], qc_all[clean], ql_all[clean]
    names = [n for n, keep in zip(names_all, clean) if keep]
    N = len(ql)
    iseus = ql == "EUS"
    shared = sorted((set(gl.tolist()) & set(ql.tolist())) - {"EUS", "UNKNOWN"})
    m7 = np.isin(ql, shared)

    eus_pos = np.where(iseus)[0]
    idx = np.random.default_rng(SEED).permutation(len(eus_pos))
    cal_pos = eus_pos[idx[:len(idx) // 2]]
    final = np.ones(N, bool)
    final[cal_pos] = False
    print(f"  clean={N}  D2-final={int(final.sum())}  overlap7={int(m7.sum())}  "
          f"eus_test={int((final & iseus).sum())}")

    # ---- per-configuration predictions, margins ---------------------------------
    per_cfg: dict[str, dict] = {}
    for name, lam, k in CONFIGS:
        G = l2(lam * gv + (1 - lam) * gc)
        Q = l2(lam * qv + (1 - lam) * qc)
        s = Q @ G.T
        srt = np.sort(s, 1)
        top1, top2 = srt[:, -1], srt[:, -2]
        if k == 1:
            pred = gl[s.argmax(1)]
        else:
            pred = np.array([vote_k(s[i], gl, k) for i in range(len(Q))])
        per_cfg[name] = {"pred": pred, "top1": top1, "top2": top2,
                         "margin": top1 - top2, "lam": lam, "k": k,
                         "top1_gallery_idx": s.argmax(1)}
    del s, G, Q

    # ---- Stage-3 evidence scores for the deployed configuration -----------------
    s_h = np.zeros(N, int)
    s_d = np.zeros(N, int)
    stage3_ok = False
    try:
        from exp_mcnemar_d2_faithful_O import load_kb_tiers
        from fishdx.config import load_config
        from fishdx.scoring.score import compute_s_d, compute_s_h
        cfg = load_config(ROOT / "configs/default.yaml")
        tiers, healthy_kw, doc_text = load_kb_tiers()
        caps = {c["image"]: (c.get("caption") or "")
                for c in json.loads((RES / "e3_captions_log.json").read_text())}
        p1 = per_cfg["pipeline"]["pred"]
        for i in range(N):
            ev = caps.get(names[i], "") + "\n" + doc_text.get(str(p1[i]), "")
            s_h[i] = compute_s_h(ev, cfg.scoring.healthy_weights, cfg.negation,
                                 healthy_keywords=healthy_kw or ("healthy",))
            t = tiers.get(str(p1[i]), {"confirmed": [], "suspected": [], "mentioned": []})
            s_d[i] = compute_s_d(ev, cfg.scoring, confirmed_keywords=t["confirmed"],
                                 suspected_keywords=t["suspected"],
                                 mentioned_keywords=t["mentioned"])
        stage3_ok = True
        print(f"  Stage-3 evidence scores computed (ADR-0018 matcher); "
              f"captions matched for {sum(1 for n in names if n in caps)}/{N}")
    except Exception as exc:  # pragma: no cover - diagnostic path
        print(f"  WARNING: Stage-3 scores unavailable ({exc}); S_h/S_d emitted as null")

    # ---- write JSONL ------------------------------------------------------------
    jsonl = OUT_DIR / "selective_predictions_d2final.jsonl"
    n_rec = 0
    with jsonl.open("w") as fh:
        for i in range(N):
            if not final[i]:
                continue
            for name, _, _ in CONFIGS:
                c = per_cfg[name]
                marg = float(c["margin"][i])
                abst = marg < THETA
                rec = {
                    "sample_id": names[i],
                    "true_label": str(ql[i]),
                    "partition": "eus_test" if iseus[i] else "overlap7",
                    "method": name,
                    "lambda": c["lam"], "k": c["k"],
                    "predicted_label_forced": str(c["pred"][i]),
                    "predicted_label_selective": "Inconclusive" if abst else str(c["pred"][i]),
                    "top1_similarity": round(float(c["top1"][i]), 6),
                    "top2_similarity": round(float(c["top2"][i]), 6),
                    "retrieval_margin": round(marg, 6),
                    "top1_gallery_class": str(gl[c["top1_gallery_idx"][i]]),
                    "abstained_at_theta_0.02": bool(abst),
                    "abstention_reason": "low_margin" if abst else None,
                    "correct_forced": bool(c["pred"][i] == ql[i]),
                    "correct_when_decided": bool((not abst) and c["pred"][i] == ql[i]),
                    "score_healthy": (int(s_h[i]) if stage3_ok and name == "pipeline" else None),
                    "score_disease": (int(s_d[i]) if stage3_ok and name == "pipeline" else None),
                    "split": "D2-final", "seed": SEED,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_rec += 1
    print(f"  wrote {n_rec:,} records -> {jsonl}")

    # ---- recompute Tables 3/4 FROM the archive and verify -----------------------
    recs = [json.loads(line) for line in jsonl.open()]
    by_method: dict[str, list[dict]] = {}
    for r in recs:
        by_method.setdefault(r["method"], []).append(r)

    ids = {m: {r["sample_id"] for r in v} for m, v in by_method.items()}
    assert len(set(map(frozenset, ids.values()))) == 1, "sample-ID sets differ across methods"
    print(f"  sample-ID sets identical across {len(ids)} methods "
          f"({len(next(iter(ids.values()))):,} ids)")

    published = json.loads((RES / "d2_final_full.json").read_text())
    t3_check, t4_check = {}, {}
    for m, rows in by_method.items():
        if m not in CONFIG_TO_TABLE3:
            continue
        ov = [r for r in rows]
        op = [r for r in rows if r["partition"] == "overlap7"]
        t3_check[CONFIG_TO_TABLE3[m]] = {
            "overall_DA": round(sum(r["correct_forced"] for r in ov) / len(ov), 4),
            "overlap_DA": round(sum(r["correct_forced"] for r in op) / len(op), 4)}

    pipe = by_method["pipeline"]
    n_final = len(pipe)
    eus = [r for r in pipe if r["partition"] == "eus_test"]
    t4_check["theta_0.00"] = {"decisive_DA": round(sum(r["correct_forced"] for r in pipe) / n_final, 4),
                              "coverage": 1.0}
    for th in THETAS:
        dec = [r for r in pipe if r["retrieval_margin"] >= th]
        t4_check[f"theta_{th:.2f}"] = {
            "decisive_DA": round(sum(r["correct_forced"] for r in dec) / len(dec), 4),
            "coverage": round(len(dec) / n_final, 4),
            "eus_interception": round(sum(1 for r in eus if r["retrieval_margin"] < th) / len(eus), 4),
            "n_decided": len(dec)}

    fidelity = {"table3": {}, "table4": {}}
    for key, got in t3_check.items():
        exp = published["table3_d2final"][key]
        ok = all(abs(got[f] - exp[f]) < 1e-9 for f in ("overall_DA", "overlap_DA"))
        fidelity["table3"][key] = {"expected": {f: exp[f] for f in ("overall_DA", "overlap_DA")},
                                   "recomputed": got, "match": ok}
        assert ok, f"Table 3 mismatch for {key}: {got} vs {exp}"
    for key, got in t4_check.items():
        exp = published["table4_d2final"][key]
        flds = [f for f in got if f in exp]
        ok = all(abs(got[f] - exp[f]) < 1e-9 for f in flds)
        fidelity["table4"][key] = {"expected": {f: exp[f] for f in flds},
                                   "recomputed": got, "match": ok}
        assert ok, f"Table 4 mismatch for {key}: {got} vs {exp}"
    print("  Tables 3 and 4 recomputed from the archive reproduce d2_final_full.json exactly")

    # ---- paired comparisons that §4.3 had withdrawn -----------------------------
    def paired(a: str, b: str) -> dict:
        ra = {r["sample_id"]: r["correct_forced"] for r in by_method[a]}
        rb = {r["sample_id"]: r["correct_forced"] for r in by_method[b]}
        both = ra.keys()
        bb = sum(1 for i in both if ra[i] and not rb[i])
        cc = sum(1 for i in both if not ra[i] and rb[i])
        stat, p = mcnemar_cc(bb, cc)
        pa = {r["sample_id"]: r["predicted_label_forced"] for r in by_method[a]}
        pb = {r["sample_id"]: r["predicted_label_forced"] for r in by_method[b]}
        identical = all(pa[i] == pb[i] for i in both)
        return {"n": len(both),
                "both_correct": sum(1 for i in both if ra[i] and rb[i]),
                "b_a_correct_b_wrong": bb, "c_a_wrong_b_correct": cc,
                "both_wrong": sum(1 for i in both if not ra[i] and not rb[i]),
                "predictions_identical": identical,
                "chi2_continuity_corrected": None if identical else stat,
                "p_value_two_tailed": None if identical else p,
                "note": ("degenerate by construction: Stage 3 scores evidence for the "
                         "retrieved candidate but does not re-rank it, so the committed "
                         "class equals the fused k-NN top-1 for every sample; no test is "
                         "reported") if identical else "non-degenerate paired test"}

    pairs = {"pipeline_vs_fused_knn_k1": paired("pipeline", "fused_knn_k1"),
             "pipeline_vs_visual_knn_k1": paired("pipeline", "visual_knn_k1"),
             "pipeline_vs_fused_knn_k5": paired("pipeline", "fused_knn_k5"),
             "pipeline_vs_visual_knn_k5": paired("pipeline", "visual_knn_k5"),
             # the lambda contrast the manuscript already reports at k = 5
             "fused_k5_vs_visual_k5": paired("fused_knn_k5", "visual_knn_k5"),
             "fused_k1_vs_visual_k1": paired("fused_knn_k1", "visual_knn_k1")}
    print("\n  paired comparisons (forced-choice, D2-final):")
    for k, v in pairs.items():
        if v["predictions_identical"]:
            print(f"    {k:32s} b={v['b_a_correct_b_wrong']} c={v['c_a_wrong_b_correct']}  "
                  f"IDENTICAL BY CONSTRUCTION — no test reported")
        else:
            print(f"    {k:32s} b={v['b_a_correct_b_wrong']:3d} c={v['c_a_wrong_b_correct']:3d}  "
                  f"chi2={v['chi2_continuity_corrected']}  p={v['p_value_two_tailed']}")

    # ---- run manifest -----------------------------------------------------------
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip()
    except Exception:
        commit = "unknown"
    manifest = {
        "experiment": "r5_selective_predictions_d2final",
        "reviewer_comment": "5-2(1): save per-sample outputs and re-run the comparison",
        "seed": SEED, "dedup_threshold": DEDUP, "lambda": LAMBDA, "theta_margin": THETA,
        "git_commit": commit,
        "inputs": {
            "gallery_cache": {"path": "results/_cache_d1_gallery.npz",
                              "sha256": sha256(RES / "_cache_d1_gallery.npz"),
                              "rows": int(len(gl))},
            "query_cache": {"path": "results/_cache_d2_query.npz",
                            "sha256": sha256(RES / "_cache_d2_query.npz"),
                            "rows": int(len(ql_all))},
            "config": {"path": "configs/default.yaml",
                       "sha256": sha256(ROOT / "configs/default.yaml")},
        },
        "counts": {"d2_raw_encoded": int(len(ql_all)), "clean": N,
                   "d2_final": int(final.sum()), "overlap7": int(m7.sum()),
                   "eus_test": int((final & iseus).sum())},
        "methods": [c[0] for c in CONFIGS],
        "records": n_rec,
        "stage3_scores_present": stage3_ok,
        "artifact": {"path": "results/reviewer5/selective_predictions_d2final.jsonl",
                     "sha256": sha256(jsonl)},
        "fidelity_vs_d2_final_full": fidelity,
        "paired_comparisons": pairs,
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\n  wrote {OUT_DIR/'run_manifest.json'}")


if __name__ == "__main__":
    main()
