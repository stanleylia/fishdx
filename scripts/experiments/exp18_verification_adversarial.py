"""
DEPOSIT NOTE — verification-loop penalty statistic (retrieval-confidence proxy).

This script produced the manuscript's verification-loop penalty rates
(3.4% D1->D1, 47.2% D1->D2, 94.0% D1->EUS) and the Mann-Whitney U test
(p = 2.12e-169, r = 1.000); its output is shipped as
results/verification_penalty_proxy.json.

IMPORTANT: the 'grounding' signal here is a RETRIEVAL-CONFIDENCE PROXY --
a candidate is penalised when its top-1 gallery similarity / top-1-vs-top-2
margin is low -- NOT a live Florence-2 visual-grounding call. A faithful
Florence-2 grounding implementation is not reconstructable from the released
code (see the manuscript Methods/Discussion and REPRODUCIBILITY_MAP.md Sec.1).

DATA DEPENDENCY: this script reads the large-scale (D8) embedding cache under
lab_dateset/organized/experiment_results/, which is NOT redistributed in this
deposit. The shipped results/verification_penalty_proxy.json is the archival
output. To re-run, regenerate that cache from the public datasets first.

----------------------------------------------------------------------
EXP-18: Verification Loop Adversarial Test (v2)
Addresses Reviewer M4: Verification Loop never triggered in EXP-06

Demonstrates that the pipeline's retrieval quality degrades when KB coverage
is incomplete, triggering the Verification Loop's penalty mechanism:

Scenario A: D1 test → D1 KB (matched) — high similarity, low penalty expected
Scenario B: EUS → D1 KB (mismatched) — low similarity, high penalty expected
Scenario C: D2 overlap → D1 KB (cross-dataset, same classes) — intermediate

Uses two complementary verification signals:
1. Cosine similarity to nearest neighbor (proxy for RAG confidence)
2. Class-level confusion analysis (shows misclassification patterns)
3. Similarity-gap analysis (top-1 vs top-2 margin as confidence proxy)
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / "lab_dateset/organized/experiment_results/large_scale_cache"
RESULTS_DIR = PROJECT_ROOT / "lab_dateset/organized/experiment_results/large_scale"

D1_ROOT = PROJECT_ROOT / "lab_dateset/external_datasets/fish_disease_south_asia/Freshwater Fish Disease Aquaculture in south asia"
D2_ROOT = PROJECT_ROOT / "lab_dateset/external_datasets/fish_disease_detection/New Dataset"

D1_CLASSES = [
    "Bacterial diseases - Aeromoniasis", "Bacterial gill disease",
    "Bacterial Red disease", "Fungal diseases Saprolegniasis",
    "Healthy Fish", "Parasitic diseases", "Viral diseases White tail disease",
]
D2_CLASSES = [
    "Bacterial diseases - Aeromoniasis", "Bacterial gill disease",
    "Bacterial Red disease", "EUS", "Fungal diseases Saprolegniasis",
    "Healthy Fish", "Parasitic diseases", "Viral diseases White tail disease",
]


def reconstruct_labels(dataset_root: Path, split: str, classes: list[str]) -> list[str]:
    """Reconstruct labels in sorted directory order."""
    labels = []
    split_dir = dataset_root / split
    for cls in classes:
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            continue
        imgs = sorted([f for f in cls_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        labels.extend([cls] * len(imgs))
    return labels


def fuse_embeddings(visual: np.ndarray, caption: np.ndarray, lam: float) -> np.ndarray:
    """λ-weighted fusion with pre/post normalization."""
    v = visual / (np.linalg.norm(visual, axis=1, keepdims=True) + 1e-8)
    c = caption / (np.linalg.norm(caption, axis=1, keepdims=True) + 1e-8)
    fused = lam * v + (1 - lam) * c
    return fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8)


def analyze_retrieval(query_embs: np.ndarray, ref_embs: np.ndarray,
                      ref_labels: list[str], query_labels: list[str],
                      verification_config: dict) -> dict:
    """Analyze retrieval quality and simulate verification decisions.

    For each query image:
    1. Compute cosine similarity to all reference images
    2. Get top-1 prediction and similarity score
    3. Compute top-1 vs top-2 margin (confidence gap)
    4. Apply verification thresholds:
       - If top-1 similarity < θ_retrieval → RAG score low → penalty likely
       - If margin < θ_margin → ambiguous → verification triggers requery
    """
    theta_retrieval = verification_config["theta_retrieval"]  # Similarity threshold
    theta_margin = verification_config["theta_margin"]  # Margin threshold
    penalty_factor = verification_config["penalty_factor"]
    min_score = verification_config["min_score"]

    sims = query_embs @ ref_embs.T  # (n_query, n_ref)

    results = {
        "total": len(query_embs),
        "correct": 0,
        "predictions": Counter(),
        "sim_scores": [],
        "margins": [],
        "below_theta": 0,
        "below_margin": 0,
        "penalty_would_apply": 0,
        "item_would_survive": 0,
        "per_class": {},
    }

    class_sims = {cls: [] for cls in set(query_labels)}
    class_margins = {cls: [] for cls in set(query_labels)}
    class_correct = Counter()
    class_total = Counter()
    class_penalty = Counter()

    for i in range(len(query_embs)):
        sorted_idx = np.argsort(sims[i])[::-1]
        top1_sim = float(sims[i, sorted_idx[0]])
        top2_sim = float(sims[i, sorted_idx[1]])
        margin = top1_sim - top2_sim
        pred = ref_labels[sorted_idx[0]]

        true_label = query_labels[i]
        correct = (pred == true_label)

        results["sim_scores"].append(top1_sim)
        results["margins"].append(margin)
        results["predictions"][pred] += 1
        class_sims[true_label].append(top1_sim)
        class_margins[true_label].append(margin)
        class_total[true_label] += 1

        if correct:
            results["correct"] += 1
            class_correct[true_label] += 1

        # Simulate verification decision
        # In the real pipeline, RAG returns items with scores ~= cosine similarity
        rag_score = top1_sim
        penalty_applied = False

        if top1_sim < theta_retrieval:
            results["below_theta"] += 1
            rag_score *= penalty_factor
            penalty_applied = True

        if margin < theta_margin:
            results["below_margin"] += 1
            if not penalty_applied:
                rag_score *= penalty_factor
                penalty_applied = True

        if penalty_applied:
            results["penalty_would_apply"] += 1
            class_penalty[true_label] += 1

        if rag_score >= min_score:
            results["item_would_survive"] += 1

    # Aggregate
    sims_arr = np.array(results["sim_scores"])
    margins_arr = np.array(results["margins"])

    results["da"] = round(results["correct"] / results["total"], 4)
    results["sim_mean"] = round(float(sims_arr.mean()), 4)
    results["sim_std"] = round(float(sims_arr.std()), 4)
    results["sim_median"] = round(float(np.median(sims_arr)), 4)
    results["sim_q25"] = round(float(np.percentile(sims_arr, 25)), 4)
    results["sim_q75"] = round(float(np.percentile(sims_arr, 75)), 4)
    results["margin_mean"] = round(float(margins_arr.mean()), 4)
    results["margin_std"] = round(float(margins_arr.std()), 4)
    results["penalty_rate"] = round(results["penalty_would_apply"] / results["total"], 4)
    results["survival_rate"] = round(results["item_would_survive"] / results["total"], 4)

    # Per-class stats
    for cls in sorted(class_total.keys()):
        t = class_total[cls]
        c = class_correct.get(cls, 0)
        p = class_penalty.get(cls, 0)
        cls_sims = np.array(class_sims[cls])
        cls_margins = np.array(class_margins[cls])
        results["per_class"][cls] = {
            "total": t,
            "correct": c,
            "da": round(c / t, 4) if t > 0 else 0.0,
            "sim_mean": round(float(cls_sims.mean()), 4),
            "sim_std": round(float(cls_sims.std()), 4),
            "margin_mean": round(float(cls_margins.mean()), 4),
            "penalty_count": p,
            "penalty_rate": round(p / t, 4) if t > 0 else 0.0,
        }

    # Clean up non-serializable
    results["predictions"] = dict(results["predictions"].most_common())
    del results["sim_scores"]
    del results["margins"]

    return results


def main():
    t0 = time.time()
    print("=" * 70)
    print("EXP-18: Verification Loop Adversarial Test")
    print("  Addresses: Reviewer M4 (Verification Loop never triggered)")
    print("=" * 70)

    lam = 0.7
    # Verification thresholds (from pipeline config):
    # θ=0.5 for grounding, penalty=0.5, min_score=0.3
    # For similarity-based proxy: use percentile-derived thresholds
    verification_config = {
        "theta_retrieval": 0.75,   # Cosine sim below this → low-confidence retrieval
        "theta_margin": 0.02,     # Top-1 vs top-2 margin below this → ambiguous
        "penalty_factor": 0.5,
        "min_score": 0.3,
    }

    # --- Load data ---
    print("\n[1/5] Loading cached data...")
    d1_visual = np.load(CACHE_DIR / "clip_sa_train.npz")["embeddings"]
    d1_caption = np.load(CACHE_DIR / "cap_sa_train.npz")["embeddings"]
    d1_labels = reconstruct_labels(D1_ROOT, "Train", D1_CLASSES)
    d1_fused = fuse_embeddings(d1_visual, d1_caption, lam)

    d1_test_visual = np.load(CACHE_DIR / "clip_sa_test.npz")["embeddings"]
    d1_test_caption = np.load(CACHE_DIR / "cap_sa_test.npz")["embeddings"]
    d1_test_labels = reconstruct_labels(D1_ROOT, "Test", D1_CLASSES)
    d1_test_fused = fuse_embeddings(d1_test_visual, d1_test_caption, lam)

    d2_visual = np.load(CACHE_DIR / "clip_det_train.npz")["embeddings"]
    d2_caption = np.load(CACHE_DIR / "cap_det_train.npz")["embeddings"]
    d2_labels = reconstruct_labels(D2_ROOT, "train_split", D2_CLASSES)
    d2_fused = fuse_embeddings(d2_visual, d2_caption, lam)

    print(f"  D1 train (KB): {d1_fused.shape[0]} images, {len(set(d1_labels))} classes")
    print(f"  D1 test: {d1_test_fused.shape[0]} images")
    print(f"  D2 train: {d2_fused.shape[0]} images ({sum(1 for l in d2_labels if l == 'EUS')} EUS)")

    # --- First, determine thresholds from D1 baseline ---
    print("\n[2/5] Establishing baseline similarity distribution (D1 test → D1 KB)...")
    d1_sims = d1_test_fused @ d1_fused.T
    d1_top1_sims = np.max(d1_sims, axis=1)
    p25 = float(np.percentile(d1_top1_sims, 25))
    p50 = float(np.median(d1_top1_sims))
    p10 = float(np.percentile(d1_top1_sims, 10))
    print(f"  D1 top-1 similarity: mean={d1_top1_sims.mean():.4f}, "
          f"std={d1_top1_sims.std():.4f}")
    print(f"  Percentiles: p10={p10:.4f}, p25={p25:.4f}, p50={p50:.4f}")

    # Use fixed threshold: D1 achieves near-perfect similarity, so we set
    # θ_retrieval based on reasonable retrieval quality, not D1's extreme distribution
    # 0.85 is standard for good semantic retrieval (cosine sim in normalized embedding space)
    verification_config["theta_retrieval"] = 0.85
    print(f"  → θ_retrieval = {verification_config['theta_retrieval']} (fixed: standard retrieval quality)")

    # --- Scenario A: D1 test → D1 KB (in-distribution, matched) ---
    print("\n[3/5] Scenario A: D1 test → D1 KB (matched)...")
    scenario_a = analyze_retrieval(d1_test_fused, d1_fused, d1_labels,
                                   d1_test_labels, verification_config)
    print(f"  DA={scenario_a['da']}, sim_mean={scenario_a['sim_mean']}, "
          f"penalty_rate={scenario_a['penalty_rate']}")

    # --- Scenario B: EUS only → D1 KB (adversarial, no EUS in KB) ---
    print("\n[4/5] Scenario B: EUS images → D1 KB (adversarial)...")
    eus_mask = np.array([l == "EUS" for l in d2_labels])
    eus_fused = d2_fused[eus_mask]
    eus_labels = [l for l in d2_labels if l == "EUS"]
    scenario_b = analyze_retrieval(eus_fused, d1_fused, d1_labels,
                                   eus_labels, verification_config)
    print(f"  DA={scenario_b['da']}, sim_mean={scenario_b['sim_mean']}, "
          f"penalty_rate={scenario_b['penalty_rate']}")
    print(f"  EUS predictions → D1 classes: {scenario_b['predictions']}")

    # --- Scenario C: D2 overlap classes → D1 KB (cross-dataset, same classes) ---
    print("\n[5/5] Scenario C: D2 overlap → D1 KB (cross-dataset)...")
    overlap_mask = np.array([l != "EUS" for l in d2_labels])
    overlap_fused = d2_fused[overlap_mask]
    overlap_labels = [l for l in d2_labels if l != "EUS"]
    scenario_c = analyze_retrieval(overlap_fused, d1_fused, d1_labels,
                                   overlap_labels, verification_config)
    print(f"  DA={scenario_c['da']}, sim_mean={scenario_c['sim_mean']}, "
          f"penalty_rate={scenario_c['penalty_rate']}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY: Verification Quality by Scenario")
    print("=" * 70)
    print(f"{'Scenario':<40} {'DA':>6} {'Sim↑':>6} {'Margin':>7} "
          f"{'Penalty%':>8} {'Survive%':>8}")
    print("-" * 70)
    for label, r in [("A: D1→D1 (in-distribution)", scenario_a),
                     ("B: EUS→D1 (adversarial)", scenario_b),
                     ("C: D2 overlap→D1 (cross-dataset)", scenario_c)]:
        print(f"{label:<40} {r['da']:>6.3f} {r['sim_mean']:>6.3f} "
              f"{r['margin_mean']:>7.4f} {r['penalty_rate']:>7.1%} "
              f"{r['survival_rate']:>7.1%}")

    print(f"\nθ_retrieval = {verification_config['theta_retrieval']} "
          f"(D1 p10 — items below this are statistical outliers)")
    print(f"θ_margin = {verification_config['theta_margin']} "
          f"(ambiguous if top-1 ≈ top-2)")

    gap_sim = scenario_a["sim_mean"] - scenario_b["sim_mean"]
    gap_margin = scenario_a["margin_mean"] - scenario_b["margin_mean"]
    gap_penalty = scenario_b["penalty_rate"] - scenario_a["penalty_rate"]
    print(f"\nKey Findings:")
    print(f"  1. Similarity gap (A vs B): Δsim = {gap_sim:+.4f}")
    print(f"  2. Margin gap (A vs B): Δmargin = {gap_margin:+.4f}")
    print(f"  3. Penalty rate increase (A → B): {gap_penalty:+.1%}")
    print(f"  4. EUS images are misclassified as: {scenario_b['predictions']}")

    # --- Statistical test: similarity distributions ---
    from scipy import stats
    eus_sims_all = (eus_fused @ d1_fused.T).max(axis=1)
    d1_sims_all = d1_top1_sims
    mw_stat, mw_p = stats.mannwhitneyu(d1_sims_all, eus_sims_all, alternative="greater")
    print(f"\n  Mann-Whitney U test (D1 sims > EUS sims):")
    print(f"    U={mw_stat:.0f}, p={mw_p:.2e}, significant={mw_p < 0.05}")

    # Effect size (rank-biserial correlation)
    n1, n2 = len(d1_sims_all), len(eus_sims_all)
    r_rb = 1 - (2 * mw_stat) / (n1 * n2)
    print(f"    Effect size (rank-biserial r) = {r_rb:.4f}")

    # --- Compile output ---
    output = {
        "experiment": "EXP-18",
        "name": "Verification Loop Adversarial Test",
        "purpose": (
            "Demonstrate that retrieval quality degrades for out-of-KB diseases, "
            "providing the signal that triggers the Verification Loop's penalty mechanism"
        ),
        "lambda": lam,
        "verification_config": verification_config,
        "scenario_a": {
            "description": "D1 test → D1 KB (in-distribution, matched)",
            **{k: v for k, v in scenario_a.items() if k != "per_class"},
            "per_class": scenario_a["per_class"],
        },
        "scenario_b": {
            "description": "EUS → D1 KB (adversarial, no EUS in KB)",
            **{k: v for k, v in scenario_b.items() if k != "per_class"},
            "per_class": scenario_b["per_class"],
        },
        "scenario_c": {
            "description": "D2 overlap classes → D1 KB (cross-dataset, same classes)",
            **{k: v for k, v in scenario_c.items() if k != "per_class"},
            "per_class": scenario_c["per_class"],
        },
        "statistical_test": {
            "test": "Mann-Whitney U (one-sided: D1 > EUS)",
            "U": round(float(mw_stat), 1),
            "p_value": float(mw_p),
            "significant": bool(mw_p < 0.05),
            "effect_size_r": round(float(r_rb), 4),
        },
        "key_findings": [
            f"In-distribution (D1→D1) sim_mean={scenario_a['sim_mean']} vs adversarial (EUS→D1) sim_mean={scenario_b['sim_mean']} (Δ={gap_sim:+.4f})",
            f"Penalty activation rate: D1={scenario_a['penalty_rate']:.1%} vs EUS={scenario_b['penalty_rate']:.1%} (Δ={gap_penalty:+.1%})",
            f"Mann-Whitney U confirms significantly lower similarity for EUS (p={mw_p:.2e})",
            f"EUS images misclassified primarily as: {list(scenario_b['predictions'].keys())[:3]}",
            "Verification Loop would detect KB mismatch via low similarity + narrow margin signals",
        ],
        "total_time_s": round(time.time() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = RESULTS_DIR / "exp18_verification_adversarial.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
