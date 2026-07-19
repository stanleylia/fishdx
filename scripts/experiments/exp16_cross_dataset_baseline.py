"""
EXP-16: Cross-Dataset Baseline Comparison (D1 → D2)
Addresses Reviewer M1 + M2: kNN baseline on D2 + Fusion statistical significance

Compares λ=1.0 (pure CLIP visual), λ=0.0 (caption-only), λ=0.7 (fusion)
on D1→D2 cross-dataset task. Includes McNemar test for fusion significance.
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
    "Bacterial diseases - Aeromoniasis",
    "Bacterial gill disease",
    "Bacterial Red disease",
    "Fungal diseases Saprolegniasis",
    "Healthy Fish",
    "Parasitic diseases",
    "Viral diseases White tail disease",
]

D2_CLASSES = [
    "Bacterial diseases - Aeromoniasis",
    "Bacterial gill disease",
    "Bacterial Red disease",
    "EUS",
    "Fungal diseases Saprolegniasis",
    "Healthy Fish",
    "Parasitic diseases",
    "Viral diseases White tail disease",
]


def reconstruct_labels(dataset_root: Path, split: str, classes: list[str]) -> list[str]:
    """Reconstruct labels in the same order as exp_large_scale.py loaded them."""
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
    """λ-weighted fusion with pre/post normalization (Eq.3-5)."""
    # Pre-normalize
    v_norm = visual / (np.linalg.norm(visual, axis=1, keepdims=True) + 1e-8)
    c_norm = caption / (np.linalg.norm(caption, axis=1, keepdims=True) + 1e-8)
    # Fuse
    fused = lam * v_norm + (1.0 - lam) * c_norm
    # Post-normalize
    fused = fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8)
    return fused


def knn_classify(query_embs: np.ndarray, ref_embs: np.ndarray,
                 ref_labels: list[str], k: int = 1) -> list[str]:
    """kNN classification with majority voting."""
    sims = query_embs @ ref_embs.T
    predictions = []
    for i in range(len(query_embs)):
        if k == 1:
            pred_idx = np.argmax(sims[i])
            predictions.append(ref_labels[pred_idx])
        else:
            top_k_idx = np.argsort(sims[i])[-k:]
            votes = Counter([ref_labels[j] for j in top_k_idx])
            predictions.append(votes.most_common(1)[0][0])
    return predictions


def compute_metrics(predictions: list[str], ground_truth: list[str],
                    classes: list[str]) -> dict:
    """Compute DA, per-class P/R/F1, macro metrics."""
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    total = len(ground_truth)
    da = correct / total if total > 0 else 0.0

    per_class = {c: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for c in classes}
    for pred, true in zip(predictions, ground_truth):
        per_class[true]["support"] += 1
        if pred == true:
            per_class[true]["tp"] += 1
        else:
            per_class[true]["fn"] += 1
            if pred in per_class:
                per_class[pred]["fp"] += 1

    class_metrics = {}
    macro_p, macro_r, macro_f1 = [], [], []
    for cls in classes:
        tp = per_class[cls]["tp"]
        fp = per_class[cls]["fp"]
        fn = per_class[cls]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        class_metrics[cls] = {
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "support": per_class[cls]["support"],
            "tp": tp, "fp": fp, "fn": fn,
        }
        macro_p.append(p)
        macro_r.append(r)
        macro_f1.append(f1)

    return {
        "da": round(da, 4),
        "correct": correct,
        "total": total,
        "macro_precision": round(float(np.mean(macro_p)), 4),
        "macro_recall": round(float(np.mean(macro_r)), 4),
        "macro_f1": round(float(np.mean(macro_f1)), 4),
        "class_metrics": class_metrics,
    }


def mcnemar_test(pred_a: list[str], pred_b: list[str], ground_truth: list[str]) -> dict:
    """McNemar test comparing two classifiers."""
    n = len(ground_truth)
    # a_correct & b_wrong, a_wrong & b_correct
    b_c = 0  # a correct, b wrong
    c_b = 0  # a wrong, b correct
    both_correct = 0
    both_wrong = 0
    for i in range(n):
        a_ok = pred_a[i] == ground_truth[i]
        b_ok = pred_b[i] == ground_truth[i]
        if a_ok and b_ok:
            both_correct += 1
        elif a_ok and not b_ok:
            b_c += 1
        elif not a_ok and b_ok:
            c_b += 1
        else:
            both_wrong += 1

    # McNemar statistic (with continuity correction)
    discordant = b_c + c_b
    if discordant == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(b_c - c_b) - 1) ** 2 / (b_c + c_b)
        from scipy import stats
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return {
        "a_correct_b_wrong": b_c,
        "a_wrong_b_correct": c_b,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "discordant_pairs": discordant,
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "significant_0.05": bool(p_value < 0.05),
    }


def wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval."""
    if total == 0:
        return (0.0, 0.0)
    p_hat = correct / total
    denom = 1 + z**2 / total
    center = (p_hat + z**2 / (2 * total)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom
    return (round(max(0, center - margin), 4), round(min(1, center + margin), 4))


def main():
    t0 = time.time()
    print("=" * 70)
    print("EXP-16: Cross-Dataset Baseline Comparison (D1 → D2)")
    print("  Addresses: Reviewer M1 (kNN on D2) + M2 (Fusion significance)")
    print("=" * 70)

    # --- Load cached embeddings ---
    print("\n[1/4] Loading cached embeddings...")
    d1_visual = np.load(CACHE_DIR / "clip_sa_train.npz")["embeddings"]  # (1747, 512)
    d1_caption = np.load(CACHE_DIR / "cap_sa_train.npz")["embeddings"]  # (1747, 512)
    d2_visual = np.load(CACHE_DIR / "clip_det_train.npz")["embeddings"]  # (3200, 512)
    d2_caption = np.load(CACHE_DIR / "cap_det_train.npz")["embeddings"]  # (3200, 512)

    print(f"  D1 train: {d1_visual.shape[0]} images (visual + caption)")
    print(f"  D2 test:  {d2_visual.shape[0]} images (visual + caption)")

    # --- Reconstruct labels ---
    print("\n[2/4] Reconstructing labels from directory structure...")
    d1_labels = reconstruct_labels(D1_ROOT, "Train", D1_CLASSES)
    d2_labels = reconstruct_labels(D2_ROOT, "train_split", D2_CLASSES)
    print(f"  D1 labels: {len(d1_labels)} ({Counter(d1_labels).most_common()})")
    print(f"  D2 labels: {len(d2_labels)} ({Counter(d2_labels).most_common()})")

    assert len(d1_labels) == d1_visual.shape[0], f"D1 label count mismatch: {len(d1_labels)} vs {d1_visual.shape[0]}"
    assert len(d2_labels) == d2_visual.shape[0], f"D2 label count mismatch: {len(d2_labels)} vs {d2_visual.shape[0]}"

    # --- Run experiments across lambda values ---
    print("\n[3/4] Running cross-dataset kNN experiments...")
    lambdas = [0.0, 0.3, 0.5, 0.7, 1.0]
    results = {"experiment": "EXP-16", "name": "Cross-Dataset Baseline (D1→D2)",
               "train_dataset": "D1 (fish_disease_south_asia)", "train_images": len(d1_labels),
               "test_dataset": "D2 (fish_disease_detection)", "test_images": len(d2_labels),
               "configurations": []}

    all_predictions = {}  # store for McNemar comparison

    for lam in lambdas:
        print(f"\n--- λ = {lam} ---")
        # Compute fused embeddings for reference (D1) and query (D2)
        ref_embs = fuse_embeddings(d1_visual, d1_caption, lam)
        query_embs = fuse_embeddings(d2_visual, d2_caption, lam)

        for k in [1, 5]:
            label = f"λ={lam}_k={k}"
            preds = knn_classify(query_embs, ref_embs, d1_labels, k=k)
            all_predictions[label] = preds

            # Overall metrics (8 classes)
            overall = compute_metrics(preds, d2_labels, D2_CLASSES)
            ci = wilson_ci(overall["correct"], overall["total"])

            # Overlap metrics (7 classes, exclude EUS)
            overlap_preds = [p for p, g in zip(preds, d2_labels) if g != "EUS"]
            overlap_gt = [g for g in d2_labels if g != "EUS"]
            overlap = compute_metrics(overlap_preds, overlap_gt, D1_CLASSES)
            overlap_ci = wilson_ci(overlap["correct"], overlap["total"])

            # EUS-specific
            eus_preds = [p for p, g in zip(preds, d2_labels) if g == "EUS"]
            eus_correct = sum(p == "EUS" for p in eus_preds)
            eus_total = len(eus_preds)

            config = {
                "lambda": lam,
                "k": k,
                "overall_da": overall["da"],
                "overall_95ci": list(ci),
                "overall_correct": overall["correct"],
                "overall_total": overall["total"],
                "overall_macro_f1": overall["macro_f1"],
                "overlap_da": overlap["da"],
                "overlap_95ci": list(overlap_ci),
                "overlap_correct": overlap["correct"],
                "overlap_total": overlap["total"],
                "eus_da": round(eus_correct / eus_total, 4) if eus_total > 0 else 0.0,
                "eus_correct": eus_correct,
                "eus_total": eus_total,
                "class_metrics": overall["class_metrics"],
            }
            results["configurations"].append(config)
            print(f"  λ={lam} k={k}: Overall DA={overall['da']:.4f} CI{ci}, "
                  f"Overlap DA={overlap['da']:.4f}, EUS={eus_correct}/{eus_total}")

    # --- McNemar tests ---
    print("\n[4/4] Running McNemar tests...")
    mcnemar_results = []
    comparisons = [
        ("λ=0.7_k=1", "λ=1.0_k=1", "Fusion vs Visual-only (k=1)"),
        ("λ=0.7_k=1", "λ=0.0_k=1", "Fusion vs Caption-only (k=1)"),
        ("λ=0.7_k=5", "λ=1.0_k=5", "Fusion vs Visual-only (k=5)"),
        ("λ=0.7_k=1", "λ=0.7_k=5", "k=1 vs k=5 (λ=0.7)"),
    ]
    for a_label, b_label, desc in comparisons:
        if a_label in all_predictions and b_label in all_predictions:
            mc = mcnemar_test(all_predictions[a_label], all_predictions[b_label], d2_labels)
            mc["comparison"] = desc
            mc["a"] = a_label
            mc["b"] = b_label
            mcnemar_results.append(mc)
            sig = "***" if mc["significant_0.05"] else "n.s."
            print(f"  {desc}: χ²={mc['chi2']}, p={mc['p_value']}, "
                  f"discordant={mc['discordant_pairs']} [{sig}]")

    results["mcnemar_tests"] = mcnemar_results
    results["total_time_s"] = round(time.time() - t0, 1)
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY TABLE — Cross-Dataset D1→D2 (n=3,200)")
    print("=" * 70)
    print(f"{'Config':<20} {'Overall DA':>10} {'95% CI':>16} {'Overlap DA':>10} {'EUS DA':>8}")
    print("-" * 70)
    for c in results["configurations"]:
        lbl = f"λ={c['lambda']} k={c['k']}"
        ci_str = f"[{c['overall_95ci'][0]:.3f},{c['overall_95ci'][1]:.3f}]"
        print(f"{lbl:<20} {c['overall_da']:>10.4f} {ci_str:>16} {c['overlap_da']:>10.4f} {c['eus_da']:>8.4f}")

    # --- Save ---
    out_path = RESULTS_DIR / "exp16_cross_dataset_baseline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
