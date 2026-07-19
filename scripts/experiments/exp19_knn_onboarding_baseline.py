"""
EXP-19: kNN EUS Onboarding Baseline
Addresses W1: Compare kNN onboarding vs Pipeline onboarding (EXP-15)

Adds n EUS reference images directly to kNN index and measures DA on
56 EUS test images. Directly comparable to EXP-15 pipeline results.
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
    v = visual / (np.linalg.norm(visual, axis=1, keepdims=True) + 1e-8)
    c = caption / (np.linalg.norm(caption, axis=1, keepdims=True) + 1e-8)
    fused = lam * v + (1 - lam) * c
    return fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8)


def knn_classify(query_embs: np.ndarray, ref_embs: np.ndarray,
                 ref_labels: list[str], k: int = 1) -> list[str]:
    sims = query_embs @ ref_embs.T
    predictions = []
    for i in range(len(query_embs)):
        if k == 1:
            predictions.append(ref_labels[np.argmax(sims[i])])
        else:
            top_k_idx = np.argsort(sims[i])[-k:]
            votes = Counter([ref_labels[j] for j in top_k_idx])
            predictions.append(votes.most_common(1)[0][0])
    return predictions


def wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple:
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
    print("EXP-19: kNN EUS Onboarding Baseline")
    print("  Addresses: W1 (kNN vs Pipeline onboarding comparison)")
    print("=" * 70)

    lam = 0.7

    # --- Load D1 KB ---
    print("\n[1/3] Loading data...")
    d1_visual = np.load(CACHE_DIR / "clip_sa_train.npz")["embeddings"]
    d1_caption = np.load(CACHE_DIR / "cap_sa_train.npz")["embeddings"]
    d1_labels = reconstruct_labels(D1_ROOT, "Train", D1_CLASSES)
    d1_fused = fuse_embeddings(d1_visual, d1_caption, lam)

    # --- Load D2 (all EUS images) ---
    d2_visual = np.load(CACHE_DIR / "clip_det_train.npz")["embeddings"]
    d2_caption = np.load(CACHE_DIR / "cap_det_train.npz")["embeddings"]
    d2_labels = reconstruct_labels(D2_ROOT, "train_split", D2_CLASSES)
    d2_fused = fuse_embeddings(d2_visual, d2_caption, lam)

    # Extract EUS indices
    eus_indices = [i for i, l in enumerate(d2_labels) if l == "EUS"]
    eus_fused = d2_fused[eus_indices]
    print(f"  D1 KB: {d1_fused.shape[0]} images, {len(set(d1_labels))} classes")
    print(f"  EUS pool: {len(eus_indices)} images")

    # --- Match EXP-15 protocol ---
    # EXP-15 used: first 50 as ref pool, last 56 as test
    # (sorted by filename within the EUS directory)
    eus_ref_pool = eus_fused[:50]   # Reference pool (to add to KB)
    eus_test = eus_fused[50:106]    # Test set (56 images, matching EXP-15)
    eus_test_size = len(eus_test)
    print(f"  EUS ref pool: {len(eus_ref_pool)}, EUS test: {eus_test_size}")

    # But wait - EXP-15 used 50 ref + 56 test = 106, but we have 400 total
    # Let me check EXP-15's protocol more carefully
    # EXP-15: eus_ref_pool_size=50, eus_test_size=56
    # Total EUS in D2 train_split = 400
    # So they used first 50 for ref, next 56 for test (indices 50-105)

    # Actually, looking at exp15 JSON: it says eus_ref_pool_size=50, eus_test_size=56
    # The total is 400 EUS images. Let's verify by checking if 50+56=106 makes sense.
    # For a fair comparison, we must use the EXACT same split.

    # Load EXP-15 results for comparison
    with open(RESULTS_DIR / "exp15_eus_onboarding.json") as f:
        exp15 = json.load(f)

    print(f"\n  EXP-15 reference: ref_pool={exp15['eus_ref_pool_size']}, "
          f"test={exp15['eus_test_size']}")

    # --- Onboarding curve (matching EXP-15 n values) ---
    print("\n[2/3] Running kNN onboarding curve...")
    n_values = [0, 1, 3, 5, 10, 20, 50]
    onboarding_results = []

    # Also load EXP-15 results for side-by-side comparison
    exp15_curve = {item["n_eus_references"]: item for item in exp15["onboarding_curve"]}

    for n in n_values:
        # Build augmented KB: D1 + n EUS reference images
        if n == 0:
            aug_fused = d1_fused
            aug_labels = d1_labels.copy()
        else:
            # Use deterministic selection: first n from ref pool
            np.random.seed(42)
            ref_indices = np.random.choice(len(eus_ref_pool), size=min(n, len(eus_ref_pool)), replace=False)
            ref_indices = np.sort(ref_indices)
            aug_fused = np.vstack([d1_fused, eus_ref_pool[ref_indices]])
            aug_labels = d1_labels + ["EUS"] * len(ref_indices)

        # Classify EUS test images
        for k in [1, 5]:
            preds = knn_classify(eus_test, aug_fused, aug_labels, k=k)
            correct = sum(p == "EUS" for p in preds)
            da = correct / eus_test_size
            ci = wilson_ci(correct, eus_test_size)
            pred_dist = dict(Counter(preds).most_common())

            # Get EXP-15 pipeline result for comparison
            exp15_item = exp15_curve.get(n, {})
            pipeline_da_k1 = exp15_item.get("eus_da_k1", None)
            pipeline_da_k5 = exp15_item.get("eus_da_k5", None)

            result = {
                "n_eus_references": n,
                "k": k,
                "eus_test_size": eus_test_size,
                "knn_da": round(da, 4),
                "knn_correct": correct,
                "knn_95ci": list(ci),
                "prediction_distribution": pred_dist,
            }
            if k == 1 and pipeline_da_k1 is not None:
                result["pipeline_da"] = pipeline_da_k1
                result["delta_knn_minus_pipeline"] = round(da - pipeline_da_k1, 4)
            elif k == 5 and pipeline_da_k5 is not None:
                result["pipeline_da"] = pipeline_da_k5
                result["delta_knn_minus_pipeline"] = round(da - pipeline_da_k5, 4)

            onboarding_results.append(result)

        # Print k=1 summary
        k1_res = [r for r in onboarding_results if r["n_eus_references"] == n and r["k"] == 1][-1]
        k5_res = [r for r in onboarding_results if r["n_eus_references"] == n and r["k"] == 5][-1]
        pipe_k1 = exp15_curve.get(n, {}).get("eus_da_k1", "N/A")
        pipe_k5 = exp15_curve.get(n, {}).get("eus_da_k5", "N/A")
        print(f"  n={n:>2}: kNN k=1 DA={k1_res['knn_da']:.4f} (pipeline={pipe_k1}), "
              f"kNN k=5 DA={k5_res['knn_da']:.4f} (pipeline={pipe_k5})")

    # --- Also test overall D2 performance with 50 EUS in KB ---
    print("\n[3/3] Overall D2 performance with augmented KB (D1 + 50 EUS)...")
    np.random.seed(42)
    ref_indices = np.random.choice(50, size=50, replace=False)
    ref_indices = np.sort(ref_indices)
    aug_fused_50 = np.vstack([d1_fused, eus_ref_pool[ref_indices]])
    aug_labels_50 = d1_labels + ["EUS"] * 50

    # Test on ALL D2 images (3200) excluding the 50 used as references
    # Remove the 50 ref EUS from query set
    eus_ref_global = set(eus_indices[i] for i in ref_indices)
    d2_test_mask = [i for i in range(len(d2_labels)) if i not in eus_ref_global]
    d2_test_fused = d2_fused[d2_test_mask]
    d2_test_labels = [d2_labels[i] for i in d2_test_mask]

    for k in [1, 5]:
        preds = knn_classify(d2_test_fused, aug_fused_50, aug_labels_50, k=k)
        correct = sum(p == g for p, g in zip(preds, d2_test_labels))
        total = len(d2_test_labels)
        da = correct / total
        ci = wilson_ci(correct, total)

        # Per-class
        class_correct = Counter()
        class_total = Counter()
        for p, g in zip(preds, d2_test_labels):
            class_total[g] += 1
            if p == g:
                class_correct[g] += 1

        print(f"  k={k}: Overall DA={da:.4f} CI{ci} ({correct}/{total})")
        for cls in D2_CLASSES:
            t = class_total.get(cls, 0)
            c = class_correct.get(cls, 0)
            cls_da = c / t if t > 0 else 0
            print(f"    {cls}: DA={cls_da:.4f} ({c}/{t})")

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("COMPARISON: kNN Onboarding vs Pipeline Onboarding (k=1)")
    print("=" * 70)
    print(f"{'n_EUS':>6} {'kNN DA':>8} {'Pipeline DA':>12} {'Δ(kNN-Pipe)':>12} {'kNN 95%CI':>18}")
    print("-" * 60)
    for r in onboarding_results:
        if r["k"] != 1:
            continue
        pipe = r.get("pipeline_da", "N/A")
        delta = r.get("delta_knn_minus_pipeline", "N/A")
        ci_str = f"[{r['knn_95ci'][0]:.3f},{r['knn_95ci'][1]:.3f}]"
        pipe_str = f"{pipe:.4f}" if isinstance(pipe, float) else str(pipe)
        delta_str = f"{delta:+.4f}" if isinstance(delta, float) else str(delta)
        print(f"{r['n_eus_references']:>6} {r['knn_da']:>8.4f} {pipe_str:>12} "
              f"{delta_str:>12} {ci_str:>18}")

    print("\n" + "=" * 70)
    print("COMPARISON: kNN Onboarding vs Pipeline Onboarding (k=5)")
    print("=" * 70)
    print(f"{'n_EUS':>6} {'kNN DA':>8} {'Pipeline DA':>12} {'Δ(kNN-Pipe)':>12}")
    print("-" * 45)
    for r in onboarding_results:
        if r["k"] != 5:
            continue
        pipe = r.get("pipeline_da", "N/A")
        delta = r.get("delta_knn_minus_pipeline", "N/A")
        pipe_str = f"{pipe:.4f}" if isinstance(pipe, float) else str(pipe)
        delta_str = f"{delta:+.4f}" if isinstance(delta, float) else str(delta)
        print(f"{r['n_eus_references']:>6} {r['knn_da']:>8.4f} {pipe_str:>12} {delta_str:>12}")

    # --- Compile output ---
    output = {
        "experiment": "EXP-19",
        "name": "kNN EUS Onboarding Baseline",
        "purpose": "Compare simple kNN onboarding with Pipeline onboarding (EXP-15)",
        "lambda": lam,
        "d1_kb_size": len(d1_labels),
        "eus_ref_pool_size": 50,
        "eus_test_size": eus_test_size,
        "onboarding_curve": onboarding_results,
        "key_findings": [],
        "total_time_s": round(time.time() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Add key findings based on results
    k1_50 = [r for r in onboarding_results if r["n_eus_references"] == 50 and r["k"] == 1][0]
    k1_10 = [r for r in onboarding_results if r["n_eus_references"] == 10 and r["k"] == 1][0]
    output["key_findings"] = [
        f"kNN k=1 with 50 EUS refs: DA={k1_50['knn_da']} vs Pipeline DA={k1_50.get('pipeline_da', 'N/A')}",
        f"kNN k=1 with 10 EUS refs: DA={k1_10['knn_da']} vs Pipeline DA={k1_10.get('pipeline_da', 'N/A')}",
        f"Delta at n=50: {k1_50.get('delta_knn_minus_pipeline', 'N/A')}",
    ]

    out_path = RESULTS_DIR / "exp19_knn_onboarding_baseline.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
