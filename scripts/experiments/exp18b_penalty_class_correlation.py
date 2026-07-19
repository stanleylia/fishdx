"""
EXP-18b: Per-Class Penalty Correlation Analysis
Supplements EXP-18: Correlates penalty rate with cross-dataset DA

Checks if the classes that get penalized most frequently in verification
are the same classes that perform worst in cross-dataset transfer (EXP-05).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "lab_dateset/organized/experiment_results/large_scale"


def main():
    t0 = time.time()
    print("=" * 70)
    print("EXP-18b: Per-Class Penalty Correlation Analysis")
    print("  Supplements EXP-18: Penalty rate vs Cross-dataset DA correlation")
    print("=" * 70)

    # --- Load EXP-18 results ---
    with open(RESULTS_DIR / "exp18_verification_adversarial.json") as f:
        exp18 = json.load(f)

    # --- Load EXP-05 cross-dataset results ---
    with open(RESULTS_DIR / "exp05_cross_dataset.json") as f:
        exp05 = json.load(f)

    # --- Load EXP-16 cross-dataset baseline ---
    with open(RESULTS_DIR / "exp16_cross_dataset_baseline.json") as f:
        exp16 = json.load(f)

    # Extract per-class data from EXP-18 Scenario C (D2 overlap → D1)
    print("\n[1/3] Extracting per-class penalty rates from EXP-18 Scenario C...")
    scenario_c = exp18["scenario_c"]
    per_class_c = scenario_c["per_class"]

    # Extract per-class DA from EXP-16 (λ=0.7, k=1 — matching EXP-05)
    print("\n[2/3] Extracting per-class DA from EXP-16...")
    exp16_config = None
    for cfg in exp16["configurations"]:
        if cfg["lambda"] == 0.7 and cfg["k"] == 1:
            exp16_config = cfg
            break

    if exp16_config is None:
        print("ERROR: Could not find λ=0.7 k=1 in EXP-16")
        return

    # --- Build correlation table ---
    print("\n[3/3] Building correlation table...")
    overlap_classes = [
        "Bacterial diseases - Aeromoniasis",
        "Bacterial gill disease",
        "Bacterial Red disease",
        "Fungal diseases Saprolegniasis",
        "Healthy Fish",
        "Parasitic diseases",
        "Viral diseases White tail disease",
    ]

    class_data = []
    print(f"\n{'Class':<40} {'Penalty%':>8} {'SimMean':>8} {'DA(EXP16)':>10} {'Margin':>8}")
    print("-" * 78)

    for cls in overlap_classes:
        penalty_info = per_class_c.get(cls, {})
        exp16_class = exp16_config["class_metrics"].get(cls, {})

        penalty_rate = penalty_info.get("penalty_rate", 0)
        sim_mean = penalty_info.get("sim_mean", 0)
        margin_mean = penalty_info.get("margin_mean", 0)

        # DA from EXP-16 per-class
        cls_tp = exp16_class.get("tp", 0)
        cls_support = exp16_class.get("support", 0)
        cls_da = cls_tp / cls_support if cls_support > 0 else 0

        class_data.append({
            "class": cls,
            "penalty_rate": penalty_rate,
            "sim_mean": sim_mean,
            "margin_mean": margin_mean,
            "cross_da": round(cls_da, 4),
        })

        print(f"{cls:<40} {penalty_rate:>7.1%} {sim_mean:>8.4f} {cls_da:>9.4f} {margin_mean:>8.4f}")

    # --- Compute correlations ---
    penalty_rates = np.array([d["penalty_rate"] for d in class_data])
    cross_das = np.array([d["cross_da"] for d in class_data])
    sim_means = np.array([d["sim_mean"] for d in class_data])
    margin_means = np.array([d["margin_mean"] for d in class_data])

    from scipy import stats

    # Penalty rate vs DA (expect negative correlation)
    r_penalty_da, p_penalty_da = stats.pearsonr(penalty_rates, cross_das)
    rho_penalty_da, p_rho_penalty_da = stats.spearmanr(penalty_rates, cross_das)

    # Similarity vs DA (expect positive correlation)
    r_sim_da, p_sim_da = stats.pearsonr(sim_means, cross_das)

    # Margin vs DA (expect positive correlation)
    r_margin_da, p_margin_da = stats.pearsonr(margin_means, cross_das)

    print(f"\nCorrelations (n={len(overlap_classes)} classes):")
    print(f"  Penalty rate ↔ Cross-DA: r={r_penalty_da:.4f} (p={p_penalty_da:.4f}), "
          f"ρ={rho_penalty_da:.4f} (p={p_rho_penalty_da:.4f})")
    print(f"  Similarity   ↔ Cross-DA: r={r_sim_da:.4f} (p={p_sim_da:.4f})")
    print(f"  Margin       ↔ Cross-DA: r={r_margin_da:.4f} (p={p_margin_da:.4f})")

    print(f"\nInterpretation:")
    if r_penalty_da < -0.5:
        print(f"  Strong negative correlation (r={r_penalty_da:.3f}): classes with "
              f"higher penalty rates have lower cross-dataset DA → Verification Loop "
              f"correctly identifies unreliable predictions")
    elif r_penalty_da < 0:
        print(f"  Weak negative correlation (r={r_penalty_da:.3f}): penalty rate is "
              f"directionally aligned with prediction difficulty")
    else:
        print(f"  No negative correlation (r={r_penalty_da:.3f}): penalty rate does "
              f"not strongly predict cross-dataset difficulty at class level")

    # --- Compile output ---
    output = {
        "experiment": "EXP-18b",
        "name": "Per-Class Penalty Correlation Analysis",
        "purpose": "Correlate verification penalty rate with cross-dataset classification accuracy",
        "class_analysis": class_data,
        "correlations": {
            "penalty_vs_da": {
                "pearson_r": round(float(r_penalty_da), 4),
                "pearson_p": round(float(p_penalty_da), 4),
                "spearman_rho": round(float(rho_penalty_da), 4),
                "spearman_p": round(float(p_rho_penalty_da), 4),
            },
            "similarity_vs_da": {
                "pearson_r": round(float(r_sim_da), 4),
                "pearson_p": round(float(p_sim_da), 4),
            },
            "margin_vs_da": {
                "pearson_r": round(float(r_margin_da), 4),
                "pearson_p": round(float(p_margin_da), 4),
            },
        },
        "total_time_s": round(time.time() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = RESULTS_DIR / "exp18b_penalty_class_correlation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
