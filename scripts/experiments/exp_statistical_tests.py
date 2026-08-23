"""
CI-09: Statistical Tests for All Key Metrics
Computes Clopper-Pearson CI, Bootstrap CI, McNemar Test, Cohen's kappa CI
"""
import json
import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import binom_test
from pathlib import Path

RESULTS_DIR = (Path(__file__).resolve().parents[2] / "lab_dateset/organized/experiment_results/large_scale")

def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial (Clopper-Pearson) confidence interval."""
    if k == 0:
        lo = 0.0
    else:
        lo = beta_dist.ppf(alpha / 2, k, n - k + 1)
    if k == n:
        hi = 1.0
    else:
        hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)

def bootstrap_ci(successes, total, n_bootstrap=10000, alpha=0.05, seed=42):
    """Bootstrap confidence interval for a proportion."""
    rng = np.random.RandomState(seed)
    data = np.array([1]*successes + [0]*(total-successes))
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=total, replace=True)
        means.append(sample.mean())
    means = np.array(means)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi

def mcnemar_exact(b, c):
    """Exact McNemar test (binomial) for discordant pairs b, c."""
    n = b + c
    if n == 0:
        return 1.0
    p_value = 2 * sum(
        binom_test(i, n, 0.5) if False else 0
        for i in range(min(b, c) + 1)
    )
    # Use scipy binom_test properly
    from scipy.stats import binom
    if b <= c:
        p_value = 2 * binom.cdf(b, b + c, 0.5)
    else:
        p_value = 2 * binom.cdf(c, b + c, 0.5)
    return min(float(p_value), 1.0)

def cohens_kappa_ci(kappa, n, alpha=0.05):
    """Approximate CI for Cohen's kappa using Fleiss formula."""
    se = np.sqrt((1 - kappa**2) / (n - 1)) if n > 1 else 0
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    return float(kappa - z * se), float(kappa + z * se)

results = {"experiment": "Statistical Tests for v14 Revision", "tests": {}}

# === 1. EXP-06: DA = 0.999 (1046/1047) ===
k, n = 1046, 1047
cp_lo, cp_hi = clopper_pearson(k, n)
bs_lo, bs_hi = bootstrap_ci(k, n)
results["tests"]["EXP-06_DA"] = {
    "metric": "Diagnostic Accuracy",
    "value": round(k/n, 4),
    "k": k, "n": n,
    "clopper_pearson_95ci": [round(cp_lo, 4), round(cp_hi, 4)],
    "bootstrap_95ci": [round(bs_lo, 4), round(bs_hi, 4)]
}
print(f"EXP-06 DA={k/n:.4f}, CP 95% CI: [{cp_lo:.4f}, {cp_hi:.4f}], BS 95% CI: [{bs_lo:.4f}, {bs_hi:.4f}]")

# === 2. EXP-02: Pareidolia FPR = 0 (0/1503) ===
k, n = 0, 1503
cp_lo, cp_hi = clopper_pearson(k, n)
results["tests"]["EXP-02_FPR"] = {
    "metric": "Pareidolia False Positive Rate",
    "value": 0.0,
    "k": k, "n": n,
    "clopper_pearson_95ci": [round(cp_lo, 6), round(cp_hi, 6)],
    "note": "Upper bound = maximum plausible FPR given 0 false positives in 1503 tests"
}
print(f"EXP-02 FPR=0.000, CP 95% CI: [{cp_lo:.6f}, {cp_hi:.6f}]")

# === 3. EXP-05: Cross-dataset DA = 0.8113 (2596/3200) ===
k, n = 2596, 3200
cp_lo, cp_hi = clopper_pearson(k, n)
bs_lo, bs_hi = bootstrap_ci(k, n)
results["tests"]["EXP-05_DA"] = {
    "metric": "Cross-Dataset Diagnostic Accuracy",
    "value": round(k/n, 4),
    "k": k, "n": n,
    "clopper_pearson_95ci": [round(cp_lo, 4), round(cp_hi, 4)],
    "bootstrap_95ci": [round(bs_lo, 4), round(bs_hi, 4)]
}
print(f"EXP-05 DA={k/n:.4f}, CP 95% CI: [{cp_lo:.4f}, {cp_hi:.4f}]")

# === 4. EXP-01: McNemar λ=0.7 vs λ=0.0 ===
# λ=0.7: 697/697 correct, λ=0.0: 694/697 correct
# Discordant: 3 correct by 0.7 but wrong by 0.0, 0 correct by 0.0 but wrong by 0.7
b, c = 0, 3  # b = wrong by 0.7 & correct by 0.0; c = correct by 0.7 & wrong by 0.0
p_val_07_vs_00 = mcnemar_exact(b, c)
results["tests"]["EXP-01_McNemar_07_vs_00"] = {
    "metric": "McNemar Test: λ=0.7 vs λ=0.0",
    "discordant_b": b, "discordant_c": c,
    "p_value": round(p_val_07_vs_00, 4),
    "significant_005": p_val_07_vs_00 < 0.05,
    "note": "b=correct by λ=0.0 only, c=correct by λ=0.7 only"
}
print(f"EXP-01 McNemar λ=0.7 vs 0.0: b={b}, c={c}, p={p_val_07_vs_00:.4f}")

# λ=0.7 vs λ=1.0: 697/697 vs 695/697
b, c = 0, 2
p_val_07_vs_10 = mcnemar_exact(b, c)
results["tests"]["EXP-01_McNemar_07_vs_10"] = {
    "metric": "McNemar Test: λ=0.7 vs λ=1.0",
    "discordant_b": b, "discordant_c": c,
    "p_value": round(p_val_07_vs_10, 4),
    "significant_005": p_val_07_vs_10 < 0.05,
    "note": "b=correct by λ=1.0 only, c=correct by λ=0.7 only"
}
print(f"EXP-01 McNemar λ=0.7 vs 1.0: b={b}, c={c}, p={p_val_07_vs_10:.4f}")

# === 5. EXP-11: Cohen's κ CI ===
kappa = 0.1441
n_exp11 = 49
kappa_lo, kappa_hi = cohens_kappa_ci(kappa, n_exp11)
results["tests"]["EXP-11_kappa_CI"] = {
    "metric": "Cohen's Kappa",
    "value": kappa,
    "n": n_exp11,
    "approximate_95ci": [round(kappa_lo, 4), round(kappa_hi, 4)],
    "interpretation": "Slight agreement (Landis & Koch scale)"
}
print(f"EXP-11 κ={kappa}, 95% CI: [{kappa_lo:.4f}, {kappa_hi:.4f}]")

# === 6. EXP-11: Agreement Rate CI ===
k, n = 21, 49
cp_lo, cp_hi = clopper_pearson(k, n)
results["tests"]["EXP-11_Agreement_CI"] = {
    "metric": "LLM-Algorithm Agreement Rate",
    "value": round(k/n, 4),
    "k": k, "n": n,
    "clopper_pearson_95ci": [round(cp_lo, 4), round(cp_hi, 4)]
}
print(f"EXP-11 Agreement={k/n:.4f}, CP 95% CI: [{cp_lo:.4f}, {cp_hi:.4f}]")

# === 7. EXP-01: DA=1.000 CI (697/697) ===
k, n = 697, 697
cp_lo, cp_hi = clopper_pearson(k, n)
results["tests"]["EXP-01_DA_lambda07"] = {
    "metric": "DA at λ=0.7",
    "value": 1.0,
    "k": k, "n": n,
    "clopper_pearson_95ci": [round(cp_lo, 4), round(cp_hi, 4)]
}
print(f"EXP-01 DA(λ=0.7)=1.000, CP 95% CI: [{cp_lo:.4f}, {cp_hi:.4f}]")

# === 8. EXP-09: Scoring SI=0 for 3/5 params ===
# All 1047 images correct across all sweep values for 3 params
k, n = 1047, 1047
cp_lo, cp_hi = clopper_pearson(k, n)
results["tests"]["EXP-09_SI0_DA"] = {
    "metric": "DA for SI=0 parameters (all sweep values)",
    "value": 1.0,
    "k": k, "n": n,
    "clopper_pearson_95ci": [round(cp_lo, 4), round(cp_hi, 4)]
}
print(f"EXP-09 SI=0 DA=1.000, CP 95% CI: [{cp_lo:.4f}, {cp_hi:.4f}]")

# === 9. EXP-10: RAG Robustness DA CI ===
k, n = 1046, 1047
results["tests"]["EXP-10_DA"] = {
    "metric": "RAG Robustness DA (KB=57)",
    "value": round(k/n, 4),
    "k": k, "n": n,
    "clopper_pearson_95ci": [round(cp_lo, 4), round(cp_hi, 4)]  # same as EXP-06
}

# Save results
output_path = RESULTS_DIR / "statistical_tests_v14.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nResults saved to {output_path}")
print(f"Total tests computed: {len(results['tests'])}")
