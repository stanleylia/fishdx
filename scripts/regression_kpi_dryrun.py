"""KPI regression — dry-run with synthetic predictions.

This script does NOT load the real Pipeline. It substitutes the per-image
prediction loop with synthetic outputs that mirror the paper's reported
numbers, so the user can verify:

  (a) the KPI tolerance bands resolve correctly (PASS/FAIL semantics),
  (b) the JSON report shape is acceptable to downstream tooling,
  (c) the comparator logic for ``≈`` / ``>=`` / ``<=`` works as expected,

before scheduling the (expensive) real-data regression on RTX 3090.

Usage
-----
    python scripts/regression_kpi_dryrun.py
    python scripts/regression_kpi_dryrun.py --inject-failure D2_Balanced_Decisive_DA

The second form injects a synthetic 5-pp regression on a chosen KPI to
verify the tolerance-band check actually triggers FAIL.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# Re-use the paper-target table and check_kpi function from the real script.
sys.path.insert(0, str(Path(__file__).parent))
from regression_kpi import PAPER_KPIS, check_kpi  # noqa: E402


SYNTHETIC_OBSERVED: dict[str, float] = {
    "config_path":              "configs/default.yaml",
    "lambda_star":              0.7,
    "similarity_cutoff":        0.5,
    "theta_margin":             0.02,
    # KPIs (mirrored to paper targets — should all PASS)
    "D1_Test_Retrieval_DA":     1.0000,
    "D1_Test_Decision_DA":      0.9627,   # canonical released-code result
    "D2_Balanced_Decisive_DA":  0.9240,
    "D2_Balanced_Coverage":     0.6430,
    "D2_Balanced_Inconclusive_pct": 0.3570,
    "EUS_Interception_Rate":    39 / 56,  # 0.6964 (paper says 69.6%)
    "EUS_Interception_count":   39.0,
    "EUS_Interception_CI_lo":   0.5670,
    "EUS_Interception_CI_hi":   0.8010,
    "EUS_n50_DA":               0.7140,
    "EUS_n50_correct":          40.0,
    "Median_Latency_ms":        533.0,
    "P95_Latency_ms":           611.0,
    "Peak_GPU_MB":              921.0,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="dry-run KPI regression on synthetic numbers")
    ap.add_argument("--inject-failure", default=None, type=str,
                    help="KPI name to perturb by 5 pp (to verify FAIL semantics)")
    ap.add_argument("--output", default=Path("regression_dryrun_report.json"), type=Path)
    args = ap.parse_args()

    observed = dict(SYNTHETIC_OBSERVED)
    if args.inject_failure:
        if args.inject_failure not in observed:
            print(f"[error] unknown KPI: {args.inject_failure!r}", file=sys.stderr)
            print(f"[error] available KPIs: {sorted(PAPER_KPIS.keys())}", file=sys.stderr)
            return 2
        observed[args.inject_failure] = float(observed[args.inject_failure]) - 0.05
        print(f"[inject] perturbed {args.inject_failure} by -0.05 → {observed[args.inject_failure]:.4f}")

    print(f"\n── KPI dry-run checks (synthetic data) ──")
    all_pass = True
    for name, spec in PAPER_KPIS.items():
        if name not in observed:
            print(f"  [SKIP] {name}")
            continue
        ok, line = check_kpi(name, observed[name], spec)
        print(line)
        all_pass &= ok

    args.output.write_text(json.dumps({
        "elapsed_s": 0.0,
        "observed": observed,
        "paper_targets": PAPER_KPIS,
        "all_pass": all_pass,
        "dryrun": True,
    }, indent=2))
    print(f"\n[dryrun] report written: {args.output}")
    print(f"[dryrun] result: {'ALL PASS' if all_pass else 'AT LEAST ONE FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
