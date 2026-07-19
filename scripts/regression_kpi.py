"""Step-5 KPI regression — evaluate paper headline numbers under both tags.

Run this script TWICE:

  1. Against ``release/fishdx-paper-frozen/`` — pre-fix snapshot (commit b8edc4c
     era).  Expect Decision DA ≈ 0.9627 on D1 Test (n=697), per
     ``docs/m3/p_prime_6a/delta_full_eval/best-config.md``.

  2. Against ``release/fishdx/``                — engineering-main with patches
     #1, #2, #4, #5, #6, #7 applied.  Expect ±0.005 of paper-frozen result.

If you adopted Option A from ``release/TASK1_LINEAGE_FINDING.md`` and
amended the paper's Layer 3 DA from 0.999 to 0.963, the paper target table
below already encodes that decision (Decision DA target = 0.963 ± 0.005).
If you instead pursued Option B / continue searching for the 0.999 commit,
adjust ``PAPER_KPIS["D1_Test_Decision_DA"]["target"]`` accordingly.

Usage
-----
    cd <release-dir>
    # Build the image gallery once (offline, deterministic):
    python scripts/build_image_gallery.py --d1-train <path> --output /tmp/fishdx_gallery
    export FISHDX_GALLERY_PERSIST=/tmp/fishdx_gallery

    python scripts/regression_kpi.py \
        --d1-test     <path>/D1/Test \
        --d2-balanced <path>/D2/balanced \
        --eus-subset  <path>/D2/EUS_56 \
        --eus-onboarding <path>/EUS_for_onboarding \
        --config configs/default.yaml \
        --output regression_report.json

The script writes a JSON report ready for Supplementary Data submission and
returns exit-code 0 iff every reported KPI passes its tolerance check.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# ── Paper headline numbers + tolerance bands ─────────────────────────────────
# Targets reflect Option A erratum (Decision DA = 0.963 instead of 0.999).
# See release/TASK1_LINEAGE_FINDING.md for the rationale.
PAPER_KPIS: dict[str, dict[str, Any]] = {
    "D1_Test_Retrieval_DA":         {"target": 1.000, "tol": 0.005, "comparator": ">="},
    "D1_Test_Decision_DA":          {"target": 0.963, "tol": 0.005, "comparator": "≈"},
    "D2_Balanced_Decisive_DA":      {"target": 0.924, "tol": 0.005, "comparator": "≈"},
    "D2_Balanced_Coverage":         {"target": 0.643, "tol": 0.010, "comparator": "≈"},
    "EUS_Interception_Rate":        {"target": 0.696, "tol": 0.018, "comparator": "≈"},
    "EUS_n50_DA":                   {"target": 0.714, "tol": 0.010, "comparator": "≈"},
    "Median_Latency_ms":            {"target": 533,   "tol": 100,   "comparator": "≈"},
    "Peak_GPU_MB":                  {"target": 921,   "tol": 200,   "comparator": "<="},
}

# Image extensions the loader accepts.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# D1 class folder name → KB doc id.  Adjust to match your local dataset
# directory names if they differ.
D1_CLASS_TO_DOC_ID: dict[str, str] = {
    "Bacterial diseases - Aeromoniasis":   "AER_001",
    "Bacterial gill disease":              "BGD_001",
    "Bacterial Red disease":               "BRD_001",
    "Fungal diseases Saprolegniasis":      "SAP_001",
    "Healthy Fish":                         "HLT_001",
    "Parasitic diseases":                   "PAR_001",
    "Viral diseases White tail disease":    "WTD_001",
}


# ─── helpers ────────────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided Wilson score CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96 if abs(alpha - 0.05) < 1e-9 else _z_from_alpha(alpha)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - margin), min(1.0, center + margin))


def _z_from_alpha(alpha: float) -> float:
    """Inverse-normal one-tail critical value for given two-sided alpha."""
    from scipy.stats import norm
    return float(norm.ppf(1 - alpha / 2))


def _list_class_images(root: Path, class_to_id: dict[str, str]) -> list[tuple[Path, str]]:
    """Return [(image_path, class_label), ...] for every image whose parent
    directory name is a key of ``class_to_id``."""
    images = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.parent.name in class_to_id:
            images.append((p, p.parent.name))
    images.sort()
    return images


# ─── evaluation primitives ──────────────────────────────────────────────────
def _build_pipeline(config_path: Path):
    """Construct fishdx.Pipeline from the given YAML; warm up encoders."""
    from fishdx.config import load_config
    from fishdx.pipeline import Pipeline           # canonical 3-stage orchestrator
    from fishdx.utils.seeding import set_global_seed

    set_global_seed(42)
    cfg = load_config(config_path)
    pipeline = Pipeline(cfg)
    pipeline.warmup()
    return pipeline, cfg


def _evaluate_d1_test(pipeline, d1_test: Path) -> dict[str, float]:
    """D1 Test — forced-choice + Layer-3 Decision DA."""
    images = _list_class_images(d1_test, D1_CLASS_TO_DOC_ID)
    if not images:
        raise RuntimeError(f"No D1 Test images found under {d1_test}")
    retrieval_correct = 0
    decision_correct = 0
    n = len(images)
    for path, cls in images:
        report = pipeline.run(str(path))
        gold_doc_id = D1_CLASS_TO_DOC_ID[cls]
        # Retrieval-level: top-1 doc_id matches the class
        if report.retrieval_top_k and report.retrieval_top_k[0].doc_id.startswith(
            gold_doc_id.split("_")[0]
        ):
            retrieval_correct += 1
        # Decision-level: pipeline decision aligns with the class
        decision_class = (
            "Healthy"
            if cls == "Healthy Fish"
            else "Disease"
        )
        if report.decision == decision_class and (
            decision_class == "Healthy"
            or (report.disease_class and report.disease_class.startswith(gold_doc_id.split("_")[0]))
        ):
            decision_correct += 1
    return {
        "n": float(n),
        "D1_Test_Retrieval_DA": retrieval_correct / n,
        "D1_Test_Decision_DA": decision_correct / n,
    }


def _evaluate_d2_balanced(pipeline, d2_balanced: Path) -> dict[str, float]:
    """D2 — Decisive DA on non-Inconclusive subset; Coverage."""
    # D2 has 8 classes including EUS; for the balanced evaluation, use D2's
    # own class folder names (assumed identical to the D2 ground-truth folder
    # naming convention).  Adjust D2_CLASS_TO_DOC_ID to match your D2 layout.
    images = []
    for p in d2_balanced.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append((p, p.parent.name))
    if not images:
        raise RuntimeError(f"No D2 balanced images found under {d2_balanced}")
    n = len(images)
    decisive_correct = 0
    decisive_total = 0
    for path, cls in images:
        report = pipeline.run(str(path))
        if report.decision == "Inconclusive":
            continue
        decisive_total += 1
        # Match by either Healthy / Disease(D_k); details depend on D2 class
        # folder naming; the user MAY need to adapt this matching rule.
        is_correct = (
            (cls.lower().startswith("healthy") and report.decision == "Healthy")
            or (
                report.decision == "Disease"
                and report.disease_class
                and any(
                    cls.lower().startswith(prefix)
                    for prefix in [report.disease_class.split("_")[0].lower()]
                )
            )
        )
        if is_correct:
            decisive_correct += 1
    decisive_da = (decisive_correct / decisive_total) if decisive_total else 0.0
    coverage = decisive_total / n
    return {
        "n": float(n),
        "D2_Balanced_Decisive_DA": decisive_da,
        "D2_Balanced_Coverage": coverage,
        "D2_Balanced_Inconclusive_pct": 1 - coverage,
    }


def _evaluate_eus_interception(pipeline, eus_subset: Path) -> dict[str, float]:
    """EUS subset — count |margin < θ_margin| as interceptions."""
    images = [p for p in eus_subset.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise RuntimeError(f"No EUS subset images found under {eus_subset}")
    n = len(images)
    intercepted = 0
    for path in images:
        report = pipeline.run(str(path))
        if report.decision == "Inconclusive":
            intercepted += 1
    rate = intercepted / n
    lo, hi = wilson_ci(intercepted, n)
    return {
        "n": float(n),
        "EUS_Interception_Rate": rate,
        "EUS_Interception_count": float(intercepted),
        "EUS_Interception_CI_lo": lo,
        "EUS_Interception_CI_hi": hi,
    }


def _evaluate_eus_onboarding(pipeline, onboarding_root: Path) -> dict[str, float]:
    """Gradient-free EUS onboarding — DA at n_ref = 50.

    Expects ``onboarding_root`` to contain two sibling sub-directories:
        gallery_50_refs/  — the 50 EUS reference images to add to the gallery
        test_56/          — the held-out 56 EUS test images
    """
    gallery = onboarding_root / "gallery_50_refs"
    test = onboarding_root / "test_56"
    if not gallery.exists() or not test.exists():
        return {"EUS_n50_DA": float("nan"), "EUS_n50_n": 0.0}
    pipeline.add_class_references("EUS", list(gallery.glob("*")))
    test_images = [p for p in test.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    n = len(test_images)
    correct = 0
    for path in test_images:
        report = pipeline.run(str(path))
        if report.decision == "Disease" and report.disease_class and "EUS" in report.disease_class:
            correct += 1
    return {
        "n": float(n),
        "EUS_n50_DA": correct / n if n else 0.0,
        "EUS_n50_correct": float(correct),
    }


def _evaluate_latency(pipeline, sample_image: Path, n_warmup: int = 3, n_measure: int = 25) -> dict[str, float]:
    """Median latency + peak GPU memory over n_measure inferences on a single image."""
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        torch = None  # type: ignore
    for _ in range(n_warmup):
        pipeline.run(str(sample_image))
    durations: list[float] = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        pipeline.run(str(sample_image))
        durations.append((time.perf_counter() - t0) * 1000)
    peak_mb = 0.0
    if torch is not None and torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return {
        "Median_Latency_ms": statistics.median(durations),
        "P95_Latency_ms":    sorted(durations)[int(0.95 * len(durations)) - 1],
        "Peak_GPU_MB":        peak_mb,
    }


# ─── orchestration ──────────────────────────────────────────────────────────
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run the full KPI battery and return the observed-numbers dict."""
    pipeline, cfg = _build_pipeline(args.config)
    print(
        f"[KPI regression] config={args.config} "
        f"λ*={cfg.fusion.lambda_star} "
        f"cutoff={cfg.retrieval.similarity_cutoff} "
        f"θ_margin={cfg.margin.retrieval_margin_theta}"
    )
    observed: dict[str, Any] = {
        "config_path": str(args.config),
        "lambda_star": cfg.fusion.lambda_star,
        "similarity_cutoff": cfg.retrieval.similarity_cutoff,
        "theta_margin": cfg.margin.retrieval_margin_theta,
    }

    print("\n[1/4] D1 Test — Retrieval DA + Decision DA")
    observed.update(_evaluate_d1_test(pipeline, args.d1_test))

    print("\n[2/4] D2 Balanced — Decisive DA + Coverage")
    observed.update(_evaluate_d2_balanced(pipeline, args.d2_balanced))

    print("\n[3/4] EUS subset — Interception rate")
    observed.update(_evaluate_eus_interception(pipeline, args.eus_subset))

    if args.eus_onboarding:
        print("\n[3b/4] EUS gradient-free onboarding @ n_ref=50")
        observed.update(_evaluate_eus_onboarding(pipeline, args.eus_onboarding))

    print("\n[4/4] Latency + GPU memory")
    sample = next(args.d1_test.rglob("*.jpg"), None)
    if sample is None:
        print("  [skip] no .jpg sample found for latency test")
    else:
        observed.update(_evaluate_latency(pipeline, sample))

    return observed


def check_kpi(name: str, observed: float, spec: dict[str, Any]) -> tuple[bool, str]:
    target = float(spec["target"])
    tol = float(spec["tol"])
    comp = spec["comparator"]
    if isinstance(observed, float) and math.isnan(observed):
        return False, f"  [SKIP] {name:<28s}  (observed=NaN — likely missing input)"
    delta = observed - target
    if comp == ">=":
        ok = observed >= (target - tol)
    elif comp == "<=":
        ok = observed <= (target + tol)
    else:
        ok = abs(delta) <= tol
    status = "PASS" if ok else "FAIL"
    return ok, (
        f"  [{status}] {name:<28s} obs={observed:>9.4f}  target={target:>9.4f}  "
        f"Δ={delta:+.4f}  (comparator={comp}, tol±{tol})"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="fishdx KPI regression for paper headline numbers")
    p.add_argument("--d1-test",         required=True, type=Path)
    p.add_argument("--d2-balanced",     required=True, type=Path)
    p.add_argument("--eus-subset",      required=True, type=Path)
    p.add_argument("--eus-onboarding",  type=Path, default=None,
                   help="optional: dir with gallery_50_refs/ and test_56/ subdirs")
    p.add_argument("--config",          default=Path("configs/default.yaml"), type=Path)
    p.add_argument("--output",          default=Path("regression_report.json"), type=Path)
    args = p.parse_args()

    t0 = time.time()
    observed = evaluate(args)
    elapsed = time.time() - t0

    print(f"\n── KPI checks (elapsed {elapsed:.1f}s) ──")
    all_pass = True
    for name, spec in PAPER_KPIS.items():
        if name not in observed:
            print(f"  [SKIP] {name:<28s}  (not in observed report)")
            continue
        ok, line = check_kpi(name, observed[name], spec)
        print(line)
        all_pass &= ok

    args.output.write_text(json.dumps({
        "elapsed_s": elapsed,
        "observed": observed,
        "paper_targets": PAPER_KPIS,
        "all_pass": all_pass,
    }, indent=2))
    print(f"\n[KPI regression] report written: {args.output}")
    print(f"[KPI regression] result: {'ALL PASS' if all_pass else 'AT LEAST ONE FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
