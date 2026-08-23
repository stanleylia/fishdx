"""EXP-02: Two-Tier Pareidolia Detection Ablation (Eq.1a, 1b, 1, 2).

Hypothesis: Hard+Soft detection outperforms any single-tier mode.

Tests 4 detection modes:
  - None:       No pareidolia detection (all labels pass through)
  - Hard-only:  Tier 1 keyword matching only (Eq.1a)
  - Soft-only:  Tier 2 CLIP semantic similarity, simulated (Eq.1b)
  - Hard+Soft:  Combined two-tier (Eq.1)

Metrics:
  - PSR:  Pareidolia Suppression Rate (suppressed false / total false)
  - FPR:  False Positive Rate (incorrectly removed real / total real)
  - Precision, Recall, F1 over the pareidolia detection task

Level 1: Offline, algorithm-only. No HTTP, no GPU.
CLIP Tier 2 is simulated via keyword membership (biological_keywords).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.algorithms.pareidolia import (  # noqa: E402
    PareidoliaResult,
    detect_hard,
)
from core.config import load_config, PareidoliaConfig  # noqa: E402
from scripts.experiments.experiment_harness import (  # noqa: E402
    SCENE_SYNTHETIC_CAPTIONS,
    add_common_args,
    compute_stats,
    cohens_d,
    friedman_test,
    get_coco_captions,
    load_scenes,
    override_config,
    parse_common_args,
    save_result,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EXPERIMENT_ID = "exp02"

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
# Each test case: (description, object_labels, ground_truth_pareidolia)
# ground_truth_pareidolia: set of labels that SHOULD be flagged as pareidolia
#   (i.e., labels that are false detections in an aquaculture context)

TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "normal_objects",
        "description": "Real aquaculture objects - should NOT be flagged",
        "labels": ["fish", "tilapia", "underwater plant"],
        "pareidolia_labels": set(),  # none are false
    },
    {
        "name": "injected_false",
        "description": "Non-aquatic objects injected by VLM hallucination",
        "labels": ["fish", "person", "kite", "television"],
        "pareidolia_labels": {"person", "kite", "television"},
    },
    {
        "name": "structural",
        "description": "Net/cage structural elements with pareidolia risk",
        "labels": ["fish", "net", "cage", "mesh"],
        "pareidolia_labels": set(),  # net/cage/mesh are valid environment labels
    },
    {
        "name": "edge_case",
        "description": "Ambiguous marine species that should survive detection",
        "labels": ["jellyfish", "crab", "starfish"],
        "pareidolia_labels": set(),  # these are real aquatic organisms
    },
]

# Captions for different contexts
STRUCTURAL_CAPTIONS = [
    "An underwater view of a fish net cage with mesh structure and wire pen",
    "The image shows a net enclosure in an aquaculture farm with cage and grid",
    "Underwater lattice fence structure in a fish farming net cage pen",
]

NORMAL_CAPTIONS = [
    "Several tilapia fish swimming in clear pond water",
    "The image shows tilapia fish swimming in a pond with clear water",
    "Healthy fish swimming in aquaculture tank underwater",
    "Multiple fish visible in underwater aquaculture environment",
]

DETECTION_MODES = ["none", "hard_only", "soft_only", "hard_soft"]


# ---------------------------------------------------------------------------
# Simulated Soft (Tier 2) detection
# ---------------------------------------------------------------------------

def _simulate_soft_detection(
    label: str,
    config: PareidoliaConfig,
) -> bool:
    """Simulate CLIP-based soft detection by checking if a label
    matches any biological keyword.

    In a real deployment, this would compute:
      P_soft(l_i) = sigma(cos(CLIP_T(G), C_s) - tau) * sigma(cos(CLIP_T(l_i), C_b) - tau)

    For offline experiments we approximate: if the label contains a
    biological keyword (face, person, animal, human, head, eye, body),
    then it is flagged.

    Returns:
        True if the label is detected as pareidolia by the soft path.
    """
    label_lower = label.lower()
    return any(kw in label_lower for kw in config.biological_keywords)


# ---------------------------------------------------------------------------
# Detection mode runners
# ---------------------------------------------------------------------------

def _detect_none(
    caption: str,
    labels: list[str],
    config: PareidoliaConfig,
) -> list[PareidoliaResult]:
    """No detection: all labels pass through unflagged."""
    return [
        PareidoliaResult(
            label=lab,
            is_pareidolia=False,
            hard_score=0.0,
            soft_score=0.0,
            combined_score=0.0,
        )
        for lab in labels
    ]


def _detect_hard_only(
    caption: str,
    labels: list[str],
    config: PareidoliaConfig,
) -> list[PareidoliaResult]:
    """Tier 1 only: keyword-based hard detection (Eq.1a)."""
    return detect_hard(caption, labels, config)


def _detect_soft_only(
    caption: str,
    labels: list[str],
    config: PareidoliaConfig,
) -> list[PareidoliaResult]:
    """Tier 2 only: simulated CLIP-based soft detection (Eq.1b)."""
    results: list[PareidoliaResult] = []
    for lab in labels:
        flagged = _simulate_soft_detection(lab, config)
        results.append(PareidoliaResult(
            label=lab,
            is_pareidolia=flagged,
            hard_score=0.0,
            soft_score=1.0 if flagged else 0.0,
            combined_score=1.0 if flagged else 0.0,
        ))
    return results


def _detect_hard_soft(
    caption: str,
    labels: list[str],
    config: PareidoliaConfig,
) -> list[PareidoliaResult]:
    """Combined two-tier: P = max(P_hard, P_soft) (Eq.1)."""
    hard_results = detect_hard(caption, labels, config)
    combined: list[PareidoliaResult] = []
    for hr in hard_results:
        soft_flagged = _simulate_soft_detection(hr.label, config)
        soft_score = 1.0 if soft_flagged else 0.0
        combined_score = max(hr.hard_score, soft_score)
        combined.append(PareidoliaResult(
            label=hr.label,
            is_pareidolia=combined_score > 0.5,
            hard_score=hr.hard_score,
            soft_score=soft_score,
            combined_score=combined_score,
        ))
    return combined


DETECT_FNS = {
    "none": _detect_none,
    "hard_only": _detect_hard_only,
    "soft_only": _detect_soft_only,
    "hard_soft": _detect_hard_soft,
}


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_metrics(
    results: list[PareidoliaResult],
    ground_truth_pareidolia: set[str],
) -> dict[str, float]:
    """Compute PSR, FPR, Precision, Recall, F1 from detection results.

    Positive class = "is pareidolia" (should be flagged).

    - TP: label IS pareidolia AND detector flagged it
    - FP: label is NOT pareidolia AND detector flagged it
    - FN: label IS pareidolia AND detector did NOT flag it
    - TN: label is NOT pareidolia AND detector did NOT flag it

    PSR  = TP / (TP + FN)  if any positives exist, else 1.0
    FPR  = FP / (FP + TN)  if any negatives exist, else 0.0
    """
    tp = fp = fn = tn = 0
    for r in results:
        is_true_pareidolia = r.label in ground_truth_pareidolia
        if is_true_pareidolia and r.is_pareidolia:
            tp += 1
        elif not is_true_pareidolia and r.is_pareidolia:
            fp += 1
        elif is_true_pareidolia and not r.is_pareidolia:
            fn += 1
        else:
            tn += 1

    # PSR = Pareidolia Suppression Rate = Recall of pareidolia detection
    psr = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    # FPR = False Positive Rate = incorrectly flagged reals / total reals
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # Precision, Recall, F1 for pareidolia detection
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = psr  # same as PSR
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "psr": psr,
        "fpr": fpr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Build caption-test-case pairs
# ---------------------------------------------------------------------------

def _build_test_pairs(
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Build all (caption, test_case) pairs for evaluation.

    Combines structural and normal captions with the 4 test cases,
    plus any COCO-derived or synthetic captions for scenes.

    Returns:
        List of dicts with keys: caption, test_case_name, labels,
        pareidolia_labels, context.
    """
    pairs: list[dict[str, Any]] = []

    # Core test cases with structural captions (pareidolia-prone context)
    for tc in TEST_CASES:
        captions = STRUCTURAL_CAPTIONS[:1] if dry_run else STRUCTURAL_CAPTIONS
        for cap in captions:
            pairs.append({
                "caption": cap,
                "test_case_name": tc["name"],
                "labels": tc["labels"],
                "pareidolia_labels": tc["pareidolia_labels"],
                "context": "structural",
            })

    # Core test cases with normal captions (non-structural context)
    for tc in TEST_CASES:
        captions = NORMAL_CAPTIONS[:1] if dry_run else NORMAL_CAPTIONS
        for cap in captions:
            pairs.append({
                "caption": cap,
                "test_case_name": tc["name"],
                "labels": tc["labels"],
                "pareidolia_labels": tc["pareidolia_labels"],
                "context": "normal",
            })

    # Add synthetic scene captions
    for sid, syn_captions in SCENE_SYNTHETIC_CAPTIONS.items():
        caps = syn_captions[:1] if dry_run else syn_captions
        for cap in caps:
            # Use injected_false test case with scene captions
            tc = TEST_CASES[1]  # injected_false
            pairs.append({
                "caption": cap,
                "test_case_name": f"scene_{sid}",
                "labels": tc["labels"],
                "pareidolia_labels": tc["pareidolia_labels"],
                "context": f"scene:{sid}",
            })

    # Try to load COCO captions from actual dataset
    try:
        scenes = load_scenes()
    except Exception:
        log.warning("Could not load scenes from manifest; skipping COCO captions")
        scenes = []
    for scene in scenes:
        if scene["has_coco"]:
            coco_caps = get_coco_captions(scene["coco_data"])
            caps = coco_caps[:1] if dry_run else coco_caps[:3]  # limit per scene
            for cap in caps:
                tc = TEST_CASES[1]  # injected_false
                pairs.append({
                    "caption": cap,
                    "test_case_name": f"coco_{scene['scene_id']}",
                    "labels": tc["labels"],
                    "pareidolia_labels": tc["pareidolia_labels"],
                    "context": f"coco:{scene['scene_id']}",
                })

    log.info("Built %d test pairs", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    dry_run: bool = False,
    n_runs: int = 3,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Execute the full pareidolia ablation experiment.

    Args:
        dry_run: If True, reduce test pairs to minimum.
        n_runs: Number of independent runs (deterministic here, but kept
                for consistency with harness).
        config_path: Optional path to override config YAML.

    Returns:
        Complete result dict.
    """
    base_config = load_config(config_path)
    config_p = base_config.pareidolia

    test_pairs = _build_test_pairs(dry_run=dry_run)

    log.info(
        "EXP-02 starting: %d modes x %d test pairs",
        len(DETECTION_MODES), len(test_pairs),
    )
    t0 = time.perf_counter()

    # Run each mode
    mode_results: dict[str, dict[str, Any]] = {}

    for mode in DETECTION_MODES:
        detect_fn = DETECT_FNS[mode]
        all_metrics: list[dict[str, float]] = []
        per_pair: list[dict[str, Any]] = []

        for pair in test_pairs:
            results = detect_fn(
                pair["caption"],
                pair["labels"],
                config_p,
            )
            metrics = _compute_metrics(results, pair["pareidolia_labels"])
            all_metrics.append(metrics)

            per_pair.append({
                "test_case": pair["test_case_name"],
                "context": pair["context"],
                "labels": pair["labels"],
                "ground_truth_pareidolia": list(pair["pareidolia_labels"]),
                "detected": [r.label for r in results if r.is_pareidolia],
                "metrics": metrics,
            })

        # Aggregate metrics across all test pairs
        psr_vals = [m["psr"] for m in all_metrics]
        fpr_vals = [m["fpr"] for m in all_metrics]
        precision_vals = [m["precision"] for m in all_metrics]
        recall_vals = [m["recall"] for m in all_metrics]
        f1_vals = [m["f1"] for m in all_metrics]

        mode_results[mode] = {
            "mode": mode,
            "n_pairs": len(test_pairs),
            "psr": asdict(compute_stats(psr_vals)) if psr_vals else {},
            "fpr": asdict(compute_stats(fpr_vals)) if fpr_vals else {},
            "precision": asdict(compute_stats(precision_vals)) if precision_vals else {},
            "recall": asdict(compute_stats(recall_vals)) if recall_vals else {},
            "f1": asdict(compute_stats(f1_vals)) if f1_vals else {},
            "per_pair": per_pair,
        }

    # --- Statistical comparisons ---
    # Cohen's d: hard_soft F1 vs each other mode
    effect_sizes: dict[str, float] = {}
    hs_f1 = [m["f1"] for m in mode_results["hard_soft"]["per_pair"]
             if "metrics" in m for m in [m["metrics"]]]
    # Recompute more cleanly
    hs_f1_list = [p["metrics"]["f1"] for p in mode_results["hard_soft"]["per_pair"]]
    for other_mode in ["none", "hard_only", "soft_only"]:
        other_f1_list = [p["metrics"]["f1"] for p in mode_results[other_mode]["per_pair"]]
        if hs_f1_list and other_f1_list:
            effect_sizes[f"d_hard_soft_vs_{other_mode}"] = cohens_d(
                hs_f1_list, other_f1_list,
            )

    # Friedman test across modes (using F1 per pair)
    if not dry_run and len(test_pairs) >= 3:
        friedman_groups = [
            [p["metrics"]["f1"] for p in mode_results[mode]["per_pair"]]
            for mode in DETECTION_MODES
        ]
        friedman_result = friedman_test(friedman_groups)
    else:
        friedman_result = {"statistic": float("nan"), "p_value": float("nan")}

    # --- Context breakdown: structural vs normal ---
    context_breakdown: dict[str, dict[str, dict[str, Any]]] = {}
    for mode in DETECTION_MODES:
        for ctx in ["structural", "normal"]:
            ctx_pairs = [
                p for p in mode_results[mode]["per_pair"]
                if p["context"] == ctx
            ]
            if ctx_pairs:
                ctx_f1 = [p["metrics"]["f1"] for p in ctx_pairs]
                ctx_psr = [p["metrics"]["psr"] for p in ctx_pairs]
                ctx_fpr = [p["metrics"]["fpr"] for p in ctx_pairs]
                context_breakdown.setdefault(mode, {})[ctx] = {
                    "n": len(ctx_pairs),
                    "f1_mean": float(np.mean(ctx_f1)),
                    "psr_mean": float(np.mean(ctx_psr)),
                    "fpr_mean": float(np.mean(ctx_fpr)),
                }

    elapsed = time.perf_counter() - t0

    # Find best mode by F1
    best_mode = max(
        DETECTION_MODES,
        key=lambda m: mode_results[m]["f1"].get("mean", 0.0),
    )

    result = {
        "experiment": "EXP-02: Two-Tier Pareidolia Detection Ablation",
        "hypothesis": "Hard+Soft > single-tier detection",
        "equations": ["Eq.1a", "Eq.1b", "Eq.1", "Eq.2"],
        "parameters": {
            "modes": DETECTION_MODES,
            "n_test_cases": len(TEST_CASES),
            "n_test_pairs": len(test_pairs),
            "structural_keywords": config_p.structural_keywords,
            "biological_keywords": config_p.biological_keywords,
            "clip_threshold": config_p.clip_threshold,
            "dry_run": dry_run,
        },
        "mode_results": {m: {
            "mode": mode_results[m]["mode"],
            "n_pairs": mode_results[m]["n_pairs"],
            "psr": mode_results[m]["psr"],
            "fpr": mode_results[m]["fpr"],
            "precision": mode_results[m]["precision"],
            "recall": mode_results[m]["recall"],
            "f1": mode_results[m]["f1"],
            # Omit per_pair from summary to keep output concise; it is in raw
        } for m in DETECTION_MODES},
        "best_mode": best_mode,
        "context_breakdown": context_breakdown,
        "effect_sizes": effect_sizes,
        "friedman_test": friedman_result,
        "raw_per_pair": {m: mode_results[m]["per_pair"] for m in DETECTION_MODES},
        "elapsed_seconds": elapsed,
    }

    return result


def print_summary(result: dict[str, Any]) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 90)
    print("EXP-02: Two-Tier Pareidolia Detection Ablation")
    print("=" * 90)

    # Main results table
    print(
        f"{'Mode':>12s}  "
        f"{'PSR(mean)':>10s}  "
        f"{'FPR(mean)':>10s}  "
        f"{'Prec(mean)':>11s}  "
        f"{'Rec(mean)':>10s}  "
        f"{'F1(mean)':>9s}  "
        f"{'F1(std)':>8s}  "
        f"{'Best':>5s}"
    )
    print("-" * 90)

    best_mode = result["best_mode"]
    for mode in DETECTION_MODES:
        mr = result["mode_results"][mode]
        marker = " <--" if mode == best_mode else ""
        print(
            f"{mode:>12s}  "
            f"{mr['psr'].get('mean', 0.0):>10.4f}  "
            f"{mr['fpr'].get('mean', 0.0):>10.4f}  "
            f"{mr['precision'].get('mean', 0.0):>11.4f}  "
            f"{mr['recall'].get('mean', 0.0):>10.4f}  "
            f"{mr['f1'].get('mean', 0.0):>9.4f}  "
            f"{mr['f1'].get('std', 0.0):>8.4f}  "
            f"{marker:>5s}"
        )

    print("-" * 90)
    print(f"Best mode: {best_mode}")

    # Context breakdown
    ctx = result.get("context_breakdown", {})
    if ctx:
        print("\nContext Breakdown (F1 mean):")
        print(f"{'Mode':>12s}  {'Structural':>12s}  {'Normal':>10s}")
        print("-" * 40)
        for mode in DETECTION_MODES:
            mode_ctx = ctx.get(mode, {})
            s_f1 = mode_ctx.get("structural", {}).get("f1_mean", float("nan"))
            n_f1 = mode_ctx.get("normal", {}).get("f1_mean", float("nan"))
            print(f"{mode:>12s}  {s_f1:>12.4f}  {n_f1:>10.4f}")

    # Effect sizes
    es = result.get("effect_sizes", {})
    if es:
        print("\nEffect sizes (Cohen's d, hard_soft vs others):")
        for key, val in es.items():
            print(f"  {key}: {val:.4f}")

    # Friedman test
    fr = result.get("friedman_test", {})
    if not np.isnan(fr.get("p_value", float("nan"))):
        print(f"\nFriedman test: chi2={fr['statistic']:.4f}, p={fr['p_value']:.6f}")

    n_pairs = result["parameters"]["n_test_pairs"]
    print(f"\nTotal test pairs: {n_pairs}")
    print(f"Elapsed: {result['elapsed_seconds']:.3f}s")
    print("=" * 90)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP-02: Two-Tier Pareidolia Detection Ablation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python exp02_pareidolia_ablation.py --dry-run\n"
            "  python exp02_pareidolia_ablation.py --runs 5\n"
        ),
    )
    add_common_args(parser)
    args = parser.parse_args()
    common = parse_common_args(args)

    log.info("Configuration: %s", common)

    result = run_experiment(
        dry_run=common["dry_run"],
        n_runs=common["runs"],
        config_path=common["config_path"],
    )

    # Save
    filepath = save_result(EXPERIMENT_ID, result)
    log.info("Results saved to %s", filepath)

    # Print summary
    print_summary(result)


if __name__ == "__main__":
    main()
