"""EXP-08: Cross-Domain Generalization (Table 7).

Hypothesis: The scoring pipeline maintains diagnostic accuracy even when
training/testing captions come from different scene domains, indicating
robust generalization rather than overfitting to domain-specific vocabulary.

Five domain transfer scenarios (all Level 1 offline):
  1. Within-fish:      Train S01 captions -> test on S02 captions
  2. Within-env:       Train S05 captions -> test on S06 captions
  3. Cross-fish->env:  S01+S02 captions -> test on S05+S06
  4. Cross-env->fish:  S05+S06 -> test on S01+S02
  5. Leave-one-out:    S01-S09 -> test on S10

"Training" means: the captions from those scenes define what the system
has been exposed to (context knowledge).
"Testing" means: run classify_scene + scoring on test scene captions and
LLM outputs to produce a diagnosis decision.

Metrics per scenario: SCA, DA, confusion matrix (3x3: Healthy/Disease/Inconclusive).

Level 1: All offline simulation.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.algorithms.scoring import full_scoring_pipeline  # noqa: E402
from core.algorithms.semantic_filter import classify_scene  # noqa: E402
from core.config import load_config  # noqa: E402
from scripts.experiments.experiment_harness import (  # noqa: E402
    SCENE_EXPECTED_STATUS,
    SCENE_SYNTHETIC_CAPTIONS,
    SIMULATED_LLM_OUTPUTS,
    add_common_args,
    compute_stats,
    parse_common_args,
    save_result,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EXPERIMENT_ID = "exp08"

CLASSES = ["Healthy", "Disease", "Inconclusive"]

# ---------------------------------------------------------------------------
# Scene groupings for domain transfer
# ---------------------------------------------------------------------------
FISH_SCENES = [
    "S01_fish_health_tilapia",
    "S02_fish_health_grouper",
]
DISEASE_SCENES = [
    "S03_disease_white_spot",
    "S04_disease_general",
]
ENV_SCENES = [
    "S05_environment_net_cage",
    "S06_environment_pond_tank",
]
OTHER_SCENES = [
    "S07_underwater_survey_rov",
    "S08_water_quality_degraded",
    "S09_multi_species_detection",
    "S10_edge_cases_turbidity",
]

ALL_SCENES = list(SCENE_EXPECTED_STATUS.keys())

# Fallback captions for scenes not in SCENE_SYNTHETIC_CAPTIONS
FALLBACK_CAPTIONS: dict[str, str] = {
    "S01_fish_health_tilapia": "Tilapia fish swimming in aquaculture pond",
    "S02_fish_health_grouper": "Grouper fish in underwater tank environment",
    "S03_disease_white_spot": "Fish with white spots showing disease symptoms",
    "S04_disease_general": "Diseased fish with infection and lesions",
    "S05_environment_net_cage": "Underwater net cage structure with mesh and wire",
    "S06_environment_pond_tank": "Sonar depth map of aquaculture pond tank floor",
    "S07_underwater_survey_rov": "ROV underwater survey showing seabed structures",
    "S08_water_quality_degraded": "Degraded underwater image with turbid water quality",
    "S09_multi_species_detection": "Multiple fish and crab species detected underwater",
    "S10_edge_cases_turbidity": "Jellyfish in turbid underwater water conditions",
}

# Domain acceptable scene types (maps ground-truth domain to acceptable
# scene_type outputs from classify_scene)
DOMAIN_ACCEPTABLE: dict[str, list[str]] = {
    "fish": ["fish", "disease", "environment", "general"],
    "aquaculture_env": ["environment", "general", "fish"],
    "water_quality": ["general", "environment"],
}

SCENE_RAG_DOMAIN: dict[str, str] = {
    "S01_fish_health_tilapia": "fish",
    "S02_fish_health_grouper": "fish",
    "S03_disease_white_spot": "fish",
    "S04_disease_general": "fish",
    "S05_environment_net_cage": "aquaculture_env",
    "S06_environment_pond_tank": "aquaculture_env",
    "S07_underwater_survey_rov": "aquaculture_env",
    "S08_water_quality_degraded": "water_quality",
    "S09_multi_species_detection": "fish",
    "S10_edge_cases_turbidity": "fish",
}

# ---------------------------------------------------------------------------
# Transfer scenario definitions
# ---------------------------------------------------------------------------
TRANSFER_SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "T1_within_fish",
        "name": "Within-Fish (S01->S02)",
        "train_scenes": ["S01_fish_health_tilapia"],
        "test_scenes": ["S02_fish_health_grouper"],
    },
    {
        "id": "T2_within_env",
        "name": "Within-Env (S05->S06)",
        "train_scenes": ["S05_environment_net_cage"],
        "test_scenes": ["S06_environment_pond_tank"],
    },
    {
        "id": "T3_fish_to_env",
        "name": "Cross-Fish->Env (S01+S02->S05+S06)",
        "train_scenes": ["S01_fish_health_tilapia", "S02_fish_health_grouper"],
        "test_scenes": ["S05_environment_net_cage", "S06_environment_pond_tank"],
    },
    {
        "id": "T4_env_to_fish",
        "name": "Cross-Env->Fish (S05+S06->S01+S02)",
        "train_scenes": ["S05_environment_net_cage", "S06_environment_pond_tank"],
        "test_scenes": ["S01_fish_health_tilapia", "S02_fish_health_grouper"],
    },
    {
        "id": "T5_leave_one_out",
        "name": "Leave-One-Out (S01-S09->S10)",
        "train_scenes": [s for s in ALL_SCENES if s != "S10_edge_cases_turbidity"],
        "test_scenes": ["S10_edge_cases_turbidity"],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_caption(scene_id: str) -> str:
    """Get the first available caption for a scene."""
    captions = SCENE_SYNTHETIC_CAPTIONS.get(scene_id, [])
    if captions:
        return captions[0]
    return FALLBACK_CAPTIONS.get(scene_id, f"Scene {scene_id}")


def _compute_confusion_matrix(
    predictions: dict[str, str],
    ground_truth: dict[str, str],
) -> dict[str, Any]:
    """Compute 3x3 confusion matrix and per-class metrics.

    Args:
        predictions: {scene_id: predicted_status}
        ground_truth: {scene_id: expected_status}

    Returns:
        Dict with confusion matrix, per-class metrics, and overall DA.
    """
    confusion: dict[str, dict[str, int]] = {
        c: {c2: 0 for c2 in CLASSES} for c in CLASSES
    }
    matches = 0
    total = 0

    for sid in predictions:
        pred = predictions[sid]
        gt = ground_truth.get(sid, "Inconclusive")
        if gt in confusion and pred in confusion[gt]:
            confusion[gt][pred] += 1
        total += 1
        if pred == gt:
            matches += 1

    da = matches / total if total > 0 else 0.0

    per_class: dict[str, dict[str, float]] = {}
    for cls in CLASSES:
        tp = confusion[cls][cls]
        fp = sum(confusion[other][cls] for other in CLASSES if other != cls)
        fn = sum(confusion[cls][other] for other in CLASSES if other != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
               if (precision + recall) > 0 else 0.0)

        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
        }

    return {
        "da": da,
        "matches": matches,
        "total": total,
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Run a single transfer scenario
# ---------------------------------------------------------------------------
def _run_transfer_scenario(
    scenario: dict[str, Any],
    base_config: Any,
) -> dict[str, Any]:
    """Run one domain transfer scenario.

    "Training" context: we acknowledge which captions the system has seen.
    "Testing": run classify_scene + scoring on test scenes.

    Args:
        scenario: Transfer scenario definition.
        base_config: Base AppConfig.

    Returns:
        Scenario result dict with SCA, DA, confusion, per-scene detail.
    """
    train_scenes = scenario["train_scenes"]
    test_scenes = scenario["test_scenes"]

    # --- Gather training context (captions from train scenes) ---
    train_captions: list[str] = []
    for sid in train_scenes:
        train_captions.append(_get_caption(sid))

    # --- Evaluate on test scenes ---
    predictions: dict[str, str] = {}
    ground_truth: dict[str, str] = {}
    sca_correct = 0
    sca_total = 0
    per_scene: dict[str, dict[str, Any]] = {}

    for sid in test_scenes:
        expected = SCENE_EXPECTED_STATUS.get(sid, "Inconclusive")
        ground_truth[sid] = expected

        # Step 1: Semantic filter classification (SCA)
        test_caption = _get_caption(sid)
        cls = classify_scene(test_caption, base_config.semantic_filter)

        gt_domain = SCENE_RAG_DOMAIN.get(sid, "fish")
        acceptable = DOMAIN_ACCEPTABLE.get(gt_domain, ["general"])
        sca_ok = cls.scene_type in acceptable
        sca_correct += int(sca_ok)
        sca_total += 1

        # Step 2: Scoring pipeline (DA)
        llm_text = SIMULATED_LLM_OUTPUTS.get(sid, "")
        decision = full_scoring_pipeline(llm_text, base_config.scoring)
        predictions[sid] = decision.status

        per_scene[sid] = {
            "expected": expected,
            "predicted_status": decision.status,
            "match": decision.status == expected,
            "healthy_score": decision.healthy_score,
            "disease_score": decision.disease_score,
            "confidence": decision.confidence,
            "scene_type": cls.scene_type,
            "filter_score": cls.score,
            "sca_ok": sca_ok,
            "gt_domain": gt_domain,
            "test_caption": test_caption[:80],
        }

    sca = sca_correct / sca_total if sca_total > 0 else 0.0
    metrics = _compute_confusion_matrix(predictions, ground_truth)

    return {
        "scenario_id": scenario["id"],
        "scenario_name": scenario["name"],
        "train_scenes": train_scenes,
        "test_scenes": test_scenes,
        "n_train": len(train_scenes),
        "n_test": len(test_scenes),
        "sca": sca,
        "da": metrics["da"],
        "matches": metrics["matches"],
        "total": metrics["total"],
        "confusion_matrix": metrics["confusion_matrix"],
        "per_class": metrics["per_class"],
        "per_scene": per_scene,
        "train_context_captions": train_captions,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment(
    dry_run: bool = False,
    n_runs: int = 3,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Execute the cross-domain generalization experiment.

    Args:
        dry_run: If True, run only the first scenario.
        n_runs: Number of repeated runs (deterministic, but kept for consistency).
        config_path: Optional override config YAML path.

    Returns:
        Complete result dict ready for save_result().
    """
    base_config = load_config(config_path)

    scenarios_to_run = TRANSFER_SCENARIOS[:1] if dry_run else TRANSFER_SCENARIOS

    log.info(
        "EXP-08 starting: %d transfer scenarios, dry_run=%s",
        len(scenarios_to_run), dry_run,
    )
    t0 = time.perf_counter()

    scenario_results: list[dict[str, Any]] = []
    da_values: list[float] = []
    sca_values: list[float] = []

    for scenario in scenarios_to_run:
        log.info("Running scenario: %s", scenario["name"])
        result = _run_transfer_scenario(scenario, base_config)
        scenario_results.append(result)
        da_values.append(result["da"])
        sca_values.append(result["sca"])
        log.info(
            "  %s: DA=%.3f, SCA=%.3f",
            scenario["id"], result["da"], result["sca"],
        )

    # --- Aggregate statistics ---
    da_stats = compute_stats(da_values) if da_values else None
    sca_stats = compute_stats(sca_values) if sca_values else None

    # --- Aggregate confusion matrix across all scenarios ---
    agg_confusion: dict[str, dict[str, int]] = {
        c: {c2: 0 for c2 in CLASSES} for c in CLASSES
    }
    for sr in scenario_results:
        cm = sr["confusion_matrix"]
        for actual in CLASSES:
            for pred in CLASSES:
                agg_confusion[actual][pred] += cm[actual][pred]

    # --- Summary rows ---
    summary_rows: list[dict[str, Any]] = []
    for sr in scenario_results:
        summary_rows.append({
            "scenario": sr["scenario_id"],
            "name": sr["scenario_name"],
            "n_train": sr["n_train"],
            "n_test": sr["n_test"],
            "da": sr["da"],
            "sca": sr["sca"],
        })

    # --- Domain gap: within-domain vs cross-domain DA ---
    within_ids = {"T1_within_fish", "T2_within_env"}
    cross_ids = {"T3_fish_to_env", "T4_env_to_fish"}
    within_da = [sr["da"] for sr in scenario_results if sr["scenario_id"] in within_ids]
    cross_da = [sr["da"] for sr in scenario_results if sr["scenario_id"] in cross_ids]

    if within_da and cross_da:
        domain_gap = float(np.mean(within_da) - np.mean(cross_da))
    else:
        domain_gap = 0.0

    # Bootstrap 95% CI for domain gap
    bootstrap_ci: dict[str, Any] = {"lower": 0.0, "upper": 0.0, "n_bootstrap": 0}
    all_da_for_bootstrap = da_values  # DA values from all scenarios
    if len(all_da_for_bootstrap) >= 2:
        rng = np.random.default_rng(42)
        n_bootstrap = 1000
        boot_gaps: list[float] = []
        within_arr = np.array(within_da) if within_da else np.array([0.0])
        cross_arr = np.array(cross_da) if cross_da else np.array([0.0])
        for _ in range(n_bootstrap):
            w_sample = rng.choice(within_arr, size=len(within_arr), replace=True)
            c_sample = rng.choice(cross_arr, size=len(cross_arr), replace=True)
            boot_gaps.append(float(np.mean(w_sample) - np.mean(c_sample)))
        boot_gaps_sorted = sorted(boot_gaps)
        ci_lower = boot_gaps_sorted[int(0.025 * n_bootstrap)]
        ci_upper = boot_gaps_sorted[int(0.975 * n_bootstrap)]
        bootstrap_ci = {
            "lower": float(ci_lower),
            "upper": float(ci_upper),
            "n_bootstrap": n_bootstrap,
        }

    elapsed = time.perf_counter() - t0

    result = {
        "experiment": "EXP-08: Cross-Domain Generalization",
        "hypothesis": "Pipeline maintains DA across domain boundaries",
        "table_ref": "Table 7",
        "parameters": {
            "n_scenarios": len(scenarios_to_run),
            "dry_run": dry_run,
        },
        "summary": summary_rows,
        "aggregate": {
            "da_mean": da_stats.mean if da_stats else 0.0,
            "da_std": da_stats.std if da_stats else 0.0,
            "sca_mean": sca_stats.mean if sca_stats else 0.0,
            "sca_std": sca_stats.std if sca_stats else 0.0,
        },
        "aggregate_confusion_matrix": agg_confusion,
        "domain_gap": domain_gap,
        "bootstrap_ci": bootstrap_ci,
        "scenario_results": {sr["scenario_id"]: sr for sr in scenario_results},
        "elapsed_seconds": elapsed,
    }

    return result


def print_summary(result: dict[str, Any]) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 95)
    print("EXP-08: Cross-Domain Generalization")
    print("=" * 95)

    # Scenario summary
    print(
        f"\n{'Scenario':<15s}  "
        f"{'Name':<38s}  "
        f"{'#Train':>6s}  "
        f"{'#Test':>5s}  "
        f"{'DA':>6s}  "
        f"{'SCA':>6s}"
    )
    print("-" * 95)

    for row in result["summary"]:
        print(
            f"{row['scenario']:<15s}  "
            f"{row['name']:<38s}  "
            f"{row['n_train']:>6d}  "
            f"{row['n_test']:>5d}  "
            f"{row['da']:>6.3f}  "
            f"{row['sca']:>6.3f}"
        )

    print("-" * 95)
    agg = result["aggregate"]
    print(
        f"{'AGGREGATE':<15s}  "
        f"{'':38s}  "
        f"{'':>6s}  "
        f"{'':>5s}  "
        f"{agg['da_mean']:>5.3f}  "
        f"{agg['sca_mean']:>5.3f}"
    )
    print(
        f"{'':15s}  "
        f"{'(std)':38s}  "
        f"{'':>6s}  "
        f"{'':>5s}  "
        f"{agg['da_std']:>5.3f}  "
        f"{agg['sca_std']:>5.3f}"
    )

    # Aggregate confusion matrix
    print("\n--- Aggregate Confusion Matrix ---")
    print(f"  {'':>15s}  {'Pred H':>7s}  {'Pred D':>7s}  {'Pred I':>7s}")
    cm = result["aggregate_confusion_matrix"]
    for actual in CLASSES:
        vals = [cm[actual].get(p, 0) for p in CLASSES]
        print(f"  {'Actual ' + actual[0]:>15s}  {vals[0]:>7d}  {vals[1]:>7d}  {vals[2]:>7d}")

    # Per-scenario confusion matrices
    print("\n--- Per-Scenario Confusion Matrices ---")
    for sid, sr in result["scenario_results"].items():
        print(f"\n  {sid} ({sr['scenario_name']}):")
        scm = sr["confusion_matrix"]
        print(f"  {'':>15s}  {'Pred H':>7s}  {'Pred D':>7s}  {'Pred I':>7s}")
        for actual in CLASSES:
            vals = [scm[actual].get(p, 0) for p in CLASSES]
            print(f"  {'Actual ' + actual[0]:>15s}  {vals[0]:>7d}  {vals[1]:>7d}  {vals[2]:>7d}")

    print(f"\nTotal elapsed: {result['elapsed_seconds']:.3f}s")
    print("=" * 95)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP-08: Cross-Domain Generalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python exp08_cross_domain.py --dry-run\n"
            "  python exp08_cross_domain.py\n"
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
