"""EXP-09: Bilingual Prompt Effect Analysis (Table 8).

Hypothesis: Structured bilingual prompts with domain-specific guidance terms
produce higher DA and more reliable scoring indicators than generic or
single-language prompts.

Four configurations (all Level 1 offline simulation):
  ZH-Full:     Chinese prompts with full guidance terms (current default)
  ZH-NoGuide:  Chinese prompts without standardized guidance terms
  EN-Full:     English prompts with scoring keywords
  EN-NoGuide:  English prompts without scoring keywords (generic)

Simulation approach:
  - ZH-Full uses SIMULATED_LLM_OUTPUTS (current system, Chinese-friendly terms)
  - ZH-NoGuide replaces standardized terms with colloquial equivalents
  - EN-Full uses English with full scoring keywords
  - EN-NoGuide uses generic English without scoring keywords

Metrics per config:
  - DA: Diagnostic Accuracy (3-class)
  - Indicator hit count: S_h (total healthy indicator matches), S_d (total disease)
  - JSON parse success rate (simulated: structured -> 1.0, unstructured -> 0.7)

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
from core.config import load_config  # noqa: E402
from scripts.experiments.experiment_harness import (  # noqa: E402
    SCENE_EXPECTED_STATUS,
    SIMULATED_LLM_OUTPUTS,
    add_common_args,
    compute_stats,
    cohens_d,
    parse_common_args,
    save_result,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EXPERIMENT_ID = "exp09"

CLASSES = ["Healthy", "Disease", "Inconclusive"]

# ---------------------------------------------------------------------------
# Simulated LLM outputs per prompt configuration
# ---------------------------------------------------------------------------

# ZH-Full: Current default (SIMULATED_LLM_OUTPUTS) -- contains standardized
# bilingual terms like "appears healthy", "diagnosed", "confirmed"
SIMULATED_ZH_FULL = SIMULATED_LLM_OUTPUTS

# ZH-NoGuide: Chinese-like but without standardized diagnostic terms.
# Colloquial equivalents that may miss scoring keywords.
SIMULATED_ZH_NOGUIDE: dict[str, str] = {
    "S01_fish_health_tilapia": "The tilapia looks ok, swimming around in the pond. Seems fine.",
    "S02_fish_health_grouper": "Grouper looks alright. Nothing weird about it.",
    "S03_disease_white_spot": "There are white dots on the fish. It looks sick.",
    "S04_disease_general": "The fish has marks on its skin. Might be bacteria causing it.",
    "S05_environment_net_cage": "This is an underwater cage with some sea creatures around it.",
    "S06_environment_pond_tank": "Looks like a depth map of a pond bottom. Monitoring equipment data.",
    "S07_underwater_survey_rov": "ROV footage of the seabed. Nothing special visible.",
    "S08_water_quality_degraded": "The water is really murky. Hard to see anything clearly.",
    "S09_multi_species_detection": "I can see fish, crabs, and shrimp. They all look ok.",
    "S10_edge_cases_turbidity": "There is a jellyfish in cloudy water. It seems ok.",
}

# EN-Full: English with full scoring keywords (structured, high-quality)
SIMULATED_EN_FULL: dict[str, str] = {
    "S01_fish_health_tilapia": "The tilapia fish appears healthy with normal body condition. Active swimming behavior observed. No disease detected.",
    "S02_fish_health_grouper": "The grouper fish is healthy with clear skin. No lesions observed. Normal condition confirmed.",
    "S03_disease_white_spot": "White spot disease confirmed. Diagnosed with Ichthyophthirius multifiliis parasite infection on body surface.",
    "S04_disease_general": "Fish diagnosed with bacterial infection. Suspected Vibrio species. Confirmed infected with visible lesions.",
    "S05_environment_net_cage": "Underwater net cage environment with marine organisms. No fish disease assessment possible.",
    "S06_environment_pond_tank": "Sonar depth map of pond floor. Environmental monitoring data only. No biological specimens visible.",
    "S07_underwater_survey_rov": "ROV survey footage showing seabed and marine structures. General environment assessment only.",
    "S08_water_quality_degraded": "Degraded underwater image. Water quality compromised. Environmental conditions noted.",
    "S09_multi_species_detection": "Multiple aquatic species detected. Fish, crab, and shrimp all appear healthy. No disease indicators.",
    "S10_edge_cases_turbidity": "Jellyfish observed in turbid water. No disease detected. Appears healthy with normal morphology.",
}

# EN-NoGuide: Generic English without scoring keywords
SIMULATED_EN_NOGUIDE: dict[str, str] = {
    "S01_fish_health_tilapia": "I see some tilapia in a pond. They are swimming normally.",
    "S02_fish_health_grouper": "A grouper is present in the image. It is in an underwater setting.",
    "S03_disease_white_spot": "The fish has some white markings. It does not look well.",
    "S04_disease_general": "The fish seems to have some skin issues. There are marks visible.",
    "S05_environment_net_cage": "An underwater structure is shown with some marine life around it.",
    "S06_environment_pond_tank": "This appears to be sensor data from some kind of tank.",
    "S07_underwater_survey_rov": "Footage from an underwater vehicle showing the ocean floor.",
    "S08_water_quality_degraded": "The image is very unclear, the water seems dirty.",
    "S09_multi_species_detection": "Several types of sea creatures are visible in the image.",
    "S10_edge_cases_turbidity": "A jellyfish is floating in the water. The water is cloudy.",
}

# Config definitions
PROMPT_CONFIGS: list[dict[str, Any]] = [
    {
        "id": "ZH-Full",
        "name": "Chinese Full Guidance",
        "llm_outputs": SIMULATED_ZH_FULL,
        "json_parse_rate": 1.0,  # Structured prompt yields reliable JSON
        "description": "Bilingual prompts with domain-specific guidance terms",
    },
    {
        "id": "ZH-NoGuide",
        "name": "Chinese No Guidance",
        "llm_outputs": SIMULATED_ZH_NOGUIDE,
        "json_parse_rate": 0.7,  # Unstructured output, less reliable JSON
        "description": "Chinese prompts without standardized diagnostic terms",
    },
    {
        "id": "EN-Full",
        "name": "English Full Guidance",
        "llm_outputs": SIMULATED_EN_FULL,
        "json_parse_rate": 1.0,
        "description": "English prompts with full scoring keywords",
    },
    {
        "id": "EN-NoGuide",
        "name": "English No Guidance",
        "llm_outputs": SIMULATED_EN_NOGUIDE,
        "json_parse_rate": 0.7,
        "description": "Generic English without scoring keywords",
    },
]


# ---------------------------------------------------------------------------
# Run a single prompt configuration
# ---------------------------------------------------------------------------
def _run_prompt_config(
    config_def: dict[str, Any],
    base_config: Any,
) -> dict[str, Any]:
    """Run scoring pipeline on all scenes for a given prompt configuration.

    Args:
        config_def: Prompt configuration definition.
        base_config: Base AppConfig.

    Returns:
        Config result dict with DA, indicator counts, per-scene detail.
    """
    llm_outputs = config_def["llm_outputs"]
    predictions: dict[str, str] = {}
    per_scene: dict[str, dict[str, Any]] = {}
    total_s_h = 0
    total_s_d = 0
    matches = 0
    total = 0
    da_per_scene: list[float] = []

    for sid, llm_text in llm_outputs.items():
        expected = SCENE_EXPECTED_STATUS.get(sid, "Inconclusive")
        decision = full_scoring_pipeline(llm_text, base_config.scoring)

        predictions[sid] = decision.status
        is_match = decision.status == expected
        matches += int(is_match)
        total += 1
        da_per_scene.append(1.0 if is_match else 0.0)

        total_s_h += decision.healthy_score
        total_s_d += decision.disease_score

        per_scene[sid] = {
            "expected": expected,
            "predicted": decision.status,
            "match": is_match,
            "healthy_score": decision.healthy_score,
            "disease_score": decision.disease_score,
            "confidence": decision.confidence,
            "healthy_patterns": decision.healthy_detail.matched_patterns,
            "disease_patterns": decision.disease_detail.matched_patterns,
        }

    da = matches / total if total > 0 else 0.0

    # Confusion matrix
    confusion: dict[str, dict[str, int]] = {
        c: {c2: 0 for c2 in CLASSES} for c in CLASSES
    }
    for sid in predictions:
        pred = predictions[sid]
        gt = SCENE_EXPECTED_STATUS.get(sid, "Inconclusive")
        if gt in confusion and pred in confusion[gt]:
            confusion[gt][pred] += 1

    return {
        "config_id": config_def["id"],
        "config_name": config_def["name"],
        "description": config_def["description"],
        "da": da,
        "da_per_scene": da_per_scene,
        "matches": matches,
        "total": total,
        "total_s_h": total_s_h,
        "total_s_d": total_s_d,
        "avg_s_h": total_s_h / total if total > 0 else 0.0,
        "avg_s_d": total_s_d / total if total > 0 else 0.0,
        "json_parse_rate": config_def["json_parse_rate"],
        "confusion_matrix": confusion,
        "per_scene": per_scene,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment(
    dry_run: bool = False,
    n_runs: int = 3,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Execute the bilingual prompt effect analysis experiment.

    Args:
        dry_run: If True, run only the first 2 configurations.
        n_runs: Number of repeated runs (deterministic, kept for consistency).
        config_path: Optional override config YAML path.

    Returns:
        Complete result dict ready for save_result().
    """
    base_config = load_config(config_path)

    configs_to_run = PROMPT_CONFIGS[:2] if dry_run else PROMPT_CONFIGS

    log.info(
        "EXP-09 starting: %d prompt configs, dry_run=%s",
        len(configs_to_run), dry_run,
    )
    t0 = time.perf_counter()

    config_results: list[dict[str, Any]] = []

    for cfg_def in configs_to_run:
        log.info("Running config: %s (%s)", cfg_def["id"], cfg_def["name"])
        result = _run_prompt_config(cfg_def, base_config)
        config_results.append(result)
        log.info(
            "  %s: DA=%.3f, S_h=%d, S_d=%d, JSON_parse=%.2f",
            cfg_def["id"], result["da"],
            result["total_s_h"], result["total_s_d"],
            result["json_parse_rate"],
        )

    # --- Summary table ---
    summary_rows: list[dict[str, Any]] = []
    for cr in config_results:
        summary_rows.append({
            "config": cr["config_id"],
            "name": cr["config_name"],
            "da": cr["da"],
            "total_s_h": cr["total_s_h"],
            "total_s_d": cr["total_s_d"],
            "avg_s_h": cr["avg_s_h"],
            "avg_s_d": cr["avg_s_d"],
            "json_parse_rate": cr["json_parse_rate"],
        })

    # --- Indicator breakdown for stacked bar chart ---
    indicator_breakdown: list[dict[str, Any]] = []
    for cr in config_results:
        healthy_hits = 0
        disease_hits = 0
        no_match = 0
        for sid, sd in cr["per_scene"].items():
            if sd["healthy_score"] > 0 or sd["disease_score"] > 0:
                if sd["healthy_score"] > sd["disease_score"]:
                    healthy_hits += 1
                else:
                    disease_hits += 1
            else:
                no_match += 1
        indicator_breakdown.append({
            "config": cr["config_id"],
            "healthy_hits": healthy_hits,
            "disease_hits": disease_hits,
            "no_match": no_match,
        })

    # --- Chi-squared test on JSON parse rates ---
    # Contingency: configs x (parse_success, parse_fail) counts
    # Use total scenes as denominator for each config
    chi_squared_result: dict[str, Any] = {}
    if len(config_results) >= 2:
        n_scenes = config_results[0]["total"]
        observed = []
        for cr in config_results:
            success = int(round(cr["json_parse_rate"] * n_scenes))
            fail = n_scenes - success
            observed.append([success, fail])
        from scipy.stats import chi2_contingency as _chi2
        try:
            chi2_stat, chi2_p, chi2_dof, chi2_expected = _chi2(observed)
            chi_squared_result = {
                "chi2": float(chi2_stat),
                "p_value": float(chi2_p),
                "dof": int(chi2_dof),
            }
        except Exception:
            chi_squared_result = {"chi2": float("nan"), "p_value": float("nan"), "dof": 0}
    else:
        chi_squared_result = {"chi2": float("nan"), "p_value": float("nan"), "dof": 0}

    # --- Fisher's exact test: Guidance(yes/no) x indicator_hit(yes/no) ---
    fishers_exact_result: dict[str, Any] = {}
    full_hits = 0
    full_no = 0
    noguide_hits = 0
    noguide_no = 0
    for cr in config_results:
        is_guided = "Full" in cr["config_id"]
        for sid, sd in cr["per_scene"].items():
            has_hit = (sd["healthy_score"] > 0 or sd["disease_score"] > 0)
            if is_guided:
                if has_hit:
                    full_hits += 1
                else:
                    full_no += 1
            else:
                if has_hit:
                    noguide_hits += 1
                else:
                    noguide_no += 1
    contingency_2x2 = [[full_hits, full_no], [noguide_hits, noguide_no]]
    from scipy.stats import fisher_exact as _fisher
    try:
        odds_ratio, fisher_p = _fisher(contingency_2x2)
        fishers_exact_result = {
            "odds_ratio": float(odds_ratio),
            "p_value": float(fisher_p),
            "table": contingency_2x2,
        }
    except Exception:
        fishers_exact_result = {
            "odds_ratio": float("nan"),
            "p_value": float("nan"),
            "table": contingency_2x2,
        }

    # --- Pairwise comparisons ---
    pairwise: list[dict[str, Any]] = []

    # Compare Full vs NoGuide within same language
    full_configs = [cr for cr in config_results if "Full" in cr["config_id"]]
    noguide_configs = [cr for cr in config_results if "NoGuide" in cr["config_id"]]

    for full_cr in full_configs:
        lang = full_cr["config_id"].split("-")[0]  # "ZH" or "EN"
        matching_no = [cr for cr in noguide_configs if cr["config_id"].startswith(lang)]
        if matching_no:
            no_cr = matching_no[0]
            da_diff = full_cr["da"] - no_cr["da"]
            s_h_diff = full_cr["total_s_h"] - no_cr["total_s_h"]
            s_d_diff = full_cr["total_s_d"] - no_cr["total_s_d"]

            # Effect size (Cohen's d) on per-scene DA
            d = cohens_d(full_cr["da_per_scene"], no_cr["da_per_scene"])

            pairwise.append({
                "comparison": f"{full_cr['config_id']} vs {no_cr['config_id']}",
                "language": lang,
                "da_full": full_cr["da"],
                "da_noguide": no_cr["da"],
                "da_diff": da_diff,
                "s_h_diff": s_h_diff,
                "s_d_diff": s_d_diff,
                "cohens_d": d,
                "json_parse_diff": full_cr["json_parse_rate"] - no_cr["json_parse_rate"],
            })

    # Compare ZH vs EN within same guidance level
    for guidance in ["Full", "NoGuide"]:
        zh_matches = [cr for cr in config_results if cr["config_id"] == f"ZH-{guidance}"]
        en_matches = [cr for cr in config_results if cr["config_id"] == f"EN-{guidance}"]
        if zh_matches and en_matches:
            zh_cr = zh_matches[0]
            en_cr = en_matches[0]
            da_diff = zh_cr["da"] - en_cr["da"]
            d = cohens_d(zh_cr["da_per_scene"], en_cr["da_per_scene"])

            pairwise.append({
                "comparison": f"{zh_cr['config_id']} vs {en_cr['config_id']}",
                "guidance": guidance,
                "da_zh": zh_cr["da"],
                "da_en": en_cr["da"],
                "da_diff": da_diff,
                "cohens_d": d,
            })

    # --- Best config ---
    best_config = max(config_results, key=lambda cr: cr["da"])

    elapsed = time.perf_counter() - t0

    result = {
        "experiment": "EXP-09: Bilingual Prompt Effect Analysis",
        "hypothesis": "Structured bilingual guidance improves DA and indicator quality",
        "table_ref": "Table 8",
        "parameters": {
            "n_configs": len(configs_to_run),
            "dry_run": dry_run,
        },
        "summary": summary_rows,
        "pairwise_comparisons": pairwise,
        "best_config": {
            "id": best_config["config_id"],
            "da": best_config["da"],
        },
        "indicator_breakdown": indicator_breakdown,
        "chi_squared": chi_squared_result,
        "fishers_exact": fishers_exact_result,
        "config_results": {cr["config_id"]: cr for cr in config_results},
        "elapsed_seconds": elapsed,
    }

    return result


def print_summary(result: dict[str, Any]) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 105)
    print("EXP-09: Bilingual Prompt Effect Analysis")
    print("=" * 105)

    # Summary table
    print(
        f"\n{'Config':<12s}  "
        f"{'Name':<30s}  "
        f"{'DA':>6s}  "
        f"{'S_h(tot)':>8s}  "
        f"{'S_d(tot)':>8s}  "
        f"{'S_h(avg)':>8s}  "
        f"{'S_d(avg)':>8s}  "
        f"{'JSON%':>6s}"
    )
    print("-" * 105)

    for row in result["summary"]:
        print(
            f"{row['config']:<12s}  "
            f"{row['name']:<30s}  "
            f"{row['da']:>6.3f}  "
            f"{row['total_s_h']:>8d}  "
            f"{row['total_s_d']:>8d}  "
            f"{row['avg_s_h']:>8.2f}  "
            f"{row['avg_s_d']:>8.2f}  "
            f"{row['json_parse_rate']:>6.2f}"
        )

    print("-" * 105)

    # Pairwise comparisons
    print("\n--- Pairwise Comparisons ---")
    for pw in result["pairwise_comparisons"]:
        print(f"\n  {pw['comparison']}:")
        if "da_diff" in pw:
            print(f"    DA diff:   {pw['da_diff']:+.3f}")
        if "cohens_d" in pw:
            effect = "large" if abs(pw["cohens_d"]) > 0.8 else (
                "medium" if abs(pw["cohens_d"]) > 0.5 else "small")
            print(f"    Cohen's d: {pw['cohens_d']:.3f} ({effect})")
        if "s_h_diff" in pw:
            print(f"    S_h diff:  {pw['s_h_diff']:+d}")
            print(f"    S_d diff:  {pw['s_d_diff']:+d}")
        if "json_parse_diff" in pw:
            print(f"    JSON% diff:{pw['json_parse_diff']:+.2f}")

    # Best config
    bc = result["best_config"]
    print(f"\n--- Best Config: {bc['id']} (DA={bc['da']:.3f}) ---")

    # Per-scene detail for best config
    best_detail = result["config_results"][bc["id"]]["per_scene"]
    print(f"\n  {'Scene':<35s}  {'Expected':>12s}  {'Predicted':>12s}  {'S_h':>4s}  {'S_d':>4s}")
    for sid in sorted(best_detail.keys()):
        sd = best_detail[sid]
        mark = " ok" if sd["match"] else " XX"
        print(
            f"  {sid:<35s}  "
            f"{sd['expected']:>12s}  "
            f"{sd['predicted'] + mark:>12s}  "
            f"{sd['healthy_score']:>4d}  "
            f"{sd['disease_score']:>4d}"
        )

    print(f"\nTotal elapsed: {result['elapsed_seconds']:.3f}s")
    print("=" * 105)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP-09: Bilingual Prompt Effect Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python exp09_bilingual_prompt.py --dry-run\n"
            "  python exp09_bilingual_prompt.py\n"
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
