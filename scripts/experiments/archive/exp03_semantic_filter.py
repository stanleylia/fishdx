"""EXP-03: Adaptive Semantic Filter Ablation (Eq.8).

Hypothesis: The adaptive Jaccard/Overlap switching gate threshold at
tau_gate=0.10-0.15 outperforms both fixed-metric strategies (Jaccard-only,
Overlap-only) and no filtering (bypass).

Strategies tested:
  - Adaptive:          tau_gate in {0.05, 0.075, 0.10, 0.125, 0.15, 0.20}
  - None (bypass):     always returns "general" scene_type
  - Jaccard-only:      tau_gate=0.10, forces Jaccard similarity (no Overlap)
  - Overlap-only:      tau_gate=0.10, forces Overlap coefficient (no Jaccard)

Metrics:
  - SCA:  Scene Classification Accuracy (match vs DOMAIN_ACCEPTABLE)
  - RAG trigger rate:   % of fish-domain scenes triggering RAG
  - False trigger rate: % of environment-only scenes triggering RAG

Statistical tests:
  - Cohen's d: Adaptive vs Bypass, Adaptive vs Jaccard-only, Adaptive vs Overlap-only
  - Friedman test: across adaptive tau sweep
  - Cochran's Q test: binary correct/incorrect across 4 strategies

Level 1: Offline, algorithm-only. No HTTP, no GPU, no LLM.
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

from core.algorithms.semantic_filter import (  # noqa: E402
    SceneClassification,
    classify_scene,
)
from core.config import load_config  # noqa: E402
from scripts.experiments.experiment_harness import (  # noqa: E402
    SCENE_EXPECTED_STATUS,
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

EXPERIMENT_ID = "exp03"

# ---------------------------------------------------------------------------
# Domain-acceptable scene types for SCA computation
# ---------------------------------------------------------------------------
# For each rag_domain, which scene_type outputs are considered "acceptable"
# (correct or reasonably close).
DOMAIN_ACCEPTABLE: dict[str, list[str]] = {
    "fish": ["fish", "disease", "environment", "general"],
    "aquaculture_env": ["environment", "general", "fish"],
    "water_quality": ["general", "environment"],
}

# Mapping from scene_id to its rag_domain (ground truth domain)
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

# Scenes classified by domain for trigger-rate computation
FISH_DOMAIN_SCENES = [
    "S01_fish_health_tilapia",
    "S02_fish_health_grouper",
    "S03_disease_white_spot",
    "S04_disease_general",
    "S09_multi_species_detection",
    "S10_edge_cases_turbidity",
]
ENV_DOMAIN_SCENES = [
    "S05_environment_net_cage",
    "S06_environment_pond_tank",
    "S07_underwater_survey_rov",
    "S08_water_quality_degraded",
]

# Adaptive tau_gate sweep values
ADAPTIVE_TAU_VALUES = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20]

# Named strategies (tau_gate value, label)
NAMED_STRATEGIES: list[dict[str, Any]] = [
    {"name": "None (bypass)", "tau_gate": None},
    {"name": "Jaccard-only", "tau_gate": 0.10},
    {"name": "Overlap-only", "tau_gate": 0.10},
]


# ---------------------------------------------------------------------------
# Caption collection
# ---------------------------------------------------------------------------

def _collect_captions(dry_run: bool = False) -> dict[str, list[str]]:
    """Collect captions for each scene: synthetic + COCO-derived.

    Returns:
        {scene_id: [caption_1, caption_2, ...]}
    """
    scene_captions: dict[str, list[str]] = {}

    # Start with synthetic captions for all scenes
    for sid in SCENE_RAG_DOMAIN:
        captions: list[str] = []

        # Add synthetic captions if available
        if sid in SCENE_SYNTHETIC_CAPTIONS:
            synth = SCENE_SYNTHETIC_CAPTIONS[sid]
            captions.extend(synth[:1] if dry_run else synth)

        scene_captions[sid] = captions

    # Add COCO-derived captions from actual dataset
    try:
        scenes = load_scenes()
    except Exception:
        log.warning("Could not load scenes from manifest; using synthetic only")
        scenes = []

    for scene in scenes:
        sid = scene["scene_id"]
        if sid not in SCENE_RAG_DOMAIN:
            continue
        if scene["has_coco"]:
            coco_caps = get_coco_captions(scene["coco_data"])
            limit = 1 if dry_run else 5
            scene_captions.setdefault(sid, []).extend(coco_caps[:limit])

    # Ensure every scene has at least one caption (fallback to description-based)
    fallback_captions: dict[str, str] = {
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
    for sid in SCENE_RAG_DOMAIN:
        if not scene_captions.get(sid):
            scene_captions[sid] = [fallback_captions.get(sid, f"Scene {sid}")]

    total = sum(len(v) for v in scene_captions.values())
    log.info("Collected %d captions across %d scenes", total, len(scene_captions))
    return scene_captions


# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------

def _run_bypass(
    scene_captions: dict[str, list[str]],
) -> dict[str, list[SceneClassification]]:
    """Strategy: None (bypass). Always returns 'general' scene_type."""
    results: dict[str, list[SceneClassification]] = {}
    for sid, captions in scene_captions.items():
        scene_results: list[SceneClassification] = []
        for _ in captions:
            scene_results.append(SceneClassification(
                scene_type="general",
                score=0.0,
                rag_triggered=False,
                matched_keywords=[],
                profile_name="general",
            ))
        results[sid] = scene_results
    return results


def _run_adaptive(
    scene_captions: dict[str, list[str]],
    tau_gate: float,
    base_config: Any,
) -> dict[str, list[SceneClassification]]:
    """Strategy: Adaptive with given tau_gate threshold."""
    cfg = override_config(base_config, {"semantic_filter.gate_threshold": tau_gate})

    results: dict[str, list[SceneClassification]] = {}
    for sid, captions in scene_captions.items():
        scene_results: list[SceneClassification] = []
        for cap in captions:
            classification = classify_scene(cap, cfg.semantic_filter)
            scene_results.append(classification)
        results[sid] = scene_results
    return results


def _run_jaccard_only(
    scene_captions: dict[str, list[str]],
    base_config: Any,
) -> dict[str, list[SceneClassification]]:
    """Strategy: Jaccard-only (tau_gate=0.10).

    Forces Jaccard similarity by temporarily patching compute_score()
    with a very high small_set_threshold (999), so the Overlap branch
    is never triggered regardless of keyword count.
    """
    from core.algorithms import semantic_filter as sf_module

    cfg = override_config(base_config, {"semantic_filter.gate_threshold": 0.10})

    original_compute_score = sf_module.compute_score

    def _jaccard_only_score(keywords: list[str], profile: Any) -> float:
        """Jaccard similarity only: |K & D| / |K | D|."""
        if not keywords or not profile.keywords:
            return 0.0
        k_set = set(kw.lower() for kw in keywords)
        d_set = set(kw.lower() for kw in profile.keywords)
        intersection = k_set & d_set
        if not intersection:
            return 0.0
        union = k_set | d_set
        return len(intersection) / len(union)

    try:
        sf_module.compute_score = _jaccard_only_score
        results: dict[str, list[SceneClassification]] = {}
        for sid, captions in scene_captions.items():
            scene_results: list[SceneClassification] = []
            for cap in captions:
                classification = classify_scene(cap, cfg.semantic_filter)
                scene_results.append(classification)
            results[sid] = scene_results
    finally:
        sf_module.compute_score = original_compute_score

    return results


def _run_overlap_only(
    scene_captions: dict[str, list[str]],
    base_config: Any,
) -> dict[str, list[SceneClassification]]:
    """Strategy: Overlap-only (tau_gate=0.10).

    Forces Overlap coefficient by temporarily patching compute_score()
    with small_set_threshold=0, so Overlap is always used regardless
    of keyword count.
    """
    from core.algorithms import semantic_filter as sf_module

    cfg = override_config(base_config, {"semantic_filter.gate_threshold": 0.10})

    original_compute_score = sf_module.compute_score

    def _overlap_only_score(keywords: list[str], profile: Any) -> float:
        """Overlap coefficient only: |K & D| / min(|K|, |D|)."""
        if not keywords or not profile.keywords:
            return 0.0
        k_set = set(kw.lower() for kw in keywords)
        d_set = set(kw.lower() for kw in profile.keywords)
        intersection = k_set & d_set
        if not intersection:
            return 0.0
        min_size = min(len(k_set), len(d_set))
        if min_size == 0:
            return 0.0
        return len(intersection) / min_size

    try:
        sf_module.compute_score = _overlap_only_score
        results: dict[str, list[SceneClassification]] = {}
        for sid, captions in scene_captions.items():
            scene_results: list[SceneClassification] = []
            for cap in captions:
                classification = classify_scene(cap, cfg.semantic_filter)
                scene_results.append(classification)
            results[sid] = scene_results
    finally:
        sf_module.compute_score = original_compute_score

    return results


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_sca(
    results: dict[str, list[SceneClassification]],
) -> dict[str, Any]:
    """Compute Scene Classification Accuracy (SCA).

    A classification is acceptable if scene_type is in
    DOMAIN_ACCEPTABLE[ground_truth_domain].

    Returns:
        Dict with overall SCA, per-scene breakdown, per-domain breakdown.
    """
    total = 0
    correct = 0
    per_scene: dict[str, dict[str, Any]] = {}
    domain_counts: dict[str, dict[str, int]] = {}

    for sid, classifications in results.items():
        gt_domain = SCENE_RAG_DOMAIN.get(sid, "fish")
        acceptable = DOMAIN_ACCEPTABLE.get(gt_domain, ["general"])

        scene_correct = 0
        scene_total = len(classifications)
        for cls in classifications:
            is_acceptable = cls.scene_type in acceptable
            if is_acceptable:
                scene_correct += 1
            total += 1
            correct += int(is_acceptable)

        scene_sca = scene_correct / scene_total if scene_total > 0 else 0.0
        per_scene[sid] = {
            "sca": scene_sca,
            "correct": scene_correct,
            "total": scene_total,
            "gt_domain": gt_domain,
            "predicted_types": [c.scene_type for c in classifications],
        }

        # Aggregate by domain
        dc = domain_counts.setdefault(gt_domain, {"correct": 0, "total": 0})
        dc["correct"] += scene_correct
        dc["total"] += scene_total

    per_domain: dict[str, float] = {}
    for domain, counts in domain_counts.items():
        per_domain[domain] = (
            counts["correct"] / counts["total"]
            if counts["total"] > 0 else 0.0
        )

    overall_sca = correct / total if total > 0 else 0.0

    return {
        "sca": overall_sca,
        "correct": correct,
        "total": total,
        "per_scene": per_scene,
        "per_domain": per_domain,
    }


def _compute_trigger_rates(
    results: dict[str, list[SceneClassification]],
) -> dict[str, float]:
    """Compute RAG trigger rate for fish-domain and false trigger rate
    for environment-domain scenes.

    Returns:
        Dict with rag_trigger_rate and false_trigger_rate.
    """
    # RAG trigger rate: % of fish-domain captions that trigger RAG
    fish_total = 0
    fish_triggered = 0
    for sid in FISH_DOMAIN_SCENES:
        if sid in results:
            for cls in results[sid]:
                fish_total += 1
                if cls.rag_triggered:
                    fish_triggered += 1

    # False trigger rate: % of env-domain captions that trigger RAG
    env_total = 0
    env_triggered = 0
    for sid in ENV_DOMAIN_SCENES:
        if sid in results:
            for cls in results[sid]:
                env_total += 1
                if cls.rag_triggered:
                    env_triggered += 1

    rag_trigger_rate = fish_triggered / fish_total if fish_total > 0 else 0.0
    false_trigger_rate = env_triggered / env_total if env_total > 0 else 0.0

    return {
        "rag_trigger_rate": rag_trigger_rate,
        "false_trigger_rate": false_trigger_rate,
        "fish_triggered": fish_triggered,
        "fish_total": fish_total,
        "env_triggered": env_triggered,
        "env_total": env_total,
    }


def _compute_per_scene_sca_values(
    results: dict[str, list[SceneClassification]],
) -> list[float]:
    """Compute a list of per-scene SCA values for statistical tests."""
    values: list[float] = []
    for sid, classifications in results.items():
        gt_domain = SCENE_RAG_DOMAIN.get(sid, "fish")
        acceptable = DOMAIN_ACCEPTABLE.get(gt_domain, ["general"])
        scene_total = len(classifications)
        if scene_total == 0:
            continue
        scene_correct = sum(
            1 for c in classifications if c.scene_type in acceptable
        )
        values.append(scene_correct / scene_total)
    return values


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def _cochrans_q_test(binary_matrices: list[list[int]]) -> dict[str, float]:
    """Cochran's Q test for k related binary samples.

    Tests whether k treatments (strategies) produce different proportions
    of successes (correct classifications) across n subjects (scenes).

    Args:
        binary_matrices: List of k binary vectors (one per treatment),
            each of length n (one entry per subject). 1=correct, 0=incorrect.

    Returns:
        Dict with 'statistic' (Q) and 'p_value'.
    """
    import scipy.stats as st

    k = len(binary_matrices)
    if k < 3:
        return {"statistic": float("nan"), "p_value": float("nan")}

    # Convert to numpy array: k x n
    data = np.array(binary_matrices)
    n = data.shape[1]
    T = data.sum(axis=1)  # row totals (per treatment)
    C = data.sum(axis=0)  # column totals (per subject)
    N = data.sum()

    numerator = (k - 1) * (k * (T ** 2).sum() - N ** 2)
    denominator = k * N - (C ** 2).sum()
    if denominator == 0:
        return {"statistic": float("nan"), "p_value": float("nan")}
    Q = numerator / denominator
    p_value = 1 - st.chi2.cdf(Q, df=k - 1)
    return {"statistic": float(Q), "p_value": float(p_value)}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    dry_run: bool = False,
    n_runs: int = 3,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Execute the full semantic filter ablation experiment.

    Args:
        dry_run: If True, use 1 caption per scene and 2 tau values only.
        n_runs: Number of independent runs (deterministic for this experiment,
                but kept for interface consistency).
        config_path: Optional path to override config YAML.

    Returns:
        Complete result dict ready for save_result().
    """
    base_config = load_config(config_path)
    scene_captions = _collect_captions(dry_run=dry_run)

    # Build full strategy list
    strategies: list[dict[str, Any]] = []

    # Named strategies
    for ns in NAMED_STRATEGIES:
        strategies.append(ns)

    # Adaptive sweep
    tau_values = ADAPTIVE_TAU_VALUES[:2] if dry_run else ADAPTIVE_TAU_VALUES
    for tau in tau_values:
        strategies.append({"name": f"Adaptive(tau={tau})", "tau_gate": tau})

    log.info(
        "EXP-03 starting: %d strategies x %d scenes",
        len(strategies), len(scene_captions),
    )
    t0 = time.perf_counter()

    # --- Run all strategies ---
    strategy_results: list[dict[str, Any]] = []

    for strat in strategies:
        strat_name = strat["name"]
        tau_gate = strat["tau_gate"]

        log.info("Running strategy: %s", strat_name)

        if tau_gate is None:
            # Bypass strategy
            classifications = _run_bypass(scene_captions)
        elif strat_name == "Jaccard-only":
            classifications = _run_jaccard_only(scene_captions, base_config)
        elif strat_name == "Overlap-only":
            classifications = _run_overlap_only(scene_captions, base_config)
        else:
            classifications = _run_adaptive(
                scene_captions, tau_gate, base_config,
            )

        sca_result = _compute_sca(classifications)
        trigger_rates = _compute_trigger_rates(classifications)
        per_scene_sca = _compute_per_scene_sca_values(classifications)

        strategy_results.append({
            "strategy": strat_name,
            "tau_gate": tau_gate,
            "sca": sca_result["sca"],
            "sca_per_domain": sca_result["per_domain"],
            "sca_per_scene": sca_result["per_scene"],
            "rag_trigger_rate": trigger_rates["rag_trigger_rate"],
            "false_trigger_rate": trigger_rates["false_trigger_rate"],
            "trigger_details": trigger_rates,
            "per_scene_sca_values": per_scene_sca,
        })

    # --- Summary table: adaptive-only sweep ---
    adaptive_results = [
        r for r in strategy_results if r["strategy"].startswith("Adaptive")
    ]
    sweep_rows: list[dict[str, Any]] = []
    for ar in adaptive_results:
        sweep_rows.append({
            "tau_gate": ar["tau_gate"],
            "sca": ar["sca"],
            "rag_trigger_rate": ar["rag_trigger_rate"],
            "false_trigger_rate": ar["false_trigger_rate"],
        })

    # --- Find optimal tau_gate ---
    # Optimize for SCA, breaking ties by lower false_trigger_rate
    if adaptive_results:
        best_adaptive = max(
            adaptive_results,
            key=lambda r: (r["sca"], -r["false_trigger_rate"]),
        )
        optimal_tau = best_adaptive["tau_gate"]
    else:
        optimal_tau = base_config.semantic_filter.gate_threshold

    # --- Statistical comparisons ---
    # Collect per-scene SCA arrays for each strategy
    bypass_sca = []
    optimal_adaptive_sca = []
    all_strategy_sca: list[list[float]] = []

    for sr in strategy_results:
        sca_vals = sr["per_scene_sca_values"]
        all_strategy_sca.append(sca_vals)
        if sr["tau_gate"] is None:
            bypass_sca = sca_vals
        if sr["tau_gate"] == optimal_tau:
            optimal_adaptive_sca = sca_vals

    # Cohen's d: optimal adaptive vs bypass
    effect_sizes: dict[str, float] = {}
    if optimal_adaptive_sca and bypass_sca and len(optimal_adaptive_sca) == len(bypass_sca):
        effect_sizes["d_optimal_vs_bypass"] = cohens_d(
            optimal_adaptive_sca, bypass_sca,
        )

    # Cohen's d: optimal adaptive vs Jaccard-only and Overlap-only
    for sr in strategy_results:
        if sr["strategy"] == "Jaccard-only":
            jaccard_sca = sr["per_scene_sca_values"]
            if optimal_adaptive_sca and jaccard_sca and len(optimal_adaptive_sca) == len(jaccard_sca):
                effect_sizes["d_optimal_vs_jaccard"] = cohens_d(
                    optimal_adaptive_sca, jaccard_sca,
                )
        elif sr["strategy"] == "Overlap-only":
            overlap_sca = sr["per_scene_sca_values"]
            if optimal_adaptive_sca and overlap_sca and len(optimal_adaptive_sca) == len(overlap_sca):
                effect_sizes["d_optimal_vs_overlap"] = cohens_d(
                    optimal_adaptive_sca, overlap_sca,
                )

    # Friedman test across all adaptive strategies
    adaptive_sca_groups = [
        r["per_scene_sca_values"] for r in adaptive_results
    ]
    if (not dry_run
            and len(adaptive_sca_groups) >= 3
            and all(len(g) >= 3 for g in adaptive_sca_groups)):
        # Ensure equal length for Friedman
        min_len = min(len(g) for g in adaptive_sca_groups)
        trimmed = [g[:min_len] for g in adaptive_sca_groups]
        friedman_result = friedman_test(trimmed)
    else:
        friedman_result = {"statistic": float("nan"), "p_value": float("nan")}

    # Cochran's Q test: binary correct/incorrect across 4 strategies
    # (None, Jaccard-only, Overlap-only, best Adaptive)
    cochran_strategies = ["None (bypass)", "Jaccard-only", "Overlap-only"]
    # Add best adaptive strategy
    if adaptive_results:
        best_name = f"Adaptive(tau={optimal_tau})"
        cochran_strategies.append(best_name)

    cochran_binary: list[list[int]] = []
    for sr in strategy_results:
        if sr["strategy"] in cochran_strategies:
            # Convert per-scene SCA to binary: 1.0 = correct, <1.0 = incorrect
            binary = [1 if v >= 1.0 else 0 for v in sr["per_scene_sca_values"]]
            cochran_binary.append(binary)

    if (not dry_run
            and len(cochran_binary) >= 3
            and all(len(b) > 0 for b in cochran_binary)):
        # Ensure equal length
        min_len = min(len(b) for b in cochran_binary)
        trimmed_binary = [b[:min_len] for b in cochran_binary]
        cochran_result = _cochrans_q_test(trimmed_binary)
    else:
        cochran_result = {"statistic": float("nan"), "p_value": float("nan")}

    elapsed = time.perf_counter() - t0

    result = {
        "experiment": "EXP-03: Adaptive Semantic Filter Ablation",
        "hypothesis": (
            "Adaptive gate threshold at tau_gate=0.10-0.15 outperforms "
            "both extremes and bypass"
        ),
        "equations": ["Eq.8"],
        "parameters": {
            "strategies": [s["name"] for s in strategies],
            "adaptive_tau_values": tau_values,
            "n_scenes": len(scene_captions),
            "total_captions": sum(len(v) for v in scene_captions.values()),
            "domain_acceptable": DOMAIN_ACCEPTABLE,
            "rag_trigger_scenes": base_config.semantic_filter.rag_trigger_scenes,
            "dry_run": dry_run,
        },
        "strategy_results": [
            {
                "strategy": sr["strategy"],
                "tau_gate": sr["tau_gate"],
                "sca": sr["sca"],
                "sca_per_domain": sr["sca_per_domain"],
                "rag_trigger_rate": sr["rag_trigger_rate"],
                "false_trigger_rate": sr["false_trigger_rate"],
                "trigger_details": sr["trigger_details"],
            }
            for sr in strategy_results
        ],
        "adaptive_sweep": sweep_rows,
        "optimal_tau_gate": optimal_tau,
        "effect_sizes": effect_sizes,
        "friedman_test": friedman_result,
        "cochrans_q_test": cochran_result,
        "elapsed_seconds": elapsed,
    }

    return result


def print_summary(result: dict[str, Any]) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 95)
    print("EXP-03: Adaptive Semantic Filter Ablation (Eq.8)")
    print("=" * 95)

    # Main table
    print(
        f"{'Strategy':>25s}  "
        f"{'tau_gate':>8s}  "
        f"{'SCA':>6s}  "
        f"{'RAG_trig':>9s}  "
        f"{'FalseTrig':>10s}  "
        f"{'fish_SCA':>9s}  "
        f"{'env_SCA':>8s}  "
        f"{'Optimal':>7s}"
    )
    print("-" * 95)

    optimal_tau = result["optimal_tau_gate"]
    for sr in result["strategy_results"]:
        tau_str = f"{sr['tau_gate']:.3f}" if sr["tau_gate"] is not None else "   N/A"
        is_optimal = sr["tau_gate"] == optimal_tau and sr["tau_gate"] is not None
        marker = "  <--" if is_optimal else ""
        fish_sca = sr["sca_per_domain"].get("fish", float("nan"))
        env_sca = sr["sca_per_domain"].get("aquaculture_env", float("nan"))

        print(
            f"{sr['strategy']:>25s}  "
            f"{tau_str:>8s}  "
            f"{sr['sca']:>6.3f}  "
            f"{sr['rag_trigger_rate']:>9.3f}  "
            f"{sr['false_trigger_rate']:>10.3f}  "
            f"{fish_sca:>9.3f}  "
            f"{env_sca:>8.3f}  "
            f"{marker:>7s}"
        )

    print("-" * 95)
    print(f"Optimal tau_gate: {optimal_tau}")

    # Adaptive sweep sub-table
    if result.get("adaptive_sweep"):
        print("\nAdaptive Sweep Detail:")
        print(f"  {'tau_gate':>8s}  {'SCA':>6s}  {'RAG_trig':>9s}  {'FalseTrig':>10s}")
        print("  " + "-" * 40)
        for row in result["adaptive_sweep"]:
            print(
                f"  {row['tau_gate']:>8.3f}  "
                f"{row['sca']:>6.3f}  "
                f"{row['rag_trigger_rate']:>9.3f}  "
                f"{row['false_trigger_rate']:>10.3f}"
            )

    # Effect sizes
    es = result.get("effect_sizes", {})
    if es:
        print("\nEffect sizes (Cohen's d):")
        for key, val in es.items():
            print(f"  {key}: {val:.4f}")

    # Friedman test
    fr = result.get("friedman_test", {})
    if not np.isnan(fr.get("p_value", float("nan"))):
        print(f"\nFriedman test (adaptive sweep): "
              f"chi2={fr['statistic']:.4f}, p={fr['p_value']:.6f}")

    # Cochran's Q test
    cq = result.get("cochrans_q_test", {})
    if not np.isnan(cq.get("p_value", float("nan"))):
        print(f"\nCochran's Q test (Bypass vs Jaccard vs Overlap vs Adaptive): "
              f"Q={cq['statistic']:.4f}, p={cq['p_value']:.6f}")

    print(f"\nTotal elapsed: {result['elapsed_seconds']:.3f}s")
    print("=" * 95)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP-03: Adaptive Semantic Filter Ablation (Eq.8)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python exp03_semantic_filter.py --dry-run\n"
            "  python exp03_semantic_filter.py --runs 5\n"
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
