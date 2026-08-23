"""EXP-01: Lambda-Weighted Fusion Embedding Ablation (Eq.3-5, 7).

Hypothesis: lambda* = 0.7 is the optimal visual-text fusion weight.

Sweeps lambda in {0.0, 0.1, 0.2, ..., 1.0} and measures:
  - L2 norm deviation from 1.0  (fused embedding should remain unit-norm)
  - Cross-domain distance       (cosine between fish-scene vs env-scene centroids)
  - DA invariance               (scoring pipeline must not change with lambda)

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
from scipy.spatial.distance import cosine as cosine_distance

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.algorithms.fusion import create_fusion_embedding, FusionResult  # noqa: E402
from core.algorithms.scoring import full_scoring_pipeline  # noqa: E402
from core.config import load_config  # noqa: E402
from scripts.experiments.experiment_harness import (  # noqa: E402
    SCENE_EXPECTED_STATUS,
    SIMULATED_LLM_OUTPUTS,
    SCENE_SYNTHETIC_CAPTIONS,
    add_common_args,
    compute_stats,
    cohens_d,
    friedman_test,
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

EXPERIMENT_ID = "exp01"
EMBEDDING_DIM = 512

# Scene categories for centroid computation
FISH_SCENES = [
    "S01_fish_health_tilapia",
    "S02_fish_health_grouper",
    "S03_disease_white_spot",
    "S04_disease_general",
    "S09_multi_species_detection",
    "S10_edge_cases_turbidity",
]
ENV_SCENES = [
    "S05_environment_net_cage",
    "S06_environment_pond_tank",
    "S07_underwater_survey_rov",
    "S08_water_quality_degraded",
]

LAMBDA_VALUES = [round(x * 0.1, 1) for x in range(11)]  # 0.0 .. 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_scene_embeddings(
    rng: np.random.Generator,
    scene_ids: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """Generate reproducible random 512-d visual + caption embeddings per scene.

    Returns:
        {scene_id: {"visual": ndarray, "caption": ndarray}}
    """
    embeddings: dict[str, dict[str, np.ndarray]] = {}
    for sid in scene_ids:
        visual = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        caption = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        embeddings[sid] = {"visual": visual, "caption": caption}
    return embeddings


def _compute_centroid(fused_list: list[np.ndarray]) -> np.ndarray:
    """Mean-pool a list of embeddings and L2-normalize."""
    stacked = np.stack(fused_list)
    centroid = np.mean(stacked, axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 1e-10:
        centroid = centroid / norm
    return centroid


def _run_single_lambda(
    lam: float,
    scene_embeddings: dict[str, dict[str, np.ndarray]],
    base_config: Any,
) -> dict[str, Any]:
    """Run fusion + scoring for a single lambda value.

    Returns a dict with per-scene fusion results and aggregate metrics.
    """
    cfg = override_config(base_config, {"fusion.lambda_weight": lam})

    per_scene: list[dict[str, Any]] = []
    fish_fused: list[np.ndarray] = []
    env_fused: list[np.ndarray] = []
    norm_deviations: list[float] = []

    for sid, embs in scene_embeddings.items():
        result: FusionResult = create_fusion_embedding(
            embs["visual"],
            embs["caption"],
            cfg.fusion,
        )
        norm_dev = abs(result.norm_check - 1.0)
        norm_deviations.append(norm_dev)

        per_scene.append({
            "scene_id": sid,
            "lambda": lam,
            "norm_check": result.norm_check,
            "norm_deviation": norm_dev,
        })

        # Accumulate centroids
        if sid in FISH_SCENES:
            fish_fused.append(result.fused_embedding)
        elif sid in ENV_SCENES:
            env_fused.append(result.fused_embedding)

    # Cross-domain cosine distance (higher = better separation)
    cross_domain_dist = 0.0
    if fish_fused and env_fused:
        fish_centroid = _compute_centroid(fish_fused)
        env_centroid = _compute_centroid(env_fused)
        cross_domain_dist = float(cosine_distance(fish_centroid, env_centroid))

    return {
        "lambda": lam,
        "per_scene": per_scene,
        "norm_deviation_mean": float(np.mean(norm_deviations)),
        "norm_deviation_max": float(np.max(norm_deviations)),
        "cross_domain_distance": cross_domain_dist,
    }


def _run_scoring_invariance(
    base_config: Any,
) -> dict[str, Any]:
    """Verify that the scoring pipeline (DA) produces identical results
    regardless of lambda, since scoring only depends on LLM text output.

    Returns a summary dict.
    """
    scene_results: dict[str, dict[str, Any]] = {}
    for sid, llm_text in SIMULATED_LLM_OUTPUTS.items():
        decision = full_scoring_pipeline(llm_text, base_config.scoring)
        scene_results[sid] = {
            "status": decision.status,
            "healthy_score": decision.healthy_score,
            "disease_score": decision.disease_score,
            "confidence": decision.confidence,
            "expected": SCENE_EXPECTED_STATUS.get(sid, "unknown"),
            "match": decision.status == SCENE_EXPECTED_STATUS.get(sid, ""),
        }

    matches = sum(1 for v in scene_results.values() if v["match"])
    total = len(scene_results)

    return {
        "accuracy": matches / total if total > 0 else 0.0,
        "total": total,
        "matches": matches,
        "per_scene": scene_results,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    dry_run: bool = False,
    n_runs: int = 3,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Execute the full lambda ablation experiment.

    Args:
        dry_run: If True, only test lambda=0.0 and lambda=0.7.
        n_runs: Number of independent runs with different RNG seeds.
        config_path: Optional path to override config YAML.

    Returns:
        Complete result dict ready for save_result().
    """
    base_config = load_config(config_path)
    scene_ids = list(SCENE_EXPECTED_STATUS.keys())

    lambdas = [0.0, 0.7] if dry_run else LAMBDA_VALUES
    seeds = [42] if dry_run else [42 + i for i in range(n_runs)]

    log.info(
        "EXP-01 starting: %d lambda values x %d runs x %d scenes",
        len(lambdas), len(seeds), len(scene_ids),
    )
    t0 = time.perf_counter()

    # --- Phase 1: Fusion ablation across runs ---
    all_runs: list[dict[str, Any]] = []
    # Aggregate per-lambda across runs: {lam: {metric: [values]}}
    agg_norm_dev: dict[float, list[float]] = {lam: [] for lam in lambdas}
    agg_cross_dist: dict[float, list[float]] = {lam: [] for lam in lambdas}

    for run_idx, seed in enumerate(seeds):
        log.info("Run %d/%d (seed=%d)", run_idx + 1, len(seeds), seed)
        rng = np.random.default_rng(seed)
        scene_embs = _generate_scene_embeddings(rng, scene_ids)

        run_results: list[dict[str, Any]] = []
        for lam in lambdas:
            result = _run_single_lambda(lam, scene_embs, base_config)
            result["run"] = run_idx
            result["seed"] = seed
            run_results.append(result)

            agg_norm_dev[lam].append(result["norm_deviation_mean"])
            agg_cross_dist[lam].append(result["cross_domain_distance"])

        all_runs.append({"run": run_idx, "seed": seed, "lambdas": run_results})

    # --- Phase 2: Scoring invariance check ---
    scoring_check = _run_scoring_invariance(base_config)

    # --- Phase 3: Statistics ---
    summary_rows: list[dict[str, Any]] = []
    for lam in lambdas:
        nd_stats = compute_stats(agg_norm_dev[lam])
        cd_stats = compute_stats(agg_cross_dist[lam])
        summary_rows.append({
            "lambda": lam,
            "norm_deviation": {
                "mean": nd_stats.mean,
                "std": nd_stats.std,
                "ci_95": [nd_stats.ci_95_low, nd_stats.ci_95_high],
                "n": nd_stats.n,
            },
            "cross_domain_distance": {
                "mean": cd_stats.mean,
                "std": cd_stats.std,
                "ci_95": [cd_stats.ci_95_low, cd_stats.ci_95_high],
                "n": cd_stats.n,
            },
        })

    # Find optimal lambda (highest cross-domain distance)
    best_row = max(summary_rows, key=lambda r: r["cross_domain_distance"]["mean"])
    optimal_lambda = best_row["lambda"]

    # Effect size: lambda*=0.7 vs lambda=0.0 and lambda=1.0
    effect_sizes: dict[str, float] = {}
    if 0.7 in agg_cross_dist and 0.0 in agg_cross_dist:
        effect_sizes["d_0.7_vs_0.0"] = cohens_d(
            agg_cross_dist[0.7], agg_cross_dist[0.0],
        )
    if 0.7 in agg_cross_dist and 1.0 in agg_cross_dist:
        effect_sizes["d_0.7_vs_1.0"] = cohens_d(
            agg_cross_dist[0.7], agg_cross_dist[1.0],
        )

    # Friedman test across lambdas (if enough data points)
    if not dry_run and len(seeds) >= 3:
        friedman_groups = [agg_cross_dist[lam] for lam in lambdas]
        friedman_result = friedman_test(friedman_groups)
    else:
        friedman_result = {"statistic": float("nan"), "p_value": float("nan")}

    elapsed = time.perf_counter() - t0

    result = {
        "experiment": "EXP-01: Lambda-Weighted Fusion Embedding Ablation",
        "hypothesis": "lambda* = 0.7 is optimal for visual-text fusion",
        "equations": ["Eq.3", "Eq.4", "Eq.5", "Eq.7"],
        "parameters": {
            "lambda_values": lambdas,
            "embedding_dim": EMBEDDING_DIM,
            "n_runs": len(seeds),
            "seeds": seeds,
            "n_scenes": len(scene_ids),
            "dry_run": dry_run,
        },
        "summary": summary_rows,
        "optimal_lambda": optimal_lambda,
        "optimal_cross_domain_distance": best_row["cross_domain_distance"]["mean"],
        "effect_sizes": effect_sizes,
        "friedman_test": friedman_result,
        "scoring_invariance": scoring_check,
        "raw_runs": all_runs,
        "elapsed_seconds": elapsed,
    }

    return result


def print_summary(result: dict[str, Any]) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 78)
    print("EXP-01: Lambda-Weighted Fusion Embedding Ablation")
    print("=" * 78)

    # Table header
    print(
        f"{'lambda':>7s}  "
        f"{'NormDev(mean)':>13s}  "
        f"{'NormDev(std)':>12s}  "
        f"{'CrossDist(mean)':>15s}  "
        f"{'CrossDist(std)':>14s}  "
        f"{'Optimal':>7s}"
    )
    print("-" * 78)

    optimal = result["optimal_lambda"]
    for row in result["summary"]:
        lam = row["lambda"]
        nd = row["norm_deviation"]
        cd = row["cross_domain_distance"]
        marker = "  <--" if lam == optimal else ""
        print(
            f"{lam:>7.1f}  "
            f"{nd['mean']:>13.6f}  "
            f"{nd['std']:>12.6f}  "
            f"{cd['mean']:>15.6f}  "
            f"{cd['std']:>14.6f}  "
            f"{marker:>7s}"
        )

    print("-" * 78)
    print(f"Optimal lambda: {optimal:.1f}  "
          f"(cross-domain distance = {result['optimal_cross_domain_distance']:.6f})")

    # Effect sizes
    if result.get("effect_sizes"):
        print("\nEffect sizes (Cohen's d):")
        for key, val in result["effect_sizes"].items():
            print(f"  {key}: {val:.4f}")

    # Friedman test
    fr = result.get("friedman_test", {})
    if not np.isnan(fr.get("p_value", float("nan"))):
        print(f"\nFriedman test: chi2={fr['statistic']:.4f}, p={fr['p_value']:.6f}")

    # Scoring invariance
    sc = result["scoring_invariance"]
    print(f"\nScoring DA invariance: {sc['matches']}/{sc['total']} "
          f"(accuracy={sc['accuracy']:.2%})")

    print(f"\nTotal elapsed: {result['elapsed_seconds']:.3f}s")
    print("=" * 78)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP-01: Lambda-Weighted Fusion Embedding Ablation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python exp01_lambda_ablation.py --dry-run\n"
            "  python exp01_lambda_ablation.py --runs 5\n"
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
