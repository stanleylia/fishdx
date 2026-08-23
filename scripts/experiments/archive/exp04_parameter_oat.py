"""EXP-04: Multi-Parameter OAT (One-At-a-Time) Sensitivity Analysis.

Hypothesis: System performance is most sensitive to healthy_threshold and
tau_gate, while embedding-only parameters (lambda, CLIP threshold) have
minimal impact on offline diagnostic accuracy.

Sweeps 9 parameters one at a time (all others held at defaults) and
measures the sensitivity index: max(metric) - min(metric) across sweep.

Metrics per parameter:
  - DA:  Diagnostic Accuracy from scoring pipeline (Eq.9-11)
  - SCA: Scene Classification Accuracy from semantic filter (Eq.8)
  - PSR: Pareidolia Suppression Rate from hard detection (Eq.1a)

Some parameters only affect specific metrics; others are recorded as N/A.

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

from core.algorithms.fusion import create_fusion_embedding  # noqa: E402
from core.algorithms.pareidolia import detect_hard  # noqa: E402
from core.algorithms.scoring import full_scoring_pipeline  # noqa: E402
from core.algorithms.semantic_filter import classify_scene  # noqa: E402
from core.config import load_config  # noqa: E402
from scripts.experiments.experiment_harness import (  # noqa: E402
    SCENE_EXPECTED_STATUS,
    SIMULATED_LLM_OUTPUTS,
    SCENE_SYNTHETIC_CAPTIONS,
    add_common_args,
    compute_stats,
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

EXPERIMENT_ID = "exp04"
EMBEDDING_DIM = 512

# ---------------------------------------------------------------------------
# Parameter definitions
# ---------------------------------------------------------------------------
# Each entry: config_path, default, sweep_values, affected_metrics
PARAMETER_DEFS: list[dict[str, Any]] = [
    {
        "name": "lambda",
        "symbol": "lambda",
        "config_path": "fusion.lambda_weight",
        "default": 0.7,
        "sweep": [0.0, 0.3, 0.5, 0.7, 1.0],
        "affects": ["embedding"],  # Only affects embedding quality, not DA/SCA
        "equation": "Eq.4",
    },
    {
        "name": "tau_clip",
        "symbol": "tau",
        "config_path": "pareidolia.clip_threshold",
        "default": 0.75,
        "sweep": [0.5, 0.625, 0.75, 0.875, 1.0],
        "affects": ["psr"],  # Affects soft path; hard path tested via PSR
        "equation": "Eq.1b",
    },
    {
        "name": "tau_gate",
        "symbol": "tau_gate",
        "config_path": "semantic_filter.gate_threshold",
        "default": 0.10,
        "sweep": [0.05, 0.075, 0.10, 0.125, 0.15, 0.20],
        "affects": ["sca"],
        "equation": "Eq.8",
    },
    {
        "name": "healthy_threshold",
        "symbol": "T_h",
        "config_path": "scoring.healthy_threshold",
        "default": 2,
        "sweep": [1, 2, 3, 4],
        "affects": ["da"],
        "equation": "Eq.11",
    },
    {
        "name": "inconclusive_margin",
        "symbol": "m",
        "config_path": "scoring.inconclusive_margin",
        "default": 1,
        "sweep": [0, 1, 2],
        "affects": ["da"],
        "equation": "Eq.11",
    },
    {
        "name": "grounding_threshold",
        "symbol": "theta",
        "config_path": "verification.grounding_threshold",
        "default": 0.5,
        "sweep": [0.3, 0.5, 0.7],
        "affects": ["offline_na"],  # Requires grounding_fn, record N/A offline
        "equation": "Alg.1",
    },
    {
        "name": "penalty_factor",
        "symbol": "p",
        "config_path": "verification.penalty_factor",
        "default": 0.5,
        "sweep": [0.2, 0.35, 0.5, 0.65, 0.8],
        "affects": ["offline_na"],  # Requires grounding_fn, record N/A offline
        "equation": "Alg.1",
    },
    {
        "name": "top_k",
        "symbol": "k",
        "config_path": "chromadb.top_k",
        "default": 5,
        "sweep": [3, 5, 7, 10],
        "affects": ["offline_na"],  # Requires ChromaDB, record N/A offline
        "equation": "RAG",
    },
    {
        "name": "similarity_cutoff",
        "symbol": "s",
        "config_path": "chromadb.similarity_cutoff",
        "default": 0.5,
        "sweep": [0.3, 0.4, 0.5, 0.6, 0.7],
        "affects": ["offline_na"],  # Requires ChromaDB, record N/A offline
        "equation": "RAG",
    },
]

# Domain acceptable types for SCA (consistent with EXP-03)
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

# Pareidolia test cases (structural context with injected false labels)
PAREIDOLIA_TEST_CASES: list[dict[str, Any]] = [
    {
        "caption": "An underwater view of a fish net cage with mesh structure",
        "labels": ["fish", "person", "net", "cage"],
        "expected_pareidolia": {"person"},
    },
    {
        "caption": "Net enclosure in aquaculture farm with cage and grid",
        "labels": ["fish", "face", "mesh", "human"],
        "expected_pareidolia": {"face", "human"},
    },
    {
        "caption": "Underwater lattice fence structure in fish farming net cage",
        "labels": ["fish", "animal", "tilapia", "head"],
        "expected_pareidolia": {"animal", "head"},
    },
    {
        "caption": "Several tilapia fish swimming in clear pond water",
        "labels": ["fish", "tilapia", "underwater plant"],
        "expected_pareidolia": set(),  # Normal scene, no pareidolia
    },
    {
        "caption": "Healthy fish swimming in aquaculture tank underwater",
        "labels": ["fish", "crab", "shrimp"],
        "expected_pareidolia": set(),  # All real aquatic organisms
    },
]


# ---------------------------------------------------------------------------
# Metric computation: DA (Diagnostic Accuracy)
# ---------------------------------------------------------------------------

def _compute_da(base_config: Any, overrides: dict[str, Any]) -> dict[str, Any]:
    """Run scoring pipeline for all scenes and compute DA.

    Args:
        base_config: Base AppConfig.
        overrides: Parameter overrides dict.

    Returns:
        Dict with da, per_scene details, total, matches.
    """
    cfg = override_config(base_config, overrides) if overrides else base_config

    per_scene: dict[str, dict[str, Any]] = {}
    matches = 0
    total = 0

    for sid, llm_text in SIMULATED_LLM_OUTPUTS.items():
        decision = full_scoring_pipeline(llm_text, cfg.scoring)
        expected = SCENE_EXPECTED_STATUS.get(sid, "")
        is_match = decision.status == expected
        matches += int(is_match)
        total += 1

        per_scene[sid] = {
            "status": decision.status,
            "expected": expected,
            "match": is_match,
            "healthy_score": decision.healthy_score,
            "disease_score": decision.disease_score,
            "confidence": decision.confidence,
        }

    da = matches / total if total > 0 else 0.0
    return {"da": da, "matches": matches, "total": total, "per_scene": per_scene}


# ---------------------------------------------------------------------------
# Metric computation: SCA (Scene Classification Accuracy)
# ---------------------------------------------------------------------------

def _collect_sca_captions(dry_run: bool = False) -> dict[str, list[str]]:
    """Collect captions for SCA computation."""
    scene_captions: dict[str, list[str]] = {}

    for sid in SCENE_RAG_DOMAIN:
        captions: list[str] = []
        if sid in SCENE_SYNTHETIC_CAPTIONS:
            synth = SCENE_SYNTHETIC_CAPTIONS[sid]
            captions.extend(synth[:1] if dry_run else synth)
        scene_captions[sid] = captions

    # COCO captions
    try:
        scenes = load_scenes()
    except Exception:
        scenes = []

    for scene in scenes:
        sid = scene["scene_id"]
        if sid not in SCENE_RAG_DOMAIN:
            continue
        if scene["has_coco"]:
            coco_caps = get_coco_captions(scene["coco_data"])
            limit = 1 if dry_run else 3
            scene_captions.setdefault(sid, []).extend(coco_caps[:limit])

    # Fallback
    fallback: dict[str, str] = {
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
            scene_captions[sid] = [fallback.get(sid, f"Scene {sid}")]

    return scene_captions


def _compute_sca(
    base_config: Any,
    overrides: dict[str, Any],
    scene_captions: dict[str, list[str]],
) -> dict[str, Any]:
    """Run semantic filter for all scenes and compute SCA.

    Args:
        base_config: Base AppConfig.
        overrides: Parameter overrides dict.
        scene_captions: {scene_id: [captions]}.

    Returns:
        Dict with sca, per_scene details.
    """
    cfg = override_config(base_config, overrides) if overrides else base_config

    total = 0
    correct = 0
    per_scene: dict[str, dict[str, Any]] = {}

    for sid, captions in scene_captions.items():
        gt_domain = SCENE_RAG_DOMAIN.get(sid, "fish")
        acceptable = DOMAIN_ACCEPTABLE.get(gt_domain, ["general"])

        scene_correct = 0
        scene_total = len(captions)
        predicted: list[str] = []

        for cap in captions:
            cls = classify_scene(cap, cfg.semantic_filter)
            is_ok = cls.scene_type in acceptable
            scene_correct += int(is_ok)
            total += 1
            correct += int(is_ok)
            predicted.append(cls.scene_type)

        per_scene[sid] = {
            "sca": scene_correct / scene_total if scene_total > 0 else 0.0,
            "predicted_types": predicted,
            "gt_domain": gt_domain,
        }

    sca = correct / total if total > 0 else 0.0
    return {"sca": sca, "correct": correct, "total": total, "per_scene": per_scene}


# ---------------------------------------------------------------------------
# Metric computation: PSR (Pareidolia Suppression Rate)
# ---------------------------------------------------------------------------

def _compute_psr(
    base_config: Any,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Run pareidolia hard detection on test cases and compute PSR.

    PSR = TP / (TP + FN), where TP = correctly flagged pareidolia labels.

    Args:
        base_config: Base AppConfig.
        overrides: Parameter overrides dict.

    Returns:
        Dict with psr, tp, fn, fp, tn, per_case details.
    """
    cfg = override_config(base_config, overrides) if overrides else base_config

    tp = fp = fn = tn = 0
    per_case: list[dict[str, Any]] = []

    for tc in PAREIDOLIA_TEST_CASES:
        results = detect_hard(
            tc["caption"],
            tc["labels"],
            cfg.pareidolia,
        )

        case_tp = case_fp = case_fn = case_tn = 0
        for r in results:
            is_true = r.label in tc["expected_pareidolia"]
            if is_true and r.is_pareidolia:
                case_tp += 1
            elif not is_true and r.is_pareidolia:
                case_fp += 1
            elif is_true and not r.is_pareidolia:
                case_fn += 1
            else:
                case_tn += 1

        tp += case_tp
        fp += case_fp
        fn += case_fn
        tn += case_tn

        per_case.append({
            "caption_prefix": tc["caption"][:50],
            "labels": tc["labels"],
            "expected_pareidolia": list(tc["expected_pareidolia"]),
            "detected": [r.label for r in results if r.is_pareidolia],
            "tp": case_tp, "fp": case_fp, "fn": case_fn, "tn": case_tn,
        })

    psr = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return {
        "psr": psr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "per_case": per_case,
    }


# ---------------------------------------------------------------------------
# Metric computation: Embedding norm deviation
# ---------------------------------------------------------------------------

def _compute_embedding_metric(
    base_config: Any,
    overrides: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Compute embedding norm deviation for fusion parameter sweeps.

    Args:
        base_config: Base AppConfig.
        overrides: Parameter overrides.
        rng: Random number generator.

    Returns:
        Dict with mean_norm_deviation, max_norm_deviation.
    """
    cfg = override_config(base_config, overrides) if overrides else base_config

    deviations: list[float] = []
    for _ in range(10):
        visual = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        caption = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        result = create_fusion_embedding(visual, caption, cfg.fusion)
        deviations.append(abs(result.norm_check - 1.0))

    return {
        "mean_norm_deviation": float(np.mean(deviations)),
        "max_norm_deviation": float(np.max(deviations)),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    dry_run: bool = False,
    n_runs: int = 3,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Execute the full OAT sensitivity analysis.

    For each of the 9 parameters, sweeps its values while holding all
    others at default. Computes DA, SCA, PSR (where applicable) and
    derives the sensitivity index.

    Args:
        dry_run: If True, test only 2 values per parameter.
        n_runs: Not used directly (deterministic), kept for interface.
        config_path: Optional path to override config YAML.

    Returns:
        Complete result dict ready for save_result().
    """
    base_config = load_config(config_path)
    scene_captions = _collect_sca_captions(dry_run=dry_run)
    rng = np.random.default_rng(42)

    # Prepare parameter defs with dry-run trimming
    param_defs = []
    for pdef in PARAMETER_DEFS:
        entry = dict(pdef)
        if dry_run:
            # Keep first and last value only
            sweep = entry["sweep"]
            entry["sweep"] = [sweep[0], sweep[-1]] if len(sweep) > 1 else sweep
        param_defs.append(entry)

    total_evals = sum(len(p["sweep"]) for p in param_defs)
    log.info(
        "EXP-04 starting: %d parameters, %d total evaluations",
        len(param_defs), total_evals,
    )
    t0 = time.perf_counter()

    # --- Sweep each parameter ---
    parameter_results: list[dict[str, Any]] = []

    for pdef in param_defs:
        param_name = pdef["name"]
        config_path_key = pdef["config_path"]
        default_val = pdef["default"]
        sweep_vals = pdef["sweep"]
        affects = pdef["affects"]

        log.info(
            "Sweeping %s (%s): %s",
            param_name, config_path_key, sweep_vals,
        )

        sweep_points: list[dict[str, Any]] = []
        da_values: list[float] = []
        sca_values: list[float] = []
        psr_values: list[float] = []
        emb_values: list[float] = []

        for val in sweep_vals:
            overrides = {config_path_key: val}
            point: dict[str, Any] = {"value": val}

            # DA: Diagnostic Accuracy (scoring pipeline)
            if "da" in affects:
                da_result = _compute_da(base_config, overrides)
                point["da"] = da_result["da"]
                da_values.append(da_result["da"])
                point["da_detail"] = {
                    "matches": da_result["matches"],
                    "total": da_result["total"],
                }
            elif "offline_na" not in affects:
                # Compute DA even for non-DA params to measure cross-impact
                da_result = _compute_da(base_config, overrides)
                point["da"] = da_result["da"]
                da_values.append(da_result["da"])
            else:
                point["da"] = None

            # SCA: Scene Classification Accuracy
            if "sca" in affects:
                sca_result = _compute_sca(base_config, overrides, scene_captions)
                point["sca"] = sca_result["sca"]
                sca_values.append(sca_result["sca"])
            elif "offline_na" not in affects:
                sca_result = _compute_sca(base_config, overrides, scene_captions)
                point["sca"] = sca_result["sca"]
                sca_values.append(sca_result["sca"])
            else:
                point["sca"] = None

            # PSR: Pareidolia Suppression Rate
            if "psr" in affects:
                psr_result = _compute_psr(base_config, overrides)
                point["psr"] = psr_result["psr"]
                psr_values.append(psr_result["psr"])
            elif "offline_na" not in affects:
                psr_result = _compute_psr(base_config, overrides)
                point["psr"] = psr_result["psr"]
                psr_values.append(psr_result["psr"])
            else:
                point["psr"] = None

            # Embedding norm deviation (fusion-specific)
            if "embedding" in affects:
                emb_result = _compute_embedding_metric(
                    base_config, overrides, rng,
                )
                point["embedding"] = emb_result
                emb_values.append(emb_result["mean_norm_deviation"])
            else:
                point["embedding"] = None

            sweep_points.append(point)

        # Compute sensitivity indices: max - min across sweep
        sensitivity: dict[str, float | None] = {}
        if da_values:
            sensitivity["da"] = max(da_values) - min(da_values)
        else:
            sensitivity["da"] = None
        if sca_values:
            sensitivity["sca"] = max(sca_values) - min(sca_values)
        else:
            sensitivity["sca"] = None
        if psr_values:
            sensitivity["psr"] = max(psr_values) - min(psr_values)
        else:
            sensitivity["psr"] = None
        if emb_values:
            sensitivity["embedding"] = max(emb_values) - min(emb_values)
        else:
            sensitivity["embedding"] = None

        # Primary sensitivity index: use DA if available, else SCA, else PSR
        if sensitivity["da"] is not None:
            primary_si = sensitivity["da"]
        elif sensitivity["sca"] is not None:
            primary_si = sensitivity["sca"]
        elif sensitivity["psr"] is not None:
            primary_si = sensitivity["psr"]
        else:
            primary_si = 0.0

        parameter_results.append({
            "name": param_name,
            "symbol": pdef["symbol"],
            "config_path": config_path_key,
            "default": default_val,
            "equation": pdef["equation"],
            "affects": affects,
            "sweep_values": sweep_vals,
            "sweep_points": sweep_points,
            "sensitivity": sensitivity,
            "primary_sensitivity_index": primary_si,
        })

    # --- Tornado chart data: sort by primary sensitivity index descending ---
    tornado_data = sorted(
        [
            {
                "name": pr["name"],
                "symbol": pr["symbol"],
                "config_path": pr["config_path"],
                "primary_si": pr["primary_sensitivity_index"],
                "si_da": pr["sensitivity"]["da"],
                "si_sca": pr["sensitivity"]["sca"],
                "si_psr": pr["sensitivity"]["psr"],
            }
            for pr in parameter_results
        ],
        key=lambda x: x["primary_si"] if x["primary_si"] is not None else -1.0,
        reverse=True,
    )

    # --- Kruskal-Wallis H test per parameter ---
    kruskal_wallis_results: list[dict[str, Any]] = []
    for pr in parameter_results:
        # Collect DA values across sweep points (skip None)
        da_vals = [sp["da"] for sp in pr["sweep_points"] if sp.get("da") is not None]
        if len(da_vals) >= 3:
            # Group DA values per sweep point (each is a single observation per setting)
            # For Kruskal-Wallis, we need groups; treat each sweep value as a group
            # with per-scene DA (binary match) as observations within each group
            groups: list[list[float]] = []
            for sp in pr["sweep_points"]:
                if sp.get("da") is not None and "da_detail" in sp:
                    # Use per-scene match values if available via recomputation
                    groups.append([sp["da"]])
                elif sp.get("da") is not None:
                    groups.append([sp["da"]])
            if len(groups) >= 3:
                from scipy.stats import kruskal as _kruskal
                try:
                    h_stat, p_val = _kruskal(*groups)
                except Exception:
                    h_stat, p_val = float("nan"), float("nan")
                kw_entry = {
                    "parameter": pr["name"],
                    "n_groups": len(groups),
                    "kruskal_wallis": {"H": float(h_stat), "p_value": float(p_val)},
                }
                kruskal_wallis_results.append(kw_entry)
                pr["kruskal_wallis"] = {"H": float(h_stat), "p_value": float(p_val)}
            else:
                pr["kruskal_wallis"] = None
        else:
            pr["kruskal_wallis"] = None

    # --- Identify most sensitive parameter ---
    most_sensitive = tornado_data[0]["name"] if tornado_data else "unknown"

    # --- Default accuracy baseline ---
    baseline_da = _compute_da(base_config, {})
    baseline_sca = _compute_sca(base_config, {}, scene_captions)
    baseline_psr = _compute_psr(base_config, {})

    elapsed = time.perf_counter() - t0

    result = {
        "experiment": "EXP-04: Multi-Parameter OAT Sensitivity Analysis",
        "hypothesis": (
            "System most sensitive to healthy_threshold and tau_gate; "
            "embedding-only parameters have minimal impact on DA"
        ),
        "parameters_tested": len(param_defs),
        "total_evaluations": total_evals,
        "baseline": {
            "da": baseline_da["da"],
            "sca": baseline_sca["sca"],
            "psr": baseline_psr["psr"],
        },
        "parameter_results": parameter_results,
        "tornado_chart": tornado_data,
        "most_sensitive_parameter": most_sensitive,
        "kruskal_wallis_results": kruskal_wallis_results,
        "dry_run": dry_run,
        "elapsed_seconds": elapsed,
    }

    return result


def print_summary(result: dict[str, Any]) -> None:
    """Print a human-readable summary including tornado chart."""
    print("\n" + "=" * 100)
    print("EXP-04: Multi-Parameter OAT Sensitivity Analysis")
    print("=" * 100)

    # Baseline
    bl = result["baseline"]
    print(f"\nBaseline (all defaults): DA={bl['da']:.3f}  "
          f"SCA={bl['sca']:.3f}  PSR={bl['psr']:.3f}")

    # Tornado chart
    print("\n--- Tornado Chart (sorted by primary sensitivity index) ---")
    print(
        f"{'Rank':>4s}  "
        f"{'Parameter':>22s}  "
        f"{'Symbol':>7s}  "
        f"{'SI(DA)':>8s}  "
        f"{'SI(SCA)':>9s}  "
        f"{'SI(PSR)':>9s}  "
        f"{'Primary':>8s}"
    )
    print("-" * 80)

    for rank, entry in enumerate(result["tornado_chart"], 1):
        si_da = f"{entry['si_da']:.4f}" if entry["si_da"] is not None else "   N/A"
        si_sca = f"{entry['si_sca']:.4f}" if entry["si_sca"] is not None else "   N/A"
        si_psr = f"{entry['si_psr']:.4f}" if entry["si_psr"] is not None else "   N/A"
        primary = f"{entry['primary_si']:.4f}" if entry["primary_si"] is not None else "   N/A"

        print(
            f"{rank:>4d}  "
            f"{entry['name']:>22s}  "
            f"{entry['symbol']:>7s}  "
            f"{si_da:>8s}  "
            f"{si_sca:>9s}  "
            f"{si_psr:>9s}  "
            f"{primary:>8s}"
        )

    print("-" * 80)
    print(f"Most sensitive parameter: {result['most_sensitive_parameter']}")

    # Per-parameter sweep detail
    print("\n--- Per-Parameter Sweep Curves ---")
    for pr in result["parameter_results"]:
        affects_str = ", ".join(pr["affects"])
        print(f"\n  {pr['name']} ({pr['config_path']}, {pr['equation']}) "
              f"[affects: {affects_str}]")
        print(f"  Default: {pr['default']}")

        header_parts = [f"{'Value':>10s}"]
        if pr["sensitivity"]["da"] is not None:
            header_parts.append(f"{'DA':>8s}")
        if pr["sensitivity"]["sca"] is not None:
            header_parts.append(f"{'SCA':>8s}")
        if pr["sensitivity"]["psr"] is not None:
            header_parts.append(f"{'PSR':>8s}")
        if pr["sensitivity"].get("embedding") is not None:
            header_parts.append(f"{'NormDev':>10s}")
        print("  " + "  ".join(header_parts))
        print("  " + "-" * (len("  ".join(header_parts)) + 2))

        for sp in pr["sweep_points"]:
            row_parts = [f"{sp['value']:>10}"]
            if pr["sensitivity"]["da"] is not None and sp["da"] is not None:
                marker = " <-- default" if sp["value"] == pr["default"] else ""
                row_parts.append(f"{sp['da']:>8.3f}{marker}")
            if pr["sensitivity"]["sca"] is not None and sp["sca"] is not None:
                row_parts.append(f"{sp['sca']:>8.3f}")
            if pr["sensitivity"]["psr"] is not None and sp["psr"] is not None:
                row_parts.append(f"{sp['psr']:>8.3f}")
            if (pr["sensitivity"].get("embedding") is not None
                    and sp.get("embedding") is not None):
                row_parts.append(f"{sp['embedding']['mean_norm_deviation']:>10.6f}")
            print("  " + "  ".join(row_parts))

    print(f"\nTotal evaluations: {result['total_evaluations']}")
    print(f"Elapsed: {result['elapsed_seconds']:.3f}s")
    print("=" * 100)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP-04: Multi-Parameter OAT Sensitivity Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python exp04_parameter_oat.py --dry-run\n"
            "  python exp04_parameter_oat.py --runs 5\n"
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
