"""Experiment Harness — Unified infrastructure for all experiments.

Provides:
  - Config override mechanism (swap parameters without modifying YAML)
  - Dataset/scene iteration
  - Result collection and JSON persistence
  - Statistical helpers (mean, std, CI, Wilcoxon, Friedman)
  - Dry-run mode for smoke testing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import AppConfig, load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "lab_dateset" / "organized" / "experiment_results"
SCENES_DIR = PROJECT_ROOT / "lab_dateset" / "organized" / "scenes"

SCENE_EXPECTED_STATUS = {
    "S01_fish_health_tilapia": "Healthy",
    "S02_fish_health_grouper": "Healthy",
    "S03_disease_white_spot": "Disease",
    "S04_disease_general": "Disease",
    "S05_environment_net_cage": "Inconclusive",
    "S06_environment_pond_tank": "Inconclusive",
    "S07_underwater_survey_rov": "Inconclusive",
    "S08_water_quality_degraded": "Inconclusive",
    "S09_multi_species_detection": "Healthy",
    "S10_edge_cases_turbidity": "Healthy",
}

SIMULATED_LLM_OUTPUTS = {
    "S01_fish_health_tilapia": "The tilapia fish appears healthy with good condition. Normal body color, active swimming. No disease detected.",
    "S02_fish_health_grouper": "The grouper fish is healthy with normal appearance. Clear skin, no lesions observed. Good condition overall.",
    "S03_disease_white_spot": "Confirmed white spot disease diagnosed. White spots visible on fish body surface. Fish is infected with Ichthyophthirius multifiliis parasite.",
    "S04_disease_general": "Fish is diagnosed with bacterial infection, suspected vibriosis. Lesions observed on skin. Confirmed infected with Vibrio species.",
    "S05_environment_net_cage": "This is an underwater net cage environment with crab and starfish. No fish disease assessment possible from this structural view.",
    "S06_environment_pond_tank": "Sonar depth map showing pond floor topography. Environmental monitoring data, no biological specimens visible.",
    "S07_underwater_survey_rov": "ROV underwater survey footage showing seabed and marine structures. General environment assessment only.",
    "S08_water_quality_degraded": "Degraded underwater image with poor visibility. Water quality appears compromised. Environmental conditions noted.",
    "S09_multi_species_detection": "Multiple aquatic species detected: fish, crab, and shrimp all appear healthy. Normal behavior observed, no disease indicators.",
    "S10_edge_cases_turbidity": "Jellyfish observed in turbid water. No disease detected. The jellyfish appears healthy with normal morphology.",
}

# Synthetic captions for scenes without COCO
SCENE_SYNTHETIC_CAPTIONS = {
    "S01_fish_health_tilapia": [
        "The image shows tilapia fish swimming in a pond with clear water",
        "Several tilapia fish visible underwater in aquaculture pond",
        "Healthy tilapia swimming near the surface of a fish farm tank",
    ],
    "S03_disease_white_spot": [
        "Fish with white spots on body surface, showing signs of disease",
        "Infected fish with visible lesions and discoloration on skin",
    ],
    "S04_disease_general": [
        "Fish showing signs of bacterial infection with skin lesions",
        "Diseased fish with hemorrhage and ulcer on body",
    ],
    "S06_environment_pond_tank": [
        "Sonar depth map of pond floor showing environmental monitoring data",
        "Underwater sensor view of aquaculture tank bottom",
    ],
    "S07_underwater_survey_rov": [
        "ROV footage showing underwater marine structures and seabed",
        "Diver with yellow underwater vehicle in deep blue water",
    ],
    "S08_water_quality_degraded": [
        "Degraded underwater image with poor visibility and turbid water",
        "UFO-120 camera showing murky underwater conditions",
    ],
}


# ---------------------------------------------------------------------------
# Config override
# ---------------------------------------------------------------------------
def override_config(config: AppConfig, overrides: dict[str, Any]) -> AppConfig:
    """Create a new AppConfig with specific parameter overrides.

    Args:
        config: Base configuration.
        overrides: Dict of dotted paths to values.
            e.g. {"fusion.lambda_weight": 0.5, "scoring.healthy_threshold": 3}

    Returns:
        New AppConfig with overrides applied.
    """
    data = config.model_dump()
    for path, value in overrides.items():
        keys = path.split(".")
        d = data
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value
    return AppConfig(**data)


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------
def load_scenes(scene_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Load scene metadata and file paths.

    Args:
        scene_ids: Optional list of scene IDs to load.
            If None, loads all scenes from manifest.

    Returns:
        List of scene dicts with keys: scene_id, description, rag_domain,
        source, file_count, scene_dir, image_paths, has_coco, coco_data.
    """
    manifest_path = SCENES_DIR.parent / "dataset_manifest.json"
    if not manifest_path.exists():
        log.warning("Manifest not found at %s", manifest_path)
        return []

    with open(manifest_path) as f:
        manifest = json.load(f)

    scenes = []
    raw_scenes = manifest.get("scenes", {})
    # Support both dict-keyed and list-of-dicts formats
    if isinstance(raw_scenes, dict):
        scene_items = [(k, v) for k, v in raw_scenes.items()]
    else:
        scene_items = [(e["scene_id"], e) for e in raw_scenes]

    for sid, entry in scene_items:
        if scene_ids and sid not in scene_ids:
            continue

        scene_dir = SCENES_DIR / sid
        if not scene_dir.exists():
            continue

        # Gather image files
        image_paths = sorted(
            p for p in scene_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            and "annotation" not in str(p).lower()
        )

        # Check for COCO annotations
        coco_path = scene_dir / "annotations" / "_annotations_filtered.coco.json"
        coco_data = None
        if coco_path.exists():
            with open(coco_path) as f:
                coco_data = json.load(f)

        scenes.append({
            "scene_id": sid,
            "description": entry.get("description", ""),
            "rag_domain": entry.get("rag_domain", "general"),
            "source": entry.get("source", ""),
            "file_count": entry.get("file_count", len(image_paths)),
            "scene_dir": scene_dir,
            "image_paths": image_paths,
            "has_coco": coco_data is not None,
            "coco_data": coco_data,
        })

    log.info("Loaded %d scenes", len(scenes))
    return scenes


def get_coco_captions(coco_data: dict) -> list[str]:
    """Generate Florence-2 style captions from COCO annotations.

    Builds captions like 'The image shows fish, crab, starfish' from
    category names in annotations.
    """
    if not coco_data:
        return []

    cat_map = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])

    # Group annotations by image
    img_anns: dict[int, set[str]] = {}
    for ann in annotations:
        img_id = ann["image_id"]
        cat_name = cat_map.get(ann["category_id"], "unknown")
        img_anns.setdefault(img_id, set()).add(cat_name)

    captions = []
    for img in images:
        cats = img_anns.get(img["id"], set())
        if cats:
            cat_str = ", ".join(sorted(cats))
            captions.append(f"The image shows {cat_str}")

    return captions


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
@dataclass
class StatResult:
    """Statistical summary of a metric."""

    mean: float
    std: float
    ci_95_low: float
    ci_95_high: float
    n: int
    values: list[float] = field(default_factory=list)


def compute_stats(values: list[float]) -> StatResult:
    """Compute mean, std, and 95% CI for a list of values."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    # 95% CI via t-distribution approximation
    if n > 1:
        import scipy.stats as st
        t_crit = st.t.ppf(0.975, df=n - 1)
        margin = t_crit * std / np.sqrt(n)
    else:
        margin = 0.0

    return StatResult(
        mean=mean,
        std=std,
        ci_95_low=mean - margin,
        ci_95_high=mean + margin,
        n=n,
        values=values,
    )


def wilcoxon_test(x: list[float], y: list[float]) -> dict[str, float]:
    """Wilcoxon signed-rank test for paired samples."""
    import scipy.stats as st
    if len(x) < 5 or len(x) != len(y):
        return {"statistic": float("nan"), "p_value": float("nan")}
    # Check if all differences are zero (Wilcoxon undefined in this case)
    diffs = [a - b for a, b in zip(x, y)]
    if all(d == 0 for d in diffs):
        return {"statistic": 0.0, "p_value": 1.0}
    try:
        stat, p = st.wilcoxon(x, y)
        return {"statistic": float(stat), "p_value": float(p)}
    except ValueError:
        return {"statistic": float("nan"), "p_value": float("nan")}


def friedman_test(groups: list[list[float]]) -> dict[str, float]:
    """Friedman test for k related samples."""
    import scipy.stats as st
    if len(groups) < 3 or any(len(g) < 3 for g in groups):
        return {"statistic": float("nan"), "p_value": float("nan")}
    stat, p = st.friedmanchisquare(*groups)
    return {"statistic": float(stat), "p_value": float(p)}


def cohens_d(x: list[float], y: list[float]) -> float:
    """Cohen's d effect size for two groups."""
    nx, ny = np.array(x), np.array(y)
    pooled_std = np.sqrt(((len(nx) - 1) * np.var(nx, ddof=1) +
                           (len(ny) - 1) * np.var(ny, ddof=1)) /
                          (len(nx) + len(ny) - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(nx) - np.mean(ny)) / pooled_std)


def cliffs_delta(x: list[float], y: list[float]) -> tuple[float, str]:
    """Cliff's delta non-parametric effect size.

    Returns (delta, interpretation) where interpretation is one of:
    negligible (|d|<0.147), small (<0.33), medium (<0.474), large (>=0.474).
    """
    nx, ny = np.array(x), np.array(y)
    n1, n2 = len(nx), len(ny)
    if n1 == 0 or n2 == 0:
        return 0.0, "negligible"
    # Count dominance pairs
    more = 0
    less = 0
    for xi in nx:
        for yj in ny:
            if xi > yj:
                more += 1
            elif xi < yj:
                less += 1
    delta = (more - less) / (n1 * n2)
    # Interpret
    abs_d = abs(delta)
    if abs_d < 0.147:
        interp = "negligible"
    elif abs_d < 0.33:
        interp = "small"
    elif abs_d < 0.474:
        interp = "medium"
    else:
        interp = "large"
    return float(delta), interp


def clopper_pearson_ci(successes: int, n: int,
                        alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson exact confidence interval for a proportion.

    Returns (lower, upper) bounds.
    """
    from scipy.stats import beta as beta_dist
    if n == 0:
        return 0.0, 1.0
    lower = beta_dist.ppf(alpha / 2, successes, n - successes + 1) if successes > 0 else 0.0
    upper = beta_dist.ppf(1 - alpha / 2, successes + 1, n - successes) if successes < n else 1.0
    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------
def save_result(experiment_id: str, result: dict[str, Any],
                sub_dir: str | None = None) -> Path:
    """Save experiment result as JSON.

    Args:
        experiment_id: e.g. "exp01"
        result: Dict to serialize.
        sub_dir: Optional subdirectory under experiment_results/expNN/.

    Returns:
        Path to saved file.
    """
    exp_dir = RESULTS_DIR / experiment_id
    if sub_dir:
        exp_dir = exp_dir / sub_dir
    exp_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_id}_{timestamp}.json"
    filepath = exp_dir / filename

    # Add metadata
    result["_metadata"] = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "config_version": "3.0.0",
    }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, Path):
                return str(obj)
            return super().default(obj)

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    log.info("Saved result to %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# Common CLI args
# ---------------------------------------------------------------------------
def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common experiment CLI arguments."""
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Smoke test: process only 1 sample per scene",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of independent runs (default: 3)",
    )
    parser.add_argument(
        "--scenes", type=str, default=None,
        help="Comma-separated scene IDs (default: all)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Override config YAML path",
    )


def parse_common_args(args: argparse.Namespace) -> dict[str, Any]:
    """Parse common args into a dict."""
    scene_ids = None
    if args.scenes:
        scene_ids = [s.strip() for s in args.scenes.split(",")]

    return {
        "dry_run": args.dry_run,
        "runs": 1 if args.dry_run else args.runs,
        "scene_ids": scene_ids,
        "output_dir": args.output_dir,
        "config_path": args.config,
    }
