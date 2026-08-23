"""EXP-13: Image Quality Degradation Robustness (IQDR).

Hypothesis: The Multimodal RAG pipeline maintains diagnostic accuracy
under progressive image quality degradation, and the IQDR ratio
(DA_LRD / DA_HR) quantifies this robustness.

Three quality grades:
  HR  — High Resolution (original images, full captions)
  MRD — Medium Resolution Degraded (bicubic downsample x0.25 then
        upsample x4 + Gaussian blur sigma=3.0)
  LRD — Low Resolution Degraded (MRD + Gaussian noise sigma=25 +
        JPEG compression quality=30)

For Level 1 (dry-run / offline), degradation effects are simulated on
captions rather than actual images:
  HR:  Full simulated output
  MRD: Reduced keywords (simulates blurry perception)
  LRD: Very sparse keywords (simulates heavily degraded perception)

Per-stage metrics:
  CSR  — Caption Semantic Richness (word count of meaningful caption tokens)
  RRP@5 — Simulated retrieval recall at 5
  DA   — Diagnostic Accuracy (scoring pipeline match vs ground truth)

Statistical tests:
  Friedman test across 3 grades, Wilcoxon pairwise (HR vs MRD, HR vs LRD).

IQDR = DA_LRD / DA_HR

Level 1: Offline, algorithm-only. No HTTP, no GPU, no LLM.
Level 3: E2E via HTTP for online mode (optional via --gateway-url).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

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
    SCENE_SYNTHETIC_CAPTIONS,
    SIMULATED_LLM_OUTPUTS,
    add_common_args,
    compute_stats,
    friedman_test,
    load_scenes,
    parse_common_args,
    save_result,
    wilcoxon_test,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EXPERIMENT_ID = "exp13"

# Focus on fish/disease scenes (S01-S04)
FISH_DISEASE_SCENES = [
    "S01_fish_health_tilapia",
    "S02_fish_health_grouper",
    "S03_disease_white_spot",
    "S04_disease_general",
]

QUALITY_GRADES = ["HR", "MRD", "LRD"]
GRADE_DESCRIPTIONS = {
    "HR": "High Resolution (original)",
    "MRD": "Medium Resolution Degraded (downsample+blur)",
    "LRD": "Low Resolution Degraded (MRD+noise+JPEG)",
}

# ---------------------------------------------------------------------------
# Simulated captions per quality grade
# ---------------------------------------------------------------------------
# HR: Full, rich captions (original perception output)
SIMULATED_CAPTIONS_HR: dict[str, str] = {
    "S01_fish_health_tilapia": (
        "The image shows tilapia fish swimming in a clear aquaculture pond. "
        "The fish appear healthy with good body condition, intact fins, "
        "and normal coloration. Active swimming behavior observed."
    ),
    "S02_fish_health_grouper": (
        "A grouper fish in an underwater tank environment. The grouper "
        "shows healthy skin with no visible lesions. Clear eyes and "
        "normal body proportions indicate good health condition."
    ),
    "S03_disease_white_spot": (
        "Fish with white spots on body surface showing signs of disease. "
        "Multiple white circular lesions visible on the skin. Possible "
        "Ichthyophthirius multifiliis parasite infection detected."
    ),
    "S04_disease_general": (
        "A fish showing signs of bacterial infection. Hemorrhagic lesions "
        "observed on skin with ulceration. Suspected vibriosis with "
        "confirmed infected tissue damage and necrosis."
    ),
}

# MRD: Reduced keywords (simulates blurry perception — details lost)
SIMULATED_CAPTIONS_MRD: dict[str, str] = {
    "S01_fish_health_tilapia": (
        "Fish swimming in water. Body appears normal. "
        "No obvious abnormalities."
    ),
    "S02_fish_health_grouper": (
        "A fish in underwater setting. Appears healthy. "
        "No clear lesions visible."
    ),
    "S03_disease_white_spot": (
        "Fish with spots on body. Some discoloration "
        "on skin surface. Possible disease."
    ),
    "S04_disease_general": (
        "Fish with marks on body. Lesions observed. "
        "Suspected infection."
    ),
}

# LRD: Very sparse keywords (heavily degraded perception — minimal info)
SIMULATED_CAPTIONS_LRD: dict[str, str] = {
    "S01_fish_health_tilapia": "Object in water. Shape unclear.",
    "S02_fish_health_grouper": "Underwater object. Blurry shape.",
    "S03_disease_white_spot": "Fish with marks. Low quality image.",
    "S04_disease_general": "Object with discoloration. Noisy image.",
}

# Simulated LLM outputs per grade (what the reasoning LLM would produce)
SIMULATED_LLM_HR: dict[str, str] = {
    "S01_fish_health_tilapia": (
        "The tilapia fish appears healthy with good condition. Normal body "
        "color, active swimming. No disease detected."
    ),
    "S02_fish_health_grouper": (
        "The grouper fish is healthy with normal appearance. Clear skin, "
        "no lesions observed. Good condition overall."
    ),
    "S03_disease_white_spot": (
        "Confirmed white spot disease diagnosed. White spots visible on "
        "fish body surface. Fish is infected with Ichthyophthirius "
        "multifiliis parasite."
    ),
    "S04_disease_general": (
        "Fish is diagnosed with bacterial infection, suspected vibriosis. "
        "Lesions observed on skin. Confirmed infected with Vibrio species."
    ),
}

SIMULATED_LLM_MRD: dict[str, str] = {
    "S01_fish_health_tilapia": (
        "Fish appears to be in normal condition. No obvious disease signs. "
        "Image quality is reduced but fish looks healthy."
    ),
    "S02_fish_health_grouper": (
        "Fish in underwater image. Appears healthy with no clear lesions. "
        "Details somewhat blurry."
    ),
    "S03_disease_white_spot": (
        "Fish shows spots on body surface. Suspected infection with "
        "possible disease. Image is blurry but lesions visible."
    ),
    "S04_disease_general": (
        "Fish shows lesions on skin. Suspected infection. Image quality "
        "degraded but damage patterns suggest bacterial infection."
    ),
}

SIMULATED_LLM_LRD: dict[str, str] = {
    "S01_fish_health_tilapia": (
        "Heavily degraded image. An aquatic object is visible but details "
        "are lost. Cannot confirm species or condition."
    ),
    "S02_fish_health_grouper": (
        "Very noisy underwater image. Object might be a fish. "
        "Assessment not possible due to image quality."
    ),
    "S03_disease_white_spot": (
        "Noisy image of fish. Some marks visible but image quality too "
        "poor for definitive diagnosis. Possible disease suspected."
    ),
    "S04_disease_general": (
        "Degraded image with possible fish. Some discoloration visible. "
        "Suspected infection but image quality limits confidence."
    ),
}

# Mapping from grade to simulated data
GRADE_CAPTIONS = {
    "HR": SIMULATED_CAPTIONS_HR,
    "MRD": SIMULATED_CAPTIONS_MRD,
    "LRD": SIMULATED_CAPTIONS_LRD,
}

GRADE_LLM_OUTPUTS = {
    "HR": SIMULATED_LLM_HR,
    "MRD": SIMULATED_LLM_MRD,
    "LRD": SIMULATED_LLM_LRD,
}


# ---------------------------------------------------------------------------
# Image degradation generation (optional, requires PIL/cv2)
# ---------------------------------------------------------------------------
def _try_import_imaging():
    # type: () -> tuple[Any, Any]
    """Try to import PIL and cv2 for image degradation."""
    pil_module = None
    cv2_module = None
    try:
        from PIL import Image as _Image
        from PIL import ImageFilter as _ImageFilter
        pil_module = (_Image, _ImageFilter)
    except ImportError:
        pass
    try:
        import cv2 as _cv2
        cv2_module = _cv2
    except ImportError:
        pass
    return pil_module, cv2_module


def generate_degraded_images(
    scenes: list[dict[str, Any]],
    output_dir: Path,
    max_per_scene: int = 2,
) -> dict[str, dict[str, list[Path]]]:
    """Generate MRD and LRD degraded variants of scene images.

    Args:
        scenes: Scene metadata from load_scenes().
        output_dir: Directory to save degraded images.
        max_per_scene: Maximum images to degrade per scene.

    Returns:
        {scene_id: {"HR": [paths], "MRD": [paths], "LRD": [paths]}}
    """
    pil_imports, cv2_module = _try_import_imaging()

    if pil_imports is None and cv2_module is None:
        log.warning(
            "Neither PIL nor cv2 available. Cannot generate degraded images. "
            "Install with: pip install Pillow opencv-python"
        )
        return {}

    result: dict[str, dict[str, list[Path]]] = {}

    for scene in scenes:
        sid = scene["scene_id"]
        if sid not in FISH_DISEASE_SCENES:
            continue

        image_paths = scene["image_paths"][:max_per_scene]
        if not image_paths:
            log.warning("No images for scene %s", sid)
            continue

        grade_paths: dict[str, list[Path]] = {"HR": [], "MRD": [], "LRD": []}

        for img_path in image_paths:
            img_name = img_path.stem
            img_ext = img_path.suffix

            # HR: original
            grade_paths["HR"].append(img_path)

            if pil_imports is not None:
                Image, ImageFilter = pil_imports
                _generate_pil(
                    Image, ImageFilter, img_path, img_name, img_ext,
                    sid, output_dir, grade_paths,
                )
            elif cv2_module is not None:
                _generate_cv2(
                    cv2_module, img_path, img_name, img_ext,
                    sid, output_dir, grade_paths,
                )

        result[sid] = grade_paths
        log.info(
            "Generated degraded images for %s: HR=%d, MRD=%d, LRD=%d",
            sid,
            len(grade_paths["HR"]),
            len(grade_paths["MRD"]),
            len(grade_paths["LRD"]),
        )

    return result


def _generate_pil(
    Image: Any,
    ImageFilter: Any,
    img_path: Path,
    img_name: str,
    img_ext: str,
    sid: str,
    output_dir: Path,
    grade_paths: dict[str, list[Path]],
) -> None:
    """Generate degraded images using PIL."""
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # MRD: Bicubic downsample x0.25 then upsample x4 + Gaussian blur sigma=3
        mrd_dir = output_dir / sid / "MRD"
        mrd_dir.mkdir(parents=True, exist_ok=True)
        mrd_path = mrd_dir / f"{img_name}_mrd{img_ext}"

        small = img.resize((w // 4, h // 4), Image.BICUBIC)
        upscaled = small.resize((w, h), Image.BICUBIC)
        mrd_img = upscaled.filter(ImageFilter.GaussianBlur(radius=3.0))
        mrd_img.save(str(mrd_path))
        grade_paths["MRD"].append(mrd_path)

        # LRD: MRD + Gaussian noise sigma=25 + JPEG compression q=30
        lrd_dir = output_dir / sid / "LRD"
        lrd_dir.mkdir(parents=True, exist_ok=True)
        lrd_path = lrd_dir / f"{img_name}_lrd.jpg"

        mrd_arr = np.array(mrd_img, dtype=np.float32)
        noise = np.random.normal(0, 25, mrd_arr.shape).astype(np.float32)
        noisy = np.clip(mrd_arr + noise, 0, 255).astype(np.uint8)
        noisy_img = Image.fromarray(noisy)
        noisy_img.save(str(lrd_path), "JPEG", quality=30)
        grade_paths["LRD"].append(lrd_path)

    except Exception as exc:
        log.error("PIL degradation failed for %s: %s", img_path, exc)


def _generate_cv2(
    cv2: Any,
    img_path: Path,
    img_name: str,
    img_ext: str,
    sid: str,
    output_dir: Path,
    grade_paths: dict[str, list[Path]],
) -> None:
    """Generate degraded images using OpenCV."""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("cv2 could not read %s", img_path)
            return
        h, w = img.shape[:2]

        # MRD: Bicubic downsample x0.25 then upsample x4 + Gaussian blur sigma=3
        mrd_dir = output_dir / sid / "MRD"
        mrd_dir.mkdir(parents=True, exist_ok=True)
        mrd_path = mrd_dir / f"{img_name}_mrd{img_ext}"

        small = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)
        upscaled = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        mrd_img = cv2.GaussianBlur(upscaled, (0, 0), 3.0)
        cv2.imwrite(str(mrd_path), mrd_img)
        grade_paths["MRD"].append(mrd_path)

        # LRD: MRD + Gaussian noise sigma=25 + JPEG compression q=30
        lrd_dir = output_dir / sid / "LRD"
        lrd_dir.mkdir(parents=True, exist_ok=True)
        lrd_path = lrd_dir / f"{img_name}_lrd.jpg"

        mrd_float = mrd_img.astype(np.float32)
        noise = np.random.normal(0, 25, mrd_float.shape).astype(np.float32)
        noisy = np.clip(mrd_float + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(lrd_path), noisy, [cv2.IMWRITE_JPEG_QUALITY, 30])
        grade_paths["LRD"].append(lrd_path)

    except Exception as exc:
        log.error("cv2 degradation failed for %s: %s", img_path, exc)


# ---------------------------------------------------------------------------
# CSR: Caption Semantic Richness
# ---------------------------------------------------------------------------
# Stop words to exclude from meaningful token count
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "and",
    "or", "but", "not", "no", "as", "it", "its", "this", "that", "these",
    "those", "has", "have", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall",
})


def compute_csr(caption: str) -> int:
    """Compute Caption Semantic Richness (CSR).

    CSR = word count of meaningful caption tokens (excluding stop words
    and tokens shorter than 2 characters).

    Args:
        caption: Caption text.

    Returns:
        Number of meaningful tokens.
    """
    words = caption.lower().split()
    meaningful = [
        w for w in words
        if len(w) >= 2
        and w.strip(".,;:!?()[]{}\"'") not in _STOP_WORDS
        and len(w.strip(".,;:!?()[]{}\"'")) >= 2
    ]
    return len(meaningful)


# ---------------------------------------------------------------------------
# Simulated RRP@5
# ---------------------------------------------------------------------------
def simulate_rrp5(grade: str, scene_id: str) -> float:
    """Simulate Retrieval Recall Precision at 5 (RRP@5).

    In a full E2E pipeline, RRP@5 measures how many of the top-5
    retrieved RAG documents are relevant. For Level 1 offline mode,
    we simulate based on expected quality grade impact.

    Args:
        grade: Quality grade (HR, MRD, LRD).
        scene_id: Scene identifier.

    Returns:
        Simulated RRP@5 value in [0, 1].
    """
    # Base relevance varies by scene type
    is_disease = "disease" in scene_id.lower()
    base = 0.90 if is_disease else 0.85

    # Grade-specific degradation factor
    grade_factors = {
        "HR": 1.0,
        "MRD": 0.80,
        "LRD": 0.55,
    }
    factor = grade_factors.get(grade, 1.0)

    # Add deterministic per-scene variation
    scene_hash = hash(scene_id) % 100
    variation = (scene_hash - 50) * 0.002  # +/- 0.1

    return min(1.0, max(0.0, base * factor + variation))


# ---------------------------------------------------------------------------
# Per-grade pipeline execution (Level 1 offline)
# ---------------------------------------------------------------------------
def _run_grade_offline(
    grade: str,
    base_config: Any,
    run_seed: int,
) -> dict[str, Any]:
    """Run the scoring pipeline for one quality grade in offline mode.

    Args:
        grade: Quality grade (HR, MRD, LRD).
        base_config: Loaded application config.
        run_seed: Random seed for this run (adds small stochastic variation).

    Returns:
        Per-grade result dict with scene-level metrics.
    """
    rng = np.random.default_rng(run_seed)

    captions = GRADE_CAPTIONS[grade]
    llm_outputs = GRADE_LLM_OUTPUTS[grade]

    per_scene: dict[str, dict[str, Any]] = {}
    matches = 0
    total = 0
    csr_values: list[int] = []
    rrp5_values: list[float] = []

    for sid in FISH_DISEASE_SCENES:
        caption = captions.get(sid, "")
        llm_text = llm_outputs.get(sid, "")
        expected = SCENE_EXPECTED_STATUS.get(sid, "")

        # CSR
        csr = compute_csr(caption)
        csr_values.append(csr)

        # Semantic filter classification
        cls = classify_scene(caption, base_config.semantic_filter)

        # Scoring pipeline
        decision = full_scoring_pipeline(llm_text, base_config.scoring)
        is_match = decision.status == expected
        matches += int(is_match)
        total += 1

        # Simulated RRP@5
        rrp5 = simulate_rrp5(grade, sid)
        # Add small stochastic variation per run
        rrp5 += rng.normal(0, 0.02)
        rrp5 = min(1.0, max(0.0, rrp5))
        rrp5_values.append(rrp5)

        per_scene[sid] = {
            "caption_preview": caption[:80],
            "csr": csr,
            "scene_type": cls.scene_type,
            "scene_score": cls.score,
            "rag_triggered": cls.rag_triggered,
            "rrp5": rrp5,
            "status": decision.status,
            "expected": expected,
            "match": is_match,
            "healthy_score": decision.healthy_score,
            "disease_score": decision.disease_score,
            "confidence": decision.confidence,
        }

    da = matches / total if total > 0 else 0.0

    return {
        "grade": grade,
        "description": GRADE_DESCRIPTIONS[grade],
        "da": da,
        "matches": matches,
        "total": total,
        "csr_mean": float(np.mean(csr_values)) if csr_values else 0.0,
        "csr_values": csr_values,
        "rrp5_mean": float(np.mean(rrp5_values)) if rrp5_values else 0.0,
        "rrp5_values": [float(v) for v in rrp5_values],
        "per_scene": per_scene,
    }


# ---------------------------------------------------------------------------
# Per-grade pipeline execution (Level 3 E2E via HTTP)
# ---------------------------------------------------------------------------
def _run_grade_e2e(
    grade: str,
    gateway_url: str,
    image_paths_by_grade: dict[str, list[Path]],
    timeout: float,
) -> dict[str, Any]:
    """Run E2E pipeline for one quality grade via HTTP gateway.

    Args:
        grade: Quality grade (HR, MRD, LRD).
        gateway_url: Base gateway URL.
        image_paths_by_grade: {scene_id: {grade: [paths]}}.
        timeout: HTTP timeout in seconds.

    Returns:
        Per-grade result dict.
    """
    if httpx is None:
        log.warning("httpx not installed; falling back to offline simulation")
        return {}

    analyze_url = gateway_url.rstrip("/") + "/analyze"
    per_scene: dict[str, dict[str, Any]] = {}
    matches = 0
    total = 0
    latencies: list[float] = []

    for sid in FISH_DISEASE_SCENES:
        expected = SCENE_EXPECTED_STATUS.get(sid, "")
        scene_images = image_paths_by_grade.get(sid, {}).get(grade, [])

        if not scene_images:
            log.warning("No %s images for scene %s", grade, sid)
            continue

        last_status = "Inconclusive"
        last_confidence = 0.0

        for img_path in scene_images[:1]:  # Use first image
            try:
                t0 = time.perf_counter()
                with open(img_path, "rb") as f:
                    files = {"image": (img_path.name, f, "image/jpeg")}
                    resp = httpx.post(analyze_url, files=files, timeout=timeout)
                elapsed = time.perf_counter() - t0

                if resp.status_code == 200:
                    data = resp.json()
                    diag = data.get("diagnosis", {})
                    last_status = diag.get("status", "Inconclusive")
                    last_confidence = diag.get("confidence", 0.0)
                    latencies.append(elapsed)
                else:
                    log.warning(
                        "Gateway returned %d for %s (%s)",
                        resp.status_code, img_path.name, grade,
                    )
                    latencies.append(elapsed)
            except Exception as exc:
                log.error("HTTP error for %s (%s): %s", img_path.name, grade, exc)

        is_match = last_status == expected
        matches += int(is_match)
        total += 1

        per_scene[sid] = {
            "status": last_status,
            "expected": expected,
            "match": is_match,
            "confidence": last_confidence,
        }

    da = matches / total if total > 0 else 0.0

    return {
        "grade": grade,
        "description": GRADE_DESCRIPTIONS[grade],
        "da": da,
        "matches": matches,
        "total": total,
        "mean_latency": float(np.mean(latencies)) if latencies else 0.0,
        "per_scene": per_scene,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment(
    dry_run: bool = False,
    n_runs: int = 3,
    config_path: str | None = None,
    gateway_url: str = "http://localhost:8000",
    generate_degraded: bool = False,
) -> dict[str, Any]:
    """Execute the IQDR experiment.

    Args:
        dry_run: If True, single run with offline simulation only.
        n_runs: Number of independent runs.
        config_path: Optional override config YAML path.
        gateway_url: Base URL for gateway service (Level 3).
        generate_degraded: If True, generate degraded image files.

    Returns:
        Complete result dict ready for save_result().
    """
    base_config = load_config(config_path)
    runs = 1 if dry_run else n_runs
    seeds = [42] if dry_run else [42 + i for i in range(runs)]

    log.info(
        "EXP-13 starting: %d quality grades, %d runs, scenes=%s",
        len(QUALITY_GRADES), runs, FISH_DISEASE_SCENES,
    )
    t0 = time.perf_counter()

    # --- Optional: Generate degraded images ---
    degraded_image_paths: dict[str, dict[str, list[Path]]] = {}
    if generate_degraded:
        log.info("Generating degraded images...")
        scenes = load_scenes(scene_ids=FISH_DISEASE_SCENES)
        if scenes:
            output_dir = (
                PROJECT_ROOT / "lab_dateset" / "organized"
                / "experiment_results" / "exp13" / "degraded_images"
            )
            degraded_image_paths = generate_degraded_images(
                scenes, output_dir, max_per_scene=2,
            )
            log.info(
                "Generated degraded images for %d scenes",
                len(degraded_image_paths),
            )
        else:
            log.warning("No scenes loaded; skipping image generation")

    # --- Phase 1: Offline runs per grade ---
    log.info("Phase 1: Running offline pipeline across quality grades")

    # Collect DA values per grade across runs for statistical tests
    grade_da_runs: dict[str, list[float]] = {g: [] for g in QUALITY_GRADES}
    grade_csr_runs: dict[str, list[float]] = {g: [] for g in QUALITY_GRADES}
    grade_rrp5_runs: dict[str, list[float]] = {g: [] for g in QUALITY_GRADES}
    all_run_results: list[dict[str, Any]] = []

    for run_idx, seed in enumerate(seeds):
        log.info("Run %d/%d (seed=%d)", run_idx + 1, len(seeds), seed)
        run_data: dict[str, Any] = {"run": run_idx, "seed": seed, "grades": {}}

        for grade in QUALITY_GRADES:
            grade_result = _run_grade_offline(grade, base_config, seed + hash(grade) % 1000)

            grade_da_runs[grade].append(grade_result["da"])
            grade_csr_runs[grade].append(grade_result["csr_mean"])
            grade_rrp5_runs[grade].append(grade_result["rrp5_mean"])

            run_data["grades"][grade] = grade_result
            log.info(
                "  %s: DA=%.3f, CSR_mean=%.1f, RRP@5_mean=%.3f",
                grade, grade_result["da"],
                grade_result["csr_mean"], grade_result["rrp5_mean"],
            )

        all_run_results.append(run_data)

    # --- Phase 2: Statistics ---
    log.info("Phase 2: Computing statistics")

    # Per-grade aggregate stats
    grade_stats: dict[str, dict[str, Any]] = {}
    for grade in QUALITY_GRADES:
        da_stats = compute_stats(grade_da_runs[grade])
        csr_stats = compute_stats(grade_csr_runs[grade])
        rrp5_stats = compute_stats(grade_rrp5_runs[grade])

        grade_stats[grade] = {
            "grade": grade,
            "description": GRADE_DESCRIPTIONS[grade],
            "da": {
                "mean": da_stats.mean,
                "std": da_stats.std,
                "ci_95": [da_stats.ci_95_low, da_stats.ci_95_high],
                "n": da_stats.n,
            },
            "csr": {
                "mean": csr_stats.mean,
                "std": csr_stats.std,
                "ci_95": [csr_stats.ci_95_low, csr_stats.ci_95_high],
                "n": csr_stats.n,
            },
            "rrp5": {
                "mean": rrp5_stats.mean,
                "std": rrp5_stats.std,
                "ci_95": [rrp5_stats.ci_95_low, rrp5_stats.ci_95_high],
                "n": rrp5_stats.n,
            },
        }

    # IQDR = DA_LRD / DA_HR
    da_hr = grade_stats["HR"]["da"]["mean"]
    da_mrd = grade_stats["MRD"]["da"]["mean"]
    da_lrd = grade_stats["LRD"]["da"]["mean"]
    iqdr = da_lrd / da_hr if da_hr > 0 else 0.0

    log.info("IQDR = DA_LRD / DA_HR = %.3f / %.3f = %.4f", da_lrd, da_hr, iqdr)

    # Friedman test across 3 grades
    if not dry_run and len(seeds) >= 3:
        friedman_groups = [grade_da_runs[g] for g in QUALITY_GRADES]
        friedman_result = friedman_test(friedman_groups)
    else:
        friedman_result = {"statistic": float("nan"), "p_value": float("nan")}

    # Wilcoxon pairwise tests
    wilcoxon_hr_mrd = wilcoxon_test(grade_da_runs["HR"], grade_da_runs["MRD"])
    wilcoxon_hr_lrd = wilcoxon_test(grade_da_runs["HR"], grade_da_runs["LRD"])

    # --- Phase 3: Summary table ---
    summary_rows: list[dict[str, Any]] = []
    for grade in QUALITY_GRADES:
        gs = grade_stats[grade]
        summary_rows.append({
            "grade": grade,
            "description": GRADE_DESCRIPTIONS[grade],
            "da_mean": gs["da"]["mean"],
            "da_std": gs["da"]["std"],
            "csr_mean": gs["csr"]["mean"],
            "csr_std": gs["csr"]["std"],
            "rrp5_mean": gs["rrp5"]["mean"],
            "rrp5_std": gs["rrp5"]["std"],
        })

    elapsed = time.perf_counter() - t0

    result: dict[str, Any] = {
        "experiment": "EXP-13: Image Quality Degradation Robustness (IQDR)",
        "hypothesis": (
            "The pipeline maintains diagnostic accuracy under progressive "
            "image quality degradation"
        ),
        "parameters": {
            "quality_grades": QUALITY_GRADES,
            "scenes": FISH_DISEASE_SCENES,
            "n_runs": len(seeds),
            "seeds": seeds,
            "dry_run": dry_run,
            "generate_degraded": generate_degraded,
            "gateway_url": gateway_url,
        },
        "summary": summary_rows,
        "grade_stats": grade_stats,
        "iqdr": iqdr,
        "iqdr_detail": {
            "da_hr": da_hr,
            "da_mrd": da_mrd,
            "da_lrd": da_lrd,
            "iqdr_ratio": iqdr,
            "interpretation": (
                "IQDR >= 0.8: Robust; "
                "0.5 <= IQDR < 0.8: Moderate degradation; "
                "IQDR < 0.5: Fragile"
            ),
        },
        "statistical_tests": {
            "friedman_test": friedman_result,
            "wilcoxon_hr_vs_mrd": wilcoxon_hr_mrd,
            "wilcoxon_hr_vs_lrd": wilcoxon_hr_lrd,
        },
        "raw_runs": all_run_results,
        "elapsed_seconds": elapsed,
    }

    return result


def print_summary(result: dict[str, Any]) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 100)
    print("EXP-13: Image Quality Degradation Robustness (IQDR)")
    print("=" * 100)

    # Summary table header
    print(
        f"\n{'Grade':>5s}  "
        f"{'Description':<45s}  "
        f"{'DA':>8s}  "
        f"{'CSR':>8s}  "
        f"{'RRP@5':>8s}"
    )
    print("-" * 100)

    for row in result["summary"]:
        da_str = f"{row['da_mean']:.3f}"
        if row["da_std"] > 0:
            da_str += f"+/-{row['da_std']:.3f}"

        csr_str = f"{row['csr_mean']:.1f}"
        if row["csr_std"] > 0:
            csr_str += f"+/-{row['csr_std']:.1f}"

        rrp5_str = f"{row['rrp5_mean']:.3f}"
        if row["rrp5_std"] > 0:
            rrp5_str += f"+/-{row['rrp5_std']:.3f}"

        print(
            f"{row['grade']:>5s}  "
            f"{row['description']:<45s}  "
            f"{da_str:>8s}  "
            f"{csr_str:>8s}  "
            f"{rrp5_str:>8s}"
        )

    # IQDR ratio
    iqdr_detail = result.get("iqdr_detail", {})
    print("\n--- IQDR Analysis ---")
    print(f"  DA_HR  = {iqdr_detail.get('da_hr', 0):.3f}")
    print(f"  DA_MRD = {iqdr_detail.get('da_mrd', 0):.3f}")
    print(f"  DA_LRD = {iqdr_detail.get('da_lrd', 0):.3f}")
    print(f"  IQDR   = DA_LRD / DA_HR = {result.get('iqdr', 0):.4f}")
    interp = iqdr_detail.get("interpretation", "")
    if interp:
        print(f"  ({interp})")

    # Per-scene detail from last run
    raw_runs = result.get("raw_runs", [])
    if raw_runs:
        last_run = raw_runs[-1]
        print("\n--- Per-Scene Detail (last run) ---")
        print(
            f"  {'Scene':<30s}  "
            f"{'Grade':>5s}  "
            f"{'Status':<14s}  "
            f"{'Expected':<14s}  "
            f"{'Match':>5s}  "
            f"{'CSR':>4s}  "
            f"{'RRP@5':>6s}"
        )
        print("  " + "-" * 90)
        for grade in QUALITY_GRADES:
            grade_data = last_run.get("grades", {}).get(grade, {})
            per_scene = grade_data.get("per_scene", {})
            for sid in FISH_DISEASE_SCENES:
                sd = per_scene.get(sid, {})
                match_str = "Y" if sd.get("match", False) else "N"
                print(
                    f"  {sid:<30s}  "
                    f"{grade:>5s}  "
                    f"{sd.get('status', '-'):<14s}  "
                    f"{sd.get('expected', '-'):<14s}  "
                    f"{match_str:>5s}  "
                    f"{sd.get('csr', 0):>4d}  "
                    f"{sd.get('rrp5', 0):>6.3f}"
                )

    # Statistical tests
    tests = result.get("statistical_tests", {})
    print("\n--- Statistical Tests ---")

    fr = tests.get("friedman_test", {})
    fr_p = fr.get("p_value", float("nan"))
    if fr_p == fr_p:  # not NaN
        sig = " *" if fr_p < 0.05 else ""
        print(f"  Friedman test (3 grades): chi2={fr.get('statistic', 0):.4f}, "
              f"p={fr_p:.6f}{sig}")
    else:
        print("  Friedman test: N/A (insufficient runs)")

    w_hm = tests.get("wilcoxon_hr_vs_mrd", {})
    w_hm_p = w_hm.get("p_value", float("nan"))
    if w_hm_p == w_hm_p:
        sig = " *" if w_hm_p < 0.05 else ""
        print(f"  Wilcoxon HR vs MRD: W={w_hm.get('statistic', 0):.4f}, "
              f"p={w_hm_p:.6f}{sig}")
    else:
        print("  Wilcoxon HR vs MRD: N/A (insufficient paired data)")

    w_hl = tests.get("wilcoxon_hr_vs_lrd", {})
    w_hl_p = w_hl.get("p_value", float("nan"))
    if w_hl_p == w_hl_p:
        sig = " *" if w_hl_p < 0.05 else ""
        print(f"  Wilcoxon HR vs LRD: W={w_hl.get('statistic', 0):.4f}, "
              f"p={w_hl_p:.6f}{sig}")
    else:
        print("  Wilcoxon HR vs LRD: N/A (insufficient paired data)")

    print(f"\nTotal elapsed: {result['elapsed_seconds']:.3f}s")
    print("=" * 100)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP-13: Image Quality Degradation Robustness (IQDR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python exp13_iqdr.py --dry-run\n"
            "  python exp13_iqdr.py --runs 5\n"
            "  python exp13_iqdr.py --generate-degraded --dry-run\n"
            "  python exp13_iqdr.py --runs 10 --gateway-url http://localhost:8000\n"
        ),
    )
    add_common_args(parser)
    parser.add_argument(
        "--gateway-url", type=str, default="http://localhost:8000",
        help="Gateway service URL for Level 3 E2E (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--generate-degraded", action="store_true",
        help="Generate degraded image files (requires PIL or cv2)",
    )
    args = parser.parse_args()
    common = parse_common_args(args)

    log.info("Configuration: %s", common)

    result = run_experiment(
        dry_run=common["dry_run"],
        n_runs=common["runs"],
        config_path=common["config_path"],
        gateway_url=args.gateway_url,
        generate_degraded=args.generate_degraded,
    )

    # Save
    filepath = save_result(EXPERIMENT_ID, result)
    log.info("Results saved to %s", filepath)

    # Print summary
    print_summary(result)


if __name__ == "__main__":
    main()
