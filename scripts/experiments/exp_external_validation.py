"""External Dataset Validation Experiments (EXP-NEW-01 through EXP-NEW-05).

Validates the Multimodal RAG Pipeline against real fish disease datasets.
All results are from actual execution — no fabricated data.

Experiments:
  EXP-NEW-01: Fish Disease Classification (South Asia, 7 classes)
  EXP-NEW-02: Binary Disease Detection (Alaa, Fresh vs Infected)
  EXP-NEW-03: Florence-2 Caption Quality (Stage 1 only)
  EXP-NEW-04: CLIP Embedding Disease Separability
  EXP-NEW-05: Cross-Dataset Generalization

Usage:
  python exp_external_validation.py --exp 4          # Run EXP-NEW-04 only (fastest)
  python exp_external_validation.py --exp 3          # Run EXP-NEW-03 (Stage 1 only)
  python exp_external_validation.py --exp 1          # Run EXP-NEW-01 (full pipeline)
  python exp_external_validation.py --exp all        # Run all experiments
  python exp_external_validation.py --exp 4 --dry-run  # Quick smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.experiment_harness import (  # noqa: E402
    compute_stats,
    clopper_pearson_ci,
    save_result,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
EXTERNAL_DIR = PROJECT_ROOT / "lab_dateset" / "external_datasets"
RESULTS_DIR = PROJECT_ROOT / "lab_dateset" / "organized" / "experiment_results" / "external_validation"

SOUTH_ASIA_DIR = EXTERNAL_DIR / "fish_disease_south_asia" / "Freshwater Fish Disease Aquaculture in south asia"
DETECTION_DIR = EXTERNAL_DIR / "fish_disease_detection" / "New Dataset"
ALAA_DIR = EXTERNAL_DIR / "fish_disease_alaa" / "Fish Disease Dataset"
CLEANED_DIR = EXTERNAL_DIR / "fish_disease_cleaned" / "Fish Disease Dataset"

# Disease class mapping: folder name → normalized label
DISEASE_CLASSES = {
    "Bacterial diseases - Aeromoniasis": "Aeromoniasis",
    "Bacterial gill disease": "Bacterial Gill Disease",
    "Bacterial Red disease": "Bacterial Red Disease",
    "Fungal diseases Saprolegniasis": "Fungal Saprolegniasis",
    "Healthy Fish": "Healthy",
    "Parasitic diseases": "Parasitic Disease",
    "Viral diseases White tail disease": "Viral White Tail Disease",
    "EUS": "EUS",
}

# For binary classification (Alaa dataset)
BINARY_CLASSES = {
    "FreshFish": "Healthy",
    "InfectedFish": "Disease",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def find_images(directory: Path) -> list[Path]:
    """Recursively find all image files in a directory."""
    images = []
    if not directory.exists():
        return images
    for p in sorted(directory.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file():
            images.append(p)
    return images


def sample_images(
    class_dirs: dict[str, Path],
    n_per_class: int,
    seed: int = 42,
) -> list[tuple[Path, str]]:
    """Sample n images per class from directories.

    Returns:
        List of (image_path, ground_truth_label) tuples.
    """
    rng = random.Random(seed)
    samples = []
    for class_name, class_dir in sorted(class_dirs.items()):
        images = find_images(class_dir)
        if not images:
            log.warning(f"No images found in {class_dir}")
            continue
        selected = rng.sample(images, min(n_per_class, len(images)))
        label = DISEASE_CLASSES.get(class_name, class_name)
        for img in selected:
            samples.append((img, label))
    rng.shuffle(samples)
    return samples


def extract_diagnosis_from_report(report: Any) -> str:
    """Extract the predicted diagnosis label from a DiagnosisReport.

    Maps the report's status + disease_id to one of our class labels.
    """
    status = getattr(report.diagnosis, "status", "Inconclusive")
    disease_id = getattr(report.diagnosis, "disease_id", "")
    llm_text = getattr(report, "llm_response", "")
    if hasattr(report, "metadata") and isinstance(report.metadata, dict):
        llm_text = report.metadata.get("llm_raw_response", llm_text)

    if status == "Healthy":
        return "Healthy"

    # Try to match disease_id or LLM text to known classes
    text_lower = f"{disease_id} {llm_text}".lower()

    disease_keywords = {
        "Aeromoniasis": ["aeromon", "aeromonas"],
        "Bacterial Gill Disease": ["gill disease", "gill rot", "bacterial gill"],
        "Bacterial Red Disease": ["red disease", "red spot", "hemorrhag"],
        "Fungal Saprolegniasis": ["saprolegnia", "fungal", "fungus", "cotton"],
        "Parasitic Disease": ["parasit", "argulus", "ich ", "ichthyo", "white spot"],
        "Viral White Tail Disease": ["white tail", "viral", "wtd"],
        "EUS": ["eus", "epizootic ulcerative", "ulcerative syndrome"],
    }

    for label, keywords in disease_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                return label

    # Fallback: if status is Disease but no specific match
    if status == "Disease":
        return "Disease (Unspecified)"
    return "Inconclusive"


def compute_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict[str, Any]:
    """Compute a confusion matrix and per-class metrics.

    Returns dict with matrix, per_class metrics, and macro averages.
    """
    n = len(labels)
    label_idx = {l: i for i, l in enumerate(labels)}
    matrix = [[0] * n for _ in range(n)]

    for true, pred in zip(y_true, y_pred):
        ti = label_idx.get(true, -1)
        pi = label_idx.get(pred, -1)
        if ti >= 0 and pi >= 0:
            matrix[ti][pi] += 1

    # Per-class precision, recall, F1
    per_class = {}
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(n)) - tp
        fn = sum(matrix[i][j] for j in range(n)) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[label] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    # Macro averages
    precisions = [v["precision"] for v in per_class.values()]
    recalls = [v["recall"] for v in per_class.values()]
    f1s = [v["f1"] for v in per_class.values()]

    return {
        "labels": labels,
        "matrix": matrix,
        "per_class": per_class,
        "macro_precision": round(np.mean(precisions), 4),
        "macro_recall": round(np.mean(recalls), 4),
        "macro_f1": round(np.mean(f1s), 4),
    }


# ---------------------------------------------------------------------------
# EXP-NEW-04: CLIP Embedding Disease Separability (NO LLM, fastest)
# ---------------------------------------------------------------------------

def run_exp04_clip_separability(
    n_per_class: int = 10,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, Any]:
    """EXP-NEW-04: CLIP Embedding Disease Separability.

    Encodes fish disease images with CLIP and measures inter-class
    cosine distance to evaluate embedding quality for disease separability.
    """
    log.info("=" * 60)
    log.info("EXP-NEW-04: CLIP Embedding Disease Separability")
    log.info("=" * 60)

    from core.config import load_config
    from core.models.clip_wrapper import CLIPWrapper

    config = load_config()
    clip = CLIPWrapper(config.models.clip)

    # Collect class directories from South Asia Test/Train
    base_dir = SOUTH_ASIA_DIR
    if not base_dir.exists():
        log.error(f"Dataset not found: {base_dir}")
        return {"error": f"Dataset not found: {base_dir}"}

    # Use Train split for more images per class
    train_dir = base_dir / "Train"
    if not train_dir.exists():
        train_dir = base_dir / "Test"

    class_dirs = {}
    for d in sorted(train_dir.iterdir()):
        if d.is_dir():
            class_dirs[d.name] = d

    actual_n = 2 if dry_run else n_per_class
    samples = sample_images(class_dirs, actual_n, seed)
    log.info(f"Sampled {len(samples)} images from {len(class_dirs)} classes (n_per_class={actual_n})")

    # Encode all images
    t0 = time.perf_counter()
    class_embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
    per_image_results = []

    for img_path, label in samples:
        try:
            from PIL import Image
            pil_img = Image.open(str(img_path)).convert("RGB")
            emb = clip.encode_image(pil_img)
            # Normalize
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            class_embeddings[label].append(emb)
            per_image_results.append({
                "image": img_path.name,
                "class": label,
                "embedding_norm": float(np.linalg.norm(emb)),
            })
        except Exception as e:
            log.warning(f"Failed to encode {img_path}: {e}")
            per_image_results.append({
                "image": img_path.name,
                "class": label,
                "error": str(e),
            })

    encode_time = time.perf_counter() - t0
    log.info(f"Encoded {len(per_image_results)} images in {encode_time:.2f}s")

    # Compute class centroids
    centroids: dict[str, np.ndarray] = {}
    for label, embs in class_embeddings.items():
        if embs:
            centroid = np.mean(np.stack(embs), axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
            centroids[label] = centroid

    # Inter-class cosine distance matrix
    labels_sorted = sorted(centroids.keys())
    n_classes = len(labels_sorted)
    distance_matrix = np.zeros((n_classes, n_classes))

    for i, li in enumerate(labels_sorted):
        for j, lj in enumerate(labels_sorted):
            if i == j:
                distance_matrix[i][j] = 0.0
            else:
                cos_sim = float(np.dot(centroids[li], centroids[lj]))
                distance_matrix[i][j] = 1.0 - cos_sim  # cosine distance

    # Intra-class variance
    intra_class_variance = {}
    for label, embs in class_embeddings.items():
        if len(embs) > 1:
            centroid = centroids[label]
            dists = [float(1.0 - np.dot(e, centroid)) for e in embs]
            intra_class_variance[label] = {
                "mean_distance": round(np.mean(dists), 6),
                "std_distance": round(np.std(dists), 6),
                "n_samples": len(embs),
            }

    # Summary metrics
    inter_distances = []
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            inter_distances.append(distance_matrix[i][j])

    intra_means = [v["mean_distance"] for v in intra_class_variance.values()]

    # Separability ratio: inter-class / intra-class (higher = better)
    avg_inter = np.mean(inter_distances) if inter_distances else 0.0
    avg_intra = np.mean(intra_means) if intra_means else 1.0
    separability_ratio = avg_inter / avg_intra if avg_intra > 0 else float("inf")

    elapsed = time.perf_counter() - t0

    result = {
        "experiment": "EXP-NEW-04: CLIP Embedding Disease Separability",
        "dataset": "fish_disease_south_asia (Train)",
        "model": f"{config.models.clip.model_name} / {config.models.clip.pretrained}",
        "embedding_dim": config.models.clip.embedding_dim,
        "n_per_class": actual_n,
        "total_images": len(per_image_results),
        "n_classes": n_classes,
        "class_labels": labels_sorted,
        "inter_class_distance_matrix": {
            "labels": labels_sorted,
            "matrix": distance_matrix.tolist(),
        },
        "intra_class_variance": intra_class_variance,
        "summary": {
            "avg_inter_class_distance": round(float(avg_inter), 6),
            "avg_intra_class_variance": round(float(avg_intra), 6),
            "separability_ratio": round(float(separability_ratio), 4),
            "min_inter_class_distance": round(float(min(inter_distances)) if inter_distances else 0.0, 6),
            "max_inter_class_distance": round(float(max(inter_distances)) if inter_distances else 0.0, 6),
        },
        "encode_time_seconds": round(encode_time, 3),
        "total_elapsed_seconds": round(elapsed, 3),
        "per_image_seconds": round(encode_time / max(len(per_image_results), 1), 4),
        "per_image_results": per_image_results,
    }

    log.info(f"Avg inter-class distance: {avg_inter:.4f}")
    log.info(f"Avg intra-class variance: {avg_intra:.4f}")
    log.info(f"Separability ratio: {separability_ratio:.4f}")

    return result


# ---------------------------------------------------------------------------
# EXP-NEW-03: Florence-2 Caption Quality (Stage 1 only, no LLM)
# ---------------------------------------------------------------------------

def run_exp03_florence2_caption(
    n_per_class: int = 3,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, Any]:
    """EXP-NEW-03: Florence-2 Caption Quality Assessment.

    Runs Stage 1 perception on fish disease images and evaluates
    whether Florence-2 captions contain disease-relevant keywords.
    """
    log.info("=" * 60)
    log.info("EXP-NEW-03: Florence-2 Caption Quality Assessment")
    log.info("=" * 60)

    from core.config import load_config
    from core.pipeline import MultimodalRAGPipeline

    config = load_config()
    pipeline = MultimodalRAGPipeline(config=config, enable_learning=False)

    # Use South Asia Test split
    test_dir = SOUTH_ASIA_DIR / "Test"
    if not test_dir.exists():
        return {"error": f"Test dir not found: {test_dir}"}

    class_dirs = {}
    for d in sorted(test_dir.iterdir()):
        if d.is_dir():
            class_dirs[d.name] = d

    actual_n = 1 if dry_run else n_per_class
    samples = sample_images(class_dirs, actual_n, seed)
    log.info(f"Sampled {len(samples)} images for caption evaluation")

    # Disease keyword sets for quality assessment
    disease_keywords_map = {
        "Aeromoniasis": {"red", "ulcer", "lesion", "hemorrhage", "wound", "sore", "blood", "infected", "disease", "spot"},
        "Bacterial Gill Disease": {"gill", "swollen", "pale", "disease", "infected", "damage", "lesion"},
        "Bacterial Red Disease": {"red", "hemorrhage", "blood", "lesion", "spot", "disease", "infected", "wound"},
        "Fungal Saprolegniasis": {"cotton", "white", "fungus", "fuzzy", "growth", "mold", "patch", "disease"},
        "Healthy": {"healthy", "normal", "clean", "clear", "good", "swimming", "fish"},
        "Parasitic Disease": {"parasite", "spot", "white", "lesion", "infected", "disease", "worm"},
        "Viral White Tail Disease": {"white", "tail", "disease", "viral", "discolor", "pale", "lesion"},
    }

    t0 = time.perf_counter()
    per_image_results = []
    class_sca_scores = defaultdict(list)

    for img_path, label in samples:
        try:
            log.info(f"Processing {img_path.name} (GT: {label})")
            stage1 = pipeline.analyze_stage1_only(img_path)

            caption = stage1.caption_refined
            raw_caption = stage1.raw_caption
            objects = [obj.label for obj in stage1.objects_refined]
            keywords = stage1.scene_keywords

            # SCA: Semantic Consistency Assessment
            # Check if caption contains disease-relevant keywords
            expected_keywords = disease_keywords_map.get(label, set())
            caption_lower = caption.lower()
            matched_keywords = [kw for kw in expected_keywords if kw in caption_lower]
            sca = len(matched_keywords) / len(expected_keywords) if expected_keywords else 0.0

            # Fish detection: did Florence-2 detect a fish?
            fish_detected = any("fish" in obj.lower() for obj in objects) or "fish" in caption_lower

            class_sca_scores[label].append(sca)

            per_image_results.append({
                "image": img_path.name,
                "ground_truth": label,
                "caption_refined": caption,
                "raw_caption": raw_caption,
                "objects_detected": objects,
                "scene_keywords": keywords,
                "pareidolia_triggered": stage1.pareidolia_triggered,
                "bubble_detected": stage1.bubble_detected,
                "fish_detected": fish_detected,
                "matched_disease_keywords": matched_keywords,
                "sca_score": round(sca, 4),
            })

        except Exception as e:
            log.error(f"Error processing {img_path}: {e}")
            per_image_results.append({
                "image": img_path.name,
                "ground_truth": label,
                "error": str(e),
            })

    elapsed = time.perf_counter() - t0

    # Aggregate SCA per class
    per_class_sca = {}
    for label, scores in class_sca_scores.items():
        stats = compute_stats(scores) if len(scores) > 1 else None
        per_class_sca[label] = {
            "mean_sca": round(np.mean(scores), 4),
            "n": len(scores),
            "std": round(float(np.std(scores)), 4) if len(scores) > 1 else 0.0,
        }

    # Fish detection rate
    total_processed = sum(1 for r in per_image_results if "error" not in r)
    fish_detected_count = sum(1 for r in per_image_results if r.get("fish_detected", False))
    fish_detection_rate = fish_detected_count / total_processed if total_processed > 0 else 0.0

    # Overall SCA
    all_sca = [r["sca_score"] for r in per_image_results if "sca_score" in r]
    overall_sca = np.mean(all_sca) if all_sca else 0.0

    result = {
        "experiment": "EXP-NEW-03: Florence-2 Caption Quality Assessment",
        "dataset": "fish_disease_south_asia (Test)",
        "model": config.models.florence2.model_name,
        "n_per_class": actual_n,
        "total_images": len(per_image_results),
        "total_processed": total_processed,
        "summary": {
            "overall_sca": round(float(overall_sca), 4),
            "fish_detection_rate": round(fish_detection_rate, 4),
            "fish_detected_count": fish_detected_count,
            "total_processed": total_processed,
        },
        "per_class_sca": per_class_sca,
        "per_image_results": per_image_results,
        "elapsed_seconds": round(elapsed, 3),
        "per_image_seconds": round(elapsed / max(total_processed, 1), 3),
    }

    log.info(f"Overall SCA: {overall_sca:.4f}")
    log.info(f"Fish detection rate: {fish_detection_rate:.2%}")

    return result


# ---------------------------------------------------------------------------
# EXP-NEW-01: Full Pipeline Disease Classification
# ---------------------------------------------------------------------------

def run_exp01_disease_classification(
    n_per_class: int = 5,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, Any]:
    """EXP-NEW-01: Fish Disease Classification Validation.

    Runs the full Multimodal RAG Pipeline on real fish disease images
    and evaluates diagnostic accuracy per class.
    """
    log.info("=" * 60)
    log.info("EXP-NEW-01: Fish Disease Classification (Full Pipeline)")
    log.info("=" * 60)

    from core.config import load_config
    from core.pipeline import MultimodalRAGPipeline

    config = load_config()
    pipeline = MultimodalRAGPipeline(config=config, enable_learning=False)

    # Use South Asia Test split
    test_dir = SOUTH_ASIA_DIR / "Test"
    if not test_dir.exists():
        return {"error": f"Test dir not found: {test_dir}"}

    class_dirs = {}
    for d in sorted(test_dir.iterdir()):
        if d.is_dir():
            class_dirs[d.name] = d

    actual_n = 1 if dry_run else n_per_class
    samples = sample_images(class_dirs, actual_n, seed)
    log.info(f"Sampled {len(samples)} images for full pipeline evaluation")

    t0 = time.perf_counter()
    y_true = []
    y_pred = []
    per_image_results = []
    class_labels = sorted(set(DISEASE_CLASSES[k] for k in class_dirs.keys() if k in DISEASE_CLASSES))

    for idx, (img_path, label) in enumerate(samples):
        log.info(f"[{idx+1}/{len(samples)}] Processing {img_path.name} (GT: {label})")
        try:
            report = pipeline.analyze(img_path)

            predicted = extract_diagnosis_from_report(report)
            y_true.append(label)
            y_pred.append(predicted)

            per_image_results.append({
                "image": img_path.name,
                "ground_truth": label,
                "predicted": predicted,
                "correct": label == predicted,
                "status": report.diagnosis.status,
                "disease_id": report.diagnosis.disease_id,
                "healthy_score": report.diagnosis.healthy_score,
                "disease_score": report.diagnosis.disease_score,
                "confidence": report.diagnosis.confidence,
                "rag_triggered": report.metadata.get("rag_triggered", False),
                "latency": report.metadata.get("total_latency", 0.0),
                "provider": report.metadata.get("provider", "unknown"),
            })

            log.info(f"  GT={label}, Pred={predicted}, "
                     f"Status={report.diagnosis.status}, "
                     f"Latency={report.metadata.get('total_latency', 0):.1f}s")

        except Exception as e:
            log.error(f"Error processing {img_path}: {e}")
            y_true.append(label)
            y_pred.append("Error")
            per_image_results.append({
                "image": img_path.name,
                "ground_truth": label,
                "predicted": "Error",
                "error": str(e),
            })

    elapsed = time.perf_counter() - t0

    # Compute metrics
    # Overall DA
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    da = correct / total if total > 0 else 0.0

    # Binary DA: correct disease/healthy classification
    binary_correct = sum(
        1 for t, p in zip(y_true, y_pred)
        if (t == "Healthy") == (p == "Healthy")
    )
    binary_da = binary_correct / total if total > 0 else 0.0

    # Clopper-Pearson CI for DA
    da_ci = clopper_pearson_ci(correct, total)
    binary_ci = clopper_pearson_ci(binary_correct, total)

    # Confusion matrix
    all_labels = class_labels + [l for l in sorted(set(y_pred)) if l not in class_labels]
    cm = compute_confusion_matrix(y_true, y_pred, all_labels)

    # Per-class accuracy
    per_class_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
    for t, p in zip(y_true, y_pred):
        per_class_accuracy[t]["total"] += 1
        if t == p:
            per_class_accuracy[t]["correct"] += 1

    per_class_da = {}
    for label, counts in per_class_accuracy.items():
        acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
        ci = clopper_pearson_ci(counts["correct"], counts["total"])
        per_class_da[label] = {
            "accuracy": round(acc, 4),
            "correct": counts["correct"],
            "total": counts["total"],
            "ci_95": [round(ci[0], 4), round(ci[1], 4)],
        }

    result = {
        "experiment": "EXP-NEW-01: Fish Disease Classification",
        "dataset": "fish_disease_south_asia (Test)",
        "pipeline": "MultimodalRAGPipeline (Unified, 4-stage)",
        "n_per_class": actual_n,
        "total_images": total,
        "class_labels": class_labels,
        "summary": {
            "diagnostic_accuracy": round(da, 4),
            "da_ci_95": [round(da_ci[0], 4), round(da_ci[1], 4)],
            "binary_accuracy": round(binary_da, 4),
            "binary_ci_95": [round(binary_ci[0], 4), round(binary_ci[1], 4)],
            "correct": correct,
            "total": total,
        },
        "confusion_matrix": cm,
        "per_class_accuracy": per_class_da,
        "per_image_results": per_image_results,
        "elapsed_seconds": round(elapsed, 3),
        "avg_latency_seconds": round(elapsed / max(total, 1), 2),
    }

    log.info(f"Overall DA: {da:.2%} ({correct}/{total})")
    log.info(f"Binary DA: {binary_da:.2%} ({binary_correct}/{total})")
    log.info(f"Macro F1: {cm['macro_f1']:.4f}")

    return result


# ---------------------------------------------------------------------------
# EXP-NEW-02: Binary Disease Detection
# ---------------------------------------------------------------------------

def run_exp02_binary_detection(
    n_per_class: int = 10,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, Any]:
    """EXP-NEW-02: Binary Disease Detection (Fresh vs Infected).

    Uses the Alaa dataset for simple binary classification:
    FreshFish → Healthy, InfectedFish → Disease.
    """
    log.info("=" * 60)
    log.info("EXP-NEW-02: Binary Disease Detection")
    log.info("=" * 60)

    from core.config import load_config
    from core.pipeline import MultimodalRAGPipeline

    config = load_config()
    pipeline = MultimodalRAGPipeline(config=config, enable_learning=False)

    if not ALAA_DIR.exists():
        return {"error": f"Dataset not found: {ALAA_DIR}"}

    class_dirs = {}
    for d in sorted(ALAA_DIR.iterdir()):
        if d.is_dir():
            class_dirs[d.name] = d

    actual_n = 2 if dry_run else n_per_class

    rng = random.Random(seed)
    samples = []
    for class_name, class_dir in sorted(class_dirs.items()):
        images = find_images(class_dir)
        selected = rng.sample(images, min(actual_n, len(images)))
        label = BINARY_CLASSES.get(class_name, class_name)
        for img in selected:
            samples.append((img, label))
    rng.shuffle(samples)

    log.info(f"Sampled {len(samples)} images for binary detection")

    t0 = time.perf_counter()
    y_true = []
    y_pred = []
    per_image_results = []

    for idx, (img_path, label) in enumerate(samples):
        log.info(f"[{idx+1}/{len(samples)}] Processing {img_path.name} (GT: {label})")
        try:
            report = pipeline.analyze(img_path)
            status = report.diagnosis.status
            predicted = "Healthy" if status == "Healthy" else "Disease"

            y_true.append(label)
            y_pred.append(predicted)

            per_image_results.append({
                "image": img_path.name,
                "ground_truth": label,
                "predicted": predicted,
                "correct": label == predicted,
                "status": status,
                "healthy_score": report.diagnosis.healthy_score,
                "disease_score": report.diagnosis.disease_score,
                "confidence": report.diagnosis.confidence,
                "latency": report.metadata.get("total_latency", 0.0),
            })

        except Exception as e:
            log.error(f"Error processing {img_path}: {e}")
            y_true.append(label)
            y_pred.append("Error")
            per_image_results.append({
                "image": img_path.name,
                "ground_truth": label,
                "predicted": "Error",
                "error": str(e),
            })

    elapsed = time.perf_counter() - t0

    # Metrics
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    da = correct / total if total > 0 else 0.0
    da_ci = clopper_pearson_ci(correct, total)

    # Sensitivity (recall for Disease) and Specificity (recall for Healthy)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "Disease" and p == "Disease")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "Disease" and p != "Disease")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == "Healthy" and p == "Healthy")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "Healthy" and p != "Healthy")

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    cm = compute_confusion_matrix(y_true, y_pred, ["Healthy", "Disease"])

    result = {
        "experiment": "EXP-NEW-02: Binary Disease Detection",
        "dataset": "fish_disease_alaa (Fresh vs Infected)",
        "pipeline": "MultimodalRAGPipeline (Unified, 4-stage)",
        "n_per_class": actual_n,
        "total_images": total,
        "summary": {
            "diagnostic_accuracy": round(da, 4),
            "da_ci_95": [round(da_ci[0], 4), round(da_ci[1], 4)],
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        },
        "confusion_matrix": cm,
        "per_image_results": per_image_results,
        "elapsed_seconds": round(elapsed, 3),
        "avg_latency_seconds": round(elapsed / max(total, 1), 2),
    }

    log.info(f"Binary DA: {da:.2%} ({correct}/{total})")
    log.info(f"Sensitivity: {sensitivity:.2%}, Specificity: {specificity:.2%}")
    log.info(f"F1: {f1:.4f}")

    return result


# ---------------------------------------------------------------------------
# EXP-NEW-05: Cross-Dataset Generalization
# ---------------------------------------------------------------------------

def run_exp05_cross_dataset(
    n_train_per_class: int = 5,
    n_test_per_class: int = 5,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, Any]:
    """EXP-NEW-05: Cross-Dataset Generalization.

    Trains ChromaDB knowledge base from South Asia dataset,
    tests on fish_disease_detection dataset (different source).
    Measures cross-domain RAG retrieval quality.
    """
    log.info("=" * 60)
    log.info("EXP-NEW-05: Cross-Dataset Generalization")
    log.info("=" * 60)

    from core.config import load_config
    from core.models.clip_wrapper import CLIPWrapper
    from core.models.chromadb_client import ChromaDBClient
    from core.algorithms.fusion import create_fusion_embedding
    from PIL import Image

    config = load_config()
    clip = CLIPWrapper(config.models.clip)

    # Use a temporary ChromaDB collection for this experiment
    import chromadb
    temp_client = chromadb.Client()
    collection = temp_client.get_or_create_collection(
        name="exp05_cross_dataset",
        metadata={"hnsw:space": "cosine"},
    )

    # Phase 1: Build knowledge base from South Asia Train
    train_dir = SOUTH_ASIA_DIR / "Train"
    if not train_dir.exists():
        return {"error": f"Train dir not found: {train_dir}"}

    class_dirs = {}
    for d in sorted(train_dir.iterdir()):
        if d.is_dir():
            class_dirs[d.name] = d

    actual_train = 2 if dry_run else n_train_per_class
    train_samples = sample_images(class_dirs, actual_train, seed)
    log.info(f"Building knowledge base from {len(train_samples)} South Asia images")

    t0 = time.perf_counter()
    train_results = []
    for idx, (img_path, label) in enumerate(train_samples):
        try:
            pil_img = Image.open(str(img_path)).convert("RGB")
            visual_emb = clip.encode_image(pil_img)
            caption_emb = clip.encode_text(f"Fish with {label}")
            fusion = create_fusion_embedding(visual_emb, caption_emb, config.fusion)

            doc_id = f"train_{idx:04d}"
            collection.add(
                ids=[doc_id],
                embeddings=[fusion.fused_embedding.tolist()],
                documents=[f"Fish disease: {label}. Source: South Asia dataset."],
                metadatas=[{"label": label, "source": "south_asia"}],
            )
            train_results.append({"image": img_path.name, "label": label, "status": "ok"})
        except Exception as e:
            train_results.append({"image": img_path.name, "label": label, "error": str(e)})

    build_time = time.perf_counter() - t0
    log.info(f"Knowledge base built in {build_time:.2f}s ({collection.count()} entries)")

    # Phase 2: Test on Detection dataset
    # Detection test_split has flat structure with class prefixes in filename
    test_split_dir = DETECTION_DIR / "test_split"
    # Also try train_split for structured subdirs
    test_source_dir = DETECTION_DIR / "train_split"

    if test_source_dir.exists():
        test_class_dirs = {}
        for d in sorted(test_source_dir.iterdir()):
            if d.is_dir():
                test_class_dirs[d.name] = d
        actual_test = 1 if dry_run else n_test_per_class
        test_samples = sample_images(test_class_dirs, actual_test, seed + 100)
    else:
        return {"error": f"Detection dataset not found: {test_source_dir}"}

    log.info(f"Testing on {len(test_samples)} Detection dataset images")

    y_true = []
    y_pred = []
    per_image_results = []

    for idx, (img_path, label) in enumerate(test_samples):
        log.info(f"[{idx+1}/{len(test_samples)}] Querying {img_path.name} (GT: {label})")
        try:
            pil_img = Image.open(str(img_path)).convert("RGB")
            visual_emb = clip.encode_image(pil_img)
            caption_emb = clip.encode_text(f"Fish image")
            fusion = create_fusion_embedding(visual_emb, caption_emb, config.fusion)

            # Query knowledge base
            results = collection.query(
                query_embeddings=[fusion.fused_embedding.tolist()],
                n_results=min(5, collection.count()),
            )

            # Get top-1 predicted label
            predicted = "Unknown"
            top_similarity = 0.0
            if results and results.get("metadatas") and results["metadatas"][0]:
                predicted = results["metadatas"][0][0].get("label", "Unknown")
                if results.get("distances") and results["distances"][0]:
                    top_similarity = 1.0 - results["distances"][0][0]  # cosine distance → similarity

            y_true.append(label)
            y_pred.append(predicted)

            per_image_results.append({
                "image": img_path.name,
                "ground_truth": label,
                "predicted": predicted,
                "correct": label == predicted,
                "top_similarity": round(top_similarity, 4),
                "top_k_labels": [m.get("label", "?") for m in results["metadatas"][0]] if results.get("metadatas") else [],
            })

        except Exception as e:
            log.error(f"Error: {e}")
            y_true.append(label)
            y_pred.append("Error")
            per_image_results.append({
                "image": img_path.name, "ground_truth": label,
                "predicted": "Error", "error": str(e),
            })

    elapsed = time.perf_counter() - t0

    # Metrics
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    total = len(y_true)
    da = correct / total if total > 0 else 0.0
    da_ci = clopper_pearson_ci(correct, total)

    # Cross-dataset label overlap analysis
    train_labels = set(s[1] for s in train_samples)
    test_labels = set(s[1] for s in test_samples)
    label_overlap = train_labels & test_labels
    unique_to_test = test_labels - train_labels

    all_labels = sorted(set(y_true) | set(y_pred))
    cm = compute_confusion_matrix(y_true, y_pred, all_labels)

    result = {
        "experiment": "EXP-NEW-05: Cross-Dataset Generalization",
        "train_dataset": "fish_disease_south_asia (Train)",
        "test_dataset": "fish_disease_detection (train_split)",
        "n_train_per_class": actual_train,
        "n_test_per_class": actual_test,
        "knowledge_base_size": collection.count(),
        "total_test_images": total,
        "summary": {
            "cross_dataset_da": round(da, 4),
            "da_ci_95": [round(da_ci[0], 4), round(da_ci[1], 4)],
            "correct": correct,
            "total": total,
            "label_overlap": sorted(label_overlap),
            "unique_to_test": sorted(unique_to_test),
            "domain_gap_indicator": round(1.0 - da, 4),
        },
        "confusion_matrix": cm,
        "per_image_results": per_image_results,
        "train_build_time": round(build_time, 3),
        "total_elapsed_seconds": round(elapsed, 3),
    }

    log.info(f"Cross-dataset DA: {da:.2%} ({correct}/{total})")
    log.info(f"Label overlap: {sorted(label_overlap)}")
    log.info(f"Unique to test: {sorted(unique_to_test)}")

    # Cleanup
    temp_client.delete_collection("exp05_cross_dataset")

    return result


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_exp04_summary(result: dict[str, Any]) -> None:
    """Print EXP-NEW-04 summary."""
    print("\n" + "=" * 70)
    print("EXP-NEW-04: CLIP Embedding Disease Separability")
    print("=" * 70)
    s = result["summary"]
    print(f"Model: {result['model']}")
    print(f"Images: {result['total_images']} ({result['n_classes']} classes)")
    print(f"Encoding time: {result['encode_time_seconds']:.2f}s ({result['per_image_seconds']:.4f}s/image)")
    print(f"\nAvg inter-class distance: {s['avg_inter_class_distance']:.4f}")
    print(f"Avg intra-class variance: {s['avg_intra_class_variance']:.6f}")
    print(f"Separability ratio:       {s['separability_ratio']:.2f}")
    print(f"Min inter-class distance: {s['min_inter_class_distance']:.4f}")
    print(f"Max inter-class distance: {s['max_inter_class_distance']:.4f}")

    # Distance matrix
    dm = result["inter_class_distance_matrix"]
    labels = dm["labels"]
    matrix = dm["matrix"]
    # Print abbreviated labels
    short = [l[:12] for l in labels]
    print(f"\nInter-class cosine distance matrix:")
    header = "             " + "  ".join(f"{s:>12s}" for s in short)
    print(header)
    for i, row in enumerate(matrix):
        vals = "  ".join(f"{v:>12.4f}" for v in row)
        print(f"{short[i]:>12s}  {vals}")

    print("=" * 70)


def print_exp03_summary(result: dict[str, Any]) -> None:
    """Print EXP-NEW-03 summary."""
    print("\n" + "=" * 70)
    print("EXP-NEW-03: Florence-2 Caption Quality Assessment")
    print("=" * 70)
    s = result["summary"]
    print(f"Model: {result['model']}")
    print(f"Images: {result['total_images']} ({result['total_processed']} processed)")
    print(f"Time: {result['elapsed_seconds']:.2f}s ({result['per_image_seconds']:.2f}s/image)")
    print(f"\nOverall SCA: {s['overall_sca']:.4f}")
    print(f"Fish detection rate: {s['fish_detection_rate']:.2%}")

    print(f"\nPer-class SCA:")
    for label, data in sorted(result["per_class_sca"].items()):
        print(f"  {label:<30s}  SCA={data['mean_sca']:.4f}  (n={data['n']})")

    print("\nSample captions:")
    for r in result["per_image_results"][:5]:
        if "caption_refined" in r:
            print(f"  [{r['ground_truth']}] {r['image']}")
            print(f"    Caption: {r['caption_refined'][:100]}")
            print(f"    Keywords: {r.get('matched_disease_keywords', [])}")
    print("=" * 70)


def print_exp01_summary(result: dict[str, Any]) -> None:
    """Print EXP-NEW-01 summary."""
    print("\n" + "=" * 70)
    print("EXP-NEW-01: Fish Disease Classification (Full Pipeline)")
    print("=" * 70)
    s = result["summary"]
    print(f"Dataset: {result['dataset']}")
    print(f"Images: {s['total']} ({result['n_per_class']}/class)")
    print(f"Time: {result['elapsed_seconds']:.1f}s ({result['avg_latency_seconds']:.1f}s/image)")
    print(f"\nDiagnostic Accuracy: {s['diagnostic_accuracy']:.2%} "
          f"({s['correct']}/{s['total']}) "
          f"CI95=[{s['da_ci_95'][0]:.2%}, {s['da_ci_95'][1]:.2%}]")
    print(f"Binary Accuracy:     {s['binary_accuracy']:.2%} "
          f"CI95=[{s['binary_ci_95'][0]:.2%}, {s['binary_ci_95'][1]:.2%}]")

    cm = result["confusion_matrix"]
    print(f"\nMacro Precision: {cm['macro_precision']:.4f}")
    print(f"Macro Recall:    {cm['macro_recall']:.4f}")
    print(f"Macro F1:        {cm['macro_f1']:.4f}")

    print(f"\nPer-class accuracy:")
    for label, data in sorted(result["per_class_accuracy"].items()):
        print(f"  {label:<30s}  DA={data['accuracy']:.2%} ({data['correct']}/{data['total']})")
    print("=" * 70)


def print_exp02_summary(result: dict[str, Any]) -> None:
    """Print EXP-NEW-02 summary."""
    print("\n" + "=" * 70)
    print("EXP-NEW-02: Binary Disease Detection")
    print("=" * 70)
    s = result["summary"]
    print(f"Dataset: {result['dataset']}")
    print(f"Images: {s.get('tp', 0) + s.get('fn', 0) + s.get('tn', 0) + s.get('fp', 0)}")
    print(f"Time: {result['elapsed_seconds']:.1f}s ({result['avg_latency_seconds']:.1f}s/image)")
    print(f"\nDiagnostic Accuracy: {s['diagnostic_accuracy']:.2%} "
          f"CI95=[{s['da_ci_95'][0]:.2%}, {s['da_ci_95'][1]:.2%}]")
    print(f"Sensitivity:  {s['sensitivity']:.2%}")
    print(f"Specificity:  {s['specificity']:.2%}")
    print(f"Precision:    {s['precision']:.2%}")
    print(f"F1:           {s['f1']:.4f}")
    print(f"\nConfusion Matrix: TP={s['tp']} FP={s['fp']} FN={s['fn']} TN={s['tn']}")
    print("=" * 70)


def print_exp05_summary(result: dict[str, Any]) -> None:
    """Print EXP-NEW-05 summary."""
    print("\n" + "=" * 70)
    print("EXP-NEW-05: Cross-Dataset Generalization")
    print("=" * 70)
    s = result["summary"]
    print(f"Train: {result['train_dataset']} ({result['knowledge_base_size']} entries)")
    print(f"Test:  {result['test_dataset']} ({s['total']} images)")
    print(f"Time: {result['total_elapsed_seconds']:.1f}s")
    print(f"\nCross-dataset DA: {s['cross_dataset_da']:.2%} "
          f"({s['correct']}/{s['total']}) "
          f"CI95=[{s['da_ci_95'][0]:.2%}, {s['da_ci_95'][1]:.2%}]")
    print(f"Domain gap: {s['domain_gap_indicator']:.4f}")
    print(f"Label overlap: {s['label_overlap']}")
    print(f"Unique to test: {s['unique_to_test']}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="External Dataset Validation Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python exp_external_validation.py --exp 4          # CLIP only\n"
            "  python exp_external_validation.py --exp 3          # Florence-2 only\n"
            "  python exp_external_validation.py --exp 1          # Full pipeline\n"
            "  python exp_external_validation.py --exp all        # All experiments\n"
            "  python exp_external_validation.py --exp 4 --dry-run\n"
        ),
    )
    parser.add_argument(
        "--exp", type=str, required=True,
        help="Experiment number (1-5) or 'all'",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Quick smoke test with minimal samples",
    )
    parser.add_argument(
        "--n-per-class", type=int, default=None,
        help="Override samples per class",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments = args.exp.split(",") if args.exp != "all" else ["4", "3", "1", "2", "5"]

    for exp_id in experiments:
        exp_id = exp_id.strip()

        if exp_id == "4":
            n = args.n_per_class or 10
            result = run_exp04_clip_separability(n, args.seed, args.dry_run)
            if "error" not in result:
                print_exp04_summary(result)
                save_result("exp_new_04", result, sub_dir="external_validation")

        elif exp_id == "3":
            n = args.n_per_class or 3
            result = run_exp03_florence2_caption(n, args.seed, args.dry_run)
            if "error" not in result:
                print_exp03_summary(result)
                save_result("exp_new_03", result, sub_dir="external_validation")

        elif exp_id == "1":
            n = args.n_per_class or 5
            result = run_exp01_disease_classification(n, args.seed, args.dry_run)
            if "error" not in result:
                print_exp01_summary(result)
                save_result("exp_new_01", result, sub_dir="external_validation")

        elif exp_id == "2":
            n = args.n_per_class or 10
            result = run_exp02_binary_detection(n, args.seed, args.dry_run)
            if "error" not in result:
                print_exp02_summary(result)
                save_result("exp_new_02", result, sub_dir="external_validation")

        elif exp_id == "5":
            n = args.n_per_class or 5
            result = run_exp05_cross_dataset(n, n, args.seed, args.dry_run)
            if "error" not in result:
                print_exp05_summary(result)
                save_result("exp_new_05", result, sub_dir="external_validation")

        else:
            log.error(f"Unknown experiment: {exp_id}")
            continue

        if "error" in result:
            log.error(f"Experiment {exp_id} failed: {result['error']}")


if __name__ == "__main__":
    main()
