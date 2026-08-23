#!/usr/bin/env python3
"""
E2: Matched-Coverage kNN Deferral Baseline

Sweep CLIP kNN k=1 with similarity-threshold rejection to match pipeline coverage.
Determine whether pipeline beats kNN-deferral at same coverage level.

THIS IS THE LINCHPIN EXPERIMENT FOR PAPER ACCEPTANCE.

Output:
  - results/e2_coverage_accuracy_curve.json
    {
      "target_coverage": 0.643,
      "matched_threshold": float,
      "matched_coverage": float,
      "matched_selective_accuracy": float,
      "pipeline_selective_accuracy": 0.924,
      "comparison": "pipeline_wins" | "tied" | "knn_wins"
    }

Timeline: ~2–3 days
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import open_clip
import logging
import sys
from PIL import Image

# ============================================================================
# CONFIG
# ============================================================================

D2_DIR = Path("./data/datasets/D2")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
TARGET_COVERAGE = 0.643  # Pipeline's coverage @ θ_margin=0.02
PIPELINE_SELECTIVE_ACC = 0.924  # Expected from pipeline

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# ============================================================================
# LOGGING
# ============================================================================

Path("logs").mkdir(parents=True, exist_ok=True)  # create log dir on demand (not shipped)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/e2.log"),
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# MAIN LOGIC
# ============================================================================

def get_image_files(directory: Path) -> list:
    """Get all image files from directory."""
    files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        files.extend(sorted(directory.glob(ext)))
    return files

def compute_clip_embedding(image_path: Path, model, preprocessor) -> np.ndarray:
    """Compute CLIP image embedding."""
    try:
        img = Image.open(image_path).convert('RGB')
        with torch.no_grad():
            image_input = preprocessor(img).unsqueeze(0).to(DEVICE)
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features[0].cpu().numpy()
    except Exception as e:
        logger.warning(f"Error embedding {image_path}: {e}")
        return None

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalized vectors."""
    return float(np.dot(vec1, vec2))

def sweep_similarity_threshold(embeddings: np.ndarray, labels: np.ndarray, k: int = 1,
                                threshold_range=None) -> list:
    """
    Sweep similarity threshold for kNN k=1 rejection.

    For each threshold τ:
      - Compute k-NN for each query
      - Reject if similarity(query, nn) < τ
      - Record: coverage (% decided), accuracy on decided cases
    """

    if threshold_range is None:
        threshold_range = np.arange(0.0, 1.01, 0.01)

    results = []
    n_total = len(embeddings)

    logger.info("Sweeping similarity thresholds...")

    for tau in tqdm(threshold_range, desc="Threshold sweep"):
        num_decided = 0
        num_correct = 0

        for i in range(n_total):
            query_emb = embeddings[i]

            # Compute cosine similarity with all images
            sims = np.dot(embeddings, query_emb)
            sims[i] = -np.inf  # Exclude self

            # Find nearest neighbor
            nn_idx = np.argmax(sims)
            nn_sim = sims[nn_idx]

            # Decide or defer based on threshold
            if nn_sim >= tau:
                num_decided += 1
                if labels[nn_idx] == labels[i]:
                    num_correct += 1

        coverage = num_decided / n_total
        accuracy = num_correct / num_decided if num_decided > 0 else np.nan

        results.append({
            'threshold': float(tau),
            'coverage': float(coverage),
            'selective_accuracy': float(accuracy),
            'num_decided': int(num_decided),
        })

    return results

def find_matched_threshold(results: list, target_coverage: float = 0.643) -> dict:
    """Find threshold closest to target coverage."""
    return min(results, key=lambda x: abs(x['coverage'] - target_coverage))

def main():
    logger.info("=" * 80)
    logger.info("E2: MATCHED-COVERAGE kNN DEFERRAL BASELINE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("CRITICAL EXPERIMENT: Does pipeline beat kNN-deferral at matched coverage?")
    logger.info("")

    # Get D2 image files
    d2_files = [f for f in get_image_files(D2_DIR) if "_aug" not in f.stem]  # exclude augmented
    logger.info(f"D2 images found (noaug): {len(d2_files)}")

    if len(d2_files) == 0:
        logger.error(f"No images found in {D2_DIR}")
        sys.exit(1)

    # Load CLIP model
    logger.info("\nLoading CLIP model (ViT-B/32)...")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='laion2b_s34b_b79k',
            device=DEVICE
        )
        model.eval()
        logger.info(f"✓ CLIP model loaded on device: {DEVICE}")
    except Exception as e:
        logger.error(f"Failed to load CLIP model: {e}")
        sys.exit(1)

    # Compute CLIP embeddings for all D2 images
    logger.info(f"\n[PHASE 1] Computing CLIP embeddings for D2 images ({len(d2_files)})...")
    logger.info("(GPU compute-bound phase)")

    embeddings_list = []
    labels_list = []
    valid_files = []

    for img_file in tqdm(d2_files, desc="D2 CLIP"):
        emb = compute_clip_embedding(img_file, model, preprocess)
        if emb is not None:
            embeddings_list.append(emb)
            # Extract label from filename (simple heuristic: before first digit/space)
            label_str = img_file.stem.split('_')[0] if '_' in img_file.stem else img_file.stem[:10]
            labels_list.append(label_str)
            valid_files.append(img_file)

    embeddings = np.array(embeddings_list)
    labels = np.array(labels_list)

    logger.info(f"✓ Computed embeddings for {len(embeddings)} images")

    # Sweep similarity thresholds
    logger.info(f"\n[PHASE 2] Sweeping similarity thresholds...")
    threshold_range = np.arange(0.0, 1.01, 0.01)
    results = sweep_similarity_threshold(embeddings, labels, k=1, threshold_range=threshold_range)

    # Find matched threshold
    logger.info(f"\n[PHASE 3] Finding matched threshold...")
    matched = find_matched_threshold(results, TARGET_COVERAGE)

    # Comparison
    knn_selective_acc = matched['selective_accuracy']

    if knn_selective_acc > PIPELINE_SELECTIVE_ACC + 0.01:
        comparison = "knn_wins"
        verdict = "❌ kNN beats pipeline"
        implication = "Pipeline contribution narrowed to diagnostic + deployment only"
    elif knn_selective_acc < PIPELINE_SELECTIVE_ACC - 0.01:
        comparison = "pipeline_wins"
        verdict = "✓ Pipeline beats kNN"
        implication = "Pipeline has genuine selective-prediction edge"
    else:
        comparison = "tied"
        verdict = "~ Tied within margin"
        implication = "Contribution unclear; may need deeper analysis"

    # Report results
    logger.info(f"\n{'='*80}")
    logger.info(f"E2 RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Target coverage: {TARGET_COVERAGE*100:.1f}%")
    logger.info(f"Matched threshold τ_reject: {matched['threshold']:.4f}")
    logger.info(f"Actual coverage: {matched['coverage']*100:.1f}%")
    logger.info(f"kNN Selective Accuracy @ matched coverage: {knn_selective_acc:.4f}")
    logger.info(f"Pipeline Selective Accuracy (expected): {PIPELINE_SELECTIVE_ACC:.4f}")
    logger.info(f"\n{verdict}")
    logger.info(f"Implication: {implication}")
    logger.info(f"\n{'='*80}")

    # Save results
    summary = {
        "target_coverage": TARGET_COVERAGE,
        "matched_threshold": matched['threshold'],
        "matched_coverage": matched['coverage'],
        "matched_selective_accuracy": float(knn_selective_acc),
        "pipeline_selective_accuracy": PIPELINE_SELECTIVE_ACC,
        "comparison": comparison,
        "verdict": verdict,
        "implication": implication,
        "full_curve": results,
    }

    results_file = RESULTS_DIR / "e2_noaug_curve.json"
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n✓ Results saved to {results_file}")
    logger.info(f"\n[PHASE 4] E2 COMPLETE")
    logger.info(f"{'='*80}\n")

if __name__ == "__main__":
    main()
