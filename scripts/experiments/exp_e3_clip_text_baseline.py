#!/usr/bin/env python3
"""
E3: CLIP-Text Baseline — Isolate Modality Contribution

DEPOSIT NOTE — EXPLORATORY / SIMULATED SCAFFOLD. This script contains simulated or placeholder logic and does not perform a live pipeline run; its output is NOT shipped in results/ and backs NO number reported in the manuscript. Retained only for transparency about exploratory scaffolding.

Evaluate D2 selective prediction using ONLY Florence-2 caption text embeddings.
Compare to pipeline's dual-path (image+text) to isolate modality value.

Does multi-modal fusion add value, or is text-only sufficient?

Output:
  - results/e3_clip_text_baseline.json
    {
      "clip_text_only_selective_accuracy": float,
      "pipeline_dual_path_accuracy": 0.924,
      "modality_delta": float,
      "verdict": "fusion_valuable" | "fusion_marginal" | "text_sufficient"
    }

Timeline: ~20 seconds (CLIP embedding reused if available)
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
TARGET_COVERAGE = 0.643
PIPELINE_SELECTIVE_ACC = 0.924  # Expected pipeline result

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
        logging.FileHandler("logs/e3.log"),
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

def compute_clip_text_embedding(caption: str, model, tokenizer) -> np.ndarray:
    """Compute CLIP text embedding from Florence-2 caption."""
    try:
        with torch.no_grad():
            text_input = tokenizer([caption]).to(DEVICE)
            text_features = model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features[0].cpu().numpy()
    except Exception as e:
        logger.warning(f"Error embedding caption '{caption}': {e}")
        return None

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between L2-normalized vectors."""
    return float(np.dot(vec1, vec2))

def main():
    logger.info("=" * 80)
    logger.info("E3: CLIP-TEXT BASELINE — MODALITY ISOLATION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Question: Does multi-modal fusion (image+text) beat text-only?")
    logger.info("")

    # Get D2 image files
    d2_files = get_image_files(D2_DIR)
    logger.info(f"D2 images found: {len(d2_files)}")

    if len(d2_files) == 0:
        logger.error(f"No images found in {D2_DIR}")
        sys.exit(1)

    # Load CLIP model (text encoder)
    logger.info("\nLoading CLIP model (ViT-B/32)...")
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='laion2b_s34b_b79k',
            device=DEVICE
        )
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        model.eval()
        logger.info(f"✓ CLIP model loaded on device: {DEVICE}")
    except Exception as e:
        logger.error(f"Failed to load CLIP model: {e}")
        sys.exit(1)

    # Simulate Florence-2 captions + extract text embeddings
    logger.info(f"\n[PHASE 1] Extracting text embeddings from simulated captions...")
    logger.info("(Using disease names as caption proxies for demonstration)")

    text_embeddings = []
    labels = []
    valid_files = []

    for img_file in tqdm(d2_files, desc="Text embedding"):
        # Simulate Florence-2 caption from filename
        label_str = img_file.stem.split('_')[0] if '_' in img_file.stem else img_file.stem[:15]
        caption = f"a fish with {label_str}"  # Simplified caption proxy

        emb = compute_clip_text_embedding(caption, model, tokenizer)
        if emb is not None:
            text_embeddings.append(emb)
            labels.append(label_str)
            valid_files.append(img_file)

    embeddings = np.array(text_embeddings)
    labels = np.array(labels)
    logger.info(f"✓ Extracted {len(embeddings)} text embeddings")

    # Evaluate text-only selective prediction
    logger.info(f"\n[PHASE 2] Evaluating text-only selective prediction...")
    logger.info(f"Sweeping similarity thresholds to match {TARGET_COVERAGE*100:.1f}% coverage...")

    threshold_range = np.arange(0.5, 1.01, 0.01)  # Start higher for text-only
    results = []

    for tau in tqdm(threshold_range, desc="Threshold sweep"):
        num_decided = 0
        num_correct = 0
        n_total = len(embeddings)

        for i in range(n_total):
            query_emb = embeddings[i]
            sims = np.dot(embeddings, query_emb)
            sims[i] = -np.inf

            nn_idx = np.argmax(sims)
            nn_sim = sims[nn_idx]

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

    # Find matched threshold
    matched = min(results, key=lambda x: abs(x['coverage'] - TARGET_COVERAGE))
    text_only_acc = matched['selective_accuracy']

    # Comparison
    delta = text_only_acc - PIPELINE_SELECTIVE_ACC

    if delta > 0.02:
        verdict = "text_sufficient"
        implication = "Text-only is sufficient; dual-path fusion doesn't add value"
    elif delta > -0.02:
        verdict = "fusion_marginal"
        implication = "Fusion effect is marginal (< 2%); modest benefit"
    else:
        verdict = "fusion_valuable"
        implication = "Fusion adds value; image component meaningful"

    # Report results
    logger.info(f"\n{'='*80}")
    logger.info(f"E3 RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Target coverage: {TARGET_COVERAGE*100:.1f}%")
    logger.info(f"Matched threshold τ: {matched['threshold']:.4f}")
    logger.info(f"Actual coverage: {matched['coverage']*100:.1f}%")
    logger.info(f"Text-only Selective Accuracy: {text_only_acc:.4f}")
    logger.info(f"Pipeline Dual-Path Accuracy (expected): {PIPELINE_SELECTIVE_ACC:.4f}")
    logger.info(f"Delta (text - dual-path): {delta:+.4f} ({delta*100:+.1f} pp)")
    logger.info(f"\nVerdict: {verdict}")
    logger.info(f"Implication: {implication}")
    logger.info(f"\n{'='*80}")

    # Save results
    summary = {
        "target_coverage": TARGET_COVERAGE,
        "matched_threshold": matched['threshold'],
        "matched_coverage": matched['coverage'],
        "clip_text_only_selective_accuracy": float(text_only_acc),
        "pipeline_dual_path_accuracy": PIPELINE_SELECTIVE_ACC,
        "modality_delta": float(delta),
        "verdict": verdict,
        "implication": implication,
        "full_curve": results,
    }

    results_file = RESULTS_DIR / "e3_clip_text_baseline.json"
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✓ Results saved to {results_file}")
    logger.info(f"\n[PHASE 3] E3 COMPLETE")
    logger.info(f"{'='*80}\n")

if __name__ == "__main__":
    main()
