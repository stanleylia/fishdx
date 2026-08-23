#!/usr/bin/env python3
"""
E1: De-duplicate D1 Train ↔ D1 Test using pHash + CLIP cosine distance.

This script audits whether D1 Train and D1 Test contain near-duplicates,
which would invalidate or inflate performance metrics.

Output:
  - results/e1_dedup_report.json
    {
      "total_test_images": 697,
      "near_dupes_detected": int,
      "near_dup_fraction": float,
      "near_dup_details": [...],
      "assessment": "LOW|MODERATE|HIGH",
      "cleaned_d1_test_images": [list of clean file paths]
    }

Timeline: ~3–4 days (mostly GPU compute for CLIP embeddings)
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
import open_clip
from imagehash import phash
import logging
import sys

# ============================================================================
# CONFIG
# ============================================================================

D1_TRAIN_DIR = Path("./data/datasets/D1/train")
D1_TEST_DIR = Path("./data/datasets/D1/test")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

PHASH_HAMMING_THRESHOLD = 5  # Pixels that differ in hash
CLIP_COSINE_THRESHOLD = 0.98  # Visual embedding similarity
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

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
        logging.FileHandler("logs/e1.log"),
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

def compute_phash(image_path: Path) -> str:
    """Compute perceptual hash of image."""
    try:
        img = Image.open(image_path).convert('RGB')
        return str(phash(img))
    except Exception as e:
        logger.warning(f"Error hashing {image_path}: {e}")
        return None

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

def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hash strings."""
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors (already L2-normalized)."""
    return float(np.dot(vec1, vec2))

def main():
    logger.info("=" * 80)
    logger.info("E1: D1 TRAIN-TEST DE-DUPLICATION AUDIT")
    logger.info("=" * 80)

    # Get file lists
    train_files = get_image_files(D1_TRAIN_DIR)
    test_files = get_image_files(D1_TEST_DIR)

    logger.info(f"\nD1 Train: {len(train_files)} images")
    logger.info(f"D1 Test: {len(test_files)} images")

    if len(train_files) == 0 or len(test_files) == 0:
        logger.error("No images found in D1 Train or D1 Test directories!")
        logger.error(f"Train dir: {D1_TRAIN_DIR}")
        logger.error(f"Test dir: {D1_TEST_DIR}")
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

    # Compute pHash for all train images
    logger.info("\n[PHASE 2] Computing pHash for D1 Train images...")
    train_phashes = {}
    for f in tqdm(train_files, desc="Train pHash"):
        h = compute_phash(f)
        if h:
            train_phashes[f.name] = h

    logger.info(f"✓ Computed pHash for {len(train_phashes)} train images")

    # Compute CLIP embeddings for all train images
    logger.info("\n[PHASE 3] Computing CLIP embeddings for D1 Train images...")
    logger.info("(This is the longest phase — GPU compute-bound)")
    train_embeddings = {}
    for f in tqdm(train_files, desc="Train CLIP"):
        emb = compute_clip_embedding(f, model, preprocess)
        if emb is not None:
            train_embeddings[f.name] = emb

    logger.info(f"✓ Computed CLIP embeddings for {len(train_embeddings)} train images")

    # Compute pHash + embeddings for test images & compare
    logger.info("\n[PHASE 4] Scanning D1 Test images for near-duplicates...")
    near_dupes = []
    clean_test_files = []

    for test_file in tqdm(test_files, desc="Test scan"):
        test_phash = compute_phash(test_file)
        test_emb = compute_clip_embedding(test_file, model, preprocess)

        if test_phash is None or test_emb is None:
            continue

        is_duplicate = False

        # Check against all train images
        for train_name, train_phash in train_phashes.items():
            hamming = hamming_distance(test_phash, train_phash)
            cosine_sim = cosine_similarity(test_emb, train_embeddings.get(train_name, np.zeros(512)))

            # Flag as near-dup if either condition met
            if hamming <= PHASH_HAMMING_THRESHOLD or cosine_sim >= CLIP_COSINE_THRESHOLD:
                near_dupes.append({
                    'test_file': test_file.name,
                    'train_file': train_name,
                    'hamming_distance': int(hamming),
                    'cosine_similarity': float(cosine_sim)
                })
                is_duplicate = True
                break

        if not is_duplicate:
            clean_test_files.append(str(test_file))

    # Report results
    dup_fraction = len(near_dupes) / len(test_files)

    if dup_fraction < 0.05:
        assessment = "LOW"
        assessment_msg = "✓ Low duplication risk (<5%). D1 Test DA is likely clean."
    elif dup_fraction < 0.20:
        assessment = "MODERATE"
        assessment_msg = "~ Moderate duplication (5–20%). D1 results valid with caveat."
    else:
        assessment = "HIGH"
        assessment_msg = "⚠️  High duplication risk (>20%). D1 Test DA may be inflated."

    logger.info(f"\n{'='*80}")
    logger.info(f"E1 RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Near-duplicates found: {len(near_dupes)} / {len(test_files)} ({dup_fraction*100:.1f}%)")
    logger.info(f"Assessment: {assessment} — {assessment_msg}")
    logger.info(f"Cleaned D1 Test: {len(clean_test_files)} images (excluded {len(near_dupes)})")

    # Save results
    results = {
        "total_test_images": len(test_files),
        "near_dupes_detected": len(near_dupes),
        "near_dup_fraction": float(dup_fraction),
        "assessment": assessment,
        "assessment_message": assessment_msg,
        "near_dup_details": near_dupes[:100],  # First 100 for inspection
        "cleaned_test_file_paths": clean_test_files,
    }

    results_file = RESULTS_DIR / "e1_dedup_report.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Results saved to {results_file}")
    logger.info(f"\n[PHASE 5] E1 COMPLETE")
    logger.info(f"{'='*80}\n")

if __name__ == "__main__":
    main()
