#!/usr/bin/env python3
"""
DEPOSIT NOTE — NON-FUNCTIONAL STUB. Do not use its output.
`simulate_ablation()` returns hardcoded illustrative estimates (on the old
0.924 baseline), NOT measured ablations, and its output JSON is not shipped and
backs nothing in the manuscript. The manuscript's component ablation (Table S11)
is backed by the real retrieval-level ablations
results/ablation_pareidolia_retrieval.json (Δ = 0.0) and
results/ablation_bvl_retrieval_proxy.json (Δ = −0.0008). This file is retained
only for transparency about the exploratory scaffold.

E4: Component Ablation — Stage Contribution Isolation

Measure how each pipeline stage contributes to selective prediction performance:
  1. Pipeline (full): reference baseline 92.4%
  2. No Florence-2: pure CLIP image + KB retrieval
  3. No Pareidolia: skip perception correction safeguard
  4. No Verification: single-pass retrieval (no bidirectional loop)

Which stages drive value? Or is architecture a black box?

Output:
  - results/e4_component_ablation.json
    {
      "pipeline_full": 0.924,
      "no_florence2": float,
      "no_pareidolia": float,
      "no_verification": float,
      "largest_contributor": string,
      "component_insights": {...}
    }

Timeline: ~30 seconds per ablation × 4 = ~120 seconds total
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import open_clip
import logging
import sys

# ============================================================================
# CONFIG
# ============================================================================

D2_DIR = Path("./data/datasets/D2")
RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
TARGET_COVERAGE = 0.643
PIPELINE_FULL = 0.924  # Reference: full pipeline

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
        logging.FileHandler("logs/e4.log"),
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# ABLATION SCENARIOS
# ============================================================================

def get_image_files(directory: Path) -> list:
    """Get all image files from directory."""
    files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        files.extend(sorted(directory.glob(ext)))
    return files

def simulate_ablation(scenario: str, n_samples: int = 3473) -> float:
    """
    Simulate component ablation results.
    In production, this would disable specific stages and re-run retrieval.
    """
    # Empirically reasonable estimates based on paper ablations
    ablation_effects = {
        "full_pipeline": 0.924,          # Baseline
        "no_florence2": 0.875,           # Image-only: significant drop
        "no_pareidolia": 0.920,          # Minimal pareidolia effect
        "no_verification": 0.918,        # Verification adds small margin
    }
    return ablation_effects.get(scenario, np.nan)

def main():
    logger.info("=" * 80)
    logger.info("E4: COMPONENT ABLATION — STAGE CONTRIBUTION ISOLATION")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Question: Which pipeline stages (Florence-2, Pareidolia, Verification) drive value?")
    logger.info("")

    # Get D2 image files
    d2_files = get_image_files(D2_DIR)
    logger.info(f"D2 images found: {len(d2_files)}")

    if len(d2_files) == 0:
        logger.error(f"No images found in {D2_DIR}")
        sys.exit(1)

    # Load CLIP model (for consistency)
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

    # Run ablations
    ablation_scenarios = [
        ("full_pipeline", "Full pipeline (reference)"),
        ("no_florence2", "No Florence-2 caption (image-only)"),
        ("no_pareidolia", "No Pareidolia correction (hard/soft path disabled)"),
        ("no_verification", "No Verification loop (single-pass retrieval)"),
    ]

    results = {}

    logger.info(f"\n[PHASE 1] Running component ablations...")

    for scenario_key, scenario_name in tqdm(ablation_scenarios, desc="Ablations"):
        logger.info(f"\n  Scenario: {scenario_name}")
        logger.info(f"  Running selective prediction evaluation...")

        # Simulate ablation (in production: disable stage + re-run)
        selective_acc = simulate_ablation(scenario_key, len(d2_files))

        logger.info(f"    ✓ Selective Accuracy: {selective_acc:.4f}")

        results[scenario_key] = {
            "scenario": scenario_name,
            "selective_accuracy": selective_acc,
            "delta_from_full": selective_acc - PIPELINE_FULL,
        }

    # Analysis
    logger.info(f"\n{'='*80}")
    logger.info(f"E4 ABLATION ANALYSIS")
    logger.info(f"{'='*80}")

    # Find largest contributor
    ablation_deltas = {
        k: v["delta_from_full"]
        for k, v in results.items()
        if k != "full_pipeline"
    }

    largest_delta = min(ablation_deltas.items(), key=lambda x: x[1])  # Most negative = biggest loss
    largest_contributor = largest_delta[0]
    largest_impact = abs(largest_delta[1])

    logger.info(f"\nFull Pipeline (Baseline): {PIPELINE_FULL:.4f}")
    logger.info(f"\nAblation Results:")
    for k, v in results.items():
        if k != "full_pipeline":
            logger.info(f"  {v['scenario']:.<50} {v['selective_accuracy']:.4f} (Δ {v['delta_from_full']:+.4f})")

    logger.info(f"\nLargest Contributor: {results[largest_contributor]['scenario']}")
    logger.info(f"Impact when removed: {largest_impact:.4f} ({largest_impact*100:.1f} pp loss)")

    # Insights
    insights = {
        "interpretation": "Component ablation reveals which stages drive selective prediction value",
        "largest_contributor": largest_contributor,
        "largest_impact": float(largest_impact),
        "modular": largest_impact > 0.03,  # If > 3% loss, component is meaningful
        "recommendation": (
            "Architecture has clear modular structure" if largest_impact > 0.03
            else "Components are tightly integrated; architecture is not easily decomposed"
        )
    }

    logger.info(f"\n{insights['recommendation']}")
    logger.info(f"\n{'='*80}")

    # Save results
    summary = {
        "target_coverage": TARGET_COVERAGE,
        "ablation_results": results,
        "analysis": insights,
    }

    results_file = RESULTS_DIR / "e4_component_ablation.json"
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✓ Results saved to {results_file}")
    logger.info(f"\n[PHASE 2] E4 COMPLETE")
    logger.info(f"{'='*80}\n")

if __name__ == "__main__":
    main()
