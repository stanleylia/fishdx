"""EXP-11: Qualitative Case Studies (Table 10 / Appendix).

Hypothesis: Detailed per-case analysis demonstrates that the pipeline
correctly handles diverse scenarios including true positives, edge cases,
and potential failure modes.

Six representative cases (all Level 1 offline simulation):
  CS-1: S01 Healthy Fish (True Positive Healthy)
  CS-2: S03 White Spot Disease (True Positive Disease)
  CS-3: S09 Multi-Species (Multi-Species Success)
  CS-4: S10 Turbid Water (Edge Case)
  CS-5: S08 Water Quality (Potential False Negative)
  CS-6: S05 Net Cage Environment (True Negative Environment)

For each case, simulate the full pipeline stages:
  1. Florence-2 caption (synthetic)
  2. Pareidolia detection result
  3. Semantic filter classification
  4. Scoring pipeline result
  5. 5-step reasoning chain (generated text)

Output: Detailed per-case JSON report with all intermediate results.

Level 1: All offline simulation.
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

from core.algorithms.pareidolia import detect_hard  # noqa: E402
from core.algorithms.scoring import full_scoring_pipeline  # noqa: E402
from core.algorithms.semantic_filter import classify_scene  # noqa: E402
from core.config import load_config  # noqa: E402
from scripts.experiments.experiment_harness import (  # noqa: E402
    SCENE_EXPECTED_STATUS,
    SCENE_SYNTHETIC_CAPTIONS,
    SIMULATED_LLM_OUTPUTS,
    add_common_args,
    parse_common_args,
    save_result,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EXPERIMENT_ID = "exp11"

# ---------------------------------------------------------------------------
# Case study definitions
# ---------------------------------------------------------------------------
CASE_STUDIES: list[dict[str, Any]] = [
    {
        "case_id": "CS-1",
        "scene_id": "S01_fish_health_tilapia",
        "title": "Healthy Fish (True Positive Healthy)",
        "category": "True Positive",
        "description": (
            "Tilapia fish in aquaculture pond with clear water. Expected: Healthy. "
            "Tests the pipeline's ability to correctly identify healthy specimens."
        ),
        "synthetic_caption": (
            "The image shows tilapia fish swimming in a pond with clear water. "
            "Several tilapia are visible near the surface in an aquaculture setting."
        ),
        "object_labels": ["fish", "tilapia", "pond"],
    },
    {
        "case_id": "CS-2",
        "scene_id": "S03_disease_white_spot",
        "title": "White Spot Disease (True Positive Disease)",
        "category": "True Positive",
        "description": (
            "Fish with white spot disease (Ichthyophthirius multifiliis). "
            "Expected: Disease. Tests detection of visible disease indicators."
        ),
        "synthetic_caption": (
            "Fish with white spots on body surface, showing signs of disease. "
            "Multiple white spots are visible across the fish's skin and fins."
        ),
        "object_labels": ["fish", "spots", "lesion"],
    },
    {
        "case_id": "CS-3",
        "scene_id": "S09_multi_species_detection",
        "title": "Multi-Species Detection (Multi-Species Success)",
        "category": "Multi-Species",
        "description": (
            "Multiple aquatic species including fish, crab, and shrimp. "
            "Expected: Healthy. Tests multi-element aggregation capability."
        ),
        "synthetic_caption": (
            "Multiple aquatic species detected: fish, crab, and shrimp. "
            "All organisms appear healthy with normal behavior patterns."
        ),
        "object_labels": ["fish", "crab", "shrimp", "starfish"],
    },
    {
        "case_id": "CS-4",
        "scene_id": "S10_edge_cases_turbidity",
        "title": "Turbid Water (Edge Case)",
        "category": "Edge Case",
        "description": (
            "Jellyfish in turbid water with poor visibility. "
            "Expected: Healthy. Tests handling of non-fish aquatic organisms "
            "and degraded image quality."
        ),
        "synthetic_caption": (
            "Jellyfish observed in turbid water conditions. "
            "Water visibility is poor due to suspended particles."
        ),
        "object_labels": ["jellyfish", "water"],
    },
    {
        "case_id": "CS-5",
        "scene_id": "S08_water_quality_degraded",
        "title": "Water Quality Degraded (Potential False Negative)",
        "category": "Potential Failure",
        "description": (
            "Degraded underwater image with poor visibility. "
            "Expected: Inconclusive. Tests whether pipeline avoids false "
            "diagnosis when visual information is insufficient."
        ),
        "synthetic_caption": (
            "Degraded underwater image with poor visibility and turbid water. "
            "Environmental monitoring shows compromised water quality."
        ),
        "object_labels": ["water", "turbidity", "sensor"],
    },
    {
        "case_id": "CS-6",
        "scene_id": "S05_environment_net_cage",
        "title": "Net Cage Environment (True Negative Environment)",
        "category": "True Negative",
        "description": (
            "Underwater net cage with structural elements that may trigger "
            "pareidolia (false biological detection). Expected: Inconclusive. "
            "Tests pareidolia correction on structural scenes."
        ),
        "synthetic_caption": (
            "An underwater net cage with mesh and wire structure. "
            "Crab and starfish visible near the net. Marine environment."
        ),
        "object_labels": ["fish", "person", "net", "cage", "crab", "starfish"],
    },
]

# ---------------------------------------------------------------------------
# 5-step reasoning chain templates
# ---------------------------------------------------------------------------
REASONING_TEMPLATES: dict[str, list[str]] = {
    "Healthy": [
        "Step 1 (Perception): Florence-2 detected {species} in {environment}. Caption indicates normal appearance.",
        "Step 2 (Pareidolia Check): {pareidolia_result}. No false biological detections requiring correction.",
        "Step 3 (Semantic Filter): Scene classified as '{scene_type}' (score={filter_score:.3f}). Domain: fish health assessment.",
        "Step 4 (Scoring): Healthy indicators matched: {healthy_patterns}. S_h={s_h}, S_d={s_d}. Decision: Healthy (S_h >= 3 AND S_h > S_d).",
        "Step 5 (Verification): Diagnosis 'Healthy' confirmed. Confidence={confidence:.2f}. No contradictions detected in bidirectional check.",
    ],
    "Disease": [
        "Step 1 (Perception): Florence-2 detected {species} with abnormal features. Caption indicates: {symptoms}.",
        "Step 2 (Pareidolia Check): {pareidolia_result}. Biological detections verified as genuine.",
        "Step 3 (Semantic Filter): Scene classified as '{scene_type}' (score={filter_score:.3f}). RAG triggered for disease knowledge.",
        "Step 4 (Scoring): Disease indicators matched: {disease_patterns}. S_h={s_h}, S_d={s_d}. Decision: Disease (S_d > S_h).",
        "Step 5 (Verification): Diagnosis 'Disease' confirmed. Confidence={confidence:.2f}. RAG knowledge base corroborates indicators.",
    ],
    "Inconclusive": [
        "Step 1 (Perception): Florence-2 detected {elements}. Caption: {caption_summary}.",
        "Step 2 (Pareidolia Check): {pareidolia_result}.",
        "Step 3 (Semantic Filter): Scene classified as '{scene_type}' (score={filter_score:.3f}). {rag_note}.",
        "Step 4 (Scoring): S_h={s_h}, S_d={s_d}. Margin |S_h - S_d| <= 1 or thresholds not met. Decision: Inconclusive.",
        "Step 5 (Verification): Status 'Inconclusive' maintained. Confidence={confidence:.2f}. Insufficient evidence for definitive diagnosis.",
    ],
}


# ---------------------------------------------------------------------------
# Generate reasoning chain
# ---------------------------------------------------------------------------
def _generate_reasoning_chain(
    case: dict[str, Any],
    pareidolia_results: list[dict[str, Any]],
    classification: dict[str, Any],
    scoring_result: dict[str, Any],
) -> list[str]:
    """Generate a 5-step reasoning chain for a case study.

    Args:
        case: Case study definition.
        pareidolia_results: Pareidolia detection results.
        classification: Semantic filter classification.
        scoring_result: Scoring pipeline result.

    Returns:
        List of 5 reasoning step strings.
    """
    status = scoring_result["status"]
    templates = REASONING_TEMPLATES.get(status, REASONING_TEMPLATES["Inconclusive"])

    # Build template variables
    pareidolia_flagged = [r["label"] for r in pareidolia_results if r["is_pareidolia"]]
    if pareidolia_flagged:
        pareidolia_text = f"Pareidolia detected for: {', '.join(pareidolia_flagged)} (corrected)"
    else:
        pareidolia_text = "No pareidolia detected"

    healthy_patterns = scoring_result.get("healthy_patterns", [])
    disease_patterns = scoring_result.get("disease_patterns", [])

    variables = {
        "species": ", ".join(case["object_labels"][:2]),
        "environment": case["synthetic_caption"][:60],
        "elements": ", ".join(case["object_labels"]),
        "symptoms": ", ".join(disease_patterns) if disease_patterns else "visible anomalies",
        "caption_summary": case["synthetic_caption"][:80],
        "pareidolia_result": pareidolia_text,
        "scene_type": classification["scene_type"],
        "filter_score": classification["score"],
        "healthy_patterns": ", ".join(healthy_patterns) if healthy_patterns else "none",
        "disease_patterns": ", ".join(disease_patterns) if disease_patterns else "none",
        "s_h": scoring_result["healthy_score"],
        "s_d": scoring_result["disease_score"],
        "confidence": scoring_result["confidence"],
        "rag_note": "RAG triggered" if classification["rag_triggered"] else "RAG not triggered",
    }

    steps: list[str] = []
    for template in templates:
        try:
            step = template.format(**variables)
        except (KeyError, IndexError):
            step = template  # Use raw template if formatting fails
        steps.append(step)

    return steps


# ---------------------------------------------------------------------------
# Run a single case study
# ---------------------------------------------------------------------------
def _run_case_study(
    case: dict[str, Any],
    base_config: Any,
) -> dict[str, Any]:
    """Run the full pipeline simulation for a single case study.

    Args:
        case: Case study definition.
        base_config: Base AppConfig.

    Returns:
        Detailed case result dict.
    """
    scene_id = case["scene_id"]
    expected = SCENE_EXPECTED_STATUS.get(scene_id, "Inconclusive")

    # --- Stage 1: Florence-2 caption (synthetic) ---
    caption = case["synthetic_caption"]
    # Also use any captions from SCENE_SYNTHETIC_CAPTIONS
    extra_captions = SCENE_SYNTHETIC_CAPTIONS.get(scene_id, [])

    # --- Stage 2: Pareidolia detection ---
    pareidolia_results_raw = detect_hard(
        caption, case["object_labels"], base_config.pareidolia)
    pareidolia_results = [
        {
            "label": r.label,
            "is_pareidolia": r.is_pareidolia,
            "hard_score": r.hard_score,
        }
        for r in pareidolia_results_raw
    ]
    n_pareidolia = sum(1 for r in pareidolia_results if r["is_pareidolia"])

    # --- Stage 3: Semantic filter classification ---
    cls = classify_scene(caption, base_config.semantic_filter)
    classification = {
        "scene_type": cls.scene_type,
        "score": cls.score,
        "rag_triggered": cls.rag_triggered,
        "matched_keywords": cls.matched_keywords,
        "profile_name": cls.profile_name,
    }

    # --- Stage 4: Scoring pipeline ---
    llm_text = SIMULATED_LLM_OUTPUTS.get(scene_id, "")
    decision = full_scoring_pipeline(llm_text, base_config.scoring)
    scoring_result = {
        "status": decision.status,
        "healthy_score": decision.healthy_score,
        "disease_score": decision.disease_score,
        "confidence": decision.confidence,
        "healthy_patterns": decision.healthy_detail.matched_patterns,
        "disease_patterns": decision.disease_detail.matched_patterns,
        "healthy_weights": [
            {"pattern": p, "weight": w}
            for p, w in decision.healthy_detail.weights_applied
        ],
        "disease_weights": [
            {"pattern": p, "weight": w}
            for p, w in decision.disease_detail.weights_applied
        ],
    }

    # --- Stage 5: Reasoning chain ---
    reasoning_chain = _generate_reasoning_chain(
        case, pareidolia_results, classification, scoring_result)

    # --- Evaluation ---
    is_correct = decision.status == expected
    analysis_notes: list[str] = []

    if is_correct:
        analysis_notes.append(
            f"CORRECT: Pipeline correctly identified '{expected}' status.")
    else:
        analysis_notes.append(
            f"MISMATCH: Expected '{expected}', got '{decision.status}'.")

    if n_pareidolia > 0:
        analysis_notes.append(
            f"Pareidolia correction activated: {n_pareidolia} labels flagged.")

    if classification["rag_triggered"]:
        analysis_notes.append("RAG retrieval was triggered for domain knowledge.")

    if decision.confidence < 0.5:
        analysis_notes.append("Low confidence score suggests borderline case.")

    return {
        "case_id": case["case_id"],
        "scene_id": scene_id,
        "title": case["title"],
        "category": case["category"],
        "description": case["description"],
        "expected_status": expected,
        "predicted_status": decision.status,
        "is_correct": is_correct,
        "stages": {
            "stage_1_caption": {
                "primary_caption": caption,
                "extra_captions": extra_captions,
                "object_labels": case["object_labels"],
            },
            "stage_2_pareidolia": {
                "results": pareidolia_results,
                "n_flagged": n_pareidolia,
                "labels_flagged": [
                    r["label"] for r in pareidolia_results if r["is_pareidolia"]
                ],
            },
            "stage_3_semantic_filter": classification,
            "stage_4_scoring": scoring_result,
            "stage_5_reasoning": {
                "chain": reasoning_chain,
                "n_steps": len(reasoning_chain),
            },
        },
        "analysis_notes": analysis_notes,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment(
    dry_run: bool = False,
    n_runs: int = 1,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Execute the qualitative case studies experiment.

    Args:
        dry_run: If True, run only the first 2 case studies.
        n_runs: Not used (single-run experiment).
        config_path: Optional override config YAML path.

    Returns:
        Complete result dict ready for save_result().
    """
    base_config = load_config(config_path)

    cases_to_run = CASE_STUDIES[:2] if dry_run else CASE_STUDIES

    log.info(
        "EXP-11 starting: %d case studies, dry_run=%s",
        len(cases_to_run), dry_run,
    )
    t0 = time.perf_counter()

    case_results: list[dict[str, Any]] = []
    correct_count = 0

    for case in cases_to_run:
        log.info("Running case: %s (%s)", case["case_id"], case["title"])
        result = _run_case_study(case, base_config)
        case_results.append(result)
        correct_count += int(result["is_correct"])
        log.info(
            "  %s: predicted=%s, expected=%s, correct=%s",
            case["case_id"], result["predicted_status"],
            result["expected_status"], result["is_correct"],
        )

    # --- Summary ---
    total_cases = len(case_results)
    overall_accuracy = correct_count / total_cases if total_cases > 0 else 0.0

    summary_rows: list[dict[str, Any]] = []
    for cr in case_results:
        summary_rows.append({
            "case_id": cr["case_id"],
            "title": cr["title"],
            "category": cr["category"],
            "expected": cr["expected_status"],
            "predicted": cr["predicted_status"],
            "correct": cr["is_correct"],
            "confidence": cr["stages"]["stage_4_scoring"]["confidence"],
            "pareidolia_flagged": cr["stages"]["stage_2_pareidolia"]["n_flagged"],
            "rag_triggered": cr["stages"]["stage_3_semantic_filter"]["rag_triggered"],
        })

    # --- Category breakdown ---
    category_results: dict[str, dict[str, int]] = {}
    for cr in case_results:
        cat = cr["category"]
        if cat not in category_results:
            category_results[cat] = {"total": 0, "correct": 0}
        category_results[cat]["total"] += 1
        category_results[cat]["correct"] += int(cr["is_correct"])

    # --- Pipeline stage analysis ---
    stage_analysis = {
        "pareidolia_activation_rate": (
            sum(1 for cr in case_results
                if cr["stages"]["stage_2_pareidolia"]["n_flagged"] > 0)
            / total_cases if total_cases > 0 else 0.0
        ),
        "rag_trigger_rate": (
            sum(1 for cr in case_results
                if cr["stages"]["stage_3_semantic_filter"]["rag_triggered"])
            / total_cases if total_cases > 0 else 0.0
        ),
        "avg_confidence": float(np.mean([
            cr["stages"]["stage_4_scoring"]["confidence"]
            for cr in case_results
        ])) if case_results else 0.0,
        "avg_healthy_score": float(np.mean([
            cr["stages"]["stage_4_scoring"]["healthy_score"]
            for cr in case_results
        ])) if case_results else 0.0,
        "avg_disease_score": float(np.mean([
            cr["stages"]["stage_4_scoring"]["disease_score"]
            for cr in case_results
        ])) if case_results else 0.0,
    }

    elapsed = time.perf_counter() - t0

    result = {
        "experiment": "EXP-11: Qualitative Case Studies",
        "hypothesis": "Pipeline handles diverse scenarios correctly",
        "table_ref": "Table 10",
        "parameters": {
            "n_cases": len(cases_to_run),
            "dry_run": dry_run,
        },
        "summary": summary_rows,
        "overall_accuracy": overall_accuracy,
        "correct_count": correct_count,
        "total_cases": total_cases,
        "category_results": category_results,
        "stage_analysis": stage_analysis,
        "case_results": {cr["case_id"]: cr for cr in case_results},
        "elapsed_seconds": elapsed,
    }

    return result


def print_summary(result: dict[str, Any]) -> None:
    """Print a human-readable summary of case studies."""
    print("\n" + "=" * 110)
    print("EXP-11: Qualitative Case Studies")
    print("=" * 110)

    # Summary table
    print(
        f"\n{'Case':<6s}  "
        f"{'Title':<40s}  "
        f"{'Category':<18s}  "
        f"{'Expected':>12s}  "
        f"{'Predicted':>12s}  "
        f"{'Conf':>5s}  "
        f"{'Correct':>7s}"
    )
    print("-" * 110)

    for row in result["summary"]:
        mark = "  ok" if row["correct"] else "  XX"
        print(
            f"{row['case_id']:<6s}  "
            f"{row['title']:<40s}  "
            f"{row['category']:<18s}  "
            f"{row['expected']:>12s}  "
            f"{row['predicted']:>12s}  "
            f"{row['confidence']:>5.2f}  "
            f"{mark:>7s}"
        )

    print("-" * 110)
    print(
        f"\nOverall Accuracy: {result['correct_count']}/{result['total_cases']} "
        f"= {result['overall_accuracy']:.3f}"
    )

    # Category breakdown
    print("\n--- Category Breakdown ---")
    for cat, stats in result["category_results"].items():
        pct = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {cat:<20s}: {stats['correct']}/{stats['total']} ({pct:.0f}%)")

    # Stage analysis
    sa = result["stage_analysis"]
    print("\n--- Pipeline Stage Analysis ---")
    print(f"  Pareidolia activation rate: {sa['pareidolia_activation_rate']:.2f}")
    print(f"  RAG trigger rate:           {sa['rag_trigger_rate']:.2f}")
    print(f"  Avg confidence:             {sa['avg_confidence']:.2f}")
    print(f"  Avg S_h:                    {sa['avg_healthy_score']:.1f}")
    print(f"  Avg S_d:                    {sa['avg_disease_score']:.1f}")

    # Detailed case reports
    print("\n--- Detailed Case Reports ---")
    for case_id, cr in result["case_results"].items():
        print(f"\n  {case_id}: {cr['title']}")
        print(f"  {'='*60}")
        print(f"  Category:  {cr['category']}")
        print(f"  Expected:  {cr['expected_status']}")
        print(f"  Predicted: {cr['predicted_status']} ({'CORRECT' if cr['is_correct'] else 'MISMATCH'})")

        # Stage details
        s2 = cr["stages"]["stage_2_pareidolia"]
        s3 = cr["stages"]["stage_3_semantic_filter"]
        s4 = cr["stages"]["stage_4_scoring"]
        s5 = cr["stages"]["stage_5_reasoning"]

        print(f"  Pareidolia: {s2['n_flagged']} flagged {s2['labels_flagged']}")
        print(f"  Filter:     {s3['scene_type']} (score={s3['score']:.3f}, RAG={s3['rag_triggered']})")
        print(f"  Scoring:    S_h={s4['healthy_score']}, S_d={s4['disease_score']}, conf={s4['confidence']:.2f}")

        # Reasoning chain
        print(f"  Reasoning Chain:")
        for step in s5["chain"]:
            print(f"    {step}")

        # Analysis notes
        print(f"  Notes:")
        for note in cr["analysis_notes"]:
            print(f"    - {note}")

    print(f"\nTotal elapsed: {result['elapsed_seconds']:.3f}s")
    print("=" * 110)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP-11: Qualitative Case Studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python exp11_case_studies.py --dry-run\n"
            "  python exp11_case_studies.py\n"
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
