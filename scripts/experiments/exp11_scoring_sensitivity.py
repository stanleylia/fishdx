#!/usr/bin/env python3
"""EXP-11: Scoring Weight Sensitivity Analysis.

Tests how Eq.9-11 weight parameters affect Diagnostic Accuracy (DA).
Uses pre-computed EXP-NEW-01 per-image results to re-run scoring with
different weight configurations.

Parameters swept (One-At-a-Time):
  1. w_d^confirmed  ∈ {1, 2, 3, 4, 5}
  2. w_d^suspected  ∈ {1, 2, 3}
  3. w_h^explicit   ∈ {1, 2, 3, 4}
  4. T_h (healthy_threshold) ∈ {1, 2, 3, 4}
  5. m (inconclusive_margin) ∈ {0, 1, 2, 3}

Output: Sensitivity Index = max(DA) - min(DA) per parameter.
"""
from __future__ import annotations

import json
import sys
import copy
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.config import get_config, ScoringConfig
from core.algorithms.scoring import full_scoring_pipeline


@dataclass
class SweepResult:
    """Result of a single parameter sweep point."""
    param_name: str
    param_value: float
    da: float
    correct: int
    total: int
    healthy_correct: int
    healthy_total: int
    disease_correct: int
    disease_total: int
    inconclusive_count: int


def load_experiment_data(path: str) -> list[dict]:
    """Load per-image results from EXP-NEW-01."""
    with open(path) as f:
        data = json.load(f)
    return data["per_image_results"]


def get_scoring_text(image_result: dict) -> str:
    """Extract the actual scoring text used in the original experiment.

    Uses the pre-recorded scoring_text_used field from per-image results,
    which is the concatenated RAG content that was scored in the pipeline.
    """
    stage3 = image_result.get("stage3_scoring", {})
    text = stage3.get("scoring_text_used", "")
    if text:
        return text

    # Fallback: reconstruct from RAG results
    rag_results = image_result.get("stage2", {}).get("rag_results", [])
    if not rag_results:
        return ""
    parts = [r.get("content", "") for r in rag_results[:3] if r.get("similarity", 0) > 0.5]
    return " ".join(parts)


def make_config_variant(
    base_config: ScoringConfig,
    param_name: str,
    param_value: float,
) -> ScoringConfig:
    """Create a config variant with one parameter changed."""
    cfg = copy.deepcopy(base_config)

    if param_name == "w_d_confirmed":
        cfg.weights.disease_confirmed = int(param_value)
    elif param_name == "w_d_suspected":
        cfg.weights.disease_suspected = int(param_value)
    elif param_name == "w_h_explicit":
        cfg.weights.healthy_explicit = int(param_value)
    elif param_name == "T_h":
        cfg.healthy_threshold = int(param_value)
    elif param_name == "m":
        cfg.inconclusive_margin = int(param_value)
    else:
        raise ValueError(f"Unknown param: {param_name}")

    return cfg


def evaluate_with_config(
    images: list[dict],
    config: ScoringConfig,
) -> tuple[int, int, int, int, int, int, int]:
    """Evaluate DA with a given scoring config.

    Returns: (correct, total, healthy_correct, healthy_total,
              disease_correct, disease_total, inconclusive_count)
    """
    correct = 0
    total = 0
    healthy_correct = 0
    healthy_total = 0
    disease_correct = 0
    disease_total = 0
    inconclusive_count = 0

    for img in images:
        gt = img["ground_truth"]
        expected = "Healthy" if gt == "Healthy Fish" else "Disease"
        text = get_scoring_text(img)

        if not text:
            total += 1
            continue

        result = full_scoring_pipeline(text, config)

        total += 1

        if result.status == "Inconclusive":
            inconclusive_count += 1
            # Inconclusive is neither correct nor incorrect for DA
            # but we count it as incorrect for strict DA
            if expected == "Healthy":
                healthy_total += 1
            else:
                disease_total += 1
            continue

        if expected == "Healthy":
            healthy_total += 1
            if result.status == "Healthy":
                healthy_correct += 1
                correct += 1
        else:
            disease_total += 1
            if result.status == "Disease":
                disease_correct += 1
                correct += 1

    return (correct, total, healthy_correct, healthy_total,
            disease_correct, disease_total, inconclusive_count)


def run_sweep(
    images: list[dict],
    base_config: ScoringConfig,
    param_name: str,
    values: list[float],
) -> list[SweepResult]:
    """Run a parameter sweep for one parameter."""
    results = []
    for val in values:
        cfg = make_config_variant(base_config, param_name, val)
        (correct, total, hc, ht, dc, dt, inc) = evaluate_with_config(images, cfg)
        da = correct / total if total > 0 else 0.0
        results.append(SweepResult(
            param_name=param_name,
            param_value=val,
            da=da,
            correct=correct,
            total=total,
            healthy_correct=hc,
            healthy_total=ht,
            disease_correct=dc,
            disease_total=dt,
            inconclusive_count=inc,
        ))
    return results


def main():
    # Load config
    config = get_config()
    base_scoring = config.scoring

    # Load experiment data
    data_path = Path(__file__).resolve().parents[2] / \
        "lab_dateset/organized/experiment_results/large_scale/exp_new01_classification.json"
    images = load_experiment_data(str(data_path))
    print(f"Loaded {len(images)} images from EXP-NEW-01\n")

    # Define parameter sweeps
    sweeps = {
        "w_d_confirmed": [1, 2, 3, 4, 5],
        "w_d_suspected": [1, 2, 3],
        "w_h_explicit": [1, 2, 3, 4],
        "T_h": [1, 2, 3, 4],
        "m": [0, 1, 2, 3],
    }

    # Show baseline
    print("=" * 80)
    print("BASELINE CONFIG:")
    print(f"  w_d_confirmed = {base_scoring.weights.disease_confirmed}")
    print(f"  w_d_suspected = {base_scoring.weights.disease_suspected}")
    print(f"  w_h_explicit  = {base_scoring.weights.healthy_explicit}")
    print(f"  T_h           = {base_scoring.healthy_threshold}")
    print(f"  m             = {base_scoring.inconclusive_margin}")
    print("=" * 80)

    # Run baseline first
    (bc, bt, bhc, bht, bdc, bdt, binc) = evaluate_with_config(images, base_scoring)
    baseline_da = bc / bt if bt > 0 else 0.0
    print(f"\nBASELINE DA = {baseline_da:.4f} ({bc}/{bt})")
    print(f"  Healthy: {bhc}/{bht}, Disease: {bdc}/{bdt}, Inconclusive: {binc}")
    print()

    # Run all sweeps
    all_results = {}
    sensitivity_summary = []

    for param_name, values in sweeps.items():
        print(f"\n{'='*60}")
        print(f"Sweeping: {param_name} ∈ {values}")
        print(f"{'='*60}")

        results = run_sweep(images, base_scoring, param_name, values)
        all_results[param_name] = results

        das = [r.da for r in results]
        sensitivity = max(das) - min(das)

        for r in results:
            marker = " ← BASELINE" if (
                (param_name == "w_d_confirmed" and r.param_value == base_scoring.weights.disease_confirmed) or
                (param_name == "w_d_suspected" and r.param_value == base_scoring.weights.disease_suspected) or
                (param_name == "w_h_explicit" and r.param_value == base_scoring.weights.healthy_explicit) or
                (param_name == "T_h" and r.param_value == base_scoring.healthy_threshold) or
                (param_name == "m" and r.param_value == base_scoring.inconclusive_margin)
            ) else ""
            print(f"  {param_name}={r.param_value:.0f}: DA={r.da:.4f} "
                  f"({r.correct}/{r.total}) "
                  f"H={r.healthy_correct}/{r.healthy_total} "
                  f"D={r.disease_correct}/{r.disease_total} "
                  f"Inc={r.inconclusive_count}{marker}")

        print(f"\n  Sensitivity Index: {sensitivity:.4f} "
              f"(min DA={min(das):.4f}, max DA={max(das):.4f})")

        sensitivity_summary.append({
            "parameter": param_name,
            "sensitivity_index": round(sensitivity, 4),
            "min_da": round(min(das), 4),
            "max_da": round(max(das), 4),
            "baseline_value": {
                "w_d_confirmed": base_scoring.weights.disease_confirmed,
                "w_d_suspected": base_scoring.weights.disease_suspected,
                "w_h_explicit": base_scoring.weights.healthy_explicit,
                "T_h": base_scoring.healthy_threshold,
                "m": base_scoring.inconclusive_margin,
            }[param_name],
            "sweep_values": values,
            "da_values": [round(d, 4) for d in das],
        })

    # Final summary
    print("\n" + "=" * 80)
    print("SENSITIVITY SUMMARY")
    print("=" * 80)
    print(f"{'Parameter':<20} {'Baseline':>10} {'Sensitivity':>12} {'Min DA':>10} {'Max DA':>10}")
    print("-" * 62)
    for s in sorted(sensitivity_summary, key=lambda x: x["sensitivity_index"], reverse=True):
        print(f"{s['parameter']:<20} {s['baseline_value']:>10} "
              f"{s['sensitivity_index']:>12.4f} {s['min_da']:>10.4f} {s['max_da']:>10.4f}")

    # Save results to JSON
    output = {
        "experiment": "EXP-11",
        "name": "Scoring Weight Sensitivity Analysis",
        "description": "One-At-a-Time parameter sweep for Eq.9-11 scoring weights",
        "total_images": len(images),
        "baseline": {
            "da": round(baseline_da, 4),
            "correct": bc,
            "total": bt,
            "healthy_correct": bhc,
            "healthy_total": bht,
            "disease_correct": bdc,
            "disease_total": bdt,
            "inconclusive": binc,
            "config": {
                "w_d_confirmed": base_scoring.weights.disease_confirmed,
                "w_d_suspected": base_scoring.weights.disease_suspected,
                "w_h_explicit": base_scoring.weights.healthy_explicit,
                "T_h": base_scoring.healthy_threshold,
                "m": base_scoring.inconclusive_margin,
            },
        },
        "sensitivity_summary": sensitivity_summary,
        "detailed_results": {
            name: [
                {
                    "value": r.param_value,
                    "da": round(r.da, 4),
                    "correct": r.correct,
                    "total": r.total,
                    "healthy_correct": r.healthy_correct,
                    "healthy_total": r.healthy_total,
                    "disease_correct": r.disease_correct,
                    "disease_total": r.disease_total,
                    "inconclusive": r.inconclusive_count,
                }
                for r in results
            ]
            for name, results in all_results.items()
        },
    }

    out_path = data_path.parent / "exp11_scoring_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
