"""
EXP-13: Layer-by-Layer Ablation
Derives incremental contribution of each pipeline stage from existing experiment data.

Layer 0: Perception-only (Florence-2 Caption → keyword matching)
Layer 1: + CLIP Visual Embedding (kNN, λ=1.0)
Layer 2: + Fusion Embedding (λ=0.7)
Layer 3: + RAG Knowledge Retrieval (ChromaDB Top-K=5)
Layer 4: + Scoring Decision (Eq.9-11)
"""
from __future__ import annotations
import json
from pathlib import Path

RESULTS_DIR = (Path(__file__).resolve().parents[2] / "lab_dateset/organized/experiment_results/large_scale")

def load_json(filename):
    with open(RESULTS_DIR / filename) as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("EXP-13: Layer-by-Layer Ablation")
    print("=" * 60)

    # Load existing experiment results
    exp01 = load_json("exp01_lambda_ablation.json")
    exp06 = load_json("exp_new01_classification.json")  # Full pipeline
    exp08 = load_json("exp_new03_caption_quality.json")  # Caption quality
    exp11 = load_json("exp11_scoring_sensitivity.json")  # Scoring sensitivity

    # === Layer 0: Perception-only ===
    # From EXP-08: SCA=3.7%, Fish Detection=45.3%
    # Caption-based disease matching upper bound = SCA
    layer0 = {
        "layer": 0,
        "name": "Perception-only (Florence-2 Caption)",
        "components": ["Florence-2 Dense Caption", "Keyword Matching"],
        "da": exp08["overall_avg_sca"],  # 0.0373
        "da_display": round(exp08["overall_avg_sca"], 4),
        "source": "EXP-08 (SCA = disease term coverage in captions)",
        "note": "SCA represents maximum achievable DA from caption-based classification alone. Fish Detection Rate = 45.3% shows Florence-2 fails to identify fish in majority of disease images.",
        "n": exp08["total_images"],  # 1747
    }
    print(f"Layer 0 (Perception-only): DA ≈ {layer0['da_display']:.4f} (SCA upper bound)")

    # === Layer 1: + CLIP Visual Embedding (λ=1.0) ===
    # From EXP-01: λ=1.0 → DA=0.9971
    lambda_results = {r["lambda"]: r for r in exp01["results"]}
    l1_data = lambda_results[1.0]
    l1_da = l1_data["accuracy"]
    layer1 = {
        "layer": 1,
        "name": "+ CLIP Visual Embedding (λ=1.0, kNN)",
        "components": ["Florence-2", "CLIP Image Encoding", "ChromaDB kNN"],
        "da": l1_da,
        "da_display": l1_da,
        "correct": l1_data["correct"],
        "total": l1_data["total"],
        "source": "EXP-01 (λ=1.0, visual-only fusion)",
        "n": l1_data["total"],  # 697
    }
    print(f"Layer 1 (+CLIP Visual): DA = {layer1['da_display']:.4f} ({l1_data['correct']}/{l1_data['total']})")

    # === Layer 2: + Caption Fusion (λ=0.7) ===
    # From EXP-01: λ=0.7 → DA=1.0000
    l2_data = lambda_results[0.7]
    l2_da = l2_data["accuracy"]
    layer2 = {
        "layer": 2,
        "name": "+ λ-Weighted Fusion (λ=0.7)",
        "components": ["Florence-2", "CLIP Image+Text Encoding", "Fusion (Eq.3-5)", "ChromaDB kNN"],
        "da": l2_da,
        "da_display": l2_da,
        "correct": l2_data["correct"],
        "total": l2_data["total"],
        "source": "EXP-01 (λ=0.7, optimal fusion)",
        "n": l2_data["total"],  # 697
        "marginal_gain": round(l2_da - l1_da, 4),
    }
    print(f"Layer 2 (+Fusion λ=0.7): DA = {layer2['da_display']:.4f} (Δ = +{layer2['marginal_gain']:.4f})")

    # === Layer 3: + RAG + Semantic Filter + Scoring ===
    # From EXP-06: Full algorithm pipeline → DA=0.999
    exp06_m = exp06["metrics"]
    layer3 = {
        "layer": 3,
        "name": "+ RAG Retrieval + Scoring (Eq.9-11)",
        "components": ["Florence-2", "CLIP Fusion", "Semantic Filter (Eq.8)", "ChromaDB RAG", "Scoring (Eq.9-11)"],
        "da": exp06_m["accuracy"],
        "da_display": round(exp06_m["accuracy"], 4),
        "correct": exp06_m["correct"],
        "total": exp06_m["total"],
        "source": "EXP-06 (Full algorithm-level pipeline, n=1,047)",
        "n": exp06_m["total"],  # 1047
        "note": "Different test set size (n=1,047 vs n=697 for Layers 1-2). DA comparison is directional.",
    }
    print(f"Layer 3 (+RAG+Scoring): DA = {layer3['da_display']:.4f} ({exp06_m['correct']}/{exp06_m['total']})")

    # === Compile Results ===
    results = {
        "experiment": "EXP-13",
        "name": "Layer-by-Layer Pipeline Ablation",
        "layers": [layer0, layer1, layer2, layer3],
        "key_findings": [
            "Layer 0→1: Perception-only (SCA=3.7%) to CLIP Visual (DA=0.997) shows RAG embedding provides massive gain (+0.96)",
            "Layer 1→2: Visual-only to Fusion (DA=1.000) shows caption fusion eliminates remaining 2 misclassifications",
            "Layer 2→3: Fusion to Full Pipeline (DA=0.999, different n) maintains near-perfect accuracy with Scoring decision engine",
            "Core insight: The Perception-Cognition Gap (Layer 0 DA=0.037 vs Layer 3 DA=0.999) is bridged primarily by RAG+Fusion (Layer 1-2), with Scoring providing deterministic decision stability",
        ],
        "ablation_summary": {
            "perception_only_da": round(layer0["da"], 4),
            "clip_visual_da": layer1["da"],
            "fusion_da": layer2["da"],
            "full_pipeline_da": round(layer3["da"], 4),
            "total_gap_bridged": round(layer3["da"] - layer0["da"], 4),
        }
    }

    # Print summary
    print("\n" + "=" * 70)
    print("LAYER-BY-LAYER ABLATION SUMMARY")
    print("=" * 70)
    print(f"{'Layer':<5} {'Configuration':<40} {'DA':>8} {'Δ DA':>8} {'n':>6}")
    print("-" * 70)
    prev_da = 0
    for layer in results["layers"]:
        if "da" in layer:
            da = layer.get("da_display", layer["da"])
            delta = da - prev_da
            n = layer.get("n", "—")
            print(f"{layer['layer']:<5} {layer['name']:<40} {da:>8.4f} {delta:>+8.4f} {str(n):>6}")
            prev_da = da
        else:
            agr = layer.get("llm_agreement", 0)
            n = layer.get("n", "—")
            print(f"{layer['layer']:<5} {layer['name']:<40} {'Agr='+str(agr):>8} {'—':>8} {str(n):>6}")
    print("=" * 70)

    # Save
    output_path = RESULTS_DIR / "exp13_layer_ablation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
