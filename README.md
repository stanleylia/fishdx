# fishdx — Training-Free Fish-Disease Classification via CLIP Fusion Retrieval

> Reference implementation for the *Scientific Reports* manuscript
> **"Visual embedding retrieval improves fish-disease classification when generated
> captions lack diagnostic vocabulary"** (Liao, Shih, Chang, 2026).

A three-stage **deterministic** pipeline that classifies freshwater fish
disease images by combining (i) Florence-2 visual perception with two-tier
pareidolia correction, (ii) λ-weighted CLIP fusion retrieval against a
1,639-image fused-embedding gallery, and (iii) candidate-conditioned evidence
scoring with a retrieval-margin abstention rule. No language-model free-form
generation occurs at inference time. The measured median latency is 559 ms per
image on an NVIDIA RTX 3070 (8 GB VRAM).

---

## Headline numbers

| Setting | Metric | Hardware |
|---|---|---|
| D2-final forced-choice (n = 2,402) | Overall DA = 0.812 | RTX 3070 |
| D2-final selective prediction | Decisive DA = 0.928 at 54.4% coverage (θ_margin = 0.02) | RTX 3070 |
| Visual-only retrieval (shared-class D2 subset) | DA = 0.893 | RTX 3070 |
| Caption-only retrieval (same subset) | DA = 0.495 | RTX 3070 |
| EUS OOD interception (held-out n = 226) | 71.2% (161/226) | RTX 3070 |
| EUS gradient-free onboarding (50 references) | DA = 0.696 | RTX 3070 |
| End-to-end median latency / peak memory | 559 ms / approximately 1.2 GB | RTX 3070 |

---

## Architecture

```
                           ┌────────────────────────────────────────────┐
                           │ Stage 1 — Visual Perception                │
 Input image  I  ────────▶ │   Florence-2-base (0.23 B, beam = 3)       │
                           │   ↓                                        │
                           │   Two-tier Pareidolia Correction (Eqs. 1–4)│
                           └─────────────────┬──────────────────────────┘
                                             ▼
                           ┌────────────────────────────────────────────┐
                           │ Stage 2 — Knowledge Retrieval              │
                           │   λ-Weighted Fusion (Eqs. 5–7, λ* = 0.7)   │
                           │   ▶ Image Gallery (1,639 fused embeddings) │
                           │     ChromaDB · HNSW · Top-K = 5            │
                           │   ▶ Auxiliary confidence score + flag     │
                           │     (logged only; raw margin decides)         │
                           └─────────────────┬──────────────────────────┘
                                             ▼
                           ┌────────────────────────────────────────────┐
                           │ Stage 3 — Scoring Decision                 │
                           │   Scoring  S_h, S_d  (Eqs. 8–9)            │
                           │   ↓                                        │
                           │   Priority-Ordered Decision Rule           │
                           │   (Eq. 10 + Eq. 11, θ_margin = 0.02)       │
                           └─────────────────┬──────────────────────────┘
                                             ▼
                       ┌────────────────────────────────────┐
                       │ Healthy │ Disease(D_k) │ Inconclusive│
                       └────────────────────────────────────┘
```

The **Text Knowledge Base** contains eight documents (seven D1 classes plus EUS,
loaded conditionally). Stage 3 uses the top-1 candidate document for evidence
scoring; the auxiliary confidence branch is logged and does not change the
retrieved class or final decision.

---

## Quick start

### Install

```bash
git clone https://github.com/stanleylia/fishdx.git
cd fishdx
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Single-image inference

```python
from pathlib import Path

from fishdx import Pipeline
from fishdx.config import load_config

pipeline = Pipeline(load_config("configs/default.yaml"))
report = pipeline.diagnose(Path("path/to/fish_image.jpg"))

print(report.decision)        # "Healthy" | "Disease" | "Inconclusive"
print(report.disease_class)   # e.g. "Bacterial Red Disease"
print(report.retrieval_margin)
```

---

## Repository layout

```
fishdx/
├── src/fishdx/                  # Canonical paper-faithful Python package
│   ├── pipeline.py              # 3-stage orchestrator
│   ├── config.py                # Pydantic-validated YAML loader
│   ├── perception/              # Stage 1 — Florence-2 + Pareidolia (Eqs. 1–4)
│   ├── retrieval/               # Stage 2 — Fusion + HNSW + auxiliary confidence
│   ├── scoring/                 # Stage 3 — Scoring + Precedence (Eqs. 8–11)
│   ├── kb/                      # Knowledge-base build/load
│   ├── metrics/                 # DA, SCA, Wilson CI, McNemar, bootstrap
│   ├── schemas/                 # Pydantic data-transfer models
│   └── utils/                   # Seeding, logging
│
├── tests/                       # Unit + integration
├── experiments/                 # Supporting development experiments
├── scripts/experiments/         # Manuscript and per-sample evaluation analyses
├── results/reviewer5/           # Per-sample and aggregate evaluation artefacts
├── REPRODUCIBILITY_MAP.md       # Equation, script and result map
├── MANIFEST.txt                 # Archival file inventory
├── scripts/                     # CLI tools (KB build, dataset organise, latency bench)
├── configs/                     # YAML hyperparameter store
├── knowledge_base/seed_data/    # 8 disease documents (Markdown source)
└── deploy/                      # Top-level docker-compose
```

---

## Per-sample evaluation archive (D2-final)

The complete per-sample archive for the D2-final evaluation set is released
under `results/reviewer5/`, so that every D2-final number in the manuscript can
be recomputed directly from the published records without re-running inference:

- `selective_predictions_d2final.jsonl`: 14,412 sample–method records spanning
  the identical ordered 2,402-image evaluation set across all six methods;
- `run_manifest.json`: configuration, input and archive SHA-256 hashes;
- `matched_coverage_knn.json`, `per_class_abstention.json` and
  `stage3_document_ablation.json`: matched-coverage, class-level abstention and
  candidate-conditioned evidence analyses;
- `diagnostic_phrase_substitution.json` and `.jsonl`: aggregate and per-caption
  intervention results.

The producing scripts are `scripts/experiments/exp_r5_*.py`. See
`REPRODUCIBILITY_MAP.md` and `results/README.md` for the complete mapping.

---

## Datasets

| ID | Source | Images | Used for |
|---|---|---|---|
| D1 | Freshwater Fish Disease in South Asia | 2,444 | Primary train/test (1,747 / 697) |
| D2 | Fish Disease Detection | 3,503 | Cross-dataset test + 56-image EUS subset |
| D3–D7 | Five auxiliary fish-disease sets | 28,658 | Subexperiments (consistency, robustness, screening) |
| D8 | Large-Scale Fish Species | 18,282 | Pareidolia negative control |
| D9 | Brackish Underwater Scene | 14,674 | Pareidolia negative control |

All nine datasets are publicly available on Kaggle under open licences.
Dataset files themselves are **not** included in this repository — the
build scripts download them on demand. **No live animals were handled in
the production of this work.**

---

## Citing this work

```bibtex
@article{liao2026perception,
  title={Visual embedding retrieval improves fish-disease classification when generated
         captions lack diagnostic vocabulary},
  author={Liao, Yen-Hsiang and Shih, Chao-Feng and Chang, Chung-Cheng},
  journal={Scientific Reports},
  year={2026},
  publisher={Nature Publishing Group}
}

@software{fishdx2026,
  title={fishdx --- Zero-Shot Fish Disease Classification via CLIP Fusion
         Embedding},
  author={Liao, Yen-Hsiang and Shih, Chao-Feng and Chang, Chung-Cheng},
  year={2026},
  version={0.1.0},
  url={https://github.com/stanleylia/fishdx}
}
```

---

## License

This project is licensed under the **GNU Affero General Public License v3.0
or later**; see [`LICENSE`](LICENSE).

Third-party model weights (Florence-2, OpenCLIP) are subject to their own
licences and are not covered by this LICENSE file.
