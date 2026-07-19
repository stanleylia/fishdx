# fishdx — Zero-Shot Fish Disease Classification via CLIP Fusion Embedding

> Reference implementation for the *Scientific Reports* manuscript
> **"Visual embedding retrieval shows potential to mitigate the perception–cognition gap of
> vision-language models in aquatic pathology"** (Liao, Shih, Chang, 2026).

A three-stage **deterministic** pipeline that classifies freshwater fish
disease images by combining (i) Florence-2 visual perception with two-tier
pareidolia correction, (ii) λ-weighted CLIP fusion retrieval against a
1,747-image fused-embedding gallery, and (iii) evidence-pool scoring with a
four-step precedence chain. No language-model free-form generation occurs at
inference time. The whole pipeline runs in < 600 ms per image on a single
NVIDIA RTX 3070 (8 GB VRAM).

---

## Headline numbers

| Setting | Metric | Hardware |
|---|---|---|
| D1 Test forced-choice (n = 697) | DA ≥ 0.999 | RTX 3090 / RTX 3070 |
| D1 → D2 cross-dataset (n = 3,200) | Decisive DA = 0.924 at 64.3% coverage (θ_margin = 0.02) | RTX 3090 |
| EUS gradient-free onboarding (n_test = 56) | DA = 0.714 at 50 reference images, 95% bootstrap CI [0.589, 0.821] | RTX 3090 |
| EUS OOD interception | ≈ 70% at θ_margin = 0.02 | RTX 3090 |
| RTX 3070 reimplementation | Retrieval DA = 1.000; Decision DA = 0.963 (95% Wilson CI 0.945–0.974) | RTX 3070 / 8 GB |
| End-to-end median latency | 533 ms / image | RTX 3070 |
| Peak GPU memory | 921 MB | RTX 3070 |

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
                           │   ▶ Image Gallery (1,747 fused embeddings) │
                           │     ChromaDB · HNSW · Top-K = 5            │
                           │   ▶ Bidirectional Verification Loop        │
                           │     (Algorithm 1, θ = 0.5, max_iter = 2)   │
                           └─────────────────┬──────────────────────────┘
                                             ▼
                           ┌────────────────────────────────────────────┐
                           │ Stage 3 — Scoring Decision                 │
                           │   Scoring  S_h, S_d  (Eqs. 8–9)            │
                           │   ↓                                        │
                           │   Four-Step Precedence Chain               │
                           │   (Eq. 10 + Eq. 11, θ_margin = 0.02)       │
                           └─────────────────┬──────────────────────────┘
                                             ▼
                       ┌────────────────────────────────────┐
                       │ Healthy │ Disease(D_k) │ Inconclusive│
                       └────────────────────────────────────┘
```

The right-side **Text Knowledge Base** (8 documents: 7 D1 classes + 1 EUS
loaded conditionally) is queried by the Bidirectional Verification Loop
(verify) and Stage 3 Scoring (evidence).

---

## Quick start

### Install

```bash
git clone https://github.com/<organisation>/fishdx.git
cd fishdx
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Single-image inference

```python
from fishdx import Pipeline
from fishdx.config import load_config

pipeline = Pipeline(load_config("configs/default.yaml"))
report = pipeline.run("path/to/fish_image.jpg")

print(report.decision)        # "Healthy" | "Disease(D_k)" | "Inconclusive"
print(report.disease_class)   # e.g. "Bacterial Red Disease"
print(report.confidence)
print(report.retrieval_top_k) # full Top-5 with similarities
```

---

## Repository layout

```
fishdx/
├── src/fishdx/                  # Canonical paper-faithful Python package
│   ├── pipeline.py              # 3-stage orchestrator
│   ├── config.py                # Pydantic-validated YAML loader
│   ├── perception/              # Stage 1 — Florence-2 + Pareidolia (Eqs. 1–4)
│   ├── retrieval/               # Stage 2 — Fusion + HNSW + BVL (Eqs. 5–7, Algo 1)
│   ├── scoring/                 # Stage 3 — Scoring + Precedence (Eqs. 8–11)
│   ├── kb/                      # Knowledge-base build/load
│   ├── metrics/                 # DA, SCA, Wilson CI, McNemar, bootstrap
│   ├── schemas/                 # Pydantic data-transfer models
│   └── utils/                   # Seeding, logging
│
├── tests/                       # Unit + integration
├── experiments/                 # EXP-1 through EXP-7 reproducibility scripts
├── scripts/                     # CLI tools (KB build, dataset organise, latency bench)
├── configs/                     # YAML hyperparameter store
├── knowledge_base/seed_data/    # 8 disease documents (Markdown source)
└── deploy/                      # Top-level docker-compose
```

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
  title={Visual embedding retrieval shows potential to mitigate the perception--cognition gap
         of vision-language models in aquatic pathology},
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
  url={https://github.com/<organisation>/fishdx}
}
```

---

## License

This project is licensed under the **fishdx Academic Research License v1.0**
(see [`LICENSE`](LICENSE)). The licence is **stricter than MIT / BSD /
Apache** and is **not OSI-approved** as "open source" in the conventional
sense. Key terms:

- **Permitted**: academic research, classroom teaching, personal study,
  non-commercial publication, reproduction of the paper's results.
- **Prohibited without a separate written agreement**: any commercial
  use, integration into a commercial product or paid service, production
  deployment by a for-profit entity, or training of any model/system
  offered commercially.
- **Source disclosure required**: any redistribution — source, binary,
  or network service — must include the complete corresponding source
  code under this same licence, free of additional charge.
- **Share-alike**: derivative works must remain under this licence; you
  may not relicense to MIT, BSD, Apache, GPL, AGPL, or any other licence
  without prior written permission.
- **Citation required**: academic use must cite the accompanying
  *Scientific Reports* paper.

For commercial licensing inquiries, contact **ccchang@mail.ntou.edu.tw**.

Third-party model weights (Florence-2, OpenCLIP) are subject to their own
licences and are not covered by this LICENSE file.
