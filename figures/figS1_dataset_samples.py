#!/usr/bin/env python3
"""Fig. S1 - one clean, representative D2 image per class (larger single-image atlas).

Every panel is a *raw, unaugmented* D2 original: no rotation / shear / affine warp and no
border-replication or letterbox padding. This D2 (Kaggle) set stores many pre-augmented
copies on disk, so filename filtering ("aug") is not sufficient; each candidate is scored by
_smear_score() and only images clean on all four borders are eligible (see that function).
The four disease panels re-picked after review are Aeromoniasis (_70, haemorrhagic flank
ulcer), Bacterial Gill Disease (_45, inflamed red gill in an open mouth), Parasitic Diseases
(_6, an Argulus fish louse) and Fungal Saprolegniasis (_33, white cotton-wool growth). Images
are loaded via PIL in RGB, so there is no BGR/RGB colour-channel mismatch. Deterministic;
requires the public D2 dataset under data/datasets/D2 (see Data availability).

Usage:  python figures_v3l/figS1_dataset_samples.py
Output: figures_v3l/Fig_dataset_samples.png  +  results/figS1_samples_manifest.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
D2_DIR = ROOT / "data" / "datasets" / "D2"
OUT = Path(__file__).resolve().parent / "Fig_dataset_samples.png"
MANIFEST = ROOT / "results" / "figS1_samples_manifest.json"

# (display name, D2 filename) - one manually verified, raw/unaugmented image per class.
# The re-picked panels (_70 / _45 / _6 / _33) are the cleanest originals surviving an automatic
# border-replication screen (see _smear_score), each showing an unambiguous diagnostic feature:
# haemorrhagic flank ulcer / inflamed red gill / Argulus fish louse / white cotton-wool fungus.
SELECTION = [
    ("Aeromoniasis", "Bacterial diseases - Aeromoniasis_70.jpg"),
    ("Bacterial Gill Disease", "Bacterial gill disease_45.jpg"),
    ("Bacterial Red Disease", "Bacterial Red disease_143.jpg"),
    ("Fungal Saprolegniasis", "Fungal diseases Saprolegniasis_33.jpg"),
    ("Healthy Fish", "Healthy Fish_119.jpg"),
    ("Parasitic Diseases", "Parasitic diseases_6.jpg"),
    ("Viral White Tail Disease", "Viral diseases White tail disease_180.jpg"),
    ("EUS (D2-only, OOD)", "EUS_1.jpg"),
]

# Reject threshold for the border-replication screen. A rotated + BORDER_REPLICATE image leaves a
# smooth directional smear along one edge: gradient perpendicular to that edge collapses to ~0
# while gradient along it stays high. Clean D2 originals score < 0 on every edge; the augmented
# panels flagged in review scored +0.7 .. +2.0. Calibrated on Healthy_119/EUS_1 (clean) vs the
# rejected _242/_117/_142 (dirty).
_SMEAR_REJECT = 0.3


def _smear_score(im: Image.Image) -> float:
    """Max over the four borders of the replicate-smear signature (higher = more augmented)."""
    a = np.asarray(im.convert("L"), dtype=np.float32)
    h, w = a.shape
    b = max(6, int(0.08 * min(h, w)))

    def edge(strip: np.ndarray, perp_axis: int, along_axis: int) -> float:
        along = float(np.abs(np.diff(strip, axis=along_axis)).mean())
        perp = float(np.abs(np.diff(strip, axis=perp_axis)).mean())
        return along - 2.0 * perp

    return max(
        edge(a[:, :b], 1, 0), edge(a[:, -b:], 1, 0),
        edge(a[:b, :], 0, 1), edge(a[-b:, :], 0, 1),
    )


def main() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12, 7))
    # generous spacing so per-image titles never touch the row above
    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.03, wspace=0.10, hspace=0.32)
    manifest = {}
    for ax, (disp, fname) in zip(axes.ravel(), SELECTION):
        p = D2_DIR / fname
        if p.exists():
            im = Image.open(p).convert("RGB")  # PIL -> RGB, no BGR mismatch
            score = _smear_score(im)
            if score > _SMEAR_REJECT:
                raise ValueError(
                    f"{disp} panel {fname!r} fails the border-replication screen "
                    f"(smear={score:.2f} > {_SMEAR_REJECT}); pick a clean, unaugmented original."
                )
            ax.imshow(im)
            manifest[disp] = {"file": fname, "smear_score": round(score, 2)}
        else:
            ax.text(0.5, 0.5, "image not found\n(place D2 under data/datasets/D2)",
                    ha="center", va="center", fontsize=8, color="#888")
        ax.set_title(disp, fontsize=10, pad=6)
        ax.axis("off")
    fig.suptitle("Fig. S1 | Representative D2 disease image per class (one clean example)",
                 fontsize=12, y=0.975)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    MANIFEST.write_text(json.dumps(
        {"source": "manually verified raw D2 originals; all pass the border-replication screen "
                   f"(smear < {_SMEAR_REJECT})",
         "selection": manifest}, indent=2, ensure_ascii=False))
    print("saved:", OUT, "| picked:", len(manifest))


if __name__ == "__main__":
    main()
