#!/usr/bin/env python3
"""Fig. S2 - lambda sensitivity on cross-dataset D2 (leak-free D2 -> D1 gallery).

Regenerates Supplementary Fig. S2 DIRECTLY from the shipped result artefact
results/d2_clean_tables.json -> "lambda_sweep_7class" (the values reported in
Supplementary Note S9 and consistent with Table 3 Overlap DA). Deterministic; no
image data required. This is the source of record for the S9 lambda sweep; the
older results/lambda_d2_ablation.json is a superseded pre-dedup scaffold and is
NOT used here.

Usage:  python figures_v3l/figS2_lambda_d2.py
Output: figures_v3l/Fig_lambda_d2.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results" / "d2_clean_tables.json"
OUT = Path(__file__).resolve().parent / "Fig_lambda_d2.png"

sw = json.loads(SRC.read_text())["lambda_sweep_7class"]
xs = sorted(sw, key=float)
lam = [float(x) for x in xs]
da = [sw[x] for x in xs]
i7, i8 = xs.index("0.7"), xs.index("0.8")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.linewidth": 0.8, "axes.edgecolor": "#333333"})
fig, ax = plt.subplots(figsize=(5.0, 3.4), dpi=300)
ax.axvspan(0.6, 1.0, color="#4C78A8", alpha=0.07, zorder=0)
ax.text(0.80, 0.523, "broad plateau\nλ ∈ [0.6, 1.0]", ha="center",
        va="center", fontsize=7.5, color="#2A4E6C")
ax.plot(lam, da, "-o", color="#2F5C8F", markerfacecolor="white",
        markeredgecolor="#2F5C8F", markeredgewidth=1.2, markersize=5,
        linewidth=1.8, zorder=3)
# One unified leader-line style for every annotation: thin "->" arrow, lw 0.9,
# colour-matched to its label. All four callouts sit in empty space below/around
# the curve so none collide with the title band or with each other.
_arrow = dict(arrowstyle="->", lw=0.9)
ax.annotate(f"caption-only\n(λ=0): {da[0]:.3f}", xy=(0.0, da[0]),
            xytext=(0.10, da[0] + 0.085), fontsize=7.5, color="#B5482E",
            arrowprops=dict(_arrow, color="#B5482E"))
ax.plot(0.7, da[i7], marker="*", markersize=17, color="#E4A11B",
        markeredgecolor="#8A5B00", markeredgewidth=0.7, zorder=5)
ax.annotate(f"λ* = 0.7 (paper)\n{da[i7]:.3f}", xy=(0.7, da[i7]),
            xytext=(0.40, 0.855), fontsize=8, color="#6B4A00", ha="center",
            fontweight="bold", arrowprops=dict(_arrow, color="#8A5B00"))
ax.annotate(f"D2-optimal\n(λ=0.8): {da[i8]:.3f}", xy=(0.8, da[i8]),
            xytext=(0.865, 0.77), fontsize=7.5, color="#333333", ha="center",
            arrowprops=dict(_arrow, color="#555555"))
ax.annotate(f"visual-only\n(λ=1): {da[-1]:.3f}", xy=(1.0, da[-1]),
            xytext=(0.92, 0.635), fontsize=7.5, color="#3B7A57", ha="center",
            arrowprops=dict(_arrow, color="#3B7A57"))
ax.set_xlabel("Fusion weight  λ  (visual share)", fontsize=9.5)
ax.set_ylabel("Cross-dataset D2 accuracy\n(Overlap DA, 7 shared classes, n = 2,176)", fontsize=9)
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(0.45, 0.965)  # ~6% headroom above the 0.905 peak -> clears the title band
ax.xaxis.set_major_locator(MultipleLocator(0.1))
ax.yaxis.set_major_locator(MultipleLocator(0.1))
ax.grid(True, linewidth=0.4, color="#DDDDDD", zorder=0)
ax.set_title("Fig. S2 | λ sensitivity on cross-dataset D2 (leak-free D2→D1 gallery)",
             fontsize=8.5, loc="left", pad=12)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=300)
print("saved:", OUT, "| lambda*=0.7 ->", da[i7], "| lambda=0.8 ->", da[i8],
      "| caption-only", da[0], "| visual", da[-1])
