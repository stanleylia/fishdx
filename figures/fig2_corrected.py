"""Fig. 2 (corrected) — Caption-level semantic deficit and cross-dataset
modality comparison, as a readable 3-panel layout.

  (a) Florence-2 keyword SCA per class (D1 Train)
  (b) Cross-dataset D2 modality comparison (caption / visual / fusion)
  (c) Second-VLM control (Florence-2 vs Qwen2-VL-2B) keyword SCA per class,
      with the caption-retrieval DA (0.426 vs 0.381) annotated.

All canonical numbers are preserved verbatim from the verified result files /
main-text values. Unified Arial-substitute font + colour-blind-safe palette.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter

from paper_style import (COLORS, apply_style, save_pair, format_da, wilson_ci_array,
                         CI_ERROR_KW, barh_value_label)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = str(ROOT / "figures_v3l")

BLUE = COLORS["KNN_BLUE"]          # #0072B2 visual / D1-train
LBLUE = COLORS["KNN_LIGHT_BLUE"]   # #56B4E9 second comparison (Qwen)
GRAY = COLORS["CLIP_ZS_GRAY"]      # #7F7F7F caption / inactive
ORANGE = COLORS["PIPELINE_RED"]    # #D55E00 pipeline / fusion
REF = COLORS["REFERENCE_GRAY"]
GRID = "#E6E6E6"

# Font sizes (>= 7.5 pt everywhere)
FS_PANEL, FS_AX, FS_TICK, FS_VAL = 9.5, 8.5, 8.0, 8.0

DIS = ['Healthy', 'Fungal', 'Bact. gill', 'Aeromon.', 'Viral WT', 'Bact. red', 'Parasitic']
DIS_KEY = ['Healthy Fish', 'Fungal Saprolegniasis', 'Bacterial Gill Disease',
           'Aeromoniasis', 'Viral White Tail Dis.', 'Bacterial Red Disease',
           'Parasitic Diseases']


def _grid(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", linestyle="-", linewidth=0.5, color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FS_TICK)


def _make() -> str:
    apply_style()
    t = json.loads((ROOT / "results/d2_clean_tables.json").read_text())
    n_ov = int(t["n_overlap_7class"])
    mod = t["modality"]

    n_per_class = np.array([250, 250, 250, 250, 247, 250, 250])
    sca_values = np.array([0.0005, 0.0023, 0.0056, 0.0040, 0.0005, 0.0019, 0.0008])  # D1-Train Confirmed-PHRASES (order: Healthy,Fungal,Bact.gill,Aeromon,ViralWT,Bact.red,Parasitic)
    y = np.arange(len(DIS))

    fig = plt.figure(figsize=(6.6, 4.7))
    gs = GridSpec(2, 2, figure=fig, hspace=0.62, wspace=0.52,
                  left=0.115, right=0.975, top=0.905, bottom=0.19)
    ax_a = fig.add_subplot(gs[0, 0]); ax_c = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, :])

    # ── (a) keyword SCA ──
    ax_a.barh(y, sca_values, color=BLUE, edgecolor="black", linewidth=0.5, height=0.68)
    ax_a.set_yticks(y); ax_a.set_yticklabels(DIS, fontsize=FS_TICK); ax_a.invert_yaxis()
    ax_a.set_xlim(0, 0.008); ax_a.set_xlabel("Avg keyword SCA", fontsize=FS_AX, fontweight="bold")
    ax_a.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    for i, v in enumerate(sca_values):
        barh_value_label(ax_a, v, i, f"{v*100:.2f}%", COLORS["DA_BLACK"], gap=0.00025, fontsize=7.5)
    ax_a.set_title("(a) Florence-2 keyword SCA", fontsize=FS_PANEL, fontweight="bold",
                   loc="left", pad=6, color=COLORS["DA_BLACK"])
    _grid(ax_a)

    # ── (b) cross-dataset modality ──
    meth = ['Caption-text', 'Visual k-NN', 'Fused λ=0.7']
    da_c = np.array([float(mod["caption_only"]), float(mod["visual"]), float(mod["fusion"])])
    lab_c = ["0.495", "0.893", "0.897"]
    col_c = [GRAY, BLUE, ORANGE]
    err_c = wilson_ci_array(da_c, np.array([n_ov]*3))
    yc = np.arange(3)
    ax_c.barh(yc, da_c, color=col_c, edgecolor="black", linewidth=0.5, height=0.6)
    for i, (v, e0, e1, col) in enumerate(zip(da_c, err_c[0], err_c[1], col_c)):
        ax_c.errorbar(v, i, xerr=[[e0], [e1]], fmt="none", ecolor=col, elinewidth=1.1, capsize=0, zorder=4)
    ax_c.set_yticks(yc); ax_c.set_yticklabels(meth, fontsize=FS_TICK); ax_c.invert_yaxis()
    ax_c.set_xlim(0, 1.20); ax_c.set_xlabel("Cross-dataset D2 DA (95% Wilson CI)", fontsize=FS_AX, fontweight="bold")
    for i, (v, e, l) in enumerate(zip(da_c, err_c[1], lab_c)):
        barh_value_label(ax_c, v, i, l, COLORS["DA_BLACK"], fixed_x=0.98, fontsize=FS_VAL)
    ax_c.set_title("(b) Cross-dataset modality (D2)", fontsize=FS_PANEL, fontweight="bold",
                   loc="left", pad=6, color=COLORS["DA_BLACK"])
    _grid(ax_c)

    # ── (d) second-VLM control ──
    sv = json.loads((ROOT / "results/second_vlm_sca_phrases.json").read_text())
    key = {'Healthy Fish': 'Healthy Fish', 'Fungal Saprolegniasis': 'Fungal diseases Saprolegniasis',
           'Bacterial Gill Disease': 'Bacterial gill disease', 'Aeromoniasis': 'Bacterial diseases - Aeromoniasis',
           'Viral White Tail Dis.': 'Viral diseases White tail disease', 'Bacterial Red Disease': 'Bacterial Red disease',
           'Parasitic Diseases': 'Parasitic diseases'}
    # Round the source SCA to the 1-dp precision that is *displayed*, then use the
    # SAME rounded array for both the bars and the labels. This guarantees equal
    # displayed values (e.g. Parasitic 3.7 / 3.7) render as exactly equal-length
    # bars — no hidden sub-decimal difference between bar and label.
    flo_pct = np.round(np.array([sv['per_class'][key[d]]['florence_sca'] for d in DIS_KEY]) * 100, 2)
    qwe_pct = np.round(np.array([sv['per_class'][key[d]]['qwen_sca'] for d in DIS_KEY]) * 100, 2)
    flo, qwe = flo_pct / 100.0, qwe_pct / 100.0
    hd = 0.38
    # Solid bars for BOTH models (consistent with panels a–c); Florence-2 on top
    # (dark blue), Qwen2-VL-2B below (light blue). No hatch.
    ax_d.barh(y - hd/2, flo, hd, color=BLUE, edgecolor="black", linewidth=0.4, label="Florence-2")
    ax_d.barh(y + hd/2, qwe, hd, color=LBLUE, edgecolor="black", linewidth=0.4, label="Qwen2-VL-2B")
    for i in range(len(DIS)):
        barh_value_label(ax_d, 0, y[i], f"{flo_pct[i]:.2f} / {qwe_pct[i]:.2f}%",
                         COLORS["DA_BLACK"], fixed_x=0.0066, fontsize=7.0)
    ax_d.set_yticks(y); ax_d.set_yticklabels(DIS, fontsize=FS_TICK); ax_d.invert_yaxis()
    ax_d.set_xlim(0, 0.008)
    ax_d.set_xticks([0.0, 0.002, 0.004, 0.006])
    ax_d.set_xlabel("Avg keyword SCA (Florence-2 / Qwen2-VL-2B)",
                    fontsize=FS_AX, fontweight="bold")
    ax_d.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    ax_d.set_title("(c) Second-VLM control: keyword SCA", fontsize=FS_PANEL, fontweight="bold",
                   loc="left", pad=6, color=COLORS["DA_BLACK"])
    ax_d.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
                frameon=False, fontsize=8.5, handlelength=1.6, columnspacing=1.8)
    _grid(ax_d)
    fig.text(0.5, 0.015,
             "Second-VLM caption-retrieval DA: Florence-2 0.426 vs Qwen2-VL-2B 0.381 "
             "(matched D1 sample, n = 700).",
             ha="center", va="bottom", fontsize=7.3, style="italic", color=REF)

    print(f"Fig2 modality n={n_ov}: caption {da_c[0]:.4f} visual {da_c[1]:.4f} fusion {da_c[2]:.4f}")
    return save_pair(fig, "Fig2_corrected", OUT_DIR)


if __name__ == "__main__":
    print(_make())
    plt.close("all")
