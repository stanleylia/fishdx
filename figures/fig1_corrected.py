"""Fig. 1 — publication data-flow diagram (padding-checked, vector PNG/PDF/SVG)."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams["font.family"] = "Liberation Sans"

ROOT = Path(__file__).resolve().parent.parent
D2 = ROOT / "data/datasets/D2"
OUT = Path(__file__).resolve().parent

QUERY = D2 / "Bacterial diseases - Aeromoniasis_70.jpg"
GALLERY = [D2 / "Fungal diseases Saprolegniasis_33.jpg", D2 / "Bacterial diseases - Aeromoniasis_167.jpg",
           D2 / "EUS_1.jpg", D2 / "Bacterial Red disease_143.jpg"]

BLUE, BLUE_T = "#2C6FB5", "#EAF2FB"
GREEN, GREEN_T = "#1F9E77", "#E7F6F1"
ORANGE, ORANGE_T = "#D9772B", "#FCEFE3"
AMBER, RED = "#E0A32E", "#D1462F"
INK, SUB, ARROW = "#20242C", "#3B4152", "#454B57"
PANEL, PANEL_EDGE = "#FFFFFF", "#C6CFDB"

W, H = 20.7, 7.0
BY0, BY1 = 0.5, 5.55
LW_MAIN, LW_SUB = 3.0, 2.4

FS_STAGE, FS_BODY, FS_BOX, FS_MATH, FS_HEAD, FS_ROW, FS_OUT, FS_ILL, FS_PATH = 23, 15, 14.5, 14.5, 14, 13.5, 17, 12.5, 14.5


def rbox(ax, x0, y0, x1, y1, fc, ec, lw=1.5, z=2, r=0.03):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0.01,rounding_size={r}",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z))


def txt(ax, x, y, s, fs, color=INK, weight="normal", style="normal", ha="center", va="center", z=6, ls=1.2):
    ax.text(x, y, s, fontsize=fs, color=color, fontweight=weight, style=style, ha=ha, va=va, zorder=z, linespacing=ls)


def arrow(ax, x0, y0, x1, y1, color=ARROW, lw=LW_MAIN, ms=19, z=5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, linewidth=lw, color=color, zorder=z))


def polyarrow(ax, pts, color=ARROW, lw=LW_SUB, ms=15, z=5):
    for i in range(len(pts) - 2):
        ax.add_line(Line2D([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]], color=color, linewidth=lw, zorder=z, solid_capstyle="round"))
    arrow(ax, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], color=color, lw=lw, ms=ms, z=z)


def thumb(ax, path, x0, y0, x1, y1, z=4, ec="#8A93A6"):
    try:
        ax.imshow(Image.open(path).convert("RGB"), extent=[x0, x1, y0, y1], aspect="auto", zorder=z, interpolation="lanczos")
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=ec, linewidth=1.1, zorder=z + 1))
    except OSError:
        pass


def sidebar(ax, x0, accent):
    ax.add_patch(FancyBboxPatch((x0 + 0.05, BY0 + 0.12), 0.08, BY1 - BY0 - 0.24, boxstyle="round,pad=0,rounding_size=0.02",
                                linewidth=0, facecolor=accent, zorder=3))


def make():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    # ---- Input + single branch node
    thumb(ax, QUERY, 0.35, 3.0, 1.9, 4.5)
    ax.add_patch(Rectangle((0.8, 3.3), 0.62, 0.62, fill=False, linestyle=(0, (3, 2)), edgecolor="#F2C14E", linewidth=1.9, zorder=6))
    txt(ax, 1.13, 4.86, "Query fish image  $I$", FS_BODY, INK, "bold")
    txt(ax, 1.13, 2.66, "Visible disease signs", FS_BODY, SUB, style="italic")
    node1 = (2.13, 3.75)
    ax.add_line(Line2D([1.9, node1[0]], [3.75, 3.75], color=ARROW, lw=LW_MAIN, zorder=5, solid_capstyle="round"))
    ax.add_patch(Circle(node1, 0.075, facecolor=ARROW, edgecolor=ARROW, zorder=8))
    arrow(ax, node1[0] + 0.04, 3.75, 2.42, 3.75, lw=LW_MAIN)  # -> Stage 1

    # ---- Stage 1
    s1x0, s1x1 = 2.45, 5.05
    rbox(ax, s1x0, BY0, s1x1, BY1, BLUE_T, "#BBD2EC", lw=1.7); sidebar(ax, s1x0, BLUE)
    cx1 = (s1x0 + s1x1) / 2 + 0.05
    txt(ax, cx1, 5.16, "1  Perception", FS_STAGE, BLUE, "bold")
    txt(ax, cx1, 4.66, "Florence-2 dense captioning", FS_BODY, INK)
    txt(ax, cx1, 4.33, "Object-label extraction", FS_BODY, INK)
    rbox(ax, s1x0 + 0.22, 3.25, s1x1 - 0.12, 4.05, "#FBFDFF", PANEL_EDGE, lw=1.2, z=3)
    txt(ax, cx1, 3.85, "Generated caption  ($G$)", FS_HEAD, BLUE, "bold")
    txt(ax, cx1, 3.52, '"a fish with a red ulcer\non its lateral body"', FS_ROW, INK, style="italic", ls=1.2)
    arrow(ax, cx1, 3.21, cx1, 2.98, lw=LW_SUB, ms=13)
    txt(ax, cx1, 2.78, "Pareidolia correction", FS_BODY, INK, "bold")
    rbox(ax, s1x0 + 0.22, 1.52, s1x1 - 0.12, 2.5, "#FBFDFF", PANEL_EDGE, lw=1.2, z=3)
    txt(ax, cx1, 2.3, "Structural-label check", FS_HEAD, SUB, "bold")
    txt(ax, cx1, 1.98, "cage / mesh / net", FS_BOX, SUB, style="italic")
    txt(ax, cx1, 1.7, r"$\rightarrow$ net_damage", FS_BOX, GREEN, "bold")
    txt(ax, cx1, 1.02, "Outputs:  $G'$,  $L'$", FS_HEAD, INK, "bold")

    # ---- Stage 2
    s2x0, s2x1 = 5.45, 15.55
    rbox(ax, s2x0, BY0, s2x1, BY1, GREEN_T, "#B7E3D4", lw=1.7); sidebar(ax, s2x0, GREEN)
    txt(ax, (s2x0 + s2x1) / 2, 5.16, "2  Retrieval", FS_STAGE, GREEN, "bold")

    evx0, evx1 = 5.95, 7.2; evc = (evx0 + evx1) / 2
    rbox(ax, evx0, 3.9, evx1, 4.68, PANEL, "#9FC9E8", lw=1.4, z=4)
    txt(ax, evc, 4.29, "CLIP visual\nembedding  $E_v$", FS_BOX, INK)
    rbox(ax, evx0, 1.98, evx1, 2.76, PANEL, "#B7E3D4", lw=1.4, z=4)
    txt(ax, evc, 2.37, "CLIP text\nembedding  $E_t$", FS_BOX, INK)

    # image pathway from the single node, over the top into E_v (thinner than main arrows)
    polyarrow(ax, [(node1[0], 3.86), (node1[0], 6.25), (evc, 6.25), (evc, 4.7)], lw=1.9, ms=15)
    txt(ax, 4.0, 6.48, "Image pathway  ($I$)", FS_PATH, BLUE, "bold")
    # caption pathway: label directly above the Stage1 -> E_t arrow
    arrow(ax, s1x1 + 0.02, 2.37, evx0 - 0.02, 2.37, lw=LW_MAIN, ms=19)
    txt(ax, evc, 3.05, "Caption pathway  ($G'$)", FS_PATH, GREEN, "bold")

    # fusion node with the equation
    fx0, fx1 = 7.4, 9.05
    rbox(ax, fx0, 2.72, fx1, 3.95, "#FFF6E9", AMBER, lw=1.7, z=4)
    txt(ax, (fx0 + fx1) / 2, 3.68, "$\\lambda$-weighted fusion", FS_BOX, INK, "bold")
    txt(ax, (fx0 + fx1) / 2, 3.31, "$E_{\\mathrm{final}}=\\lambda E_v+(1{-}\\lambda)E_t$", 12, SUB)
    txt(ax, (fx0 + fx1) / 2, 2.96, "$\\lambda^{*}=0.7$", FS_MATH, ORANGE, "bold")
    arrow(ax, evx1, 4.15, fx0, 3.6, lw=LW_SUB, ms=12)
    arrow(ax, evx1, 2.5, fx0, 3.05, lw=LW_SUB, ms=12)

    # reference gallery deck
    gx, gy, half, step = 10.05, 3.33, 0.5, 0.1
    for i, g in enumerate(GALLERY):
        thumb(ax, g, gx - half + i * step, gy - half - i * step, gx + half + i * step, gy + half - i * step, z=4 + i)
    txt(ax, 10.15, 2.12, "Reference gallery\n1,639 fused embeddings", FS_ROW, SUB, ls=1.3)
    arrow(ax, fx1, 3.33, gx - half + 0.05, 3.33, lw=LW_SUB, ms=13)

    # Top-K panel (ranked neighbours; similarities illustrative, omitted for clarity)
    tx0, tx1 = 11.35, 14.05
    rbox(ax, tx0, 2.52, tx1, 4.15, PANEL, PANEL_EDGE, lw=1.4, z=6)
    txt(ax, (tx0 + tx1) / 2, 3.94, "Top-$K$ neighbours,  $K=5$", FS_HEAD, GREEN, "bold", z=7)
    for j, (r, nm) in enumerate([("1", "Aeromoniasis"), ("2", "Bacterial Red Disease"),
                                 ("3", "Fungal Saprolegniasis")]):
        yy = 3.6 - j * 0.29
        txt(ax, tx0 + 0.18, yy, r, FS_ROW, SUB, ha="left", z=7)
        txt(ax, tx0 + 0.42, yy, nm, FS_ROW, INK, ha="left", z=7)
    txt(ax, (tx0 + tx1) / 2, 2.68, "ranked by fused-embedding similarity", FS_ILL, SUB, style="italic", z=7)
    arrow(ax, gx + half + 0.28, 3.33, tx0 - 0.02, 3.33, lw=LW_SUB, ms=13)

    # retrieval-margin check (two centred lines, comfortably inside the box)
    rx0, rx1 = 14.2, 15.5
    rbox(ax, rx0, 2.78, rx1, 3.92, "#F4FBF8", "#8FCDB8", lw=1.5, z=6)
    txt(ax, (rx0 + rx1) / 2, 3.66, "Retrieval-", FS_ROW, GREEN, "bold", z=7)
    txt(ax, (rx0 + rx1) / 2, 3.4, "margin check", FS_ROW, GREEN, "bold", z=7)
    txt(ax, (rx0 + rx1) / 2, 3.06, "$\\Delta = s_1{-}s_2$", FS_BOX, INK, z=7)
    arrow(ax, tx1, 3.33, rx0 - 0.02, 3.33, lw=LW_SUB, ms=13)

    # ---- Auxiliary retrieval-confidence log as an explicit DASHED SIDE BRANCH
    # (Reviewer 5 comment 3(1), ADR-0017). The reweighted score and flag are logged only;
    # they never re-rank the candidates and never enter the decision, so the branch is
    # drawn dashed and has no outgoing arrow. It is no longer a numbered algorithm in the
    # manuscript, so the label must not say "Algorithm 1".
    ax0, ax1_, ay0, ay1 = 11.15, 14.25, 0.80, 1.86
    ax.add_patch(FancyBboxPatch((ax0, ay0), ax1_ - ax0, ay1 - ay0,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor="#FAFAFA", edgecolor="#9AA3B0",
                                linewidth=1.3, linestyle=(0, (4, 3)), zorder=5))
    txt(ax, (ax0 + ax1_) / 2, 1.63, "Auxiliary retrieval-", 12.5, "#6B7280", "bold", z=7)
    txt(ax, (ax0 + ax1_) / 2, 1.40, "confidence log", 12.5, "#6B7280", "bold", z=7)
    txt(ax, (ax0 + ax1_) / 2, 1.13, "returns (score, flag); logged only —", 11.0, "#6B7280", z=7)
    txt(ax, (ax0 + ax1_) / 2, 0.94, "does not affect the decision", 11.0, "#6B7280", z=7)
    ax.add_patch(FancyArrowPatch((12.7, 2.50), (12.7, ay1 + 0.02),
                                 arrowstyle="-|>", mutation_scale=13, color="#9AA3B0",
                                 linewidth=1.6, linestyle=(0, (4, 3)),
                                 shrinkA=0, shrinkB=0, zorder=5))

    # ---- Stage 3
    s3x0, s3x1 = 15.85, 18.35
    rbox(ax, s3x0, BY0, s3x1, BY1, ORANGE_T, "#F0C9A3", lw=1.7); sidebar(ax, s3x0, ORANGE)
    cx3 = (s3x0 + s3x1) / 2 + 0.05
    bx0, bx1 = s3x0 + 0.18, s3x1 - 0.1
    txt(ax, cx3, 5.16, "3  Decision", FS_STAGE, ORANGE, "bold")
    arrow(ax, rx1 + 0.02, 3.33, s3x0 - 0.02, 4.62, lw=LW_MAIN, ms=19)  # margin check -> Stage 3 (into evidence pool)

    rbox(ax, bx0, 4.28, bx1, 5.0, PANEL, "#F0C9A3", lw=1.4, z=4)
    txt(ax, cx3, 4.74, "Evidence pool", FS_HEAD, INK, "bold")
    txt(ax, cx3, 4.44, "$\\mathcal{O}=G'+r_1.\\mathrm{text}$", FS_MATH, SUB)
    arrow(ax, cx3, 4.25, cx3, 4.06, lw=LW_SUB, ms=11)
    rbox(ax, bx0, 3.46, bx1, 4.04, PANEL, "#8FCDB8", lw=1.4, z=4)
    txt(ax, cx3, 3.83, "Healthy evidence", FS_HEAD, INK, "bold")
    txt(ax, cx3, 3.58, "$S_h$", FS_MATH, INK)
    arrow(ax, cx3, 3.43, cx3, 3.24, lw=LW_SUB, ms=11)
    rbox(ax, bx0, 2.64, bx1, 3.22, PANEL, "#F1B7A6", lw=1.4, z=4)
    txt(ax, cx3, 3.01, "Disease evidence", FS_HEAD, INK, "bold")
    txt(ax, cx3, 2.76, "$S_d^{(k)}$", FS_MATH, INK)
    arrow(ax, cx3, 2.61, cx3, 2.42, lw=LW_SUB, ms=11)
    rbox(ax, bx0, 1.35, bx1, 2.4, "#FFF6E9", AMBER, lw=1.7, z=4)
    txt(ax, cx3, 2.18, "Selective decision gate", FS_ROW, INK, "bold")
    txt(ax, cx3, 1.88, "Score and margin rules", FS_ROW, SUB)
    txt(ax, cx3, 1.56, "$\\theta_{\\mathrm{margin}}=0.02$", FS_HEAD, ORANGE, "bold")

    # ---- Single branching node (moved right) + outputs
    node2 = (18.72, 1.875)
    arrow(ax, s3x1 + 0.02, 1.875, node2[0] - 0.04, 1.875, lw=LW_MAIN, ms=17)  # one horizontal arrow
    ax.add_patch(Circle(node2, 0.08, facecolor=ARROW, edgecolor=ARROW, zorder=8))
    ox0, ox1 = 19.05, 20.6
    outs = [("Healthy", GREEN, 2.95, 3.6), (r"Disease  $D_{k^{*}}$", RED, 1.55, 2.2),
            ("Inconclusive\nExpert referral", AMBER, 0.2, 1.2)]
    for label, ec, y0, y1 in outs:
        rbox(ax, ox0, y0, ox1, y1, PANEL, ec, lw=2.1, z=6)
        txt(ax, (ox0 + ox1) / 2, (y0 + y1) / 2, label, FS_OUT, ec, "bold", z=7, ls=1.2)
        arrow(ax, node2[0] + 0.04, node2[1], ox0 - 0.02, (y0 + y1) / 2, lw=2.2, ms=15)

    for ext in ("png", "pdf", "svg"):
        dpi = 400 if ext == "png" else None
        fig.savefig(OUT / f"Fig1_corrected.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    print("saved Fig1_corrected (png/pdf/svg)")
    plt.close(fig)


if __name__ == "__main__":
    make()
