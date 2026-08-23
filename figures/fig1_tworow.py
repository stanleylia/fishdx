"""Fig. 1 — two-row data-flow diagram (taller aspect for larger print fonts; R3)."""
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

W, H = 13.2, 8.4
LW_MAIN, LW_SUB = 3.0, 2.3
FS_STAGE, FS_BODY, FS_BOX, FS_MATH, FS_HEAD, FS_ROW, FS_OUT, FS_ILL, FS_PATH = 25, 16, 15, 15, 15, 14, 18, 13, 15

R1Y0, R1Y1 = 4.5, 7.75
R2Y0, R2Y1 = 0.55, 3.55


def rbox(ax, x0, y0, x1, y1, fc, ec, lw=1.5, z=2, r=0.04):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0.01,rounding_size={r}",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z))


def txt(ax, x, y, s, fs, color=INK, weight="normal", style="normal", ha="center", va="center", z=6, ls=1.2):
    ax.text(x, y, s, fontsize=fs, color=color, fontweight=weight, style=style, ha=ha, va=va, zorder=z, linespacing=ls)


def arrow(ax, x0, y0, x1, y1, color=ARROW, lw=LW_MAIN, ms=20, z=5):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, linewidth=lw, color=color, zorder=z))


def polyarrow(ax, pts, color=ARROW, lw=LW_MAIN, ms=18, z=5):
    for i in range(len(pts) - 2):
        ax.add_line(Line2D([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]], color=color, linewidth=lw, zorder=z, solid_capstyle="round"))
    arrow(ax, pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], color=color, lw=lw, ms=ms, z=z)


def thumb(ax, path, x0, y0, x1, y1, z=4, ec="#8A93A6"):
    try:
        ax.imshow(Image.open(path).convert("RGB"), extent=[x0, x1, y0, y1], aspect="auto", zorder=z, interpolation="lanczos")
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=ec, linewidth=1.1, zorder=z + 1))
    except OSError:
        pass


def sidebar(ax, x0, y0, y1, accent):
    ax.add_patch(FancyBboxPatch((x0 + 0.05, y0 + 0.1), 0.08, y1 - y0 - 0.2, boxstyle="round,pad=0,rounding_size=0.02",
                                linewidth=0, facecolor=accent, zorder=3))


def make():
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    # ============ ROW 1: Input -> 1 Perception -> 2 Retrieval ============
    # Input
    thumb(ax, QUERY, 0.2, 5.5, 1.55, 6.85)
    ax.add_patch(Rectangle((0.6, 5.78), 0.55, 0.55, fill=False, linestyle=(0, (3, 2)), edgecolor="#F2C14E", linewidth=1.9, zorder=6))
    txt(ax, 0.87, 7.08, "Query fish image  $I$", FS_BODY, INK, "bold")
    txt(ax, 0.87, 5.28, "Visible disease signs", FS_ROW, SUB, style="italic")
    node1 = (1.78, 6.18)
    ax.add_line(Line2D([1.55, node1[0]], [6.18, 6.18], color=ARROW, lw=LW_MAIN, zorder=5, solid_capstyle="round"))
    ax.add_patch(Circle(node1, 0.07, facecolor=ARROW, edgecolor=ARROW, zorder=8))
    arrow(ax, node1[0] + 0.04, 6.18, 2.02, 6.18)

    # Stage 1
    s1x0, s1x1 = 2.05, 5.15
    rbox(ax, s1x0, R1Y0, s1x1, R1Y1, BLUE_T, "#BBD2EC", lw=1.7); sidebar(ax, s1x0, R1Y0, R1Y1, BLUE)
    cx1 = (s1x0 + s1x1) / 2 + 0.05
    txt(ax, cx1, 7.42, "1  Perception", FS_STAGE, BLUE, "bold")
    txt(ax, cx1, 6.95, "Florence-2 dense captioning", FS_ROW, INK)
    rbox(ax, s1x0 + 0.2, 6.05, s1x1 - 0.12, 6.72, "#FBFDFF", PANEL_EDGE, lw=1.2, z=3)
    txt(ax, cx1, 6.5, "Generated caption  ($G$)", FS_BOX, BLUE, "bold")
    txt(ax, cx1, 6.2, '"a fish with a red ulcer …"', FS_ILL, INK, style="italic")
    arrow(ax, cx1, 6.0, cx1, 5.82, lw=LW_SUB, ms=13)
    txt(ax, cx1, 5.62, "Pareidolia correction", FS_ROW, INK, "bold")
    rbox(ax, s1x0 + 0.2, 4.85, s1x1 - 0.12, 5.42, "#FBFDFF", PANEL_EDGE, lw=1.2, z=3)
    txt(ax, cx1, 5.2, "cage / mesh / net", FS_BOX, SUB, style="italic")
    txt(ax, cx1, 4.98, r"$\rightarrow$ net_damage", FS_BOX, GREEN, "bold")
    txt(ax, cx1, 4.68, "Outputs:  $G'$,  $L'$", FS_BOX, INK, "bold")

    # Stage 2
    s2x0, s2x1 = 5.55, 12.95
    rbox(ax, s2x0, R1Y0, s2x1, R1Y1, GREEN_T, "#B7E3D4", lw=1.7); sidebar(ax, s2x0, R1Y0, R1Y1, GREEN)
    txt(ax, (s2x0 + s2x1) / 2, 7.42, "2  Retrieval", FS_STAGE, GREEN, "bold")

    evx0, evx1 = 5.95, 7.15; evc = (evx0 + evx1) / 2
    rbox(ax, evx0, 6.15, evx1, 6.85, PANEL, "#9FC9E8", lw=1.4, z=4)
    txt(ax, evc, 6.5, "CLIP visual\nembedding $E_v$", FS_ILL, INK)
    rbox(ax, evx0, 5.0, evx1, 5.7, PANEL, "#B7E3D4", lw=1.4, z=4)
    txt(ax, evc, 5.35, "CLIP text\nembedding $E_t$", FS_ILL, INK)
    # pathways — image pathway routed ABOVE the panels to avoid crossing the Perception box
    polyarrow(ax, [(node1[0], 6.32), (node1[0], 8.08), (evc, 8.08), (evc, 6.87)], lw=2.0, ms=15)
    txt(ax, 4.1, 8.26, "Image pathway ($I$)", FS_ILL, BLUE, "bold")
    arrow(ax, s1x1 + 0.02, 5.35, evx0 - 0.02, 5.35, lw=LW_MAIN, ms=18)
    txt(ax, evc, 5.9, "Caption ($G'$)", FS_ILL, GREEN, "bold")

    # fusion
    fx0, fx1 = 7.35, 8.75
    rbox(ax, fx0, 5.35, fx1, 6.5, "#FFF6E9", AMBER, lw=1.7, z=4)
    txt(ax, (fx0 + fx1) / 2, 6.24, "$\\lambda$-fusion", FS_BOX, INK, "bold")
    txt(ax, (fx0 + fx1) / 2, 5.9, "$E_{\\mathrm{f}}=\\lambda E_v+(1{-}\\lambda)E_t$", FS_ILL, SUB)
    txt(ax, (fx0 + fx1) / 2, 5.58, "$\\lambda^{*}=0.7$", FS_BOX, ORANGE, "bold")
    arrow(ax, evx1, 6.4, fx0, 6.1, lw=LW_SUB, ms=12)
    arrow(ax, evx1, 5.4, fx0, 5.75, lw=LW_SUB, ms=12)

    # gallery
    gx, gy, half, step = 9.55, 5.9, 0.46, 0.09
    for i, g in enumerate(GALLERY):
        thumb(ax, g, gx - half + i * step, gy - half - i * step, gx + half + i * step, gy + half - i * step, z=4 + i)
    txt(ax, 9.62, 4.86, "Reference gallery\n1,639 fused embeddings", FS_ILL, SUB, ls=1.25)
    arrow(ax, fx1, 5.92, gx - half + 0.02, 5.92, lw=LW_SUB, ms=13)

    # Top-K
    tx0, tx1 = 10.75, 12.85
    rbox(ax, tx0, 5.28, tx1, 6.72, PANEL, PANEL_EDGE, lw=1.4, z=6)
    txt(ax, (tx0 + tx1) / 2, 6.5, "Top-$K$ neighbours, $K=5$", FS_BOX, GREEN, "bold", z=7)
    for j, (r, nm) in enumerate([("1", "Aeromoniasis"), ("2", "Bacterial Red Disease"),
                                 ("3", "Fungal Saprolegniasis")]):
        yy = 6.16 - j * 0.28
        txt(ax, tx0 + 0.16, yy, r, FS_ROW, SUB, ha="left", z=7)
        txt(ax, tx0 + 0.4, yy, nm, FS_ROW, INK, ha="left", z=7)
    txt(ax, (tx0 + tx1) / 2, 5.44, "ranked by fused-embedding similarity", 11, SUB, style="italic", z=7)
    arrow(ax, gx + half + 0.24, 5.92, tx0 - 0.02, 5.92, lw=LW_SUB, ms=13)

    # ============ RETURN CONNECTOR: Retrieval -> Decision ============
    chy = 4.05
    polyarrow(ax, [((tx0 + tx1) / 2, 5.26), ((tx0 + tx1) / 2, chy), (2.3, chy), (2.3, R2Y1 + 0.02)], lw=LW_MAIN, ms=19)
    txt(ax, 7.4, chy + 0.2, "Retrieval margin $\\Delta = s_1 - s_2$  →  decision", FS_ILL, ARROW, style="italic")

    # ============ ROW 2: 3 Decision (horizontal) -> outputs ============
    s3x0, s3x1 = 2.05, 8.7
    rbox(ax, s3x0, R2Y0, s3x1, R2Y1, ORANGE_T, "#F0C9A3", lw=1.7); sidebar(ax, s3x0, R2Y0, R2Y1, ORANGE)
    txt(ax, s3x0 + 1.15, 3.18, "3  Decision", FS_STAGE, ORANGE, "bold")

    # evidence pool
    epx0, epx1 = 2.35, 3.95
    rbox(ax, epx0, 1.35, epx1, 2.55, PANEL, "#F0C9A3", lw=1.4, z=4)
    txt(ax, (epx0 + epx1) / 2, 2.24, "Evidence pool", FS_BOX, INK, "bold")
    txt(ax, (epx0 + epx1) / 2, 1.9, "$\\mathcal{O}=G'+r_1.\\mathrm{text}$", FS_ILL, SUB)
    txt(ax, (epx0 + epx1) / 2, 1.58, "(caption + KB doc)", FS_ILL, SUB, style="italic")
    arrow(ax, epx1, 1.95, 4.35, 1.95, lw=LW_SUB, ms=14)

    # scores (two stacked)
    scx0, scx1 = 4.4, 5.95
    rbox(ax, scx0, 2.05, scx1, 2.62, PANEL, "#8FCDB8", lw=1.4, z=4)
    txt(ax, (scx0 + scx1) / 2, 2.34, "Healthy evidence $S_h$", FS_ILL, INK, "bold")
    rbox(ax, scx0, 1.28, scx1, 1.85, PANEL, "#F1B7A6", lw=1.4, z=4)
    txt(ax, (scx0 + scx1) / 2, 1.57, "Disease evidence $S_d^{(k)}$", FS_ILL, INK, "bold")
    arrow(ax, scx1, 1.95, 6.35, 1.95, lw=LW_SUB, ms=14)

    # decision gate
    dgx0, dgx1 = 6.4, 8.5
    rbox(ax, dgx0, 1.3, dgx1, 2.6, "#FFF6E9", AMBER, lw=1.7, z=4)
    txt(ax, (dgx0 + dgx1) / 2, 2.3, "Selective decision gate", FS_BOX, INK, "bold")
    txt(ax, (dgx0 + dgx1) / 2, 1.95, "score + margin rules", FS_ILL, SUB)
    txt(ax, (dgx0 + dgx1) / 2, 1.6, "$\\theta_{\\mathrm{margin}}=0.02$", FS_BOX, ORANGE, "bold")

    # node -> outputs
    node2 = (8.95, 1.95)
    arrow(ax, s3x1 + 0.02, 1.95, node2[0] - 0.04, 1.95, lw=LW_MAIN, ms=17)
    ax.add_patch(Circle(node2, 0.08, facecolor=ARROW, edgecolor=ARROW, zorder=8))
    ox0, ox1 = 9.35, 12.95
    outs = [("Healthy", GREEN, 2.62, 3.32), (r"Disease  $D_{k^{*}}$", RED, 1.6, 2.3),
            ("Inconclusive —\nexpert referral", AMBER, 0.5, 1.36)]
    for label, ec, y0, y1 in outs:
        rbox(ax, ox0, y0, ox1, y1, PANEL, ec, lw=2.1, z=6)
        txt(ax, (ox0 + ox1) / 2, (y0 + y1) / 2, label, FS_OUT, ec, "bold", z=7)
        arrow(ax, node2[0] + 0.04, node2[1], ox0 - 0.02, (y0 + y1) / 2, lw=2.2, ms=15)

    for ext in ("png", "pdf", "svg"):
        dpi = 400 if ext == "png" else None
        fig.savefig(OUT / f"Fig1_tworow.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.22, facecolor="white")
    print("saved Fig1_tworow (png/pdf/svg)")
    plt.close(fig)


if __name__ == "__main__":
    make()
