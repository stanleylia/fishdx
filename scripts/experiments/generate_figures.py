"""Generate all figures and tables for the paper.

Reads experiment results from experiment_results/expNN/ directories and
produces publication-quality PDF figures (300 DPI).

Figures:
  1. fig_exp01_lambda.pdf       -- Dual Y-axis: X=lambda, left=norm_dev, right=cross_domain_dist
  2. fig_exp02_pareidolia.pdf   -- Grouped bar: 4 modes x F1 score
  3. fig_exp03_heatmap.pdf      -- Heatmap: X=tau_gate, Y=strategy, color=SCA
  4. fig_exp04_tornado.pdf      -- Horizontal bar: parameters sorted by sensitivity_index
  5. fig_exp04_grid.pdf         -- 3x3 grid of parameter sweep curves
  6. fig_exp05_progressive.pdf  -- Stacked DA progression C0->C5
  7. fig_exp06_baseline.pdf     -- Bar chart with error bars: 5 methods x DA
  8. fig_exp07_degradation.pdf  -- Line: X=fallback level, Y=DA with threshold line
  9. fig_exp10_pareto.pdf       -- Scatter: X=latency, Y=DA with Pareto frontier
 10. fig_exp08_confusion.pdf    -- 2x3 confusion matrix grid
 11. fig_exp12_verification.pdf -- Grouped bar: C-NoVerify vs C-Verify (DA, HallucinRate)
 12. fig_exp13_iqdr.pdf         -- Grouped bar: HR/MRD/LRD quality grades (DA, CSR, RRP@5)

All figures gracefully skip if input data is unavailable.
Saves to experiment_results/figures/ as PDF.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.experiment_harness import RESULTS_DIR  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FIGURES_DIR = RESULTS_DIR / "figures"

# Publication style settings
STYLE_CONFIG = {
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

COLORS = {
    "primary": "#2196F3",
    "secondary": "#FF9800",
    "accent": "#4CAF50",
    "danger": "#F44336",
    "purple": "#9C27B0",
    "teal": "#009688",
    "grey": "#607D8B",
    "healthy": "#4CAF50",
    "disease": "#F44336",
    "inconclusive": "#FF9800",
}

PALETTE = [
    COLORS["primary"], COLORS["secondary"], COLORS["accent"],
    COLORS["danger"], COLORS["purple"], COLORS["teal"],
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_latest_result(exp_id: str) -> dict[str, Any] | None:
    """Load the most recent result JSON from an experiment directory.

    Args:
        exp_id: Experiment ID, e.g. "exp01".

    Returns:
        Parsed JSON dict, or None if no results found.
    """
    exp_dir = RESULTS_DIR / exp_id
    if not exp_dir.exists():
        log.warning("No results directory for %s", exp_id)
        return None

    json_files = sorted(exp_dir.glob(f"{exp_id}_*.json"), reverse=True)
    if not json_files:
        log.warning("No result files in %s", exp_dir)
        return None

    latest = json_files[0]
    log.info("Loading %s", latest)
    with open(latest) as f:
        return json.load(f)


def _save_figure(fig: Any, filename: str) -> Path:
    """Save a matplotlib figure as PDF.

    Args:
        fig: matplotlib Figure object.
        filename: Output filename (e.g., "fig_exp01_lambda.pdf").

    Returns:
        Path to saved file.
    """
    filepath = FIGURES_DIR / filename
    fig.savefig(filepath, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure: %s", filepath)
    return filepath


def _apply_style() -> None:
    """Apply publication-quality style settings."""
    if HAS_MATPLOTLIB:
        plt.rcParams.update(STYLE_CONFIG)
    if HAS_SEABORN:
        sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# Figure 1: EXP-01 Lambda Ablation (Dual Y-axis)
# ---------------------------------------------------------------------------
def generate_fig_exp01(dry_run: bool = False) -> Path | None:
    """Generate fig_exp01_lambda.pdf: Dual Y-axis line plot.

    X = lambda, left Y = norm_deviation, right Y = cross_domain_distance.
    """
    data = _load_latest_result("exp01")
    if not data:
        return None

    # Data lives in "summary" (list of dicts with nested stat objects)
    sweep = data.get("summary", data.get("sweep_results", []))
    if not sweep:
        log.warning("No summary/sweep_results in exp01 data")
        return None

    lambdas = [s["lambda"] for s in sweep]
    # norm_deviation and cross_domain_distance are stat dicts with "mean" key
    norm_devs = []
    cross_dists = []
    for s in sweep:
        nd = s.get("norm_deviation", {})
        cd = s.get("cross_domain_distance", {})
        if isinstance(nd, dict):
            norm_devs.append(nd.get("mean", 0.0))
        else:
            norm_devs.append(float(nd))
        if isinstance(cd, dict):
            cross_dists.append(cd.get("mean", 0.0))
        else:
            cross_dists.append(float(cd))

    fig, ax1 = plt.subplots(figsize=(6, 4))

    color1 = COLORS["primary"]
    color2 = COLORS["secondary"]

    ax1.set_xlabel(r"$\lambda$ (Visual Weight)")
    ax1.set_ylabel("Norm Deviation", color=color1)
    line1 = ax1.plot(lambdas, norm_devs, "o-", color=color1, linewidth=2,
                     markersize=6, label="Norm Deviation")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Cross-Domain Distance", color=color2)
    line2 = ax2.plot(lambdas, cross_dists, "s--", color=color2, linewidth=2,
                     markersize=6, label="Cross-Domain Dist.")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Mark optimal lambda
    optimal_lambda = data.get("optimal_lambda", 0.7)
    ax1.axvline(x=optimal_lambda, color=COLORS["accent"], linestyle=":",
                alpha=0.7, label=f"$\\lambda^*$={optimal_lambda}")

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center", framealpha=0.9)

    ax1.set_title(r"EXP-01: $\lambda$-Weighted Fusion Ablation")
    fig.tight_layout()

    if dry_run:
        plt.close(fig)
        return None
    return _save_figure(fig, "fig_exp01_lambda.pdf")


# ---------------------------------------------------------------------------
# Figure 2: EXP-02 Pareidolia Detection (Grouped Bar)
# ---------------------------------------------------------------------------
def generate_fig_exp02(dry_run: bool = False) -> Path | None:
    """Generate fig_exp02_pareidolia.pdf: Grouped bar chart of 4 modes x F1."""
    data = _load_latest_result("exp02")
    if not data:
        return None

    mode_results = data.get("mode_results", {})
    if not mode_results:
        log.warning("No mode_results in exp02 data")
        return None

    # mode_results can be a dict {mode_name: {psr, fpr, precision, recall, f1, ...}}
    # or a list of dicts with "mode" key
    if isinstance(mode_results, list):
        modes = [mr.get("mode", f"mode_{i}") for i, mr in enumerate(mode_results)]
        mr_list = mode_results
    else:
        modes = list(mode_results.keys())
        mr_list = [mode_results[m] for m in modes]

    f1_scores = []
    precision_scores = []
    recall_scores = []

    for mr in mr_list:
        # metrics may be nested or flat; values can be floats or stat dicts
        metrics = mr.get("metrics", mr)
        for key, target in [("f1", f1_scores), ("precision", precision_scores),
                            ("recall", recall_scores)]:
            val = metrics.get(key, 0.0)
            if isinstance(val, dict):
                val = val.get("mean", 0.0)
            target.append(float(val))

    x = np.arange(len(modes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width, precision_scores, width, label="Precision",
                   color=COLORS["primary"], alpha=0.85)
    bars2 = ax.bar(x, recall_scores, width, label="Recall",
                   color=COLORS["secondary"], alpha=0.85)
    bars3 = ax.bar(x + width, f1_scores, width, label="F1",
                   color=COLORS["accent"], alpha=0.85)

    ax.set_xlabel("Detection Mode")
    ax.set_ylabel("Score")
    ax.set_title("EXP-02: Pareidolia Detection — Mode Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{height:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7)

    fig.tight_layout()

    if dry_run:
        plt.close(fig)
        return None
    return _save_figure(fig, "fig_exp02_pareidolia.pdf")


# ---------------------------------------------------------------------------
# Figure 3: EXP-03 Semantic Filter Heatmap
# ---------------------------------------------------------------------------
def generate_fig_exp03(dry_run: bool = False) -> Path | None:
    """Generate fig_exp03_heatmap.pdf: Heatmap of tau_gate x strategy -> SCA."""
    data = _load_latest_result("exp03")
    if not data:
        return None

    # EXP-03 stores "strategy_results" (list) and "adaptive_sweep" (list)
    strategy_results = data.get("strategy_results", [])
    adaptive_sweep = data.get("adaptive_sweep", [])
    grid = data.get("grid_results", [])

    # Merge all available data points
    all_points = strategy_results + adaptive_sweep + grid
    if not all_points:
        log.warning("No strategy_results/adaptive_sweep/grid_results in exp03 data")
        return None

    # Extract unique tau_gate values and strategies
    tau_values_raw: list[float | None] = []
    strategies_raw: list[str] = []
    for g in all_points:
        tv = g.get("tau_gate", g.get("gate_threshold"))
        s = g.get("strategy", "adaptive")
        if tv is not None:
            tau_values_raw.append(float(tv))
        strategies_raw.append(str(s))

    tau_values = sorted(set(t for t in tau_values_raw if t is not None))
    strategies = sorted(set(strategies_raw))

    if len(tau_values) < 2 or len(strategies) < 2:
        # Not enough grid data; fallback to bar chart of all data points
        sca_values = [g.get("sca", 0.0) for g in all_points]
        labels = []
        for g in all_points:
            strat = g.get("strategy", "")
            tau = g.get("tau_gate", g.get("gate_threshold", ""))
            label = f"{strat}" if strat else f"tau={tau}"
            if tau and strat:
                label = f"{strat[:12]} t={tau}"
            labels.append(label)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(sca_values)), sca_values, color=COLORS["primary"])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("SCA")
        ax.set_title("EXP-03: Semantic Filter — SCA by Configuration")
        ax.set_ylim(0, 1.1)
        fig.tight_layout()

        if dry_run:
            plt.close(fig)
            return None
        return _save_figure(fig, "fig_exp03_heatmap.pdf")

    # Build matrix
    sca_matrix = np.zeros((len(strategies), len(tau_values)))
    tau_idx = {t: i for i, t in enumerate(tau_values)}
    strat_idx = {s: i for i, s in enumerate(strategies)}

    for g in grid:
        t = g.get("tau_gate", g.get("gate_threshold", 0.0))
        s = g.get("strategy", g.get("rag_trigger_strategy", "default"))
        sca = g.get("sca", 0.0)
        if t in tau_idx and s in strat_idx:
            sca_matrix[strat_idx[s], tau_idx[t]] = sca

    fig, ax = plt.subplots(figsize=(7, 4))

    if HAS_SEABORN:
        sns.heatmap(sca_matrix, annot=True, fmt=".3f",
                    xticklabels=[f"{t:.2f}" for t in tau_values],
                    yticklabels=strategies,
                    cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
    else:
        im = ax.imshow(sca_matrix, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(tau_values)))
        ax.set_xticklabels([f"{t:.2f}" for t in tau_values])
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(strategies)
        plt.colorbar(im, ax=ax)
        for i in range(len(strategies)):
            for j in range(len(tau_values)):
                ax.text(j, i, f"{sca_matrix[i, j]:.3f}",
                        ha="center", va="center", fontsize=8)

    ax.set_xlabel(r"$\tau_{gate}$")
    ax.set_ylabel("Strategy")
    ax.set_title("EXP-03: Semantic Filter — SCA Heatmap")
    fig.tight_layout()

    if dry_run:
        plt.close(fig)
        return None
    return _save_figure(fig, "fig_exp03_heatmap.pdf")


# ---------------------------------------------------------------------------
# Figure 4a: EXP-04 Tornado (Sensitivity Analysis)
# ---------------------------------------------------------------------------
def generate_fig_exp04_tornado(dry_run: bool = False) -> Path | None:
    """Generate fig_exp04_tornado.pdf: Horizontal bar sorted by sensitivity_index."""
    data = _load_latest_result("exp04")
    if not data:
        return None

    param_results = data.get("parameter_results", data.get("sensitivity", []))
    # Also try tornado_chart as alternate source
    tornado = data.get("tornado_chart", [])

    if not param_results and not tornado:
        log.warning("No parameter_results in exp04 data")
        return None

    params: list[str] = []
    sensitivities: list[float] = []

    if isinstance(param_results, list):
        for pr in param_results:
            name = pr.get("name", pr.get("symbol", "?"))
            si = pr.get("sensitivity_index", pr.get("da_range", 0.0))
            params.append(name)
            sensitivities.append(float(si))
    elif isinstance(param_results, dict):
        for param_name, pr in param_results.items():
            si = pr.get("sensitivity_index", pr.get("da_range", 0.0))
            params.append(param_name)
            sensitivities.append(float(si))
    elif tornado:
        for t in tornado:
            params.append(t.get("name", t.get("parameter", "?")))
            sensitivities.append(float(t.get("sensitivity_index", t.get("range", 0.0))))

    # Sort by sensitivity
    sorted_pairs = sorted(zip(params, sensitivities), key=lambda p: p[1])
    params = [p[0] for p in sorted_pairs]
    sensitivities = [p[1] for p in sorted_pairs]

    fig, ax = plt.subplots(figsize=(7, max(3, len(params) * 0.5)))

    colors = [COLORS["danger"] if s > 0.1 else COLORS["primary"] for s in sensitivities]
    ax.barh(params, sensitivities, color=colors, alpha=0.85)
    ax.set_xlabel("Sensitivity Index (DA Range)")
    ax.set_title("EXP-04: Parameter Sensitivity — Tornado Diagram")
    ax.axvline(x=0.1, color=COLORS["grey"], linestyle="--", alpha=0.5, label="Threshold (0.1)")
    ax.legend(loc="lower right")

    for i, (p, s) in enumerate(zip(params, sensitivities)):
        ax.text(s + 0.005, i, f"{s:.3f}", va="center", fontsize=8)

    fig.tight_layout()

    if dry_run:
        plt.close(fig)
        return None
    return _save_figure(fig, "fig_exp04_tornado.pdf")


# ---------------------------------------------------------------------------
# Figure 4b: EXP-04 Grid (Parameter Sweep Curves)
# ---------------------------------------------------------------------------
def generate_fig_exp04_grid(dry_run: bool = False) -> Path | None:
    """Generate fig_exp04_grid.pdf: 3x3 grid of parameter sweep curves."""
    data = _load_latest_result("exp04")
    if not data:
        return None

    param_results_raw = data.get("parameter_results", [])
    if not param_results_raw:
        return None

    # Normalize to list of dicts with "name" and "sweep_points" keys
    if isinstance(param_results_raw, dict):
        param_list = [
            {"name": k, **v} for k, v in param_results_raw.items()
        ]
    else:
        param_list = param_results_raw

    n_params = len(param_list)
    if n_params == 0:
        return None

    # Determine grid size
    ncols = min(3, n_params)
    nrows = (n_params + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx, pr in enumerate(param_list):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        param_name = pr.get("name", pr.get("symbol", f"param_{idx}"))
        sweep = pr.get("sweep_points", pr.get("sweep", pr.get("values", [])))

        if not sweep:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(param_name, fontsize=9)
            continue

        x_vals = []
        y_vals = []
        for i, s in enumerate(sweep):
            xv = s.get("value", s.get("param_value", None))
            yv = s.get("da", None)
            if xv is not None and yv is not None:
                x_vals.append(float(xv))
                y_vals.append(float(yv))

        if not x_vals:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(param_name, fontsize=9)
            continue

        ax.plot(x_vals, y_vals, "o-", color=COLORS["primary"], linewidth=1.5, markersize=4)
        ax.set_title(param_name, fontsize=9)
        ax.set_ylabel("DA", fontsize=8)
        ax.set_ylim(0, 1.05)

        # Mark optimal
        if y_vals:
            best_idx = int(np.argmax(y_vals))
            ax.axvline(x=x_vals[best_idx], color=COLORS["accent"], linestyle=":",
                       alpha=0.5)

    # Hide unused subplots
    for idx in range(n_params, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    fig.suptitle("EXP-04: Parameter Sweep Curves", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if dry_run:
        plt.close(fig)
        return None
    return _save_figure(fig, "fig_exp04_grid.pdf")


# ---------------------------------------------------------------------------
# Figure 5: EXP-05 Progressive Assembly (Stacked DA Progression)
# ---------------------------------------------------------------------------
def generate_fig_exp05(dry_run: bool = False) -> Path | None:
    """Generate fig_exp05_progressive.pdf: Stacked DA progression C0->C5."""
    data = _load_latest_result("exp05")
    if not data:
        return None

    progression = data.get("da_progression", [])
    if not progression:
        # Fallback to summary
        summary = data.get("summary", [])
        if summary:
            progression = [{"config": s["config"], "da": s["da"],
                            "delta_da": 0.0} for s in summary]

    if not progression:
        return None

    configs = [p["config"] for p in progression]
    da_values = [p["da"] for p in progression]
    deltas = [p.get("delta_da", 0.0) for p in progression]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Bars for DA
    bars = ax.bar(configs, da_values, color=PALETTE[:len(configs)], alpha=0.85,
                  edgecolor="white", linewidth=0.5)

    # Add delta labels
    for i, (bar, delta) in enumerate(zip(bars, deltas)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                f"{height:.3f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")
        if i > 0 and delta != 0:
            sign = "+" if delta > 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.04,
                    f"({sign}{delta:.3f})", ha="center", va="bottom",
                    fontsize=7, color=COLORS["accent"] if delta > 0 else COLORS["danger"])

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Diagnostic Accuracy (DA)")
    ax.set_title("EXP-05: Progressive Component Assembly")
    ax.set_ylim(0, 1.15)

    # Add reference line at DA=1.0
    ax.axhline(y=1.0, color=COLORS["grey"], linestyle="--", alpha=0.3)

    fig.tight_layout()

    if dry_run:
        plt.close(fig)
        return None
    return _save_figure(fig, "fig_exp05_progressive.pdf")


# ---------------------------------------------------------------------------
# Figure 6: EXP-06 Baseline Comparison (Bar Chart with Error Bars)
# ---------------------------------------------------------------------------
def generate_fig_exp06(dry_run: bool = False) -> Path | None:
    """Generate fig_exp06_baseline.pdf: Bar chart with method DA + macro-F1."""
    data = _load_latest_result("exp06")
    if not data:
        return None

    summary = data.get("summary", [])
    if not summary:
        return None

    methods = [s["method"] for s in summary]
    da_values = [s["da"] for s in summary]
    f1_values = [s.get("macro_f1", 0.0) for s in summary]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - width / 2, da_values, width, label="DA",
                   color=COLORS["primary"], alpha=0.85)
    bars2 = ax.bar(x + width / 2, f1_values, width, label="Macro-F1",
                   color=COLORS["secondary"], alpha=0.85)

    ax.set_xlabel("Method")
    ax.set_ylabel("Score")
    ax.set_title("EXP-06: Baseline System Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1.15)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{height:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7)

    fig.tight_layout()

    if dry_run:
        plt.close(fig)
        return None
    return _save_figure(fig, "fig_exp06_baseline.pdf")


# ---------------------------------------------------------------------------
# Figure 7: EXP-07 Fallback Degradation (Line Chart)
# ---------------------------------------------------------------------------
def generate_fig_exp07(dry_run: bool = False) -> Path | None:
    """Generate fig_exp07_degradation.pdf: Line plot of DA across fallback levels."""
    data = _load_latest_result("exp07")
    if not data:
        return None

    summary = data.get("summary", [])
    if not summary:
        return None

    levels = [s["level"] for s in summary]
    da_values = [s["da"] for s in summary]
    confidence_means = [s.get("confidence_mean", 0.0) for s in summary]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    # DA line
    ax1.plot(levels, da_values, "o-", color=COLORS["primary"], linewidth=2.5,
             markersize=8, label="DA", zorder=5)
    ax1.fill_between(range(len(levels)), da_values, alpha=0.1, color=COLORS["primary"])
    ax1.set_xlabel("Fallback Level")
    ax1.set_ylabel("Diagnostic Accuracy (DA)")
    ax1.set_ylim(0, 1.1)

    # Threshold line at DA=0.7
    ax1.axhline(y=0.7, color=COLORS["danger"], linestyle="--", alpha=0.5,
                label="Acceptable Threshold (0.7)")

    # Confidence on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(levels, confidence_means, "s--", color=COLORS["secondary"],
             linewidth=1.5, markersize=6, alpha=0.7, label="Confidence (mean)")
    ax2.set_ylabel("Confidence", color=COLORS["secondary"])
    ax2.tick_params(axis="y", labelcolor=COLORS["secondary"])
    ax2.set_ylim(0, 1.1)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", framealpha=0.9)

    ax1.set_title("EXP-07: LLM Fallback Chain Degradation")

    # Annotate DA values
    for i, (level, da) in enumerate(zip(levels, da_values)):
        ax1.annotate(f"{da:.2f}", (i, da), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8, fontweight="bold")

    fig.tight_layout()

    if dry_run:
        plt.close(fig)
        return None
    return _save_figure(fig, "fig_exp07_degradation.pdf")


# ---------------------------------------------------------------------------
# Figure 8: EXP-10 Pareto Frontier (Scatter Plot)
# ---------------------------------------------------------------------------
def generate_fig_exp10(dry_run: bool = False) -> Path | None:
    """Generate fig_exp10_pareto.pdf: Scatter plot with Pareto frontier."""
    data = _load_latest_result("exp10")
    if not data:
        return None

    summary = data.get("summary", [])
    pareto_ids = data.get("pareto_frontier", [])
    if not summary:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot all points
    for point in summary:
        is_pareto = point.get("is_pareto", point["config"] in pareto_ids)
        color = COLORS["accent"] if is_pareto else COLORS["grey"]
        marker = "*" if is_pareto else "o"
        size = 150 if is_pareto else 60

        ax.scatter(point["latency"], point["da"], c=color, s=size,
                   marker=marker, edgecolors="black", linewidths=0.5,
                   zorder=5 if is_pareto else 3)
        ax.annotate(point["config"],
                    (point["latency"], point["da"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, fontweight="bold" if is_pareto else "normal")

    # Draw Pareto frontier line
    pareto_points = sorted(
        [p for p in summary if p.get("is_pareto", p["config"] in pareto_ids)],
        key=lambda p: p["latency"],
    )
    if len(pareto_points) >= 2:
        px = [p["latency"] for p in pareto_points]
        py = [p["da"] for p in pareto_points]
        ax.plot(px, py, "--", color=COLORS["accent"], alpha=0.5, linewidth=1.5,
                label="Pareto Frontier")

    # Color-coded source regions
    c_points = [p for p in summary if p["config"].startswith("C")]
    l_points = [p for p in summary if p["config"].startswith("L")]

    if c_points and l_points:
        legend_handles = [
            mpatches.Patch(color=COLORS["primary"], alpha=0.2, label="EXP-05 (C configs)"),
            mpatches.Patch(color=COLORS["secondary"], alpha=0.2, label="EXP-07 (L configs)"),
            plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=COLORS["accent"],
                       markersize=12, label="Pareto Optimal"),
        ]
        ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

    ax.set_xlabel("Estimated Latency (seconds)")
    ax.set_ylabel("Diagnostic Accuracy (DA)")
    ax.set_title("EXP-10: Latency-Accuracy Pareto Analysis")
    ax.set_ylim(0, 1.1)

    fig.tight_layout()

    if dry_run:
        plt.close(fig)
        return None
    return _save_figure(fig, "fig_exp10_pareto.pdf")


# ---------------------------------------------------------------------------
# Figure 9: EXP-08 Confusion Matrices (2x3 Grid)
# ---------------------------------------------------------------------------
def generate_fig_exp08(dry_run: bool = False) -> Path | None:
    """Generate fig_exp08_confusion.pdf: 2x3 confusion matrix grid."""
    data = _load_latest_result("exp08")
    if not data:
        return None

    scenario_results = data.get("scenario_results", {})
    if not scenario_results:
        return None

    classes = ["Healthy", "Disease", "Inconclusive"]
    scenarios = list(scenario_results.items())
    n = len(scenarios)

    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (sid, sr) in enumerate(scenarios):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        cm = sr.get("confusion_matrix", {})
        matrix = np.zeros((len(classes), len(classes)))
        for i, actual in enumerate(classes):
            for j, pred in enumerate(classes):
                matrix[i, j] = cm.get(actual, {}).get(pred, 0)

        if HAS_SEABORN:
            sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues",
                        xticklabels=["H", "D", "I"],
                        yticklabels=["H", "D", "I"],
                        ax=ax, cbar=False, square=True)
        else:
            im = ax.imshow(matrix, cmap="Blues", aspect="equal")
            ax.set_xticks(range(3))
            ax.set_xticklabels(["H", "D", "I"])
            ax.set_yticks(range(3))
            ax.set_yticklabels(["H", "D", "I"])
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{int(matrix[i, j])}",
                            ha="center", va="center", fontsize=10)

        short_name = sr.get("scenario_name", sid)
        if len(short_name) > 25:
            short_name = short_name[:22] + "..."
        da = sr.get("da", 0.0)
        ax.set_title(f"{short_name}\nDA={da:.2f}", fontsize=9)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    fig.suptitle("EXP-08: Cross-Domain Confusion Matrices", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if dry_run:
        plt.close(fig)
        return None
    return _save_figure(fig, "fig_exp08_confusion.pdf")


# ---------------------------------------------------------------------------
# Figure 10: EXP-05 Radar Chart (5 Dimensions per Config)
# ---------------------------------------------------------------------------
def generate_fig_exp05_radar(dry_run: bool = False) -> Path | None:
    """Generate fig_exp05_radar.pdf: Radar/spider chart with 5 dimensions.

    Dimensions: DA, SCA, PSR, RAG_Trigger, Speed.
    Overlays polygons for C0-C5 configurations.
    """
    data = _load_latest_result("exp05")
    if not data:
        return None

    # Try radar_data key first, fallback to computing from summary
    radar_data = data.get("radar_data")
    if not radar_data:
        summary = data.get("summary", [])
        if not summary:
            log.warning("No radar_data or summary in exp05 data")
            return None
        # Build radar_data from summary
        radar_data = []
        for s in summary:
            entry = {
                "config": s.get("config", ""),
                "DA": s.get("da", 0.0),
                "SCA": s.get("sca", 0.0),
                "PSR": s.get("psr", 0.0),
                "RAG_Trigger": s.get("rag_trigger", s.get("rag_trigger_rate", 0.0)),
                "Speed": 1.0 - min(s.get("latency", 0.0), 40.0) / 40.0,
            }
            radar_data.append(entry)

    if not radar_data:
        return None

    dimensions = ["DA", "SCA", "PSR", "RAG_Trigger", "Speed"]
    n_dims = len(dimensions)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    for i, entry in enumerate(radar_data):
        config_name = entry.get("config", f"C{i}")
        values = []
        for dim in dimensions:
            val = entry.get(dim, 0.0)
            # DA and SCA are already 0-1; normalize Speed if raw latency
            values.append(float(val))
        values += values[:1]  # Close the polygon

        color = PALETTE[i % len(PALETTE)]
        ax.plot(angles, values, "o-", linewidth=2, markersize=5,
                color=color, label=config_name)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("EXP-05: Configuration Radar Chart", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    fig.tight_layout()

    if dry_run:
        plt.close(fig)
        return None
    return _save_figure(fig, "fig_exp05_radar.pdf")


# ---------------------------------------------------------------------------
# Figure 11: EXP-07 Confidence Box Plot
# ---------------------------------------------------------------------------
def generate_fig_exp07_boxplot(dry_run: bool = False) -> Path | None:
    """Generate fig_exp07_boxplot.pdf: Box plot of confidence distributions per level.

    X-axis: L0-L5, Y-axis: confidence score.
    """
    data = _load_latest_result("exp07")
    if not data:
        return None

    # Try confidence_distributions first, then build from level_results
    conf_dists = data.get("confidence_distributions")
    if not conf_dists:
        level_results = data.get("level_results", {})
        if not level_results:
            # Fallback: try summary with per_scene data
            summary = data.get("summary", [])
            if not summary:
                log.warning("No confidence data in exp07")
                return None
            # Build from summary if per_scene exists
            conf_dists = {}
            for s in summary:
                level_key = f"L{s.get('level', 0)}"
                scenes = s.get("per_scene", [])
                if scenes:
                    conf_dists[level_key] = [
                        sc.get("confidence", 0.0) for sc in scenes
                    ]
        else:
            conf_dists = {}
            for level_key, lr in level_results.items():
                scenes = lr.get("per_scene", lr.get("scenes", []))
                if isinstance(scenes, list):
                    conf_dists[level_key] = [
                        sc.get("confidence", 0.0) for sc in scenes
                        if isinstance(sc, dict)
                    ]

    if not conf_dists:
        log.warning("Could not build confidence distributions for exp07")
        return None

    # Sort levels naturally (L0, L1, ..., L5)
    sorted_levels = sorted(conf_dists.keys(), key=lambda k: int(k.replace("L", "")))
    box_data = [conf_dists[lv] for lv in sorted_levels]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bp = ax.boxplot(box_data, labels=sorted_levels, patch_artist=True,
                    showmeans=True, meanprops={"marker": "D", "markerfacecolor": "white",
                                               "markersize": 5})

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.7)

    ax.set_xlabel("Fallback Level")
    ax.set_ylabel("Confidence Score")
    ax.set_title("EXP-07: Confidence Score Distribution per Fallback Level")
    ax.set_ylim(0, 1.1)

    fig.tight_layout()

    if dry_run:
        plt.close(fig)
        return None
    return _save_figure(fig, "fig_exp07_boxplot.pdf")


# ---------------------------------------------------------------------------
# Figure 12: EXP-08 Domain Gap (Grouped Bar)
# ---------------------------------------------------------------------------
def generate_fig_exp08_domain_gap(dry_run: bool = False) -> Path | None:
    """Generate fig_exp08_domain_gap.pdf: Grouped bar comparing within vs cross-domain DA.

    Blue bars = within-domain (T1, T2), Orange bars = cross-domain (T3, T4, T5).
    """
    data = _load_latest_result("exp08")
    if not data:
        return None

    summary = data.get("summary", [])
    if not summary:
        log.warning("No summary in exp08 data")
        return None

    # Classify scenarios as within-domain or cross-domain
    within_ids = {"T1", "T2"}
    labels = []
    da_values = []
    bar_colors = []

    for s in summary:
        scenario = s.get("scenario", s.get("scenario_id", ""))
        da = s.get("da", 0.0)
        labels.append(str(scenario))
        da_values.append(float(da))
        # Determine color based on scenario ID
        scenario_upper = str(scenario).upper()
        if any(wid in scenario_upper for wid in within_ids):
            bar_colors.append(COLORS["primary"])  # Blue = within-domain
        else:
            bar_colors.append(COLORS["secondary"])  # Orange = cross-domain

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(labels))
    bars = ax.bar(x, da_values, color=bar_colors, alpha=0.85, edgecolor="white",
                  linewidth=0.5)

    # Mean DA dashed line
    mean_da = float(np.mean(da_values))
    ax.axhline(y=mean_da, color=COLORS["grey"], linestyle="--", alpha=0.7,
               label=f"Mean DA = {mean_da:.3f}")

    # Value labels
    for bar, da in zip(bars, da_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                f"{da:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Diagnostic Accuracy (DA)")
    ax.set_title("EXP-08: Within-Domain vs Cross-Domain Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15)

    # Legend for within/cross
    legend_handles = [
        mpatches.Patch(color=COLORS["primary"], alpha=0.85, label="Within-domain (T1, T2)"),
        mpatches.Patch(color=COLORS["secondary"], alpha=0.85, label="Cross-domain (T3, T4, T5)"),
        plt.Line2D([0], [0], color=COLORS["grey"], linestyle="--",
                   label=f"Mean DA = {mean_da:.3f}"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9)

    fig.tight_layout()

    if dry_run:
        plt.close(fig)
        return None
    return _save_figure(fig, "fig_exp08_domain_gap.pdf")


# ---------------------------------------------------------------------------
# Figure 13: EXP-09 Indicator Breakdown (Stacked Bar)
# ---------------------------------------------------------------------------
def generate_fig_exp09_indicators(dry_run: bool = False) -> Path | None:
    """Generate fig_exp09_indicators.pdf: Stacked bar of indicator breakdown.

    X-axis: ZH-Full, ZH-NoGuide, EN-Full, EN-NoGuide.
    Stacked colors: healthy_hits (green), disease_hits (red), no_match (gray).
    """
    data = _load_latest_result("exp09")
    if not data:
        return None

    # Try indicator_breakdown key first
    breakdown = data.get("indicator_breakdown")
    if not breakdown:
        # Compute from config_results or summary
        config_results = data.get("config_results", data.get("summary", []))
        if not config_results:
            log.warning("No indicator_breakdown or config_results in exp09 data")
            return None
        breakdown = []
        for cr in config_results:
            config_name = cr.get("config", cr.get("config_name", ""))
            total_s_h = cr.get("total_s_h", cr.get("healthy_hits", 0))
            total_s_d = cr.get("total_s_d", cr.get("disease_hits", 0))
            no_match = cr.get("no_match", cr.get("unmatched", 0))
            breakdown.append({
                "config": config_name,
                "healthy_hits": int(total_s_h),
                "disease_hits": int(total_s_d),
                "no_match": int(no_match),
            })

    if not breakdown:
        return None

    configs = [b.get("config", f"cfg_{i}") for i, b in enumerate(breakdown)]
    healthy = [b.get("healthy_hits", 0) for b in breakdown]
    disease = [b.get("disease_hits", 0) for b in breakdown]
    no_match = [b.get("no_match", 0) for b in breakdown]

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(configs))
    width = 0.5

    ax.bar(x, healthy, width, label="Healthy Hits", color=COLORS["healthy"])
    ax.bar(x, disease, width, bottom=healthy, label="Disease Hits",
           color=COLORS["disease"])
    bottom_no = [h + d for h, d in zip(healthy, disease)]
    ax.bar(x, no_match, width, bottom=bottom_no, label="No Match",
           color=COLORS["grey"])

    ax.set_xlabel("Prompt Configuration")
    ax.set_ylabel("Indicator Count")
    ax.set_title("EXP-09: Scoring Indicator Breakdown by Prompt Config")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=15, ha="right", fontsize=9)
    ax.legend(loc="upper right")

    # Add total labels on top
    for i, (h, d, n) in enumerate(zip(healthy, disease, no_match)):
        total = h + d + n
        if total > 0:
            ax.text(i, total + 0.3, str(total), ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    fig.tight_layout()

    if dry_run:
        plt.close(fig)
        return None
    return _save_figure(fig, "fig_exp09_indicators.pdf")


# ---------------------------------------------------------------------------
# Figure 14: EXP-10 Stacked Latency Breakdown (Horizontal Bar)
# ---------------------------------------------------------------------------
def generate_fig_exp10_stacked_latency(dry_run: bool = False) -> Path | None:
    """Generate fig_exp10_stacked_latency.pdf: Stacked horizontal bar of latency components.

    Components: Florence-2 (blue), CLIP (green), Algorithms (yellow),
                ChromaDB (orange), LLM (red), Network (gray).
    """
    data = _load_latest_result("exp10")
    if not data:
        return None

    latency_breakdown = data.get("latency_breakdown", [])
    if not latency_breakdown:
        log.warning("No latency_breakdown in exp10 data")
        return None

    components_order = ["Florence-2", "CLIP", "Algorithms", "ChromaDB", "LLM", "Network"]
    component_colors = {
        "Florence-2": COLORS["primary"],
        "CLIP": COLORS["accent"],
        "Algorithms": "#FFC107",  # Yellow
        "ChromaDB": COLORS["secondary"],
        "LLM": COLORS["danger"],
        "Network": COLORS["grey"],
    }

    configs = [lb.get("config", f"cfg_{i}") for i, lb in enumerate(latency_breakdown)]
    # Build component arrays
    component_values: dict[str, list[float]] = {c: [] for c in components_order}
    for lb in latency_breakdown:
        components = lb.get("components", lb.get("latency_components", {}))
        for comp in components_order:
            # Try exact match, then lowercase, then snake_case
            val = components.get(comp)
            if val is None:
                val = components.get(comp.lower())
            if val is None:
                val = components.get(comp.lower().replace("-", "_"))
            if val is None:
                val = 0.0
            component_values[comp].append(float(val))

    fig, ax = plt.subplots(figsize=(9, max(3, len(configs) * 0.6)))

    y = np.arange(len(configs))
    left = np.zeros(len(configs))

    for comp in components_order:
        vals = np.array(component_values[comp])
        ax.barh(y, vals, left=left, label=comp,
                color=component_colors.get(comp, COLORS["grey"]), alpha=0.85,
                edgecolor="white", linewidth=0.5)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(configs, fontsize=9)
    ax.set_xlabel("Latency (seconds)")
    ax.set_title("EXP-10: Latency Component Breakdown per Configuration")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # Add total latency label at end of each bar
    for i, total in enumerate(left):
        ax.text(total + 0.2, i, f"{total:.1f}s", va="center", fontsize=8)

    fig.tight_layout()

    if dry_run:
        plt.close(fig)
        return None
    return _save_figure(fig, "fig_exp10_stacked_latency.pdf")


# ---------------------------------------------------------------------------
# Figure 15: EXP-10 Latency Pie Chart (Best Config)
# ---------------------------------------------------------------------------
def generate_fig_exp10_pie(dry_run: bool = False) -> Path | None:
    """Generate fig_exp10_pie.pdf: Pie chart of latency proportions for best config.

    Uses first Pareto point from config_points, or C5_L0 if available.
    """
    data = _load_latest_result("exp10")
    if not data:
        return None

    # Find best config's latency components
    raw_cp = data.get("config_points", data.get("summary", []))
    pareto_ids = data.get("pareto_frontier", [])

    # Normalize config_points to list of dicts
    if isinstance(raw_cp, dict):
        config_points = list(raw_cp.values())
    else:
        config_points = raw_cp

    target_point = None

    # Try first Pareto point
    if pareto_ids and config_points:
        for cp in config_points:
            cfg = cp.get("config", cp.get("id", ""))
            if cfg in pareto_ids or cfg == pareto_ids[0]:
                target_point = cp
                break

    # Fallback: look for C5_L0 or C5
    if not target_point and config_points:
        for cp in config_points:
            cfg = cp.get("config", cp.get("id", ""))
            if cfg in ("C5_L0", "C5"):
                target_point = cp
                break

    # Fallback: last config point (most complete pipeline)
    if not target_point and config_points:
        target_point = config_points[-1]

    if not target_point:
        log.warning("No config point found for pie chart in exp10")
        return None

    components = target_point.get("latency_components", target_point.get("components", {}))
    if not components:
        log.warning("No latency_components in target config point for exp10 pie")
        return None

    labels = []
    sizes = []
    colors = []
    component_colors = {
        "florence_2": COLORS["primary"],
        "florence-2": COLORS["primary"],
        "clip": COLORS["accent"],
        "algorithms": "#FFC107",
        "chromadb": COLORS["secondary"],
        "llm": COLORS["danger"],
        "network": COLORS["grey"],
    }

    for comp_name, comp_val in components.items():
        val = float(comp_val)
        if val > 0:
            labels.append(comp_name)
            sizes.append(val)
            colors.append(component_colors.get(comp_name.lower().replace("-", "_"),
                                               COLORS["grey"]))

    if not sizes:
        return None

    config_name = target_point.get("config", "Best Config")

    fig, ax = plt.subplots(figsize=(7, 6))

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.8,
        textprops={"fontsize": 9},
    )
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_fontweight("bold")

    ax.set_title(f"EXP-10: Latency Proportion — {config_name}")

    fig.tight_layout()

    if dry_run:
        plt.close(fig)
        return None
    return _save_figure(fig, "fig_exp10_pie.pdf")


# ---------------------------------------------------------------------------
# Figure 16: EXP-12 Verification Isolation (Grouped Bar)
# ---------------------------------------------------------------------------
def generate_fig_exp12(dry_run: bool = False) -> Path | None:
    """Generate fig_exp12_verification.pdf: Grouped bar of C-NoVerify vs C-Verify.

    Metrics: DA (green), Hallucination Rate (red), Penalty Count (orange).
    """
    data = _load_latest_result("exp12")
    if not data:
        return None

    summary = data.get("summary", [])
    if not summary:
        return None

    configs = [s["config"] for s in summary]
    da_values = [s.get("da", 0.0) for s in summary]
    halluc_rates = [s.get("hallucination_rate", 0.0) for s in summary]
    penalties = [s.get("penalty_count", 0.0) for s in summary]

    x = np.arange(len(configs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width, da_values, width, label="DA",
                   color=COLORS["accent"], alpha=0.85)
    bars2 = ax.bar(x, halluc_rates, width, label="Hallucination Rate",
                   color=COLORS["danger"], alpha=0.85)
    bars3 = ax.bar(x + width, penalties, width, label="Penalty Count",
                   color=COLORS["secondary"], alpha=0.85)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Score")
    ax.set_title("EXP-12: Verification Loop Ablation")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.legend()
    ax.set_ylim(0, 1.15)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{height:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

    # Add statistical annotation
    stats = data.get("statistical_tests", {})
    delta_da = stats.get("cliffs_delta_da", {})
    if delta_da:
        delta_val = delta_da.get("delta", 0.0)
        interp = delta_da.get("interpretation", "")
        ax.text(0.98, 0.02, f"Cliff's $\\delta$(DA) = {delta_val:.2f} ({interp})",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                style="italic", color=COLORS["grey"])

    fig.tight_layout()

    if dry_run:
        plt.close(fig)
        return None
    return _save_figure(fig, "fig_exp12_verification.pdf")


# ---------------------------------------------------------------------------
# Figure 17: EXP-13 IQDR (Grouped Bar)
# ---------------------------------------------------------------------------
def generate_fig_exp13(dry_run: bool = False) -> Path | None:
    """Generate fig_exp13_iqdr.pdf: Grouped bar of HR/MRD/LRD quality grades.

    Metrics: DA (blue), CSR normalized (green), RRP@5 (orange).
    """
    data = _load_latest_result("exp13")
    if not data:
        return None

    summary = data.get("summary", [])
    if not summary:
        return None

    grades = [s["grade"] for s in summary]
    da_values = [s.get("da_mean", 0.0) for s in summary]
    # Normalize CSR to 0-1 scale (max observed CSR / max possible)
    csr_raw = [s.get("csr_mean", 0.0) for s in summary]
    max_csr = max(csr_raw) if csr_raw and max(csr_raw) > 0 else 1.0
    csr_norm = [c / max_csr for c in csr_raw]
    rrp5_values = [s.get("rrp5_mean", 0.0) for s in summary]

    x = np.arange(len(grades))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar(x - width, da_values, width, label="DA",
                   color=COLORS["primary"], alpha=0.85)
    bars2 = ax.bar(x, csr_norm, width, label="CSR (normalized)",
                   color=COLORS["accent"], alpha=0.85)
    bars3 = ax.bar(x + width, rrp5_values, width, label="RRP@5",
                   color=COLORS["secondary"], alpha=0.85)

    ax.set_xlabel("Image Quality Grade")
    ax.set_ylabel("Score")
    ax.set_title("EXP-13: Image Quality Degradation Robustness (IQDR)")
    ax.set_xticks(x)
    ax.set_xticklabels(grades, fontsize=10)
    ax.legend()
    ax.set_ylim(0, 1.15)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{height:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

    # IQDR annotation
    iqdr = data.get("iqdr", 0.0)
    iqdr_detail = data.get("iqdr_detail", {})
    interp = iqdr_detail.get("interpretation", "")
    ax.text(0.98, 0.02, f"IQDR = {iqdr:.3f} ({interp})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            fontweight="bold", color=COLORS["grey"])

    fig.tight_layout()

    if dry_run:
        plt.close(fig)
        return None
    return _save_figure(fig, "fig_exp13_iqdr.pdf")


# ---------------------------------------------------------------------------
# Master generation function
# ---------------------------------------------------------------------------
FIGURE_GENERATORS: list[dict[str, Any]] = [
    {"name": "fig_exp01_lambda", "func": generate_fig_exp01, "exp": "exp01"},
    {"name": "fig_exp02_pareidolia", "func": generate_fig_exp02, "exp": "exp02"},
    {"name": "fig_exp03_heatmap", "func": generate_fig_exp03, "exp": "exp03"},
    {"name": "fig_exp04_tornado", "func": generate_fig_exp04_tornado, "exp": "exp04"},
    {"name": "fig_exp04_grid", "func": generate_fig_exp04_grid, "exp": "exp04"},
    {"name": "fig_exp05_progressive", "func": generate_fig_exp05, "exp": "exp05"},
    {"name": "fig_exp05_radar", "func": generate_fig_exp05_radar, "exp": "exp05"},
    {"name": "fig_exp06_baseline", "func": generate_fig_exp06, "exp": "exp06"},
    {"name": "fig_exp07_degradation", "func": generate_fig_exp07, "exp": "exp07"},
    {"name": "fig_exp07_boxplot", "func": generate_fig_exp07_boxplot, "exp": "exp07"},
    {"name": "fig_exp08_confusion", "func": generate_fig_exp08, "exp": "exp08"},
    {"name": "fig_exp08_domain_gap", "func": generate_fig_exp08_domain_gap, "exp": "exp08"},
    {"name": "fig_exp09_indicators", "func": generate_fig_exp09_indicators, "exp": "exp09"},
    {"name": "fig_exp10_pareto", "func": generate_fig_exp10, "exp": "exp10"},
    {"name": "fig_exp10_stacked_latency", "func": generate_fig_exp10_stacked_latency, "exp": "exp10"},
    {"name": "fig_exp10_pie", "func": generate_fig_exp10_pie, "exp": "exp10"},
    {"name": "fig_exp12_verification", "func": generate_fig_exp12, "exp": "exp12"},
    {"name": "fig_exp13_iqdr", "func": generate_fig_exp13, "exp": "exp13"},
]


def generate_all_figures(
    dry_run: bool = False,
    only: str | None = None,
) -> dict[str, Any]:
    """Generate all available figures.

    Args:
        dry_run: If True, create figures but do not save.
        only: If specified, generate only the matching figure(s).

    Returns:
        Summary of generated figures.
    """
    if not HAS_MATPLOTLIB:
        log.error("matplotlib not available; cannot generate figures")
        return {"error": "matplotlib not installed"}

    _apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    generated = 0
    skipped = 0

    for fg in FIGURE_GENERATORS:
        if only and only not in fg["name"]:
            continue

        log.info("Generating %s (from %s)...", fg["name"], fg["exp"])
        try:
            path = fg["func"](dry_run=dry_run)
            if path:
                results.append({
                    "figure": fg["name"],
                    "path": str(path),
                    "status": "generated",
                })
                generated += 1
            else:
                results.append({
                    "figure": fg["name"],
                    "status": "skipped" if not dry_run else "dry_run",
                    "reason": "no data or dry_run",
                })
                skipped += 1
        except Exception as exc:
            log.error("Error generating %s: %s", fg["name"], exc)
            results.append({
                "figure": fg["name"],
                "status": "error",
                "error": str(exc),
            })
            skipped += 1

    return {
        "generated": generated,
        "skipped": skipped,
        "total": len(results),
        "figures": results,
        "output_dir": str(FIGURES_DIR),
    }


def print_summary(result: dict[str, Any]) -> None:
    """Print figure generation summary."""
    print("\n" + "=" * 70)
    print("Figure Generation Summary")
    print("=" * 70)
    print(
        f"  Generated: {result['generated']}, "
        f"Skipped: {result['skipped']}, "
        f"Total: {result['total']}"
    )
    print(f"  Output dir: {result['output_dir']}")

    print(f"\n{'Figure':<30s}  {'Status':<12s}  {'Path/Reason'}")
    print("-" * 70)
    for fg in result["figures"]:
        path_or_reason = fg.get("path", fg.get("reason", fg.get("error", "")))
        print(f"  {fg['figure']:<28s}  {fg['status']:<12s}  {path_or_reason}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate all paper figures from experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generate_figures.py --dry-run\n"
            "  python generate_figures.py\n"
            "  python generate_figures.py --only exp05\n"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Create figures but do not save to disk",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Generate only figures matching this substring (e.g., 'exp05')",
    )
    args = parser.parse_args()

    if not HAS_MATPLOTLIB:
        log.error("matplotlib is required. Install with: pip install matplotlib seaborn")
        sys.exit(1)

    result = generate_all_figures(dry_run=args.dry_run, only=args.only)
    print_summary(result)


if __name__ == "__main__":
    main()
