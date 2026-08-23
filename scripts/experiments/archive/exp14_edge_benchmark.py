"""EXP-14: Edge Deployment Benchmark — KV260 (Table 14 / Figure).

Hypothesis: YOLOv7-tiny INT8 deployed on Xilinx Kria KV260 meets real-time
aquaculture monitoring requirements: FPS >= 15, Latency <= 67 ms,
Power <= 5.0 W, FPS/Watt >= 3.0, mAP@0.5 >= 0.70.

Data sources (in priority order):
  1. Real benchmark data from edge/benchmark_results.json (live KV260 run)
  2. Reference values from ICMT 2026 report (--use-reference flag)
  3. Simulated representative data with noise (--dry-run)

Reference values (ICMT 2026):
  FPS ~ 16.26, Latency ~ 61.5 ms, Power ~ 4.87 W,
  FPS/Watt ~ 3.34, mAP@0.5 ~ 0.72

Metrics collected per frame batch:
  - FPS (frames per second)
  - Latency (ms per frame)
  - Power draw (W, measured via INA260 sensor or estimated)
  - FPS/Watt (energy efficiency)
  - mAP@0.5 (detection accuracy on validation set)

Level 1: Offline / simulated unless KV260 hardware available.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Project bootstrap
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.experiment_harness import (  # noqa: E402
    add_common_args,
    compute_stats,
    parse_common_args,
    save_result,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EXPERIMENT_ID = "exp14"

# ---------------------------------------------------------------------------
# Reference values (ICMT 2026 report)
# ---------------------------------------------------------------------------
REFERENCE_VALUES: dict[str, float] = {
    "fps": 16.26,
    "latency_ms": 61.5,
    "power_w": 4.87,
    "fps_per_watt": 3.34,
    "map50": 0.72,
}

# ---------------------------------------------------------------------------
# Target thresholds (pass/fail criteria)
# ---------------------------------------------------------------------------
TARGETS: dict[str, dict[str, Any]] = {
    "fps":          {"threshold": 15.0,  "op": ">=", "unit": "FPS"},
    "latency_ms":   {"threshold": 67.0,  "op": "<=", "unit": "ms"},
    "power_w":      {"threshold": 5.0,   "op": "<=", "unit": "W"},
    "fps_per_watt": {"threshold": 3.0,   "op": ">=", "unit": "FPS/W"},
    "map50":        {"threshold": 0.70,  "op": ">=", "unit": ""},
}

# ---------------------------------------------------------------------------
# Benchmark data path (populated by real KV260 runs)
# ---------------------------------------------------------------------------
EDGE_BENCHMARK_PATH = PROJECT_ROOT / "edge" / "benchmark_results.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def _check_pass(metric: str, value: float) -> bool:
    """Check whether a metric value passes its target threshold.

    Args:
        metric: Metric key from TARGETS.
        value: Observed value.

    Returns:
        True if the value meets or exceeds the target.
    """
    target = TARGETS[metric]
    if target["op"] == ">=":
        return value >= target["threshold"]
    else:  # "<="
        return value <= target["threshold"]


def _generate_simulated_data(
    n_frames: int,
    n_runs: int,
    rng: np.random.RandomState,
) -> list[dict[str, Any]]:
    """Generate representative simulated benchmark data.

    Models realistic KV260 performance with Gaussian noise around
    reference values. Includes run-to-run variance.

    Args:
        n_frames: Number of frames per run.
        n_runs: Number of independent runs.
        rng: NumPy random state for reproducibility.

    Returns:
        List of per-run result dicts.
    """
    runs: list[dict[str, Any]] = []

    for run_idx in range(n_runs):
        # Per-frame latency samples (ms) — slight run-to-run drift
        run_offset = rng.normal(0, 1.0)
        latencies = rng.normal(
            REFERENCE_VALUES["latency_ms"] + run_offset,
            2.5,
            size=n_frames,
        )
        # Clamp to realistic range
        latencies = np.clip(latencies, 45.0, 85.0)

        fps_values = 1000.0 / latencies

        # Power draw samples (W) — less variable
        power_samples = rng.normal(
            REFERENCE_VALUES["power_w"],
            0.15,
            size=n_frames,
        )
        power_samples = np.clip(power_samples, 3.5, 6.0)

        # mAP is evaluated per-run, not per-frame
        map50 = float(np.clip(
            rng.normal(REFERENCE_VALUES["map50"], 0.015),
            0.60,
            0.82,
        ))

        fps_per_watt = fps_values / power_samples

        runs.append({
            "run_idx": run_idx,
            "n_frames": n_frames,
            "fps": fps_values.tolist(),
            "latency_ms": latencies.tolist(),
            "power_w": power_samples.tolist(),
            "fps_per_watt": fps_per_watt.tolist(),
            "map50": map50,
        })

    return runs


def _load_real_data(path: Path) -> list[dict[str, Any]] | None:
    """Attempt to load real benchmark data from JSON file.

    Expected format: list of run dicts, each containing keys:
      fps, latency_ms, power_w, fps_per_watt, map50

    Args:
        path: Path to benchmark_results.json.

    Returns:
        List of run dicts if file exists and is valid, else None.
    """
    if not path.exists():
        log.info("Real benchmark data not found at %s", path)
        return None

    try:
        with open(path) as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.get("runs", [data])
        if not isinstance(data, list) or len(data) == 0:
            log.warning("Benchmark file has unexpected format")
            return None

        log.info("Loaded real benchmark data: %d runs from %s", len(data), path)
        return data
    except (json.JSONDecodeError, KeyError) as exc:
        log.warning("Failed to parse benchmark data: %s", exc)
        return None


def _build_reference_runs(n_runs: int) -> list[dict[str, Any]]:
    """Build pseudo-runs from ICMT 2026 reference values (no variance).

    Used when --use-reference is set. Each run returns a single
    measurement equal to the reference value.

    Args:
        n_runs: Number of identical reference runs to generate.

    Returns:
        List of run dicts with scalar reference values.
    """
    runs: list[dict[str, Any]] = []
    for run_idx in range(n_runs):
        runs.append({
            "run_idx": run_idx,
            "n_frames": 1,
            "fps": [REFERENCE_VALUES["fps"]],
            "latency_ms": [REFERENCE_VALUES["latency_ms"]],
            "power_w": [REFERENCE_VALUES["power_w"]],
            "fps_per_watt": [REFERENCE_VALUES["fps_per_watt"]],
            "map50": REFERENCE_VALUES["map50"],
        })
    return runs


def _compute_descriptive_stats(values: list[float]) -> dict[str, float]:
    """Compute descriptive statistics including median and IQR.

    Args:
        values: List of numeric values.

    Returns:
        Dict with mean, std, median, q1, q3, iqr, min, max.
    """
    arr = np.array(values, dtype=float)
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "median": float(np.median(arr)),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(arr),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run_experiment(
    dry_run: bool = False,
    n_runs: int = 3,
    n_frames: int = 100,
    use_reference: bool = False,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Execute the KV260 edge deployment benchmark.

    Args:
        dry_run: If True, generate simulated data with fewer frames.
        n_runs: Number of independent runs.
        n_frames: Number of frames per run (simulated / dry-run).
        use_reference: If True, use ICMT 2026 reference values directly.
        config_path: Optional override config YAML path (unused but kept
            for CLI consistency with other experiments).

    Returns:
        Complete result dict.
    """
    t0 = time.perf_counter()

    if dry_run:
        n_frames = min(n_frames, 20)
        n_runs = min(n_runs, 2)

    log.info(
        "EXP-14 starting: runs=%d, frames=%d, dry_run=%s, use_reference=%s",
        n_runs, n_frames, dry_run, use_reference,
    )

    # --- Determine data source ---
    data_source = "simulated"
    runs: list[dict[str, Any]]

    # Priority 1: real benchmark data
    real_data = _load_real_data(EDGE_BENCHMARK_PATH)
    if real_data is not None:
        runs = real_data
        data_source = "real_kv260"
        log.info("Using REAL KV260 benchmark data (%d runs)", len(runs))
    elif use_reference:
        # Priority 2: reference values
        runs = _build_reference_runs(n_runs)
        data_source = "reference_icmt2026"
        log.info("Using ICMT 2026 reference values")
    else:
        # Priority 3: simulated data
        rng = np.random.RandomState(seed=42)
        runs = _generate_simulated_data(n_frames, n_runs, rng)
        data_source = "simulated"
        log.info("Using simulated data (%d runs x %d frames)", n_runs, n_frames)

    # --- Aggregate metrics across all runs ---
    all_fps: list[float] = []
    all_latency: list[float] = []
    all_power: list[float] = []
    all_fps_per_watt: list[float] = []
    all_map50: list[float] = []
    per_run_summary: list[dict[str, Any]] = []

    for run in runs:
        run_fps = run["fps"] if isinstance(run["fps"], list) else [run["fps"]]
        run_lat = run["latency_ms"] if isinstance(run["latency_ms"], list) else [run["latency_ms"]]
        run_pow = run["power_w"] if isinstance(run["power_w"], list) else [run["power_w"]]
        run_fpw = run["fps_per_watt"] if isinstance(run["fps_per_watt"], list) else [run["fps_per_watt"]]
        run_map = run["map50"] if isinstance(run["map50"], (int, float)) else run["map50"]

        all_fps.extend(run_fps)
        all_latency.extend(run_lat)
        all_power.extend(run_pow)
        all_fps_per_watt.extend(run_fpw)
        if isinstance(run_map, (int, float)):
            all_map50.append(float(run_map))
        elif isinstance(run_map, list):
            all_map50.extend([float(v) for v in run_map])

        per_run_summary.append({
            "run_idx": run.get("run_idx", len(per_run_summary)),
            "n_frames": run.get("n_frames", len(run_fps)),
            "fps_mean": float(np.mean(run_fps)),
            "latency_mean_ms": float(np.mean(run_lat)),
            "power_mean_w": float(np.mean(run_pow)),
            "fps_per_watt_mean": float(np.mean(run_fpw)),
            "map50": float(run_map) if isinstance(run_map, (int, float)) else float(np.mean(run_map)),
        })

    # --- Compute descriptive statistics ---
    stats = {
        "fps": _compute_descriptive_stats(all_fps),
        "latency_ms": _compute_descriptive_stats(all_latency),
        "power_w": _compute_descriptive_stats(all_power),
        "fps_per_watt": _compute_descriptive_stats(all_fps_per_watt),
        "map50": _compute_descriptive_stats(all_map50),
    }

    # --- Compute harness-compatible stats (mean, std, CI) ---
    harness_stats = {
        "fps": compute_stats(all_fps),
        "latency_ms": compute_stats(all_latency),
        "power_w": compute_stats(all_power),
        "fps_per_watt": compute_stats(all_fps_per_watt),
        "map50": compute_stats(all_map50),
    }

    # --- Pass/fail against targets ---
    target_results: dict[str, dict[str, Any]] = {}
    for metric, target_def in TARGETS.items():
        observed = stats[metric]["median"]
        passed = _check_pass(metric, observed)
        target_results[metric] = {
            "observed_median": observed,
            "observed_mean": stats[metric]["mean"],
            "threshold": target_def["threshold"],
            "operator": target_def["op"],
            "unit": target_def["unit"],
            "pass": passed,
        }

    all_passed = all(v["pass"] for v in target_results.values())

    elapsed = time.perf_counter() - t0

    result: dict[str, Any] = {
        "experiment": "EXP-14: Edge Deployment Benchmark (KV260)",
        "hypothesis": (
            "YOLOv7-tiny INT8 on Xilinx Kria KV260 meets real-time "
            "aquaculture monitoring requirements"
        ),
        "table_ref": "Table 14",
        "parameters": {
            "model": "YOLOv7-tiny INT8",
            "platform": "Xilinx Kria KV260",
            "data_source": data_source,
            "n_runs": len(runs),
            "n_frames_total": len(all_fps),
            "dry_run": dry_run,
            "use_reference": use_reference,
        },
        "reference_values": REFERENCE_VALUES,
        "targets": {
            k: {"threshold": v["threshold"], "op": v["op"], "unit": v["unit"]}
            for k, v in TARGETS.items()
        },
        "descriptive_stats": {
            k: v for k, v in stats.items()
        },
        "harness_stats": {
            k: {
                "mean": v.mean,
                "std": v.std,
                "ci_95_low": v.ci_95_low,
                "ci_95_high": v.ci_95_high,
                "n": v.n,
            }
            for k, v in harness_stats.items()
        },
        "target_results": target_results,
        "all_targets_met": all_passed,
        "per_run_summary": per_run_summary,
        "elapsed_seconds": elapsed,
    }

    return result


# ---------------------------------------------------------------------------
# Pretty-print summary
# ---------------------------------------------------------------------------
def print_summary(result: dict[str, Any]) -> None:
    """Print a human-readable summary table with pass/fail indicators."""
    print_header("EXP-14: Edge Deployment Benchmark (KV260)")

    params = result["parameters"]
    print(f"\n  Model:       {params['model']}")
    print(f"  Platform:    {params['platform']}")
    print(f"  Data Source: {params['data_source']}")
    print(f"  Runs:        {params['n_runs']}")
    print(f"  Frames:      {params['n_frames_total']}")

    # --- Metric table ---
    print(f"\n{'Metric':<16s}  "
          f"{'Median':>8s}  "
          f"{'Mean':>8s}  "
          f"{'Std':>7s}  "
          f"{'IQR':>7s}  "
          f"{'Target':>10s}  "
          f"{'Status':>6s}")
    print("-" * 80)

    metric_labels = {
        "fps":          "FPS",
        "latency_ms":   "Latency (ms)",
        "power_w":      "Power (W)",
        "fps_per_watt": "FPS/Watt",
        "map50":        "mAP@0.5",
    }

    ds = result["descriptive_stats"]
    tr = result["target_results"]

    for metric in ["fps", "latency_ms", "power_w", "fps_per_watt", "map50"]:
        label = metric_labels[metric]
        s = ds[metric]
        t = tr[metric]
        status = "PASS" if t["pass"] else "FAIL"
        target_str = f"{t['operator']}{t['threshold']:.1f}"
        if metric == "map50":
            target_str = f"{t['operator']}{t['threshold']:.2f}"

        print(
            f"{label:<16s}  "
            f"{s['median']:>8.2f}  "
            f"{s['mean']:>8.2f}  "
            f"{s['std']:>7.2f}  "
            f"{s['iqr']:>7.2f}  "
            f"{target_str:>10s}  "
            f"{'[' + status + ']':>6s}"
        )

    print("-" * 80)

    # --- Overall verdict ---
    if result["all_targets_met"]:
        verdict = "ALL TARGETS MET"
    else:
        failed = [m for m, t in tr.items() if not t["pass"]]
        verdict = "TARGETS NOT MET: " + ", ".join(failed)
    print(f"\n  Verdict: {verdict}")

    # --- Reference comparison ---
    print("\n--- Reference Comparison (ICMT 2026) ---")
    ref = result["reference_values"]
    print(f"  {'Metric':<16s}  {'Reference':>10s}  {'Observed':>10s}  {'Delta':>8s}")
    print("  " + "-" * 52)
    for metric in ["fps", "latency_ms", "power_w", "fps_per_watt", "map50"]:
        label = metric_labels[metric]
        ref_val = ref[metric]
        obs_val = ds[metric]["median"]
        delta = obs_val - ref_val
        sign = "+" if delta >= 0 else ""
        fmt = ".2f" if metric != "map50" else ".3f"
        print(
            f"  {label:<16s}  "
            f"{ref_val:>10{fmt}}  "
            f"{obs_val:>10{fmt}}  "
            f"{sign}{delta:>7{fmt}}"
        )

    # --- Per-run summary ---
    print("\n--- Per-Run Summary ---")
    print(
        f"  {'Run':>4s}  "
        f"{'Frames':>6s}  "
        f"{'FPS':>8s}  "
        f"{'Lat(ms)':>8s}  "
        f"{'Power(W)':>8s}  "
        f"{'FPS/W':>8s}  "
        f"{'mAP@0.5':>8s}"
    )
    print("  " + "-" * 62)
    for run in result["per_run_summary"]:
        print(
            f"  {run['run_idx']:>4d}  "
            f"{run['n_frames']:>6d}  "
            f"{run['fps_mean']:>8.2f}  "
            f"{run['latency_mean_ms']:>8.2f}  "
            f"{run['power_mean_w']:>8.2f}  "
            f"{run['fps_per_watt_mean']:>8.2f}  "
            f"{run['map50']:>8.3f}"
        )

    print(f"\n  Total elapsed: {result['elapsed_seconds']:.3f}s")
    print("=" * 90)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="EXP-14: Edge Deployment Benchmark (KV260)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python exp14_edge_benchmark.py --dry-run\n"
            "  python exp14_edge_benchmark.py --use-reference\n"
            "  python exp14_edge_benchmark.py --runs 5 --frames 200\n"
        ),
    )
    add_common_args(parser)
    parser.add_argument(
        "--use-reference", action="store_true",
        help="Use ICMT 2026 reference values instead of simulating",
    )
    parser.add_argument(
        "--frames", type=int, default=100,
        help="Number of frames per run for simulation (default: 100)",
    )
    args = parser.parse_args()
    common = parse_common_args(args)

    log.info("Configuration: %s", common)

    result = run_experiment(
        dry_run=common["dry_run"],
        n_runs=common["runs"],
        n_frames=args.frames,
        use_reference=args.use_reference,
        config_path=common["config_path"],
    )

    # Save
    filepath = save_result(EXPERIMENT_ID, result)
    log.info("Results saved to %s", filepath)

    # Print summary
    print_summary(result)


if __name__ == "__main__":
    main()
