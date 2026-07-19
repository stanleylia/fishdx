"""Latency Benchmark — Measure per-component timing.

Profiles each pipeline stage to validate E2E latency targets (Table 15b).

Usage: python scripts/benchmark_latency.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config
from core.algorithms.fusion import create_fusion_embedding, normalize_l2
from core.algorithms.semantic_filter import classify_scene
from core.algorithms.scoring import full_scoring_pipeline
from core.algorithms.pareidolia import detect_hard
from core.algorithms.verification import RAGItem, run_verification_loop

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    component: str
    latency_ms: float
    iterations: int


def benchmark(name: str, fn: object, iterations: int = 100) -> BenchmarkResult:
    """Benchmark a function over multiple iterations."""
    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_ms = np.mean(times)
    return BenchmarkResult(component=name, latency_ms=avg_ms, iterations=iterations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark component latency")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    config = load_config()
    rng = np.random.RandomState(42)
    n_iter = args.iterations

    results: list[BenchmarkResult] = []

    # 1. Pareidolia Detection
    results.append(benchmark(
        "Pareidolia (Hard)",
        lambda: detect_hard("A net cage with mesh", ["face", "fish"], config.pareidolia),
        n_iter,
    ))

    # 2. Fusion Embedding
    e_v = rng.randn(512).astype(np.float32)
    e_c = rng.randn(512).astype(np.float32)
    results.append(benchmark(
        "Fusion Embedding",
        lambda: create_fusion_embedding(e_v, e_c, config.fusion),
        n_iter,
    ))

    # 3. Semantic Filter
    results.append(benchmark(
        "Semantic Filter",
        lambda: classify_scene("A tilapia fish with gill damage in pond", config.semantic_filter),
        n_iter,
    ))

    # 4. Scoring Pipeline
    text = "The fish is diagnosed with white spot disease, confirmed infected"
    results.append(benchmark(
        "Scoring Pipeline",
        lambda: full_scoring_pipeline(text, config.scoring),
        n_iter,
    ))

    # 5. L2 Normalization
    raw = rng.randn(512).astype(np.float32)
    results.append(benchmark(
        "L2 Normalize",
        lambda: normalize_l2(raw),
        n_iter,
    ))

    # Report
    print("\n═══ Latency Benchmark Results ═══")
    print(f"{'Component':<25s} {'Avg (ms)':>10s} {'Iterations':>12s}")
    print("─" * 50)
    total_ms = 0.0
    for r in results:
        print(f"  {r.component:<23s} {r.latency_ms:>8.3f}ms  {r.iterations:>10d}")
        total_ms += r.latency_ms
    print("─" * 50)
    print(f"  {'Total (algorithms)':<23s} {total_ms:>8.3f}ms")
    print(f"\n  Note: Florence-2, CLIP, and ChromaDB latencies")
    print(f"  are not included (require GPU/model loading).")
    print(f"  See Table 15b for full E2E latency profile (~7.1s total).")


if __name__ == "__main__":
    main()
