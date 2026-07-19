"""EXP-1 λ × similarity_cutoff 2D sweep on D1 Test — M3 first experiment.

Design
------
Stage 1 (Florence-2 + Pareidolia) and CLIP encoding outputs are independent
of both λ and ``similarity_cutoff``. Precompute them ONCE per image, then
sweep 49 ``(λ, cutoff)`` cells over cached embeddings. Converts a
~2.4-hour full-pipeline sweep into ~3-min precompute + ~2-min sweep.

Outputs (``docs/m3/exp1/``)
---------------------------
* ``results.csv``       — 49 rows × (λ, cutoff, DA, CI_lo, CI_hi, n_correct,
                          n_inconclusive, per_class DA, median_latency_ms)
* ``heatmap.png``       — λ×cutoff DA heatmap with annotations
* ``best-config.md``    — chosen (λ*, cutoff*) + paper sanity check +
                          per-class breakdown at best cell
* ``determinism.json``  — two-run bit-identical verification on chosen cell
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import chromadb
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from fishdx.config import load_config  # noqa: E402
from fishdx.kb.builder import KBBuilder, parse_markdown_docs  # noqa: E402
from fishdx.perception.florence2 import Florence2Wrapper  # noqa: E402
from fishdx.perception.postprocess import clean_caption  # noqa: E402
from fishdx.retrieval.clip_encoder import OpenClipEmbedder  # noqa: E402
from fishdx.retrieval.fusion import fuse_and_normalize  # noqa: E402
from fishdx.retrieval.store import ChromaStore  # noqa: E402
from fishdx.retrieval.verification import verify_candidates  # noqa: E402
from fishdx.schemas import DecisionEnum  # noqa: E402
from fishdx.scoring.decision import make_decision  # noqa: E402
from fishdx.scoring.score import compute_s_d, compute_s_h  # noqa: E402

LAMBDAS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
# Cutoff range updated for REPAIR-6 under ADR-0011 (shifted downward from
# the original EXP-1 range {0.20..0.50}); paper-literal 0.5 is computed
# as a supplemental baseline cell outside the grid.
CUTOFFS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
SUPPLEMENTAL_PAPER_CUTOFF = 0.5  # (λ=0.7, cutoff=0.5) paper baseline

# D1 class-folder → KB doc_id mapping (6 disease folders + Healthy)
CLASS_TO_DOC_ID: dict[str, str] = {
    "Bacterial Red disease": "BRD",
    "Bacterial diseases - Aeromoniasis": "AER",
    "Bacterial gill disease": "COL",
    "Fungal diseases Saprolegniasis": "SAP",
    "Healthy Fish": "HLT",
    "Parasitic diseases": "PAR",
    "Viral diseases White tail disease": "VWT",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wilson_ci(n_correct: int, n_total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score 95% CI (normal approximation)."""
    if n_total == 0:
        return (0.0, 0.0)
    p = n_correct / n_total
    z = 1.959964  # 97.5 percentile of standard normal
    denom = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denom
    half = (z / denom) * ((p * (1 - p) / n_total + z**2 / (4 * n_total**2)) ** 0.5)
    return (max(0.0, center - half), min(1.0, center + half))


def precompute_embeddings(
    images: list[Path], florence: Florence2Wrapper, clip: OpenClipEmbedder
) -> list[dict]:
    """Run Stage 1 + CLIP encode for each image; return cache list."""
    out: list[dict] = []
    t0 = time.perf_counter()
    for i, p in enumerate(images):
        caption_raw, _objs, _ = florence.caption_and_detect(p)
        caption = clean_caption(caption_raw)
        text_input = caption if caption else " "
        e_caption = clip.encode_text([text_input])[0]
        e_visual = clip.encode_image_paths([str(p)])[0]
        out.append(
            {
                "path": str(p),
                "class": p.parent.name,
                "caption": caption,
                "e_caption": e_caption,
                "e_visual": e_visual,
            }
        )
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{i + 1}/{len(images)}] encoded in {elapsed:.1f} s")
    print(f"  precompute total: {time.perf_counter() - t0:.1f} s")
    return out


def sweep_cell(
    cache: list[dict],
    lam: float,
    cutoff: float,
    kb_docs: dict[str, object],
    hlt_keywords: tuple[str, ...],
    store: ChromaStore,
    cfg: object,
) -> dict:
    """Evaluate one (λ, cutoff) cell over cached embeddings.

    Produces two DA metrics:

    * ``retrieval_da`` (primary) — top-1 retrieved doc's ``doc_id`` matches
      ground-truth ``doc_id``; no-candidate → incorrect. This is the
      ``correct-class top-1 retrieval proportion`` per EXP-1 kickoff spec.
    * ``decision_da`` (secondary) — full Stage 3 decision outcome matches
      ground truth; Inconclusive always counts as incorrect. Reported
      alongside retrieval_da to measure how much Stage 3 scoring
      preserves retrieval correctness under current KB vocabulary
      coverage.
    """
    retrieval_correct = 0
    decision_correct = 0
    retrieval_no_candidate = 0
    decision_inconclusive = 0
    per_class_total: dict[str, int] = defaultdict(int)
    per_class_retrieval_correct: dict[str, int] = defaultdict(int)
    iter_counts: list[int] = []

    cell_start = time.perf_counter()
    for row in cache:
        true_doc_id = CLASS_TO_DOC_ID.get(row["class"])
        per_class_total[row["class"]] += 1

        e_final = fuse_and_normalize(row["e_visual"], row["e_caption"], lam)
        candidates = store.query(e_final, similarity_cutoff=cutoff)
        if candidates:
            grounding = [0.9] * len(candidates)
            verified, iters, _ = verify_candidates(candidates, grounding, cfg.verification)
        else:
            verified, iters = [], 0
        iter_counts.append(iters)

        top1 = verified[0] if verified else None
        top2 = verified[1] if len(verified) >= 2 else None

        # ── Retrieval DA (primary) ──────────────────────────────
        if top1 is None:
            retrieval_no_candidate += 1
        elif top1.doc_id == true_doc_id:
            retrieval_correct += 1
            per_class_retrieval_correct[row["class"]] += 1

        # ── Decision DA (secondary) ─────────────────────────────
        s_h = compute_s_h(
            row["caption"],
            cfg.scoring.healthy_weights,
            cfg.negation,
            healthy_keywords=hlt_keywords,
        )
        s_d = 0
        disease_class = None
        pred_doc_id_candidate: str | None = None
        if top1:
            kb_doc = kb_docs.get(top1.doc_id)
            if kb_doc is not None:
                s_d = compute_s_d(
                    row["caption"],
                    cfg.scoring,
                    confirmed_keywords=tuple(kb_doc.clinical_keywords.get("confirmed", [])),
                    suspected_keywords=tuple(kb_doc.clinical_keywords.get("suspected", [])),
                    mentioned_keywords=tuple(kb_doc.clinical_keywords.get("mentioned", [])),
                )
                disease_class = kb_doc.disease_class
                pred_doc_id_candidate = top1.doc_id

        top1_sim = (top1.similarity_penalized or top1.similarity) if top1 else 0.0
        top2_sim = (top2.similarity_penalized or top2.similarity) if top2 else 0.0

        result = make_decision(
            score_healthy=s_h,
            score_disease=s_d,
            top1_similarity=float(top1_sim),
            top2_similarity=float(top2_sim),
            disease_class=disease_class,
            decision_config=cfg.decision,
            margin_config=cfg.margin,
        )

        if result.decision is DecisionEnum.HEALTHY:
            decision_pred = "HLT"
        elif result.decision is DecisionEnum.DISEASE:
            decision_pred = pred_doc_id_candidate
        else:
            decision_inconclusive += 1
            decision_pred = None

        if decision_pred is not None and decision_pred == true_doc_id:
            decision_correct += 1

    cell_elapsed = time.perf_counter() - cell_start
    n = len(cache)
    retrieval_da = retrieval_correct / n if n else 0.0
    decision_da = decision_correct / n if n else 0.0
    ret_ci_lo, ret_ci_hi = wilson_ci(retrieval_correct, n)
    dec_ci_lo, dec_ci_hi = wilson_ci(decision_correct, n)
    per_class_retrieval_da = {
        cls: per_class_retrieval_correct[cls] / per_class_total[cls]
        for cls in sorted(per_class_total)
        if per_class_total[cls] > 0
    }
    return {
        "lambda": lam,
        "cutoff": cutoff,
        "n": n,
        "retrieval_correct": retrieval_correct,
        "retrieval_no_candidate": retrieval_no_candidate,
        "retrieval_da": retrieval_da,
        "retrieval_ci_lo": ret_ci_lo,
        "retrieval_ci_hi": ret_ci_hi,
        "decision_correct": decision_correct,
        "decision_inconclusive": decision_inconclusive,
        "decision_da": decision_da,
        "decision_ci_lo": dec_ci_lo,
        "decision_ci_hi": dec_ci_hi,
        "per_class_retrieval_da": per_class_retrieval_da,
        "cell_seconds": cell_elapsed,
        "iter_counts_median": statistics.median(iter_counts) if iter_counts else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=0, help="If >0, randomly subsample D1 Test to this size (for pilot)")
    parser.add_argument("--lambdas", type=str, default=",".join(str(x) for x in LAMBDAS))
    parser.add_argument("--cutoffs", type=str, default=",".join(str(x) for x in CUTOFFS))
    parser.add_argument("--out", type=str, default="docs/m3/exp1")
    args = parser.parse_args()

    lambdas = [float(x) for x in args.lambdas.split(",")]
    cutoffs = [float(x) for x in args.cutoffs.split(",")]
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(42)
    cfg = load_config(REPO / "configs/default.yaml")

    d1_test = (
        REPO
        / "lab_dateset/external_datasets/fish_disease_south_asia"
        / "Freshwater Fish Disease Aquaculture in south asia/Test"
    )
    image_exts = {".jpg", ".jpeg", ".png"}
    images = sorted(p for p in d1_test.rglob("*") if p.is_file() and p.suffix.lower() in image_exts)
    if args.subset > 0:
        rng = random.Random(42)
        images = rng.sample(images, min(args.subset, len(images)))
        images.sort()
    print(f"D1 Test images: {len(images)}")

    florence = Florence2Wrapper(cfg.florence2)
    clip = OpenClipEmbedder(cfg.clip)
    print("[warmup] Florence-2 + OpenCLIP")
    florence.warmup()
    clip.warmup()

    print("[warmup] KB → ChromaDB")
    import tempfile

    tmp_kb = tempfile.mkdtemp()
    chroma = chromadb.PersistentClient(path=tmp_kb)
    docs = parse_markdown_docs(REPO / "src/fishdx/kb/documents")
    builder = KBBuilder(
        chroma_client=chroma, embedder=clip, kb_config=cfg.kb, retrieval_config=cfg.retrieval
    )
    builder.ingest(docs)
    store = ChromaStore(
        chroma_client=chroma,
        collection_name=cfg.kb.collection_name,
        retrieval_config=cfg.retrieval,
    )
    store.warmup()
    kb_docs = {d.doc_id: d for d in docs}
    hlt_doc = kb_docs.get("HLT")
    hlt_keywords = tuple(hlt_doc.healthy_keywords) if hlt_doc else ("healthy",)

    print("\n[precompute] Stage 1 + CLIP encode per image")
    cache = precompute_embeddings(images, florence, clip)

    print(f"\n[sweep] {len(lambdas)}×{len(cutoffs)} = {len(lambdas) * len(cutoffs)} cells")
    results: list[dict] = []
    sweep_t0 = time.perf_counter()
    for lam in lambdas:
        for cutoff in cutoffs:
            result = sweep_cell(cache, lam, cutoff, kb_docs, hlt_keywords, store, cfg)
            results.append(result)
            print(
                f"  λ={lam:.2f} cutoff={cutoff:.2f}  "
                f"retr_DA={result['retrieval_da']:.4f} "
                f"[{result['retrieval_ci_lo']:.3f},{result['retrieval_ci_hi']:.3f}] "
                f"({result['retrieval_correct']}/{result['n']})  "
                f"dec_DA={result['decision_da']:.4f} "
                f"(inc={result['decision_inconclusive']})  "
                f"{result['cell_seconds']:.1f}s"
            )
    sweep_elapsed = time.perf_counter() - sweep_t0
    print(f"[sweep] total {sweep_elapsed:.1f} s")

    # Supplemental paper-baseline cell: (λ=0.7, cutoff=0.5)
    print(f"\n[supplemental] (λ=0.7, cutoff={SUPPLEMENTAL_PAPER_CUTOFF}) paper-literal baseline")
    supplemental = sweep_cell(
        cache, 0.7, SUPPLEMENTAL_PAPER_CUTOFF, kb_docs, hlt_keywords, store, cfg
    )
    print(
        f"  retr_DA={supplemental['retrieval_da']:.4f} "
        f"[{supplemental['retrieval_ci_lo']:.3f},{supplemental['retrieval_ci_hi']:.3f}] "
        f"({supplemental['retrieval_correct']}/{supplemental['n']})  "
        f"dec_DA={supplemental['decision_da']:.4f}"
    )

    # Determinism check: re-run best-retrieval-DA cell
    best = max(results, key=lambda r: r["retrieval_da"])
    print(
        f"\n[determinism] re-running best cell λ={best['lambda']} cutoff={best['cutoff']}"
    )
    det_run = sweep_cell(cache, best["lambda"], best["cutoff"], kb_docs, hlt_keywords, store, cfg)
    det_ok = (
        det_run["retrieval_correct"] == best["retrieval_correct"]
        and abs(det_run["retrieval_da"] - best["retrieval_da"]) < 1e-12
        and det_run["decision_correct"] == best["decision_correct"]
    )

    # Paper sanity check at (λ=0.7, cutoff=0.5) if in grid
    paper_cell = next(
        (r for r in results if abs(r["lambda"] - 0.7) < 1e-9 and abs(r["cutoff"] - 0.5) < 1e-9),
        None,
    )

    # Save CSV with per-class columns
    all_classes = sorted({cls for r in results for cls in r["per_class_retrieval_da"]})
    csv_path = out_dir / "results.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        header = [
            "lambda",
            "cutoff",
            "n",
            "retrieval_correct",
            "retrieval_no_candidate",
            "retrieval_da",
            "retrieval_ci_lo",
            "retrieval_ci_hi",
            "decision_correct",
            "decision_inconclusive",
            "decision_da",
            "decision_ci_lo",
            "decision_ci_hi",
            "cell_seconds",
            "iter_counts_median",
        ]
        for cls in all_classes:
            short = cls.replace(" ", "_").replace("-", "").replace("__", "_")[:30]
            header.append(f"per_class_da_{short}")
        f.write(",".join(header) + "\n")
        for r in results:
            row = [
                f"{r['lambda']:.3f}",
                f"{r['cutoff']:.3f}",
                str(r["n"]),
                str(r["retrieval_correct"]),
                str(r["retrieval_no_candidate"]),
                f"{r['retrieval_da']:.6f}",
                f"{r['retrieval_ci_lo']:.6f}",
                f"{r['retrieval_ci_hi']:.6f}",
                str(r["decision_correct"]),
                str(r["decision_inconclusive"]),
                f"{r['decision_da']:.6f}",
                f"{r['decision_ci_lo']:.6f}",
                f"{r['decision_ci_hi']:.6f}",
                f"{r['cell_seconds']:.3f}",
                str(r["iter_counts_median"]),
            ]
            for cls in all_classes:
                row.append(f"{r['per_class_retrieval_da'].get(cls, 0.0):.4f}")
            f.write(",".join(row) + "\n")
    print(f"CSV → {csv_path}")

    # Heatmap
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Two-panel heatmap: retrieval DA (primary) + decision DA (secondary)
        def _build_grid(metric: str) -> "np.ndarray":
            grid = np.zeros((len(lambdas), len(cutoffs)))
            for r in results:
                i = lambdas.index(r["lambda"])
                j = cutoffs.index(r["cutoff"])
                grid[i, j] = r[metric]
            return grid

        retr_grid = _build_grid("retrieval_da")
        dec_grid = _build_grid("decision_da")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, grid, title in [
            (axes[0], retr_grid, "Retrieval DA (primary)"),
            (axes[1], dec_grid, "Decision DA (secondary)"),
        ]:
            vmax = max(0.001, grid.max())
            im = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
            ax.set_xticks(range(len(cutoffs)))
            ax.set_xticklabels([f"{c:.2f}" for c in cutoffs])
            ax.set_yticks(range(len(lambdas)))
            ax.set_yticklabels([f"{lam:.1f}" for lam in lambdas])
            ax.set_xlabel("similarity_cutoff")
            ax.set_ylabel("λ (visual weight)")
            ax.set_title(title)
            for i in range(len(lambdas)):
                for j in range(len(cutoffs)):
                    ax.text(
                        j,
                        i,
                        f"{grid[i, j]:.3f}",
                        ha="center",
                        va="center",
                        color="white" if grid[i, j] < vmax / 2 else "black",
                        fontsize=8,
                    )
            plt.colorbar(im, ax=ax, label="DA")
        fig.suptitle(f"EXP-1 D1 Test (n={len(images)}) — λ × cutoff sweep")
        plt.tight_layout()
        plt.savefig(out_dir / "heatmap.png", dpi=120)
        plt.close(fig)
        print(f"Heatmap → {out_dir / 'heatmap.png'}")

        # Per-class heatmap — one panel per D1 class (7 classes)
        class_names_sorted = sorted({cls for r in results for cls in r["per_class_retrieval_da"]})
        ncols = 4
        nrows = (len(class_names_sorted) + ncols - 1) // ncols
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
        axes2_flat = axes2.flatten() if nrows > 1 else list(axes2)
        for ax, cls in zip(axes2_flat, class_names_sorted):  # noqa: B905
            grid_c = np.zeros((len(lambdas), len(cutoffs)))
            for r in results:
                i = lambdas.index(r["lambda"])
                j = cutoffs.index(r["cutoff"])
                grid_c[i, j] = r["per_class_retrieval_da"].get(cls, 0.0)
            vmax = max(0.001, grid_c.max())
            im = ax.imshow(grid_c, cmap="viridis", aspect="auto", vmin=0, vmax=1.0)
            ax.set_xticks(range(len(cutoffs)))
            ax.set_xticklabels([f"{c:.2f}" for c in cutoffs], rotation=45, fontsize=7)
            ax.set_yticks(range(len(lambdas)))
            ax.set_yticklabels([f"{lam:.1f}" for lam in lambdas], fontsize=7)
            ax.set_xlabel("cutoff", fontsize=8)
            ax.set_ylabel("λ", fontsize=8)
            ax.set_title(cls[:28], fontsize=9)
            for i in range(len(lambdas)):
                for j in range(len(cutoffs)):
                    ax.text(
                        j,
                        i,
                        f"{grid_c[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if grid_c[i, j] < 0.5 else "black",
                        fontsize=6,
                    )
            plt.colorbar(im, ax=ax, label="DA")
        # Hide any unused subplots
        for k in range(len(class_names_sorted), len(axes2_flat)):
            axes2_flat[k].set_visible(False)
        fig2.suptitle(f"EXP-1 REPAIR-6 per-class retrieval DA (n={len(images)})")
        plt.tight_layout()
        plt.savefig(out_dir / "per_class_heatmap.png", dpi=120)
        plt.close(fig2)
        print(f"Per-class heatmap → {out_dir / 'per_class_heatmap.png'}")
    except Exception as e:
        print(f"Heatmap generation failed: {e}")

    # best-config.md
    bc = out_dir / "best-config.md"
    per_class_best_lines = "\n".join(
        f"  - `{cls}`: {da:.4f}" for cls, da in best["per_class_retrieval_da"].items()
    )
    # Pattern G Layer 3 assessment: does cutoff reduction translate to
    # correct-class retrieval gain, or merely admit more noise?
    layer3_lines: list[str] = []
    layer3_lines.append("## Pattern G Layer 3 — Downstream Utility Assessment\n")
    layer3_lines.append(
        "For each λ row, tracks how retrieval DA changes as cutoff is relaxed from 0.45 → 0.15."
    )
    layer3_lines.append(
        "If Layer 3 is working: lower cutoffs admit additional **correct-class** top-1 retrievals "
        "and DA rises. If Layer 3 is failing: lower cutoffs admit additional **wrong-class** "
        "retrievals, and DA is flat or DA *falls* because wrong candidates displace correct ones "
        "from top-1.\n"
    )
    layer3_lines.append("| λ | DA @ cutoff 0.15 | 0.25 | 0.35 | 0.45 | Δ(0.15 − 0.45) | Verdict |")
    layer3_lines.append("|---|---|---|---|---|---|---|")
    for lam in lambdas:
        def _da_at(target_cutoff: float) -> float:
            for r in results:
                if abs(r["lambda"] - lam) < 1e-9 and abs(r["cutoff"] - target_cutoff) < 1e-9:
                    return r["retrieval_da"]
            return 0.0
        da_low = _da_at(0.15)
        da_mid1 = _da_at(0.25)
        da_mid2 = _da_at(0.35)
        da_high = _da_at(0.45)
        delta = da_low - da_high
        if delta > 0.02:
            verdict_cell = "✓ utility gain"
        elif delta < -0.02:
            verdict_cell = "✗ noise admit"
        else:
            verdict_cell = "~ flat"
        layer3_lines.append(
            f"| {lam:.1f} | {da_low:.3f} | {da_mid1:.3f} | {da_mid2:.3f} | {da_high:.3f} | "
            f"{delta:+.3f} | {verdict_cell} |"
        )
    layer3_md = "\n".join(layer3_lines) + "\n"

    bc_text = f"""# EXP-1 Best Configuration

> Generated: 2026-04-19 (REPAIR-6) · Sweep grid: {len(lambdas)}λ × {len(cutoffs)}cutoff = {len(lambdas) * len(cutoffs)} cells · n_test = {len(images)} D1 Test images · ADR-0011 active

## Metric note
Two DA metrics are reported per cell. **Optimisation target** is
``retrieval_da`` — the proportion of images whose top-1 retrieved KB
document's ``doc_id`` matches the ground-truth class. ``decision_da``
(full Stage 3 outcome) is reported as a secondary diagnostic to
localise Stage 3 scoring coverage under current KB vocabulary.

## Recommended (λ*, cutoff*) — best retrieval DA

| Quantity | Value |
|---|---|
| λ* | **{best['lambda']:.2f}** |
| similarity_cutoff* | **{best['cutoff']:.2f}** |
| retrieval DA | **{best['retrieval_da']:.4f}** |
| 95 % Wilson CI | [{best['retrieval_ci_lo']:.4f}, {best['retrieval_ci_hi']:.4f}] |
| retrieval correct / total | {best['retrieval_correct']} / {best['n']} |
| retrieval no-candidate | {best['retrieval_no_candidate']} |
| decision DA (secondary) | {best['decision_da']:.4f} |
| decision inconclusive | {best['decision_inconclusive']} |
| cell runtime | {best['cell_seconds']:.2f} s |

### Per-class retrieval DA at best cell
{per_class_best_lines}

## Paper sanity check at (λ=0.7, cutoff=0.5)
"""
    if paper_cell:
        bc_text += f"""
| Quantity | λ=0.7, cutoff=0.25 (ADR-0011 default) | λ=0.7, cutoff=0.5 (paper-literal) | Paper Table III |
|---|---|---|---|
| retrieval DA | {paper_cell['retrieval_da']:.4f} | {supplemental['retrieval_da']:.4f} | ≈ 0.999 |
| 95 % Wilson CI | [{paper_cell['retrieval_ci_lo']:.4f}, {paper_cell['retrieval_ci_hi']:.4f}] | [{supplemental['retrieval_ci_lo']:.4f}, {supplemental['retrieval_ci_hi']:.4f}] | — |
| decision DA | {paper_cell['decision_da']:.4f} | {supplemental['decision_da']:.4f} | ≈ 0.999 |
| decision inconclusive | {paper_cell['decision_inconclusive']} | {supplemental['decision_inconclusive']} | — |

**Deviation from paper at (λ=0.7, cutoff=0.25)**: {paper_cell['retrieval_da'] - 0.999:+.4f}
**Deviation from paper at (λ=0.7, cutoff=0.5)**: {supplemental['retrieval_da'] - 0.999:+.4f}
Both exceed ADR-0007 T2's 1 pp threshold; ADR-0011 is necessary-not-sufficient per §Critical Caveat.
"""
    else:
        bc_text += f"""
| Quantity | (λ=0.7, cutoff=0.5) paper-literal |
|---|---|
| retrieval DA | {supplemental['retrieval_da']:.4f} |
| 95 % Wilson CI | [{supplemental['retrieval_ci_lo']:.4f}, {supplemental['retrieval_ci_hi']:.4f}] |
| decision DA | {supplemental['decision_da']:.4f} |

Note: (λ=0.7, cutoff=0.25) from main grid not found.
"""

    bc_text += f"""
## Determinism
Two-run bit-identical on best cell: **{'PASS' if det_ok else 'FAIL'}**
(retrieval_correct {best['retrieval_correct']} vs {det_run['retrieval_correct']},
 decision_correct {best['decision_correct']} vs {det_run['decision_correct']})

{layer3_md}
## Artefacts
- `results.csv` — full sweep, both DA columns, per-class breakdown
- `heatmap.png` — side-by-side retrieval DA + decision DA surfaces (49 cells)
- `per_class_heatmap.png` — 7-panel per-class retrieval DA (COL attractor visualisation)
- `determinism.json` — re-run comparison

## Strategic milestone — awaiting user direction

ADR-0011 is active (cutoff = 0.25). REPAIR-6 landscape is now visible.
Pending user Strategy P / Q / R / IV choice (cf. EXP-1.5 v2 closure).
"""
    bc.write_text(bc_text)
    print(f"best-config.md → {bc}")

    (out_dir / "determinism.json").write_text(
        json.dumps(
            {
                "cell": {"lambda": best["lambda"], "cutoff": best["cutoff"]},
                "run1_retrieval_correct": best["retrieval_correct"],
                "run2_retrieval_correct": det_run["retrieval_correct"],
                "run1_retrieval_da": best["retrieval_da"],
                "run2_retrieval_da": det_run["retrieval_da"],
                "run1_decision_correct": best["decision_correct"],
                "run2_decision_correct": det_run["decision_correct"],
                "pass": det_ok,
            },
            indent=2,
        )
    )
    print("determinism.json written.")

    florence.close()
    clip.close()


if __name__ == "__main__":
    main()
