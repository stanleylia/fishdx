"""P'-5 — full 49-cell EXP-1 rerun under ADR-0012d image_gallery + Stage 3.

Architecture-corrected EXP-1: Stage 2 retrieves from the pre-built
``image_gallery`` ChromaCollection (1,747 D1 Train fused embeddings,
P'-3 baseline DA = 0.9986); Stage 3 looks up KB doc keyword tiers via
``top1.metadata['doc_id']``.

Reports both DA metrics per ADR-0012d kickoff:
  - retrieval_da : top-1 retrieval doc_id matches ground-truth class
  - decision_da  : full Stage 3 outcome matches ground-truth class
                   (Inconclusive counts as wrong)

Outputs
-------
``docs/m3/p_prime_5/`` :
  - results.csv        49 cells × {retrieval_da, decision_da, per-class}
  - heatmap.png        side-by-side retrieval + decision DA panels
  - per_class_heatmap.png  per-class retrieval DA grid
  - best-config.md     tier verdict + paper-cell comparison + decision dist
  - determinism.json   re-run on best cell
  - summary.json       machine-readable Pattern G Layer 4 report
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import chromadb
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from fishdx.config import load_config  # noqa: E402
from fishdx.kb.builder import parse_markdown_docs  # noqa: E402
from fishdx.perception.florence2 import Florence2Wrapper  # noqa: E402
from fishdx.perception.postprocess import clean_caption  # noqa: E402
from fishdx.retrieval.clip_encoder import OpenClipEmbedder  # noqa: E402
from fishdx.retrieval.fusion import fuse_and_normalize  # noqa: E402
from fishdx.retrieval.store import ChromaStore  # noqa: E402
from fishdx.retrieval.verification import verify_candidates  # noqa: E402
from fishdx.schemas import DecisionEnum  # noqa: E402
from fishdx.scoring.decision import make_decision  # noqa: E402
from fishdx.scoring.score import compute_s_d, compute_s_h  # noqa: E402

SEED = 42
LAMBDAS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
CUTOFFS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
SUPPLEMENTAL_PAPER_CUTOFF = 0.5
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
CLASS_TO_DOC_ID: dict[str, str] = {
    "Bacterial Red disease": "BRD",
    "Bacterial diseases - Aeromoniasis": "AER",
    "Bacterial gill disease": "COL",
    "Fungal diseases Saprolegniasis": "SAP",
    "Healthy Fish": "HLT",
    "Parasitic diseases": "PAR",
    "Viral diseases White tail disease": "VWT",
}


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    return (max(0.0, center - half), min(1.0, center + half))


def sweep_cell(
    cache: list[dict],
    lam: float,
    cutoff: float,
    store: ChromaStore,
    kb_by_id: dict,
    hlt_keywords: tuple[str, ...],
    cfg,
) -> dict:
    retrieval_correct = 0
    decision_correct = 0
    retrieval_no_candidate = 0
    decision_inconclusive = 0
    decision_dist: Counter = Counter()
    per_class_total: dict[str, int] = defaultdict(int)
    per_class_retrieval_correct: dict[str, int] = defaultdict(int)
    per_class_decision_correct: dict[str, int] = defaultdict(int)
    cell_t0 = time.perf_counter()

    for row in cache:
        true_class = row["true_class"]
        true_doc_id = CLASS_TO_DOC_ID[true_class]
        per_class_total[true_class] += 1

        e_final = fuse_and_normalize(row["e_visual"], row["e_caption"], lam)
        candidates = store.query(e_final, similarity_cutoff=cutoff)
        if candidates:
            grounding = [0.9] * len(candidates)
            verified, _iters, _ = verify_candidates(
                candidates, grounding, cfg.verification
            )
        else:
            verified = []

        top1 = verified[0] if verified else None
        top2 = verified[1] if len(verified) >= 2 else None

        # Retrieval DA — top1 metadata['doc_id'] matches GT
        if top1 is None:
            retrieval_no_candidate += 1
        else:
            top1_doc_id = top1.metadata.get("doc_id") or top1.doc_id
            if top1_doc_id == true_doc_id:
                retrieval_correct += 1
                per_class_retrieval_correct[true_class] += 1

        # Decision DA — full Stage 3
        # ADR-0012e: evidence 𝒪 = caption ∪ top-1 retrieved KB doc text
        kb_doc = None
        pred_doc_id_candidate: str | None = None
        if top1:
            top1_doc_id = top1.metadata.get("doc_id") or top1.doc_id
            kb_doc = kb_by_id.get(top1_doc_id)
            if kb_doc:
                pred_doc_id_candidate = top1_doc_id
        top1_text = kb_doc.text if kb_doc is not None else ""
        evidence = f"{row['caption']}\n{top1_text}" if top1_text else row["caption"]

        s_h = compute_s_h(
            evidence,
            cfg.scoring.healthy_weights,
            cfg.negation,
            healthy_keywords=hlt_keywords,
        )
        s_d = 0
        disease_class = None
        if kb_doc is not None:
            s_d = compute_s_d(
                evidence,
                cfg.scoring,
                confirmed_keywords=tuple(kb_doc.clinical_keywords.get("confirmed", [])),
                suspected_keywords=tuple(kb_doc.clinical_keywords.get("suspected", [])),
                mentioned_keywords=tuple(kb_doc.clinical_keywords.get("mentioned", [])),
            )
            disease_class = kb_doc.disease_class

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
        decision_dist[result.decision.value] += 1

        if result.decision is DecisionEnum.HEALTHY:
            decision_pred = "HLT"
        elif result.decision is DecisionEnum.DISEASE:
            decision_pred = pred_doc_id_candidate
        else:
            decision_inconclusive += 1
            decision_pred = None
        if decision_pred and decision_pred == true_doc_id:
            decision_correct += 1
            per_class_decision_correct[true_class] += 1

    n = len(cache)
    return {
        "lambda": lam,
        "cutoff": cutoff,
        "n": n,
        "retrieval_correct": retrieval_correct,
        "retrieval_no_candidate": retrieval_no_candidate,
        "retrieval_da": retrieval_correct / n if n else 0.0,
        "retrieval_ci_lo": wilson_ci(retrieval_correct, n)[0],
        "retrieval_ci_hi": wilson_ci(retrieval_correct, n)[1],
        "decision_correct": decision_correct,
        "decision_inconclusive": decision_inconclusive,
        "decision_da": decision_correct / n if n else 0.0,
        "decision_ci_lo": wilson_ci(decision_correct, n)[0],
        "decision_ci_hi": wilson_ci(decision_correct, n)[1],
        "decision_dist": dict(decision_dist),
        "per_class_retrieval_da": {
            cls: per_class_retrieval_correct[cls] / per_class_total[cls]
            for cls in sorted(per_class_total)
        },
        "per_class_decision_da": {
            cls: per_class_decision_correct[cls] / per_class_total[cls]
            for cls in sorted(per_class_total)
        },
        "cell_seconds": time.perf_counter() - cell_t0,
    }


def main() -> None:
    import os

    out_dir = REPO / os.environ.get("FISHDX_OUT_DIR", "docs/m3/p_prime_5")
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    cfg = load_config(REPO / "configs/default.yaml")
    print(
        f"config: λ*={cfg.fusion.lambda_star} cutoff={cfg.retrieval.similarity_cutoff} "
        f"retrieval.collection={cfg.retrieval.collection} "
        f"scoring.keyword_dict={cfg.scoring.keyword_dict}"
    )

    test_root = (
        REPO
        / "lab_dateset/external_datasets/fish_disease_south_asia"
        / "Freshwater Fish Disease Aquaculture in south asia/Test"
    )
    images = sorted(
        (p, p.parent.name)
        for p in test_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.parent.name in CLASS_TO_DOC_ID
    )
    print(f"D1 Test: {len(images)} images")

    # Image-gallery persistence directory.
    # Override via FISHDX_GALLERY_PERSIST environment variable; defaults to a
    # generic /tmp path. Build the gallery first via experiments/p_prime_3_gallery.py
    # or scripts/import_seed_data_fusion.py before running this evaluation.
    import os as _os
    gallery_persist = _os.environ.get("FISHDX_GALLERY_PERSIST", "/tmp/fishdx_image_gallery")
    print(f"image_gallery from {gallery_persist}")
    chroma = chromadb.PersistentClient(path=gallery_persist)

    florence = Florence2Wrapper(cfg.florence2)
    clip = OpenClipEmbedder(cfg.clip)
    print("[warmup] Florence-2 + OpenCLIP")
    florence.warmup()
    clip.warmup()

    store = ChromaStore(
        chroma_client=chroma,
        collection_name=cfg.retrieval.collection,
        retrieval_config=cfg.retrieval,
        expected_source_type="fused",
    )
    store.warmup()
    print("[Layer 4 check] image_gallery embedding_source_type=fused → PASS")

    kb_docs = parse_markdown_docs(REPO / "src/fishdx/kb/documents")
    kb_by_id = {d.doc_id: d for d in kb_docs}
    hlt = kb_by_id.get("HLT")
    hlt_keywords = tuple(hlt.healthy_keywords) if hlt else ("healthy",)

    # Precompute Stage 1 + CLIP per image (once)
    print(f"\n[precompute] Stage 1 + CLIP per image (n={len(images)})")
    t_pre = time.perf_counter()
    cache: list[dict] = []
    log_step = max(1, len(images) // 10)
    for idx, (path, cls) in enumerate(images):
        raw_cap, _, _ = florence.caption_and_detect(path)
        cap = clean_caption(raw_cap)
        text_for_clip = cap if cap else " "
        e_caption = clip.encode_text([text_for_clip])[0]
        e_visual = clip.encode_image_paths([str(path)])[0]
        cache.append(
            {
                "path": str(path),
                "true_class": cls,
                "caption": cap,
                "e_visual": e_visual,
                "e_caption": e_caption,
            }
        )
        if (idx + 1) % log_step == 0 or idx + 1 == len(images):
            print(f"  [{idx + 1}/{len(images)}] {time.perf_counter() - t_pre:.1f}s")
    print(f"[precompute] total {time.perf_counter() - t_pre:.1f}s")

    # Sweep
    print(f"\n[sweep] {len(LAMBDAS)}×{len(CUTOFFS)} = {len(LAMBDAS) * len(CUTOFFS)} cells")
    results: list[dict] = []
    sweep_t0 = time.perf_counter()
    for lam in LAMBDAS:
        for cutoff in CUTOFFS:
            r = sweep_cell(cache, lam, cutoff, store, kb_by_id, hlt_keywords, cfg)
            results.append(r)
            print(
                f"  λ={lam:.2f} cutoff={cutoff:.2f}  "
                f"retr_DA={r['retrieval_da']:.4f} "
                f"[{r['retrieval_ci_lo']:.3f},{r['retrieval_ci_hi']:.3f}]  "
                f"dec_DA={r['decision_da']:.4f} "
                f"(inc={r['decision_inconclusive']})  {r['cell_seconds']:.1f}s"
            )
    print(f"[sweep] total {time.perf_counter() - sweep_t0:.1f}s")

    # Supplemental paper-literal cell
    print(f"\n[supplemental] (λ=0.7, cutoff={SUPPLEMENTAL_PAPER_CUTOFF}) paper-literal")
    sup = sweep_cell(cache, 0.7, SUPPLEMENTAL_PAPER_CUTOFF, store, kb_by_id, hlt_keywords, cfg)
    print(
        f"  retr_DA={sup['retrieval_da']:.4f} "
        f"[{sup['retrieval_ci_lo']:.3f},{sup['retrieval_ci_hi']:.3f}]  "
        f"dec_DA={sup['decision_da']:.4f}"
    )

    best = max(results, key=lambda r: r["retrieval_da"])
    print(f"\n[determinism] re-run best cell λ={best['lambda']} cutoff={best['cutoff']}")
    det = sweep_cell(cache, best["lambda"], best["cutoff"], store, kb_by_id, hlt_keywords, cfg)
    det_ok = (
        det["retrieval_correct"] == best["retrieval_correct"]
        and abs(det["retrieval_da"] - best["retrieval_da"]) < 1e-12
        and det["decision_correct"] == best["decision_correct"]
    )
    print(f"  determinism: {'PASS' if det_ok else 'FAIL'}")

    # CSV
    all_classes = sorted({c for r in results for c in r["per_class_retrieval_da"]})
    csv_path = out_dir / "results.csv"
    with csv_path.open("w") as f:
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
        ]
        for cls in all_classes:
            short = cls.replace(" ", "_").replace("-", "")[:30]
            header.append(f"retr_da_{short}")
        for cls in all_classes:
            short = cls.replace(" ", "_").replace("-", "")[:30]
            header.append(f"dec_da_{short}")
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
            ]
            for cls in all_classes:
                row.append(f"{r['per_class_retrieval_da'].get(cls, 0.0):.4f}")
            for cls in all_classes:
                row.append(f"{r['per_class_decision_da'].get(cls, 0.0):.4f}")
            f.write(",".join(row) + "\n")
    print(f"CSV → {csv_path}")

    # Heatmaps
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def _grid(metric: str) -> "np.ndarray":
            g = np.zeros((len(LAMBDAS), len(CUTOFFS)))
            for r in results:
                i = LAMBDAS.index(r["lambda"])
                j = CUTOFFS.index(r["cutoff"])
                g[i, j] = r[metric]
            return g

        retr_grid = _grid("retrieval_da")
        dec_grid = _grid("decision_da")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, grid, title in [
            (axes[0], retr_grid, "Retrieval DA (image_gallery)"),
            (axes[1], dec_grid, "Decision DA (full pipeline)"),
        ]:
            vmax = max(0.001, grid.max())
            im = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=0, vmax=1.0)
            ax.set_xticks(range(len(CUTOFFS)))
            ax.set_xticklabels([f"{c:.2f}" for c in CUTOFFS])
            ax.set_yticks(range(len(LAMBDAS)))
            ax.set_yticklabels([f"{lam:.1f}" for lam in LAMBDAS])
            ax.set_xlabel("similarity_cutoff")
            ax.set_ylabel("λ (visual weight)")
            ax.set_title(title)
            for i in range(len(LAMBDAS)):
                for j in range(len(CUTOFFS)):
                    ax.text(
                        j,
                        i,
                        f"{grid[i, j]:.3f}",
                        ha="center",
                        va="center",
                        color="white" if grid[i, j] < 0.5 else "black",
                        fontsize=8,
                    )
            plt.colorbar(im, ax=ax, label="DA")
        fig.suptitle(f"P'-5 EXP-1 D1 Test (n={len(images)}) — image_gallery dual-collection")
        plt.tight_layout()
        plt.savefig(out_dir / "heatmap.png", dpi=120)
        plt.close(fig)
        print(f"heatmap.png → {out_dir / 'heatmap.png'}")
    except Exception as e:
        print(f"heatmap failed: {e}")

    # Find paper cells
    paper_025 = next(
        (r for r in results if abs(r["lambda"] - 0.7) < 1e-9 and abs(r["cutoff"] - 0.25) < 1e-9),
        None,
    )

    # best-config.md
    bc_lines = [
        "# P'-5 Best Configuration — image_gallery EXP-1 rerun\n",
        f"> Generated 2026-04-19 · Sweep grid: {len(LAMBDAS)}λ × {len(CUTOFFS)}cutoff = "
        f"{len(LAMBDAS) * len(CUTOFFS)} cells · n_test = {len(images)} D1 Test images\n",
        "> Stage 2 retrieval: ADR-0012d image_gallery (1,747 D1 Train fused)\n",
        "> Stage 3 scoring: in-memory KB doc lookup via top-1 metadata['doc_id']\n",
        "",
        "## Best cell (retrieval_da)\n",
        "| Quantity | Value |",
        "|---|---|",
        f"| λ* | **{best['lambda']:.2f}** |",
        f"| cutoff* | **{best['cutoff']:.2f}** |",
        f"| retrieval DA | **{best['retrieval_da']:.4f}** |",
        f"| 95 % Wilson CI | [{best['retrieval_ci_lo']:.4f}, {best['retrieval_ci_hi']:.4f}] |",
        f"| retrieval correct / total | {best['retrieval_correct']} / {best['n']} |",
        f"| decision DA (secondary) | {best['decision_da']:.4f} |",
        f"| decision inconclusive | {best['decision_inconclusive']} |",
        f"| decision distribution | {best['decision_dist']} |",
        f"| cell runtime | {best['cell_seconds']:.2f} s |",
        "",
        "## Per-class retrieval DA at best cell\n",
    ]
    for cls in sorted(best["per_class_retrieval_da"]):
        bc_lines.append(f"  - `{cls}`: {best['per_class_retrieval_da'][cls]:.4f}")
    bc_lines.append("")
    bc_lines.append("## Paper cells comparison\n")
    bc_lines.append(
        "| Quantity | (λ=0.7, cutoff=0.25) ADR-0011 | (λ=0.7, cutoff=0.5) paper-literal | Paper |"
    )
    bc_lines.append("|---|---|---|---|")
    p25_retr = paper_025["retrieval_da"] if paper_025 else "n/a"
    p25_dec = paper_025["decision_da"] if paper_025 else "n/a"
    bc_lines.append(
        f"| retrieval DA | {p25_retr:.4f} | {sup['retrieval_da']:.4f} | ≈ 1.000 (paper Table V CLIP kNN) |"
    )
    bc_lines.append(
        f"| decision DA | {p25_dec:.4f} | {sup['decision_da']:.4f} | ≈ 0.999 (paper Table III) |"
    )
    bc_lines.append(
        f"| inconclusive | {paper_025['decision_inconclusive'] if paper_025 else 'n/a'} | {sup['decision_inconclusive']} | — |"
    )
    bc_lines.append("")
    bc_lines.append("## Determinism\n")
    bc_lines.append(
        f"Best-cell two-run: **{'PASS' if det_ok else 'FAIL'}** "
        f"(retr {best['retrieval_correct']}={det['retrieval_correct']}, "
        f"dec {best['decision_correct']}={det['decision_correct']})"
    )
    bc_lines.append("")
    bc_lines.append("## Pattern G Layer 4 end-to-end")
    bc_lines.append(
        f"- image_gallery `embedding_source_type=fused` verified at warmup → PASS"
    )
    bc_lines.append(
        f"- Architecture-corrected pipeline successfully retrieves to ≥ {best['retrieval_da']:.2f} retrieval DA"
    )
    bc_lines.append(
        f"- Strategy P' success criterion (best DA ≥ 0.90): "
        f"**{'MET' if best['retrieval_da'] >= 0.90 else 'NOT MET'}**"
    )
    (out_dir / "best-config.md").write_text("\n".join(bc_lines))
    print(f"best-config.md → {out_dir / 'best-config.md'}")

    (out_dir / "determinism.json").write_text(
        json.dumps(
            {
                "cell": {"lambda": best["lambda"], "cutoff": best["cutoff"]},
                "run1_retrieval_correct": best["retrieval_correct"],
                "run2_retrieval_correct": det["retrieval_correct"],
                "run1_retrieval_da": best["retrieval_da"],
                "run2_retrieval_da": det["retrieval_da"],
                "run1_decision_correct": best["decision_correct"],
                "run2_decision_correct": det["decision_correct"],
                "pass": det_ok,
            },
            indent=2,
        )
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "best": {
                    "lambda": best["lambda"],
                    "cutoff": best["cutoff"],
                    "retrieval_da": best["retrieval_da"],
                    "retrieval_ci": [best["retrieval_ci_lo"], best["retrieval_ci_hi"]],
                    "decision_da": best["decision_da"],
                    "per_class_retrieval_da": best["per_class_retrieval_da"],
                    "per_class_decision_da": best["per_class_decision_da"],
                    "decision_dist": best["decision_dist"],
                },
                "paper_cell_025": {
                    "retrieval_da": paper_025["retrieval_da"] if paper_025 else None,
                    "decision_da": paper_025["decision_da"] if paper_025 else None,
                    "decision_inconclusive": paper_025["decision_inconclusive"] if paper_025 else None,
                },
                "paper_cell_05_supplemental": {
                    "retrieval_da": sup["retrieval_da"],
                    "decision_da": sup["decision_da"],
                },
                "strategy_p_prime_success": best["retrieval_da"] >= 0.90,
                "determinism_pass": det_ok,
            },
            indent=2,
        )
    )

    florence.close()
    clip.close()


if __name__ == "__main__":
    main()
