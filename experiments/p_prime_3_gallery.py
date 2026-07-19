"""P'-3 runner — build image_gallery + Layer 4 kNN k=1 pre-validation.

Executes ADR-0012d §Validation Plan P'-3 section:
  Step 1-2: build 1,747 D1 Train fusion-embedding gallery at λ=0.7
  Step 3:   CLIP kNN k=1 on 697 D1 Test → per-class DA + Wilson CI
  Step 4:   tier-assess (Optimal / Acceptable / Marginal / Failure)
  Step 5:   write docs/m3/p_prime_3/gallery_build.md + layer4_prevalidation.md

Runs foreground. Expected ~12–20 min on RTX 3070 with ADR-0008 sub-budget.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import chromadb
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from fishdx.config import load_config  # noqa: E402
from fishdx.kb.image_gallery_builder import (  # noqa: E402
    GalleryImage,
    build_image_gallery,
)
from fishdx.perception.florence2 import Florence2Wrapper  # noqa: E402
from fishdx.perception.postprocess import clean_caption  # noqa: E402
from fishdx.retrieval.clip_encoder import OpenClipEmbedder  # noqa: E402
from fishdx.retrieval.fusion import fuse_and_normalize  # noqa: E402

SEED = 42
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


def list_class_images(root: Path) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    for cls_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        if cls_dir.name not in CLASS_TO_DOC_ID:
            continue
        files = sorted(
            p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        out.extend((p, cls_dir.name) for p in files)
    return out


def precompute_test_embeddings(
    test_images: list[tuple[Path, str]],
    florence: Florence2Wrapper,
    clip: OpenClipEmbedder,
    fusion_lambda: float,
    verbose: bool = True,
) -> list[dict]:
    rows: list[dict] = []
    n = len(test_images)
    log_step = max(1, n // 10)
    t0 = time.perf_counter()
    for idx, (path, cls) in enumerate(test_images):
        raw_cap, _, _ = florence.caption_and_detect(path)
        cap = clean_caption(raw_cap)
        text_for_clip = cap if cap else " "
        e_caption = clip.encode_text([text_for_clip])[0]
        e_visual = clip.encode_image_paths([str(path)])[0]
        e_final = fuse_and_normalize(e_visual, e_caption, fusion_lambda)
        rows.append(
            {
                "query_id": f"D1_Test/{cls}/{path.name}",
                "true_class": cls,
                "true_doc_id": CLASS_TO_DOC_ID[cls],
                "e_final": e_final,
            }
        )
        if verbose and ((idx + 1) % log_step == 0 or idx + 1 == n):
            elapsed = time.perf_counter() - t0
            print(f"  [test {idx + 1}/{n}] encoded in {elapsed:.1f}s")
    return rows


def knn_eval(test_rows: list[dict], gallery_collection) -> dict:
    correct = 0
    per_class_total: dict[str, int] = defaultdict(int)
    per_class_correct: dict[str, int] = defaultdict(int)
    top1_doc_counts: dict[str, int] = defaultdict(int)
    for row in test_rows:
        per_class_total[row["true_class"]] += 1
        result = gallery_collection.query(query_embeddings=[row["e_final"]], n_results=1)
        metadatas = result.get("metadatas", [[]])[0]
        if not metadatas:
            continue
        top1 = metadatas[0]
        pred_doc_id = top1.get("doc_id")
        top1_doc_counts[pred_doc_id] += 1
        if pred_doc_id == row["true_doc_id"]:
            correct += 1
            per_class_correct[row["true_class"]] += 1
    n = len(test_rows)
    ci_lo, ci_hi = wilson_ci(correct, n)
    per_class_da = {
        cls: per_class_correct[cls] / per_class_total[cls] if per_class_total[cls] else 0.0
        for cls in per_class_total
    }
    return {
        "n": n,
        "n_correct": correct,
        "da": correct / n if n else 0.0,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "per_class_da": per_class_da,
        "per_class_total": dict(per_class_total),
        "per_class_correct": dict(per_class_correct),
        "top1_doc_distribution": dict(top1_doc_counts),
    }


def tier_assess(da: float) -> tuple[str, str]:
    if da >= 0.95:
        return ("OPTIMAL", "Architecture fidelity confirmed. Proceed P'-4 directly.")
    if da >= 0.80:
        return (
            "ACCEPTABLE",
            "Proceed P'-4 with caveat in Layer 4 assessment report. Document gap to paper 1.000.",
        )
    if da >= 0.70:
        return (
            "MARGINAL",
            "HALT before P'-4. Investigate: (a) fusion bit-identity (b) CLIP preprocessing (c) caption quality (d) Pareidolia consistency.",
        )
    return (
        "FAILURE",
        "T1 HALT. Escalate to user; likely 10th consulting-layer gap candidate.",
    )


def main() -> None:
    out_dir = REPO / "docs/m3/p_prime_3"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    cfg = load_config(REPO / "configs/default.yaml")
    print(
        f"config: λ*={cfg.fusion.lambda_star} cutoff={cfg.retrieval.similarity_cutoff} "
        f"retrieval.collection={cfg.retrieval.collection}"
    )

    train_root = (
        REPO
        / "lab_dateset/external_datasets/fish_disease_south_asia"
        / "Freshwater Fish Disease Aquaculture in south asia/Train"
    )
    test_root = train_root.parent / "Test"

    train_images = list_class_images(train_root)
    test_images = list_class_images(test_root)
    print(f"D1 Train: {len(train_images)} | D1 Test: {len(test_images)}")

    # warmup
    print("\n[warmup] Florence-2 + OpenCLIP")
    florence = Florence2Wrapper(cfg.florence2)
    clip = OpenClipEmbedder(cfg.clip)
    t0 = time.perf_counter()
    florence.warmup()
    clip.warmup()
    print(f"  warmup {time.perf_counter() - t0:.1f}s")

    # ChromaDB persistent at fresh tmp dir (deterministic baseline)
    tmp_chroma = tempfile.mkdtemp(prefix="p_prime_3_")
    chroma = chromadb.PersistentClient(path=tmp_chroma)

    # ── Step 1-2: Gallery construction ──────────────────────────────
    print(
        f"\n[P'-3 Step 1-2] Building image_gallery ({len(train_images)} D1 Train) at λ={cfg.fusion.lambda_star}"
    )
    gallery_items: list[GalleryImage] = []
    for p, cls in train_images:
        doc_id = CLASS_TO_DOC_ID[cls]
        gallery_items.append(
            GalleryImage(
                image_id=f"D1_Train/{cls}/{p.name}",
                image_path=p,
                true_class=cls,
                doc_id=doc_id,
            )
        )
    receipt = build_image_gallery(
        gallery_items,
        florence=florence,
        clip=clip,
        chroma_client=chroma,
        collection_name=cfg.retrieval.collection,
        fusion_lambda=cfg.fusion.lambda_star,
        created_under_adr="0012d",
        seed=SEED,
        batch_size=16,
    )
    print(
        f"  gallery receipt: n={receipt.n_images_ingested}/{receipt.n_images_requested}  "
        f"empty_cap={receipt.n_caption_empty}  VRAM peak {receipt.vram_peak_mb:.0f} MB  "
        f"{receipt.duration_seconds:.1f}s"
    )
    print(f"  per-class: {dict(sorted(receipt.per_class_counts.items()))}")

    # ── Step 3: Layer 4 pre-validation ─────────────────────────────
    print(
        f"\n[P'-3 Step 3] Precompute D1 Test embeddings ({len(test_images)} at λ={cfg.fusion.lambda_star})"
    )
    test_rows = precompute_test_embeddings(test_images, florence, clip, cfg.fusion.lambda_star)

    gallery = chroma.get_collection(cfg.retrieval.collection)
    print(f"\n[P'-3 Step 3] Run 1: CLIP kNN k=1 on {len(test_rows)} D1 Test")
    t0 = time.perf_counter()
    run1 = knn_eval(test_rows, gallery)
    run1["duration_seconds"] = time.perf_counter() - t0
    print(
        f"  run1: DA={run1['da']:.4f}  [{run1['ci_lo']:.4f}, {run1['ci_hi']:.4f}]  "
        f"{run1['n_correct']}/{run1['n']}  {run1['duration_seconds']:.1f}s"
    )

    print("\n[P'-3 Step 3 determinism] Run 2: re-eval same test set")
    t0 = time.perf_counter()
    run2 = knn_eval(test_rows, gallery)
    run2["duration_seconds"] = time.perf_counter() - t0
    det_ok = (
        run1["n_correct"] == run2["n_correct"]
        and abs(run1["da"] - run2["da"]) < 1e-12
        and run1["per_class_correct"] == run2["per_class_correct"]
    )
    print(f"  run2: DA={run2['da']:.4f}  determinism={'PASS' if det_ok else 'FAIL'}")

    # ── Step 4: Tier assessment ─────────────────────────────────────
    tier, tier_message = tier_assess(run1["da"])
    print(f"\n[P'-3 Step 4] Tier verdict: **{tier}** — {tier_message}")

    # ── Step 5: Reports ────────────────────────────────────────────
    gallery_md = (
        f"""# P'-3 Image Gallery Construction Report

> Generated 2026-04-19 · ADR-0012d · seed={SEED}

## Configuration
- `retrieval.collection`: `{cfg.retrieval.collection}`
- `fusion.lambda_star`: {cfg.fusion.lambda_star}
- `clip.architecture`: {cfg.clip.architecture}
- `clip.pretrained`: {cfg.clip.pretrained}
- `florence2.revision`: {cfg.florence2.revision[:16]}…

## Source
- D1 Train root: `{train_root.relative_to(REPO)}`
- n (requested): **{receipt.n_images_requested}**
- n (ingested): **{receipt.n_images_ingested}**
- captions empty after `clean_caption`: {receipt.n_caption_empty} ({receipt.n_caption_empty / receipt.n_images_ingested:.1%})

## Per-class ingest counts
| class | count |
|---|---|
"""
        + "\n".join(f"| {cls} | {n} |" for cls, n in sorted(receipt.per_class_counts.items()))
        + f"""

## Runtime + resource
- duration: **{receipt.duration_seconds:.1f} s** ({receipt.duration_seconds / 60:.1f} min)
- VRAM peak: **{receipt.vram_peak_mb:.0f} MB** ({receipt.vram_peak_mb / 1024:.2f} GB)
- ADR-0008 sub-budget ≤ 6 GB: **{"PASS" if receipt.vram_peak_mb / 1024 <= 6.0 else "FAIL"}**
- ADR-0012d T3 (runtime ≤ 45 min): **{"PASS" if receipt.duration_seconds <= 2700 else "FAIL"}**

## Collection metadata
- `embedding_source_type`: "fused"
- `fusion_lambda`: {receipt.fusion_lambda}
- `created_under_adr`: {receipt.created_under_adr}
- `seed`: {SEED}
- HNSW: M=16, ef_construction=200, ef_search=100, num_threads=1

## Determinism
- Seed: {SEED}
- Florence-2 beam=3, do_sample=False
- CLIP preprocessing deterministic
- Gallery build is idempotent (ChromaDB `upsert`)
- Layer 4 determinism verified at Step 3 (see layer4_prevalidation.md)
"""
    )
    (out_dir / "gallery_build.md").write_text(gallery_md)
    print(f"  report: {out_dir / 'gallery_build.md'}")

    layer4_md = (
        f"""# P'-3 Pattern G Layer 4 Pre-validation Report

> Generated 2026-04-19 · ADR-0012d §Pattern G Layer 4 Acceptance Tiers

## Result: **{tier}** (DA = {run1['da']:.4f})

{tier_message}

## CLIP kNN k=1 on D1 Test (n={run1['n']})

| Quantity | Value |
|---|---|
| DA | **{run1['da']:.4f}** |
| 95 % Wilson CI | [{run1['ci_lo']:.4f}, {run1['ci_hi']:.4f}] |
| n correct / total | {run1['n_correct']} / {run1['n']} |
| Paper Table V expectation | 1.000 |
| Deviation from paper | {run1['da'] - 1.000:+.4f} |
| Run duration | {run1['duration_seconds']:.1f} s |

## Tier assessment vs ADR-0012d threshold table

| Tier | Range | Our DA is in range? |
|---|---|---|
| Optimal | DA ≥ 0.95 | {"✓" if run1['da'] >= 0.95 else "✗"} |
| Acceptable | 0.80 ≤ DA < 0.95 | {"✓" if 0.80 <= run1['da'] < 0.95 else "✗"} |
| Marginal | 0.70 ≤ DA < 0.80 | {"✓" if 0.70 <= run1['da'] < 0.80 else "✗"} |
| Failure | DA < 0.70 | {"✓" if run1['da'] < 0.70 else "✗"} |

## Per-class DA
| class | DA | correct / total |
|---|---|---|
"""
        + "\n".join(
            f"| {cls} | {run1['per_class_da'].get(cls, 0.0):.4f} | "
            f"{run1['per_class_correct'].get(cls, 0)} / {run1['per_class_total'].get(cls, 0)} |"
            for cls in sorted(run1["per_class_total"])
        )
        + f"""

## Top-1 doc distribution (attractor check)
Uniform would be ~14 % per class. Concentrated into one class signals
an image-embedding attractor.

| doc_id | count | rate |
|---|---|---|
"""
        + "\n".join(
            f"| {d} | {c} | {c / run1['n']:.1%} |"
            for d, c in sorted(run1["top1_doc_distribution"].items(), key=lambda x: -x[1])
        )
        + f"""

## Determinism (two-run)
- Run 1 DA: {run1['da']:.6f} ({run1['n_correct']}/{run1['n']})
- Run 2 DA: {run2['da']:.6f} ({run2['n_correct']}/{run2['n']})
- Per-class identical: {"PASS" if run1['per_class_correct'] == run2['per_class_correct'] else "FAIL"}
- Overall determinism: **{"PASS" if det_ok else "FAIL"}**

## Predicted-vs-observed tier commentary

User's expected-scenario table from ADR-0012d kickoff:
- If DA ≥ 0.90: Paper reproduction SUCCESSFUL
- If 0.70 ≤ DA < 0.90: Paper reproduction PARTIAL
- If 0.40 ≤ DA < 0.70: Architecture helped but gap remains
- If DA < 0.40: Further investigation required

Observed: DA = **{run1['da']:.4f}** → tier **{tier}**.
"""
    )
    (out_dir / "layer4_prevalidation.md").write_text(layer4_md)
    print(f"  report: {out_dir / 'layer4_prevalidation.md'}")

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "gallery": {
                    "n_ingested": receipt.n_images_ingested,
                    "duration_s": receipt.duration_seconds,
                    "vram_peak_mb": receipt.vram_peak_mb,
                    "caption_empty": receipt.n_caption_empty,
                    "per_class_counts": receipt.per_class_counts,
                },
                "layer4": {
                    "tier": tier,
                    "tier_message": tier_message,
                    "run1_da": run1["da"],
                    "run1_ci_lo": run1["ci_lo"],
                    "run1_ci_hi": run1["ci_hi"],
                    "run1_correct": run1["n_correct"],
                    "n": run1["n"],
                    "per_class_da": run1["per_class_da"],
                    "top1_doc_distribution": run1["top1_doc_distribution"],
                    "determinism_pass": det_ok,
                },
            },
            indent=2,
        )
    )

    florence.close()
    clip.close()


if __name__ == "__main__":
    main()
