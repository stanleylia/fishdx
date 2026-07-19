"""EXP-1.5 v2 — 6-hypothesis diagnostic under REPAIR-1 captions.

Images: 35 stratified (per original EXP-1.5) ∪ 5 REPAIR-4 sample. Dedup
by path. Runs Stage 1 + CLIP encoding once per image; all hypothesis
tests derive from the per-image cache.

Hypotheses (v2 rewrites the v1 tests against real captions):
    H1 v2 — fraction of captions containing ≥ 1 KB keyword (any tier)
             from the CORRECT-class doc; threshold < 30 % → confirmed
    H2 v2 — KB-doc attractor quantification (COL centre hypothesis)
    H3 v2 — CLIP caption→correct-doc vs caption→other-doc separation
             gap; threshold < 0.05 → confirmed
    H4 v2 — fraction of caption tokens in clinical medical vocabulary;
             threshold < 5 % → confirmed
    H5 v2 — λ sensitivity on top-1 class correctness (10 images ×
             5 λ values)
    H6 v2 — best-achievable top-1 doc distribution; threshold
             COL ≥ 60 % → COL-attractor structural
"""

from __future__ import annotations

import json
import random
import re
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
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

SEED = 42
PER_CLASS = 5
CLASS_TO_DOC_ID: dict[str, str] = {
    "Bacterial Red disease": "BRD",
    "Bacterial diseases - Aeromoniasis": "AER",
    "Bacterial gill disease": "COL",
    "Fungal diseases Saprolegniasis": "SAP",
    "Healthy Fish": "HLT",
    "Parasitic diseases": "PAR",
    "Viral diseases White tail disease": "VWT",
}
LAMBDAS_H5 = [0.0, 0.3, 0.5, 0.7, 1.0]

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
H1_CONFIRM_THRESHOLD = 0.30
H3_CONFIRM_THRESHOLD = 0.05
H4_CONFIRM_THRESHOLD = 0.05
H6_CONFIRM_FRACTION = 0.60


def tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_images(test_root: Path) -> list[tuple[Path, str]]:
    """Reproduce EXP-1.5 v1 sample (35) + REPAIR-4 sample (5), dedup."""
    image_exts = {".jpg", ".jpeg", ".png"}
    rng = random.Random(SEED)
    sampled: list[tuple[Path, str]] = []
    class_dirs = sorted(d for d in test_root.iterdir() if d.is_dir())

    # EXP-1.5 v1 sample (35): 5 per class × 7 classes
    for cls_dir in class_dirs:
        files = sorted(
            p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts
        )
        chosen = rng.sample(files, min(PER_CLASS, len(files)))
        chosen.sort()
        for p in chosen:
            sampled.append((p, cls_dir.name))

    # REPAIR-4 sample (5): first 5 class_dirs, random.choice with seed=42
    rng2 = random.Random(SEED)
    for cls_dir in class_dirs[:5]:
        files = sorted(
            p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts
        )
        if files:
            sampled.append((rng2.choice(files), cls_dir.name))

    # Dedup preserving first occurrence
    seen = set()
    unique: list[tuple[Path, str]] = []
    for p, c in sampled:
        if p not in seen:
            seen.add(p)
            unique.append((p, c))
    return unique


def main() -> None:
    out_dir = REPO / "docs/m3/exp1_5_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    cfg = load_config(REPO / "configs/default.yaml")

    test_root = (
        REPO
        / "lab_dateset/external_datasets/fish_disease_south_asia"
        / "Freshwater Fish Disease Aquaculture in south asia/Test"
    )
    sample = sample_images(test_root)
    print(f"[sample] {len(sample)} unique images (35 EXP-1.5 ∪ 5 REPAIR-4)")

    florence = Florence2Wrapper(cfg.florence2)
    clip = OpenClipEmbedder(cfg.clip)
    florence.warmup()
    clip.warmup()

    # KB ingest
    tmp_kb = tempfile.mkdtemp()
    chroma = chromadb.PersistentClient(path=tmp_kb)
    kb_docs = parse_markdown_docs(REPO / "src/fishdx/kb/documents")
    builder = KBBuilder(
        chroma_client=chroma, embedder=clip, kb_config=cfg.kb, retrieval_config=cfg.retrieval
    )
    builder.ingest(kb_docs)
    store = ChromaStore(
        chroma_client=chroma,
        collection_name=cfg.kb.collection_name,
        retrieval_config=cfg.retrieval,
    )
    store.warmup()
    kb_by_id = {d.doc_id: d for d in kb_docs}
    kb_doc_ids_sorted = sorted(kb_by_id)

    # Keywords by doc, by tier, + flat
    per_doc_all_kws: dict[str, set[str]] = {}
    medical_vocab: set[str] = set()
    for d in kb_docs:
        doc_kws: set[str] = set()
        for tier_kws in d.clinical_keywords.values():
            for kw in tier_kws:
                doc_kws.add(kw.lower())
        for kw in d.healthy_keywords:
            doc_kws.add(kw.lower())
        per_doc_all_kws[d.doc_id] = doc_kws
        for kw in d.clinical_keywords.get("confirmed", []):
            medical_vocab.add(kw.lower())

    # Tokenise medical vocab to a per-token set for H4 v2
    medical_tokens: set[str] = set()
    for kw in medical_vocab:
        for t in tokens(kw):
            medical_tokens.add(t)

    # KB doc text embeddings for H2 / H3
    kb_text_embeddings = clip.encode_text([kb_by_id[did].text for did in kb_doc_ids_sorted])
    kb_emb = np.array(kb_text_embeddings, dtype=np.float64)
    kb_pairwise = kb_emb @ kb_emb.T

    # ── Per-image loop ────────────────────────────────────────────
    print("[run] Florence-2 + CLIP per image …")
    rows = []
    for i, (img_path, cls) in enumerate(sample):
        true_doc_id = CLASS_TO_DOC_ID[cls]
        caption_raw, _, _ = florence.caption_and_detect(img_path)
        caption = clean_caption(caption_raw)
        text_for_clip = caption if caption else " "
        e_caption = clip.encode_text([text_for_clip])[0]
        e_visual = clip.encode_image_paths([str(img_path)])[0]

        # Similarity of caption vs each KB doc (all 8)
        e_cap_np = np.array(e_caption)
        sim_cap_to_docs = {
            did: float(e_cap_np @ np.array(kb_text_embeddings[j]))
            for j, did in enumerate(kb_doc_ids_sorted)
        }

        # Fused retrieval at default λ=0.7 no cutoff
        e_final = fuse_and_normalize(e_visual, e_caption, cfg.fusion.lambda_star)
        cands = store.query(e_final, similarity_cutoff=0.0)
        top1 = cands[0] if cands else None

        row = {
            "idx": i,
            "image_path": str(img_path.relative_to(REPO)),
            "true_class": cls,
            "true_doc_id": true_doc_id,
            "caption": caption,
            "caption_tokens": tokens(caption),
            "e_caption": e_caption,
            "e_visual": e_visual,
            "sim_cap_to_docs": sim_cap_to_docs,
            "fused_top1_doc_id": top1.doc_id if top1 else None,
            "fused_top1_sim": top1.similarity if top1 else 0.0,
        }
        rows.append(row)
        print(
            f"  [{i + 1:2d}/{len(sample)}] {true_doc_id:4s} | top1_fused={row['fused_top1_doc_id']}"
            f"/{row['fused_top1_sim']:.3f} | caption len={len(caption.split()):3d}w"
        )

    # ─────────────────────────────────────────────────────────────
    # H1 v2 — correct-class keyword hit rate
    # ─────────────────────────────────────────────────────────────
    n_hit_correct_kw = 0
    for r in rows:
        correct_kws = per_doc_all_kws[r["true_doc_id"]]
        cap_lower = r["caption"].lower()
        if any(kw in cap_lower for kw in correct_kws if kw):
            n_hit_correct_kw += 1
    h1_frac = n_hit_correct_kw / len(rows) if rows else 0.0
    h1_verdict = "CONFIRMED" if h1_frac < H1_CONFIRM_THRESHOLD else "REJECTED"

    # ─────────────────────────────────────────────────────────────
    # H2 v2 — doc-centre / COL attractor
    # ─────────────────────────────────────────────────────────────
    doc_char_lens = {did: len(kb_by_id[did].text) for did in kb_doc_ids_sorted}
    # avg cosine of each doc to the other 7
    avg_cos_to_others = {}
    for i_did, did in enumerate(kb_doc_ids_sorted):
        other_vals = [kb_pairwise[i_did, j] for j in range(len(kb_doc_ids_sorted)) if j != i_did]
        avg_cos_to_others[did] = float(np.mean(other_vals))
    centre_doc = max(avg_cos_to_others, key=lambda d: avg_cos_to_others[d])
    non_centre_avg = np.mean([v for did, v in avg_cos_to_others.items() if did != centre_doc])
    col_centrality = avg_cos_to_others["COL"] - float(non_centre_avg)
    # Frequency of each doc as fused-top1 in the sample
    fused_top1_counts = Counter(r["fused_top1_doc_id"] for r in rows)
    col_top1_rate = fused_top1_counts.get("COL", 0) / len(rows) if rows else 0.0
    h2_verdict = (
        "CONFIRMED"
        if (centre_doc == "COL" and col_top1_rate > 0.3)
        else "PARTIAL"
        if (centre_doc == "COL" or col_top1_rate > 0.3)
        else "REJECTED"
    )

    # ─────────────────────────────────────────────────────────────
    # H3 v2 — caption→correct-doc vs caption→other-doc separation
    # ─────────────────────────────────────────────────────────────
    correct_sims = []
    incorrect_sims = []
    for r in rows:
        correct_sim = r["sim_cap_to_docs"].get(r["true_doc_id"], 0.0)
        correct_sims.append(correct_sim)
        for did, sim in r["sim_cap_to_docs"].items():
            if did != r["true_doc_id"]:
                incorrect_sims.append(sim)
    mean_correct = float(np.mean(correct_sims)) if correct_sims else 0.0
    mean_incorrect = float(np.mean(incorrect_sims)) if incorrect_sims else 0.0
    separation_gap = mean_correct - mean_incorrect
    h3_verdict = "CONFIRMED" if separation_gap < H3_CONFIRM_THRESHOLD else "REJECTED"

    # ─────────────────────────────────────────────────────────────
    # H4 v2 — caption vocabulary sparsity vs medical vocab
    # ─────────────────────────────────────────────────────────────
    total_caption_tokens = 0
    medical_overlap_tokens = 0
    for r in rows:
        for t in r["caption_tokens"]:
            total_caption_tokens += 1
            if t in medical_tokens:
                medical_overlap_tokens += 1
    medical_overlap_frac = (
        medical_overlap_tokens / total_caption_tokens if total_caption_tokens else 0.0
    )
    h4_verdict = "CONFIRMED" if medical_overlap_frac < H4_CONFIRM_THRESHOLD else "REJECTED"

    # ─────────────────────────────────────────────────────────────
    # H5 v2 — λ sensitivity on 10 stratified images
    # ─────────────────────────────────────────────────────────────
    # Take first image per class (7 classes) + 3 additional = 10
    by_class: dict[str, list] = defaultdict(list)
    for r in rows:
        by_class[r["true_doc_id"]].append(r)
    h5_images = []
    for did in sorted(by_class):
        if by_class[did]:
            h5_images.append(by_class[did][0])
    # fill to 10
    for r in rows:
        if len(h5_images) >= 10:
            break
        if r not in h5_images:
            h5_images.append(r)
    h5_per_lambda_correct: dict[float, int] = {lam: 0 for lam in LAMBDAS_H5}
    for r in h5_images:
        for lam in LAMBDAS_H5:
            e_final = fuse_and_normalize(r["e_visual"], r["e_caption"], lam)
            cands = store.query(e_final, similarity_cutoff=0.0)
            if cands and cands[0].doc_id == r["true_doc_id"]:
                h5_per_lambda_correct[lam] += 1
    h5_best_lambda = max(LAMBDAS_H5, key=lambda lam: h5_per_lambda_correct[lam])
    h5_best_correct = h5_per_lambda_correct[h5_best_lambda]
    h5_verdict = (
        "IMAGE_ONLY_BEST"
        if h5_best_lambda == 1.0 and h5_best_correct > h5_per_lambda_correct.get(0.7, 0)
        else "CAPTION_ONLY_BEST"
        if h5_best_lambda == 0.0 and h5_best_correct > h5_per_lambda_correct.get(0.7, 0)
        else "MIDDLE_BEST"
    )

    # ─────────────────────────────────────────────────────────────
    # H6 v2 — best-achievable top-1 doc distribution
    # ─────────────────────────────────────────────────────────────
    best_achievable_counts: Counter = Counter()
    for r in rows:
        sims = r["sim_cap_to_docs"]  # caption-only similarity to all 8 KB docs
        best_doc = max(sims, key=lambda d: sims[d])
        best_achievable_counts[best_doc] += 1
    col_best_rate = best_achievable_counts.get("COL", 0) / len(rows) if rows else 0.0
    h6_verdict = "CONFIRMED" if col_best_rate >= H6_CONFIRM_FRACTION else "REJECTED"

    # ─────────────────────────────────────────────────────────────
    # Scenario classification
    # ─────────────────────────────────────────────────────────────
    vocabulary_dom = h1_verdict == "CONFIRMED" or h4_verdict == "CONFIRMED"
    attractor_dom = h2_verdict == "CONFIRMED" or h6_verdict == "CONFIRMED"
    clip_dom = h3_verdict == "CONFIRMED"
    image_best = h5_verdict == "IMAGE_ONLY_BEST"

    active = {
        "α_vocab": vocabulary_dom,
        "β_attractor": attractor_dom,
        "γ_clip": clip_dom,
        "δ_image_best": image_best,
    }
    active_count = sum(active.values())
    if active_count == 0:
        scenario, recommendation = (
            "ε (no single hypothesis dominates)",
            "ADR-0011 alone (cutoff reduction); re-run REPAIR-6 EXP-1 to gather more data in the new landscape.",
        )
    elif active_count == 1:
        if vocabulary_dom:
            scenario = "α (vocabulary gap dominant)"
            recommendation = "ADR-0011 (cutoff reduction) + ADR-0012a KB Vocabulary Reform."
        elif attractor_dom:
            scenario = "β (KB-doc attractor dominant)"
            recommendation = "ADR-0011 + ADR-0012c Retrieval Strategy (centroid-aware rerank or per-class calibration)."
        elif clip_dom:
            scenario = "γ (CLIP separability insufficient)"
            recommendation = "ADR-0011 + ADR-0012b CLIP Encoder Assessment (ViT-L/14, SigLIP, BiomedCLIP)."
        else:  # image_best
            scenario = "δ (image-only retrieval outperforms caption-heavy)"
            recommendation = "ADR-0011 + methodology revision (lower λ default); may trigger paper-deviation note."
    else:
        active_list = [k for k, v in active.items() if v]
        scenario = f"mixed ({'+'.join(active_list)}) → collapse to ε with co-causes noted"
        recommendation = "ADR-0011 alone first, then REPAIR-6 EXP-1 to see which co-cause persists after cutoff-fix."

    # ── Write analysis.md ─────────────────────────────────────────
    pairwise_rows_md = "\n".join(
        "| " + did + " | " + " | ".join(f"{kb_pairwise[i, j]:.3f}" for j in range(len(kb_doc_ids_sorted))) + " |"
        for i, did in enumerate(kb_doc_ids_sorted)
    )
    doc_centrality_rows = "\n".join(
        f"| {did} | {doc_char_lens[did]:>6d} | {avg_cos_to_others[did]:.3f} | {fused_top1_counts.get(did, 0):>3d} | {best_achievable_counts.get(did, 0):>3d} |"
        for did in kb_doc_ids_sorted
    )
    h5_rows = "\n".join(
        f"| λ = {lam:.1f} | {h5_per_lambda_correct[lam]} / {len(h5_images)} | "
        f"{h5_per_lambda_correct[lam] / len(h5_images):.2f} |"
        for lam in LAMBDAS_H5
    )

    analysis = f"""# EXP-1.5 v2 — Extended Diagnostic with REPAIR-1 Captions

> Generated 2026-04-18 · n = {len(rows)} unique images (35 EXP-1.5 ∪ 5 REPAIR-4, dedup) · seed = {SEED}

## Summary

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1 v2 vocabulary gap (correct-class KB keyword in caption) | **{h1_verdict}** | {n_hit_correct_kw}/{len(rows)} ({h1_frac:.1%}) captions contain ≥ 1 correct-class KB keyword; threshold < {H1_CONFIRM_THRESHOLD:.0%} |
| H2 v2 COL attractor | **{h2_verdict}** | centre doc by avg cosine = **{centre_doc}**; COL centrality Δ vs non-centre avg = {col_centrality:+.3f}; COL top-1 rate = {col_top1_rate:.1%} |
| H3 v2 CLIP separability | **{h3_verdict}** | caption→correct = {mean_correct:.3f}, →incorrect = {mean_incorrect:.3f}, gap = **{separation_gap:+.3f}**; threshold < {H3_CONFIRM_THRESHOLD} |
| H4 v2 vocabulary sparsity | **{h4_verdict}** | medical-vocab overlap = {medical_overlap_tokens}/{total_caption_tokens} ({medical_overlap_frac:.2%}); threshold < {H4_CONFIRM_THRESHOLD:.0%} |
| H5 v2 λ sensitivity | **{h5_verdict}** | best λ = **{h5_best_lambda:.1f}** ({h5_best_correct}/{len(h5_images)} correct); distribution below |
| H6 v2 COL best-achievable-top-1 rate | **{h6_verdict}** | COL is caption-only best for {best_achievable_counts.get("COL", 0)}/{len(rows)} ({col_best_rate:.1%}); threshold ≥ {H6_CONFIRM_FRACTION:.0%} |

**Scenario**: {scenario}
**Recommendation**: {recommendation}

---

## H1 v2 — correct-class KB keyword coverage in real captions

Under REPAIR-1 fix (`<MORE_DETAILED_CAPTION>`), captions are non-empty
natural-language descriptions. This test asks: do they contain any
keyword from the correct-class KB doc's three tiers (confirmed /
suspected / mentioned) or `HLT.healthy_keywords`?

- Captions with ≥ 1 correct-class keyword: **{n_hit_correct_kw}/{len(rows)}** ({h1_frac:.1%})
- Threshold for confirmation: < {H1_CONFIRM_THRESHOLD:.0%}
- Verdict: **{h1_verdict}**

### Per-class breakdown
| doc_id | captions with ≥ 1 correct-class kw |
|---|---|
"""

    per_class_h1 = defaultdict(list)
    for r in rows:
        correct_kws = per_doc_all_kws[r["true_doc_id"]]
        cap_lower = r["caption"].lower()
        hit = any(kw in cap_lower for kw in correct_kws if kw)
        per_class_h1[r["true_doc_id"]].append(hit)
    for did in kb_doc_ids_sorted:
        if did in per_class_h1 and per_class_h1[did]:
            n_hit = sum(per_class_h1[did])
            n_tot = len(per_class_h1[did])
            analysis += f"| {did} | {n_hit}/{n_tot} |\n"

    analysis += f"""
---

## H2 v2 — KB-doc attractor (COL centre hypothesis)

8×8 CLIP-text cosine matrix of KB doc text embeddings:

| doc | {" | ".join(kb_doc_ids_sorted)} |
|---|{"---|" * len(kb_doc_ids_sorted)}
{pairwise_rows_md}

Per-doc stats:

| doc_id | chars | avg cos to other 7 | fused top-1 count | best-achievable top-1 |
|---|---|---|---|---|
{doc_centrality_rows}

- Centre doc (max avg cosine): **{centre_doc}**
- COL centrality Δ vs non-centre mean: {col_centrality:+.3f}
- COL top-1 rate in fused retrieval (λ = 0.7): {col_top1_rate:.1%}
- Verdict: **{h2_verdict}**

---

## H3 v2 — CLIP caption→doc separation gap

Across {len(rows)} images, mean caption-to-doc cosine:

- Mean similarity to **correct** class doc: **{mean_correct:.3f}**
- Mean similarity to **any other** doc: **{mean_incorrect:.3f}**
- Separation gap: **{separation_gap:+.3f}**
- Threshold for confirmation: gap < {H3_CONFIRM_THRESHOLD}
- Verdict: **{h3_verdict}**

---

## H4 v2 — caption vocabulary sparsity vs clinical medical vocab

Medical vocabulary = union of all 8 KB docs' `confirmed` keywords, tokenised.

- Total caption tokens across {len(rows)} images: {total_caption_tokens}
- Tokens in medical vocab: {medical_overlap_tokens}
- Overlap fraction: **{medical_overlap_frac:.2%}**
- Threshold for confirmation: < {H4_CONFIRM_THRESHOLD:.0%}
- Verdict: **{h4_verdict}**

---

## H5 v2 — λ sensitivity on top-1 class correctness

10-image stratified subset; top-1 correctness at 5 λ values:

| λ | correct | rate |
|---|---|---|
{h5_rows}

- Best λ = **{h5_best_lambda:.1f}**
- Verdict: **{h5_verdict}**

---

## H6 v2 — best-achievable top-1 doc distribution (caption-only)

For each caption, we find the KB doc with highest cosine similarity
(ignoring cutoff, ignoring fusion). This measures what the caption
*could* retrieve in the absence of visual path contribution.

Distribution of best-achievable top-1 doc:

| doc_id | count | rate |
|---|---|---|
"""
    for did in kb_doc_ids_sorted:
        cnt = best_achievable_counts.get(did, 0)
        analysis += f"| {did} | {cnt} | {cnt / len(rows):.1%} |\n"
    analysis += f"""
- COL best-achievable rate: **{col_best_rate:.1%}**
- Threshold for confirmation: ≥ {H6_CONFIRM_FRACTION:.0%}
- Verdict: **{h6_verdict}**

---

## Scenario classification

Active hypothesis families:

| Family | Trigger | Active now |
|---|---|---|
| α vocabulary gap | H1 or H4 CONFIRMED | {vocabulary_dom} |
| β attractor | H2 or H6 CONFIRMED | {attractor_dom} |
| γ CLIP insufficient | H3 CONFIRMED | {clip_dom} |
| δ image-only best | H5 IMAGE_ONLY_BEST | {image_best} |

**Scenario**: {scenario}

**Recommendation**: {recommendation}

---

## Next-step proposal (awaiting user authorisation)

1. **ADR-0011 Configuration Tuning — similarity_cutoff** — triple-confirmed
   empirically; cutoff = 0.5 is over-tight given observed similarity
   distribution (top-1 always < 0.5 with this KB + CLIP). Proposed new
   value to be chosen from EXP-1 re-run data, not from any single sample.
2. **ADR-0012 variant** per scenario as recommended above (if applicable).
3. **REPAIR-6 EXP-1 re-run** after (1) and (2) land; expected runtime
   ~4 min given caching design in `exp1_cutoff_lambda_sweep.py`.

"""
    (out_dir / "analysis.md").write_text(analysis)
    print(f"\nanalysis.md → {out_dir / 'analysis.md'}")

    # Persist machine-readable summary
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "n": len(rows),
                "h1": {"n_hit": n_hit_correct_kw, "frac": h1_frac, "verdict": h1_verdict},
                "h2": {
                    "centre_doc": centre_doc,
                    "col_centrality": col_centrality,
                    "col_top1_rate": col_top1_rate,
                    "fused_top1_counts": dict(fused_top1_counts),
                    "verdict": h2_verdict,
                },
                "h3": {
                    "mean_correct": mean_correct,
                    "mean_incorrect": mean_incorrect,
                    "separation_gap": separation_gap,
                    "verdict": h3_verdict,
                },
                "h4": {
                    "medical_overlap_tokens": medical_overlap_tokens,
                    "total_caption_tokens": total_caption_tokens,
                    "frac": medical_overlap_frac,
                    "verdict": h4_verdict,
                },
                "h5": {
                    "per_lambda_correct": {f"{lam:.1f}": h5_per_lambda_correct[lam] for lam in LAMBDAS_H5},
                    "best_lambda": h5_best_lambda,
                    "best_correct": h5_best_correct,
                    "n_images": len(h5_images),
                    "verdict": h5_verdict,
                },
                "h6": {
                    "best_achievable_counts": dict(best_achievable_counts),
                    "col_best_rate": col_best_rate,
                    "verdict": h6_verdict,
                },
                "scenario": scenario,
                "recommendation": recommendation,
            },
            indent=2,
            default=str,
        )
    )
    print(f"summary.json → {out_dir / 'summary.json'}")

    florence.close()
    clip.close()


if __name__ == "__main__":
    main()
