"""EXP-1.5 root-cause diagnostic — isolate H1/H2/H3/H4 failure modes.

Selects 5 stratified images per D1 Test class (35 total) and produces
per-image retrieval diagnostics at three λ points (0.0 / 0.7 / 1.0)
plus KB-structural analyses. All writes land in ``docs/m3/exp1_5/``.

Hypotheses under test (per M3 EXP-1.5 authorization):
    H1  KB vocabulary mismatch          — captions vs KB keyword tiers
    H2  KB coverage imbalance           — doc-length variance + class freq
    H3  CLIP class-separability failure — 8×8 KB doc pairwise similarity
    H4  Aggressive clean_caption        — token diff vs KB vocabulary
    H5  Stack-version drift             — deferred to EXP-2

Acceptance: diagnostic.csv, analysis.md with H1-H4 verdicts,
determinism.json, scenario identification α/β/γ/δ.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
import sys
import tempfile
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
LAMBDAS_DIAG = [0.0, 0.7, 1.0]

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_KB_CLUSTER_THRESHOLD = 0.8  # H3 pairwise similarity threshold
_KB_CLUSTER_MIN = 6  # H3 min docs mutually clustered
_H1_CAPTION_KEYWORD_MIN_FRAC = 0.10  # <10% of captions hit ≥1 keyword → H1 confirmed


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def sample_stratified(test_root: Path, per_class: int, seed: int) -> list[tuple[Path, str]]:
    """Deterministic stratified sample: per_class images from each D1 class folder."""
    rng = random.Random(seed)
    sampled: list[tuple[Path, str]] = []
    image_exts = {".jpg", ".jpeg", ".png"}
    for cls_dir in sorted(test_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        files = sorted(
            p
            for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in image_exts
        )
        if len(files) <= per_class:
            chosen = files
        else:
            chosen = rng.sample(files, per_class)
            chosen.sort()
        for p in chosen:
            sampled.append((p, cls_dir.name))
    return sampled


def _top_k_from_candidates(candidates, true_doc_id: str, k: int = 3):
    out = []
    for cand in candidates[:k]:
        sim = cand.similarity_penalized if cand.similarity_penalized is not None else cand.similarity
        out.append((cand.doc_id, float(sim), cand.doc_id == true_doc_id))
    while len(out) < k:
        out.append((None, 0.0, False))
    return out


def main() -> None:
    out_dir = REPO / "docs/m3/exp1_5"
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    cfg = load_config(REPO / "configs/default.yaml")

    test_root = (
        REPO
        / "lab_dateset/external_datasets/fish_disease_south_asia"
        / "Freshwater Fish Disease Aquaculture in south asia/Test"
    )
    sample = sample_stratified(test_root, PER_CLASS, SEED)
    print(f"[sample] {len(sample)} images (5 per class × 7 classes)")

    florence = Florence2Wrapper(cfg.florence2)
    clip = OpenClipEmbedder(cfg.clip)
    florence.warmup()
    clip.warmup()

    # KB ingest into tmp Chroma
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

    # Build flat keyword→set-of-doc-ids map across all tiers + healthy
    all_keywords: set[str] = set()
    for d in kb_docs:
        for tier, kws in d.clinical_keywords.items():
            for kw in kws:
                all_keywords.add(kw.lower())
        for kw in d.healthy_keywords:
            all_keywords.add(kw.lower())

    # KB doc text embeddings for H3 pairwise matrix
    kb_text_embeddings = clip.encode_text([kb_by_id[did].text for did in kb_doc_ids_sorted])
    kb_emb_arr = np.array(kb_text_embeddings, dtype=np.float64)
    kb_pairwise = kb_emb_arr @ kb_emb_arr.T  # cosine (already L2-normalised)

    rows: list[dict] = []

    print("[diagnostic] running 35-image loop…")
    for idx, (img_path, true_class) in enumerate(sample):
        true_doc_id = CLASS_TO_DOC_ID[true_class]

        caption_raw, _objs, _telem = florence.caption_and_detect(img_path)
        caption_clean = clean_caption(caption_raw)
        tokens_raw = tokenize(caption_raw)
        tokens_clean = tokenize(caption_clean)
        stripped = sorted(set(tokens_raw) - set(tokens_clean))

        text_for_clip = caption_clean if caption_clean else " "
        e_caption = clip.encode_text([text_for_clip])[0]
        e_visual = clip.encode_image_paths([str(img_path)])[0]

        # Per-λ retrievals (no cutoff — see all 8, sorted)
        per_lambda_top3 = {}
        per_lambda_all8 = {}
        for lam in LAMBDAS_DIAG:
            e_final = fuse_and_normalize(e_visual, e_caption, lam)
            all_candidates = store.query(e_final, similarity_cutoff=0.0)
            per_lambda_top3[lam] = _top_k_from_candidates(all_candidates, true_doc_id, k=3)
            by_doc = {c.doc_id: float(c.similarity) for c in all_candidates}
            per_lambda_all8[lam] = [by_doc.get(did, 0.0) for did in kb_doc_ids_sorted]

        # Caption keyword coverage (H1): count how many unique KB keywords appear in caption
        caption_lower = caption_clean.lower()
        kw_hits = sum(1 for kw in all_keywords if kw and kw in caption_lower)

        row = {
            "idx": idx,
            "image_path": str(img_path.relative_to(REPO)),
            "true_class": true_class,
            "true_doc_id": true_doc_id,
            "caption_raw": caption_raw.replace("\n", " ").replace("\r", " "),
            "caption_clean": caption_clean.replace("\n", " ").replace("\r", " "),
            "caption_token_count_raw": len(tokens_raw),
            "caption_token_count_clean": len(tokens_clean),
            "stripped_tokens": "|".join(stripped),
            "caption_kb_keyword_hits": kw_hits,
            "top3_lambda_0_0": per_lambda_top3[0.0],
            "top3_lambda_0_7": per_lambda_top3[0.7],
            "top3_lambda_1_0": per_lambda_top3[1.0],
            "sim_all8_lambda_0_0": per_lambda_all8[0.0],
            "sim_all8_lambda_0_7": per_lambda_all8[0.7],
            "sim_all8_lambda_1_0": per_lambda_all8[1.0],
        }
        rows.append(row)
        print(
            f"  [{idx + 1}/{len(sample)}] {true_doc_id:4s} {img_path.name[:40]:40s} "
            f"kw_hits={kw_hits:2d} "
            f"top1@λ=0.7={row['top3_lambda_0_7'][0][0]}/"
            f"{row['top3_lambda_0_7'][0][1]:.3f}/"
            f"{'✓' if row['top3_lambda_0_7'][0][2] else '×'}"
        )

    # ── Determinism on image 0 ────────────────────────────────────
    (first_path, first_class) = sample[0]
    caption_raw_2, _, _ = florence.caption_and_detect(first_path)
    caption_clean_2 = clean_caption(caption_raw_2)
    e_vis_2 = clip.encode_image_paths([str(first_path)])[0]
    e_txt_2 = clip.encode_text([caption_clean_2 if caption_clean_2 else " "])[0]
    e_final_2 = fuse_and_normalize(e_vis_2, e_txt_2, 0.7)
    rerun_cands = store.query(e_final_2, similarity_cutoff=0.0)
    rerun_top3 = _top_k_from_candidates(
        rerun_cands, CLASS_TO_DOC_ID[first_class], k=3
    )
    det_ok = (
        rerun_top3 == rows[0]["top3_lambda_0_7"]
        and caption_raw_2 == rows[0]["caption_raw"].replace(" ", " ")
        or caption_clean_2 == rows[0]["caption_clean"]
    )
    det_ok = rerun_top3 == rows[0]["top3_lambda_0_7"] and caption_clean_2 == rows[0]["caption_clean"]

    (out_dir / "determinism.json").write_text(
        json.dumps(
            {
                "image": str(first_path.relative_to(REPO)),
                "run1_top3_lambda_0_7": rows[0]["top3_lambda_0_7"],
                "run2_top3_lambda_0_7": rerun_top3,
                "run1_caption_clean_sha": hashlib.sha256(
                    rows[0]["caption_clean"].encode()
                ).hexdigest()[:16],
                "run2_caption_clean_sha": hashlib.sha256(
                    caption_clean_2.encode()
                ).hexdigest()[:16],
                "pass": det_ok,
            },
            indent=2,
        )
    )

    # ── Save diagnostic.csv ───────────────────────────────────────
    csv_path = out_dir / "diagnostic.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        header = [
            "idx",
            "image_path",
            "true_class",
            "true_doc_id",
            "caption_raw",
            "caption_clean",
            "caption_token_count_raw",
            "caption_token_count_clean",
            "stripped_tokens",
            "caption_kb_keyword_hits",
        ]
        for lam in LAMBDAS_DIAG:
            ltag = f"{lam:.1f}".replace(".", "_")
            for rank in (1, 2, 3):
                header += [
                    f"top{rank}_doc_id_lambda_{ltag}",
                    f"top{rank}_sim_lambda_{ltag}",
                    f"top{rank}_correct_lambda_{ltag}",
                ]
        for did in kb_doc_ids_sorted:
            for lam in LAMBDAS_DIAG:
                ltag = f"{lam:.1f}".replace(".", "_")
                header.append(f"sim_{did}_lambda_{ltag}")
        f.write(",".join(header) + "\n")

        for r in rows:
            cells: list[str] = [
                str(r["idx"]),
                r["image_path"],
                f"\"{r['true_class']}\"",
                r["true_doc_id"],
                f"\"{r['caption_raw'].replace(chr(34), chr(39))}\"",
                f"\"{r['caption_clean'].replace(chr(34), chr(39))}\"",
                str(r["caption_token_count_raw"]),
                str(r["caption_token_count_clean"]),
                f"\"{r['stripped_tokens']}\"",
                str(r["caption_kb_keyword_hits"]),
            ]
            for lam in LAMBDAS_DIAG:
                key = f"top3_lambda_{str(lam).replace('.', '_')}"
                for rank in range(3):
                    did_val, sim_val, corr = r[key][rank]
                    cells += [
                        did_val if did_val is not None else "",
                        f"{sim_val:.4f}",
                        str(corr),
                    ]
            for did_idx, did in enumerate(kb_doc_ids_sorted):
                for lam in LAMBDAS_DIAG:
                    key = f"sim_all8_lambda_{str(lam).replace('.', '_')}"
                    cells.append(f"{r[key][did_idx]:.4f}")
            f.write(",".join(cells) + "\n")
    print(f"diagnostic.csv → {csv_path}")

    # ── H1 test ───────────────────────────────────────────────────
    n_with_any_hit = sum(1 for r in rows if r["caption_kb_keyword_hits"] > 0)
    h1_frac = n_with_any_hit / len(rows)
    h1_verdict = "CONFIRMED" if h1_frac < _H1_CAPTION_KEYWORD_MIN_FRAC else "REJECTED"

    # ── H2 test ───────────────────────────────────────────────────
    doc_char_lens = {did: len(kb_by_id[did].text) for did in kb_doc_ids_sorted}
    length_vals = list(doc_char_lens.values())
    length_std = statistics.stdev(length_vals) if len(length_vals) > 1 else 0.0
    length_mean = statistics.mean(length_vals)
    length_cv = length_std / length_mean if length_mean else 0.0

    # Retrieval frequency per doc at λ=0.7
    retrieval_top1_counts = defaultdict(int)
    for r in rows:
        top1 = r["top3_lambda_0_7"][0]
        if top1[0] is not None:
            retrieval_top1_counts[top1[0]] += 1
    freq_vals = [retrieval_top1_counts.get(did, 0) for did in kb_doc_ids_sorted]
    # Correlation between doc length and retrieval frequency
    if len(set(length_vals)) > 1 and len(set(freq_vals)) > 1:
        corr_matrix = np.corrcoef(length_vals, freq_vals)
        length_freq_corr = float(corr_matrix[0, 1])
    else:
        length_freq_corr = 0.0
    h2_verdict = (
        "CONFIRMED"
        if (length_cv > 0.5 and abs(length_freq_corr) > 0.5)
        else "PARTIAL"
        if (length_cv > 0.5 or abs(length_freq_corr) > 0.5)
        else "REJECTED"
    )

    # ── H3 test — 8×8 KB doc pairwise cosine ──────────────────────
    pairs_above = 0
    mutual_cluster_size = 0
    for i, did_i in enumerate(kb_doc_ids_sorted):
        above_neighbours = sum(
            1
            for j in range(len(kb_doc_ids_sorted))
            if j != i and kb_pairwise[i, j] > _KB_CLUSTER_THRESHOLD
        )
        if above_neighbours >= len(kb_doc_ids_sorted) - 2:
            mutual_cluster_size += 1
    for i in range(len(kb_doc_ids_sorted)):
        for j in range(i + 1, len(kb_doc_ids_sorted)):
            if kb_pairwise[i, j] > _KB_CLUSTER_THRESHOLD:
                pairs_above += 1
    h3_verdict = "CONFIRMED" if mutual_cluster_size >= _KB_CLUSTER_MIN else "REJECTED"

    # ── H4 test — stripped-token vs KB keyword overlap ────────────
    stripped_all = set()
    stripped_disease_relevant = set()
    for r in rows:
        if r["stripped_tokens"]:
            toks = [t for t in r["stripped_tokens"].split("|") if t]
            stripped_all.update(toks)
            for t in toks:
                if t.lower() in all_keywords:
                    stripped_disease_relevant.add(t)
    h4_verdict = "CONFIRMED" if stripped_disease_relevant else "REJECTED"

    # ── Scenario classification ───────────────────────────────────
    if h1_verdict == "CONFIRMED":
        scenario = "α (KB vocabulary mismatch dominant)"
        next_adr = "ADR-0012 — KB Vocabulary Reform"
    elif h3_verdict == "CONFIRMED":
        scenario = "β (CLIP separability failure)"
        next_adr = "ADR-0012 — CLIP Encoder Reassessment"
    elif h4_verdict == "CONFIRMED":
        scenario = "γ (aggressive clean_caption)"
        next_adr = "no ADR — narrow clean_caption fix"
    else:
        scenario = "δ (no single hypothesis dominates)"
        next_adr = "proceed to EXP-2 Florence-2-only DA to isolate Stage 1 vs Stage 2"

    # ── analysis.md ───────────────────────────────────────────────
    per_class_kw_hits = defaultdict(list)
    for r in rows:
        per_class_kw_hits[r["true_doc_id"]].append(r["caption_kb_keyword_hits"])

    per_class_hit_summary = "\n".join(
        f"| {did} | {statistics.mean(v):.2f} | {max(v)} | {min(v)} |"
        for did, v in sorted(per_class_kw_hits.items())
    )

    pairwise_rows = "\n".join(
        "| " + did + " | " + " | ".join(f"{kb_pairwise[i, j]:.3f}" for j in range(len(kb_doc_ids_sorted))) + " |"
        for i, did in enumerate(kb_doc_ids_sorted)
    )

    doc_len_rows = "\n".join(
        f"| {did} | {doc_char_lens[did]:>6d} | {retrieval_top1_counts.get(did, 0):>3d} |"
        for did in kb_doc_ids_sorted
    )

    analysis = f"""# EXP-1.5 Diagnostic Analysis

> Generated 2026-04-18 · n = {len(rows)} images (5 stratified per class × 7) · seed = {SEED}

## Summary

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1 KB vocabulary mismatch | **{h1_verdict}** | {n_with_any_hit}/{len(rows)} captions ({h1_frac:.1%}) contain ≥1 KB keyword; threshold < {_H1_CAPTION_KEYWORD_MIN_FRAC:.0%} → confirmed |
| H2 KB coverage imbalance | **{h2_verdict}** | length CV = {length_cv:.3f}; length↔freq corr = {length_freq_corr:+.3f} |
| H3 CLIP separability | **{h3_verdict}** | {mutual_cluster_size}/{len(kb_doc_ids_sorted)} docs in mutual cluster (> {_KB_CLUSTER_THRESHOLD}); {pairs_above} pairs above |
| H4 clean_caption stripping | **{h4_verdict}** | {len(stripped_disease_relevant)} disease-relevant tokens stripped out of {len(stripped_all)} total |
| H5 stack-version drift | deferred | See EXP-2 |

**Scenario**: {scenario}
**Proposed next step**: {next_adr}

---

## H1 — Caption vs KB keyword coverage

Fraction of images whose cleaned caption contains at least one KB
keyword (across all 7 disease docs, all three clinical tiers, plus
``HLT.healthy_keywords``): **{h1_frac:.1%}** ({n_with_any_hit}/{len(rows)}).

Threshold for H1 confirmation: < {_H1_CAPTION_KEYWORD_MIN_FRAC:.0%} of captions
carry any KB keyword. H1 is **{h1_verdict}**.

### Per-class caption keyword hits
| doc_id | mean hits | max | min |
|---|---|---|---|
{per_class_hit_summary}

---

## H2 — KB coverage imbalance

| doc_id | chars | top-1 @ λ=0.7 |
|---|---|---|
{doc_len_rows}

- length mean = {length_mean:.1f}, stdev = {length_std:.1f}, CV = {length_cv:.3f}
- correlation (length × retrieval frequency at λ=0.7) = {length_freq_corr:+.3f}

H2 verdict: **{h2_verdict}** (CONFIRMED requires CV > 0.5 **and** |corr| > 0.5).

---

## H3 — 8×8 KB doc pairwise cosine similarity

| doc | {" | ".join(kb_doc_ids_sorted)} |
|---|{"---|" * len(kb_doc_ids_sorted)}
{pairwise_rows}

- pairs with cos > {_KB_CLUSTER_THRESHOLD}: **{pairs_above}** out of {len(kb_doc_ids_sorted) * (len(kb_doc_ids_sorted) - 1) // 2}
- docs in mutual cluster (≥ 6 of 8 neighbours above threshold): **{mutual_cluster_size}**

H3 verdict: **{h3_verdict}** (CONFIRMED requires ≥ {_KB_CLUSTER_MIN} in mutual cluster).

---

## H4 — clean_caption token diff

- total stripped tokens across all 35 captions: **{len(stripped_all)}**
- stripped tokens that ALSO appear in any KB keyword set: **{len(stripped_disease_relevant)}**
- stripped disease-relevant tokens: `{", ".join(sorted(stripped_disease_relevant)) or "(none)"}`

H4 verdict: **{h4_verdict}**.

---

## Determinism

Two-run bit-identical on image 0 at λ=0.7: **{"PASS" if det_ok else "FAIL"}**.

See `determinism.json`.

---

## Concrete next step

{next_adr}

"""
    if scenario.startswith("α"):
        analysis += """
Under Scenario α, next actions (awaiting user authorization):
1. Author **ADR-0012 KB Vocabulary Reform** with context citing this H1 verdict
2. Rewrite 8 KB docs so each clinical term is paired with a
   Florence-2-compatible colloquial phrase (e.g. "haemorrhagic
   ulceration (red sore / open wound)")
3. Re-run a reduced EXP-1 grid {λ ∈ [0.3, 0.5, 0.7], cutoff ∈ [0.2, 0.3, 0.4]}
   to verify lift
"""
    elif scenario.startswith("β"):
        analysis += """
Under Scenario β, next actions (awaiting user authorization):
1. Author **ADR-0012 CLIP Encoder Reassessment**
2. Benchmark ViT-L/14 and SigLIP on the 8-doc separability task
3. Scope change to CLIP encoder requires paper-deviation flag
"""
    elif scenario.startswith("γ"):
        analysis += """
Under Scenario γ, next actions:
1. No ADR required (implementation fix)
2. Narrow clean_caption to preserve disease-relevant tokens
3. Re-run EXP-1 reduced grid
"""
    else:
        analysis += """
Under Scenario δ, proceed to EXP-2 to isolate Stage 1 vs Stage 2 contribution.
If EXP-2 Florence-2-only DA matches paper 0.037 ± 0.01, Stage 2 is the problem
layer and H3/retrieval-level interventions are needed.
"""
    (out_dir / "analysis.md").write_text(analysis)
    print(f"analysis.md → {out_dir / 'analysis.md'}")

    florence.close()
    clip.close()


if __name__ == "__main__":
    main()
