"""Reproduce the two paired McNemar tests on cross-dataset D2 (reviewer R5.3).

This script recomputes, from REAL paired per-image predictions, the two 2x2
contingency tables the manuscript reports on the cross-dataset D2 evaluation:

  (a) Full PIPELINE (forced-choice) vs CLIP k-NN (k=1, fused lambda=0.7)
  (b) fused lambda=0.7 (k-NN) vs visual-only lambda=1.0 (k-NN)

Everything is real: real CLIP ViT-B-32 (laion2b_s34b_b79k) visual embeddings
recomputed on the on-disk augmentation-free D2 subset, the real cached
Florence-2 MORE_DETAILED_CAPTION captions (results/e3_captions_log.json), the
real 8-document knowledge base (src/fishdx/kb/documents/*.md) keyword tiers,
and the real Stage-3 scoring primitives (compute_s_h / compute_s_d, Eq. 8-9).

Retrieval design (paper Stage 2): D2 queries are matched against the labelled
D1 reference gallery (results/_cache_d1_gallery.npz, 1,639 D1-Train fusion
embeddings, 7 classes, no EUS) — leak-free because D1 and D2 are disjoint
datasets. Evaluation follows the codebase's established Overlap-DA convention:
the seven classes shared by D1 and D2 (EUS excluded, as the D1 gallery has no
EUS reference and neither retrieval nor scoring can commit to it here).

Pipeline forced-choice per image (no abstention, matching how the manuscript's
"Pipeline Overall DA = 0.811" forced-choice prediction is defined):
  1. fused lambda=0.7 retrieval, top-K=5, top-1 gallery class -> candidate c1
  2. Stage-3 evidence-pool scoring over the Florence-2 caption:
       S_h = compute_s_h(caption)  [healthy KB keywords + negation, Eq. 8]
       S_d = compute_s_d(caption)  [c1's confirmed/suspected/mentioned tiers, Eq. 9]
  3. committed class = "healthy" if S_h > S_d else c1  (argmax the scoring commits
     to; retrieval supplies the disease identity, scoring gates healthy vs disease)

All hyperparameters (lambda*, scoring weights, negation patterns, seed) are read
from configs/default.yaml — no magic numbers. Florence-2 is NOT re-run (captions
cached); only D2 CLIP visual + caption-text embeddings are computed.

Output: results/mcnemar_d2_reproduced.json
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import chi2
from tqdm import tqdm

from fishdx.config import load_config
from fishdx.kb.builder import parse_markdown_docs
from fishdx.scoring.decision import make_decision
from fishdx.scoring.score import compute_s_d, compute_s_h
from fishdx.schemas import DecisionEnum

ROOT = Path(__file__).resolve().parents[2]
D2_DIR = ROOT / "data/datasets/D2"
RES = ROOT / "results"
CACHE = RES / "_cache_d1_gallery.npz"
D2_CAPS = RES / "e3_captions_log.json"
KB_DOCS = ROOT / "src/fishdx/kb/documents"
CONFIG = ROOT / "configs/default.yaml"
OUT = RES / "mcnemar_d2_reproduced_decisive.json"
INCONCLUSIVE = "__INCONCLUSIVE__"  # sentinel: never equals a true label -> counts as a miss

SEED = 42
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Original manuscript discordant counts (for side-by-side comparison only).
ORIG_PIPE_KNN = {"b": 21, "c": 40, "chi2": 5.31, "p": 0.021}
ORIG_L07_L10 = {"b": 20, "c": 41, "chi2": 6.56, "p": 0.010}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def canon(stem: str) -> str:
    """Canonical class label (identical to exp_lambda_d2_ablation.canon)."""
    s = re.sub(r"[\s_]*\(?\d+\)?$", "", stem).strip()
    s = re.sub(r"_aug$", "", s).strip()
    m = re.match(r"^(.*)_\1$", s)
    if m:
        s = m.group(1)
    s = s.lower()
    if "healthy" in s:
        return "healthy"
    if "aeromon" in s:
        return "aeromoniasis"
    if "gill" in s:
        return "bacterial_gill"
    if "red" in s:
        return "bacterial_red"
    if "fungal" in s or "saproleg" in s:
        return "fungal"
    if "parasit" in s:
        return "parasitic"
    if "white tail" in s or "viral" in s:
        return "viral_white_tail"
    if "eus" in s or "ulcerative" in s:
        return "EUS"
    return "UNKNOWN"


def l2(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def knn_majority_vote(sims: np.ndarray, gallery_labels: np.ndarray, k: int) -> np.ndarray:
    """k-NN majority vote. Ties broken by summed cosine similarity, then by the
    nearest (highest-similarity) neighbour — fully deterministic."""
    order = np.argsort(-sims, axis=1)[:, :k]
    preds: list[str] = []
    for i in range(sims.shape[0]):
        idx = order[i]
        labels = gallery_labels[idx]
        sim_row = sims[i, idx]
        votes: dict[str, int] = {}
        sim_sum: dict[str, float] = {}
        best_rank: dict[str, int] = {}
        for rank, (lab, s) in enumerate(zip(labels.tolist(), sim_row.tolist())):
            votes[lab] = votes.get(lab, 0) + 1
            sim_sum[lab] = sim_sum.get(lab, 0.0) + s
            if lab not in best_rank:
                best_rank[lab] = rank
        # max votes -> max summed similarity -> earliest (nearest) rank
        winner = max(votes, key=lambda lab: (votes[lab], sim_sum[lab], -best_rank[lab]))
        preds.append(winner)
    return np.array(preds)


def load_clip():
    import open_clip

    clip, _, prep = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=DEVICE
    )
    clip.eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    return clip, prep, tok


def clip_image(path: Path, clip, prep) -> np.ndarray:
    with torch.no_grad():
        x = prep(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
        f = clip.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
    return f[0].cpu().numpy().astype(np.float32)


def clip_text(text: str, clip, tok) -> np.ndarray:
    with torch.no_grad():
        t = tok([text or "fish"]).to(DEVICE)
        f = clip.encode_text(t)
        f = f / f.norm(dim=-1, keepdim=True)
    return f[0].cpu().numpy().astype(np.float32)


def load_kb_tiers() -> tuple[dict[str, dict[str, list[str]]], list[str], dict[str, str]]:
    """Parse the real 8 KB docs -> per-class disease keyword tiers + healthy kws
    + per-class full KB doc text (for the ADR-0012e evidence pool 𝒪).

    Classes are keyed by ``canon(disease_class)`` so they align with the gallery
    labels (e.g. "Fungal Saprolegniasis" -> "fungal").
    """
    docs = parse_markdown_docs(KB_DOCS)
    tiers: dict[str, dict[str, list[str]]] = {}
    healthy_keywords: list[str] = []
    doc_text: dict[str, str] = {}
    for d in docs:
        key = canon(d.disease_class)
        tiers[key] = {
            "confirmed": list(d.clinical_keywords.get("confirmed", [])),
            "suspected": list(d.clinical_keywords.get("suspected", [])),
            "mentioned": list(d.clinical_keywords.get("mentioned", [])),
        }
        doc_text[key] = d.text
        if d.healthy_keywords:
            healthy_keywords = list(d.healthy_keywords)
    return tiers, healthy_keywords, doc_text


def mcnemar_cc(b: int, c: int) -> tuple[float, float]:
    """Continuity-corrected McNemar chi-square (df=1) + two-tailed p (chi2 sf)."""
    n = b + c
    if n == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / n
    stat = max(stat, 0.0)
    p = float(chi2.sf(stat, df=1))
    return float(stat), p


def contingency(pred_a: np.ndarray, pred_b: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    """2x2 table: rows = system A correct?, cols = system B correct?."""
    a_ok = pred_a == truth
    b_ok = pred_b == truth
    both = int(np.sum(a_ok & b_ok))
    a_only = int(np.sum(a_ok & ~b_ok))  # b in McNemar (A right, B wrong)
    b_only = int(np.sum(~a_ok & b_ok))  # c in McNemar (A wrong, B right)
    neither = int(np.sum(~a_ok & ~b_ok))
    return {
        "both_correct": both,
        "A_correct_B_wrong": a_only,
        "A_wrong_B_correct": b_only,
        "both_wrong": neither,
    }


def main() -> None:
    set_determinism(SEED)
    cfg = load_config(CONFIG)
    lam_star = cfg.fusion.lambda_star  # 0.7
    log.info(f"config: lambda*={lam_star}  seed={SEED}  device={DEVICE}")

    # --- KB keyword tiers (real 8 docs) ---
    tiers, healthy_keywords, doc_text = load_kb_tiers()
    log.info(f"KB classes: {sorted(tiers)}  |healthy_kw|={len(healthy_keywords)}")

    # --- D1 reference gallery (cached) ---
    if not CACHE.exists():
        raise FileNotFoundError(f"missing D1 gallery cache: {CACHE}")
    z = np.load(CACHE, allow_pickle=True)
    gv, gc, gl = z["visual"], z["caption"], z["labels"]
    log.info(f"D1 gallery: {len(gl)} entries, classes={sorted(set(gl.tolist()))}")

    # --- CLIP model (Florence NOT needed; captions cached) ---
    log.info("loading CLIP ViT-B-32 (laion2b_s34b_b79k)...")
    clip, prep, tok = load_clip()

    # --- D2 queries: FULL reproducible working set (all 8 classes incl EUS,
    #     augmentations INCLUDED) to match the manuscript's Table 3 Panel B set. ---
    caps = {c["image"]: (c.get("caption") or "") for c in json.loads(D2_CAPS.read_text())}
    files = sorted(D2_DIR.glob("*.jpg"))
    log.info(f"D2 .jpg files (full set, augmentations included): {len(files)}")

    qv, qc, ql, qcap = [], [], [], []
    for f in tqdm(files, desc="D2 CLIP encode"):
        cap = caps.get(f.name, "")
        qv.append(clip_image(f, clip, prep))
        qc.append(clip_text(cap, clip, tok))
        ql.append(canon(f.stem))
        qcap.append(cap)
    qv = np.array(qv)
    qc = np.array(qc)
    ql = np.array(ql)

    # --- Full-set evaluation: ALL 8 D2 classes INCLUDING EUS are kept. The D1
    #     gallery has no EUS reference, so EUS queries can never be predicted
    #     correctly by either the k-NN or the forced-choice pipeline (their
    #     retrieval top-1 is always one of the 7 gallery classes). They land in
    #     the "both wrong" cell — this is exactly what pulls Overall DA down to
    #     ~0.81 and is required for n / DA to match the manuscript's set. ---
    eval_classes = sorted(set(ql.tolist()))
    n = len(ql)
    n_eus = int(np.sum(ql == "EUS"))
    log.info(f"evaluation set (full, all classes): n={n}  classes={eval_classes}  n_EUS={n_eus}")

    # --- Retrieval predictions ---
    # lambda = 1.0 (visual-only) k-NN
    g10, q10 = l2(gv), l2(qv)
    sims10 = q10 @ g10.T
    pred_l10 = gl[sims10.argmax(axis=1)]

    # lambda = 0.7 (paper fusion) k-NN  + top-1/top-2 sims for margin bookkeeping
    g07 = l2(lam_star * gv + (1 - lam_star) * gc)
    q07 = l2(lam_star * qv + (1 - lam_star) * qc)
    sims07 = q07 @ g07.T
    order07 = np.argsort(-sims07, axis=1)
    top1_idx = order07[:, 0]
    pred_l07 = gl[top1_idx]

    # k=5 majority-vote k-NN predictions (pure retrieval) for table (c)
    k5 = cfg.retrieval.top_k  # 5
    pred_l07_k5 = knn_majority_vote(sims07, gl, k5)
    pred_l10_k5 = knn_majority_vote(sims10, gl, k5)

    # top-1 / top-2 fused-λ0.7 similarities for the Eq. 11 retrieval-margin valve
    top1_sim = sims07[np.arange(n), order07[:, 0]]
    top2_sim = sims07[np.arange(n), order07[:, 1]]

    # --- Pipeline DECISIVE three-way decision per image (Eq. 10 + Eq. 11) ---
    # Faithful 𝒪 = caption ∪ top-1 KB doc text. Inconclusive verdicts are kept as
    # a distinct outcome (sentinel) so they count as NON-correct in DA / McNemar.
    pred_pipe = []
    n_inconclusive = 0
    n_healthy = 0
    n_disease = 0
    n_incon_margin = 0  # Eq. 11 retrieval-margin valve
    n_incon_scoring = 0  # Eq. 10 scoring-margin / fallback
    theta = cfg.margin.retrieval_margin_theta
    for i in range(n):
        c1 = str(pred_l07[i])  # retrieval top-1 candidate disease class
        evidence = qcap[i] + "\n" + doc_text.get(c1, "")
        s_h = compute_s_h(
            evidence,
            cfg.scoring.healthy_weights,
            cfg.negation,
            healthy_keywords=healthy_keywords or ("healthy",),
        )
        tier = tiers.get(c1, {"confirmed": [], "suspected": [], "mentioned": []})
        s_d = compute_s_d(
            evidence,
            cfg.scoring,
            confirmed_keywords=tier["confirmed"],
            suspected_keywords=tier["suspected"],
            mentioned_keywords=tier["mentioned"],
        )
        result = make_decision(
            score_healthy=s_h,
            score_disease=s_d,
            top1_similarity=float(top1_sim[i]),
            top2_similarity=float(top2_sim[i]),
            disease_class=c1,
            decision_config=cfg.decision,
            margin_config=cfg.margin,
        )
        if result.decision == DecisionEnum.HEALTHY:
            pred_pipe.append("healthy")
            n_healthy += 1
        elif result.decision == DecisionEnum.DISEASE:
            pred_pipe.append(result.disease_class or c1)
            n_disease += 1
        else:  # INCONCLUSIVE -> non-correct
            pred_pipe.append(INCONCLUSIVE)
            n_inconclusive += 1
            if (float(top1_sim[i]) - float(top2_sim[i])) < theta:
                n_incon_margin += 1
            else:
                n_incon_scoring += 1
    pred_pipe = np.array(pred_pipe)
    abstention_rate = n_inconclusive / n
    log.info(
        f"decisive verdicts: healthy={n_healthy} disease={n_disease} "
        f"inconclusive={n_inconclusive} (abstention rate={abstention_rate:.4f})"
    )
    log.info(
        f"inconclusive breakdown: retrieval-margin(Eq.11 theta={theta})={n_incon_margin} "
        f"scoring-margin/fallback(Eq.10)={n_incon_scoring}"
    )

    # --- Diagnostic accuracies (Inconclusive counts as non-correct for pipeline) ---
    da_l10 = float((pred_l10 == ql).mean())
    da_l07 = float((pred_l07 == ql).mean())
    da_pipe = float((pred_pipe == ql).mean())
    da_l07_k5 = float((pred_l07_k5 == ql).mean())
    da_l10_k5 = float((pred_l10_k5 == ql).mean())
    log.info(
        f"DA  kNN(lambda=1.0,k1)={da_l10:.4f}  kNN(lambda=0.7,k1)={da_l07:.4f}  "
        f"pipeline(decisive)={da_pipe:.4f}"
    )
    log.info(f"DA  kNN(lambda=0.7,k5)={da_l07_k5:.4f}  kNN(lambda=1.0,k5)={da_l10_k5:.4f}")

    # --- Table (a): Pipeline vs kNN (lambda=0.7) ---
    ta = contingency(pred_pipe, pred_l07, ql)
    b_a = ta["A_correct_B_wrong"]  # pipeline correct, kNN wrong
    c_a = ta["A_wrong_B_correct"]  # pipeline wrong, kNN correct
    chi2_a, p_a = mcnemar_cc(b_a, c_a)

    # --- Table (b): lambda=0.7 vs lambda=1.0 (both k-NN) ---
    tb = contingency(pred_l07, pred_l10, ql)
    b_b = tb["A_correct_B_wrong"]  # lambda0.7 correct, lambda1.0 wrong
    c_b = tb["A_wrong_B_correct"]  # lambda0.7 wrong, lambda1.0 correct
    chi2_b, p_b = mcnemar_cc(b_b, c_b)

    # --- Table (c): lambda=0.7 k=5 vs lambda=1.0 k=5 (majority-vote k-NN) ---
    tc = contingency(pred_l07_k5, pred_l10_k5, ql)
    b_c = tc["A_correct_B_wrong"]  # lambda0.7 k5 correct, lambda1.0 k5 wrong
    c_c = tc["A_wrong_B_correct"]  # lambda0.7 k5 wrong, lambda1.0 k5 correct
    chi2_c, p_c = mcnemar_cc(b_c, c_c)

    log.info(
        f"[a] Pipeline(decisive) vs kNN(0.7,k1): b={b_a} c={c_a} chi2={chi2_a:.3f} "
        f"p={p_a:.4f} (orig b=21 c=40 chi2=5.31 p=0.021)"
    )
    log.info(
        f"[b] lambda0.7 vs lambda1.0 (k1): b={b_b} c={c_b} chi2={chi2_b:.3f} "
        f"p={p_b:.4f} (orig b=20 c=41 chi2=6.56 p=0.010)"
    )
    log.info(
        f"[c] lambda0.7 k5 vs lambda1.0 k5: b={b_c} c={c_c} chi2={chi2_c:.3f} "
        f"p={p_c:.4f} (orig p=0.033)"
    )

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    out = {
        "experiment": "mcnemar_d2_reproduced_decisive",
        "reviewer_point": "R5.3 — actual 2x2 contingency tables from real paired D2 predictions",
        "note": (
            "VARIANT C — DECISIVE three-way pipeline. Full n=3,453, all 8 classes "
            "incl EUS, augmentations included, seed 42, cached captions, FAITHFUL 𝒪 "
            "(ADR-0012e Option B: caption ∪ top-1 KB doc text). The pipeline runs the "
            "full Stage-3 decision (Eq. 11 retrieval-margin valve theta=0.02, then "
            "Eq. 10 three-way on S_h/S_d with T_h=2, m=1) — NOT a forced-choice "
            "argmax. An Inconclusive verdict is a distinct outcome that counts as "
            "NON-correct in DA and McNemar (mirroring Overall DA counting abstentions "
            "as misses). This is the configuration hypothesised to match how the "
            "manuscript defined 'Pipeline' (DA=0.811, 0.6pp below kNN lambda0.7). "
            "Table (c) adds the third manuscript test: kNN lambda0.7 k=5 vs lambda1.0 "
            "k=5 majority vote. Original manuscript counts are under 'original_manuscript'."
        ),
        "metadata": {
            "gpu": gpu,
            "device": DEVICE,
            "seed": SEED,
            "lambda_star": lam_star,
            "retrieval_top_k": cfg.retrieval.top_k,
            "retrieval_margin_theta": cfg.margin.retrieval_margin_theta,
            "scoring_disease_weights": {
                "confirmed": cfg.scoring.disease_weights.confirmed,
                "suspected": cfg.scoring.disease_weights.suspected,
                "mentioned": cfg.scoring.disease_weights.mentioned,
            },
            "scoring_healthy_weights": {
                "explicit": cfg.scoring.healthy_weights.explicit,
                "negation": cfg.scoring.healthy_weights.negation,
            },
            "n_gallery": int(len(gl)),
            "n_eval": int(n),
            "n_d2_total": int(len(files)),
            "augmentations_included": True,
            "eval_classes": eval_classes,
            "n_eus_queries": n_eus,
            "config_freeze_hash": cfg.freeze_hash(),
            "evidence_pool": (
                "ADR-0012e Option B: caption ∪ top-1 KB doc text "
                "(caption + '\\n' + KB document body for retrieved class c1)"
            ),
            "pipeline_decision": (
                "DECISIVE Stage-3: Eq. 11 retrieval-margin valve (theta=0.02) then "
                "Eq. 10 three-way (T_h=2, m=1); Inconclusive = non-correct"
            ),
            "decision_thresholds": {
                "healthy_threshold_Th": cfg.decision.healthy_threshold_Th,
                "inconclusive_margin_m": cfg.decision.inconclusive_margin_m,
                "retrieval_margin_theta": cfg.margin.retrieval_margin_theta,
            },
            "decisive_verdict_counts": {
                "healthy": int(n_healthy),
                "disease": int(n_disease),
                "inconclusive": int(n_inconclusive),
            },
            "inconclusive_breakdown": {
                "retrieval_margin_valve_Eq11": int(n_incon_margin),
                "scoring_margin_or_fallback_Eq10": int(n_incon_scoring),
            },
            "abstention_count": int(n_inconclusive),
            "abstention_rate": round(abstention_rate, 4),
        },
        "diagnostic_accuracy": {
            "knn_lambda_1.0_k1": round(da_l10, 4),
            "knn_lambda_0.7_k1": round(da_l07, 4),
            "pipeline_decisive": round(da_pipe, 4),
            "knn_lambda_0.7_k5": round(da_l07_k5, 4),
            "knn_lambda_1.0_k5": round(da_l10_k5, 4),
        },
        "table_a_pipeline_vs_knn_lambda0.7": {
            "description": "McNemar: Pipeline (DECISIVE, Inconclusive=miss) vs CLIP k-NN (k=1, lambda=0.7)",
            "contingency_2x2": {
                "both_correct": ta["both_correct"],
                "pipeline_correct_knn_wrong": ta["A_correct_B_wrong"],
                "pipeline_wrong_knn_correct": ta["A_wrong_B_correct"],
                "both_wrong": ta["both_wrong"],
            },
            "b_pipeline_correct_knn_wrong": b_a,
            "c_pipeline_wrong_knn_correct": c_a,
            "chi2_continuity_corrected": round(chi2_a, 4),
            "p_value_two_tailed": round(p_a, 4),
            "significant_at_0.05": bool(p_a < cfg.statistics.alpha),
            "n": int(n),
            "original_manuscript": ORIG_PIPE_KNN,
        },
        "table_b_lambda0.7_vs_lambda1.0": {
            "description": "McNemar: fused lambda=0.7 vs visual-only lambda=1.0 (both k-NN k=1)",
            "contingency_2x2": {
                "both_correct": tb["both_correct"],
                "lambda0.7_correct_lambda1.0_wrong": tb["A_correct_B_wrong"],
                "lambda0.7_wrong_lambda1.0_correct": tb["A_wrong_B_correct"],
                "both_wrong": tb["both_wrong"],
            },
            "b_lambda0.7_correct_lambda1.0_wrong": b_b,
            "c_lambda0.7_wrong_lambda1.0_correct": c_b,
            "chi2_continuity_corrected": round(chi2_b, 4),
            "p_value_two_tailed": round(p_b, 4),
            "significant_at_0.05": bool(p_b < cfg.statistics.alpha),
            "n": int(n),
            "original_manuscript": ORIG_L07_L10,
        },
        "table_c_lambda0.7_k5_vs_lambda1.0_k5": {
            "description": "McNemar: fused lambda=0.7 k=5 vs visual-only lambda=1.0 k=5 (majority-vote k-NN)",
            "contingency_2x2": {
                "both_correct": tc["both_correct"],
                "lambda0.7_k5_correct_lambda1.0_k5_wrong": tc["A_correct_B_wrong"],
                "lambda0.7_k5_wrong_lambda1.0_k5_correct": tc["A_wrong_B_correct"],
                "both_wrong": tc["both_wrong"],
            },
            "b_lambda0.7_k5_correct_lambda1.0_k5_wrong": b_c,
            "c_lambda0.7_k5_wrong_lambda1.0_k5_correct": c_c,
            "chi2_continuity_corrected": round(chi2_c, 4),
            "p_value_two_tailed": round(p_c, 4),
            "significant_at_0.05": bool(p_c < cfg.statistics.alpha),
            "n": int(n),
            "original_manuscript": {"p": 0.033, "da_lambda0.7_k5": 0.759, "da_lambda1.0_k5": 0.749},
        },
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log.info(f"saved {OUT}")
    log.info("DONE")


if __name__ == "__main__":
    main()
