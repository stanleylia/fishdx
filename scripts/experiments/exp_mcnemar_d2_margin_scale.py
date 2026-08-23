"""Margin-scale diagnostic for D2 Table 3 Panel A (reviewer R5.3).

Question: is the manuscript's Panel A (b=21/c=40, Pipeline DA=0.811, ~1%
abstention) *correct-but-computed-on-a-different-margin-scale*, or wrong?

The decisive pipeline (faithful 𝒪 = caption ∪ top-1 KB doc text) over-abstains
at 37% when the Eq. 11 retrieval-margin valve uses raw fused cosine (sim1-sim2)
with θ=0.02. This script holds EVERYTHING fixed (full n=3,453, all 8 classes
incl EUS, faithful 𝒪, seed 42, θ_margin=0.02 from config) and varies ONLY the
retrieval-margin definition, reporting abstention rate, Pipeline decisive DA,
and table (a) 2×2 (b/c/χ²/p) for each:

  1. raw cosine        margin = sim1 − sim2
  2. squared-L2        margin = 2(sim1 − sim2)          (ChromaDB 'l2' default)
  3. relative/rank     margin = (sim1 − sim2)/sim1
  4. class-prototype   margin = simproto1 − simproto2   (L2-norm class means)
  5. raw-cosine θ sweep: θ ∈ {0.02, 0.01, 0.005, 0.002}

Pipeline committed class is the faithful-𝒪 decision WITHOUT the retrieval-margin
valve (base class); each margin definition only decides the Inconclusive valve.
An Inconclusive verdict counts as NON-correct (a miss).

Output: results/mcnemar_d2_margin_scale_diagnostic.json
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
from fishdx.scoring.score import compute_s_d, compute_s_h

ROOT = Path(__file__).resolve().parents[2]
D2_DIR = ROOT / "data/datasets/D2"
RES = ROOT / "results"
CACHE = RES / "_cache_d1_gallery.npz"
D2_CAPS = RES / "e3_captions_log.json"
KB_DOCS = ROOT / "src/fishdx/kb/documents"
CONFIG = ROOT / "configs/default.yaml"
OUT = RES / "mcnemar_d2_margin_scale_diagnostic.json"

SEED = 42
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
INCONCLUSIVE = "__INCONCLUSIVE__"
ORIG_PIPE_KNN = {"b": 21, "c": 40, "chi2": 5.31, "p": 0.021, "pipeline_DA": 0.811, "abstention": "~1%"}

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


def load_kb():
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
    n = b + c
    if n == 0:
        return 0.0, 1.0
    stat = max((abs(b - c) - 1) ** 2 / n, 0.0)
    return float(stat), float(chi2.sf(stat, df=1))


def contingency(pred_a: np.ndarray, pred_b: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    a_ok, b_ok = pred_a == truth, pred_b == truth
    return {
        "both_correct": int(np.sum(a_ok & b_ok)),
        "A_correct_B_wrong": int(np.sum(a_ok & ~b_ok)),
        "A_wrong_B_correct": int(np.sum(~a_ok & b_ok)),
        "both_wrong": int(np.sum(~a_ok & ~b_ok)),
    }


def main() -> None:
    set_determinism(SEED)
    cfg = load_config(CONFIG)
    lam = cfg.fusion.lambda_star
    theta = cfg.margin.retrieval_margin_theta
    t_h = cfg.decision.healthy_threshold_Th
    m_margin = cfg.decision.inconclusive_margin_m
    log.info(f"config: lambda*={lam} theta_margin={theta} T_h={t_h} m={m_margin} seed={SEED}")

    tiers, healthy_keywords, doc_text = load_kb()

    z = np.load(CACHE, allow_pickle=True)
    gv, gc, gl = z["visual"], z["caption"], z["labels"]

    clip, prep, tok = load_clip()
    caps = {c["image"]: (c.get("caption") or "") for c in json.loads(D2_CAPS.read_text())}
    files = sorted(D2_DIR.glob("*.jpg"))
    log.info(f"D2 files (full, aug incl): {len(files)}")

    qv, qc, ql, qcap = [], [], [], []
    for f in tqdm(files, desc="D2 CLIP encode"):
        qv.append(clip_image(f, clip, prep))
        qc.append(clip_text(caps.get(f.name, ""), clip, tok))
        ql.append(canon(f.stem))
        qcap.append(caps.get(f.name, ""))
    qv, qc, ql = np.array(qv), np.array(qc), np.array(ql)
    n = len(ql)
    log.info(f"eval set (all 8 classes): n={n}  n_EUS={int(np.sum(ql == 'EUS'))}")

    # --- fused lambda=0.7 retrieval ---
    g07 = l2(lam * gv + (1 - lam) * gc)
    q07 = l2(lam * qv + (1 - lam) * qc)
    sims07 = q07 @ g07.T
    order07 = np.argsort(-sims07, axis=1)
    pred_knn07 = gl[order07[:, 0]]  # kNN lambda=0.7 k=1 (reference system B)
    sim1 = sims07[np.arange(n), order07[:, 0]]
    sim2 = sims07[np.arange(n), order07[:, 1]]
    da_knn07 = float((pred_knn07 == ql).mean())
    log.info(f"kNN(lambda=0.7,k1) DA={da_knn07:.4f}")

    # --- class prototypes (L2-norm class means of fused lambda=0.7 gallery) ---
    gal_classes = sorted(set(gl.tolist()))
    protos = np.array([l2(g07[gl == c].mean(axis=0, keepdims=True))[0] for c in gal_classes])
    psims = q07 @ protos.T
    porder = np.argsort(-psims, axis=1)
    psim1 = psims[np.arange(n), porder[:, 0]]
    psim2 = psims[np.arange(n), porder[:, 1]]

    # --- base decisive class per image (faithful 𝒪, WITHOUT retrieval-margin valve) ---
    base_pred = []
    base_inconclusive = np.zeros(n, dtype=bool)
    for i in range(n):
        c1 = str(pred_knn07[i])
        evidence = qcap[i] + "\n" + doc_text.get(c1, "")
        s_h = compute_s_h(
            evidence, cfg.scoring.healthy_weights, cfg.negation,
            healthy_keywords=healthy_keywords or ("healthy",),
        )
        tier = tiers.get(c1, {"confirmed": [], "suspected": [], "mentioned": []})
        s_d = compute_s_d(
            evidence, cfg.scoring,
            confirmed_keywords=tier["confirmed"],
            suspected_keywords=tier["suspected"],
            mentioned_keywords=tier["mentioned"],
        )
        # Eq. 10 steps 2-5 (retrieval-margin valve applied separately per definition)
        if abs(s_h - s_d) <= m_margin:
            base_pred.append(INCONCLUSIVE)
            base_inconclusive[i] = True
        elif s_h >= t_h and s_h > s_d:
            base_pred.append("healthy")
        elif s_d > s_h:
            base_pred.append(c1)
        else:
            base_pred.append(INCONCLUSIVE)
            base_inconclusive[i] = True
    base_pred = np.array(base_pred)
    log.info(f"base scoring-only inconclusive (no retrieval valve): {int(base_inconclusive.sum())}")

    def evaluate(name: str, margin_arr: np.ndarray, thr: float) -> dict[str, object]:
        abstain = (margin_arr < thr) | base_inconclusive
        pred = np.where(abstain, INCONCLUSIVE, base_pred)
        da = float((pred == ql).mean())
        t = contingency(pred, pred_knn07, ql)
        b, c = t["A_correct_B_wrong"], t["A_wrong_B_correct"]
        stat, p = mcnemar_cc(b, c)
        rate = float(abstain.mean())
        log.info(
            f"[{name}] abstain={rate*100:.1f}%  PipeDA={da:.4f}  "
            f"b={b} c={c} chi2={stat:.3f} p={p:.4f}"
        )
        return {
            "definition": name,
            "theta": thr,
            "abstention_count": int(abstain.sum()),
            "abstention_rate": round(rate, 4),
            "pipeline_decisive_DA": round(da, 4),
            "contingency_2x2": {
                "both_correct": t["both_correct"],
                "pipeline_correct_knn_wrong": b,
                "pipeline_wrong_knn_correct": c,
                "both_wrong": t["both_wrong"],
            },
            "b_pipeline_correct_knn_wrong": b,
            "c_pipeline_wrong_knn_correct": c,
            "chi2_continuity_corrected": round(stat, 4),
            "p_value_two_tailed": round(p, 4),
            "significant_at_0.05": bool(p < cfg.statistics.alpha),
        }

    raw = sim1 - sim2
    results = [
        evaluate("1_raw_cosine", raw, theta),
        evaluate("2_squared_L2_chromadb", 2.0 * raw, theta),
        evaluate("3_relative_rank", raw / np.clip(sim1, 1e-9, None), theta),
        evaluate("4_class_prototype", psim1 - psim2, theta),
    ]
    sweep = [
        evaluate(f"5_raw_cosine_theta_{t}", raw, t) for t in (0.02, 0.01, 0.005, 0.002)
    ]

    # verdict
    def near(r: dict[str, object]) -> bool:
        return (
            0.005 <= float(r["abstention_rate"]) <= 0.02  # ~1%
            and 15 <= int(r["b_pipeline_correct_knn_wrong"]) <= 27  # b≈21
            and 33 <= int(r["c_pipeline_wrong_knn_correct"]) <= 47  # c≈40
            and 0.806 <= float(r["pipeline_decisive_DA"]) <= 0.816  # DA≈0.811
        )

    all_defs = results + sweep
    matches = [r["definition"] for r in all_defs if near(r)]
    b_always_zero = all(int(r["b_pipeline_correct_knn_wrong"]) == 0 for r in all_defs)

    verdict = {
        "any_definition_reproduces_manuscript_panel_A": bool(matches),
        "matching_definitions": matches,
        "b_is_zero_for_all_definitions": b_always_zero,
        "structural_finding": (
            "Under faithful 𝒪 the decisive pipeline's committed class is always the "
            "kNN lambda=0.7 top-1 class (scoring confirms c1; the only class-changing "
            "path — healthy rescue of a mis-retrieved disease — is suppressed because "
            "c1's own KB doc text saturates S_d). Therefore 'pipeline correct & kNN "
            "wrong' (cell b) is structurally 0 for EVERY margin definition: the "
            "retrieval-margin valve can only turn kNN-correct predictions into misses "
            "(inflating c) or abstain on kNN-wrong ones (no effect on b). The "
            "manuscript's b=21 (pipeline right where kNN wrong) cannot arise from any "
            "margin scale under faithful 𝒪 — it requires a class-changing mechanism "
            "(caption-driven healthy rescue), which is the lighter caption-only 𝒪."
            if b_always_zero else
            "b is non-zero for at least one definition; see matching_definitions."
        ),
    }

    out = {
        "experiment": "mcnemar_d2_margin_scale_diagnostic",
        "reviewer_point": "R5.3 — is manuscript Panel A correct-on-a-different-margin-scale, or wrong?",
        "note": (
            "Full n=3,453, all 8 classes incl EUS, faithful 𝒪 (caption ∪ top-1 KB "
            "doc text), seed 42. Only the Eq. 11 retrieval-margin valve DEFINITION "
            "varies; committed class = faithful-𝒪 decisive class without the valve. "
            "Inconclusive = non-correct. Manuscript Panel A target under "
            "'original_manuscript'."
        ),
        "metadata": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "seed": SEED,
            "lambda_star": lam,
            "theta_margin_config": theta,
            "T_h": t_h,
            "m": m_margin,
            "n_eval": int(n),
            "n_gallery": int(len(gl)),
            "knn_lambda0.7_k1_DA": round(da_knn07, 4),
            "evidence_pool": "faithful 𝒪 = caption ∪ top-1 KB doc text (ADR-0012e)",
            "config_freeze_hash": cfg.freeze_hash(),
        },
        "original_manuscript": ORIG_PIPE_KNN,
        "margin_definitions": results,
        "raw_cosine_theta_sweep": sweep,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log.info(f"saved {OUT}")
    log.info(f"VERDICT: reproduces={bool(matches)} matches={matches} b_always_zero={b_always_zero}")
    log.info("DONE")


if __name__ == "__main__":
    main()
