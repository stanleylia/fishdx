"""Positively identify the manuscript's Table 3 Panel A configuration (R5.3).

Hypothesis (from the margin-scale diagnostic): the manuscript's Panel A
(b≈21, c≈40, Pipeline DA≈0.811, ~1% abstention) is NOT reproducible under the
faithful ADR-0012e evidence pool (𝒪 = caption ∪ top-1 KB doc text), because
that pool structurally forces cell b = 0 (pipeline never diverges from the kNN
top-1 class). The b=21 signature requires a *class-changing healthy-rescue* that
only the lighter CAPTION-ONLY pool permits (S_h can flip a mis-retrieved disease
to Healthy, giving a pipeline-correct / kNN-wrong pair).

This run tests that hypothesis directly, holding everything else fixed
(full n=3,453, all 8 classes incl EUS, seed 42):
  - 𝒪 = CAPTION ONLY.
  - Stage-3 = full Eq. 10 three-way + Eq. 11 retrieval-margin valve.
  - Sweep θ_margin ∈ {0.005, 0.003, 0.002, 0.001, 0.0} to reach ~1% abstention.
For each θ: abstention%, Pipeline decisive DA (Inconclusive = miss), and table
(a) Pipeline vs kNN λ0.7 k=1 full 2×2 + χ² + p. Flags any θ hitting
abstention≈1% AND DA≈0.811 AND b≈21 AND c≈40.

Output: results/mcnemar_d2_panelA_identification.json
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
OUT = RES / "mcnemar_d2_panelA_identification.json"

SEED = 42
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
INCONCLUSIVE = "__INCONCLUSIVE__"
THETA_SWEEP = [0.005, 0.003, 0.002, 0.001, 0.0]
ORIG = {"b": 21, "c": 40, "chi2": 5.31, "p": 0.021, "pipeline_DA": 0.811, "abstention": "~1%"}

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
    for d in docs:
        key = canon(d.disease_class)
        tiers[key] = {
            "confirmed": list(d.clinical_keywords.get("confirmed", [])),
            "suspected": list(d.clinical_keywords.get("suspected", [])),
            "mentioned": list(d.clinical_keywords.get("mentioned", [])),
        }
        if d.healthy_keywords:
            healthy_keywords = list(d.healthy_keywords)
    return tiers, healthy_keywords


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
    t_h = cfg.decision.healthy_threshold_Th
    m_margin = cfg.decision.inconclusive_margin_m
    log.info(f"config: lambda*={lam} T_h={t_h} m={m_margin} seed={SEED} (theta swept: {THETA_SWEEP})")

    tiers, healthy_keywords = load_kb()

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

    # fused lambda=0.7 retrieval
    g07 = l2(lam * gv + (1 - lam) * gc)
    q07 = l2(lam * qv + (1 - lam) * qc)
    sims07 = q07 @ g07.T
    order07 = np.argsort(-sims07, axis=1)
    pred_knn07 = gl[order07[:, 0]]
    sim1 = sims07[np.arange(n), order07[:, 0]]
    sim2 = sims07[np.arange(n), order07[:, 1]]
    raw_margin = sim1 - sim2
    da_knn07 = float((pred_knn07 == ql).mean())
    log.info(f"kNN(lambda=0.7,k1) DA={da_knn07:.4f}")

    # --- CAPTION-ONLY base decision per image (Eq. 10 steps 2-5; retrieval valve
    #     applied per-theta afterwards). The healthy-rescue path (S_h >= T_h and
    #     S_h > S_d with c1 a disease) can flip the committed class to 'healthy'. ---
    base_pred = []
    base_inconclusive = np.zeros(n, dtype=bool)
    n_healthy_rescue = 0
    for i in range(n):
        c1 = str(pred_knn07[i])
        evidence = qcap[i]  # CAPTION ONLY
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
        if abs(s_h - s_d) <= m_margin:
            base_pred.append(INCONCLUSIVE)
            base_inconclusive[i] = True
        elif s_h >= t_h and s_h > s_d:
            base_pred.append("healthy")
            if c1 != "healthy":
                n_healthy_rescue += 1
        elif s_d > s_h:
            base_pred.append(c1)
        else:
            base_pred.append(INCONCLUSIVE)
            base_inconclusive[i] = True
    base_pred = np.array(base_pred)
    log.info(
        f"caption-only base: scoring-inconclusive={int(base_inconclusive.sum())} "
        f"healthy-rescues(class flip off kNN)={n_healthy_rescue}"
    )

    def evaluate(thr: float) -> dict[str, object]:
        abstain = (raw_margin < thr) | base_inconclusive if thr > 0 else base_inconclusive.copy()
        pred = np.where(abstain, INCONCLUSIVE, base_pred)
        da = float((pred == ql).mean())
        t = contingency(pred, pred_knn07, ql)
        b, c = t["A_correct_B_wrong"], t["A_wrong_B_correct"]
        stat, p = mcnemar_cc(b, c)
        rate = float(abstain.mean())
        hit = bool(
            0.005 <= rate <= 0.02
            and 0.806 <= da <= 0.816
            and 15 <= b <= 27
            and 33 <= c <= 47
        )
        log.info(
            f"[theta={thr}] abstain={rate*100:.2f}%  PipeDA={da:.4f}  "
            f"b={b} c={c} chi2={stat:.3f} p={p:.4f}  panelA_hit={hit}"
        )
        return {
            "theta_margin": thr,
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
            "panelA_match": hit,
        }

    sweep = [evaluate(t) for t in THETA_SWEEP]

    # max achievable b (theta=0 keeps only scoring-driven class changes = healthy rescues)
    max_b = max(int(r["b_pipeline_correct_knn_wrong"]) for r in sweep)
    matches = [r["theta_margin"] for r in sweep if r["panelA_match"]]
    # closest point to the manuscript by |DA-0.811| + |abstain-0.01|
    closest = min(
        sweep,
        key=lambda r: abs(float(r["pipeline_decisive_DA"]) - 0.811)
        + abs(float(r["abstention_rate"]) - 0.01),
    )

    if matches:
        verdict_text = (
            f"YES — caption-only 𝒪 + retrieval-margin theta in {matches} reproduces "
            f"the manuscript's Panel A (abstention≈1%, Pipeline DA≈0.811, b≈21, c≈40). "
            f"This identifies how Panel A was computed and vindicates it: "
            f"𝒪=caption-only, theta={matches[0]}."
        )
    else:
        verdict_text = (
            f"NO — caption-only 𝒪 + low-theta margin does NOT reproduce Panel A. The "
            f"class-changing healthy-rescue path yields at most b={max_b} "
            f"(manuscript b=21), so cell b tops out {'below' if max_b < 15 else 'near but the joint (DA, abstain, b, c) target is not met'}. "
            f"Closest point: theta={closest['theta_margin']}, abstain="
            f"{float(closest['abstention_rate'])*100:.2f}%, PipeDA={closest['pipeline_decisive_DA']}, "
            f"b={closest['b_pipeline_correct_knn_wrong']}, c={closest['c_pipeline_wrong_knn_correct']}."
        )
    log.info(verdict_text)

    out = {
        "experiment": "mcnemar_d2_panelA_identification",
        "reviewer_point": "R5.3 — positively identify the manuscript's Table 3 Panel A configuration",
        "note": (
            "Full n=3,453, all 8 classes incl EUS, seed 42. Evidence pool 𝒪 = "
            "CAPTION ONLY (permits class-changing healthy-rescue). Stage-3 = full "
            "Eq. 10 three-way + Eq. 11 retrieval-margin valve; theta swept low to "
            "target ~1% abstention. Inconclusive = non-correct. Manuscript Panel A "
            "target under 'original_manuscript'."
        ),
        "metadata": {
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "seed": SEED,
            "lambda_star": lam,
            "T_h": t_h,
            "m": m_margin,
            "evidence_pool": "CAPTION ONLY",
            "theta_sweep": THETA_SWEEP,
            "n_eval": int(n),
            "n_gallery": int(len(gl)),
            "knn_lambda0.7_k1_DA": round(da_knn07, 4),
            "caption_only_healthy_rescues": int(n_healthy_rescue),
            "caption_only_scoring_inconclusive": int(base_inconclusive.sum()),
            "max_achievable_b": int(max_b),
            "config_freeze_hash": cfg.freeze_hash(),
        },
        "original_manuscript": ORIG,
        "theta_sweep_results": sweep,
        "panelA_matching_thetas": matches,
        "closest_point": closest,
        "verdict": verdict_text,
        "reproduces_manuscript_panelA": bool(matches),
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    log.info(f"saved {OUT}")
    log.info(f"VERDICT reproduces={bool(matches)} matches={matches} max_b={max_b}")
    log.info("DONE")


if __name__ == "__main__":
    main()
