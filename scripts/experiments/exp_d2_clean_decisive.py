"""Clean Decisive-DA / selective-prediction sweep on the D1-DISJOINT D2 subset.

Recomputes the paper's headline selective-prediction result (Decisive DA = 0.924 at
theta_margin = 0.02, originally on contaminated D2) on the clean subset (D2 images whose
perceptual hash is > 5 bits from every D1 image). For each theta:
  * decided = (top1_sim - top2_sim) >= theta        (else Inconclusive / abstain)
  * Decisive DA = accuracy over decided images
  * Coverage    = fraction decided
Also reports how the gate treats the out-of-distribution EUS class (which the D1 gallery
lacks) -- the selective-prediction thesis requires the gate to abstain on those.

Real OpenCLIP ViT-B-32 LAION-2B; cached Florence-2 captions; fused lambda=0.7.
Writes results/d2_clean_decisive.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import imagehash
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
D1_DIR = ROOT / "data/datasets/D1"
D2_DIR = ROOT / "data/datasets/D2"
RES = ROOT / "results"
CACHE = RES / "_cache_d1_gallery.npz"
D2_CAPS = RES / "e3_captions_log.json"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LAM = 0.7
PHASH_T = 5
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
THETAS = [0.00, 0.01, 0.02, 0.05]


def canon(stem: str) -> str:
    s = re.sub(r"[\s_]*\(?\d+\)?$", "", stem).strip()
    s = re.sub(r"_aug$", "", s).strip()
    m = re.match(r"^(.*)_\1$", s)
    if m:
        s = m.group(1)
    s = s.lower()
    if "healthy" in s: return "healthy"
    if "aeromon" in s: return "aeromoniasis"
    if "gill" in s: return "bacterial_gill"
    if "red" in s: return "bacterial_red"
    if "fungal" in s or "saproleg" in s: return "fungal"
    if "parasit" in s: return "parasitic"
    if "white tail" in s or "viral" in s: return "viral_white_tail"
    if "eus" in s or "ulcerative" in s: return "EUS"
    return "UNKNOWN"


def imgs(root): return sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)
def phash_bits(p):
    with Image.open(p) as im:
        return imagehash.phash(im.convert("RGB")).hash.flatten().astype(np.uint8)
def l2(a): return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def main() -> None:
    z = np.load(CACHE, allow_pickle=True)
    gv, gc, gl = z["visual"], z["caption"], z["labels"]

    d1_all, d2_all = imgs(D1_DIR), imgs(D2_DIR)
    B1 = np.stack([phash_bits(p) for p in tqdm(d1_all, desc="D1 pHash")]).astype(np.float32)
    B2, paths = [], []
    for p in tqdm(d2_all, desc="D2 pHash"):
        B2.append(phash_bits(p)); paths.append(p)
    B2 = np.stack(B2).astype(np.float32)
    mind = (B2 @ (1 - B1).T + (1 - B2) @ B1.T).min(axis=1)
    clean_paths = [p for p, ok in zip(paths, mind > PHASH_T) if ok]

    import open_clip
    clip, _, prep = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=DEVICE)
    clip.eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    caps = {c["image"]: (c.get("caption") or "") for c in json.loads(D2_CAPS.read_text())}

    def enc_img(path):
        with torch.no_grad():
            x = prep(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
            f = clip.encode_image(x); return (f / f.norm(dim=-1, keepdim=True))[0].cpu().numpy().astype(np.float32)

    def enc_txt(text):
        with torch.no_grad():
            t = tok([text or "fish"]).to(DEVICE)
            f = clip.encode_text(t); return (f / f.norm(dim=-1, keepdim=True))[0].cpu().numpy().astype(np.float32)

    qv, qc, ql = [], [], []
    for p in tqdm(clean_paths, desc="clean D2 encode"):
        qv.append(enc_img(p)); qc.append(enc_txt(caps.get(p.name, ""))); ql.append(canon(p.stem))
    qv, qc, ql = np.array(qv), np.array(qc), np.array(ql)

    g = l2(LAM * gv + (1 - LAM) * gc)
    q = l2(LAM * qv + (1 - LAM) * qc)
    sims = q @ g.T
    order = np.sort(sims, axis=1)
    top1, top2 = order[:, -1], order[:, -2]
    margin = top1 - top2
    pred = gl[sims.argmax(axis=1)]
    correct = pred == ql
    is_eus = ql == "EUS"
    n = len(ql)

    sweep = {}
    for th in THETAS:
        decided = margin >= th
        cov = float(decided.mean())
        dec_da = float(correct[decided].mean()) if decided.any() else 0.0
        # EUS handling: what fraction of EUS is (correctly) abstained?
        eus_abstained = float((~decided & is_eus).sum() / max(1, is_eus.sum()))
        sweep[f"theta_{th:.2f}"] = {
            "coverage": round(cov, 4),
            "abstention": round(1 - cov, 4),
            "decisive_DA": round(dec_da, 4),
            "n_decided": int(decided.sum()),
            "eus_abstained_frac": round(eus_abstained, 4),
        }

    out = {
        "experiment": "d2_clean_decisive_selective_prediction",
        "subset": "D2 disjoint from D1 (pHash>5)",
        "n_clean": n, "n_eus": int(is_eus.sum()),
        "lambda": LAM,
        "forced_choice_overall_DA": round(float(correct.mean()), 4),
        "sweep": sweep,
        "original_contaminated": {"decisive_DA": 0.924, "coverage": 0.643, "at_theta": 0.02},
    }
    RES.mkdir(exist_ok=True)
    (RES / "d2_clean_decisive.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
