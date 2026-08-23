"""Re-run the D2 retrieval analyses on the D1-DISJOINT clean subset (post D1<->D2 audit).

The D1<->D2 overlap audit found ~44% of D2 is a (near-)duplicate of D1, so D2-query ->
D1-gallery retrieval self-retrieves duplicates and inflates accuracy. This script rebuilds
a clean D2 subset (images whose perceptual hash is > T bits from EVERY D1 image), then
re-computes the affected numbers against the cached D1 gallery:

  * modality comparison  : caption-only (lambda=0) vs visual (lambda=1) top-1 DA
  * lambda-sweep         : caption-only / fusion(0.7) / visual top-1 DA, 7 shared classes
  * 8-class Overall DA   : lambda=0.7 (incl. EUS, which the gallery lacks -> always wrong)

Real models only (OpenCLIP ViT-B-32 LAION-2B); cached Florence-2 captions reused.
Writes results/d2_clean_rerun.json.
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
PHASH_T = 5          # disjoint = pHash Hamming > 5 from every D1 image
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def imgs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)


def phash_bits(p: Path) -> np.ndarray:
    with Image.open(p) as im:
        return imagehash.phash(im.convert("RGB")).hash.flatten().astype(np.uint8)


def l2(a): return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def main() -> None:
    # ---- D1 gallery (cached) ----
    z = np.load(CACHE, allow_pickle=True)
    gv, gc, gl = z["visual"], z["caption"], z["labels"]
    print(f"D1 gallery: {len(gl)}")

    # ---- clean mask: D2 images disjoint from ALL D1 (pHash Hamming > T) ----
    d1_all, d2_all = imgs(D1_DIR), imgs(D2_DIR)
    B1 = np.stack([phash_bits(p) for p in tqdm(d1_all, desc="D1 pHash")]).astype(np.float32)
    d2_bits, d2_paths = [], []
    for p in tqdm(d2_all, desc="D2 pHash"):
        d2_bits.append(phash_bits(p)); d2_paths.append(p)
    B2 = np.stack(d2_bits).astype(np.float32)
    ham = B2 @ (1 - B1).T + (1 - B2) @ B1.T      # (N2, N1)
    mind = ham.min(axis=1)
    clean = mind > PHASH_T
    print(f"D2 total={len(d2_paths)}  clean(disjoint)={int(clean.sum())}  contaminated={int((~clean).sum())}")

    # ---- encode clean D2 (visual via CLIP, caption via cached Florence text) ----
    import open_clip
    clip, _, prep = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=DEVICE)
    clip.eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    caps = {c["image"]: (c.get("caption") or "") for c in json.loads(D2_CAPS.read_text())}

    def enc_img(path):
        with torch.no_grad():
            x = prep(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
            f = clip.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
        return f[0].cpu().numpy().astype(np.float32)

    def enc_txt(text):
        with torch.no_grad():
            t = tok([text or "fish"]).to(DEVICE)
            f = clip.encode_text(t); f = f / f.norm(dim=-1, keepdim=True)
        return f[0].cpu().numpy().astype(np.float32)

    clean_paths = [p for p, c in zip(d2_paths, clean) if c]
    qv, qc, ql, noaug = [], [], [], []
    for p in tqdm(clean_paths, desc="clean D2 encode"):
        qv.append(enc_img(p)); qc.append(enc_txt(caps.get(p.name, "")))
        ql.append(canon(p.stem)); noaug.append("_aug" not in p.stem)
    qv, qc, ql, noaug = np.array(qv), np.array(qc), np.array(ql), np.array(noaug)

    def da(qvv, qcc, qll, lam):
        g = l2(lam * gv + (1 - lam) * gc)
        q = l2(lam * qvv + (1 - lam) * qcc)
        pred = gl[(q @ g.T).argmax(axis=1)]
        return round(float((pred == qll).mean()), 4), len(qll)

    shared = sorted((set(gl.tolist()) & set(ql.tolist())) - {"EUS", "UNKNOWN"})
    m7 = np.isin(ql, shared)                       # 7 shared classes
    m7na = m7 & noaug                              # + augmentation-free
    results = {
        "n_d2_total": len(d2_paths),
        "n_d2_clean_disjoint": int(clean.sum()),
        "n_clean_7class": int(m7.sum()),
        "n_clean_7class_noaug": int(m7na.sum()),
        "clean_7class_noaug": {
            "caption_only_lambda0.0": da(qv[m7na], qc[m7na], ql[m7na], 0.0)[0],
            "fusion_lambda0.7": da(qv[m7na], qc[m7na], ql[m7na], 0.7)[0],
            "visual_only_lambda1.0": da(qv[m7na], qc[m7na], ql[m7na], 1.0)[0],
        },
        "clean_7class_full": {
            "caption_only_lambda0.0": da(qv[m7], qc[m7], ql[m7], 0.0)[0],
            "fusion_lambda0.7": da(qv[m7], qc[m7], ql[m7], 0.7)[0],
            "visual_only_lambda1.0": da(qv[m7], qc[m7], ql[m7], 1.0)[0],
        },
        "clean_8class_overall_lambda0.7": da(qv, qc, ql, 0.7)[0],
        "lambda_sweep_7class_noaug": {
            f"{lam:.1f}": da(qv[m7na], qc[m7na], ql[m7na], lam)[0]
            for lam in [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]
        },
        "original_contaminated_for_comparison": {
            "caption_only": 0.720, "visual_only": 0.940, "fusion_0.7": 0.943,
            "modality_caption": "0.536/0.569", "modality_visual": 0.934,
            "note": "original numbers were on D2 including ~44% D1 duplicates",
        },
    }
    (RES / "d2_clean_rerun.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
