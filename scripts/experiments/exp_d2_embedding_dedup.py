"""Embedding-space dedup of D2 vs D1 gallery (catches augmentation leakage pHash misses).

pHash>5 removes exact/near byte-duplicates, but an AUGMENTED copy of a D1 image can pass
pHash while still sitting almost on top of its D1 original in CLIP space -> it self-retrieves
and leaks. This script deduplicates D2 against the D1 gallery in the actual retrieval space
(max CLIP visual cosine) and reports how the retrieval DA moves as the dedup threshold
tightens, so we can pick a defensible clean number.

Writes results/d2_embedding_dedup.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
D2_DIR = ROOT / "data/datasets/D2"
RES = ROOT / "results"
CACHE = RES / "_cache_d1_gallery.npz"
D2_CAPS = RES / "e3_captions_log.json"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def canon(stem: str) -> str:
    s = re.sub(r"[\s_]*\(?\d+\)?$", "", stem).strip()
    s = re.sub(r"_aug$", "", s).strip()
    m = re.match(r"^(.*)_\1$", s)
    if m: s = m.group(1)
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


def l2(a): return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def main() -> None:
    z = np.load(CACHE, allow_pickle=True)
    gv, gc, gl = z["visual"], z["caption"], z["labels"]

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

    paths = sorted(p for p in D2_DIR.rglob("*") if p.suffix.lower() in EXTS)
    qv, qc, ql, isaug = [], [], [], []
    for p in tqdm(paths, desc="D2 encode"):
        qv.append(enc_img(p)); qc.append(enc_txt(caps.get(p.name, "")))
        ql.append(canon(p.stem)); isaug.append("_aug" in p.stem)
    qv, qc, ql, isaug = np.array(qv), np.array(qc), np.array(ql), np.array(isaug)

    # max VISUAL cosine of each D2 image to the D1 gallery (duplicate detector)
    max_cos = (qv @ gv.T).max(axis=1)
    print("max-cos-to-gallery percentiles:",
          {p: round(float(np.percentile(max_cos, p)), 3) for p in (50, 75, 90, 95, 99)})

    def da(mask, lam):
        shared = sorted((set(gl.tolist()) & set(ql[mask].tolist())) - {"EUS", "UNKNOWN"})
        m = mask & np.isin(ql, shared)
        g = l2(lam * gv + (1 - lam) * gc)
        q = l2(lam * qv[m] + (1 - lam) * qc[m])
        pred = gl[(q @ g.T).argmax(axis=1)]
        return round(float((pred == ql[m]).mean()), 4), int(m.sum())

    rows = {}
    all_mask = np.ones(len(ql), bool)
    for name, mask in [
        ("raw_all", all_mask),
        ("phash_clean_proxy(cos<0.99)", max_cos < 0.99),
        ("cos<0.98", max_cos < 0.98),
        ("cos<0.95", max_cos < 0.95),
        ("cos<0.90", max_cos < 0.90),
        ("noaug_only", ~isaug),
        ("noaug_AND_cos<0.95", (~isaug) & (max_cos < 0.95)),
    ]:
        v, nv = da(mask, 1.0)
        f, _ = da(mask, 0.7)
        c, _ = da(mask, 0.0)
        rows[name] = {"n_7class": nv, "caption_0.0": c, "fusion_0.7": f, "visual_1.0": v}

    out = {
        "experiment": "d2_embedding_dedup",
        "purpose": "Dedup D2 vs D1 in CLIP retrieval space; find the defensible clean number.",
        "max_cos_percentiles": {int(p): round(float(np.percentile(max_cos, p)), 4)
                                for p in (50, 75, 90, 95, 99)},
        "n_d2": len(ql),
        "n_with_cos_ge_0.95": int((max_cos >= 0.95).sum()),
        "n_with_cos_ge_0.98": int((max_cos >= 0.98).sum()),
        "retrieval_by_dedup_level_7class": rows,
    }
    (RES / "d2_embedding_dedup.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["retrieval_by_dedup_level_7class"], indent=2))
    print("n cos>=0.95:", out["n_with_cos_ge_0.95"], " n cos>=0.98:", out["n_with_cos_ge_0.98"])


if __name__ == "__main__":
    main()
