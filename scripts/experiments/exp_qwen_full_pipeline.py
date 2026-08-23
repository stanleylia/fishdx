"""Cross-VLM at the FULL fused-pipeline level (addresses the 'complete pipeline
cross-VLM comparison' request).

Extends exp_qwen_pipeline_retrieval.py from caption-only (lambda=0) to the full
lambda=0.7 fused pipeline. On the SAME matched stratified subset (100/class D1
gallery + 100/class clean-D2 query, seed 42, 7 shared classes), the only thing
that changes between arms is the VLM captioner (Florence-2 vs Qwen2-VL-2B); the
CLIP visual embeddings are shared. Qwen captions are read from cache (no re-run);
we only CLIP-text-encode them and fuse.

Reports k=1 retrieval DA for caption-only (lambda=0), fused (lambda=0.7) and
visual-only (lambda=1.0). Writes results/qwen_full_pipeline.json.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import open_clip
import torch

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
DEDUP = 0.95
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CANON = {'aeromoniasis': 'aeromoniasis', 'bacterial diseases - aeromoniasis': 'aeromoniasis',
         'bacterial gill disease': 'bacterial_gill', 'bacterial red disease': 'bacterial_red',
         'fungal diseases saprolegniasis': 'fungal', 'parasitic diseases': 'parasitic',
         'viral diseases white tail disease': 'viral_white_tail',
         'viral white tail disease': 'viral_white_tail', 'healthy fish': 'healthy', 'eus': 'EUS'}


def canon(s: str) -> str:
    s = s.lower().replace('_', ' ').strip()
    for k, v in CANON.items():
        if k in s:
            return v
    return 'UNKNOWN'


def l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def strat(paths, per):
    by: dict = {}
    for p in paths:
        by.setdefault(canon(p.stem), []).append(p)
    out = []
    for c in sorted(by):
        if c in ('EUS', 'UNKNOWN'):
            continue
        random.shuffle(by[c])
        out += by[c][:per]
    return out


def da(gal_emb, gal_lab, qry_emb, qry_lab):
    return round(float((gal_lab[(l2(qry_emb) @ l2(gal_emb).T).argmax(1)] == qry_lab).mean()), 4)


def main() -> None:
    zg = np.load(RES / "_cache_d1_gallery.npz", allow_pickle=True)
    zq = np.load(RES / "_cache_d2_query.npz", allow_pickle=True)
    gv_all, gc_all = zg["visual"], zg["caption"]
    qv_all, qc_all = zq["visual"], zq["caption"]

    random.seed(42)  # module-level once, sequential strat calls — matches exp_qwen_pipeline_retrieval
    d1paths = sorted((ROOT / "data/datasets/D1/train").glob("*.jpg"))
    d2paths = sorted(p for p in (ROOT / "data/datasets/D2").rglob("*") if p.suffix.lower() in EXTS)
    clean = (qv_all @ gv_all.T).max(1) < DEDUP
    d2clean = [p for p, c in zip(d2paths, clean) if c]

    gal = strat(d1paths, 100)
    qry = strat(d2clean, 100)
    name2gv = {p.name: gv_all[i] for i, p in enumerate(d1paths)}
    name2gc = {p.name: gc_all[i] for i, p in enumerate(d1paths)}
    name2qv = {p.name: qv_all[i] for i, p in enumerate(d2paths)}
    name2qc = {p.name: qc_all[i] for i, p in enumerate(d2paths)}

    gl = np.array([canon(p.stem) for p in gal])
    ql = np.array([canon(p.stem) for p in qry])
    g_vis = l2(np.array([name2gv[p.name] for p in gal]))
    q_vis = l2(np.array([name2qv[p.name] for p in qry]))
    g_flo = l2(np.array([name2gc[p.name] for p in gal]))
    q_flo = l2(np.array([name2qc[p.name] for p in qry]))

    qcap = json.loads((RES / "qwen_subset_captions.json").read_text())
    for prior in ["qwen_d1_captions.json"]:
        if (RES / prior).exists():
            qcap.update({k: v for k, v in json.loads((RES / prior).read_text()).items() if k not in qcap})

    clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=DEV)
    clip.eval()
    tok = open_clip.get_tokenizer("ViT-B-32")

    def et(t):
        with torch.no_grad():
            f = clip.encode_text(tok([t or "fish"]).to(DEV))
            return (f / f.norm(dim=-1, keepdim=True))[0].cpu().numpy().astype(np.float32)

    g_qwe = l2(np.array([et(qcap.get(p.name, "")) for p in gal]))
    q_qwe = l2(np.array([et(qcap.get(p.name, "")) for p in qry]))

    LAM = 0.7
    out = {
        "design": "matched stratified subset (100/class D1 gallery + 100/class clean D2 query, "
                  "seed 42, 7 shared classes); only the VLM captioner differs; CLIP visual shared",
        "n_gallery": len(gal), "n_query": len(qry),
        "visual_only_DA_lambda1.0": da(g_vis, gl, q_vis, ql),
        "florence": {
            "caption_only_DA_lambda0": da(g_flo, gl, q_flo, ql),
            "fused_DA_lambda0.7": da(l2(LAM * g_vis + (1 - LAM) * g_flo), gl,
                                     l2(LAM * q_vis + (1 - LAM) * q_flo), ql),
        },
        "qwen2vl": {
            "caption_only_DA_lambda0": da(g_qwe, gl, q_qwe, ql),
            "fused_DA_lambda0.7": da(l2(LAM * g_vis + (1 - LAM) * g_qwe), gl,
                                     l2(LAM * q_vis + (1 - LAM) * q_qwe), ql),
        },
    }
    (RES / "qwen_full_pipeline.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
