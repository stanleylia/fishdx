"""CLIP embedding-geometry + CRMR artifact (traceability for the §4.2/Discussion numbers).

Recomputes, from the released D1 gallery cache and the real Florence-2 D1-Train captions,
the numbers the manuscript cites in the mechanistic discussion:
  - Separability Ratio = mean inter-class prototype cosine distance / mean intra-class
    (to-own-prototype) cosine distance
  - healthy-vs-disease and inter-disease prototype cosine distances (per pair)
  - CRMR (Correct Reference Match Rate): fraction of captions whose top sentence-embedding
    (all-MiniLM-L6-v2) match to a KB disease document coincides with the ground-truth class

Deterministic; measured only. Writes results/clip_geometry.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results/_cache_d1_gallery.npz"
CAPS = ROOT / "results/d1_train_captions_log.json"
KB = ROOT / "src/fishdx/kb/documents"
OUT = ROOT / "results/clip_geometry.json"


def canon(s: str) -> str:
    s = s.lower()
    for k, v in [("healthy", "healthy"), ("aeromon", "aeromoniasis"), ("gill", "bacterial_gill"),
                 ("red", "bacterial_red"), ("saprol", "fungal"), ("fungal", "fungal"),
                 ("parasit", "parasitic"), ("white", "viral_white_tail"), ("viral", "viral_white_tail")]:
        if k in s:
            return v
    return "?"


def geometry() -> dict:
    z = np.load(CACHE, allow_pickle=True)
    gv, gl = z["visual"], z["labels"]
    gv = gv / (np.linalg.norm(gv, axis=1, keepdims=True) + 1e-12)
    classes = sorted(set(gl.tolist()))
    proto = {c: gv[gl == c].mean(0) for c in classes}
    for c in proto:
        proto[c] = proto[c] / (np.linalg.norm(proto[c]) + 1e-12)
    intra = float(np.mean([(1 - (gv[gl == c] @ proto[c])).mean() for c in classes]))
    pairs = {}
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            pairs[f"{a} <-> {b}"] = round(1 - float(proto[a] @ proto[b]), 4)
    inter = float(np.mean(list(pairs.values())))
    return {
        "n_gallery": int(len(gl)), "n_classes": len(classes),
        "intra_class_mean_cos_dist": round(intra, 4),
        "inter_class_mean_cos_dist": round(inter, 4),
        "separability_ratio_inter_over_intra": round(inter / intra, 4),
        "pairwise_prototype_cos_dist": dict(sorted(pairs.items(), key=lambda x: x[1])),
    }


def crmr() -> dict:
    from sentence_transformers import SentenceTransformer, util
    caps = json.loads(CAPS.read_text())
    docs = {}
    for f in sorted(KB.glob("*.md")):
        t = f.read_text()
        docs[canon(f.stem + " " + t[:200])] = t
    doc_cls = list(docs.keys())
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    demb = m.encode([docs[c] for c in doc_cls], convert_to_tensor=True, normalize_embeddings=True)
    by: dict[str, list[int]] = {}
    for r in caps:
        gt = canon(r.get("label", ""))
        if gt == "?":
            continue
        ce = m.encode(r.get("caption", "") or "", convert_to_tensor=True, normalize_embeddings=True)
        pred = doc_cls[int(util.cos_sim(ce, demb)[0].argmax())]
        by.setdefault(gt, []).append(int(pred == gt))
    per = {c: round(sum(v) / len(v), 4) for c, v in by.items()}
    overall = sum(sum(v) for v in by.values()) / sum(len(v) for v in by.values())
    return {"crmr_overall": round(overall, 4), "crmr_per_class": per,
            "scorer": "all-MiniLM-L6-v2 top-1 caption->KB-doc match vs ground-truth class"}


def main() -> None:
    res = {"experiment": "clip_geometry_and_crmr",
           "purpose": "traceability artifact for §4.2/Discussion geometry + CRMR numbers",
           "geometry": geometry(), "crmr": crmr()}
    OUT.write_text(json.dumps(res, indent=2))
    g = res["geometry"]
    print(f"Separability Ratio = {g['separability_ratio_inter_over_intra']} "
          f"(inter {g['inter_class_mean_cos_dist']} / intra {g['intra_class_mean_cos_dist']})")
    print("smallest inter-disease pairs:",
          {k: v for k, v in list(g["pairwise_prototype_cos_dist"].items())[:4]})
    print(f"CRMR = {res['crmr']['crmr_overall']}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
