"""Reviewer 5.1 — caption -> KB-document text-only retrieval baseline.

Disentangles "caption vocabulary deficiency" from "CLIP text-embedding quality" by
retrieving each Florence-2 caption against the eight knowledge-base disease documents,
with BOTH query and target encoded by the CLIP text encoder (OpenCLIP ViT-B-32,
laion2b_s34b_b79k). Predicted class = argmax cosine over the KB documents.

Two target variants are reported:
  (A) full KB document text (CLIP truncates to its 77-token context);
  (B) the plain-language "Lay-visible appearance" paragraph only (closest in register to
      a Florence-2 caption, so the fairest test of whether captions carry disease signal).

Real cached Florence-2 captions (results/e3_noaug_captions_log.json, n = 1,537, D2 noaug),
real KB docs (src/fishdx/kb/documents/*.md), seed 42. No fabrication: prints measured DA.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import open_clip
import torch

ROOT = Path(__file__).resolve().parents[2]
CAPS = ROOT / "results" / "e3_noaug_captions_log.json"
KB_DIR = ROOT / "src" / "fishdx" / "kb" / "documents"
OUT = ROOT / "results" / "caption_to_kb_baseline.json"
SEED = 42
ARCH, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"

# caption-label string -> canonical KB disease_class
LABEL_MAP = {
    "Bacterial Red disease": "Bacterial Red Disease",
    "Bacterial diseases - Aeromoniasis": "Aeromoniasis",
    "Bacterial gill disease": "Bacterial Gill Disease",
    "EUS": "Epizootic Ulcerative Syndrome",
    "Fungal diseases Saprolegniasis": "Fungal Saprolegniasis",
    "Healthy Fish": "Healthy Fish",
    "Parasitic diseases": "Parasitic Diseases",
    "Viral diseases White tail disease": "Viral White Tail Disease",
}


def kb_docs() -> list[dict]:
    docs = []
    for f in sorted(KB_DIR.glob("*.md")):
        raw = f.read_text()
        m = re.search(r"disease_class:\s*(.+)", raw)
        cls = m.group(1).strip()
        body = raw.split("---", 2)[-1]
        lay = ""
        lm = re.search(r"##\s*Lay-visible appearance\s*(.+?)(?:\n##|\Z)", body, re.S)
        if lm:
            lay = lm.group(1).strip()
        docs.append({"class": cls, "full": body.strip(), "lay": lay or body.strip()})
    return docs


@torch.no_grad()
def encode(texts: list[str], model, tok, device) -> np.ndarray:
    out = []
    for i in range(0, len(texts), 256):
        t = tok(texts[i : i + 256]).to(device)  # open_clip truncates to 77 tokens
        e = model.encode_text(t).float()
        e = e / e.norm(dim=-1, keepdim=True)
        out.append(e.cpu().numpy())
    return np.concatenate(out, axis=0)


def da(pred: list[str], true: list[str], keep: set[str] | None) -> tuple[float, int]:
    idx = [i for i in range(len(true)) if keep is None or true[i] in keep]
    correct = sum(pred[i] == true[i] for i in idx)
    return correct / len(idx), len(idx)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED, device=device)
    model.eval()
    tok = open_clip.get_tokenizer(ARCH)

    recs = json.loads(CAPS.read_text())
    captions = [r["caption"] for r in recs]
    true = [LABEL_MAP[r["label"]] for r in recs]
    docs = kb_docs()
    doc_classes = [d["class"] for d in docs]

    q = encode(captions, model, tok, device)
    shared = {c for c in set(true) if c != "Epizootic Ulcerative Syndrome"}
    import collections

    maj = collections.Counter(true).most_common(1)[0]
    majority_da = round(maj[1] / len(true), 4)

    results = {
        "experiment": "caption_to_kb_baseline",
        "reviewer_point": "R5.1 — caption->KB-document text-only retrieval (both CLIP-text encoded)",
        "encoder": f"{ARCH} / {PRETRAINED}",
        "n_captions": len(captions),
        "caption_source": "results/e3_noaug_captions_log.json (D2 noaug, real Florence-2)",
        "note": "CLIP text encoder truncates targets to 77 tokens; query=caption, target=KB doc.",
        "seed": SEED,
        "random_baseline_DA_8class": round(1 / 8, 4),
        "majority_class_DA": majority_da,
        "majority_class": maj[0],
        "variants": {},
        "reference_numbers": {
            "caption_to_caption_gallery_DA": 0.495,
            "visual_embedding_DA": 0.893,
            "note": "reference values from the manuscript modality comparison (7 shared classes)",
        },
    }
    for name, key in (("full_document", "full"), ("lay_appearance_paragraph", "lay")):
        t = encode([d[key] for d in docs], model, tok, device)
        sims = q @ t.T
        pred = [doc_classes[j] for j in sims.argmax(axis=1)]
        da_all, n_all = da(pred, true, None)
        da_shared, n_shared = da(pred, true, shared)
        results["variants"][name] = {
            "target": f"KB doc {name}",
            "DA_all_8_classes": round(da_all, 4),
            "n_all": n_all,
            "DA_7_shared_classes": round(da_shared, 4),
            "n_shared": n_shared,
            "prediction_distribution": dict(collections.Counter(pred)),
        }
        print(f"[{name}] DA(8-class)={da_all:.4f} (n={n_all})  DA(7-shared)={da_shared:.4f} (n={n_shared})")

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
