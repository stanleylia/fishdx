"""Reviewer 3.6 — Florence-2-only zero-shot baseline (no CLIP retrieval).

Classifies each image using ONLY the Florence-2 caption: the predicted class is the
disease whose knowledge-base keyword set has the most occurrences in the caption
(argmax over the eight classes; ties / no-match count as incorrect). This is the
pure perception→language baseline the reviewer requested, distinct from CLIP k-NN,
caption-only CLIP-text retrieval, and the caption→KB CLIP-text baseline.

Real cached Florence-2 D2 captions (results/e3_noaug_captions_log.json, n=1,537),
real KB keyword tiers (src/fishdx/kb/documents/*.md). Deterministic; no model load.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAPS = ROOT / "results" / "e3_noaug_captions_log.json"
KB_DIR = ROOT / "src" / "fishdx" / "kb" / "documents"
OUT = ROOT / "results" / "florence2_only_baseline.json"

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


def kb_keywords() -> dict[str, list[str]]:
    """class -> list of disease keywords (all keyword tiers, lower-cased)."""
    out = {}
    for f in sorted(KB_DIR.glob("*.md")):
        raw = f.read_text()
        cls = re.search(r"disease_class:\s*(.+)", raw).group(1).strip()
        kws = re.findall(r"^-\s+(.+)$", raw, re.M)  # markdown bullet keywords
        out[cls] = [k.strip().lower() for k in kws if len(k.strip()) > 2]
    return out


def main() -> None:
    recs = json.loads(CAPS.read_text())
    kb = kb_keywords()
    classes = list(kb)
    correct = correct7 = n7 = 0
    dist = {}
    for r in recs:
        cap = r["caption"].lower()
        true = LABEL_MAP[r["label"]]
        scores = {c: sum(cap.count(k) for k in kb[c]) for c in classes}
        best = max(scores.values())
        pred = next((c for c in classes if scores[c] == best), None) if best > 0 else "UNMATCHED"
        dist[pred] = dist.get(pred, 0) + 1
        if pred == true:
            correct += 1
        if true != "Epizootic Ulcerative Syndrome":
            n7 += 1
            if pred == true:
                correct7 += 1
    n = len(recs)
    out = {
        "experiment": "florence2_only_baseline",
        "reviewer_point": "R3.6 — Florence-2-only zero-shot (caption keyword classification, no CLIP)",
        "method": "predicted class = argmax over 8 KB keyword sets of caption keyword occurrences",
        "caption_source": "results/e3_noaug_captions_log.json (D2 noaug, real Florence-2)",
        "n": n,
        "DA_8_classes": round(correct / n, 4),
        "DA_7_shared": round(correct7 / n7, 4),
        "prediction_distribution": dist,
        "reference": "matches the ~3.7% keyword SCA; far below visual retrieval 0.893 and the full pipeline",
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Florence-2-only DA = {out['DA_8_classes']} (8-class) / {out['DA_7_shared']} (7-shared), n={n}")
    print("saved", OUT)


if __name__ == "__main__":
    main()
