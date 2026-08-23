"""A6 Direction 2 — official keyword-SCA under the documented Confirmed-PHRASES vocabulary.

Generates real Florence-2 MORE_DETAILED_CAPTION on the 1,747 D1-Train images (the headline
corpus; captions cached to results/d1_train_captions_log.json for reproducible re-runs), then
computes the keyword-SCA under the *described* token-level formula restricted to KB Confirmed
disease phrases: a caption token counts only if it participates in a KB Confirmed disease
phrase that actually appears in the caption.

  SCA_i = |Ki ∩ (tokens of KB-Confirmed phrases present in caption i)| / |Ki|

Reports D1-Train (headline) + per-class + between-class range + zero-vocab %, the cached D2
no-aug cross-check, and saves phrase-hit / zero-hit caption examples for a Table-S10/S11-style
manual over-strictness spot-check. Deterministic (beam=3), fp16. Measured only, no fabrication.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
D1_TRAIN = ROOT / "lab_dateset/external_datasets/fish_disease_south_asia/Freshwater Fish Disease Aquaculture in south asia/Train"
KB = ROOT / "src/fishdx/kb/documents"
D2_CAPS = ROOT / "results/e3_noaug_captions_log.json"
CAP_CACHE = ROOT / "results/d1_train_captions_log.json"
OUT = ROOT / "results/sca_direction2.json"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
STOP = set("a an the of and or to in on with at is are be this that fish its it as for from by".split())


def toks(t: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", t.lower()) if len(w) > 2 and w not in STOP]


def section(body: str, name: str) -> list[str]:
    m = re.search(rf"##\s*{re.escape(name)}\s*(.+?)(?:\n##|\Z)", body, re.S)
    return [x.strip() for x in re.findall(r"^-\s+(.+)$", m.group(1), re.M)] if m else []


def confirmed_phrases() -> list[str]:
    ph = []
    for f in sorted(KB.glob("*.md")):
        b = f.read_text().split("---", 2)[-1]
        ph += [p.lower() for p in section(b, "Confirmed keywords")]
    return sorted(set(ph))


def gen_d1_captions() -> list[dict]:
    if CAP_CACHE.exists():
        return json.loads(CAP_CACHE.read_text())
    proc = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch.float16,
                                               trust_remote_code=True).to(DEVICE).eval()
    prompt = "<MORE_DETAILED_CAPTION>"
    recs = []
    files = sorted(p for cd in sorted(D1_TRAIN.iterdir()) if cd.is_dir()
                   for p in sorted(cd.iterdir()) if p.suffix.lower() in IMG_EXT)
    for i, p in enumerate(files):
        im = Image.open(p).convert("RGB")
        with torch.no_grad():
            inp = proc(text=prompt, images=im, return_tensors="pt").to(DEVICE)
            inp = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inp.items()}
            gid = mdl.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                               max_new_tokens=256, num_beams=3, do_sample=False)
            raw = proc.batch_decode(gid, skip_special_tokens=False)[0]
            cap = proc.post_process_generation(raw, task=prompt, image_size=(im.width, im.height))[prompt].strip()
        recs.append({"image": p.name, "label": p.parent.name, "caption": cap})
        if (i + 1) % 200 == 0:
            print(f"  Florence {i+1}/{len(files)}", flush=True)
    CAP_CACHE.write_text(json.dumps(recs, indent=2, ensure_ascii=False))
    return recs


def score(recs: list[dict], phrases: list[str]) -> dict:
    per_class: dict[str, list[float]] = {}
    zero_by_class: dict[str, list[int]] = {}
    hits, zeros = [], []
    for r in recs:
        c = (r.get("caption") or "").lower()
        k = toks(c)
        if not k:
            continue
        matched = [p for p in phrases if p in c]
        mt = set()
        for p in matched:
            mt |= set(toks(p))
        hit = sum(1 for w in k if w in mt)
        sca = hit / len(k)
        lab = r["label"]
        per_class.setdefault(lab, []).append(sca)
        zero_by_class.setdefault(lab, []).append(int(hit == 0))
        rec = {"image": r["image"], "label": lab, "caption": r["caption"],
               "matched_phrases": matched, "hit_tokens": hit, "sca_i": round(sca, 4)}
        (zeros if hit == 0 else hits).append(rec)
    allsca = [s for v in per_class.values() for s in v]
    n = len(allsca)
    class_mean = {c: round(sum(v) / len(v), 4) for c, v in per_class.items()}
    return {
        "n": n,
        "mean_keyword_sca_pct": round(100 * sum(allsca) / n, 3),
        "zero_vocab_pct": round(100 * sum(sum(z) for z in zero_by_class.values()) / n, 2),
        "per_class_mean_pct": {c: round(100 * m, 3) for c, m in class_mean.items()},
        "between_class_range_pct": [round(100 * min(class_mean.values()), 3), round(100 * max(class_mean.values()), 3)],
        "_hits": hits, "_zeros": zeros,
    }


def main() -> None:
    phrases = confirmed_phrases()
    print(f"KB Confirmed phrases: {len(phrases)}")
    d1 = gen_d1_captions()
    print(f"D1-Train captions: {len(d1)}")
    d2 = json.loads(D2_CAPS.read_text())

    s1 = score(d1, phrases)
    s2 = score(d2, phrases)

    # spot-check samples (Table S10/S11 style): a few hits + a few zeros per corpus
    import random
    rng = random.Random(42)

    def sample(recs, k=8):
        return rng.sample(recs, min(k, len(recs)))

    res = {
        "experiment": "sca_direction2_confirmed_phrases",
        "vocabulary": "KB Confirmed disease phrases (Note S6 sources), substring-matched; "
                      "SCA_i = |Ki ∩ tokens-of-present-Confirmed-phrases| / |Ki|",
        "n_confirmed_phrases": len(phrases),
        "D1_Train_headline": {k: v for k, v in s1.items() if not k.startswith("_")},
        "D2_noaug_crosscheck": {k: v for k, v in s2.items() if not k.startswith("_")},
        "old_published": {"D1_Train_3.7pct": "classify_scene overlap (different method)",
                          "D2_0.17pct_92.6zero": "unpreserved strict metric (different method)"},
        "spotcheck_D1_phrase_hits": [{k: r[k] for k in ("label", "caption", "matched_phrases", "hit_tokens", "sca_i")}
                                     for r in sample(s1["_hits"])],
        "spotcheck_D1_zero_hits": [{k: r[k] for k in ("label", "caption", "hit_tokens")}
                                   for r in sample(s1["_zeros"])],
    }
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("\n=== Confirmed-PHRASES keyword-SCA ===")
    print(f"D1-Train (headline): mean {s1['mean_keyword_sca_pct']}%  zero {s1['zero_vocab_pct']}%  "
          f"range {s1['between_class_range_pct']}%  (n={s1['n']})")
    print(f"D2 no-aug (x-check): mean {s2['mean_keyword_sca_pct']}%  zero {s2['zero_vocab_pct']}%  (n={s2['n']})")
    print("per-class D1:", s1["per_class_mean_pct"])
    print("saved", OUT)


if __name__ == "__main__":
    main()
