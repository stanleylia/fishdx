"""A5-v2 — exhaustive test of whether real Florence-2 grounding can supply a
DISCRIMINATIVE per-candidate scalar for the Bidirectional Verification Loop.

Goes beyond A5 (presence only): for each real D2 image we probe the class's TRUE
disease phrase vs a DISTRACTOR (wrong-class) phrase under BOTH grounding tasks, and
extract three candidate scalars — box presence, box COUNT, and max box AREA fraction —
then test whether any of them separates true from distractor (paired, per image).
If none separates, a genuine Florence-2 grounding score cannot replace the proxy; this
is a proof, not a lost-code excuse. Real Florence-2-base (fp16, beam=3, deterministic).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
D2 = ROOT / "data/datasets/D2"
OUT = ROOT / "results" / "florence_grounding_signal_v2.json"
N_PER_CLASS = 15
TASKS = ["<OPEN_VOCABULARY_DETECTION>", "<CAPTION_TO_PHRASE_GROUNDING>"]

PHRASE = {
    "Bacterial diseases - Aeromoniasis": "red skin ulcer",
    "Bacterial gill disease": "swollen gill",
    "Bacterial Red disease": "red haemorrhage on body",
    "Fungal diseases Saprolegniasis": "white cotton fungus",
    "Parasitic diseases": "parasite on skin",
    "Viral diseases White tail disease": "white tail",
    "Healthy Fish": "healthy intact fish",
}
CLASSES = list(PHRASE)


def probe(proc, mdl, dev, im, task, phrase):
    """Return (n_boxes, max_area_frac) for phrase grounding on image."""
    inp = proc(text=task + phrase, images=im, return_tensors="pt").to(dev, torch.float16)
    with torch.no_grad():
        gid = mdl.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                           max_new_tokens=128, num_beams=3, do_sample=False)
    txt = proc.batch_decode(gid, skip_special_tokens=False)[0]
    out = proc.post_process_generation(txt, task=task, image_size=(im.width, im.height))[task]
    boxes = out.get("bboxes", []) if isinstance(out, dict) else []
    area = im.width * im.height
    max_frac = 0.0
    for b in boxes:
        w = max(0.0, b[2] - b[0]); h = max(0.0, b[3] - b[1])
        max_frac = max(max_frac, (w * h) / area)
    return len(boxes), round(max_frac, 4)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True,
                                               torch_dtype=torch.float16).to(dev).eval()
    results = {}
    for task in TASKS:
        per_class = {}
        # paired accumulators for a discrimination test on box-count and area
        n_true_gt_dist = n_dist_gt_true = n_tie = 0
        t_pres = d_pres = total = 0
        for c in CLASSES:
            imgs = sorted(p for p in D2.glob("*.jpg")
                          if p.name.startswith(c) and "aug" not in p.name.lower())[:N_PER_CLASS]
            distractor = PHRASE[CLASSES[(CLASSES.index(c) + 1) % len(CLASSES)]]
            rows = []
            for p in imgs:
                im = Image.open(p).convert("RGB")
                tn, ta = probe(proc, mdl, dev, im, task, PHRASE[c])
                dn, da = probe(proc, mdl, dev, im, task, distractor)
                rows.append((tn, ta, dn, da))
                t_pres += int(tn > 0); d_pres += int(dn > 0); total += 1
                # discrimination by (count, then area) — does TRUE score strictly higher?
                key_t, key_d = (tn, ta), (dn, da)
                if key_t > key_d: n_true_gt_dist += 1
                elif key_d > key_t: n_dist_gt_true += 1
                else: n_tie += 1
            per_class[c] = {
                "n": len(imgs), "true_phrase": PHRASE[c], "distractor_phrase": distractor,
                "true_grounded": sum(r[0] > 0 for r in rows),
                "distractor_grounded": sum(r[2] > 0 for r in rows),
                "mean_true_boxes": round(sum(r[0] for r in rows) / max(1, len(rows)), 2),
                "mean_distractor_boxes": round(sum(r[2] for r in rows) / max(1, len(rows)), 2),
            }
            print(f"[{task[1:20]}] {c[:30]:32s} true {per_class[c]['true_grounded']}/{len(imgs)} "
                  f"({per_class[c]['mean_true_boxes']} box) | dist {per_class[c]['distractor_grounded']}/{len(imgs)} "
                  f"({per_class[c]['mean_distractor_boxes']} box)")
        results[task] = {
            "n_total": total,
            "true_presence_rate": round(t_pres / total, 4),
            "distractor_presence_rate": round(d_pres / total, 4),
            "discrimination_true>dist": n_true_gt_dist,
            "discrimination_dist>true": n_dist_gt_true,
            "discrimination_tie": n_tie,
            "signal_discriminates": (n_true_gt_dist - n_dist_gt_true) / total if total else 0.0,
            "per_class": per_class,
        }
        print(f"  -> presence true {t_pres/total:.3f} vs dist {d_pres/total:.3f}; "
              f"(count,area) true>dist {n_true_gt_dist}, dist>true {n_dist_gt_true}, tie {n_tie}")
    res = {
        "experiment": "florence_grounding_signal_v2 (exhaustive discriminative test)",
        "model": "microsoft/Florence-2-base (fp16, beam=3, deterministic)",
        "note": "Florence-2 grounding returns boxes+labels only (no scalar confidence). This probes "
                "whether box presence/count/area discriminates a class's TRUE disease phrase from a "
                "wrong-class DISTRACTOR, under both grounding tasks, on real D2 images.",
        "tasks": results,
    }
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("saved", OUT)


if __name__ == "__main__":
    main()
