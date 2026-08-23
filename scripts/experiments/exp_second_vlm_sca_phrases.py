"""Note S13 re-run under the Direction-2 Confirmed-PHRASES keyword-SCA (consistency with the
corrected 0.2% headline). Matched Florence-2 vs Qwen2-VL-2B on the SAME 100/class D1-Train
sample, scored by SCA_i = |Ki ∩ tokens-of-present-KB-Confirmed-phrases| / |Ki|. Saves raw
captions. Real models, deterministic; measured only.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data/datasets/D1/train"
KB = ROOT / "src/fishdx/kb/documents"
OUT = ROOT / "results/second_vlm_sca_phrases.json"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 42
IMG_EXT = {".jpg", ".jpeg", ".png"}
STOP = set("a an the of and or to in on with at is are be this that fish its it as for from by".split())


def toks(t):
    return [w for w in re.findall(r"[a-z]+", t.lower()) if len(w) > 2 and w not in STOP]


def confirmed_phrases():
    ph = []
    for f in sorted(KB.glob("*.md")):
        b = f.read_text().split("---", 2)[-1]
        m = re.search(r"##\s*Confirmed keywords\s*(.+?)(?:\n##|\Z)", b, re.S)
        if m:
            ph += [x.strip().lower() for x in re.findall(r"^-\s+(.+)$", m.group(1), re.M)]
    return sorted(set(ph))


PHRASES = confirmed_phrases()


def sca(caption: str) -> float:
    c = (caption or "").lower()
    k = toks(c)
    if not k:
        return 0.0
    mt = set()
    for p in PHRASES:
        if p in c:
            mt |= set(toks(p))
    return sum(1 for w in k if w in mt) / len(k)


def cls_of(name):
    return re.sub(r"\s*\(\d+\)\s*$", "", Path(name).stem).strip()


def sample_images(n):
    by = {}
    for f in sorted(TRAIN.iterdir()):
        if f.suffix.lower() in IMG_EXT:
            by.setdefault(cls_of(f.name), []).append(f)
    rng = random.Random(SEED)
    out = []
    for c in sorted(by):
        fs = sorted(by[c])
        rng.shuffle(fs)
        out += fs[:n]
    return out


def load_florence():
    from transformers import AutoModelForCausalLM, AutoProcessor
    p = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch.float16,
                                             trust_remote_code=True).to(DEVICE).eval()
    return p, m


def cap_florence(img, proc, mdl):
    pr = "<MORE_DETAILED_CAPTION>"
    inp = proc(text=pr, images=img, return_tensors="pt").to(DEVICE)
    inp = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inp.items()}
    with torch.no_grad():
        gid = mdl.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                           max_new_tokens=256, num_beams=3, do_sample=False)
    raw = proc.batch_decode(gid, skip_special_tokens=False)[0]
    return proc.post_process_generation(raw, task=pr, image_size=(img.width, img.height))[pr].strip()


def load_qwen():
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    p = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    m = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct",
                                                        torch_dtype=torch.float16).to(DEVICE).eval()
    return p, m


def cap_qwen(img, proc, mdl):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image in detail."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        gid = mdl.generate(**inp, max_new_tokens=256, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inp.input_ids, gid)]
    return proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    imgs = sample_images(n_per)
    print(f"sample {len(imgs)} ({n_per}/class); {len(PHRASES)} Confirmed phrases", flush=True)
    rows = []
    fp, fm = load_florence()
    for k, p in enumerate(imgs):
        im = Image.open(p).convert("RGB")
        c = cap_florence(im, fp, fm)
        rows.append({"file": p.name, "cls": cls_of(p.name), "florence_caption": c, "florence_sca": sca(c)})
        if (k + 1) % 100 == 0:
            print(f"  florence {k+1}/{len(imgs)}", flush=True)
    del fm
    torch.cuda.empty_cache()
    qp, qm = load_qwen()
    for k, r in enumerate(rows):
        im = Image.open(TRAIN / r["file"]).convert("RGB")
        c = cap_qwen(im, qp, qm)
        r["qwen_caption"] = c
        r["qwen_sca"] = sca(c)
        if (k + 1) % 100 == 0:
            print(f"  qwen {k+1}/{len(rows)}", flush=True)
    fl = [r["florence_sca"] for r in rows]
    qw = [r["qwen_sca"] for r in rows]
    res = {
        "experiment": "second_vlm_sca_phrases",
        "scorer": "Confirmed-PHRASES keyword-SCA (Direction 2): |Ki ∩ tokens-of-present-Confirmed-phrases|/|Ki|",
        "n_images": len(imgs), "n_per_class": n_per, "seed": SEED,
        "florence2_base": {"avg_keyword_sca_pct": round(100 * sum(fl) / len(fl), 3)},
        "qwen2_vl_2b": {"avg_keyword_sca_pct": round(100 * sum(qw) / len(qw), 3)},
        "old_classify_scene": {"florence": 0.0413, "qwen": 0.0412},
        "captions": rows,
    }
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nFlorence keyword-SCA = {res['florence2_base']['avg_keyword_sca_pct']}%  |  "
          f"Qwen = {res['qwen2_vl_2b']['avg_keyword_sca_pct']}%  (Confirmed-PHRASES)")
    print("saved", OUT)


if __name__ == "__main__":
    main()
