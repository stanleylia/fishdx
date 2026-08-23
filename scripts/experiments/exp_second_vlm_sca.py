"""Second-VLM SCA control (ADR-0015): recompute the keyword-level SCA on a matched
D1-Train stratified sample with a second, different-family generative VLM
(Qwen2-VL-2B-Instruct) alongside Florence-2, using the IDENTICAL scorer
(core.algorithms.semantic_filter.classify_scene(caption).score) that produced the
manuscript's Florence-2 SCA = 0.037. Off-path control only; the pipeline is
unchanged.

Usage: python3 scripts/experiments/exp_second_vlm_sca.py [N_PER_CLASS]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.algorithms.semantic_filter import classify_scene  # noqa: E402
from core.config import SemanticFilterConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data/datasets/D1/train"
OUT = ROOT / "results/second_vlm_sca.json"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SEED = 42
SF_CFG = SemanticFilterConfig()
IMG_EXT = {".jpg", ".jpeg", ".png"}


def cls_of(name: str) -> str:
    return re.sub(r"\s*\(\d+\)\s*$", "", Path(name).stem).strip()


def sample_images(n_per_class: int) -> list[Path]:
    by_cls: dict[str, list[Path]] = {}
    for f in sorted(TRAIN.iterdir()):
        if f.suffix.lower() in IMG_EXT:
            by_cls.setdefault(cls_of(f.name), []).append(f)
    rng = random.Random(SEED)
    out = []
    for c in sorted(by_cls):
        files = sorted(by_cls[c])
        rng.shuffle(files)
        out.extend(files[:n_per_class])
    return out


def sca(caption: str) -> float:
    return float(classify_scene(caption, SF_CFG).score)


def load_florence():
    from transformers import AutoModelForCausalLM, AutoProcessor
    proc = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base", torch_dtype=torch.float16,
        trust_remote_code=True).to(DEVICE).eval()
    return proc, mdl


def cap_florence(img, proc, mdl) -> str:
    prompt = "<MORE_DETAILED_CAPTION>"
    inp = proc(text=prompt, images=img, return_tensors="pt").to(DEVICE)
    inp = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inp.items()}
    with torch.no_grad():
        gid = mdl.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                           max_new_tokens=256, num_beams=3, do_sample=False)
    raw = proc.batch_decode(gid, skip_special_tokens=False)[0]
    return proc.post_process_generation(raw, task=prompt,
                                        image_size=(img.width, img.height))[prompt].strip()


def load_qwen():
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    mdl = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.float16).to(DEVICE).eval()
    return proc, mdl


def cap_qwen(img, proc, mdl) -> str:
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "Describe this image in detail."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(text=[text], images=[img], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        gid = mdl.generate(**inp, max_new_tokens=256, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inp.input_ids, gid)]
    return proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    imgs = sample_images(n_per)
    print(f"sample: {len(imgs)} images ({n_per}/class), scorer=classify_scene(.score)")

    rows = []
    t0 = time.time()
    fp, fm = load_florence()
    for k, path in enumerate(imgs):
        im = Image.open(path).convert("RGB")
        c = cap_florence(im, fp, fm)
        rows.append({"file": path.name, "cls": cls_of(path.name),
                     "florence_caption": c, "florence_sca": sca(c)})
        if k % 50 == 0:
            print(f"  florence {k}/{len(imgs)}")
    del fm
    torch.cuda.empty_cache()
    print(f"florence done ({time.time()-t0:.0f}s); loading Qwen2-VL-2B…")

    qp, qm = load_qwen()
    for k, r in enumerate(rows):
        im = Image.open(TRAIN / r["file"]).convert("RGB")
        c = cap_qwen(im, qp, qm)
        r["qwen_caption"] = c
        r["qwen_sca"] = sca(c)
        if k % 50 == 0:
            print(f"  qwen {k}/{len(rows)}")

    fl = [r["florence_sca"] for r in rows]
    qw = [r["qwen_sca"] for r in rows]
    result = {
        "experiment": "second_vlm_sca_control",
        "adr": "0015",
        "scorer": "core.algorithms.semantic_filter.classify_scene(caption).score",
        "n_images": len(rows), "n_per_class": n_per, "seed": SEED,
        "florence2_base": {"model": "microsoft/Florence-2-base",
                           "avg_keyword_sca": round(sum(fl) / len(fl), 4)},
        "qwen2_vl_2b": {"model": "Qwen/Qwen2-VL-2B-Instruct",
                        "avg_keyword_sca": round(sum(qw) / len(qw), 4)},
        "manuscript_florence_sca": 0.037,
        "per_class": {},
    }
    for c in sorted(set(r["cls"] for r in rows)):
        cr = [r for r in rows if r["cls"] == c]
        result["per_class"][c] = {
            "n": len(cr),
            "florence_sca": round(sum(r["florence_sca"] for r in cr) / len(cr), 4),
            "qwen_sca": round(sum(r["qwen_sca"] for r in cr) / len(cr), 4)}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("\n=== RESULT ===")
    print("Florence-2 avg keyword SCA:", result["florence2_base"]["avg_keyword_sca"],
          "(manuscript 0.037)")
    print("Qwen2-VL-2B avg keyword SCA:", result["qwen2_vl_2b"]["avg_keyword_sca"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
