"""Novel-disease OOD interception on two real, never-before-seen fish diseases.

Two pathogen classes were added to the corpus under ``Journal/fish_data/`` with full
provenance (README per class): **Fish Louse / Argulosis** (*Argulus* spp., a crustacean
ectoparasite) and **Velvet / Oodiniosis** (*Piscinoodinium/Amyloodinium*, a dinoflagellate
ectoparasite) across three hosts (cichlid, corydoras, goldfish). Neither disease is present
in the D1 gallery's seven classes, so both are genuine out-of-distribution (OOD) inputs.

This experiment mirrors ``exp_eus56_margin_and_latency.py`` exactly — the same real models
(Florence-2-base MORE_DETAILED_CAPTION beam=3 deterministic fp16; OpenCLIP ViT-B-32
LAION-2B), the same fused gallery (results/_cache_d1_gallery.npz, 1,639 D1 refs, 7 classes),
the same fusion (lambda=0.7, Eqs. 5-7) and margin definition (top1-top2 sim, Eq. 11) — and
asks whether the margin safety valve (theta_margin) that intercepts EUS also intercepts
these two novel diseases as Inconclusive rather than confidently mis-mapping them.

A harder test than EUS: both new diseases are *parasitic* and D1 contains a ``parasitic``
class, so a semantically adjacent in-gallery class exists. We therefore also report, for the
DECIDED (non-intercepted) images, which gallery class they retrieved to.

Honesty about frame correlation: most images are fps-sampled video frames from a handful of
source clips (near-duplicates). We report the interception rate at both the frame level (all
images) AND the source level (each video clip / still photo counts once, majority-intercept),
so the correlated frames cannot inflate the headline.

All embeddings are cached to results/_cache_newdx_gallery.npz for the onboarding follow-up
(exp_newdx_onboarding.py) so Florence-2 need not be re-run. No simulation, no hard-coded outputs.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LAMBDA = 0.7
THETAS = [0.01, 0.02, 0.05]          # same interception sweep as the EUS experiment
THETA_PRIMARY = 0.02
CACHE = ROOT / "results/_cache_d1_gallery.npz"
DATA_DIR = ROOT / "Journal/fish_data"
EMB_CACHE = ROOT / "results/_cache_newdx_gallery.npz"
OUT = ROOT / "results/newdx_ood_interception.json"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# class-directory -> (disease class label, host subgroup)
DIR_MAP = {
    "Fish_Louse_Argulus": ("argulus", "angelfish"),
    "Velvet_Oodinium_Cichlid": ("oodinium", "cichlid"),
    "Velvet_Oodinium_Corydoras": ("oodinium", "corydoras"),
    "Velvet_Oodinium_Goldfish": ("oodinium", "goldfish"),
}


def l2(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def wilson_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = correct / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def source_key(rel_dir: str, stem: str) -> str:
    """Group correlated frames: video frames share a source-clip key; photos are singletons."""
    m = re.match(r"(louse_[0-9a-f]+)_frame_\d+", stem)
    if m:
        return f"{rel_dir}:{m.group(1)}"
    m = re.match(r"([a-z]+)_video_frame_\d+", stem)
    if m:
        return f"{rel_dir}:{m.group(1)}_video"
    return f"{rel_dir}:{stem}"          # a still photo — its own source


def enumerate_images() -> list[dict]:
    items: list[dict] = []
    for d, (cls, host) in DIR_MAP.items():
        cdir = DATA_DIR / d
        for f in sorted(cdir.iterdir()):
            if f.is_file() and f.suffix.lower() in IMG_EXT:
                items.append({
                    "path": f, "cls": cls, "host": host,
                    "src": source_key(d, f.stem),
                    "is_frame": "frame" in f.stem,
                })
    return items


def main() -> None:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print("=" * 72)
    print("Novel-disease OOD interception (Argulus + Oodinium/Velvet)")
    print(f"  GPU: {gpu_name}")
    print("=" * 72)

    z = np.load(CACHE, allow_pickle=True)
    gv, gc, gl = z["visual"], z["caption"], z["labels"]
    g_fus = l2(LAMBDA * gv + (1 - LAMBDA) * gc)
    gallery_classes = sorted(set(gl.tolist()))
    assert "argulus" not in gallery_classes and "oodinium" not in gallery_classes
    print(f"  gallery: {len(gl)} refs, classes={gallery_classes}")

    items = enumerate_images()
    print(f"  new-disease images: {len(items)}  "
          f"(argulus={sum(i['cls']=='argulus' for i in items)}, "
          f"oodinium={sum(i['cls']=='oodinium' for i in items)})")

    print("\n[loading real models: Florence-2-base + CLIP ViT-B-32 LAION-2B ...]")
    import open_clip
    from transformers import AutoModelForCausalLM, AutoProcessor

    clip, _, prep = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=DEVICE)
    clip.eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    proc = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    flor = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base", torch_dtype=torch.float16, trust_remote_code=True).to(DEVICE)
    prompt = "<MORE_DETAILED_CAPTION>"

    vis_emb, cap_emb, rows = [], [], []
    for k, it in enumerate(items, 1):
        image = Image.open(it["path"]).convert("RGB")
        with torch.no_grad():
            inp = proc(text=prompt, images=image, return_tensors="pt").to(DEVICE)
            inp = {kk: (v.half() if v.dtype == torch.float32 else v) for kk, v in inp.items()}
            gid = flor.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                                max_new_tokens=256, num_beams=3, do_sample=False)
            raw = proc.batch_decode(gid, skip_special_tokens=False)[0]
            caption = proc.post_process_generation(
                raw, task=prompt, image_size=(image.width, image.height))[prompt].strip()
            v = clip.encode_image(prep(image).unsqueeze(0).to(DEVICE))
            v = v / v.norm(dim=-1, keepdim=True)
            t = clip.encode_text(tok([caption or "fish"]).to(DEVICE))
            t = t / t.norm(dim=-1, keepdim=True)

        qv = v[0].cpu().numpy().astype(np.float32)
        qc = t[0].cpu().numpy().astype(np.float32)
        q_fus = l2(LAMBDA * qv + (1 - LAMBDA) * qc)
        sims = g_fus @ q_fus
        order = np.argsort(-sims)
        margin = float(sims[order[0]] - sims[order[1]])
        top1 = str(gl[order[0]])

        vis_emb.append(qv)
        cap_emb.append(qc)
        rows.append({"cls": it["cls"], "host": it["host"], "src": it["src"],
                     "is_frame": it["is_frame"], "file": it["path"].name,
                     "margin": margin, "top1": top1})
        if k % 20 == 0 or k == len(items):
            print(f"  [{k:3d}/{len(items)}] {it['cls']:8s} margin={margin:.4f} "
                  f"{'INTERCEPT' if margin < THETA_PRIMARY else 'decide->' + top1}")

    np.savez(EMB_CACHE,
             visual=np.array(vis_emb, dtype=np.float32),
             caption=np.array(cap_emb, dtype=np.float32),
             labels=np.array([r["cls"] for r in rows]),
             hosts=np.array([r["host"] for r in rows]),
             srcs=np.array([r["src"] for r in rows]),
             files=np.array([r["file"] for r in rows]))
    print(f"  cached embeddings -> {EMB_CACHE}")

    def block(subset: list[dict]) -> dict:
        m = np.array([r["margin"] for r in subset])
        n = len(m)
        sweep = {}
        for theta in THETAS:
            ni = int((m < theta).sum())
            sweep[f"theta_{theta}"] = {
                "n_intercepted": ni, "interception_rate": round(ni / n, 4),
                "wilson_ci95": list(wilson_ci(ni, n)),
            }
        ni_p = int((m < THETA_PRIMARY).sum())
        decided = [r for r in subset if r["margin"] >= THETA_PRIMARY]
        # source-level: a source is "intercepted" if the majority of its frames are
        srcs: dict[str, list[float]] = {}
        for r in subset:
            srcs.setdefault(r["src"], []).append(r["margin"])
        src_int = sum(1 for v in srcs.values()
                      if np.mean([x < THETA_PRIMARY for x in v]) >= 0.5)
        return {
            "n_frames": n,
            "n_sources": len(srcs),
            "theta_sweep": sweep,
            "primary_theta": THETA_PRIMARY,
            "frame_level": {
                "n_intercepted": ni_p, "interception_rate": round(ni_p / n, 4),
                "wilson_ci95": list(wilson_ci(ni_p, n)),
            },
            "source_level": {
                "n_intercepted": src_int, "interception_rate": round(src_int / len(srcs), 4),
                "wilson_ci95": list(wilson_ci(src_int, len(srcs))),
            },
            "margin_mean": round(float(m.mean()), 4),
            "margin_min": round(float(m.min()), 4),
            "margin_max": round(float(m.max()), 4),
            "decided_top1_distribution": dict(Counter(r["top1"] for r in decided)),
        }

    result = {
        "experiment": "newdx_ood_interception",
        "purpose": "Test whether the margin safety valve (theta_margin) that intercepts EUS "
                   "also intercepts two novel parasitic diseases (Argulus fish-louse, "
                   "Oodinium/Velvet) not present in the D1 gallery.",
        "gpu": gpu_name,
        "gallery": "results/_cache_d1_gallery.npz (D1 fused, 7 classes; no argulus/oodinium)",
        "lambda": LAMBDA, "theta_margin": THETA_PRIMARY,
        "margin_def": "top1_sim - top2_sim over fused gallery (Eq. 11)",
        "data_source": "Journal/fish_data (provenance README per class)",
        "overall": block(rows),
        "by_class": {c: block([r for r in rows if r["cls"] == c])
                     for c in sorted({r["cls"] for r in rows})},
        "oodinium_by_host": {h: block([r for r in rows if r["cls"] == "oodinium" and r["host"] == h])
                             for h in sorted({r["host"] for r in rows if r["cls"] == "oodinium"})},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 72)
    for name, blk in [("OVERALL", result["overall"]),
                      ("argulus", result["by_class"]["argulus"]),
                      ("oodinium", result["by_class"]["oodinium"])]:
        fl, sl = blk["frame_level"], blk["source_level"]
        print(f"  {name:9s} frame {fl['n_intercepted']}/{blk['n_frames']}="
              f"{fl['interception_rate']:.3f}  source {sl['n_intercepted']}/{blk['n_sources']}="
              f"{sl['interception_rate']:.3f}  decided->{blk['decided_top1_distribution']}")
    print(f"  saved -> {OUT}")
    print("=" * 72)


if __name__ == "__main__":
    main()
