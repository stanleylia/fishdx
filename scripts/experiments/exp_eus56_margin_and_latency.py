"""Real EUS-test-subset margin interception (56 images) + single-image latency/VRAM.

Addresses two audit findings:
  (1) Manuscript's headline "EUS OOD interception 69.6% (39/56)" had NO committed
      generating artifact (only a hard-coded value in a self-described 'synthetic'
      regression stub). This recomputes it on the REAL 56-image EUS *test* subset
      (D2 test_split/EUS_*), retrieving against the real D1 fused gallery, using the
      exact fusion (lambda=0.7, Eqs. 5-7) and margin definition (top1-top2, Eq. 11)
      as exp16b / demo_live_trace.
  (2) Manuscript's headline latency "median 533 ms/image, 921 MB peak GPU on RTX 3070"
      had NO measurement artifact (the one real benchmark used np.random inputs and
      excluded the models). This measures the REAL pipeline (Florence-2 caption + CLIP
      encode + fused retrieval + margin) per image on the actual GPU, reporting median /
      p95 latency and peak CUDA memory.

All models real: Florence-2-base (MORE_DETAILED_CAPTION, beam=3, deterministic, fp16),
OpenCLIP ViT-B-32 LAION-2B. Gallery: results/_cache_d1_gallery.npz (1,639 real D1
reference embeddings, 7 classes, no EUS). No simulation, no hard-coded outputs.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LAMBDA = 0.7
THETAS = [0.01, 0.02, 0.05]          # manuscript's EUS interception sweep
THETA_PRIMARY = 0.02
CACHE = ROOT / "results/_cache_d1_gallery.npz"
EUS_TEST_DIR = ROOT / "lab_dateset/external_datasets/fish_disease_detection/New Dataset/test_split"
OUT = ROOT / "results/eus56_margin_and_latency.json"
BOOT_RESAMPLES, SEED = 10000, 42


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


def bootstrap_ci(outcomes: np.ndarray, b: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(outcomes)
    means = np.empty(b, dtype=np.float64)
    for i in range(b):
        means[i] = outcomes[rng.integers(0, n, n)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("EUS-56 margin interception + single-image latency/VRAM")
    print(f"  GPU: {gpu_name}")
    print("=" * 70)

    z = np.load(CACHE, allow_pickle=True)
    gv, gc, gl = z["visual"], z["caption"], z["labels"]
    g_fus = l2(LAMBDA * gv + (1 - LAMBDA) * gc)
    assert "EUS" not in set(gl.tolist()), "gallery must not contain EUS"
    print(f"  gallery: {len(gl)} refs, classes={sorted(set(gl.tolist()))}")

    eus_files = sorted(EUS_TEST_DIR.glob("EUS_*.jpg"))
    print(f"  EUS test images: {len(eus_files)}")

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

    margins, top1_labels = [], []
    latencies_ms = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for k, f in enumerate(eus_files, 1):
        image = Image.open(f).convert("RGB")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

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
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        margins.append(margin)
        top1_labels.append(str(gl[order[0]]))
        if k % 10 == 0 or k == len(eus_files):
            print(f"  [{k:2d}/{len(eus_files)}] margin={margin:.4f} "
                  f"{'INTERCEPT' if margin < THETA_PRIMARY else 'decide->' + str(gl[order[0]])}")

    margins_arr = np.array(margins)
    n = len(margins_arr)
    sweep = {}
    for theta in THETAS:
        out_t = (margins_arr < theta).astype(np.float64)
        n_int_t = int(out_t.sum())
        sweep[f"theta_{theta}"] = {
            "n_intercepted": n_int_t,
            "interception_rate": round(n_int_t / n, 4),
            "wilson_ci95": list(wilson_ci(n_int_t, n)),
        }
    prim = (margins_arr < THETA_PRIMARY).astype(np.float64)
    n_int = int(prim.sum())
    rate = n_int / n
    ci_lo, ci_hi = bootstrap_ci(prim, BOOT_RESAMPLES, SEED)
    w_lo, w_hi = wilson_ci(n_int, n)
    outcomes = prim

    peak_mb = (torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0
    lat = np.array(latencies_ms)

    result = {
        "experiment": "eus56_margin_interception_and_latency",
        "purpose": "Recompute EUS OOD interception on the real 56-image EUS test subset "
                   "and measure real single-image latency/VRAM (audit fixes 1 & 2).",
        "gpu": gpu_name,
        "eus_interception": {
            "subset": "D2 test_split/EUS_* (the manuscript's 56-image EUS test set)",
            "gallery": "results/_cache_d1_gallery.npz (D1 fused, 7 classes, no EUS)",
            "lambda": LAMBDA, "theta_margin": THETA_PRIMARY,
            "margin_def": "top1_sim - top2_sim over fused gallery (Eq. 11)",
            "n_test": n,
            "theta_sweep": sweep,
            "primary_theta": THETA_PRIMARY,
            "n_intercepted": n_int,
            "interception_rate": round(rate, 4),
            "wilson_ci95": [round(w_lo, 4), round(w_hi, 4)],
            "bootstrap_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
            "bootstrap_resamples": BOOT_RESAMPLES, "seed": SEED,
            "margin_mean": round(float(np.mean(margins)), 4),
            "margin_min": round(float(np.min(margins)), 4),
            "margin_max": round(float(np.max(margins)), 4),
            "decided_top1_distribution": {c: int((np.array(top1_labels)[~outcomes.astype(bool)] == c).sum())
                                          for c in sorted(set(top1_labels))},
        },
        "latency": {
            "pipeline": "Florence-2 MORE_DETAILED_CAPTION (beam=3) + CLIP image+text "
                        "encode + fused retrieval + margin, per image",
            "n_images_timed": int(len(lat)),
            "median_ms": round(float(np.median(lat)), 1),
            "mean_ms": round(float(lat.mean()), 1),
            "p95_ms": round(float(np.percentile(lat, 95)), 1),
            "min_ms": round(float(lat.min()), 1),
            "max_ms": round(float(lat.max()), 1),
            "peak_cuda_mem_mb": round(float(peak_mb), 1),
            "note": "Retrieval-only pipeline (no LLM reasoning stage). Measured on the "
                    "actual GPU above; not a simulated benchmark.",
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("\n" + "=" * 70)
    print(f"  EUS interception sweep (of {n}): " +
          "  ".join(f"θ={t}:{sweep[f'theta_{t}']['n_intercepted']}"
                    f"({sweep[f'theta_{t}']['interception_rate']:.3f})" for t in THETAS))
    print(f"  primary θ={THETA_PRIMARY}: {n_int}/{n} = {rate:.4f}  "
          f"(Wilson CI {w_lo:.3f}-{w_hi:.3f}; bootstrap {ci_lo:.3f}-{ci_hi:.3f})")
    print(f"  Latency: median {np.median(lat):.0f} ms, p95 {np.percentile(lat, 95):.0f} ms, "
          f"peak CUDA {peak_mb:.0f} MB")
    print(f"  saved -> {OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
