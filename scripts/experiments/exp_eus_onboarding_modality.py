"""Clean matched EUS onboarding: visual-only vs λ=0.7 fused k-NN, on the audited 56/50 set.

Resolves an integrity question about the manuscript's onboarding claim. The manuscript's
Table 5 / §4.6 compares a "Pipeline (k=1, Scoring)" column (0.714 at n_ref=50) against a
"kNN (k=1)" column (0.446). Inspection of the generating code shows the two columns actually
differ only in RETRIEVAL MODALITY, not in a scoring stage:
  - exp15 ("Pipeline") classifies EUS test images by nearest D1+ref *image* embedding — i.e.
    VISUAL-ONLY k-NN (encode_image only; the text embeddings it computes are never used; no
    scoring call).  → 0.714
  - exp19 ("kNN") uses λ=0.7 FUSED embeddings. → 0.446
Both use the same 56 EUS test images (D2 test_split/EUS_*) and the same 50-image reference pool
(D2 train_split/EUS), same D1 gallery. So "60.1% Pipeline-over-kNN" is really "visual-only
onboarding beats λ=0.7 fused onboarding" — consistent with the paper's central finding that
visual embeddings dominate and fusion adds ≈0 in forced choice (§4.2–4.3).

This script recomputes that comparison cleanly and freshly on the audited set: for each n_ref,
onboard the same n_ref EUS references and classify the same 56 test images by k=1 nearest
neighbour in (a) visual-only space and (b) λ=0.7 fused space. Reuses the D1 gallery cache; runs
Florence-2 + CLIP only on the 106 EUS images. seed=42. No simulation, no hard-coded outputs.
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
SEED = 42
N_REFS = [0, 5, 10, 20, 50]
CACHE = ROOT / "results/_cache_d1_gallery.npz"
D2 = ROOT / "lab_dateset/external_datasets/fish_disease_detection/New Dataset"
TEST_DIR = D2 / "test_split"
REF_DIR = D2 / "train_split/EUS"
OUT = ROOT / "results/eus_onboarding_modality.json"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


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


def encode(files, clip, prep, tok, proc, flor, prompt):
    vis, cap = [], []
    for k, f in enumerate(files, 1):
        image = Image.open(f).convert("RGB")
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
        vis.append(v[0].cpu().numpy().astype(np.float32))
        cap.append(t[0].cpu().numpy().astype(np.float32))
    return np.array(vis), np.array(cap)


def main() -> None:
    print("=" * 72)
    print("EUS onboarding — visual-only vs λ=0.7 fused (matched 56/50)")
    print("=" * 72)
    z = np.load(CACHE, allow_pickle=True)
    gv, gc, gl = z["visual"], z["caption"], z["labels"].astype("<U16")

    test_files = sorted(TEST_DIR.glob("EUS_*.jpg"))
    ref_all = sorted(p for p in REF_DIR.iterdir() if p.suffix.lower() in IMG_EXT)
    rng = np.random.default_rng(SEED)
    ref_files = [ref_all[i] for i in rng.permutation(len(ref_all))[:50]]
    print(f"  test: {len(test_files)}  ref pool: {len(ref_files)} (of {len(ref_all)})")

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

    print("  encoding EUS test + refs (Florence-2 + CLIP)...")
    tv, tc = encode(test_files, clip, prep, tok, proc, flor, prompt)
    rv, rc = encode(ref_files, clip, prep, tok, proc, flor, prompt)

    def curve(mode: str):
        if mode == "visual":
            g_base, q = gv.copy(), l2(tv)
            g_base = l2(g_base); ref_emb = l2(rv)
        else:  # fused
            g_base = l2(LAMBDA * gv + (1 - LAMBDA) * gc)
            q = l2(LAMBDA * tv + (1 - LAMBDA) * tc)
            ref_emb = l2(LAMBDA * rv + (1 - LAMBDA) * rc)
        pts = []
        for n in N_REFS:
            if n == 0:
                gg, ll = g_base, gl
            else:
                gg = np.concatenate([g_base, ref_emb[:n]], axis=0)
                ll = np.concatenate([gl, np.array(["EUS"] * n)], axis=0)
            pred = ll[(q @ gg.T).argmax(axis=1)]
            nok = int((pred == "EUS").sum())
            pts.append({"n_ref": n, "recognition_DA": round(nok / len(q), 4),
                        "n_correct": nok, "n_test": len(q), "wilson_ci95": list(wilson_ci(nok, len(q)))})
        return pts

    vis_curve = curve("visual")
    fus_curve = curve("fused")
    v50 = vis_curve[-1]["recognition_DA"]
    f50 = fus_curve[-1]["recognition_DA"]
    rel = round((v50 - f50) / f50 * 100, 1) if f50 > 0 else None

    result = {
        "experiment": "eus_onboarding_modality",
        "purpose": "Freshly recompute EUS onboarding as visual-only vs λ=0.7 fused k-NN on the "
                   "audited matched set (56 test / 50 ref), resolving the mislabeled "
                   "'Pipeline(Scoring) vs kNN' framing in Table 5 / §4.6.",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "test": "D2 test_split/EUS_* (56)", "ref_pool": "D2 train_split/EUS (50, seed=42)",
        "gallery": "results/_cache_d1_gallery.npz (D1, 7 classes)", "lambda": LAMBDA, "seed": SEED,
        "visual_only_curve": vis_curve,
        "fused_lambda07_curve": fus_curve,
        "n_ref_50": {"visual_only_DA": v50, "fused_DA": f50,
                     "visual_over_fused_relative_pct": rel,
                     "interpretation": "The manuscript's '0.714 Pipeline' is this visual-only DA; "
                                       "its '0.446 kNN' is this fused DA. The difference is modality, "
                                       "not a scoring stage."},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("\n  visual-only:", " ".join(f"{p['n_ref']}:{p['recognition_DA']:.3f}" for p in vis_curve))
    print("  fused λ0.7 :", " ".join(f"{p['n_ref']}:{p['recognition_DA']:.3f}" for p in fus_curve))
    print(f"  n_ref=50: visual {v50:.4f} vs fused {f50:.4f}  (visual over fused = {rel}%)")
    print(f"  saved -> {OUT}")


if __name__ == "__main__":
    main()
