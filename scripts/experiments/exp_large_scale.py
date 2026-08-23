#!/usr/bin/env python3
"""Large-Scale Experiments — 1000+ real images per experiment.

Phase A: Batch Florence-2 + CLIP inference with JSON/NPZ caching
Phase B: Run 9 experiments using cached model outputs
Phase C: Generate comprehensive experiment report with embedded images

All model inference is REAL GPU execution. No simulated data.
Stage 3 uses the deterministic evidence-pool scoring (Eqs. 8-11); no free-form LLM is used.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (
    load_config, AppConfig, ChromaDBConfig, FusionConfig,
)
from core.models.florence2_wrapper import Florence2Wrapper
from core.models.clip_wrapper import CLIPWrapper
from core.models.chromadb_client import ChromaDBClient
from core.algorithms.fusion import (
    create_fusion_embedding, fuse_embeddings, normalize_l2,
)
from core.algorithms.pareidolia import (
    detect_hard, detect_soft, detect_combined, remap_labels,
)
from core.algorithms.semantic_filter import classify_scene
from core.algorithms.scoring import full_scoring_pipeline
from core.algorithms.verification import (
    run_verification_loop, extract_disease_keywords, RAGItem,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("exp_large_scale")

# ════════════════════════════════════════════════════════════
# Paths
# ════════════════════════════════════════════════════════════
EXT = PROJECT_ROOT / "lab_dateset" / "external_datasets"
RESULTS_DIR = PROJECT_ROOT / "lab_dateset" / "organized" / "experiment_results" / "large_scale"
CACHE_DIR = PROJECT_ROOT / "lab_dateset" / "organized" / "experiment_results" / "large_scale_cache"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures" / "large_scale_samples"
REPORT_FILE = PROJECT_ROOT / "docs" / "EXPERIMENT_RESULTS_LARGE_SCALE.md"

DS_SOUTH_ASIA = EXT / "fish_disease_south_asia" / "Freshwater Fish Disease Aquaculture in south asia"
DS_DETECTION = EXT / "fish_disease_detection" / "New Dataset"
DS_CLEANED = EXT / "fish_disease_cleaned" / "Fish Disease Dataset"
DS_ALAA = EXT / "fish_disease_alaa" / "Fish Disease Dataset"
DS_LARGE_SCALE = EXT / "large_scale_fish" / "Fish_Dataset" / "Fish_Dataset"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DISEASE_DESCRIPTIONS = {
    "Bacterial diseases - Aeromoniasis": (
        "This fish is infected with Aeromonas bacteria causing Aeromoniasis. "
        "Symptoms include hemorrhagic septicemia, skin ulceration, and fin erosion. "
        "The fish shows signs of bacterial infection confirmed as Aeromoniasis disease."
    ),
    "Bacterial gill disease": (
        "This fish has bacterial gill disease. The gills show inflammation, "
        "swelling, and necrosis. Infected with pathogenic bacteria affecting "
        "the gill tissue. Suspected bacterial gill infection confirmed."
    ),
    "Bacterial Red disease": (
        "This fish is infected with bacterial red disease. Shows red spots, "
        "hemorrhage and redness on the body surface. The infection has caused "
        "visible hemorrhagic lesions. Confirmed bacterial red disease."
    ),
    "Fungal diseases Saprolegniasis": (
        "This fish shows fungal infection caused by Saprolegnia species. "
        "White cotton-like growths visible on body and fins. The fungus has "
        "colonized damaged tissue. Diagnosed as Saprolegniasis fungal disease."
    ),
    "Parasitic diseases": (
        "This fish is infected with parasites. Shows white spots, excess mucus "
        "and signs of irritation. Parasitic organisms visible on body surface. "
        "Suspected parasitic disease infection confirmed."
    ),
    "Viral diseases White tail disease": (
        "This fish shows viral white tail disease. The tail and posterior body "
        "have characteristic white discoloration. Viral infection has caused "
        "tissue damage. Confirmed viral white tail disease."
    ),
    "Healthy Fish": (
        "This fish appears healthy with no visible signs of disease. Good body "
        "condition, normal coloration, and no lesions observed. The fish shows "
        "no abnormalities and appears to be in healthy condition."
    ),
    "EUS": (
        "This fish shows epizootic ulcerative syndrome (EUS). Deep ulcers "
        "visible on body surface with secondary fungal or bacterial co-infection. "
        "The disease has caused severe tissue damage. Confirmed EUS."
    ),
    "FreshFish": (
        "This fish appears healthy and fresh with no disease signs. Normal "
        "appearance, good body condition. No abnormalities detected."
    ),
    "InfectedFish": (
        "This fish is infected with disease. Shows abnormal appearance with "
        "visible lesions and discoloration. Signs of infection confirmed."
    ),
}

# ════════════════════════════════════════════════════════════
# Dataset Scanning
# ════════════════════════════════════════════════════════════

def scan_class_dataset(base_dir: Path) -> list[tuple[str, str]]:
    """Scan dataset organized as base_dir/{class_name}/{images}."""
    samples = []
    if not base_dir.exists():
        logger.warning(f"Dataset dir not found: {base_dir}")
        return samples
    for class_dir in sorted(base_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in IMG_EXT:
                samples.append((str(img_file), label))
    logger.info(f"Scanned {base_dir.name}: {len(samples)} images, "
                f"{len(set(s[1] for s in samples))} classes")
    return samples


def scan_large_scale_fish(base_dir: Path, max_per_species: int = 0) -> list[tuple[str, str]]:
    """Scan large_scale_fish dataset (nested dirs, skip GT)."""
    samples = []
    if not base_dir.exists():
        logger.warning(f"Dataset dir not found: {base_dir}")
        return samples
    for species_dir in sorted(base_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        label = species_dir.name
        img_subdir = species_dir / label
        if not img_subdir.exists():
            continue
        count = 0
        for img_file in sorted(img_subdir.iterdir()):
            if img_file.suffix.lower() in IMG_EXT:
                samples.append((str(img_file), label))
                count += 1
                if max_per_species > 0 and count >= max_per_species:
                    break
    logger.info(f"Scanned large_scale_fish: {len(samples)} images, "
                f"{len(set(s[1] for s in samples))} species")
    return samples


# ════════════════════════════════════════════════════════════
# Phase A — Data Collection with Caching
# ════════════════════════════════════════════════════════════

def collect_florence2_batch(
    dataset_name: str,
    samples: list[tuple[str, str]],
    cache_file: Path,
    florence2: Florence2Wrapper,
) -> dict:
    """Run Florence-2 perceive on all images with JSON caching."""
    if cache_file.exists():
        logger.info(f"Loading Florence-2 cache: {cache_file.name}")
        with open(cache_file) as f:
            cached = json.load(f)
        logger.info(f"  Loaded {cached['count']} results")
        return cached

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    total = len(samples)
    errors = 0
    t_start = time.time()

    for i, (img_path, label) in enumerate(samples):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            logger.info(f"  [F2 {dataset_name}] {i+1}/{total} "
                        f"({rate:.1f} img/s, ETA {eta:.0f}s)")
        try:
            t0 = time.time()
            img = Image.open(img_path).convert("RGB")
            perception = florence2.perceive(img)
            latency = (time.time() - t0) * 1000
            results[img_path] = {
                "caption": perception["caption"],
                "objects": perception["objects"],
                "class_label": label,
                "latency_ms": round(latency, 1),
            }
        except Exception as e:
            errors += 1
            logger.warning(f"  F2 error {img_path}: {e}")
            results[img_path] = {
                "caption": "", "objects": [], "class_label": label,
                "latency_ms": 0.0, "error": str(e),
            }

    total_time = time.time() - t_start
    cache_data = {
        "dataset": dataset_name, "count": len(results), "errors": errors,
        "total_time_s": round(total_time, 1),
        "avg_latency_ms": round(total_time * 1000 / max(len(results), 1), 1),
        "timestamp": datetime.now().isoformat(), "results": results,
    }
    with open(cache_file, "w") as f:
        json.dump(cache_data, f, ensure_ascii=False)
    logger.info(f"  Florence-2 {dataset_name}: {len(results)} saved "
                f"({errors} errors, {total_time:.1f}s)")
    return cache_data


def collect_clip_batch(
    dataset_name: str,
    samples: list[tuple[str, str]],
    cache_file: Path,
    clip: CLIPWrapper,
) -> dict:
    """Run CLIP image encoding on all images with NPZ caching."""
    npz_file = cache_file.with_suffix(".npz")
    meta_file = cache_file.with_suffix(".json")

    if npz_file.exists() and meta_file.exists():
        logger.info(f"Loading CLIP cache: {npz_file.name}")
        data = np.load(npz_file)
        with open(meta_file) as f:
            meta = json.load(f)
        logger.info(f"  Loaded {len(meta['paths'])} embeddings")
        return {"paths": meta["paths"], "labels": meta["labels"],
                "embeddings": data["embeddings"]}

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    paths, labels, embeddings = [], [], []
    errors = 0
    total = len(samples)
    t_start = time.time()

    for i, (img_path, label) in enumerate(samples):
        if (i + 1) % 200 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            logger.info(f"  [CLIP {dataset_name}] {i+1}/{total} "
                        f"({rate:.1f} img/s, ETA {eta:.0f}s)")
        try:
            img = Image.open(img_path).convert("RGB")
            emb = clip.encode_image(img)
            paths.append(img_path)
            labels.append(label)
            embeddings.append(emb)
        except Exception as e:
            errors += 1
            logger.warning(f"  CLIP error {img_path}: {e}")

    emb_arr = np.stack(embeddings).astype(np.float32)
    total_time = time.time() - t_start

    np.savez_compressed(npz_file, embeddings=emb_arr)
    with open(meta_file, "w") as f:
        json.dump({"dataset": dataset_name, "count": len(paths),
                    "errors": errors, "total_time_s": round(total_time, 1),
                    "timestamp": datetime.now().isoformat(),
                    "paths": paths, "labels": labels}, f, ensure_ascii=False)
    logger.info(f"  CLIP {dataset_name}: {len(paths)} embeddings "
                f"({errors} errors, {total_time:.1f}s)")
    return {"paths": paths, "labels": labels, "embeddings": emb_arr}


def get_caption_embeddings(
    s1_cache: dict, paths: list[str], cache_file: Path, clip: CLIPWrapper,
) -> np.ndarray:
    """Get CLIP text embeddings for Florence-2 captions (batch + cache)."""
    npz_file = cache_file.with_suffix(".npz")
    if npz_file.exists():
        logger.info(f"Loading caption emb cache: {npz_file.name}")
        return np.load(npz_file)["embeddings"]

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    captions = []
    for p in paths:
        entry = s1_cache["results"].get(p, {})
        cap = entry.get("caption", "")
        captions.append(cap if cap else "fish image")

    # Batch encode
    all_embs = []
    bs = 128
    for start in range(0, len(captions), bs):
        batch = captions[start:start + bs]
        batch_embs = clip.encode_texts(batch)
        all_embs.append(batch_embs)
    result = np.concatenate(all_embs, axis=0).astype(np.float32)
    np.savez_compressed(npz_file, embeddings=result)
    logger.info(f"  Caption embeddings: {len(paths)} encoded")
    return result


# ════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════

def save_result(filepath: Path, data: dict) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved: {filepath.name}")


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Compute accuracy, per-class P/R/F1, macro averages."""
    classes = sorted(set(y_true) | set(y_pred))
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / len(y_true) if y_true else 0.0
    cm = {}
    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        cm[cls] = {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4), "support": sum(1 for t in y_true if t == cls),
                    "tp": tp, "fp": fp, "fn": fn}
    macro_p = np.mean([m["precision"] for m in cm.values()]) if cm else 0.0
    macro_r = np.mean([m["recall"] for m in cm.values()]) if cm else 0.0
    macro_f1 = np.mean([m["f1"] for m in cm.values()]) if cm else 0.0
    return {"accuracy": round(acc, 4), "correct": correct, "total": len(y_true),
            "macro_precision": round(float(macro_p), 4),
            "macro_recall": round(float(macro_r), 4),
            "macro_f1": round(float(macro_f1), 4),
            "class_metrics": cm}


def stratified_sample(
    results: list[dict], key: str, n_per_class: int = 3,
) -> list[dict]:
    """Pick n_per_class samples from each unique class (first, mid, last)."""
    from collections import defaultdict
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_class[r[key]].append(r)
    out: list[dict] = []
    for cls in sorted(by_class.keys()):
        items = by_class[cls]
        indices = [0]
        if len(items) > 1:
            indices.append(len(items) // 2)
        if len(items) > 2:
            indices.append(len(items) - 1)
        for idx in indices[:n_per_class]:
            out.append(items[idx])
    return out


def build_experiment_db(
    name: str, embeddings: np.ndarray, labels: list[str],
    paths: list[str], documents: list[str] | None = None,
) -> ChromaDBClient:
    """Build a temporary ChromaDB collection for experiments."""
    persist_dir = f"/tmp/exp_chromadb/{name}"
    if Path(persist_dir).exists():
        shutil.rmtree(persist_dir)
    config = ChromaDBConfig(
        persist_directory=persist_dir, collection_name=name,
        distance_metric="cosine", top_k=5, similarity_cutoff=0.0,
    )
    client = ChromaDBClient(config)
    batch_size = 5000
    total = len(labels)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        ids = [f"doc_{i:06d}" for i in range(start, end)]
        embs = embeddings[start:end].tolist()
        docs = [(documents[i] if documents else labels[i]) for i in range(start, end)]
        metas = [{"source": labels[i], "class_label": labels[i]} for i in range(start, end)]
        client.add_documents(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
    logger.info(f"  ChromaDB '{name}': {client.count()} documents")
    return client


def make_grounding_fn(florence2: Florence2Wrapper):
    """Create Florence-2 grounding function for verification loop."""
    _cache = {"path": None, "img": None}
    def fn(image_path, keyword):
        if _cache["path"] != image_path:
            _cache["path"] = image_path
            _cache["img"] = Image.open(image_path).convert("RGB")
        return florence2.phrase_grounding(_cache["img"], keyword)
    return fn


def copy_sample_images(samples: list[tuple[str, str]], n_per_class: int = 3) -> dict:
    """Copy representative sample images to figures dir. Returns {class: [paths]}."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    by_class: dict[str, list[str]] = {}
    for path, label in samples:
        by_class.setdefault(label, []).append(path)

    copied = {}
    for cls, paths in sorted(by_class.items()):
        copied[cls] = []
        for i, p in enumerate(paths[:n_per_class]):
            safe_cls = cls.replace(" ", "_").replace("-", "_")
            dest = FIGURES_DIR / f"{safe_cls}_sample_{i+1}{Path(p).suffix}"
            if not dest.exists():
                shutil.copy2(p, dest)
            copied[cls].append(str(dest))
    return copied


# ════════════════════════════════════════════════════════════
# EXP-01: λ Ablation Study (Fusion Weight)
# ════════════════════════════════════════════════════════════

def run_exp01(
    train_clip: dict, test_clip: dict,
    train_s1: dict, test_s1: dict,
    clip: CLIPWrapper, config: AppConfig,
) -> dict:
    logger.info("=" * 60)
    logger.info("EXP-01: λ Ablation Study (2,444 images)")
    logger.info("=" * 60)
    t0 = time.time()

    tr_paths, tr_labels = train_clip["paths"], train_clip["labels"]
    te_paths, te_labels = test_clip["paths"], test_clip["labels"]
    tr_vis = train_clip["embeddings"]
    te_vis = test_clip["embeddings"]

    # Caption embeddings
    tr_cap = get_caption_embeddings(
        train_s1, tr_paths, CACHE_DIR / "cap_sa_train", clip)
    te_cap = get_caption_embeddings(
        test_s1, te_paths, CACHE_DIR / "cap_sa_test", clip)

    lambda_values = [round(x * 0.1, 1) for x in range(11)]
    lambda_results = []
    sample_details = []  # 3 sample images per λ=0.7

    for lam in lambda_values:
        logger.info(f"  λ = {lam:.1f}")
        fusion_cfg = FusionConfig(lambda_weight=lam)

        # Fuse train
        tr_fused = np.stack([
            create_fusion_embedding(tr_vis[i], tr_cap[i], fusion_cfg).fused_embedding
            for i in range(len(tr_paths))
        ])
        # Fuse test
        te_fused = np.stack([
            create_fusion_embedding(te_vis[i], te_cap[i], fusion_cfg).fused_embedding
            for i in range(len(te_paths))
        ])

        db = build_experiment_db(f"exp01_l{lam}", tr_fused, tr_labels, tr_paths)
        y_pred = []
        per_image = []

        for i in range(len(te_paths)):
            res = db.query(te_fused[i], top_k=1)
            pred = res[0]["source"] if res else "unknown"
            sim = res[0]["similarity"] if res else 0.0
            y_pred.append(pred)
            per_image.append({
                "path": te_paths[i], "ground_truth": te_labels[i],
                "predicted": pred, "similarity": round(float(sim), 4),
                "correct": pred == te_labels[i],
            })

        metrics = compute_metrics(te_labels, y_pred)
        lambda_results.append({"lambda": lam, **metrics})
        logger.info(f"    DA={metrics['accuracy']:.4f} F1={metrics['macro_f1']:.4f}")

        # Collect detailed samples for λ*=0.7
        if abs(lam - 0.7) < 0.01:
            sample_details = stratified_sample(per_image, "ground_truth", 3)

        db.delete_collection()

    best = max(lambda_results, key=lambda x: x["accuracy"])
    return {
        "experiment": "EXP-01", "name": "Lambda Ablation Study",
        "train_images": len(tr_paths), "test_images": len(te_paths),
        "total_images": len(tr_paths) + len(te_paths),
        "results": lambda_results,
        "best_lambda": best["lambda"], "best_accuracy": best["accuracy"],
        "sample_details": sample_details,
        "total_time_s": round(time.time() - t0, 1),
        "timestamp": datetime.now().isoformat(),
    }


# ════════════════════════════════════════════════════════════
# EXP-02: Two-Tier Pareidolia Detection
# ════════════════════════════════════════════════════════════

def run_exp02(
    disease_s1: dict, general_s1: dict,
    clip: CLIPWrapper, config: AppConfig,
) -> dict:
    logger.info("=" * 60)
    n_disease = disease_s1["count"]
    n_general = general_s1["count"]
    logger.info(f"EXP-02: Pareidolia Detection ({n_disease + n_general} images)")
    logger.info("=" * 60)
    t0 = time.time()

    clip_text_fn = clip.encode_text
    pareidolia_cfg = config.pareidolia
    results_disease = []
    results_general = []

    for tag, s1_cache, out_list in [
        ("disease", disease_s1, results_disease),
        ("general", general_s1, results_general),
    ]:
        for i, (img_path, data) in enumerate(s1_cache["results"].items()):
            if (i + 1) % 200 == 0:
                logger.info(f"  [Pareidolia {tag}] {i+1}/{len(s1_cache['results'])}")
            caption = data.get("caption", "")
            objects = data.get("objects", [])
            obj_labels = [o["label"] for o in objects if isinstance(o, dict)]
            if not obj_labels:
                obj_labels = ["unknown"]

            hard = detect_hard(caption, obj_labels, pareidolia_cfg)
            combined = detect_combined(
                caption, obj_labels, pareidolia_cfg,
                clip_text_fn=clip_text_fn,
            )
            remapped = remap_labels(obj_labels, combined)

            out_list.append({
                "path": img_path,
                "class_label": data.get("class_label", ""),
                "caption": caption,
                "object_labels": obj_labels,
                "hard_triggered": any(r.is_pareidolia for r in hard),
                "hard_scores": [round(r.hard_score, 4) for r in hard],
                "combined_triggered": any(r.is_pareidolia for r in combined),
                "combined_scores": [round(r.combined_score, 4) for r in combined],
                "soft_scores": [round(r.soft_score, 4) for r in combined],
                "remapped_labels": remapped,
            })

    # Aggregate
    d_hard_rate = sum(1 for r in results_disease if r["hard_triggered"]) / max(len(results_disease), 1)
    d_comb_rate = sum(1 for r in results_disease if r["combined_triggered"]) / max(len(results_disease), 1)
    g_hard_rate = sum(1 for r in results_general if r["hard_triggered"]) / max(len(results_general), 1)
    g_comb_rate = sum(1 for r in results_general if r["combined_triggered"]) / max(len(results_general), 1)

    # Per-class breakdown for disease dataset
    class_rates = {}
    for r in results_disease:
        cls = r["class_label"]
        class_rates.setdefault(cls, {"total": 0, "hard": 0, "combined": 0})
        class_rates[cls]["total"] += 1
        if r["hard_triggered"]:
            class_rates[cls]["hard"] += 1
        if r["combined_triggered"]:
            class_rates[cls]["combined"] += 1

    return {
        "experiment": "EXP-02", "name": "Two-Tier Pareidolia Detection",
        "disease_images": len(results_disease),
        "general_images": len(results_general),
        "total_images": len(results_disease) + len(results_general),
        "disease_hard_trigger_rate": round(d_hard_rate, 4),
        "disease_combined_trigger_rate": round(d_comb_rate, 4),
        "general_hard_trigger_rate": round(g_hard_rate, 4),
        "general_combined_trigger_rate": round(g_comb_rate, 4),
        "class_breakdown": {
            cls: {"hard_rate": round(v["hard"] / v["total"], 4),
                  "combined_rate": round(v["combined"] / v["total"], 4),
                  "count": v["total"]}
            for cls, v in sorted(class_rates.items())
        },
        "sample_disease": stratified_sample(results_disease, "class_label", 3),
        "sample_general": stratified_sample(results_general, "class_label", 2),
        "total_time_s": round(time.time() - t0, 1),
        "timestamp": datetime.now().isoformat(),
    }


# ════════════════════════════════════════════════════════════
# EXP-03: Semantic Filter Effectiveness
# ════════════════════════════════════════════════════════════

def run_exp03(
    disease_s1: dict, general_s1: dict, config: AppConfig,
) -> dict:
    logger.info("=" * 60)
    n_total = disease_s1["count"] + general_s1["count"]
    logger.info(f"EXP-03: Semantic Filter ({n_total} images)")
    logger.info("=" * 60)
    t0 = time.time()

    sf_cfg = config.semantic_filter
    results_disease = []
    results_general = []

    for tag, s1_cache, out_list in [
        ("disease", disease_s1, results_disease),
        ("general", general_s1, results_general),
    ]:
        for i, (img_path, data) in enumerate(s1_cache["results"].items()):
            if (i + 1) % 200 == 0:
                logger.info(f"  [SemFilter {tag}] {i+1}/{len(s1_cache['results'])}")
            caption = data.get("caption", "")
            sc = classify_scene(caption, sf_cfg)
            out_list.append({
                "path": img_path,
                "class_label": data.get("class_label", ""),
                "caption": caption[:200],
                "scene_type": sc.scene_type,
                "score": round(sc.score, 4),
                "rag_triggered": sc.rag_triggered,
                "matched_keywords": sc.matched_keywords,
                "profile_name": sc.profile_name,
            })

    d_rag = sum(1 for r in results_disease if r["rag_triggered"]) / max(len(results_disease), 1)
    g_rag = sum(1 for r in results_general if r["rag_triggered"]) / max(len(results_general), 1)

    # Per-class for disease
    class_rates = {}
    for r in results_disease:
        cls = r["class_label"]
        class_rates.setdefault(cls, {"total": 0, "triggered": 0, "scores": []})
        class_rates[cls]["total"] += 1
        class_rates[cls]["scores"].append(r["score"])
        if r["rag_triggered"]:
            class_rates[cls]["triggered"] += 1

    # Scene type distribution for general
    general_scenes = {}
    for r in results_general:
        st = r["scene_type"]
        general_scenes[st] = general_scenes.get(st, 0) + 1

    return {
        "experiment": "EXP-03", "name": "Semantic Filter Effectiveness",
        "disease_images": len(results_disease),
        "general_images": len(results_general),
        "total_images": len(results_disease) + len(results_general),
        "disease_rag_trigger_rate": round(d_rag, 4),
        "general_rag_trigger_rate": round(g_rag, 4),
        "class_rag_rates": {
            cls: {"trigger_rate": round(v["triggered"] / v["total"], 4),
                  "avg_score": round(float(np.mean(v["scores"])), 4),
                  "count": v["total"]}
            for cls, v in sorted(class_rates.items())
        },
        "general_scene_distribution": general_scenes,
        "sample_disease": stratified_sample(results_disease, "class_label", 3),
        "sample_general": stratified_sample(results_general, "class_label", 2),
        "total_time_s": round(time.time() - t0, 1),
        "timestamp": datetime.now().isoformat(),
    }


# ════════════════════════════════════════════════════════════
# EXP-04: CLIP Embedding Separability
# ════════════════════════════════════════════════════════════

def run_exp04(train_clip: dict) -> dict:
    logger.info("=" * 60)
    logger.info(f"EXP-04: CLIP Separability ({len(train_clip['paths'])} images)")
    logger.info("=" * 60)
    t0 = time.time()

    paths = train_clip["paths"]
    labels = train_clip["labels"]
    embs = train_clip["embeddings"]

    # Normalize embeddings
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_norm = embs / (norms + 1e-10)

    # Per-class
    classes = sorted(set(labels))
    class_embs = {cls: [] for cls in classes}
    for i, label in enumerate(labels):
        class_embs[label].append(embs_norm[i])

    centroids = {}
    intra_distances = {}
    for cls in classes:
        arr = np.stack(class_embs[cls])
        centroid = arr.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        centroids[cls] = centroid
        dists = 1.0 - np.dot(arr, centroid)
        intra_distances[cls] = {
            "mean": round(float(dists.mean()), 4),
            "std": round(float(dists.std()), 4),
            "min": round(float(dists.min()), 4),
            "max": round(float(dists.max()), 4),
            "count": len(arr),
        }

    # Inter-class distance matrix
    inter_matrix = {}
    inter_dists = []
    for ci in classes:
        inter_matrix[ci] = {}
        for cj in classes:
            d = 1.0 - float(np.dot(centroids[ci], centroids[cj]))
            inter_matrix[ci][cj] = round(d, 4)
            if ci != cj:
                inter_dists.append(d)

    avg_inter = float(np.mean(inter_dists)) if inter_dists else 0.0
    avg_intra = float(np.mean([v["mean"] for v in intra_distances.values()]))
    sep_ratio = avg_inter / avg_intra if avg_intra > 0 else 0.0

    # Per-image distances to own centroid
    per_image_samples = []
    for cls in classes:
        arr = np.stack(class_embs[cls])
        dists = 1.0 - np.dot(arr, centroids[cls])
        sorted_idx = np.argsort(dists)
        # Closest and farthest
        for idx in [sorted_idx[0], sorted_idx[len(sorted_idx)//2], sorted_idx[-1]]:
            label_paths = [p for p, l in zip(paths, labels) if l == cls]
            per_image_samples.append({
                "path": label_paths[idx] if idx < len(label_paths) else "",
                "class": cls,
                "distance_to_centroid": round(float(dists[idx]), 4),
                "position": "closest" if idx == sorted_idx[0] else
                           ("median" if idx == sorted_idx[len(sorted_idx)//2] else "farthest"),
            })

    return {
        "experiment": "EXP-04", "name": "CLIP Embedding Separability",
        "total_images": len(paths), "num_classes": len(classes),
        "separability_ratio": round(sep_ratio, 4),
        "avg_inter_class_distance": round(avg_inter, 4),
        "avg_intra_class_distance": round(avg_intra, 4),
        "intra_class_distances": intra_distances,
        "inter_class_matrix": inter_matrix,
        "per_image_samples": per_image_samples,
        "total_time_s": round(time.time() - t0, 1),
        "timestamp": datetime.now().isoformat(),
    }


# ════════════════════════════════════════════════════════════
# EXP-05: Cross-Dataset Generalization
# ════════════════════════════════════════════════════════════

def run_exp05(
    train_clip: dict, det_clip: dict,
    train_s1: dict, det_s1: dict,
    clip: CLIPWrapper, config: AppConfig,
) -> dict:
    logger.info("=" * 60)
    n_total = len(train_clip["paths"]) + len(det_clip["paths"])
    logger.info(f"EXP-05: Cross-Dataset Generalization ({n_total} images)")
    logger.info("=" * 60)
    t0 = time.time()

    tr_paths, tr_labels = train_clip["paths"], train_clip["labels"]
    tr_vis = train_clip["embeddings"]
    de_paths, de_labels = det_clip["paths"], det_clip["labels"]
    de_vis = det_clip["embeddings"]

    # Caption embeddings
    tr_cap = get_caption_embeddings(train_s1, tr_paths, CACHE_DIR / "cap_sa_train", clip)
    de_cap = get_caption_embeddings(det_s1, de_paths, CACHE_DIR / "cap_det_train", clip)

    fusion_cfg = FusionConfig(lambda_weight=0.7)

    # Fuse train
    tr_fused = np.stack([
        create_fusion_embedding(tr_vis[i], tr_cap[i], fusion_cfg).fused_embedding
        for i in range(len(tr_paths))
    ])
    # Fuse detection test
    de_fused = np.stack([
        create_fusion_embedding(de_vis[i], de_cap[i], fusion_cfg).fused_embedding
        for i in range(len(de_paths))
    ])

    # Build ChromaDB with train data + disease descriptions
    documents = [DISEASE_DESCRIPTIONS.get(l, l) for l in tr_labels]
    db = build_experiment_db("exp05", tr_fused, tr_labels, tr_paths, documents)

    y_pred = []
    per_image = []
    for i in range(len(de_paths)):
        if (i + 1) % 500 == 0:
            logger.info(f"  [EXP-05 query] {i+1}/{len(de_paths)}")
        res = db.query(de_fused[i], top_k=5)
        pred = res[0]["source"] if res else "unknown"
        y_pred.append(pred)
        per_image.append({
            "path": de_paths[i], "ground_truth": de_labels[i],
            "predicted": pred,
            "top5": [{"label": r["source"], "sim": round(r["similarity"], 4)}
                     for r in res[:5]],
            "correct": pred == de_labels[i],
        })

    metrics = compute_metrics(de_labels, y_pred)
    db.delete_collection()

    # Analyze label overlap
    train_classes = set(tr_labels)
    det_classes = set(de_labels)
    overlap = train_classes & det_classes
    unique_to_det = det_classes - train_classes

    # Metrics for overlapping vs unique classes
    overlap_pred = [(t, p) for t, p in zip(de_labels, y_pred) if t in overlap]
    unique_pred = [(t, p) for t, p in zip(de_labels, y_pred) if t in unique_to_det]

    return {
        "experiment": "EXP-05", "name": "Cross-Dataset Generalization",
        "train_images": len(tr_paths), "test_images": len(de_paths),
        "total_images": n_total,
        "lambda": 0.7,
        "metrics": metrics,
        "label_overlap": sorted(overlap),
        "unique_to_detection": sorted(unique_to_det),
        "overlap_accuracy": round(
            sum(1 for t, p in overlap_pred if t == p) / max(len(overlap_pred), 1), 4),
        "unique_class_accuracy": round(
            sum(1 for t, p in unique_pred if t == p) / max(len(unique_pred), 1), 4),
        "sample_details": stratified_sample(per_image, "ground_truth", 3),
        "total_time_s": round(time.time() - t0, 1),
        "timestamp": datetime.now().isoformat(),
    }


# ════════════════════════════════════════════════════════════
# EXP-NEW-01: Disease Classification (Full Pipeline)
# ════════════════════════════════════════════════════════════

def run_exp_new01(
    train_clip: dict, test_clip: dict, cl_clip: dict,
    train_s1: dict, test_s1: dict, cl_s1: dict,
    clip: CLIPWrapper, florence2: Florence2Wrapper, config: AppConfig,
) -> dict:
    logger.info("=" * 60)
    # Combine SA Test + Cleaned Valid
    te_paths = test_clip["paths"] + cl_clip["paths"]
    te_labels = test_clip["labels"] + cl_clip["labels"]
    te_vis = np.concatenate([test_clip["embeddings"], cl_clip["embeddings"]], axis=0)
    logger.info(f"EXP-NEW-01: Classification ({len(te_paths)} test, "
                f"{len(train_clip['paths'])} train)")
    logger.info("=" * 60)
    t0 = time.time()

    tr_paths, tr_labels = train_clip["paths"], train_clip["labels"]
    tr_vis = train_clip["embeddings"]

    # Caption embeddings
    tr_cap = get_caption_embeddings(train_s1, tr_paths, CACHE_DIR / "cap_sa_train", clip)

    # Test caption embeddings — merge caches
    te_cap_sa = get_caption_embeddings(test_s1, test_clip["paths"],
                                        CACHE_DIR / "cap_sa_test", clip)
    te_cap_cl = get_caption_embeddings(cl_s1, cl_clip["paths"],
                                        CACHE_DIR / "cap_cl_valid", clip)
    te_cap = np.concatenate([te_cap_sa, te_cap_cl], axis=0)

    fusion_cfg = FusionConfig(lambda_weight=0.7)
    sf_cfg = config.semantic_filter
    scoring_cfg = config.scoring
    quality_gate_cfg = config.quality_gate

    # Fuse train + build DB with disease descriptions
    tr_fused = np.stack([
        create_fusion_embedding(tr_vis[i], tr_cap[i], fusion_cfg).fused_embedding
        for i in range(len(tr_paths))
    ])
    documents = [DISEASE_DESCRIPTIONS.get(l, l) for l in tr_labels]
    db = build_experiment_db("exp_new01", tr_fused, tr_labels, tr_paths, documents)

    grounding_fn = make_grounding_fn(florence2)
    verification_cfg = config.verification
    per_image = []
    y_pred = []

    for i in range(len(te_paths)):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(f"  [NEW-01] {i+1}/{len(te_paths)} ({rate:.1f}/s)")

        p = te_paths[i]

        # Stage 1: Florence-2 perception (from cache)
        s1_data = {}
        if p in test_s1["results"]:
            s1_data = test_s1["results"][p]
        elif p in cl_s1["results"]:
            s1_data = cl_s1["results"][p]
        caption = s1_data.get("caption", "")
        objects = s1_data.get("objects", [])

        # Stage 2a: Semantic filter
        sc = classify_scene(caption, sf_cfg)

        # Stage 2b: Fusion embedding + ChromaDB query
        fused = create_fusion_embedding(te_vis[i], te_cap[i], fusion_cfg)
        rag_results = db.query(fused.fused_embedding, top_k=5)

        # Stage 2c: Verification loop (with grounding)
        rag_items = [
            RAGItem(content=r["content"], score=r["similarity"],
                    source=r["source"], doc_id=r["id"])
            for r in rag_results
        ]
        ver_result = None
        if rag_items and sc.rag_triggered:
            try:
                ver_result = run_verification_loop(
                    rag_items, p, grounding_fn, None, verification_cfg)
            except Exception as e:
                logger.warning(f"  Verification error {p}: {e}")

        # Stage 3: Scoring on RAG content
        combined_text = " ".join(r["content"] for r in rag_results[:3])
        scoring_result = full_scoring_pipeline(combined_text, scoring_cfg)

        # Predicted label from top-1 RAG
        pred = rag_results[0]["source"] if rag_results else "unknown"
        y_pred.append(pred)

        # Learning: Quality gate decision
        max_score = max(scoring_result.healthy_score, scoring_result.disease_score)
        if max_score >= quality_gate_cfg.min_confidence_score:
            quality_decision = "auto_ingest"
        elif max_score >= 2:
            quality_decision = "hitl_review"
        else:
            quality_decision = "discard"

        record = {
            "path": p,
            "ground_truth": te_labels[i],
            "stage1": {
                "caption": caption[:300],
                "objects": objects[:5],
                "latency_ms": s1_data.get("latency_ms", 0),
            },
            "stage2": {
                "semantic_filter": {
                    "scene_type": sc.scene_type, "score": round(sc.score, 4),
                    "rag_triggered": sc.rag_triggered,
                    "matched_keywords": sc.matched_keywords,
                },
                "fusion_lambda": 0.7,
                "rag_results": [
                    {"content": r["content"][:150], "similarity": round(r["similarity"], 4),
                     "source": r["source"]}
                    for r in rag_results[:5]
                ],
                "verification": {
                    "iterations_run": ver_result.iterations_run if ver_result else 0,
                    "keywords_checked": ver_result.keywords_checked if ver_result else [],
                    "items_removed": ver_result.items_removed if ver_result else 0,
                    "converged": ver_result.converged if ver_result else False,
                    "grounding_results": [
                        {"keyword": gr.keyword, "grounded": gr.grounded,
                         "confidence": round(gr.confidence, 4)}
                        for gr in (ver_result.grounding_results if ver_result else [])
                    ],
                } if ver_result else None,
            },
            "stage3_scoring": {
                "status": scoring_result.status,
                "healthy_score": scoring_result.healthy_score,
                "disease_score": scoring_result.disease_score,
                "confidence": round(scoring_result.confidence, 4),
                "matched_healthy": scoring_result.healthy_detail.matched_patterns,
                "matched_disease": scoring_result.disease_detail.matched_patterns,
                "scoring_text_used": combined_text[:200],
            },
            "learning": {
                "quality_decision": quality_decision,
                "max_score": max_score,
                "llm_available": False,
                "reason": "LLM unavailable (nested session)",
            },
            "predicted_label": pred,
            "correct": pred == te_labels[i],
        }
        per_image.append(record)

    metrics = compute_metrics(te_labels, y_pred)
    db.delete_collection()

    # Scoring statistics
    scoring_stats = {
        "healthy": sum(1 for r in per_image if r["stage3_scoring"]["status"] == "Healthy"),
        "disease": sum(1 for r in per_image if r["stage3_scoring"]["status"] == "Disease"),
        "inconclusive": sum(1 for r in per_image if r["stage3_scoring"]["status"] == "Inconclusive"),
    }
    quality_stats = {
        "auto_ingest": sum(1 for r in per_image if r["learning"]["quality_decision"] == "auto_ingest"),
        "hitl_review": sum(1 for r in per_image if r["learning"]["quality_decision"] == "hitl_review"),
        "discard": sum(1 for r in per_image if r["learning"]["quality_decision"] == "discard"),
    }

    return {
        "experiment": "EXP-NEW-01", "name": "Disease Classification (Full Pipeline)",
        "train_images": len(tr_paths), "test_images": len(te_paths),
        "total_images": len(tr_paths) + len(te_paths),
        "metrics": metrics,
        "scoring_distribution": scoring_stats,
        "quality_gate_distribution": quality_stats,
        "per_image_results": per_image,
        "total_time_s": round(time.time() - t0, 1),
        "timestamp": datetime.now().isoformat(),
    }


# ════════════════════════════════════════════════════════════
# EXP-NEW-02: Binary Detection (Fresh vs Infected)
# ════════════════════════════════════════════════════════════

def run_exp_new02(
    train_clip: dict, alaa_clip: dict,
    train_s1: dict, alaa_s1: dict,
    clip: CLIPWrapper, config: AppConfig,
) -> dict:
    logger.info("=" * 60)
    logger.info(f"EXP-NEW-02: Binary Detection ({len(alaa_clip['paths'])} test)")
    logger.info("=" * 60)
    t0 = time.time()

    tr_paths, tr_labels = train_clip["paths"], train_clip["labels"]
    tr_vis = train_clip["embeddings"]
    al_paths, al_labels = alaa_clip["paths"], alaa_clip["labels"]
    al_vis = alaa_clip["embeddings"]

    tr_cap = get_caption_embeddings(train_s1, tr_paths, CACHE_DIR / "cap_sa_train", clip)
    al_cap = get_caption_embeddings(alaa_s1, al_paths, CACHE_DIR / "cap_alaa", clip)

    fusion_cfg = FusionConfig(lambda_weight=0.7)
    sf_cfg = config.semantic_filter
    scoring_cfg = config.scoring

    tr_fused = np.stack([
        create_fusion_embedding(tr_vis[i], tr_cap[i], fusion_cfg).fused_embedding
        for i in range(len(tr_paths))
    ])
    al_fused = np.stack([
        create_fusion_embedding(al_vis[i], al_cap[i], fusion_cfg).fused_embedding
        for i in range(len(al_paths))
    ])

    documents = [DISEASE_DESCRIPTIONS.get(l, l) for l in tr_labels]
    db = build_experiment_db("exp_new02", tr_fused, tr_labels, tr_paths, documents)

    # Map alaa labels to binary
    label_map = {"FreshFish": "Healthy", "InfectedFish": "Disease"}
    per_image = []
    y_true_binary = []
    y_pred_binary = []

    for i in range(len(al_paths)):
        p = al_paths[i]
        s1_data = alaa_s1["results"].get(p, {})
        caption = s1_data.get("caption", "")

        sc = classify_scene(caption, sf_cfg)
        rag_results = db.query(al_fused[i], top_k=5)
        combined_text = " ".join(r["content"] for r in rag_results[:3])
        scoring_result = full_scoring_pipeline(combined_text, scoring_cfg)

        gt_binary = label_map.get(al_labels[i], al_labels[i])
        pred_label = rag_results[0]["source"] if rag_results else "unknown"
        pred_binary = "Healthy" if pred_label == "Healthy Fish" else "Disease"

        y_true_binary.append(gt_binary)
        y_pred_binary.append(pred_binary)

        per_image.append({
            "path": p,
            "ground_truth_original": al_labels[i],
            "ground_truth_binary": gt_binary,
            "stage1": {
                "caption": caption[:300],
                "objects": s1_data.get("objects", [])[:5],
            },
            "stage2": {
                "scene_type": sc.scene_type, "score": round(sc.score, 4),
                "rag_triggered": sc.rag_triggered,
                "top3_rag": [{"source": r["source"], "sim": round(r["similarity"], 4)}
                             for r in rag_results[:3]],
            },
            "stage3_scoring": {
                "status": scoring_result.status,
                "healthy_score": scoring_result.healthy_score,
                "disease_score": scoring_result.disease_score,
                "confidence": round(scoring_result.confidence, 4),
            },
            "predicted_rag_label": pred_label,
            "predicted_binary": pred_binary,
            "correct": gt_binary == pred_binary,
        })

    metrics = compute_metrics(y_true_binary, y_pred_binary)

    # Scoring agreement
    scoring_agree = sum(
        1 for r in per_image
        if (r["stage3_scoring"]["status"] == "Healthy" and r["ground_truth_binary"] == "Healthy")
        or (r["stage3_scoring"]["status"] == "Disease" and r["ground_truth_binary"] == "Disease")
    ) / max(len(per_image), 1)

    db.delete_collection()

    return {
        "experiment": "EXP-NEW-02", "name": "Binary Detection (Fresh vs Infected)",
        "total_images": len(al_paths),
        "metrics": metrics,
        "scoring_agreement": round(scoring_agree, 4),
        "per_image_results": per_image,
        "total_time_s": round(time.time() - t0, 1),
        "timestamp": datetime.now().isoformat(),
    }


# ════════════════════════════════════════════════════════════
# EXP-NEW-03: Florence-2 Caption Quality (SCA)
# ════════════════════════════════════════════════════════════

def run_exp_new03(train_s1: dict, config: AppConfig) -> dict:
    logger.info("=" * 60)
    logger.info(f"EXP-NEW-03: Caption Quality ({train_s1['count']} images)")
    logger.info("=" * 60)
    t0 = time.time()

    sf_cfg = config.semantic_filter
    fish_keywords = {"fish", "fin", "tail", "gill", "scale", "eye", "body",
                     "swim", "aquatic", "underwater", "marine"}
    disease_keywords = {"disease", "infection", "lesion", "ulcer", "spot",
                        "red", "white", "damaged", "abnormal", "wound",
                        "fungus", "parasite", "bacteria", "virus",
                        "discoloration", "hemorrhage", "necrosis", "inflammation"}

    per_class = {}
    per_image = []

    for img_path, data in train_s1["results"].items():
        caption = data.get("caption", "")
        objects = data.get("objects", [])
        cls = data.get("class_label", "")
        caption_lower = caption.lower()

        # Semantic filter
        sc = classify_scene(caption, sf_cfg)

        # Fish detection in objects
        obj_labels = [o["label"].lower() for o in objects if isinstance(o, dict)]
        fish_in_objects = any("fish" in l for l in obj_labels)

        # Keyword analysis
        fish_kw_found = [kw for kw in fish_keywords if kw in caption_lower]
        disease_kw_found = [kw for kw in disease_keywords if kw in caption_lower]

        per_class.setdefault(cls, {
            "count": 0, "sca_scores": [], "rag_triggered": 0,
            "fish_detected": 0, "fish_kw_counts": [], "disease_kw_counts": [],
            "scene_types": {},
        })
        pc = per_class[cls]
        pc["count"] += 1
        pc["sca_scores"].append(sc.score)
        if sc.rag_triggered:
            pc["rag_triggered"] += 1
        if fish_in_objects:
            pc["fish_detected"] += 1
        pc["fish_kw_counts"].append(len(fish_kw_found))
        pc["disease_kw_counts"].append(len(disease_kw_found))
        pc["scene_types"][sc.scene_type] = pc["scene_types"].get(sc.scene_type, 0) + 1

        per_image.append({
            "path": img_path, "class_label": cls,
            "caption": caption[:200],
            "objects_count": len(objects),
            "fish_in_objects": fish_in_objects,
            "scene_type": sc.scene_type,
            "sca_score": round(sc.score, 4),
            "rag_triggered": sc.rag_triggered,
            "fish_keywords": fish_kw_found,
            "disease_keywords": disease_kw_found,
        })

    # Aggregate per-class
    class_summary = {}
    for cls, pc in sorted(per_class.items()):
        class_summary[cls] = {
            "count": pc["count"],
            "avg_sca_score": round(float(np.mean(pc["sca_scores"])), 4),
            "sca_above_threshold": round(
                sum(1 for s in pc["sca_scores"] if s >= sf_cfg.gate_threshold) / pc["count"], 4),
            "rag_trigger_rate": round(pc["rag_triggered"] / pc["count"], 4),
            "fish_detection_rate": round(pc["fish_detected"] / pc["count"], 4),
            "avg_fish_keywords": round(float(np.mean(pc["fish_kw_counts"])), 2),
            "avg_disease_keywords": round(float(np.mean(pc["disease_kw_counts"])), 2),
            "scene_types": pc["scene_types"],
        }

    overall_sca = float(np.mean([s["avg_sca_score"] for s in class_summary.values()]))
    overall_rag = float(np.mean([s["rag_trigger_rate"] for s in class_summary.values()]))
    overall_fish = float(np.mean([s["fish_detection_rate"] for s in class_summary.values()]))

    return {
        "experiment": "EXP-NEW-03", "name": "Florence-2 Caption Quality (SCA)",
        "total_images": train_s1["count"],
        "overall_avg_sca": round(overall_sca, 4),
        "overall_rag_trigger_rate": round(overall_rag, 4),
        "overall_fish_detection_rate": round(overall_fish, 4),
        "class_summary": class_summary,
        "sample_images": stratified_sample(per_image, "class_label", 3),
        "total_time_s": round(time.time() - t0, 1),
        "timestamp": datetime.now().isoformat(),
    }


# ════════════════════════════════════════════════════════════
# EXP-12: Bidirectional Verification Loop
# ════════════════════════════════════════════════════════════

def run_exp12(
    test_clip: dict, train_clip: dict,
    test_s1: dict, train_s1: dict,
    clip: CLIPWrapper, florence2: Florence2Wrapper, config: AppConfig,
) -> dict:
    logger.info("=" * 60)
    logger.info(f"EXP-12: Verification Loop ({len(test_clip['paths'])} images)")
    logger.info("=" * 60)
    t0 = time.time()

    tr_paths, tr_labels = train_clip["paths"], train_clip["labels"]
    tr_vis = train_clip["embeddings"]
    te_paths, te_labels = test_clip["paths"], test_clip["labels"]
    te_vis = test_clip["embeddings"]

    tr_cap = get_caption_embeddings(train_s1, tr_paths, CACHE_DIR / "cap_sa_train", clip)
    te_cap = get_caption_embeddings(test_s1, te_paths, CACHE_DIR / "cap_sa_test", clip)

    fusion_cfg = FusionConfig(lambda_weight=0.7)
    verification_cfg = config.verification

    tr_fused = np.stack([
        create_fusion_embedding(tr_vis[i], tr_cap[i], fusion_cfg).fused_embedding
        for i in range(len(tr_paths))
    ])
    te_fused = np.stack([
        create_fusion_embedding(te_vis[i], te_cap[i], fusion_cfg).fused_embedding
        for i in range(len(te_paths))
    ])

    documents = [DISEASE_DESCRIPTIONS.get(l, l) for l in tr_labels]
    db = build_experiment_db("exp12", tr_fused, tr_labels, tr_paths, documents)

    grounding_fn = make_grounding_fn(florence2)
    per_image = []
    total_iterations = 0
    total_keywords = 0
    total_grounded = 0
    total_ungrounded = 0
    convergence_count = 0
    penalty_count = 0
    removal_count = 0

    for i in range(len(te_paths)):
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(f"  [EXP-12] {i+1}/{len(te_paths)} ({rate:.2f}/s)")

        rag_results = db.query(te_fused[i], top_k=5)
        rag_items = [
            RAGItem(content=r["content"], score=r["similarity"],
                    source=r["source"], doc_id=r["id"])
            for r in rag_results
        ]

        if not rag_items:
            per_image.append({
                "path": te_paths[i], "ground_truth": te_labels[i],
                "rag_items_count": 0, "verification": None,
            })
            continue

        try:
            ver = run_verification_loop(
                rag_items, te_paths[i], grounding_fn, None, verification_cfg)

            total_iterations += ver.iterations_run
            total_keywords += len(ver.keywords_checked)
            for gr in ver.grounding_results:
                if gr.grounded:
                    total_grounded += 1
                else:
                    total_ungrounded += 1
            if ver.converged:
                convergence_count += 1
            removal_count += ver.items_removed
            penalty_count += sum(
                1 for item in ver.verified_items if item.penalty_applied)

            per_image.append({
                "path": te_paths[i], "ground_truth": te_labels[i],
                "rag_items_count": len(rag_items),
                "verification": {
                    "iterations_run": ver.iterations_run,
                    "keywords_checked": ver.keywords_checked,
                    "grounding_results": [
                        {"keyword": gr.keyword, "grounded": gr.grounded,
                         "confidence": round(gr.confidence, 4),
                         "bbox": [round(b, 1) for b in gr.bbox] if gr.bbox else []}
                        for gr in ver.grounding_results
                    ],
                    "items_removed": ver.items_removed,
                    "converged": ver.converged,
                    "verified_items": [
                        {"source": item.source,
                         "score_after": round(item.score, 4),
                         "verified": item.verified,
                         "penalty_applied": item.penalty_applied}
                        for item in ver.verified_items
                    ],
                },
            })
        except Exception as e:
            logger.warning(f"  Verification error: {e}")
            per_image.append({
                "path": te_paths[i], "ground_truth": te_labels[i],
                "rag_items_count": len(rag_items), "verification": None,
                "error": str(e),
            })

    n_with_ver = sum(1 for r in per_image if r.get("verification"))
    db.delete_collection()

    return {
        "experiment": "EXP-12", "name": "Bidirectional Verification Loop",
        "total_images": len(te_paths),
        "images_with_verification": n_with_ver,
        "avg_iterations": round(total_iterations / max(n_with_ver, 1), 2),
        "total_keywords_checked": total_keywords,
        "avg_keywords_per_image": round(total_keywords / max(n_with_ver, 1), 2),
        "grounded_count": total_grounded,
        "ungrounded_count": total_ungrounded,
        "grounding_rate": round(
            total_grounded / max(total_grounded + total_ungrounded, 1), 4),
        "convergence_rate": round(convergence_count / max(n_with_ver, 1), 4),
        "penalty_applied_count": penalty_count,
        "items_removed_total": removal_count,
        "avg_items_removed": round(removal_count / max(n_with_ver, 1), 2),
        "per_image_results": per_image,
        "total_time_s": round(time.time() - t0, 1),
        "timestamp": datetime.now().isoformat(),
    }


# ════════════════════════════════════════════════════════════
# Phase C — Report Generation
# ════════════════════════════════════════════════════════════

def _img_md(path: str) -> str:
    """Copy image to figures dir and return markdown reference with safe path.

    GitLab cannot display images from uncommitted dataset dirs or paths with
    spaces/parentheses.  We copy each referenced image into
    docs/figures/large_scale_samples/ with a sanitised filename and return a
    relative markdown link that works on GitLab.
    """
    src = Path(path)
    if not src.exists():
        return f"*(image not found: {src.name})*"

    # Build a safe filename: dataset_split_classname_filename
    parts = src.parts
    # Detect dataset + split from path
    try:
        ext_idx = parts.index("external_datasets")
        dataset = parts[ext_idx + 1]          # e.g. fish_disease_south_asia
        remaining = parts[ext_idx + 2:]       # after dataset name
        # Try to find Train/Test/Valid split
        split = ""
        for p in remaining:
            if p.lower() in ("train", "test", "valid"):
                split = p.lower()
                break
        class_name = src.parent.name
    except (ValueError, IndexError):
        dataset = "unknown"
        split = ""
        class_name = src.parent.name

    safe_name = (
        f"{dataset}_{split}_{class_name}_{src.name}"
        .replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
    )
    dest = FIGURES_DIR / safe_name
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(src, dest)

    # Return path relative to docs/ (report lives in docs/)
    return f"![sample](figures/large_scale_samples/{safe_name})"


def generate_report(all_results: dict, report_path: Path) -> None:
    """Generate comprehensive markdown experiment report."""
    logger.info("Generating report...")
    lines = []
    w = lines.append

    w("# Large-Scale Experiment Results")
    w(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"**Total images processed**: ~12,000+")
    w(f"**All model inference**: Real GPU execution (Florence-2 + CLIP)")
    w("**Stage 3**: deterministic evidence-pool scoring (Eqs. 8-11); no free-form LLM used")
    w("")

    # ── Executive Summary ──
    w("## Executive Summary\n")
    w("| Experiment | Images | Key Metric | Value |")
    w("|:-----------|-------:|:-----------|------:|")

    if "exp01" in all_results:
        r = all_results["exp01"]
        w(f"| EXP-01 λ Ablation | {r['total_images']} | "
          f"Best DA (λ={r['best_lambda']}) | {r['best_accuracy']:.4f} |")
    if "exp02" in all_results:
        r = all_results["exp02"]
        w(f"| EXP-02 Pareidolia | {r['total_images']} | "
          f"Disease Trigger Rate | {r['disease_combined_trigger_rate']:.4f} |")
    if "exp03" in all_results:
        r = all_results["exp03"]
        w(f"| EXP-03 Semantic Filter | {r['total_images']} | "
          f"Disease RAG Rate | {r['disease_rag_trigger_rate']:.4f} |")
    if "exp04" in all_results:
        r = all_results["exp04"]
        w(f"| EXP-04 CLIP Separability | {r['total_images']} | "
          f"Sep. Ratio | {r['separability_ratio']:.4f} |")
    if "exp05" in all_results:
        r = all_results["exp05"]
        w(f"| EXP-05 Cross-Dataset | {r['total_images']} | "
          f"DA | {r['metrics']['accuracy']:.4f} |")
    if "exp_new01" in all_results:
        r = all_results["exp_new01"]
        w(f"| EXP-NEW-01 Classification | {r['total_images']} | "
          f"DA | {r['metrics']['accuracy']:.4f} |")
    if "exp_new02" in all_results:
        r = all_results["exp_new02"]
        w(f"| EXP-NEW-02 Binary | {r['total_images']} | "
          f"DA | {r['metrics']['accuracy']:.4f} |")
    if "exp_new03" in all_results:
        r = all_results["exp_new03"]
        w(f"| EXP-NEW-03 Caption Quality | {r['total_images']} | "
          f"Avg SCA | {r['overall_avg_sca']:.4f} |")
    if "exp12" in all_results:
        r = all_results["exp12"]
        w(f"| EXP-12 Verification | {r['total_images']} | "
          f"Convergence Rate | {r['convergence_rate']:.4f} |")
    w("")

    # ── EXP-01 ──
    if "exp01" in all_results:
        r = all_results["exp01"]
        w("---\n## EXP-01: λ-Weighted Fusion Ablation Study\n")
        w(f"- **Training set**: south_asia Train ({r['train_images']} images, 7 classes)")
        w(f"- **Test set**: south_asia Test ({r['test_images']} images, 7 classes)")
        w(f"- **Total**: {r['total_images']} images")
        w(f"- **Equation**: Eq.3-5 (λ-Weighted Fusion)")
        w(f"- **Best λ**: {r['best_lambda']} (DA={r['best_accuracy']:.4f})\n")

        w("### λ vs Accuracy Table\n")
        w("| λ | Accuracy | Correct/Total | Macro-P | Macro-R | Macro-F1 |")
        w("|--:|--------:|--------------:|--------:|--------:|---------:|")
        for lr in r["results"]:
            w(f"| {lr['lambda']:.1f} | {lr['accuracy']:.4f} | "
              f"{lr['correct']}/{lr['total']} | "
              f"{lr.get('macro_precision', 0):.4f} | "
              f"{lr.get('macro_recall', 0):.4f} | "
              f"{lr.get('macro_f1', 0):.4f} |")
        w("")

        # Per-class at best λ
        best_lr = [lr for lr in r["results"] if lr["lambda"] == r["best_lambda"]][0]
        if "class_metrics" in best_lr:
            w(f"### Per-Class Metrics at λ*={r['best_lambda']}\n")
            w("| Class | Precision | Recall | F1 | Support |")
            w("|:------|----------:|-------:|---:|--------:|")
            for cls, m in sorted(best_lr["class_metrics"].items()):
                w(f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | "
                  f"{m['f1']:.4f} | {m['support']} |")
            w("")

        # Sample images
        if r.get("sample_details"):
            w("### Sample Results (λ=0.7)\n")
            for s in r["sample_details"][:14]:
                mark = "✓" if s["correct"] else "✗"
                w(f"- {mark} `{Path(s['path']).name}` GT=**{s['ground_truth']}** "
                  f"→ Pred=**{s['predicted']}** (sim={s['similarity']:.4f})")
            w("")

    # ── EXP-02 ──
    if "exp02" in all_results:
        r = all_results["exp02"]
        w("---\n## EXP-02: Two-Tier Pareidolia Detection\n")
        w(f"- **Disease images**: {r['disease_images']} (south_asia Train)")
        w(f"- **General fish images**: {r['general_images']} (large_scale_fish)")
        w(f"- **Total**: {r['total_images']} images")
        w(f"- **Equation**: Eq.1a, 1b, 1, 2\n")

        w("### Overall Trigger Rates\n")
        w("| Dataset | Hard Trigger | Combined Trigger |")
        w("|:--------|------------:|-----------------:|")
        w(f"| Disease Fish | {r['disease_hard_trigger_rate']:.4f} | "
          f"{r['disease_combined_trigger_rate']:.4f} |")
        w(f"| General Fish | {r['general_hard_trigger_rate']:.4f} | "
          f"{r['general_combined_trigger_rate']:.4f} |")
        w("")

        if r.get("class_breakdown"):
            w("### Per-Class Trigger Rates (Disease Dataset)\n")
            w("| Class | Hard Rate | Combined Rate | Count |")
            w("|:------|----------:|--------------:|------:|")
            for cls, v in r["class_breakdown"].items():
                w(f"| {cls} | {v['hard_rate']:.4f} | {v['combined_rate']:.4f} | {v['count']} |")
            w("")

        # Sample with stage data — one per class
        if r.get("sample_disease"):
            w("### Sample Stage 1 + Pareidolia Results (Disease)\n")
            shown = set()
            for s in r["sample_disease"]:
                if s["class_label"] in shown:
                    continue
                shown.add(s["class_label"])
                w(f"\n**{s['class_label']}** — `{Path(s['path']).name}`")
                w(f"\n{_img_md(s['path'])}\n")
                w(f"- **Caption**: {s['caption'][:150]}")
                w(f"- **Objects**: {s['object_labels']}")
                w(f"- **Hard triggered**: {s['hard_triggered']} (scores: {s['hard_scores']})")
                w(f"- **Combined triggered**: {s['combined_triggered']} "
                  f"(scores: {s['combined_scores']})")
                w(f"- **Remapped**: {s['remapped_labels']}")
            w("")

    # ── EXP-03 ──
    if "exp03" in all_results:
        r = all_results["exp03"]
        w("---\n## EXP-03: Semantic Filter Effectiveness (Eq.8)\n")
        w(f"- **Disease images**: {r['disease_images']} (south_asia Train)")
        w(f"- **General images**: {r['general_images']} (large_scale_fish)")
        w(f"- **Total**: {r['total_images']} images\n")

        w("### RAG Trigger Rates\n")
        w("| Dataset | RAG Trigger Rate |")
        w("|:--------|--:|")
        w(f"| Disease Fish | {r['disease_rag_trigger_rate']:.4f} |")
        w(f"| General Fish | {r['general_rag_trigger_rate']:.4f} |")
        w("")

        if r.get("class_rag_rates"):
            w("### Per-Class RAG Trigger (Disease)\n")
            w("| Class | Trigger Rate | Avg Score | Count |")
            w("|:------|------------:|----------:|------:|")
            for cls, v in r["class_rag_rates"].items():
                w(f"| {cls} | {v['trigger_rate']:.4f} | {v['avg_score']:.4f} | {v['count']} |")
            w("")

        if r.get("general_scene_distribution"):
            w("### Scene Type Distribution (General Fish)\n")
            w("| Scene Type | Count |")
            w("|:-----------|------:|")
            for st, cnt in sorted(r["general_scene_distribution"].items()):
                w(f"| {st} | {cnt} |")
            w("")

        # Samples — one per class
        if r.get("sample_disease"):
            w("### Sample Semantic Filter Results\n")
            shown = set()
            for s in r["sample_disease"]:
                if s["class_label"] in shown:
                    continue
                shown.add(s["class_label"])
                w(f"\n**{s['class_label']}** — `{Path(s['path']).name}`")
                w(f"\n{_img_md(s['path'])}\n")
                w(f"- **Caption**: {s['caption']}")
                w(f"- **Scene**: {s['scene_type']} (score={s['score']:.4f})")
                w(f"- **RAG triggered**: {s['rag_triggered']}")
                w(f"- **Matched keywords**: {s['matched_keywords']}")
            w("")

    # ── EXP-04 ──
    if "exp04" in all_results:
        r = all_results["exp04"]
        w("---\n## EXP-04: CLIP Embedding Separability\n")
        w(f"- **Images**: {r['total_images']} (south_asia Train, {r['num_classes']} classes)")
        w(f"- **Separability Ratio**: {r['separability_ratio']:.4f}")
        w(f"- **Avg Inter-class Distance**: {r['avg_inter_class_distance']:.4f}")
        w(f"- **Avg Intra-class Distance**: {r['avg_intra_class_distance']:.4f}\n")

        w("### Intra-Class Distance Statistics\n")
        w("| Class | Mean | Std | Min | Max | Count |")
        w("|:------|-----:|----:|----:|----:|------:|")
        for cls, v in sorted(r["intra_class_distances"].items()):
            w(f"| {cls} | {v['mean']:.4f} | {v['std']:.4f} | "
              f"{v['min']:.4f} | {v['max']:.4f} | {v['count']} |")
        w("")

        w("### Inter-Class Cosine Distance Matrix\n")
        classes = sorted(r["inter_class_matrix"].keys())
        short = {c: c[:8] for c in classes}
        header = "| | " + " | ".join(short[c] for c in classes) + " |"
        sep = "|:--|" + "|".join("-:" for _ in classes) + "|"
        w(header)
        w(sep)
        for ci in classes:
            row = f"| {short[ci]} |"
            for cj in classes:
                d = r["inter_class_matrix"][ci][cj]
                row += f" {d:.4f} |"
            w(row)
        w("")

    # ── EXP-05 ──
    if "exp05" in all_results:
        r = all_results["exp05"]
        w("---\n## EXP-05: Cross-Dataset Generalization\n")
        w(f"- **Train**: south_asia Train ({r['train_images']} images)")
        w(f"- **Test**: detection train_split ({r['test_images']} images)")
        w(f"- **Total**: {r['total_images']} images")
        w(f"- **λ**: {r['lambda']}\n")
        w(f"- **Overall DA**: {r['metrics']['accuracy']:.4f}")
        w(f"- **Overlap classes DA**: {r['overlap_accuracy']:.4f}")
        w(f"- **Unique class (EUS) DA**: {r['unique_class_accuracy']:.4f}")
        w(f"- **Label overlap**: {r['label_overlap']}")
        w(f"- **Unique to detection**: {r['unique_to_detection']}\n")

        w("### Per-Class Metrics\n")
        w("| Class | Precision | Recall | F1 | Support |")
        w("|:------|----------:|-------:|---:|--------:|")
        for cls, m in sorted(r["metrics"]["class_metrics"].items()):
            w(f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | "
              f"{m['f1']:.4f} | {m['support']} |")
        w("")

    # ── EXP-NEW-01 ──
    if "exp_new01" in all_results:
        r = all_results["exp_new01"]
        w("---\n## EXP-NEW-01: Disease Classification (Full Pipeline)\n")
        w(f"- **Train**: south_asia Train ({r['train_images']} images)")
        w(f"- **Test**: south_asia Test + cleaned Valid ({r['test_images']} images)")
        w(f"- **Total**: {r['total_images']} images")
        w(f"- **Pipeline**: Stage 1 → Semantic Filter → Fusion → RAG → "
          f"Verification → Scoring\n")
        w(f"- **DA**: {r['metrics']['accuracy']:.4f}")
        w(f"- **Macro F1**: {r['metrics']['macro_f1']:.4f}\n")

        w("### Per-Class Metrics\n")
        w("| Class | Precision | Recall | F1 | Support |")
        w("|:------|----------:|-------:|---:|--------:|")
        for cls, m in sorted(r["metrics"]["class_metrics"].items()):
            w(f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | "
              f"{m['f1']:.4f} | {m['support']} |")
        w("")

        w("### Scoring Distribution (Eq.9-11)\n")
        w("| Status | Count | Percentage |")
        w("|:-------|------:|-----------:|")
        total = r['test_images']
        for status, cnt in r["scoring_distribution"].items():
            w(f"| {status} | {cnt} | {cnt/total*100:.1f}% |")
        w("")

        w("### Quality Gate Distribution\n")
        w("| Decision | Count | Percentage |")
        w("|:---------|------:|-----------:|")
        for dec, cnt in r["quality_gate_distribution"].items():
            w(f"| {dec} | {cnt} | {cnt/total*100:.1f}% |")
        w("")

        # Full pipeline sample (3 per class)
        w("### Complete Pipeline Trace (Sample Images)\n")
        shown_classes = set()
        for rec in r.get("per_image_results", []):
            cls = rec["ground_truth"]
            if cls in shown_classes:
                continue
            shown_classes.add(cls)
            if len(shown_classes) > 7:
                break

            mark = "✓" if rec["correct"] else "✗"
            w(f"\n#### {mark} {cls} — `{Path(rec['path']).name}`\n")
            w(f"{_img_md(rec['path'])}\n")

            s1 = rec.get("stage1", {})
            w(f"**Stage 1 — Visual Semantics (Florence-2):**\n")
            w(f"- Caption: *{s1.get('caption', 'N/A')}*")
            w(f"- Objects detected: {len(s1.get('objects', []))}")
            if s1.get("objects"):
                obj_str = ", ".join(o.get("label", "?") for o in s1["objects"][:5])
                w(f"- Labels: {obj_str}")
            w(f"- Latency: {s1.get('latency_ms', 0):.0f}ms\n")

            s2 = rec.get("stage2", {})
            sf = s2.get("semantic_filter", {})
            w(f"**Stage 2 — RAG Retrieval:**\n")
            w(f"- Scene: {sf.get('scene_type', '?')} (score={sf.get('score', 0):.4f})")
            w(f"- RAG triggered: {sf.get('rag_triggered', False)}")
            w(f"- Matched keywords: {sf.get('matched_keywords', [])}")
            if s2.get("rag_results"):
                w(f"- Top RAG results:")
                for rr in s2["rag_results"][:3]:
                    w(f"  - [{rr['source']}] sim={rr['similarity']:.4f}: "
                      f"{rr['content'][:80]}...")
            ver = s2.get("verification")
            if ver:
                w(f"- Verification: {ver.get('iterations_run', 0)} iterations, "
                  f"converged={ver.get('converged', False)}, "
                  f"items_removed={ver.get('items_removed', 0)}")
                if ver.get("grounding_results"):
                    for gr in ver["grounding_results"][:3]:
                        g_mark = "✓" if gr["grounded"] else "✗"
                        w(f"  - {g_mark} `{gr['keyword']}` "
                          f"conf={gr['confidence']:.4f}")
            w("")

            s3 = rec.get("stage3_scoring", {})
            w(f"**Stage 3 — Cognitive Reasoning (Scoring Eq.9-11):**\n")
            w(f"- S_h={s3.get('healthy_score', 0)}, S_d={s3.get('disease_score', 0)}")
            w(f"- Decision: **{s3.get('status', '?')}** "
              f"(confidence={s3.get('confidence', 0):.4f})")
            w(f"- Healthy patterns: {s3.get('matched_healthy', [])}")
            w(f"- Disease patterns: {s3.get('matched_disease', [])}")
            w(f"- Text scored: *{s3.get('scoring_text_used', '')[:100]}...*\n")

            learn = rec.get("learning", {})
            w(f"**Adaptive Learning Layer:**\n")
            w(f"- Quality decision: **{learn.get('quality_decision', '?')}**")
            w(f"- Max score: {learn.get('max_score', 0)}")
            w(f"- LLM available: {learn.get('llm_available', False)}")
            w(f"- Predicted: **{rec['predicted_label']}** | "
              f"GT: **{rec['ground_truth']}** | "
              f"{'Correct ✓' if rec['correct'] else 'Wrong ✗'}")
            w("")

    # ── EXP-NEW-02 ──
    if "exp_new02" in all_results:
        r = all_results["exp_new02"]
        w("---\n## EXP-NEW-02: Binary Detection (Fresh vs Infected)\n")
        w(f"- **Dataset**: fish_disease_alaa ({r['total_images']} images)")
        w(f"- **DA**: {r['metrics']['accuracy']:.4f}")
        w(f"- **Scoring Agreement**: {r['scoring_agreement']:.4f}\n")

        w("### Per-Class Metrics\n")
        w("| Class | Precision | Recall | F1 | Support |")
        w("|:------|----------:|-------:|---:|--------:|")
        for cls, m in sorted(r["metrics"]["class_metrics"].items()):
            w(f"| {cls} | {m['precision']:.4f} | {m['recall']:.4f} | "
              f"{m['f1']:.4f} | {m['support']} |")
        w("")

        # Samples with images — one per class
        w("### Sample Results with Pipeline Trace\n")
        shown_cls = set()
        for rec in r.get("per_image_results", []):
            gt_orig = rec.get("ground_truth_original", "")
            if gt_orig in shown_cls:
                continue
            shown_cls.add(gt_orig)
            if len(shown_cls) > 4:
                break
            mark = "✓" if rec["correct"] else "✗"
            w(f"\n{mark} **{gt_orig}** → "
              f"Pred: **{rec['predicted_binary']}** — "
              f"`{Path(rec['path']).name}`\n")
            w(f"{_img_md(rec['path'])}\n")
            w(f"- Stage 1 caption: *{rec['stage1']['caption'][:150]}*")
            s2 = rec["stage2"]
            w(f"- Stage 2: scene={s2['scene_type']}, score={s2['score']:.4f}, "
              f"RAG={s2['rag_triggered']}")
            if s2.get("top3_rag"):
                for rr in s2["top3_rag"]:
                    w(f"  - [{rr['source']}] sim={rr['sim']:.4f}")
            s3 = rec["stage3_scoring"]
            w(f"- Stage 3: S_h={s3['healthy_score']}, S_d={s3['disease_score']} "
              f"→ **{s3['status']}** (conf={s3['confidence']:.4f})")
            w("")

    # ── EXP-NEW-03 ──
    if "exp_new03" in all_results:
        r = all_results["exp_new03"]
        w("---\n## EXP-NEW-03: Florence-2 Caption Quality (SCA)\n")
        w(f"- **Dataset**: south_asia Train ({r['total_images']} images)")
        w(f"- **Overall SCA**: {r['overall_avg_sca']:.4f}")
        w(f"- **RAG Trigger Rate**: {r['overall_rag_trigger_rate']:.4f}")
        w(f"- **Fish Detection Rate**: {r['overall_fish_detection_rate']:.4f}\n")

        w("### Per-Class Caption Quality\n")
        w("| Class | SCA Score | SCA≥τ Rate | RAG Rate | Fish Det. | "
          "Avg Fish KW | Avg Disease KW | Count |")
        w("|:------|----------:|-----------:|---------:|----------:|"
          "-----------:|---------------:|------:|")
        for cls, v in sorted(r["class_summary"].items()):
            w(f"| {cls} | {v['avg_sca_score']:.4f} | "
              f"{v['sca_above_threshold']:.4f} | "
              f"{v['rag_trigger_rate']:.4f} | "
              f"{v['fish_detection_rate']:.4f} | "
              f"{v['avg_fish_keywords']:.1f} | "
              f"{v['avg_disease_keywords']:.1f} | {v['count']} |")
        w("")

        # Samples
        if r.get("sample_images"):
            w("### Sample Florence-2 Outputs\n")
            shown = set()
            for s in r["sample_images"]:
                if s["class_label"] in shown:
                    continue
                shown.add(s["class_label"])
                w(f"\n**{s['class_label']}** — `{Path(s['path']).name}`\n")
                w(f"{_img_md(s['path'])}\n")
                w(f"- Caption: *{s['caption']}*")
                w(f"- Objects: {s['objects_count']}, Fish detected: {s['fish_in_objects']}")
                w(f"- Scene: {s['scene_type']} (SCA={s['sca_score']:.4f})")
                w(f"- Fish keywords: {s['fish_keywords']}")
                w(f"- Disease keywords: {s['disease_keywords']}")
                w(f"- RAG triggered: {s['rag_triggered']}")
            w("")

    # ── EXP-12 ──
    if "exp12" in all_results:
        r = all_results["exp12"]
        w("---\n## EXP-12: Bidirectional Verification Loop (Algorithm 1)\n")
        w(f"- **Dataset**: south_asia Test ({r['total_images']} images)")
        w(f"- **Images with verification**: {r['images_with_verification']}\n")

        w("### Verification Statistics\n")
        w("| Metric | Value |")
        w("|:-------|------:|")
        w(f"| Avg iterations per image | {r['avg_iterations']:.2f} |")
        w(f"| Total keywords checked | {r['total_keywords_checked']} |")
        w(f"| Avg keywords per image | {r['avg_keywords_per_image']:.2f} |")
        w(f"| Grounded keywords | {r['grounded_count']} |")
        w(f"| Ungrounded keywords | {r['ungrounded_count']} |")
        w(f"| Grounding success rate | {r['grounding_rate']:.4f} |")
        w(f"| Convergence rate | {r['convergence_rate']:.4f} |")
        w(f"| Total items removed | {r['items_removed_total']} |")
        w(f"| Avg items removed/image | {r['avg_items_removed']:.2f} |")
        w(f"| Penalty applied count | {r['penalty_applied_count']} |")
        w("")

        # Samples — one per class for diversity
        w("### Sample Verification Traces\n")
        shown_cls = set()
        for rec in r.get("per_image_results", []):
            if not rec.get("verification") or len(shown_cls) >= 7:
                continue
            ver = rec["verification"]
            if not ver.get("grounding_results"):
                continue
            gt = rec["ground_truth"]
            if gt in shown_cls:
                continue
            shown_cls.add(gt)
            w(f"\n**{gt}** — `{Path(rec['path']).name}`\n")
            w(f"{_img_md(rec['path'])}\n")
            w(f"- RAG items: {rec['rag_items_count']}")
            w(f"- Iterations: {ver['iterations_run']}")
            w(f"- Keywords: {ver['keywords_checked']}")
            w(f"- Converged: {ver['converged']}")
            w(f"- Items removed: {ver['items_removed']}")
            w(f"- Grounding results:")
            for gr in ver["grounding_results"]:
                g_mark = "✓" if gr["grounded"] else "✗"
                bbox_str = f" bbox={gr['bbox']}" if gr.get("bbox") else ""
                w(f"  - {g_mark} `{gr['keyword']}` conf={gr['confidence']:.4f}{bbox_str}")
            if ver.get("verified_items"):
                w(f"- Verified items:")
                for vi in ver["verified_items"][:3]:
                    p_mark = " [PENALIZED]" if vi["penalty_applied"] else ""
                    w(f"  - [{vi['source']}] score={vi['score_after']:.4f} "
                      f"verified={vi['verified']}{p_mark}")
        w("")

    # ── Footer ──
    w("\n---\n## Methodology Notes\n")
    w("1. **All Florence-2 and CLIP inference is real GPU execution** — "
      "no simulated or synthetic embeddings")
    w("2. **Stage 3** uses deterministic evidence-pool scoring (Eqs. 8-11); "
      "no free-form language-model generation is used in the inference path")
    w("3. **Scoring (Eq.9-11) operates on RAG-retrieved disease descriptions** — "
      "real keyword matching and negation detection")
    w("4. **Quality gate uses real scoring output** — LLM verification flag "
      "set to False due to nested session limitation")
    w("5. **All per-image data preserved** in JSON files under "
      "`lab_dateset/organized/experiment_results/large_scale/`")
    w("6. **Cache files** stored in "
      "`lab_dateset/organized/experiment_results/large_scale_cache/`")
    w("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Report saved: {report_path}")


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main() -> None:
    logger.info("=" * 60)
    logger.info("Large-Scale Experiments — Starting")
    logger.info("=" * 60)

    config = load_config()
    t_global = time.time()

    # Initialize models
    logger.info("Loading Florence-2...")
    florence2 = Florence2Wrapper(config.models.florence2)
    logger.info("Loading CLIP...")
    clip_model = CLIPWrapper(config.models.clip)

    # ── Scan Datasets ──
    logger.info("\n=== Scanning Datasets ===")
    sa_train = scan_class_dataset(DS_SOUTH_ASIA / "Train")
    sa_test = scan_class_dataset(DS_SOUTH_ASIA / "Test")
    det_train = scan_class_dataset(DS_DETECTION / "train_split")
    cl_valid = scan_class_dataset(DS_CLEANED / "Valid")
    alaa_all = scan_class_dataset(DS_ALAA)
    lsf_sampled = scan_large_scale_fish(DS_LARGE_SCALE, max_per_species=167)

    logger.info(f"\nDataset summary:")
    logger.info(f"  south_asia Train: {len(sa_train)}")
    logger.info(f"  south_asia Test:  {len(sa_test)}")
    logger.info(f"  detection Train:  {len(det_train)}")
    logger.info(f"  cleaned Valid:    {len(cl_valid)}")
    logger.info(f"  alaa:             {len(alaa_all)}")
    logger.info(f"  large_scale_fish: {len(lsf_sampled)}")
    total_imgs = (len(sa_train) + len(sa_test) + len(det_train) +
                  len(cl_valid) + len(alaa_all) + len(lsf_sampled))
    logger.info(f"  TOTAL: {total_imgs}")

    # ── Phase A: Data Collection ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase A: Data Collection (Model Inference)")
    logger.info("=" * 60)

    # Florence-2 batches
    sa_train_s1 = collect_florence2_batch(
        "sa_train", sa_train, CACHE_DIR / "s1_sa_train.json", florence2)
    sa_test_s1 = collect_florence2_batch(
        "sa_test", sa_test, CACHE_DIR / "s1_sa_test.json", florence2)
    det_train_s1 = collect_florence2_batch(
        "det_train", det_train, CACHE_DIR / "s1_det_train.json", florence2)
    lsf_s1 = collect_florence2_batch(
        "lsf", lsf_sampled, CACHE_DIR / "s1_lsf.json", florence2)
    cl_valid_s1 = collect_florence2_batch(
        "cl_valid", cl_valid, CACHE_DIR / "s1_cl_valid.json", florence2)
    alaa_s1 = collect_florence2_batch(
        "alaa", alaa_all, CACHE_DIR / "s1_alaa.json", florence2)

    # CLIP image batches
    sa_train_clip = collect_clip_batch(
        "sa_train", sa_train, CACHE_DIR / "clip_sa_train", clip_model)
    sa_test_clip = collect_clip_batch(
        "sa_test", sa_test, CACHE_DIR / "clip_sa_test", clip_model)
    det_train_clip = collect_clip_batch(
        "det_train", det_train, CACHE_DIR / "clip_det_train", clip_model)
    lsf_clip = collect_clip_batch(
        "lsf", lsf_sampled, CACHE_DIR / "clip_lsf", clip_model)
    cl_valid_clip = collect_clip_batch(
        "cl_valid", cl_valid, CACHE_DIR / "clip_cl_valid", clip_model)
    alaa_clip = collect_clip_batch(
        "alaa", alaa_all, CACHE_DIR / "clip_alaa", clip_model)

    logger.info(f"\nPhase A complete: {time.time() - t_global:.0f}s")

    # ── Copy sample images for report ──
    copy_sample_images(sa_train[:21], n_per_class=3)
    copy_sample_images(alaa_all[:6], n_per_class=3)

    # ── Phase B: Run Experiments ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase B: Running 9 Experiments")
    logger.info("=" * 60)

    all_results = {}

    # EXP-01
    all_results["exp01"] = run_exp01(
        sa_train_clip, sa_test_clip, sa_train_s1, sa_test_s1, clip_model, config)
    save_result(RESULTS_DIR / "exp01_lambda_ablation.json", all_results["exp01"])

    # EXP-02
    all_results["exp02"] = run_exp02(sa_train_s1, lsf_s1, clip_model, config)
    save_result(RESULTS_DIR / "exp02_pareidolia.json", all_results["exp02"])

    # EXP-03
    all_results["exp03"] = run_exp03(sa_train_s1, lsf_s1, config)
    save_result(RESULTS_DIR / "exp03_semantic_filter.json", all_results["exp03"])

    # EXP-04
    all_results["exp04"] = run_exp04(sa_train_clip)
    save_result(RESULTS_DIR / "exp04_clip_separability.json", all_results["exp04"])

    # EXP-05
    all_results["exp05"] = run_exp05(
        sa_train_clip, det_train_clip, sa_train_s1, det_train_s1, clip_model, config)
    save_result(RESULTS_DIR / "exp05_cross_dataset.json", all_results["exp05"])

    # EXP-NEW-01
    all_results["exp_new01"] = run_exp_new01(
        sa_train_clip, sa_test_clip, cl_valid_clip,
        sa_train_s1, sa_test_s1, cl_valid_s1,
        clip_model, florence2, config)
    save_result(RESULTS_DIR / "exp_new01_classification.json", all_results["exp_new01"])

    # EXP-NEW-02
    all_results["exp_new02"] = run_exp_new02(
        sa_train_clip, alaa_clip, sa_train_s1, alaa_s1, clip_model, config)
    save_result(RESULTS_DIR / "exp_new02_binary.json", all_results["exp_new02"])

    # EXP-NEW-03
    all_results["exp_new03"] = run_exp_new03(sa_train_s1, config)
    save_result(RESULTS_DIR / "exp_new03_caption_quality.json", all_results["exp_new03"])

    # EXP-12
    all_results["exp12"] = run_exp12(
        sa_test_clip, sa_train_clip, sa_test_s1, sa_train_s1,
        clip_model, florence2, config)
    save_result(RESULTS_DIR / "exp12_verification.json", all_results["exp12"])

    logger.info(f"\nPhase B complete: {time.time() - t_global:.0f}s")

    # ── Phase C: Report ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase C: Generating Report")
    logger.info("=" * 60)

    generate_report(all_results, REPORT_FILE)

    total_time = time.time() - t_global
    logger.info(f"\n{'=' * 60}")
    logger.info(f"ALL EXPERIMENTS COMPLETE")
    logger.info(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.info(f"Results: {RESULTS_DIR}")
    logger.info(f"Report: {REPORT_FILE}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
