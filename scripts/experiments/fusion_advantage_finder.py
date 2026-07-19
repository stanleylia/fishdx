#!/usr/bin/env python3
"""
Find test images where lambda=0.7 (fusion) correctly classifies but
lambda=0.0 (caption-only) or lambda=1.0 (visual-only) fails.

Demonstrates the value of the fusion embedding from Eq.3/4/5/7 in the paper.

Key finding: Florence-2 generates identical captions for visually distinct
fish diseases, making caption-only embeddings unable to distinguish classes.
The visual component in the fusion embedding breaks these ties.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

# Override via environment variable FISHDX_CACHE_DIR; defaults to the
# repository's lab_dateset/organized/experiment_results/large_scale_cache
# directory (resolved relative to this file, two levels up to the repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(
    os.environ.get(
        "FISHDX_CACHE_DIR",
        PROJECT_ROOT / "lab_dateset" / "organized" / "experiment_results" / "large_scale_cache",
    )
)

TOP_K = 3  # Show top-K nearest neighbors for each lambda


def load_data():
    """Load all cached embeddings and metadata."""
    with open(CACHE_DIR / "clip_sa_train.json") as f:
        train_meta = json.load(f)
    with open(CACHE_DIR / "clip_sa_test.json") as f:
        test_meta = json.load(f)

    # Florence-2 captions
    with open(CACHE_DIR / "s1_sa_test.json") as f:
        test_captions = json.load(f)
    with open(CACHE_DIR / "s1_sa_train.json") as f:
        train_captions = json.load(f)

    # Embeddings
    train_visual = np.load(CACHE_DIR / "clip_sa_train.npz")["embeddings"]   # (1747, 512)
    test_visual = np.load(CACHE_DIR / "clip_sa_test.npz")["embeddings"]     # (697, 512)
    train_caption = np.load(CACHE_DIR / "cap_sa_train.npz")["embeddings"]   # (1747, 512)
    test_caption = np.load(CACHE_DIR / "cap_sa_test.npz")["embeddings"]     # (697, 512)

    return (
        train_meta, test_meta,
        test_captions, train_captions,
        train_visual, test_visual,
        train_caption, test_caption,
    )


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize each row."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return x / norms


def fuse_embeddings(visual: np.ndarray, caption: np.ndarray, lam: float) -> np.ndarray:
    """Compute L2-normalized fused embedding: lam * vis_norm + (1-lam) * cap_norm."""
    v_norm = l2_normalize(visual)
    c_norm = l2_normalize(caption)
    fused = lam * v_norm + (1.0 - lam) * c_norm
    return l2_normalize(fused)


def classify_nn(
    test_fused: np.ndarray,
    train_fused: np.ndarray,
    train_labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sim_matrix, nn_indices) for nearest-neighbor classification."""
    sim_matrix = test_fused @ train_fused.T  # (n_test, n_train)
    nn_indices = np.argmax(sim_matrix, axis=1)
    return sim_matrix, nn_indices


def analyze_caption_duplicates(
    train_caption: np.ndarray,
    train_labels: list[str],
    train_paths: list[str],
    train_captions_data: dict,
):
    """Analyze cross-class duplicate caption embeddings in the training set."""
    tc_norm = l2_normalize(train_caption)
    sim_mat = tc_norm @ tc_norm.T
    np.fill_diagonal(sim_mat, 0)

    # Find cross-class duplicates (cosine sim > 0.9999)
    threshold = 0.9999
    pairs = np.argwhere(sim_mat > threshold)
    cross_class = []
    for i, j in pairs:
        if i < j and train_labels[i] != train_labels[j]:
            cross_class.append((i, j, sim_mat[i, j]))

    print("=" * 100)
    print("CAPTION EMBEDDING ANALYSIS: Cross-Class Duplicates in Training Set")
    print("=" * 100)
    print(f"Total near-duplicate pairs (cosine > {threshold}): {len(pairs) // 2}")
    print(f"Cross-class near-duplicate pairs: {len(cross_class)}")
    print()

    train_results = train_captions_data.get("results", {})
    for i, j, sim in cross_class:
        cap_i = train_results.get(train_paths[i], {}).get("caption", "N/A")
        cap_j = train_results.get(train_paths[j], {}).get("caption", "N/A")
        identical = np.allclose(train_caption[i], train_caption[j], atol=1e-8)
        print(f"  [{i}] {train_labels[i]}")
        print(f"    Caption: {cap_i[:120]}...")
        print(f"  [{j}] {train_labels[j]}")
        print(f"    Caption: {cap_j[:120]}...")
        print(f"    Cosine similarity: {sim:.10f}, byte-identical: {identical}")
        print(f"    Captions identical: {cap_i == cap_j}")
        print()

    return cross_class


def main():
    print("Loading cached data...")
    (
        train_meta, test_meta,
        test_captions_data, train_captions_data,
        train_visual, test_visual,
        train_caption, test_caption,
    ) = load_data()

    train_labels = train_meta["labels"]
    test_labels = test_meta["labels"]
    test_paths = test_meta["paths"]
    train_paths = train_meta["paths"]
    n_test = len(test_labels)
    n_train = len(train_labels)

    print(f"Train: {n_train} images, Test: {n_test} images")
    print(f"Classes ({len(set(train_labels))}): {sorted(set(train_labels))}")
    print()

    # -------------------------------------------------------------------
    # Phase 1: Cross-class caption duplicate analysis
    # -------------------------------------------------------------------
    cross_class_dups = analyze_caption_duplicates(
        train_caption, train_labels, train_paths, train_captions_data
    )

    # -------------------------------------------------------------------
    # Phase 2: Classify at lambda = 0.0, 0.7, 1.0
    # -------------------------------------------------------------------
    lambdas = [0.0, 0.7, 1.0]
    results = {}

    for lam in lambdas:
        test_fused = fuse_embeddings(test_visual, test_caption, lam)
        train_fused = fuse_embeddings(train_visual, train_caption, lam)
        sim_matrix, nn_indices = classify_nn(test_fused, train_fused, train_labels)
        nn_sims = sim_matrix[np.arange(n_test), nn_indices]
        preds = [train_labels[idx] for idx in nn_indices]
        correct = sum(1 for p, t in zip(preds, test_labels) if p == t)
        acc = correct / n_test * 100
        results[lam] = {
            "preds": preds,
            "sims": nn_sims,
            "nn_idxs": nn_indices,
            "sim_matrix": sim_matrix,
        }
        print(f"lambda={lam:.1f}: accuracy={acc:.2f}% ({correct}/{n_test})")

    print()

    # -------------------------------------------------------------------
    # Phase 3: Find fusion-advantage images
    # -------------------------------------------------------------------
    fusion_advantage = []
    both_fail = []
    only_caption_fails = []
    only_visual_fails = []

    for i in range(n_test):
        gt = test_labels[i]
        c07 = results[0.7]["preds"][i] == gt
        c00 = results[0.0]["preds"][i] == gt
        c10 = results[1.0]["preds"][i] == gt

        if c07 and (not c00 or not c10):
            entry = {
                "index": i,
                "path": test_paths[i],
                "ground_truth": gt,
            }
            for lam in lambdas:
                lk = f"lam{lam}"
                entry[f"{lk}_pred"] = results[lam]["preds"][i]
                entry[f"{lk}_sim"] = float(results[lam]["sims"][i])
                entry[f"{lk}_correct"] = results[lam]["preds"][i] == gt
                entry[f"{lk}_nn_idx"] = int(results[lam]["nn_idxs"][i])

            fusion_advantage.append(entry)
            if not c00 and not c10:
                both_fail.append(entry)
            elif not c00 and c10:
                only_caption_fails.append(entry)
            elif c00 and not c10:
                only_visual_fails.append(entry)

    print("=" * 100)
    print(f"FUSION ADVANTAGE IMAGES: {len(fusion_advantage)} total")
    print(f"  Both lambda=0.0 AND lambda=1.0 fail, fusion succeeds: {len(both_fail)}")
    print(f"  Only lambda=0.0 (caption-only) fails:                 {len(only_caption_fails)}")
    print(f"  Only lambda=1.0 (visual-only) fails:                  {len(only_visual_fails)}")
    print("=" * 100)
    print()

    # Caption lookup
    test_cap_results = test_captions_data.get("results", {})
    train_cap_results = train_captions_data.get("results", {})

    # -------------------------------------------------------------------
    # Phase 4: Print all fusion-advantage images with detailed analysis
    # -------------------------------------------------------------------

    def print_section(title: str, entries: list):
        if not entries:
            return
        print()
        print("=" * 100)
        print(title)
        print("=" * 100)

        for entry in entries:
            path = entry["path"]
            gt = entry["ground_truth"]
            idx = entry["index"]
            print()
            print(f"  TEST IMAGE [{idx}]")
            print(f"  Path:         ...{path[-80:]}")
            print(f"  Ground Truth: {gt}")

            # Florence-2 caption
            cap_info = test_cap_results.get(path, {})
            caption = cap_info.get("caption", "(no caption found)")
            print(f"  Florence-2 Caption: {caption}")
            print()

            # Per-lambda results with top-K neighbors
            for lam in lambdas:
                lk = f"lam{lam}"
                pred = entry[f"{lk}_pred"]
                sim = entry[f"{lk}_sim"]
                correct = entry[f"{lk}_correct"]
                nn_idx = entry[f"{lk}_nn_idx"]
                status = "CORRECT" if correct else "** WRONG **"

                print(f"    lambda={lam:.1f}: pred={pred} [{status}]")
                print(f"      NN match: train[{nn_idx}], sim={sim:.8f}")
                print(f"      NN path:  ...{train_paths[nn_idx][-70:]}")

                # Show top-K neighbors from sim_matrix
                sim_row = results[lam]["sim_matrix"][idx]
                top_k = np.argsort(sim_row)[-TOP_K:][::-1]
                print(f"      Top-{TOP_K} neighbors:")
                for rank, ti in enumerate(top_k, 1):
                    t_label = train_labels[ti]
                    t_sim = sim_row[ti]
                    match = "==" if t_label == gt else "!="
                    t_cap = train_cap_results.get(train_paths[ti], {}).get("caption", "")
                    print(
                        f"        #{rank}: train[{ti}] {t_label} "
                        f"(sim={t_sim:.8f}) {match} GT"
                    )
                    if t_cap:
                        print(f"             Caption: {t_cap[:100]}...")
                print()

            # Explanation
            if not entry["lam0.0_correct"]:
                wrong_nn = entry["lam0.0_nn_idx"]
                test_cap_emb = train_caption[wrong_nn]  # compare with test caption
                test_emb = test_caption[idx]  # renamed for clarity
                cap_sim = float(
                    np.dot(test_emb, test_cap_emb)
                    / (np.linalg.norm(test_emb) * np.linalg.norm(test_cap_emb) + 1e-10)
                )
                identical = bool(np.allclose(test_emb, test_cap_emb, atol=1e-8))
                print(
                    f"    ROOT CAUSE: Test caption embedding is "
                    f"{'IDENTICAL to' if identical else f'very similar (cos={cap_sim:.6f}) to'} "
                    f"wrong-class train[{wrong_nn}] ({train_labels[wrong_nn]})"
                )
                # Also check if it is identical to the correct-class match
                correct_nn = entry["lam0.7_nn_idx"]
                correct_cap_emb = train_caption[correct_nn]
                correct_identical = bool(np.allclose(test_emb, correct_cap_emb, atol=1e-8))
                if correct_identical:
                    print(
                        f"    DETAIL:     Also IDENTICAL to correct-class train[{correct_nn}] "
                        f"({train_labels[correct_nn]}) -- pure tie-breaking issue"
                    )
                    print(
                        "    CONCLUSION: Caption alone cannot distinguish these two diseases. "
                        "Visual embedding breaks the tie correctly."
                    )

            print(f"  {'─' * 96}")

    # Print each category
    if both_fail:
        print_section(
            "STRONGEST: Both caption-only AND visual-only FAIL, fusion SUCCEEDS",
            both_fail,
        )

    if only_caption_fails:
        print_section(
            "Caption-only (lambda=0.0) FAILS, visual-only and fusion SUCCEED",
            only_caption_fails,
        )

    if only_visual_fails:
        print_section(
            "Visual-only (lambda=1.0) FAILS, caption-only and fusion SUCCEED",
            only_visual_fails,
        )

    # -------------------------------------------------------------------
    # Phase 5: Also check for cases where fusion fails but unimodal succeeds
    #          (to show any trade-off)
    # -------------------------------------------------------------------
    fusion_disadvantage = []
    for i in range(n_test):
        gt = test_labels[i]
        c07 = results[0.7]["preds"][i] == gt
        c00 = results[0.0]["preds"][i] == gt
        c10 = results[1.0]["preds"][i] == gt
        if not c07 and (c00 or c10):
            fusion_disadvantage.append(i)

    print()
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    for lam in lambdas:
        correct = sum(1 for i in range(n_test) if results[lam]["preds"][i] == test_labels[i])
        print(f"  lambda={lam:.1f}: {correct}/{n_test} = {correct / n_test * 100:.2f}%")
    print()
    print(f"  Fusion advantage cases (fusion correct, unimodal fails): {len(fusion_advantage)}")
    print(f"  Fusion disadvantage cases (fusion fails, unimodal correct): {len(fusion_disadvantage)}")
    print()

    if fusion_advantage:
        gt_counts = Counter(e["ground_truth"] for e in fusion_advantage)
        print("  Class distribution of fusion-advantage cases:")
        for cls, cnt in gt_counts.most_common():
            print(f"    {cls}: {cnt}")
    print()

    # Cross-class caption duplication summary
    print(f"  Cross-class identical caption pairs in training set: {len(cross_class_dups)}")
    print("  This causes caption-only (lambda=0.0) to fail via nearest-neighbor tie-breaking.")
    print("  Fusion with visual embeddings (lambda=0.7) breaks these ties correctly,")
    print("  demonstrating the value of the lambda-weighted fusion from Eq.3-7.")


if __name__ == "__main__":
    main()
