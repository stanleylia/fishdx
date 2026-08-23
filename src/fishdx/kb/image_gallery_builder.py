"""Image-gallery builder — ADR-0012d + Pattern G Layer 4.

Constructs the ChromaDB ``image_gallery`` collection from a set of reference
images per the paper §III-E architecture (1,639 encoded D1 Train fused embeddings
at λ=0.7 fixed; embedding_source_type="fused"; seed=42 determinism).

This module is **only** responsible for gallery construction; Layer 4
pre-validation (CLIP kNN k=1 on the test set) is orchestrated by the
P'-3 runner script that calls :func:`build_image_gallery`.

Pipeline per image (paper Eq. 5-7):
    Stage 1 Florence-2 caption + Pareidolia → caption_clean
    CLIP image encode → E_visual (L2-normalised by encoder)
    CLIP text encode(caption_clean or " ") → E_caption (L2-normalised)
    fusion : λ·E_visual + (1-λ)·E_caption  (Eq. 6, λ=0.7 fixed)
    L2 post-normalise → E_final  (Eq. 7)

ADR-0008 sub-budget: VRAM peak ≤ 6 GB sustained; ADR-0012d T3 triggers halt
if exceeded during gallery construction. Batched processing with a progress
counter logged every ~10% of the input enables operator inspection without
flooding logs.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from fishdx.errors import KBIngestError, ReproducibilityError
from fishdx.perception.postprocess import clean_caption
from fishdx.retrieval.fusion import fuse_and_normalize

if TYPE_CHECKING:  # pragma: no cover
    from fishdx.perception.florence2 import Florence2Wrapper
    from fishdx.retrieval.clip_encoder import OpenClipEmbedder


_VRAM_BUDGET_GB = 6.0
_LOG_EVERY_PCT = 10


@dataclass(frozen=True)
class GalleryImage:
    """Input specification for a single gallery image."""

    image_id: str
    image_path: Path
    true_class: str
    doc_id: str


@dataclass(frozen=True)
class GalleryReceipt:
    """Summary produced by :func:`build_image_gallery`."""

    collection_name: str
    n_images_requested: int
    n_images_ingested: int
    n_caption_empty: int
    vram_peak_mb: float
    duration_seconds: float
    fusion_lambda: float
    created_under_adr: str
    per_class_counts: dict[str, int]


def _vram_peak_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def _reset_vram_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def build_image_gallery(
    images: Iterable[GalleryImage],
    *,
    florence: Florence2Wrapper,
    clip: OpenClipEmbedder,
    chroma_client: Any,
    collection_name: str = "image_gallery",
    fusion_lambda: float = 0.7,
    created_under_adr: str = "0012d",
    seed: int = 42,
    batch_size: int = 16,
    verbose: bool = True,
) -> GalleryReceipt:
    """Build the ``image_gallery`` ChromaCollection per ADR-0012d.

    Parameters
    ----------
    images : Iterable[GalleryImage]
        Reference images (1,639 encoded D1 Train per paper §IV-A Table 2). Each
        must carry ``true_class`` (folder name) and ``doc_id`` (paper's
        7-class canonical label).
    florence, clip
        Warmed-up wrappers; pipeline deterministic per ADR-0001 / seed=42.
    chroma_client
        Either a :class:`chromadb.PersistentClient` or
        :class:`chromadb.EphemeralClient`; caller owns persistence path.
    collection_name
        ADR-0012d value ``"image_gallery"``; caller may override for
        experimental reruns.
    fusion_lambda
        Paper Table IX λ* = 0.7 fixed at gallery construction time.
    created_under_adr
        Governance pointer stored on the collection metadata.
    seed
        Passed through for documentation; Stage 1 beam-search is
        already deterministic per ADR-0001.
    batch_size
        CLIP batched encode; 16 is the conservative default on RTX 3070.
    verbose
        Logs per-10 % progress when True.

    Returns
    -------
    GalleryReceipt
        Summary statistics (counts, VRAM peak, runtime) for the run.
    """
    image_list = list(images)
    n_total = len(image_list)
    if n_total == 0:
        raise KBIngestError("image_gallery build received empty image list")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _reset_vram_peak(device)
    t_start = time.perf_counter()

    collection_metadata = {
        "embedding_source_type": "fused",
        "fusion_lambda": float(fusion_lambda),
        "n_images_expected": n_total,
        "created_under_adr": created_under_adr,
        "seed": int(seed),
        "clip_architecture": clip._architecture,  # noqa: SLF001
        "clip_pretrained": clip._pretrained,  # noqa: SLF001
        "hnsw:space": "cosine",
        "hnsw:M": 16,
        "hnsw:construction_ef": 200,
        "hnsw:search_ef": 100,
        "hnsw:num_threads": 1,
    }
    collection = chroma_client.get_or_create_collection(
        name=collection_name, metadata=collection_metadata
    )

    n_caption_empty = 0
    per_class: dict[str, int] = {}
    log_step = max(1, n_total * _LOG_EVERY_PCT // 100)

    def _ingest_batch(batch: list[tuple[GalleryImage, str]]) -> None:
        nonlocal n_caption_empty
        # CLIP image encode and text encode per batch.
        image_paths = [str(item.image_path) for item, _ in batch]
        captions = [cap if cap else " " for _, cap in batch]
        e_visuals = clip.encode_image_paths(image_paths)
        e_captions = clip.encode_text(captions)
        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        for (item, cap), e_vis, e_cap in zip(batch, e_visuals, e_captions):  # noqa: B905
            e_final = fuse_and_normalize(e_vis, e_cap, fusion_lambda)
            ids.append(item.image_id)
            embeddings.append(e_final)
            metadatas.append(
                {
                    "true_class": item.true_class,
                    "doc_id": item.doc_id,
                    "image_path_relative": str(item.image_path),
                    "caption_tokens": len(cap.split()),
                    "caption_non_empty": bool(cap.strip()),
                    "fusion_lambda": float(fusion_lambda),
                    "embedding_source_type": "fused",
                }
            )
            if not cap.strip():
                n_caption_empty += 1
            per_class[item.true_class] = per_class.get(item.true_class, 0) + 1
        collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    current_batch: list[tuple[GalleryImage, str]] = []
    for idx, item in enumerate(image_list):
        raw_caption, _objs, _telem = florence.caption_and_detect(item.image_path)
        cap = clean_caption(raw_caption)
        current_batch.append((item, cap))
        if len(current_batch) >= batch_size:
            _ingest_batch(current_batch)
            current_batch = []
        if verbose and ((idx + 1) % log_step == 0 or idx + 1 == n_total):
            pct = (idx + 1) * 100 / n_total
            vram = _vram_peak_mb(device)
            print(
                f"  [{idx + 1}/{n_total} · {pct:4.0f}%] "
                f"VRAM peak {vram:6.0f} MB · caption_empty_so_far {n_caption_empty}"
            )
            if vram / 1024 > _VRAM_BUDGET_GB:
                raise ReproducibilityError(
                    f"VRAM peak {vram:.0f} MB > ADR-0008 sub-budget {_VRAM_BUDGET_GB} GB",
                    context={"vram_mb": vram, "idx": idx + 1},
                )
    if current_batch:
        _ingest_batch(current_batch)

    duration = time.perf_counter() - t_start
    vram_peak_mb = _vram_peak_mb(device)
    n_ingested = sum(per_class.values())

    return GalleryReceipt(
        collection_name=collection_name,
        n_images_requested=n_total,
        n_images_ingested=n_ingested,
        n_caption_empty=n_caption_empty,
        vram_peak_mb=vram_peak_mb,
        duration_seconds=duration,
        fusion_lambda=fusion_lambda,
        created_under_adr=created_under_adr,
        per_class_counts=per_class,
    )


__all__ = ["GalleryImage", "GalleryReceipt", "build_image_gallery"]
