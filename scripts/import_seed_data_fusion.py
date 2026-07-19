"""Import seed data into ChromaDB with CLIP fusion embeddings.

Creates λ-weighted fusion embeddings (Eq.6-8) that match the pipeline's
query space. For diseases with representative images in the dataset,
computes fusion centroids from multiple images. For diseases without
images, falls back to text-only embeddings.

Usage: python scripts/import_seed_data_fusion.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Dataset class → disease_id mapping
# Maps directory names in fish_disease_south_asia/Train/ to our disease IDs
DATASET_CLASS_MAP: dict[str, str] = {
    "Bacterial Red disease": "BRD",
    "Bacterial diseases - Aeromoniasis": "AER",
    "Bacterial gill disease": "BGD",
    "Fungal diseases Saprolegniasis": "SAP",
    "Healthy Fish": "HEALTHY",
    "Parasitic diseases": "WSD",
    "Viral diseases White tail disease": "WTD",
}


def load_seed_data(seed_dir: str) -> list[dict]:
    """Load all seed data JSON files."""
    seed_path = Path(seed_dir)
    all_data: list[dict] = []

    for json_file in seed_path.rglob("*.json"):
        logger.info(f"Loading {json_file}")
        with open(json_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)

    return all_data


def find_representative_images(
    dataset_dir: Path,
    disease_id: str,
    max_images: int = 5,
) -> list[Path]:
    """Find representative images for a disease from the dataset.

    Args:
        dataset_dir: Root dataset directory containing Train/ subdirectory.
        disease_id: Disease ID to look up in DATASET_CLASS_MAP.
        max_images: Maximum number of images to use.

    Returns:
        List of image paths, up to max_images.
    """
    # Reverse lookup: disease_id → class directory name
    class_name = None
    for cls, did in DATASET_CLASS_MAP.items():
        if did == disease_id:
            class_name = cls
            break

    if class_name is None:
        return []

    class_dir = dataset_dir / "Train" / class_name
    if not class_dir.exists():
        logger.warning(f"Class directory not found: {class_dir}")
        return []

    # Collect image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = sorted(
        p for p in class_dir.iterdir()
        if p.suffix.lower() in image_extensions
    )

    if not images:
        logger.warning(f"No images found in {class_dir}")
        return []

    # Select evenly spaced images for diversity
    if len(images) <= max_images:
        selected = images
    else:
        indices = np.linspace(0, len(images) - 1, max_images, dtype=int)
        selected = [images[i] for i in indices]

    logger.info(f"  {disease_id}: selected {len(selected)}/{len(images)} images from {class_name}")
    return selected


def compute_fusion_centroid(
    clip_wrapper,
    images: list[Path],
    text_content: str,
    lambda_weight: float,
) -> NDArray[np.float32]:
    """Compute fusion embedding centroid from multiple images + text.

    For each image:
      visual_emb = clip.encode_image(image)
      text_emb = clip.encode_text(content[:77])
      fusion = λ * normalize(visual) + (1-λ) * normalize(text)  (Eq.6-8)

    Then average all fusion embeddings and re-normalize.

    Args:
        clip_wrapper: CLIPWrapper instance.
        images: List of image paths.
        text_content: Disease description text.
        lambda_weight: Visual weight λ (default 0.7).

    Returns:
        L2-normalized fusion centroid embedding (512-dim).
    """
    from PIL import Image

    # Pre-compute text embedding (same for all images of this disease)
    text_emb = clip_wrapper.encode_text(text_content[:77])
    text_norm = text_emb / (np.linalg.norm(text_emb) + 1e-10)

    fusion_embeddings: list[NDArray[np.float32]] = []

    for img_path in images:
        try:
            img = Image.open(img_path).convert("RGB")
            visual_emb = clip_wrapper.encode_image(img)
            visual_norm = visual_emb / (np.linalg.norm(visual_emb) + 1e-10)

            # Eq.6-8: λ-weighted fusion
            fused = lambda_weight * visual_norm + (1.0 - lambda_weight) * text_norm
            fused = fused / (np.linalg.norm(fused) + 1e-10)
            fusion_embeddings.append(fused.astype(np.float32))
        except Exception as e:
            logger.warning(f"  Failed to process {img_path.name}: {e}")

    if not fusion_embeddings:
        # Fallback to text-only if all images failed
        logger.warning("  All images failed, using text-only embedding")
        return (text_norm / (np.linalg.norm(text_norm) + 1e-10)).astype(np.float32)

    # Average fusion embeddings → centroid
    stacked = np.stack(fusion_embeddings)
    centroid = np.mean(stacked, axis=0).astype(np.float32)

    # L2-normalize centroid
    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)

    return centroid


def main() -> None:
    parser = argparse.ArgumentParser(description="Import seed data with fusion embeddings")
    parser.add_argument("--config", default=None, help="Config file path")
    parser.add_argument("--seed-dir", default="knowledge_base/seed_data/fish_diseases")
    parser.add_argument(
        "--dataset-dir",
        default="lab_dateset/external_datasets/fish_disease_south_asia/"
                "Freshwater Fish Disease Aquaculture in south asia",
    )
    parser.add_argument("--max-images", type=int, default=5,
                        help="Max images per disease for fusion centroid")
    parser.add_argument("--delete-existing", action="store_true",
                        help="Delete existing collection before import")
    parser.add_argument("--use-random-embeddings", action="store_true",
                        help="Use random embeddings (for testing without CLIP)")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_data = load_seed_data(args.seed_dir)

    if not seed_data:
        logger.error("No seed data found")
        sys.exit(1)

    logger.info(f"Loaded {len(seed_data)} seed documents")

    dataset_dir = Path(args.dataset_dir)
    lambda_weight = config.fusion.lambda_weight
    logger.info(f"Fusion λ = {lambda_weight}")

    # Generate embeddings
    ids: list[str] = []
    documents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []

    if args.use_random_embeddings:
        logger.warning("Using random embeddings (testing mode)")
        for doc in seed_data:
            ids.append(doc["id"])
            documents.append(doc["content"])
            rng = np.random.RandomState(hash(doc["id"]) % 2**31)
            emb = rng.randn(config.models.clip.embedding_dim).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb.tolist())
            metadatas.append({
                "source": doc.get("source", "seed"),
                "disease_id": doc.get("disease_id", ""),
                "severity": doc.get("severity", ""),
                "title": doc.get("title", ""),
            })
    else:
        try:
            from core.models.clip_wrapper import CLIPWrapper
            clip = CLIPWrapper(config.models.clip)

            for doc in seed_data:
                disease_id = doc.get("disease_id", "")
                ids.append(doc["id"])
                documents.append(doc["content"])

                # Try to find representative images for fusion embedding
                images = find_representative_images(
                    dataset_dir, disease_id, args.max_images,
                )

                if images:
                    # Fusion centroid: average of (λ*visual + (1-λ)*text) per image
                    logger.info(f"  Computing fusion centroid for {doc['id']} "
                                f"({len(images)} images, λ={lambda_weight})")
                    emb = compute_fusion_centroid(
                        clip, images, doc["content"], lambda_weight,
                    )
                else:
                    # Text-only fallback for diseases without dataset images
                    logger.info(f"  Text-only embedding for {doc['id']} (no dataset images)")
                    emb = clip.encode_text(doc["content"][:77])
                    emb = emb / (np.linalg.norm(emb) + 1e-10)

                embeddings.append(emb.tolist())
                metadatas.append({
                    "source": doc.get("source", "seed"),
                    "disease_id": disease_id,
                    "severity": doc.get("severity", ""),
                    "title": doc.get("title", ""),
                })
                logger.info(f"  Encoded: {doc['id']} (norm={np.linalg.norm(emb):.4f})")

        except ImportError as e:
            logger.error(f"CLIP not available: {e}. Use --use-random-embeddings for testing.")
            sys.exit(1)

    # Import into ChromaDB
    from core.models.chromadb_client import ChromaDBClient
    db = ChromaDBClient(config.chromadb)

    if args.delete_existing:
        try:
            db.delete_collection()
            logger.info("Deleted existing collection")
            # Re-initialize after deletion
            db = ChromaDBClient(config.chromadb)
        except Exception:
            logger.info("No existing collection to delete")

    db.add_documents(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info(f"Successfully imported {len(ids)} documents into ChromaDB")
    logger.info(f"Collection: {config.chromadb.collection_name}")
    logger.info(f"Total documents: {db.count()}")

    # Print summary
    disease_ids = [m.get("disease_id", "?") for m in metadatas]
    has_images = set()
    for doc in seed_data:
        did = doc.get("disease_id", "")
        imgs = find_representative_images(dataset_dir, did, 1) if not args.use_random_embeddings else []
        if imgs:
            has_images.add(did)

    logger.info("--- Embedding Summary ---")
    for doc, meta in zip(seed_data, metadatas):
        did = meta.get("disease_id", "?")
        mode = "FUSION" if did in has_images else "TEXT-ONLY"
        logger.info(f"  {doc['id']:10s} | {did:8s} | {mode}")


if __name__ == "__main__":
    main()
