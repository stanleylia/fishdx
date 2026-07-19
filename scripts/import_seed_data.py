"""Import seed data into ChromaDB knowledge base.

Reads fish disease profiles from seed_data and creates embeddings
for ChromaDB storage. Used for initial knowledge base population.

Usage: python scripts/import_seed_data.py [--config configs/default.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Import seed data into ChromaDB")
    parser.add_argument("--config", default=None, help="Config file path")
    parser.add_argument("--seed-dir", default="knowledge_base/seed_data/fish_diseases")
    parser.add_argument("--use-random-embeddings", action="store_true",
                        help="Use random embeddings (for testing without CLIP)")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_data = load_seed_data(args.seed_dir)

    if not seed_data:
        logger.error("No seed data found")
        sys.exit(1)

    logger.info(f"Loaded {len(seed_data)} seed documents")

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
                ids.append(doc["id"])
                documents.append(doc["content"])

                # Encode document text with CLIP
                text_emb = clip.encode_text(doc["content"][:77])  # CLIP max 77 tokens
                text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-10)
                embeddings.append(text_emb.tolist())

                metadatas.append({
                    "source": doc.get("source", "seed"),
                    "disease_id": doc.get("disease_id", ""),
                    "severity": doc.get("severity", ""),
                    "title": doc.get("title", ""),
                })
                logger.info(f"  Encoded: {doc['id']}")

        except ImportError:
            logger.error("CLIP not available. Use --use-random-embeddings for testing.")
            sys.exit(1)

    # Import into ChromaDB
    from core.models.chromadb_client import ChromaDBClient
    db = ChromaDBClient(config.chromadb)

    db.add_documents(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info(f"Successfully imported {len(ids)} documents into ChromaDB")
    logger.info(f"Collection: {config.chromadb.collection_name}")
    logger.info(f"Total documents: {db.count()}")


if __name__ == "__main__":
    main()
