"""ChromaDB Client for vector storage and RAG retrieval.

Persistent ChromaDB with HNSW cosine distance for:
- Storing fish disease knowledge embeddings
- Top-k similarity retrieval for RAG pipeline
- Continuous learning data injection
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.config import ChromaDBConfig

logger = logging.getLogger(__name__)


class ChromaDBClient:
    """ChromaDB persistent client for knowledge base storage and retrieval."""

    def __init__(self, config: ChromaDBConfig) -> None:
        self.config = config
        self._client: Any = None
        self._collection: Any = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Initialize ChromaDB client and collection on first use."""
        if self._initialized:
            return

        try:
            import chromadb

            logger.info(f"Initializing ChromaDB at {self.config.persist_directory}")
            self._client = chromadb.PersistentClient(
                path=self.config.persist_directory
            )
            self._collection = self._client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": self.config.distance_metric},
            )
            self._initialized = True

            count = self._collection.count()
            logger.info(
                f"ChromaDB initialized: collection='{self.config.collection_name}', "
                f"documents={count}"
            )
        except ImportError:
            logger.error("chromadb package not installed")
            raise
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            raise

    def add_documents(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Add documents with embeddings to the collection.

        Args:
            ids: Unique document IDs.
            embeddings: List of embedding vectors.
            documents: Document content strings.
            metadatas: Optional metadata dicts per document.
        """
        self._ensure_initialized()

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Added {len(ids)} documents to ChromaDB")

    def query(
        self,
        query_embedding: list[float] | NDArray[np.float32],
        top_k: int | None = None,
        where: dict | None = None,
    ) -> list[dict]:
        """Query ChromaDB for similar documents.

        Args:
            query_embedding: Query vector (512-dim).
            top_k: Number of results to return (default from config).
            where: Optional metadata filter.

        Returns:
            List of dicts with keys: content, similarity, source, id.
        """
        self._ensure_initialized()

        k = top_k or self.config.top_k

        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": k,
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        parsed: list[dict] = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                # Convert cosine distance to similarity (clamp for float precision)
                similarity = max(0.0, min(1.0, 1.0 - distance))

                if similarity < self.config.similarity_cutoff:
                    continue

                source = ""
                if results["metadatas"] and results["metadatas"][0]:
                    source = results["metadatas"][0][i].get("source", "")

                doc_id = results["ids"][0][i] if results["ids"] else ""

                parsed.append({
                    "content": doc,
                    "similarity": similarity,
                    "source": source,
                    "id": doc_id,
                })

        return parsed

    def get_documents(
        self,
        ids: list[str],
        include: list[str] | None = None,
    ) -> list[dict]:
        """Retrieve specific documents by their IDs.

        Args:
            ids: Document IDs to retrieve.
            include: Fields to include (default: documents, metadatas).

        Returns:
            List of dicts with keys: id, content, metadata, embedding.
        """
        self._ensure_initialized()
        include = include or ["documents", "metadatas"]

        results = self._collection.get(ids=ids, include=include)

        parsed: list[dict] = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                entry: dict[str, Any] = {"id": doc_id}
                if "documents" in include and results.get("documents"):
                    entry["content"] = results["documents"][i]
                if "metadatas" in include and results.get("metadatas"):
                    entry["metadata"] = results["metadatas"][i]
                if "embeddings" in include and results.get("embeddings"):
                    entry["embedding"] = results["embeddings"][i]
                parsed.append(entry)
        return parsed

    def update_metadata(
        self,
        ids: list[str],
        metadatas: list[dict],
    ) -> None:
        """Update metadata for existing documents.

        Args:
            ids: Document IDs to update.
            metadatas: New metadata dicts.
        """
        self._ensure_initialized()
        self._collection.update(ids=ids, metadatas=metadatas)
        logger.debug(f"Updated metadata for {len(ids)} documents")

    def delete_documents(self, ids: list[str]) -> None:
        """Delete specific documents by ID.

        Args:
            ids: Document IDs to delete.
        """
        self._ensure_initialized()
        self._collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents from ChromaDB")

    def query_by_metadata(
        self,
        where: dict,
        top_k: int = 100,
        include: list[str] | None = None,
    ) -> list[dict]:
        """Query documents by metadata filter (no embedding needed).

        Args:
            where: ChromaDB where clause.
            top_k: Max results.
            include: Fields to include.

        Returns:
            List of dicts with keys: id, content, metadata.
        """
        self._ensure_initialized()
        include = include or ["documents", "metadatas"]

        results = self._collection.get(
            where=where,
            limit=top_k,
            include=include,
        )

        parsed: list[dict] = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                entry: dict[str, Any] = {"id": doc_id}
                if results.get("documents"):
                    entry["content"] = results["documents"][i]
                if results.get("metadatas"):
                    entry["metadata"] = results["metadatas"][i]
                parsed.append(entry)
        return parsed

    def get_all_documents(
        self,
        limit: int = 100,
        include: list[str] | None = None,
    ) -> list[dict]:
        """Retrieve all documents from the collection (up to limit).

        Args:
            limit: Maximum number of documents to return.
            include: Fields to include (default: documents, metadatas).

        Returns:
            List of dicts with keys: id, content, metadata.
        """
        self._ensure_initialized()
        include = include or ["documents", "metadatas"]

        results = self._collection.get(limit=limit, include=include)

        parsed: list[dict] = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                entry: dict[str, Any] = {"id": doc_id}
                if "documents" in include and results.get("documents"):
                    entry["content"] = results["documents"][i]
                if "metadatas" in include and results.get("metadatas"):
                    entry["metadata"] = results["metadatas"][i]
                if "embeddings" in include and results.get("embeddings"):
                    entry["embedding"] = results["embeddings"][i]
                parsed.append(entry)
        return parsed

    def count(self) -> int:
        """Get the number of documents in the collection."""
        self._ensure_initialized()
        return self._collection.count()

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self._ensure_initialized()
        self._client.delete_collection(self.config.collection_name)
        self._initialized = False
        logger.warning(f"Deleted collection '{self.config.collection_name}'")

    @property
    def is_initialized(self) -> bool:
        return self._initialized
