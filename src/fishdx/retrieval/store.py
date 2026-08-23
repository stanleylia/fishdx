"""ChromaDB read-side wrapper — paper §III.C, ADR-0002, ADR-0012d.

Thin query layer over a ChromaDB collection. Enforces Top-K,
cosine-similarity cutoff, and deterministic fetch (HNSW single-thread
per ADR-0002 M2). Per ADR-0012d, Stage 2 retrieval queries the
``image_gallery`` collection (1,639 encoded D1 Train fused embeddings) while
Stage 3 keyword scoring uses the ``fishdx_kb`` collection in a
declarative role; this wrapper is collection-agnostic and parameterised
by ``collection_name`` + ``expected_source_type`` (Pattern G Layer 4
runtime architecture-fidelity check).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fishdx.config import RetrievalConfig
from fishdx.errors import ChromaStoreError
from fishdx.schemas import RetrievedDoc


class ChromaStore:
    """Query-side wrapper for an already-populated ChromaDB collection.

    Pattern G Layer 4 (Architecture Fidelity) is enforced at
    :meth:`warmup`: when ``expected_source_type`` is provided, the
    collection's metadata ``embedding_source_type`` field is checked
    against the expected value. Mismatch raises :class:`ChromaStoreError`
    so an architectural misroute (e.g. Stage 2 accidentally pointed at
    ``fishdx_kb`` instead of ``image_gallery``) surfaces at query time
    rather than silently producing wrong-pipeline results.
    """

    def __init__(
        self,
        chroma_client: Any,
        *,
        collection_name: str,
        retrieval_config: RetrievalConfig,
        expected_source_type: str | None = None,
    ) -> None:
        self._client = chroma_client
        self._collection_name = collection_name
        self._config = retrieval_config
        self._expected_source_type = expected_source_type
        self._collection: Any = None

    def warmup(self) -> None:
        if self._collection is not None:
            return
        try:
            self._collection = self._client.get_collection(name=self._collection_name)
        except Exception as e:
            raise ChromaStoreError(
                f"collection not found: {self._collection_name}",
                context={"cause": str(e)},
            ) from e
        if self._expected_source_type is not None:
            actual = (self._collection.metadata or {}).get("embedding_source_type")
            if actual != self._expected_source_type:
                raise ChromaStoreError(
                    f"Pattern G Layer 4 architecture-fidelity mismatch: "
                    f"collection {self._collection_name!r} has "
                    f"embedding_source_type={actual!r} but caller expected "
                    f"{self._expected_source_type!r}",
                    context={
                        "collection": self._collection_name,
                        "actual": actual,
                        "expected": self._expected_source_type,
                    },
                )

    def query(
        self,
        embedding: Sequence[float],
        *,
        top_k: int | None = None,
        similarity_cutoff: float | None = None,
    ) -> list[RetrievedDoc]:
        """Cosine-similarity query over the KB; returns Top-K above cutoff.

        ChromaDB's cosine distance is ``1 - cos``; we invert to similarity.
        """
        self.warmup()
        assert self._collection is not None
        k = top_k if top_k is not None else self._config.top_k
        cutoff = (
            similarity_cutoff if similarity_cutoff is not None else self._config.similarity_cutoff
        )
        try:
            result = self._collection.query(query_embeddings=[list(embedding)], n_results=k)
        except Exception as e:
            raise ChromaStoreError(
                f"collection.query failed: {e}", context={"cause": str(e)}
            ) from e

        # ChromaDB returns per-batch lists; batch size = 1 here.
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]

        retrieved: list[RetrievedDoc] = []
        for i, d, m, dist in zip(ids, docs, metas, dists):
            sim = 1.0 - float(dist)  # cosine distance → cosine similarity
            if sim < cutoff:
                continue
            meta_clean = {str(k): str(v) for k, v in (m or {}).items()}
            retrieved.append(
                RetrievedDoc(
                    doc_id=str(i),
                    text=str(d),
                    disease_class=meta_clean.get("disease_class"),
                    similarity=sim,
                    metadata=meta_clean,
                )
            )
        return retrieved


__all__ = ["ChromaStore"]
