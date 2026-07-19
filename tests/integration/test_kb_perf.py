"""KB build performance — configs/default.yaml SLA (p95 ≤ 90 s for full index).

Since M1 does not yet load real CLIP weights, this perf test asserts the
**parse + ingest with a fake embedder** stays under a very loose ceiling
(5 s). The real 90 s SLA will apply in M2 when the OpenClipEmbedder
landing lights up.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from fishdx.config import KBConfig, RetrievalConfig
from fishdx.kb.builder import KBBuilder, parse_markdown_docs


class _NoopEmbedder:
    @property
    def embedding_dim(self) -> int:
        return 4

    def encode_text(self, texts):  # type: ignore[no-untyped-def]
        return [[0.0, 0.0, 0.0, 0.0] for _ in texts]

    def encode_image_paths(self, paths):  # type: ignore[no-untyped-def]
        raise NotImplementedError


class _NoopCollection:
    def upsert(self, *, ids, documents, metadatas, embeddings):  # type: ignore[no-untyped-def]
        pass


class _NoopClient:
    def get_or_create_collection(self, *, name, metadata):  # type: ignore[no-untyped-def]
        return _NoopCollection()


@pytest.mark.integration
def test_kb_parse_ingest_under_budget() -> None:
    """KBPERF1 | parse + ingest (no model) completes in < 5 s (M1 budget)."""
    root = Path(__file__).resolve().parents[2] / "src" / "fishdx" / "kb" / "documents"
    kb_cfg = KBConfig(
        documents_count=8,
        root_path="./data/kb/",
        chromadb_persist="./data/kb/chromadb",
        collection_name="fishdx_kb",
        source_reference="Roberts 2012 4th ed.",
    )
    rt_cfg = RetrievalConfig(
        index_type="hnsw",
        metric="cosine",
        top_k=5,
        similarity_cutoff=0.5,
        hnsw_ef_construction=200,
        hnsw_ef_search=100,
        hnsw_M=16,
        hnsw_num_threads=1,
    )
    builder = KBBuilder(
        chroma_client=_NoopClient(),
        embedder=_NoopEmbedder(),
        kb_config=kb_cfg,
        retrieval_config=rt_cfg,
    )
    t0 = time.perf_counter()
    docs = parse_markdown_docs(root)
    receipt = builder.ingest(docs)
    elapsed = time.perf_counter() - t0
    assert receipt.n_documents == 8
    assert elapsed < 5.0
