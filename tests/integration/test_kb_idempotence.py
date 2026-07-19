"""KB idempotent ingest — M1 architecture §6.3."""

from __future__ import annotations

from pathlib import Path

import pytest

from fishdx.config import KBConfig, RetrievalConfig
from fishdx.errors import KBIngestError
from fishdx.kb.builder import KBBuilder, parse_markdown_docs
from fishdx.schemas import KBDocument


class _FakeCollection:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, object]] = {}

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, str]],
        embeddings: list[list[float]],
    ) -> None:
        for i, d, m, e in zip(ids, documents, metadatas, embeddings):  # noqa: B905
            self.rows[i] = {"doc": d, "meta": m, "emb": e}


class _FakeClient:
    def __init__(self) -> None:
        self._coll = _FakeCollection()

    def get_or_create_collection(self, name: str, metadata: dict[str, object]) -> _FakeCollection:
        return self._coll


class _CountingEmbedder:
    calls: int = 0

    @property
    def embedding_dim(self) -> int:
        return 4

    def encode_text(self, texts):  # type: ignore[no-untyped-def]
        _CountingEmbedder.calls += 1
        return [[float(len(t)), 1.0, 2.0, 3.0] for t in texts]

    def encode_image_paths(self, paths):  # type: ignore[no-untyped-def]
        raise NotImplementedError


@pytest.fixture
def builder() -> KBBuilder:
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
    _CountingEmbedder.calls = 0
    return KBBuilder(
        chroma_client=_FakeClient(),
        embedder=_CountingEmbedder(),
        kb_config=kb_cfg,
        retrieval_config=rt_cfg,
    )


def _load_docs() -> list[KBDocument]:
    here = Path(__file__).resolve().parents[2] / "src" / "fishdx" / "kb" / "documents"
    return parse_markdown_docs(here)


def test_kb_ingest_count(builder: KBBuilder) -> None:
    """KBI1 | 8 docs ingested, receipt.n_documents == 8."""
    docs = _load_docs()
    receipt = builder.ingest(docs)
    assert receipt.n_documents == 8


def test_kb_ingest_sha_stable(builder: KBBuilder) -> None:
    """KBI2 | Repeat ingest produces identical content SHA (idempotence)."""
    docs = _load_docs()
    r1 = builder.ingest(docs)
    r2 = builder.ingest(docs)
    assert r1.sha256_content == r2.sha256_content


def test_kb_ingest_wrong_count_raises(builder: KBBuilder) -> None:
    """KBI3 | documents_count != 8 → KBIngestError."""
    docs = _load_docs()[:7]
    with pytest.raises(KBIngestError):
        builder.ingest(docs)
