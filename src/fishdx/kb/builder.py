"""Knowledge-base builder — M1 architecture §6.

Ingests 8 Markdown disease documents into a ChromaDB collection with
``hnsw:M=16`` / ``ef_construction=200`` / single-thread build
(ADR-0002 M2).

Idempotence: a repeat ``ingest()`` of the same documents is a no-op —
existing ids are overwritten with identical content so the receipt SHA is
stable.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fishdx.config import KBConfig, RetrievalConfig
from fishdx.errors import KBIngestError
from fishdx.schemas import IngestReceipt, KBDocument

if TYPE_CHECKING:
    from fishdx.retrieval.clip_encoder import EmbedderProtocol


_VERSION = "0.1.0"
_FRONT_MATTER_EXPECTED_PARTS = 3  # split("---", 2) → prefix, front, body


_CLINICAL_TIER_HEADINGS = {
    "confirmed keywords": "confirmed",
    "suspected keywords": "suspected",
    "mentioned keywords": "mentioned",
}
_HEALTHY_HEADING = "healthy keywords"
_NEGATION_CN_HEADING = "negation patterns (cn)"
_NEGATION_EN_HEADING = "negation patterns (en)"


def _extract_bullet_list(block: str) -> list[str]:
    """Return the sequence of ``- item`` or ``* item`` bullets in ``block``."""
    items: list[str] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if line.startswith(("- ", "* ")):
            items.append(line[2:].strip())
    return items


def _parse_body_sections(body: str) -> dict[str, str]:
    """Split a Markdown body into ``{heading_lower: block_text}`` sections.

    Headings recognised are level-2 ATX (``## Heading``). Text before the
    first heading is stored under the ``"_intro"`` key as the clinical
    presentation paragraph.
    """
    sections: dict[str, str] = {"_intro": ""}
    current_key = "_intro"
    buffer: list[str] = []
    for raw_line in body.splitlines():
        if raw_line.startswith("## "):
            sections[current_key] = "\n".join(buffer).strip()
            current_key = raw_line[3:].strip().lower()
            buffer = []
        else:
            buffer.append(raw_line)
    sections[current_key] = "\n".join(buffer).strip()
    return sections


def parse_markdown_docs(root: Path) -> list[KBDocument]:
    """Parse the 8 KB Markdown docs under ``root`` per ADR-0007 format.

    Frontmatter keys (7 mandatory under ADR-0007): ``doc_id``,
    ``disease_class``, ``title``, ``source_reference``, ``source_url``,
    ``substitution_date``, ``substituted_under_adr``.

    Body sections (5 mandatory; 6 for HLT): ``## Clinical presentation``
    (free paragraph before any heading also accepted), ``## Confirmed
    keywords``, ``## Suspected keywords``, ``## Mentioned keywords``,
    ``## Healthy keywords`` (HLT only), ``## Negation patterns (CN)``,
    ``## Negation patterns (EN)``.
    """
    docs: list[KBDocument] = []
    for path in sorted(root.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        parts = raw.split("---", 2)
        if len(parts) < _FRONT_MATTER_EXPECTED_PARTS:
            raise KBIngestError(
                f"KB doc missing front-matter: {path.name}",
                context={"path": str(path)},
            )
        front = parts[1].strip()
        body = parts[2].strip()
        meta: dict[str, str] = {}
        for line in front.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()

        sections = _parse_body_sections(body)
        clinical_keywords: dict[str, list[str]] = {}
        for heading, tier in _CLINICAL_TIER_HEADINGS.items():
            if heading in sections:
                clinical_keywords[tier] = _extract_bullet_list(sections[heading])

        healthy_keywords = (
            _extract_bullet_list(sections[_HEALTHY_HEADING]) if _HEALTHY_HEADING in sections else []
        )
        neg_cn = (
            _extract_bullet_list(sections[_NEGATION_CN_HEADING])
            if _NEGATION_CN_HEADING in sections
            else []
        )
        neg_en = (
            _extract_bullet_list(sections[_NEGATION_EN_HEADING])
            if _NEGATION_EN_HEADING in sections
            else []
        )

        adr_raw = (meta.get("substituted_under_adr") or "").strip()
        if adr_raw.startswith("[") and adr_raw.endswith("]"):
            # ADR-0012a list form, e.g. "[0007, 0012a]"
            adr_list = [s.strip() for s in adr_raw[1:-1].split(",") if s.strip()]
        elif adr_raw:
            # ADR-0007 single-int legacy form
            adr_list = [adr_raw]
        else:
            adr_list = []

        docs.append(
            KBDocument(
                doc_id=meta["doc_id"],
                disease_class=meta.get("disease_class", meta["doc_id"]),
                title=meta.get("title", meta["doc_id"]),
                text=body,
                source_reference=meta.get(
                    "source_reference", "Fish Pathology 4th ed. (Roberts 2012)"
                ),
                source_url=meta.get("source_url", ""),
                substitution_date=meta.get("substitution_date"),
                rewrite_date=meta.get("rewrite_date"),
                substituted_under_adr=adr_list,
                clinical_keywords=clinical_keywords,
                healthy_keywords=healthy_keywords,
                negation_patterns_cn=neg_cn,
                negation_patterns_en=neg_en,
            )
        )
    return docs


def _content_sha(docs: Sequence[KBDocument]) -> str:
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda x: x.doc_id):
        h.update(f"{d.doc_id}|{d.disease_class}|{d.text}".encode())
        h.update(b"\x00")
    return h.hexdigest()


class KBBuilder:
    """Deterministic ChromaDB ingester.

    Accepts a pre-built ``chroma_client`` (usually an in-process
    :class:`chromadb.PersistentClient`) and an embedder implementing
    :class:`EmbedderProtocol`.
    """

    def __init__(
        self,
        *,
        chroma_client: Any,
        embedder: EmbedderProtocol,
        kb_config: KBConfig,
        retrieval_config: RetrievalConfig,
        ingestion_seed: int = 42,
    ) -> None:
        self._client = chroma_client
        self._embedder = embedder
        self._kb = kb_config
        self._retrieval = retrieval_config
        self._ingestion_seed = ingestion_seed

    def _collection(self) -> Any:
        metadata = {
            "hnsw:space": self._retrieval.metric,
            "hnsw:M": self._retrieval.hnsw_M,
            "hnsw:construction_ef": self._retrieval.hnsw_ef_construction,
            "hnsw:search_ef": self._retrieval.hnsw_ef_search,
            "hnsw:num_threads": self._retrieval.hnsw_num_threads,
        }
        return self._client.get_or_create_collection(
            name=self._kb.collection_name, metadata=metadata
        )

    def ingest(self, docs: Sequence[KBDocument]) -> IngestReceipt:
        if len(docs) != self._kb.documents_count:
            raise KBIngestError(
                f"documents_count mismatch: got {len(docs)}, expected {self._kb.documents_count}",
                context={"observed": len(docs), "expected": self._kb.documents_count},
            )
        t0 = time.perf_counter()
        collection = self._collection()
        ids = [d.doc_id for d in docs]
        texts = [d.text for d in docs]
        metadatas: list[dict[str, str]] = [
            {
                "disease_class": d.disease_class,
                "title": d.title,
                "source_reference": d.source_reference,
            }
            for d in docs
        ]
        embeddings = self._embedder.encode_text(texts)
        collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        elapsed = time.perf_counter() - t0
        return IngestReceipt(
            collection_name=self._kb.collection_name,
            n_documents=len(docs),
            ingestion_seed=self._ingestion_seed,
            sha256_content=_content_sha(docs),
            duration_seconds=elapsed,
            fishdx_version=_VERSION,
        )


__all__ = ["KBBuilder", "parse_markdown_docs"]
