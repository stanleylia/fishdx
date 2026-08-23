"""fishdx end-to-end pipeline orchestrator — architecture.md §4.3 + ADR-0012d.

Three-stage deterministic flow with **dual-collection** routing per
ADR-0012d (P'-2):

    Stage 1  — Florence-2 caption + detection → clean_caption
                → Pareidolia Eq. 1-4 correction on detected objects
    Stage 2  — CLIP text encode + image encode → λ-fusion (Eq. 5-7)
                → ChromaStore Top-K query against ``image_gallery``
                (1,639 encoded D1 Train fused embeddings, paper §III-E target)
                → Algorithm 1 verification
    Stage 3  — top-1 retrieval's ``metadata['doc_id']`` resolves to the
                in-memory KB doc (parsed from ADR-0007 Wikipedia files);
                compute_s_h (Eq. 8) + compute_s_d (Eq. 9) on caption
                against doc keyword tiers → make_decision (Eq. 10 + 11)

The text-KB ``fishdx_kb`` collection is **also** built for declarative
ADR-0012d compliance (`scoring.keyword_dict` config key) but the
keyword scoring path uses the in-memory ``KBDocument`` list which is
the source of truth.

Returns a frozen :class:`DiagnosisResult` with full telemetry. Pattern G
Layer 4 architecture-fidelity check fires at warmup via ChromaStore's
``expected_source_type="fused"`` guard.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from fishdx.config import AppConfig
from fishdx.kb.builder import KBBuilder, parse_markdown_docs
from fishdx.perception.florence2 import Florence2Wrapper
from fishdx.perception.pareidolia import (
    compute_p_hard,
    compute_p_soft,
    merge_pareidolia,
    remap_label,
)
from fishdx.perception.postprocess import clean_caption
from fishdx.retrieval.clip_encoder import OpenClipEmbedder
from fishdx.retrieval.fusion import fuse_and_normalize, l2_normalize
from fishdx.retrieval.store import ChromaStore
from fishdx.retrieval.reweighting import reweight_candidates
from fishdx.schemas import (
    DecisionEnum,
    DetectedObject,
    DiagnosisResult,
    HealthStatus,
    KBDocument,
    RetrievedDoc,
)
from fishdx.scoring.decision import make_decision
from fishdx.scoring.score import compute_s_d, compute_s_h

_LOG = logging.getLogger(__name__)
_HLT_DOC_ID = "HLT"


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    return float(sum(x * y for x, y in zip(a, b)))


def _centroid_l2(vectors: list[list[float]]) -> list[float]:
    """Element-wise mean then L2-normalise → concept centroid."""
    arr = np.array(vectors, dtype=np.float64)
    mean = arr.mean(axis=0).tolist()
    return l2_normalize(mean)


class Pipeline:
    """End-to-end deterministic diagnostic pipeline.

    Lifecycle: ``__init__(config, kb_docs_dir) → warmup() →
    diagnose(image)* → close()``.
    """

    def __init__(self, config: AppConfig, kb_docs_dir: Path | None = None) -> None:
        self._config = config
        self._kb_docs_dir = kb_docs_dir or (Path(__file__).resolve().parent / "kb" / "documents")
        self._florence = Florence2Wrapper(config.florence2)
        self._clip = OpenClipEmbedder(config.clip)
        self._chroma_client: Any = None
        self._kb_builder: KBBuilder | None = None
        self._retrieval_store: ChromaStore | None = None  # image_gallery (Stage 2)
        self._kb_docs: list[KBDocument] = []
        self._kb_doc_by_id: dict[str, KBDocument] = {}
        self._c_s: list[float] = []
        self._c_b: list[float] = []
        self._warm = False

    # ── lifecycle ──────────────────────────────────────────────────
    def warmup(
        self,
        chroma_persist: Path | str | None = None,
        *,
        build_text_kb: bool = True,
    ) -> None:
        """Load models + open image_gallery + build text KB + cache centroids.

        ADR-0012d dual-collection routing:
        - ``self._retrieval_store`` opens the existing
          ``self._config.retrieval.collection`` (= ``"image_gallery"``)
          via Pattern G Layer 4 ``expected_source_type="fused"`` check.
          Caller MUST ensure the gallery exists at ``chroma_persist``
          (built by :func:`fishdx.kb.image_gallery_builder.build_image_gallery`).
        - ``self._kb_docs`` parsed from Markdown for in-memory keyword
          lookup at Stage 3.
        - ``self._config.scoring.keyword_dict`` (= ``"fishdx_kb"``) is
          declaratively the Stage 3 source; we materialise it as a
          ChromaCollection for ADR-0012d compliance + future query
          support, but Stage 3 reads from the in-memory dict.
        """
        if self._warm:
            return
        self._florence.warmup()
        self._clip.warmup()

        persist = (
            str(chroma_persist) if chroma_persist is not None else self._config.kb.chromadb_persist
        )
        self._chroma_client = chromadb.PersistentClient(path=persist)

        # Stage 2: retrieval store opens the pre-existing image_gallery
        # collection. Pattern G Layer 4 fidelity check fires on warmup.
        self._retrieval_store = ChromaStore(
            chroma_client=self._chroma_client,
            collection_name=self._config.retrieval.collection,
            retrieval_config=self._config.retrieval,
            expected_source_type="fused",
        )
        self._retrieval_store.warmup()

        # Stage 3: in-memory KB docs (source of truth for keyword tiers).
        self._kb_docs = parse_markdown_docs(self._kb_docs_dir)
        self._kb_doc_by_id = {d.doc_id: d for d in self._kb_docs}

        # Optionally build the text-KB ChromaCollection (declarative
        # ADR-0012d `scoring.keyword_dict` materialisation; not queried
        # by Stage 3 — kept for forward-compat).
        if build_text_kb:
            self._kb_builder = KBBuilder(
                chroma_client=self._chroma_client,
                embedder=self._clip,
                kb_config=self._config.kb,
                retrieval_config=self._config.retrieval,
            )
            self._kb_builder.ingest(self._kb_docs)

        # Pareidolia concept centroids (Eq. 2): encode structural / biological
        # keyword lists via CLIP text encoder, average → L2 normalise.
        ks = list(self._config.pareidolia.structural_keywords)
        kb = list(self._config.pareidolia.biological_keywords)
        self._c_s = _centroid_l2(self._clip.encode_text(ks))
        self._c_b = _centroid_l2(self._clip.encode_text(kb))

        self._warm = True

    def close(self) -> None:
        self._florence.close()
        self._clip.close()
        self._chroma_client = None
        self._retrieval_store = None
        self._warm = False

    def health_check(self) -> HealthStatus:
        return HealthStatus(
            component="pipeline",
            ready=self._warm,
            status="ok" if self._warm else "failed",
            details={"warmed": str(self._warm)},
        )

    # ── inference ──────────────────────────────────────────────────
    def diagnose(self, image_path: Path) -> DiagnosisResult:
        if not self._warm:
            self.warmup()
        assert self._retrieval_store is not None

        t_total_start = time.perf_counter()

        # ── Stage 1 ──────────────────────────────────────────────
        t_s1 = time.perf_counter()
        caption_raw, objects_raw, _s1_telem = self._florence.caption_and_detect(image_path)
        caption_clean = clean_caption(caption_raw)
        objects_corrected, pareidolia_count = self._apply_pareidolia(caption_clean, objects_raw)
        stage1_ms = (time.perf_counter() - t_s1) * 1000

        # ── Stage 2 (ADR-0012d image_gallery retrieval) ─────────
        t_s2 = time.perf_counter()
        if caption_clean:
            e_caption = self._clip.encode_text([caption_clean])[0]
        else:
            # Empty caption → use zero-norm-safe placeholder: encode a single
            # space, which CLIP reduces to a near-uniform baseline vector.
            e_caption = self._clip.encode_text([" "])[0]
        e_visual = self._clip.encode_image_paths([str(image_path)])[0]
        e_final = fuse_and_normalize(e_visual, e_caption, self._config.fusion.lambda_star)
        candidates = self._retrieval_store.query(e_final)
        # Algorithm 1 (retrieval-confidence reweighting) is AUXILIARY and LOGGED
        # ONLY (ADR-0016): it flags low-confidence retrievals but does NOT alter
        # the Stage-3 decision, which uses the raw retrieval top-1 (matching
        # Fig. 1 and the reported analysis scripts).
        retrieval_score, retrieval_low_confidence = self._retrieval_confidence_indicator(candidates)
        _LOG.debug(
            "retrieval-confidence indicator (auxiliary): score=%.6f low_confidence=%s",
            retrieval_score,
            retrieval_low_confidence,
        )
        stage2_ms = (time.perf_counter() - t_s2) * 1000

        # ── Stage 3 (decides on the RAW retrieval top-1) ─────────
        t_s3 = time.perf_counter()
        result = self._score_and_decide(caption_clean, candidates)
        stage3_ms = (time.perf_counter() - t_s3) * 1000

        total_ms = (time.perf_counter() - t_total_start) * 1000

        # Assemble final result with telemetry.
        return result.model_copy(
            update={
                "compute_time_ms": total_ms,
                "stage_latencies_ms": {
                    "stage_1": stage1_ms,
                    "stage_2": stage2_ms,
                    "stage_3": stage3_ms,
                    "total": total_ms,
                },
                "retrieval_count": len(candidates),
                "verification_iterations": 1 if candidates else 0,  # single-pass reweighting (ADR-0016)
                "caption_empty": not caption_clean,
                "pareidolia_correction_count": pareidolia_count,
            }
        )

    # ── internals ──────────────────────────────────────────────────
    def _apply_pareidolia(
        self, caption: str, objects: list[DetectedObject]
    ) -> tuple[list[DetectedObject], int]:
        """Eq. 1-4 correction on each detected object."""
        cfg = self._config.pareidolia
        ks = list(cfg.structural_keywords)
        kb = list(cfg.biological_keywords)
        corrected: list[DetectedObject] = []
        count = 0

        # CLIP-text of caption once (needed for every object's soft path).
        cos_caption_to_cs: float
        if caption and self._c_s:
            e_caption = self._clip.encode_text([caption])[0]
            cos_caption_to_cs = _cosine(e_caption, self._c_s)
        else:
            cos_caption_to_cs = 0.0

        for obj in objects:
            p_hard = compute_p_hard(caption, obj.label, ks, kb)
            # Soft path: cos(CLIP_T(label), C_b)
            if self._c_b:
                e_label = self._clip.encode_text([obj.label])[0]
                cos_label_to_cb = _cosine(e_label, self._c_b)
            else:
                cos_label_to_cb = 0.0
            p_soft = compute_p_soft(cos_caption_to_cs, cos_label_to_cb, cfg.clip_threshold_tau)
            merged = merge_pareidolia(float(p_hard), p_soft)
            new_label = remap_label(obj.label, merged, cfg.remap_label)
            if new_label != obj.label:
                count += 1
            corrected.append(obj.model_copy(update={"label": new_label}))
        return corrected, count

    def _retrieval_confidence_indicator(self, candidates: list[RetrievedDoc]) -> tuple[float, bool]:
        """Algorithm 1 — retrieval-confidence reweighting as an AUXILIARY indicator (ADR-0016).

        Runs the single-pass reweighting and reports whether it flags the retrieval
        as low-confidence. **Logged only**: it does NOT alter the Stage-3 decision,
        which uses the raw retrieval top-1 (matching Fig. 1 and the reported scripts).
        """
        if not candidates:
            return 0.0, True
        ranked = sorted(candidates, key=lambda c: c.similarity, reverse=True)
        s1 = ranked[0].similarity
        s2 = ranked[1].similarity if len(ranked) > 1 else s1
        cfg = self._config.reweighting
        low_confidence = s1 < cfg.similarity_threshold or (s1 - s2) < cfg.margin_threshold
        reweighted = reweight_candidates(candidates, self._config.reweighting)
        score = reweighted[0].similarity_penalized if reweighted else 0.0
        return float(score or 0.0), low_confidence

    def _score_and_decide(self, caption: str, candidates: list[RetrievedDoc]) -> DiagnosisResult:
        if len(candidates) < 2:
            return DiagnosisResult(
                decision=DecisionEnum.INCONCLUSIVE,
                disease_class=None,
                score_healthy=0.0,
                score_disease=0.0,
                retrieval_margin=0.0,
                inconclusive_reason="insufficient_candidates",
            )
        top1 = candidates[0] if candidates else None
        top2 = candidates[1] if len(candidates) >= 2 else None

        # Healthy keywords: always from HLT KB doc (fixed reference).
        hlt = self._kb_doc_by_id.get(_HLT_DOC_ID)
        healthy_kw: tuple[str, ...] = tuple(hlt.healthy_keywords) if hlt else ("healthy",)

        # ADR-0012e evidence pool: 𝒪 = caption ∪ top-1 retrieved KB doc
        # text. Paper §3.4 Eq. 8-9 require the retrieved doc text to be
        # scored alongside the caption; caption-only 𝒪 is inconsistent
        # with paper Table 8 Layer 3 DA = 0.999 under SCA = 3.7% (Pattern
        # J #4 mathematical-consistency reasoning).
        top1_doc_id: str | None = None
        kb_doc = None
        if top1:
            top1_doc_id = top1.metadata.get("doc_id") or top1.doc_id
            kb_doc = self._kb_doc_by_id.get(top1_doc_id)
        top1_text = kb_doc.text if kb_doc is not None else ""
        evidence = f"{caption}\n{top1_text}" if top1_text else caption

        s_h = compute_s_h(
            evidence,
            self._config.scoring.healthy_weights,
            self._config.negation,
            healthy_keywords=healthy_kw,
        )

        # Disease keywords: top-1 retrieved row's metadata['doc_id']
        # resolves to the in-memory KB doc (ADR-0012d dual-collection
        # routing — image_gallery rows store doc_id as metadata, not as
        # the chroma row id which is the image path).
        s_d = 0
        disease_class: str | None = None
        if kb_doc is not None:
            s_d = compute_s_d(
                evidence,
                self._config.scoring,
                confirmed_keywords=tuple(kb_doc.clinical_keywords.get("confirmed", [])),
                suspected_keywords=tuple(kb_doc.clinical_keywords.get("suspected", [])),
                mentioned_keywords=tuple(kb_doc.clinical_keywords.get("mentioned", [])),
            )
            disease_class = kb_doc.disease_class

        top1_sim = (
            top1.similarity_penalized
            if top1 and top1.similarity_penalized is not None
            else (top1.similarity if top1 else 0.0)
        )
        top2_sim = (
            top2.similarity_penalized
            if top2 and top2.similarity_penalized is not None
            else (top2.similarity if top2 else 0.0)
        )

        return make_decision(
            score_healthy=s_h,
            score_disease=s_d,
            top1_similarity=float(top1_sim),
            top2_similarity=float(top2_sim),
            disease_class=disease_class,
            decision_config=self._config.decision,
            margin_config=self._config.margin,
        )


__all__ = ["Pipeline"]


# Guard against pyflakes; DecisionEnum re-exported for callers.
assert DecisionEnum is not None
assert math is not None  # reserved for future pareidolia extensions
