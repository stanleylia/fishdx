"""Bidirectional Verification Loop — Algorithm 1 (§3.5).

RAG-to-Visual verification: uses Florence-2 phrase grounding to validate
RAG results against the actual image, penalizing ungrounded claims.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field

from core.config import VerificationConfig

logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    """Result of Florence-2 phrase grounding for a single keyword."""

    keyword: str
    grounded: bool
    confidence: float
    bbox: list[float] = field(default_factory=list)  # [x1, y1, x2, y2]


@dataclass
class RAGItem:
    """RAG retrieval item with mutable score for verification."""

    content: str
    score: float
    source: str
    verified: bool = False
    penalty_applied: bool = False
    keywords: list[str] = field(default_factory=list)
    doc_id: str = ""


@dataclass
class VerificationResult:
    """Result of the complete bidirectional verification loop."""

    verified_items: list[RAGItem]
    iterations_run: int
    keywords_checked: list[str]
    grounding_results: list[GroundingResult]
    items_removed: int
    converged: bool


def extract_disease_keywords(rag_items: list[RAGItem]) -> list[str]:
    """Extract disease-related keywords from RAG results.

    Scans RAG content for common disease terminology to build
    the keyword list for phrase grounding verification.

    Args:
        rag_items: RAG retrieval results.

    Returns:
        Deduplicated list of disease keywords.
    """
    disease_terms = {
        # English
        "disease", "infection", "lesion", "ulcer", "parasite", "fungus",
        "bacteria", "virus", "necrosis", "hemorrhage", "inflammation",
        "white spot", "red spot", "columnaris", "vibriosis", "streptococcosis",
        "aeromonas", "edwardsiella", "ich", "saprolegnia",
        # Chinese
        "白點病", "赤鰭病", "爛鰓病", "弧菌", "鏈球菌",
        "氣單胞菌", "水黴病", "潰瘍", "出血", "壞死",
    }

    found: list[str] = []
    seen: set[str] = set()

    for item in rag_items:
        content_lower = item.content.lower()
        for term in disease_terms:
            term_lower = term.lower()
            if term_lower in content_lower and term_lower not in seen:
                found.append(term)
                seen.add(term_lower)

    return found


def verify_with_grounding(
    keywords: list[str],
    grounding_fn: object,
    image_path: str,
    config: VerificationConfig,
) -> list[GroundingResult]:
    """Verify keywords using Florence-2 phrase grounding.

    For each keyword, calls Florence-2's CAPTION_TO_PHRASE_GROUNDING
    task and checks if the confidence exceeds threshold θ.

    Args:
        keywords: Disease keywords to verify.
        grounding_fn: Florence-2 grounding callable(image_path, keyword) -> (confidence, bbox).
        image_path: Path to the image.
        config: Verification configuration (contains θ).

    Returns:
        List of GroundingResult per keyword.
    """
    results: list[GroundingResult] = []

    for kw in keywords:
        try:
            confidence, bbox = grounding_fn(image_path, kw)
            grounded = confidence >= config.grounding_threshold
            results.append(GroundingResult(
                keyword=kw,
                grounded=grounded,
                confidence=confidence,
                bbox=bbox,
            ))
            logger.debug(f"Grounding '{kw}': conf={confidence:.3f}, grounded={grounded}")
        except Exception as e:
            logger.warning(f"Grounding failed for '{kw}': {e}")
            results.append(GroundingResult(
                keyword=kw,
                grounded=False,
                confidence=0.0,
            ))

    return results


def apply_penalty(
    rag_items: list[RAGItem],
    grounding_results: list[GroundingResult],
    config: VerificationConfig,
) -> list[RAGItem]:
    """Apply penalty to RAG items with ungrounded keywords.

    For each ungrounded keyword, penalizes RAG items that contain it:
        r.score ← r.score × penalty

    Then filters items below min_score.

    Args:
        rag_items: RAG items with current scores.
        grounding_results: Grounding verification results.
        config: Verification configuration (penalty_factor, min_score).

    Returns:
        Penalized and filtered RAG items.
    """
    # Build set of ungrounded keywords
    ungrounded = {
        gr.keyword.lower()
        for gr in grounding_results
        if not gr.grounded
    }

    penalized: list[RAGItem] = []
    for item in rag_items:
        item_copy = deepcopy(item)
        content_lower = item_copy.content.lower()

        for kw in ungrounded:
            if kw in content_lower:
                item_copy.score *= config.penalty_factor
                item_copy.penalty_applied = True
                logger.debug(
                    f"Penalty applied to '{item_copy.source}' for ungrounded '{kw}', "
                    f"new score={item_copy.score:.3f}"
                )

        # Filter by min_score
        if item_copy.score >= config.min_score:
            penalized.append(item_copy)
        else:
            logger.info(
                f"Removed '{item_copy.source}' (score={item_copy.score:.3f} < {config.min_score})"
            )

    return penalized


def _check_convergence(
    prev_items: list[RAGItem],
    curr_items: list[RAGItem],
) -> bool:
    """Check if the RAG results have converged (stable between iterations)."""
    if len(prev_items) != len(curr_items):
        return False

    for prev, curr in zip(prev_items, curr_items):
        if abs(prev.score - curr.score) > 1e-6:
            return False
        if prev.source != curr.source:
            return False

    return True


def run_verification_loop(
    rag_items: list[RAGItem],
    image_path: str,
    grounding_fn: object,
    requery_fn: object | None,
    config: VerificationConfig,
) -> VerificationResult:
    """Run the complete Bidirectional Verification Loop (Algorithm 1).

    Algorithm 1: RAG-to-Visual Bidirectional Verification
    ─────────────────────────────────────────────────────
    for iter = 1 to max_iter:
        Keywords ← extract_disease_keywords(R)
        for each keyword kw in Keywords:
            grounding ← F.phrase_grounding(I, kw)
            if grounding.confidence < θ:
                for each r in R where kw ∈ r:
                    r.score ← r.score × penalty
        R ← filter(R, score ≥ min_score)
        if R is stable: break
        else: R ← supplementary_query(ChromaDB, top_k=3)
    return R

    Args:
        rag_items: Initial RAG retrieval results.
        image_path: Path to the source image.
        grounding_fn: Florence-2 grounding callable.
        requery_fn: ChromaDB supplementary query callable (optional).
        config: Verification configuration.

    Returns:
        VerificationResult with verified items and metadata.
    """
    current_items = deepcopy(rag_items)
    initial_count = len(current_items)
    all_keywords: list[str] = []
    all_grounding_results: list[GroundingResult] = []
    converged = False

    for iteration in range(1, config.max_iterations + 1):
        logger.info(f"Verification iteration {iteration}/{config.max_iterations}")

        # Extract disease keywords from current RAG results
        keywords = extract_disease_keywords(current_items)
        all_keywords.extend(keywords)

        if not keywords:
            logger.info("No disease keywords found, marking all as verified")
            for item in current_items:
                item.verified = True
            converged = True
            break

        # Grounding verification
        grounding_results = verify_with_grounding(keywords, grounding_fn, image_path, config)
        all_grounding_results.extend(grounding_results)

        # Store previous state for convergence check
        prev_items = deepcopy(current_items)

        # Apply penalties and filter
        current_items = apply_penalty(current_items, grounding_results, config)

        # Check convergence
        if _check_convergence(prev_items, current_items):
            converged = True
            logger.info(f"Converged at iteration {iteration}")
            break

        # Supplementary query if items were removed and requery function available
        if len(current_items) < len(prev_items) and requery_fn is not None:
            try:
                supplementary = requery_fn(top_k=config.requery_top_k)
                for item in supplementary:
                    if item.source not in {r.source for r in current_items}:
                        current_items.append(item)
            except Exception as e:
                logger.warning(f"Supplementary query failed: {e}")

    # Mark surviving items as verified
    for item in current_items:
        if not item.penalty_applied:
            item.verified = True

    items_removed = initial_count - len(current_items)

    return VerificationResult(
        verified_items=current_items,
        iterations_run=iteration,
        keywords_checked=list(set(all_keywords)),
        grounding_results=all_grounding_results,
        items_removed=items_removed,
        converged=converged,
    )
