"""Dynamic Semantic Filter (v19 Eq.5; removed from v20 paper).

Adaptive Jaccard/Overlap scoring to determine if RAG retrieval
should be triggered for a given scene. Retained in codebase as
pipeline component though not included in v20 paper equations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from core.config import SemanticFilterConfig
from core.utils.text_utils import extract_keywords

logger = logging.getLogger(__name__)


@dataclass
class DomainProfile:
    """A domain profile containing keywords for a scene type."""

    name: str
    keywords: set[str]
    description: str = ""


@dataclass
class SceneClassification:
    """Result of scene classification via semantic filtering."""

    scene_type: str
    score: float
    rag_triggered: bool
    matched_keywords: list[str]
    profile_name: str


# ─── Default Domain Profiles ───────────────────────

DEFAULT_DOMAIN_PROFILES: dict[str, DomainProfile] = {
    "fish": DomainProfile(
        name="fish",
        keywords={
            # Core species & anatomy
            "fish", "tilapia", "grouper", "salmon", "trout", "shrimp", "prawn",
            "gill", "fin", "scale", "tail", "eye", "mouth", "body",
            "swim", "swimming", "school", "aquaculture", "pond",
            # Florence-2 descriptive vocabulary
            "underwater", "aquatic", "marine", "seafood",
            "jellyfish", "crab", "starfish",
            # Chinese
            "魚", "蝦", "養殖", "鰓", "鰭", "鱗",
        },
        description="Fish and aquaculture species",
    ),
    "disease": DomainProfile(
        name="disease",
        keywords={
            "disease", "infection", "lesion", "ulcer", "parasite", "fungus",
            "bacteria", "virus", "necrosis", "hemorrhage", "inflammation",
            "spot", "patch", "discoloration", "erosion", "swelling",
            "病", "感染", "潰瘍", "寄生蟲", "出血", "壞死",
            # Visual symptom descriptors
            "damaged", "abnormal", "wound", "sore", "blister", "rot",
            "white", "red", "dark", "cloudy", "bulging", "protruding",
            "異常", "損傷", "腐爛", "白點", "紅斑",
        },
        description="Fish disease indicators",
    ),
    "environment": DomainProfile(
        name="environment",
        keywords={
            "water", "tank", "cage", "net", "pen", "pond", "pipe", "filter",
            "aerator", "pump", "feed", "feeder", "sensor", "camera",
            "水", "池", "網", "箱", "管",
        },
        description="Aquaculture environment and equipment",
    ),
    "general": DomainProfile(
        name="general",
        keywords=set(),
        description="Default fallback scene",
    ),
}


def compute_score(keywords: list[str], profile: DomainProfile) -> float:
    """Compute adaptive Jaccard/Overlap score (Eq.8).

                |K ∩ Dⱼ|
    Score =     ─────────    if |K| > 3    (Jaccard Similarity)
                |K ∪ Dⱼ|

                |K ∩ Dⱼ|
             =  ──────────   if |K| ≤ 3    (Overlap Coefficient)
                min(|K|,|Dⱼ|)

    Args:
        keywords: Extracted keywords K from caption.
        profile: Domain profile Dⱼ.

    Returns:
        Semantic filter score.
    """
    if not keywords or not profile.keywords:
        return 0.0

    k_set = set(kw.lower() for kw in keywords)
    d_set = set(kw.lower() for kw in profile.keywords)

    intersection = k_set & d_set

    if not intersection:
        return 0.0

    if len(k_set) > 3:
        # Jaccard Similarity
        union = k_set | d_set
        return len(intersection) / len(union)
    else:
        # Overlap Coefficient (for small keyword sets)
        min_size = min(len(k_set), len(d_set))
        if min_size == 0:
            return 0.0
        return len(intersection) / min_size


def classify_scene(
    caption: str,
    config: SemanticFilterConfig,
    profiles: dict[str, DomainProfile] | None = None,
) -> SceneClassification:
    """Classify a scene by computing scores against all domain profiles.

    Args:
        caption: Input caption text.
        config: Semantic filter configuration.
        profiles: Domain profiles to match against.

    Returns:
        SceneClassification with best-matching profile.
    """
    if profiles is None:
        profiles = DEFAULT_DOMAIN_PROFILES

    keywords = extract_keywords(caption)

    best_score = 0.0
    best_profile = "general"
    best_matched: list[str] = []

    for name, profile in profiles.items():
        if name == "general":
            continue

        score = compute_score(keywords, profile)
        if score > best_score:
            best_score = score
            best_profile = name
            k_set = set(kw.lower() for kw in keywords)
            d_set = set(kw.lower() for kw in profile.keywords)
            best_matched = list(k_set & d_set)

    # Fall back to general if no profile matches well enough
    if best_score < config.gate_threshold:
        best_profile = "general"
        best_matched = []

    # Determine RAG trigger
    rag_triggered = should_trigger_rag(best_profile, best_score, config)

    logger.info(
        f"Scene classified: {best_profile} (score={best_score:.3f}, "
        f"rag_triggered={rag_triggered})"
    )

    return SceneClassification(
        scene_type=best_profile,
        score=best_score,
        rag_triggered=rag_triggered,
        matched_keywords=best_matched,
        profile_name=best_profile,
    )


def should_trigger_rag(
    scene_type: str,
    score: float,
    config: SemanticFilterConfig,
) -> bool:
    """Determine if RAG retrieval should be triggered.

    RAG is triggered when:
    1. Score exceeds gate_threshold AND
    2. Scene type is in rag_trigger_scenes list

    Args:
        scene_type: Classified scene type.
        score: Semantic filter score.
        config: Configuration.

    Returns:
        Whether RAG should be triggered.
    """
    if score < config.gate_threshold:
        return False

    # Check if scene type matches any trigger scene
    return any(
        trigger in scene_type.lower()
        for trigger in config.rag_trigger_scenes
    )


def classify_scene_with_clip(
    caption: str,
    config: SemanticFilterConfig,
    clip_similarity_fn: Callable[[Any, list[str]], Any] | None = None,
    pil_image: Any = None,
    profiles: dict[str, DomainProfile] | None = None,
) -> SceneClassification:
    """Scene classification with CLIP visual fallback.

    When Florence-2 produces non-fish captions (e.g. "green vase"),
    text-based classification returns "general". CLIP can independently
    verify whether the image content is fish-related.

    1. First tries text-based classify_scene()
    2. If result is "general" AND clip_similarity_fn + image available,
       computes CLIP similarity against fish reference prompts
    3. If max similarity >= clip_scene_threshold, overrides to "fish"

    Args:
        caption: Input caption text.
        config: Semantic filter configuration.
        clip_similarity_fn: CLIP compute_similarity(image, texts) -> 1D array.
        pil_image: PIL Image for CLIP inference.
        profiles: Domain profiles to match against.

    Returns:
        SceneClassification, potentially overridden by CLIP.
    """
    # Step 1: text-based classification
    text_result = classify_scene(caption, config, profiles)

    # Step 2: CLIP fallback when text-based returns "general"
    if (
        text_result.scene_type == "general"
        and clip_similarity_fn is not None
        and pil_image is not None
    ):
        try:
            fish_prompts = config.clip_fish_prompts
            similarities = clip_similarity_fn(pil_image, fish_prompts)
            max_sim = float(max(similarities))
            best_idx = int(similarities.argmax()) if hasattr(similarities, "argmax") else 0

            logger.info(
                f"CLIP scene fallback: max_sim={max_sim:.4f}, "
                f"prompt='{fish_prompts[best_idx]}', "
                f"threshold={config.clip_scene_threshold}"
            )

            if max_sim >= config.clip_scene_threshold:
                logger.info(
                    f"CLIP override: general -> fish "
                    f"(sim={max_sim:.4f} >= {config.clip_scene_threshold})"
                )
                return SceneClassification(
                    scene_type="fish",
                    score=max_sim,
                    rag_triggered=True,
                    matched_keywords=[f"clip:{fish_prompts[best_idx]}"],
                    profile_name="fish",
                )
        except Exception as e:
            logger.warning(f"CLIP scene fallback error: {e}")

    return text_result
