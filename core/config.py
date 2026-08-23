"""Configuration management — Pydantic Settings v2 → YAML.

All hyperparameters are loaded from configs/default.yaml.
No hardcoded values allowed in business logic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from core.knowledge.negation_patterns import NEGATION_PATTERNS
from core.knowledge.scoring_indicators import DISEASE_INDICATORS, HEALTHY_INDICATORS


# ─── Sub-Config Models ──────────────────────────────

class SystemConfig(BaseModel):
    version: str = "3.0.0"
    environment: str = "development"
    log_level: str = "INFO"
    temp_dir: str = "/tmp/multimodal_rag"


class Florence2Config(BaseModel):
    model_name: str = "microsoft/Florence-2-base"
    device: str = "cuda"
    num_beams: int = 3
    max_new_tokens: int = 1024
    task_prompts: dict[str, list[str]] = Field(default_factory=dict)


class CLIPConfig(BaseModel):
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    embedding_dim: int = 512
    device: str = "cuda"


class ModelsConfig(BaseModel):
    florence2: Florence2Config = Field(default_factory=Florence2Config)
    clip: CLIPConfig = Field(default_factory=CLIPConfig)


class ChromaDBConfig(BaseModel):
    persist_directory: str = "/data/chroma_db"
    collection_name: str = "fish_disease_knowledge"
    distance_metric: str = "cosine"
    top_k: int = 5
    similarity_cutoff: float = 0.5


class PareidoliaConfig(BaseModel):
    """Configuration for Two-Tier Pareidolia Detection (Eq.1a, 1b)."""

    structural_keywords: list[str] = Field(default_factory=lambda: [
        "net", "fence", "mesh", "cage", "grid", "wire", "pen", "enclosure", "lattice", "netting"
    ])
    biological_keywords: list[str] = Field(default_factory=lambda: [
        "face", "person", "animal", "human", "head", "eye", "body"
    ])
    clip_threshold: float = 0.75  # τ
    centroid_cache: str = "/data/cache/pareidolia_centroids.pt"


class FusionConfig(BaseModel):
    """Configuration for λ-Weighted Fusion Embedding (Eq.4)."""

    lambda_weight: float = 0.7  # λ*


class SemanticFilterConfig(BaseModel):
    """Configuration for Dynamic Semantic Filter (Eq.8)."""

    gate_threshold: float = 0.10  # τ_gate (default.yaml scene_classification.rag_gate_threshold)
    small_set_threshold: int = 3
    rag_trigger_scenes: list[str] = Field(default_factory=lambda: ["fish"])
    # CLIP visual fallback for scene classification
    clip_scene_threshold: float = 0.24
    clip_fish_prompts: list[str] = Field(default_factory=lambda: [
        "a photo of a fish",
        "a photo of a fish in water",
        "a photo of a diseased fish",
        "a photo of fish in an aquaculture pond",
        "a photo of tilapia",
    ])


class VerificationConfig(BaseModel):
    """Configuration for Bidirectional Verification Loop (Algorithm 1)."""

    grounding_threshold: float = 0.5  # θ
    penalty_factor: float = 0.5
    min_score: float = 0.3
    max_iterations: int = 2
    requery_top_k: int = 3


class ScoringWeights(BaseModel):
    disease_confirmed: int = 3
    disease_suspected: int = 2
    disease_mentioned: int = 1
    healthy_explicit: int = 2
    healthy_negation: int = 1


class ScoringConfig(BaseModel):
    """Configuration for Scoring-Based Diagnosis (Eq.8-10, v20 §3.4)."""

    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    healthy_threshold: int = 2  # T_h (v20 Eq.10)
    inconclusive_margin: int = 1  # m (v20 Eq.10)
    negation_patterns: dict[str, list[str]] = Field(
        default_factory=lambda: {k: list(v) for k, v in NEGATION_PATTERNS.items()}
    )
    healthy_indicators: dict[str, list[str]] = Field(
        default_factory=lambda: {k: list(v) for k, v in HEALTHY_INDICATORS.items()}
    )
    disease_indicators: dict[str, list[str]] = Field(
        default_factory=lambda: {k: list(v) for k, v in DISEASE_INDICATORS.items()}
    )


class QualityGateConfig(BaseModel):
    """Configuration for Quality Gate (§3.7)."""

    min_confidence_score: int = 4
    require_verified: bool = True
    learning_queue_path: str = "/data/learning_queue/"


class LearningConfig(BaseModel):
    """Configuration for Adaptive Learning Layer — knowledge lifecycle management."""

    # Deduplication
    dedup_similarity_threshold: float = 0.92
    dedup_top_k: int = 5

    # Contradiction detection
    contradiction_similarity_threshold: float = 0.80
    contradiction_top_k: int = 10

    # Trust management
    initial_trust_score: float = 0.6
    seed_trust_score: float = 1.0
    trust_reinforcement_boost: int = 1
    trust_decay_rate: float = 0.01
    min_trust_for_retrieval: float = 0.3

    # Cleanup
    cleanup_interval_analyses: int = 50
    stale_days_threshold: int = 90
    min_trust_for_retention: float = 0.2
    max_collection_size: int = 10000

    # Feedback
    enable_feedback_tracking: bool = True


class FrameExtractionConfig(BaseModel):
    """Configuration for video frame extraction."""

    short_clip_fps: float = 1.0
    short_clip_max: int = 30
    medium_clip_fps: float = 0.1
    medium_clip_max: int = 50
    long_clip_scene_threshold: float = 0.3
    long_clip_max: int = 50
    size_boundary_medium_mb: int = 100
    size_boundary_large_mb: int = 1000


class DatasetConfig(BaseModel):
    """Configuration for lab dataset organization."""

    lab_dir: str = "lab_dateset"
    organized_dir: str = "lab_dateset/organized"
    manifest_path: str = "lab_dateset/organized/dataset_manifest.json"
    scenes_dir: str = "lab_dateset/organized/scenes"
    frame_extraction: FrameExtractionConfig = Field(default_factory=FrameExtractionConfig)


class ServiceEndpoint(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    total_timeout: int | None = None


class ServicesConfig(BaseModel):
    gateway: ServiceEndpoint = Field(default_factory=lambda: ServiceEndpoint(port=8000, total_timeout=60))
    perception: ServiceEndpoint = Field(default_factory=lambda: ServiceEndpoint(host="perception-svc", port=8002))
    embedding: ServiceEndpoint = Field(default_factory=lambda: ServiceEndpoint(host="embedding-svc", port=8003))
    reasoning: ServiceEndpoint = Field(default_factory=lambda: ServiceEndpoint(port=8004))
    learning: ServiceEndpoint = Field(default_factory=lambda: ServiceEndpoint(port=8005))


# ─── Root Config ────────────────────────────────────

class AppConfig(BaseModel):
    """Root application configuration.

    All hyperparameters flow from configs/default.yaml through this model.
    No business logic module should hardcode any value that exists here.
    """

    system: SystemConfig = Field(default_factory=SystemConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    chromadb: ChromaDBConfig = Field(default_factory=ChromaDBConfig)
    pareidolia: PareidoliaConfig = Field(default_factory=PareidoliaConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    semantic_filter: SemanticFilterConfig = Field(default_factory=SemanticFilterConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    quality_gate: QualityGateConfig = Field(default_factory=QualityGateConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    services: ServicesConfig = Field(default_factory=ServicesConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)


# ─── Loader ─────────────────────────────────────────

def _find_config_dir() -> Path:
    """Walk upward from this file to find the configs/ directory."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        candidate = current / "configs"
        if candidate.is_dir():
            return candidate
        current = current.parent
    return Path("configs")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    config_path: str | Path | None = None,
    environment: str | None = None,
) -> AppConfig:
    """Load application configuration from YAML files.

    Resolution order:
    1. configs/default.yaml (base)
    2. configs/{environment}.yaml (overlay, if exists)
    3. Explicit config_path (override, if provided)

    Args:
        config_path: Optional explicit path to a YAML config file.
        environment: Optional environment name (e.g., 'production', 'test').
            Falls back to MULTIMODAL_RAG_ENV env var, then 'development'.

    Returns:
        Fully validated AppConfig instance.
    """
    config_dir = _find_config_dir()

    # Load base config
    default_path = config_dir / "default.yaml"
    if default_path.exists():
        with open(default_path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
    else:
        data = {}

    # Determine environment
    env = environment or os.environ.get("MULTIMODAL_RAG_ENV", "development")

    # Overlay environment-specific config
    env_path = config_dir / f"{env}.yaml"
    if env_path.exists() and env != "development":
        with open(env_path) as f:
            env_data: dict[str, Any] = yaml.safe_load(f) or {}
        data = _deep_merge(data, env_data)

    # Override with explicit config path
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file) as f:
                override_data: dict[str, Any] = yaml.safe_load(f) or {}
            data = _deep_merge(data, override_data)

    return AppConfig(**data)


# ─── Singleton ──────────────────────────────────────

_config_instance: AppConfig | None = None


def get_config(
    config_path: str | Path | None = None,
    environment: str | None = None,
    force_reload: bool = False,
) -> AppConfig:
    """Get or create the singleton AppConfig instance.

    Args:
        config_path: Optional override config path.
        environment: Optional environment name.
        force_reload: Force re-reading config files.

    Returns:
        Singleton AppConfig instance.
    """
    global _config_instance
    if _config_instance is None or force_reload:
        _config_instance = load_config(config_path, environment)
    return _config_instance


def reset_config() -> None:
    """Reset the singleton config (for testing)."""
    global _config_instance
    _config_instance = None
