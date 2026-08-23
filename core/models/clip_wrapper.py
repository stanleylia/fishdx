"""CLIP Wrapper for embedding generation.

Wraps OpenCLIP ViT-B-32 (laion2b_s34b_b79k) for:
- Image embedding (512-dim)
- Text embedding (512-dim)
- Used by Fusion (Eq.4) and Pareidolia Soft Path (Eq.1b)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.config import CLIPConfig

logger = logging.getLogger(__name__)


class CLIPWrapper:
    """Wrapper for OpenCLIP model with lazy loading.

    Provides image and text embedding capabilities for:
    - λ-Weighted Fusion Embedding (Eq.4)
    - Pareidolia Soft Detection (Eq.1b)
    - ChromaDB vector storage and retrieval
    """

    def __init__(self, config: CLIPConfig) -> None:
        self.config = config
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load the CLIP model on first use."""
        if self._loaded:
            return

        try:
            import open_clip
            import torch

            logger.info(f"Loading CLIP: {self.config.model_name} ({self.config.pretrained})")

            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.config.model_name,
                pretrained=self.config.pretrained,
            )
            self._tokenizer = open_clip.get_tokenizer(self.config.model_name)

            device = self.config.device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, using CPU")
                device = "cpu"

            self._model = self._model.to(device)
            self._model.eval()
            self._loaded = True
            logger.info("CLIP loaded successfully")

        except ImportError as e:
            logger.error(f"Missing dependency for CLIP: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            raise

    def encode_image(self, image: Any) -> NDArray[np.float32]:
        """Encode an image to a 512-dim embedding.

        Args:
            image: PIL Image (RGB).

        Returns:
            512-dim numpy embedding vector (not normalized).
        """
        self._ensure_loaded()
        import torch

        device = next(self._model.parameters()).device
        image_tensor = self._preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = self._model.encode_image(image_tensor)

        return embedding.cpu().numpy().flatten().astype(np.float32)

    def encode_text(self, text: str) -> NDArray[np.float32]:
        """Encode text to a 512-dim embedding.

        Args:
            text: Input text string.

        Returns:
            512-dim numpy embedding vector (not normalized).
        """
        self._ensure_loaded()
        import torch

        device = next(self._model.parameters()).device
        tokens = self._tokenizer([text]).to(device)

        with torch.no_grad():
            embedding = self._model.encode_text(tokens)

        return embedding.cpu().numpy().flatten().astype(np.float32)

    def encode_texts(self, texts: list[str]) -> NDArray[np.float32]:
        """Batch encode multiple texts.

        Args:
            texts: List of text strings.

        Returns:
            Array of shape (N, 512) with embeddings.
        """
        self._ensure_loaded()
        import torch

        device = next(self._model.parameters()).device
        tokens = self._tokenizer(texts).to(device)

        with torch.no_grad():
            embeddings = self._model.encode_text(tokens)

        return embeddings.cpu().numpy().astype(np.float32)

    def compute_similarity(self, image: Any, texts: list[str]) -> NDArray[np.float32]:
        """Compute cosine similarity between an image and multiple texts.

        Args:
            image: PIL Image (RGB).
            texts: List of text candidates.

        Returns:
            1D array of similarity scores.
        """
        img_emb = self.encode_image(image)
        txt_embs = self.encode_texts(texts)

        # Normalize
        img_norm = img_emb / (np.linalg.norm(img_emb) + 1e-10)
        txt_norms = txt_embs / (np.linalg.norm(txt_embs, axis=1, keepdims=True) + 1e-10)

        similarities = txt_norms @ img_norm
        return similarities.astype(np.float32)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim
