"""OpenCLIP encoder + Embedder Protocol — Skill S3 §2.1 hard binding.

Exposes :class:`EmbedderProtocol` for research-extension visibility
(SigLIP, DINOv2, EVA-02 per paper §VI Future Work) but production
binding is to :class:`OpenClipEmbedder` with ``ViT-B-32`` +
``laion2b_s34b_b79k``. **No config-driven runtime switching in M1-M4**;
any change to the embedder requires a new ADR (Q3 resolution).

Implementation notes
--------------------
- Model / tokenizer / image-preprocess loaded lazily inside :meth:`warmup`
  so pytest collection does not pay the download cost.
- Text encoding uses **no prompt template** on the fusion path (Skill S3
  §2.2). The 7 zero-shot templates in ``configs/default.yaml`` are
  reserved for the EXP-2 baseline experiment only.
- Encoder inference runs in fp16 on GPU (per config ``clip.precision``),
  but outputs are always upcast to Python ``float`` for Eq. 5–7 fusion,
  which runs in fp32 for numerical stability.
- Every embedding is L2-normalised on return (``text_normalize``,
  ``image_normalize`` are always ``"l2"`` in Table IX).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch

from fishdx.config import ClipConfig
from fishdx.errors import ClipEncodingError, FusionNormalizationError

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image as PILImage

_DTYPE_MAP = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
_ZERO_NORM_EPS = 1e-12


class EmbedderProtocol(Protocol):
    """Text + image embedder contract — NOT swappable via config."""

    @property
    def embedding_dim(self) -> int: ...

    def encode_text(self, texts: Sequence[str]) -> list[list[float]]: ...

    def encode_image_paths(self, paths: Sequence[str]) -> list[list[float]]: ...


class OpenClipEmbedder:
    """OpenCLIP ViT-B-32 + laion2b_s34b_b79k (Skill S3 §2.1)."""

    def __init__(
        self,
        config: ClipConfig | None = None,
        *,
        architecture: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ) -> None:
        if config is not None:
            self._architecture = config.architecture
            self._pretrained = config.pretrained
            self._dtype = _DTYPE_MAP[config.precision]
        else:
            self._architecture = architecture
            self._pretrained = pretrained
            self._dtype = torch.float16
        self._device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None

    @property
    def embedding_dim(self) -> int:
        return 512  # ViT-B-32 invariant

    def warmup(self) -> None:
        """Load model, preprocess, tokenizer. Idempotent."""
        if self._model is not None:
            return
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            self._architecture, pretrained=self._pretrained
        )
        self._model = model.to(self._device).eval()
        if self._device.type == "cuda" and self._dtype == torch.float16:
            self._model = self._model.half()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(self._architecture)

    def close(self) -> None:
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── text ──────────────────────────────────────────────────────
    @torch.inference_mode()
    def encode_text(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        self.warmup()
        try:
            tokens = self._tokenizer(list(texts)).to(self._device)
            feats = self._model.encode_text(tokens)
            feats = feats.float()  # upcast to fp32 for L2 normalisation
            feats = self._l2_normalize(feats)
        except FusionNormalizationError:
            raise
        except Exception as e:
            raise ClipEncodingError(
                f"encode_text failed: {e}", context={"n_texts": len(texts), "cause": str(e)}
            ) from e
        return feats.cpu().tolist()

    # ── image ─────────────────────────────────────────────────────
    @torch.inference_mode()
    def encode_image_paths(self, paths: Sequence[str]) -> list[list[float]]:
        if not paths:
            return []
        self.warmup()
        from PIL import Image

        batch: list[torch.Tensor] = []
        try:
            for p in paths:
                img = Image.open(Path(p)).convert("RGB")
                batch.append(self._preprocess(img))
            stacked = torch.stack(batch, dim=0).to(self._device, dtype=self._dtype)
            feats = self._model.encode_image(stacked)
            feats = feats.float()
            feats = self._l2_normalize(feats)
        except FusionNormalizationError:
            raise
        except Exception as e:
            raise ClipEncodingError(
                f"encode_image_paths failed: {e}",
                context={"n_paths": len(paths), "cause": str(e)},
            ) from e
        return feats.cpu().tolist()

    @torch.inference_mode()
    def encode_images(self, pil_images: Sequence[PILImage]) -> list[list[float]]:
        """Encode already-loaded PIL images (avoids re-reading from disk)."""
        if not pil_images:
            return []
        self.warmup()
        try:
            stacked = torch.stack([self._preprocess(img) for img in pil_images], dim=0).to(
                self._device, dtype=self._dtype
            )
            feats = self._model.encode_image(stacked)
            feats = feats.float()
            feats = self._l2_normalize(feats)
        except FusionNormalizationError:
            raise
        except Exception as e:
            raise ClipEncodingError(
                f"encode_images failed: {e}",
                context={"n_images": len(pil_images), "cause": str(e)},
            ) from e
        return feats.cpu().tolist()

    @staticmethod
    def _l2_normalize(t: torch.Tensor) -> torch.Tensor:
        norms = t.norm(dim=-1, keepdim=True)
        if torch.any(norms < _ZERO_NORM_EPS):
            raise FusionNormalizationError(
                "CLIP produced zero-norm embedding",
                context={"min_norm": float(norms.min())},
            )
        return t / norms


__all__ = ["EmbedderProtocol", "OpenClipEmbedder"]
