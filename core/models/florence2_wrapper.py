"""Florence-2 VLM Wrapper for Stage 1 Perception.

Wraps Microsoft Florence-2-base (0.23B) for:
- Detailed Caption (<DETAILED_CAPTION>)
- Object Detection (<OD>)
- Phrase Grounding (<CAPTION_TO_PHRASE_GROUNDING>)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from core.config import Florence2Config

logger = logging.getLogger(__name__)


class Florence2Wrapper:
    """Wrapper for Florence-2 VLM with lazy loading.

    Provides caption generation, object detection, and phrase grounding
    capabilities for the perception layer (Stage 1).
    """

    def __init__(self, config: Florence2Config) -> None:
        self.config = config
        self._model: Any = None
        self._processor: Any = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load the model and processor on first use."""
        if self._loaded:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            logger.info(f"Loading Florence-2: {self.config.model_name}")
            self._processor = AutoProcessor.from_pretrained(
                self.config.model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name, trust_remote_code=True
            )

            if self.config.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self._model = self._model.to("cuda")
                else:
                    logger.warning("CUDA not available, using CPU")
                    self._model = self._model.to("cpu")
            else:
                self._model = self._model.to(self.config.device)

            self._loaded = True
            logger.info("Florence-2 loaded successfully")

        except ImportError as e:
            logger.error(f"Missing dependency for Florence-2: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load Florence-2: {e}")
            raise

    def _run_inference(self, image: Any, task_prompt: str, text_input: str = "") -> dict:
        """Run a single Florence-2 inference task.

        Args:
            image: PIL Image.
            task_prompt: Task prompt (e.g., '<DETAILED_CAPTION>').
            text_input: Optional text input for conditional tasks.

        Returns:
            Parsed result dictionary.
        """
        self._ensure_loaded()
        import torch

        prompt = task_prompt if not text_input else f"{task_prompt}{text_input}"
        inputs = self._processor(text=prompt, images=image, return_tensors="pt")

        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                num_beams=self.config.num_beams,
            )

        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        result = self._processor.post_process_generation(
            generated_text, task=task_prompt, image_size=image.size
        )
        return result

    def generate_caption(self, image: Any) -> str:
        """Generate a detailed caption for an image.

        Uses <DETAILED_CAPTION> task prompt.

        Args:
            image: PIL Image (RGB).

        Returns:
            Detailed caption string.
        """
        result = self._run_inference(image, "<DETAILED_CAPTION>")
        return result.get("<DETAILED_CAPTION>", "")

    def detect_objects(self, image: Any) -> list[dict]:
        """Detect objects in an image.

        Uses <OD> task prompt.

        Args:
            image: PIL Image (RGB).

        Returns:
            List of {label, bbox, confidence} dicts.
        """
        result = self._run_inference(image, "<OD>")
        od_result = result.get("<OD>", {})

        objects = []
        labels = od_result.get("labels", [])
        bboxes = od_result.get("bboxes", [])

        for label, bbox in zip(labels, bboxes):
            objects.append({
                "label": label,
                "bbox": bbox,
                "confidence": 1.0,  # Florence-2 OD doesn't return confidence
            })
        return objects

    def phrase_grounding(self, image: Any, phrase: str) -> tuple[float, list[float]]:
        """Ground a phrase in an image using Florence-2.

        Uses <CAPTION_TO_PHRASE_GROUNDING> task prompt.
        Used by Algorithm 1 (Bidirectional Verification).

        Args:
            image: PIL Image (RGB).
            phrase: Text phrase to ground.

        Returns:
            Tuple of (confidence, bbox) where bbox is [x1, y1, x2, y2].
        """
        result = self._run_inference(
            image, "<CAPTION_TO_PHRASE_GROUNDING>", text_input=phrase
        )
        grounding = result.get("<CAPTION_TO_PHRASE_GROUNDING>", {})

        bboxes = grounding.get("bboxes", [])
        labels = grounding.get("labels", [])

        if bboxes:
            # Return the first matching bounding box
            bbox = bboxes[0] if isinstance(bboxes[0], list) else list(bboxes[0])
            confidence = 1.0 if labels else 0.5
            return confidence, bbox
        else:
            return 0.0, []

    def perceive(self, image: Any) -> dict:
        """Run full perception pipeline: caption + object detection.

        Args:
            image: PIL Image (RGB).

        Returns:
            Dict with 'caption' and 'objects' keys.
        """
        caption = self.generate_caption(image)
        objects = self.detect_objects(image)
        return {
            "caption": caption,
            "objects": objects,
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded
