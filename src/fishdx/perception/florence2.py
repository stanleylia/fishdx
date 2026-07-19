"""Florence-2 wrapper — paper §III.B, Skill S2 §2.1.

Loads ``microsoft/Florence-2-base`` at the SHA pinned in config, with
``trust_remote_code=True`` per ADR-0004 M1, and exposes a deterministic
caption + open-vocabulary detection interface.

Hard-coded HF generation kwargs (per Skill S2 §2.1):
    num_beams       = config.florence2.num_beams        (3)
    max_new_tokens  = config.florence2.max_new_tokens   (1024)
    do_sample       = False  (G2 determinism hard rule)

Task prompts (per paper §III.B):
    "<DENSE_CAPTION>"     → dense caption (field ``caption_raw``)
    "<OD>"                → open-vocabulary object detection
                             (populates ``objects_raw`` with label+bbox)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image

from fishdx.config import Florence2Config
from fishdx.errors import Florence2InferenceError, InputValidationError, RevisionPinError
from fishdx.schemas import DetectedObject

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

# Florence-2 caption task selection — REPAIR-1 fix, 2026-04-18
#
# Prior value ``<DENSE_CAPTION>`` is NOT a task token recognised by this
# model's ``processing_florence2.py::tasks_answer_post_processing_type``
# map. Verified at Diagnostic 3 (docs/m3/exp1_5/analysis.md): grep of the
# cached snapshot (5ca5edf5…) ships zero occurrences of ``DENSE_CAPTION``.
# Supported caption-tier tasks returning ``pure_text`` are
# ``<CAPTION>``, ``<DETAILED_CAPTION>``, ``<MORE_DETAILED_CAPTION>``.
# Paper §III.B's English phrase "dense caption" has no direct token
# counterpart in the shipped model; ``<MORE_DETAILED_CAPTION>`` is the
# closest semantic match (most descriptive of the three caption tiers).
#
# Consequence for M2 closure: the prior determinism verification was
# empty-output-trivially-identical. See docs/m2/m2-closure-amendment.md
# for the formal retraction + REPAIR re-certification record.
_TASK_CAPTION = "<MORE_DETAILED_CAPTION>"
_TASK_DETECTION = "<OD>"

_DTYPE_MAP = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}

_SHA_LENGTH = 40  # ADR-0004 M1 revision pin length
_BBOX_XYXY_LEN = 4  # Florence-2 <OD> bbox format


class Florence2Wrapper:
    """Deterministic Florence-2 caption + detection wrapper."""

    def __init__(self, config: Florence2Config) -> None:
        self._config = config
        self._model: PreTrainedModel | None = None
        self._processor: ProcessorMixin | None = None
        self._device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self._dtype = _DTYPE_MAP[config.precision]

    # ── lifecycle ───────────────────────────────────────────────
    def warmup(self) -> None:
        """Eagerly load weights + processor. Idempotent."""
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoProcessor

        # Enforce ADR-0004 M1: revision must already be SHA-validated by
        # Florence2Config's field validator (RevisionPinError path).
        if not self._config.revision or len(self._config.revision) != _SHA_LENGTH:
            raise RevisionPinError(
                "florence2.revision must be a 40-char SHA before load",
                context={"revision": self._config.revision},
            )

        self._processor = AutoProcessor.from_pretrained(
            self._config.model_id,
            revision=self._config.revision,
            trust_remote_code=self._config.trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._config.model_id,
            revision=self._config.revision,
            trust_remote_code=self._config.trust_remote_code,
            torch_dtype=self._dtype,
        ).to(self._device)
        self._model.eval()

    def close(self) -> None:
        """Release model + processor + CUDA cache."""
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── inference ────────────────────────────────────────────────
    @torch.inference_mode()
    def caption_and_detect(
        self, image_path: Path
    ) -> tuple[str, list[DetectedObject], dict[str, float]]:
        """Run `<DENSE_CAPTION>` then `<OD>` on ``image_path``.

        Returns ``(caption_raw, objects_raw, telemetry)``. Telemetry keys:
        ``caption_ms``, ``detect_ms``, ``vram_peak_mb``.
        """
        if self._model is None or self._processor is None:
            self.warmup()
        assert self._model is not None
        assert self._processor is not None

        if not image_path.exists():
            raise InputValidationError(
                f"image path does not exist: {image_path}",
                context={"path": str(image_path)},
            )
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise InputValidationError(
                f"failed to open image: {image_path}", context={"path": str(image_path)}
            ) from e

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self._device)

        try:
            t0 = time.perf_counter()
            caption_raw = self._run_task(image, _TASK_CAPTION)
            caption_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            detect_output = self._run_task_raw(image, _TASK_DETECTION)
            detect_ms = (time.perf_counter() - t0) * 1000
            objects_raw = self._parse_detection(detect_output)
        except Exception as e:
            raise Florence2InferenceError(
                f"Florence-2 inference failed: {e}",
                context={"path": str(image_path), "cause": str(e)},
            ) from e

        vram_peak_mb = (
            torch.cuda.max_memory_allocated(self._device) / (1024**2)
            if torch.cuda.is_available()
            else 0.0
        )
        telemetry = {
            "caption_ms": caption_ms,
            "detect_ms": detect_ms,
            "vram_peak_mb": vram_peak_mb,
        }
        return caption_raw, objects_raw, telemetry

    # ── internals ────────────────────────────────────────────────
    def _run_task(self, image: Image.Image, task_prompt: str) -> str:
        raw = self._run_task_raw(image, task_prompt)
        value = raw.get(task_prompt, "")
        return value if isinstance(value, str) else str(value)

    def _run_task_raw(self, image: Image.Image, task_prompt: str) -> dict[str, Any]:
        assert self._processor is not None
        assert self._model is not None

        inputs = self._processor(text=task_prompt, images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device, dtype=self._dtype)
        input_ids = inputs["input_ids"].to(self._device)

        generated_ids = self._model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=self._config.max_new_tokens,
            num_beams=self._config.num_beams,
            do_sample=self._config.do_sample,
        )
        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )
        return parsed  # type: ignore[no-any-return]

    @staticmethod
    def _parse_detection(parsed: dict[str, Any]) -> list[DetectedObject]:
        detection = parsed.get(_TASK_DETECTION, {})
        if not isinstance(detection, dict):
            return []
        bboxes = detection.get("bboxes", []) or []
        labels = detection.get("labels", []) or []
        objects: list[DetectedObject] = []
        for bbox, label in zip(bboxes, labels):  # noqa: B905
            if not isinstance(bbox, (list, tuple)) or len(bbox) != _BBOX_XYXY_LEN:
                continue
            objects.append(
                DetectedObject(
                    label=str(label),
                    bbox_xyxy=(
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ),
                    confidence=1.0,  # Florence-2 <OD> returns labels w/o confidence
                )
            )
        return objects


__all__ = ["Florence2Wrapper"]
