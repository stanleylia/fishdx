"""Florence-2 output post-processing — M2.2.a caption cleanup.

Strips Florence-2 task-token residue and spatial-grounding locator tokens
from decoded caption strings so downstream CLIP text encoding receives the
clinical payload cleanly.

Empirical observation recorded under ADR-0009 V2 (stack-version validation)
-----------------------------------------------------------------------------
Under ``transformers 4.46.3`` (per ADR-0009 widened pin), Florence-2
``AutoProcessor.batch_decode(..., skip_special_tokens=False)`` emits the
task token with a leading letter truncated — e.g. ``<DENSE_CAPTION>`` comes
back as ``ENSE_CAPTION``. The same call under the original ``transformers
==4.44.*`` pin reportedly emitted the full token (per HF release notes
4.45 Generation API change-log). This is a benign decoding-layer
difference: it does not affect the generated token IDs, only how
``batch_decode`` renders them. Both the full and partial residues are
stripped below by design, so downstream captions are identical across
transformers versions.

If ADR-0009 V2 investigation later finds EXP-2 Florence-2-only DA
deviating > 1 pp from paper 0.037 and attribution points at generation
rather than decoding, revisit this docstring.
"""

from __future__ import annotations

import re

# Known Florence-2 task tokens (HF model-card reference).
_TASK_TOKENS: tuple[str, ...] = (
    "<DENSE_CAPTION>",
    "<OD>",
    "<CAPTION>",
    "<DETAILED_CAPTION>",
    "<MORE_DETAILED_CAPTION>",
    "<CAPTION_TO_PHRASE_GROUNDING>",
    "<REGION_TO_DESCRIPTION>",
    "<OPEN_VOCABULARY_DETECTION>",
    "<DENSE_REGION_CAPTION>",
)

# Partial-decode variants observed under transformers 4.46.3 (see module
# docstring). Full list derived by dropping 0–2 leading chars from each
# task token name plus an optional trailing ``>``.
_PARTIAL_RESIDUES: tuple[str, ...] = (
    "DENSE_CAPTION>",
    "ENSE_CAPTION>",
    "NSE_CAPTION>",
    "DENSE_CAPTION",
    "ENSE_CAPTION",
    "NSE_CAPTION",
    "CAPTION>",
    "CAPTION",
    "D>",
    "DETAILED_CAPTION>",
    "ETAILED_CAPTION>",
    "TAILED_CAPTION>",
    "MORE_DETAILED_CAPTION>",
    "ORE_DETAILED_CAPTION>",
)

_LOC_TOKEN_RE = re.compile(r"<loc_\d+>")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_caption(raw: str) -> str:
    """Strip Florence-2 task-token residue + ``<loc_N>`` markers.

    Parameters
    ----------
    raw : str
        Caption as returned by ``Florence2Wrapper._run_task`` (may carry
        leading task-token residue and interleaved ``<loc_N>`` spatial
        grounding tokens).

    Returns
    -------
    str
        Caption with task-token residue removed, all ``<loc_N>`` markers
        stripped, and whitespace collapsed. Never returns ``None``; an
        empty string is valid if the raw caption was all residue.
    """
    cleaned = raw
    # Remove full task tokens first (longest match wins).
    for token in _TASK_TOKENS:
        cleaned = cleaned.replace(token, "")
    # Remove partial decode residues (longest prefix first).
    for partial in _PARTIAL_RESIDUES:
        cleaned = cleaned.replace(partial, "")
    # Strip locator tokens.
    cleaned = _LOC_TOKEN_RE.sub("", cleaned)
    # Collapse whitespace and trim.
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


__all__ = ["clean_caption"]
