"""Scoring indicator vocabulary — reference data for Eq. 8-9 scoring.

Canonical keyword tiers consumed by ``core.algorithms.scoring``:

- ``HEALTHY_INDICATORS`` — explicit healthy-condition phrases (Eq. 8 first
  term), keyed by language.
- ``DISEASE_INDICATORS`` — disease evidence keyed by tier and language:
  ``confirmed_* > suspected_* > mentioned_*`` mapping to weights 3 / 2 / 1
  (Eq. 9). Token-level dedup keeps the highest tier per keyword.

These mirror the reference-data pattern already used by
``negation_patterns.py`` and ``pareidolia_keywords.py`` so that a default
``ScoringConfig`` is functional without an explicit YAML override; any
``configs/*.yaml`` ``scoring.{healthy,disease}_indicators`` block still
overrides these defaults.
"""

from __future__ import annotations

HEALTHY_INDICATORS: dict[str, list[str]] = {
    "zh": [
        "外觀健康", "未見疾病", "正常體色", "狀態良好", "活力充沛",
        "無異常", "無病徵", "外觀正常", "無明顯病灶", "偏向健康",
        "狀態正常", "魚體健康", "體色正常", "無病灶", "未見明顯病",
    ],
    "en": [
        "appears healthy", "good condition", "no disease", "no abnormalities",
        "no lesion", "fish healthy", "no visible disease", "no signs of disease",
        "healthy condition", "normal appearance",
    ],
}

DISEASE_INDICATORS: dict[str, list[str]] = {
    "confirmed_zh": ["確診", "診斷為"],
    "confirmed_en": ["confirmed", "diagnosed"],
    "suspected_zh": ["患有", "感染", "疑似"],
    "suspected_en": ["suspected", "infected", "likely"],
    "mentioned_zh": [
        "潰瘍", "出血", "紅斑", "充血", "病灶",
        "壞死", "腫脹", "發紅", "敗血", "爛鰭",
    ],
    "mentioned_en": [
        "red patches", "hemorrhage", "bleeding", "ulcer", "lesion",
        "necrosis", "reddening", "swelling", "erosion", "hemorrhagic",
        "red spots", "red discoloration", "skin redness",
    ],
}

__all__ = ["DISEASE_INDICATORS", "HEALTHY_INDICATORS"]
