"""Caption post-processing tests — M2.2.a clean_caption().

Covers Florence-2 task-token residue + <loc_N> stripping.
"""

from __future__ import annotations

import pytest

from fishdx.perception.postprocess import clean_caption


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(
            "ENSE_CAPTION<loc_0><loc_0><loc_998><loc_998>A fish with lesions.",
            "A fish with lesions.",
            id="CLN1-transformers-4.46-residue",
        ),
        pytest.param(
            "<DENSE_CAPTION>A healthy fish.",
            "A healthy fish.",
            id="CLN2-full-task-token",
        ),
        pytest.param(
            "<loc_12><loc_34>ulcer on caudal peduncle<loc_56>",
            "ulcer on caudal peduncle",
            id="CLN3-locator-only",
        ),
        pytest.param(
            "   ENSE_CAPTION   <loc_0>   healthy   fish   ",
            "healthy fish",
            id="CLN4-whitespace-collapsing",
        ),
        pytest.param(
            "fish with white cotton-like growth",
            "fish with white cotton-like growth",
            id="CLN5-already-clean-idempotent",
        ),
    ],
)
def test_clean_caption(raw: str, expected: str) -> None:
    """Case CLN{1-5} — clean_caption removes residue + preserves clinical content.

    CLN1 reproduces the exact residue pattern observed during M2.1
    Florence-2 empirical validation.
    """
    assert clean_caption(raw) == expected
