"""fishdx — visual-embedding retrieval for training-free fish-disease classification.

Reference implementation accompanying Liao, Shih & Chang (2026).

The reported empirical results are produced by the analysis scripts in
``scripts/experiments/`` operating on the cached Florence-2/CLIP embeddings in
``results/_cache_*.npz`` (see REPRODUCIBILITY_MAP.md §3 and §9). This package
implements the same caption-based pareidolia guard, boundary-aware membership
scoring, logged auxiliary confidence rule and priority-ordered decision
semantics described in the manuscript. ``architecture.md`` documents module
boundaries.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__: list[str] = []
