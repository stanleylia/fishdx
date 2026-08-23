"""Cache-level regression tests for the released deposit.

Locks the reported KPIs, the deterministic k-NN tie-break, config hyperparameters, McNemar
2x2 row sums, and cache shapes so future edits cannot silently drift the numbers. Run: pytest -q
"""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def _load(name):
    return json.loads((RES / name).read_text())


def test_config_hyperparameters():
    """The shipped configs/default.yaml carries the paper's Table IX hyperparameters."""
    import yaml

    y = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    assert y["meta"]["seed"] == 42
    assert y["fusion"]["lambda_star"] == 0.7
    assert y["retrieval"]["top_k"] == 5
    assert y["retrieval"]["similarity_cutoff"] == 0.25
    assert y["margin"]["retrieval_margin_theta"] == 0.02
    assert y["decision"]["healthy_threshold_Th"] == 2
    assert y["decision"]["inconclusive_margin_m"] == 1


def test_d2final_k5_tiebreak_deterministic():
    """The k=5 majority-vote must use a float64 similarity tie-break and reproduce 0.786/0.868."""
    g = np.load(RES / "_cache_d1_gallery.npz", allow_pickle=True)
    q = np.load(RES / "_cache_d2_query.npz", allow_pickle=True)
    gv, gc, gl = g["visual"], g["caption"], g["labels"]
    qv, qc, ql = q["visual"], q["caption"], q["labels"]

    def l2(a):
        return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)

    clean = (qv @ gv.T).max(1) < 0.95
    qv, qc, ql = qv[clean], qc[clean], ql[clean]
    eus = np.where(ql == "EUS")[0]
    rng = np.random.default_rng(42)
    cal = eus[rng.permutation(len(eus))[: len(eus) // 2]]
    final = np.ones(len(ql), bool)
    final[cal] = False
    shared = sorted((set(gl.tolist()) & set(ql.tolist())) - {"EUS", "UNKNOWN"})
    m7 = np.isin(ql, shared)

    G, Q = l2(0.7 * gv + 0.3 * gc), l2(0.7 * qv + 0.3 * qc)
    s = Q @ G.T
    ix = np.argpartition(-s, 5, axis=1)[:, :5]
    pred = []
    for i in range(len(Q)):
        u = {}
        for lab, sc in zip(gl[ix[i]], s[i, ix[i]]):
            u[lab] = u.get(lab, 0.0) + 1.0 + 1e-6 * float(sc)  # float64 tie-break
        pred.append(max(u, key=u.get))
    pred = np.array(pred)
    ok = pred == ql
    overall = round(int(ok[final].sum()) / int(final.sum()), 4)
    overlap = round(int(ok[m7].sum()) / int(m7.sum()), 4)
    assert overall == 0.786, overall
    assert round(overlap, 3) == 0.868, overlap


def test_mcnemar_rowsums():
    d = _load("d2_clean_mcnemar_k5.json")
    c = d["contingency_2x2"]
    a, b, cc, dd = c["both_correct"], c["l07_correct_l10_wrong"], c["l07_wrong_l10_correct"], c["both_wrong"]
    assert a + b + cc + dd == d["n"] == 2628
    assert b == d["b"] and cc == d["c"]


def test_caption_to_kb_baseline():
    d = _load("caption_to_kb_baseline.json")
    assert d["variants"]["full_document"]["DA_all_8_classes"] == 0.1672
    assert d["majority_class_DA"] == 0.1789


def test_florence2_only_baseline():
    d = _load("florence2_only_baseline.json")
    assert d["DA_8_classes"] == 0.0423
    assert d["DA_7_shared"] == 0.0357


def test_cache_shapes():
    g = np.load(RES / "_cache_d1_gallery.npz", allow_pickle=True)
    q = np.load(RES / "_cache_d2_query.npz", allow_pickle=True)
    assert g["visual"].shape[1] == 512 and g["caption"].shape[1] == 512
    assert q["visual"].shape[0] == q["labels"].shape[0]
    assert g["visual"].shape[0] == g["labels"].shape[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
