"""Reviewer 5 comment 3(3) — evidence-layer ablations over the Stage-3 document.

"Separate Stage 3 into 'retrieval candidate generation' and 'independent evidence
verification', or add ablations that remove the top-1 document, substitute documents, and
use random documents."

We take the second option. The three ablations the reviewer names are implemented, plus an
oracle upper bound that makes the results interpretable:

  A  remove   — 𝒪 = caption only (the top-1 document is withheld)
  B  deployed — 𝒪 = caption + the retrieved top-1 class's document          (baseline)
  C  substitute — 𝒪 = caption + a deterministically chosen wrong-class document
  D  random   — 𝒪 = caption + a randomly drawn non-top-1 document, over 5 seeds
  F  oracle   — 𝒪 = caption + the *true* class's document (upper bound; not deployable)

In C, D and F the substituted document supplies **both** the evidence text and the keyword
tiers, which is what "substituting the document" means in the deployed design where the
top-1 doc_id resolves to the tiers used by Eq. (9).

Retrieval is held fixed in every condition: the candidate class c1 and the retrieval margin
come from the archived per-sample records of comment 2(1). Only the evidence pool changes.
The commitment rule is the one used throughout the reported experiments,
``committed = "healthy" if S_h > S_d else c1``.

Everything runs on CPU from archived captions and embeddings; no model is re-run.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/experiments"))

from fishdx.config import load_config  # noqa: E402
from fishdx.scoring.score import compute_s_d, compute_s_h  # noqa: E402

RES = ROOT / "results"
R5 = RES / "reviewer5"
OUT = R5 / "stage3_document_ablation.json"

THETA = 0.02
RANDOM_SEEDS = (42, 43, 44, 45, 46)   # condition D, protocol: >= 5 seeds


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 4), round(float(c + h), 4)]


def main() -> None:
    from exp_mcnemar_d2_faithful_O import load_kb_tiers

    cfg = load_config(ROOT / "configs/default.yaml")
    tiers, healthy_kw, doc_text = load_kb_tiers()
    kb_classes = sorted(tiers)

    recs = [json.loads(l) for l in (R5 / "selective_predictions_d2final.jsonl").open()]
    pipe = [r for r in recs if r["method"] == "pipeline"]
    caps = {c["image"]: (c.get("caption") or "")
            for c in json.loads((RES / "e3_captions_log.json").read_text())}

    n = len(pipe)
    print("=" * 100)
    print("Reviewer 5 / 3(3) — Stage-3 evidence-pool document ablations")
    print(f"  n = {n:,} D2-final samples | KB classes = {len(kb_classes)} | theta = {THETA}")
    print("=" * 100)

    c1 = [r["top1_gallery_class"] for r in pipe]
    truth = [r["true_label"] for r in pipe]
    margin = np.array([r["retrieval_margin"] for r in pipe])
    caption = [caps.get(r["sample_id"], "") for r in pipe]
    decided_mask = margin >= THETA

    # Substituting the *Healthy* document is qualitatively different from substituting
    # another disease document: the Healthy document supplies healthy-tier keywords and
    # negation patterns instead of disease tiers, so S_d collapses and the healthy gate
    # fires mechanically. We therefore report both a full-pool variant (any wrong class)
    # and a disease-only variant that isolates the question the reviewer is asking —
    # whether the *disease identity* of the document drives the output.
    disease_classes = [c for c in kb_classes if c not in ("healthy", "EUS")]

    def wrong_class_of(c: str, pool: list[str]) -> str:
        """Deterministic substitute: the next class in `pool` after c (never equals c)."""
        if c in pool:
            return pool[(pool.index(c) + 1) % len(pool)]
        return pool[0]

    def score(evidence: str, tier_class: str) -> tuple[int, int]:
        s_h = compute_s_h(evidence, cfg.scoring.healthy_weights, cfg.negation,
                          healthy_keywords=healthy_kw or ("healthy",))
        t = tiers.get(tier_class, {"confirmed": [], "suspected": [], "mentioned": []})
        s_d = compute_s_d(evidence, cfg.scoring, confirmed_keywords=t["confirmed"],
                          suspected_keywords=t["suspected"], mentioned_keywords=t["mentioned"])
        return int(s_h), int(s_d)

    def run(doc_class_of, label: str) -> dict:
        """doc_class_of(i) -> class supplying evidence text AND tiers, or None for caption-only."""
        committed, sh, sd = [], [], []
        for i in range(n):
            dc = doc_class_of(i)
            ev = caption[i] + ("\n" + doc_text.get(dc, "") if dc else "")
            tier_class = dc if dc else c1[i]
            a, b = score(ev, tier_class)
            sh.append(a); sd.append(b)
            committed.append("healthy" if a > b else c1[i])
        committed = np.array(committed)
        corr = committed == np.array(truth)
        agree_c1 = float((committed == np.array(c1)).mean())
        dec_k = int(corr[decided_mask].sum()); dec_n = int(decided_mask.sum())
        return {
            "condition": label,
            "agreement_with_retrieval_top1": round(agree_c1, 4),
            "n_overridden_to_healthy": int((committed == "healthy").sum()
                                           - sum(1 for x in c1 if x == "healthy")),
            "overall_DA": round(float(corr.mean()), 4),
            "overall_wilson95": wilson(int(corr.sum()), n),
            "decisive_DA": round(dec_k / dec_n, 4),
            "decisive_wilson95": wilson(dec_k, dec_n),
            "coverage": round(dec_n / n, 4),
            "S_h": {"mean": round(float(np.mean(sh)), 3), "median": float(np.median(sh)),
                    "max": int(np.max(sh))},
            "S_d": {"mean": round(float(np.mean(sd)), 3), "median": float(np.median(sd)),
                    "max": int(np.max(sd))},
            "frac_S_h_gt_S_d": round(float(np.mean(np.array(sh) > np.array(sd))), 4),
            "committed_distribution": dict(Counter(committed.tolist()).most_common()),
        }

    conditions = {}
    print("\n  running conditions ...", flush=True)
    conditions["A_remove_top1_document"] = run(lambda i: None, "A — caption only (document removed)")
    conditions["B_deployed_top1_document"] = run(lambda i: c1[i], "B — caption + top-1 document (deployed)")
    conditions["C_substitute_wrong_class_any"] = run(
        lambda i: wrong_class_of(c1[i], kb_classes),
        "C — caption + fixed wrong-class document (any KB class)")
    conditions["C2_substitute_wrong_disease"] = run(
        lambda i: wrong_class_of(c1[i], disease_classes),
        "C2 — caption + fixed wrong-DISEASE document")
    conditions["F_oracle_true_class"] = run(lambda i: truth[i],
                                            "F — caption + true-class document (oracle)")

    # ---- D: random non-top-1 document over several seeds ------------------------
    def random_runs(base_pool: list[str], tag: str) -> list[dict]:
        out_runs = []
        for sd_seed in RANDOM_SEEDS:
            rng = np.random.default_rng(sd_seed)
            choice = []
            for i in range(n):
                pool = [c for c in base_pool if c != c1[i]] or base_pool
                choice.append(pool[int(rng.integers(0, len(pool)))])
            out_runs.append(run(lambda i, ch=choice: ch[i],
                                f"{tag} (seed {sd_seed})"))
        return out_runs

    d_runs = random_runs(kb_classes, "D — random non-top-1 document (any KB class)")
    d2_runs = random_runs(disease_classes, "D2 — random non-top-1 DISEASE document")
    def agg(key: str, runs=None) -> dict:
        vals = [r[key] for r in (runs if runs is not None else d_runs)]
        return {"mean": round(float(np.mean(vals)), 4), "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4), "per_seed": vals}
    conditions["D_random_non_top1"] = {
        "condition": "D — caption + random non-top-1 document",
        "seeds": list(RANDOM_SEEDS),
        "agreement_with_retrieval_top1": agg("agreement_with_retrieval_top1"),
        "overall_DA": agg("overall_DA"),
        "decisive_DA": agg("decisive_DA"),
        "frac_S_h_gt_S_d": agg("frac_S_h_gt_S_d"),
        "per_seed_detail": d_runs,
    }
    conditions["D2_random_non_top1_disease"] = {
        "condition": "D2 — caption + random non-top-1 DISEASE document",
        "seeds": list(RANDOM_SEEDS),
        "agreement_with_retrieval_top1": agg("agreement_with_retrieval_top1", d2_runs),
        "overall_DA": agg("overall_DA", d2_runs),
        "decisive_DA": agg("decisive_DA", d2_runs),
        "frac_S_h_gt_S_d": agg("frac_S_h_gt_S_d", d2_runs),
        "per_seed_detail": d2_runs,
    }

    # ---- structural observation --------------------------------------------------
    distinct_disease_labels = {c for cond in ("A_remove_top1_document", "B_deployed_top1_document",
                                              "C2_substitute_wrong_disease", "F_oracle_true_class")
                               for c in conditions[cond]["committed_distribution"]}
    structural = {
        "claim": "the evidence layer cannot change the retrieved disease identity",
        "reason": ("the commitment rule is 'healthy if S_h > S_d else c1'; the disease label "
                   "always comes from the Stage-2 retrieval candidate c1, so varying the "
                   "document can only move a sample between c1 and Healthy, never between "
                   "two diseases"),
        "labels_ever_committed": sorted(distinct_disease_labels),
    }

    out = {
        "experiment": "r5_stage3_document_ablation",
        "reviewer_comment": "5-3(3): remove / substitute / random document ablations",
        "design": ("retrieval held fixed from the comment-2(1) archive; only the evidence "
                   "pool varies; substituted documents supply both text and keyword tiers"),
        "commitment_rule": "healthy if S_h > S_d else c1",
        "n_d2final": n, "theta_margin": THETA,
        "models_rerun": False,
        "conditions": conditions,
        "structural_finding": structural,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    hdr = f"  {'condition':<44}{'agree c1':>10}{'S_h>S_d':>10}{'Overall DA':>12}{'Decisive DA':>13}"
    print("\n" + hdr); print("  " + "-" * (len(hdr) - 2))
    for key in ("A_remove_top1_document", "B_deployed_top1_document",
                "C_substitute_wrong_class_any", "C2_substitute_wrong_disease",
                "F_oracle_true_class"):
        v = conditions[key]
        print(f"  {v['condition']:<44}{v['agreement_with_retrieval_top1']:>10.4f}"
              f"{v['frac_S_h_gt_S_d']:>10.4f}{v['overall_DA']:>12.4f}{v['decisive_DA']:>13.4f}")
    for dk in ("D_random_non_top1", "D2_random_non_top1_disease"):
        d = conditions[dk]
        print(f"  {d['condition']:<44}{d['agreement_with_retrieval_top1']['mean']:>10.4f}"
              f"{d['frac_S_h_gt_S_d']['mean']:>10.4f}{d['overall_DA']['mean']:>12.4f}"
              f"{d['decisive_DA']['mean']:>13.4f}")
    print(f"\n  labels ever committed across conditions: {structural['labels_ever_committed']}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
