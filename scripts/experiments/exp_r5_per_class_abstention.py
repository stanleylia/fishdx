"""Reviewer 5 comment 2(3) — per-class abstention rates and error distributions.

"Report abstention rates and error distributions per class, for both known diseases and EUS,
to avoid masking inter-class differences with a single overall interception rate."

Everything is derived from the per-sample archive written under comment 2(1); no model is
re-run and no new retrieval is performed.

Two semantics are deliberately kept apart:

  * **Seven known classes** — abstention withholds a prediction the system could have made.
    It is decomposed into *beneficial* abstention (the forced-choice prediction would have
    been wrong, so the deferral intercepts an error) and *costly* abstention (the
    forced-choice prediction would have been right, so the deferral discards a correct
    answer). Their ratio is the quantity that says whether deferral pays off for that class.
  * **EUS** — the reference gallery contains no EUS class, so every committed prediction is
    necessarily wrong and abstention is the correct behaviour. EUS is therefore reported as
    an interception rate and is never pooled into the known-class accuracy figures.

Consistency assertions (all must hold before anything is written):
  - the eight rows sum to the D2-final size
  - per-class abstention counts sum to the overall abstention count of Table 4
  - decided-confusion-matrix row sums equal the per-class decided counts
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
R5 = RES / "reviewer5"
OUT = R5 / "per_class_abstention.json"

THETAS = (0.01, 0.02, 0.05)
THETA_PRIMARY = 0.02

PRETTY = {"aeromoniasis": "Aeromoniasis", "bacterial_gill": "Bacterial Gill Disease",
          "bacterial_red": "Bacterial Red Disease", "fungal": "Fungal Saprolegniasis",
          "healthy": "Healthy Fish", "parasitic": "Parasitic Diseases",
          "viral_white_tail": "Viral White Tail Disease", "EUS": "Epizootic Ulcerative Syndrome"}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(float(c - h), 4), round(float(c + h), 4)]


def main() -> None:
    recs = [json.loads(l) for l in (R5 / "selective_predictions_d2final.jsonl").open()]
    pipe = [r for r in recs if r["method"] == "pipeline"]
    n_total = len(pipe)
    print("=" * 100)
    print("Reviewer 5 / 2(3) — per-class abstention and error distribution")
    print(f"  source: per-sample archive, {n_total:,} D2-final samples")
    print("=" * 100)

    classes = sorted({r["true_label"] for r in pipe}, key=lambda c: (c == "EUS", c))
    published = json.loads((RES / "d2_final_full.json").read_text())["table4_d2final"]

    sweep: dict[str, dict] = {}
    for theta in THETAS:
        panel: dict[str, dict] = {}
        for cls in classes:
            rows = [r for r in pipe if r["true_label"] == cls]
            n = len(rows)
            abst = [r for r in rows if r["retrieval_margin"] < theta]
            dec = [r for r in rows if r["retrieval_margin"] >= theta]
            entry: dict = {
                "class": PRETTY.get(cls, cls),
                "n": n,
                "n_abstained": len(abst),
                "abstention_rate": round(len(abst) / n, 4),
                "abstention_wilson95": wilson(len(abst), n),
                "n_decided": len(dec),
                "coverage": round(len(dec) / n, 4),
            }
            if cls == "EUS":
                # out-of-gallery class: abstention IS the correct behaviour
                entry["semantics"] = ("out-of-gallery: abstention is correct interception; "
                                      "every committed prediction is necessarily wrong")
                entry["interception_rate"] = entry["abstention_rate"]
                entry["interception_wilson95"] = entry["abstention_wilson95"]
                entry["committed_label_distribution"] = dict(
                    Counter(r["predicted_label_forced"] for r in dec).most_common())
            else:
                n_ok = sum(r["correct_forced"] for r in dec)
                acc = n_ok / len(dec) if dec else 0.0
                beneficial = sum(1 for r in abst if not r["correct_forced"])
                costly = sum(1 for r in abst if r["correct_forced"])
                entry.update({
                    "semantics": "in-gallery class: abstention withholds an available prediction",
                    "n_correct_decided": n_ok,
                    "selective_accuracy": round(acc, 4),
                    "selective_accuracy_wilson95": wilson(n_ok, len(dec)),
                    "selective_risk": round(1 - acc, 4),
                    "forced_choice_accuracy": round(sum(r["correct_forced"] for r in rows) / n, 4),
                    "abstention_beneficial": beneficial,
                    "abstention_costly": costly,
                    "beneficial_to_costly_ratio": (round(beneficial / costly, 3) if costly
                                                   else None),
                    # Under abstention that ignores correctness, the expected ratio is
                    # (1-p)/p with p the forced-choice accuracy of the class. Comparing the
                    # observed ratio with that baseline separates genuine error-targeting
                    # from the arithmetic consequence of a high baseline accuracy.
                    "beneficial_to_costly_if_random": None,
                    "error_targeting_lift": None,
                    "decided_error_distribution": dict(
                        Counter(r["predicted_label_forced"] for r in dec
                                if not r["correct_forced"]).most_common()),
                })
                p_fc = entry["forced_choice_accuracy"]
                if 0 < p_fc < 1 and costly:
                    rand_ratio = (1 - p_fc) / p_fc
                    entry["beneficial_to_costly_if_random"] = round(rand_ratio, 3)
                    entry["error_targeting_lift"] = round(
                        (beneficial / costly) / rand_ratio, 2)
            panel[cls] = entry
        sweep[f"theta_{theta:.2f}"] = panel

    # ---- consistency assertions -------------------------------------------------
    primary = sweep[f"theta_{THETA_PRIMARY:.2f}"]
    assert sum(v["n"] for v in primary.values()) == n_total, "class sizes do not sum to D2-final"
    tot_abst = sum(v["n_abstained"] for v in primary.values())
    tot_dec = sum(v["n_decided"] for v in primary.values())
    exp_dec = published[f"theta_{THETA_PRIMARY:.2f}"]["n_decided"]
    assert tot_dec == exp_dec, f"decided {tot_dec} != Table 4 {exp_dec}"
    assert tot_abst == n_total - exp_dec, "abstention counts do not reconcile with Table 4"
    exp_eus = published[f"theta_{THETA_PRIMARY:.2f}"]["eus_interception"]
    got_eus = primary["EUS"]["interception_rate"]
    assert abs(got_eus - exp_eus) < 1e-9, f"EUS interception {got_eus} != {exp_eus}"
    print(f"  consistency OK: 8 rows sum to {n_total:,}; decided {tot_dec} and "
          f"abstained {tot_abst} reconcile with Table 4; EUS interception {got_eus}")

    # ---- decided confusion matrix over the seven known classes ------------------
    known = [c for c in classes if c != "EUS"]
    dec_rows = [r for r in pipe if r["retrieval_margin"] >= THETA_PRIMARY
                and r["true_label"] != "EUS"]
    cm = {t: {p: 0 for p in known} for t in known}
    for r in dec_rows:
        cm[r["true_label"]][r["predicted_label_forced"]] += 1
    for t in known:
        assert sum(cm[t].values()) == primary[t]["n_decided"], f"confusion row {t} mismatch"
    print("  confusion-matrix row sums match per-class decided counts")

    out = {
        "experiment": "r5_per_class_abstention",
        "reviewer_comment": "5-2(3): per-class abstention rates and error distributions",
        "source_archive": "results/reviewer5/selective_predictions_d2final.jsonl",
        "models_rerun": False,
        "n_d2final": n_total,
        "primary_theta": THETA_PRIMARY,
        "eus_note": ("EUS is absent from the reference gallery, so abstention is the correct "
                     "behaviour and its rate is reported as interception, never pooled with "
                     "the in-gallery selective accuracies"),
        "per_class_sweep": sweep,
        "decided_confusion_matrix_theta_0.02": cm,
        "consistency_checks": {"rows_sum_to_n": True, "decided_matches_table4": True,
                               "eus_interception_matches_table4": True,
                               "confusion_rows_match": True},
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # ---- console table ----------------------------------------------------------
    print(f"\n  --- theta = {THETA_PRIMARY} ---")
    hdr = f"  {'class':<30}{'N':>5}{'abst%':>7}{'dec':>6}{'fc.acc':>8}{'sel.acc':>9}{'benef':>7}{'costly':>7}{'b/c':>6}{'rand':>6}{'lift':>6}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for cls in classes:
        v = primary[cls]
        if cls == "EUS":
            print(f"  {v['class']:<30}{v['n']:>5}{v['abstention_rate']*100:>6.1f}%"
                  f"{v['n_decided']:>6}{'—':>8}{'— (OOD)':>9}"
                  f"{'—':>7}{'—':>7}{'—':>6}{'—':>6}{'—':>6}")
        else:
            r = v["beneficial_to_costly_ratio"]; rr = v["beneficial_to_costly_if_random"]
            lf = v["error_targeting_lift"]
            print(f"  {v['class']:<30}{v['n']:>5}{v['abstention_rate']*100:>6.1f}%"
                  f"{v['n_decided']:>6}{v['forced_choice_accuracy']:>8.3f}"
                  f"{v['selective_accuracy']:>9.4f}"
                  f"{v['abstention_beneficial']:>7}{v['abstention_costly']:>7}"
                  f"{(f'{r:.2f}' if r is not None else '—'):>6}"
                  f"{(f'{rr:.3f}' if rr is not None else '—'):>6}"
                  f"{(f'{lf:.1f}x' if lf is not None else '—'):>6}")
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
