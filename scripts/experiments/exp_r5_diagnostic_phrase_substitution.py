"""Reviewer 5 comment 1(3) — controlled diagnostic-phrase substitution (CDPS).

"Use controlled diagnostic-phrase substitution or expert-judgment experiments to test
whether supplementing terminology genuinely improves text-pathway performance. The
substitution experiment itself can serve as a practical proxy for SCA validation without
requiring a full human annotation study."

Design. The image set, the CLIP text encoder, the retrieval targets and the decision rule
are all held fixed. Only the caption text changes, and every condition inserts exactly one
phrase at the same position (appended as a final sentence):

  A  original    — the real Florence-2 caption, unmodified
  B  correct     — one Confirmed diagnostic phrase of the image's TRUE class
  C  wrong-class — one Confirmed diagnostic phrase of a DIFFERENT class, word-length matched
  D  placebo     — a neutral, non-diagnostic fish descriptor, word-length matched

C controls for "any medical-sounding vocabulary helps" and D controls for "a longer caption
helps". Without both, a gain in B would be uninterpretable.

B is selected using the reference label, so it is an **oracle intervention** that probes
pathway sensitivity. It is not a deployable classification result and must never be
reported as one.

Two text pathways are measured, both fixed:
  * caption -> the 8 knowledge-base documents (the published DA = 0.167 baseline)
  * caption -> the D1 gallery caption embeddings (the published caption-only retrieval)

Endpoints: top-1 accuracy, mean reciprocal rank of the correct knowledge-base document,
and the Eq. 12 keyword-SCA under the ADR-0018 boundary-aware matcher.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fishdx.scoring.score import _count_occurrences  # noqa: E402  (ADR-0018 matcher)

RES = ROOT / "results"
R5 = RES / "reviewer5"
KB_DIR = ROOT / "src/fishdx/kb/documents"
CAPS = RES / "e3_noaug_captions_log.json"
OUT = R5 / "diagnostic_phrase_substitution.json"
JSONL = R5 / "diagnostic_phrase_substitution.jsonl"

SEED = 42
N_BOOT = 1000
DOSES = (1, 2, 3, 5, "all")   # phrases inserted in the dose-response arm
ARCH, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"

LABEL_MAP = {
    "Bacterial Red disease": "Bacterial Red Disease",
    "Bacterial diseases - Aeromoniasis": "Aeromoniasis",
    "Bacterial gill disease": "Bacterial Gill Disease",
    "EUS": "Epizootic Ulcerative Syndrome",
    "Fungal diseases Saprolegniasis": "Fungal Saprolegniasis",
    "Healthy Fish": "Healthy Fish",
    "Parasitic diseases": "Parasitic Diseases",
    "Viral diseases White tail disease": "Viral White Tail Disease",
}
CANON_SHORT = {"Aeromoniasis": "aeromoniasis", "Bacterial Gill Disease": "bacterial_gill",
               "Bacterial Red Disease": "bacterial_red", "Fungal Saprolegniasis": "fungal",
               "Healthy Fish": "healthy", "Parasitic Diseases": "parasitic",
               "Viral White Tail Disease": "viral_white_tail",
               "Epizootic Ulcerative Syndrome": "EUS"}

# Neutral, non-diagnostic descriptors spanning 1-6 words. Asserted below to contain no
# knowledge-base Confirmed phrase.
PLACEBOS = [
    "swimming", "photographed", "aquarium",
    "in water", "near plants", "on gravel", "beside stones",
    "next to a plant", "under bright light", "against a plain background",
    "held above the water surface", "resting near the tank floor",
    "photographed from the side at close range",
    "positioned in the centre of the frame area",
]

STOP = set("a an the of and or to in on with at is are be this that fish its it as for from by".split())


def toks(t: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", t.lower()) if len(w) > 2 and w not in STOP]


def kb_docs() -> list[dict]:
    docs = []
    for f in sorted(KB_DIR.glob("*.md")):
        raw = f.read_text()
        cls = re.search(r"disease_class:\s*(.+)", raw).group(1).strip()
        body = raw.split("---", 2)[-1]
        m = re.search(r"##\s*Confirmed keywords\s*(.+?)(?:\n##|\Z)", body, re.S)
        phrases = [x.strip() for x in re.findall(r"^-\s+(.+)$", m.group(1), re.M)] if m else []
        docs.append({"class": cls, "full": body.strip(), "phrases": phrases})
    return docs


def sca(caption: str, all_phrases: list[str]) -> float:
    c = (caption or "").lower()
    k = toks(c)
    if not k:
        return 0.0
    matched: set[str] = set()
    for p in all_phrases:
        if _count_occurrences(c, p) > 0:
            matched |= set(toks(p))
    return sum(1 for w in k if w in matched) / len(k)


@torch.no_grad()
def encode(texts: list[str], model, tok) -> np.ndarray:
    out = []
    for i in range(0, len(texts), 256):
        e = model.encode_text(tok(texts[i: i + 256]).to(DEV)).float()
        out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out, 0)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-tailed exact binomial McNemar p-value."""
    from scipy.stats import binomtest
    if b + c == 0:
        return 1.0
    return float(binomtest(b, b + c, 0.5, alternative="two-sided").pvalue)


def main() -> None:
    rng = np.random.default_rng(SEED)
    docs = kb_docs()
    doc_classes = [d["class"] for d in docs]
    all_phrases = sorted({p.lower() for d in docs for p in d["phrases"]})
    phrases_by_class = {d["class"]: d["phrases"] for d in docs}

    for p in PLACEBOS:
        hits = [q for q in all_phrases if _count_occurrences(p.lower(), q) > 0]
        assert not hits, f"placebo {p!r} contains diagnostic phrase(s) {hits}"

    caps = json.loads(CAPS.read_text())
    rows = [r for r in caps if (r.get("caption") or "").strip()
            and r["label"] in LABEL_MAP]
    n = len(rows)
    print("=" * 100)
    print("Reviewer 5 / 1(3) — controlled diagnostic-phrase substitution")
    print(f"  captions = {n:,} | KB classes = {len(docs)} | Confirmed phrases = {len(all_phrases)}")
    print(f"  device = {DEV} | seed = {SEED}")
    print("=" * 100)

    def wlen(s: str) -> int:
        return len(s.split())

    def pick_matched(pool: list[str], target_len: int) -> str:
        best = min(pool, key=lambda p: (abs(wlen(p) - target_len), p))
        ties = [p for p in pool if abs(wlen(p) - target_len) == abs(wlen(best) - target_len)]
        return ties[int(rng.integers(0, len(ties)))]

    # ---- build the four conditions ---------------------------------------------
    built = []
    for r in rows:
        true_cls = LABEL_MAP[r["label"]]
        base = (r["caption"] or "").strip().rstrip(".")
        correct_pool = phrases_by_class[true_cls]
        b_phrase = correct_pool[int(rng.integers(0, len(correct_pool)))]
        L = wlen(b_phrase)
        wrong_pool = [p for cls, ps in phrases_by_class.items() if cls != true_cls for p in ps]
        c_phrase = pick_matched(wrong_pool, L)
        d_phrase = pick_matched(PLACEBOS, L)
        # dose-response arm: k correct diagnostic phrases, k = 1, 2, 3, 5, all
        dose_pool = list(correct_pool)
        rng.shuffle(dose_pool)
        dose = {}
        for k in DOSES:
            take = dose_pool if k == "all" else dose_pool[:int(k)]
            dose[f"K{k}"] = f"{base}. The fish shows " + ", ".join(take) + "."
        dose_pre = {}
        for k in DOSES:
            take = dose_pool if k == "all" else dose_pool[:int(k)]
            dose_pre[f"KP{k}"] = "The fish shows " + ", ".join(take) + ". " + base + "."
        built.append({
            "image": r["image"], "true_class": true_cls,
            **dose, **dose_pre,
            # PREPENDED variants: 96.7% of captions already reach the 77-token CLIP limit,
            # so an appended phrase is truncated away before it reaches the encoder.
            # Prepending guarantees the intervention is actually seen. B/C/D prepend the
            # same number of words at the same position, so BP vs CP and BP vs DP remain
            # controlled; BP vs A additionally loses the caption tail and is reported as such.
            "BP": f"The fish shows {b_phrase}. {base}.",
            "CP": f"The fish shows {c_phrase}. {base}.",
            "DP": f"The fish is {d_phrase}. {base}.",
            "n_phrases_all": len(dose_pool),
            "A": base + ".",
            "B": f"{base}. The fish shows {b_phrase}.",
            "C": f"{base}. The fish shows {c_phrase}.",
            "D": f"{base}. The fish is {d_phrase}.",
            "phrase_B": b_phrase, "phrase_C": c_phrase, "phrase_D": d_phrase,
            "phrase_words": L,
        })

    # ---- encode ------------------------------------------------------------------
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED, device=DEV)
    model.eval()
    tok = open_clip.get_tokenizer(ARCH)

    kb_emb = encode([d["full"] for d in docs], model, tok)
    g = np.load(RES / "_cache_d1_gallery.npz", allow_pickle=True)
    gal_cap = g["caption"] / (np.linalg.norm(g["caption"], axis=-1, keepdims=True) + 1e-12)
    gal_lab = g["labels"]

    truth_long = np.array([b["true_class"] for b in built])
    truth_short = np.array([CANON_SHORT[c] for c in truth_long])
    shared = sorted(set(gal_lab.tolist()) - {"EUS", "UNKNOWN"})
    in_shared = np.isin(truth_short, shared)

    results, per_cond = {}, {}
    COND_ORDER = (["A", "B", "C", "D", "BP", "CP", "DP"]
                  + [f"K{k}" for k in DOSES] + [f"KP{k}" for k in DOSES])
    for cond in COND_ORDER:
        texts = [b[cond] for b in built]
        emb = encode(texts, model, tok)

        sims_kb = emb @ kb_emb.T
        pred_kb = np.array([doc_classes[i] for i in sims_kb.argmax(1)])
        correct_kb = pred_kb == truth_long
        order = np.argsort(-sims_kb, axis=1)
        true_idx = np.array([doc_classes.index(c) for c in truth_long])
        rank = np.array([int(np.where(order[i] == true_idx[i])[0][0]) + 1 for i in range(n)])
        mrr = 1.0 / rank

        sims_g = emb @ gal_cap.T
        pred_g = gal_lab[sims_g.argmax(1)]
        correct_g = (pred_g == truth_short) & in_shared

        s = np.array([sca(t, all_phrases) for t in texts])
        per_cond[cond] = {"correct_kb": correct_kb, "mrr": mrr, "correct_g": correct_g,
                          "sca": s, "rank": rank}
        results[cond] = {
            "kb_top1_accuracy": round(float(correct_kb.mean()), 4),
            "kb_mrr": round(float(mrr.mean()), 4),
            "kb_mean_rank": round(float(rank.mean()), 3),
            "caption_gallery_top1_accuracy_7shared": round(
                float(correct_g[in_shared].sum() / in_shared.sum()), 4),
            "keyword_sca_pct": round(float(100 * s.mean()), 3),
            "mean_caption_words": round(float(np.mean([len(t.split()) for t in texts])), 2),
        }
        print(f"  {cond}: KB top-1 {results[cond]['kb_top1_accuracy']:.4f}  "
              f"MRR {results[cond]['kb_mrr']:.4f}  "
              f"caption-gallery {results[cond]['caption_gallery_top1_accuracy_7shared']:.4f}  "
              f"SCA {results[cond]['keyword_sca_pct']:.3f}%", flush=True)

    # ---- paired statistics -------------------------------------------------------
    def paired(x: str, y: str) -> dict:
        cx, cy = per_cond[x]["correct_kb"], per_cond[y]["correct_kb"]
        b = int((cx & ~cy).sum()); c = int((~cx & cy).sum())
        d_mrr = per_cond[x]["mrr"] - per_cond[y]["mrr"]
        boot = np.array([d_mrr[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
        return {
            "contrast": f"{x} vs {y}",
            "kb_top1_delta": round(float(cx.mean() - cy.mean()), 4),
            "mcnemar_b": b, "mcnemar_c": c,
            "mcnemar_exact_p": float(f"{mcnemar_exact(b, c):.3g}"),
            "mrr_delta": round(float(d_mrr.mean()), 4),
            "mrr_delta_ci95": [round(float(np.percentile(boot, 2.5)), 4),
                               round(float(np.percentile(boot, 97.5)), 4)],
        }

    contrasts = {
        # appended arm (truncated away for 96.7% of captions — reported as a null by design)
        "B_vs_A": paired("B", "A"), "B_vs_C": paired("B", "C"), "B_vs_D": paired("B", "D"),
        # PREPENDED arm (primary): the intervention actually reaches the encoder
        "BP_vs_A": paired("BP", "A"), "BP_vs_CP": paired("BP", "CP"),
        "BP_vs_DP": paired("BP", "DP"), "CP_vs_DP": paired("CP", "DP"),
        "DP_vs_A": paired("DP", "A"),
    }

    per_class = {}
    for cls in sorted(set(truth_long.tolist())):
        m = truth_long == cls
        per_class[cls] = {cond: round(float(per_cond[cond]["correct_kb"][m].mean()), 4)
                          for cond in ("A", "B", "C", "D")}
        per_class[cls]["n"] = int(m.sum())

    with JSONL.open("w") as fh:
        for i, b in enumerate(built):
            fh.write(json.dumps({
                "image": b["image"], "true_class": b["true_class"],
                "phrase_B": b["phrase_B"], "phrase_C": b["phrase_C"], "phrase_D": b["phrase_D"],
                "phrase_words": b["phrase_words"],
                **{f"kb_correct_{c}": bool(per_cond[c]["correct_kb"][i]) for c in "ABCD"},
                **{f"kb_rank_{c}": int(per_cond[c]["rank"][i]) for c in "ABCD"},
                **{f"sca_{c}": round(float(per_cond[c]["sca"][i]), 6) for c in "ABCD"},
            }, ensure_ascii=False) + "\n")

    out = {
        "experiment": "r5_diagnostic_phrase_substitution",
        "reviewer_comment": "5-1(3): controlled diagnostic-phrase substitution as an SCA proxy",
        "oracle_warning": ("condition B selects the inserted phrase using the reference "
                           "label; it probes pathway sensitivity and is NOT deployable "
                           "classification performance"),
        "corpus": "results/e3_noaug_captions_log.json (the set used for the published "
                  "caption-to-KB DA = 0.167 baseline)",
        "n_captions": n, "seed": SEED, "encoder": f"{ARCH}/{PRETRAINED}",
        "insertion": "one phrase appended as a final sentence, same position in every condition",
        "conditions": results,
        "dose_response_appended": {f"K{k}": results[f"K{k}"] for k in DOSES},
        "dose_response_prepended": {f"KP{k}": results[f"KP{k}"] for k in DOSES},
        "truncation_note": ("96.7% of the Florence-2 captions already reach the CLIP "
                            "77-token limit, so appended text never reaches the encoder. "
                            "The prepended arm is therefore the primary analysis; the "
                            "appended arm is retained because its null result documents "
                            "the truncation mechanism."),
        "dose_note": ("K1..Kall insert k Confirmed phrases of the TRUE class; all are oracle "
                      "interventions. Kall is the per-caption analogue of the diagnostic-rich "
                      "oracle query control, which reaches 8/8 when the caption is replaced "
                      "entirely by diagnostic text."),
        "paired_contrasts": contrasts,
        "per_class_kb_top1": per_class,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print("\n  dose-response, APPENDED (truncated away for 96.7% of captions):")
    for k in DOSES:
        v = results[f"K{k}"]
        print(f"    k={str(k):<4} KB top-1 {v['kb_top1_accuracy']:.4f}  MRR {v['kb_mrr']:.4f}  "
              f"SCA {v['keyword_sca_pct']:6.3f}%")
    print("\n  dose-response, PREPENDED (primary — reaches the encoder):")
    for k in DOSES:
        v = results[f"KP{k}"]
        print(f"    k={str(k):<4} KB top-1 {v['kb_top1_accuracy']:.4f}  MRR {v['kb_mrr']:.4f}  "
              f"caption-gallery {v['caption_gallery_top1_accuracy_7shared']:.4f}  "
              f"SCA {v['keyword_sca_pct']:6.3f}%")

    print("\n  paired contrasts (KB top-1 and MRR):")
    for k, v in contrasts.items():
        print(f"    {k:8s} dTop1={v['kb_top1_delta']:+.4f}  b={v['mcnemar_b']:>4} c={v['mcnemar_c']:>4}"
              f"  p={v['mcnemar_exact_p']:<10} dMRR={v['mrr_delta']:+.4f} {v['mrr_delta_ci95']}")
    print(f"\n  wrote {OUT}\n  wrote {JSONL}")


if __name__ == "__main__":
    main()
