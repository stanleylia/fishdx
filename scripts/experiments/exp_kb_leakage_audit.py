"""KB leakage audit + keyword-tier census (addresses R1.5, R5.2, and the data-isolation protocol).

Verifies two things about the 8 on-disk KB documents:
 1. LEAKAGE: no D1/D2 dataset-specific token (image filename stems, split names,
    dataset ids) appears in any KB document body.
 2. CENSUS: the per-document keyword-tier counts (Confirmed / Suspected / Mentioned /
    Healthy / Negation CN / EN) actually present on disk, so Note S6 reports real counts.

Writes results/kb_leakage_audit.json. Read-only over the KB; no models loaded.
"""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "src/fishdx/kb/documents"
D1 = ROOT / "data/datasets/D1"
D2 = ROOT / "data/datasets/D2"
RES = ROOT / "results"

TIER_HEADINGS = {
    "confirmed": r"confirmed keywords",
    "suspected": r"suspected keywords",
    "mentioned": r"mentioned keywords",
    "healthy": r"healthy keywords",
    "negation_cn": r"negation patterns \(cn\)",
    "negation_en": r"negation patterns \(en\)",
    "bridge": r"bridge",
}

def sections(md: str) -> dict[str, list[str]]:
    """Split markdown body into ## sections -> list of bullet items."""
    out, cur = {}, None
    for line in md.splitlines():
        h = re.match(r"^#{2,3}\s+(.*)", line.strip())
        if h:
            cur = h.group(1).strip().lower(); out[cur] = []
        elif cur is not None:
            b = re.match(r"^[-*]\s+(.+)", line.strip())
            if b:
                out[cur].append(b.group(1).strip())
    return out

def tier_count(secs: dict[str, list[str]], pattern: str) -> int:
    n = 0
    for name, items in secs.items():
        if re.search(pattern, name):
            n += len(items)
    return n

# ---- build D1/D2 leakage token set (filename stems + split/dataset ids) ----
def dataset_tokens() -> set[str]:
    toks = set()
    for d in (D1, D2):
        if not d.exists():
            continue
        for f in d.rglob("*.jpg"):
            stem = f.stem.lower()
            # split on non-alnum, keep tokens that look dataset-specific (digits/underscores/aug)
            for t in re.split(r"[^a-z0-9]+", stem):
                if t and (t.isdigit() or t in {"aug"} or re.match(r".*\d.*", t)):
                    toks.add(t)
    toks |= {"d1", "d2", "train_split", "test_split", "_aug", "train/", "test/",
             "kaggle", "d1_train", "d1_test", "d2_test", "leave-one-out"}
    return toks

def main() -> None:
    tokens = dataset_tokens()
    docs, census, leaks = [], [], []
    for f in sorted(KB.glob("*.md")):
        raw = f.read_text()
        body = raw.split("---", 2)[-1] if raw.startswith("---") else raw
        secs = sections(body)
        counts = {k: tier_count(secs, v) for k, v in TIER_HEADINGS.items()}
        census.append({"file": f.name, **counts})
        # leakage: check each dataset token as a whole-word match in the body (lowered)
        low = body.lower()
        hit = sorted({t for t in tokens if len(t) >= 3 and re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", low)})
        if hit:
            leaks.append({"file": f.name, "tokens": hit})
        docs.append(f.name)

    out = {
        "experiment": "kb_leakage_audit",
        "n_docs": len(docs),
        "docs": docs,
        "n_dataset_tokens_checked": len(tokens),
        "leakage_hits": leaks,
        "leakage_clean": len(leaks) == 0,
        "keyword_census": census,
    }
    RES.mkdir(exist_ok=True)
    (RES / "kb_leakage_audit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"leakage_clean": out["leakage_clean"], "n_docs": out["n_docs"],
                      "leaks": leaks}, indent=2))
    print("\nCENSUS (Confirmed/Suspected/Mentioned/Healthy/NegCN/NegEN):")
    for c in census:
        print(f"  {c['file']:20s} C={c['confirmed']:2d} S={c['suspected']:2d} "
              f"M={c['mentioned']:2d} H={c['healthy']:2d} nCN={c['negation_cn']} nEN={c['negation_en']}")

if __name__ == "__main__":
    main()
