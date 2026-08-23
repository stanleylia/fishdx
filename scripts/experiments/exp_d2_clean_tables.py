"""Authoritative clean-D2 recompute of Table 3 + Table 4 + supporting numbers, with CIs.

Clean set = D2 images whose max CLIP visual cosine to the D1 gallery < 0.95 (removes exact
duplicates AND CLIP-near-identical augmentations that self-retrieve). Everything the D1<->D2
overlap contaminated is recomputed on this set from real embeddings:

  Table 3 : Pipeline / kNN(lambda,k) / caption-only  -> Overall DA (8-cls) + Overlap DA (7-cls) + Wilson CI
  Table 4 : theta_margin sweep -> Decisive DA (bootstrap CI) / coverage / Inconclusive% / EUS interception
  S9      : class-prototype retrieval vs kNN
  S2/R1.3 : deferring visual-kNN Decisive DA at the pipeline's operating coverage
  lambda-sweep (7-cls), modality comparison

Pipeline forced-choice == kNN(lambda=0.7,k=1) under the paper's faithful evidence pool
(Supplementary Note S5: scoring confirms the retrieval top-1 class), so the Pipeline row
equals the kNN(0.7,k=1) row by construction; reported as such.

Writes results/d2_clean_tables.json.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
D2_DIR = ROOT / "data/datasets/D2"; RES = ROOT / "results"
CACHE = RES / "_cache_d1_gallery.npz"; D2_CAPS = RES / "e3_captions_log.json"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEDUP_COS = 0.95; EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}
SEED = 42; rng = np.random.default_rng(SEED)

def canon(stem):
    s = re.sub(r"[\s_]*\(?\d+\)?$","",stem).strip(); s = re.sub(r"_aug$","",s).strip()
    m = re.match(r"^(.*)_\1$", s); s = (m.group(1) if m else s).lower()
    for k,v in [("healthy","healthy"),("aeromon","aeromoniasis"),("gill","bacterial_gill"),
                ("red","bacterial_red"),("saproleg","fungal"),("fungal","fungal"),
                ("parasit","parasitic"),("white tail","viral_white_tail"),("viral","viral_white_tail"),
                ("ulcerative","EUS"),("eus","EUS")]:
        if k in s: return v
    return "UNKNOWN"
def l2(a): return a/(np.linalg.norm(a,axis=-1,keepdims=True)+1e-12)
def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (round(float(c-h),4), round(float(c+h),4))

z = np.load(CACHE, allow_pickle=True); gv,gc,gl = z["visual"],z["caption"],z["labels"]
import open_clip
clip,_,prep = open_clip.create_model_and_transforms("ViT-B-32",pretrained="laion2b_s34b_b79k",device=DEVICE)
clip.eval(); tok = open_clip.get_tokenizer("ViT-B-32")
caps = {c["image"]:(c.get("caption") or "") for c in json.loads(D2_CAPS.read_text())}
def ei(p):
    with torch.no_grad():
        x=prep(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE); f=clip.encode_image(x)
        return (f/f.norm(dim=-1,keepdim=True))[0].cpu().numpy().astype(np.float32)
def et(t):
    with torch.no_grad():
        tt=tok([t or "fish"]).to(DEVICE); f=clip.encode_text(tt)
        return (f/f.norm(dim=-1,keepdim=True))[0].cpu().numpy().astype(np.float32)
paths = sorted(p for p in D2_DIR.rglob("*") if p.suffix.lower() in EXTS)
qv,qc,ql = [],[],[]
for p in tqdm(paths, desc="D2 encode"):
    qv.append(ei(p)); qc.append(et(caps.get(p.name,""))); ql.append(canon(p.stem))
qv,qc,ql = np.array(qv),np.array(qc),np.array(ql)
clean = (qv@gv.T).max(axis=1) < DEDUP_COS
qv,qc,ql = qv[clean],qc[clean],ql[clean]
n_removed = int((~clean).sum())
shared = sorted((set(gl.tolist())&set(ql.tolist()))-{"EUS","UNKNOWN"}); m7 = np.isin(ql,shared)
iseus = ql == "EUS"

def preds(lam, k):
    g = l2(lam*gv+(1-lam)*gc); q = l2(lam*qv+(1-lam)*qc); sims = q@g.T
    if k == 1:
        return gl[sims.argmax(1)]
    idx = np.argpartition(-sims, k, axis=1)[:, :k]
    out = []
    for i in range(len(q)):
        labs = gl[idx[i]]; s = sims[i, idx[i]]
        # majority vote, tie-break by summed similarity
        uniq = {}
        for lab, sc in zip(labs, s):
            uniq[lab] = uniq.get(lab, 0.0) + 1.0 + 1e-6*float(sc)  # float64 deterministic tie-break
        out.append(max(uniq, key=uniq.get))
    return np.array(out)

def da_ci(pred, mask):
    p, t = pred[mask], ql[mask]; k = int((p==t).sum()); n = int(mask.sum())
    return round(k/n,4), n, wilson(k,n)

# ---- Table 3 ----
table3 = {}
for name, lam, k in [("Pipeline (λ=0.7)",0.7,1),("kNN (λ=0.7,k=1)",0.7,1),("kNN (λ=1.0,k=1)",1.0,1),
                     ("kNN (λ=0.7,k=5)",0.7,5),("kNN (λ=1.0,k=5)",1.0,5),("Caption-only (λ=0.0,k=1)",0.0,1)]:
    pr = preds(lam,k)
    ov,n8,ci8 = da_ci(pr, np.ones(len(ql),bool))
    op,n7,ci7 = da_ci(pr, m7)
    table3[name] = {"overall_DA":ov,"overall_wilson":ci8,"overlap_DA":op,"overlap_wilson":ci7}

# ---- lambda sweep (7-class) ----
lam_sweep = {f"{x:.1f}": da_ci(preds(x,1), m7)[0] for x in [round(i,1) for i in np.arange(0,1.01,0.1)]}

# ---- Table 4: theta sweep (lambda=0.7) with bootstrap Decisive-DA CI ----
g = l2(0.7*gv+0.3*gc); q = l2(0.7*qv+0.3*qc); sims = q@g.T
order = np.sort(sims,1); margin = order[:,-1]-order[:,-2]; pred07 = gl[sims.argmax(1)]; corr = pred07==ql
def boot_ci(dec_mask, B=1000):
    idx = np.where(dec_mask)[0]
    if len(idx)==0: return (0.0,0.0)
    accs = [ (corr[rng.choice(idx,len(idx),replace=True)]).mean() for _ in range(B) ]
    return (round(float(np.percentile(accs,2.5)),4), round(float(np.percentile(accs,97.5)),4))
table4 = {}
for th in [0.00,0.01,0.02,0.05]:
    dec = margin>=th; cov=float(dec.mean())
    dd = float(corr[dec].mean()) if dec.any() else 0.0
    table4[f"theta_{th:.2f}"] = {"decisive_DA":round(dd,4),"decisive_boot_ci":boot_ci(dec),
        "coverage":round(cov,4),"inconclusive_pct":round(1-cov,4),
        "eus_interception":round(float((~dec & iseus).sum()/max(1,iseus.sum())),4)}

# ---- deferring visual kNN at pipeline operating coverage (R1.3/S2) ----
gv_n = l2(gv); qv_n = l2(qv); sv = qv_n@gv_n.T
ov = np.sort(sv,1); vmargin = ov[:,-1]-ov[:,-2]; vpred = gl[sv.argmax(1)]; vcorr = vpred==ql
# operating coverage = pipeline coverage at theta=0.02
target_cov = table4["theta_0.02"]["coverage"]
thr = np.quantile(vmargin, 1-target_cov)
vdec = vmargin>=thr
deferring_knn = {"coverage":round(float(vdec.mean()),4),
                 "decisive_DA":round(float(vcorr[vdec].mean()),4)}

# ---- class-prototype (S9) ----
proto = {}
for lam in [1.0,0.7,0.0]:
    g = l2(lam*gv+(1-lam)*gc); q = l2(lam*qv[m7]+(1-lam)*qc[m7])
    gcls = sorted(set(gl.tolist()))
    P = np.array([l2(g[gl==c].mean(0,keepdims=True))[0] for c in gcls])
    pr = np.array(gcls)[(q@P.T).argmax(1)]
    proto[f"lambda_{lam}"] = {"class_prototype_DA":round(float((pr==ql[m7]).mean()),4),
                              "knn_DA":table3[{1.0:"kNN (λ=1.0,k=1)",0.7:"kNN (λ=0.7,k=1)",0.0:"Caption-only (λ=0.0,k=1)"}[lam]]["overlap_DA"]}

out = {"experiment":"d2_clean_tables","dedup":f"CLIP visual cosine < {DEDUP_COS}",
 "seed":SEED,"n_clean":int(len(ql)),"n_removed_dup":n_removed,"n_eus_clean":int(iseus.sum()),
 "n_overlap_7class":int(m7.sum()),
 "table3":table3,"lambda_sweep_7class":lam_sweep,"table4_theta_sweep":table4,
 "deferring_visual_knn_matched_coverage":deferring_knn,"class_prototype_S9":proto,
 "modality":{"caption_only":table3["Caption-only (λ=0.0,k=1)"]["overlap_DA"],
             "visual":table3["kNN (λ=1.0,k=1)"]["overlap_DA"],
             "fusion":table3["kNN (λ=0.7,k=1)"]["overlap_DA"]},
 "original_contaminated":{"pipeline_overall":0.811,"knn07_overall":0.817,"knn10_overall":0.819,
     "visual_overlap":0.934,"fusion_overlap":0.943,"caption":0.720,"decisive_DA":0.924,
     "decisive_cov":0.643,"eus_interception":0.679,"lambda_opt":"0.8=0.947","deferring_knn":0.934,
     "prototype":"0.650 vs 0.943"}}
(RES/"d2_clean_tables.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
