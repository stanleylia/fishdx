"""k=5 McNemar (fused lambda=0.7 vs visual-only lambda=1.0, majority vote) on the CLEAN
D1-disjoint D2 subset (cos<0.95, n=2,628), so the main-text k=5 test matches the clean
evaluation set rather than the full pre-dedup D2. Real CLIP; seed 42.
Writes results/d2_clean_mcnemar_k5.json.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np, torch
from PIL import Image
from tqdm import tqdm
from scipy.stats import chi2 as chi2dist

ROOT=Path(__file__).resolve().parents[2]; RES=ROOT/"results"
z=np.load(RES/"_cache_d1_gallery.npz",allow_pickle=True); gv,gc,gl=z["visual"],z["caption"],z["labels"]
D2=ROOT/"data/datasets/D2"; CAPS=json.loads((RES/"e3_captions_log.json").read_text())
caps={c["image"]:(c.get("caption") or "") for c in CAPS}
EXTS={'.jpg','.jpeg','.png','.bmp','.webp'}
def canon(s):
    s=re.sub(r"[\s_]*\(?\d+\)?$","",s).strip(); s=re.sub(r"_aug$","",s).strip()
    m=re.match(r"^(.*)_\1$",s); s=(m.group(1) if m else s).lower()
    for k,v in [("healthy","healthy"),("aeromon","aeromoniasis"),("gill","bacterial_gill"),("red","bacterial_red"),("saproleg","fungal"),("fungal","fungal"),("parasit","parasitic"),("white tail","viral_white_tail"),("viral","viral_white_tail"),("ulcerative","EUS"),("eus","EUS")]:
        if k in s: return v
    return "UNKNOWN"
def l2(a): return a/(np.linalg.norm(a,axis=-1,keepdims=True)+1e-12)
import open_clip
clip,_,prep=open_clip.create_model_and_transforms("ViT-B-32",pretrained="laion2b_s34b_b79k",device="cuda:0"); clip.eval()
tok=open_clip.get_tokenizer("ViT-B-32")
def ei(p):
    with torch.no_grad():
        x=prep(Image.open(p).convert("RGB")).unsqueeze(0).to("cuda:0"); f=clip.encode_image(x)
        return (f/f.norm(dim=-1,keepdim=True))[0].cpu().numpy().astype(np.float32)
def et(t):
    with torch.no_grad():
        tt=tok([t or "fish"]).to("cuda:0"); f=clip.encode_text(tt)
        return (f/f.norm(dim=-1,keepdim=True))[0].cpu().numpy().astype(np.float32)
paths=sorted(p for p in D2.rglob("*") if p.suffix.lower() in EXTS)
qv,qc,ql=[],[],[]
for p in tqdm(paths,desc="enc"):
    qv.append(ei(p)); qc.append(et(caps.get(p.name,""))); ql.append(canon(p.stem))
qv,qc,ql=np.array(qv),np.array(qc),np.array(ql)
clean=(qv@gv.T).max(1)<0.95; qv,qc,ql=qv[clean],qc[clean],ql[clean]
def knn5(lam):
    g=l2(lam*gv+(1-lam)*gc); q=l2(lam*qv+(1-lam)*qc); sims=q@g.T
    idx=np.argpartition(-sims,5,axis=1)[:,:5]; out=[]
    for i in range(len(q)):
        labs=gl[idx[i]]; sc=sims[i,idx[i]]; u={}
        for lb,s in zip(labs,sc): u[lb]=u.get(lb,0)+1+1e-6*s
        out.append(max(u,key=u.get))
    return np.array(out)
p07=knn5(0.7); p10=knn5(1.0); c07=p07==ql; c10=p10==ql
b=int((c07&~c10).sum()); c=int((~c07&c10).sum())
chi2=(abs(b-c)-1)**2/(b+c) if (b+c)>0 else 0.0
p=float(chi2dist.sf(chi2,1))
out={"experiment":"d2_clean_mcnemar_k5","subset":"D1-disjoint clean D2 (cos<0.95)","n":int(len(ql)),
     "test":"lambda0.7 k=5 vs lambda1.0 k=5 (majority vote), 8-class",
     "DA_lambda0.7_k5":round(float(c07.mean()),4),"DA_lambda1.0_k5":round(float(c10.mean()),4),
     "contingency_2x2":{"both_correct":int((c07&c10).sum()),"l07_correct_l10_wrong":b,
                        "l07_wrong_l10_correct":c,"both_wrong":int((~c07&~c10).sum())},
     "b":b,"c":c,"chi2_cc":round(chi2,4),"p_value":round(p,4),
     "original_fullset":{"p_full_n3453":0.023,"p_manuscript":0.033}}
(RES/"d2_clean_mcnemar_k5.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
