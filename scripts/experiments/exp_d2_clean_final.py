"""Definitive clean-D2 numbers: embedding-space dedup (cos<0.95) + full analysis in one run.

Dedup: remove D2 images whose max CLIP visual cosine to the D1 gallery >= 0.95 (a duplicate
or CLIP-near-identical augmentation that would self-retrieve). On the clean remainder, compute
every affected number consistently: modality/lambda retrieval DA (7-class Overlap), 8-class
Overall DA, and the theta_margin selective-prediction sweep (Decisive DA / coverage / EUS
interception). Writes results/d2_clean_final.json.
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
DEDUP_COS = 0.95; LAM = 0.7; EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}
THETAS = [0.00, 0.01, 0.02, 0.05]

def canon(stem):
    s = re.sub(r"[\s_]*\(?\d+\)?$","",stem).strip(); s = re.sub(r"_aug$","",s).strip()
    m = re.match(r"^(.*)_\1$", s);  s = (m.group(1) if m else s).lower()
    for k,v in [("healthy","healthy"),("aeromon","aeromoniasis"),("gill","bacterial_gill"),
                ("red","bacterial_red"),("saproleg","fungal"),("fungal","fungal"),
                ("parasit","parasitic"),("white tail","viral_white_tail"),("viral","viral_white_tail"),
                ("ulcerative","EUS"),("eus","EUS")]:
        if k in s: return v
    return "UNKNOWN"
def l2(a): return a/(np.linalg.norm(a,axis=-1,keepdims=True)+1e-12)

z=np.load(CACHE,allow_pickle=True); gv,gc,gl=z["visual"],z["caption"],z["labels"]
import open_clip
clip,_,prep=open_clip.create_model_and_transforms("ViT-B-32",pretrained="laion2b_s34b_b79k",device=DEVICE)
clip.eval(); tok=open_clip.get_tokenizer("ViT-B-32")
caps={c["image"]:(c.get("caption") or "") for c in json.loads(D2_CAPS.read_text())}
def ei(p):
    with torch.no_grad():
        x=prep(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE); f=clip.encode_image(x)
        return (f/f.norm(dim=-1,keepdim=True))[0].cpu().numpy().astype(np.float32)
def et(t):
    with torch.no_grad():
        tt=tok([t or "fish"]).to(DEVICE); f=clip.encode_text(tt)
        return (f/f.norm(dim=-1,keepdim=True))[0].cpu().numpy().astype(np.float32)
paths=sorted(p for p in D2_DIR.rglob("*") if p.suffix.lower() in EXTS)
qv,qc,ql=[],[],[]
for p in tqdm(paths,desc="D2 encode"):
    qv.append(ei(p)); qc.append(et(caps.get(p.name,""))); ql.append(canon(p.stem))
qv,qc,ql=np.array(qv),np.array(qc),np.array(ql)
max_cos=(qv@gv.T).max(axis=1); clean=max_cos<DEDUP_COS
qv,qc,ql=qv[clean],qc[clean],ql[clean]
def da(qvv,qcc,qll,lam):
    g=l2(lam*gv+(1-lam)*gc); q=l2(lam*qvv+(1-lam)*qcc)
    return round(float((gl[(q@g.T).argmax(1)]==qll).mean()),4)
shared=sorted((set(gl.tolist())&set(ql.tolist()))-{"EUS","UNKNOWN"}); m7=np.isin(ql,shared)
# margin sweep (8-class, lambda=0.7)
g=l2(LAM*gv+(1-LAM)*gc); q=l2(LAM*qv+(1-LAM)*qc); sims=q@g.T
order=np.sort(sims,1); margin=order[:,-1]-order[:,-2]; pred=gl[sims.argmax(1)]
corr=pred==ql; iseus=ql=="EUS"
sweep={f"theta_{t:.2f}":{"coverage":round(float((margin>=t).mean()),4),
        "decisive_DA":round(float(corr[margin>=t].mean()),4) if (margin>=t).any() else 0.0,
        "eus_intercepted":round(float((~(margin>=t)&iseus).sum()/max(1,iseus.sum())),4)} for t in THETAS}
out={"experiment":"d2_clean_final","dedup":"CLIP visual cosine < 0.95 vs D1 gallery",
 "n_clean":int(len(ql)),"n_removed_dup":int((~clean).sum()),"n_eus_clean":int(iseus.sum()),
 "overlap_DA_7class":{"caption_0.0":da(qv[m7],qc[m7],ql[m7],0.0),
    "fusion_0.7":da(qv[m7],qc[m7],ql[m7],0.7),"visual_1.0":da(qv[m7],qc[m7],ql[m7],1.0)},
 "overall_DA_8class_lambda0.7":da(qv,qc,ql,0.7),
 "lambda_sweep_7class":{f"{x:.1f}":da(qv[m7],qc[m7],ql[m7],x) for x in [round(i,1) for i in np.arange(0,1.01,0.1)]},
 "decisive_sweep":sweep,
 "original_contaminated":{"visual":0.934,"fusion":0.943,"caption":"0.536/0.569",
    "overall":0.811,"decisive_DA":0.924,"decisive_cov":0.643,"eus_interception":0.679}}
(RES/"d2_clean_final.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
