"""Real ablations for reviewer response: class-prototype baseline + sanity, on clean D2.
Reuses the exact recipe from exp_d2_clean_tables.py (CLIP ViT-B-32 laion2b, D1 gallery cache)."""
from pathlib import Path
import json, numpy as np, torch, open_clip
from PIL import Image
from tqdm import tqdm

ROOT=Path(__file__).resolve().parents[2]; RES=ROOT/"results"
D2_DIR=ROOT/"data/datasets/D2"; CACHE=RES/"_cache_d1_gallery.npz"
DEV="cuda:0" if torch.cuda.is_available() else "cpu"; DEDUP=0.95
EXTS={".jpg",".jpeg",".png",".bmp",".webp"}
CANON={'aeromoniasis':'aeromoniasis','bacterial diseases - aeromoniasis':'aeromoniasis',
 'bacterial gill disease':'bacterial_gill','bacterial red disease':'bacterial_red',
 'fungal diseases saprolegniasis':'fungal','parasitic diseases':'parasitic',
 'viral diseases white tail disease':'viral_white_tail','viral white tail disease':'viral_white_tail',
 'healthy fish':'healthy','eus':'EUS'}
def canon(s):
    s=s.lower().replace('_',' ').strip()
    for k,v in CANON.items():
        if k in s: return v
    return 'UNKNOWN'
def l2(x): return x/np.linalg.norm(x,axis=-1,keepdims=True)

z=np.load(CACHE,allow_pickle=True); gv,gc,gl=z["visual"],z["caption"],z["labels"]
clip,_,prep=open_clip.create_model_and_transforms("ViT-B-32",pretrained="laion2b_s34b_b79k",device=DEV)
clip.eval(); tok=open_clip.get_tokenizer("ViT-B-32")
caps={c["image"]:(c.get("caption") or "") for c in json.loads((RES/"e3_captions_log.json").read_text())}
def ei(p):
    with torch.no_grad():
        x=prep(Image.open(p).convert("RGB")).unsqueeze(0).to(DEV); f=clip.encode_image(x)
        return (f/f.norm(dim=-1,keepdim=True))[0].cpu().numpy().astype(np.float32)
def et(t):
    with torch.no_grad():
        f=clip.encode_text(tok([t or "fish"]).to(DEV))
        return (f/f.norm(dim=-1,keepdim=True))[0].cpu().numpy().astype(np.float32)

paths=sorted(p for p in D2_DIR.rglob("*") if p.suffix.lower() in EXTS)
qv,qc,ql=[],[],[]
for p in tqdm(paths,desc="D2 encode"):
    qv.append(ei(p)); qc.append(et(caps.get(p.name,""))); ql.append(canon(p.stem))
qv,qc,ql=np.array(qv),np.array(qc),np.array(ql)
np.savez(RES/"_cache_d2_query.npz",visual=qv,caption=qc,labels=ql)  # cache for Qwen job
clean=(qv@gv.T).max(1)<DEDUP; qv,qc,ql=qv[clean],qc[clean],ql[clean]
shared=sorted((set(gl.tolist())&set(ql.tolist()))-{"EUS","UNKNOWN"}); m7=np.isin(ql,shared)
print(f"n_clean={len(ql)} n_overlap7={int(m7.sum())} shared={shared}")

def da(pred,mask):
    p,t=pred[mask],ql[mask]; return round(float((p==t).mean()),4),int(mask.sum())
def preds(lam):
    g=l2(lam*gv+(1-lam)*gc); q=l2(lam*qv+(1-lam)*qc); return gl[(q@g.T).argmax(1)]
res={}
res['caption_only_florence']=da(preds(0.0),m7)   # sanity ~0.495
res['visual_only']=da(preds(1.0),m7)             # sanity ~0.893
res['fusion_0.7']=da(preds(0.7),m7)              # sanity ~0.897

# CLASS-PROTOTYPE baseline (NEW): gallery = per-class mean visual embedding
classes=sorted(set(gl.tolist()))
proto=l2(np.array([gv[gl==c].mean(0) for c in classes]))
proto_pred=np.array(classes)[ (l2(qv)@proto.T).argmax(1) ]
res['class_prototype_visual']=da(proto_pred,m7)
# also caption-prototype
protoc=l2(np.array([gc[gl==c].mean(0) for c in classes]))
protoc_pred=np.array(classes)[ (l2(qc)@protoc.T).argmax(1) ]
res['class_prototype_caption']=da(protoc_pred,m7)

print(json.dumps(res,indent=1))
json.dump({'n_clean':int(clean.sum()),'n_overlap7':int(m7.sum()),'results':res,
  'note':'DA=top-1 NN label match, 7 shared classes, clean D1-disjoint D2; class_prototype = per-class mean embedding gallery'},
  open(RES/"class_prototype_ablation.json","w"),indent=1)
print("saved results/class_prototype_ablation.json")
