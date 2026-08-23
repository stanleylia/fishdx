"""Recompute evaluation-of-record on D2-final = 2,176 shared + 226 held-out EUS test
(excludes the 226 EUS calibration images used to select theta_margin).
Faithful to exp_d2_clean_tables.py (margin/pred/corr) + exp_eus_theta_calibration.py (226/226 split)."""
import json, numpy as np
from pathlib import Path
RES=Path(__file__).resolve().parents[2] / "results"
SEED=42; DEDUP=0.95
def l2(a): return a/(np.linalg.norm(a,axis=-1,keepdims=True)+1e-12)
def wilson(k,n,z=1.96):
    if n==0: return (0.0,0.0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(float(c-h),4),round(float(c+h),4))

g=np.load(RES/"_cache_d1_gallery.npz",allow_pickle=True); gv,gc,gl=g["visual"],g["caption"],g["labels"]
q=np.load(RES/"_cache_d2_query.npz",allow_pickle=True); qv,qc,ql=q["visual"],q["caption"],q["labels"]
clean=(qv@gv.T).max(1)<DEDUP
qv,qc,ql=qv[clean],qc[clean],ql[clean]
N=len(ql); iseus=ql=="EUS"
shared=sorted((set(gl.tolist())&set(ql.tolist()))-{"EUS","UNKNOWN"}); m7=np.isin(ql,shared)
print("n_clean",N,"n_eus",int(iseus.sum()),"n_shared",int(m7.sum()))

# fused lambda=0.7 retrieval
gf=l2(0.7*gv+0.3*gc); qf=l2(0.7*qv+0.3*qc); sims=qf@gf.T
order=np.sort(sims,1); margin=order[:,-1]-order[:,-2]
pred=gl[sims.argmax(1)]; corr=pred==ql

# EUS 226/226 split (identical to exp_eus_theta_calibration)
eus_pos=np.where(iseus)[0]
rng=np.random.default_rng(SEED); idx=rng.permutation(len(eus_pos)); half=len(idx)//2
cal_pos=eus_pos[idx[:half]]; tst_pos=eus_pos[idx[half:]]
print("eus split: cal",len(cal_pos),"test",len(tst_pos))

# D2-final = everything EXCEPT the calibration EUS
final=np.ones(N,bool); final[cal_pos]=False
print("D2-final n =",int(final.sum()),"= shared",int(m7.sum()),"+ eus_test",int(final[eus_pos].sum()))

def macro_f1(dec_mask):
    # over 7 shared classes; predictions among DECIDED images; EUS-decided count as errors (never a shared TP)
    sel=dec_mask
    yp=pred[sel]; yt=ql[sel]
    fs=[]
    for c in shared:
        tp=int(((yp==c)&(yt==c)).sum()); fp=int(((yp==c)&(yt!=c)).sum()); fn=int(((yp!=c)&(yt==c)).sum())
        prec=tp/(tp+fp) if tp+fp else 0.0; rec=tp/(tp+fn) if tp+fn else 0.0
        fs.append(2*prec*rec/(prec+rec) if prec+rec else 0.0)
    return round(float(np.mean(fs)),4)

def sweep(mask,label):
    out={}
    sub=np.where(mask)[0]
    # forced choice (theta=0) DA on shared-only (Overlap) and overall
    ov_k=int(corr[mask].sum()); ov_n=int(mask.sum())
    mm7=mask&m7
    op_k=int(corr[mm7].sum()); op_n=int(mm7.sum())
    out["forced_choice_overall_DA"]=[round(ov_k/ov_n,4),ov_n,wilson(ov_k,ov_n)]
    out["forced_choice_overlap_DA"]=[round(op_k/op_n,4),op_n,wilson(op_k,op_n)]
    out["overlap_macro_f1"]=macro_f1(mm7)
    for th in [0.01,0.02,0.05]:
        dec=(margin>=th)&mask
        nd=int(dec.sum()); cov=nd/ov_n
        k=int(corr[dec].sum()); dd=k/nd if nd else 0.0
        eus_test_mask=mask&iseus
        eus_int=int(((margin<th)&eus_test_mask).sum())/max(1,int(eus_test_mask.sum()))
        out[f"theta_{th:.2f}"]={"decisive_DA":round(dd,4),"decisive_wilson":wilson(k,nd),
            "coverage":round(cov,4),"inconclusive_pct":round(1-cov,4),"n_decided":nd,
            "decided_macro_f1":macro_f1(dec),
            "eus_interception":round(eus_int,4),"n_eus_in_set":int(eus_test_mask.sum())}
    return out

res={"D2_clean_2628":sweep(np.ones(N,bool),"clean"),
     "D2_final_2402_excl_calib":sweep(final,"final")}
print(json.dumps(res,indent=1))
(RES/"d2_final_no_calib.json").write_text(json.dumps(res,indent=1))
