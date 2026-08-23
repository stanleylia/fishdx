"""Full D2-final (2,402, excl 226 calib EUS) recompute: Table 3 (all configs), Table 4 sweep,
deferring-kNN at matched coverage. Faithful to exp_d2_clean_tables + exp_eus_theta_calibration."""
import json, numpy as np
from pathlib import Path
RES=Path(__file__).resolve().parents[2] / "results"; SEED=42; DEDUP=0.95
def l2(a): return a/(np.linalg.norm(a,axis=-1,keepdims=True)+1e-12)
def wil(k,n,z=1.96):
    if n==0: return (0.0,0.0)
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return (round(float(c-h),4),round(float(c+h),4))
g=np.load(RES/"_cache_d1_gallery.npz",allow_pickle=True); gv,gc,gl=g["visual"],g["caption"],g["labels"]
q=np.load(RES/"_cache_d2_query.npz",allow_pickle=True); qv,qc,ql=q["visual"],q["caption"],q["labels"]
clean=(qv@gv.T).max(1)<DEDUP; qv,qc,ql=qv[clean],qc[clean],ql[clean]
N=len(ql); iseus=ql=="EUS"
shared=sorted((set(gl.tolist())&set(ql.tolist()))-{"EUS","UNKNOWN"}); m7=np.isin(ql,shared)
eus_pos=np.where(iseus)[0]; rng=np.random.default_rng(SEED); idx=rng.permutation(len(eus_pos)); half=len(idx)//2
cal_pos=eus_pos[idx[:half]]
final=np.ones(N,bool); final[cal_pos]=False   # D2-final = 2402
def preds(lam,k):
    G=l2(lam*gv+(1-lam)*gc); Q=l2(lam*qv+(1-lam)*qc); s=Q@G.T
    if k==1: return gl[s.argmax(1)]
    ix=np.argpartition(-s,k,axis=1)[:,:k]; out=[]
    for i in range(len(Q)):
        u={};
        # deterministic k-NN majority vote: float64 count + tiny similarity tie-break
        # (float(sc) forces float64 so the 1e-6 increment is never lost to float32 rounding)
        for lab,sc in zip(gl[ix[i]],s[i,ix[i]]): u[lab]=u.get(lab,0.0)+1.0+1e-6*float(sc)
        out.append(max(u,key=u.get))
    return np.array(out)
def macf1(yp,yt):
    fs=[]
    for c in shared:
        tp=int(((yp==c)&(yt==c)).sum());fp=int(((yp==c)&(yt!=c)).sum());fn=int(((yp!=c)&(yt==c)).sum())
        pr=tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0
        fs.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    return round(float(np.mean(fs)),4)

# ---- Table 3 on D2-final: Overall(2402) + Overlap(2176) ----
t3={}
for nm,lam,k in [("Pipeline/kNN(0.7,k1)",0.7,1),("kNN(1.0,k1)",1.0,1),("kNN(0.7,k5)",0.7,5),
                 ("kNN(1.0,k5)",1.0,5),("Caption-only(0.0,k1)",0.0,1)]:
    pr=preds(lam,k)
    ok=(pr==ql)
    ov_k=int(ok[final].sum()); ov_n=int(final.sum())
    op_k=int(ok[m7].sum()); op_n=int(m7.sum())
    t3[nm]={"overall_DA":round(ov_k/ov_n,4),"overall_wilson":wil(ov_k,ov_n),
            "overlap_DA":round(op_k/op_n,4),"overlap_wilson":wil(op_k,op_n)}
mac_overlap=macf1(preds(0.7,1)[m7],ql[m7])

# ---- Table 4 D2-final sweep ----
G=l2(0.7*gv+0.3*gc); Q=l2(0.7*qv+0.3*qc); s=Q@G.T
order=np.sort(s,1); margin=order[:,-1]-order[:,-2]; pred=gl[s.argmax(1)]; corr=pred==ql
t4={}
fc_k=int(corr[final].sum()); fc_n=int(final.sum())
t4["theta_0.00"]={"decisive_DA":round(fc_k/fc_n,4),"wilson":wil(fc_k,fc_n),"coverage":1.0,
                  "inconclusive_pct":0.0,"eus_interception":0.0}
for th in [0.01,0.02,0.05]:
    dec=(margin>=th)&final; nd=int(dec.sum()); k=int(corr[dec].sum())
    et=final&iseus; ei=int(((margin<th)&et).sum())/max(1,int(et.sum()))
    t4[f"theta_{th:.2f}"]={"decisive_DA":round(k/nd,4),"wilson":wil(k,nd),
        "coverage":round(nd/fc_n,4),"inconclusive_pct":round(1-nd/fc_n,4),
        "decided_macro_f1":macf1(pred[dec],ql[dec]),
        "eus_interception":round(ei,4),"n_eus":int(et.sum()),"n_decided":nd}
# ---- deferring visual kNN at matched coverage (on D2-final) ----
gvn=l2(gv); qvn=l2(qv); sv=qvn@gvn.T; ov=np.sort(sv,1); vm=ov[:,-1]-ov[:,-2]; vcorr=(gl[sv.argmax(1)]==ql)
tcov=t4["theta_0.02"]["coverage"]
# threshold on D2-final margins to match coverage
fm=vm[final]; thr=np.quantile(fm,1-tcov); vdec=(vm>=thr)&final
defknn={"coverage":round(float(vdec.sum()/final.sum()),4),"decisive_DA":round(float(vcorr[vdec].mean()),4)}
out={"n_d2final":int(final.sum()),"n_overlap":int(m7.sum()),"n_eus_test":int((final&iseus).sum()),
     "table3_d2final":t3,"overlap_macro_f1":mac_overlap,"table4_d2final":t4,
     "deferring_knn_matched_cov":defknn,
     "gain_pp_decisive_vs_forcedchoice":round(t4["theta_0.02"]["decisive_DA"]-t4["theta_0.00"]["decisive_DA"],4)}
(RES/"d2_final_full.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
