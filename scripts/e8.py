import pandas as pd, numpy as np, re, os, torch
from scipy import stats

MUT="data/Processed_K50_dG_datasets/mutation_datasets"
KD=dict(zip("ACDEFGHIKLMNPQRSTVWY",
  [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))

def burial(p):
    f="/groups/keasar_group/casp15/meytav/protein_tensors/%s/coords_tensor.pt"%p
    if not os.path.exists(f): return None
    x=torch.load(f,map_location="cpu",weights_only=False).float()
    cb=x[:,3,:] if x.shape[1]>=4 else x[:,1,:]
    n=((torch.cdist(cb,cb)<10.0).float().sum(1)-1)
    return (n/20.0).clamp(0,1).numpy()

def muts(p):
    f=os.path.join(MUT,p+".csv")
    if not os.path.exists(f): return None
    d=pd.read_csv(f); rows=[]
    for nm,dg in zip(d["name"].astype(str), d["deltaG"].values):
        m=re.search(r"\.pdb_([A-Z])(\d+)([A-Z])$", nm)
        if m: rows.append((round(float(dg),5), m.group(1), int(m.group(2)), m.group(3)))
    r=pd.DataFrame(rows,columns=["k","wt_aa","pos","mut_aa"])
    return r.drop_duplicates("k")

def ev(t):
    d=pd.read_csv("eval_results/abl_%s.csv"%t); d=d[d.protein!="2K5H"].copy()
    d["k"]=d.deltaG.round(5); return d

CTRL=["calib_ctrl_repro2_e14","sigma_seed1_e13","sigma_seed2_e10","sigma_seed3_e13",
      "sigma_seed4_e14","sigma_seed42_e9"]
W12=["w12_s1_e14","w12_s2_e12"]
print("E8 - W12 mechanism: is the gain concentrated where packing matters?")
print("")
# ---- per-protein: does the gain track packing_frac? ----
M=pd.read_csv("results/08_data/structfeat/master_28.csv")
key=[c for c in M.columns if M[c].dtype==object][0]
def pp(t):
    d=ev(t)
    return {p:np.corrcoef(g.ddG,g.pred_ddG)[0,1] for p,g in d.groupby("protein")
            if g.ddG.std()>1e-9 and g.pred_ddG.std()>1e-9}
C={}; W={}
for t in CTRL:
    for p,v in pp(t).items(): C.setdefault(p,[]).append(v)
for t in W12:
    for p,v in pp(t).items(): W.setdefault(p,[]).append(v)
rows=[]
for p in sorted(set(C)&set(W)):
    m=M[M[key].astype(str).str.contains(str(p),case=False,na=False)]
    if not len(m): continue
    r=m.iloc[0]
    try: pf=float(r["packing_frac"]); vv=float(r["void_vol_per_res"])
    except Exception: continue
    rows.append((p, np.mean(W[p])-np.mean(C[p]), pf, vv))
D=pd.DataFrame(rows,columns=["protein","gain","packing_frac","void"])
print("PER-PROTEIN (n=%d):" % len(D))
for c in ("packing_frac","void"):
    rr,ppv=stats.pearsonr(D[c],D.gain)
    print("  corr(W12 gain, %-14s) = %+.4f   p=%.4f" % (c,rr,ppv))
hi=D[D.packing_frac>=D.packing_frac.median()]; lo=D[D.packing_frac<D.packing_frac.median()]
print("  tightly packed (top half): mean gain %+.4f   loosely packed: %+.4f"
      % (hi.gain.mean(), lo.gain.mean()))
print("")
# ---- per-mutation: buried vs exposed, hydrophobic vs polar ----
acc=[]
for p in sorted(set(C)&set(W)):
    b=burial(p); mu=muts(p)
    if b is None or mu is None: continue
    c=ev(CTRL[0]); c=c[c.protein==p][["k","ddG","pred_ddG"]]
    j=c.merge(mu,on="k",how="inner")
    j=j[(j.pos>=1)&(j.pos<=len(b))]
    if len(j)<30: continue
    j["bur"]=b[j.pos.values-1]
    j["e_c"]=(j.pred_ddG-j.ddG).abs()
    for t in W12:
        w=ev(t); w=w[w.protein==p][["k","pred_ddG"]]
        jj=j.merge(w,on="k",how="inner",suffixes=("","_w"))
        if len(jj)<30: continue
        jj["gain"]=jj.e_c-(jj.pred_ddG_w-jj.ddG).abs()
        acc.append(jj[["bur","gain","mut_aa"]])
A=pd.concat(acc,ignore_index=True)
bu=A[A.bur>=0.5]; ex=A[A.bur<0.25]
print("PER-MUTATION (n=%d mutation-seed pairs, %d proteins):" % (len(A), len(acc)//2))
print("  %-24s %8s %11s %11s" % ("","n","mean gain","median"))
print("  %-24s %8d %11.5f %11.5f" % ("BURIED  (>=0.50)",len(bu),bu.gain.mean(),bu.gain.median()))
print("  %-24s %8d %11.5f %11.5f" % ("EXPOSED (<0.25)",len(ex),ex.gain.mean(),ex.gain.median()))
t=stats.ttest_ind(bu.gain,ex.gain,equal_var=False)
print("  buried - exposed = %+.5f   Welch t=%.3f  p=%.4g" % (bu.gain.mean()-ex.gain.mean(),t.statistic,t.pvalue))
print("  corr(burial, gain) = %+.4f" % np.corrcoef(A.bur,A.gain)[0,1])
h=A[A.mut_aa.map(KD)>1.5]; pol=A[A.mut_aa.map(KD)<-1.5]
print("  to HYDROPHOBIC: n=%6d mean %+.5f median %+.5f" % (len(h),h.gain.mean(),h.gain.median()))
print("  to POLAR      : n=%6d mean %+.5f median %+.5f" % (len(pol),pol.gain.mean(),pol.gain.median()))
bh=A[(A.bur>=0.5)&(A.mut_aa.map(KD)>1.5)]
print("  BURIED+HYDROPHOBIC (the FINDINGS 3.4 deficit): n=%6d mean %+.5f median %+.5f"
      % (len(bh),bh.gain.mean(),bh.gain.median()))
