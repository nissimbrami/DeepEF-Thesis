import pandas as pd, numpy as np, glob, os
sc=pd.read_csv("results/SCORE_ALL76.tsv",sep="\t")
# healthy D0 runs only
bad=("D1_uemb","loroWdesc2","sa_uembmean","sa_coil","anchor_w3.0","dg_coil")
healthy=[t for t in sc.tag if not any(b in t for b in bad)]
healthy=[t for t in healthy if sc[sc.tag==t].iloc[0].pooled>0.45]
print(f"healthy runs: {len(healthy)}")
def load(t):
    d=pd.read_csv(f"eval_results/abl_{t}.csv")
    d=d[d.protein!='2K5H']
    d=d.groupby(["protein","deltaG"],as_index=False).first()   # key
    return d.sort_values(["protein","deltaG"]).reset_index(drop=True)
base=load(healthy[0])
key=base[["protein","deltaG"]].copy()
M={}; kept=[]
for t in healthy:
    d=load(t)
    if len(d)!=len(base) or not (d.protein.values==base.protein.values).all(): continue
    if not np.allclose(d.deltaG.values,base.deltaG.values,atol=1e-6): continue
    M[t]=d.pred_ddG.values; kept.append(t)
print(f"aligned: {len(kept)}  rows={len(base)}  proteins={base.protein.nunique()}")
true=base.ddG.values
np.save("scripts/_ens_true.npy",true)
np.save("scripts/_ens_prot.npy",base.protein.values)
import pickle; pickle.dump({"M":M,"kept":kept},open("scripts/_ens.pkl","wb"))
print("\npairwise corr between run predictions:")
K=kept[:12]; C=np.corrcoef(np.array([M[t] for t in K]))
off=C[np.triu_indices(len(K),1)]
print(f"  n_pairs={len(off)}  mean {off.mean():.4f}  min {off.min():.4f}  max {off.max():.4f}")
