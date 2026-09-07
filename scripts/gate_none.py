"""Proper reproduction gate: the eval dataloader shuffles proteins, so compare by KEY."""
import numpy as np, pandas as pd
ref=pd.read_csv("eval_results/abl_calib_ctrl_repro2_e14.csv")
non=pd.read_csv("results/seqabl/abl_none_s42.csv")
print("rows ref=%d none=%d"%(len(ref),len(non)))
def key(d):
    d=d.copy(); d["r"]=d.groupby("protein").cumcount(); return d.set_index(["protein","r"])
a=key(ref).sort_index(); b=key(non).sort_index()
print("index identical:", a.index.equals(b.index))
common=a.index.intersection(b.index); print("common keys:",len(common))
a=a.loc[common]; b=b.loc[common]
ok=True
for c in ["deltaG","pred_deltaG","ddG","pred_ddG"]:
    d=np.abs(a[c].values-b[c].values); mx=np.nanmax(d)
    print("  %-12s maxabsdiff=%.3e  mean=%.3e"%(c,mx,np.nanmean(d)))
    if c.startswith("pred") and mx>1e-4: ok=False
print("GATE:", "PASS (bit-for-bit within 1e-4)" if ok else "FAIL")
# also compare protein-level aggregates, which are order-invariant
print("\nprotein-level order-invariant check (mean pred_ddG per protein):")
ga=ref.groupby("protein").pred_ddG.mean(); gb=non.groupby("protein").pred_ddG.mean()
j=pd.concat([ga,gb],axis=1,keys=["ref","none"]).dropna()
print("  n proteins=%d  max abs diff=%.3e  corr=%.8f"%(len(j),np.max(np.abs(j.ref-j.none)),np.corrcoef(j.ref,j.none)[0,1]))
gs=ref.groupby("protein").pred_ddG.std(); gt=non.groupby("protein").pred_ddG.std()
k=pd.concat([gs,gt],axis=1,keys=["ref","none"]).dropna()
print("  std  max abs diff=%.3e"%np.max(np.abs(k.ref-k.none)))
print("\npooled ddG PCC: ref=%.6f none=%.6f"%(
  np.corrcoef(ref.query("not(ddG==0 and pred_ddG==0)").ddG,ref.query("not(ddG==0 and pred_ddG==0)").pred_ddG)[0,1],
  np.corrcoef(non.query("not(ddG==0 and pred_ddG==0)").ddG,non.query("not(ddG==0 and pred_ddG==0)").pred_ddG)[0,1]))
