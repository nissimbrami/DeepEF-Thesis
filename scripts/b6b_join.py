import torch, os, pandas as pd, numpy as np
D="data/S669"
ids=torch.load(os.path.join(D,"all_ids.pt"),map_location="cpu",weights_only=False)
uniq=sorted(set(str(i).split("_")[0].upper() for i in ids))
c=pd.read_csv("data/FINAL_DATASET_100k_030926.csv", low_memory=False)
print("catalogue:", c.shape)
# find the id column by content, not by name
best=None
for col in c.columns:
    if c[col].dtype==object:
        v=c[col].astype(str).str.upper().str.strip()
        hit=v.str[:4].isin(uniq).sum()
        if hit>0 and (best is None or hit>best[1]): best=(col,hit)
print("best id-like column:", best)
if best:
    col=best[0]
    c["_k"]=c[col].astype(str).str.upper().str.strip().str[:4]
    sub=c[c["_k"].isin(uniq)].copy()
    print("S669 PDBs matched: %d of %d  (%d catalogue rows)" % (sub["_k"].nunique(), len(uniq), len(sub)))
    het=pd.read_csv("data/het_classification.csv")
    bound=set(het[het.counts_as_bound_ligand.astype(str).str.strip().str.lower()=="yes"].het_code.astype(str).str.upper())
    print("het codes counting as a bound ligand: %d" % len(bound))
    L=sub["Ligands_y"].fillna("").astype(str)
    def codes(s):
        return [t.strip().upper() for t in s.replace(";",",").split(",") if t.strip()]
    sub["_has_real"]=L.map(lambda s: any(t in bound for t in codes(s)))
    sub["_nlig"]=L.map(lambda s: len(codes(s)))
    per=sub.groupby("_k").agg(has_real=("_has_real","max"), nlig=("_nlig","max"))
    print("")
    print("  S669 proteins with ANY het code      : %d of %d" % ((per.nlig>0).sum(), len(per)))
    print("  S669 proteins with a REAL bound ligand: %d of %d" % (per.has_real.sum(), len(per)))
    print("")
    print("  -> our 27 MegaScale test proteins: 0 of 27 (all single-chain, ligand-free, metal-free)")
    if per.has_real.sum()>0:
        print("  examples:", list(per[per.has_real].index[:12]))
        ex=sub[sub._has_real].groupby("_k")["Ligands_y"].first().head(8)
        for k,v in ex.items(): print("     %s: %s" % (k, str(v)[:70]))
    metal=set(het[het.classification.astype(str).str.lower()=="metal"].het_code.astype(str).str.upper())
    sub["_has_metal"]=L.map(lambda s: any(t in metal for t in codes(s)))
    pm=sub.groupby("_k")["_has_metal"].max()
    print("")
    print("  S669 proteins with a METAL: %d of %d" % (pm.sum(), len(pm)))
