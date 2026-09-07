"""Ofir Ezrielev enrichment metric (thesis p.21 eq.) applied to DeepEF eval CSVs.

Enrichment = |argmax_{k}(y) INTERSECT argmax_{k}(yhat)| / ( N * (k/N)^2 )
where k = floor(X% * N).  Chance = 1.0.  Theoretical max = 1/(k/N) = N/k.
"""
import os, glob, math, json
import numpy as np, pandas as pd

EV = "eval_results"

def enrich(y, yhat, pct, low=False):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    N = len(y)
    k = int(math.floor(pct * N))
    if k < 1: return float('nan'), 0, N, float('nan')
    s = 1.0 if low else -1.0     # low=True -> ascending (bottom); else descending (top)
    ty = set(np.argsort(s*y, kind='stable')[:k].tolist())
    tp = set(np.argsort(s*yhat, kind='stable')[:k].tolist())
    hits = len(ty & tp)
    denom = N * (k/N)**2
    return hits/denom, hits, N, N/k   # value, hits, N, theoretical max

def pcc(a, b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)<3 or np.std(a)==0 or np.std(b)==0: return float('nan')
    return float(np.corrcoef(a,b)[0,1])

files = sorted(glob.glob(os.path.join(EV,"*.csv")))
print("FILES", len(files))
rows=[]
per_prot_rows=[]
for f in files:
    d = pd.read_csv(f)
    tag = os.path.basename(f).replace(".csv","")
    # drop WT self rows (ddG==0 and pred==0 exact) ? keep both variants
    for label, dd in (("all", d), ("nonWT", d[~((d.ddG==0)&(d.pred_ddG==0))])):
        if len(dd)==0: continue
        r = {"file":tag, "subset":label, "N":len(dd),
             "PCC_pooled": pcc(dd.ddG, dd.pred_ddG),
             "PCC_dG_pooled": pcc(dd.deltaG, dd.pred_deltaG)}
        for pct,nm in ((0.05,"5"),(0.15,"15")):
            # ddG: destabilizing = HIGH ddG in this convention? check both ends
            for low,end in ((False,"top"),(True,"bot")):
                v,h,N,mx = enrich(dd.ddG, dd.pred_ddG, pct, low)
                if False: pass
            v,h,N,mx = enrich(dd.ddG, dd.pred_ddG, pct, False)
            r[f"E{nm}_top_ddG"]=v; r[f"E{nm}_max"]=mx; r[f"E{nm}_top_hits"]=h
            v,h,N,mx = enrich(dd.ddG, dd.pred_ddG, pct, True)
            r[f"E{nm}_bot_ddG"]=v; r[f"E{nm}_bot_hits"]=h
            v,_,_,_ = enrich(dd.deltaG, dd.pred_deltaG, pct, False)
            r[f"E{nm}_top_dG"]=v
            v,_,_,_ = enrich(dd.deltaG, dd.pred_deltaG, pct, True)
            r[f"E{nm}_bot_dG"]=v
        rows.append(r)
    # per-protein enrichment (within-protein design use case) on nonWT
    dd = d[~((d.ddG==0)&(d.pred_ddG==0))]
    for p,g in dd.groupby("protein"):
        if len(g) < 40: continue
        e5t,_,N,mx5 = enrich(g.ddG, g.pred_ddG, 0.05, False)
        e5b,_,_,_   = enrich(g.ddG, g.pred_ddG, 0.05, True)
        e15t,_,_,mx15=enrich(g.ddG, g.pred_ddG, 0.15, False)
        e15b,_,_,_  = enrich(g.ddG, g.pred_ddG, 0.15, True)
        per_prot_rows.append({"file":tag,"protein":p,"n":len(g),
            "pcc":pcc(g.ddG,g.pred_ddG),
            "E5top":e5t,"E5bot":e5b,"E5max":mx5,
            "E15top":e15t,"E15bot":e15b,"E15max":mx15})

R = pd.DataFrame(rows); P = pd.DataFrame(per_prot_rows)
R.to_csv("results/enrichment_pooled.csv", index=False)
P.to_csv("results/enrichment_perprotein.csv", index=False)

pd.set_option("display.width",250); pd.set_option("display.max_columns",50)
print("\n===== POOLED (all 28 proteins mixed, nonWT rows) =====")
sub = R[R.subset=="nonWT"]
print(sub[["file","N","PCC_pooled","E5_top_ddG","E5_bot_ddG","E5_max","E15_top_ddG","E15_bot_ddG","E15_max"]].to_string(index=False))
print("\nmean E5_top=%.3f  E5_bot=%.3f  E15_top=%.3f  E15_bot=%.3f  (chance=1.0, max5=%.1f max15=%.2f)"%(
    sub.E5_top_ddG.mean(), sub.E5_bot_ddG.mean(), sub.E15_top_ddG.mean(), sub.E15_bot_ddG.mean(),
    sub.E5_max.mean(), sub.E15_max.mean()))
print("\n--- dG pooled enrichment (reference-state sensitive) ---")
print(sub[["file","E5_top_dG","E5_bot_dG","E15_top_dG","E15_bot_dG","PCC_dG_pooled"]].to_string(index=False))

print("\n===== PER-PROTEIN (design use case: rank mutations within one protein) =====")
agg = P.groupby("protein").agg(n=("n","first"), pcc=("pcc","mean"),
      E5top=("E5top","mean"), E5bot=("E5bot","mean"), E5max=("E5max","first"),
      E15top=("E15top","mean"), E15bot=("E15bot","mean")).sort_values("E5top", ascending=False)
print(agg.to_string())
print("\nAcross %d proteins x %d ckpts: mean E5top=%.3f  E5bot=%.3f  E15top=%.3f  E15bot=%.3f"%(
    P.protein.nunique(), P.file.nunique(), P.E5top.mean(), P.E5bot.mean(), P.E15top.mean(), P.E15bot.mean()))
print("median E5top=%.3f E5bot=%.3f"%(P.E5top.median(), P.E5bot.median()))
print("frac protein-ckpt cells with E5top>1 (better than chance): %.3f  E5bot>1: %.3f"%(
    (P.E5top>1).mean(), (P.E5bot>1).mean()))
print("frac E5top==0 (total miss): %.3f   E5bot==0: %.3f"%((P.E5top==0).mean(),(P.E5bot==0).mean()))
n=agg.shape[0]
print("\nn proteins = %d ; |r| threshold p=.05 = %.3f"%(n, 2/math.sqrt(n-2)))
print("corr(per-protein PCC, E5top) = %.4f"%pcc(agg.pcc, agg.E5top))
print("corr(per-protein PCC, E5bot) = %.4f"%pcc(agg.pcc, agg.E5bot))
print("corr(per-protein PCC, E15top)= %.4f"%pcc(agg.pcc, agg.E15top))
print("corr(n, E5top) = %.4f"%pcc(agg.n, agg.E5top))
print("\nWROTE results/enrichment_pooled.csv results/enrichment_perprotein.csv")
