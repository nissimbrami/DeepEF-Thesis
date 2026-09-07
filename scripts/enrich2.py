"""Deep enrichment analysis: (a) does the offset b_p destroy enrichment? (Ofir's claim: enrichment
is the metric that SURVIVES a mean shift, because it is rank-based within a set.)
(b) enrichment vs a_p slope. (c) baselines: hydrophobicity/position-only nulls.
"""
import os, glob, math
import numpy as np, pandas as pd

def enrich(y, yhat, pct, low=False):
    y=np.asarray(y,float); yhat=np.asarray(yhat,float); N=len(y)
    k=int(math.floor(pct*N))
    if k<1: return float('nan')
    s=1.0 if low else -1.0
    ty=set(np.argsort(s*y,kind='stable')[:k].tolist()); tp=set(np.argsort(s*yhat,kind='stable')[:k].tolist())
    return len(ty&tp)/(N*(k/N)**2)

def pcc(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)<3 or np.std(a)==0 or np.std(b)==0: return float('nan')
    return float(np.corrcoef(a,b)[0,1])

files=sorted(glob.glob("eval_results/*.csv"))
rows=[]
for f in files:
    d=pd.read_csv(f); tag=os.path.basename(f)[:-4]
    d=d[~((d.ddG==0)&(d.pred_ddG==0))]
    # per-protein a_p,b_p on dG
    ab={}
    for p,g in d.groupby("protein"):
        if len(g)<40: continue
        A=np.polyfit(g.deltaG, g.pred_deltaG, 1)
        ab[p]=(A[0],A[1])
    # (1) POOLED enrichment before vs after removing the per-protein offset b_p
    dd=d[d.protein.isin(ab)].copy()
    dd["bp"]=dd.protein.map(lambda p: ab[p][1]); dd["ap"]=dd.protein.map(lambda p: ab[p][0])
    dd["pred_dG_nob"]=dd.pred_deltaG-dd.bp
    dd["pred_dG_oracle"]=(dd.pred_deltaG-dd.bp)/dd.ap
    for pct in (0.05,0.15):
        rows.append(dict(file=tag,pct=pct,
          dG_raw_top=enrich(dd.deltaG,dd.pred_deltaG,pct),
          dG_raw_bot=enrich(dd.deltaG,dd.pred_deltaG,pct,True),
          dG_nob_top=enrich(dd.deltaG,dd.pred_dG_nob,pct),
          dG_nob_bot=enrich(dd.deltaG,dd.pred_dG_nob,pct,True),
          dG_orc_top=enrich(dd.deltaG,dd.pred_dG_oracle,pct),
          dG_orc_bot=enrich(dd.deltaG,dd.pred_dG_oracle,pct,True),
          ddG_top=enrich(dd.ddG,dd.pred_ddG,pct),
          ddG_bot=enrich(dd.ddG,dd.pred_ddG,pct,True),
          N=len(dd)))
R=pd.DataFrame(rows)
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
print("=== POOLED dG enrichment: raw vs offset-removed vs full-oracle (mean over 10 ckpts) ===")
print(R.groupby("pct").mean(numeric_only=True).round(3).to_string())
print("\nper-file, pct=0.05:")
print(R[R.pct==0.05].round(3).to_string(index=False))
R.to_csv("results/enrichment_offset.csv",index=False)

# (2) per-protein: does enrichment care about b_p at all?
pr=[]
for f in files:
    d=pd.read_csv(f); tag=os.path.basename(f)[:-4]
    d=d[~((d.ddG==0)&(d.pred_ddG==0))]
    for p,g in d.groupby("protein"):
        if len(g)<40: continue
        A=np.polyfit(g.deltaG,g.pred_deltaG,1)
        pr.append(dict(file=tag,protein=p,n=len(g),ap=A[0],bp=A[1],
            pcc=pcc(g.ddG,g.pred_ddG),
            E5top=enrich(g.ddG,g.pred_ddG,0.05),E5bot=enrich(g.ddG,g.pred_ddG,0.05,True),
            E15top=enrich(g.ddG,g.pred_ddG,0.15),E15bot=enrich(g.ddG,g.pred_ddG,0.15,True),
            E5dGtop=enrich(g.deltaG,g.pred_deltaG,0.05),E5dGbot=enrich(g.deltaG,g.pred_deltaG,0.05,True),
            rng=g.ddG.max()-g.ddG.min(), sd=g.ddG.std(),
            skew=float(pd.Series(g.ddG).skew())))
P=pd.DataFrame(pr); P.to_csv("results/enrichment_bp.csv",index=False)
A=P.groupby("protein").mean(numeric_only=True)
n=len(A); thr=2/math.sqrt(n-2)
print("\n=== PER-PROTEIN: what predicts enrichment? n=%d, |r|>%.3f significant at p=.05 ==="%(n,thr))
for tgt in ["E5top","E5bot","E15top","E15bot"]:
    print("  %-7s : ap %+.3f | bp %+.3f | |bp| %+.3f | pcc %+.3f | n %+.3f | sd %+.3f | skew %+.3f"%(
      tgt,pcc(A["ap"],A[tgt]),pcc(A["bp"],A[tgt]),pcc(A["bp"].abs(),A[tgt]),pcc(A["pcc"],A[tgt]),
      pcc(A["n"],A[tgt]),pcc(A["sd"],A[tgt]),pcc(A["skew"],A[tgt])))
print("\nNOTE: within a protein, ddG enrichment is INVARIANT to b_p by construction (b_p is a")
print("constant added to every mutant -> ranks unchanged). The bp column above is a sanity")
print("check that it really is ~0, not evidence of anything.")

# (3) NULL baselines for enrichment
print("\n=== NULL BASELINES (per-protein E5, mean over 28 proteins) ===")
rng=np.random.RandomState(0)
d=pd.read_csv(files[0]); d=d[~((d.ddG==0)&(d.pred_ddG==0))]
nt=[];nb=[]
for p,g in d.groupby("protein"):
    if len(g)<40: continue
    v=[];w=[]
    for _ in range(200):
        r=rng.permutation(len(g))
        v.append(enrich(g.ddG.values,g.ddG.values[r],0.05))
        w.append(enrich(g.ddG.values,g.ddG.values[r],0.05,True))
    nt.append(np.mean(v)); nb.append(np.mean(w))
print("random shuffle null: E5top=%.3f  E5bot=%.3f (expected 1.0)"%(np.mean(nt),np.mean(nb)))
print("95th pct of shuffle null across proteins: top=%.3f bot=%.3f"%(np.percentile(nt,95),np.percentile(nb,95)))
print("\nWROTE results/enrichment_offset.csv results/enrichment_bp.csv")
