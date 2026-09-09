import pandas as pd, numpy as np, glob, os, json
rows=[]
for f in sorted(glob.glob("eval_results/*.csv")):
    tag=os.path.basename(f)[:-4].replace("abl_","")
    try: d=pd.read_csv(f)
    except Exception as e: continue
    d=d[d.protein!='2K5H']
    if len(d)<1000 or d.pred_ddG.std()<1e-12: continue
    pooled=np.corrcoef(d.ddG,d.pred_ddG)[0,1]
    # per-protein mean PCC + a_p (median per-protein slope pred~true)
    pp=[]; ap=[]
    for p in d.protein.unique():
        g=d[d.protein==p]
        if g.ddG.std()>1e-9 and g.pred_ddG.std()>1e-9:
            pp.append(np.corrcoef(g.ddG,g.pred_ddG)[0,1])
            ap.append(np.polyfit(g.ddG.values,g.pred_ddG.values,1)[0])
    # offset-only oracle
    T,P=[],[]
    for p in d.protein.unique():
        g=d[d.protein==p]; x=g.pred_ddG.values; y=g.ddG.values
        T.append(y); P.append(x-(x.mean()-y.mean()))
    orc_off=np.corrcoef(np.concatenate(T),np.concatenate(P))[0,1]
    # k=20 ridge lam=3
    prots=sorted(d.protein.unique()); G={p:d[d.protein==p] for p in prots}
    R=[]
    for s in range(60):
        rng=np.random.default_rng(1000+s); TT,PP=[],[]
        for p in prots:
            g=G[p]; n=len(g); y=g.ddG.values; x=g.pred_ddG.values
            if n<=22: continue
            idx=rng.choice(n,20,replace=False); m=np.zeros(n,bool); m[idx]=True
            xa,ya=x[m],y[m]; xc=xa-xa.mean(); yc=ya-ya.mean()
            a=(xc@yc+3.0)/(xc@xc+3.0); b=ya.mean()-a*xa.mean()
            TT.append(y[~m]); PP.append(a*x[~m]+b)
        R.append(np.corrcoef(np.concatenate(TT),np.concatenate(PP))[0,1])
    rows.append(dict(tag=tag,pooled=pooled,ppmean=np.mean(pp),ap=np.median(ap),
                     orc=orc_off,k20=np.mean(R),n=len(d)))
df=pd.DataFrame(rows).sort_values("pooled",ascending=False)
df.to_csv("results/SCORE_ALL76.tsv",sep="\t",index=False,float_format="%.4f")
print(f"scored {len(df)} of 76\n")
print(f"{'tag':<40}{'pooled':>8}{'ppmean':>8}{'a_p':>7}{'orc':>8}{'k=20':>8}")
for _,r in df.iterrows():
    print(f"{r.tag:<40}{r.pooled:>8.4f}{r.ppmean:>8.4f}{r.ap:>7.3f}{r.orc:>8.4f}{r.k20:>8.4f}")
