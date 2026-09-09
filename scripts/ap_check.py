import pandas as pd, numpy as np, sys, os
print(f"{'run':<32}{'a_p':>8}{'r':>8}{'s':>8}{'k=0':>9}{'k=20':>9}")
for t in sys.argv[1:]:
    f=f"eval_results/{t}.csv"
    if not os.path.exists(f): print(f"{t:<32}  MISSING"); continue
    d=pd.read_csv(f); d=d[d.protein!='2K5H']
    a=np.polyfit(d.pred_ddG,d.ddG,1)[0]
    r=np.corrcoef(d.ddG,d.pred_ddG)[0,1]
    s=d.pred_ddG.std()/d.ddG.std()
    # k=20 ridge
    prots=sorted(d.protein.unique()); G={p:d[d.protein==p] for p in prots}
    R=[]
    for sd in range(200):
        rng=np.random.default_rng(1000+sd); T,P=[],[]
        for p in prots:
            g=G[p]; n=len(g); y=g.ddG.values; x=g.pred_ddG.values
            if n<=22: continue
            idx=rng.choice(n,20,replace=False); m=np.zeros(n,bool); m[idx]=True
            xa,ya=x[m],y[m]; xc=xa-xa.mean(); yc=ya-ya.mean()
            aa=(xc@yc+3.0)/(xc@xc+3.0); bb=ya.mean()-aa*xa.mean()
            T.append(y[~m]); P.append(aa*x[~m]+bb)
        R.append(np.corrcoef(np.concatenate(T),np.concatenate(P))[0,1])
    print(f"{t:<32}{a:>8.3f}{r:>8.3f}{s:>8.3f}{np.corrcoef(d.ddG,d.pred_ddG)[0,1]:>9.4f}{np.mean(R):>9.4f}")
