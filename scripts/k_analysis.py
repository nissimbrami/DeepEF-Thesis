import pandas as pd, numpy as np, os, sys
def load(p):
    d=pd.read_csv(p); return d[d.protein!='2K5H']
def pooled(d): return np.corrcoef(d.ddG,d.pred_ddG)[0,1]
def perprot(d):
    r=[]
    for p in d.protein.unique():
        g=d[d.protein==p]
        if g.ddG.std()>1e-9 and g.pred_ddG.std()>1e-9:
            r.append(np.corrcoef(g.ddG,g.pred_ddG)[0,1])
    return float(np.mean(r))
def kshot(d,k,nboot=200,lam=3.0):
    if k==0: return pooled(d),0.0
    prots=sorted(d.protein.unique()); G={p:d[d.protein==p] for p in prots}
    R=[]
    for s in range(nboot):
        rng=np.random.default_rng(1000+s); T,P=[],[]
        for p in prots:
            g=G[p]; n=len(g); y=g.ddG.values; x=g.pred_ddG.values
            if n<=k+2: continue
            idx=rng.choice(n,k,replace=False); m=np.zeros(n,bool); m[idx]=True
            xa,ya=x[m],y[m]; xc=xa-xa.mean(); yc=ya-ya.mean()
            a=(xc@yc+lam)/(xc@xc+lam); b=ya.mean()-a*xa.mean()
            T.append(y[~m]); P.append(a*x[~m]+b)
        R.append(np.corrcoef(np.concatenate(T),np.concatenate(P))[0,1])
    return float(np.mean(R)),float(np.std(R))
def oracle(d):
    T,P=[],[]
    for p in d.protein.unique():
        g=d[d.protein==p]; x=g.pred_ddG.values; y=g.ddG.values
        if x.std()<1e-9: continue
        a,b=np.polyfit(x,y,1); T.append(y); P.append(a*x+b)
    return np.corrcoef(np.concatenate(T),np.concatenate(P))[0,1]
print(f"{'run':<36}{'k=0':>8}{'k=5':>8}{'k=20':>8}{'sd20':>7}{'orac':>8}{'perP':>8}{'a_p':>7}{'n':>7}")
for t in sys.argv[1:]:
    f=f"eval_results/{t}.csv"
    if not os.path.exists(f): print(f"{t:<36}  MISSING"); continue
    d=load(f); a=np.polyfit(d.pred_ddG,d.ddG,1)[0]
    k0,_=kshot(d,0); k5,_=kshot(d,5); k20,s20=kshot(d,20)
    print(f"{t:<36}{k0:>8.4f}{k5:>8.4f}{k20:>8.4f}{s20:>7.4f}{oracle(d):>8.4f}{perprot(d):>8.4f}{a:>7.3f}{len(d):>7}")
