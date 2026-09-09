import pandas as pd, numpy as np
def load(t):
    d=pd.read_csv(f"eval_results/{t}.csv"); return d[d.protein!='2K5H']
def ks(d,k,lam,nboot=400):
    if k==0: return np.corrcoef(d.ddG,d.pred_ddG)[0,1],0.0
    prots=sorted(d.protein.unique()); G={p:d[d.protein==p] for p in prots}
    R=[]
    for s in range(nboot):
        rng=np.random.default_rng(7000+s); T,P=[],[]
        for p in prots:
            g=G[p]; n=len(g); y=g.ddG.values; x=g.pred_ddG.values
            if n<=k+2: continue
            idx=rng.choice(n,k,replace=False); m=np.zeros(n,bool); m[idx]=True
            xa,ya=x[m],y[m]
            if lam is None:  # FREE fit = the old, broken method
                a,b=(1.0,ya.mean()-xa.mean()) if xa.std()<1e-9 else np.polyfit(xa,ya,1)
            else:
                xc=xa-xa.mean(); yc=ya-ya.mean()
                a=(xc@yc+lam)/(xc@xc+lam); b=ya.mean()-a*xa.mean()
            T.append(y[~m]); P.append(a*x[~m]+b)
        R.append(np.corrcoef(np.concatenate(T),np.concatenate(P))[0,1])
    return float(np.mean(R)),float(np.std(R))

C=load("abl_calib_ctrl_repro2_e14"); W=load("abl_w12_s2_e12")
print("PAIRED bootstrap: same anchor draws for both arms, ridge lam=3\n")
print(f"{'k':>4} {'control':>18} {'W12':>18} {'W12-control':>22}")
for k in [0,5,10,20,50]:
    if k==0:
        c=np.corrcoef(C.ddG,C.pred_ddG)[0,1]; w=np.corrcoef(W.ddG,W.pred_ddG)[0,1]
        print(f"{k:>4} {c:>18.4f} {w:>18.4f} {w-c:>+22.4f}")
        continue
    prots=sorted(C.protein.unique())
    GC={p:C[C.protein==p] for p in prots}; GW={p:W[W.protein==p] for p in prots}
    D=[]; CS=[]; WS=[]
    for s in range(400):
        rng=np.random.default_rng(9000+s)
        Tc,Pc,Tw,Pw=[],[],[],[]
        for p in prots:
            gc=GC[p]; gw=GW[p]; n=len(gc)
            if n<=k+2: continue
            idx=rng.choice(n,k,replace=False); m=np.zeros(n,bool); m[idx]=True
            for g,T,P in ((gc,Tc,Pc),(gw,Tw,Pw)):
                y=g.ddG.values; x=g.pred_ddG.values
                xa,ya=x[m],y[m]; xc=xa-xa.mean(); yc=ya-ya.mean()
                a=(xc@yc+3.0)/(xc@xc+3.0); b=ya.mean()-a*xa.mean()
                T.append(y[~m]); P.append(a*x[~m]+b)
        rc=np.corrcoef(np.concatenate(Tc),np.concatenate(Pc))[0,1]
        rw=np.corrcoef(np.concatenate(Tw),np.concatenate(Pw))[0,1]
        CS.append(rc); WS.append(rw); D.append(rw-rc)
    D=np.array(D); lo,hi=np.percentile(D,[2.5,97.5])
    sig = "SIGNIFICANT" if (lo>0 or hi<0) else "n.s."
    print(f"{k:>4} {np.mean(CS):>18.4f} {np.mean(WS):>18.4f} {np.mean(D):>+12.4f} [{lo:+.4f},{hi:+.4f}] {sig}")
