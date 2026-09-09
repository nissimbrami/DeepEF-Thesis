import numpy as np, pickle, pandas as pd
D=pickle.load(open("scripts/_ens.pkl","rb")); M=D["M"]
true=np.load("scripts/_ens_true.npy"); prot=np.load("scripts/_ens_prot.npy",allow_pickle=True)
prots=sorted(set(prot)); idx={p:np.where(prot==p)[0] for p in prots}
sc=pd.read_csv("results/SCORE_ALL76.tsv",sep="\t")
order=[t for t in sc.sort_values("pooled",ascending=False).tag if t in M]
def k20(p,nb=250,seed0=5000):
    R=[]
    for s in range(nb):
        rng=np.random.default_rng(seed0+s); T,P=[],[]
        for pr in prots:
            ii=idx[pr]; y=true[ii]; x=p[ii]; n=len(ii)
            if n<=22: continue
            sel=rng.choice(n,20,replace=False); m=np.zeros(n,bool); m[sel]=True
            xa,ya=x[m],y[m]; xc=xa-xa.mean(); yc=ya-ya.mean()
            a=(xc@yc+3.0)/(xc@xc+3.0); b=ya.mean()-a*xa.mean()
            T.append(y[~m]); P.append(a*x[~m]+b)
        R.append(np.corrcoef(np.concatenate(T),np.concatenate(P))[0,1])
    return np.array(R)
best=M[order[0]]; ens5=np.mean([M[t] for t in order[:5]],axis=0); ctrl=M["calib_ctrl_repro2_e14"]
a=k20(best); b=k20(ens5); c=k20(ctrl)
for nm,x,y in [("top5 - best_single",a,b),("top5 - control",c,b),("best_single - control",c,a)]:
    d=y-x; lo,hi=np.percentile(d,[2.5,97.5])
    print(f"{nm:<24} {d.mean():+.4f}  CI [{lo:+.4f},{hi:+.4f}]  P(>0)={(d>0).mean():.3f}")
print(f"\nk20 means: best_single {a.mean():.4f}  top5_ens {b.mean():.4f}  control {c.mean():.4f}")
