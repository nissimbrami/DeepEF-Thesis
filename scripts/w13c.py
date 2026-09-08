# VERIFY the convergence: is the k=20 ceiling really independent of the arm?
# If yes, that is a structural fact about the DATA, not about any model we trained.
import glob,os,numpy as np,pandas as pd
def load(f):
    d=pd.read_csv(f)
    if not {'protein','ddG','pred_ddG'}.issubset(d.columns): return None
    d=d[['protein','ddG','pred_ddG']].dropna(); d=d[d.protein.astype(str).str.upper()!='2K5H']
    return d if len(d)>100 and d.protein.nunique()>=20 else None
def pooled(y,p): return float(np.corrcoef(p,y)[0,1])
def oracle(d):
    ys,ps=[],[]
    for _,g in d.groupby('protein'):
        ys.append(g.ddG.values); ps.append(g.pred_ddG.values-(g.pred_ddG.values-g.ddG.values).mean())
    return pooled(np.concatenate(ys),np.concatenate(ps))
rows=[]
for f in sorted(glob.glob('/home/nissimb/DeepPEF/eval_results/abl_*_e*.csv')):
    d=load(f)
    if d is None: continue
    k0=pooled(d.ddG.values,d.pred_ddG.values)
    rows.append((os.path.basename(f)[4:-4],k0,oracle(d)))
df=pd.DataFrame(rows,columns=['run','k0','oracle'])
df=df[df.k0>0.4]
print(f"n runs with pooled>0.4: {len(df)}")
print(f"  zero-shot pooled : mean {df.k0.mean():.4f}  sd {df.k0.std():.4f}  range {df.k0.min():.4f}-{df.k0.max():.4f}")
print(f"  ORACLE (offset removed): mean {df.oracle.mean():.4f}  sd {df.oracle.std():.4f}  range {df.oracle.min():.4f}-{df.oracle.max():.4f}")
print(f"\n  spread COLLAPSES: sd {df.k0.std():.4f} -> {df.oracle.std():.4f}  ({100*(1-df.oracle.std()/df.k0.std()):.0f}% reduction)")
print(f"  corr(zero-shot, oracle) = {np.corrcoef(df.k0,df.oracle)[0,1]:+.3f}")
print("\n=> if the oracle spread is much smaller than the zero-shot spread, then almost all the")
print("   difference between our arms IS the offset, and removing it makes them interchangeable.")
