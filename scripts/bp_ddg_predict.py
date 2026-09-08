#!/usr/bin/env python3
"""Attempt #10 at predicting b_p -- but in ddG SPACE, which was never tried.

WHY THIS IS NOT A REPEAT. All nine prior attempts targeted the dG-space WT error, which is
96.3% one global constant (-5.07) and therefore nearly unlearnable by construction. The
ddG-space intercept is only 38-55% global (measured today on three checkpoints), so 45-62%
of it is genuinely per-protein. That part has never been tested for predictability.

Leave-one-protein-out ridge. n=27, so |r| < 0.381 is indistinguishable from zero at p=0.05.
Degenerate control included: predicting the global mean must score R^2 = 0 by definition.
"""
import glob, os, warnings, numpy as np, pandas as pd, torch
warnings.filterwarnings('ignore')
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut

ROOT='/home/nissimb/DeepPEF'
def load(f):
    d=pd.read_csv(f)
    if not {'protein','ddG','pred_ddG'}.issubset(d.columns): return None
    d=d[['protein','ddG','pred_ddG']].dropna()
    return d[d.protein.astype(str).str.upper()!='2K5H']

def feats(prot):
    """Protein-level features available WITHOUT any label."""
    out={}
    for p in prot:
        g=glob.glob(f'{ROOT}/data/Processed_K50_dG_datasets/training_data/{p}*/coords_tensor.pt')
        if not g: out[p]=None; continue
        c=torch.load(g[0],map_location='cpu',weights_only=False).float().numpy()
        ca=c[:,1,:]; cb=c[:,3,:]
        N=len(ca)
        d=np.linalg.norm(ca[:,None,:]-ca[None,:,:],axis=-1)
        contacts=((d<10)&(d>0)).sum(1)
        # radius of gyration, contact order, burial stats -- all label-free
        rg=np.sqrt(((ca-ca.mean(0))**2).sum(1).mean())
        sep=np.abs(np.arange(N)[:,None]-np.arange(N)[None,:])
        co=(sep*((d<8)&(d>0))).sum()/max(1,((d<8)&(d>0)).sum())/N
        out[p]=[N, rg, rg/N**0.6, contacts.mean(), contacts.std(), co,
                float(np.median(np.linalg.norm(cb-ca,axis=1)))]
    return out

for tag in ['abl_p3_slope1.0_s42_e13','abl_calib_ctrl_repro2_e14','abl_sigma_seed2_e10']:
    f=f'{ROOT}/eval_results/{tag}.csv'
    if not os.path.exists(f): continue
    d=load(f)
    bp={p:(g.pred_ddG.values-g.ddG.values).mean() for p,g in d.groupby('protein')}
    prot=sorted(bp)
    F=feats(prot)
    ok=[p for p in prot if F[p] is not None]
    if len(ok)<20: print(f"{tag}: only {len(ok)} proteins with coords"); continue
    X=np.array([F[p] for p in ok]); y=np.array([bp[p] for p in ok])
    Xs=(X-X.mean(0))/(X.std(0)+1e-9)
    pr=np.zeros(len(y))
    for tr,te in LeaveOneOut().split(Xs):
        pr[te]=RidgeCV(alphas=np.logspace(-2,4,25)).fit(Xs[tr],y[tr]).predict(Xs[te])
    r2=1-((y-pr)**2).sum()/((y-y.mean())**2).sum()
    r=np.corrcoef(pr,y)[0,1] if pr.std()>1e-12 else 0.0
    print(f"{tag}")
    print(f"   n={len(y)}  std(b_p)={y.std(ddof=1):.4f}  global={y.mean():+.4f}")
    print(f"   LOPO R2 = {r2:+.4f}   corr = {r:+.3f}   (|r|<0.381 = zero at n=27)")
    print(f"   VERDICT: {'PREDICTABLE' if r2>0.05 else 'not predictable'}")
