#!/usr/bin/env python3
"""W13b -- does the MEASURED offset compose with the slope lever?

They target different terms of pred ~= a_p*true + b_p:
  --slope_weight  fixes a_p (the compression), measured 0.496 -> 0.782 at the canonical epoch
  W13             fixes b_p (the offset), measured +0.112 pooled at k=20
If they are independent, the combination should give roughly the sum. If they overlap, less.
This is a pure post-processing question -- ZERO GPU.

Also tests the ORDER question: does removing the offset AFTER a slope-corrected model work better
or worse than on the control? A slope arm has larger std(b_p) (it trades offset for slope), so it
is not obvious.
"""
import glob, os, numpy as np, pandas as pd

def load(f):
    d = pd.read_csv(f)
    if not {'protein','ddG','pred_ddG'}.issubset(d.columns): return None
    d = d[['protein','ddG','pred_ddG']].dropna()
    d = d[d.protein.astype(str).str.upper() != '2K5H']
    return d if len(d) > 100 and d.protein.nunique() >= 20 else None

def pooled(y,p): return float(np.corrcoef(p,y)[0,1]) if len(y)>2 and np.std(p)>1e-12 else np.nan

def w13(d,k,R,rng):
    out=[]
    for _ in range(R):
        ys,ps=[],[]
        for _,g in d.groupby('protein'):
            n=len(g)
            if n<k+5: continue
            idx=rng.permutation(n); cal,ev=idx[:k],idx[k:]
            gy,gp=g.ddG.values,g.pred_ddG.values
            off=(gp[cal]-gy[cal]).mean() if k>0 else 0.0
            ys.append(gy[ev]); ps.append(gp[ev]-off)
        if ys: out.append(pooled(np.concatenate(ys),np.concatenate(ps)))
    a=np.array([x for x in out if np.isfinite(x)])
    return (a.mean(), a.std(ddof=1) if len(a)>1 else 0.0) if len(a) else (np.nan,np.nan)

def stats(d):
    aps,rs,ss,bps=[],[],[],[]
    for _,g in d.groupby('protein'):
        if len(g)>4 and g.ddG.std()>1e-9 and g.pred_ddG.std()>1e-9:
            aps.append(np.polyfit(g.ddG,g.pred_ddG,1)[0])
            rs.append(np.corrcoef(g.pred_ddG,g.ddG)[0,1])
            ss.append(g.pred_ddG.std()/g.ddG.std())
        bps.append((g.pred_ddG.values-g.ddG.values).mean())
    return np.mean(aps),np.mean(rs),np.mean(ss),np.std(bps,ddof=1)

ARMS={
 'control        ':'abl_calib_ctrl_repro2_e14.csv',
 'slope (canon e10)':'abl_gld_slope1.0_s42_e10.csv',
 'slope (e13)    ':'abl_p3_slope1.0_s42_e13.csv',
 'best pooled    ':'abl_sigma_seed2_e10.csv',
}
rng=np.random.RandomState(0)
print("W13b -- does the measured offset COMPOSE with the slope lever?")
print("27 proteins, 2K5H dropped, ddG. 20 draws per k.\n")
print(f"{'arm':<20}{'a_p':>7}{'std(b_p)':>10}{'k=0':>9}{'k=3':>9}{'k=10':>9}{'k=20':>9}{'gain':>9}")
print("-"*82)
for name,fn in ARMS.items():
    f='/home/nissimb/DeepPEF/eval_results/'+fn
    if not os.path.exists(f): print(f"{name:<20} MISSING"); continue
    d=load(f)
    if d is None: print(f"{name:<20} unusable"); continue
    ap,r,s,sb=stats(d)
    k0,_=w13(d,0,1,rng); k3,_=w13(d,3,20,rng); k10,_=w13(d,10,20,rng); k20,_=w13(d,20,20,rng)
    print(f"{name:<20}{ap:>7.3f}{sb:>10.3f}{k0:>9.4f}{k3:>9.4f}{k10:>9.4f}{k20:>9.4f}{k20-k0:>+9.4f}")
print("\nIf slope and W13 were independent, the slope arm's k=20 should exceed the control's k=20")
print("by roughly the slope lever's own zero-shot gain. If they land together, they OVERLAP.")
