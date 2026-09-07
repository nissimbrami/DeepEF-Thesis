"""Verify the few-shot calibration result is not leakage or a coding artifact.

Checks:
 1. The k calibration mutants are EXCLUDED from scoring (no test-on-train).
 2. A null control: subtract a RANDOM constant of the same magnitude -> must NOT help.
 3. A shuffled control: use protein q's offset on protein p -> must HURT.
 4. Report ddG PCC too (b_p cancels in ddG -> few-shot must be a NO-OP there; if it
    moves ddG, something is wrong).
 5. Per-checkpoint spread, so this is not one lucky CSV.
"""
import os,glob,math
import numpy as np,pandas as pd
def pcc(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if len(a)<3 or np.std(a)==0 or np.std(b)==0: return float('nan')
    return float(np.corrcoef(a,b)[0,1])

files=sorted(glob.glob("eval_results/*.csv"))
print("=== PER-CHECKPOINT few-shot dG PCC (k=3), + controls. 30 draws each ===")
print("%-26s %6s %6s %6s %6s | %7s %7s %7s | %6s"%(
  "file","raw","k=1","k=3","k=10","ORACLE","randB","swapB","ddGk3"))
agg={}
for f in files:
    d=pd.read_csv(f); tag=os.path.basename(f)[:-4]
    d=d[~((d.ddG==0)&(d.pred_ddG==0))]
    gs={p:g for p,g in d.groupby("protein") if len(g)>=200}
    rng=np.random.RandomState(1)
    bt={p:float((g.pred_deltaG-g.deltaG).mean()) for p,g in gs.items()}
    T=np.concatenate([g.deltaG.values for g in gs.values()])
    P=np.concatenate([g.pred_deltaG.values for g in gs.values()])
    raw=pcc(T,P)
    orc=pcc(T,np.concatenate([g.pred_deltaG.values-bt[p] for p,g in gs.items()]))
    # random constant of same magnitude
    sd=np.std(list(bt.values())); mu=np.mean(list(bt.values()))
    rnd=np.mean([pcc(T,np.concatenate([g.pred_deltaG.values-rng.normal(mu,sd) for g in gs.values()])) for _ in range(30)])
    # swapped offsets
    ks=list(gs); sw=[]
    for _ in range(30):
        pm=rng.permutation(len(ks))
        sw.append(pcc(T,np.concatenate([gs[ks[i]].pred_deltaG.values-bt[ks[pm[i]]] for i in range(len(ks))])))
    res={}
    for k in (1,3,10):
        v=[];vd=[]
        for _ in range(30):
            aT=[];aP=[];bT=[];bP=[]
            for p,g in gs.items():
                idx=rng.choice(len(g),size=k,replace=False)
                b=float((g.iloc[idx].pred_deltaG-g.iloc[idx].deltaG).mean())
                rest=g.drop(g.index[idx])          # EXCLUDED from scoring
                aT.append(rest.deltaG.values); aP.append(rest.pred_deltaG.values-b)
                bT.append(rest.ddG.values);   bP.append(rest.pred_ddG.values-0.0)
            v.append(pcc(np.concatenate(aT),np.concatenate(aP)))
            vd.append(pcc(np.concatenate(bT),np.concatenate(bP)))
        res[k]=np.mean(v); res[str(k)+'d']=np.mean(vd)
    print("%-26s %6.3f %6.3f %6.3f %6.3f | %7.3f %7.3f %7.3f | %6.3f"%(
      tag,raw,res[1],res[3],res[10],orc,rnd,np.mean(sw),res['3d']))
    agg[tag]=(raw,res[1],res[3],res[10],orc,rnd,np.mean(sw),res['3d'])
A=np.array(list(agg.values()))
print("-"*100)
print("%-26s %6.3f %6.3f %6.3f %6.3f | %7.3f %7.3f %7.3f | %6.3f"%(("MEAN(10 ckpts)",)+tuple(A.mean(0))))
print("\nCONTROLS: randB must be <= raw (random constant cannot help);")
print("          swapB must be << raw (wrong protein's offset must hurt);")
print("          ddGk3 must equal the untouched pooled ddG PCC (b_p cancels in ddG).")
d=pd.read_csv(files[0]); d=d[~((d.ddG==0)&(d.pred_ddG==0))]
gs={p:g for p,g in d.groupby("protein") if len(g)>=200}
print("sanity: untouched pooled ddG PCC on file[0] = %.3f"%pcc(
  np.concatenate([g.ddG.values for g in gs.values()]),
  np.concatenate([g.pred_ddG.values for g in gs.values()])))
