"""Ofir p.34: 'with a small number of proteins with the new AA that are experimentally evaluated
... the RMSE can be used to calibrate the predictions of the protein.'

Transferred: for a HELD-OUT protein, measure b_p from only k experimentally-known mutations of
that protein and correct the rest. This is few-shot calibration, NOT the failed LOPO regression
of b_p on protein-level features. Ofir's point is that you do not PREDICT the offset, you MEASURE
it from a handful of samples.

Reports: pooled ddG PCC and dG enrichment vs k, plus the 'RMSE ~= |mean gap|' identity check.
"""
import os, glob, math
import numpy as np, pandas as pd

def pcc(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if len(a)<3 or np.std(a)==0 or np.std(b)==0: return float('nan')
    return float(np.corrcoef(a,b)[0,1])
def enrich(y,yh,pct,low=False):
    y=np.asarray(y,float);yh=np.asarray(yh,float);N=len(y);k=int(math.floor(pct*N))
    if k<1: return float('nan')
    s=1.0 if low else -1.0
    return len(set(np.argsort(s*y,kind='stable')[:k].tolist())&set(np.argsort(s*yh,kind='stable')[:k].tolist()))/(N*(k/N)**2)

files=sorted(glob.glob("eval_results/*.csv"))
KS=[1,2,3,5,10,20,50,100]
NREP=60
rows=[]
# ---- Part A: Ofir's identity, RMSE(dG) vs |mean gap|, per protein ----
d=pd.read_csv(files[0]); d=d[~((d.ddG==0)&(d.pred_ddG==0))]
ident=[]
for p,g in d.groupby("protein"):
    if len(g)<40: continue
    err=g.pred_deltaG-g.deltaG
    rmse=float(np.sqrt((err**2).mean())); gap=abs(float(err.mean()))
    resid=float(np.sqrt(max((err**2).mean()-err.mean()**2,0)))
    ident.append(dict(protein=p,n=len(g),rmse_dG=rmse,mean_gap=gap,resid_rmse=resid))
I=pd.DataFrame(ident)
n=len(I); thr=2/math.sqrt(n-2)
print("=== OFIR IDENTITY on OUR data (dG, ckpt %s), n=%d proteins, |r|>%.3f sig ==="%(os.path.basename(files[0]),n,thr))
print("corr(RMSE_dG, |mean gap|)   = %+.4f    [Ofir LOAO: +0.9883, n=18]"%pcc(I.rmse_dG,I.mean_gap))
print("RMSE_dG range %.3f-%.3f (%.2fx);  after removing offset: %.3f-%.3f (%.2fx)"%(
  I.rmse_dG.min(),I.rmse_dG.max(),I.rmse_dG.max()/I.rmse_dG.min(),
  I.resid_rmse.min(),I.resid_rmse.max(),I.resid_rmse.max()/I.resid_rmse.min()))
print("mean RMSE_dG=%.3f  mean |gap|=%.3f  -> offset is %.1f%% of MSE"%(
  I.rmse_dG.mean(),I.mean_gap.mean(),100*(I.mean_gap**2).sum()/(I.rmse_dG**2).sum()))
print("corr(resid_RMSE, |mean gap|)= %+.4f  (should collapse toward 0 if offset is the whole story)"%pcc(I.resid_rmse,I.mean_gap))
I.to_csv("results/ofir_identity.csv",index=False)

# ---- Part B: few-shot measured-offset calibration ----
print("\n=== FEW-SHOT OFFSET CALIBRATION (measure b_p from k known mutants of the SAME protein) ===")
print("k = #experimentally-known mutations per held-out protein; %d random draws each"%NREP)
out=[]
for f in files:
    d=pd.read_csv(f); tag=os.path.basename(f)[:-4]
    d=d[~((d.ddG==0)&(d.pred_ddG==0))]
    gs={p:g for p,g in d.groupby("protein") if len(g)>=200}
    rng=np.random.RandomState(0)
    base_p=[]; base_e=[]
    for k in KS:
        ps=[];es=[]
        for rep in range(NREP):
            allT=[];allP=[]
            for p,g in gs.items():
                idx=rng.choice(len(g),size=k,replace=False)
                sub=g.iloc[idx]
                b=float((sub.pred_deltaG-sub.deltaG).mean())   # MEASURED offset
                rest=g.drop(g.index[idx])
                allT.append(rest.deltaG.values); allP.append(rest.pred_deltaG.values-b)
            T=np.concatenate(allT);P=np.concatenate(allP)
            ps.append(pcc(T,P)); es.append(enrich(T,P,0.05,True))
        out.append(dict(file=tag,k=k,dG_pcc=np.mean(ps),dG_pcc_sd=np.std(ps),E5bot=np.mean(es)))
    # raw (k=0) reference on the same protein set
    T=np.concatenate([g.deltaG.values for g in gs.values()])
    P=np.concatenate([g.pred_deltaG.values for g in gs.values()])
    out.append(dict(file=tag,k=0,dG_pcc=pcc(T,P),dG_pcc_sd=0.0,E5bot=enrich(T,P,0.05,True)))
    # oracle
    Po=np.concatenate([g.pred_deltaG.values-float((g.pred_deltaG-g.deltaG).mean()) for g in gs.values()])
    out.append(dict(file=tag,k=-1,dG_pcc=pcc(T,Po),dG_pcc_sd=0.0,E5bot=enrich(T,Po,0.05,True)))
O=pd.DataFrame(out); O.to_csv("results/fewshot_calib.csv",index=False)
S=O.groupby("k").agg(dG_pcc=("dG_pcc","mean"),sd=("dG_pcc_sd","mean"),E5bot=("E5bot","mean")).sort_index()
S.index=["ORACLE(all)" if i==-1 else ("RAW(k=0)" if i==0 else "k=%d"%i) for i in S.index]
print(S.round(4).to_string())
print("\nproteins used (n>=200 muts): %d"%len(gs))
print("WROTE results/ofir_identity.csv results/fewshot_calib.csv")
