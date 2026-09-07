"""Enrichment translated to the design decision it actually supports, plus budget curves.

Ofir's argument (p.31-32): enrichment has ABSOLUTE meaning -- 'the relative prospective of
success using the predictions, compared with a random sample'. Here we express it as
'how many mutants must I make to find M stabilizing ones', which is the number an
experimentalist budgets against.
"""
import os,glob,math
import numpy as np,pandas as pd

def topk(y,yh,k,low=False):
    s=1.0 if low else -1.0
    return set(np.argsort(s*np.asarray(yh,float),kind='stable')[:k].tolist())

files=sorted(glob.glob("eval_results/*.csv"))
print("=== HIT-RATE CURVES (per protein, averaged over 28 proteins x 10 checkpoints) ===")
print("A 'hit' = a mutation truly in the top/bottom X%% of that protein's ddG.\n")
rec=[]
for f in files:
    d=pd.read_csv(f); tag=os.path.basename(f)[:-4]
    d=d[~((d.ddG==0)&(d.pred_ddG==0))]
    for p,g in d.groupby("protein"):
        if len(g)<40: continue
        N=len(g); y=g.ddG.values; yh=g.pred_ddG.values
        for pct in (0.05,0.10,0.15):
            k=int(math.floor(pct*N))
            for low,end in ((False,'top'),(True,'bot')):
                true=topk(y,y,k,low); pred=topk(y,yh,k,low)
                hit=len(true&pred)
                rec.append(dict(file=tag,protein=p,N=N,pct=pct,end=end,k=k,hits=hit,
                                rate=hit/k, chance=k/N, enr=(hit/k)/(k/N)))
R=pd.DataFrame(rec); R.to_csv("results/enrichment_hitrate.csv",index=False)
S=R.groupby(["pct","end"]).agg(k=("k","mean"),hit_rate=("rate","mean"),chance=("chance","mean"),
       enrichment=("enr","mean"),N=("N","mean")).reset_index()
S["screened_per_hit_model"]=1/S.hit_rate
S["screened_per_hit_random"]=1/S.chance
S["saving_x"]=S.screened_per_hit_random/S.screened_per_hit_model
print(S.round(3).to_string(index=False))
print("\nREADING: at pct=0.05 'bot' (the most DESTABILIZING 5%), picking the model's top-k")
print("gives a hit rate of %.1f%% vs %.1f%% by chance -> you screen %.1f candidates per hit"%(
  100*S[(S.pct==0.05)&(S.end=='bot')].hit_rate.iloc[0],
  100*S[(S.pct==0.05)&(S.end=='bot')].chance.iloc[0],
  S[(S.pct==0.05)&(S.end=='bot')].screened_per_hit_model.iloc[0]))

# Budget curve: to find M true top-5% stabilizers, how many must you make?
print("\n=== BUDGET: mutants to synthesize to obtain M true top-5%% STABILIZING hits ===")
print(" M   model_ranked   random   saving")
r5=R[(R.pct==0.05)&(R.end=='top')]
hr=r5.rate.mean(); ch=r5.chance.mean()
for M in (1,3,5,10):
    print(" %-3d %12.1f %8.1f   %.1fx"%(M,M/hr,M/ch,(M/ch)/(M/hr)))

# per-checkpoint spread -> is enrichment a stable model-selection signal?
print("\n=== IS ENRICHMENT STABLE ENOUGH TO SELECT A CHECKPOINT? (10 ckpts) ===")
C=R[(R.pct==0.05)].groupby(["file","end"]).enr.mean().unstack()
C["ddG_pcc"]=[float(np.corrcoef(pd.read_csv("eval_results/%s.csv"%f).query("not (ddG==0 and pred_ddG==0)").ddG,
              pd.read_csv("eval_results/%s.csv"%f).query("not (ddG==0 and pred_ddG==0)").pred_ddG)[0,1]) for f in C.index]
print(C.round(4).to_string())
print("\nspread across ckpts: E5top %.3f-%.3f (%.2fx), E5bot %.3f-%.3f (%.2fx), PCC %.4f-%.4f"%(
  C.top.min(),C.top.max(),C.top.max()/C.top.min(),C.bot.min(),C.bot.max(),C.bot.max()/C.bot.min(),
  C.ddG_pcc.min(),C.ddG_pcc.max()))
print("corr(pooled ddG PCC, mean per-protein E5top) = %+.4f  (n=10, |r|>0.632 sig)"%np.corrcoef(C.ddG_pcc,C.top)[0,1])
print("corr(pooled ddG PCC, mean per-protein E5bot) = %+.4f"%np.corrcoef(C.ddG_pcc,C.bot)[0,1])
print("best ckpt by PCC: %s | by E5top: %s | by E5bot: %s"%(
  C.ddG_pcc.idxmax(),C.top.idxmax(),C.bot.idxmax()))
print("\nWROTE results/enrichment_hitrate.csv")
