import pandas as pd, numpy as np, glob, os
from scipy import stats
M=pd.read_csv("results/08_data/structfeat/master_28.csv")
key=[c for c in M.columns if M[c].dtype==object][0]
print("master_28: %d rows, key '%s'" % (len(M), key))
bad=("D1_uemb","loroWdesc2","sa_","anchor_w3.0","dg_coil","coil_nu588","coiledge","loroW_desc","descunit")
B={};A={};R={};n=0
for f in sorted(glob.glob("eval_results/*.csv")):
    t=os.path.basename(f)
    if any(b in t for b in bad): continue
    d=pd.read_csv(f); d=d[d.protein!='2K5H']
    if d.pred_ddG.std()<1e-9: continue
    n+=1
    for p,g in d.groupby('protein'):
        if g.ddG.std()<1e-9 or g.pred_ddG.std()<1e-9: continue
        a,b=np.polyfit(g.ddG.values,g.pred_ddG.values,1)
        A.setdefault(p,[]).append(a); B.setdefault(p,[]).append(b)
        R.setdefault(p,[]).append(np.corrcoef(g.ddG,g.pred_ddG)[0,1])
print("healthy runs used: %d" % n)
rows=[]
for p in sorted(B):
    m=M[M[key].astype(str).str.contains(str(p),case=False,na=False)]
    if len(m)==0: continue
    r=m.iloc[0]
    rec=dict(protein=p,b_p=np.mean(B[p]),a_p=np.mean(A[p]),r=np.mean(R[p]))
    for c in M.columns:
        if c==key: continue
        try:
            val=float(r[c]); rec[c]=val
        except Exception: pass
    rows.append(rec)
D=pd.DataFrame(rows)
print("joined: %d proteins" % len(D))
tgts=['b_p','a_p','r']
feats=[c for c in D.columns if c not in tgts+['protein']]
print("")
print("%-28s %16s %16s %16s" % ("feature","vs b_p","vs a_p","vs r"))
out=[]
for c in feats:
    v=D[c].values.astype(float)
    if np.isnan(v).any() or v.std()<1e-12: continue
    line="%-28s" % c
    for t in tgts:
        rr,pp=stats.pearsonr(v,D[t].values)
        line+=" %+9.3f(%.3f)" % (rr,pp)
        out.append((c,t,rr,pp))
    print(line)
if out:
    thr=0.05/len(out)
    sig=[o for o in out if o[3]<0.05]
    print("")
    print("n=%d proteins; %d tests; nominal p<0.05: %d; Bonferroni thr %.5f" % (len(D),len(out),len(sig),thr))
    for c,t,rr,pp in sorted(sig,key=lambda x:x[3])[:8]:
        tag="SURVIVES Bonferroni" if pp<thr else ""
        print("   %s vs %s: r=%+.3f p=%.4f %s" % (c,t,rr,pp,tag))
