import pandas as pd, numpy as np, os
from scipy import stats
def ev(t):
    d=pd.read_csv("eval_results/abl_%s.csv"%t); return d[d.protein!="2K5H"]
def stat(t):
    d=ev(t)
    r=[np.corrcoef(g.ddG,g.pred_ddG)[0,1] for _,g in d.groupby("protein")
       if g.ddG.std()>1e-9 and g.pred_ddG.std()>1e-9]
    return np.corrcoef(d.ddG,d.pred_ddG)[0,1], np.mean(r), np.median(r)
def pp(t):
    d=ev(t)
    return {p:np.corrcoef(g.ddG,g.pred_ddG)[0,1] for p,g in d.groupby("protein")
            if g.ddG.std()>1e-9 and g.pred_ddG.std()>1e-9}
CTRL=["calib_ctrl_repro2_e14","sigma_seed1_e13","sigma_seed2_e10","sigma_seed3_e13",
      "sigma_seed4_e14","sigma_seed42_e9"]
C={}
for t in CTRL:
    for p,v in pp(t).items(): C.setdefault(p,[]).append(v)
cs=[stat(t) for t in CTRL]
cp,cm,cd=np.mean([a for a,_,_ in cs]),np.mean([b for _,b,_ in cs]),np.mean([c for _,_,c in cs])
sp,sm,sd_=np.std([a for a,_,_ in cs],ddof=1),np.std([b for _,b,_ in cs],ddof=1),np.std([c for _,_,c in cs],ddof=1)
print("control (6 seeds): pooled %.4f+/-%.4f  mean r %.4f+/-%.4f  med r %.4f+/-%.4f"%(cp,sp,cm,sm,cd,sd_))
print("")
print("%-24s %8s %8s %8s %7s %8s %9s"%("arm","pooled","mean r","med r","sd(mr)","impr","Wilcox"))
for t,nm in [("w15_s42_e10","W15 (NEW)"),
             ("disto0.01_s42_e14","disto 0.01"),("disto0.1_s42_e14","disto 0.1"),
             ("disto1.0_s42_e14","disto 1.0"),
             ("ctrl_s1_e14","ctrl_s1"),("p_ctrl_s2_e14","ctrl_s2"),
             ("p_descunit_s42_e14","descunit"),
             ("pooled0.3_s42_e14","pooled_corr .3"),("pooled1.0_s42_e14","pooled_corr 1.0"),
             ("pooled_slope_s42_e14","pooled+slope")]:
    if not os.path.exists("eval_results/abl_%s.csv"%t): print("%-24s  pending"%nm); continue
    P,M,D=stat(t); W=pp(t)
    pr=sorted(set(C)&set(W)); dd=np.array([W[x]-np.mean(C[x]) for x in pr])
    w=stats.wilcoxon(dd).pvalue
    print("%-24s %8.4f %8.4f %8.4f %+7.2f %5d/%d %9.4f"%(nm,P,M,D,(M-cm)/sm,sum(dd>0),len(dd),w))
print("")
print("=== B1d CLAIM TEST: are the 3 disto arms identical? (they should be, flag was inert) ===")
v=[stat("disto%s_s42_e14"%x) for x in ("0.01","0.1","1.0")]
print("  pooled: %s   spread %.4f"%(["%.4f"%a for a,_,_ in v], max(a for a,_,_ in v)-min(a for a,_,_ in v)))
print("  mean r: %s   spread %.4f"%(["%.4f"%b for _,b,_ in v], max(b for _,b,_ in v)-min(b for _,b,_ in v)))
print("  control-family pooled sd = %.4f  -> spread is %.2f sd"%(sp,(max(a for a,_,_ in v)-min(a for a,_,_ in v))/sp))
