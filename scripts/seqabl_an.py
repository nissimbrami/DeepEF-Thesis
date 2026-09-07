"""Analyse Ofir-style sequence ablations on the frozen DeepEF checkpoint.
GATE FIRST: mode 'none' must reproduce the shipped abl_calib_ctrl_repro2_e14.csv.
"""
import os,glob,math
import numpy as np,pandas as pd
def pcc(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if len(a)<3 or np.std(a)==0 or np.std(b)==0: return float('nan')
    return float(np.corrcoef(a,b)[0,1])
def enrich(y,yh,pct,low=False):
    y=np.asarray(y,float);yh=np.asarray(yh,float);N=len(y);k=int(math.floor(pct*N))
    if k<1: return float('nan')
    s=1.0 if low else -1.0
    return len(set(np.argsort(s*y,kind='stable')[:k].tolist())&set(np.argsort(s*yh,kind='stable')[:k].tolist()))/(N*(k/N)**2)

REF="eval_results/abl_calib_ctrl_repro2_e14.csv"
ref=pd.read_csv(REF)
none=pd.read_csv("results/seqabl/abl_none_s42.csv")
print("=== GATE: mode 'none' must reproduce the shipped CSV ===")
print("shipped rows %d, none rows %d"%(len(ref),len(none)))
cols=[c for c in ref.columns if c in none.columns]
print("shared cols:",cols)
m=min(len(ref),len(none))
for c in ["deltaG","pred_deltaG","ddG","pred_ddG"]:
    if c in cols:
        a=ref[c].values[:m]; b=none[c].values[:m]
        print("  %-12s maxabsdiff=%.3e  corr=%.8f"%(c,np.nanmax(np.abs(a-b)),pcc(a,b)))
GATE = np.nanmax(np.abs(ref.pred_ddG.values[:m]-none.pred_ddG.values[:m]))
print("GATE %s (pred_ddG max abs diff %.3e)"%("PASS" if GATE<1e-4 else "FAIL -- ordering differs, compare by (protein,ddG) key instead",GATE))

print("\n=== ABLATION RESULTS (all vs mode 'none' baseline) ===")
rows=[]
for f in sorted(glob.glob("results/seqabl/abl_*.csv")):
    tag=os.path.basename(f)[4:-4]
    d=pd.read_csv(f); d=d[~((d.ddG==0)&(d.pred_ddG==0))]
    pp=[];e5t=[];e5b=[];aps=[]
    for p,g in d.groupby("protein"):
        if len(g)<40: continue
        pp.append(pcc(g.ddG,g.pred_ddG))
        e5t.append(enrich(g.ddG,g.pred_ddG,0.05)); e5b.append(enrich(g.ddG,g.pred_ddG,0.05,True))
        aps.append(np.polyfit(g.deltaG,g.pred_deltaG,1)[0])
    rows.append(dict(mode=tag,N=len(d),
      ddG_PCC_pooled=pcc(d.ddG,d.pred_ddG), ddG_PCC_perprot=np.mean(pp),
      dG_PCC=pcc(d.deltaG,d.pred_deltaG),
      ddG_RMSE=float(np.sqrt(((d.ddG-d.pred_ddG)**2).mean())),
      E5top=np.mean(e5t), E5bot=np.mean(e5b), ap_med=np.median(aps),
      sd_pred=float(d.pred_ddG.std())))
R=pd.DataFrame(rows).set_index("mode")
order=["none_s42","perm_s42","perm_s7","perm_emb_s42","perm_oh_s42","collapse_s42","collapse_oh_s42"]
R=R.reindex([o for o in order if o in R.index])
pd.set_option("display.width",250)
print(R.round(4).to_string())
b=R.loc["none_s42"]
print("\n=== RETAINED FRACTION vs intact model (Ofir Table 3.2 analogue) ===")
for m_ in R.index:
    if m_=="none_s42": continue
    r=R.loc[m_]
    print("  %-16s ddG PCC %.3f (%.0f%% of intact)  E5top %.2f (%.0f%%)  E5bot %.2f (%.0f%%)  RMSE %.3f (%+.0f%%)"%(
      m_,r.ddG_PCC_pooled,100*r.ddG_PCC_pooled/b.ddG_PCC_pooled,
      r.E5top,100*r.E5top/b.E5top, r.E5bot,100*r.E5bot/b.E5bot,
      r.ddG_RMSE,100*(r.ddG_RMSE/b.ddG_RMSE-1)))
print("\nOfir Table 3.2 for reference: permuting TRAIN dropped his CNN correlation 0.953->0.870")
print("(91.3%% retained) and enrichment 5.575->4.631 (83.1%%) -- he called that 'a small factor'")
print("and concluded much of the learned information is composition-dependent.")
R.to_csv("results/seqabl_summary.csv")
print("\nWROTE results/seqabl_summary.csv")
