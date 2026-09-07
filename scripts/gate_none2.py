"""Re-gate: shipped CSV = affine(a,b) applied to raw pred_deltaG. My runs are RAW.
So the gate is: a*raw_pred_dG + b  ==  shipped pred_deltaG."""
import json, numpy as np, pandas as pd
A=json.load(open("Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_calib_ctrl_repro2/affine.json"))
a,b=A["a"],A["b"]; print("affine a=%.10f b=%.10f"%(a,b))
ref=pd.read_csv("eval_results/abl_calib_ctrl_repro2_e14.csv")
non=pd.read_csv("results/seqabl/abl_none_s42.csv")
def key(d):
    d=d.copy(); d["r"]=d.groupby("protein").cumcount(); return d.set_index(["protein","r"]).sort_index()
x=key(ref); y=key(non)
assert x.index.equals(y.index)
pred_aff = a*y.pred_deltaG.values + b
d=np.abs(x.pred_deltaG.values-pred_aff)
print("dG  : max=%.3e mean=%.3e"%(d.max(),d.mean()))
# ddG = pred_dG(mut) - pred_dG(wt) per protein -> a cancels? no: ddG scales by a, b cancels
wt=y.groupby(level=0).pred_deltaG.transform("first")
ddg_aff = a*(y.pred_deltaG.values - wt.values)
d2=np.abs(x.pred_ddG.values-ddg_aff)
print("ddG : max=%.3e mean=%.3e   (b cancels, a scales)"%(d2.max(),d2.mean()))
ok = d.max()<1e-3 and d2.max()<1e-3
print("GATE:", "PASS -- mode 'none' reproduces the shipped CSV exactly once the affine sidecar is applied" if ok
      else "still off; residual investigated below")
if not ok:
    fit=np.polyfit(y.pred_deltaG.values, x.pred_deltaG.values,1)
    print("  empirical affine recovered from data: a=%.6f b=%.6f (json a=%.6f b=%.6f)"%(fit[0],fit[1],a,b))
    r=np.corrcoef(y.pred_deltaG.values,x.pred_deltaG.values)[0,1]
    print("  corr(raw, shipped) = %.8f  -> %s"%(r,"pure affine" if r>0.99999 else "NOT a pure affine map"))
