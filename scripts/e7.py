import pandas as pd, numpy as np
from scipy import stats

NINE = ["calib_ctrl_repro2_e14","sigma_seed1_e13","sigma_seed2_e10","sigma_seed3_e13",
        "sigma_seed4_e14","sigma_seed42_e9","anchor_w0.3_s42_e14","anchor_w1.0_s42_e14",
        "p3_slope1.0_s42_e13"]
W12 = ["w12_s1_e14","w12_s2_e12"]

def load(t):
    d = pd.read_csv("eval_results/abl_%s.csv" % t)
    return d[d.protein != "2K5H"]

def pooled(d): return np.corrcoef(d.ddG, d.pred_ddG)[0,1]

def oracle_offset(d):
    T,P = [],[]
    for p,g in d.groupby("protein"):
        x,y = g.pred_ddG.values, g.ddG.values
        T.append(y); P.append(x - (x.mean()-y.mean()))
    return np.corrcoef(np.concatenate(T), np.concatenate(P))[0,1]

def perprot(d):
    r = [np.corrcoef(g.ddG,g.pred_ddG)[0,1] for _,g in d.groupby("protein")
         if g.ddG.std()>1e-9 and g.pred_ddG.std()>1e-9]
    return np.mean(r), np.median(r)

print("E7 - W12 on the CANONICAL basis (27 proteins, ddG, val-selected epoch)")
print("")
# reproduce the canonical population first (verification the basis is intact)
ok = [t for t in NINE if __import__("os").path.exists("eval_results/abl_%s.csv" % t)]
pv = [pooled(load(t)) for t in ok]
ov = [oracle_offset(load(t)) for t in ok]
print("canonical population: n=%d" % len(ok))
print("  pooled %.4f +/- %.4f    (recorded: 0.5772 +/- 0.0316)" % (np.mean(pv), np.std(pv, ddof=1)))
print("  oracle %.4f +/- %.4f    (recorded: 0.7156 +/- 0.0474)" % (np.mean(ov), np.std(ov, ddof=1)))
print("")
# the CONTROL family only (the fair comparator for a lever)
CTRL = ["calib_ctrl_repro2_e14","sigma_seed1_e13","sigma_seed2_e10","sigma_seed3_e13",
        "sigma_seed4_e14","sigma_seed42_e9"]
cp = [pooled(load(t)) for t in CTRL]
co = [oracle_offset(load(t)) for t in CTRL]
cm = [perprot(load(t)) for t in CTRL]
print("control family (6 same-config seeds):")
print("  pooled  %.4f +/- %.4f" % (np.mean(cp), np.std(cp, ddof=1)))
print("  oracle  %.4f +/- %.4f" % (np.mean(co), np.std(co, ddof=1)))
print("  mean r  %.4f +/- %.4f" % (np.mean([a for a,_ in cm]), np.std([a for a,_ in cm], ddof=1)))
print("  med  r  %.4f +/- %.4f" % (np.mean([b for _,b in cm]), np.std([b for _,b in cm], ddof=1)))
print("")
wp = [pooled(load(t)) for t in W12]
wo = [oracle_offset(load(t)) for t in W12]
wm = [perprot(load(t)) for t in W12]
print("W12 (2 seeds):")
for t,a,b,(m,md) in zip(W12,wp,wo,wm):
    print("  %-14s pooled %.4f  oracle %.4f  mean r %.4f  med r %.4f" % (t,a,b,m,md))
print("  MEAN           pooled %.4f  oracle %.4f  mean r %.4f  med r %.4f"
      % (np.mean(wp), np.mean(wo), np.mean([a for a,_ in wm]), np.mean([b for _,b in wm])))
print("")
sdp = np.std(cp, ddof=1); sdm = np.std([a for a,_ in cm], ddof=1)
print("GAIN over the control family, in units of that channel's own seed sd:")
print("  pooled   %+.4f  = %+.2f sd   (sd %.4f)" % (np.mean(wp)-np.mean(cp), (np.mean(wp)-np.mean(cp))/sdp, sdp))
print("  mean r   %+.4f  = %+.2f sd   (sd %.4f)" % (np.mean([a for a,_ in wm])-np.mean([a for a,_ in cm]),
      (np.mean([a for a,_ in wm])-np.mean([a for a,_ in cm]))/sdm, sdm))
print("")
print("vs the canonical headline 0.5772: W12 mean pooled %.4f -> %+.4f" % (np.mean(wp), np.mean(wp)-0.5772))
print("vs the best single canonical run 0.6382: %+.4f" % (np.mean(wp)-0.6382))
