"""Is the IFUM equilibrium-mixture label actually informative on OUR data?

Eq 2:  r_U^2 = 6 * 1.927 * |i-j|^0.598
Eq 4:  [F]:[U] = exp(dG/RT) : 1
Eq 5:  p_F = exp(dG/RT)/(1+exp(dG/RT)),  p_U = 1 - p_F

If p_F is ~1.0 for every protein in our set, the mixed label is just the folded distogram and
the whole mechanism carries no stability information HERE, whatever it does on their data.
"""
import pandas as pd, numpy as np, glob, os, torch

RT = 0.001987 * 298.15   # kcal/mol
print("RT = %.4f kcal/mol at 298 K" % RT)

d = pd.read_csv("eval_results/abl_calib_ctrl_repro2_e14.csv")
d = d[d.protein != "2K5H"]
dg = d.deltaG.values
print("\nour dG labels (27 test proteins, %d variants):" % len(dg))
print("  min %.3f  p5 %.3f  median %.3f  p95 %.3f  max %.3f"
      % (dg.min(), np.percentile(dg,5), np.median(dg), np.percentile(dg,95), dg.max()))

pf = 1.0/(1.0+np.exp(-dg/RT))
print("\np_F = exp(dG/RT)/(1+exp(dG/RT))  -- the folded weight in Eq 5:")
for q in (0,5,25,50,75,95,100):
    print("   p%-3d  dG=%+7.3f -> p_F = %.6f" % (q, np.percentile(dg,q),
          1.0/(1.0+np.exp(-np.percentile(dg,q)/RT))))
print("\n  fraction of variants with p_F > 0.99 : %.4f" % (pf>0.99).mean())
print("  fraction with p_F in [0.01, 0.99]     : %.4f" % ((pf>0.01)&(pf<0.99)).mean())
print("  fraction with p_F < 0.01              : %.4f" % (pf<0.01).mean())

print("\n-- how wide is dG in units of RT? --")
print("  dG range %.3f kcal/mol = %.1f RT" % (dg.max()-dg.min(), (dg.max()-dg.min())/RT))
print("  a 1 kcal/mol change is %.1f RT, so p_F saturates unless |dG| < ~%.2f kcal/mol" % (1/RT, 5*RT))

print("\n-- the unfolded map, Eq 2 --")
sep = np.arange(1,61)
rU = np.sqrt(6*1.927*sep**0.598)
print("  |i-j|=1 -> %.2f A   |i-j|=10 -> %.2f A   |i-j|=30 -> %.2f A   |i-j|=59 -> %.2f A"
      % (rU[0], rU[9], rU[29], rU[58]))
print("  (our _coil_bond_length-based map used b*|i-j|^nu with b ~ 3.8; theirs is fixed)")
