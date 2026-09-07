"""Per-protein calibration parameters from an eval CSV.

b_p = the WT error (an error in ABSOLUTE dG); a_p = the ddG compression slope.
The two are deliberately computed on DIFFERENT metrics, because they live on
different metrics: b_p is a dG-side quantity, a_p is a ddG-side quantity.
Conflating them is the mistake this project keeps paying for.
"""
import json, csv, math
from collections import defaultdict

SRC = 'eval_results/abl_calib_ctrl_repro2_e14.csv'
rows = defaultdict(list)
with open(SRC) as f:
    for r in csv.DictReader(f):
        rows[r['protein']].append(
            (float(r['deltaG']), float(r['pred_deltaG']),
             float(r['ddG']), float(r['pred_ddG'])))

def pearson(x, y):
    n = len(x)
    if n < 3: return float('nan')
    mx, my = sum(x)/n, sum(y)/n
    sxy = sum((a-mx)*(b-my) for a, b in zip(x, y))
    sxx = sum((a-mx)**2 for a in x); syy = sum((b-my)**2 for b in y)
    if sxx <= 0 or syy <= 0: return float('nan')
    return sxy/math.sqrt(sxx*syy)

def fit(x, y):
    """least squares y = a*x + b"""
    n = len(x)
    if n < 3: return float('nan'), float('nan')
    mx, my = sum(x)/n, sum(y)/n
    sxx = sum((a-mx)**2 for a in x)
    if sxx <= 0: return float('nan'), float('nan')
    a = sum((p-mx)*(q-my) for p, q in zip(x, y))/sxx
    return a, my - a*mx

out = {}
for p, v in rows.items():
    dg   = [t[0] for t in v]; pdg  = [t[1] for t in v]
    ddg  = [t[2] for t in v]; pddg = [t[3] for t in v]
    # b_p: the WT error. WT is the row with ddG == 0 (WS-1 convention, row 0).
    wt = [i for i, t in enumerate(v) if t[2] == 0.0]
    b_wt = (pdg[wt[0]] - dg[wt[0]]) if wt else float('nan')
    a_dg, b_dg = fit(dg, pdg)        # dG-side affine fit
    a_ddg, b_ddg = fit(ddg, pddg)    # ddG-side: a_ddg IS the compression a_p
    out[p] = dict(n=len(v), b_p_wt_error=b_wt,
                  a_dg=a_dg, b_dg=b_dg, a_p=a_ddg, b_ddg=b_ddg,
                  pcc_ddg=pearson(ddg, pddg), pcc_dg=pearson(dg, pdg),
                  mean_dg=sum(dg)/len(dg))

ok = [p for p in out if not math.isnan(out[p]['b_p_wt_error'])]
print('proteins: %d  (with a WT row: %d)' % (len(out), len(ok)))
bs = sorted(out[p]['b_p_wt_error'] for p in ok)
as_ = sorted(out[p]['a_p'] for p in out if not math.isnan(out[p]['a_p']))
def med(z): return z[len(z)//2] if z else float('nan')
mb = sum(bs)/len(bs)
print('b_p  (WT error): mean %.4f  median %.4f  std %.4f  min %.4f  max %.4f'
      % (mb, med(bs), math.sqrt(sum((x-mb)**2 for x in bs)/len(bs)), bs[0], bs[-1]))
print('a_p  (ddG slope): median %.4f  min %.4f  max %.4f' % (med(as_), as_[0], as_[-1]))
pcs = [out[p]['pcc_ddg'] for p in out if not math.isnan(out[p]['pcc_ddg'])]
print('per-protein ddG PCC: median %.4f' % med(sorted(pcs)))
print('corr(a_p, per-protein PCC) = %.4f'
      % pearson([out[p]['a_p'] for p in out if not math.isnan(out[p]['a_p']) and not math.isnan(out[p]['pcc_ddg'])],
                [out[p]['pcc_ddg'] for p in out if not math.isnan(out[p]['a_p']) and not math.isnan(out[p]['pcc_ddg'])]))
print('corr(|b_p|, per-protein PCC) = %.4f'
      % pearson([abs(out[p]['b_p_wt_error']) for p in ok if not math.isnan(out[p]['pcc_ddg'])],
                [out[p]['pcc_ddg'] for p in ok if not math.isnan(out[p]['pcc_ddg'])]))
json.dump(out, open('results/calib_per_protein.json','w'), indent=1)
print('wrote results/calib_per_protein.json')
