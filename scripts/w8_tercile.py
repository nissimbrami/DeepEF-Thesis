"""W8 pLDDT tercile analysis (CPU, read-only).

Stratifies the 28 test proteins by mean pLDDT into terciles and reports, per
tercile: per-protein ddG PCC, b_p (WT error, dG-side), a_p (ddG slope), dG MAE.

Qualifiers: single eval run abl_calib_ctrl_repro2_e14 (the calibration control
repro), test split, 28 held-out proteins. b_p and dG MAE are dG-side; a_p and
PCC are ddG-side (per-protein). No selection over runs.
"""
import json, csv, math
from collections import defaultdict

SRC   = 'eval_results/abl_calib_ctrl_repro2_e14.csv'
PLDDT = 'data/Processed_K50_dG_datasets/plddt.csv'
CALIB = 'results/calib_per_protein.json'

# --- dG MAE per protein, from the same CSV calc_bp.py used -------------------
mae_num = defaultdict(float); mae_cnt = defaultdict(int)
with open(SRC) as f:
    for r in csv.DictReader(f):
        p = r['protein']
        mae_num[p] += abs(float(r['pred_deltaG']) - float(r['deltaG']))
        mae_cnt[p] += 1
mae = {p: mae_num[p]/mae_cnt[p] for p in mae_num}

calib = json.load(open(CALIB))
assert set(mae) == set(calib), "protein set mismatch between eval CSV and calib json"

# --- mean pLDDT per protein --------------------------------------------------
pl = defaultdict(list)
with open(PLDDT) as f:
    for r in csv.DictReader(f):
        pl[r['protein']].append(float(r['plddt']))
plddt_mean = {p: sum(v)/len(v) for p, v in pl.items()}

test = sorted(calib)
have    = [p for p in test if p in plddt_mean]
missing = [p for p in test if p not in plddt_mean]
print('test proteins: %d   with pLDDT: %d   without: %d' % (len(test), len(have), len(missing)))
print('NO pLDDT (reported, not silently dropped): %s' % (', '.join(missing) if missing else 'none'))
print('plddt.csv covers %d proteins total (superset incl. non-test)' % len(plddt_mean))

def pearson(x, y):
    n = len(x)
    if n < 3: return float('nan')
    mx, my = sum(x)/n, sum(y)/n
    sxy = sum((a-mx)*(b-my) for a, b in zip(x, y))
    sxx = sum((a-mx)**2 for a in x); syy = sum((b-my)**2 for b in y)
    if sxx <= 0 or syy <= 0: return float('nan')
    return sxy/math.sqrt(sxx*syy)

def med(z):
    z = sorted(z); n = len(z)
    if not n: return float('nan')
    return z[n//2] if n % 2 else 0.5*(z[n//2-1]+z[n//2])

# --- terciles on mean pLDDT over the 24 that have it -------------------------
order = sorted(have, key=lambda p: plddt_mean[p])
n = len(order); k = n // 3
# n=24 -> 8/8/8
bounds = [(0, k), (k, 2*k), (2*k, n)]
names  = ['T1 low pLDDT', 'T2 mid pLDDT', 'T3 high pLDDT']

print()
print('=== per-protein table (sorted by mean pLDDT) ===')
print('%-22s %8s %8s %8s %8s %8s %6s' % ('protein','plddt','pcc_ddG','b_p','a_p','dG_MAE','n'))
for p in order:
    c = calib[p]
    print('%-22s %8.2f %8.4f %8.4f %8.4f %8.4f %6d'
          % (p, plddt_mean[p], c['pcc_ddg'], c['b_p_wt_error'], c['a_p'], mae[p], c['n']))
print('--- no pLDDT ---')
for p in missing:
    c = calib[p]
    print('%-22s %8s %8.4f %8.4f %8.4f %8.4f %6d'
          % (p, 'NA', c['pcc_ddg'], c['b_p_wt_error'], c['a_p'], mae[p], c['n']))

print()
print('=== terciles (n=24 with pLDDT, 8/8/8) ===')
print('%-14s %5s %13s %9s %9s %9s %9s %9s'
      % ('tercile','n','plddt range','pcc_med','b_p_med','|b_p|_med','a_p_med','MAE_med'))
rowsum = []
for (lo, hi), nm in zip(bounds, names):
    g = order[lo:hi]
    pl_r = (plddt_mean[g[0]], plddt_mean[g[-1]])
    pcc  = med([calib[p]['pcc_ddg'] for p in g])
    bp   = med([calib[p]['b_p_wt_error'] for p in g])
    abp  = med([abs(calib[p]['b_p_wt_error']) for p in g])
    ap   = med([calib[p]['a_p'] for p in g])
    mm   = med([mae[p] for p in g])
    rowsum.append((nm, len(g), pl_r, pcc, bp, abp, ap, mm))
    print('%-14s %5d %6.1f-%5.1f %9.4f %9.4f %9.4f %9.4f %9.4f'
          % (nm, len(g), pl_r[0], pl_r[1], pcc, bp, abp, ap, mm))

# means too (medians on n=8 are coarse)
print()
print('%-14s %5s %9s %9s %9s %9s %9s' % ('tercile','n','pcc_mean','b_p_mean','|b_p|_mean','a_p_mean','MAE_mean'))
for (lo, hi), nm in zip(bounds, names):
    g = order[lo:hi]
    f = lambda v: sum(v)/len(v)
    print('%-14s %5d %9.4f %9.4f %9.4f %9.4f %9.4f'
          % (nm, len(g), f([calib[p]['pcc_ddg'] for p in g]),
             f([calib[p]['b_p_wt_error'] for p in g]),
             f([abs(calib[p]['b_p_wt_error']) for p in g]),
             f([calib[p]['a_p'] for p in g]),
             f([mae[p] for p in g])))

# --- continuous correlations over the 24 -------------------------------------
x = [plddt_mean[p] for p in have]
print()
print('=== continuous, n=24 (Pearson r) ===')
for lab, y in [('pcc_ddG',   [calib[p]['pcc_ddg'] for p in have]),
               ('b_p',       [calib[p]['b_p_wt_error'] for p in have]),
               ('|b_p|',     [abs(calib[p]['b_p_wt_error']) for p in have]),
               ('a_p',       [calib[p]['a_p'] for p in have]),
               ('dG_MAE',    [mae[p] for p in have])]:
    r = pearson(x, y)
    # two-sided p via t, series approximation for the incomplete beta
    df = len(x) - 2
    t = abs(r)*math.sqrt(df/max(1e-12, 1-r*r))
    # regularized incomplete beta I_{df/(df+t^2)}(df/2, 1/2) = p
    def betacf(a, b, xx):
        MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
        qab, qap, qam = a+b, a+1, a-1
        c = 1.0; d = 1-qab*xx/qap
        if abs(d) < FPMIN: d = FPMIN
        d = 1/d; h = d
        for m in range(1, MAXIT+1):
            m2 = 2*m
            aa = m*(b-m)*xx/((qam+m2)*(a+m2))
            d = 1+aa*d;  c = 1+aa/c
            if abs(d) < FPMIN: d = FPMIN
            if abs(c) < FPMIN: c = FPMIN
            d = 1/d; h *= d*c
            aa = -(a+m)*(qab+m)*xx/((a+m2)*(qap+m2))
            d = 1+aa*d;  c = 1+aa/c
            if abs(d) < FPMIN: d = FPMIN
            if abs(c) < FPMIN: c = FPMIN
            d = 1/d; de = d*c; h *= de
            if abs(de-1) < EPS: break
        return h
    def betai(a, b, xx):
        if xx <= 0: return 0.0
        if xx >= 1: return 1.0
        lbeta = (math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)
                 + a*math.log(xx) + b*math.log(1-xx))
        bt = math.exp(lbeta)
        if xx < (a+1)/(a+b+2): return bt*betacf(a, b, xx)/a
        return 1 - bt*betacf(b, a, 1-xx)/b
    p_two = betai(df/2, 0.5, df/(df+t*t))
    print('  plddt_mean vs %-8s r=%+.4f  p=%.3g  %s'
          % (lab, r, p_two, 'PASSES' if p_two < 5.6e-4 else 'below Bonferroni bar 5.6e-4 (suggestive only)'))
