#!/usr/bin/env python3
"""Analyse an eval CSV: dG MAE, std(b_p), per-protein PCC, a_p, r, s.
Usage: arm_analyze.py <csv> [<csv> ...]
Prints one block per CSV. All quantities defined exactly as in FINDINGS.md:
  per protein, fit pred_ddG ~ a_p*true_ddG + b_p_ddg  (within-protein slope)
  b_p (the OFFSET of record) is the dG-space wild-type error: pred_dG - dG on the WT row.
"""
import sys, csv, math
from collections import OrderedDict

def pearson(x, y):
    n = len(x)
    if n < 3: return float('nan')
    mx = sum(x)/n; my = sum(y)/n
    sxx = sum((a-mx)**2 for a in x); syy = sum((b-my)**2 for b in y)
    sxy = sum((a-mx)*(b-my) for a, b in zip(x, y))
    if sxx <= 0 or syy <= 0: return float('nan')
    return sxy/math.sqrt(sxx*syy)

def std(v):
    n = len(v)
    if n < 2: return float('nan')
    m = sum(v)/n
    return math.sqrt(sum((a-m)**2 for a in v)/(n-1))

def median(v):
    s = sorted(v); n = len(s)
    if n == 0: return float('nan')
    return s[n//2] if n % 2 else 0.5*(s[n//2-1]+s[n//2])

def analyse(path):
    prots = OrderedDict()
    with open(path) as f:
        for row in csv.DictReader(f):
            p = row['protein']
            prots.setdefault(p, []).append((
                float(row['deltaG']), float(row['pred_deltaG']),
                float(row['ddG']), float(row['pred_ddG'])))

    aps, rs, ss, ppcc, bps_ddg, bp_wt, dg_ae = [], [], [], [], [], [], []
    pooled_t, pooled_p = [], []
    for p, rows in prots.items():
        dg  = [r[0] for r in rows]; pdg = [r[1] for r in rows]
        t   = [r[2] for r in rows]; pr  = [r[3] for r in rows]
        # dG MAE over every row of every protein
        dg_ae.extend(abs(a-b) for a, b in zip(dg, pdg))
        # b_p as the dG-space WT error (row 0 is the wild type)
        bp_wt.append(pdg[0] - dg[0])
        if len(t) < 3: continue
        r = pearson(t, pr)
        st, sp = std(t), std(pr)
        if not (st > 0) or not (sp > 0) or r != r: continue
        s = sp/st
        a = r*s
        mt = sum(t)/len(t); mp = sum(pr)/len(pr)
        aps.append(a); rs.append(r); ss.append(s)
        ppcc.append(r)
        bps_ddg.append(mp - a*mt)
        pooled_t.extend(t); pooled_p.extend(pr)

    n_prot = len(prots)
    out = OrderedDict()
    out['file'] = path.split('/')[-1]
    out['n_proteins'] = n_prot
    out['n_rows'] = sum(len(v) for v in prots.values())
    out['dG_MAE'] = sum(dg_ae)/len(dg_ae)
    out['std_bp_dG_WTerr'] = std(bp_wt)
    out['mean_bp_dG_WTerr'] = sum(bp_wt)/len(bp_wt)
    out['std_bp_ddG_intercept'] = std(bps_ddg)
    out['perprot_ddG_PCC'] = sum(ppcc)/len(ppcc)
    out['a_p_median'] = median(aps)
    out['r_median'] = median(rs)
    out['s_median'] = median(ss)
    out['pooled_ddG_PCC'] = pearson(pooled_t, pooled_p)
    # offset-removal oracle on ddG: subtract each protein's own mean residual
    res_t, res_p = [], []
    for p, rows in prots.items():
        t = [r[2] for r in rows]; pr = [r[3] for r in rows]
        if len(t) < 3: continue
        d = [b-a for a, b in zip(t, pr)]
        md = sum(d)/len(d)
        res_t.extend(t); res_p.extend(x-md for x in pr)
    out['oracle_offset_removed_PCC'] = pearson(res_t, res_p)
    return out

for path in sys.argv[1:]:
    try:
        o = analyse(path)
    except Exception as e:
        print(f"ERROR {path}: {e}"); continue
    print("=" * 68)
    for k, v in o.items():
        print(f"  {k:26s} = {v:.4f}" if isinstance(v, float) else f"  {k:26s} = {v}")
