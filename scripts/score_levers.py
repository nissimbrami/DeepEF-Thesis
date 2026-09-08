"""Unified lever scorer. Reports pooled ddG PCC, per-protein PCC, a_p, r, s, std(b_p), dG MAE.

a_p = r * s decomposition per PART III of FINDINGS: for a least-squares fit of pred on true,
a = r * (std(pred)/std(true)). Reported r/s/a_p are the MEDIANS over proteins, matching
the control triple (a_p 0.499, r 0.798, s 0.635) quoted in FINDINGS 3.2.
"""
import sys, csv, math
from collections import defaultdict

def pearson(x, y):
    n = len(x)
    if n < 3: return float('nan')
    mx, my = sum(x)/n, sum(y)/n
    sxy = sum((a-mx)*(b-my) for a, b in zip(x, y))
    sxx = sum((a-mx)**2 for a in x); syy = sum((b-my)**2 for b in y)
    if sxx <= 0 or syy <= 0: return float('nan')
    return sxy/math.sqrt(sxx*syy)

def std(v):
    n = len(v)
    if n < 2: return float('nan')
    m = sum(v)/n
    return math.sqrt(sum((a-m)**2 for a in v)/n)

def med(z):
    z = sorted(a for a in z if not math.isnan(a))
    if not z: return float('nan')
    n = len(z)
    return z[n//2] if n % 2 else 0.5*(z[n//2-1]+z[n//2])

def score(path):
    rows = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r['protein']].append((float(r['deltaG']), float(r['pred_deltaG']),
                                       float(r['ddG']), float(r['pred_ddG'])))
    all_ddg, all_pddg, all_dg, all_pdg = [], [], [], []
    pp_pcc, aps, rs, ss, bps = [], [], [], [], []
    for p, v in rows.items():
        dg  = [t[0] for t in v]; pdg  = [t[1] for t in v]
        ddg = [t[2] for t in v]; pddg = [t[3] for t in v]
        all_ddg += ddg; all_pddg += pddg; all_dg += dg; all_pdg += pdg
        r = pearson(ddg, pddg)
        st, sp = std(ddg), std(pddg)
        s = sp/st if (st and st > 0) else float('nan')
        if not math.isnan(r) and not math.isnan(s):
            pp_pcc.append(r); rs.append(r); ss.append(s); aps.append(r*s)
        # b_p = WT error, the row with ddG == 0
        wt = [i for i, t in enumerate(v) if t[2] == 0.0]
        if wt: bps.append(pdg[wt[0]] - dg[wt[0]])
    dg_mae = sum(abs(a-b) for a, b in zip(all_dg, all_pdg))/len(all_dg)
    ddg_mae = sum(abs(a-b) for a, b in zip(all_ddg, all_pddg))/len(all_ddg)
    return dict(path=path, n_prot=len(rows), n_rows=len(all_ddg),
                pooled_ddg_pcc=pearson(all_ddg, all_pddg),
                pooled_dg_pcc=pearson(all_dg, all_pdg),
                per_protein_pcc=med(pp_pcc), a_p=med(aps), r=med(rs), s=med(ss),
                std_bp=std(bps), mean_bp=sum(bps)/len(bps) if bps else float('nan'),
                n_bp=len(bps), dg_mae=dg_mae, ddg_mae=ddg_mae)

FMT = ('%-28s prot=%2d rows=%5d | pooled_ddG_PCC=%.4f per-prot_PCC=%.4f a_p=%.4f r=%.4f s=%.4f '
       '| std(b_p)=%.4f mean(b_p)=%+.4f | dG_MAE=%.4f ddG_MAE=%.4f pooled_dG_PCC=%.4f')
if __name__ == '__main__':
    for p in sys.argv[1:]:
        try:
            d = score(p)
        except Exception as e:
            print('%-28s ERROR %s' % (p.split('/')[-1], e)); continue
        print(FMT % (d['path'].split('/')[-1][:28], d['n_prot'], d['n_rows'],
                     d['pooled_ddg_pcc'], d['per_protein_pcc'], d['a_p'], d['r'], d['s'],
                     d['std_bp'], d['mean_bp'], d['dg_mae'], d['ddg_mae'], d['pooled_dg_pcc']))
