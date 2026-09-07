"""Is a_p a loss artefact (regression dilution) or a model failure?

CPU only. Reads eval CSVs + MsDs label files + AlphaFold PDBs (SASA).
Writes results/SLOPE_ORIGIN.md. Touches NO feature-vector code.
"""
import csv, math, glob, os, json, collections

EVALS = sorted(glob.glob('eval_results/*.csv'))
CTRL  = 'eval_results/abl_calib_ctrl_repro2_e14.csv'

def pearson(x, y):
    n = len(x)
    if n < 3: return float('nan')
    mx, my = sum(x)/n, sum(y)/n
    sxy = sum((a-mx)*(b-my) for a,b in zip(x,y))
    sxx = sum((a-mx)**2 for a in x); syy = sum((b-my)**2 for b in y)
    if sxx <= 0 or syy <= 0: return float('nan')
    return sxy/math.sqrt(sxx*syy)

def spearman(x, y):
    def rk(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0]*len(v); i = 0
        while i < len(order):
            j = i
            while j+1 < len(order) and v[order[j+1]] == v[order[i]]: j += 1
            avg = (i+j)/2.0 + 1
            for k in range(i, j+1): r[order[k]] = avg
            i = j+1
        return r
    return pearson(rk(x), rk(y))

def fit(x, y):
    n = len(x)
    if n < 3: return float('nan'), float('nan')
    mx, my = sum(x)/n, sum(y)/n
    sxx = sum((a-mx)**2 for a in x)
    if sxx <= 0: return float('nan'), float('nan')
    a = sum((p-mx)*(q-my) for p,q in zip(x,y))/sxx
    return a, my - a*mx

def med(z):
    z = sorted(z); n = len(z)
    if not n: return float('nan')
    return z[n//2] if n % 2 else 0.5*(z[n//2-1]+z[n//2])

def load_eval(path):
    rows = collections.defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r['protein']].append((float(r['deltaG']), float(r['pred_deltaG']),
                                       float(r['ddG']), float(r['pred_ddG'])))
    return rows

# ---------------- per-protein a_p, per eval file ----------------
per_file_ap = {}
for e in EVALS:
    rows = load_eval(e)
    ap = {}
    for p, v in rows.items():
        ddg  = [t[2] for t in v]; pddg = [t[3] for t in v]
        a, _ = fit(ddg, pddg)
        if not math.isnan(a): ap[p] = a
    per_file_ap[os.path.basename(e)] = ap

ctrl_rows = load_eval(CTRL)
ctrl_ap   = per_file_ap[os.path.basename(CTRL)]

# per-protein stats on the control eval
stats = {}
for p, v in ctrl_rows.items():
    ddg  = [t[2] for t in v]; pddg = [t[3] for t in v]
    n = len(ddg)
    mu = sum(ddg)/n
    sd = math.sqrt(sum((d-mu)**2 for d in ddg)/n)
    a, _ = fit(ddg, pddg)
    stats[p] = dict(n=n, sd_ddG=sd, var_ddG=sd*sd, a_p=a,
                    pcc=pearson(ddg, pddg),
                    rng=max(ddg)-min(ddg),
                    mean_abs_ddG=sum(abs(d) for d in ddg)/n)

print("proteins:", len(stats))
print("a_p median %.4f  min %.4f  max %.4f" %
      (med([s['a_p'] for s in stats.values()]),
       min(s['a_p'] for s in stats.values()),
       max(s['a_p'] for s in stats.values())))
json.dump(stats, open('results/_slope_stats.json','w'), indent=1)
