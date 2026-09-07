"""SLOPE_ORIGIN: is a_p regression dilution (loss artefact) or model failure?
CPU only. Reads eval CSVs + cached structural table. Writes results/slope_origin.json.
Touches NO feature-vector code.
"""
import csv, math, glob, os, json, collections

CTRL  = 'eval_results/abl_calib_ctrl_repro2_e14.csv'
EVALS = sorted(glob.glob('eval_results/*.csv'))


def pearson(x, y):
    n = len(x)
    if n < 3:
        return float('nan')
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return float('nan')
    return sxy / math.sqrt(sxx * syy)


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x, y):
    return pearson(rank(x), rank(y))


def fit(x, y):
    n = len(x)
    if n < 3:
        return float('nan'), float('nan')
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx <= 0:
        return float('nan'), float('nan')
    a = sum((p - mx) * (q - my) for p, q in zip(x, y)) / sxx
    return a, my - a * mx


def med(z):
    z = sorted(z)
    n = len(z)
    if not n:
        return float('nan')
    return z[n // 2] if n % 2 else 0.5 * (z[n // 2 - 1] + z[n // 2])


def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def tp(r, n):
    """two-sided p-value for a pearson r, via the regularized incomplete beta"""
    if n <= 2 or math.isnan(r):
        return float('nan')
    if abs(r) >= 1.0:
        return 0.0
    t = abs(r) * math.sqrt((n - 2) / (1 - r * r))
    df = n - 2
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    if not (0.0 < x < 1.0):
        return float('nan')
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        ib = math.exp(lb) * betacf(a, b, x) / a
    else:
        ib = 1.0 - math.exp(lb) * betacf(b, a, 1 - x) / b
    return max(0.0, min(1.0, ib))


def load_eval(path):
    rows = collections.defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r['protein']].append((float(r['deltaG']), float(r['pred_deltaG']),
                                       float(r['ddG']), float(r['pred_ddG'])))
    return rows


ctrl = load_eval(CTRL)

stats = {}
for p, v in ctrl.items():
    ddg = [t[2] for t in v]
    pddg = [t[3] for t in v]
    n = len(ddg)
    mu = sum(ddg) / n
    var = sum((d - mu) ** 2 for d in ddg) / n
    a, _ = fit(ddg, pddg)
    sd = sorted(ddg)
    stats[p] = dict(n=n, var_ddG=var, sd_ddG=math.sqrt(var), a_p=a,
                    pcc=pearson(ddg, pddg),
                    rng=max(ddg) - min(ddg),
                    iqr=sd[int(.75 * n)] - sd[int(.25 * n)])

cat = json.load(open('results/catalogue_vs_bp.json'))
mt = {r['protein']: r for r in cat['merged_table']}

P = sorted(stats)
ap = [stats[p]['a_p'] for p in P]
sasa = [mt[p]['mean_rel_SASA'] for p in P]
nmut = [float(stats[p]['n']) for p in P]
sdd = [stats[p]['sd_ddG'] for p in P]
vard = [stats[p]['var_ddG'] for p in P]
rng = [stats[p]['rng'] for p in P]
pcc = [stats[p]['pcc'] for p in P]
L = [float(mt[p]['length']) for p in P]
N = len(P)

res = {'n_proteins': N, 'eval_csv': CTRL}


def rec(k, x, y):
    r = pearson(x, y)
    s = spearman(x, y)
    pv = tp(r, N)
    res.setdefault('correlations', {})[k] = dict(r=r, p=pv, rho=s, n=N)
    print("%-34s r=%+.4f  p=%.4g  rho=%+.4f" % (k, r, pv, s))


print("=== N = %d proteins ===" % N)
print("a_p: median %.4f  mean %.4f  min %.4f  max %.4f"
      % (med(ap), sum(ap) / N, min(ap), max(ap)))
print()
print("--- item 4: the trivial statistical confounds ---")
rec('a_p ~ n_mutations', ap, nmut)
rec('a_p ~ sd_true_ddG', ap, sdd)
rec('a_p ~ var_true_ddG', ap, vard)
rec('a_p ~ range_true_ddG', ap, rng)
rec('a_p ~ length', ap, L)
rec('a_p ~ per_protein_PCC', ap, pcc)
print()
print("--- the structural signal ---")
rec('a_p ~ mean_rel_SASA', ap, sasa)
print()
print("--- item 3: does any noise proxy track SASA? ---")
rec('n_mutations ~ mean_rel_SASA', nmut, sasa)
rec('sd_true_ddG ~ mean_rel_SASA', sdd, sasa)
rec('var_true_ddG ~ mean_rel_SASA', vard, sasa)
rec('length ~ mean_rel_SASA', L, sasa)

# ---------- attenuation arithmetic ----------
obs = med(ap)
lam = (1 - obs) / obs
vt = med(vard)
vn_from_obs = vt * (1 - obs)
print()
print("=== ATTENUATION ARITHMETIC ===")
print("median a_p                       = %.4f" % obs)
print("implied var_noise/var_true       = %.4f" % lam)
print("median var(observed ddG)         = %.4f   (sd %.4f kcal/mol)" % (vt, math.sqrt(vt)))
print("=> implied var_noise             = %.4f   (sigma_noise %.4f kcal/mol)"
      % (vn_from_obs, math.sqrt(vn_from_obs)))
print("=> implied var_true              = %.4f   (sd_true     %.4f kcal/mol)"
      % (vt * obs, math.sqrt(vt * obs)))
res['attenuation'] = dict(median_a_p=obs, lambda_noise_over_true=lam,
                          median_var_obs_ddG=vt, sd_obs_ddG=math.sqrt(vt),
                          implied_var_noise=vn_from_obs,
                          implied_sigma_noise=math.sqrt(vn_from_obs),
                          implied_var_true=vt * obs,
                          implied_sd_true=math.sqrt(vt * obs))

per = []
for p in P:
    a = stats[p]['a_p']
    vo = stats[p]['var_ddG']
    if 0 < a < 1:
        per.append((p, a, math.sqrt(vo), math.sqrt(vo * (1 - a))))
print()
print("per-protein sigma_noise REQUIRED to explain a_p by attenuation alone:")
print("  median %.4f   min %.4f   max %.4f   (n=%d of %d with 0<a_p<1)"
      % (med([t[3] for t in per]), min(t[3] for t in per),
         max(t[3] for t in per), len(per), N))
res['per_protein_sigma_required'] = dict(
    median=med([t[3] for t in per]), min=min(t[3] for t in per),
    max=max(t[3] for t in per), n=len(per))

sig = [math.sqrt(stats[p]['var_ddG'] * (1 - stats[p]['a_p'])) for p in P if 0 < stats[p]['a_p'] < 1]
ss = [mt[p]['mean_rel_SASA'] for p in P if 0 < stats[p]['a_p'] < 1]
r = pearson(sig, ss)
print("required sigma_noise ~ mean_rel_SASA:  r=%+.4f  p=%.4g  (n=%d)"
      % (r, tp(r, len(sig)), len(sig)))
res['required_sigma_vs_sasa'] = dict(r=r, p=tp(r, len(sig)), n=len(sig))

# ---------- replication ----------
print()
print("=== replication across %d eval CSVs ===" % len(EVALS))
reps = []
for e in EVALS:
    rows = load_eval(e)
    aps, sas, ns, sds = [], [], [], []
    for p in sorted(rows):
        if p not in mt:
            continue
        ddg = [t[2] for t in rows[p]]
        pddg = [t[3] for t in rows[p]]
        a, _ = fit(ddg, pddg)
        if math.isnan(a):
            continue
        n = len(ddg)
        mu = sum(ddg) / n
        aps.append(a)
        sas.append(mt[p]['mean_rel_SASA'])
        ns.append(float(n))
        sds.append(math.sqrt(sum((d - mu) ** 2 for d in ddg) / n))
    r1, r2, r3 = pearson(aps, sas), pearson(aps, ns), pearson(aps, sds)
    reps.append(dict(file=os.path.basename(e), median_a_p=med(aps),
                     r_sasa=r1, r_nmut=r2, r_sd=r3, n=len(aps)))
    print("%-36s med_a_p=%.3f  r(SASA)=%+.3f  r(n)=%+.3f  r(sd)=%+.3f"
          % (os.path.basename(e), med(aps), r1, r2, r3))
res['replication'] = reps
rs = [x['r_sasa'] for x in reps]
rn = [x['r_nmut'] for x in reps]
rd = [x['r_sd'] for x in reps]
print("mean r(a_p,SASA) = %+.4f   positive %d/%d" % (sum(rs) / len(rs), sum(1 for v in rs if v > 0), len(rs)))
print("mean r(a_p,n)    = %+.4f   positive %d/%d" % (sum(rn) / len(rn), sum(1 for v in rn if v > 0), len(rn)))
print("mean r(a_p,sd)   = %+.4f   positive %d/%d" % (sum(rd) / len(rd), sum(1 for v in rd if v > 0), len(rd)))
res['replication_summary'] = dict(
    mean_r_sasa=sum(rs) / len(rs), pos_sasa=sum(1 for v in rs if v > 0),
    mean_r_nmut=sum(rn) / len(rn), pos_nmut=sum(1 for v in rn if v > 0),
    mean_r_sd=sum(rd) / len(rd), pos_sd=sum(1 for v in rd if v > 0), k=len(rs))


def partial(x, y, z):
    rxy, rxz, ryz = pearson(x, y), pearson(x, z), pearson(y, z)
    d = math.sqrt((1 - rxz * rxz) * (1 - ryz * ryz))
    return (rxy - rxz * ryz) / d if d > 0 else float('nan')


print()
print("=== partial r(a_p, mean_rel_SASA | confound) ===")
for lbl, z in (('n_mutations', nmut), ('sd_true_ddG', sdd),
               ('var_true_ddG', vard), ('length', L)):
    pr = partial(ap, sasa, z)
    print("  controlling %-14s partial r = %+.4f" % (lbl, pr))
    res.setdefault('partial', {})[lbl] = pr

json.dump(res, open('results/slope_origin.json', 'w'), indent=1)
print()
print("wrote results/slope_origin.json")
