"""Forward simulation: can pure attenuation reproduce r(a_p, mean_rel_SASA)=+0.71?

Take the REAL per-protein true ddG vectors. Build a synthetic "predictor" that is
attenuation-only: pred = a*ddG_true + noise, where a comes ONLY from label noise
that is INDEPENDENT of structure (the attenuation null). Refit a_p exactly as
calc_bp.py does, and ask how often r(a_p, SASA) reaches the observed +0.7139.
CPU only, no model, no feature vector.
"""
import csv, math, json, random, collections

CTRL = 'eval_results/abl_calib_ctrl_repro2_e14.csv'
OBS_R = 0.7139
TRIALS = 2000
random.seed(20260907)


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


def fit(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx <= 0:
        return float('nan')
    return sum((p - mx) * (q - my) for p, q in zip(x, y)) / sxx


def med(z):
    z = sorted(z)
    n = len(z)
    return z[n // 2] if n % 2 else 0.5 * (z[n // 2 - 1] + z[n // 2])


rows = collections.defaultdict(list)
with open(CTRL) as f:
    for r in csv.DictReader(f):
        rows[r['protein']].append(float(r['ddG']))

cat = json.load(open('results/catalogue_vs_bp.json'))
mt = {r['protein']: r for r in cat['merged_table']}
P = sorted(rows)
sasa = [mt[p]['mean_rel_SASA'] for p in P]

# global sigma_noise implied by the median a_p (structure-independent, by construction)
allv = []
for p in P:
    d = rows[p]
    mu = sum(d) / len(d)
    allv.append(sum((x - mu) ** 2 for x in d) / len(d))
vt = med(allv)
sigma_n = math.sqrt(vt * (1 - 0.4975))
print("attenuation null: one global sigma_noise = %.4f kcal/mol (structure-independent)" % sigma_n)

hits = 0
rs = []
for t in range(TRIALS):
    aps = []
    for p in P:
        d = rows[p]
        # errors-in-variables: the LABEL we regress on carries noise; the model
        # learned the true signal. pred = true, observed label = true + noise.
        # Fitting pred on the noisy label attenuates the slope.
        obs = [x + random.gauss(0, sigma_n) for x in d]
        aps.append(fit(obs, d))
    r = pearson(aps, sasa)
    rs.append(r)
    if r >= OBS_R:
        hits += 1

rs.sort()
print("simulated r(a_p, mean_rel_SASA) under pure attenuation, %d trials:" % TRIALS)
print("  mean   %+.4f" % (sum(rs) / len(rs)))
print("  sd     %.4f" % math.sqrt(sum((v - sum(rs) / len(rs)) ** 2 for v in rs) / len(rs)))
print("  2.5%%   %+.4f   97.5%%  %+.4f" % (rs[int(.025 * len(rs))], rs[int(.975 * len(rs))]))
print("  max    %+.4f" % rs[-1])
print("  P(r >= observed %.4f) = %d/%d = %.5f" % (OBS_R, hits, TRIALS, hits / TRIALS))

json.dump(dict(sigma_noise=sigma_n, trials=TRIALS, observed_r=OBS_R,
               sim_mean_r=sum(rs) / len(rs), sim_max_r=rs[-1],
               sim_p975=rs[int(.975 * len(rs))],
               p_value=hits / TRIALS),
          open('results/slope_origin_sim.json', 'w'), indent=1)
print("wrote results/slope_origin_sim.json")
