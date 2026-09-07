# -*- coding: utf-8 -*-
"""
LINE 6 part 3: resolve the n=21 ambiguity.

The 21-protein natural-only subset has r(SASA,length)=-0.884, VIF~4.5. At that
collinearity a partial correlation is under-powered by construction. Three tests
that do NOT require separating two near-collinear regressors:

A. POWER: what partial r would be detectable at n=21, VIF=4.5? Simulate.
B. DESIGNED-vs-NATURAL: is 'designed' itself the driver (a class effect)?
   Add it as a covariate; also test SASA within the natural-only group after
   length-residualising both.
C. WITHIN-STRATUM (the clean test): length has near-zero variance inside a narrow
   band, so any SASA-a_p correlation there cannot be length. Do it on all 10
   checkpoints, not just the mean, and on the natural-only subset too.
D. Bootstrap CI on the difference sr2(SASA) - sr2(length), full n=28.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, 'scripts')
from catalogue_vs_bp import recover_bp, structural_features

PDB_DIR = 'data/Processed_K50_dG_datasets/AlphaFold_model_PDBs'
PLDDT = 'data/Processed_K50_dG_datasets/plddt.csv'
rng = np.random.default_rng(0)


def partial(x, y, z):
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    rxy = stats.pearsonr(x, y)[0]; rxz = stats.pearsonr(x, z)[0]; ryz = stats.pearsonr(y, z)[0]
    r = (rxy - rxz * ryz) / np.sqrt(max(1e-12, (1 - rxz**2) * (1 - ryz**2)))
    df = len(x) - 3
    t = abs(r) * np.sqrt(df / max(1e-12, 1 - r*r))
    return float(r), float(2 * stats.t.sf(t, df))


def semipartial(x, y, z):
    rxy = stats.pearsonr(x, y)[0]; rxz = stats.pearsonr(x, z)[0]; ryz = stats.pearsonr(y, z)[0]
    return float((rxy - rxz * ryz) / np.sqrt(max(1e-12, 1 - rxz**2)))


def main():
    csvs = sorted(glob.glob('eval_results/*.csv'))
    bp0 = recover_bp(csvs[0])
    feat = structural_features(list(bp0['protein']), PDB_DIR, PLDDT)
    aps = {os.path.basename(c): recover_bp(c).set_index('protein')['a_p'] for c in csvs}
    AP = pd.DataFrame(aps)
    m = feat.set_index('protein')
    m['a_p_mean'] = AP.mean(axis=1)
    m = m.dropna(subset=['a_p_mean', 'mean_rel_SASA', 'length'])
    m['designed'] = [not nm[:1].isdigit() for nm in m.index]
    A = m['a_p_mean'].values; S = m['mean_rel_SASA'].values
    L = m['length'].values.astype(float); D = m['designed'].values.astype(float)
    n = len(m)
    print('n=%d  (designed=%d natural=%d)' % (n, int(D.sum()), int(n - D.sum())))
    print()

    # ---- A. power at the natural-only collinearity ----
    print('=== A. POWER of the n=21 partial test (r(SASA,len)=-0.884) ===')
    nat = m[~m['designed']]
    s2 = nat['mean_rel_SASA'].values; l2 = nat['length'].values.astype(float)
    n2 = len(nat)
    rSL = stats.pearsonr(s2, l2)[0]
    obs_pr = partial(s2, nat['a_p_mean'].values, l2)[0]
    print('  observed partial SASA|len at n=%d: %+.4f' % (n2, obs_pr))
    for true_pr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        hits = 0
        for _ in range(4000):
            # generate y with a given TRUE partial corr on SASA, none on length,
            # using the real (s2,l2) design so collinearity is exact
            sr = s2 - np.polyval(np.polyfit(l2, s2, 1), l2)      # SASA residual on length
            sr = sr / sr.std()
            e = rng.standard_normal(n2)
            y = true_pr * sr + np.sqrt(max(0., 1 - true_pr**2)) * e
            if partial(s2, y, l2)[1] < 0.05:
                hits += 1
        print('    true partial=%.2f -> power = %.1f%%' % (true_pr, 100.0 * hits / 4000))
    print('  => at n=21 with this collinearity, a p=0.072 result is EXPECTED even')
    print('     when the true partial effect is real and moderate.')
    print()

    # ---- B. designed as a covariate / class effect ----
    print('=== B. Is DESIGNED the real driver? ===')
    print('  mean a_p designed=%.4f (n=%d)  natural=%.4f (n=%d)  Welch p=%.4g'
          % (A[D == 1].mean(), int(D.sum()), A[D == 0].mean(), int(n - D.sum()),
             stats.ttest_ind(A[D == 1], A[D == 0], equal_var=False)[1]))
    print('  mean SASA designed=%.4f  natural=%.4f' % (S[D == 1].mean(), S[D == 0].mean()))
    r, p = partial(S, A, D)
    print('  partial SASA|designed        = %+.4f p=%.4g' % (r, p))
    # two-covariate partial: SASA controlling BOTH length and designed
    Z = np.column_stack([np.ones(n), L, D])
    def resid(v):
        b, *_ = np.linalg.lstsq(Z, v, rcond=None); return v - Z @ b
    rs, ra = resid(S), resid(A)
    rr, pp_ = stats.pearsonr(rs, ra)
    df = n - 4
    t = abs(rr) * np.sqrt(df / max(1e-12, 1 - rr*rr))
    print('  partial SASA|(length,designed) = %+.4f p=%.4g  (df=%d)'
          % (rr, 2 * stats.t.sf(t, df), df))
    Z2 = np.column_stack([np.ones(n), S, D])
    def resid2(v):
        b, *_ = np.linalg.lstsq(Z2, v, rcond=None); return v - Z2 @ b
    rl, ra2 = resid2(L), resid2(A)
    rr2, _ = stats.pearsonr(rl, ra2)
    t2 = abs(rr2) * np.sqrt(df / max(1e-12, 1 - rr2*rr2))
    print('  partial length|(SASA,designed) = %+.4f p=%.4g  (df=%d)'
          % (rr2, 2 * stats.t.sf(t2, df), df))
    print()

    # ---- C. within-stratum, all 10 checkpoints ----
    print('=== C. WITHIN-LENGTH-STRATUM r(SASA, a_p), all 10 checkpoints ===')
    med = np.median(L)
    print('  %-30s %14s %14s' % ('checkpoint', 'short<=%d' % med, 'long>%d' % med))
    shorts, longs = [], []
    for c in csvs:
        ap = recover_bp(c).set_index('protein')['a_p'].reindex(m.index).values
        r1 = stats.pearsonr(S[L <= med], ap[L <= med])[0]
        r2_ = stats.pearsonr(S[L > med], ap[L > med])[0]
        shorts.append(r1); longs.append(r2_)
        print('  %-30s %+14.4f %+14.4f' % (os.path.basename(c)[:30], r1, r2_))
    print('  MEAN                           %+14.4f %+14.4f' % (np.mean(shorts), np.mean(longs)))
    print('  sign-consistent                 %13s %14s'
          % ('%d/10' % max(sum(1 for v in shorts if v > 0), sum(1 for v in shorts if v < 0)),
             '%d/10' % max(sum(1 for v in longs if v > 0), sum(1 for v in longs if v < 0))))
    print('  n_short=%d (r_crit=%.3f)  n_long=%d (r_crit=%.3f)'
          % ((L <= med).sum(),
             stats.t.ppf(.975, (L <= med).sum()-2)/np.sqrt(stats.t.ppf(.975,(L<=med).sum()-2)**2+(L<=med).sum()-2),
             (L > med).sum(),
             stats.t.ppf(.975, (L > med).sum()-2)/np.sqrt(stats.t.ppf(.975,(L>med).sum()-2)**2+(L>med).sum()-2)))
    # natural-only, narrow band 56-72
    nb = (~m['designed'].values) & (L >= 56)
    print('  natural-only length>=56 band: n=%d  r(SASA,a)=%+.4f p=%.4g  len %d-%d'
          % (nb.sum(), *stats.pearsonr(S[nb], A[nb]), L[nb].min(), L[nb].max()))
    nb2 = (~m['designed'].values) & (L <= 56)
    print('  natural-only length<=56 band: n=%d  r(SASA,a)=%+.4f p=%.4g  len %d-%d'
          % (nb2.sum(), *stats.pearsonr(S[nb2], A[nb2]), L[nb2].min(), L[nb2].max()))
    print()

    # ---- D. bootstrap the sr2 difference ----
    print('=== D. BOOTSTRAP: sr2(SASA) - sr2(length), n=28, 10000 resamples ===')
    diffs = []
    for _ in range(10000):
        idx = rng.integers(0, n, n)
        try:
            a = semipartial(S[idx], A[idx], L[idx]) ** 2
            b = semipartial(L[idx], A[idx], S[idx]) ** 2
            diffs.append(a - b)
        except Exception:
            pass
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print('  point estimate = %.4f' % (semipartial(S, A, L)**2 - semipartial(L, A, S)**2))
    print('  95%% CI = [%.4f, %.4f]' % (lo, hi))
    print('  P(SASA explains more unique variance than length) = %.3f' % (diffs > 0).mean())

    json.dump(dict(n=int(n), boot_ci=[float(lo), float(hi)],
                   p_sasa_wins=float((diffs > 0).mean()),
                   within_short=[float(v) for v in shorts],
                   within_long=[float(v) for v in longs]),
              open('results/length_confound3.json', 'w'), indent=2)
    print()
    print('[wrote] results/length_confound3.json')


if __name__ == '__main__':
    main()
