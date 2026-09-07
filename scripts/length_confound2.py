# -*- coding: utf-8 -*-
"""
LINE 6 part 2: robustness of the SASA-survives-length result.

1. Multivariate OLS a_p ~ SASA + length (standardised betas, t-stats, VIF).
2. Semipartial (part) correlations: unique variance each explains.
3. Leave-one-protein-out on the partial corr (is it 1-2 proteins?).
4. Drop the 4 designed 43aa proteins (they anchor the short/exposed corner).
5. Length-matched: restrict to the 43-56aa and 57-72aa halves separately.
6. Sanity: is SASA -> a_p just SASA_per_residue or frac_buried in disguise?
7. Also check b_p_wt_error channel for the same confound (frac_buried vs length).
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, 'scripts')
from catalogue_vs_bp import recover_bp, structural_features

PDB_DIR = 'data/Processed_K50_dG_datasets/AlphaFold_model_PDBs'
PLDDT = 'data/Processed_K50_dG_datasets/plddt.csv'


def partial(x, y, z):
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    rxy = stats.pearsonr(x, y)[0]; rxz = stats.pearsonr(x, z)[0]; ryz = stats.pearsonr(y, z)[0]
    den = np.sqrt(max(1e-12, (1 - rxz**2) * (1 - ryz**2)))
    r = (rxy - rxz * ryz) / den
    n = len(x); df = n - 3
    t = abs(r) * np.sqrt(df / max(1e-12, 1 - r*r))
    return float(r), float(2 * stats.t.sf(t, df))


def semipartial(x, y, z):
    """corr(y, x controlling z for x only) = unique contribution of x to y."""
    rxy = stats.pearsonr(x, y)[0]; rxz = stats.pearsonr(x, z)[0]; ryz = stats.pearsonr(y, z)[0]
    return float((rxy - rxz * ryz) / np.sqrt(max(1e-12, 1 - rxz**2)))


def ols(y, X, names):
    """standardised OLS with t-stats."""
    y = (y - y.mean()) / y.std(ddof=1)
    Xs = (X - X.mean(0)) / X.std(0, ddof=1)
    A = np.column_stack([np.ones(len(y)), Xs])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    n, k = A.shape
    s2 = resid @ resid / (n - k)
    cov = s2 * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), n - k)
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    return dict(names=['const'] + names, beta=beta.tolist(), t=t.tolist(),
                p=p.tolist(), r2=float(r2))


def vif(X):
    out = []
    for j in range(X.shape[1]):
        y = X[:, j]; Z = np.delete(X, j, axis=1)
        A = np.column_stack([np.ones(len(y)), Z])
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ b
        r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
        out.append(float(1 / max(1e-9, 1 - r2)))
    return out


def main():
    csvs = sorted(glob.glob('eval_results/*.csv'))
    bp0 = recover_bp(csvs[0])
    feat = structural_features(list(bp0['protein']), PDB_DIR, PLDDT)

    # per-checkpoint a_p matrix -> also build the MEAN a_p across checkpoints
    aps = {}
    bps = {}
    for c in csvs:
        b = recover_bp(c).set_index('protein')
        aps[os.path.basename(c)] = b['a_p']
        bps[os.path.basename(c)] = b['b_p_wt_error']
    AP = pd.DataFrame(aps)
    BP = pd.DataFrame(bps)
    m = feat.set_index('protein')
    m['a_p_mean'] = AP.mean(axis=1)
    m['a_p_sd_across_ckpt'] = AP.std(axis=1)
    m['bp_wt_mean'] = BP.mean(axis=1)
    m = m.dropna(subset=['a_p_mean', 'mean_rel_SASA', 'length'])
    m['designed'] = [not nm[:1].isdigit() for nm in m.index]

    A = m['a_p_mean'].values; S = m['mean_rel_SASA'].values
    L = m['length'].values.astype(float); n = len(m)
    print('n=%d proteins; a_p averaged over %d checkpoints' % (n, len(csvs)))
    print('ICC-style: mean across-checkpoint SD of a_p = %.4f vs between-protein SD = %.4f'
          % (m['a_p_sd_across_ckpt'].mean(), m['a_p_mean'].std(ddof=1)))
    print()

    print('=== 1. MULTIVARIATE OLS  a_p ~ SASA + length (standardised) ===')
    X = np.column_stack([S, L])
    res = ols(A, X, ['mean_rel_SASA', 'length'])
    for nm, b, t, p in zip(res['names'], res['beta'], res['t'], res['p']):
        print('  %-16s beta=%+.4f  t=%+.3f  p=%.4g' % (nm, b, t, p))
    print('  R2=%.4f    VIF: SASA=%.2f length=%.2f' % (res['r2'], *vif(X)))
    print()

    print('=== 2. SEMIPARTIAL (unique variance) on mean a_p ===')
    sp_S = semipartial(S, A, L); sp_L = semipartial(L, A, S)
    print('  SASA unique   sr=%+.4f  sr2=%.4f (%.1f%% of variance)' % (sp_S, sp_S**2, 100*sp_S**2))
    print('  length unique sr=%+.4f  sr2=%.4f (%.1f%% of variance)' % (sp_L, sp_L**2, 100*sp_L**2))
    print('  raw r(SASA,a)=%+.4f  raw r(len,a)=%+.4f  r(SASA,len)=%+.4f'
          % (stats.pearsonr(S,A)[0], stats.pearsonr(L,A)[0], stats.pearsonr(S,L)[0]))
    print()

    print('=== 3. LEAVE-ONE-PROTEIN-OUT on partial(SASA, a | length) ===')
    vals = []
    for i in range(n):
        k = [j for j in range(n) if j != i]
        r, p = partial(S[k], A[k], L[k])
        vals.append((r, p, m.index[i]))
    rs = np.array([v[0] for v in vals])
    print('  min=%+.4f (drop %s)   max=%+.4f (drop %s)   median=%+.4f'
          % (rs.min(), vals[int(rs.argmin())][2], rs.max(), vals[int(rs.argmax())][2], np.median(rs)))
    print('  all %d LOPO partials p<0.05? %s (max p = %.4g)'
          % (n, all(v[1] < 0.05 for v in vals), max(v[1] for v in vals)))
    print()
    print('  same for partial(length, a | SASA):')
    vals2 = [partial(L[[j for j in range(n) if j != i]], A[[j for j in range(n) if j != i]],
                     S[[j for j in range(n) if j != i]]) for i in range(n)]
    rs2 = np.array([v[0] for v in vals2])
    print('  min=%+.4f max=%+.4f median=%+.4f ; p<0.05 in %d/%d LOPO folds'
          % (rs2.min(), rs2.max(), np.median(rs2), sum(1 for v in vals2 if v[1] < 0.05), n))
    print()

    print('=== 4. DROP the 4 designed 43aa proteins ===')
    sub = m[~m['designed']]
    print('  n=%d remaining, length range %d-%d' % (len(sub), sub['length'].min(), sub['length'].max()))
    a2 = sub['a_p_mean'].values; s2 = sub['mean_rel_SASA'].values; l2 = sub['length'].values.astype(float)
    print('  raw r(SASA,a)=%+.4f p=%.4g   raw r(len,a)=%+.4f p=%.4g'
          % (stats.pearsonr(s2,a2)[0], stats.pearsonr(s2,a2)[1],
             stats.pearsonr(l2,a2)[0], stats.pearsonr(l2,a2)[1]))
    r,p = partial(s2,a2,l2); print('  partial SASA|len   = %+.4f p=%.4g' % (r,p))
    r,p = partial(l2,a2,s2); print('  partial len |SASA  = %+.4f p=%.4g' % (r,p))
    print('  r(SASA,len) here   = %+.4f' % stats.pearsonr(s2,l2)[0])
    print()

    print('=== 5. LENGTH-STRATIFIED (within-stratum SASA vs a_p) ===')
    med = np.median(L)
    for lab, mask in [('short (<=%daa)' % med, L <= med), ('long (>%daa)' % med, L > med)]:
        aa = A[mask]; ss = S[mask]; ll = L[mask]
        r, p = stats.pearsonr(ss, aa)
        rl, pl = stats.pearsonr(ll, aa)
        print('  %-16s n=%2d  r(SASA,a)=%+.4f p=%.4g   r(len,a)=%+.4f p=%.4g  len %d-%d'
              % (lab, mask.sum(), r, p, rl, pl, ll.min(), ll.max()))
    print()

    print('=== 6. OTHER EXPOSURE PROXIES vs a_p (raw / partial on length) ===')
    for col in ['SASA_per_residue', 'frac_buried_rel_lt_0.25', 'frac_exposed_rel_gt_0.5',
                'total_SASA', 'mean_rel_SASA_hydrophobic', 'mean_hydropathy_KD',
                'frac_hydrophobic', 'plddt_mean', 'SASA_over_len_pow_073']:
        if col not in m.columns:
            continue
        d = m[[col, 'a_p_mean', 'length']].dropna()
        if len(d) < 8:
            continue
        x = d[col].values.astype(float); y = d['a_p_mean'].values; z = d['length'].values.astype(float)
        r, p = stats.pearsonr(x, y)
        pr, pp = partial(x, y, z)
        print('  %-26s n=%2d  raw=%+.4f (p=%.3g)   |len=%+.4f (p=%.3g)   r(x,len)=%+.4f'
              % (col, len(d), r, p, pr, pp, stats.pearsonr(x, z)[0]))
    print()

    print('=== 7. b_p CHANNEL: frac_buried vs b_p_wt_error, controlling length ===')
    d = m[['frac_buried_rel_lt_0.25', 'bp_wt_mean', 'length', 'mean_rel_SASA']].dropna()
    x = d['frac_buried_rel_lt_0.25'].values; y = d['bp_wt_mean'].values
    z = d['length'].values.astype(float)
    print('  n=%d  raw r(frac_buried, b_p)=%+.4f p=%.4g' % (len(d), *stats.pearsonr(x, y)))
    print('       raw r(length,      b_p)=%+.4f p=%.4g' % stats.pearsonr(z, y))
    r, p = partial(x, y, z); print('       partial frac_buried|len = %+.4f p=%.4g' % (r, p))
    r, p = partial(z, y, x); print('       partial len|frac_buried = %+.4f p=%.4g' % (r, p))
    print('       r(frac_buried, length)  = %+.4f' % stats.pearsonr(x, z)[0])

    out = dict(n=int(n), ols=res, vif=vif(X),
               semipartial=dict(SASA=sp_S, length=sp_L),
               lopo_partial_SASA=dict(min=float(rs.min()), max=float(rs.max()),
                                      median=float(np.median(rs))),
               lopo_partial_len=dict(min=float(rs2.min()), max=float(rs2.max()),
                                     median=float(np.median(rs2)),
                                     n_sig=int(sum(1 for v in vals2 if v[1] < 0.05))))
    with open('results/length_confound2.json', 'w') as fh:
        json.dump(out, fh, indent=2, default=float)
    print()
    print('[wrote] results/length_confound2.json')


if __name__ == '__main__':
    main()
