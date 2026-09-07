# -*- coding: utf-8 -*-
"""
LINE 6: is the mean_rel_SASA -> a_p finding actually LENGTH?

Reuses catalogue_vs_bp.recover_bp and .structural_features (same SASA code path)
so the structural numbers are identical to CHECKPOINT 8's.

Adds, per protein:
  n_mut       : number of rows (mutations+WT) the slope was fitted on
  sd_ddg_true : SD of the true ddG values -- the regressor spread.
                For a fitted slope a_p = cov(x,y)/var(x), BOTH of these are
                statistical confounds independent of any structure.

Runs on ALL 10 eval CSVs. Partial correlations use df = n-3.
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, 'scripts')
from catalogue_vs_bp import recover_bp, structural_features

PDB_DIR = 'data/Processed_K50_dG_datasets/AlphaFold_model_PDBs'
PLDDT = 'data/Processed_K50_dG_datasets/plddt.csv'


def stat_covariates(csv_path):
    """n_mut and true-ddG spread per protein -- the statistical confounds."""
    df = pd.read_csv(csv_path)
    rows = []
    for name, g in df.groupby('protein', sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 3:
            continue
        dt = g['deltaG'].values - g['deltaG'].values[0]
        rows.append(dict(protein=str(name), n_mut=int(len(g)),
                         sd_ddg_true=float(np.std(dt, ddof=1)),
                         range_ddg_true=float(np.max(dt) - np.min(dt)),
                         mean_abs_ddg_true=float(np.mean(np.abs(dt)))))
    return pd.DataFrame(rows)


def partial(x, y, z):
    """corr(x,y | z) with p-value at df = n-3."""
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    rxy = stats.pearsonr(x, y)[0]
    rxz = stats.pearsonr(x, z)[0]
    ryz = stats.pearsonr(y, z)[0]
    den = np.sqrt(max(1e-12, (1 - rxz**2) * (1 - ryz**2)))
    r = (rxy - rxz * ryz) / den
    n = len(x); df = n - 3
    t = abs(r) * np.sqrt(df / max(1e-12, 1 - r*r))
    p = 2 * stats.t.sf(t, df)
    return float(r), float(p), float(rxy)


def main():
    csvs = sorted(glob.glob('eval_results/*.csv'))
    print('eval CSVs found: %d' % len(csvs))
    for c in csvs:
        print('  ', c)
    print()

    bp0 = recover_bp(csvs[0])
    names = list(bp0['protein'])
    feat = structural_features(names, PDB_DIR, PLDDT)
    print('structural features computed for %d proteins' % len(feat))

    # ---- 2. the confound itself ----
    sub = feat[['length', 'mean_rel_SASA']].dropna()
    r_LS, p_LS = stats.pearsonr(sub['length'], sub['mean_rel_SASA'])
    rho_LS, prho_LS = stats.spearmanr(sub['length'], sub['mean_rel_SASA'])
    n28 = len(sub)
    tcrit = stats.t.ppf(0.975, n28 - 2)
    rcrit = float(tcrit / np.sqrt(tcrit**2 + n28 - 2))
    print()
    print('=== CONFOUND: corr(length, mean_rel_SASA), n=%d ===' % n28)
    print('  pearson  r = %+.4f  p=%.3g' % (r_LS, p_LS))
    print('  spearman r = %+.4f  p=%.3g' % (rho_LS, prho_LS))
    print('  length range: %d - %d aa' % (feat['length'].min(), feat['length'].max()))
    print('  |r| < %.3f is indistinguishable from 0 at p=0.05 (n=%d)' % (rcrit, n28))

    recs = []
    for csv in csvs:
        bp = recover_bp(csv)
        sc = stat_covariates(csv)
        m = bp.merge(feat, on='protein', how='left').merge(sc, on='protein', how='left')
        m = m.dropna(subset=['a_p', 'mean_rel_SASA', 'length'])
        n = len(m)
        A = m['a_p'].values
        S = m['mean_rel_SASA'].values
        L = m['length'].values.astype(float)
        NM = m['n_mut'].values.astype(float)
        SD = m['sd_ddg_true'].values

        d = dict(csv=os.path.basename(csv), n=n)
        d['r_a_SASA'] = float(stats.pearsonr(S, A)[0])
        d['p_a_SASA'] = float(stats.pearsonr(S, A)[1])
        d['r_a_len'] = float(stats.pearsonr(L, A)[0])
        d['p_a_len'] = float(stats.pearsonr(L, A)[1])
        d['r_a_nmut'] = float(stats.pearsonr(NM, A)[0])
        d['p_a_nmut'] = float(stats.pearsonr(NM, A)[1])
        d['r_a_sdddg'] = float(stats.pearsonr(SD, A)[0])
        d['p_a_sdddg'] = float(stats.pearsonr(SD, A)[1])
        r, p, _ = partial(S, A, L); d['pr_SASA_ctrl_len'] = r; d['pp_SASA_ctrl_len'] = p
        r, p, _ = partial(L, A, S); d['pr_len_ctrl_SASA'] = r; d['pp_len_ctrl_SASA'] = p
        r, p, _ = partial(S, A, NM); d['pr_SASA_ctrl_nmut'] = r; d['pp_SASA_ctrl_nmut'] = p
        r, p, _ = partial(S, A, SD); d['pr_SASA_ctrl_sdddg'] = r; d['pp_SASA_ctrl_sdddg'] = p
        d['rho_a_SASA'] = float(stats.spearmanr(S, A)[0])
        d['rho_a_len'] = float(stats.spearmanr(L, A)[0])
        recs.append(d)
        m.to_csv('results/_lc_merged_%s' % os.path.basename(csv), index=False)

    R = pd.DataFrame(recs)
    pd.set_option('display.width', 250)
    print()
    print('=== RAW CORRELATIONS WITH a_p, per checkpoint (n=%d each) ===' % recs[0]['n'])
    print(R[['csv','n','r_a_SASA','r_a_len','r_a_nmut','r_a_sdddg']].to_string(index=False))
    print()
    print('=== PARTIAL CORRELATIONS WITH a_p (df=n-3) ===')
    print(R[['csv','pr_SASA_ctrl_len','pp_SASA_ctrl_len','pr_len_ctrl_SASA','pp_len_ctrl_SASA']].to_string(index=False))
    print()
    print('=== SASA controlling for the STATISTICAL confounds ===')
    print(R[['csv','pr_SASA_ctrl_nmut','pp_SASA_ctrl_nmut','pr_SASA_ctrl_sdddg','pp_SASA_ctrl_sdddg']].to_string(index=False))
    print()
    print('=== SUMMARY over %d checkpoints ===' % len(R))
    for c in ['r_a_SASA','r_a_len','r_a_nmut','r_a_sdddg',
              'pr_SASA_ctrl_len','pr_len_ctrl_SASA','pr_SASA_ctrl_nmut','pr_SASA_ctrl_sdddg',
              'rho_a_SASA','rho_a_len']:
        v = R[c].values
        if c.startswith('r_a_'):
            pc = 'p_a_' + c[4:]
        elif c.startswith('pr_'):
            pc = 'pp_' + c[3:]
        else:
            pc = None
        nsig = int((R[pc] < 0.05).sum()) if pc and pc in R.columns else -1
        print('  %-22s mean=%+.4f  median=%+.4f  min=%+.4f  max=%+.4f  sign-consistent=%2d/%d  p<.05 in %s'
              % (c, v.mean(), np.median(v), v.min(), v.max(),
                 max((v > 0).sum(), (v < 0).sum()), len(v),
                 ('%d/%d' % (nsig, len(v))) if nsig >= 0 else 'n/a'))

    print()
    print('=== STRUCTURE TABLE (checkpoint-independent) ===')
    cols = ['protein','length','mean_rel_SASA','frac_buried_rel_lt_0.25','plddt_mean',
            'total_SASA','SASA_per_residue','mean_hydropathy_KD']
    cols = [c for c in cols if c in feat.columns]
    print(feat[cols].sort_values('length').to_string(index=False))

    out = dict(n_checkpoints=len(csvs), csvs=[os.path.basename(c) for c in csvs],
               n_proteins=int(n28), r_crit_p05=rcrit,
               corr_length_sasa=dict(pearson_r=float(r_LS), pearson_p=float(p_LS),
                                     spearman_r=float(rho_LS), spearman_p=float(prho_LS)),
               per_checkpoint=recs,
               structure=feat.to_dict(orient='records'))
    with open('results/length_confound.json','w') as fh:
        json.dump(out, fh, indent=2, default=float)
    print()
    print('[wrote] results/length_confound.json')


if __name__ == '__main__':
    main()
