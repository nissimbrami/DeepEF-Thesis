"""Correlate structural features against per-protein a_p / b_p across all 10 eval CSVs."""
import os, glob, json
import numpy as np, pandas as pd
from scipy import stats

EV = ['eval_results/abl_anchor_w0.3_s42_e14.csv',
      'eval_results/abl_anchor_w1.0_s42_e14.csv',
      'eval_results/abl_anchor_w3.0_s42_e13.csv',
      'eval_results/abl_calib_ctrl_repro2_e14.csv',
      'eval_results/abl_p3_slope3.0_s42_e8.csv',
      'eval_results/abl_sigma_seed1_e13.csv',
      'eval_results/abl_sigma_seed2_e10.csv',
      'eval_results/abl_sigma_seed3_e13.csv',
      'eval_results/abl_sigma_seed42_e9.csv',
      'eval_results/abl_sigma_seed4_e14.csv']  # PINNED: the 10 canonical checkpoints.
# An 11th (abl_p3_a0_d0_s0_D1_uemb_seed42_e9.csv) appeared 2026-09-07 17:04 mid-analysis
# from another job; excluded so every number in this report is over the same 10.
SF = pd.read_csv('results/structfeat/struct_features_28.csv')

DESIGNED = {'HEEH_KT_rd6_0746','HEEH_KT_rd6_0793','HHH_rd1_0142','HHH_rd1_0244',
            'r11_1081_TrROS_Hall','r12_757_TrROS_Hall','r18_3_TrROS_Hall'}
ROSETTA = {'HEEH_KT_rd6_0746','HEEH_KT_rd6_0793','HHH_rd1_0142','HHH_rd1_0244'}


def fit_ab(df):
    """CANONICAL project definitions (scripts/calc_bp.py):
       b_p = WT dG error  (pred_deltaG - deltaG on the row with ddG == 0)
       a_p = ddG-side compression slope (pred_ddG ~ a*ddG + b)
    Also keep the dG-side affine fit for reference (a_dg / b_dg)."""
    rows = []
    for p, g in df.groupby('protein'):
        dg = g['deltaG'].values; pdg = g['pred_deltaG'].values
        ddg = g['ddG'].values; pddg = g['pred_ddG'].values
        wt = np.where(ddg == 0.0)[0]
        b_p = float(pdg[wt[0]] - dg[wt[0]]) if len(wt) else np.nan
        a_p, b_ddg = np.polyfit(ddg, pddg, 1)
        a_dg, b_dg = np.polyfit(dg, pdg, 1)
        rows.append(dict(protein=p, a_p=float(a_p), b_p=b_p,
                         a_dg=float(a_dg), b_dg=float(b_dg),
                         abs_b_p=abs(b_p),
                         ddg_pcc=float(stats.pearsonr(ddg, pddg)[0]),
                         dg_pcc=float(stats.pearsonr(dg, pdg)[0]),
                         n_mut=len(g),
                         resid_sd=float((pddg - (a_p*ddg + b_ddg)).std())))
    return pd.DataFrame(rows)


def main():
    print('eval CSVs:', len(EV))
    for e in EV: print('  ', os.path.basename(e))
    fits = {}
    for e in EV:
        df = pd.read_csv(e)
        fits[os.path.basename(e)] = fit_ab(df)
    ref = fits['abl_calib_ctrl_repro2_e14.csv']
    print('SANITY on abl_calib_ctrl_repro2_e14 (brief: std(b_p)=1.5741, median a_p=0.4990, ddG PCC=0.798, corr(a_p,PCC)=+0.5714):')
    print('   std(b_p) pop=%.4f  ddof1=%.4f   median(a_p)=%.4f   median ddG PCC=%.4f   mean ddG PCC=%.4f' %
          (ref['b_p'].std(ddof=0), ref['b_p'].std(ddof=1), ref['a_p'].median(),
           ref['ddg_pcc'].median(), ref['ddg_pcc'].mean()))
    print('   corr(a_p, ddg_pcc)=%.4f   corr(|b_p|, ddg_pcc)=%.4f' %
          (stats.pearsonr(ref['a_p'], ref['ddg_pcc'])[0],
           stats.pearsonr(ref['abs_b_p'], ref['ddg_pcc'])[0]))
    print('Across 10 ckpts: mean std(b_p)=%.4f  mean median(a_p)=%.4f  mean ddG PCC=%.4f' %
          (np.mean([f['b_p'].std(ddof=0) for f in fits.values()]),
           np.mean([f['a_p'].median() for f in fits.values()]),
           np.mean([f['ddg_pcc'].mean() for f in fits.values()])))

    FEATS = [c for c in SF.select_dtypes(include=[np.number]).columns
             if SF[c].std() > 0 and SF[c].notna().all()]
    print('\nn features tested:', len(FEATS))

    results = []
    for feat in FEATS:
        for target in ('b_p', 'abs_b_p', 'a_p', 'ddg_pcc', 'a_dg'):
            rs = []
            for name, f in fits.items():
                m = f.merge(SF[['protein', feat]], on='protein')
                r, p = stats.pearsonr(m[feat], m[target])
                rho, _ = stats.spearmanr(m[feat], m[target])
                rs.append((r, p, rho))
            rs = np.array(rs)
            mr = rs[:, 0].mean()
            sgn = int(np.sum(np.sign(rs[:, 0]) == np.sign(mr)))
            results.append(dict(feature=feat, target=target, mean_r=mr,
                                sd_r=rs[:, 0].std(), min_r=rs[:, 0].min(), max_r=rs[:, 0].max(),
                                sign_consistency=sgn, mean_spearman=rs[:, 2].mean(),
                                mean_p=rs[:, 1].mean(),
                                n_sig05=int((rs[:, 1] < 0.05).sum())))
    R = pd.DataFrame(results)
    R.to_csv('results/structfeat/feature_correlations.csv', index=False)

    nfeat = len(FEATS)
    bonf = 0.05 / nfeat
    # r threshold for n=28 at alpha
    from scipy.stats import t as tdist
    def r_crit(alpha, n=28):
        tc = tdist.ppf(1 - alpha / 2, n - 2)
        return tc / np.sqrt(tc**2 + n - 2)
    print('n=28. |r| crit p<0.05 = %.4f ; Bonferroni alpha=%.2e over %d features -> |r| crit = %.4f'
          % (r_crit(0.05), bonf, nfeat, r_crit(bonf)))

    for target in ('b_p', 'abs_b_p', 'a_p', 'ddg_pcc'):
        sub = R[R.target == target].reindex(
            R[R.target == target].mean_r.abs().sort_values(ascending=False).index)
        print('\n===== TOP 15 vs %s =====' % target)
        print(sub.head(15)[['feature', 'mean_r', 'sd_r', 'min_r', 'max_r',
                            'sign_consistency', 'mean_spearman']].to_string(index=False))

    # ---- designed vs natural ----
    print('\n\n########## DESIGNED vs NATURAL ##########')
    des_rows = []
    for name, f in fits.items():
        f = f.copy()
        f['designed'] = f.protein.isin(DESIGNED)
        for target in ('b_p', 'abs_b_p', 'a_p', 'ddg_pcc', 'resid_sd'):
            d = f.loc[f.designed, target].values
            nn = f.loc[~f.designed, target].values
            u, pu = stats.mannwhitneyu(d, nn)
            tt, pt = stats.ttest_ind(d, nn, equal_var=False)
            des_rows.append(dict(ckpt=name, target=target, mean_designed=d.mean(),
                                 mean_natural=nn.mean(), diff=d.mean() - nn.mean(),
                                 p_mwu=pu, p_t=pt))
    Dd = pd.DataFrame(des_rows)
    Dd.to_csv('results/structfeat/designed_vs_natural.csv', index=False)
    print('n_designed=%d n_natural=%d' % (len(DESIGNED), 28 - len(DESIGNED)))
    print(Dd.groupby('target').agg(mean_designed=('mean_designed', 'mean'),
                                   mean_natural=('mean_natural', 'mean'),
                                   mean_diff=('diff', 'mean'),
                                   n_p_mwu_lt05=('p_mwu', lambda s: int((s < 0.05).sum())),
                                   mean_p_mwu=('p_mwu', 'mean')).to_string())

    # Rosetta-only (the 4 without pLDDT) vs rest -- the missingness bias
    print('\n########## pLDDT-MISSING (4 Rosetta) vs rest ##########')
    mb = []
    for name, f in fits.items():
        f = f.copy(); f['miss'] = f.protein.isin(ROSETTA)
        for target in ('b_p', 'abs_b_p', 'a_p', 'ddg_pcc'):
            d = f.loc[f.miss, target].values; nn = f.loc[~f.miss, target].values
            mb.append(dict(ckpt=name, target=target, mean_missing=d.mean(),
                           mean_present=nn.mean(), diff=d.mean() - nn.mean(),
                           p=stats.mannwhitneyu(d, nn)[1]))
    MB = pd.DataFrame(mb)
    MB.to_csv('results/structfeat/plddt_missing_bias.csv', index=False)
    print(MB.groupby('target').agg(mean_missing=('mean_missing', 'mean'),
                                   mean_present=('mean_present', 'mean'),
                                   mean_diff=('diff', 'mean'),
                                   mean_p=('p', 'mean')).to_string())
    # feature-space bias
    print('\nFeature-space bias of the 4 missing (std units vs other 24):')
    other = SF[~SF.protein.isin(ROSETTA)]
    miss = SF[SF.protein.isin(ROSETTA)]
    bias = []
    for c in FEATS:
        z = (miss[c].mean() - other[c].mean()) / other[c].std(ddof=1)
        bias.append((c, z))
    bias.sort(key=lambda t: -abs(t[1]))
    for c, z in bias[:15]:
        print('   %-28s z=%+.2f  (missing %.3f vs present %.3f)' %
              (c, z, miss[c].mean(), other[c].mean()))

    # mean fit table for report
    mf = None
    for name, f in fits.items():
        g = f[['protein', 'a_p', 'b_p', 'abs_b_p', 'ddg_pcc', 'a_dg']].set_index('protein')
        mf = g if mf is None else mf + g
    mf = (mf / len(fits)).reset_index()
    mf['designed'] = mf.protein.isin(DESIGNED)
    mf['has_plddt'] = ~mf.protein.isin(ROSETTA)
    mf = mf.merge(SF.drop(columns=['seq', 'ss_string']), on='protein')
    mf.to_csv('results/structfeat/master_28.csv', index=False)
    print('\nwrote results/structfeat/master_28.csv', mf.shape)
    print(mf[['protein', 'designed', 'a_p', 'b_p', 'ddg_pcc', 'RCO', 'CO_abs',
              'frac_H', 'frac_E']].sort_values('b_p').to_string(index=False))


if __name__ == '__main__':
    main()
