"""Robustness of the structural-feature -> a_p / b_p correlations.

Everything here is aimed at killing false positives:
  1. leave-one-protein-out influence (max |r| change, and r on the 21 naturals only)
  2. partial correlation controlling for n_res  (length confound)
  3. designed-vs-natural, with the Rosetta-4 and TrRosetta-3 split apart
  4. permutation null for the top feature, and a "how big is the best r you get
     from 62 random features" calibration
"""
import glob, os
import numpy as np, pandas as pd
from scipy import stats

EV = sorted(glob.glob('eval_results/*.csv'))
SF = pd.read_csv('results/structfeat/struct_features_28.csv')
DESIGNED = {'HEEH_KT_rd6_0746', 'HEEH_KT_rd6_0793', 'HHH_rd1_0142', 'HHH_rd1_0244',
            'r11_1081_TrROS_Hall', 'r12_757_TrROS_Hall', 'r18_3_TrROS_Hall'}
ROSETTA = {'HEEH_KT_rd6_0746', 'HEEH_KT_rd6_0793', 'HHH_rd1_0142', 'HHH_rd1_0244'}
TRROS = {'r11_1081_TrROS_Hall', 'r12_757_TrROS_Hall', 'r18_3_TrROS_Hall'}


def fit_ab(df):
    rows = []
    for p, g in df.groupby('protein'):
        dg = g['deltaG'].values; pdg = g['pred_deltaG'].values
        ddg = g['ddG'].values; pddg = g['pred_ddG'].values
        wt = np.where(ddg == 0.0)[0]
        b_p = float(pdg[wt[0]] - dg[wt[0]]) if len(wt) else np.nan
        a_p, b_ddg = np.polyfit(ddg, pddg, 1)
        rows.append(dict(protein=p, a_p=float(a_p), b_p=b_p, abs_b_p=abs(b_p),
                         ddg_pcc=float(stats.pearsonr(ddg, pddg)[0]),
                         resid_sd=float((pddg - (a_p * ddg + b_ddg)).std())))
    return pd.DataFrame(rows)


def partial_r(x, y, z):
    """corr(x,y | z)"""
    rx = x - np.poly1d(np.polyfit(z, x, 1))(z)
    ry = y - np.poly1d(np.polyfit(z, y, 1))(z)
    return stats.pearsonr(rx, ry)[0]


def main():
    fits = {os.path.basename(e): fit_ab(pd.read_csv(e)) for e in EV}
    FEATS = [c for c in SF.select_dtypes(include=[np.number]).columns
             if SF[c].std() > 0 and SF[c].notna().all() and c not in ('bfac_all_zero',)]
    nfeat = len(FEATS)
    from scipy.stats import t as tdist

    def r_crit(alpha, n):
        tc = tdist.ppf(1 - alpha / 2, n - 2)
        return tc / np.sqrt(tc ** 2 + n - 2)

    print('n features = %d ; Bonferroni alpha = %.3e' % (nfeat, 0.05 / nfeat))
    print('|r| crit  n=28: p05=%.3f  bonf=%.3f | n=21: p05=%.3f  bonf=%.3f'
          % (r_crit(.05, 28), r_crit(.05 / nfeat, 28), r_crit(.05, 21), r_crit(.05 / nfeat, 21)))

    TARGETS = ('a_p', 'b_p', 'abs_b_p', 'ddg_pcc')
    rows = []
    for feat in FEATS:
        fv_all = SF.set_index('protein')[feat]
        for target in TARGETS:
            r_all, r_nat, r_part, loo_min, loo_max = [], [], [], [], []
            for name, f in fits.items():
                cols = ['protein', feat] if feat == 'n_res' else ['protein', feat, 'n_res']
                m = f.merge(SF[cols], on='protein')
                r_all.append(stats.pearsonr(m[feat], m[target])[0])
                mn = m[~m.protein.isin(DESIGNED)]
                r_nat.append(stats.pearsonr(mn[feat], mn[target])[0])
                if feat == 'n_res':
                    r_part.append(np.nan)
                else:
                    r_part.append(partial_r(m[feat].values, m[target].values,
                                            m['n_res'].values.astype(float)))
                loo = [stats.pearsonr(m.drop(i)[feat], m.drop(i)[target])[0]
                       for i in range(len(m))]
                loo_min.append(min(loo)); loo_max.append(max(loo))
            r_all = np.array(r_all); r_nat = np.array(r_nat); r_part = np.array(r_part, dtype=float)
            rows.append(dict(feature=feat, target=target,
                             r_all=r_all.mean(), sign_all=int((np.sign(r_all) == np.sign(r_all.mean())).sum()),
                             r_natural21=r_nat.mean(),
                             sign_nat=int((np.sign(r_nat) == np.sign(r_nat.mean())).sum()),
                             r_partial_nres=np.nanmean(r_part) if not np.all(np.isnan(r_part)) else np.nan,
                             loo_worst=np.mean(loo_min) if r_all.mean() > 0 else np.mean(loo_max),
                             loo_min=np.mean(loo_min), loo_max=np.mean(loo_max)))
    R = pd.DataFrame(rows)
    R.to_csv('results/structfeat/robustness.csv', index=False)

    for t in TARGETS:
        sub = R[R.target == t].copy()
        sub['abs_r'] = sub.r_all.abs()
        sub = sub.sort_values('abs_r', ascending=False).head(12)
        print('\n===== %s : robustness (mean over 10 ckpts) =====' % t)
        print(sub[['feature', 'r_all', 'sign_all', 'r_natural21', 'sign_nat',
                   'r_partial_nres', 'loo_worst']].to_string(index=False,
                                                             float_format=lambda v: '%+.3f' % v))

    # ---------- permutation null: best-of-62 features ----------
    print('\n\n########## PERMUTATION NULL (best |r| over %d features, n=28) ##########' % nfeat)
    rng = np.random.default_rng(0)
    ref = fits['abl_calib_ctrl_repro2_e14.csv'].merge(SF, on='protein')
    X = ref[FEATS].values
    best_null = []
    for _ in range(2000):
        y = rng.permutation(ref['a_p'].values)
        rs = [abs(stats.pearsonr(X[:, k], y)[0]) for k in range(X.shape[1])]
        best_null.append(max(rs))
    best_null = np.array(best_null)
    obs = max(abs(stats.pearsonr(X[:, k], ref['a_p'].values)[0]) for k in range(X.shape[1]))
    print('observed best |r| vs a_p = %.4f' % obs)
    print('null best-of-%d |r|: median %.4f  p95 %.4f  p99 %.4f' %
          (nfeat, np.median(best_null), np.percentile(best_null, 95), np.percentile(best_null, 99)))
    print('family-wise p (obs beaten by null) = %.4f' % ((best_null >= obs).mean()))

    yb = ref['b_p'].values
    best_null_b = []
    for _ in range(2000):
        y = rng.permutation(yb)
        best_null_b.append(max(abs(stats.pearsonr(X[:, k], y)[0]) for k in range(X.shape[1])))
    best_null_b = np.array(best_null_b)
    obsb = max(abs(stats.pearsonr(X[:, k], yb)[0]) for k in range(X.shape[1]))
    print('observed best |r| vs b_p = %.4f ; null p95 %.4f ; family-wise p = %.4f'
          % (obsb, np.percentile(best_null_b, 95), (best_null_b >= obsb).mean()))

    # ---------- designed vs natural, split ----------
    print('\n\n########## GROUP MEANS (mean over 10 ckpts) ##########')
    grp = []
    for name, f in fits.items():
        f = f.copy()
        f['grp'] = np.where(f.protein.isin(ROSETTA), 'Rosetta4',
                            np.where(f.protein.isin(TRROS), 'TrRos3', 'Natural21'))
        for t in ('a_p', 'b_p', 'abs_b_p', 'ddg_pcc', 'resid_sd'):
            for g, sub in f.groupby('grp'):
                grp.append(dict(ckpt=name, target=t, grp=g, val=sub[t].mean()))
    G = pd.DataFrame(grp)
    piv = G.groupby(['target', 'grp']).val.mean().unstack()
    print(piv.to_string(float_format=lambda v: '%+.4f' % v))
    G.to_csv('results/structfeat/group_means.csv', index=False)

    # per-ckpt significance of the designed-vs-natural resid_sd gap
    print('\nresid_sd designed(7) vs natural(21), per checkpoint:')
    for name, f in fits.items():
        d = f[f.protein.isin(DESIGNED)].resid_sd.values
        n = f[~f.protein.isin(DESIGNED)].resid_sd.values
        u, p = stats.mannwhitneyu(d, n)
        print('  %-32s designed %.3f  natural %.3f  MWU p=%.4f' % (name, d.mean(), n.mean(), p))

    # is resid_sd gap just ddG spread? compare the SD of true ddG per group
    print('\nControl: SD of TRUE ddG per group (is the designed gap just a smaller target range?)')
    df0 = pd.read_csv('eval_results/abl_calib_ctrl_repro2_e14.csv')
    sd = df0.groupby('protein').ddG.std()
    sdp = df0.groupby('protein').pred_ddG.std()
    for g, members in (('Rosetta4', ROSETTA), ('TrRos3', TRROS),
                       ('Natural21', set(SF.protein) - DESIGNED)):
        print('  %-10s  SD(true ddG)=%.3f   SD(pred ddG)=%.3f   n=%d'
              % (g, sd[list(members)].mean(), sdp[list(members)].mean(), len(members)))


if __name__ == '__main__':
    main()
