"""pLDDT analysis: whole-protein aggregates vs b_p/a_p (n=24), and the
per-MUTATION question nobody has asked - does per-SITE pLDDT predict per-mutation error?

The per-mutation part is the point: 28,314 rows, not 28.
"""
import glob, os, re
import numpy as np, pandas as pd
from scipy import stats

EV = sorted(glob.glob('eval_results/*.csv'))
SF = pd.read_csv('results/structfeat/struct_features_28.csv')
PL = pd.read_csv('data/Processed_K50_dG_datasets/plddt.csv')
ROSETTA = {'HEEH_KT_rd6_0746', 'HEEH_KT_rd6_0793', 'HHH_rd1_0142', 'HHH_rd1_0244'}
DESIGNED = ROSETTA | {'r11_1081_TrROS_Hall', 'r12_757_TrROS_Hall', 'r18_3_TrROS_Hall'}


def fit_ab(df):
    rows = []
    for p, g in df.groupby('protein'):
        dg = g['deltaG'].values; pdg = g['pred_deltaG'].values
        ddg = g['ddG'].values; pddg = g['pred_ddG'].values
        wt = np.where(ddg == 0.0)[0]
        b_p = float(pdg[wt[0]] - dg[wt[0]]) if len(wt) else np.nan
        a_p, b_ddg = np.polyfit(ddg, pddg, 1)
        rows.append(dict(protein=p, a_p=float(a_p), b_p=b_p, abs_b_p=abs(b_p),
                         ddg_pcc=float(stats.pearsonr(ddg, pddg)[0])))
    return pd.DataFrame(rows)


def main():
    print('plddt.csv:', PL.shape, PL.resi_convention.unique())
    prots = sorted(SF.protein)
    have = sorted(set(PL.protein) & set(prots))
    print('proteins with pLDDT: %d / %d' % (len(have), len(prots)))

    # ---- whole-protein pLDDT aggregates ----
    agg = PL[PL.protein.isin(prots)].groupby('protein').plddt.agg(
        plddt_mean='mean', plddt_min='min', plddt_p10=lambda s: np.percentile(s, 10),
        plddt_std='std', plddt_frac_lt70=lambda s: (s < 70).mean(),
        plddt_frac_lt90=lambda s: (s < 90).mean(), plddt_n='count').reset_index()
    agg.to_csv('results/structfeat/plddt_agg.csv', index=False)
    print(agg.to_string(index=False))

    # length check: does plddt row count match n_res?
    chk = agg.merge(SF[['protein', 'n_res']], on='protein')
    print('\npLDDT length vs structure n_res mismatch:',
          (chk.plddt_n != chk.n_res).sum(), 'of', len(chk))
    print(chk[chk.plddt_n != chk.n_res][['protein', 'plddt_n', 'n_res']].to_string(index=False))

    fits = {os.path.basename(e): fit_ab(pd.read_csv(e)) for e in EV}
    PCOLS = [c for c in agg.columns if c.startswith('plddt')]
    from scipy.stats import t as tdist

    def r_crit(alpha, n):
        tc = tdist.ppf(1 - alpha / 2, n - 2)
        return tc / np.sqrt(tc ** 2 + n - 2)
    print('\nn=24 -> |r| crit p05 = %.3f ; Bonferroni over %d pLDDT feats -> %.3f'
          % (r_crit(.05, 24), len(PCOLS), r_crit(.05 / len(PCOLS), 24)))

    print('\n===== whole-protein pLDDT vs targets (n=24, mean over 10 ckpts) =====')
    out = []
    for c in PCOLS:
        for t in ('a_p', 'b_p', 'abs_b_p', 'ddg_pcc'):
            rs = []
            for name, f in fits.items():
                m = f.merge(agg[['protein', c]], on='protein')
                rs.append(stats.pearsonr(m[c], m[t])[0])
            rs = np.array(rs)
            out.append(dict(feature=c, target=t, mean_r=rs.mean(),
                            sign=int((np.sign(rs) == np.sign(rs.mean())).sum()),
                            min_r=rs.min(), max_r=rs.max()))
    O = pd.DataFrame(out)
    O.to_csv('results/structfeat/plddt_corr.csv', index=False)
    for t in ('a_p', 'b_p', 'abs_b_p', 'ddg_pcc'):
        s = O[O.target == t].sort_values('mean_r', key=abs, ascending=False)
        print('\n-- %s --' % t)
        print(s[['feature', 'mean_r', 'sign', 'min_r', 'max_r']].to_string(
            index=False, float_format=lambda v: '%+.3f' % v))

    # ---------- PER-MUTATION: does site pLDDT predict per-mutation error? ----------
    print('\n\n########## PER-MUTATION pLDDT (28,314 rows, not 28) ##########')
    df = pd.read_csv('eval_results/abl_calib_ctrl_repro2_e14.csv')
    print('columns available:', list(df.columns))
    print('NOTE: eval CSVs carry NO mutation-position column, so per-site pLDDT')
    print('cannot be joined to a mutation row. Checking for any position info...')
    for c in df.columns:
        print('  %-14s dtype=%s  nunique=%d' % (c, df[c].dtype, df[c].nunique()))
    # can we recover position from row order within a protein?
    g = df[df.protein == have[0]]
    print('\nrows for %s: %d ; structure n_res=%d ; 19*n_res=%d'
          % (have[0], len(g), int(SF[SF.protein == have[0]].n_res.iloc[0]),
             19 * int(SF[SF.protein == have[0]].n_res.iloc[0])))
    cnt = df.groupby('protein').size().reset_index(name='n_rows')
    cnt = cnt.merge(SF[['protein', 'n_res']], on='protein')
    cnt['ratio'] = cnt.n_rows / cnt.n_res
    print(cnt.to_string(index=False))


if __name__ == '__main__':
    main()
