"""Identifiability of b_p and a_p across the 5 abl_sigma seeds. CPU only, eval CSVs only.

TWO b_p definitions exist in this project and they are NOT the same object:
  b_p_wt   = pred_deltaG[WT] - deltaG[WT]      (calc_bp.py; the CANONICAL b_p, std=1.5741,
                                                a dG-side reference-state error)
  b_ddg    = intercept of polyfit(ddG_true, ddG_pred)  (calib_diag.py 'std_b', ~0.17)
Both are reported. a_p = slope of polyfit(ddG_true, ddG_pred), identical in both scripts.
"""
import json
import numpy as np
import pandas as pd

FILES = {
    'seed1_e13': ('eval_results/abl_sigma_seed1_e13.csv', 13),
    'seed2_e10': ('eval_results/abl_sigma_seed2_e10.csv', 10),
    'seed3_e13': ('eval_results/abl_sigma_seed3_e13.csv', 13),
    'seed4_e14': ('eval_results/abl_sigma_seed4_e14.csv', 14),
    'seed42_e9': ('eval_results/abl_sigma_seed42_e9.csv',  9),
}


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def per_protein_fit(ddg_true, ddg_pred):
    """Verbatim from calib_diag.py."""
    if np.std(ddg_true) < 1e-8:
        return 1.0, float(np.mean(ddg_pred) - np.mean(ddg_true))
    a, b = np.polyfit(ddg_true, ddg_pred, 1)
    return float(a), float(b)


def fit_seed(path):
    df = pd.read_csv(path)
    rec = {}
    pooled_true, pooled_pred, off_wt, off_ddg = [], [], [], []
    for name, g in df.groupby('protein', sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 3:
            continue
        ddg_true = g['deltaG'].values - g['deltaG'].values[0]
        ddg_pred = g['pred_deltaG'].values - g['pred_deltaG'].values[0]
        a, b_ddg = per_protein_fit(ddg_true, ddg_pred)
        # canonical b_p: WT error on absolute dG. Row 0 is WT (verified: ddG[0]==0 in all seeds).
        b_wt = float(g['pred_deltaG'].values[0] - g['deltaG'].values[0])
        rec[name] = {'a_p': a, 'b_ddg': b_ddg, 'b_p_wt': b_wt}
        pooled_true.append(ddg_true)
        pooled_pred.append(ddg_pred)
        off_ddg.append(ddg_pred - b_ddg)
        off_wt.append(ddg_pred - b_wt)
    pt, pp = np.concatenate(pooled_true), np.concatenate(pooled_pred)
    return rec, {
        'pooled_ddG_PCC': _pearson(pt, pp),
        'pooled_offset_removed_PCC_bddg': _pearson(pt, np.concatenate(off_ddg)),
        'pooled_offset_removed_PCC_bwt': _pearson(pt, np.concatenate(off_wt)),
    }


def icc1(M):
    """One-way random-effects ICC(1,1), balanced. M = [n_proteins x k_seeds]."""
    n, k = M.shape
    row_means = M.mean(axis=1)
    ss_b = k * np.sum((row_means - M.mean()) ** 2)
    ss_w = np.sum((M - row_means[:, None]) ** 2)
    msb, msw = ss_b / (n - 1), ss_w / (n * (k - 1))
    icc = (msb - msw) / (msb + (k - 1) * msw)
    var_p = max((msb - msw) / k, 0.0)
    return {'ICC1': float(icc), 'MS_between': float(msb), 'MS_within': float(msw),
            'var_protein_component': float(var_p), 'var_noise_component': float(msw),
            'sd_protein_component': float(np.sqrt(var_p)),
            'sd_noise_component': float(np.sqrt(msw)),
            'n_proteins': int(n), 'k_seeds': int(k)}


def main():
    fits, seed_metrics = {}, {}
    for k, (p, ep) in FILES.items():
        rec, m = fit_seed(p)
        fits[k] = rec
        m['epoch'] = ep
        seed_metrics[k] = m
        print(f'{k}: n={len(rec)} pooled={m["pooled_ddG_PCC"]:.4f} '
              f'offrem(b_ddg)={m["pooled_offset_removed_PCC_bddg"]:.4f}')

    seeds = list(FILES)
    prots = sorted(set.intersection(*[set(fits[s]) for s in seeds]))
    print(f'\ncommon proteins across all 5 seeds: {len(prots)}')

    out = {'seeds': seed_metrics, 'n_proteins': len(prots), 'proteins': prots,
           'note_two_bp_definitions': (
               'b_p_wt = pred_dG[WT]-dG[WT] (calc_bp.py, canonical, std~1.5741); '
               'b_ddg = polyfit intercept on ddG (calib_diag.py std_b, ~0.17)')}

    mats = {}
    for key in ('b_p_wt', 'b_ddg', 'a_p'):
        M = np.array([[fits[s][p][key] for s in seeds] for p in prots])
        mats[key] = M
        r = icc1(M)
        r['per_seed_across_protein_std'] = {s: float(np.std(M[:, i]))
                                            for i, s in enumerate(seeds)}
        r['mean_per_seed_across_protein_std'] = float(
            np.mean([np.std(M[:, i]) for i in range(M.shape[1])]))
        r['mean_within_protein_across_seed_std'] = float(np.mean(np.std(M, axis=1, ddof=1)))
        from itertools import combinations
        pw = [_pearson(M[:, i], M[:, j]) for i, j in combinations(range(len(seeds)), 2)]
        r['pairwise_seed_PCC_mean'] = float(np.mean(pw))
        r['pairwise_seed_PCC_min'] = float(np.min(pw))
        r['pairwise_seed_PCC_max'] = float(np.max(pw))
        out[key] = r
        print(f'\n=== {key} ===')
        print('  per-seed across-protein std: ' +
              ', '.join(f'{s}={v:.4f}' for s, v in r['per_seed_across_protein_std'].items()))
        print(f'  mean across-protein std      = {r["mean_per_seed_across_protein_std"]:.4f}')
        print(f'  mean within-protein seed std = {r["mean_within_protein_across_seed_std"]:.4f}')
        print(f'  sd_protein={r["sd_protein_component"]:.4f}  '
              f'sd_noise={r["sd_noise_component"]:.4f}')
        print(f'  ICC(1,1) = {r["ICC1"]:.4f}')
        print(f'  pairwise across-seed PCC mean={r["pairwise_seed_PCC_mean"]:.4f} '
              f'min={r["pairwise_seed_PCC_min"]:.4f}')

    # ---- Consequence: variance bound and achievable pooled PCC ------------------
    STD_B_REF = 1.5741
    icc_b = out['b_p_wt']['ICC1']
    out['consequence'] = {
        'std_b_reference': STD_B_REF,
        'total_var_reference': STD_B_REF ** 2,
        'ICC_b_p_wt': icc_b,
        'max_removable_var': icc_b * STD_B_REF ** 2,
        'irreducible_seed_var': (1 - icc_b) * STD_B_REF ** 2,
        'irreducible_seed_sd': float(np.sqrt((1 - icc_b) * STD_B_REF ** 2)),
    }
    print(f'\n=== CONSEQUENCE (b_p_wt, canonical) ===')
    print(f'  ICC = {icc_b:.4f};  total var = {STD_B_REF**2:.4f}')
    print(f'  max removable by any perfect feature corrector = '
          f'{icc_b*STD_B_REF**2:.4f} ({100*icc_b:.1f}%)')
    print(f'  irreducible seed-noise var = {(1-icc_b)*STD_B_REF**2:.4f} '
          f'(sd {np.sqrt((1-icc_b))*STD_B_REF:.4f})')

    with open('results/identifiability_raw.json', 'w') as fh:
        json.dump({'stats': out, 'seeds': seeds, 'proteins': prots,
                   'matrices': {k: v.tolist() for k, v in mats.items()}}, fh, indent=2)
    print('\nwrote results/identifiability_raw.json')


if __name__ == '__main__':
    main()
