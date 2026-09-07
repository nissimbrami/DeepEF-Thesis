"""Identifiability of b_p / a_p across 5 abl_sigma seeds. CPU only, eval CSVs only.
Conventions copied verbatim from cluster_run/code/calib_diag.py:
  groupby('protein', sort=False); row 0 of group = WT; ddG = deltaG - deltaG[0];
  per_protein_fit -> np.polyfit(ddg_true, ddg_pred, 1) -> (a, b); stats on len(g)>=3.
"""
import json
import numpy as np
import pandas as pd

FILES = {
    'seed1_e13':  'eval_results/abl_sigma_seed1_e13.csv',
    'seed2_e10':  'eval_results/abl_sigma_seed2_e10.csv',
    'seed3_e13':  'eval_results/abl_sigma_seed3_e13.csv',
    'seed4_e14':  'eval_results/abl_sigma_seed4_e14.csv',
    'seed42_e9':  'eval_results/abl_sigma_seed42_e9.csv',
}
EPOCH = {'seed1_e13': 13, 'seed2_e10': 10, 'seed3_e13': 13, 'seed4_e14': 14, 'seed42_e9': 9}


def per_protein_fit(ddg_true, ddg_pred):
    """Verbatim from calib_diag.py."""
    if np.std(ddg_true) < 1e-8:
        return 1.0, float(np.mean(ddg_pred) - np.mean(ddg_true))
    a, b = np.polyfit(ddg_true, ddg_pred, 1)
    return float(a), float(b)


def _pearson(x, y):
    if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def fit_seed(path):
    """Return dict protein -> (a, b), plus pooled/offset-removed PCC for that seed."""
    df = pd.read_csv(path)
    ab = {}
    pooled_true, pooled_pred, off_pred = [], [], []
    for name, g in df.groupby('protein', sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        ddg_true = g['deltaG'].values - g['deltaG'].values[0]
        ddg_pred = g['pred_deltaG'].values - g['pred_deltaG'].values[0]
        a, b = per_protein_fit(ddg_true, ddg_pred)
        pooled_true.append(ddg_true)
        pooled_pred.append(ddg_pred)
        off_pred.append(ddg_pred - b)
        if len(g) >= 3:
            ab[name] = (a, b)
    pt = np.concatenate(pooled_true)
    pp = np.concatenate(pooled_pred)
    po = np.concatenate(off_pred)
    return ab, _pearson(pt, pp), _pearson(pt, po)


def icc1(M):
    """One-way random-effects ICC(1,1) on matrix M [n_proteins x k_seeds].
    Balanced design. MSB = between-protein mean square, MSW = within-protein mean square.
    ICC = (MSB - MSW) / (MSB + (k-1)*MSW).  Also returns the variance components."""
    n, k = M.shape
    grand = M.mean()
    row_means = M.mean(axis=1)
    ss_between = k * np.sum((row_means - grand) ** 2)
    ss_within = np.sum((M - row_means[:, None]) ** 2)
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))
    icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
    var_protein = (ms_between - ms_within) / k      # sigma^2_between (true protein variance)
    var_noise = ms_within                            # sigma^2_within (seed/epoch noise)
    return {
        'ICC1': float(icc),
        'MS_between': float(ms_between),
        'MS_within': float(ms_within),
        'var_protein_component': float(var_protein),
        'var_noise_component': float(var_noise),
        'n_proteins': int(n),
        'k_seeds': int(k),
    }


def main():
    fits, seed_pcc = {}, {}
    for k, p in FILES.items():
        ab, pooled, offrem = fit_seed(p)
        fits[k] = ab
        seed_pcc[k] = {'pooled_ddG_PCC': pooled, 'pooled_offset_removed_PCC': offrem,
                       'epoch': EPOCH[k]}
        print(f'{k}: n_prot={len(ab)} pooled={pooled:.4f} offset_removed={offrem:.4f}')

    seeds = list(FILES)
    proteins = sorted(set.intersection(*[set(fits[s]) for s in seeds]))
    print(f'\nproteins common to all {len(seeds)} seeds: {len(proteins)}')

    B = np.array([[fits[s][p][1] for s in seeds] for p in proteins])  # [n_prot x k]
    A = np.array([[fits[s][p][0] for s in seeds] for p in proteins])

    out = {'seeds': seed_pcc, 'n_proteins': len(proteins), 'proteins': proteins}

    for label, M in (('b_p', B), ('a_p', A)):
        r = icc1(M)
        total_obs_var = float(np.var(M, ddof=0))
        # per-seed across-protein std (the "std(b_p)" style number, one per seed)
        per_seed_std = {s: float(np.std(M[:, i], ddof=0)) for i, s in enumerate(seeds)}
        # mean across-seed (within-protein) std
        within_std = float(np.mean(np.std(M, axis=1, ddof=1)))
        r.update({
            'total_observed_variance': total_obs_var,
            'per_seed_across_protein_std': per_seed_std,
            'mean_within_protein_across_seed_std': within_std,
            'sd_protein_component': float(np.sqrt(max(r['var_protein_component'], 0.0))),
            'sd_noise_component': float(np.sqrt(max(r['var_noise_component'], 0.0))),
        })
        out[label] = r
        print(f'\n=== {label} ===')
        print(f'  per-seed across-protein std: ' +
              ', '.join(f'{s}={v:.4f}' for s, v in per_seed_std.items()))
        print(f'  mean within-protein across-seed std : {within_std:.4f}')
        print(f'  MS_between={r["MS_between"]:.4f}  MS_within={r["MS_within"]:.4f}')
        print(f'  var_protein={r["var_protein_component"]:.4f} (sd {r["sd_protein_component"]:.4f})')
        print(f'  var_noise  ={r["var_noise_component"]:.4f} (sd {r["sd_noise_component"]:.4f})')
        print(f'  ICC(1,1) = {r["ICC1"]:.4f}   <-- protein-attributable fraction')

    # Spearman/Pearson agreement of b_p rank across seed pairs (a second, assumption-light view)
    from itertools import combinations
    def rank(v):
        o = np.argsort(np.argsort(v))
        return o.astype(float)
    for label, M in (('b_p', B), ('a_p', A)):
        pear, spear = [], []
        for i, j in combinations(range(len(seeds)), 2):
            pear.append(_pearson(M[:, i], M[:, j]))
            spear.append(_pearson(rank(M[:, i]), rank(M[:, j])))
        out[label]['pairwise_seed_pearson_mean'] = float(np.mean(pear))
        out[label]['pairwise_seed_pearson_min'] = float(np.min(pear))
        out[label]['pairwise_seed_pearson_max'] = float(np.max(pear))
        out[label]['pairwise_seed_spearman_mean'] = float(np.mean(spear))
        print(f'\n{label} pairwise across-seed PCC: mean={np.mean(pear):.4f} '
              f'min={np.min(pear):.4f} max={np.max(pear):.4f} '
              f'spearman_mean={np.mean(spear):.4f}')

    np.save('/tmp/icc_B.npy', B)
    np.save('/tmp/icc_A.npy', A)
    with open('/tmp/icc_raw.json', 'w') as fh:
        json.dump({'proteins': proteins, 'seeds': seeds,
                   'B': B.tolist(), 'A': A.tolist(), 'stats': out}, fh, indent=2)
    print('\nwrote /tmp/icc_raw.json')


if __name__ == '__main__':
    main()
