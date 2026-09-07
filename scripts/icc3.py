"""Epoch-confound controls + achievable-PCC bound for the identifiability analysis."""
import json
import numpy as np
from itertools import combinations

D = json.load(open('results/identifiability_raw.json'))
seeds = D['seeds']
prots = D['proteins']
EPOCH = {'seed1_e13': 13, 'seed2_e10': 10, 'seed3_e13': 13, 'seed4_e14': 14, 'seed42_e9': 9}


def icc1(M):
    n, k = M.shape
    rm = M.mean(axis=1)
    msb = k * np.sum((rm - M.mean()) ** 2) / (n - 1)
    msw = np.sum((M - rm[:, None]) ** 2) / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw), msb, msw


def icc2_twoway(M):
    """Two-way random ICC(C,1) = consistency, removes the seed MAIN effect (a per-seed
    additive shift, which is exactly what an epoch difference mostly looks like).
    Higher than ICC(1,1) by construction; the gap is the epoch/seed column effect."""
    n, k = M.shape
    gm = M.mean()
    rm, cm = M.mean(axis=1), M.mean(axis=0)
    ss_r = k * np.sum((rm - gm) ** 2)
    ss_c = n * np.sum((cm - gm) ** 2)
    ss_t = np.sum((M - gm) ** 2)
    ss_e = ss_t - ss_r - ss_c
    msr = ss_r / (n - 1)
    mse = ss_e / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse), msr, mse, ss_c / (k - 1)


out = {}
for key in ('b_p_wt', 'b_ddg', 'a_p'):
    M = np.array(D['matrices'][key])
    i1, msb, msw = icc1(M)
    i2, msr, mse, msc = icc2_twoway(M)
    # epoch-matched pair: seed1_e13 and seed3_e13 are the SAME epoch -> pure seed contrast
    names = D['seeds'] if isinstance(D['seeds'], list) else list(EPOCH)
    idx = {s: i for i, s in enumerate(names)}
    pair = M[:, [idx['seed1_e13'], idx['seed3_e13']]]
    i_pair, _, _ = icc1(pair)
    # column (seed/epoch) means: how much of the spread is a global per-seed shift
    col_mean = M.mean(axis=0)
    # correlation of the per-seed column mean with epoch number
    eps = np.array([EPOCH[s] for s in names], float)
    r_epoch = float(np.corrcoef(col_mean, eps)[0, 1])
    out[key] = {
        'ICC1_oneway_LOWER_bound': float(i1),
        'ICC_C1_twoway_seedshift_removed': float(i2),
        'ICC1_epoch_matched_pair_e13_only': float(i_pair),
        'MS_column_seed_effect': float(msc),
        'MS_error_after_removing_seed_shift': float(mse),
        'per_seed_column_mean': {s: float(col_mean[i]) for s, i in idx.items()},
        'corr_columnmean_vs_epoch': r_epoch,
    }
    print(f'=== {key} ===')
    print(f'  ICC(1,1)  one-way            = {i1:.4f}   (LOWER bound: epoch counted as noise)')
    print(f'  ICC(C,1)  two-way consistency= {i2:.4f}   (per-seed additive shift removed)')
    print(f'  ICC(1,1) on e13 pair only    = {i_pair:.4f}   (same epoch -> pure seed contrast)')
    print(f'  per-seed column means: ' +
          ', '.join(f'{s}={col_mean[i]:.3f}' for s, i in idx.items()))
    print(f'  corr(column mean, epoch) = {r_epoch:+.4f}\n')

# ---------------- achievable pooled-PCC bound from offset correction --------------
# Empirical anchor: measured pooled PCC and offset-removal ORACLE pooled PCC per seed.
sm = D['stats']['seeds']
raw = np.mean([sm[s]['pooled_ddG_PCC'] for s in sm])
orc = np.mean([sm[s]['pooled_offset_removed_PCC_bddg'] for s in sm])
icc_b = D['stats']['b_p_wt']['ICC1']
icc_bd = D['stats']['b_ddg']['ICC1']

# A feature corrector can only capture the protein-attributable share of the offset.
# Model the oracle gain in *variance-explained* terms: r^2 goes raw^2 -> orc^2 by removing
# offset variance. A corrector that removes only a fraction f of that offset variance
# recovers f of the r^2 gain (offset removal is a variance subtraction in the denominator).
def bound(f):
    return float(np.sqrt(raw ** 2 + f * (orc ** 2 - raw ** 2)))

out['achievable'] = {
    'mean_raw_pooled_PCC': float(raw),
    'mean_offset_oracle_pooled_PCC': float(orc),
    'ICC_b_p_wt': float(icc_b),
    'ICC_b_ddg': float(icc_bd),
    'bound_with_ICC_b_p_wt': bound(icc_b),
    'bound_with_ICC_b_ddg': bound(icc_bd),
    'oracle_gain_PCC': float(orc - raw),
    'identifiable_gain_PCC_b_ddg': float(bound(icc_bd) - raw),
    'lost_to_seed_noise_PCC': float(orc - bound(icc_bd)),
}
print('=== ACHIEVABLE POOLED PCC BOUND ===')
print(f'  mean raw pooled PCC (5 seeds)          = {raw:.4f}')
print(f'  mean offset-removal ORACLE pooled PCC  = {orc:.4f}   (gain {orc-raw:+.4f})')
print(f'  ceiling if corrector captures ICC(b_ddg)={icc_bd:.4f}: {bound(icc_bd):.4f} '
      f'(gain {bound(icc_bd)-raw:+.4f})')
print(f'  ceiling if corrector captures ICC(b_wt) ={icc_b:.4f}: {bound(icc_b):.4f}')
print(f'  PCC lost to irreducible seed noise      = {orc-bound(icc_bd):.4f}')

json.dump(out, open('results/identifiability_epoch.json', 'w'), indent=2)
print('\nwrote results/identifiability_epoch.json')
