"""GUARD against the signature failure mode: prove the ICC actually reads the per-protein
identity of b_p, and is not a number the arithmetic would emit regardless of the data."""
import json
import numpy as np

D = json.load(open('results/identifiability_raw.json'))
names = D['seeds']

def icc1(M):
    n, k = M.shape
    rm = M.mean(axis=1)
    msb = k * np.sum((rm - M.mean()) ** 2) / (n - 1)
    msw = np.sum((M - rm[:, None]) ** 2) / (n * (k - 1))
    return float((msb - msw) / (msb + (k - 1) * msw))

rng = np.random.RandomState(0)
res = {}
for key in ('b_p_wt', 'a_p'):
    M = np.array(D['matrices'][key])
    real = icc1(M)
    # NEGATIVE CONTROL 1: destroy the protein identity by shuffling proteins independently
    # within each seed column. Protein-attributable variance must vanish -> ICC ~ 0.
    perm = []
    for _ in range(200):
        S = np.column_stack([rng.permutation(M[:, j]) for j in range(M.shape[1])])
        perm.append(icc1(S))
    perm = np.array(perm)
    # NEGATIVE CONTROL 2: replace values with pure noise of the same marginal scale -> ICC ~ 0
    noise = np.array([icc1(rng.normal(M.mean(), M.std(), size=M.shape)) for _ in range(200)])
    # POSITIVE CONTROL: perfectly reproducible object (same column repeated) -> ICC = 1
    pos = icc1(np.column_stack([M[:, 0]] * M.shape[1]))
    # LEAVE-ONE-SEED-OUT stability of the real ICC
    loo = {names[j]: icc1(np.delete(M, j, axis=1)) for j in range(M.shape[1])}
    # BOOTSTRAP CI over proteins
    n = M.shape[0]
    boot = np.array([icc1(M[rng.randint(0, n, n)]) for _ in range(2000)])
    res[key] = {
        'ICC_real': real,
        'shuffled_protein_identity_mean': float(perm.mean()),
        'shuffled_protein_identity_p95': float(np.percentile(perm, 95)),
        'pure_noise_mean': float(noise.mean()),
        'positive_control_identical_columns': pos,
        'leave_one_seed_out': loo,
        'bootstrap_CI95': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        'real_exceeds_shuffle_p95': bool(real > np.percentile(perm, 95)),
    }
    print(f'=== {key} ===')
    print(f'  REAL ICC                         = {real:.4f}')
    print(f'  shuffled protein identity  mean  = {perm.mean():+.4f} (p95 {np.percentile(perm,95):+.4f})')
    print(f'  pure-noise matrix          mean  = {noise.mean():+.4f}')
    print(f'  positive ctrl (identical cols)   = {pos:.4f}')
    print(f'  leave-one-seed-out: ' + ', '.join(f'{k}={v:.4f}' for k, v in loo.items()))
    print(f'  bootstrap 95% CI over proteins   = [{np.percentile(boot,2.5):.4f}, {np.percentile(boot,97.5):.4f}]')
    print(f'  REAL > shuffle p95 ?             = {real > np.percentile(perm,95)}\n')

ok = all(r['real_exceeds_shuffle_p95'] and r['shuffled_protein_identity_mean'] < 0.15
         and r['positive_control_identical_columns'] > 0.999 for r in res.values())
print('GUARD:', 'PASS - ICC genuinely reads per-protein identity' if ok else 'FAIL')
res['guard_pass'] = bool(ok)
json.dump(res, open('results/identifiability_guard.json', 'w'), indent=2)
print('wrote results/identifiability_guard.json')
