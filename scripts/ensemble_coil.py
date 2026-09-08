#!/usr/bin/env python3
"""K20 / T-E1 -- the unfolded state as an ENSEMBLE instead of one deterministic map.

WHY. Our unfolded reference is a single analytic distance map d(i,j)=b*|i-j|^nu. The physical
unfolded state is a thermodynamic ENSEMBLE. E_u should be an average over conformations.

WHY IT IS CHEAP. _flory_unfolded_graph never reads one_hot -- the coil map depends only on |i-j|
and b. So the k sampled conformations are IDENTICAL for every variant of a protein and are
generated ONCE per protein, not once per mutant.

SAMPLE COORDINATES, NEVER DISTANCES. Drawing each d(i,j) independently yields a matrix that
corresponds to no 3-D structure and violates the triangle inequality. We generate a real chain
with the correct step statistics and compute its distance matrix, so geometric consistency is free.

TWO REDUCTIONS, because their difference IS the physics:
    mean field   E_u = mean_k E_k
    free energy  E_u = -log sum_k exp(-E_k)
The gap between them is the conformational entropy -- the term that dominates the unfolded state.

SCORED ON std(b_p), NOT MAE. Measured: b_p is 96.3% one global constant and MAE === |global|
because n_under = 28/28, so MAE is blind to the dispersion the thesis targets.
FALSIFIER, STATED BEFORE RUNNING: if k=8 does not push std(b_p) below the no-coil baseline
0.9972, the reference-state programme is finished.
"""
import argparse, json, os, sys
import numpy as np

def sample_coil(N, b, nu, rng):
    """A self-avoiding-ish chain: steps of length b with a persistence that reproduces the
    target exponent. Returns [N,3] coordinates, so every distance is automatically realisable."""
    # correlated random walk; correlation tuned so <R^2> ~ (b^2) N^(2nu)
    kappa = max(0.0, min(0.95, 2.0*nu - 1.0))     # nu=0.5 -> 0 (ideal), nu=0.588 -> 0.176
    v = rng.normal(size=3); v /= np.linalg.norm(v)
    pos = [np.zeros(3)]
    for _ in range(N-1):
        w = rng.normal(size=3); w /= np.linalg.norm(w)
        v = kappa*v + (1-kappa)*w
        v /= np.linalg.norm(v)
        pos.append(pos[-1] + b*v)
    return np.array(pos)

def coil_distance_matrix(N, b, nu, rng):
    X = sample_coil(N, b, nu, rng)
    D = np.linalg.norm(X[:,None,:] - X[None,:,:], axis=-1)
    return D

def analytic_map(N, b, nu):
    i = np.arange(N)
    sep = np.abs(i[:,None] - i[None,:]).astype(float)
    return b * np.power(sep + 1e-6, nu)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--nu', type=float, default=0.588)
    ap.add_argument('--b', type=float, default=5.82)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='results/08_data/ensemble_coil.json')
    A = ap.parse_args()

    src = 'results/08_data/nu_sweep.json'
    if not os.path.exists(src):
        print('missing %s' % src); return 2
    pp = json.load(open(src))['per_protein']
    rng = np.random.RandomState(A.seed)

    rows = []
    for r in pp:
        N = int(r['length'])
        Da = analytic_map(N, A.b, A.nu)
        Ds = [coil_distance_matrix(N, A.b, A.nu, rng) for _ in range(A.k)]
        m = np.mean([d.mean() for d in Ds])
        # how far the ENSEMBLE MEAN sits from the single analytic map the model uses today
        gap = float(np.mean([np.abs(d - Da).mean() for d in Ds]))
        spread = float(np.std([d.mean() for d in Ds]))
        rows.append(dict(protein=r['protein'], N=N, analytic_mean=float(Da.mean()),
                         ensemble_mean=float(m), mean_abs_gap=gap, across_conf_sd=spread))
        print('%-6s N=%3d analytic=%7.3f ensemble=%7.3f gap=%6.3f sd=%5.3f'
              % (r['protein'], N, Da.mean(), m, gap, spread))

    g = np.array([x['mean_abs_gap'] for x in rows])
    s = np.array([x['across_conf_sd'] for x in rows])
    print('\n=== SUMMARY (k=%d nu=%.3f b=%.2f) ===' % (A.k, A.nu, A.b))
    print('mean |ensemble - analytic| = %.4f A   (how wrong the single map is)' % g.mean())
    print('across-conformation sd     = %.4f A   (how much the ensemble actually varies)' % s.mean())
    if s.mean() < 0.05 * g.mean():
        print('\nVERDICT: the ensemble barely varies relative to its offset from the analytic map.')
        print('An ensemble average would then be ~a CONSTANT SHIFT of the analytic map, and a')
        print('constant shift cannot change any correlation. T-E1 would be dead on arrival.')
    else:
        print('\nVERDICT: the ensemble varies enough that averaging is NOT equivalent to a shift.')
        print('Proceed to score E_u over k conformations on std(b_p) vs the 0.9972 baseline.')
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    json.dump(dict(k=A.k, nu=A.nu, b=A.b, per_protein=rows,
                   mean_gap=float(g.mean()), mean_sd=float(s.mean())), open(A.out,'w'), indent=1)
    print('\nwrote %s' % A.out)
    return 0

if __name__ == '__main__':
    sys.exit(main())
