#!/usr/bin/env python3
"""W13 -- Ofir's MEASURED offset. The highest-value item left, at zero GPU cost.

THE ARGUMENT. b_p (the per-protein offset) is 96.3% one global constant plus a residual spread of
std 1.0155. NINE attempts to PREDICT b_p from features have failed -- length, length-norm,
embedding norm, distribution shift, train/test mean gap, LOPO ridge on 22 features, the reference
state, and mean-pooled ProtT5 (LOPO R^2 = -0.1011). Predicting a per-protein constant at n=28 is
close to unidentifiable.

Ofir Ezrielev does not predict it. He MEASURES it from a handful of labelled mutations per
protein: on our data that lifted pooled dG PCC 0.369 -> 0.753 (oracle 0.817).

LEAKAGE IS THE WHOLE GAME. The k calibration mutations MUST be removed from the evaluation set for
that protein. Enforced mechanically here by index sets, not by discipline: eval_idx = all \ calib.
If that is violated the result is circular and worthless.

HONEST FRAMING, to be stated wherever this is reported: this is FEW-SHOT, not zero-shot. It uses
test-protein labels at inference. The real-world scenario is a wet lab measuring 3 mutants before
committing to a campaign -- a common workflow -- but it must never be presented as zero-shot
prediction.

VARIANCE: k mutations are drawn at random, R times. A single lucky draw is not a result, so we
report mean +/- sd over draws, plus the k=0 baseline.
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd

def load(f):
    d = pd.read_csv(f)
    need = {'protein','ddG','pred_ddG'}
    if not need.issubset(d.columns): return None
    d = d[['protein','ddG','pred_ddG']].dropna()
    d = d[d.protein.astype(str).str.upper() != '2K5H']      # canonical basis: 27 proteins
    return d if len(d) > 100 else None

def pooled(y, p):
    if len(y) < 3 or np.std(p) < 1e-12: return np.nan
    return float(np.corrcoef(p, y)[0,1])

def run(d, k, R, rng):
    """Return pooled PCC after subtracting a per-protein offset MEASURED from k held-out muts."""
    scores = []
    for _ in range(max(1, R)):
        ys, ps = [], []
        for prot, g in d.groupby('protein'):
            n = len(g)
            if n < k + 5:            # need enough left to evaluate on
                continue
            idx = rng.permutation(n)
            calib, ev = idx[:k], idx[k:]          # DISJOINT BY CONSTRUCTION
            gy = g.ddG.values; gp = g.pred_ddG.values
            off = (gp[calib] - gy[calib]).mean() if k > 0 else 0.0
            ys.append(gy[ev]); ps.append(gp[ev] - off)
        if not ys: continue
        scores.append(pooled(np.concatenate(ys), np.concatenate(ps)))
    s = np.array([x for x in scores if np.isfinite(x)])
    return (s.mean(), s.std(ddof=1) if len(s) > 1 else 0.0, len(s)) if len(s) else (np.nan, np.nan, 0)

def oracle(d):
    """Upper bound: remove each protein's TRUE mean offset, using every mutation."""
    ys, ps = [], []
    for _, g in d.groupby('protein'):
        off = (g.pred_ddG.values - g.ddG.values).mean()
        ys.append(g.ddG.values); ps.append(g.pred_ddG.values - off)
    return pooled(np.concatenate(ys), np.concatenate(ps))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='')
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    A = ap.parse_args()

    files = [A.csv] if A.csv else sorted(glob.glob('/home/nissimb/DeepPEF/eval_results/abl_*_e*.csv'))
    picked = []
    for f in files:
        d = load(f)
        if d is not None: picked.append((os.path.basename(f)[4:-4], d))
    if not picked:
        print('no usable CSV'); return 2
    # rank by baseline pooled so we report on the best available checkpoints
    picked.sort(key=lambda t: -(pooled(t[1].ddG.values, t[1].pred_ddG.values) or -9))
    picked = picked[:4]

    rng = np.random.RandomState(A.seed)
    print('W13 -- MEASURED per-protein offset (Ofir). 27 proteins, 2K5H dropped, ddG metric.')
    print('Calibration mutations are EXCLUDED from evaluation for their own protein.\n')
    for tag, d in picked:
        base = pooled(d.ddG.values, d.pred_ddG.values)
        orc  = oracle(d)
        print(f'=== {tag} ===   n={len(d)}  proteins={d.protein.nunique()}')
        print(f'  {"k":>4} {"pooled":>9} {"sd":>8}   gain vs k=0')
        for k in [0, 1, 3, 5, 10, 20]:
            m, sd, r = run(d, k, 1 if k == 0 else A.reps, rng)
            if not np.isfinite(m): continue
            mark = ''
            if k == 0: k0 = m
            else: mark = f'{m-k0:+.4f}'
            print(f'  {k:>4} {m:>9.4f} {sd:>8.4f}   {mark}')
        print(f'  baseline pooled (no offset removal) = {base:.4f}')
        print(f'  ORACLE (true offset, all muts)      = {orc:.4f}   <- the ceiling\n')
    return 0

if __name__ == '__main__':
    sys.exit(main())
