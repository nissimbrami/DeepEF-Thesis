#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ITEM 18 -- the DECISIVE gate on the Mordred descriptor matrix.

PHASE 6.4: "If aspartate comes out near leucine, the matrix is wrong and no training
will fix it."

This script is the gate. It runs on the CSV, needs no rdkit or mordred, and exits
non-zero if the matrix is not chemically sensible. Run it before any training arm that
consumes the matrix, and paste its output into the run log.

    python verify_descriptors_mordred.py data/aa_descriptors_mordred.csv
    python verify_descriptors_mordred.py data/aa_descriptors_mordred_pca16.csv

THE REQUIRED CHECKS
  1. Shape and order       -- 20 rows, exactly ACDEFGHIKLMNPQRSTVWY, in that order.
                              Row i must be one-hot index i or the block is permuted
                              against the one-hot block and everything downstream is wrong.
  2. Finiteness / scale    -- no NaN, no inf, no constant column, no column whose scale is
                              so large it alone dictates every Euclidean distance.
  3. NEAREST NEIGHBOURS    -- the decisive one, from PHASE 6.4:
                                 L -> among {I, V, M}
                                 D -> among {E, N}
                                 F -> among {Y, W}
  4. ANTI-CHECK            -- D must NOT be near L, and L must NOT be near D or E. This is
                              the named failure mode; check 3 alone can pass weakly while
                              this is violated, so it is tested on its own.
  5. Extra sanity          -- charged/polar vs aliphatic separation, G and P as outliers,
                              and the aromatic trio F/Y/W mutually close.

Checks 1-4 are BLOCKING. Check 5 is reported and does not block: it encodes softer
expectations that a legitimate descriptor space may reasonably violate.

WHY THE THRESHOLD IS WHAT IT IS
  With 20 residues, a random distance matrix puts a specific residue in a specific 3-member
  allowed set with probability ~3/19 = 0.158 per draw. Requiring TWO of the top three
  neighbours to fall in the allowed set is therefore a real test, not a formality --
  under a random matrix the three families would pass together with probability well
  under 1%.
"""
from __future__ import print_function

import sys

import numpy as np
import pandas as pd

# Chemistry expectations. PHASE 6.4 defines the first three; the rest are this script's.
REQUIRED = {
    'L': set('IVM'),      # leucine: the other large aliphatics
    'D': set('EN'),       # aspartate: the other acid, and its amide
    'F': set('YW'),       # phenylalanine: the other aromatics
}
# Residues that must NOT be near each other. Violating any of these means the space has
# collapsed the polar/apolar axis, which is the axis that matters most for stability.
FORBIDDEN = [
    ('D', 'L'),   # the named catastrophe
    ('D', 'I'),
    ('D', 'V'),
    ('E', 'L'),
    ('R', 'V'),
    ('K', 'F'),
]
ORDER = list('ACDEFGHIKLMNPQRSTVWY')

APOLAR = set('AVLIMFWC')
CHARGED = set('DEKR')


def load(path):
    """Read the matrix, tolerating the '# ...' provenance header."""
    df = pd.read_csv(path, index_col=0, comment='#')
    df.index = [str(i).strip() for i in df.index]
    return df


def pdist_matrix(D):
    """Euclidean distance matrix without scipy, so this runs in a bare env."""
    X = D.values.astype('float64')
    sq = (X ** 2).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * X.dot(X.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 0.0)
    return pd.DataFrame(np.sqrt(d2), index=D.index, columns=D.index)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/aa_descriptors_mordred.csv'
    print('=' * 78)
    print('DESCRIPTOR MATRIX GATE  --  %s' % path)
    print('=' * 78)

    try:
        D = load(path)
    except Exception as e:
        print('FAIL: could not read the matrix: %s' % e)
        return 1

    blocking = []

    # ---- check 1: shape and order --------------------------------------------------
    print('\n[1] shape and residue order')
    print('    %d residues x %d descriptors' % (D.shape[0], D.shape[1]))
    got = list(D.index)
    # W10: open alphabet. The invariant is NOT "exactly the twenty" -- that was the same
    # hard closure this project removed from aa_descriptors.py. It is: the canonical
    # twenty occupy rows 0..19 in AA_MAP order (so row i == one-hot index i and the
    # canonical tensor path is byte-identical), and anything after row 19 is an extra.
    head, extras = got[:len(ORDER)], got[len(ORDER):]
    if head != ORDER:
        blocking.append('rows 0..%d are %s, must be %s (row i == one-hot index i)'
                        % (len(ORDER) - 1, ''.join(head), ''.join(ORDER)))
        print('    FAIL order: first %d rows are %s' % (len(ORDER), ''.join(head)))
    else:
        print('    ok: rows 0-19 are ACDEFGHIKLMNPQRSTVWY, matching train_utils.AA_MAP')
    if extras:
        dup = [e for e in extras if e in ORDER]
        seen, rep = set(), []
        for e in extras:
            if e in seen:
                rep.append(e)
            seen.add(e)
        if dup:
            blocking.append('extra rows %s shadow canonical residues' % ','.join(dup))
            print('    FAIL extras: %s duplicate canonical labels' % ','.join(dup))
        elif rep:
            blocking.append('extra rows repeat labels %s' % ','.join(rep))
            print('    FAIL extras: repeated labels %s' % ','.join(rep))
        else:
            print('    ok: %d NON-CANONICAL row(s) past the twenty: %s'
                  % (len(extras), ','.join(extras)))
            print('        (the alphabet is OPEN -- these are reachable by label via')
            print('         aa_descriptors.residue_descriptors_by_label)')
    if D.shape[1] < 2:
        blocking.append('only %d descriptor column(s) -- nothing to measure' % D.shape[1])

    # ---- check 2: finiteness and scale ---------------------------------------------
    print('\n[2] finiteness and scale')
    X = D.values.astype('float64')
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    print('    NaN=%d  inf=%d' % (n_nan, n_inf))
    if n_nan or n_inf:
        blocking.append('matrix contains %d NaN and %d inf -- filter 1 did not do its job'
                        % (n_nan, n_inf))
    sd = X.std(axis=0, ddof=0)
    n_const = int((sd < 1e-12).sum())
    print('    constant columns=%d   column std: min=%.4g median=%.4g max=%.4g'
          % (n_const, sd.min(), float(np.median(sd)), sd.max()))
    if n_const:
        print('    WARN: %d constant column(s) contribute nothing but cost width' % n_const)
    # A single runaway column can dominate every distance and make the gate meaningless.
    var = X.var(axis=0, ddof=0)
    if var.sum() > 0:
        top = float(var.max() / var.sum())
        print('    largest single-column share of total variance: %.3f' % top)
        if top > 0.5:
            blocking.append('one column carries %.1f%% of total variance -- distances are '
                            'that column, not the chemistry. Was the matrix normalised?'
                            % (100 * top))

    if blocking:
        # Distances are meaningless if the above failed; stop before reporting them.
        print('\n' + '=' * 78)
        for b in blocking:
            print('BLOCKING: %s' % b)
        print('VERDICT: BROKEN -- do not train on this matrix.')
        print('=' * 78)
        return 1

    dist = pdist_matrix(D)
    offdiag = dist.values[~np.eye(len(D), dtype=bool)]
    med = float(np.median(offdiag))

    # ---- check 3: nearest neighbours, the decisive one ------------------------------
    print('\n[3] nearest neighbours  (DECISIVE -- PHASE 6.4)')
    print('    %-3s %-26s %-12s %s' % ('aa', 'three nearest', 'allowed', 'verdict'))
    for aa in sorted(REQUIRED):
        if aa not in dist.index:
            blocking.append('residue %s missing from the matrix' % aa)
            continue
        s = dist[aa].drop(aa).sort_values()
        nn3 = list(s.index[:3])
        allowed = REQUIRED[aa]
        n_hit = len(set(nn3) & allowed)
        # Requirement: at least two of the top three fall in the allowed set, and the
        # single closest is one of them. Both, because either alone is too weak.
        ok = (n_hit >= 2) and (nn3[0] in allowed)
        verdict = 'PASS' if ok else ('WEAK' if n_hit >= 1 else 'FAIL')
        detail = ', '.join('%s(%.2f)' % (b, s[b]) for b in nn3)
        print('    %-3s %-26s %-12s %s  [%d/3 in set]'
              % (aa, detail, ','.join(sorted(allowed)), verdict, n_hit))
        if not ok:
            blocking.append('%s nearest neighbours %s are not among {%s} '
                            '(need >=2 of 3, and the closest must be one of them)'
                            % (aa, ','.join(nn3), ','.join(sorted(allowed))))

    # ---- check 4: the anti-check ----------------------------------------------------
    print('\n[4] anti-check  -- pairs that must stay apart')
    print('    median off-diagonal distance = %.3f' % med)
    for a, b in FORBIDDEN:
        if a not in dist.index or b not in dist.index:
            continue
        d = float(dist.loc[a, b])
        # b must not be in a's three nearest, and the distance must be at least
        # median/2 -- a pair far closer than typical is a collapsed axis.
        nn3 = list(dist[a].drop(a).sort_values().index[:3])
        bad = (b in nn3) or (d < 0.5 * med)
        rank = list(dist[a].drop(a).sort_values().index).index(b) + 1
        print('    %s vs %s: d=%.3f (rank %2d of 19)   %s'
              % (a, b, d, rank, 'FAIL' if bad else 'ok'))
        if bad:
            blocking.append('%s and %s are too close (d=%.3f, rank %d, median %.3f) -- '
                            'the polar/apolar axis has collapsed' % (a, b, d, rank, med))

    # ---- check 5: soft sanity, reported only ----------------------------------------
    print('\n[5] soft sanity  (reported, NOT blocking)')
    ap = [a for a in D.index if a in APOLAR]
    ch = [a for a in D.index if a in CHARGED]
    if ap and ch:
        within = np.mean([dist.loc[i, j] for i in ap for j in ap if i != j])
        across = np.mean([dist.loc[i, j] for i in ap for j in ch])
        print('    mean distance within apolar {%s}: %.3f' % (''.join(ap), within))
        print('    mean distance apolar<->charged : %.3f' % across)
        print('    separation ratio across/within : %.3f  %s'
              % (across / within if within else float('nan'),
                 '(>1 expected)' if across > within else '(<1 -- unexpected)'))
    for aa in ('G', 'P'):
        if aa in dist.index:
            mean_d = float(dist[aa].drop(aa).mean())
            print('    %s mean distance to others: %.3f  (median residue %.3f) %s'
                  % (aa, mean_d, med, '-- outlier, as expected' if mean_d > med else ''))
    arom = [a for a in 'FYW' if a in dist.index]
    if len(arom) == 3:
        pairs = [(arom[i], arom[j]) for i in range(3) for j in range(i + 1, 3)]
        vals = ['%s%s=%.2f' % (a, b, dist.loc[a, b]) for a, b in pairs]
        print('    aromatic trio: %s   (median %.3f)' % ('  '.join(vals), med))

    # ---- verdict --------------------------------------------------------------------
    print('\n' + '=' * 78)
    if blocking:
        for b in blocking:
            print('BLOCKING: %s' % b)
        print('')
        print('VERDICT: BROKEN -- do not train on this matrix.')
        print('A wrong SMILES, a wrong residue order, or a missing normalisation are the')
        print('three causes worth checking first, in that order.')
        print('=' * 78)
        return 1
    print('VERDICT: USABLE -- checks 1-4 all pass.')
    print('The descriptor space is chemically sensible: the aliphatic, acidic and aromatic')
    print('families each recover their own members, and the polar/apolar axis is intact.')
    print('=' * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())
