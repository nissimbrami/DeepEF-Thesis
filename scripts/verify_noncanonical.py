"""W10 -- is the OPEN part of the descriptor matrix chemically sensible?

scripts/verify_descriptors_mordred.py checks the canonical twenty (L near I/V/M, D near
E/N, F near Y/W). Nothing checked the rows PAST the twenty, and those are the whole point
of the open alphabet: a non-canonical row that landed in a random place would be worse
than no row at all, because it would look like information and be noise.

The check: each non-canonical residue's position relative to its PARENT -- the canonical
residue it is a modification of.

WHAT COUNTS AS PASSING, and why it is not simply "parent must be rank 1"
--------------------------------------------------------------------------------------
MSE (selenomethionine) and HYP (hydroxyproline) are small perturbations of M and P -- one
atom swapped, one OH added -- so their parent SHOULD be rank 1, and if it is not, the
matrix is wrong.

The phospho residues are different, and getting this wrong would be the easy mistake. A
phosphate group is large, charged and polar; it dominates a 756-column descriptor vector.
Phosphoserine is chemically much more like glutamate than like serine, and the matrix
SHOULD say so. Demanding SEP->S rank 1 would be demanding that the descriptor space be
wrong. So the phospho residues are checked against a chemically-motivated NEIGHBOURHOOD
(each other, and the acidic/polar residues) rather than against their parent alone.

This asymmetry is the honest test. Written down here so that a later reader does not
"fix" the phospho rows to sit next to S and T and quietly destroy the chemistry.
"""
from __future__ import print_function

import argparse
import sys

# parent residue, and the set that residue is ALLOWED to sit among
EXPECT = {
    'MSE': dict(parent='M', allowed={'M'}, strict=True,
                why='selenomethionine is methionine with S->Se, one atom; it must be '
                    'nearest methionine or the matrix is not tracking chemistry'),
    'HYP': dict(parent='P', allowed={'P'}, strict=True,
                why='4-hydroxyproline is proline plus one OH; the pyrrolidine ring '
                    'dominates, so proline must be nearest'),
    'PTR': dict(parent='Y', allowed={'Y', 'W', 'F', 'TPO', 'SEP'}, strict=False,
                why='phosphotyrosine keeps the aromatic ring but gains a phosphate; it '
                    'should stay among the aromatics or beside the other phospho rows'),
    'SEP': dict(parent='S', allowed={'TPO', 'PTR', 'E', 'D', 'Q', 'N'}, strict=False,
                why='the phosphate dominates: phosphoserine is chemically closer to '
                    'glutamate than to serine, and the matrix SHOULD say so'),
    'TPO': dict(parent='T', allowed={'SEP', 'PTR', 'E', 'D', 'Q', 'N'}, strict=False,
                why='same as SEP -- phosphothreonine belongs with the phospho/acidic '
                    'cluster, not next to threonine'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--k', type=int, default=4)
    A = ap.parse_args()

    import numpy as np
    import pandas as pd

    D = pd.read_csv(A.csv, index_col=0, comment='#')
    lab = [str(i) for i in D.index]
    X = D.values.astype(float)
    print('matrix: %d residues x %d descriptors' % X.shape)
    print('labels: %s' % ','.join(lab))

    extras = [l for l in lab if l in EXPECT]
    if not extras:
        print('\nNo non-canonical rows in this matrix -- nothing for this check to do.')
        print('Build one with:  build_aa_descriptors_mordred.py --include_noncanonical')
        return 0

    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    dist = pd.DataFrame(d, index=lab, columns=lab)
    iu = np.triu_indices(len(lab), 1)
    med = float(np.median(d[iu]))
    print('median pairwise distance: %.3f' % med)

    ok = True
    print('')
    for nc in extras:
        e = EXPECT[nc]
        nn = dist[nc].drop(nc).sort_values()
        order = list(nn.index)
        parent = e['parent']
        prank = order.index(parent) + 1 if parent in order else -1
        top = order[:A.k]
        hit = set(top[:2]) & e['allowed']
        if e['strict']:
            good = (order[0] == parent)
            verdict = 'PASS' if good else 'FAIL'
        else:
            good = bool(hit)
            verdict = 'PASS' if good else 'FAIL'
        ok = ok and good
        print('  %-4s (parent %s)  %s' % (nc, parent, verdict))
        print('        nearest: %s' % ', '.join('%s(%.1f)' % (i, nn[i]) for i in top))
        print('        d(%s,%s)=%.3f   parent rank %d/%d   median %.1f'
              % (nc, parent, dist.loc[nc, parent], prank, len(order), med))
        print('        expect: %s' % e['why'])
        if not good:
            print('        -> allowed neighbourhood was {%s}, none of it in the top 2'
                  % ','.join(sorted(e['allowed'])))
        print('')

    # Sanity: an extra row must not be a duplicate of a canonical one.
    for nc in extras:
        nz = dist[nc].drop(nc)
        if float(nz.min()) == 0.0:
            print('  FAIL %s is at distance 0 from %s -- duplicated row'
                  % (nc, nz.idxmin()))
            ok = False

    print('W10 NON-CANONICAL CHEMISTRY: %s'
          % ('SENSIBLE' if ok else 'BROKEN -- the extra rows are not where chemistry says'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
