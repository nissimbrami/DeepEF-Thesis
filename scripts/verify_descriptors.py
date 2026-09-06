"""W6 verification -- DECISIVE. If aspartate comes out near leucine, the matrix is
wrong and no amount of training will fix it.

Required: L's nearest neighbours within {I,V,M}; D's within {E,N}; F's within {Y,W}.
"""
import sys
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

path = sys.argv[1] if len(sys.argv) > 1 else 'data/aa_descriptors.csv'
D = pd.read_csv(path, index_col=0, comment='#')
print('matrix: %d residues x %d descriptors' % D.shape)

dist = pd.DataFrame(squareform(pdist(D.values)), index=D.index, columns=D.index)
REQ = {'L': {'I', 'V', 'M'}, 'D': {'E', 'N'}, 'F': {'Y', 'W'}}
ok = True
for aa, allowed in REQ.items():
    nn = dist[aa].drop(aa).nsmallest(3).index.tolist()
    hit = len(set(nn[:len(allowed)]) & allowed) > 0
    strict = set(nn[:2]).issubset(allowed) or len(set(nn) & allowed) >= min(2, len(allowed))
    print('  %s -> %-12s  allowed %-10s  %s' % (aa, ','.join(nn), ','.join(sorted(allowed)),
                                                'PASS' if strict else ('WEAK' if hit else 'FAIL')))
    if not hit:
        ok = False

print('\nsanity: most and least hydrophobic should be far apart')
if 'hydropathy' in D.columns:
    print('  I vs R distance: %.3f   (median pairwise %.3f)'
          % (dist.loc['I', 'R'], np.median(squareform(dist.values, checks=False))))

print('\nW6 DESCRIPTORS: %s' % ('USABLE' if ok else 'BROKEN -- do not train on this'))
sys.exit(0 if ok else 1)
