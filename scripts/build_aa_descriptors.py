"""W6: build the amino-acid physicochemical descriptor matrix, once, offline.

Ofir Ezrielev's method: chiral SMILES -> Mordred (1826 descriptors) -> drop those with
missing values -> keep those with enough unique values across the alphabet -> normalise.
His central result is that a model trained on canonical residues predicts non-canonical
effects, because the physicochemical space is continuous.

DELIBERATE DEVIATION, stated here and in the CSV header: Ofir kept descriptors with >= 40
unique values across 58 residues. With 20 residues the maximum possible unique count is
20, so the threshold scales to 15 -- a descriptor must distinguish three quarters of the
alphabet.

rdkit and mordred are NOT installed in esm2_env_py38 and must not be. This script runs
once, anywhere, and commits data/aa_descriptors.csv. Training-time code depends only on
that CSV, never on rdkit.

Fallback: if mordred is unavailable, --curated writes the 12 interpretable descriptors
from a literature table instead, with provenance recorded the same way.
"""
import argparse
import sys

# AA_MAP order (alphabetical), matching train_utils.AA_MAP exactly.
ORDER = list('ACDEFGHIKLMNPQRSTVWY')

SMILES = {
    'A': 'C[C@@H](C(=O)O)N', 'C': 'C([C@@H](C(=O)O)N)S', 'D': 'C([C@@H](C(=O)O)N)C(=O)O',
    'E': 'C(CC(=O)O)[C@@H](C(=O)O)N', 'F': 'c1ccc(cc1)C[C@@H](C(=O)O)N',
    'G': 'C(C(=O)O)N', 'H': 'c1cc(nc1)C[C@@H](C(=O)O)N', 'I': 'CC[C@H](C)[C@@H](C(=O)O)N',
    'K': 'C(CCN)C[C@@H](C(=O)O)N', 'L': 'CC(C)C[C@@H](C(=O)O)N',
    'M': 'CSCC[C@@H](C(=O)O)N', 'N': 'C([C@@H](C(=O)O)N)C(=O)N',
    'P': 'C1C[C@H](NC1)C(=O)O', 'Q': 'C(CC(=O)N)[C@@H](C(=O)O)N',
    'R': 'C(C[C@@H](C(=O)O)N)CNC(=N)N', 'S': 'C([C@@H](C(=O)O)N)O',
    'T': 'C[C@H]([C@@H](C(=O)O)N)O', 'V': 'CC(C)[C@@H](C(=O)O)N',
    'W': 'c1ccc2c(c1)c(c[nH]2)C[C@@H](C(=O)O)N',
    'Y': 'c1cc(ccc1C[C@@H](C(=O)O)N)O',
}

# Fallback table: molecular weight, van der Waals volume (A^3), hydropathy (Kyte-Doolittle),
# formal charge at pH 7, HBD, HBA, aromatic rings, rotatable bonds, polarity, flexibility,
# isoelectric point, side-chain surface area (A^2). Literature values.
CURATED = {
    #     MW     vdW    KD    chg HBD HBA arom rot  polar flex   pI    SASA
    'A': (89.1,  88.6,  1.8,  0,  1,  1,  0,  0,  0.0,  0.36, 6.00, 115, 0, 0),
    'C': (121.2, 108.5, 2.5,  0,  2,  2,  0,  1,  0.13, 0.35, 5.07, 135, 0, 0),
    'D': (133.1, 111.1, -3.5, -1, 2,  4,  0,  2,  1.0,  0.51, 2.77, 150, 0, 2),
    'E': (147.1, 138.4, -3.5, -1, 2,  4,  0,  3,  1.0,  0.50, 3.22, 190, 0, 2),
    'F': (165.2, 189.9, 2.8,  0,  1,  1,  1,  2,  0.0,  0.31, 5.48, 210, 6, 6),
    'G': (75.1,  60.1,  -0.4, 0,  1,  1,  0,  0,  0.0,  0.54, 5.97, 75, 0, 0),
    'H': (155.2, 153.2, -3.2, 0,  2,  3,  1,  2,  0.51, 0.32, 7.59, 195, 5, 6),
    'I': (131.2, 166.7, 4.5,  0,  1,  1,  0,  2,  0.0,  0.46, 6.02, 175, 0, 0),
    'K': (146.2, 168.6, -3.9, 1,  3,  2,  0,  4,  1.0,  0.47, 9.74, 200, 0, 0),
    'L': (131.2, 166.7, 3.8,  0,  1,  1,  0,  2,  0.0,  0.37, 5.98, 170, 0, 0),
    'M': (149.2, 162.9, 1.9,  0,  1,  2,  0,  3,  0.0,  0.30, 5.74, 185, 0, 0),
    'N': (132.1, 114.1, -3.5, 0,  3,  3,  0,  2,  1.0,  0.46, 5.41, 160, 0, 2),
    'P': (115.1, 112.7, -1.6, 0,  1,  1,  0,  0,  0.0,  0.51, 6.30, 145, 5, 0),
    'Q': (146.2, 143.8, -3.5, 0,  3,  3,  0,  3,  1.0,  0.49, 5.65, 180, 0, 2),
    'R': (174.2, 173.4, -4.5, 1,  5,  3,  0,  5,  1.0,  0.53, 10.76, 225, 0, 2),
    'S': (105.1, 89.0,  -0.8, 0,  2,  2,  0,  1,  1.0,  0.51, 5.68, 115, 0, 0),
    'T': (119.1, 116.1, -0.7, 0,  2,  2,  0,  1,  1.0,  0.44, 5.60, 140, 0, 0),
    'V': (117.1, 140.0, 4.2,  0,  1,  1,  0,  1,  0.0,  0.39, 5.96, 155, 0, 0),
    'W': (204.2, 227.8, -0.9, 0,  2,  1,  2,  3,  0.13, 0.31, 5.89, 255, 9, 10),
    'Y': (181.2, 193.6, -1.3, 0,  2,  2,  1,  3,  0.20, 0.42, 5.66, 230, 6, 6),
}
CURATED_COLS = ['MW', 'vdW_vol', 'hydropathy', 'charge', 'HBD', 'HBA',
                'aromatic_rings', 'rotatable_bonds', 'polarity', 'flexibility',
                'pI', 'sidechain_SASA', 'ring_atoms', 'pi_electrons']

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='data/aa_descriptors.csv')
ap.add_argument('--min_unique', type=int, default=15,
                help='Ofir used 40 of 58 residues; scaled to 15 of 20 (three quarters)')
ap.add_argument('--curated', action='store_true', help='skip mordred, write the 12-descriptor table')
ap.add_argument('--pca', type=int, default=0, help='if >0, also write a PCA-reduced matrix')
A = ap.parse_args()

import numpy as np
import pandas as pd

prov = []
if A.curated:
    df = pd.DataFrame([CURATED[a] for a in ORDER], index=ORDER, columns=CURATED_COLS)
    prov.append('source=curated literature table (12 interpretable descriptors)')
    n_raw = df.shape[1]
    n_nona = n_raw
else:
    try:
        from rdkit import Chem
        import rdkit
        from mordred import Calculator, descriptors
        import mordred
    except ImportError as e:
        print('rdkit/mordred unavailable (%s).' % e)
        print('Re-run with --curated, or install them in a SEPARATE env -- never in')
        print('esm2_env_py38, which must stay exactly as the training runs saw it.')
        sys.exit(2)
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(SMILES[a]) for a in ORDER]
    assert all(m is not None for m in mols), 'a SMILES failed to parse'
    df = calc.pandas(mols)
    df.index = ORDER
    n_raw = df.shape[1]
    df = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1)   # Ofir filter 1
    n_nona = df.shape[1]
    df = df.loc[:, df.nunique() >= A.min_unique]                   # Ofir filter 2, scaled
    prov.append('source=Mordred(ignore_3D) over chiral L-amino-acid SMILES')
    prov.append('rdkit=%s mordred=%s' % (rdkit.__version__, getattr(mordred, '__version__', '?')))

sd = df.std(ddof=0).replace(0, 1)
dfn = (df - df.mean()) / sd

prov += ['order=AA_MAP alphabetical ACDEFGHIKLMNPQRSTVWY',
         'filter1=drop columns with any missing value: %d -> %d' % (n_raw, n_nona),
         'filter2=keep nunique >= %d (Ofir used 40/58; scaled to %d/20): %d -> %d'
         % (A.min_unique, A.min_unique, n_nona, df.shape[1]),
         'normalisation=z-score across the 20 residues']

if A.pca:
    from numpy.linalg import svd
    X = dfn.values - dfn.values.mean(0)
    U, S, Vt = svd(X, full_matrices=False)
    k = min(A.pca, X.shape[0] - 1, X.shape[1])
    Z = U[:, :k] * S[:k]
    ev = (S ** 2 / (S ** 2).sum())[:k]
    prov.append('pca=%d components, variance explained=%.4f' % (k, float(ev.sum())))
    dfn = pd.DataFrame(Z, index=ORDER, columns=['PC%d' % (i + 1) for i in range(k)])

import os
d = os.path.dirname(A.out)
if d and not os.path.isdir(d):
    os.makedirs(d)
with open(A.out, 'w') as f:
    for line in prov:
        f.write('# %s\n' % line)
    dfn.to_csv(f)
print('wrote %s  shape=%s' % (A.out, dfn.shape))
for line in prov:
    print('  # ' + line)
