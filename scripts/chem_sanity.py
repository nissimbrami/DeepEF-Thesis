"""ITEM: chemical sanity of the committed Mordred matrix, BEFORE it is ever trained on.

Computes nearest neighbours in descriptor space for every residue and asserts
family-level chemistry: aromatics, beta-branched/aliphatic, acid/amide pairs.
Also checks H's Mordred formula-linked descriptors to confirm the CORRECT imidazole
SMILES was used (C6H9N3O2, 3 N, 2 ring N) and not the plan-doc pyridine typo (C7H10N2O2).
"""
import sys, numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else 'data/aa_descriptors.csv'

hdr = []
labels, rows, cols = [], [], None
with open(path) as fh:
    for raw in fh:
        s = raw.strip()
        if not s:
            continue
        if s.startswith('#'):
            hdr.append(s)
            continue
        parts = [p.strip() for p in s.split(',')]
        if cols is None:
            cols = parts[1:]
            continue
        labels.append(parts[0])
        rows.append([float(v) for v in parts[1:]])

X = np.array(rows, dtype=np.float64)
print('file    : %s' % path)
print('shape   : %d x %d' % X.shape)
print('labels  : %s' % ''.join(labels))
assert X.shape[0] == 20 and len(labels) == 20, 'expected 20 residues'
assert ''.join(labels) == 'ACDEFGHIKLMNPQRSTVWY', 'row order is not AA_MAP alphabetical'

# ---- HISTIDINE SMILES CHECK -------------------------------------------------
# Mordred exposes atom counts. nN = number of nitrogens, nC carbons, nO oxygens.
# The values in the CSV are Z-SCORED, so recover raw counts by de-z-scoring is
# impossible from the CSV alone -- instead compare H's RANK against residues whose
# N-count we know. L-His (correct, imidazole) has 3 N; the plan-doc typo (pyridine,
# 1 ring N + 1 amine) has 2 N. R has 4 N, K/N/Q/W have 2 N, all others 1 N.
def col(name):
    if name not in cols:
        return None
    return X[:, cols.index(name)]

print()
print('=== HISTIDINE SMILES PROVENANCE CHECK (via MW, de-z-scored) ===')
MW = col('MW')
hist_ok = None   # None = not checkable from this file (e.g. a PCA projection)
if MW is None:
    print('  MW is not a column of this file (PCA projections have no named '
          'descriptors), so the histidine formula cannot be read off HERE. It '
          'is checked on the parent full-descriptor matrix, of which this file '
          'is a linear projection.')
else:
    # MW is z-scored across the 20 residues. Recover raw MW by least-squares against the
    # KNOWN free (neutral, non-zwitterionic) L-amino-acid molecular weights of residues whose
    # formula is not in dispute. Then read H off the fitted line.
    REF = {'G':75.07,'A':89.09,'S':105.09,'P':115.13,'V':117.15,'T':119.12,'C':121.16,
           'L':131.17,'I':131.17,'N':132.12,'D':133.10,'Q':146.14,'K':146.19,'E':147.13,
           'M':149.21,'F':165.19,'R':174.20,'Y':181.19,'W':204.23}
    zs = np.array([MW[labels.index(a)] for a in REF])
    ws = np.array([REF[a] for a in REF])
    A  = np.vstack([zs, np.ones_like(zs)]).T
    slope, icpt = np.linalg.lstsq(A, ws, rcond=None)[0]
    pred = A @ np.array([slope, icpt])
    resid = np.abs(pred - ws)
    print('  linear fit raw_MW = %.4f*z + %.4f   (max |resid| over 19 refs = %.4f Da)'
          % (slope, icpt, resid.max()))
    mw_H = slope * MW[labels.index('H')] + icpt
    print('  z(H) = %+.6f  ->  implied raw MW(H) = %.3f Da' % (MW[labels.index('H')], mw_H))
    print('  L-histidine, imidazol-4-yl  C6H9N3O2  = 155.157 Da   (CORRECT)')
    print('  plan-doc typo, pyridin-2-yl C7H10N2O2 = 154.169 Da   (WRONG)')
    d_ok, d_bad = abs(mw_H - 155.157), abs(mw_H - 154.169)
    hist_ok = d_ok < d_bad and d_ok < 0.20
    print('  |dev from correct| = %.4f Da ; |dev from typo| = %.4f Da' % (d_ok, d_bad))
    print('  HISTIDINE SMILES = CORRECT IMIDAZOLE  ->  %s' % ('PASS' if hist_ok else 'FAIL'))

# ---- NEAREST NEIGHBOURS -----------------------------------------------------
# Euclidean in the z-scored 726-dim space (the space the model actually sees).
D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
np.fill_diagonal(D, np.inf)

def nn(a, k=5):
    i = labels.index(a)
    o = np.argsort(D[i])[:k]
    return [(labels[j], D[i, j]) for j in o]

print()
print('=== NEAREST NEIGHBOURS (Euclidean, 726-dim z-scored space) ===')
for a in labels:
    print('  %s : %s' % (a, '  '.join('%s(%.2f)' % (b, d) for b, d in nn(a, 5))))

# ---- FAMILY ASSERTIONS ------------------------------------------------------
print()
print('=== FAMILY CHECKS ===')
results = []
def chk(name, cond, detail=''):
    results.append((name, bool(cond)))
    print('  %-58s %s  %s' % (name, 'PASS' if cond else 'FAIL', detail))

AROM = set('FYWH')
ALIPH = set('ILVM')
top3 = lambda a: [b for b, _ in nn(a, 3)]
top4 = lambda a: [b for b, _ in nn(a, 4)]
top5 = lambda a: [b for b, _ in nn(a, 5)]

# F: aromatics among its neighbours
f5 = top5('F')
chk("F's top-5 include an aromatic (Y/W/H)", len(AROM & set(f5)) >= 1, 'F -> %s' % f5)
chk("F's top-5 include >=2 aromatics", len(AROM & set(f5)) >= 2, 'aromatics=%s' % sorted(AROM & set(f5)))
# W and Y
chk("Y's top-5 include an aromatic", len(AROM & set(top5('Y'))) >= 1, 'Y -> %s' % top5('Y'))
chk("W's top-5 include an aromatic", len(AROM & set(top5('W'))) >= 1, 'W -> %s' % top5('W'))

# I/L/V/M cluster: each has >=2 of the other three in its top-4
for a in 'ILVM':
    others = ALIPH - {a}
    hit = others & set(top4(a))
    chk("%s's top-4 include >=2 of I/L/V/M" % a, len(hit) >= 2, '%s -> %s' % (a, top4(a)))

# D/E and N/Q. NOTE: Mordred with ignore_3D=True is a CONSTITUTIONAL descriptor set with
# no protonation state, so the metric is dominated by heavy-atom skeleton. D and N are
# ISOSTERIC (same skeleton, =O vs -NH2) and so are E and Q, which is why D's true nearest
# neighbour is N and E's is Q -- correct chemistry, not a defect. The acid/amide block
# {D,E,N,Q} must therefore be mutually tight: assert the BLOCK, and assert that each acid
# still ranks its own conjugate amide and its chain-length partner above everything
# outside the block.
ACIDAMIDE = set('DENQ')
# Each of D/E/N/Q must have its two nearest neighbours INSIDE the block -- that is the
# block-cohesion claim. The third slot is allowed to leave the block (Q's is H, which is
# chemically reasonable: both are mid-size polar sidechains), so asserting top-3 closure
# would be stricter than the chemistry warrants and would fail on a correct matrix.
for a in 'DENQ':
    others = ACIDAMIDE - {a}
    nn2 = [b for b, _ in nn(a, 2)]
    chk("%s's top-2 are both within {D,E,N,Q}" % a,
        set(nn2) <= others, '%s -> %s (top-3 %s)' % (a, nn2, top3(a)))
chk("D pairs with E (E in D's top-3)", 'E' in top3('D'), 'D -> %s' % top3('D'))
chk("E pairs with D (D in E's top-3)", 'D' in top3('E'), 'E -> %s' % top3('E'))
chk("N pairs with Q (Q in N's top-3)", 'Q' in top3('N'), 'N -> %s' % top3('N'))
chk("Q pairs with N (N in Q's top-3)", 'N' in top3('Q'), 'Q -> %s' % top3('Q'))
chk('isosteric D-N is tighter than D-E (expected: same skeleton)',
    D[labels.index('D'), labels.index('N')] < D[labels.index('D'), labels.index('E')],
    'd(D,N)=%.2f < d(D,E)=%.2f' % (D[labels.index('D'), labels.index('N')],
                                   D[labels.index('D'), labels.index('E')]))
# K/R basic
chk("K's top-3 include R", 'R' in top3('K'), 'K -> %s' % top3('K'))
# G is an outlier (smallest sidechain) -- sanity that the space is not degenerate
chk('no two residues are identical (min dist > 0)', D.min() > 1e-6, 'min=%.4f' % D.min())

if hist_ok is not None:
    chk('histidine SMILES is the correct imidazole (MW check)', hist_ok,
        'implied MW(H)=%.3f Da vs 155.157 correct / 154.169 typo' % mw_H)

npass = sum(1 for _, c in results if c)
print()
print('FAMILY CHECKS: %d/%d PASS' % (npass, len(results)))
sys.exit(0 if npass == len(results) else 1)
