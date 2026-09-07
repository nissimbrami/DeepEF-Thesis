#!/usr/bin/env python
"""
gate_w6.py -- W6 descriptor-matrix gate.

Runs in the TRAINING env (esm2_env_py38). Imports neither rdkit nor mordred: the whole
point of the W6 design is that the CSV is the artifact and training-time code only reads it.

What it enforces, and why each check exists:

  1. Every mode's CSV exists and loads.
  2. Row labels are exactly ACDEFGHIKLMNPQRSTVWY -- all 20, no extras.
  3. The loader reorders BY LABEL, not by position. Proved by shuffling the data rows of
     the real file and asserting the loaded tensor is bit-identical.
  4. descriptor_dim(mode) == the actual column count of the file (not hard-coded).
  5. The matrix is finite (no NaN/inf) and no column is constant.
  6. The matrix is NOT the old placeholder: its provenance header must not say
     'synthetic replica' / 'curated literature table', and must name rdkit+mordred.
  7. CHEMISTRY. Nearest neighbour in descriptor space: L in {I,V,M}, D in {E,N}, F in {Y,W}.
     If aspartate lands next to leucine the matrix is wrong and no training run fixes it.
  8. HISTIDINE, specifically. The bug this gate exists for was invisible to a gate that
     tested only L, D and F. Histidine must be nearer to the aromatics/polars than to the
     aliphatics -- a pyridine-ring C7H10N2O2 impostor fails this.
  9. PCA arm is NOT whitened: the per-component stds must not all be 1.0, because
     per-component whitening destroys the eigenvalue spectrum, which is the chemistry.
"""
import os, sys, math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

ORDER = list('ACDEFGHIKLMNPQRSTVWY')
results = []


def check(name, cond, detail=''):
    results.append((name, bool(cond), detail))
    print('%-46s %-5s %s' % (name, 'PASS' if cond else 'FAIL', detail))
    return bool(cond)


def read_header(path):
    out = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#'):
                out.append(line[1:].strip())
            else:
                break
    return out


def main():
    import aa_descriptors as ad

    check('consumer imports no rdkit',
          'rdkit' not in sys.modules and 'mordred' not in sys.modules,
          'training-time code must never need the chemistry toolkit')

    modes = [m for m in ad.MODES if m != 'none']
    tables = {}

    for mode in modes:
        path = ad.descriptor_path(mode)
        if not check('%s: CSV exists' % mode, os.path.isfile(path), path):
            continue
        try:
            t = ad.load_descriptor_table(path)
        except Exception as e:
            check('%s: loads' % mode, False, repr(e))
            continue
        tables[mode] = (path, t)
        check('%s: loads' % mode, True, 'shape=%s' % (tuple(t.shape),))
        check('%s: 20 rows' % mode, t.shape[0] == 20, '%d rows' % t.shape[0])
        check('%s: descriptor_dim == ncols' % mode,
              ad.descriptor_dim(mode) == t.shape[1],
              'descriptor_dim=%d ncols=%d' % (ad.descriptor_dim(mode), t.shape[1]))
        check('%s: all finite' % mode, bool(t.isfinite().all()))
        sd = t.std(dim=0, unbiased=False)
        check('%s: no constant column' % mode, bool((sd > 0).all()),
              '%d constant' % int((sd <= 0).sum()))

        # provenance: must be the real thing, not the placeholder
        hdr = ' '.join(read_header(path)).lower()
        check('%s: header not a placeholder' % mode,
              ('synthetic replica' not in hdr) and ('curated literature table' not in hdr),
              'header must not advertise a stand-in matrix')
        check('%s: header names rdkit+mordred' % mode,
              ('rdkit=' in hdr) and ('mordred=' in hdr),
              'provenance must record the toolkit versions')

    # ---- reorder is BY LABEL, proved on the real file --------------------------------
    if tables:
        mode0, (p0, t0) = list(tables.items())[0]
        lines = open(p0).read().splitlines()
        hdr = [l for l in lines if l.startswith('#')]
        body = [l for l in lines if not l.startswith('#') and l.strip()]
        colrow, rows = body[0], body[1:]
        shuffled = list(reversed(rows))
        tmp = os.path.join(ROOT, '.gate_w6_shuffled.csv')
        with open(tmp, 'w') as fh:
            fh.write('\n'.join(hdr + [colrow] + shuffled) + '\n')
        try:
            t1 = ad.load_descriptor_table(tmp)
            same = bool((t0 == t1).all())
        finally:
            os.remove(tmp)
        check('loader reorders by LABEL not position', same,
              'row-reversed file must load bit-identically')

    # ---- chemistry -------------------------------------------------------------------
    for mode, (path, t) in tables.items():
        X = t.double()
        D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1).sqrt()
        D.fill_diagonal_(float('inf'))
        nn = {}
        for i, aa in enumerate(ORDER):
            nn[aa] = ORDER[int(D[i].argmin())]
        for q, exp in (('L', {'I', 'V', 'M'}), ('D', {'E', 'N'}), ('F', {'Y', 'W'})):
            check('%s: NN(%s) in %s' % (mode, q, sorted(exp)), nn[q] in exp,
                  'got %s' % nn[q])
        # Histidine: the residue the old three-residue gate never looked at.
        # H is aromatic AND polar/ionisable; it must not sit among the pure aliphatics.
        aliph = {'A', 'V', 'L', 'I', 'G', 'P'}
        check('%s: NN(H) not aliphatic' % mode, nn['H'] not in aliph,
              'got %s -- a pyridine-ring impostor lands here' % nn['H'])
        # and D must never be nearest to L
        check('%s: NN(D) != L' % mode, nn['D'] != 'L', 'got %s' % nn['D'])

    # ---- PCA not whitened -------------------------------------------------------------
    for mode, (path, t) in tables.items():
        if 'pca' not in mode:
            continue
        sd = t.double().std(dim=0, unbiased=False)
        allunit = bool((( sd - 1.0).abs() < 1e-6).all())
        check('%s: PCA NOT whitened' % mode, not allunit,
              'stds PC1=%.4f..PCk=%.4f ratio=%.3f (whitening would force all 1.0)'
              % (sd[0], sd[-1], sd[0] / sd[-1] if sd[-1] > 0 else float('inf')))

    print('')
    bad = [n for n, ok, _ in results if not ok]
    if bad:
        print('W6: %d FAILED -- %s' % (len(bad), '; '.join(bad)))
        return 1
    print('W6: ALL PASS (%d checks)' % len(results))
    return 0


if __name__ == '__main__':
    sys.exit(main())
