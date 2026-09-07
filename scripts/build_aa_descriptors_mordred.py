#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ITEM 18 -- the REAL Mordred pipeline for the DeepEF amino-acid descriptor matrix.

Ofir Ezrielev's method, run as he ran it, not the 14-column hand fallback:

    chiral SMILES (L-amino acid, neutral free form)
      -> Mordred Calculator(descriptors, ignore_3D=True)      ~1613 raw descriptors
         NOTE (verified against the thesis): Ofir reports 1826, which is 1613 2D + 213 3D.
         That total is only coherent if 3D descriptors WERE requested. With ignore_3D=True
         the ceiling is 1613 and his 1826->1280 missing-value drop is unreproducible -- 3D
         descriptors return NaN without an embedded conformer, which is what his drop was.
         We keep ignore_3D=True deliberately (no conformer generation), so our counts differ
         from his BY DESIGN. Do not quote 1826 as if this script produced it.
      -> filter 1: drop every column with ANY missing value   -> ~1280
      -> filter 2: keep columns with enough unique values     -> ~654
      -> normalise (z-score across the residues)
      -> data/aa_descriptors_mordred.csv, with a provenance header

His central result -- and the reason this item exists at all -- is that a model trained on
canonical residues predicts NON-canonical effects, because the physicochemical space is
continuous. That property is what a hand table of 14 columns cannot buy: it is not the
column count, it is that any molecule with a SMILES string gets a vector on the SAME axes.

--------------------------------------------------------------------------------------
THE ONE DELIBERATE DEVIATION, stated here, in the CSV header, and in REPORT.md
--------------------------------------------------------------------------------------
Ofir kept descriptors with >= 40 unique values across his 58 residues (canonical plus
non-canonical). With 20 residues the maximum possible unique count is 20, so >= 40 is
unsatisfiable. The threshold scales to 15/20 = 0.750 -- a descriptor must distinguish at
roughly two thirds of the alphabet -- 40/58 = 0.690, which scales to 13.8, so a faithful
proportional threshold would be 13 or 14, NOT 15. We keep 15 as OUR choice. Note also that
the thesis gives NO justification for 40 at all, so there is no ratio to preserve; and 2D
Mordred cannot distinguish D from L, so his effective alphabet is well under 58. Both the original and the scaled threshold are
recorded in the CSV header. --min_unique overrides it; --unique_frac derives it from a
fraction if you would rather scale Ofir's 40/58 = 0.690 exactly.

--------------------------------------------------------------------------------------
A CORRECTION TO THE SMILES TABLE IN PHASES_5_9_WITH_CODE.md, PHASE 6
--------------------------------------------------------------------------------------
The histidine string printed in the plan --  'H': 'c1cc(nc1)C[C@@H](C(=O)O)N'  -- is
WRONG. Its aromatic ring is c1cc(n c1): four carbons and ONE nitrogen. That is a
pyridine-type ring, and the whole molecule parses as C7H10N2O2. L-histidine is C6H9N3O2:
the side chain is imidazol-4-yl, a five-membered ring with TWO nitrogens, one of them an
NH. The corrected string is

    'H': 'c1c(nc[nH]1)C[C@@H](C(=O)O)N'          (PubChem CID 6274, L-histidine)

This matters concretely and not cosmetically. Mordred would have computed all ~654
descriptors for the wrong molecule: wrong nitrogen count, wrong ring type, wrong H-bond
donor count, wrong autocorrelation and charge descriptors -- and histidine would then have
sat in the wrong place in the space for every downstream arm, silently, because the
nearest-neighbour gate in PHASE 6.4 tests only L, D and F. `--strict_smiles` (default ON)
re-derives the molecular formula from the parsed molecule and refuses to write a matrix if
ANY residue disagrees with its known formula, so this class of error cannot recur.

--------------------------------------------------------------------------------------
WHERE THIS RUNS
--------------------------------------------------------------------------------------
NOT in esm2_env_py38. That environment must stay bit-identical to what the running jobs
see; installing rdkit into it would perturb numpy/pandas under 48 queued calibration runs.
See INSTALL.md -- a separate conda env, or a container. This script runs ONCE, anywhere
rdkit+mordred exist, and its OUTPUT (a CSV) is the committed artifact. Training-time code
reads the CSV and never imports rdkit.

Usage
    python build_aa_descriptors_mordred.py --out data/aa_descriptors_mordred.csv
    python build_aa_descriptors_mordred.py --out data/aa_descriptors_mordred.csv --pca 16
    python build_aa_descriptors_mordred.py --selftest        # no rdkit needed
"""
from __future__ import print_function

import argparse
import datetime
import hashlib
import os
import platform
import re
import sys

# ----------------------------------------------------------------------------------
# Residue order: MUST equal train_utils.AA_MAP, verified alphabetical ACDEFGHIKLMNPQRSTVWY
# in srclive/train_utils.py. Row i of the CSV is the residue with one-hot index i. Any
# reordering silently permutes the descriptor block against the one-hot block.
# ----------------------------------------------------------------------------------
ORDER = list('ACDEFGHIKLMNPQRSTVWY')

# Chiral SMILES, L-amino acids, NEUTRAL free form (not zwitterionic).
# Source: PubChem canonical isomeric SMILES for each L-amino acid.
# 'H' is CORRECTED relative to PHASES_5_9_WITH_CODE.md -- see the module docstring.
SMILES = {
    'A': 'C[C@@H](C(=O)O)N',                       # CID 5950   L-alanine
    'C': 'C([C@@H](C(=O)O)N)S',                    # CID 5862   L-cysteine
    'D': 'C([C@@H](C(=O)O)N)C(=O)O',               # CID 5960   L-aspartic acid
    'E': 'C(CC(=O)O)[C@@H](C(=O)O)N',              # CID 33032  L-glutamic acid
    'F': 'c1ccc(cc1)C[C@@H](C(=O)O)N',             # CID 6140   L-phenylalanine
    'G': 'C(C(=O)O)N',                             # CID 750    glycine (achiral)
    'H': 'c1c(nc[nH]1)C[C@@H](C(=O)O)N',           # CID 6274   L-histidine  <-- CORRECTED
    'I': 'CC[C@H](C)[C@@H](C(=O)O)N',              # CID 6306   L-isoleucine (2 centres)
    'K': 'C(CCN)C[C@@H](C(=O)O)N',                 # CID 5962   L-lysine
    'L': 'CC(C)C[C@@H](C(=O)O)N',                  # CID 6106   L-leucine
    'M': 'CSCC[C@@H](C(=O)O)N',                    # CID 6137   L-methionine
    'N': 'C([C@@H](C(=O)O)N)C(=O)N',               # CID 6267   L-asparagine
    'P': 'C1C[C@H](NC1)C(=O)O',                    # CID 145742 L-proline
    'Q': 'C(CC(=O)N)[C@@H](C(=O)O)N',              # CID 5961   L-glutamine
    'R': 'C(C[C@@H](C(=O)O)N)CNC(=N)N',            # CID 6322   L-arginine
    'S': 'C([C@@H](C(=O)O)N)O',                    # CID 5951   L-serine
    'T': 'C[C@H]([C@@H](C(=O)O)N)O',               # CID 6288   L-threonine (2 centres)
    'V': 'CC(C)[C@@H](C(=O)O)N',                   # CID 6287   L-valine
    'W': 'c1ccc2c(c1)c(c[nH]2)C[C@@H](C(=O)O)N',   # CID 6305   L-tryptophan
    'Y': 'c1cc(ccc1C[C@@H](C(=O)O)N)O',            # CID 6057   L-tyrosine
}

# Known heavy-atom formula of each NEUTRAL free L-amino acid, and the expected number of
# tetrahedral stereocentres. This is the ground truth --strict_smiles checks against.
FORMULA = {
    'A': dict(C=3,  N=1, O=2, S=0), 'C': dict(C=3,  N=1, O=2, S=1),
    'D': dict(C=4,  N=1, O=4, S=0), 'E': dict(C=5,  N=1, O=4, S=0),
    'F': dict(C=9,  N=1, O=2, S=0), 'G': dict(C=2,  N=1, O=2, S=0),
    'H': dict(C=6,  N=3, O=2, S=0), 'I': dict(C=6,  N=1, O=2, S=0),
    'K': dict(C=6,  N=2, O=2, S=0), 'L': dict(C=6,  N=1, O=2, S=0),
    'M': dict(C=5,  N=1, O=2, S=1), 'N': dict(C=4,  N=2, O=3, S=0),
    'P': dict(C=5,  N=1, O=2, S=0), 'Q': dict(C=5,  N=2, O=3, S=0),
    'R': dict(C=6,  N=4, O=2, S=0), 'S': dict(C=3,  N=1, O=3, S=0),
    'T': dict(C=4,  N=1, O=3, S=0), 'V': dict(C=5,  N=1, O=2, S=0),
    'W': dict(C=11, N=2, O=2, S=0), 'Y': dict(C=9,  N=1, O=3, S=0),
}
NSTEREO = dict((a, 1) for a in FORMULA)
NSTEREO['G'] = 0          # glycine is achiral
NSTEREO['I'] = 2          # Ca and Cb
NSTEREO['T'] = 2          # Ca and Cb

# ----------------------------------------------------------------------------------
# W10: THE ALPHABET IS OPEN PAST TWENTY.
#
# Ofir's claim is that a model trained on canonical residues predicts NON-canonical
# effects, because the descriptor space is continuous -- any molecule with a SMILES
# string lands on the SAME axes. That claim is untestable while the table has exactly
# twenty rows, so here are the first five rows past it. They are ordinary PDB chemical
# component codes, the same labels a structure file uses.
#
# All five are NEUTRAL free forms, to match the canonical twenty exactly: the z-score in
# the builder is taken across rows, so one zwitterionic row among twenty neutral ones
# would show up as a genuine descriptor difference that is really a protonation-state
# artefact. The phospho residues are therefore written as the free acid P(=O)(O)O, not
# the physiological dianion.
#
# Every one of them goes through the same two gates as the canonical twenty: the
# rdkit-free formula/stereocentre/ring/paren selftest, and the rdkit formula+stereo gate.
# ----------------------------------------------------------------------------------
NONCANON_SMILES = {
    # phosphoserine -- serine with the side-chain OH phosphorylated. 1 stereocentre (Ca).
    'SEP': 'C([C@@H](C(=O)O)N)OP(=O)(O)O',
    # phosphothreonine -- threonine phosphorylated. 2 stereocentres (Ca, Cb), like T.
    'TPO': 'C[C@H]([C@@H](C(=O)O)N)OP(=O)(O)O',
    # phosphotyrosine -- tyrosine phenol O phosphorylated. 1 stereocentre (Ca).
    'PTR': 'c1cc(ccc1C[C@@H](C(=O)O)N)OP(=O)(O)O',
    # selenomethionine -- methionine with S -> Se. 1 stereocentre (Ca).
    # This is THE canonical non-canonical residue: it is in a large share of the PDB as a
    # phasing aid, so it is the one an open alphabet meets first in real structures.
    'MSE': 'C[Se]CC[C@@H](C(=O)O)N',
    # 4-hydroxyproline (trans-4-hydroxy-L-proline) -- proline with 4-OH.
    # 2 stereocentres (Ca and C4).
    'HYP': 'C1[C@H](CN[C@@H]1C(=O)O)O',
}

# Heavy-atom composition of each NEUTRAL free non-canonical residue. Se is tracked
# separately from S: selenomethionine is NOT methionine with a heavier S, and a checker
# that folded Se into S would pass MSE while silently accepting a methionine SMILES.
NONCANON_FORMULA = {
    'SEP': dict(C=3,  N=1, O=6, S=0, SE=0, P=1),   # C3H8NO6P   phosphoserine
    'TPO': dict(C=4,  N=1, O=6, S=0, SE=0, P=1),   # C4H10NO6P  phosphothreonine
    'PTR': dict(C=9,  N=1, O=6, S=0, SE=0, P=1),   # C9H12NO6P  phosphotyrosine
    'MSE': dict(C=5,  N=1, O=2, S=0, SE=1, P=0),   # C5H11NO2Se selenomethionine
    'HYP': dict(C=5,  N=1, O=3, S=0, SE=0, P=0),   # C5H9NO3    4-hydroxyproline
}
NONCANON_NSTEREO = {'SEP': 1, 'TPO': 2, 'PTR': 1, 'MSE': 1, 'HYP': 2}

# The full open alphabet, canonical twenty FIRST and in AA_MAP order. That ordering is
# load-bearing: aa_descriptors.py guarantees row i == one-hot index i for i < 20, so the
# canonical tensor path stays byte-identical no matter how many rows follow.
ORDER_OPEN = list(ORDER) + ['SEP', 'TPO', 'PTR', 'MSE', 'HYP']

SMILES_OPEN = dict(SMILES)
SMILES_OPEN.update(NONCANON_SMILES)

# FORMULA entries for the canonical twenty carry no SE/P keys; normalise both tables to
# the same key set so one checker handles all rows.
_ELEMENTS = ('C', 'N', 'O', 'S', 'SE', 'P')


def _norm_formula(d):
    return dict((e, int(d.get(e, 0))) for e in _ELEMENTS)


FORMULA_OPEN = dict((a, _norm_formula(FORMULA[a])) for a in ORDER)
FORMULA_OPEN.update(dict((a, _norm_formula(NONCANON_FORMULA[a])) for a in NONCANON_FORMULA))

NSTEREO_OPEN = dict(NSTEREO)
NSTEREO_OPEN.update(NONCANON_NSTEREO)


# Ofir's thresholds, and the scaling of the second one.
OFIR_N_RESIDUES = 58
OFIR_MIN_UNIQUE = 40

# ----------------------------------------------------------------------------------
# SMILES check that needs NO rdkit -- runs in --selftest, anywhere.
# ----------------------------------------------------------------------------------
# W10: Se and P must be tokenised. 'Se' has to be tried BEFORE the single-letter
# alternatives or 'S' would match first and the 'e' would be dropped, which would
# score selenomethionine as methionine and pass.
_TOK = re.compile(r'\[[^\]]*\]|Cl|Br|Se|[BCNOPSFIbcnops]')


def formula_from_smiles_text(smi):
    """Heavy-atom composition parsed straight from the SMILES text.

    W10: counts Se and P as well as C/N/O/S, because the non-canonical rows need them.
    Se is a SEPARATE key from S -- selenomethionine differs from methionine in exactly
    that one atom, so a checker that merged them would validate the wrong molecule.
    """
    c = dict((e, 0) for e in ('C', 'N', 'O', 'S', 'SE', 'P'))
    for t in _TOK.findall(smi):
        if t.startswith('['):
            m = re.search(r'[A-Z][a-z]?|[a-z]', t[1:-1])
            el = m.group(0) if m else ''
        else:
            el = t
        el = el.upper()
        if el in c:
            c[el] += 1
    return c


def _fmt_formula(d):
    return ''.join('%s%d' % (e, d.get(e, 0)) for e in ('C', 'N', 'O', 'S', 'SE', 'P'))


def stereocentres_from_smiles_text(smi):
    """Count tetrahedral markers: collapse @@ to @ first, then count @."""
    return len(re.findall(r'@', smi.replace('@@', '@')))


def selftest_smiles(verbose=True, residues=None, smiles=None, formula=None, nstereo=None):
    """Formula / stereocentre / ring-closure / paren check, no chemistry toolkit.

    W10: parameterised over a residue list so the SAME check runs over the canonical
    twenty and over the non-canonical rows. Defaults reproduce the pre-W10 behaviour
    exactly, so an unchanged call site is an unchanged test.
    """
    residues = list(ORDER) if residues is None else list(residues)
    smiles = SMILES_OPEN if smiles is None else smiles
    formula = FORMULA_OPEN if formula is None else formula
    nstereo = NSTEREO_OPEN if nstereo is None else nstereo
    fails = []
    for aa in residues:
        smi = smiles[aa]
        got, exp = formula_from_smiles_text(smi), _norm_formula(formula[aa])
        ns = stereocentres_from_smiles_text(smi)
        stripped = re.sub(r'\[[^\]]*\]', '', smi)
        digits = {}
        for ch in stripped:
            if ch.isdigit():
                digits[ch] = digits.get(ch, 0) + 1
        ok_ring = all(v % 2 == 0 for v in digits.values())
        depth, ok_par = 0, True
        for ch in smi:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth < 0:
                    ok_par = False
        ok_par = ok_par and depth == 0
        why = []
        if got != exp:
            why.append('formula %s != %s' % (_fmt_formula(got), _fmt_formula(exp)))
        if ns != nstereo[aa]:
            why.append('stereocentres %d != %d' % (ns, nstereo[aa]))
        if not ok_ring:
            why.append('unclosed ring')
        if not ok_par:
            why.append('unbalanced parens')
        if why:
            fails.append((aa, '; '.join(why)))
        if verbose:
            print('  %-3s %-46s %s' % (aa, smi, 'ok' if not why else 'FAIL: ' + '; '.join(why)))
    return fails


# ----------------------------------------------------------------------------------
# The same check again, but through rdkit, on the molecule Mordred will actually see.
# The text check above can be fooled by exotic syntax; this one cannot.
# ----------------------------------------------------------------------------------
def check_mols_rdkit(mols, residues=None, formula=None, nstereo=None):
    """W10: parameterised the same way as selftest_smiles. Defaults are the canonical 20."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    residues = list(ORDER) if residues is None else list(residues)
    formula = FORMULA_OPEN if formula is None else formula
    nstereo = NSTEREO_OPEN if nstereo is None else nstereo
    fails = []
    for aa, m in zip(residues, mols):
        if m is None:
            fails.append((aa, 'SMILES failed to parse'))
            continue
        f = rdMolDescriptors.CalcMolFormula(m)          # e.g. 'C6H9N3O2'
        exp = _norm_formula(formula[aa])
        parsed = dict((e, 0) for e in ('C', 'N', 'O', 'S', 'SE', 'P'))
        for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', f.replace('+', '').replace('-', '')):
            el = el.upper()
            if el in parsed:
                parsed[el] = int(n) if n else 1
        why = []
        if parsed != exp:
            why.append('rdkit formula %s -> %s, expected %s'
                       % (f, _fmt_formula(parsed), _fmt_formula(exp)))
        try:
            centres = Chem.FindMolChiralCenters(m, includeUnassigned=True,
                                                useLegacyImplementation=False)
        except TypeError:
            centres = Chem.FindMolChiralCenters(m, includeUnassigned=True)
        if len(centres) != nstereo[aa]:
            why.append('rdkit stereocentres %d != %d' % (len(centres), nstereo[aa]))
        unassigned = [c for c in centres if c[1] in ('?', None)]
        if unassigned:
            why.append('unassigned stereocentre(s) %s -- chirality lost' % (unassigned,))
        if why:
            fails.append((aa, '; '.join(why)))
    return fails


def _write_csv(path, frame, header_lines, float_fmt='%.10g'):
    """Write '# ...' provenance lines, then the CSV body. pandas spells the newline
    argument differently across versions, so build the body as a string and write it
    ourselves rather than guessing the keyword."""
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d)
    body = frame.to_csv(float_format=float_fmt)
    with open(path, 'w') as f:
        for line in header_lines:
            f.write('# %s\n' % line)
        f.write(body)
    return path


# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='data/aa_descriptors_mordred.csv')
    ap.add_argument('--min_unique', type=int, default=None,
                    help='override the scaled threshold (default: 15 of 20)')
    ap.add_argument('--unique_frac', type=float, default=None,
                    help='derive threshold as ceil(frac*20); 0.690 reproduces Ofir 40/58 exactly')
    ap.add_argument('--pca', type=int, default=0,
                    help='PCA arm: also write an N-component matrix, variance explained reported')
    ap.add_argument('--pca_out', default=None,
                    help='where the PCA matrix goes (default: --out with _pca<N> before .csv)')
    ap.add_argument('--no_strict_smiles', dest='strict_smiles', action='store_false',
                    default=True,
                    help='skip the rdkit formula/stereo gate -- NOT recommended; this is the '
                         'gate that caught the wrong histidine in the plan')
    ap.add_argument('--use_3d', dest='ignore_3d', action='store_false', default=True,
                    help='compute 3D descriptors too -- requires embedded conformers, which this '
                         'script does NOT generate. NOTE: the thesis reports 1826 = 1613 2D '
                         '+ 213 3D, so Ofir most likely did NOT use ignore_3D=True; that is '
                         'our deviation, not his setting.')
    ap.add_argument('--selftest', action='store_true',
                    help='run the no-dependency SMILES check and exit')
    ap.add_argument('--include_noncanonical', action='store_true',
                    help='W10: build the OPEN alphabet -- the canonical 20 followed by '
                         'SEP, TPO, PTR, MSE, HYP. The canonical rows keep rows 0..19 in '
                         'AA_MAP order, so the canonical tensor path is unaffected. Note '
                         'the z-score is then taken over 25 rows, which changes the '
                         'canonical NUMBERS: use a separate --out, never overwrite the '
                         'committed 20-row matrix with this.')
    ap.add_argument('--nproc', type=int, default=1,
                    help='mordred worker processes; 1 is deterministic and 20 molecules is trivial')
    A = ap.parse_args()

    # ---- selftest path: no rdkit, no mordred, no pandas ---------------------------
    if A.selftest:
        res = ORDER_OPEN if A.include_noncanonical else ORDER
        print('SMILES self-test (no rdkit required) -- %d residues%s'
              % (len(res), ' (OPEN alphabet)' if A.include_noncanonical else ''))
        fails = selftest_smiles(residues=res)
        print('')
        if fails:
            for aa, why in fails:
                print('FAIL %s: %s' % (aa, why))
            print('SELFTEST FAILED (%d of %d residues)' % (len(fails), len(res)))
            return 1
        print('SELFTEST PASSED: %d/%d SMILES match their known heavy-atom formula, '
              'stereocentre count, ring closure and parenthesis balance.'
              % (len(res), len(res)))
        return 0

    # ---- threshold -----------------------------------------------------------------
    # W10: everything below runs over RESIDUES, which is ORDER or ORDER_OPEN.
    RESIDUES = ORDER_OPEN if A.include_noncanonical else ORDER
    n_res = len(RESIDUES)
    if A.min_unique is not None:
        min_unique = A.min_unique
        thr_note = 'set explicitly via --min_unique'
    elif A.unique_frac is not None:
        import math
        min_unique = int(math.ceil(A.unique_frac * n_res))
        thr_note = 'derived from --unique_frac %.4f -> ceil(%.4f * %d) = %d' % (
            A.unique_frac, A.unique_frac, n_res, min_unique)
    else:
        min_unique = 15
        thr_note = ('scaled from Ofir %d/%d (=%.3f) up to %d/%d (=%.3f), i.e. a descriptor '
                    'must distinguish at least three quarters of the alphabet'
                    % (OFIR_MIN_UNIQUE, OFIR_N_RESIDUES,
                       OFIR_MIN_UNIQUE / float(OFIR_N_RESIDUES),
                       15, n_res, 15 / float(n_res)))

    # ---- imports that need the special env -----------------------------------------
    try:
        import numpy as np
        import pandas as pd
    except ImportError as e:
        sys.stderr.write('numpy/pandas missing: %s\n' % e)
        return 2
    try:
        import rdkit
        from rdkit import Chem, RDLogger
        import mordred
        from mordred import Calculator, descriptors
    except ImportError as e:
        sys.stderr.write(
            'rdkit/mordred unavailable (%s).\n'
            'This script needs them. Do NOT install them into esm2_env_py38 -- that env must\n'
            'stay bit-identical to what the queued jobs see. Follow INSTALL.md: create the\n'
            'separate env `mordred_env` (or use the container), activate it, and re-run.\n'
            'To check the SMILES table without any of this: --selftest\n' % e)
        return 2
    RDLogger.DisableLog('rdApp.*')

    rdkit_v = rdkit.__version__
    mordred_v = getattr(mordred, '__version__', 'unknown')

    # ---- build molecules and GATE them ---------------------------------------------
    mols = [Chem.MolFromSmiles(SMILES_OPEN[a]) for a in RESIDUES]
    bad = [a for a, m in zip(RESIDUES, mols) if m is None]
    if bad:
        sys.stderr.write('SMILES failed to parse: %s\n' % ', '.join(bad))
        return 3
    if A.strict_smiles:
        fails = check_mols_rdkit(mols, residues=RESIDUES)
        if fails:
            sys.stderr.write('\nSMILES GATE FAILED -- refusing to write a descriptor matrix.\n')
            for aa, why in fails:
                sys.stderr.write('  %s: %s\n' % (aa, why))
            sys.stderr.write('This is the gate that caught the wrong histidine in the plan.\n'
                             'Fix the SMILES; do not pass --no_strict_smiles to get past it.\n')
            return 3
        print('SMILES gate: %d/%d residues match formula, stereocentre count and assignment.'
              % (len(RESIDUES), len(RESIDUES)))

    # ---- Mordred -------------------------------------------------------------------
    calc = Calculator(descriptors, ignore_3D=A.ignore_3d)
    n_registered = len(calc.descriptors)
    print('Mordred: %d descriptors registered (ignore_3D=%s); computing over %d residues...'
          % (n_registered, A.ignore_3d, len(mols)))
    df = calc.pandas(mols, nproc=A.nproc, quiet=False)
    df.index = RESIDUES
    n_raw = df.shape[1]

    # Mordred returns Error/Missing OBJECTS, not NaN, for descriptors it could not compute.
    # to_numeric(errors='coerce') is what turns them into NaN so filter 1 can see them.
    # Without this line dropna() removes nothing and filter 1 is silently a no-op.
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(axis=1, how='any')                    # Ofir filter 1
    n_nona = df.shape[1]

    df = df.loc[:, df.nunique() >= min_unique]           # Ofir filter 2, scaled
    n_keep = df.shape[1]
    if n_keep == 0:
        sys.stderr.write('every descriptor was filtered out -- threshold %d is too high\n'
                         % min_unique)
        return 4

    # ---- normalise -----------------------------------------------------------------
    # Population std (ddof=0). With n=20 the ddof choice rescales every column by the same
    # constant sqrt(19/20), so it cannot change the Euclidean-distance RANKING the
    # nearest-neighbour gate tests -- but it is recorded so the matrix is reproducible.
    sd = df.std(ddof=0).replace(0, 1.0)
    dfn = ((df - df.mean()) / sd).astype('float64')

    # ---- provenance ----------------------------------------------------------------
    stamp = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    smiles_hash = hashlib.sha256(
        ''.join('%s=%s;' % (a, SMILES_OPEN[a]) for a in RESIDUES).encode('utf-8')).hexdigest()[:16]
    prov = [
        'DeepEF amino-acid physicochemical descriptor matrix -- REAL Mordred pipeline',
        'generated_utc=%s' % stamp,
        'generator=build_aa_descriptors_mordred.py (ITEM 18, worker mordred)',
        'method=Ofir Ezrielev: chiral SMILES -> Mordred -> dropna -> unique-value filter -> z-score',
        'python=%s' % platform.python_version(),
        'platform=%s' % platform.platform(),
        'rdkit=%s' % rdkit_v,
        'mordred=%s' % mordred_v,
        'numpy=%s pandas=%s' % (np.__version__, pd.__version__),
        'smiles_source=PubChem canonical isomeric SMILES, neutral free L-amino acids '
        '(NOT zwitterionic)',
        'smiles_sha256_16=%s' % smiles_hash,
        'smiles_correction=histidine corrected relative to the PHASES_5_9_WITH_CODE.md PHASE 6 '
        'table. That table has c1cc(nc1)C[C@@H](C(=O)O)N, which parses as C7H10N2O2 '
        '(pyridine-type ring, one ring N). L-histidine is C6H9N3O2 (imidazol-4-yl, two ring N, '
        'one an NH). Corrected to c1c(nc[nH]1)C[C@@H](C(=O)O)N, PubChem CID 6274.',
        'smiles_gate=rdkit molecular formula + stereocentre count + stereo assignment; %s'
        % (('ENFORCED, %d/%d pass' % (len(RESIDUES), len(RESIDUES))) if A.strict_smiles
           else 'SKIPPED via --no_strict_smiles'),
        'mordred_config=Calculator(descriptors, ignore_3D=%s), nproc=%d'
        % (A.ignore_3d, A.nproc),
        'residue_order=' + ''.join(ORDER) + ' (train_utils.AA_MAP alphabetical) '
        + ('then NON-CANONICAL ' + ','.join(RESIDUES[len(ORDER):]) + ' '
           if len(RESIDUES) > len(ORDER) else '') + 
        '(row i == one-hot index i)',
        'n_residues=%d' % n_res,
        'raw_descriptors=%d (registered by Mordred: %d)' % (n_raw, n_registered),
        'filter1=drop columns with ANY missing value: %d -> %d (removed %d)'
        % (n_raw, n_nona, n_raw - n_nona),
        'filter2=keep columns with nunique >= %d: %d -> %d (removed %d)'
        % (min_unique, n_nona, n_keep, n_nona - n_keep),
        'DEVIATION=Ofir kept descriptors with nunique >= %d across %d residues. With %d '
        'residues the maximum possible unique count is %d, so >= %d is unsatisfiable. '
        'Threshold %s.'
        % (OFIR_MIN_UNIQUE, OFIR_N_RESIDUES, n_res, n_res, OFIR_MIN_UNIQUE, thr_note),
        'threshold_used=%d of %d (%.3f)' % (min_unique, n_res, min_unique / float(n_res)),
        'normalisation=z-score across the %d residues, per column, ddof=0; zero-std columns '
        'divided by 1.0' % n_res,
        'final_shape=%d x %d' % (dfn.shape[0], dfn.shape[1]),
        'NOTE=training-time code reads THIS CSV and never imports rdkit or mordred.',
    ]

    _write_csv(A.out, dfn, prov + ['arm=full (all %d surviving descriptors, z-scored)' % n_keep])
    print('')
    print('wrote %s   shape=%d x %d' % (A.out, dfn.shape[0], dfn.shape[1]))
    print('  raw %d -> dropna %d -> nunique>=%d %d' % (n_raw, n_nona, min_unique, n_keep))

    # ---- PCA arm --------------------------------------------------------------------
    if A.pca:
        X = dfn.values.astype('float64')
        Xc = X - X.mean(axis=0, keepdims=True)
        # SVD of the centred matrix; squared singular values are proportional to the
        # variance each component carries. A 20-row matrix has rank at most 19.
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        rank_cap = min(Xc.shape[0] - 1, Xc.shape[1])
        k = min(A.pca, rank_cap)
        var = S ** 2
        total = float(var.sum())
        evr = var / total if total > 0 else np.zeros_like(var)
        cum = float(evr[:k].sum())
        Z = U[:, :k] * S[:k]
        # DO NOT WHITEN. Dividing each PC by its own std forces every component to unit
        # variance, which erases the eigenvalue spectrum -- and that spectrum IS the
        # chemistry: PC1 carrying ~10x the variance of PC8 is a real statement about the
        # residue alphabet, not an artifact to be normalised away. Whitening would make an
        # essentially-noise trailing component shout as loudly as the dominant size/polarity
        # axis. The projection is kept on its natural scale; one global factor below puts
        # the block in the same numeric range as the z-scored arm without touching the
        # RELATIVE variance between components.
        gscale = float(Z.std(ddof=0))          # ONE scalar over the whole block
        if gscale == 0.0:
            gscale = 1.0
        Zs = Z / gscale
        pca_df = pd.DataFrame(Zs, index=ORDER,
                              columns=['PC%d' % (i + 1) for i in range(k)])
        comp_sd = Zs.std(axis=0, ddof=0)
        per = ', '.join('PC%d=%.4f' % (i + 1, evr[i]) for i in range(k))
        cumlist = ', '.join('PC1..%d=%.4f' % (i + 1, float(evr[:i + 1].sum()))
                            for i in range(k))
        pca_extra = [
            'arm=pca%d' % k,
            'pca_requested=%d, delivered=%d (rank cap = min(n_residues-1, n_cols) = %d)'
            % (A.pca, k, rank_cap),
            "pca_note=PCA IS OUR ADDITION, NOT THE THESIS METHOD. Ofir does no "
                     "dimensionality reduction of any kind: PCA, principal component and "
                     "SVD appear nowhere in the thesis, and his 654 features enter the CNN "
                     "as 654 channels. Do not attribute this projection to him. "
                     'pca_input=the %d z-scored descriptors above, so this is PCA on the correlation '
            'structure, not on raw units' % n_keep,
            'pca_variance_explained_cumulative=%.6f' % cum,
            'pca_variance_explained_per_component=%s' % per,
            'pca_variance_explained_cumulative_curve=%s' % cumlist,
            'pca_scaling=NATURAL COMPONENT SCALE PRESERVED. The projection U[:,:k]*S[:k] is '
            'divided by ONE global scalar (the std over the whole block, %.6f) and NOT by '
            'each component OWN std. Per-component whitening was deliberately NOT applied: '
            'it would set every PC to unit variance and destroy the eigenvalue spectrum, '
            'which is the chemical content of the decomposition (a dominant size/polarity '
            'axis and a long tail of near-noise directions).' % gscale,
            'pca_component_std_after_scaling=%s' % ', '.join(
                'PC%d=%.6f' % (i + 1, comp_sd[i]) for i in range(k)),
            'pca_component_std_ratio_PC1_over_PCk=%.4f'
            % (comp_sd[0] / comp_sd[k - 1] if comp_sd[k - 1] > 0 else float('inf')),
        ]
        pca_path = A.pca_out or (re.sub(r'\.csv$', '', A.out) + '_pca%d.csv' % k)
        _write_csv(pca_path, pca_df, prov + pca_extra)
        print('wrote %s   shape=%d x %d   variance explained=%.4f'
              % (pca_path, pca_df.shape[0], pca_df.shape[1], cum))
        print('  per component: %s' % per)
        if k < A.pca:
            print('  NOTE: only %d components exist. A %d-row matrix has rank at most %d, '
                  'so a PCA to %d components is not defined.'
                  % (k, n_res, n_res - 1, A.pca))

    print('')
    print('Next, and it is not optional:')
    print('  python verify_descriptors_mordred.py %s' % A.out)
    print('  L must land among {I,V,M}; D among {E,N}; F among {Y,W}.')
    print('  If aspartate lands near leucine the matrix is wrong and no training fixes it.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
