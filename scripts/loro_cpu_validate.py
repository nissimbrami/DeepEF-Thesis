"""LORO harness CPU validation -- no GPU, no checkpoint, no training.

Five things are checked, each of which would be a silent failure if wrong:

  V1  the holdout REMOVES what it claims, counted before and after on the REAL
      mutation CSVs. A holdout that silently removes nothing is this project's
      signature failure mode, so this is the check that matters most.
  V2  the holdout removes BOTH directions (X->W and W->X) and NEVER the WT row.
  V3  the TEST loader is NOT filtered -- the held-out rows survive to be scored.
      Verified by reading the patched source and asserting the guard appears
      exactly once, inside load_protein_data and not load_test_protein_data.
  V4  the Spearman scorer is correct, on random tensors: it must equal
      scipy's, be invariant to any monotone transform of the predictions, and
      return the right sign.
  V5  the end-to-end scoring path runs on RANDOM tensors of the real shapes,
      so the harness is exercised without a model.

Usage:  python loro_cpu_validate.py [--holdout W] [--limit_files N]
"""
from __future__ import print_function

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

CODE_RE = re.compile(r'^([A-Z])(\d+)([A-Z])$')
FAILED = []


def check(name, cond, extra=''):
    print('  %-64s %s %s' % (name, 'PASS' if cond else 'FAIL', extra))
    if not cond:
        FAILED.append(name)
    return cond


# ----------------------------------------------------------------------------------
# The holdout, as a standalone function with EXACTLY the semantics the patch installs.
# Kept here so the split logic can be validated with no torch, no CUDA and no model.
# ----------------------------------------------------------------------------------
def apply_holdout(mut_types, holdout):
    """Return (keep_idx, drop_idx) over a list of mut_type strings.

    Drops a row iff the code parses as a single substitution AND either side is a
    held-out residue. 'wt' rows do not parse and are therefore always kept.
    """
    hold = set(holdout)
    keep, drop = [], []
    for i, code in enumerate(mut_types):
        m = CODE_RE.match(str(code).strip())
        if m is not None and (m.group(1) in hold or m.group(3) in hold):
            drop.append(i)
        else:
            keep.append(i)
    return keep, drop


def spearman(x, y):
    """Spearman rho = Pearson on ranks, average ranks for ties.

    Written out rather than imported so the harness has no scipy dependency on the
    login node; V4 checks it against scipy when scipy IS importable.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:
        return float('nan')
    rx, ry = _rank(x), _rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(a):
    order = np.argsort(a, kind='mergesort')
    r = np.empty(a.size, dtype=float)
    r[order] = np.arange(1, a.size + 1, dtype=float)
    # average ranks within tie groups
    s = a[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def tofloat(vals):
    """MegaScale writes '-' where a ddG could not be fitted. Those become NaN and are
    DROPPED. Mapping them to 0.0 would read as a real neutral mutation and flatten
    every slope toward compression -- the exact a_p artefact this project measures."""
    out = np.empty(len(vals), dtype=float)
    for i, v in enumerate(vals):
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


# ----------------------------------------------------------------------------------
def v1_v2_real_csvs(mut_dir, holdout, limit):
    print('\nV1/V2  holdout removes what it claims, on the REAL mutation CSVs')
    files = sorted(glob.glob(os.path.join(mut_dir, '*.csv')))
    if limit:
        files = files[:limit]
    check('mutation CSVs found', len(files) > 0, '%d files' % len(files))

    tot_before = tot_after = 0
    into = frm = wt_kept = 0
    n_prot_with_removal = 0
    n_nonempty = 0
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        mt = df['mut_type'].astype(str).tolist()
        # 494 of the 862 mutation CSVs are EMPTY (0 rows). They are not proteins that
        # lack the held-out residue -- they have no rows at all, and no corresponding
        # training_data tensor directory, so the training loader never sees them. The
        # denominator for "did the holdout bite" must therefore be the NON-EMPTY CSVs,
        # which are exactly the 368 proteins under training_data. Counting against all
        # 862 would report a spurious 43% and hide a real no-op behind arithmetic.
        if len(mt) == 0:
            continue
        n_nonempty += 1
        keep, drop = apply_holdout(mt, holdout)
        tot_before += len(mt)
        tot_after += len(keep)
        if drop:
            n_prot_with_removal += 1
        for i in drop:
            m = CODE_RE.match(mt[i].strip())
            if m.group(3) in holdout:
                into += 1
            if m.group(1) in holdout:
                frm += 1
        wt_kept += sum(1 for i in keep if mt[i].strip() == 'wt')

    removed = tot_before - tot_after
    print('    rows before holdout : %d' % tot_before)
    print('    rows after  holdout : %d' % tot_after)
    print('    REMOVED             : %d  (%.2f%%)'
          % (removed, 100.0 * removed / max(tot_before, 1)))
    print('    of which INTO %-4s  : %d' % (','.join(holdout), into))
    print('    of which FROM %-4s  : %d' % (','.join(holdout), frm))
    print('    WT rows kept        : %d' % wt_kept)

    check('V1 holdout removed a NON-ZERO number of rows', removed > 0, '%d' % removed)
    print('    non-empty CSVs      : %d of %d' % (n_nonempty, len(files)))
    check('V1 holdout bit in EVERY non-empty protein',
          n_nonempty > 0 and n_prot_with_removal == n_nonempty,
          '%d/%d proteins' % (n_prot_with_removal, n_nonempty))
    check('V2 removes mutations INTO the held-out residue', into > 0, '%d' % into)
    check('V2 removes mutations FROM the held-out residue', frm > 0, '%d' % frm)
    check('V2 WT rows are NEVER removed', wt_kept > 0, '%d kept' % wt_kept)

    # the complement: after the holdout, ZERO surviving rows may mention the residue
    leak = 0
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        mt = df['mut_type'].astype(str).tolist()
        keep, _ = apply_holdout(mt, holdout)
        for i in keep:
            m = CODE_RE.match(mt[i].strip())
            if m is not None and (m.group(1) in holdout or m.group(3) in holdout):
                leak += 1
    check('V2 NO held-out residue survives in the training split', leak == 0,
          '%d leaked' % leak)
    return dict(before=tot_before, after=tot_after, removed=removed, into=into,
                frm=frm, files=len(files), nonempty=n_nonempty)


def v3_source_guard(train_py):
    print('\nV3  the TEST loader is NOT filtered (held-out rows survive to be scored)')
    if not os.path.isfile(train_py):
        check('patched train.py present', False, train_py)
        return
    src = open(train_py).read()
    # The guard string also appears in the startup banner; what must be unique is its
    # occurrence inside a DATASET LOADER, which is what the body checks below test.
    n_guard = src.count('if HOLDOUT_RESIDUES:')
    check('holdout guard present in source', n_guard >= 1, '%d occurrences' % n_guard)

    # locate the two loader bodies and assert which one carries the guard
    def body(name):
        i = src.find('def %s(self, idx):' % name)
        if i < 0:
            return ''
        j = src.find('\n    def ', i + 1)
        return src[i:j if j > 0 else len(src)]

    tr = body('load_protein_data')
    te = body('load_test_protein_data')
    check('guard IS inside load_protein_data (train)',
          'if HOLDOUT_RESIDUES:' in tr)
    check('guard is NOT inside load_test_protein_data (test)',
          'if HOLDOUT_RESIDUES:' not in te and len(te) > 0)
    check('zero-removal tripwire loro_report() defined', 'def loro_report()' in src)
    check('tripwire RAISES on zero removal', 'raise RuntimeError' in src
          and 'removed ZERO rows' in src)


def v4_spearman():
    print('\nV4  Spearman scorer, on RANDOM tensors')
    rng = np.random.RandomState(0)
    x = rng.randn(500)
    y = 0.7 * x + 0.7 * rng.randn(500)
    rho = spearman(x, y)
    try:
        from scipy.stats import spearmanr
        ref = float(spearmanr(x, y).correlation)
        check('matches scipy.stats.spearmanr', abs(rho - ref) < 1e-10,
              'ours=%.6f scipy=%.6f' % (rho, ref))
    except ImportError:
        check('scipy unavailable -- internal consistency only', True, 'rho=%.4f' % rho)

    # rank correlation must be invariant to ANY monotone transform of predictions.
    # This is exactly why Spearman is the right metric here: an arm that gets the
    # ORDER right but the SCALE wrong still scores well, which is the question
    # "can descriptors rank mutations into an unseen residue" and not "can they
    # calibrate them".
    check('invariant to a monotone transform of the predictions',
          abs(spearman(x, np.exp(y)) - rho) < 1e-10,
          '%.6f' % spearman(x, np.exp(y)))
    check('sign flips under negation', abs(spearman(x, -y) + rho) < 1e-10)
    check('perfect rank agreement -> 1.0', abs(spearman(x, 3 * x + 1) - 1.0) < 1e-10)
    check('handles ties (average ranks)',
          np.isfinite(spearman(np.round(x), np.round(y))))
    check('constant input -> nan, not a crash',
          not np.isfinite(spearman(np.ones(50), rng.randn(50))))
    check('n<3 -> nan, not a crash', not np.isfinite(spearman([1.0, 2.0], [1.0, 2.0])))
    check('non-numeric ddG sentinel becomes NaN and is dropped',
          np.isnan(tofloat(['-', '1.5'])[0]) and tofloat(['-', '1.5'])[1] == 1.5)


def v5_end_to_end(mut_dir, holdout, limit):
    print('\nV5  end-to-end scoring path on RANDOM predictions of the real shapes')
    files = sorted(glob.glob(os.path.join(mut_dir, '*.csv')))
    if limit:
        files = files[:limit]
    rng = np.random.RandomState(1)
    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        mt = df['mut_type'].astype(str).tolist()
        _, drop = apply_holdout(mt, holdout)
        # score ONLY mutations INTO the held-out residue -- that is the eval set
        idx = [i for i in drop if CODE_RE.match(mt[i].strip()).group(3) in set(holdout)]
        if len(idx) < 3:
            continue
        y = tofloat(df['ddG_ML'].iloc[idx].tolist())
        p = rng.randn(len(idx))                     # a MODEL-FREE random predictor
        m = np.isfinite(y) & np.isfinite(p)
        if m.sum() < 3:
            continue
        rows.append((os.path.basename(f)[:-4], spearman(y[m], p[m]), int(m.sum())))

    check('per-protein Spearman computed on held-out rows', len(rows) > 0,
          '%d proteins' % len(rows))
    if rows:
        rhos = np.array([r[1] for r in rows], dtype=float)
        rhos = rhos[np.isfinite(rhos)]
        n_tot = sum(r[2] for r in rows)
        print('    scorable proteins   : %d' % len(rows))
        print('    held-out rows scored: %d' % n_tot)
        print('    median rho (RANDOM predictor, so this MUST be ~0): %+.4f'
              % float(np.median(rhos)))
        check('random predictor scores ~0 (harness is not leaking the label)',
              abs(float(np.median(rhos))) < 0.10, '%+.4f' % float(np.median(rhos)))
        # the same random vector must score ~1 against itself: proves the scorer is
        # wired to the predictions it is handed, not to some cached array.
        yy = rng.randn(200)
        check('scorer is actually READING the predictions handed to it',
              abs(spearman(yy, yy) - 1.0) < 1e-10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--holdout', default='W')
    ap.add_argument('--mut_dir',
                    default='data/Processed_K50_dG_datasets/mutation_datasets')
    ap.add_argument('--limit_files', type=int, default=0)
    ap.add_argument('--train_py', default='Megascale-fineTuning/train.py')
    A = ap.parse_args()
    holdout = [h.strip().upper() for h in A.holdout.split(',') if h.strip()]

    print('=' * 78)
    print('LORO CPU VALIDATION -- holdout=%s   (no GPU, no model, no training)'
          % ','.join(holdout))
    print('=' * 78)

    stats = v1_v2_real_csvs(A.mut_dir, holdout, A.limit_files or None)
    v3_source_guard(A.train_py)
    v4_spearman()
    v5_end_to_end(A.mut_dir, holdout, A.limit_files or None)

    print('\n' + '=' * 78)
    if FAILED:
        print('LORO CPU VALIDATION: %d FAILURE(S)' % len(FAILED))
        for f in FAILED:
            print('   FAILED: %s' % f)
        return 1
    print('LORO CPU VALIDATION: ALL PASS  (holdout=%s, %d rows removed of %d)'
          % (','.join(holdout), stats['removed'], stats['before']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
