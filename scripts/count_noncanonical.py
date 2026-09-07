"""W10 -- how many NON-STANDARD residues are actually in the MegaScale corpus?

The open-alphabet work is only worth doing if the closure was actually costing us
something, so this measures it instead of asserting it. It scans every mutation CSV:
every aa_seq character, and both sides of every parsed mutation code.

Expected answer: ZERO. That is not a disappointment, it is the finding. It means the
open alphabet cannot be validated as a benchmark number ON THIS CORPUS at all, and any
claim about it must be a GENERALISATION claim demonstrated by held-out-residue proxy
(scripts/heldout_residue.py). Writing that down here is the point -- otherwise someone
later reads 'open alphabet' and expects a MegaScale improvement that no amount of code
can produce from data that contains no non-canonical residues.
"""
from __future__ import print_function

import argparse
import collections
import glob
import os
import re
import sys

CANON = set('ACDEFGHIKLMNPQRSTVWY')
NAME_RE = re.compile(r'_([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3})(?:_|$)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mut_dir', default='data/Processed_K50_dG_datasets/mutation_datasets')
    ap.add_argument('--limit', type=int, default=0)
    A = ap.parse_args()

    import pandas as pd
    files = sorted(glob.glob(os.path.join(A.mut_dir, '*.csv')))
    if A.limit:
        files = files[:A.limit]
    if not files:
        sys.stderr.write('no CSVs under %s\n' % A.mut_dir)
        return 2

    seqchars = collections.Counter()
    wt_side = collections.Counter()
    mut_side = collections.Counter()
    n_rows = n_seq = n_codes = 0

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            sys.stderr.write('skip %s: %s\n' % (f, e))
            continue
        n_rows += len(df)
        if 'aa_seq' in df.columns:
            for s in df['aa_seq'].astype(str).unique():
                n_seq += 1
                seqchars.update(s)
        for col in ('mut_type', 'name'):
            if col not in df.columns:
                continue
            for v in df[col].astype(str):
                m = NAME_RE.search(v) or NAME_RE.search('_' + v + '_')
                if m:
                    wt_side[m.group(1)] += 1
                    mut_side[m.group(3)] += 1
                    n_codes += 1
            break

    print('MegaScale non-standard residue census')
    print('  mutation CSVs scanned : %d' % len(files))
    print('  rows                  : %d' % n_rows)
    print('  unique aa_seq scanned : %d' % n_seq)
    print('  mutation codes parsed : %d' % n_codes)
    print('  distinct chars in aa_seq  : %s' % ''.join(sorted(seqchars)))
    print('  distinct WT-side letters  : %s' % ''.join(sorted(wt_side)))
    print('  distinct MUT-side letters : %s' % ''.join(sorted(mut_side)))

    non_seq = sorted((c, n) for c, n in seqchars.items() if c not in CANON)
    non_wt = sorted((c, n) for c, n in wt_side.items() if c not in CANON)
    non_mu = sorted((c, n) for c, n in mut_side.items() if c not in CANON)
    total = sum(n for _, n in non_seq) + sum(n for _, n in non_wt) + sum(n for _, n in non_mu)

    print('')
    print('  NON-CANONICAL in sequences  : %s' % (non_seq or 'NONE'))
    print('  NON-CANONICAL as WT         : %s' % (non_wt or 'NONE'))
    print('  NON-CANONICAL as mutant     : %s' % (non_mu or 'NONE'))
    print('')
    print('  TOTAL NON-STANDARD RESIDUE OCCURRENCES: %d' % total)
    if total == 0:
        print('')
        print('  => ZERO. The open alphabet therefore CANNOT be scored as a benchmark')
        print('     number on MegaScale. It is a GENERALISATION claim, and the way to')
        print('     demonstrate it on this corpus is the held-out-residue proxy:')
        print('       python scripts/heldout_residue.py --hold_out W --reference_probe')
    return 0


if __name__ == '__main__':
    sys.exit(main())
