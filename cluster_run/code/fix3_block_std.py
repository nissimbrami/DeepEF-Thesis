"""FIX 3 — is the slope term measuring a protein's spread, or a contiguous block's?

Lever C (--slope_weight) penalises |std(pred_ddG) - std(true_ddG)|, computed over ONE
mini-batch:

    for j in range(0, batch['prott5'].size(1), self.mini_batch_size):

The loop walks contiguous blocks and the variants are NOT shuffled.  If the mutation CSV
is ordered by position -- which is usual -- each block covers a narrow window of the
sequence, whose true ddG spread is systematically narrower than the whole protein's.  The
term would then calibrate the model to a block's spread rather than a protein's, which is
under-dispersion: the exact failure lever C was written to fix.

Decision rule (FIXES.md FIX 3):
    ratio >= 0.9  -> blocks are representative, the term is fine as written
    ratio <= 0.7  -> real; add shuffling behind its own default-off flag, as a 5th factor

Do NOT add shuffling now.  It changes the training regime for every run and would break
comparability with the pilots in flight and the reference runs already banked.

Reports the distribution across proteins, not one number, and separately reports whether
the CSVs are in fact position-ordered -- the whole concern rests on that.
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd


def position_of(mut_type):
    """Extract the residue index from a mut_type like 'A23G'.  None if unparseable."""
    if not isinstance(mut_type, str):
        return None
    m = re.match(r'^[A-Za-z](\d+)[A-Za-z]$', mut_type.strip())
    return int(m.group(1)) if m else None


def monotonicity(positions):
    """Fraction of consecutive pairs that are non-decreasing.  1.0 = fully sorted."""
    p = [x for x in positions if x is not None]
    if len(p) < 2:
        return float('nan')
    a = np.asarray(p)
    return float(np.mean(a[1:] >= a[:-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mut_root', default='./data/Processed_K50_dG_datasets/mutation_datasets')
    ap.add_argument('--mb', type=int, default=64, help='mini_batch_size used in training')
    ap.add_argument('--min_variants', type=int, default=0,
                    help='skip proteins with fewer than mb*2 variants unless overridden')
    ap.add_argument('--out', default='results/fix3_block_std.json')
    args = ap.parse_args()

    mb = args.mb
    need = args.min_variants if args.min_variants > 0 else mb * 2

    files = sorted(glob.glob(os.path.join(args.mut_root, '*.csv')))
    print('[fix3] %d mutation CSVs under %s' % (len(files), args.mut_root))

    rows = []
    skipped = []
    for f in files:
        name = os.path.basename(f)[:-4]
        try:
            df = pd.read_csv(f)
            df = df[~df['mut_type'].astype(str).str.contains('ins|del', na=False)]
            df = df[~df['mut_type'].astype(str).str.contains(':', na=False)]
            df = df.reset_index(drop=True)

            # WS-1 convention: row 0 is the wild type, excluded from the spread.
            d = pd.to_numeric(df['ddG_ML'].iloc[1:], errors='coerce').values
            keep = ~np.isnan(d)
            d = d[keep]
            if d.size < need:
                skipped.append((name, 'only %d usable variants (need %d)' % (d.size, need)))
                continue

            prot_std = float(np.std(d, ddof=0))
            blocks = [float(np.std(d[i:i + mb], ddof=0))
                      for i in range(0, d.size - mb + 1, mb)]
            if not blocks or prot_std == 0:
                skipped.append((name, 'no full block or zero protein std'))
                continue

            pos = [position_of(m) for m in df['mut_type'].iloc[1:][keep].tolist()]
            rows.append({
                'name': name,
                'n_variants': int(d.size),
                'n_blocks': len(blocks),
                'protein_std': prot_std,
                'mean_block_std': float(np.mean(blocks)),
                'ratio': float(np.mean(blocks) / prot_std),
                'position_monotonicity': monotonicity(pos),
            })
        except Exception as exc:                       # noqa: BLE001
            skipped.append((name, repr(exc)))

    if not rows:
        print('[fix3] FATAL: no protein produced a ratio')
        raise SystemExit(2)

    r = np.array([x['ratio'] for x in rows])
    mono = np.array([x['position_monotonicity'] for x in rows], dtype=float)
    mono = mono[~np.isnan(mono)]

    summary = {
        'n_proteins': len(rows),
        'n_skipped': len(skipped),
        'mini_batch_size': mb,
        'ratio_median': float(np.median(r)),
        'ratio_mean': float(np.mean(r)),
        'ratio_q25': float(np.percentile(r, 25)),
        'ratio_q75': float(np.percentile(r, 75)),
        'ratio_min': float(np.min(r)),
        'ratio_max': float(np.max(r)),
        'frac_below_0.7': float(np.mean(r <= 0.7)),
        'frac_above_0.9': float(np.mean(r >= 0.9)),
        'position_monotonicity_median': (float(np.median(mono)) if mono.size else None),
        'position_ordered': (bool(np.median(mono) > 0.95) if mono.size else None),
    }

    med = summary['ratio_median']
    if med >= 0.9:
        verdict = ('ratio median %.3f >= 0.9: blocks are representative of the protein. '
                   'The slope term is fine as written. Record and move on.' % med)
        action = 'none'
    elif med <= 0.7:
        verdict = ('ratio median %.3f <= 0.7: the concern is real. A mini-batch measures a '
                   'systematically narrower spread than the protein, so lever C would '
                   'calibrate toward under-dispersion.' % med)
        action = ('add shuffling behind its own default-off flag and treat it as a fifth '
                  'factor; do NOT change the regime of runs already in flight')
    else:
        verdict = ('ratio median %.3f falls between the two thresholds (0.7, 0.9): '
                   'neither decision rule fires. Report the distribution and decide with '
                   'Nissim rather than picking a side.' % med)
        action = 'report the distribution, do not change the code unilaterally'

    print('\n[fix3] proteins=%d  skipped=%d  mb=%d' % (len(rows), len(skipped), mb))
    print('[fix3] ratio  median=%.3f  IQR=[%.3f, %.3f]  min=%.3f  max=%.3f'
          % (med, summary['ratio_q25'], summary['ratio_q75'],
             summary['ratio_min'], summary['ratio_max']))
    print('[fix3] fraction <=0.7: %.2f   fraction >=0.9: %.2f'
          % (summary['frac_below_0.7'], summary['frac_above_0.9']))
    print('[fix3] position-ordered (median monotonicity %.3f): %s'
          % (summary['position_monotonicity_median'] or float('nan'),
             summary['position_ordered']))
    print('\n[fix3] VERDICT: %s' % verdict)
    print('[fix3] ACTION : %s' % action)

    out = {'schema': 'fix3.v1', 'summary': summary, 'verdict': verdict,
           'action': action, 'per_protein': rows, 'skipped': skipped}
    d = os.path.dirname(os.path.abspath(args.out))
    if d and not os.path.isdir(d):
        os.makedirs(d)
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('[fix3] wrote %s' % args.out)


if __name__ == '__main__':
    main()
