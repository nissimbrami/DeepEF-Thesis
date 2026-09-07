"""Gate: the WT reference row of every protein in an eval CSV must be a plausible wild type.

WHY THIS EXISTS. 2K5H's row 0 was a MUTANT background (dG 1.723) rather than the true wild type
(dG 4.806), so all 1125 of its ddG labels were shifted by ~3.0 kcal/mol. A constant per-protein
label shift IS a per-protein offset by construction, so the offset-removal oracle "discovered" an
offset we had introduced ourselves. Measured cost: dropping 2K5H alone removes 36% of the entire
offset-removal gain (+0.2373 -> +0.1528 over 22 eval CSVs).

Nothing raised. The numbers looked plausible. Only the percentile test exposes it.

THE TESTS
  1. EXACTLY ONE row per protein may have ddG == 0. Two means the WT convention is broken and
     row 0 may not be the reference the ddG column was actually built against.
  2. Row 0's dG must sit at or above its protein's MEDIAN. A wild type is normally among the more
     stable variants; 26 of our 28 sit at the 62nd-99th percentile. Anything below the median is
     reported, and anything below the 25th is a hard failure.
  3. Row 0 must be the row with ddG == 0 (the WS-1 convention this codebase assumes everywhere).

Usage:  python3 scripts/gate_refrow.py [eval_csv ...]     (defaults to all eval_results/abl_*.csv)
Exit 0 if every protein passes, 1 otherwise.
"""
import csv, collections, glob, sys

HARD_PCTILE = 25.0   # below this, the reference row is almost certainly wrong
WARN_PCTILE = 50.0   # below the median is worth reporting

def check(path):
    g = collections.defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            g[r['protein']].append((float(r['deltaG']), float(r['ddG'])))
    hard, warn = [], []
    for p, v in g.items():
        dg = [a for a, _ in v]
        zeros = [i for i, (_, d) in enumerate(v) if d == 0.0]
        pct = 100.0 * sum(1 for x in dg if x < dg[0]) / len(dg)
        if len(zeros) != 1:
            hard.append((p, 'has %d rows with ddG==0 (expected exactly 1)' % len(zeros)))
        elif zeros[0] != 0:
            hard.append((p, 'the ddG==0 row is index %d, not 0' % zeros[0]))
        if pct < HARD_PCTILE:
            hard.append((p, 'row0 dG=%.3f is at the %.1f-th percentile of its own protein '
                            '(max %.3f) - the reference row looks like a MUTANT, not the WT'
                         % (dg[0], pct, max(dg))))
        elif pct < WARN_PCTILE:
            warn.append((p, 'row0 dG=%.3f at %.1f-th percentile (below median)' % (dg[0], pct)))
    return hard, warn

def main():
    paths = sys.argv[1:] or sorted(glob.glob('eval_results/abl_*.csv'))
    if not paths:
        print('no eval CSVs found'); return 1
    bad = 0
    for path in paths:
        try:
            hard, warn = check(path)
        except Exception as e:
            print('%-52s ERROR %s' % (path.split('/')[-1], e)); bad += 1; continue
        name = path.split('/')[-1]
        if hard:
            bad += 1
            print('%-52s FAIL' % name)
            for p, m in hard: print('    [HARD] %-22s %s' % (p, m))
        else:
            print('%-52s PASS%s' % (name, '  (%d warnings)' % len(warn) if warn else ''))
        for p, m in warn: print('    [warn] %-22s %s' % (p, m))
    print('\nREFROW GATE: %s  (%d of %d CSVs failed)'
          % ('ALL PASS' if bad == 0 else 'FAILURES', bad, len(paths)))
    return 0 if bad == 0 else 1

sys.exit(main())
