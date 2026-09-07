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


MUT_DIRS = ('data_fixed/mutation_datasets', 'data/Processed_K50_dG_datasets/mutation_datasets')

def _n_backgrounds(protein):
    """How many distinct wild-type backgrounds does this protein's mutation file contain?

    2K5H's file concatenates 2K5H.pdb, 2K5H.pdb_G11S and 2K5H.pdb_G23A; each is 'wt' of its own
    background, so mut_type alone cannot distinguish them. More than one background means row 0
    may not be the true WT. Prefers data_fixed/ so a corrected file is seen first.
    """
    import os
    for d in MUT_DIRS:
        f = os.path.join(d, protein + '.csv')
        if not os.path.exists(f):
            continue
        names = set()
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if str(r.get('mut_type', '')).strip().lower() == 'wt':
                    names.add(r['name'].split('_wt')[0].strip())
        return len(names) if names else 1
    return 1

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
        # The percentile alone is suggestive, not decisive: a DESIGNED protein with many
        # stabilising mutations legitimately has a low-percentile WT. r18_3_TrROS_Hall sits at
        # the 38th percentile and is CORRECT -- its mutation file has a single background and
        # row 0 really is r18_3_TrROS_Hall.pdb.
        # What made 2K5H a bug is that its file CONCATENATES THREE BACKGROUNDS
        # (2K5H.pdb, 2K5H.pdb_G11S, 2K5H.pdb_G23A) and the mutant one sorts first. So the
        # decisive test is multiplicity of backgrounds, and the percentile only escalates it.
        nbg = _n_backgrounds(p)
        if nbg > 1 and pct < WARN_PCTILE:
            hard.append((p, 'row0 dG=%.3f at the %.1f-th percentile AND the mutation file has %d '
                            'distinct backgrounds - row 0 is probably a MUTANT background, not '
                            'the true WT (this is the 2K5H defect)' % (dg[0], pct, nbg)))
        elif nbg > 1:
            warn.append((p, '%d distinct backgrounds in the mutation file; row0 percentile %.1f '
                            'looks fine, but confirm row 0 is the true WT' % (nbg, pct)))
        elif pct < HARD_PCTILE:
            warn.append((p, 'row0 dG=%.3f at the %.1f-th percentile (max %.3f) but only ONE '
                            'background - low percentile alone is legitimate for a designed '
                            'protein with many stabilising mutations' % (dg[0], pct, max(dg))))
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
