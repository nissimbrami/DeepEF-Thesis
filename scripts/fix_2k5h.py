"""K1: fix 2K5H's reference row, which is a MUTANT background.

data/Processed_K50_dG_datasets/mutation_datasets/2K5H.csv concatenates THREE backgrounds --
2K5H.pdb (the true WT, dG 4.805), 2K5H.pdb_G11S (dG 1.723) and 2K5H.pdb_G23A -- and
2K5H.pdb_G11S sorts FIRST. The WS-1 convention treats row 0 as the wild type, so every ddG for
this protein is measured against a mutant background and is shifted by ~3.08 kcal/mol.

A constant per-protein label shift IS a per-protein offset by construction, so the
offset-removal oracle "discovers" an offset we introduced. Measured cost: dropping 2K5H alone
removes 36% of the entire offset-removal gain (+0.2373 -> +0.1528 over 22 eval CSVs).

2K5H is the ONLY test protein with multiple background files (checked all 28), so this is an
isolated defect, not a systemic one.

THE FIX: reorder so the TRUE WT row (name == '2K5H.pdb', mut_type == 'wt') is row 0. Nothing is
deleted -- the mutant-background rows are kept, just no longer first. The original is backed up.

--check   report only, change nothing
--apply   write the reordered file (backing up the original first)
"""
import csv, io, os, shutil, sys

SRC = 'data/Processed_K50_dG_datasets/mutation_datasets/2K5H.csv'
BAK = SRC + '.orig_mutantref'
TRUE_WT = '2K5H.pdb'

def load(p):
    with io.open(p, encoding='utf-8', errors='replace') as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames

rows, cols = load(SRC)
print('rows=%d' % len(rows))
print('row0 now : name=%s deltaG=%s mut_type=%s'
      % (rows[0]['name'], str(rows[0]['deltaG'])[:8], rows[0].get('mut_type')))

# the true WT: exact name match AND mut_type wt
idx = [i for i, x in enumerate(rows)
       if x['name'].strip() == TRUE_WT and str(x.get('mut_type', '')).strip().lower() == 'wt']
if not idx:
    print('FATAL: no row with name == %r and mut_type == wt' % TRUE_WT)
    sys.exit(1)
w = idx[0]
print('true WT  : idx=%d name=%s deltaG=%s' % (w, rows[w]['name'], str(rows[w]['deltaG'])[:8]))
shift = float(rows[w]['deltaG']) - float(rows[0]['deltaG'])
print('label shift that this introduced: %+.4f kcal/mol' % shift)

# how many rows belong to each background
import collections
bg = collections.Counter(x['name'].split('_')[0] if '_' in x['name'] else x['name'] for x in rows)
print('background row counts:', dict(list(bg.items())[:4]))

if '--apply' in sys.argv:
    if not os.path.exists(BAK):
        shutil.copy2(SRC, BAK)
        print('backed up -> %s' % BAK)
    new = [rows[w]] + rows[:w] + rows[w+1:]
    assert len(new) == len(rows), 'row count changed'
    with io.open(SRC, 'w', encoding='utf-8', newline='') as f:
        wri = csv.DictWriter(f, fieldnames=cols)
        wri.writeheader()
        wri.writerows(new)
    chk, _ = load(SRC)
    print('APPLIED. row0 is now: name=%s deltaG=%s' % (chk[0]['name'], str(chk[0]['deltaG'])[:8]))
    assert chk[0]['name'].strip() == TRUE_WT, 'reorder failed'
    print('verified: row0 is the true WT, %d rows preserved' % len(chk))
else:
    print('\n(check only -- pass --apply to write)')
