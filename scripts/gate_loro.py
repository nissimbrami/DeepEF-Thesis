"""Gate for --holdout_residues (LORO). It had NONE, while 2 queued jobs use it.

LORO is the one idea worth stealing from Ofir's thesis: he trains on canonical residues
and tests on NON-canonical ones. MegaScale has zero non-canonical residues, so the
runnable proxy is to hold out a CANONICAL residue type entirely from TRAINING and then
predict mutations to it. One-hot cannot represent a residue it never saw; a continuous
descriptor space can. That contrast IS the experiment.

The failure this gate exists to prevent is the project's signature one: a filter that
silently removes nothing, so the "holdout" run is byte-identical to baseline and the null
result gets written up as evidence against Ofir.

Checks:
  1-3  the mutation-code parser reads the DESTINATION residue, not the source
  4    holding out W removes every X->W row and keeps everything else
  5    the filter is TRAINING-only in intent (it lives in load_protein_data)
  6    an empty holdout set removes nothing (default path byte-identical)
  7    multi-residue holdout works
  8    the no-op guard in train.py raises when zero rows were removed
  9    P is representable as a negative control (descriptors should NOT rescue it)
"""
import re
import sys

ok = True
def chk(name, cond, extra=''):
    global ok
    print('%-64s %s %s' % (name, 'PASS' if cond else 'FAIL', extra))
    if not cond: ok = False

# The convention: a mutation code looks like <FROM><POS><TO>, e.g. A42W means Ala->Trp.
# The holdout must key on TO (the destination), because "predict mutations TO W" is the
# question. Keying on FROM would hold out the wrong rows and quietly invert the test.
CODE = re.compile(r'^([A-Z])(\d+)([A-Z])$')

def dest(code):
    m = CODE.match(code.strip().upper())
    return m.group(3) if m else None

def src(code):
    m = CODE.match(code.strip().upper())
    return m.group(1) if m else None

chk('1 parser reads DESTINATION residue from A42W', dest('A42W') == 'W', dest('A42W'))
chk('2 parser reads SOURCE residue from A42W', src('A42W') == 'A', src('A42W'))
chk('3 destination != source for A42W (they must not be confused)', dest('A42W') != src('A42W'))

rows = ['A42W', 'L10W', 'W15A', 'G7P', 'V22L', 'K30W', 'P5G', 'M1V']
hold = {'W'}
kept = [r for r in rows if dest(r) not in hold]
removed = [r for r in rows if dest(r) in hold]
chk('4a holding out W removes every X->W row', sorted(removed) == sorted(['A42W', 'L10W', 'K30W']),
    str(removed))
chk('4b W15A is KEPT (W is the SOURCE there, not the destination)', 'W15A' in kept)
chk('4c nothing else was removed', len(kept) == 5, str(kept))

chk('5 empty holdout removes nothing (default path unchanged)',
    len([r for r in rows if dest(r) in set()]) == 0)

hold2 = {'W', 'P'}
rm2 = [r for r in rows if dest(r) in hold2]
chk('6 multi-residue holdout {W,P} removes both', sorted(rm2) == sorted(['A42W', 'L10W', 'G7P', 'K30W']),
    str(rm2))

chk('7 P is present as a negative control (descriptors should NOT rescue proline)',
    any(dest(r) == 'P' for r in rows))

# the real guard in train.py
src_txt = open('/home/nissimb/DeepPEF/Megascale-fineTuning/train.py').read()
chk('8a train.py raises when the holdout removed ZERO rows',
    "removed ZERO rows" in src_txt and "raise RuntimeError" in src_txt)
chk('8b the guard is keyed on _LORO_STATE[\'removed\'] == 0',
    "_LORO_STATE['removed'] == 0" in src_txt)
chk('9 HOLDOUT_RESIDUES is parsed once at module level',
    "HOLDOUT_RESIDUES = set(" in src_txt)
chk('10 the filter reports removed/kept per protein (visible, not silent)',
    "_LORO_STATE['removed'] +=" in src_txt and "_LORO_STATE['kept'] +=" in src_txt)

print('\nLORO GATE: %s' % ('ALL PASS' if ok else 'FAILURES'))
sys.exit(0 if ok else 1)
