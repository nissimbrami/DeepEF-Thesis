"""LORO patch: add --holdout_residues to Megascale-fineTuning/train.py.

Verbatim anchors only. Every anchor is asserted to match EXACTLY ONCE before any write.
Idempotent: re-running detects the marker and exits 0 without a second apply.

WHAT THE PATCH DOES
  * adds the flag --holdout_residues (comma-separated residue letters)
  * filters the TRAINING loader only (load_protein_data), never the test loader
    (load_test_protein_data), so held-out rows survive in test as the eval set.
  * removes BOTH directions: mutations INTO a held-out residue AND mutations FROM one.
    Removing only "into" leaves W visible to the model as the residue being mutated
    away from, and the "never seen" premise is then false.
  * NEVER removes the WT row (mut_type == 'wt'). Row 0 is the per-protein reference the
    whole calib_diag convention depends on; dropping it would silently change b_p/a_p
    for every protein and confound the arm with a reference-state change.
  * PRINTS before/after row counts per protein and accumulates a global tally, because a
    holdout that silently removes nothing is this project's signature failure mode. A
    holdout that matches zero rows across the whole epoch RAISES.
"""
import io
import sys

TRAIN = 'Megascale-fineTuning/train.py'
MARKER = 'LORO_HOLDOUT_RESIDUES'


def die(m):
    sys.stderr.write('PATCH ABORT: %s\n' % m)
    sys.exit(1)


def anchor(src, a, name):
    n = src.count(a)
    if n != 1:
        die('anchor %r matched %d times, expected exactly 1' % (name, n))
    print('  anchor %-28s matched exactly once  OK' % name)


src = io.open(TRAIN, encoding='utf-8').read()
if MARKER in src:
    print('already patched (marker %s present) -- no-op' % MARKER)
    sys.exit(0)

# ------------------------------------------------------------- anchor 1: the flag
A1 = "_a, _ = _p.parse_known_args()\nREADOUT = _a.readout\n"
anchor(src, A1, 'argparse_tail')

HELP = (
    "LORO (leave-residue-out): comma-separated residue letters removed ENTIRELY from the "
    "TRAINING set -- both mutations INTO them and mutations FROM them -- while leaving them "
    "in TEST. This is the only runnable proxy for the continuity claim on MegaScale, which "
    "contains ZERO non-canonical residues (scripts/count_noncanonical.py: 451514 codes, 20 "
    "letters). One-hot cannot represent a residue whose column never fired in training; "
    "descriptors can, because the residue is still a point among trained neighbours. The WT "
    "row (mut_type=='wt') is NEVER removed -- it is the per-protein reference row 0 that b_p "
    "and a_p are defined against. Empty = off = byte-identical to baseline."
)
B1 = (
    "_p.add_argument('--holdout_residues', type=str, default='',\n"
    "                help=" + repr(HELP) + ")\n"
    + A1
)
src = src.replace(A1, B1, 1)

# --------------------------------------------------- anchor 2: module-level constant
A2 = "READOUT = _a.readout\nWT_ANCHOR_WEIGHT = _a.wt_anchor_weight\n"
anchor(src, A2, 'readout_const')
B2 = (
    "READOUT = _a.readout\n"
    "# " + MARKER + " -- parsed once at import; the dataset reads this module-level set.\n"
    "HOLDOUT_RESIDUES = set(r.strip().upper() for r in _a.holdout_residues.split(',') if r.strip())\n"
    "if HOLDOUT_RESIDUES:\n"
    "    print('[LORO] holding out of TRAINING (both directions): %s'\n"
    "          % ','.join(sorted(HOLDOUT_RESIDUES)))\n"
    "WT_ANCHOR_WEIGHT = _a.wt_anchor_weight\n"
)
src = src.replace(A2, B2, 1)

# ----------------------------------- anchor 3: the TRAIN-side row filter. Unique by
# trailing whitespace + self.protein_dirs[idx]; the TEST-side twin is deliberately
# left alone so held-out rows remain scorable.
A3 = (
    "        indexes = list(indexes)\n"
    "        mutations = mutations.loc[indexes]\n"
    "        delta_g_tensor = delta_g_tensor[indexes]\n"
    "        one_hot_tensor = one_hot_tensor[indexes]\n"
    "        embedding_tensor = embedding_tensor[indexes]\n"
    "            \n"
    "        mutations_data = {\n"
    "            'name': self.protein_dirs[idx],"
)
anchor(src, A3, 'train_row_filter')

B3 = (
    "        # ---- " + MARKER + " : TRAINING-SET-ONLY residue holdout ----------------\n"
    "        # Applied here and NOWHERE else. load_test_protein_data has a near-identical\n"
    "        # block that is deliberately NOT patched: the held-out rows must survive in\n"
    "        # test, or there is nothing to score.\n"
    "        if HOLDOUT_RESIDUES:\n"
    "            _n_before = len(indexes)\n"
    "            _mt = mutations.loc[list(indexes), 'mut_type'].astype(str)\n"
    "            # Parse WT/mutant off the code. 'wt' rows never match and are therefore\n"
    "            # never dropped -- they are the per-protein reference row that b_p/a_p\n"
    "            # are defined against.\n"
    "            _drop = set()\n"
    "            for _i, _code in _mt.items():\n"
    "                _m = _LORO_CODE_RE.match(_code.strip())\n"
    "                if _m is None:\n"
    "                    continue                      # 'wt', multi-mutants, ins/del: keep\n"
    "                if _m.group(1) in HOLDOUT_RESIDUES or _m.group(3) in HOLDOUT_RESIDUES:\n"
    "                    _drop.add(_i)\n"
    "            indexes = [_i for _i in indexes if _i not in _drop]\n"
    "            _n_after = len(indexes)\n"
    "            # A holdout that silently removes nothing is this project's signature\n"
    "            # failure mode, so it is COUNTED, printed and tallied -- never assumed.\n"
    "            _loro_tally(self.protein_dirs[idx], _n_before, _n_after)\n"
    "        # ------------------------------------------------------------------\n"
    "\n"
    + A3
)
src = src.replace(A3, B3, 1)

# ------------------------------ anchor 4: tallies + regex next to the dataset class.
A4 = "class AllProteinValidationDataset(Dataset):\n"
anchor(src, A4, 'dataset_class')
B4 = (
    "import re  # " + MARKER + ": mutation-code parsing for the training-set holdout\n"
    "\n"
    "_LORO_CODE_RE = re.compile(r'^([A-Z])(\\d+)([A-Z])$')\n"
    "_LORO_STATE = {'removed': 0, 'kept': 0, 'proteins': 0}\n"
    "\n"
    "\n"
    "def _loro_tally(protein, n_before, n_after):\n"
    "    _LORO_STATE['removed'] += (n_before - n_after)\n"
    "    _LORO_STATE['kept'] += n_after\n"
    "    _LORO_STATE['proteins'] += 1\n"
    "    if _LORO_STATE['proteins'] <= 5 or n_before == n_after:\n"
    "        print('[LORO] %-14s rows %6d -> %6d  (removed %5d)'\n"
    "              % (protein, n_before, n_after, n_before - n_after))\n"
    "\n"
    "\n"
    "def loro_report():\n"
    "    \"\"\"Global holdout tally. RAISES if the holdout removed nothing: a flag that\n"
    "    claims to remove a residue and removes zero rows would train an arm silently\n"
    "    identical to baseline and be written up as a null result.\"\"\"\n"
    "    if not HOLDOUT_RESIDUES:\n"
    "        return\n"
    "    print('[LORO] TOTAL over %d train proteins: removed %d rows, kept %d'\n"
    "          % (_LORO_STATE['proteins'], _LORO_STATE['removed'], _LORO_STATE['kept']))\n"
    "    if _LORO_STATE['proteins'] and _LORO_STATE['removed'] == 0:\n"
    "        raise RuntimeError(\n"
    "            '--holdout_residues %s removed ZERO rows across %d proteins. The flag is '\n"
    "            'a no-op and this run would be silently identical to baseline.'\n"
    "            % (','.join(sorted(HOLDOUT_RESIDUES)), _LORO_STATE['proteins']))\n"
    "\n"
    "\n"
    + A4
)
src = src.replace(A4, B4, 1)

io.open(TRAIN + '.loro_new', 'w', encoding='utf-8').write(src)
print('\nwrote %s.loro_new (NOT yet installed)' % TRAIN)
