"""W10 GATE -- the 20-residue closure is gone, and nothing canonical moved.

Asserts, in order:

  (a) CANONICAL OUTPUT IS BYTE-IDENTICAL to the pre-W10 implementation. Not "close",
      not "same shape" -- torch.equal on the tensor a REAL get_graph call returns, with
      the descriptor lever ON, against the same call computed by a faithful re-execution
      of the OLD code path (the original one_hot @ [20,K] matmul, reconstructed here from
      the CSV independently of the module under test). Table equality alone would NOT be
      enough: the bug this guards against lives in how the table is USED.

  (b) A 25-ROW TABLE LOADS and the extra 5 rows are REACHABLE by label. The canonical 20
      still occupy rows 0..19 in AA_MAP order in that table, so the canonical tensor path
      is unaffected by their presence -- checked by loading BOTH tables and comparing.

  (c) AN UNKNOWN LABEL RAISES A CLEAR ERROR rather than silently returning zeros. This is
      the one that matters most in practice: a zero descriptor row is indistinguishable
      from a legitimately zero descriptor vector, so a silent fallthrough trains to
      garbage while passing every shape and finiteness check there is.

  (d) THE HELD-OUT-RESIDUE HARNESS RETURNS PREDICTIONS for the held-out residue.

Runs on CPU in seconds. No GPU, no checkpoint, no job submission.
"""
from __future__ import print_function

import os
import sys
import tempfile
import traceback

sys.path.insert(0, os.getcwd())
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

os.environ.setdefault('WANDB_MODE', 'disabled')

import torch

OK = [True]


def check(name, cond, extra=''):
    print('%-62s %s %s' % (name, 'PASS' if cond else 'FAIL', extra))
    if not cond:
        OK[0] = False
    return cond


def section(t):
    print('\n' + '-' * 78)
    print(t)
    print('-' * 78)


ORDER = list('ACDEFGHIKLMNPQRSTVWY')

# The mode and the canonical CSV are read FROM THE MODULE, never hard-coded here. This
# gate used to hard-code MODE; that mode was later retired and renamed, so
# aa_descriptor_mode() raised and this gate died in its first section -- it had
# silently stopped testing anything. Deriving both from the module means a rename can
# never turn this gate into a no-op again.
def _live_mode_and_csv():
    import aa_descriptors as _AD
    for _m in _AD.MODES:
        if _m == 'none':
            continue
        try:
            _p = _AD.descriptor_path(_m, None)
        except Exception:                                # noqa: BLE001
            continue
        if int(_AD.load_descriptor_table(_p).shape[1]) > 16:
            return _m, _p                                # the full-width matrix
    for _m in _AD.MODES:
        if _m != 'none':
            return _m, _AD.descriptor_path(_m, None)
    raise RuntimeError('no live descriptor mode')


MODE, CANON_CSV = _live_mode_and_csv()
print('  live descriptor mode = %r   canonical CSV = %r' % (MODE, CANON_CSV))


# ======================================================================================
def read_csv_reference(path):
    """Re-implement the PRE-W10 reader from scratch, here, so (a) is a real comparison.

    Importing the module under test to build its own reference would prove nothing. This
    is a deliberately independent 15-line reader: skip '#', first non-comment line is the
    header, reorder rows by ORDER, exactly as the old loader did.
    """
    header, rows = None, {}
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if header is None:
                header = parts[1:]
                continue
            rows[parts[0]] = [float(v) for v in parts[1:]]
    return torch.tensor([rows[a] for a in ORDER], dtype=torch.float32)


def resolve(p):
    for c in (p, os.path.join(os.path.dirname(_HERE), p)):
        if os.path.isfile(c):
            return c
    raise IOError('cannot find %s' % p)


# ======================================================================================
section('(a) CANONICAL PATH IS BYTE-IDENTICAL  --  torch.equal on a real get_graph call')

csv_path = resolve(CANON_CSV)
ref_table = read_csv_reference(csv_path)
print('  reference table rebuilt independently from %s : %s' % (csv_path, tuple(ref_table.shape)))

from aa_descriptors import (ORDER as MOD_ORDER, load_descriptor_table, descriptor_labels,
                            residue_descriptors, residue_descriptors_by_label,
                            one_hot_from_labels, descriptor_dim, descriptor_columns)
import train_utils
from train_utils import AA_MAP, get_graph, get_unfolded_graph, get_one_hot
from model.model_cfg import CFG

check('ORDER == list(AA_MAP), alphabetical', MOD_ORDER == list(AA_MAP) == ORDER)

mod_table = load_descriptor_table(csv_path)
check('load_descriptor_table default is [20,K] and byte-identical to reference',
      tuple(mod_table.shape) == tuple(ref_table.shape) and torch.equal(mod_table, ref_table),
      '%s' % (tuple(mod_table.shape),))

# --- the real thing: a full get_graph, descriptor lever ON, compared byte for byte -----
torch.manual_seed(1234)
N = 41
x = torch.randn(N, 4, 3) * 7.0
seq = ''.join(ORDER[i] for i in torch.randint(0, 20, (N,)).tolist())
oh = get_one_hot(seq)
emb = torch.randn(N, int(CFG.emb_input_dim))
mask = torch.ones(N)

_saved = dict(aa_descriptors=getattr(CFG, 'aa_descriptors', 'none'),
              aa_descriptor_csv=getattr(CFG, 'aa_descriptor_csv', None),
              burial_features=getattr(CFG, 'burial_features', False),
              flory_unfolded=getattr(CFG, 'flory_unfolded', False),
              unfolded_emb=getattr(CFG, 'unfolded_emb', 'full'))


def restore():
    for k, v in _saved.items():
        setattr(CFG, k, v)


try:
    CFG.aa_descriptors = MODE
    CFG.aa_descriptor_csv = csv_path
    CFG.burial_features = False
    CFG.flory_unfolded = False
    CFG.unfolded_emb = 'full'

    g_new = get_graph(x, oh, emb, mask)
    u_new = get_unfolded_graph(x, oh, emb, mask)

    # Reconstruct what the OLD code produced, by monkeypatching the descriptor block back
    # to the literal pre-W10 expression: one_hot @ (independently rebuilt [20,K] table).
    _orig = train_utils._desc_or_none
    train_utils._desc_or_none = lambda one_hot: one_hot @ ref_table.to(one_hot.dtype)
    try:
        g_old = get_graph(x, oh, emb, mask)
        u_old = get_unfolded_graph(x, oh, emb, mask)
    finally:
        train_utils._desc_or_none = _orig

    check('get_graph          byte-identical (torch.equal)', torch.equal(g_new, g_old),
          'shape=%s' % (tuple(g_new.shape),))
    check('get_unfolded_graph byte-identical (torch.equal)', torch.equal(u_new, u_old),
          'shape=%s' % (tuple(u_new.shape),))
    check('descriptor block is actually present (K>0)', descriptor_dim(MODE, CFG) > 0,
          'K=%d' % descriptor_dim(MODE, CFG))

    # And the gather agrees with the matmul on the canonical twenty.
    g_by_label = residue_descriptors_by_label(seq, MODE, CFG)
    g_by_matmul = residue_descriptors(oh, MODE, CFG)
    check('label GATHER == one-hot MATMUL on canonical residues (torch.equal)',
          torch.equal(g_by_label, g_by_matmul), 'n=%d' % N)

    check('one_hot_from_labels[:, :20] == get_one_hot',
          torch.equal(one_hot_from_labels(seq, MODE, CFG)[:, :20], oh))

    # ==================================================================================
    section('(b) A 25-ROW TABLE LOADS AND THE EXTRA 5 ARE REACHABLE')

    # ==================================================================================
    # (b0) THE REAL COMMITTED TABLE. This gate used to skip straight to a /tmp fixture
    # built by appending rows to the canonical body -- which trivially preserves K and
    # trivially has a byte-identical canonical block, so it passed while proving
    # NOTHING about the file the open-alphabet arm would actually load. Test the real
    # file first, and FAIL LOUDLY on a K mismatch rather than papering over it.
    # ==================================================================================
    REAL25 = os.path.join(os.path.dirname(_HERE), 'data', 'aa_descriptors_open25.csv')
    check('(b0) the REAL committed open-alphabet table exists', os.path.isfile(REAL25),
          REAL25)
    if os.path.isfile(REAL25):
        _hdr, _rows = None, []
        for _raw in open(REAL25):
            _l = _raw.strip()
            if not _l or _l.startswith('#'):
                continue
            _p = [c.strip() for c in _l.split(',')]
            if _hdr is None:
                _hdr = _p[1:]
                continue
            _rows.append(_p[0])
        _K25 = len(_hdr)
        _Kc = ref_table.shape[1]
        _cols_c = descriptor_columns(csv_path)
        print('      real open25: %d rows, K=%d   |   canonical: 20 rows, K=%d'
              % (len(_rows), _K25, _Kc))
        _same_K = (_K25 == _Kc)
        _same_cols = (list(_hdr) == list(_cols_c))
        check('(b0.1) REAL open25 preserves K (a column count must not move)', _same_K,
              'K=%d vs canonical K=%d' % (_K25, _Kc))
        check('(b0.2) REAL open25 has the SAME columns, in the same order', _same_cols)
        if not _same_cols:
            _drop = [c for c in _cols_c if c not in set(_hdr)]
            _add = [c for c in _hdr if c not in set(_cols_c)]
            print('      FINDING: %d canonical columns DROPPED, %d ADDED.'
                  % (len(_drop), len(_add)))
            print('               dropped e.g. %s   added e.g. %s'
                  % (','.join(_drop[:4]), ','.join(_add[:4])))
            print('               The committed 25-row table is therefore NOT an open')
            print('               alphabet over the canonical matrix -- it is a')
            print('               DIFFERENT descriptor matrix. Any layer sized from the')
            print('               canonical K would silently read a different width,')
            print('               and the canonical residues would silently change.')
        # Whatever the answer above, the LOADER must refuse to load it rather than
        # quietly handing back a [20,K'] slice that every caller would treat as the
        # canonical table.
        import aa_descriptors as _ADmod
        _ADmod._CACHE.pop(os.path.abspath(REAL25), None)
        _r25, _m25 = False, ''
        try:
            load_descriptor_table(REAL25, canonical_only=False)
        except ValueError as _e:
            _r25, _m25 = True, str(_e)
        if _same_K and _same_cols:
            check('(b0.3) a CONFORMING real table loads without raising', not _r25, _m25[:60])
        else:
            check('(b0.3) the open-alphabet arm REFUSES the non-conforming real table',
                  _r25, _m25[:60] if not _r25 else '')
            _ADmod._CACHE.pop(os.path.abspath(REAL25), None)
            _r25b = False
            try:
                load_descriptor_table(REAL25, canonical_only=True)
            except ValueError:
                _r25b = True
            check('(b0.4) ...and canonical_only=True refuses it too (that was the '
                  'silent path)', _r25b)

    tmpdir = tempfile.mkdtemp(prefix='w10gate_')
    open25 = os.path.join(tmpdir, 'aa_descriptors_open25.csv')
    EXTRA = ['SEP', 'TPO', 'PTR', 'MSE', 'HYP']
    with open(csv_path) as fh:
        body = fh.read().rstrip('\n')
    K = ref_table.shape[1]
    with open(open25, 'w') as fh:
        fh.write('# W10 gate fixture: canonical 20 + %s\n' % ','.join(EXTRA))
        fh.write(body + '\n')
        for j, lab in enumerate(EXTRA):
            # Deliberately NOT zeros and NOT a copy of any canonical row: a fixture that
            # duplicated serine would pass a reachability test while proving nothing about
            # the rows being distinct.
            vals = [float(j + 1) + 0.01 * i for i in range(K)]
            fh.write('%s,%s\n' % (lab, ','.join('%.10g' % v for v in vals)))

    CFG.aa_descriptor_csv = open25
    labels25 = descriptor_labels(open25)
    check('25-row table loads', len(labels25) == 25, 'M=%d' % len(labels25))
    check('rows 0..19 are still ORDER, in AA_MAP order', labels25[:20] == ORDER)
    check('rows 20..24 are the 5 extras', labels25[20:] == EXTRA, ','.join(labels25[20:]))

    full = load_descriptor_table(open25, canonical_only=False)
    canon_from_open = load_descriptor_table(open25, canonical_only=True)
    check('full table is [25,K]', tuple(full.shape) == (25, K), '%s' % (tuple(full.shape),))
    check('canonical slice of the 25-row table == the 20-row table (torch.equal)',
          torch.equal(canon_from_open, ref_table))

    # The decisive one: the extras are REACHABLE, and reachable to the RIGHT rows.
    got = residue_descriptors_by_label(EXTRA, MODE, CFG)
    check('all 5 non-canonical rows reachable by label', tuple(got.shape) == (5, K),
          '%s' % (tuple(got.shape),))
    exact = all(torch.equal(got[j], full[20 + j]) for j in range(5))
    check('each extra label gathers ITS OWN row (torch.equal, per row)', exact)
    distinct = len(set(tuple(got[j].tolist()) for j in range(5))) == 5
    check('the 5 extra rows are mutually distinct (not all zeros/duplicates)', distinct)
    nonzero = bool((got.abs().sum(dim=1) > 0).all())
    check('no extra row came back all-zero', nonzero)

    mixed = residue_descriptors_by_label(['A', 'SEP', 'W', 'MSE'], MODE, CFG)
    check('mixed canonical + non-canonical gather works', tuple(mixed.shape) == (4, K))
    check('  ...and its canonical entries still match the canonical table',
          torch.equal(mixed[0], ref_table[ORDER.index('A')]) and
          torch.equal(mixed[2], ref_table[ORDER.index('W')]))

    oh25 = one_hot_from_labels(['A', 'SEP', 'W'], MODE, CFG)
    check('one_hot_from_labels widens to M=25 (SEP gets its OWN column, not zeros)',
          tuple(oh25.shape) == (3, 25) and float(oh25[1].sum()) == 1.0
          and int(oh25[1].argmax()) == 20, 'argmax(SEP)=%d' % int(oh25[1].argmax()))

    check('descriptor_dim unchanged by the extra ROWS (K is a column count)',
          descriptor_dim(MODE, CFG) == K, 'K=%d' % K)

    # ==================================================================================
    section('(c) AN UNKNOWN LABEL RAISES CLEARLY -- it does NOT return zeros')

    raised, msg = False, ''
    try:
        residue_descriptors_by_label(['A', 'ZZZ'], MODE, CFG)
    except KeyError as e:
        raised, msg = True, str(e)
    except Exception as e:                                   # any other type is a fail
        raised, msg = False, 'wrong exception type %s: %s' % (type(e).__name__, e)
    check('unknown label raises (does not return zeros)', raised)
    good_msg = raised and ('ZZZ' in msg) and ('zero' in msg.lower()) and ('label' in msg.lower())
    check('  ...and the message names the label and says why not zeros', good_msg)
    if raised:
        print('      message: %s' % msg.strip('"\'')[:200])

    raised2 = False
    try:
        one_hot_from_labels(['A', 'ZZZ'], MODE, CFG)
    except KeyError:
        raised2 = True
    check('one_hot_from_labels also raises on an unknown label', raised2)

    # Regression guard for the OLD closure: an extra row must no longer be an error.
    loaded_ok = True
    try:
        load_descriptor_table(open25, canonical_only=False)
    except Exception as e:
        loaded_ok = False
        print('      %s' % e)
    check('a >20-row CSV no longer RAISES (the closure is really gone)', loaded_ok)

    # A duplicate label must still be an error -- open is not the same as sloppy.
    dup = os.path.join(tmpdir, 'dup.csv')
    with open(dup, 'w') as fh:
        fh.write(body + '\n')
        fh.write('A,%s\n' % ','.join(['0'] * K))
    dup_raised = False
    try:
        load_descriptor_table(dup)
    except ValueError as e:
        dup_raised = 'duplicate' in str(e).lower()
    check('a DUPLICATE row label still raises (open != unvalidated)', dup_raised)

    # ==================================================================================
    section('(d) HELD-OUT-RESIDUE HARNESS RETURNS PREDICTIONS FOR THE HELD-OUT RESIDUE')

    from heldout_residue import (split_by_held_out, descriptor_neighbours,
                                 parse_mutation, _reference_probe, metrics)

    check('parse_mutation handles 1-letter codes', parse_mutation('L7S') == ('L', 7, 'S'))
    check('parse_mutation handles MULTI-CHAR codes (open alphabet in the parser too)',
          parse_mutation('S7SEP') == ('S', 7, 'SEP'), '%s' % (parse_mutation('S7SEP'),))

    CFG.aa_descriptor_csv = csv_path
    nb = descriptor_neighbours(['W'], csv_path, k=4)
    h, nn, err = nb[0]
    check('descriptor neighbours of W computed', err is None and nn is not None)
    if nn:
        print('      W nearest: %s' % ', '.join('%s(%.2f)' % (l, d) for l, d, _ in nn))
        arom = set(l for l, _, _ in nn) & set('FYH')
        check('  ...and they include aromatic residues (F/Y/H) -- something to interpolate from',
              len(arom) > 0, ','.join(sorted(arom)))

    # Synthetic rows, so (d) needs no corpus on disk and is a real end-to-end run of the
    # split + probe. The ddG signal is a deterministic function of the descriptor table,
    # so a descriptor model CAN learn it and a one-hot model cannot generalise to unseen W.
    tbl = ref_table
    idx = dict((a, i) for i, a in enumerate(ORDER))
    rows = []
    for wt in ORDER:
        for mu in ORDER:
            if wt == mu:
                continue
            dv = tbl[idx[mu]] - tbl[idx[wt]]
            y = float(0.9 * dv[2] - 0.6 * dv[1] + 0.3 * dv[0])   # hydropathy/vol/MW
            rows.append(dict(mut_type='%s%d%s' % (wt, 10, mu), ddG_ML=y, _protein='SYN'))

    train, held, wt_side = split_by_held_out(rows, ['W'])
    check('split: held-out rows are non-empty', len(held) > 0, 'n_held=%d' % len(held))
    check('split: no TRAIN row has W as its mutant',
          all(r['_mut'] != 'W' for r in train))
    check('split: no TRAIN row has W as its WT either (the premise "never seen")',
          all(r['_wt'] != 'W' for r in train), 'n_train=%d n_dropped=%d' % (len(train), len(wt_side)))

    res = _reference_probe(train, held, ['W'], csv_path)
    for arm in ('onehot', 'desc'):
        m = res[arm]
        check('harness RETURNS PREDICTIONS for held-out W, arm=%-7s' % arm,
              m['n'] > 0, 'n=%d PCC=%.4f MAE=%.4f' % (m['n'], m['pcc'], m['mae']))
    print('      dPCC(desc - onehot) = %+.4f   dMAE = %+.4f'
          % (res['desc']['pcc'] - res['onehot']['pcc'],
             res['desc']['mae'] - res['onehot']['mae']))
    print('      (this synthetic fixture has a descriptor-linear target by construction,')
    print('       so it verifies the HARNESS runs, not the scientific claim. The claim is')
    print('       measured on the real corpus by heldout_residue.py --reference_probe.)')

except Exception:
    OK[0] = False
    traceback.print_exc()
finally:
    restore()

print('\n' + '=' * 78)
print('W10 OPEN-ALPHABET GATE: %s' % ('ALL PASS' if OK[0] else 'FAILURES'))
print('=' * 78)
sys.exit(0 if OK[0] else 1)
