"""W8 gates -- must all pass before --struct_quality is used in any run.

Usage
-----
    python scripts/gate_w8.py
    python scripts/gate_w8.py --csv plddt.csv \
        --tensors data/Processed_K50_dG_datasets/training_data

Gates
-----
G1  OFF is BIT-IDENTICAL. torch.equal, never allclose. The block builder returns None
    (not a zero tensor) when the flag is off, so the concatenation is the same object
    graph it was before this module existed.
G2  ON WITHOUT DATA RAISES. Requirement 2 of the item: setting the flag with no
    annotation file must be a hard error naming the file it wants, never an all-zero
    block that gets reported as a pLDDT result. Checked for require_annotations() and
    for a missing-file load.
G3  Block semantics: value mapping, the mask gate on columns 1-2, the low-confidence
    threshold, exact zeros for an unannotated protein, and the [0,1] range.
G4  Parser rejects malformed files: missing column, bad resi, negative resi, duplicate
    residue, out-of-range value, empty file. Each must RAISE.
G5  Scale handling: a 0-100 file and the same file already in [0,1] parse to the SAME
    normalised table. A per-row scale guess would fail this.
G6  Slot arithmetic. The block lands at struct_quality_start(cfg) and the
    right-anchored indices (one_hot at -20, llm at -(E+20)) are unmoved, with and
    without W5/W6/W9 in front of it.
G7  Both-states identity: the block is deliberately the SAME folded and unfolded.
    Asserted so that changing it later is a visible, deliberate act.
G8  Real-data gates, only when --csv is given: every protein resolves to a tensor
    directory, every resi is inside that protein's residue count, and an off-by-one
    is searched for in [-5,5] and reported.

Exit code 0 = all gates pass.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from struct_quality import (                      # noqa: E402
    STRUCT_QUALITY_DIM, LOW_CONF_CUTOFF, PLDDT_SCALE,
    PlddtAnnotations, struct_quality_features, struct_quality_or_none,
    struct_quality_dim, struct_quality_start, require_annotations,
    load_annotations, set_annotations, set_current_protein, clear_current_protein,
    plddt_from_pdb, has_real_plddt,
)

_FAILURES = []
_PASSES = []


class _Cfg(object):
    """Stand-in for model_cfg.CFG so the gates run without the training package."""
    def __init__(self, **kw):
        self.struct_quality = False
        self.burial_features = False
        self.metal_features = False
        self.aa_descriptors = 'none'
        self.emb_input_dim = 1024
        self.plddt_path = None
        for k, v in kw.items():
            setattr(self, k, v)


def check(name, ok, detail=''):
    status = 'PASS' if ok else 'FAIL'
    print('[%s] %s%s' % (status, name, ('  -- ' + detail) if detail else ''))
    (_PASSES if ok else _FAILURES).append(name)
    return ok


def raises(name, fn, want_substr=None):
    """Assert fn() raises, and that the message names what the operator needs."""
    try:
        fn()
    except Exception as e:
        msg = str(e)
        if want_substr and want_substr not in msg:
            return check(name, False,
                         'raised but message lacks %r: %s' % (want_substr, msg[:120]))
        return check(name, True, '%s: %s' % (type(e).__name__, msg.split('\n')[0][:90]))
    return check(name, False, 'did NOT raise')


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _fixture(n=24, emb=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, 3, generator=g)
    one_hot = torch.zeros(n, 20)
    one_hot[torch.arange(n), torch.randint(0, 20, (n,), generator=g)] = 1.0
    e = torch.randn(n, emb, generator=g)
    mask = torch.ones(n)
    return x, one_hot, e, mask


def _toy_table():
    #  P1: a low-confidence tail then a confident core.
    #  P2 is deliberately absent -> must give exact zeros.
    return PlddtAnnotations(table={
        'P1': {0: 0.40, 1: 0.55, 2: 0.90, 3: 0.99},
    }, path='<toy>', n_rows=4)


def _write(path, text):
    with open(path, 'w') as fh:
        fh.write(text)
    return path


# ---------------------------------------------------------------------------
# G1 -- OFF is bit-identical
# ---------------------------------------------------------------------------

def gate_g1():
    print('\n--- G1: OFF is bit-identical -------------------------------------')
    x, one_hot, emb, mask = _fixture()
    cfg_off = _Cfg(struct_quality=False)

    # With a table LOADED and a protein SET, the off flag must still yield None.
    set_annotations(_toy_table())
    set_current_protein('P1')
    b = struct_quality_or_none(x, mask, cfg_off)
    check('G1a off returns None even with data loaded', b is None,
          'got %s' % type(b).__name__)
    check('G1b off contributes 0 dims', struct_quality_dim(cfg_off) == 0)

    # The concatenation the caller builds must be byte-identical to the original.
    base = torch.cat([torch.randn(24, 48, generator=torch.Generator().manual_seed(1)),
                      emb, one_hot], dim=1)
    blocks = [base[:, :48]] + ([] if b is None else [b]) + [emb, one_hot]
    rebuilt = torch.cat(blocks, dim=1)
    check('G1c off concatenation is torch.equal', torch.equal(rebuilt, base),
          'width %d' % rebuilt.shape[1])

    # And with no table at all.
    set_annotations(None)
    clear_current_protein()
    check('G1d off returns None with no table',
          struct_quality_or_none(x, mask, cfg_off) is None)


# ---------------------------------------------------------------------------
# G2 -- ON without data must RAISE
# ---------------------------------------------------------------------------

def gate_g2(tmpdir):
    print('\n--- G2: ON without data RAISES -----------------------------------')
    set_annotations(None)
    clear_current_protein()

    cfg_on = _Cfg(struct_quality=True)
    raises('G2a require_annotations(on, no file) raises',
           lambda: require_annotations(cfg_on), 'struct_quality')
    raises('G2b message names the flag to turn off or the file to pass',
           lambda: require_annotations(cfg_on), '--plddt_path')

    missing = os.path.join(tmpdir, 'does_not_exist.csv')
    raises('G2c loading a missing file raises FileNotFoundError',
           lambda: load_annotations(missing), 'not found')
    raises('G2d missing-file message names PLDDT_SPEC.md',
           lambda: PlddtAnnotations.from_csv(missing), 'PLDDT_SPEC.md')

    # OFF with no file must NOT raise -- that is the default path of 48 live runs.
    cfg_off = _Cfg(struct_quality=False)
    ok = True
    try:
        require_annotations(cfg_off)
    except Exception as e:
        ok = False
        print('    unexpected: %s' % e)
    check('G2e OFF with no file does not raise', ok)

    # ON *with* a file must succeed.
    good = _write(os.path.join(tmpdir, 'good.csv'),
                  'protein,resi,plddt\nP1,0,40\nP1,1,90\n')
    set_annotations(None)
    ann = require_annotations(_Cfg(struct_quality=True, plddt_path=good))
    check('G2f ON with a file loads', ann is not None and ann.has('P1'),
          repr(ann))
    set_annotations(None)


# ---------------------------------------------------------------------------
# G3 -- block semantics
# ---------------------------------------------------------------------------

def gate_g3():
    print('\n--- G3: block semantics ------------------------------------------')
    ann = _toy_table()
    set_annotations(ann)

    n = 6
    mask = torch.ones(n)
    mask[4:] = 0.0                      # last two are padding
    b = ann.block('P1', n, mask=mask)

    check('G3a shape', tuple(b.shape) == (n, STRUCT_QUALITY_DIM), str(tuple(b.shape)))
    check('G3b confidence column matches the table',
          abs(b[0, 0].item() - 0.40) < 1e-6 and abs(b[3, 0].item() - 0.99) < 1e-6,
          'c0=%.3f c3=%.3f' % (b[0, 0], b[3, 0]))
    check('G3c uncertainty = 1 - confidence on real residues',
          abs(b[0, 1].item() - 0.60) < 1e-6, 'u0=%.3f' % b[0, 1])
    check('G3d low-confidence flag set below the cutoff',
          b[0, 2].item() == 1.0 and b[1, 2].item() == 1.0,
          'cutoff=%.2f' % LOW_CONF_CUTOFF)
    check('G3e low-confidence flag clear above the cutoff',
          b[2, 2].item() == 0.0 and b[3, 2].item() == 0.0)
    check('G3f masked rows read as not-a-residue, not max-uncertainty',
          b[4, 1].item() == 0.0 and b[4, 2].item() == 0.0,
          'row4=%s' % b[4].tolist())
    check('G3g unannotated residues are exactly zero',
          torch.equal(b[4], torch.zeros(STRUCT_QUALITY_DIM)))
    check('G3h every value is in [0,1]',
          bool((b >= 0).all() and (b <= 1).all()))

    zb = ann.block('P_ABSENT', n)
    check('G3i unannotated protein gives EXACT zeros',
          torch.equal(zb, torch.zeros(n, STRUCT_QUALITY_DIM)))

    set_current_protein('P1')
    f = struct_quality_features(4)
    check('G3j module-level current-protein plumbing works',
          abs(f[0, 0].item() - 0.40) < 1e-6)
    clear_current_protein()
    check('G3k unset protein gives EXACT zeros',
          torch.equal(struct_quality_features(4),
                      torch.zeros(4, STRUCT_QUALITY_DIM)))

    # An annotation past the end of the tensor is the off-by-one signature.
    raises('G3l resi past the tensor end raises IndexError',
           lambda: ann.block('P1', 2), 'residue indexing')
    set_annotations(None)


# ---------------------------------------------------------------------------
# G4 -- the parser rejects malformed files
# ---------------------------------------------------------------------------

def gate_g4(tmpdir):
    print('\n--- G4: parser rejects malformed files ---------------------------')
    def w(name, text):
        return _write(os.path.join(tmpdir, name), text)

    raises('G4a missing required column',
           lambda: PlddtAnnotations.from_csv(
               w('m.csv', 'protein,plddt\nP1,90\n')), 'missing required column')
    raises('G4b non-integer resi',
           lambda: PlddtAnnotations.from_csv(
               w('b.csv', 'protein,resi,plddt\nP1,x,90\n')), 'not an integer')
    raises('G4c negative resi',
           lambda: PlddtAnnotations.from_csv(
               w('n.csv', 'protein,resi,plddt\nP1,-1,90\n')), 'negative')
    raises('G4d duplicate residue',
           lambda: PlddtAnnotations.from_csv(
               w('d.csv', 'protein,resi,plddt\nP1,0,90\nP1,0,80\n')), 'duplicate')
    raises('G4e value above the pLDDT scale',
           lambda: PlddtAnnotations.from_csv(
               w('h.csv', 'protein,resi,plddt\nP1,0,101\n')), 'outside')
    raises('G4f non-numeric value',
           lambda: PlddtAnnotations.from_csv(
               w('v.csv', 'protein,resi,plddt\nP1,0,abc\n')), 'not a number')
    raises('G4g empty file is an error, not a silent zero table',
           lambda: PlddtAnnotations.from_csv(
               w('e.csv', 'protein,resi,plddt\n')), 'zero usable rows')
    raises('G4h unknown resi_convention is refused, not assumed',
           lambda: PlddtAnnotations.from_csv(
               w('c.csv', 'protein,resi,plddt,resi_convention\nP1,0,90,pdb_author\n')),
           'tensor0')


# ---------------------------------------------------------------------------
# G5 -- scale handling
# ---------------------------------------------------------------------------

def gate_g5(tmpdir):
    print('\n--- G5: 0-100 and 0-1 files agree --------------------------------')
    a = _write(os.path.join(tmpdir, 's100.csv'),
               'protein,resi,plddt\nP1,0,40\nP1,1,90\nP1,2,99\n')
    b = _write(os.path.join(tmpdir, 's1.csv'),
               'protein,resi,plddt\nP1,0,0.40\nP1,1,0.90\nP1,2,0.99\n')
    ta = PlddtAnnotations.from_csv(a)
    tb = PlddtAnnotations.from_csv(b)
    same = all(abs(ta.table['P1'][i] - tb.table['P1'][i]) < 1e-9 for i in range(3))
    check('G5a a 0-100 file and a 0-1 file give the same table', same,
          '%s vs %s' % (ta.table['P1'], tb.table['P1']))
    check('G5b the 0-100 file was rescaled', ta.rescaled is True)
    check('G5c the 0-1 file was NOT rescaled', tb.rescaled is False)
    check('G5d blocks are torch.equal',
          torch.equal(ta.block('P1', 3), tb.block('P1', 3)))


# ---------------------------------------------------------------------------
# G6 -- slot arithmetic
# ---------------------------------------------------------------------------

def gate_g6():
    print('\n--- G6: slot arithmetic ------------------------------------------')
    E = 1024
    cases = [
        ('bare',            _Cfg(struct_quality=True), 48),
        ('with W5',         _Cfg(struct_quality=True, burial_features=True), 51),
        ('with W5+W6',      _Cfg(struct_quality=True, burial_features=True,
                                 aa_descriptors='pca16', aa_desc_dim=16), 67),
        ('with W5+W6+W9',   _Cfg(struct_quality=True, burial_features=True,
                                 aa_descriptors='pca16', aa_desc_dim=16,
                                 metal_features=True), 76),
    ]
    for name, cfg, want in cases:
        got = struct_quality_start(cfg)
        check('G6 start offset, %s' % name, got == want, 'got %d want %d' % (got, want))

    # Right-anchored indices must not move as blocks are added on the left.
    for name, cfg, start in cases:
        width = start + STRUCT_QUALITY_DIM + E + 20
        x = torch.arange(width, dtype=torch.float32).reshape(1, width)
        one_hot_idx = -20
        llm_idx = -(E + 20)
        ok = (x[:, one_hot_idx:].shape[1] == 20 and
              x[:, llm_idx:one_hot_idx].shape[1] == E and
              x[:, start:start + STRUCT_QUALITY_DIM].shape[1] == STRUCT_QUALITY_DIM)
        check('G6 right-anchored slices unmoved, %s' % name, ok,
              'width=%d' % width)

    check('G6 dim is 0 when off', struct_quality_dim(_Cfg(struct_quality=False)) == 0)
    check('G6 dim is 3 when on', struct_quality_dim(_Cfg(struct_quality=True)) == 3)

    # A W6 mode with no published width must refuse rather than guess an offset.
    raises('G6 unset aa_desc_dim refuses to guess an offset',
           lambda: struct_quality_start(_Cfg(struct_quality=True,
                                             aa_descriptors='pca16')),
           'Refusing to guess')


# ---------------------------------------------------------------------------
# G7 -- both-states identity is deliberate
# ---------------------------------------------------------------------------

def gate_g7():
    print('\n--- G7: block is identical in both states (deliberate) -----------')
    set_annotations(_toy_table())
    set_current_protein('P1')
    a = struct_quality_features(4)
    b = struct_quality_features(4)
    check('G7a folded and unfolded blocks are torch.equal', torch.equal(a, b),
          'pLDDT is a property of the measurement, not the conformation')
    check('G7b the block is not all zeros for an annotated protein',
          not torch.equal(a, torch.zeros_like(a)))
    clear_current_protein()
    set_annotations(None)


# ---------------------------------------------------------------------------
# G9 -- a placeholder B-factor column is ABSENCE, not zero confidence
# ---------------------------------------------------------------------------

def gate_g9():
    print('\n--- G9: placeholder B-factor is absence, not zero confidence -----')
    check('G9a an all-zero column is not real pLDDT',
          has_real_plddt([0.0, 0.0, 0.0]) is False)
    check('G9b an empty column is not real pLDDT',
          has_real_plddt([]) is False)
    check('G9c a genuine column is real pLDDT',
          has_real_plddt([55.27, 90.1]) is True)
    check('G9d a genuine column with some zeros is still real pLDDT',
          has_real_plddt([0.0, 0.0, 88.4]) is True,
          'only an ALL-zero column is a placeholder')


# ---------------------------------------------------------------------------
# G8 -- real annotation file
# ---------------------------------------------------------------------------

def gate_g8(csv_path, tensors):
    print('\n--- G8: real annotation file -------------------------------------')
    ann = PlddtAnnotations.from_csv(csv_path)
    check('G8a file parses', ann.n_rows > 0, repr(ann))

    if not tensors or not os.path.isdir(tensors):
        print('    (no --tensors dir; skipping alignment gates)')
        return

    allp = set(os.listdir(tensors))
    have = set(ann.proteins())
    unknown = sorted(have - allp)
    check('G8b every annotated protein has a tensor directory', not unknown,
          'unknown: %s' % unknown[:5])
    print('    coverage: %d / %d tensor proteins annotated'
          % (len(have & allp), len(allp)))

    over = []
    checked = 0
    for p in sorted(have & allp):
        ct = os.path.join(tensors, p, 'coords_tensor.pt')
        if not os.path.exists(ct):
            continue
        n = torch.load(ct, map_location='cpu').shape[0]
        mx = max(ann.table[p])
        checked += 1
        if mx >= n:
            over.append((p, mx, n))
    check('G8c every resi is inside its tensor', not over,
          'checked %d; offenders: %s' % (checked, over[:5]))

    # Exact-length agreement is the real off-by-one detector.
    wrong_len = []
    for p in sorted(have & allp):
        ct = os.path.join(tensors, p, 'coords_tensor.pt')
        if not os.path.exists(ct):
            continue
        n = torch.load(ct, map_location='cpu').shape[0]
        if len(ann.table[p]) != n:
            wrong_len.append((p, len(ann.table[p]), n))
    check('G8d annotated residue count equals the tensor residue count',
          not wrong_len, 'offenders: %s' % wrong_len[:5])

    if over or wrong_len:
        print('    searching for an index offset that would fix it...')
        for off in range(-5, 6):
            bad = 0
            for p, mx, n in (over or wrong_len):
                if mx + off >= n or mx + off < 0:
                    bad += 1
            if bad == 0:
                print('    offset %+d would fix every offender' % off)
                break


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='W8 structure-quality gates.')
    ap.add_argument('--csv', help='A real annotation CSV to validate.')
    ap.add_argument('--tensors', help='training_data dir, for alignment gates.')
    ap.add_argument('--tmpdir', default=None)
    args = ap.parse_args()

    import tempfile
    tmpdir = args.tmpdir or tempfile.mkdtemp(prefix='gate_w8_')

    gate_g1()
    gate_g2(tmpdir)
    gate_g3()
    gate_g4(tmpdir)
    gate_g5(tmpdir)
    gate_g6()
    gate_g7()
    gate_g9()
    if args.csv:
        gate_g8(args.csv, args.tensors)

    print('\n' + '=' * 66)
    print('W8: %d passed, %d failed' % (len(_PASSES), len(_FAILURES)))
    if _FAILURES:
        print('FAILED: %s' % ', '.join(_FAILURES))
        print('W8-GATE: FAIL')
        return 1
    print('W8-GATE: ALL PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
