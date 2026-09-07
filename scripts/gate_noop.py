"""GATE: the three SILENT-NO-OP guards actually FIRE. A guard nobody tests is not a guard.

Each section does two things:
  1. proves the hole was REAL -- it reconstructs the silent path and shows that, without
     the guard, the run completes and returns a correctly-shaped tensor / number that is
     NOT what it claims to be;
  2. proves the GUARD FIRES on exactly that input, and that the OFF path is untouched.

CPU only. No GPU, no checkpoint, no job submission.
"""
from __future__ import print_function

import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == 'scripts' else _HERE
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
os.environ.setdefault('WANDB_MODE', 'disabled')

import torch                                             # noqa: E402

OK = [True]


def check(name, cond, extra=''):
    print('%-64s %s %s' % (name, 'PASS' if cond else 'FAIL', extra))
    if not cond:
        OK[0] = False
    return cond


def section(t):
    print('\n' + '-' * 80)
    print(t)
    print('-' * 80)


from model.model_cfg import CFG                          # noqa: E402
import model.hydro_net as HN                             # noqa: E402
import aa_descriptors as AD                              # noqa: E402

_LEVERS = ('aa_descriptors', 'aa_descriptor_csv', 'burial_features', 'metal_features',
           'struct_quality', 'ligand_nodes', 'ligand_annotations', 'model_arch')
_SAVED = dict((k, getattr(CFG, k, None)) for k in _LEVERS)


def restore():
    for k, v in _SAVED.items():
        setattr(CFG, k, v)


# =====================================================================================
section('(A) PEMGraphTransformer refuses inserted blocks instead of dropping them')
# =====================================================================================
try:
    # --- A0: the hole is REAL. The slice arithmetic silently drops a middle block. -----
    # PEMGraphTransformer.forward slices flat_x at [:16], [16:48], [-1044:-20], [-20:].
    # Show directly that inserting K columns at 48 leaves every slice IN BOUNDS and
    # correctly shaped, so nothing raises and the descriptor columns are never read.
    K = 726
    SENTINEL = -12345.0                # outside the baseline range, so a hit is real
    base = torch.arange(16 + 32 + 1024 + 20, dtype=torch.float32).unsqueeze(0)
    assert float((base == SENTINEL).sum()) == 0.0
    withk = torch.cat([base[:, :48], torch.full((1, K), SENTINEL), base[:, 48:]], dim=1)
    def _reassemble(fx):
        return torch.cat([fx[:, :16], fx[:, 16:48], fx[:, -1044:-20], fx[:, -20:]], dim=-1)
    r0, r1 = _reassemble(base), _reassemble(withk)
    check('A.0 inserted block does NOT change the reassembled shape (nothing raises)',
          r0.shape == r1.shape, '%s == %s' % (tuple(r0.shape), tuple(r1.shape)))
    check('A.0b ...and the reassembled vector is IDENTICAL -- the K cols are DROPPED',
          torch.equal(r0, r1))
    check('A.0c ...and the sentinel NEVER appears in the output (block unread)',
          float((r1 == SENTINEL).sum()) == 0.0 and float((withk == SENTINEL).sum()) == K,
          'K=%d sentinel cols present in flat_x, 0 reached the node vector: this is '
          'the silent no-op -- a baseline number reported as a descriptor result' % K)

    # --- A1: OFF path still constructs. The guard must not break the baseline. --------
    restore()
    CFG.aa_descriptors = 'none'
    for f in ('burial_features', 'metal_features', 'struct_quality', 'ligand_nodes'):
        setattr(CFG, f, False)
    built = True
    try:
        HN.PEMGraphTransformer()
    except Exception as e:                                # noqa: BLE001
        built = False
        print('      %s: %s' % (type(e).__name__, e))
    check('A.1 graph_transformer STILL CONSTRUCTS with every lever off', built)

    # --- A2..A6: each lever fires the guard, at CONSTRUCTION, with a clear message ----
    cases = [
        ('A.2 --aa_descriptors', dict(aa_descriptors='mordred726'), 'aa_descriptors'),
        ('A.3 --burial_features', dict(burial_features=True), 'burial_features'),
        ('A.4 --metal_features', dict(metal_features=True), 'metal_features'),
        ('A.5 --struct_quality', dict(struct_quality=True), 'struct_quality'),
        ('A.6 --ligand_nodes', dict(ligand_nodes=True), 'ligand_nodes'),
    ]
    for label, levers, needle in cases:
        restore()
        CFG.aa_descriptors = 'none'
        for f in ('burial_features', 'metal_features', 'struct_quality', 'ligand_nodes'):
            setattr(CFG, f, False)
        for k, v in levers.items():
            setattr(CFG, k, v)
        raised, msg = False, ''
        try:
            HN.PEMGraphTransformer()
        except ValueError as e:
            raised, msg = True, str(e)
        except Exception as e:                            # noqa: BLE001
            msg = 'wrong exception type %s: %s' % (type(e).__name__, e)
        check('%s fires the guard at __init__' % label, raised, msg[:60] if not raised else '')
        check('  ...message names the lever and says the block is dropped',
              raised and needle in msg and 'silently DROPPED' in msg and 'model_arch' in msg)

    # --- A7: the guard is in __init__, NOT forward -- construction is where it costs ---
    import inspect
    src_init = inspect.getsource(HN.PEMGraphTransformer.__init__)
    src_fwd = inspect.getsource(HN.PEMGraphTransformer.forward)
    check('A.7 guard lives in __init__ (cheap) and not in forward (after data load)',
          'CANNOT read inserted feature blocks' in src_init
          and 'CANNOT read inserted feature blocks' not in src_fwd)

    # --- A8: PEM (the default arch) is UNAFFECTED -- it sizes fc1 from the widths ------
    restore()
    CFG.aa_descriptors = 'none'
    for f in ('burial_features', 'metal_features', 'struct_quality', 'ligand_nodes'):
        setattr(CFG, f, False)
    CFG.burial_features = True
    pem_ok = True
    try:
        HN.PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef)
    except Exception as e:                                # noqa: BLE001
        pem_ok = False
        print('      %s: %s' % (type(e).__name__, e))
    check('A.8 model_arch=pem is UNAFFECTED by the guard (burial on)', pem_ok)
finally:
    restore()


# =====================================================================================
section('(B) --ligand_nodes refuses to run without a protein context')
# =====================================================================================
try:
    import ligand_features as LF

    n = 12
    ann = LF.LigandAnnotations(table={}, path='<gate>', n_rows=0,
                               cutoff_angstrom=LF.DEFAULT_CUTOFF_ANGSTROM)
    LF.set_annotations(ann)

    # --- B.1 the guard fires when the context was never set -------------------------
    LF.clear_current_protein()
    raised, msg = False, ''
    try:
        LF.ligand_features(n, folded=True)
    except RuntimeError as e:
        raised, msg = True, str(e)
    check('B.1 unset context RAISES instead of returning zeros', raised)
    check('  ...message names set_current_protein and says "silently do nothing"',
          raised and 'set_current_protein' in msg and 'silently do nothing' in msg)

    # --- B.2 what the hole USED to do: a correctly-shaped all-zero block -------------
    # Reconstruct the old behaviour to show what was being reported as a null result.
    old = torch.zeros((n, LF.LIGAND_DIM))
    check('B.2 the OLD path returned a correctly-shaped, entirely-zero block',
          tuple(old.shape) == (n, LF.LIGAND_DIM) and float(old.abs().sum()) == 0.0,
          'shape=%s sum=0 -- indistinguishable from "this dataset is ligand-free"'
          % (tuple(old.shape),))

    # --- B.3 the escape hatch still works -------------------------------------------
    expl = LF.ligand_features(n, folded=True, protein='NOT_IN_TABLE')
    check('B.3 explicit protein= still works with the context unset',
          tuple(expl.shape) == (n, LF.LIGAND_DIM))

    # --- B.4 the unfolded pass never looks a protein up, so it must NOT raise --------
    unf_ok = True
    try:
        unf = LF.ligand_features(n, folded=False)
    except Exception:                                     # noqa: BLE001
        unf_ok = False
    check('B.4 unfolded pass still returns zeros without raising (U7 unaffected)',
          unf_ok and torch.equal(unf, torch.zeros_like(unf)))

    # --- B.5 lever OFF is byte-identical: ligand_or_none returns None, never raises --
    class _Off(object):
        ligand_nodes = False
    x = torch.randn(n, 4, 3)
    mask = torch.ones(n)
    off_ok = True
    try:
        r = LF.ligand_or_none(x, mask, True, _Off())
    except Exception:                                     # noqa: BLE001
        off_ok = False
        r = 'raised'
    check('B.5 lever OFF returns None and never raises (OFF path byte-identical)',
          off_ok and r is None)

    # --- B.6 no table loaded => zeros, no raise (the lever is not really on) ---------
    LF.set_annotations(None)
    LF.clear_current_protein()
    noann = LF.ligand_features(n, folded=True)
    check('B.6 no annotation table => zeros without raising (lever not armed)',
          torch.equal(noann, torch.zeros_like(noann)))

    # --- B.7 the TRAINING path really pushes the context now ------------------------
    tp = os.path.join(_ROOT, 'Megascale-fineTuning', 'train.py')
    src = open(tp).read()
    check('B.7 train.py defines _set_ligand_context', 'def _set_ligand_context' in src)
    n_calls = src.count('self._set_ligand_context(batch)')
    check('  ...and calls it before EVERY graph-building entry point',
          n_calls >= 3, '%d call sites (get_deltaG, get_ddg_head, get_wt_deltaG)' % n_calls)
    check('  ...and it pushes the name into ligand_features',
          '_LIG.set_current_protein(str(name))' in src)
    # It must be called BEFORE the first get_graph in get_deltaG, not after.
    seg = src.split('def get_deltaG(self, batch, i, n=None):', 1)[1]
    i_ctx = seg.find('self._set_ligand_context(batch)')
    i_gg = seg.find('get_graph(')
    check('  ...and the push happens BEFORE the first get_graph call',
          0 <= i_ctx < i_gg, 'ctx@%d < get_graph@%d' % (i_ctx, i_gg))
finally:
    try:
        LF.set_annotations(None)
        LF.clear_current_protein()
    except Exception:                                     # noqa: BLE001
        pass
    restore()


# =====================================================================================
section('(C) the open-alphabet loader refuses a table whose canonical block moved')
# =====================================================================================
try:
    canon = AD._resolve(AD._CANONICAL_CSV)
    ref = AD.load_descriptor_table(canon)
    K = int(ref.shape[1])
    print('  canonical table %s : [20,%d]' % (canon, K))

    # --- C.1 the REAL committed open25 table is tested, not a /tmp fixture -----------
    real25 = os.path.join(_ROOT, 'data', 'aa_descriptors_open25.csv')
    check('C.1 the REAL committed open-alphabet table exists on disk',
          os.path.isfile(real25), real25)

    if os.path.isfile(real25):
        # Read it raw, independently of the module, so this is a real measurement.
        hdr, rows = None, {}
        for raw in open(real25):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if hdr is None:
                hdr = parts[1:]
                continue
            rows[parts[0]] = parts[1:]
        K25 = len(hdr)
        print('  real open25 table %s : [%d,%d]' % (real25, len(rows), K25))

        # --- C.2 THE FINDING: K moved and the columns differ. State it, do not paper
        # over it. The claim "canonical slice is byte-identical, K unchanged" is FALSE
        # on the live files.
        hdr_canon = AD.descriptor_columns(canon)
        same_cols = list(hdr) == list(hdr_canon)
        print('  K(canonical)=%d  K(open25)=%d  columns identical=%s'
              % (K, K25, same_cols))
        if not same_cols:
            dropped = [c for c in hdr_canon if c not in set(hdr)]
            added = [c for c in hdr if c not in set(hdr_canon)]
            print('  FINDING: %d canonical columns DROPPED, %d new columns ADDED.'
                  % (len(dropped), len(added)))
            print('           e.g. dropped %s ... / added %s ...'
                  % (','.join(dropped[:3]), ','.join(added[:3])))
            print('           So the 25-row table does NOT preserve K and is NOT an')
            print('           open alphabet over the canonical matrix. It is a')
            print('           DIFFERENT descriptor matrix. That is a real finding.')

        # --- C.3 the loader must REFUSE it, loudly ----------------------------------
        if real25 in AD._CACHE:
            del AD._CACHE[real25]
        AD._CACHE.pop(os.path.abspath(real25), None)
        raised, msg = False, ''
        try:
            AD.load_descriptor_table(real25, canonical_only=False)
        except ValueError as e:
            raised, msg = True, str(e)
        except Exception as e:                            # noqa: BLE001
            msg = 'wrong exception type %s: %s' % (type(e).__name__, e)
        check('C.3 the loader REFUSES the real open25 table', raised,
              msg[:70] if not raised else '')
        check('  ...and the message reports both K values and the column delta',
              raised and ('K=%d' % K25) in msg and ('K=%d' % K) in msg
              and 'column NAMES differ' in msg)
        if raised:
            print('      %s' % msg.split(chr(10))[0][:180])

        # --- C.4 canonical_only=True must ALSO refuse. This is the one that mattered:
        # the silent path took the [:20] slice of a table whose columns had changed,
        # got a correctly-shaped [20,K'] tensor, and never noticed.
        AD._CACHE.pop(os.path.abspath(real25), None)
        raised2 = False
        try:
            AD.load_descriptor_table(real25, canonical_only=True)
        except ValueError:
            raised2 = True
        check('C.4 canonical_only=True ALSO refuses (the slice was the silent path)',
              raised2)

    # --- C.5 a LEGITIMATE open-alphabet table -- extra rows, SAME columns -- loads ---
    import tempfile
    tmpd = tempfile.mkdtemp(prefix='noopgate_')
    good = os.path.join(tmpd, 'good_open25.csv')
    body = open(canon).read().rstrip('\n')
    EXTRA = ['SEP', 'TPO', 'PTR', 'MSE', 'HYP']
    with open(good, 'w') as fh:
        fh.write(body + '\n')
        for j, lab in enumerate(EXTRA):
            fh.write('%s,%s\n' % (lab, ','.join('%.10g' % (j + 1 + 0.01 * i)
                                                for i in range(K))))
    gt = AD.load_descriptor_table(good, canonical_only=False)
    check('C.5 a table with extra rows and the SAME columns still loads',
          tuple(gt.shape) == (25, K), '%s' % (tuple(gt.shape),))
    check('  ...and its canonical block is byte-identical to the canonical table',
          torch.equal(AD.load_descriptor_table(good, canonical_only=True), ref))
    check('  ...and K is unchanged, so no layer width moves',
          int(gt.shape[1]) == K, 'K=%d' % K)

    # --- C.6 a table with the right columns but a MUTATED canonical row is refused ---
    bad = os.path.join(tmpd, 'bad_canon.csv')
    lines = [l for l in body.split('\n')]
    out = []
    hit = False
    for l in lines:
        if l.strip() and not l.startswith('#') and l.split(',')[0].strip() == 'A' and not hit:
            p = l.split(',')
            p[1] = '%.10g' % (float(p[1]) + 1.0)          # move ONE number
            l = ','.join(p)
            hit = True
        out.append(l)
    with open(bad, 'w') as fh:
        fh.write('\n'.join(out) + '\n')
        for j, lab in enumerate(EXTRA):
            fh.write('%s,%s\n' % (lab, ','.join('%.10g' % (j + 1 + 0.01 * i)
                                                for i in range(K))))
    check('C.6 fixture really did mutate one canonical value', hit)
    raised3, msg3 = False, ''
    try:
        AD.load_descriptor_table(bad, canonical_only=False)
    except ValueError as e:
        raised3, msg3 = True, str(e)
    check('C.6b a MUTATED canonical row is refused even with the right columns', raised3)
    check('  ...and the message says the canonical block is not byte-identical',
          raised3 and 'NOT byte-identical' in msg3)

    # --- C.7 the pre-W10 path is untouched: a plain 20-row table never validates -----
    plain_ok = True
    try:
        AD.load_descriptor_table(canon)
    except Exception:                                     # noqa: BLE001
        plain_ok = False
    check('C.7 the canonical 20-row table still loads (guard never fires on it)',
          plain_ok)
except Exception:
    OK[0] = False
    traceback.print_exc()
finally:
    restore()


print('\n' + '=' * 80)
print('SILENT-NO-OP GATE: %s' % ('ALL PASS' if OK[0] else 'FAILURES'))
print('=' * 80)
sys.exit(0 if OK[0] else 1)
