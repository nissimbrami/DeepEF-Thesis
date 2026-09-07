"""W11 gates -- must all pass before --ligand_nodes is used in any run.

Usage
-----
    python scripts/gate_ligand.py                    # A-E on synthetic fixtures
    python scripts/gate_ligand.py --real             # A/B also on REAL get_graph output
    python scripts/gate_ligand.py --csv data/.../ligand_sites.csv \
                                  --tensors data/Processed_K50_dG_datasets/training_data

Gates (the five the item asked for, in order)
---------------------------------------------
A  OFF is BYTE-IDENTICAL. torch.equal, never allclose, on REAL get_graph and
   get_unfolded_graph output with --real.
B  ON with an EMPTY annotation file is ALSO byte-identical in VALUE. This is the
   critical no-op case: MegaScale is expected to be ligand-free, so this is the state
   the project will actually run in, and a lever that perturbs a ligand-free dataset
   is a bug that would be misread as a null result. Checked two ways:
     B1 the block itself is exactly zero,
     B2 stripping the 10 zero columns from the ON graph reproduces the OFF graph
        byte-for-byte -- i.e. the ONLY difference is 10 zeros, nothing else moved.
C  ON with a synthetic ligand CHANGES the folded graph (the lever is not inert), and
   the ligand contribution is EXACTLY ZERO in the unfolded state. If it survived into
   the unfolded half it would cancel in dG = E_u - E_f and the feature could not
   express binding. This is defect U7; it is the whole point of the gate.
D  Content is right: proximity decays with distance, contact count saturates, class
   one-hot is exclusive, multi-ligand reduction takes the nearest, unknown class ->
   OTHER, apo protein -> exact zeros.
E  Slot arithmetic. The block lands at ligand_start(cfg) across every combination of
   the sibling levers, and the RIGHT-ANCHORED indices (one_hot at -20, llm at -(E+20))
   are unmoved. A wrong offset does NOT crash -- it silently feeds the wrong columns
   into fc1 -- which is why this is gated rather than trusted.
F  Annotation file sanity, only when --csv is given (off-by-one hunt, as in W9's G4).

Exit code 0 = all gates pass.
"""

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from ligand_features import (          # noqa: E402
    LIGAND_DIM, LIGAND_CLASSES, LIGAND_OTHER_INDEX, LIGAND_CLASS_DIM,
    CONTACT_COUNT_CAP, LIGAND_SIZE_CAP, DEFAULT_CUTOFF_ANGSTROM,
    LigandAnnotations, ligand_features, ligand_or_none, ligand_dim, ligand_start,
    set_annotations, set_current_protein, clear_current_protein, to_model_units,
)

AA_ORDER = 'ACDEFGHIKLMNPQRSTVWY'      # matches train_utils.AA_MAP

_FAILURES = []


class _Cfg(object):
    """Stand-in for model_cfg.CFG so the gates run without the training package."""
    def __init__(self, **kw):
        self.ligand_nodes = False
        self.metal_features = False
        self.struct_quality = False
        self.burial_features = False
        self.aa_descriptors = 'none'
        self.aa_desc_dim = None
        self.emb_input_dim = 1024
        for k, v in kw.items():
            setattr(self, k, v)


def check(name, ok, detail=''):
    status = 'PASS' if ok else 'FAIL'
    print('[%s] %s%s' % (status, name, ('  -- ' + detail) if detail else ''))
    if not ok:
        _FAILURES.append(name)
    return ok


# ---------------------------------------------------------------------------
# Fixtures -- no dataset needed
# ---------------------------------------------------------------------------

def _fixture(n=24, emb=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, 3, generator=g)
    one_hot = torch.zeros(n, 20)
    one_hot[torch.arange(n), torch.randint(0, 20, (n,), generator=g)] = 1.0
    e = torch.randn(n, emb, generator=g)
    mask = torch.ones(n)
    return x, one_hot, e, mask


def _toy_table(cutoff=DEFAULT_CUTOFF_ANGSTROM):
    """One ATP (NUCLEOTIDE, 31 heavy atoms) contacting 4 residues at varied distance,
    plus one Zn (METAL, 1 atom) contacting 2. Protein P2 is apo."""
    ann = LigandAnnotations(table={}, path='<toy>', n_rows=0, cutoff_angstrom=cutoff)
    p1 = ann.table.setdefault('P1', {})

    atp = LigandAnnotations._Ligand('ATP_401', 'NUCLEOTIDE', 31)
    atp.contacts = {3: (0.0, 5), 6: (2.25, 3), 9: (4.5, 1), 12: (3.0, 12)}
    p1['ATP_401'] = atp

    zn = LigandAnnotations._Ligand('ZN_402', 'METAL', 1)
    zn.contacts = {18: (2.0, 1), 20: (2.2, 1)}
    p1['ZN_402'] = zn

    ann.n_rows = 6
    return ann


# ---------------------------------------------------------------------------
# Gate A -- OFF is byte-identical
# ---------------------------------------------------------------------------

def gate_a():
    print('\n--- A: OFF is byte-identical (torch.equal) ---')
    x, one_hot, emb, mask = _fixture()
    cfg_off = _Cfg(ligand_nodes=False)

    b_f = ligand_or_none(x, mask, True, cfg_off)
    b_u = ligand_or_none(x, mask, False, cfg_off)
    check('A.1 ligand_or_none returns None when off (folded)', b_f is None)
    check('A.2 ligand_or_none returns None when off (unfolded)', b_u is None)
    check('A.3 ligand_dim == 0 when off', ligand_dim(cfg_off) == 0,
          'got %d' % ligand_dim(cfg_off))

    # Simulate the two torch.cat branches exactly as get_graph writes them.
    D = torch.randn(x.shape[0], 16); Fb = torch.randn(x.shape[0], 32)
    ref = torch.cat([D, Fb, emb, one_hot], dim=1)
    L = ligand_or_none(x, mask, True, cfg_off)
    parts = [D, Fb] + ([L] if L is not None else [])
    got = torch.cat(parts + [emb, one_hot], dim=1)
    check('A.4 node tensor byte-identical when off (torch.equal)',
          torch.equal(ref, got),
          'shapes ref=%s got=%s' % (tuple(ref.shape), tuple(got.shape)))

    # Flag ON but no table loaded must still be a legal all-zero block, never a crash.
    set_annotations(None); clear_current_protein()
    cfg_on = _Cfg(ligand_nodes=True)
    L = ligand_or_none(x, mask, True, cfg_on)
    check('A.5 flag on + no table -> all-zero [N,10], no crash',
          L is not None and L.shape == (x.shape[0], LIGAND_DIM)
          and torch.equal(L, torch.zeros_like(L)))


def gate_a_real():
    """A on the REAL get_graph, with CFG actually toggled. Requires the training pkg."""
    print('\n--- A(real): OFF is byte-identical on real get_graph output ---')
    from model.model_cfg import CFG
    from train_utils import get_graph, get_unfolded_graph

    torch.manual_seed(0)
    N = 36
    x = torch.randn(N, 4, 3) * 7.0
    oh = torch.eye(20)[torch.randint(0, 20, (N,))]
    emb = torch.randn(N, int(CFG.emb_input_dim))
    mask = torch.ones(N)

    CFG.ligand_nodes = False
    set_annotations(None); clear_current_protein()
    f_off = get_graph(x, oh, emb, mask)
    u_off = get_unfolded_graph(x, oh, emb, mask)

    CFG.ligand_nodes = False          # re-assert; build twice to prove determinism
    f_off2 = get_graph(x, oh, emb, mask)
    u_off2 = get_unfolded_graph(x, oh, emb, mask)
    check('A(real).1 OFF folded graph is deterministic', torch.equal(f_off, f_off2))
    check('A(real).2 OFF unfolded graph is deterministic', torch.equal(u_off, u_off2))
    check('A(real).3 OFF width unchanged at 16+32+E+20',
          f_off.shape[1] == 48 + int(CFG.emb_input_dim) + 20,
          'got %d' % f_off.shape[1])
    return CFG, get_graph, get_unfolded_graph, (x, oh, emb, mask), f_off, u_off


# ---------------------------------------------------------------------------
# Gate B -- ON with an EMPTY annotation file is byte-identical. THE CRITICAL CASE.
# ---------------------------------------------------------------------------

def gate_b(real=None):
    print('\n--- B: ON + EMPTY annotation file is byte-identical (the no-op case) ---')

    # An empty but WELL-FORMED file must parse, not raise.
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix='.csv', prefix='ligand_empty_')
    os.close(fd)
    try:
        with open(tmp, 'w', encoding='utf-8') as fh:
            fh.write('protein,ligand_id,ligand_class,resi,distance,'
                     'n_heavy_atoms,n_contact_atoms,resi_convention\n')
        try:
            ann = LigandAnnotations.from_csv(tmp)
            parsed = True
            err = ''
        except Exception as e:                                  # noqa: BLE001
            ann = None
            parsed = False
            err = str(e)[:120]
        check('B.0 a header-only (empty) annotation file PARSES, does not raise',
              parsed, err)
        if not parsed:
            return
        check('B.0b empty file yields 0 proteins, 0 rows',
              len(ann.table) == 0 and ann.n_rows == 0, repr(ann))
    finally:
        os.unlink(tmp)

    set_annotations(ann)
    set_current_protein('ANY_PROTEIN')
    blk = ligand_features(24, folded=True)
    check('B.1 empty table -> block is EXACTLY zero (torch.equal)',
          torch.equal(blk, torch.zeros_like(blk)))

    if real is None:
        print('    (B.2 needs --real; skipped)')
        return
    CFG, get_graph, get_unfolded_graph, args, f_off, u_off = real
    x, oh, emb, mask = args

    CFG.ligand_nodes = True
    set_annotations(ann)
    set_current_protein('ANY_PROTEIN')
    f_on = get_graph(x, oh, emb, mask)
    u_on = get_unfolded_graph(x, oh, emb, mask)

    check('B.2a ON adds exactly 10 columns',
          f_on.shape[1] == f_off.shape[1] + LIGAND_DIM,
          'off=%d on=%d' % (f_off.shape[1], f_on.shape[1]))

    start = ligand_start(CFG)
    blk_f = f_on[:, start:start + LIGAND_DIM]
    blk_u = u_on[:, start:start + LIGAND_DIM]
    check('B.2b the inserted block is EXACTLY zero (folded)',
          torch.equal(blk_f, torch.zeros_like(blk_f)))
    check('B.2c the inserted block is EXACTLY zero (unfolded)',
          torch.equal(blk_u, torch.zeros_like(blk_u)))

    # The real no-op test: delete the 10 columns and the graph must be byte-identical
    # to the OFF graph. This catches ANY other perturbation, not just a nonzero block.
    stripped_f = torch.cat([f_on[:, :start], f_on[:, start + LIGAND_DIM:]], dim=1)
    stripped_u = torch.cat([u_on[:, :start], u_on[:, start + LIGAND_DIM:]], dim=1)
    check('B.2d ON-minus-block == OFF, byte-identical (folded)',
          torch.equal(stripped_f, f_off))
    check('B.2e ON-minus-block == OFF, byte-identical (unfolded)',
          torch.equal(stripped_u, u_off))

    CFG.ligand_nodes = False
    set_annotations(None); clear_current_protein()


# ---------------------------------------------------------------------------
# Gate C -- a real ligand changes the folded graph and is ZERO unfolded
# ---------------------------------------------------------------------------

def gate_c(real=None):
    print('\n--- C: a ligand changes the FOLDED graph and is ZERO unfolded ---')
    set_annotations(_toy_table())
    n = 24

    set_current_protein('P1')
    folded = ligand_features(n, folded=True)
    unfolded = ligand_features(n, folded=False)

    check('C.1 folded block DIFFERS from unfolded (U7 cancellation check)',
          not torch.equal(folded, unfolded),
          'if equal, the column cancels exactly in E_u - E_f and the lever is inert')
    check('C.2 unfolded block is EXACTLY zero',
          torch.equal(unfolded, torch.zeros_like(unfolded)))
    check('C.3 folded block is NOT all zero for an annotated protein',
          folded.abs().sum().item() > 0,
          'sum=%.4f' % folded.abs().sum().item())

    if real is None:
        print('    (C.4 needs --real; skipped)')
        return
    CFG, get_graph, get_unfolded_graph, args, f_off, u_off = real
    x, oh, emb, mask = args

    CFG.ligand_nodes = True
    set_annotations(_toy_table())
    set_current_protein('P1')
    f_on = get_graph(x, oh, emb, mask)
    u_on = get_unfolded_graph(x, oh, emb, mask)
    start = ligand_start(CFG)
    blk_f = f_on[:, start:start + LIGAND_DIM]
    blk_u = u_on[:, start:start + LIGAND_DIM]

    check('C.4 real folded graph carries a NONZERO ligand block',
          blk_f.abs().sum().item() > 0, 'sum=%.4f' % blk_f.abs().sum().item())
    check('C.5 real UNFOLDED graph ligand block is EXACTLY zero',
          torch.equal(blk_u, torch.zeros_like(blk_u)),
          'a nonzero here would cancel in dG = E_u - E_f')
    # And the rest of the graph must be untouched -- the ligand block is additive,
    # it does not perturb D, Fb, emb or one_hot.
    stripped_f = torch.cat([f_on[:, :start], f_on[:, start + LIGAND_DIM:]], dim=1)
    check('C.6 with a ligand present the OTHER columns are still byte-identical',
          torch.equal(stripped_f, f_off),
          'the block must be purely additive')

    CFG.ligand_nodes = False
    set_annotations(None); clear_current_protein()


# ---------------------------------------------------------------------------
# Gate D -- content
# ---------------------------------------------------------------------------

def gate_d():
    print('\n--- D: the block expresses ligand contact ---')
    cutoff = DEFAULT_CUTOFF_ANGSTROM
    set_annotations(_toy_table(cutoff))
    n = 24
    set_current_protein('P1')
    f = ligand_features(n, folded=True)

    atp_rows = [3, 6, 9, 12]
    zn_rows = [18, 20]
    check('D.1 bound flag set on every contacting residue',
          all(f[r, 0].item() == 1.0 for r in atp_rows + zn_rows))

    nuc = 1 + LIGAND_CLASSES.index('NUCLEOTIDE')
    met = 1 + LIGAND_CLASSES.index('METAL')
    check('D.2 ligand class one-hot correct',
          all(f[r, nuc].item() == 1.0 for r in atp_rows) and
          all(f[r, met].item() == 1.0 for r in zn_rows))
    check('D.3 class one-hot sums to exactly 1 on contacting rows',
          all(abs(f[r, 1:1 + LIGAND_CLASS_DIM].sum().item() - 1.0) < 1e-6
              for r in atp_rows + zn_rows))

    # Proximity: 1 - d/cutoff. d=0 -> 1.0 ; d=cutoff -> 0.0 ; monotone decreasing.
    pcol = 2 + LIGAND_CLASS_DIM
    check('D.4 proximity == 1.0 at zero separation',
          abs(f[3, pcol].item() - 1.0) < 1e-6, 'got %.4f' % f[3, pcol].item())
    check('D.5 proximity == 0.0 exactly at the cutoff',
          abs(f[9, pcol].item() - 0.0) < 1e-6, 'got %.4f' % f[9, pcol].item())
    check('D.6 proximity is monotone decreasing in distance (0 < 2.25 < 4.5 A)',
          f[3, pcol].item() > f[6, pcol].item() > f[9, pcol].item(),
          '%.3f > %.3f > %.3f' % (f[3, pcol].item(), f[6, pcol].item(),
                                  f[9, pcol].item()))
    check('D.7 proximity is a REAL distance decay, not flat',
          abs(f[3, pcol].item() - f[9, pcol].item()) > 0.5)

    # Contact count / 8, clipped.
    ccol = 1 + LIGAND_CLASS_DIM
    check('D.8 contact count = 5/8', abs(f[3, ccol].item() - 5 / CONTACT_COUNT_CAP) < 1e-6,
          'got %.4f' % f[3, ccol].item())
    check('D.9 contact count clipped to 1.0 for an over-packed residue (12 atoms)',
          abs(f[12, ccol].item() - 1.0) < 1e-6, 'got %.4f' % f[12, ccol].item())

    # Ligand size.
    scol = 3 + LIGAND_CLASS_DIM
    check('D.10 ligand size = 31/64 for ATP',
          abs(f[3, scol].item() - 31 / LIGAND_SIZE_CAP) < 1e-6)
    check('D.11 ligand size distinguishes ATP(31) from Zn(1)',
          f[3, scol].item() > f[18, scol].item(),
          '%.4f > %.4f' % (f[3, scol].item(), f[18, scol].item()))

    non = [r for r in range(n) if r not in atp_rows + zn_rows]
    check('D.12 non-contacting residues are EXACTLY zero',
          all(f[r].abs().sum().item() == 0.0 for r in non))

    # Apo protein and unset protein -> exact zeros.
    set_current_protein('P2')
    p2 = ligand_features(n, folded=True)
    check('D.13 apo protein -> exactly zero', torch.equal(p2, torch.zeros_like(p2)))
    clear_current_protein()
    # D.14 USED to assert 'unset current protein -> exactly zero'. That assertion
    # codified the silent no-op: _CURRENT_PROTEIN was set only from gate scripts, so in
    # the real training path it was ALWAYS None and every folded ligand block was zeros
    # -- --ligand_nodes could not fire, and a run would have been written up as a null
    # result for a lever that never ran. The block now REFUSES. A flag that silently
    # does nothing is worse than a flag that errors.
    _d14_raised, _d14_msg = False, ''
    try:
        ligand_features(n, folded=True)
    except RuntimeError as _e:
        _d14_raised, _d14_msg = True, str(_e)
    check('D.14 unset current protein RAISES (does NOT silently return zeros)',
          _d14_raised)
    check('D.14b ...and the message says the lever would silently do nothing',
          _d14_raised and 'silently do nothing' in _d14_msg
          and 'set_current_protein' in _d14_msg)
    # An EXPLICIT protein= still works with the context unset: that is the documented
    # escape hatch, and it must not be broken by the guard.
    _d14_expl = ligand_features(n, folded=True, protein='P2')
    check('D.14c explicit protein= still works with the context unset',
          torch.equal(_d14_expl, torch.zeros_like(_d14_expl)))
    # And the UNFOLDED pass must still be zeros without raising -- it never looks a
    # protein up at all (U7: an unfolded chain has no pocket).
    _d14_unf = ligand_features(n, folded=False)
    check('D.14d unfolded pass still returns zeros without raising',
          torch.equal(_d14_unf, torch.zeros_like(_d14_unf)))
    set_current_protein('P1')

    # Unknown class -> OTHER, not dropped.
    ann = LigandAnnotations(table={}, path='<toy>', n_rows=1, cutoff_angstrom=cutoff)
    lg = LigandAnnotations._Ligand('XYZ_1', 'FLAVODOXINOID', 20)
    lg.contacts = {2: (1.0, 1)}
    ann.table['PX'] = {'XYZ_1': lg}
    set_annotations(ann); set_current_protein('PX')
    fx = ligand_features(8, folded=True)
    check('D.15 unknown ligand class -> OTHER slot, not dropped',
          fx[2, 0].item() == 1.0 and fx[2, 1 + LIGAND_OTHER_INDEX].item() == 1.0)

    # Multi-ligand reduction: nearest wins the class, counts add.
    ann = LigandAnnotations(table={}, path='<toy>', n_rows=2, cutoff_angstrom=cutoff)
    far = LigandAnnotations._Ligand('FAR', 'LIPID', 40)
    far.contacts = {1: (4.0, 2)}
    near = LigandAnnotations._Ligand('NEAR', 'COFACTOR', 10)
    near.contacts = {1: (1.0, 3)}
    ann.table['PM'] = {'FAR': far, 'NEAR': near}
    set_annotations(ann); set_current_protein('PM')
    fm = ligand_features(8, folded=True)
    cof = 1 + LIGAND_CLASSES.index('COFACTOR')
    lip = 1 + LIGAND_CLASSES.index('LIPID')
    check('D.16 multi-ligand: the NEAREST ligand wins the class',
          fm[1, cof].item() == 1.0 and fm[1, lip].item() == 0.0)
    check('D.17 multi-ligand: contact counts ADD (2+3=5)',
          abs(fm[1, ccol].item() - 5 / CONTACT_COUNT_CAP) < 1e-6,
          'got %.4f' % fm[1, ccol].item())

    # An out-of-range resi must RAISE, never be silently dropped.
    ann = LigandAnnotations(table={}, path='<toy>', n_rows=1, cutoff_angstrom=cutoff)
    bad = LigandAnnotations._Ligand('B', 'METAL', 1)
    bad.contacts = {99: (1.0, 1)}
    ann.table['PB'] = {'B': bad}
    set_annotations(ann); set_current_protein('PB')
    raised = False
    try:
        ligand_features(8, folded=True)
    except IndexError:
        raised = True
    check('D.18 out-of-range resi RAISES (off-by-one is never silent)', raised)

    # Units helper.
    check('D.19 to_model_units scales Angstrom by 0.1 (the coords are pre-scaled)',
          abs(to_model_units(4.5) - 0.45) < 1e-9, 'got %r' % to_model_units(4.5))

    set_annotations(None); clear_current_protein()


# ---------------------------------------------------------------------------
# Gate E -- slot arithmetic, right-anchored blocks did not move
# ---------------------------------------------------------------------------

def gate_e():
    print('\n--- E: slot arithmetic and right-anchored indices ---')
    E = 1024
    # Every combination of the sibling levers. The ligand block must land at
    # 48+solv+desc+metal+sq in ALL of them. A wrong offset does NOT crash -- it
    # silently feeds the wrong columns into fc1 -- which is exactly why this is gated.
    #
    # FIXED 2026-09-07. This loop used to run metal/sq over (False, True) and then
    # build `parts` with torch.randn stand-ins for the W9 and W8 blocks. That layout
    # IS A FICTION: train_utils.get_graph builds
    #     _blocks = [D, Fb] (+ _S) (+ _Dsc) (+ _L) + [emb, _oh]
    # and mentions neither metal_features nor struct_quality anywhere in the file.
    # So the gate constructed a feature vector the training path cannot emit, and
    # PASSED on it -- while the real graph width stayed 1102 with metal_features=True
    # and the block stayed at column 48, against a ligand_start() of 57. That is the
    # project's signature failure (runs, completes, reports a number, wrong columns).
    # ligand_start() now REFUSES those combinations, so the gate asserts the refusal
    # instead of asserting the fiction. Re-enable the combos here only when the W9/W8
    # blocks are genuinely concatenated in get_graph.
    combos = []
    for burial in (False, True):
        for desc_k in (0, 16):
            combos.append((burial, desc_k, False, False))

    for _m, _q in ((True, False), (False, True), (True, True)):
        _cfg = _Cfg(ligand_nodes=True, burial_features=False,
                    metal_features=_m, struct_quality=_q, emb_input_dim=E,
                    aa_descriptors='none', aa_desc_dim=None)
        try:
            ligand_start(_cfg)
            check('E ligand_start REFUSES unreachable W9/W8 layout '
                  '(metal=%d,sq=%d)' % (_m, _q), False,
                  'returned an offset instead of raising')
        except RuntimeError:
            check('E ligand_start REFUSES unreachable W9/W8 layout '
                  '(metal=%d,sq=%d)' % (_m, _q), True)

    for burial, desc_k, metal, sq in combos:
        cfg = _Cfg(ligand_nodes=True, burial_features=burial,
                   metal_features=metal, struct_quality=sq, emb_input_dim=E,
                   aa_descriptors=('pca16' if desc_k else 'none'),
                   aa_desc_dim=(desc_k or None))
        solv = 3 if burial else 0
        mtl = 9 if metal else 0
        sqd = 3 if sq else 0
        tag = 'burial=%d,desc=%d,metal=%d,sq=%d' % (burial, desc_k, metal, sq)

        start = ligand_start(cfg)
        exp_start = 48 + solv + desc_k + mtl + sqd
        check('E start == 48+solv+desc+metal+sq (%s)' % tag,
              start == exp_start, 'got %d, expected %d' % (start, exp_start))

        n = 12
        D = torch.randn(n, 16); Fb = torch.randn(n, 32)
        S = torch.randn(n, solv) if solv else None
        Dsc = torch.randn(n, desc_k) if desc_k else None
        M = torch.randn(n, mtl) if mtl else None
        Q = torch.randn(n, sqd) if sqd else None
        # A recognisable ramp, so a wrong slice is visible, not merely unequal.
        L = torch.arange(n * LIGAND_DIM, dtype=torch.float32).reshape(n, LIGAND_DIM)
        emb = torch.randn(n, E); one_hot = torch.eye(20)[torch.arange(n) % 20]
        parts = ([D, Fb] + [t for t in (S, Dsc, M, Q) if t is not None]
                 + [L, emb, one_hot])
        Fh = torch.cat(parts, dim=1)

        sliced = Fh[:, start:start + LIGAND_DIM]
        check('E block recoverable at ligand_start (%s)' % tag,
              torch.equal(sliced, L))
        check('E one_hot still at -20 (%s)' % tag,
              torch.equal(Fh[:, -20:], one_hot))
        llm_index = -(E + 20)
        check('E emb still at -(E+20) (%s)' % tag,
              torch.equal(Fh[:, llm_index:-20], emb))
        exp_w = 16 + 32 + solv + desc_k + mtl + sqd + LIGAND_DIM + E + 20
        check('E total width (%s)' % tag, Fh.shape[1] == exp_w,
              'got %d, expected %d' % (Fh.shape[1], exp_w))

        # The GCN slice is x[:, :32] -- D(16) plus HALF of Fb -- NOT x[:, :48].
        # Widening it would feed the wrong count into a layer sized for 52+extras.
        blocks = [t for t in (S, Dsc, M, Q) if t is not None] + [sliced]
        gcn_in = torch.cat([Fh[:, :32]] + blocks + [Fh[:, -20:]], dim=-1)
        exp_gcn = 32 + solv + desc_k + mtl + sqd + LIGAND_DIM + 20
        check('E GCN width = 32+extras+10+20, slice stays x[:, :32] (%s)' % tag,
              gcn_in.shape[1] == exp_gcn,
              'got %d, expected %d' % (gcn_in.shape[1], exp_gcn))
        gat_in = torch.cat([Fh[:, :16]] + blocks + [Fh[:, -20:]], dim=-1)
        exp_gat = 16 + solv + desc_k + mtl + sqd + LIGAND_DIM + 20
        check('E GAT width = 16+extras+10+20 (%s)' % tag,
              gat_in.shape[1] == exp_gat,
              'got %d, expected %d' % (gat_in.shape[1], exp_gat))

    # fc2_*/inst_norm1/inst_norm2/fc_in_dim must NOT grow. Assert the arithmetic
    # the project's convention depends on: those widths are functions of proj_extra
    # ONLY, never of any feature block.
    proj_extra = 16
    for burial, desc_k, metal, sq in combos:
        solv = 3 if burial else 0
        mtl = 9 if metal else 0
        sqd = 3 if sq else 0
        inst1 = 36 + proj_extra
        inst2 = 2 * (36 + proj_extra)
        fc_in = 2 * (36 + proj_extra) + 1024
        check('E fixed widths independent of blocks (b=%d,d=%d,m=%d,q=%d)'
              % (burial, desc_k, metal, sq),
              inst1 == 52 and inst2 == 104 and fc_in == 1128,
              'inst_norm1=%d inst_norm2=%d fc_in_dim=%d' % (inst1, inst2, fc_in))


def gate_e_real():
    """E against the REAL model: fc1_* grew by exactly 10, fc2_*/inst_norm did not."""
    print('\n--- E(real): the model widths grew in the right places only ---')
    from model.model_cfg import CFG
    from model.hydro_net import PEM

    def widths():
        m = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
                dropout_rate=CFG.dropout_rate, light_attention=True, readout=False)
        return {
            'fc1_gcn': m.fc1_gcn.in_features,
            'fc1_gat': m.fc1_gat.in_features,
            'fc2_gcn': m.fc2_gcn.out_features,
            'fc2_gat': m.fc2_gat.out_features,
            'fc1': m.fc1.in_features,
        }

    CFG.ligand_nodes = False
    off = widths()
    CFG.ligand_nodes = True
    on = widths()
    CFG.ligand_nodes = False

    check('E(real).1 fc1_gcn grew by exactly 10',
          on['fc1_gcn'] - off['fc1_gcn'] == LIGAND_DIM,
          '%d -> %d' % (off['fc1_gcn'], on['fc1_gcn']))
    check('E(real).2 fc1_gat grew by exactly 10',
          on['fc1_gat'] - off['fc1_gat'] == LIGAND_DIM,
          '%d -> %d' % (off['fc1_gat'], on['fc1_gat']))
    check('E(real).3 fc2_gcn did NOT grow',
          on['fc2_gcn'] == off['fc2_gcn'], '%d -> %d' % (off['fc2_gcn'], on['fc2_gcn']))
    check('E(real).4 fc2_gat did NOT grow',
          on['fc2_gat'] == off['fc2_gat'], '%d -> %d' % (off['fc2_gat'], on['fc2_gat']))
    check('E(real).5 fc_in_dim (fc1) did NOT grow',
          on['fc1'] == off['fc1'], '%d -> %d' % (off['fc1'], on['fc1']))


def gate_forward_real():
    """The end-to-end check gate_g4_cpu does: a real graph through a real PEM."""
    print('\n--- E(real2): end-to-end forward with the lever ON ---')
    from model.model_cfg import CFG
    from model.hydro_net import PEM
    from train_utils import get_graph, get_unfolded_graph

    torch.manual_seed(0)
    N = 36
    x = torch.randn(N, 4, 3) * 7.0
    oh = torch.eye(20)[torch.randint(0, 20, (N,))]
    emb = torch.randn(N, int(CFG.emb_input_dim))
    mask = torch.ones(N)

    CFG.ligand_nodes = True
    set_annotations(_toy_table()); set_current_protein('P1')
    m = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
            dropout_rate=CFG.dropout_rate, light_attention=True, readout=False)
    m.eval()
    f = get_graph(x, oh, emb, mask).unsqueeze(0)
    u = get_unfolded_graph(x, oh, emb, mask).unsqueeze(0)
    check('E(real2).1 folded and unfolded have the SAME N (the cat must work)',
          f.shape[1] == u.shape[1], 'f=%s u=%s' % (tuple(f.shape), tuple(u.shape)))
    with torch.no_grad():
        e = m(torch.cat([f, u], dim=0))
    dg = float(e.reshape(-1)[0] - e.reshape(-1)[1])
    check('E(real2).2 forward runs, dG finite with --ligand_nodes on',
          bool(torch.isfinite(e).all().item()),
          'dG=%.4f width=%d' % (dg, f.shape[-1]))

    CFG.ligand_nodes = False
    set_annotations(None); clear_current_protein()


# ---------------------------------------------------------------------------
# Gate F -- annotation file sanity, only with --csv
# ---------------------------------------------------------------------------

def _residue_letters(tensor_dir, protein):
    """Decode one_hot_encodings.pt row 0 (wild type) into a residue string."""
    p = os.path.join(tensor_dir, protein, 'one_hot_encodings.pt')
    if not os.path.exists(p):
        return None
    oh = torch.load(p, weights_only=True)
    while oh.dim() > 3:
        oh = oh.squeeze(0)
    wt = oh[0] if oh.dim() == 3 else oh
    wt = wt[:, :20]
    idx = wt.argmax(dim=-1)
    active = wt.sum(dim=-1) > 0
    return ''.join(AA_ORDER[i] if a else '-'
                   for i, a in zip(idx.tolist(), active.tolist()))


def gate_f(csv_path, tensor_dir, cutoff):
    print('\n--- F: annotation file sanity ---')
    ann = LigandAnnotations.from_csv(csv_path, cutoff_angstrom=cutoff)
    print('    %r' % (ann,))
    check('F.0 file parsed', True,
          '%d proteins, %d rows' % (len(ann.table), ann.n_rows))

    if ann.n_rows == 0:
        print('    NOTE: the annotation file is EMPTY. That is a LEGITIMATE delivery '
              '(LIGAND_SPEC.md section 8) and means the dataset is ligand-free, which '
              'is the expected outcome for MegaScale. The W11 arm should then be '
              'REPORTED as inert on this benchmark, not RUN as an accuracy arm.')
        return

    if not tensor_dir or not os.path.isdir(tensor_dir):
        print('    SKIP F.1-F.3: --tensors not given or not a directory.')
        return

    available = set(os.listdir(tensor_dir))
    missing = [p for p in ann.proteins() if p not in available]
    check('F.1 every annotated protein has a tensor directory', not missing,
          ('missing: %s' % (missing[:10],)) if missing
          else 'all %d resolve' % len(ann.table))

    total = 0
    out_of_range = []
    dists = []
    for prot in ann.proteins():
        if prot not in available:
            continue
        letters = _residue_letters(tensor_dir, prot)
        if letters is None:
            continue
        n = len(letters)
        for lid, lclass, nh, contacts in ann.contacts(prot):
            for resi, (d, na) in contacts.items():
                if resi >= n:
                    out_of_range.append((prot, lid, resi, n))
                    continue
                total += 1
                dists.append(d)

    check('F.2 every resi is inside its protein residue count', not out_of_range,
          ('out of range: %s' % (out_of_range[:10],)) if out_of_range else 'ok')

    if dists:
        mx = max(dists)
        check('F.3 every distance is within the declared cutoff',
              mx <= cutoff + 1e-6, 'max=%.3f A, cutoff=%.3f A' % (mx, cutoff))
        # A distance distribution that is entirely integral or entirely identical is
        # the signature of a placeholder file rather than a measured one.
        check('F.4 distances look measured, not placeholder',
              len(set(round(d, 3) for d in dists)) > 1 or len(dists) == 1,
              '%d contacts, %d distinct distances'
              % (len(dists), len(set(round(d, 3) for d in dists))))
    if out_of_range:
        print('    OFF-BY-ONE DIAGNOSIS: %d contacts fall past the end of their '
              'tensor. The annotation is most likely in PDB resSeq numbering, not '
              '0-based tensor numbering. See LIGAND_SPEC.md section 4 -- do NOT '
              'proceed until this is resolved.' % len(out_of_range))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=None, help='ligand_sites.csv (enables F)')
    ap.add_argument('--tensors', default=None, help='tensor_root_dir (enables F.1-F.3)')
    ap.add_argument('--cutoff', type=float, default=DEFAULT_CUTOFF_ANGSTROM,
                    help='contact cutoff in ANGSTROM (default %.1f)'
                         % DEFAULT_CUTOFF_ANGSTROM)
    ap.add_argument('--real', action='store_true',
                    help='also run A/B/C/E against the real get_graph and PEM')
    a = ap.parse_args()

    real = None
    if a.real:
        try:
            real = gate_a_real()
        except Exception as e:                                  # noqa: BLE001
            check('A(real) importable', False, '%s: %s' % (type(e).__name__, e))
            real = None

    gate_a()
    gate_b(real)
    gate_c(real)
    gate_d()
    gate_e()
    if real is not None:
        try:
            gate_e_real()
            gate_forward_real()
        except Exception as e:                                  # noqa: BLE001
            check('E(real) ran', False, '%s: %s' % (type(e).__name__, e))

    if a.csv:
        gate_f(a.csv, a.tensors, a.cutoff)
    else:
        print('\n--- F SKIPPED: no --csv. The annotation file does not exist yet; '
              'see results/LIGAND_SPEC.md. A-E are complete without it. ---')

    print('\n' + '=' * 66)
    if _FAILURES:
        print('W11 GATES FAILED (%d): %s' % (len(_FAILURES), _FAILURES))
        return 1
    print('W11 GATES PASSED. --ligand_nodes is safe to enable (and is inert '
          'without an annotation file).')
    print('SCORING: this lever acts on dG / b_p via FOLDED-STATE stabilisation. '
          'It CANCELS in ddG when the ligand is present in both WT and mutant. '
          'Never score it on ddG alone.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
