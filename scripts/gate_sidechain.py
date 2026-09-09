"""Gate for W12 --sidechain_features.

The claims this must falsify, in order of how badly each would mislead us:
  1. the block is ZERO in the unfolded state (else it cancels in dG and the lever is inert)
  2. the burial-weighted columns really are folded-minus-unfolded
  3. the table is aligned to AA_MAP by LABEL, not position (a silent shift would be invisible)
  4. the chemistry is right: W/F/I/L hydrophobic, R/K/D/E hydrophilic
  5. the SCALE does not repeat the LORO failure -- that arm collapsed because the descriptor
     table carried ~16.8x the energy of one-hot
  6. dG_transfer beats VOLUME against the measured per-residue slopes (the corrected mechanism)
"""
import sys, math, io
sys.path.insert(0, '/home/nissimb/DeepPEF')
sys.path.insert(0, '/home/nissimb/DeepPEF/scripts')
import torch
from sidechain_features import (sidechain_block, sidechain_dim, DG_TRANSFER, VOLUME, AA,
                                SIDECHAIN_DIM, _DGT, _VOL)

ok = True
def chk(name, cond, extra=''):
    global ok
    print('%-64s %s %s' % (name, 'PASS' if cond else 'FAIL', extra))
    if not cond: ok = False

torch.manual_seed(0)
N = 40
idx = torch.randint(0, 20, (N,))
oh = torch.eye(20)[idx]
burial = torch.rand(N)

f = sidechain_block(oh, burial, folded=True)
u = sidechain_block(oh, burial, folded=False)

chk('1a block width is 4', f.shape == (N, 4), str(tuple(f.shape)))
chk('1b UNFOLDED burial columns are exactly ZERO',
    bool(torch.all(u[:, 1] == 0) and torch.all(u[:, 3] == 0)))
chk('1c identity columns are IDENTICAL in both states (sequence, not structure)',
    torch.equal(f[:, 0], u[:, 0]) and torch.equal(f[:, 2], u[:, 2]))
chk('2a folded burial columns are NOT all zero', bool(torch.any(f[:, 1] != 0)))
chk('2b col1 == burial * dG_transfer exactly',
    torch.allclose(f[:, 1], burial * f[:, 0], atol=1e-6))
chk('2c the folded-minus-unfolded difference IS the burial term',
    torch.allclose(f[:, 1] - u[:, 1], burial * f[:, 0], atol=1e-6))

# 3 - label alignment, the failure mode that is invisible if wrong
for a in ('W', 'R', 'G'):
    i = AA.index(a)
    one = torch.eye(20)[i:i+1]
    got = float(sidechain_block(one, None, folded=False)[0, 0])
    want = float(_DGT[i])
    chk('3 %s maps to its OWN row (label-aligned)' % a, abs(got - want) < 1e-6)

# 4 - the chemistry
order = sorted(AA, key=lambda a: -DG_TRANSFER[a])
chk('4a most hydrophobic is W', order[0] == 'W', order[:3])
chk('4b most hydrophilic is R', order[-1] == 'R', order[-3:])
chk('4c I, L, F all above +1.5 kcal/mol',
    all(DG_TRANSFER[a] > 1.5 for a in 'ILF'))
chk('4d D, E, K, R all negative', all(DG_TRANSFER[a] < 0 for a in 'DEKR'))

# 5 - THE LORO LESSON: scale must not swamp one-hot
oh_energy = float((oh ** 2).sum(1).mean())              # one-hot row energy = 1.0
blk_energy = float((sidechain_block(oh, None, folded=False) ** 2).sum(1).mean())
ratio = blk_energy / oh_energy
chk('5 block energy is comparable to one-hot, NOT 16.8x (the LORO failure)',
    ratio < 5.0, 'ratio=%.2fx  (LORO descriptor arm was 16.8x and collapsed)' % ratio)

# 6 - the corrected mechanism: transfer free energy beats volume against MEASURED slopes
measured = {'R':0.528,'K':0.537,'Q':0.546,'E':0.540,'N':0.565,'D':0.542,'H':0.518,'P':0.475,
            'Y':0.353,'W':0.341,'S':0.524,'T':0.477,'G':0.506,'A':0.419,'M':0.340,'C':0.267,
            'F':0.288,'L':0.282,'V':0.309,'I':0.311}
def pear(x, y):
    n=len(x); mx=sum(x)/n; my=sum(y)/n
    sxy=sum((a-mx)*(b-my) for a,b in zip(x,y)); sxx=sum((a-mx)**2 for a in x); syy=sum((b-my)**2 for b in y)
    return sxy/math.sqrt(sxx*syy)
res = sorted(measured)
r_dgt = pear([DG_TRANSFER[a] for a in res], [measured[a] for a in res])
r_vol = pear([VOLUME[a] for a in res],      [measured[a] for a in res])
chk('6a dG_transfer correlates with the measured slope', abs(r_dgt) > 0.6, 'r=%+.3f' % r_dgt)
chk('6b dG_transfer BEATS volume (the corrected mechanism)',
    abs(r_dgt) > abs(r_vol), 'dGt %+.3f vs vol %+.3f' % (r_dgt, r_vol))

# 7 - off is off
class C: pass
c = C(); c.sidechain_features = False
chk('7a dim is 0 when the flag is off', sidechain_dim(c) == 0)
c.sidechain_features = True
chk('7b dim is 4 when on', sidechain_dim(c) == SIDECHAIN_DIM)

# ===========================================================================
# 8 - INTEGRATION: the REAL get_graph and the REAL PEM, not a paraphrase.
#
# Sections 1-7 test sidechain_features.py in isolation. That module passed 18/18 for
# days while the block COULD NOT FIRE, because there was no --sidechain_features flag
# and get_graph never called it. An isolated gate cannot catch that. Everything below
# drives the actual training code path with CFG mutated the way train.py mutates it,
# which is the only way to prove the wiring rather than the arithmetic.
# ===========================================================================
print('\n[8] INTEGRATION  real get_graph / real PEM  (the wiring, not the block)')

import os as _os
_os.environ.setdefault('WANDB_MODE', 'disabled')
_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)
from model.model_cfg import CFG as _CFG
from model.hydro_net import PEM as _PEM
from train_utils import get_graph as _gg, get_unfolded_graph as _gu, compute_burial as _cb

_KEYS = ('sidechain_features', 'burial_features', 'burial_mode', 'aa_descriptors',
         'ligand_nodes', 'flory_unfolded', 'unfolded_emb')
_saved = {k: getattr(_CFG, k, None) for k in _KEYS}
_BASE = dict(sidechain_features=False, burial_features=False, burial_mode='count',
             aa_descriptors='none', ligand_nodes=False, flory_unfolded=False,
             unfolded_emb='full')

def _set(**kw):
    for k, v in dict(_BASE, **kw).items():
        setattr(_CFG, k, v)

try:
    _g = torch.Generator().manual_seed(0)
    _Nr = 36
    _xyz = torch.randn(_Nr, 4, 3, generator=_g) * 7.0
    _oh = torch.eye(20)[torch.randint(0, 20, (_Nr,), generator=_g)]
    _emb = torch.randn(_Nr, int(_CFG.emb_input_dim), generator=_g)
    _msk = torch.ones(_Nr)
    _ARG = (_xyz, _oh, _emb, _msk)

    # -- 8a OFF is BYTE-IDENTICAL. Not "same shape": torch.equal on real output.
    _set()
    f_off, u_off = _gg(*_ARG).clone(), _gu(*_ARG).clone()
    if hasattr(_CFG, 'sidechain_features'):
        delattr(_CFG, 'sidechain_features')          # the pre-W12 tree had no attribute
    f_abs, u_abs = _gg(*_ARG), _gu(*_ARG)
    _set()
    chk('8a get_graph OFF is byte-identical to attribute-absent',
        torch.equal(f_off, f_abs), 'width=%d' % f_off.shape[-1])
    chk('8b get_unfolded_graph OFF is byte-identical', torch.equal(u_off, u_abs))

    # -- 8c ON widens by exactly 4, and ONLY in the middle. A left- or right-anchored
    # slip would still be 4 wider and every shape check would still pass.
    _set(sidechain_features=True)
    f_on, u_on = _gg(*_ARG).clone(), _gu(*_ARG).clone()
    _SC = 48                                          # solv_dim 0 + desc_dim 0
    chk('8c folded width grows by exactly 4',
        f_on.shape[-1] == f_off.shape[-1] + 4, '%d -> %d' % (f_off.shape[-1], f_on.shape[-1]))
    chk('8d unfolded width grows by exactly 4', u_on.shape[-1] == u_off.shape[-1] + 4)
    chk('8e columns LEFT of the block are unmoved (D and Fb)',
        torch.equal(f_on[:, :_SC], f_off[:, :_SC]))
    chk('8f columns RIGHT of the block are unmoved (emb and one-hot)',
        torch.equal(f_on[:, _SC + 4:], f_off[:, _SC:]))

    # -- 8g THE LEVER'S WHOLE POINT, measured on the real graph. If the burial columns
    # were nonzero unfolded they would cancel in E_u - E_f and the block would be inert.
    _bf, _bu = f_on[:, _SC:_SC + 4], u_on[:, _SC:_SC + 4]
    chk('8g burial columns are EXACTLY zero in the real unfolded graph',
        bool(torch.all(_bu[:, 1] == 0) and torch.all(_bu[:, 3] == 0)))
    chk('8h burial columns are NONZERO in the real folded graph',
        bool(torch.any(_bf[:, 1] != 0) and torch.any(_bf[:, 3] != 0)))
    chk('8i identity columns cancel between the states (they are sequence, not structure)',
        torch.equal(_bf[:, 0], _bu[:, 0]) and torch.equal(_bf[:, 2], _bu[:, 2]))
    chk('8j the folded block equals sidechain_block on this exact input',
        torch.equal(_bf, sidechain_block(_oh, _cb(_xyz, _msk).to(_oh.dtype), folded=True)))

    # the Flory coil is a THIRD assembly site; it was patched too and must agree.
    _set(sidechain_features=True, flory_unfolded=True)
    _bc = _gu(*_ARG)[:, _SC:_SC + 4]
    chk('8k the Flory-coil unfolded graph also zeroes the burial columns',
        bool(torch.all(_bc[:, 1] == 0) and torch.all(_bc[:, 3] == 0)))

    # -- 8l the model: only fc1_* may grow.
    def _mk(**kw):
        _set(**kw)
        torch.manual_seed(0)
        m = _PEM(layers=_CFG.num_layers, gaussian_coef=_CFG.gaussian_coef,
                 dropout_rate=_CFG.dropout_rate, light_attention=True, readout=False)
        m.eval()
        return m

    m0, m1 = _mk(), _mk(sidechain_features=True)
    chk('8l fc1_gat grows by 4', m1.fc1_gat.in_features == m0.fc1_gat.in_features + 4,
        '%d -> %d' % (m0.fc1_gat.in_features, m1.fc1_gat.in_features))
    chk('8m fc1_gcn grows by 4', m1.fc1_gcn.in_features == m0.fc1_gcn.in_features + 4,
        '%d -> %d' % (m0.fc1_gcn.in_features, m1.fc1_gcn.in_features))
    chk('8n fc2_gat/fc2_gcn/fc1 UNCHANGED (fixed internal widths)',
        m1.fc2_gat.out_features == m0.fc2_gat.out_features
        and m1.fc2_gcn.out_features == m0.fc2_gcn.out_features
        and m1.fc1.in_features == m0.fc1.in_features)
    chk('8o inst_norm1/inst_norm2 UNCHANGED',
        m1.inst_norm1.inst_norm.num_features == m0.inst_norm1.inst_norm.num_features
        and m1.inst_norm2.inst_norm.num_features == m0.inst_norm2.inst_norm.num_features)
    chk('8p sc_start lands at 48 + solv_dim + desc_dim', m1.sc_start == 48 and m1.sc_dim == 4,
        'start=%d dim=%d' % (m1.sc_start, m1.sc_dim))

    # -- 8q THE READ TEST. This is the check that would have failed for the whole time
    # the block was dead code: perturb ONLY the four block columns and demand the
    # energy MOVES. A block that is concatenated but never sliced passes every width
    # assertion above and fails only here.
    _set(sidechain_features=True)
    _fin = _gg(*_ARG).unsqueeze(0)
    _uin = _gu(*_ARG).unsqueeze(0)
    _pert = _fin.clone()
    _pert[0, :, 48:52] += 3.0
    with torch.no_grad():
        _e0 = m1(torch.cat([_fin, _uin], dim=0))
        _e1 = m1(torch.cat([_pert, _uin], dim=0))
    _d = float((_e0 - _e1).abs().max())
    chk('8q perturbing ONLY cols 48:52 CHANGES the energy (block is really read)',
        _d > 1e-8, 'max|delta|=%.6g' % _d)
    chk('8r forward is finite with the lever on', bool(torch.isfinite(_e0).all()))

    # -- 8s co-existence with W5/W6/W11. The block sits BETWEEN W6 and W11, so W11's
    # start must move with it; if it does not, the ligand slice reads four columns of
    # side-chain chemistry as "ligand" and never crashes.
    sys.path.insert(0, _os.path.join(_root, 'scripts'))
    from ligand_features import ligand_start as _lstart
    class _Cfg(object):
        burial_features = False; aa_descriptors = 'none'; sidechain_features = False
        metal_features = False; struct_quality = False
    _c = _Cfg()
    _a0 = _lstart(_c)
    _c.sidechain_features = True
    _a1 = _lstart(_c)
    chk('8s ligand_start moves +4 with the block (W11 reads the right columns)',
        _a1 == _a0 + 4, '%d -> %d' % (_a0, _a1))

    m2 = _mk(sidechain_features=True, burial_features=True, aa_descriptors='mordred_pca16')
    _f2 = _gg(*_ARG)
    chk('8t stacks with W5+W6: sc_start = 48+3+16 and width +23',
        m2.sc_start == 67 and _f2.shape[-1] == f_off.shape[-1] + 23,
        'sc_start=%d width=%d' % (m2.sc_start, _f2.shape[-1]))
    with torch.no_grad():
        _e2 = m2(torch.cat([_f2.unsqueeze(0), _gu(*_ARG).unsqueeze(0)], dim=0))
    chk('8u stacked forward is finite', bool(torch.isfinite(_e2).all()))

    # -- 8v the graph_transformer must REFUSE, not silently drop the block.
    _set(sidechain_features=True)
    try:
        from model.hydro_net import PEMGraphTransformer as _GT
        _GT(layers=2, gaussian_coef=_CFG.gaussian_coef)
        chk('8v graph_transformer REFUSES --sidechain_features', False, 'it did not raise')
    except ValueError as _ex:
        chk('8v graph_transformer REFUSES --sidechain_features',
            'sidechain_features' in str(_ex), 'names the lever')
    except Exception as _ex:                                  # noqa: BLE001
        chk('8v graph_transformer REFUSES --sidechain_features', False, repr(_ex)[:90])

    # -- 8w and the flag must default OFF, or every run silently becomes a W12 run.
    _tp = _os.path.join(_root, 'Megascale-fineTuning', 'train.py')
    _src = io.open(_tp, encoding='utf-8').read()
    chk('8w --sidechain_features exists in train.py and is store_true',
        "'--sidechain_features', action='store_true'" in _src)
    chk('8x train.py assigns CFG.sidechain_features',
        'CFG.sidechain_features = _a.sidechain_features' in _src)
    chk('8y the run config logs sidechain_features (traceability)',
        "'sidechain_features': _a.sidechain_features," in _src)

finally:
    for k, v in _saved.items():
        if v is None:
            if hasattr(_CFG, k):
                delattr(_CFG, k)
        else:
            setattr(_CFG, k, v)

print('\nW12 SIDECHAIN GATE: %s' % ('ALL PASS' if ok else 'FAILURES'))
sys.exit(0 if ok else 1)
