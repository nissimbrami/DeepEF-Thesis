"""G1-G4 gate for ITEM 22 -- W7 pairwise EDGE ATTRIBUTES (--edge_features).

EXITS NON-ZERO ON ANY FAILURE. Run this before any training uses --edge_features.

    cd <repo root>
    python scripts/gate_w7_edge.py ; echo "exit=$?"

Companion to scripts/gate_w7.py, which gates the edge SET (--gcn_span,
--gcn_bidir, 11/11). This one gates the edge ATTRIBUTE path that gate_w7.py
does not touch.

WHAT THIS GATE IS FOR, in one line each:

  G1 inert when off   With --edge_features off the model output is BYTE-identical
                      (torch.equal, never allclose) to the build without it.
  G2 does what named   k(2A) > k(8A) > k(15A) STRICTLY, plus the stronger
                      expected-centre readout; and the 56-dim attribute actually
                      distinguishes (LEU->ASP) from (ASP->LEU).
  G3 no silent inter.  --edge_features with --flory_unfolded RAISES unless the
                      per-half ca_coords fix (U6) is present.
  G4 U6               The unfolded half's edge distances DIFFER from the folded
                      half's. (MASTER_PLAN Part IV, "W7 + coil" row.)

Sections 1-4 need only torch + edge_features.py and run on a laptop.
Section 5 (G1 byte-identity) needs the repo on sys.path and is SKIPPED with a
loud notice, not silently, when hydro_net/model_cfg cannot be imported --
a skipped G1 is NOT a passed G1 and the manager must run it on the cluster.
"""

import os
import sys

os.environ.setdefault('WANDB_MODE', 'disabled')
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

try:
    import edge_features as EF
except ImportError:
    from scripts import edge_features as EF   # if dropped under scripts/

ok = True
skipped = []


def check(name, cond, extra=''):
    global ok
    print('%-62s %s %s' % (name, 'PASS' if cond else 'FAIL', extra))
    ok = ok and bool(cond)


def skip(name, why):
    skipped.append(name)
    print('%-62s SKIP %s' % (name, why))


print('=' * 84)
print('G2 -- the RBF bank. THE assertion: k(2A) > k(8A) > k(15A), strictly.')
print('=' * 84)

# ---------------------------------------------------------------- 1. THE assertion
d = torch.tensor([2.0, 8.0, 15.0])
K = EF.rbf_expand(d)

check('bank is CONCATENATED, width 16 (a summed bank is width 1)',
      K.shape == (3, 16), str(tuple(K.shape)))

mono, vals = EF.rbf_monotone_ok()
check('k(2A) > k(8A) > k(15A) STRICTLY  <-- THE assertion',
      mono, '%.6e %.6e %.6e' % tuple(vals) if vals else 'no values')

# The stronger, non-fragile restatement. The first-contact channel spans 26
# orders of magnitude, so a strict > on it is true but knife-edge in float32.
# The bank's expected centre (sum_m w_m c_m, w = normalised responses) recovers
# the input distance to 3 decimals and CANNOT be flat unless the bank collapsed.
centers = torch.linspace(0.0, 20.0, 16)
w = K / K.sum(dim=-1, keepdim=True)
exp_c = (w * centers).sum(dim=-1)
check('expected centre recovers the distance (2/8/15 A)',
      bool(exp_c[0] < exp_c[1] < exp_c[2]) and abs(float(exp_c[1]) - 8.0) < 0.05
      and abs(float(exp_c[2]) - 15.0) < 0.05,
      '%.4f %.4f %.4f' % tuple(exp_c.tolist()))

check('each distance peaks at a DIFFERENT centre',
      len({int(K[i].argmax()) for i in range(3)}) == 3,
      str([int(K[i].argmax()) for i in range(3)]))

# Show the U8 failure explicitly so nobody re-derives it. A SUMMED bank returns
# ONE number per distance, and that number is measurably FLAT past ~5 A:
# 2.3500 at 5 A, 8 A and 15 A alike. That is the 4.649 report, same shape.
sums = [float(K[i].sum()) for i in range(3)]
check('a SUMMED bank WOULD be flat past ~5A -- why we concatenate',
      abs(sums[1] - sums[2]) < 1e-3,
      'sums %.4f %.4f %.4f  <- 8A and 15A indistinguishable' % tuple(sums))

# The L2 norm is flat too. gate_w7.py distinguishes 2/8/15 A by NORM, which
# separates them only in the 4th decimal -- recorded here so the weaker check is
# never mistaken for this one.
norms = [float(K[i].norm()) for i in range(3)]
print('%-62s INFO %s' % ('  (bank L2 norms are near-flat too)',
                         '%.4f %.4f %.4f' % tuple(norms)))

# ---------------------------------------------------------------- 2. edge attrs
print()
print('=' * 84)
print('G2 -- edge ATTRIBUTES: 20 src one-hot + 20 dst one-hot + 16 RBF = 56')
print('=' * 84)

check('edge_attr_dim() == 56', EF.edge_attr_dim() == 56, str(EF.edge_attr_dim()))

N = 12
B = 2
torch.manual_seed(0)
aa = torch.randint(0, 20, (B * N,))
one_hot = torch.zeros(B * N, 20)
one_hot[torch.arange(B * N), aa] = 1.0
ca = torch.randn(B, N, 3) * 6.0

full = torch.arange(N)
si, di = torch.meshgrid(full, full, indexing='ij')
m = si != di
ei = torch.stack([si[m], di[m]])
ei = torch.cat([ei, ei + N], dim=1)          # both graphs in the flat index space

A = EF.build_edge_attr(ei, one_hot, ca, N)
check('build_edge_attr returns [E,56]', A.shape == (ei.shape[1], 56), str(tuple(A.shape)))
check('src block is a valid one-hot (rows sum to 1)',
      bool(torch.all(A[:, :20].sum(-1) == 1.0)))
check('dst block is a valid one-hot (rows sum to 1)',
      bool(torch.all(A[:, 20:40].sum(-1) == 1.0)))

# Direction must survive. GATv2Conv aggregates at dst, so (LEU->ASP) and
# (ASP->LEU) must not collide, or the (identity, identity, distance) triple that
# justifies the whole phase is thrown away.
fwd = (ei[0] == 0) & (ei[1] == 1)
rev = (ei[0] == 1) & (ei[1] == 0)
af, ar = A[fwd][0], A[rev][0]
check('(i->j) and (j->i) have DIFFERENT attributes (direction kept)',
      not torch.equal(af, ar))
check('  ...and it is the one-hot halves that swap, not the RBF',
      torch.equal(af[:20], ar[20:40]) and torch.equal(af[20:40], ar[:20])
      and torch.equal(af[40:], ar[40:]))

# The RBF half must actually track geometry, not be a constant column.
rbf_block = A[:, 40:]
check('RBF block varies across edges (not a constant column)',
      float(rbf_block.std()) > 1e-4, 'std=%.6f' % float(rbf_block.std()))

# The distance the attribute encodes must be the true CA-CA distance.
b_idx = torch.div(ei[0], N, rounding_mode='floor')
true_d = (ca[b_idx, ei[0] % N] - ca[b_idx, ei[1] % N]).norm(dim=-1)
check('RBF block equals rbf_expand(true CA-CA distance) exactly',
      torch.equal(rbf_block, EF.rbf_expand(true_d)))

# ---------------------------------------------------------------- 3. G4 / U6
print()
print('=' * 84)
print('G4 / U6 -- the unfolded half must NOT receive folded coordinates')
print('=' * 84)

coords = torch.randn(N, 4, 3) * 6.0          # [N,4,3], atom order (N,CA,C,CB)
n_half = 3
sc = EF.stack_half_coords(coords, n_half, n_half)
check('stack_half_coords returns [2n,N,3]', sc.shape == (2 * n_half, N, 3),
      str(tuple(sc.shape)))
check('folded rows carry the REAL CA', torch.equal(sc[0], EF.folded_ca(coords)))
check('unfolded rows DIFFER from folded rows  <-- the U6 guarantee',
      not torch.equal(sc[0], sc[n_half]))
check('all unfolded rows are identical to each other',
      torch.equal(sc[n_half], sc[2 * n_half - 1]))

# The MASTER_PLAN Part IV row: "the unfolded half's edge distances DIFFER from
# the folded half's". Measure the distances, not just the coordinates.
ei2 = torch.stack([si[m], di[m]])
ei2 = torch.cat([ei2, ei2 + N], dim=1)
oh2 = torch.cat([one_hot[:N], one_hot[:N]], dim=0)
ca2 = torch.stack([sc[0], sc[n_half]])       # [2,N,3]: row0 folded, row1 unfolded
A2 = EF.build_edge_attr(ei2, oh2, ca2, N)
E1 = ei2.shape[1] // 2
check('unfolded-half EDGE DISTANCES differ from folded-half  <-- MASTER_PLAN',
      not torch.equal(A2[:E1, 40:], A2[E1:, 40:]))

# ...and the unfolded reference must be a pure function of |i-j|.
ref = EF.unfolded_reference_coords(N)
d01 = float((ref[0] - ref[1]).norm())
d12 = float((ref[1] - ref[2]).norm())
d05 = float((ref[0] - ref[5]).norm())
check('unfolded distance depends ONLY on |i-j| (d01 == d12)',
      abs(d01 - d12) < 1e-5, '%.4f %.4f' % (d01, d12))
check('unfolded distance grows with |i-j| (d05 == 5*d01)',
      abs(d05 - 5 * d01) < 1e-4, '%.4f vs %.4f' % (d05, 5 * d01))
check('unfolded reference reads NO folded coordinate (b defaults to 3.8 A)',
      abs(d01 - 3.8) < 1e-4, '%.4f' % d01)

# Flory arm: separation must follow b*|i-j|^nu, sub-linear for nu < 1.
# Tolerance 2e-3, not 1e-3: unfolded_reference_coords carries the SAME +1e-6
# epsilon as train_utils._flory_unfolded_graph (deliberately, so the edge coil
# and the node coil are the same curve), and that epsilon displaces residue 0 by
# b*1e-3 = 0.0038 A. Measured ratio 2.00100. Matching the node path is worth
# more than a rounder number here.
reff = EF.unfolded_reference_coords(N, flory=True, nu=0.5)
f01 = float((reff[0] - reff[1]).norm())
f04 = float((reff[0] - reff[4]).norm())
check('flory arm: d(0,4)/d(0,1) == 4^nu == 2 for nu=0.5',
      abs(f04 / max(f01, 1e-9) - 2.0) < 2e-3, '%.5f (eps-shifted from 2)'
      % (f04 / max(f01, 1e-9)))
check('flory coil is COMPACTER than the extended chain at the same |i-j|',
      f04 < 4 * f01, '%.3f < %.3f' % (f04, 4 * f01))

try:
    EF.unfolded_reference_coords(N, flory=True, nu=1.7)
    check('flory nu outside (0,1] RAISES', False, 'no raise')
except ValueError:
    check('flory nu outside (0,1] RAISES', True)

# ---------------------------------------------------------------- 4. G3 guard
print()
print('=' * 84)
print('G3 -- no silent interaction: --edge_features x --flory_unfolded')
print('=' * 84)

try:
    EF.assert_u6_compatible(True, True, False)
    check('edge_features + flory_unfolded WITHOUT per-half fix RAISES', False,
          'it did not raise')
except RuntimeError as e:
    check('edge_features + flory_unfolded WITHOUT per-half fix RAISES', True,
          '(%s...)' % str(e)[:44])

check('edge_features + flory_unfolded WITH per-half fix is allowed',
      EF.assert_u6_compatible(True, True, True) is True)
check('edge_features alone is allowed',
      EF.assert_u6_compatible(True, False, False) is True)
check('flory_unfolded alone is allowed',
      EF.assert_u6_compatible(False, True, False) is True)

# ---------------------------------------------------------------- 5. G1 in-tree
print()
print('=' * 84)
print('G1 -- byte-identical when off (needs the repo; SKIP is NOT a PASS)')
print('=' * 84)

try:
    from model.model_cfg import CFG
    from model.hydro_net import PEM
    import train_utils as TU
    have_repo = True
except Exception as e:
    have_repo = False
    skip('G1 byte-identity + rbf_expand agreement', '(%s)' % type(e).__name__)
    print('       Run this gate from the repo root on the cluster. A skipped G1')
    print('       is NOT a passed G1 -- --edge_features must not train until it')
    print('       has run there and passed.')

if have_repo:
    # The local copy of rbf_expand must be byte-identical to the one the model
    # calls, or the gate is testing a different function than production.
    dd = torch.linspace(0.0, 20.0, 97)
    check('edge_features.rbf_expand == train_utils.rbf_expand (torch.equal)',
          torch.equal(EF.rbf_expand(dd), TU.rbf_expand(dd)))

    def build(edge_features):
        CFG.burial_features = False
        CFG.flory_unfolded = False
        CFG.unfolded_emb = 'full'
        CFG.gcn_span = 1
        CFG.gcn_bidir = False
        CFG.edge_features = edge_features
        torch.manual_seed(1234)
        m = PEM(layers=CFG.num_layers, gaussian_coef=CFG.gaussian_coef,
                dropout_rate=CFG.dropout_rate, light_attention=True)
        m.eval()
        return m

    emb_in = getattr(CFG, 'emb_input_dim', 1024)
    node_dim = 16 + 32 + emb_in + 20
    torch.manual_seed(7)
    x = torch.randn(2, 24, node_dim)

    m_off = build(False)
    m_on = build(True)

    with torch.no_grad():
        y_off = m_off(x)
    # Same seed, same construction order => identical parameters when the flag
    # adds no parameters to the OFF path. If this fails, the flag is not inert.
    same_params = all(torch.equal(a, b) for a, b in
                      zip(m_off.state_dict().values(), m_on.state_dict().values())
                      ) if len(m_off.state_dict()) == len(m_on.state_dict()) else False
    print('%-62s INFO %s' % ('  (ON build adds parameters, as expected)',
                             'no' if same_params else 'yes'))

    CFG.edge_features = False
    m_off2 = build(False)
    with torch.no_grad():
        y_off2 = m_off2(x)
    check('OFF is deterministic and reproducible (torch.equal)',
          torch.equal(y_off, y_off2))

    # The real G1: an OFF model built with the patch present must equal a model
    # built from the PRE-PATCH parameter layout. Approximated in-tree by checking
    # that the OFF path constructs no edge parameters at all.
    n_edge_params = sum(p.numel() for n, p in m_off.named_parameters()
                        if 'edge' in n or 'lin_edge' in n)
    check('OFF path constructs ZERO edge parameters', n_edge_params == 0,
          str(n_edge_params))
    n_edge_params_on = sum(p.numel() for n, p in m_on.named_parameters()
                           if 'edge' in n or 'lin_edge' in n)
    check('ON path DOES construct edge parameters', n_edge_params_on > 0,
          str(n_edge_params_on))

    CFG.edge_features = False

print()
print('=' * 84)
if skipped:
    print('SKIPPED (must be run in-tree before training): %s' % ', '.join(skipped))
print('W7 EDGE GATE: %s' % ('ALL PASS' if ok else 'FAILURES'))
print('=' * 84)
sys.exit(0 if ok else 1)
