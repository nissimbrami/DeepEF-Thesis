"""G1 + G2 gates for U2 (--unfolded_emb), factor D of the calibration factorial.

G1 inertness: with the flag at its default the unfolded graph must be BYTE-identical
to the pre-patch build. torch.equal, never allclose -- an inert default is what
protects every baseline number already banked.

G2 semantics: it must actually do what it is named for -- the embedding block of the
UNFOLDED graph is all zeros under 'zero', while the FOLDED graph is untouched.
"""
import os
import sys
os.environ.setdefault('WANDB_MODE', 'disabled')
sys.path.insert(0, os.getcwd())
import torch

from model.model_cfg import CFG
from train_utils import get_graph, get_unfolded_graph

torch.manual_seed(0)
N = 40
x = torch.randn(N, 4, 3)
oh = torch.eye(20)[torch.randint(0, 20, (N,))]
emb = torch.randn(N, int(CFG.emb_input_dim))
mask = torch.ones(N)

D_W, FB_W, OH_W = 16, 32, 20
EMB_LO = D_W + FB_W
EMB_HI = EMB_LO + int(CFG.emb_input_dim)

ok = True


def check(name, cond):
    global ok
    print('%-58s %s' % (name, 'PASS' if cond else 'FAIL'))
    if not cond:
        ok = False


CFG.flory_unfolded = False

# --- G1: default is inert and reproducible ---
CFG.unfolded_emb = 'full'
a = get_unfolded_graph(x, oh, emb, mask)
b = get_unfolded_graph(x, oh, emb, mask)
check('G1 default reproducible (torch.equal)', torch.equal(a, b))

if not hasattr(CFG, 'unfolded_emb_absent_probe'):
    delattr(CFG, 'unfolded_emb')
c = get_unfolded_graph(x, oh, emb, mask)   # attribute missing -> getattr default 'full'
CFG.unfolded_emb = 'full'
check('G1 missing attribute == full (torch.equal)', torch.equal(a, c))

folded_before = get_graph(x, oh, emb, mask)

# --- G2: zero actually zeroes, and only in the unfolded pass ---
CFG.unfolded_emb = 'zero'
z = get_unfolded_graph(x, oh, emb, mask)
check('G2 zero: unfolded emb block is all zeros', bool(torch.all(z[:, EMB_LO:EMB_HI] == 0)))
check('G2 zero: unfolded graph differs from full', not torch.equal(a, z))
check('G2 zero: shape unchanged', a.shape == z.shape)
check('G2 zero: one-hot block untouched',
      torch.equal(a[:, EMB_HI:EMB_HI + OH_W], z[:, EMB_HI:EMB_HI + OH_W]))
check('G2 zero: D+Fb blocks untouched', torch.equal(a[:, :EMB_LO], z[:, :EMB_LO]))

folded_after = get_graph(x, oh, emb, mask)
check('G2 FOLDED pass unaffected by the flag (torch.equal)',
      torch.equal(folded_before, folded_after))
check('G2 folded emb block is NOT zero',
      bool(torch.any(folded_after[:, EMB_LO:EMB_HI] != 0)))

# --- mean mode ---
CFG.unfolded_emb = 'mean'
m = get_unfolded_graph(x, oh, emb, mask)
check('mean: differs from both full and zero',
      (not torch.equal(m, a)) and (not torch.equal(m, z)))
check('mean: every residue row identical in the emb block',
      bool(torch.allclose(m[:, EMB_LO:EMB_HI], m[0:1, EMB_LO:EMB_HI].expand(N, -1),
                          atol=1e-6)))

# --- composes with the coil, since the coil path got the same treatment ---
CFG.flory_unfolded = True
CFG.unfolded_emb = 'full'
cf = get_unfolded_graph(x, oh, emb, mask)
CFG.unfolded_emb = 'zero'
cz = get_unfolded_graph(x, oh, emb, mask)
check('coil path: zero also applies there',
      bool(torch.all(cz[:, EMB_LO:EMB_HI] == 0)) and not torch.equal(cf, cz))

CFG.flory_unfolded = False
CFG.unfolded_emb = 'full'

# --- bad value must raise, not silently pass through ---
CFG.unfolded_emb = 'nonsense'
try:
    get_unfolded_graph(x, oh, emb, mask)
    check('invalid mode raises', False)
except ValueError:
    check('invalid mode raises ValueError', True)
CFG.unfolded_emb = 'full'

print('')
print('U2 GATE: %s' % ('ALL PASS' if ok else 'FAILURES PRESENT -- revert, do not debug in place'))
sys.exit(0 if ok else 1)
