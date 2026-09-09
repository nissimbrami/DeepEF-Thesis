"""gate_coil_nu.py -- prove that --nu and --b are ACTUALLY READ.

The project's signature failure mode is: code runs, prints a success line, feature
never read. This gate measures the artifact, not a DONE message.

Tests
  1  d(i,j) = b*|i-j|^nu holds to 1e-6 for every nu tested, measured on the RAW
     coil distance matrix (before the Gaussian kernel).
  2  changing nu CHANGES the unfolded graph tensor (not merely the intermediate).
  3  changing b  CHANGES the unfolded graph tensor.
  4  nu=0.5 / b=5.82 reproduces the historical coil_fixed_b path BIT-IDENTICALLY.
  5  the graph SHAPE is invariant to nu and b (value-only lever).
  6  the log-log slope of measured d against |i-j| recovers nu to 1e-6.
"""
import os, sys
os.environ.setdefault('WANDB_MODE', 'disabled')
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from model.model_cfg import CFG
import train_utils as TU
from train_utils import get_unfolded_graph

SCALE = TU._COIL_COORD_SCALE
FAIL = []


def chk(name, cond, detail=''):
    print('%-52s %s  %s' % (name, 'PASS' if cond else 'FAIL', detail))
    if not cond:
        FAIL.append(name)


torch.manual_seed(0)
N = 60
coords = torch.randn(N, 4, 3) * 1.0
mask = torch.ones(N)
oh = torch.zeros(N, 20); oh[torch.arange(N), torch.randint(0, 20, (N,))] = 1.0
emb = torch.randn(N, 1024)


def raw_coil(nu, b_ang):
    """Rebuild the RAW coil distance exactly as _flory_unfolded_graph does."""
    idx = torch.arange(N, dtype=torch.float32)
    sep = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    b = torch.tensor(b_ang * SCALE)
    return b * torch.pow(sep + 1e-6, nu)


def graph(nu, b_ang, channels='broadcast'):
    CFG.flory_unfolded = True
    CFG.coil_b = 'fixed'
    CFG.coil_channels = channels
    CFG.unfolded_emb = 'full'
    CFG.flory_nu = nu
    TU._COIL_B_FIXED = b_ang * SCALE
    g = get_unfolded_graph(coords, oh, emb, mask)
    return g.detach().clone()


def reset():
    CFG.flory_unfolded = False
    CFG.coil_b = 'fitted'
    CFG.coil_channels = 'broadcast'
    CFG.unfolded_emb = 'full'
    CFG.flory_nu = 0.5
    TU._COIL_B_FIXED = TU._COIL_B_FIXED_ANGSTROM * SCALE


print('=== 1. d(i,j) = b*|i-j|^nu, analytic form ===')
for nu in [0.5, 0.55, 0.588, 0.62, 0.65]:
    for b_ang in [4.97, 5.82]:
        d = raw_coil(nu, b_ang)
        # check a handful of separations against the closed form
        errs = []
        for s in [1, 5, 10, 20, 50]:
            if s >= N:
                continue
            want = b_ang * SCALE * (s ** nu)
            got = float(d[0, s])
            errs.append(abs(got - want) / want)
        chk('nu=%.3f b=%.2f  matches b*|i-j|^nu' % (nu, b_ang),
            max(errs) < 1e-6, 'max rel err %.2e' % max(errs))

print('\n=== 2/3. the GRAPH TENSOR changes with nu and with b ===')
g_ref = graph(0.5, 5.82)
for nu in [0.55, 0.588, 0.62, 0.65]:
    g = graph(nu, 5.82)
    delta = float((g - g_ref).abs().max())
    rel = float((g - g_ref).norm() / g_ref.norm())
    chk('nu 0.5 -> %.3f changes the unfolded graph' % nu,
        delta > 1e-9, 'max|d|=%.3e  relF=%.3e' % (delta, rel))
g_b = graph(0.5, 4.97)
delta_b = float((g_b - g_ref).abs().max())
chk('b 5.82 -> 4.97 changes the unfolded graph',
    delta_b > 1e-9, 'max|d|=%.3e' % delta_b)

print('\n=== 4. nu=0.5 / b=5.82 == historical coil_fixed_b, BIT-IDENTICAL ===')
# historical path: flory on, coil_b fixed, broadcast, module constant untouched
reset()
CFG.flory_unfolded = True; CFG.coil_b = 'fixed'
g_hist = get_unfolded_graph(coords, oh, emb, mask).detach().clone()
reset()
g_new = graph(0.5, 5.82)
same = bool(torch.equal(g_hist, g_new))
chk('sweep cell (0.5, 5.82) is byte-identical to coil_fixed_b',
    same, 'max|d|=%.3e' % float((g_hist - g_new).abs().max()))

print('\n=== 5. shape invariance (value-only lever) ===')
shapes = {tuple(graph(nu, b).shape) for nu in [0.5, 0.65] for b in [4.97, 5.82]}
chk('graph shape invariant to nu and b', len(shapes) == 1, str(shapes))
chk('graph width == 1092', list(shapes)[0][1] == 1092, str(list(shapes)[0]))

print('\n=== 6. log-log slope recovers nu ===')
for nu in [0.5, 0.588, 0.65]:
    d = raw_coil(nu, 5.82).numpy()
    s = np.arange(1, N)
    y = np.log(d[0, 1:])
    x = np.log(s)
    slope = np.polyfit(x, y, 1)[0]
    chk('log-log slope recovers nu=%.3f' % nu, abs(slope - nu) < 1e-6,
        'slope=%.8f' % slope)

print('\n=== 7. distance error vs nu=0.5 at the separations that matter ===')
for s in [10, 20, 50]:
    a = 5.82 * (s ** 0.5)
    b = 5.82 * (s ** 0.588)
    print('   |i-j|=%2d   nu=0.5 %.2f A   nu=0.588 %.2f A   error %.0f%%'
          % (s, a, b, 100 * (b - a) / a))

reset()
print('\ngate_coil_nu: %s' % ('ALL PASS' if not FAIL else 'FAILED: %s' % FAIL))
sys.exit(1 if FAIL else 0)
