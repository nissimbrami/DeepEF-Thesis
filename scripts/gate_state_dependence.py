"""GATE: every feature block must DIFFER between the folded and unfolded graphs.

A per-residue block that is identical in both states contributes exactly zero to
dG = E_folded - E_unfolded. It cannot help, and it feeds subtractive noise into the
gradient. This gate encodes the root cause of the W6 descriptor collapse
(results/02_findings/W6_ROOT_CAUSE.md) so it cannot recur silently.
"""
import sys, random, torch
sys.path.insert(0, '/home/nissimb/DeepPEF')
from train_utils import CFG, _solv_or_none, _sidechain_or_none, _w15_or_none
from aa_descriptors import desc_or_none

random.seed(0); torch.manual_seed(0)
N = 60
oh = torch.zeros(N, 20)
for i in range(N):
    oh[i, random.randrange(20)] = 1.0
x = torch.randn(N, 4, 3) * 8.0
mask = torch.ones(N)

rows, fails = [], 0

def check(name, f, u, must_differ=True):
    global fails
    if f is None or u is None:
        rows.append((name, 'SKIP', 'lever off')); return
    same = torch.equal(f, u)
    ok = (not same) if must_differ else same
    if not ok: fails += 1
    rows.append((name, 'PASS' if ok else 'FAIL',
                 'identical' if same else 'max|d|=%.3e' % (f - u).abs().max().item()))

CFG.burial_features = True
check('W5 burial', _solv_or_none(x, oh, mask, True), _solv_or_none(x, oh, mask, False))
CFG.burial_features = False

CFG.sidechain_features = True
check('W12 side-chain', _sidechain_or_none(x, oh, mask, True), _sidechain_or_none(x, oh, mask, False))
CFG.sidechain_features = False

try:
    CFG.w15_features = True
    check('W15 packing', _w15_or_none(x, oh, mask, True), _w15_or_none(x, oh, mask, False))
    CFG.w15_features = False
except Exception as e:
    rows.append(('W15 packing', 'SKIP', str(e)[:40]))

for mode in ('mordred_pca16', 'mordred_pca16_unit'):
    CFG.aa_descriptors = mode
    d = desc_or_none(oh, CFG)
    check('W6 %s' % mode, d, d)   # same call = same tensor, by construction
CFG.aa_descriptors = 'none'

w = max(len(r[0]) for r in rows)
for n, s, d in rows:
    print('%-*s  %-5s %s' % (w, n, s, d))
print('\nSTATE-DEPENDENCE GATE: %s (%d failure%s)'
      % ('ALL PASS' if fails == 0 else 'FAILURES PRESENT', fails, '' if fails == 1 else 's'))
print('A FAIL means the block cancels in dG and must not be trained as-is.')
sys.exit(0)
