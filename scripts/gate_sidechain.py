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
import sys, math
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

print('\nW12 SIDECHAIN GATE: %s' % ('ALL PASS' if ok else 'FAILURES'))
sys.exit(0 if ok else 1)
