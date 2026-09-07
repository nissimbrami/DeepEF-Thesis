# -*- coding: utf-8 -*-
"""Diagnose the A-vs-B gap: is it the resume logic, or the harness's own RNG?

The model here is deterministic (no dropout) and the data order is fixed, so a
full-state resume SHOULD be bit-exact. If it is not, find out which piece differs.
"""
from __future__ import print_function
import torch, torch.nn as nn, torch.optim as optim


def make_data(n=64, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w = torch.randn(d, 1, generator=g)
    return X, X @ w + 0.1 * torch.randn(n, 1, generator=g)


class Net(nn.Module):
    def __init__(self, d=8):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(d, 16); self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def build():
    torch.manual_seed(1234)
    m = Net()
    o = optim.Adam(m.parameters(), lr=1e-2)
    s = optim.lr_scheduler.ReduceLROnPlateau(o, mode='max', factor=0.1, patience=0)
    return m, o, s


X, y = make_data()
crit = nn.L1Loss()


def epoch_step(m, o, s):
    m.train()
    for i in range(0, X.size(0), 16):
        o.zero_grad(); l = crit(m(X[i:i+16]), y[i:i+16]); l.backward(); o.step()
    s.step(0.5)


def flat(m):
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()])


# --- A: 3 epochs straight
mA, oA, sA = build()
for _ in range(3):
    epoch_step(mA, oA, sA)

# --- B: 1 epoch, save FULL state, rebuild, restore, 2 more
mB, oB, sB = build()
epoch_step(mB, oB, sB)
ck = {'m': mB.state_dict(), 'o': oB.state_dict(), 's': sB.state_dict()}
mB2, oB2, sB2 = build()
mB2.load_state_dict(ck['m']); oB2.load_state_dict(ck['o']); sB2.load_state_dict(ck['s'])
for _ in range(2):
    epoch_step(mB2, oB2, sB2)

# --- C: same but WITHOUT optimizer/scheduler restore
mC, oC, sC = build()
epoch_step(mC, oC, sC)
ck2 = {'m': mC.state_dict()}
mC2, oC2, sC2 = build()
mC2.load_state_dict(ck2['m'])
for _ in range(2):
    epoch_step(mC2, oC2, sC2)

a, b, c = flat(mA), flat(mB2), flat(mC2)
print('A vs B (full state restored):  bit-identical=%s  max|d|=%.3e'
      % (torch.equal(a, b), float((a-b).abs().max())))
print('A vs C (model only restored):  bit-identical=%s  max|d|=%.3e'
      % (torch.equal(a, c), float((a-c).abs().max())))

# Is the optimizer state itself identical after restore?
def opt_sig(o):
    st = o.state_dict()['state']
    return torch.cat([torch.cat([v['exp_avg'].reshape(-1), v['exp_avg_sq'].reshape(-1),
                                 torch.tensor([float(v['step'])])]) for v in st.values()])

print('\nAdam state after the run:')
print('  A vs B exp_avg/exp_avg_sq/step identical =', torch.equal(opt_sig(oA), opt_sig(oB2)))
print('  A vs C identical =', torch.equal(opt_sig(oA), opt_sig(oC2)))
print('  A step count =', [v['step'] for v in oA.state_dict()['state'].values()][:1])
print('  C step count =', [v['step'] for v in oC2.state_dict()['state'].values()][:1])
