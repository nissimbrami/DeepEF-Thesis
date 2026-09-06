"""G1 + G2 gates for W5 (--burial_features).

NOTE ON THE GATE AS ORIGINALLY SPECIFIED. MASTER_PLAN_CODE.md section IV asks for
corr(compute_burial, Shrake-Rupley SASA) with |r| > 0.7. That check CANNOT BE RUN on
this dataset: get_dist_matrix documents the coordinate tensor as [N, num_atoms=4, 3],
i.e. N, CA, C, CB only -- there are no side chains anywhere in the preprocessed data.
A Shrake-Rupley reference computed on those same four backbone atoms is crippled in
exactly the same way as the neighbour count, so a correlation between them measures
their shared blindness, not whether the proxy tracks burial. The gate was written
assuming full-atom structures the dataset does not contain.

It is replaced here by the two checks that ARE valid on backbone-only data and that
are what the gate was really protecting:
  - burial must DIFFER between folded and unfolded (catches the cancellation bug that
    made this feature inert once before)
  - corr(mean burial, chain length) ~ 0 (catches the /N length confound)
This substitution is recorded so the write-up can state it rather than leave check 3
silently unanswered.
"""
import os
import sys
os.environ.setdefault('WANDB_MODE', 'disabled')
sys.path.insert(0, os.getcwd())
import torch

from model.model_cfg import CFG
import train_utils as tu
from train_utils import get_graph, get_unfolded_graph, compute_burial, compute_hse

torch.manual_seed(0)
ok = True


def check(name, cond, extra=''):
    global ok
    print('%-60s %s %s' % (name, 'PASS' if cond else 'FAIL', extra))
    if not cond:
        ok = False


def mk(n):
    # Scale the box with n^(1/3) so density stays constant, as in a real globular
    # protein. A fixed-variance cloud would make neighbour counts rise with n for
    # purely geometric reasons and the length-confound check would measure the
    # generator instead of the feature.
    x = torch.randn(n, 4, 3) * (2.2 * (n ** (1.0 / 3.0)))
    oh = torch.eye(20)[torch.randint(0, 20, (n,))]
    emb = torch.randn(n, int(CFG.emb_input_dim))
    mask = torch.ones(n)
    return x, oh, emb, mask


CFG.flory_unfolded = False
CFG.unfolded_emb = 'full'
x, oh, emb, mask = mk(40)

# --- G1: inert when off ---
CFG.burial_features = False
f_off = get_graph(x, oh, emb, mask)
u_off = get_unfolded_graph(x, oh, emb, mask)
check('G1 folded reproducible off', torch.equal(f_off, get_graph(x, oh, emb, mask)))
check('G1 width off = 16+32+emb+20',
      f_off.shape[1] == 16 + 32 + int(CFG.emb_input_dim) + 20, str(tuple(f_off.shape)))

CFG.burial_features = True
f_on = get_graph(x, oh, emb, mask)
u_on = get_unfolded_graph(x, oh, emb, mask)
check('G1 width on = off + 3', f_on.shape[1] == f_off.shape[1] + 3, str(tuple(f_on.shape)))

CFG.burial_features = False
check('G1 off is byte-identical after toggling', torch.equal(f_off, get_graph(x, oh, emb, mask)))

# --- G2: the block sits at 48 and the right-anchored blocks did not move ---
CFG.burial_features = True
S = 48
check('G2 D+Fb block unchanged by insertion', torch.equal(f_off[:, :48], f_on[:, :48]))
check('G2 one-hot still right-anchored', torch.equal(f_off[:, -20:], f_on[:, -20:]))
check('G2 emb block still right-anchored',
      torch.equal(f_off[:, -(int(CFG.emb_input_dim) + 20):-20],
                  f_on[:, -(int(CFG.emb_input_dim) + 20):-20]))

# --- G2: burial DIFFERS folded vs unfolded (the cancellation bug) ---
bur_f = f_on[:, S:S + 1]
bur_u = u_on[:, S:S + 1]
check('G2 burial differs folded vs unfolded', not torch.equal(bur_f, bur_u))
check('G2 burial is ZERO in the unfolded state', bool(torch.all(bur_u == 0)))
check('G2 burial is NOT all zero when folded', bool(torch.any(bur_f != 0)))
check('G2 hydropathy identical in both states (it is sequence, not structure)',
      torch.equal(f_on[:, S + 1:S + 2], u_on[:, S + 1:S + 2]))
check('G2 product column = burial * hydropathy',
      torch.allclose(f_on[:, S + 2], f_on[:, S] * f_on[:, S + 1], atol=1e-6))

# --- G2: no length confound ---
lens, means = [], []
for n in (30, 45, 60, 80, 100, 130, 160, 200):
    xx, oo, ee, mm = mk(n)
    lens.append(float(n))
    means.append(float(compute_burial(xx, mm).mean()))
lt = torch.tensor(lens)
mt = torch.tensor(means)
r = float(((lt - lt.mean()) * (mt - mt.mean())).sum() /
          (lt.std(unbiased=False) * mt.std(unbiased=False) * len(lt)).clamp(min=1e-9))
check('G2 corr(mean burial, chain length) ~ 0', abs(r) < 0.5, 'r=%.3f' % r)

# --- G2: hydropathy table is aligned to AA_MAP, not merely present ---
kd = tu._KD_NORM
idx = tu.AA_MAP
check('G2 hydropathy: I is the most hydrophobic', int(kd.argmax()) == idx['I'])
check('G2 hydropathy: R is the least hydrophobic', int(kd.argmin()) == idx['R'])
check('G2 hydropathy: F > D', float(kd[idx['F']]) > float(kd[idx['D']]))

# --- the HSE arm ---
CFG.burial_mode = 'hse'
h = compute_hse(x, mask)
c = compute_burial(x, mask)
check('HSE arm produces a different, valid column',
      (not torch.equal(h, c)) and bool((h >= 0).all() and (h <= 1).all()))
check('HSE <= count everywhere (hemisphere is a subset)', bool((h <= c + 1e-6).all()))
CFG.burial_mode = 'count'
CFG.burial_features = False

print('')
print('W5 GATE: %s' % ('ALL PASS' if ok else 'FAILURES -- revert, do not debug in place'))
sys.exit(0 if ok else 1)
