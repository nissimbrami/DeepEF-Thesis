"""Gate for --slope_weight (factor C). It had NONE, while 40 queued jobs use it.

This is the thesis's own claimed contribution and the only lever that attacks a_p
(the compression slope, median 0.4990, corr(a_p, per-protein PCC) = +0.5714).
It was running unverified.

The term, copied verbatim from Megascale-fineTuning/train.py:
    slope_pred_ddg = output - wt_dg_slope
    slope_true_ddg = delta_g - delta_g_wt_slope
    slope_loss = |std(slope_pred_ddg, unbiased=False) - std(slope_true_ddg, unbiased=False)|
    loss = data_loss + ... + SLOPE_WEIGHT * slope_loss     [only if SLOPE_WEIGHT > 0 and n >= 2]

What must hold, and what each check protects against:
  1. ZERO when the predicted spread already matches the true spread   (no penalty when correct)
  2. POSITIVE and symmetric when compressed or expanded  (this is what makes it a SLOPE term:
     it must punish under-reaction, which is exactly the a_p < 1 failure we measured)
  3. SHIFT-INVARIANT: adding a constant to every prediction must not change it. A per-protein
     OFFSET is b_p's business, not the slope's. If this failed, the term would be silently
     fighting the anchor.
  4. SCALE-CORRECT: predictions scaled by k give |k-1| * std(true) exactly.
  5. unbiased=False really is population std (divide by n), not sample std (n-1).
  6. INERT at weight 0 -- nothing is built, so the loss is bit-identical to baseline.
  7. GRADIENT actually flows to the predictions (a term with no gradient is decoration).
  8. Minimised exactly at the true spread -- descending the gradient must REDUCE compression.
"""
import sys
sys.path.insert(0, '/home/nissimb/DeepPEF')
import torch

ok = True
def chk(name, cond, extra=''):
    global ok
    print('%-62s %s %s' % (name, 'PASS' if cond else 'FAIL', extra))
    if not cond: ok = False

def slope_loss(pred_ddg, true_ddg):
    """Verbatim reimplementation of train.py's term."""
    return torch.abs(pred_ddg.std(unbiased=False) - true_ddg.std(unbiased=False))

torch.manual_seed(0)
true = torch.tensor([0.0, -1.5, 2.0, -0.5, 1.0, -2.5, 0.75, 3.0])   # WT=0 at row 0, WS-1
s_true = float(true.std(unbiased=False))
print('true ddG spread (population std) = %.6f\n' % s_true)

# 1 - zero when spreads match
chk('1 perfect match -> loss == 0', float(slope_loss(true.clone(), true)) < 1e-7,
    'loss=%.3e' % float(slope_loss(true.clone(), true)))

# 2 - punishes compression AND expansion, symmetrically
comp, expa = slope_loss(true * 0.5, true), slope_loss(true * 1.5, true)
chk('2a compression (a_p=0.5) penalised', float(comp) > 0, 'loss=%.4f' % float(comp))
chk('2b expansion (a_p=1.5) penalised', float(expa) > 0, 'loss=%.4f' % float(expa))
chk('2c symmetric: |0.5| and |1.5| deviate equally',
    abs(float(comp) - float(expa)) < 1e-6, '%.6f vs %.6f' % (float(comp), float(expa)))

# 3 - SHIFT INVARIANCE: the offset is b_p's job, not the slope's
base = slope_loss(true * 0.5, true)
for sh in (-2.0, -0.5, 0.5, 2.0):
    chk('3 shift-invariant (offset %+.1f is b_p, not a_p)' % sh,
        abs(float(slope_loss(true * 0.5 + sh, true)) - float(base)) < 1e-6)

# 4 - exact scale relation
for k in (0.25, 0.5, 0.75, 1.25, 2.0):
    want = abs(k - 1.0) * s_true
    got = float(slope_loss(true * k, true))
    chk('4 scale k=%.2f -> |k-1|*std(true)' % k, abs(got - want) < 1e-5,
        'got %.6f want %.6f' % (got, want))

# 5 - population std, not sample std
n = true.numel()
pop = float(true.std(unbiased=False)); samp = float(true.std(unbiased=True))
chk('5 unbiased=False is population std (/n not /(n-1))',
    abs(pop - samp * ((n - 1) / n) ** 0.5) < 1e-6, 'pop=%.6f sample=%.6f' % (pop, samp))

# 6 - inert at weight 0 (the guard in train.py is `if SLOPE_WEIGHT > 0`)
W = 0.0
data_loss = torch.tensor(1.2345)
chk('6 weight 0 -> loss bit-identical to data_loss',
    float(data_loss + W * slope_loss(true * 0.5, true)) == float(data_loss))

# 7 - gradient reaches the predictions
p = (true * 0.5).clone().requires_grad_(True)
slope_loss(p, true).backward()
chk('7 gradient flows to predictions', p.grad is not None and float(p.grad.abs().sum()) > 0,
    'sum|grad|=%.4f' % float(p.grad.abs().sum()))

# 8 - descending the gradient REDUCES compression (the whole point)
p2 = (true * 0.5).clone().requires_grad_(True)
before = float(p2.std(unbiased=False))
for _ in range(200):
    l = slope_loss(p2, true)
    g, = torch.autograd.grad(l, p2)
    with torch.no_grad():
        p2 -= 0.05 * g
after = float(p2.std(unbiased=False))
chk('8 gradient descent moves spread TOWARD the true spread',
    abs(after - s_true) < abs(before - s_true),
    'spread %.4f -> %.4f (target %.4f)' % (before, after, s_true))

# 9 - the n<2 guard: a single mutation has no spread to match
chk('9 single-element input has zero spread (guard n>=2 is required)',
    float(torch.tensor([1.0]).std(unbiased=False)) == 0.0)

# 10 - the --loss_mode dg refusal really exists in train.py
src = open('/home/nissimb/DeepPEF/Megascale-fineTuning/train.py').read()
chk('10 --slope_weight with --loss_mode dg raises',
    "SLOPE_WEIGHT > 0 and LOSS_MODE == 'dg'" in src and 'raise' in src.split(
        "SLOPE_WEIGHT > 0 and LOSS_MODE == 'dg'")[1][:400])

# 11 - the built term in train.py is the one tested here
chk('11 train.py uses abs(std(pred)-std(true)) with unbiased=False',
    "torch.abs(slope_pred_ddg.std(unbiased=False) - slope_true_ddg.std(unbiased=False))" in src)
chk('12 train.py builds NOTHING when weight is 0',
    'if SLOPE_WEIGHT > 0 and output.numel() >= 2:' in src)

print('\nSLOPE GATE: %s' % ('ALL PASS' if ok else 'FAILURES'))
sys.exit(0 if ok else 1)
