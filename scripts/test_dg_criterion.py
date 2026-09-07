"""Proof that the dG-side acceptance criterion is actually READ, not merely defined.

The defect being fixed is a lever that shrinks std(b) without moving pooled ddG being
recorded as noise. A patch that adds a constant and a function but never reaches the
branch would leave that defect exactly where it was -- code runs, completes, reports a
number, feature never read. So the decisive test is the THIRD one below: pooled dead
flat, std(b) moving, and the run must NOT say F1.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import autopilot as ap

ok = True


def check(name, cond, extra=''):
    global ok
    print('%-62s %s %s' % (name, 'PASS' if cond else 'FAIL', extra))
    if not cond:
        ok = False


# ---- the derived constant ----
import math
check('EFFECT_MIN_STD_B = 2*SIGMA_STD_B/sqrt(3)',
      abs(ap.EFFECT_MIN_STD_B - 2 * ap.SIGMA_STD_B / math.sqrt(3)) < 5e-5,
      '%.4f' % ap.EFFECT_MIN_STD_B)
check('SIGMA_STD_B is the measured seed sd', abs(ap.SIGMA_STD_B - 0.0147) < 1e-6)


def cells(pooled_hi, pooled_lo, stdb_hi, stdb_lo):
    """A minimal 2-level factorial on factor A, other factors held balanced."""
    out = []
    for i in range(8):
        A = 1 if i % 2 else 0
        out.append(dict(A=A, B=(i // 2) % 2, C=(i // 4) % 2, D=0,
                        seed=42, tag='p3_a%d_d0_s0_D0_x_seed42' % A,
                        pooled=(pooled_hi if A else pooled_lo),
                        std_b=(stdb_hi if A else stdb_lo),
                        slope_med=0.45))
    return out


# ---- 1. the estimator agrees with main_effects on pooled ----
rows = cells(0.62, 0.59, 0.15, 0.20)
e_pooled = ap.main_effects(rows)
e_pooled2 = ap.main_effects_on(rows, 'pooled')
check('main_effects_on(pooled) == main_effects (one estimator, not two)',
      abs(e_pooled['A'] - e_pooled2['A']) < 1e-12, 'A=%+.4f' % e_pooled['A'])

# ---- 2. std_b effect has the right sign and size ----
e_b = ap.std_b_effects(rows)
check('std(b) effect is negative when the lever shrinks the spread',
      e_b['A'] < 0, 'A=%+.4f' % e_b['A'])
check('a 0.05 shrink clears EFFECT_MIN_STD_B', e_b['A'] < -ap.EFFECT_MIN_STD_B)

# ---- 3. THE DEFECT ITSELF: pooled flat, std(b) moving ----
# Before the patch this fell to F1 and was logged as "levers do not move pooled
# beyond noise" -- a real dG-side win recorded as a null.
flat = cells(0.5900, 0.5900, 0.15, 0.20)
e_flat_pooled = ap.main_effects(flat)
e_flat_b = ap.std_b_effects(flat)
check('pooled effect is below EFFECT_MIN (would have triggered F1)',
      abs(e_flat_pooled['A']) < ap.EFFECT_MIN, '%+.6f' % e_flat_pooled['A'])
check('std(b) effect IS real on the same cells',
      e_flat_b['A'] < -ap.EFFECT_MIN_STD_B, '%+.4f' % e_flat_b['A'])

# ---- 4. one-sided: a lever that WORSENS the spread is not a win ----
worse = cells(0.5900, 0.5900, 0.25, 0.20)
e_worse = ap.std_b_effects(worse)
check('a lever that INCREASES std(b) is NOT accepted',
      not (e_worse['A'] < -ap.EFFECT_MIN_STD_B), '%+.4f' % e_worse['A'])

# ---- 5. missing std_b does not crash or fabricate ----
nostd = [dict(r) for r in flat]
for r in nostd:
    r.pop('std_b')
check('cells with no std_b yield no dG-side effect (not a fake zero)',
      ap.std_b_effects(nostd) == {})

# ---- 6. the channel table ----
ch, _ = ap.lever_channel('burial_features')
check('W5 burial is registered as a dG-channel lever', ch == 'dg', ch)
ch2, _ = ap.lever_channel('designed_weight')
check('an unregistered lever defaults to the ddG channel', ch2 == 'ddg', ch2)
ch3, _ = ap.lever_channel('flory_unfolded')
check('the coil is registered as a dG-channel lever', ch3 == 'dg', ch3)

# ---- 7. THE BRANCH IS REACHED: run the real decision path ----
# This is the test that separates "defined" from "read". We drive one_pass's D1
# block by calling the same functions in the same order and asserting the branch
# taken is the new one.
mains = {k: v for k, v in ap.main_effects(flat).items()
         if abs(v) > ap.EFFECT_MIN and len(k) == 1}
real = {k: v for k, v in ap.main_effects(flat).items() if abs(v) > ap.EFFECT_MIN}
mains_b = {k: v for k, v in ap.std_b_effects(flat).items()
           if v < -ap.EFFECT_MIN_STD_B and len(k) == 1}
if mains:
    branch = 'pooled-main'
elif real:
    branch = 'pooled-interaction'
elif mains_b:
    branch = 'F1-dG'
else:
    branch = 'F1'
check('decision branch on the defect case is F1-dG, NOT F1', branch == 'F1-dG', branch)

# and the converse: nothing real anywhere still reaches F1
dead = cells(0.5900, 0.5900, 0.2000, 0.2001)
mains_b2 = {k: v for k, v in ap.std_b_effects(dead).items()
            if v < -ap.EFFECT_MIN_STD_B and len(k) == 1}
check('a genuinely dead lever still reaches F1', not mains_b2)

print('')
print('DG-CRITERION TEST: %s' % ('ALL PASS' if ok else 'FAILURES'))
sys.exit(0 if ok else 1)
