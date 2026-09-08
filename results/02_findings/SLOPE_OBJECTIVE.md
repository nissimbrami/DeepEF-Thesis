# Is |std(pred) - std(true)| the right objective for decompressing a_p?

**Short answer: it is the right SIGN and the wrong CEILING.** The std-matching objective can
drive the per-protein slope `a_p` no higher than the per-protein correlation `r`. Our `r` is
0.7930, so a *perfectly* std-matched DeepEF still has `a_p = 0.793`, never 1.0. That ceiling is
built into the objective algebraically, not a matter of tuning `--slope_weight`.

But the ceiling is high enough to be worth having: it recovers **64.8% of the total slope
deficit**. The objective's real defects are elsewhere -- it is under-determined (blind to `a_p`
except through `r`), and its L1 form has a constant, non-vanishing gradient.

All numbers below are CPU-only, computed on the 10 existing eval CSVs in
`/home/nissimb/DeepPEF/eval_results/` (28,314 rows each; 28 test proteins; 771-1334 variants per
protein). No training was run. `scripts/gate_g4_cpu.py` re-run after the work: `dG=-0.0030
width=1092`, ALL PASS.

---

## 1. The algebra: std-matching pins the slope to the correlation

Let `t` be true ddG and `q` predicted ddG within one protein. The least-squares slope of `q` on
`t` is

```
a_p = cov(q,t) / var(t)
```

Multiply and divide by `sd(q)*sd(t)`:

```
a_p = [ cov(q,t) / (sd(q) sd(t)) ] * [ sd(q) / sd(t) ]
    =            r_p              *      ratio_p
```

**`a_p = r_p * sd(q)/sd(t)`.** Two independent factors: a *shape* factor `r_p` (how well the
model orders and proportions the mutations) and a *scale* factor `ratio_p` (how wide its
predictions are).

The current loss term is (`Megascale-fineTuning/train.py:736`):

```python
slope_loss = torch.abs(slope_pred_ddg.std(unbiased=False) - slope_true_ddg.std(unbiased=False))
```

Its unique zero is `sd(q) = sd(t)`, i.e. `ratio_p = 1`. Substituting into the identity:

> **At the optimum of the std objective, `a_p = r_p` exactly.**

The objective cannot see `a_p` at all. It sees only one of the two factors, and it drives that
one to 1. Whatever `r_p` is, that is where the slope lands.

### 1.1 The identity verified numerically

Computed per protein on every one of the 10 checkpoints (280 protein-checkpoint cells):

| checkpoint | n | max abs(a_p - r*ratio) |
|---|---|---|
| abl_anchor_w0.3_s42_e14 | 28 | 3.3e-16 |
| abl_anchor_w1.0_s42_e14 | 28 | 6.7e-16 |
| abl_anchor_w3.0_s42_e13 | 28 | 1.1e-16 |
| abl_calib_ctrl_repro2_e14 | 28 | 4.4e-16 |
| abl_p3_slope3.0_s42_e8 | 28 | 2.2e-16 |
| abl_sigma_seed1_e13 | 28 | 3.3e-16 |
| abl_sigma_seed2_e10 | 28 | 2.2e-16 |
| abl_sigma_seed3_e13 | 28 | 2.2e-16 |
| abl_sigma_seed42_e9 | 28 | 2.2e-16 |
| abl_sigma_seed4_e14 | 28 | 2.2e-16 |

Exact to floating point. The decomposition is not an approximation.

### 1.2 The objective is blind to a_p: constructive demonstration

Synthetic `t ~ N(0,1)`, n=2000. For each target slope, build `q = a*t + sqrt(1-a^2)*e` and
rescale to `sd(q) = sd(t)`:

| target a_p | achieved a_p | sd(q)/sd(t) | r |
|---|---|---|---|
| 0.3 | 0.2763 | **1.000000** | 0.2763 |
| 0.5 | 0.4969 | **1.000000** | 0.4969 |
| 0.8 | 0.7981 | **1.000000** | 0.7981 |
| 1.0 | 1.0000 | **1.000000** | 1.0000 |
| 1.5 | *impossible* | -- | would need r > 1 |

The std objective is **perfectly satisfied (loss exactly 0) for every slope in (0,1]**. The
premise in the task brief is confirmed: a model can match the spread perfectly with a badly wrong
slope. The only thing that then determines `a_p` is `r`.

Note the last row: with `sd(q)=sd(t)` enforced, `a_p > 1` is *unreachable* -- it would require
`r > 1`. So the std objective not only has a ceiling at `r`, it makes overshoot impossible. It is
a one-sided constraint disguised as a two-sided one.

---

## 2. THE DECOMPOSITION (the deliverable)

Per protein, averaged over the **6 slope-free replicates** (`abl_sigma_seed{1,2,3,4,42}` and
`abl_calib_ctrl_repro2`). Sorted by `a_p`. `pct_from_r` is `r`'s share of the log-deficit,
`-log(r) / -log(a_p)`.

| protein | n | a_p | r | ratio | sd_true | pct_from_r |
|---|---|---|---|---|---|---|
| HEEH_KT_rd6_0793 | 796 | 0.0873 | 0.3747 | 0.2333 | 0.8056 | 40.3% |
| 2KVS | 1207 | 0.0933 | 0.2308 | 0.4501 | 1.1478 | 61.8% |
| 1QKH | 1272 | 0.2442 | 0.5317 | 0.4597 | 0.8077 | 44.8% |
| HHH_rd1_0142 | 804 | 0.2655 | 0.7713 | 0.3438 | 1.0713 | 19.6% |
| HHH_rd1_0244 | 771 | 0.2806 | 0.8372 | 0.3353 | 1.0143 | 14.0% |
| 3DKM | 1239 | 0.2919 | 0.6411 | 0.4554 | 0.9969 | 36.1% |
| 2KXD | 1173 | 0.3296 | 0.8111 | 0.4064 | 1.3631 | 18.9% |
| 6EWT | 1025 | 0.3451 | 0.8043 | 0.4289 | 1.3154 | 20.5% |
| 1QP2 | 1083 | 0.3555 | 0.5721 | 0.6226 | 1.1247 | 54.0% |
| r18_3_TrROS_Hall | 990 | 0.3606 | 0.8039 | 0.4489 | 0.6641 | 21.4% |
| 4C26 | 1160 | 0.3607 | 0.7851 | 0.4595 | 1.2810 | 23.7% |
| 6EWU | 1010 | 0.3618 | 0.8173 | 0.4423 | 1.2126 | 19.8% |
| 6EWS | 1036 | 0.3810 | 0.8107 | 0.4698 | 1.2305 | 21.8% |
| HEEH_KT_rd6_0746 | 807 | 0.3978 | 0.7337 | 0.5414 | 0.4767 | 33.6% |
| 2K5H | 1125 | 0.4259 | 0.8569 | 0.4970 | 1.5847 | 18.1% |
| r11_1081_TrROS_Hall | 898 | 0.4450 | 0.7863 | 0.5659 | 0.7702 | 29.7% |
| 2BTH | 801 | 0.4526 | 0.8496 | 0.5325 | 0.7354 | 20.6% |
| 2LQK | 990 | 0.4683 | 0.6250 | 0.7490 | 0.6718 | 62.0% |
| 1TUC | 1044 | 0.4772 | 0.7741 | 0.6167 | 1.0848 | 34.6% |
| 1GYZ | 1126 | 0.4830 | 0.7020 | 0.6886 | 0.7280 | 48.6% |
| r12_757_TrROS_Hall | 982 | 0.5056 | 0.7339 | 0.6884 | 0.6369 | 45.4% |
| 2KWH | 980 | 0.5285 | 0.8013 | 0.6581 | 0.7215 | 34.7% |
| 1PSE | 1218 | 0.5624 | 0.6973 | 0.8081 | 0.6738 | 62.6% |
| 2L33 | 1334 | 0.5968 | 0.7996 | 0.7471 | 0.8141 | 43.3% |
| 2K28 | 920 | 0.6401 | 0.8379 | 0.7639 | 0.9592 | 39.6% |
| 2K1B | 934 | 0.7737 | 0.8116 | 0.9538 | 0.8477 | 81.4% |
| 2WXC | 787 | 0.9239 | 0.8533 | 1.0823 | 0.8130 | (a_p ~ 1, share ill-defined) |
| 1W4H | 802 | 0.9904 | 0.8390 | 1.1799 | 0.7976 | (a_p ~ 1, share ill-defined) |

**Medians: `a_p = 0.4119`, `r = 0.7930`, `ratio = 0.5370`.** (`0.7930 x 0.5370 = 0.4258`; medians
do not multiply exactly, the per-protein products do.)

### 2.1 How much of a_p is r, and how much is fixable compression?

Because the identity is multiplicative, the honest split is additive in logs:
`-log a_p = -log r + -log ratio`. Averaged over the 28 proteins:

```
-log a_p  =  0.9316
          =  -log r      0.3429   ->  36.8%   IRREDUCIBLE by any spread objective
          +  -log ratio  0.5848   ->  62.8%   spread compression, addressable
```

Per checkpoint (mean of the log-shares over proteins), `r`'s share is:

| checkpoint | -log a | -log r | -log ratio | % from r |
|---|---|---|---|---|
| abl_anchor_w0.3_s42_e14 | 0.6050 | 0.3268 | 0.2783 | 54.0% |
| abl_anchor_w1.0_s42_e14 | 0.6072 | 0.3605 | 0.2466 | 59.4% |
| abl_anchor_w3.0_s42_e13 | 2.1223 | 0.4680 | 1.6543 | 22.0% |
| abl_calib_ctrl_repro2_e14 | 0.7940 | 0.3571 | 0.4369 | 45.0% |
| abl_p3_slope3.0_s42_e8 | 1.2077 | 0.3635 | 0.8443 | 30.1% |
| abl_sigma_seed1_e13 | 0.9034 | 0.3712 | 0.5322 | 41.1% |
| abl_sigma_seed2_e10 | 0.8976 | 0.3158 | 0.5818 | 35.2% |
| abl_sigma_seed3_e13 | 1.1092 | 0.3483 | 0.7609 | 31.4% |
| abl_sigma_seed42_e9 | 0.8286 | 0.3454 | 0.4832 | 41.7% |
| abl_sigma_seed4_e14 | 1.1398 | 0.3468 | 0.7930 | 30.4% |
| **mean** | | | | **39.0%** |

**In slope units:**

```
a_p now                              0.4119
a_p at the std objective's optimum   0.7930   (= r)
a_p target                           1.0

recoverable by a spread objective    0.3811  =  64.8% of the gap to 1.0
structurally unreachable             0.2070  =  35.2% of the gap to 1.0
```

**So `--slope_weight` CAN fix most of it -- roughly two thirds -- and provably cannot fix the last
third.** The task brief's hypothesis ("if a_p is mostly r, --slope_weight cannot fix it") is
answered: `a_p` is *not* mostly `r`. It is ~63% compression and ~37% ranking error. The objective
is aimed at the larger component.

### 2.2 Which factor actually drives a_p across proteins

n = 28; the p=0.05 two-sided threshold is |r| = 0.374.

| correlation | value | p | verdict |
|---|---|---|---|
| corr(a_p, ratio) | **+0.9436** | 3e-14 | a_p is essentially the ratio |
| corr(a_p, r) | +0.5773 | 0.0013 | significant, secondary |
| corr(r, ratio) | +0.3007 | 0.120 | **not distinguishable from zero** |

The two factors behave as **independent channels** at this n. Compression is not a by-product of
poor ranking, and improving one is not expected to move the other. This matters: it means a
spread objective can in principle collect its 64.8% without trading away `r`.

Two proteins (2WXC, 1W4H) already have `ratio > 1` and `a_p ~ 0.92-0.99`. A global spread
objective would push these *past* 1. Any deployed slope term should be per-protein, not global.

### 2.3 Seed stability: the ceiling is hard, the compression is soft

ICC(1) and per-protein SD over the 6 replicates:

| quantity | median 6-seed SD | between-protein SD | ICC(1) |
|---|---|---|---|
| r | **0.0098** | 0.1490 | **0.960** |
| a_p | 0.0730 | 0.2070 | 0.839 |
| ratio | 0.1003 | 0.2220 | 0.780 |

`r` is ten times more seed-stable than `ratio`. The ceiling is a near-deterministic property of
the model-plus-data; the compression is the volatile, trainable part. This is consistent with the
ceiling being real rather than an artefact of any particular run.

---

## 3. What the one existing slope_weight run actually did

`abl_p3_slope3.0_s42_e8` was trained with `--slope_weight 3.0`; `abl_sigma_seed42_e9` is its
control. Naive paired comparison (28 proteins, Wilcoxon) looks like a catastrophe:

| | control | slope3.0 | delta (median) | Wilcoxon p |
|---|---|---|---|---|
| a_p | 0.4366 | 0.3253 | -0.1324 | <1e-4 |
| ratio | 0.5573 | 0.4489 | -0.1784 | <1e-4 |
| sd_pred | 0.5998 | 0.4215 | -0.1849 | <1e-4 |
| r | 0.7921 | 0.7943 | -0.0120 | 0.011 |

i.e. the decompression lever apparently *compressed* by 31%. **This is not a real effect.**
Against the correct null -- the 6 slope-free replicates, which differ only in seed and epoch --
the slope3.0 run sits inside the band:

| quantity | 6 replicate medians | mean +- sd | slope3.0 | z | rank | two-sided p |
|---|---|---|---|---|---|---|
| a_p | 0.4068, 0.4301, 0.3387, 0.4366, 0.3271, 0.4975 | 0.4061 +- 0.0643 | 0.3253 | -1.26 | 0/6 | >= 0.29 |
| ratio | 0.5550, 0.5852, 0.4580, 0.5573, 0.4220, 0.6342 | 0.5353 +- 0.0800 | 0.4489 | -1.08 | 1/6 | >= 0.57 |
| r | 0.7877, 0.7940, 0.7911, 0.7921, 0.7930, 0.7955 | 0.7922 +- 0.0027 | 0.7943 | +0.77 | 5/6 | >= 0.57 |

The Wilcoxon p<1e-4 is a **pseudo-replication artefact**: it treats 28 proteins as independent
replicates of a *training-run-level* effect, of which there is exactly n=1. The seed-to-seed SD of
the ratio is 0.0800 -- comparable to the entire claimed effect once you account for it. Epoch is
confounded too (e8 vs e9-e14), though `corr(epoch, ratio)` within replicates is r=-0.295, p=0.57,
n=6, i.e. undetermined.

**Honest verdict: `--slope_weight 3.0` produced no detectable change in the spread ratio (|z| <
1.1 against a 6-run null).** The lever fired at weight 3.0 and moved nothing measurable. That is
the finding, and it needs explaining -- section 4 explains it.

Note also `abl_anchor_w3.0_s42_e13`: ratio 0.2272, a_p 0.1535, r 0.6669. The WT-anchor lever at
weight 3.0 *does* massively compress, and it damages `r` (0.667 vs the 0.792 +- 0.0027 replicate
band -- far outside). That is a real, large effect, and a warning that at weight 3.0 an auxiliary
term can take over the optimisation. Anchor at 0.3/1.0 instead *raises* the ratio to 0.81/0.85 --
the two highest in the whole set. **The strongest observed spread lever in this collection is the
WT anchor at low weight, not slope_weight at all.** (n=1 run per anchor weight; suggestive, not
established.)

---

## 4. Why the L1 std term is weak: gradient analysis

At the operating point (`sd_pred=0.5998`, `sd_true=0.8309`, `r=0.7921`, `a_p=0.4366`):

**(a) The sign is right.** At protein level, `sd_pred < sd_true` for 25/28 proteins (median gap
-0.3496). Sampling minibatches of size B from the eval predictions, `P(sd_pred > sd_true)` is
0.217 (B=4), 0.178 (B=8), 0.157 (B=16), 0.144 (B=32). So in 78-86% of minibatches the term pushes
spread **up**, correctly. It is not sign-confused, and the failure in section 3 is not a sign error.

**(b) The magnitude is fixed and small.** For `L = |sd(q) - sd(t)|`,

```
dL/dq_i = sign(sd_q - sd_t) * (q_i - qbar) / (n * sd_q)
||grad L||_2 = 1/sqrt(n)      exactly, for all mismatch sizes
```

An absolute-value penalty has **constant gradient norm**. It gives the identical push whether the
mismatch is 0.001 or 1.0, and it does not vanish at its own optimum -- it is non-differentiable
there and chatters. Meanwhile the L1 data loss `mean|q-t|` also has `||grad|| = 1/sqrt(n)`. So at
`--slope_weight w` the term is simply `w` times the data gradient in norm, with no adaptation to
how wrong the spread actually is. It cannot *converge* to `sd_q = sd_t`; it can only oscillate
around it, and the data loss -- which prefers shrinkage, see (c) -- wins on average.

**(c) The data loss actively wants compression.** Under any convex regression loss with imperfect
prediction, shrinkage reduces expected error. The MSE-optimal post-hoc rescale of our predictions
is `k* = r * sd_t / sd_p`, median **1.2028**, which would set `a_p -> a_p * k* = r^2 = 0.6274`.
That is *lower* than the std objective's target of `r = 0.7930`. So the data term's preferred
slope (`r^2`) and the std term's preferred slope (`r`) genuinely disagree, and every unit of
decompression is paid for in training loss. Centred per-protein MSE:

| regime | median centred MSE |
|---|---|
| now (a_p = 0.437) | 0.4296 |
| std-matched (a_p = r) | **0.3365** |
| slope-forced (a_p = 1) | 0.6236 |

Interesting: std-matching is *cheaper* in MSE than the status quo (0.3365 vs 0.4296) -- our models
are over-shrunk even by MSE's own standard, so the first part of the decompression is free.
Forcing `a_p = 1` costs 1.85x the std-matched MSE. That is the real price of the last 35% of the
slope gap, and it is a price paid in accuracy for a gain in calibration.

**(d) The term is under-determined.** `sd_q^2 = a_p^2 sd_t^2 + sd_resid^2`. The objective
constrains only the sum. Measured split at the control checkpoint: median signal component
`a_p*sd_t = 0.4342`, median noise component `= 0.3742`, noise share of predictive variance
**0.373**. The model can satisfy `sd_q = sd_t` by inflating the *noise* component alone, leaving
`a_p` untouched and degrading `r`. Nothing in the loss prevents this. (In the slope3.0 run the
noise share was 0.369 vs the control's 0.373 -- unchanged, consistent with "the term did nothing"
rather than "the term inflated noise". The loophole is available but was not exercised.)

---

## 5. Alternative objectives

Write centred within-protein vectors `t' = t - tbar`, `q' = q - qbar`, minibatch size n.

### A -- current: `L_A = |sd(q) - sd(t)|`

- Fixed point `sd_q = sd_t` => **`a_p = r` = 0.793**. Ceiling.
- `||grad||_2 = 1/sqrt(n)`, constant, non-vanishing, non-smooth at the optimum.
- Blind to `a_p`; satisfiable by noise inflation (sections 1.2, 4d).
- One-sided in effect: `a_p > 1` unreachable.

### B -- variance-ratio: `L_B = (sd(q)/sd(t) - 1)^2`

- **Same fixed point, same ceiling `a_p = r`.** No improvement on the central defect.
- But smooth, and the gradient *vanishes* at the optimum, so it converges instead of chattering,
  and it is scale-free: it treats a protein with `sd_t = 0.48` (HEEH_KT_rd6_0746) the same as one
  with `sd_t = 1.58` (2K5H). `L_A` weights each protein in proportion to its absolute mismatch,
  and our `sd_t` spans 0.4767-1.5847, a 3.3x range, so narrow proteins are systematically
  under-served. Since the lowest-`a_p` proteins are not the widest ones, this rescaling is a
  genuine if second-order improvement.
- **Strictly dominates A. Zero-cost upgrade, but does not raise the ceiling.**

### C -- direct slope penalty: `L_C = (1 - a_hat)^2`, `a_hat = <q',t'>/<t',t'>`

- `a_hat` is differentiable in `q` and cheap: a dot product over the minibatch, with the
  denominator `<t',t'>` a *constant* w.r.t. the model. So
  ```
  dL_C/dq_i = -2 (1 - a_hat) * t'_i / (n * var(t))
  ||grad L_C||_2 = 2|1 - a_hat| / (sqrt(n) * sd_t)
  ```
  At our operating point that is 1.356x the `L_A` gradient (identical ratio at n=8, 16, 32 -- both
  scale as `1/sqrt(n)`). So it is *not* a weaker signal; it is a comparably sized one that
  additionally vanishes at the target instead of chattering.
- **Fixed point `a_p = 1`. No ceiling.** This is the decisive property.
- It is aimed at the right object: it projects the error onto `t'`, so it rewards only the
  component of extra spread that is *aligned with the truth*. Noise inflation is orthogonal to
  `t'` in expectation and earns nothing. This directly closes hole (4d), which A and B both leave
  open.
- Cost: reaching `a_p = 1` requires `sd_q = sd_t / r`, i.e. **over-spreading by 1/r = 1.2624,
  26.2% wider than the truth**. So C and A/B are in *direct conflict*: A's optimum is a point C
  must leave by 26%. They must not be summed into one loss. And C accepts the 1.85x centred-MSE
  increase from section 4c.
- Risk: `a_hat` is a ratio estimate and is noisy on small minibatches. With per-protein
  minibatches of 771-1334 variants this is not a concern here, but B=4 would be.

### D -- slope + correlation: `L_D = (1 - a_hat)^2 + lam*(1 - r_hat)`

- Same fixed point as C for `a_p`, plus explicit pressure on the *other* factor. Since
  `corr(r, ratio) = +0.30, p = 0.12, n = 28` -- no evidence the channels are coupled -- there is
  no a priori reason C alone would raise `r`, and `r` is 35.2% of the gap.
- But `r` is the seed-stable, ICC 0.960 quantity: its median moved by less than 0.003 across 6
  seeds. It looks like a capacity/representation limit, not something a loss term will shift.
  **I do not expect the `lam` term to earn its complexity**, and it adds a hyperparameter.
  Recommend only as a follow-up if C succeeds and the residual `r` gap still matters.

### Recommendation

**C, at low weight, replacing A.** Ranked:

| | fixed point | max a_p | grad at optimum | blind to a_p? | verdict |
|---|---|---|---|---|---|
| A `abs(sd_q - sd_t)` | sd_q = sd_t | **0.793** | 1/sqrt(n), non-zero | **yes** | current; ceilinged |
| B `(sd_q/sd_t - 1)^2` | sd_q = sd_t | **0.793** | -> 0 | **yes** | free upgrade on A, same ceiling |
| C `(1 - a_hat)^2` | a_p = 1 | **1.0** | -> 0 | no | **recommended** |
| D C + `lam(1 - r)` | a_p = 1 | 1.0 | -> 0 | no | speculative, extra hyperparameter |

Two operational cautions from the data:

1. **Use a low weight.** `abl_anchor_w3.0_s42_e13` shows what weight 3.0 does to an auxiliary
   term: it took over, drove `r` from 0.792 to 0.667 (far outside the 0.0027-SD replicate band)
   and crushed the ratio to 0.227. Sweep 0.1-1.0, not 3.0.
2. **Make it per-protein, and consider clipping at `a_hat >= 1`.** Two of 28 proteins (2WXC
   a_p=0.924, 1W4H a_p=0.990) are already at slope 1. A symmetric `(1 - a_hat)^2` will push them
   past it. `max(0, 1 - a_hat)^2` decompresses only what is compressed.

---

## 6. Answer to the question asked

**No, `|std(pred) - std(true)|` is not the right objective -- but it is closer to right than the
framing suggested.**

- The task brief's worry ("a model could match the spread perfectly with a badly wrong slope") is
  **exactly correct and demonstrated** (section 1.2): loss 0 is attainable at every `a_p` in (0,1].
- The consequence is a **hard ceiling at `a_p = r = 0.793`**, verified across 10 checkpoints and
  seed-stable to SD 0.0027. `--slope_weight` cannot reach `a_p = 1` no matter how it is tuned.
- **But the brief's follow-on ("if a_p is mostly r, --slope_weight cannot fix it") does not
  apply.** `a_p = 0.412` decomposes as 36.8% ranking error and **62.8% spread compression**. The
  ceiling still delivers 64.8% of the slope gap. The objective is pointed at the right, larger
  component. This is the deliverable, and it is a partially positive result for the existing lever.
- The reason the one `slope_weight=3.0` run showed nothing is **not** the ceiling -- it stopped
  far short of it. It is the L1 form: constant `1/sqrt(n)` gradient, non-vanishing and chattering
  at its own optimum, opposed by a data loss whose own preferred slope is `r^2 = 0.627`, *below*
  the std target.
- Replacing it with `(1 - a_hat)^2` removes the ceiling, removes the noise-inflation loophole,
  gives a 1.36x larger gradient that vanishes at the target, and costs 1.85x centred MSE at
  `a_p = 1`. If only a partial fix is wanted at no MSE cost, note that std-matching is
  *cheaper* in MSE than the status quo (0.3365 vs 0.4296) -- the models are over-shrunk even by
  MSE's own criterion, so the first part of the decompression is free.

---

## 7. Limits of these claims

- **n = 28 proteins.** The p=0.05 two-sided threshold is |r| = 0.374. `corr(r, ratio) = +0.30` is
  *not* distinguishable from zero; the independence of the two channels is a failure to reject,
  not a demonstration of independence.
- **n = 6 training runs** for the replicate null, and **n = 1** slope_weight run. "slope_weight
  did nothing" means "no effect detectable at |z| < 1.1 with n=6"; an effect of ~0.08 in the ratio
  could hide there. It is not proof the term is inert.
- Epoch is confounded with condition across the 10 checkpoints (e8-e14) and cannot be separated
  at n=6 (`corr(epoch, ratio)` r=-0.295, p=0.57).
- The ceiling result (section 1) is **algebra, not statistics** -- it holds identically and is not
  subject to any of the above. Only the *sizes* (0.793, 64.8%, 62.8%) are estimates on this
  benchmark.
- All 28 test proteins are single-chain, ligand-free, metal-free monomers of 43-72 aa. Whether
  `r = 0.79` -- and hence the ceiling's height -- transfers to larger or multi-chain targets is a
  **generalisation claim this benchmark cannot test**.
- Sections 4c and 5C predict the *consequences* of alternative objectives from the algebra and
  from the current predictions. They are predictions. No alternative objective has been trained.

## Artifacts

- `/home/nissimb/DeepPEF/results/SLOPE_OBJECTIVE.md` (this file)
- `/home/nissimb/DeepPEF/results/_slope_decomposition.csv` -- per-protein a_p, r, ratio, sd_true
  (6-replicate means), 28 rows
- `/home/nissimb/DeepPEF/results/_slope_objective.json` -- per-checkpoint summary + all 280
  protein-checkpoint cells
- `/home/nissimb/DeepPEF/scratch_slopeobj/slope_obj{,2,3,4}.py` -- the analysis scripts

Gate after this work: `scripts/gate_g4_cpu.py` -> `dG=-0.0030 width=1092`, ALL PASS.
