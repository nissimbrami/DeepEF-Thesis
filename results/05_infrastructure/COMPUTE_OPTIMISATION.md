# COMPUTE_OPTIMISATION.md

Optimising our own compute allocation. Every number below is measured from `sacct` and
from the 22 eval CSVs on the cluster, not estimated. Reproduce with the scripts in
`/home/nissimb/DeepPEF/analysis_co/`.

Gate re-run after all work: `scripts/gate_g4_cpu.py` -> **`PASS dG=-0.0030 width=1092`** (ALL PASS).

---

## HEADLINE

**The factorial cannot resolve A, B or C on pooled ddG PCC, and the D arm is already
decided at |z| = 29 from one seed. About 139 GPU-h of the queue is buying nothing.**

Three findings, in order of operational importance:

1. **Factor D (`--unfolded_emb`) is catastrophic, not subtle.** All 5 D=1 cells score
   pooled ddG PCC 0.10-0.31 (one at **-0.078**); all 5 D=0 cells score 0.583-0.618.
   **Complete separation, zero overlap**, a gap of 0.443 = **15 seed-sigmas**. The 16
   pending D=1 trainings (138 GPU-h) re-measure a result already at |z|=29 with k=1,
   which stays at |z|=10 even at the pessimistic upper-95% bound on seed noise.

2. **A, B and C are UNRESOLVABLE on pooled ddG PCC inside the working regime.** Within
   the D=0 cells the largest main effect is 0.0120, against an MDE80(k=3) of **0.0252**.
   Detecting it would need **k=12 seeds/cell = 1,708 GPU-h**. Three seeds is not
   "underpowered by a little" on this metric; it is off by 4x in k and 16x in cost.

3. **The same design is comfortably powered on `median a_p`** (effects 0.05-0.30 vs
   MDE80 0.043). The factorial is not worthless - **it is being scored on the wrong
   metric.** This is the most important operational finding available, and it is free.

---

## 1. MEASURED SPEND

`sacct -u nissimb`, window **2026-09-05T19:09 -> 2026-09-08T00:21 = 52.9 wall-hours**,
163 jobs. GPU-hours = elapsed x allocated GPUs.

| State | jobs | GPU-h |
|---|---:|---:|
| COMPLETED | 69 | 235.95 |
| RUNNING (accrued so far) | 12 | 50.73 |
| CANCELLED | 28 | 0.38 |
| FAILED | 6 | 0.04 |
| **TOTAL BURNED** | **163** | **287.10** |

By class:

| Class | jobs | GPU-h |
|---|---:|---:|
| Factorial trainings | 52 | 166.74 |
| Other trainings (anchor sweep, abl_sigma, slope sweep) | 26 | 84.14 |
| Golden-lane probes (`gld_*`) | 9 | 35.10 |
| misc / gates | 54 | 0.98 |
| All evals | 20 | 0.23 |

Mean elapsed of a completed training = **8.6 h** (26 runs, range 6:21-14:41), which
validates the ~8 h planning figure.

### Correction to two planning assumptions

- **Eval is 0.040 GPU-h, not 0.5.** Measured mean over 17 completed evals = **145 s**.
  The 48 pending eval jobs are 1.9 GPU-h of compute in total.
- **The golden lane is not held by a labmate - it is held by us.** Eight `gld_*` jobs
  (`gld_dg_coil`, `gld_slope`, `gld_w5_dg`, `gld_w7edge`, `gld_u10bidir`, `gld_w7span4`,
  `gld_loroW_onehot`, `gld_loroW_desc`) are RUNNING on `rtx6000` under this account, about
  5 h each. That lane is already working for us; the plan should say so.

### Remaining queue

96 jobs queued. Public lane cap confirmed `gres/gpu=5` (`sacctmgr`: `gpu-part gres/gpu=5`).

| Item | jobs | GPU-h |
|---|---:|---:|
| D=1 (uemb) trainings | 16 | 137.6 |
| D=0 (coil) trainings | 13 | 111.8 |
| Running trainings (remainder) | 5 | 17.2 |
| Side arms (`sa_coil`, `sa_uembmean`) | 2 | 17.2 |
| All 48 evals | 48 | 1.9 |
| **TOTAL REMAINING** | | **285.8** |

Full 48-run factorial = 413 GPU-h. Spent 167, remaining 286 -> it would land at **453 GPU-h**.

### What each remaining cell tells us

- **D=1 cells (16 trainings, 138 GPU-h): nothing.** D is decided. See section 2.
- **D=0 cells (13 trainings, 112 GPU-h): the only cells with information**, and only
  when scored on `a_p` / `std(b_p)`, not pooled PCC.
- **`sa_uembmean` (8.6 GPU-h): genuinely informative.** `--unfolded_emb mean` is the one
  D variant not yet falsified, and it discriminates "embedding carried global scale" from
  "embedding carried per-residue identity". Highest information per GPU-hour in the queue.
- **`sa_coil` (8.6 GPU-h): redundant.** Every D=0 cell is already a coil run.

---

## 2. THE HARD QUESTION: IS 3 SEEDS x 16 CELLS RIGHT?

### Measured seed variance

The 5 `abl_sigma` runs are 5 seeds of **one identical config**, and that config
(`--wt_anchor_weight 0 --designed_weight 1`, no slope, no D flag) is exactly factorial
cell **A=0 B=1 C=0 D=0**. So this is a true replicate estimate at a design point.

| metric | mean | **sd_seed** (df=4) | 95% CI on sd |
|---|---:|---:|---|
| pooled ddG PCC | 0.6052 | **0.0302** | [0.018, 0.087] |
| mean per-protein PCC | 0.7321 | **0.0069** | [0.004, 0.020] |
| median a_p | 0.3878 | **0.0515** | [0.031, 0.148] |
| std(b_p) | 1.0981 | **0.1369** | [0.082, 0.393] |
| dG PCC | 0.4194 | **0.0470** | [0.028, 0.135] |

This sd deliberately **includes** best-epoch-selection jitter, because that is how cells
are actually scored. Decomposed from the val curves (5 seeds x 15 epochs, free from logs):

- across-seed sd at a **fixed** epoch, plateau e8-e14 = **0.0285**
- within-seed sd across epochs e8-e14 (epoch jitter) = **0.0135**

So the noise is genuinely seed-driven (about 2x the epoch jitter), not a selection
artifact. The chosen epoch nonetheless wanders **e5-e14** across runs.

### Power

Main effect = mean(8 cells at +) - mean(8 cells at -), k seeds each;
SE = sd_seed * sqrt(2/(8k)); error df = 16(k-1).

| metric | k | SE | **MDE80** | runs | GPU-h |
|---|---:|---:|---:|---:|---:|
| pooled ddG PCC | 2 | 0.0107 | 0.0319 | 32 | 275 |
| pooled ddG PCC | **3** | 0.0087 | **0.0252** | 48 | 413 |
| mean per-prot PCC | 3 | 0.0020 | 0.0058 | 48 | 413 |
| median a_p | **3** | 0.0149 | **0.0430** | 48 | 413 |
| std(b_p) | 3 | 0.0395 | 0.1142 | 48 | 413 |

### Observed effects vs MDE - the answer

Fitting +/-1-coded OLS to the 10 cells that already have evals (design rank 5/5):

| metric | A | B | C | D |
|---|---:|---:|---:|---:|
| pooled | -0.1604 | -0.0125 | -0.0190 | **-0.5432** |
| mean per-prot PCC | -0.0300 | +0.0373 | -0.0015 | **-0.5498** |
| median a_p | +0.1368 | -0.0130 | +0.0554 | **-0.4417** |

But the observed subset is **not orthogonal**: corr(A,D) = **-0.6**. That imbalance is
why the marginal A effect on pooled (+0.169) flips sign to -0.160 once D is adjusted for.
The honest estimates come from the D=0 cells only:

| metric | A given D=0 | B given D=0 | C given D=0 | MDE80(k=3) | verdict |
|---|---:|---:|---:|---:|---|
| pooled | -0.0120 | -0.0040 | -0.0115 | 0.0252 | **all < MDE - unresolvable** |
| median a_p | **+0.2962** | +0.0453 | **+0.1153** | 0.0430 | **A and C resolvable** |
| std(b_p) | +0.4049 | +0.1445 | +0.2164 | 0.1142 | A resolvable |

**So the answer is: it depends entirely on the metric, and nobody chose one.**

- On **pooled ddG PCC**: 3 seeds is badly underpowered. Effects are about 0.012 against
  an MDE80 of 0.0252. Required k = **12 seeds/cell = 1,708 GPU-h**. Even k=4 (615 GPU-h)
  only reaches 0.020. This metric should be abandoned as the factorial response.
- On **median a_p**: 3 seeds is **more than sufficient** - effects of 0.12-0.30 against
  MDE80 0.043 means k=2 (MDE80 0.054) already resolves A and C. **Two seeds would do, and
  that saves about 138 GPU-h.**
- On **D, any metric**: k=1 was already sufficient.

Robustness - repeating with the pessimistic **upper 95% bound** on sd_seed (2.87x larger,
since df=4 is a weak estimate): D stays at |z|=10 with k=1; A/B/C on pooled stay
undetectable (MDE80 rises to 0.0724). **Neither conclusion is an artifact of the small
sigma sample.**

---

## 3. WOULD A HALF-FRACTION HAVE BEEN SUFFICIENT?

Yes - and the cost comparison is stronger than it first looks.

`2^(4-1)` resolution IV, generator `I = ABCD`, 8 cells. Alias structure:

```
A = BCD      AB = CD
B = ACD      AC = BD
C = ABD      AD = BC
D = ABC
```

All four main effects are clean, aliased only with 3-factor interactions (negligible here).

**What we would have lost:** the ability to separate the three 2-factor interaction
*pairs*. AB is indistinguishable from CD, AC from BD, AD from BC. We would see that
*some* interaction is present in a pair but not which member.

**Was that worth 8 extra cells?** For A, B, C - no; they barely have main effects.
For D - **yes, and it is the one real loss.** The measured A x D interaction is
**-0.147 on pooled**: the A effect is -0.004 when D=0 but -0.298 when D=1. A
half-fraction would have aliased AD with BC and left that ambiguous. In practice we
learned it anyway, because the D damage is visible cell-by-cell without any contrast.

**The key arithmetic the original plan missed:**

| design | cells | seeds | runs | GPU-h | SE(main effect, pooled) |
|---|---:|---:|---:|---:|---:|
| full 2^4 | 16 | 3 | 48 | 413 | 0.0087 |
| **half 2^(4-1)** | 8 | **6** | 48 | 413 | **0.0087** |
| half 2^(4-1) | 8 | 3 | 24 | 206 | 0.0123 |

A half-fraction at 2k seeds costs **exactly the same** as the full design at k seeds and
gives **identical** main-effect precision. The full factorial premium is entirely
2-factor-interaction separation. Given that A/B/C main effects on pooled are below MDE at
k=3, spending the budget on **8 cells x 6 seeds** would have been strictly better: same
cost, same main-effect SE, and it would have crossed the k=12 threshold far sooner on the
metric that mattered.

**Verdict: a half-fraction would have been sufficient, and 8 cells x 6 seeds would have
been better than 16 x 3 at identical cost.**

---

## 4. OPTIMAL REMAINING ALLOCATION

Ranked by information per GPU-hour. **Recommendations only - nothing was cancelled.**

| rank | action | GPU-h | information |
|---:|---|---:|---|
| 1 | **Re-score existing checkpoints at a fixed epoch, on CPU** | **0** | Removes the 0.0135 epoch jitter from every comparison. 513 checkpoints already on disk. Free. |
| 2 | **Move all 48 pending evals to `--partition=cpu`** | **-1.9** | `evaluate.py:63` already falls back to CPU. Takes them off the 5-GPU critical path. |
| 3 | **Keep `sa_uembmean`** | 8.6 | The only unfalsified D variant. Highest info/GPU-h of any queued training. |
| 4 | **Finish the 13 D=0 trainings** | 112 | The only cells with information - scored on `a_p`, not pooled PCC. |
| 5 | **Cancel the 16 D=1 trainings** | **-138** | D decided at abs(z)=29. Buys precision on a 15-sigma loser. |
| 6 | **Cancel `sa_coil`** | -8.6 | Redundant; every D=0 cell is a coil run. |

### Concrete recommendations

**CANCEL (recommend; about 147 GPU-h freed, 1.2 days of the public lane):**
- The 16 pending `DeepEF_p3_*_D1_uemb_seed{1,2}` trainings and their 24 dependent evals.
  D is settled. If a referee wants the D main effect with replication, the 5 existing D=1
  cells plus `sa_uembmean` already document it, and at k=1 it is |z|=29.
- `DeepEF_sa_coil_seed42` - duplicates the D=0 arm.

**REORDER:**
- Resubmit all 48 evals to `--partition=cpu --qos normal` (no GPU). They are 2.4-minute
  jobs currently each holding one of only 5 RTX-6000 slots ahead of 8.6 h trainings. This
  is the cheapest throughput win available and costs nothing.
- Promote the 13 D=0 trainings into the golden lane as `gld_*` slots free. With
  public(5) + golden(8) = 13 concurrent, the surviving 139 GPU-h finishes in about
  **11 wall-hours** instead of 28 on the public lane alone.

**ADD (cheap, high information):**
- **Fix the response metric before the remaining cells land.** Score the factorial on
  `median a_p` and `std(b_p)`. On pooled ddG PCC the design provably cannot resolve
  A/B/C at any affordable k; on `a_p` it already can. This is a decision, not compute.
- **Replace best-val-epoch selection with a fixed epoch (or the mean of e10-e14).**
  Selection currently wanders e5-e14 and injects 0.0135 sd for nothing. Averaging e10-e14
  cut the across-seed sd from 0.0336 to 0.0306 in the sigma runs. Free, and applies to
  checkpoints we already have.
- **Spend about 35 GPU-h of the freed budget on 4 more seeds of the single best D=0 cell**
  rather than on breadth. With the sigma k=5 plus 4 more, that cell reaches k=9 and the
  reference point for every future comparison gains a real confidence interval.

### Bottom line

Of 286 GPU-h queued, **about 147 is uninformative**. Redirecting it costs one `scancel`
the student should run, and the two changes that matter most - score on `a_p`, fix the
eval epoch - cost **zero GPU-hours** and are worth more than the 138 GPU-h cancelled.

---

## CAVEATS

- `sd_seed` comes from **n=5** replicates (df=4); its own 95% CI spans a 2.87x range.
  Every power claim above was re-checked at the upper bound and survives, but a larger
  sigma set would tighten this if it were the target.
- The 10-cell effect estimates are **unreplicated** (one seed each) and the observed
  subset is non-orthogonal (corr(A,D) = -0.6). The D=0-subset effects (n=5) are the
  honest ones; A/B/C estimates will move when the pending seeds land. **The D conclusion
  does not depend on this** - it is complete separation, not a contrast.
- `abl_sigma` replicates cell A0/B1/C0/D0 only. Seed variance is assumed homogeneous
  across cells; the D=1 cells are visibly more dispersed (pooled range 0.39 vs 0.035), so
  seed noise is probably *larger* in the D=1 arm - which strengthens, not weakens, the
  case for cancelling it.
- Mean training elapsed 8.6 h has real spread (6:21-14:41); wall-clock projections are
  correspondingly approximate.
- n=28 proteins throughout; |r| < 0.374 is indistinguishable from zero at p=0.05. No
  claim here rests on a correlation near that threshold.
