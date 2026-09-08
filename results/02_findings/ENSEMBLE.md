# LINE 5 — Free Accuracy From Checkpoints We Already Have (Ensembling)

**Scope:** CPU only, no training, no GPU. Inputs are the 10 existing eval CSVs in
`/home/nissimb/DeepPEF/eval_results/` and the 382 weight checkpoints on disk.
n = 28 proteins, 28,304 mutations. At n=28, |r| < 0.374 is indistinguishable from zero at p=0.05.

**Headline: ensembling 5 seeds buys +0.0095 pooled ddG PCC (0.6050 -> 0.6145). It is real
(cluster-bootstrap P(gain>0) = 1.000, 27/28 proteins improve, Wilcoxon p=2.8e-6) and it is
negligible. It does nothing whatsoever to the calibration problem, and that is provable
algebraically rather than merely observed.**

---

## 0. A methodological trap that had to be cleared first

The 10 CSVs each have 28,314 rows in the SAME multiset but in DIFFERENT row order — the eval
loader shuffles. A naive `mean(pred)` across files runs, completes, and prints a number.
That number is garbage.

    abl_sigma_seed1_e13 vs abl_anchor_w0.3: protein column aligned = False
    label dG max row-wise difference = 5.95 kcal/mol   (i.e. row i is a different mutation)

Rows were joined on the key `(protein, deltaG-label)` after verifying:
- every file has an **identical key multiset** (sorted label vectors match bitwise, maxdiff 0.00e+00);
- after the join, ddG labels agree to **0.000e+00** across all 10 files;
- zero NaN predictions.

5 keys collide with multiplicity 2; those 10 rows (0.035%) are dropped. **28,304 rows survive.**
Every number below is on that verified join. This is the signature failure mode from CONTEXT —
it would have produced a plausible-looking wrong answer silently.

---

## 1. Ensemble across seeds (the 5 `abl_sigma` seeds)

Individual members and the equal-weight prediction average:

| run | pooled ddG PCC | per-prot PCC (med) | a_p (med) | std(b_p) dG | dG MAE |
|---|---|---|---|---|---|
| abl_sigma_seed1_e13  | 0.5947 | 0.7877 | 0.4076 | 1.1086 | 1.0812 |
| abl_sigma_seed2_e10  | **0.6574** | 0.7940 | 0.4301 | 1.0233 | 1.0235 |
| abl_sigma_seed3_e13  | 0.5797 | 0.7911 | 0.3395 | 1.0235 | 1.0119 |
| abl_sigma_seed42_e9  | 0.5927 | 0.7921 | 0.4366 | 1.3320 | 1.4781 |
| abl_sigma_seed4_e14  | 0.6007 | 0.7930 | 0.3271 | 1.0035 | 0.9887 |
| **member mean** | 0.6050 | 0.7916 | 0.3882 | 1.0982 | 1.1167 |
| **SIGMA-5 ENSEMBLE** | **0.6145** | **0.7982** | 0.3939 | **1.0722** | **1.0729** |
| delta vs member mean | **+0.0095** | +0.0066 | +0.0057 | -0.0259 | -0.0438 |

Ensemble size sweep (averaged over all subsets of each size) — saturating by k=3:

    k=1  pooled=0.6050   k=2  0.6110   k=3  0.6130   k=4  0.6139   k=5  0.6145

Significance of the gain, **cluster bootstrap resampling the 28 proteins** (2000 reps, the correct
unit — mutations within a protein are not independent):

    gain = +0.0085, 95% CI [+0.0044, +0.0132], P(gain > 0) = 1.000
    ensemble pooled PCC 95% CI [0.4809, 0.7286]   <-- note how wide the LEVEL is

Per-protein: the ensemble beats the member mean on **27/28 proteins**, mean delta +0.0057,
Wilcoxon signed-rank **p = 2.76e-06**.

**Read this correctly.** The *gain* is highly significant and *tiny*. The CI on the pooled PCC
*level* ([0.48, 0.73]) is two orders of magnitude wider than the gain. Ensembling reliably moves
the third decimal place of a quantity whose second decimal place is not pinned down by n=28.

---

## 2. The tension the brief asked about — and why it dissolves

The brief predicted: *"averaging predictions that disagree SHRINKS the spread, so an ensemble may
make the compression WORSE even as PCC improves."*

**That prediction is provably wrong, and this is the most interesting result here.**

Measured: mean a_p = 0.4287 for the ensemble and 0.4287 for the mean-of-members. Not close —
**identical to 6.66e-16**, i.e. floating-point exact. Same for b_p (4.44e-16).

The reason is algebra, not luck. Both calibration parameters are **linear functionals of the
prediction vector**:

- `a_p` = OLS slope of pred_ddG on the *fixed, label-side* true ddG = `cov(y, pred)/var(y)`.
  The numerator is linear in `pred`; the denominator depends only on labels.
- `b_p` = `mean(pred_dG - true_dG)` over the protein. Manifestly linear in `pred`.

Averaging is linear, so slope-of-mean = mean-of-slopes **exactly**:

    a_p(mean of members) == mean of a_p(member)      max deviation 6.66e-16
    b_p(mean of members) == mean of b_p(member)      max deviation 4.44e-16

**Ensembling cannot shrink a_p, and it cannot inflate it.** The intuition that averaging
disagreeing predictors shrinks the signal applies when the regression is of the *predictor on
itself* (and indeed the std of the ensemble's own predictions IS smaller than the members'). But
a_p regresses the prediction on the *fixed true label*, and that operation commutes with
averaging. The compression is a property of the mean member, and the mean member is exactly what
an equal-weight ensemble gives you.

Measured consequence: median a_p 0.3939 for the ensemble, vs member medians
[0.4076, 0.4301, 0.3395, 0.4366, 0.3271] — squarely inside the member range, not below it.
(The median moves a hair because the median is not a linear functional; the mean, which is, does
not move at all.) Spread across proteins is likewise essentially unchanged: std(a_p) = 0.1967
for the ensemble vs 0.1996 mean over members.

**Conclusion: the compression a_p ~ 0.5 is completely immune to ensembling. Ensembling is not a
calibration lever, and it never could have been.**

---

## 3. Does ensembling reduce std(b_p)? — prediction stated first

**Prediction made before measuring.** ICC(b_p) = 0.898 means ~90% of b_p variance is
protein-driven and shared across seeds; only ~10% is independent seed noise. Averaging m=5 seeds
cancels the independent part by 1/m, so

    std(b_p) after averaging ~= std(b_p) * sqrt(ICC + (1-ICC)/5) = sqrt(0.898 + 0.0204) = 0.958x

i.e. a ~4% reduction, essentially nothing. **Predicted: b_p barely moves.**

**Measured:** std(b_p) 1.0982 (member mean) -> **1.0722** (ensemble) = **2.36% reduction**.

**The prediction was right in direction and magnitude** (predicted ~4%, observed 2.4%). With the
ICC re-estimated on these 5 seeds (ICC = 0.856) the formula predicts 1.0330 vs observed 1.0722 —
so the real reduction is even *smaller* than variance-components theory allows, meaning the
seed-to-seed b_p component is more correlated across seeds than a pure random-effects model
assumes.

Direct decomposition of the 5x28 b_p matrix:

    seed-noise SD within a protein  = 0.4051
    std(b_p) across proteins        = 1.0982 (member mean) -> 1.0722 (ensemble)
    ICC(b_p) on these 5 seeds       = 0.8560   (independently reproduces CONTEXT's 0.898)

**The offset is genuinely protein-driven and shared. Averaging seeds cancels only the 10-14%
that is seed noise. std(b_p) stays ~1.07 — the between-protein calibration problem is untouched.**

Note also (metric rule): subtracting a per-protein *constant* from dG predictions cancels exactly
in ddG, so a pure b_p corrector cannot change pooled ddG PCC at all, by construction. b_p is a
dG-space object and must be scored there.

---

## 4. Why the gain is so small — error decomposition

Decomposing the 5 members' squared error into a shared component and a seed-specific one:

    mean member MSE          = 1.07857
    ensemble (common) MSE    = 1.05998
    diversity (reducible)    = 0.01859  =  1.7% of member MSE

**98.3% of each member's error is shared bias that no amount of seed-averaging can remove.**
Pairwise correlation between member predictions is 0.9619 (range 0.925-0.985) — the seeds are
near-copies of each other. Ensembling removes variance, and variance is 1.7% of the problem.
The error is bias, and the bias is the miscalibration.

This is the quantitative statement of why Line 5 cannot touch the thesis' actual problem.

---

## 5. Ensemble across epochs — NOT POSSIBLE from disk

Checked directly. All 15 epochs (0-14) exist as **weight** checkpoints for every one of the ~25
runs (382 .pt files, 88 MB each). But:

    per-epoch prediction dumps found:     NONE
    *.npy / *.npz on disk:                NONE
    per-epoch CSVs outside eval_results:  NONE

Each of the 10 eval CSVs is a **single epoch** of a single run (the epoch is in the filename:
`_e13`, `_e9`, ...). **Prediction-space epoch ensembling cannot be done from what is on disk** —
it requires a forward pass per epoch, which is a compute job, not free. Reported as a negative,
per the brief's instruction to say so if the output does not exist.

Weight-space averaging (a "model soup") across epochs *within* a run is the legitimate
nearly-free alternative and is the one worth trying, but it too requires one forward pass over
28k mutations through a full ESM2 finetune to produce predictions. It is **out of scope for
"already on disk, no GPU"** and is listed as next action, not claimed as a result.

Weight averaging *across seeds* would be invalid — independently initialised runs are not
mode-connected, and averaging their weights generally destroys the model.

---

## 6. The honest ceiling with no further GPU

All combination rules tried on the 10 CSVs:

| method | pooled ddG PCC |
|---|---|
| single best member, **selected on test** | 0.6574 |
| best subset over all 1023 subsets, **selected on test** (= that same single member) | 0.6574 |
| all-10 z-scored average | 0.6206 |
| **all-10 ensemble** | **0.6197** |
| sigma-5 ensemble | 0.6145 |
| all-10 median | 0.6116 |
| mean single member | 0.5993 |
| all-10 rank-average | 0.5930 |
| sigma-5 rank-average | 0.5852 |

Two things must be said plainly:

**(a) The best "ensemble" is a single model.** Greedy forward selection on pooled PCC adds
`abl_sigma_seed2_e10` and then **stops at k=1** — no ensemble beats that one member. Exhaustive
search over all 1023 subsets agrees. But 0.6574 is selected *on the test set*, so it is an
oracle, not a method — the ensembling analogue of the b_p oracle, and it does not survive
held-out selection.

**(b) Held-out selection recovers almost none of it.** Selecting the member subset by
leave-one-protein-out (choose the best subset on 27 proteins, predict the 28th):

    LOPO-selected ensemble        = 0.6224
    all-10 fixed ensemble         = 0.6197
    mean single member            = 0.5993
    test-selected best member     = 0.6574  (ORACLE)

LOPO selection (0.6224) beats the fixed all-10 ensemble by only +0.0027 and recovers just **40%**
of the gap to the test-selected oracle. Same pattern as the b_p corrector: the oracle gain does
not survive held-out prediction.

**HONEST CEILING: pooled ddG PCC ~ 0.62 (all-10 ensemble 0.6197; LOPO-selected 0.6224), versus
0.5993 for a typical single model. Free gain ~ +0.02.** Anything above that on this data is
test-set selection.

For reference, per-protein affine oracles applied to the ensemble sit far above all of this
(ensemble + oracle per-protein ddG-mean removal = 0.8047; + full oracle affine = 0.8396),
confirming once more that the headroom is entirely in between-protein calibration, which
ensembling does not touch.

Spearman view (rank-based, less sensitive to the scale problem): mean member 0.5723 -> all-10
ensemble 0.5838. Same +0.01 story.

---

## 7. Verdict

- Seed ensembling gives a **statistically certain but practically negligible** +0.01 pooled PCC,
  and ~+0.02 for the 10-model heterogeneous ensemble. It is free and there is no reason not to
  bank it.
- It is **provably not a calibration method**: a_p and b_p are linear functionals of the
  predictions, so an equal-weight ensemble reproduces the mean member's a_p exactly (to 6.7e-16).
  The compression a_p ~ 0.5 is untouched, by algebra.
- std(b_p) falls only 2.4% (1.0982 -> 1.0722), as predicted in advance from ICC. The offset is
  shared across seeds, so averaging cannot cancel it.
- 98.3% of member error is shared bias. Ensembling attacks the 1.7% that is variance.
- Epoch ensembling in prediction space is **impossible from disk** — no per-epoch predictions were
  ever written.

**Line 5 is closed. It yields ~+0.02 PCC and zero calibration improvement. The remaining headroom
(raw 0.62 -> oracle 0.80-0.84) is entirely between-protein calibration, and this line does not
reach it.**

### Generalisation caveat

All of this is measured on 28 single-chain, ligand-free, 43-72aa monomers. The *algebraic* claims
(linearity of a_p and b_p, hence exact invariance of mean a_p under averaging) are benchmark
independent and hold for any dataset. The *empirical* claims (1.7% diversity, 0.96 inter-seed
correlation, +0.02 ceiling) are properties of these 5 seeds of one architecture on this
benchmark, and a more diverse pool — different architectures or feature blocks, rather than
seeds — would be expected to show more diversity and therefore a larger ensembling gain. That
larger gain would still not change a_p, by the same algebra.

---

## Reproduction

    /home/nissimb/DeepPEF/ens.py    # members, sigma-5 ensemble, size sweep, all-10, greedy
    /home/nissimb/DeepPEF/ens2.py   # linearity identity, error decomposition, LOPO, bootstrap
    /home/nissimb/DeepPEF/ens3.py   # ceiling table, rank/z/median rules, per-protein tests
    /home/nissimb/DeepPEF/results/ensemble_stats.json

Gate: `scripts/gate_g4_cpu.py` -> `baseline ... PASS dG=-0.0030 width=1092`, G4-CPU: ALL PASS.
No repo code was modified by this line of work; only new standalone analysis scripts were added.
