# Protein difficulty is INTRINSIC, not lever-dependent — and no lever can fix it

Done locally while the cluster was unreachable, on `nu_sweep.json`: 28 proteins × 15 conditions
(base, coil, coil_fixed_b, coil_ca_only, noemb, and 10 ν×b cells).

## The finding

**Rank correlation of per-protein error between conditions: mean 0.837, min 0.212.**

The same proteins are hard **no matter which lever is applied.** Difficulty is a property of the
protein, not of the model configuration.

| hardest (mean rank of 28) | mean \|err\| | length | true dG |
|---|---|---|---|
| 2KXD | 5.98 | 71 | 4.06 |
| HEEH_KT_rd6_0793 | 5.95 | 42 | 3.00 |
| 2K5H | 5.70 | 62 | 4.81 |
| 3DKM | 5.37 | 72 | 3.12 |

| easiest | mean \|err\| | length | true dG |
|---|---|---|---|
| r12_757_TrROS_Hall | 1.81 | 54 | 1.42 |
| 2KWH | 2.24 | 52 | 2.14 |
| 2K1B | 2.86 | 49 | 2.41 |

**The spread is 3.3× between easiest and hardest**, and it is stable across every condition tested.

## What drives it — two independent factors, equally

```
corr(difficulty, length)   = +0.502
corr(difficulty, true dG)  = +0.502     <- these looked identical; they are 0.5022 and 0.5016
corr(length, true dG)      = +0.204     <- only weakly related to each other
```

Partialling each out:

```
partial corr(difficulty, length  | true dG) = +0.472
partial corr(difficulty, true dG | length ) = +0.472
```

**Both survive, and at exactly the same strength.** That symmetry is not coincidence: the two
factors are nearly orthogonal (r = 0.204) and contribute independently. **Longer proteins are
harder, and more stable proteins are harder, for separate reasons.**

## Why the "more stable is harder" half matters

```
corr(mean signed error, true dG) = -0.546
proteins under-predicted: 28/28
```

Every protein is under-predicted, and **the under-prediction grows with the true dG.** The model
does not just have an offset — **it compresses the dG scale**, so proteins at the top of the range
are hit hardest. That is the same compression already measured within proteins as `a_p ≈ 0.5`,
now visible *between* proteins as well.

## Consequence for the thesis

1. **A per-protein offset correction cannot fix this.** W13 subtracts a constant per protein; it
   does nothing about a slope that is wrong in the same direction for all 28.
2. **This is why `b_p` resisted ten prediction attempts.** Difficulty is driven by length AND by
   true dG — but true dG is the LABEL. Any predictor using it is circular, and length alone gives
   only half the signal.
3. **The honest framing:** the model has a global scale-compression problem that shows up as
   per-protein offsets. Correcting the offsets treats the symptom.

## What would test it

Train with the dG target **standardised per protein** so the scale cannot compress, and check
whether difficulty still tracks true dG. If it does not, compression is confirmed as the mechanism.
That is a clean, cheap experiment and it has never been run.

**Confidence: 85%** — 15 conditions agree, and the partial correlations separate the two factors
cleanly. Held back from 95% because n = 28 and difficulty rank is a coarse statistic.

---

# CORRECTION — I overstated the compression claim, and the measurement says so

Above I wrote: *"the model has a global scale-compression problem that shows up as per-protein
offsets, so correcting the offsets treats the symptom."* **That was too strong.** Measured:

## The global affine fit

```
pred = 0.4053 * true - 3.1983        (a = 1 would be no compression)
```

The compression is real — the slope is 0.41, not 1.0. But decomposing the variance of the 28
per-protein WT errors:

| component | variance | share |
|---|---|---|
| total variance of b_p | 0.9944 | 100% |
| explained by ONE global compression | 0.3019 | **30.4%** |
| **genuinely per-protein residual** | 0.6925 | **69.6%** |

**Both effects are real and neither dominates.** Global compression accounts for 30%; **70%
survives any global fit and is truly per-protein.** So a per-protein correction is *not* treating
a symptom — it is treating the larger of the two components.

## And the residual is still unexplainable

After removing the global compression:

```
corr(residual, length)  = -0.124    zero at n=28
corr(residual, true dG) = +0.000    zero by construction of the fit
```

**The 70% that matters most is not a function of length, and by construction not of true dG.**
This is exactly consistent with ten failed prediction attempts — and it now has a cleaner
statement: **the unpredictable part is the majority of the offset, not a leftover.**

## A trap this rules out

A global affine rescale looks attractive — it cuts MAE from 5.07 to 1.54. But:

```
corr before rescale: +0.4104
corr after  rescale: +0.4104     IDENTICAL
sd(err) before: 1.0155  ->  after: 2.0908   WORSE
```

**Pearson correlation is exactly invariant to an affine map.** The MAE improvement is cosmetic and
the error spread actually doubles. **This is the same trap as the Flory coil's "20% MAE
improvement", and the same rule catches it: never score a calibration lever on MAE.**

## Consequence for the proposed experiment

Training with a per-protein standardised dG target would remove the 30% global component. It
would **not** touch the 70% that matters. **The experiment is worth less than I implied** — it is
a clean mechanistic check, not a route to a better score.

**Confidence: 90%.** The variance decomposition is a direct computation on 28 points.
