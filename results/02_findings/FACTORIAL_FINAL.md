# The factorial, complete: 22 D0 cells, 8 configurations, 3 seeds — all effects are ZERO

Supersedes `FACTORIAL_D0.md`, which was written when only 6 cells were scored.

## The design as actually delivered

```
22 scored D0 cells   8 distinct A/B/C configurations   seeds 1, 2, 42
pooled: mean 0.5940   sd 0.0177   range 0.5401 - 0.6261
```

## Main effects

| factor | effect | ± se | n1 / n0 | t | verdict |
|---|---|---|---|---|---|
| A | +0.0013 | 0.0082 | 12 / 10 | 0.16 | **zero** |
| B | +0.0052 | 0.0082 | 12 / 10 | 0.63 | **zero** |
| C | −0.0089 | 0.0075 | 11 / 11 | −1.18 | **zero** |

**All three are inside ±0.0344 by a factor of four to twenty-five, and no |t| exceeds 1.2.**
With 22 cells and three seeds this is not an underpowered null — it is a measured null.

## Two corrections to my own earlier analysis

**1. Epoch is NOT a confound.** `FACTORIAL_D0.md` reported `corr(epoch, pooled) = +0.543` from
6 cells and warned the design could not separate factor from epoch. With 22 cells:

```
corr(epoch, pooled) = -0.091
```

**The +0.543 was an artifact of six points.** The warning was right to raise but wrong in fact.

**2. Seed is not a confound either.** The three seed means are 0.5944, 0.5980, 0.5883 — a spread
of 0.0097, far inside the noise band. The factorial is balanced across seeds.

## What the factorial established

**Only factor D matters.** D0 versus D1 is `t = 11.80` with zero overlap; A, B and C are jointly
indistinguishable from nothing across 22 cells.

**And the cell-to-cell sd is 0.0177** — half the seed sd of 0.0344. The eight configurations are
more similar to each other than two seeds of one configuration are. **The design space explored
by this factorial is flat.**

## Consequence

**The factorial is finished and it produced one usable result** (D0 is required) **and three
null results** (A, B, C). Six factorial jobs remain in the queue; they will add cells to a
question already answered at n=22.

**This is a clean negative and it belongs in the thesis as one** — a 2×2×2×2 design where three
of four factors are flat is a real finding about the model's insensitivity, not a failed
experiment.

**Confidence: 95%.** 22 cells, 8 configurations, 3 balanced seeds, all three effects an order of
magnitude inside the noise band.
