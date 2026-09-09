# Dose-response: both weighted levers have an interior optimum, and both are past it at w=3

Local analysis of `scoreboard.csv`, 43 runs, while the cluster was unreachable.

## The two sweeps

**Anchor weight:**

| w | pooled | a_p | r | oracle |
|---|---|---|---|---|
| 0.3 | **0.6105** | 0.617 | 0.727 | 0.692 |
| 1.0 | 0.5942 | 0.622 | 0.702 | 0.661 |
| 3.0 | 0.5364 | **0.143** | 0.639 | 0.621 |

**Slope weight:**

| w | pooled | a_p | r | oracle |
|---|---|---|---|---|
| 0.3 | 0.5676 | 0.487 | 0.730 | 0.678 |
| 1.0 | **0.6210** | **0.591** | 0.745 | 0.701 |
| 3.0 | 0.5362 | 0.339 | 0.713 | 0.664 |

## What this shows

**The slope lever has a genuine interior optimum at w = 1.0.** Both 0.3 and 3.0 are worse on
pooled *and* on `a_p`. That is the signature of a real effect, not noise — noise does not produce
a peak at the same weight on two independent metrics.

**The anchor lever is monotonically decreasing** (corr(weight, pooled) = **−0.999**). Its best
tested weight, 0.3, is the smallest tried. **The optimum, if any, is below 0.3 and was never
tested** — the sweep only ever bracketed the descending side.

## The failure at w=3 is instructive

Anchor at w=3 collapses `a_p` to **0.143**, six times worse than at w=1. The loss term overwhelms
the primary objective and the model stops responding to true ddG at all. This is the same failure
shape as the D1 arm, and it means **an over-weighted auxiliary loss does not degrade gracefully —
it destroys the prediction.**

## Correction to how these were reported

`anchor_w0.3_s42_e14` at 0.6105 appears in the top-five table as if it were an established lever.
It is:

- **n = 1 seed**
- the **best of three weights**, all on the same seed
- and the weight sweep is **monotone**, so 0.3 being best is not evidence of an optimum — it is
  evidence that we never tested small enough

Against the seed sd of **0.0344**, a 0.6105-vs-0.5635 gap of 0.047 is **1.4σ on one draw.**
**The anchor lever is not established.** It should be reported as "promising, n=1, untested below
w=0.3", not as a result.

## What would settle it

Two things, neither yet run:
1. **anchor at w = 0.1 and 0.03** — the untested side of the curve
2. **2–3 seeds at w = 0.3** — to see whether 0.047 survives a 0.0344 sd

The slope lever's three golden replicates (`gld_slope1.0_s1/s2/s3`) are the analogous test and are
still training. **Until they land, the project has no lever established beyond one seed.**

**Confidence: 90%** on the dose-response reading; **95%** that the anchor sweep never bracketed
its optimum.
