
---

# CHECKPOINT 7 — 2026-09-07 — THE THESIS SURVIVED ITS STRONGEST ATTACK (measured, CPU only)

## The attack

An adversarial critic raised the sharpest objection anyone has put to this project:

> b_p is not a calibration error the model could have avoided. Subtracting a per-protein
> constant FITTED ON THE TEST LABELS removes between-protein variance from a pooled
> correlation **whether or not that constant is physically meaningful**. A perfectly-ranking
> predictor polluted with PURE RANDOM per-protein noise would show the same 0.59 -> 0.71 lift.
> If so, the headline result is a property of the ESTIMATOR, not of the model.

This had to be answered before any GPU was spent, because if true the whole programme is
arithmetic. It costs seconds to test.

## The test — `scripts/null_offset.py` and `scripts/null_fit.py`

Build a synthetic predictor with the REAL per-protein structure (real protein sizes, real ddG
values from the reference eval CSV) but a MEANINGLESS offset: `pred = a*true + b + noise`,
with `b ~ N(0, 1.5741)` (our measured std(b_p)) drawn independently per protein. Then grid
search (noise_sd, slope) over 25 x 21 cells, 3 seeds each, and ask how close the null can get
to our measured triple.

## Result: THE NULL CANNOT REPRODUCE US

| | pooled | offset-removed | per-protein |
|---|---|---|---|
| **measured (ours)** | **0.590** | **0.710** | **0.798** |
| best null over the whole grid | 0.620 | 0.785 | 0.712 |
| error | +0.030 | **+0.075** | **-0.086** |

Best achievable L2 distance: **0.118**, and it fails in a STRUCTURED way, not by a little
noise everywhere:

1. **The null over-delivers on offset removal (+0.075).** Of course it does: a meaningless
   constant is perfectly removable by construction, so subtracting it recovers a near-perfect
   ranking. Our real 0.710 is much LOWER than what a pure offset predicts, which means our
   residual error is NOT a clean per-protein constant. There is real within-protein error that
   the offset cannot absorb — the model is wrong in ways beyond a shift.
2. **The null under-delivers on per-protein PCC (-0.086).** Our within-protein ranking is
   BETTER than the null's at the same pooled value.
3. **The null needs slope 1.25 to get even that close** — a predictor that EXPANDS ddG. We
   measured a_p = 0.499, compression by half. The null reaches our pooled number by the wrong
   mechanism entirely.

## What this establishes

**The 0.59 -> 0.71 lift is not merely an estimator artifact.** The two error channels are
real and separable: a genuine per-protein offset (b_p) AND a genuine compression (a_p), and
no single random-offset model produces both at once.

**But the critic was still half right, and this is the honest caveat to carry:** offset removal
uses a constant fitted on test labels, so 0.70-0.72 is an ORACLE and cannot be claimed as
model performance. The claim it licenses is precisely: *the ranking information is present;
what is lost is calibration.* Turning that into a method requires an offset predicted from
FEATURES on held-out proteins — which is exactly the 100k-corrector direction, and the reason
that direction matters more than it looked.

## Method note

Both scripts are pure-stdlib and run in seconds on a login node. They should be re-run against
the factorial's own eval CSVs when those land, since every arm's claim rests on the same
arithmetic. A result that survives its strongest attack is worth more than one that was never
attacked.
