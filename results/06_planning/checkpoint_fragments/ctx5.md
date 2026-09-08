
---

# CHECKPOINT 5 — 2026-09-07 — b_p AND a_p MEASURED DIRECTLY (no GPU)

`scripts/calc_bp.py` -> `results/calib_per_protein.json`, from
`eval_results/abl_calib_ctrl_repro2_e14.csv` (28 proteins, 28,315 rows, checkpoint e14).

| quantity | value |
|---|---|
| b_p (WT error) mean | 0.3542 |
| b_p median | 0.7207 |
| **b_p std** | **1.5741** |
| b_p range | -4.7043 .. +2.7474 |
| **a_p (ddG slope) median** | **0.4990** |
| a_p range | 0.0789 .. 1.2566 |
| **per-protein ddG PCC median** | **0.7980** |
| **corr(a_p, per-protein PCC)** | **+0.5714** |
| corr(\|b_p\|, per-protein PCC) | -0.3596 |

## What these four numbers establish

1. **The calibration gap is real and quantified.** Per-protein median PCC 0.798 against a pooled
   ~0.59. The ranking information IS present within a protein; pooling destroys it. This is the
   thesis, now measured independently of any earlier analysis.

2. **std(b_p) = 1.57 kcal/mol** is the offset spread that pooling folds into the error. It is the
   target of every reference-state lever. Any dG-side lever must be judged by whether it shrinks
   THIS number.

3. **corr(a_p, PCC) = +0.571** reproduces the +0.60 recorded in the plan docs from a completely
   separate computation. The slope is the strongest single per-protein predictor of quality, and
   `--slope_weight` (factor C) is the only lever that attacks it directly -- and it has still never
   been run.

4. **a_p min = 0.0789 explains why the affine oracle is not reproducible.** The oracle divides by
   the slope; dividing by 0.079 amplifies that protein's noise ~13x. An estimator that is not
   monotone is not a bound, which is why the "0.77-0.81" figure inverts on some checkpoints. The
   0.70-0.72 from offset removal alone survives because subtracting b_p cannot amplify anything.

## Method note worth keeping

b_p and a_p are computed on DIFFERENT metrics on purpose: b_p is the WT error, a dG-side quantity;
a_p is the ddG compression slope, a ddG-side quantity. Fitting both on one metric would silently
mix the two error modes the whole thesis is trying to separate.

## Still true, still the gap

No dG arm has ever been submitted. `--slope_weight` has never been run. Both need GPU from the
8-card cap the 48-run factorial is consuming.
