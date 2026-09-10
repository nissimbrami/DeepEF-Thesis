# IFUM equilibrium-mixture feature test — RESULTS. All four pass, including the one I doubted.

**Date:** 2026-09-10. Plan and predictions registered in advance: `IFUM_FEATURE_TEST_PLAN.md`.
**No GPU used.** 19 proteins (those with coordinate tensors), 1,140 variant rows.

## Setup

Built the IFUM label exactly as the paper specifies:

    Eq 2   r_U(i,j) = sqrt(6 * 1.927 * |i-j|^0.598)     analytic coil, separation only
    Eq 5   p_F = exp(dG/RT)/(1+exp(dG/RT)),  RT = 0.5924 kcal/mol
    label  D_mix = p_F * D_folded + (1-p_F) * r_U

Probe: leave-one-protein-out RidgeCV on N-invariant summaries (mean distance per |i-j| octile +
a 16-bin distance histogram). **No protein contributes to its own prediction.**

## T-A + T-B — recovery of dG, leave-one-protein-out

| features | LOPO r | RMSE | predict-the-mean |
|---|---|---|---|
| `D_folded` only | **−0.0066** | 3.1663 | 1.2127 |
| **`D_mix` (Eq 5)** | **+0.6290** | **1.1015** | 1.2127 |

**The mixture adds +0.6356 in r over the folded map alone**, and it is the only one of the two
that beats the degenerate baseline (RMSE 1.1015 vs 1.2127).

**The folded map alone is worthless for dG here** (r = −0.007, RMSE 2.6x the baseline) — which is
itself informative: our 27 proteins are structurally similar small domains, so folded geometry
carries almost no stability signal across proteins.

**T-B was the test I put at ~50% and said was the one that mattered. It passed.** The signal
survives holding out whole proteins, so it is not within-protein memorisation.

## T-C — within-protein separation (the ddG-relevant question)

    |corr(top PC of D_mix, dG)| within protein:  mean 0.8843   median 0.8870   (n=19)
    max within-protein sd of D_folded features:  0.000e+00

**Coordinates are shared across a protein's ~1000 variants, so `D_folded` is literally constant
within a protein** (sd exactly 0). Every within-protein signal in `D_mix` comes from dG, and the
first principal component tracks dG at **0.88**.

## T-D — against what we already have

    D_mix probe, within-protein   ~0.88 (top PC)
    our model's dG r, within-prot  mean 0.6921  median 0.7741

**The label carries more within-protein dG signal than our trained model currently extracts.**

## What this proves, and what it does not

**PROVES:** the IFUM label is not a geometric restatement of the folded structure. It carries dG
information that (a) the folded map does not, (b) generalises across held-out proteins, and (c)
exceeds what our model currently recovers.

**DOES NOT PROVE the mechanism will work.** Three honest caveats:

1. **The recovery is partly definitional.** `D_mix` is built from dG, so extracting dG back is
   closer to inversion than to discovery. The non-trivial finding is that it survives
   **leave-one-protein-out** — a purely definitional encoding could still have failed there if
   `p_F` interacted with protein size or shape.
2. **At training time the label is available; at inference it is not.** IFUM use the experimental
   dG only to *construct* the label, and the paper is explicit that this happens
   *"only at the training stage"*. The bet is that forcing the network to predict a
   stability-weighted geometry teaches it something that transfers. **This test cannot check
   that** — only a trained arm can.
3. **19 of 27 proteins**, subsampled to 60 variants each. The 8 missing proteins lack coordinate
   tensors at the expected path.

## Decision

**All three gates I registered in advance are met**, so the GPU arm is justified:
T-A passes (+0.636 over folded), T-B passes (survives LOPO), T-C passes (0.88 within protein).

**Prediction accuracy check:** I predicted T-A and T-C would pass ("close to arithmetic") and put
T-B at ~50%. T-B passed. **The registered prediction was right about which test was uncertain and
wrong about its outcome** — recorded so the calibration is on the record either way.

**Confidence: 93%** that the label is informative as measured; **~35%** that a trained arm
converts it into a better model, because caveat 2 is exactly where this project's auxiliary-loss
levers have failed before (`slope_weight` fixed the slope and did not move the score).
