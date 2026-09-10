# Feature test for the IFUM equilibrium-mixture label — PLAN, before any code

**Date:** 2026-09-10. The question Nissim asked: **does this feature carry new information, and
does it work on its own, outside the model?**

## What is being tested

IFUM's mechanism is not "a better unfolded map". It is: **the equilibrium distogram, mixed by the
protein's own experimental dG, as a dense per-residue-pair training label.**

    Eq 2   r_U(i,j)^2 = 6 * 1.927 * |i-j|^0.598          analytic coil, sequence separation only
    Eq 4   [F]:[U] = exp(dG/RT) : 1
    Eq 5   p_F = exp(dG/RT)/(1+exp(dG/RT)),  p_U = 1 - p_F
    label  D_mix(i,j) = p_F * D_folded(i,j) + p_U * r_U(i,j)

**The claim to test: `D_mix` contains information about dG that `D_folded` alone does not.**

## Why this is testable without the model

The mixture is a deterministic function of (folded coords, dG). So the honest question is the
INVERSE one: **given a mixed distogram, can dG be recovered?** If a simple probe recovers dG from
`D_mix` but not from `D_folded`, the label demonstrably carries stability information, and
training a network to predict it is injecting that information densely.

If dG is NOT recoverable, the auxiliary task is teaching geometry we already have, and the
mechanism cannot work here regardless of implementation.

## Already established (this session)

`p_F` on our 27 test proteins, 27,189 variants:

    fraction with p_F > 0.99            0.3911
    fraction with p_F in [0.01, 0.99]   0.6089      <- informative range
    fraction with p_F < 0.01            0.0000

**61% of variants sit where the mixture actually mixes.** Not degenerate. (RT = 0.5924 kcal/mol;
dG spans 10.1 RT.)

## The tests, in order

**T-A. Signal existence.** Compute `D_mix` per variant. Fit a probe from `D_mix` -> dG. Compare
to the same probe from `D_folded` -> dG. **Falsifier: if D_folded already predicts dG as well as
D_mix, the mixture adds nothing.**

**T-B. Leakage control — the one that decides validity.** `D_mix` is BUILT from dG, so recovering
dG from it is near-circular. The honest version: hold out proteins, fit on the rest, and ask
whether the recovered signal generalises. **Falsifier: if it only works within-protein, it is
memorisation, not information.**

**T-C. The variant-level question.** Coordinates are SHARED across a protein's ~1000 variants —
only dG differs. So within a protein, all variation in `D_mix` comes from dG. **Test: does
`D_mix` separate variants of the SAME protein by their dG?** This is the ddG-relevant question
and cannot be answered by the folded map alone (which is constant within a protein).

**T-D. Does it beat what we already have?** Compare the probe's dG recovery against our model's
own `pred_deltaG`. If the mixture recovers dG worse than the model already does, it is not a
useful auxiliary signal for us even if it is informative in principle.

## Method

- Probe: ridge regression AND gradient boosting on summary statistics of the distogram
  (binned distance histograms, per-separation means), not raw [N,N] which would overfit at n=27.
- Split: **leave-one-protein-out**, so no protein contributes to its own prediction.
- Metrics: Pearson r and RMSE on held-out dG; report BOTH, plus the degenerate baseline
  (predicting the mean), which is the bar any positive result must clear.
- Report mean AND median where per-protein numbers are involved, per the standing rule.

## What would make me build the GPU arm

**All three:** (1) T-A shows `D_mix` beats `D_folded`, (2) T-B survives leave-one-protein-out,
(3) T-C shows within-protein separation. **Any one failing means the arm is not worth 8 GPU-hours**,
and that is a real result about the mechanism on our data.

**Prediction, registered now:** T-A and T-C will pass (they are close to arithmetic — the mixture
is built from dG, so it must encode it). **T-B is the one I am unsure about, and it is the one
that matters.** I put it at ~50%.
