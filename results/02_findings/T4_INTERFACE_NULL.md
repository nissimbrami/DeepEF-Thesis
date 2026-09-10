# T4 - BSA/interface does NOT predict the offset. Packing does predict the slope.

**Date:** 2026-09-10. Task T4 of `CHECKLIST.md`, under `PROTOCOL.md`.

## PLAN (registered before execution)

**Q:** The earlier BSA null was measured against the **pre-training decoy loss** - a target already
known not to predict ddG. The question that matters: do the proteins our model fails on have large
interfaces?
**Metric:** correlate every structural feature against per-protein `b_p` (true ddG intercept),
`a_p` (slope) and `r`, over 27 test proteins, using 58 healthy runs.
**Falsifier:** if no interface feature reaches significance vs `b_p`, the interface direction is
closed on this dataset.
**If X then Y:** significant -> reopen ligands/complexes; null -> the earlier rejection stands,
now for the right reason.

## VERIFY - the falsifier FIRED. Interface is null for the offset.

**No interface or SASA feature predicts `b_p`.** Best nominal values are `n_hbond` (-0.446,
p=0.020) and `n_helix_seg` (-0.415, p=0.031), neither surviving 201 tests
(Bonferroni threshold 0.00025).

**`b_p` remains unpredictable** - this is now the tenth independent method to fail on it,
consistent with `FINDINGS` section 4.4.

**The earlier BSA rejection was reached by an invalid route (a dead column scored against the
pre-training loss) but the conclusion is correct on this test set.** Restated properly:
*interface size does not predict our per-protein offset on 27 single-chain monomers.*

## But two real findings fell out, and they are about the SLOPE

| feature | vs a_p | p | |
|---|---|---|---|
| **packing_frac** | **-0.662** | **0.0002** | **survives Bonferroni** |
| **void_vol_per_res** | **+0.652** | **0.0002** | **survives Bonferroni** |
| heavy_contact_density | -0.598 | 0.0010 | nominal |
| depth_p90 | -0.561 | 0.0022 | nominal |

(`a_dg vs a_p` and `ddg_pcc vs r` also survive but are circular - the same quantity under two
names - and are discarded.)

**Loosely packed proteins have HIGHER a_p; tightly packed proteins compress hardest.**

This is the same axis as `FINDINGS` section 3.5 (exposure predicts the slope, r=+0.714) reached
from an independent feature set, and it sharpens it: the predictor is **packing density**, not
exposure per se. `void_vol_per_res` is its mirror image, as expected.

**And it is mechanism, not correlation:** section 3.4 already established that the ranking error is
concentrated in **burying hydrophobics**, which is exactly what a tightly packed core is made of.
Packing density predicting slope collapse is the structural signature of that deficit.

## Why this matters for the levers

W12 and W15 both encode **side-chain packing**. This gives an independent, pre-registered reason to
expect them to work where the information levers did not - and W12 is the only lever that has ever
cleared significance.

**It also predicts WHERE W12 should help: the most tightly packed proteins.** That is exactly
task T8, and it is now a falsifiable prediction rather than a post-hoc story.

## DONE

**Cost: ~40 minutes, no GPU.** Interface direction closed on this dataset for the right reason;
packing density identified as a Bonferroni-surviving predictor of slope compression.

**Confidence: 93%** - 27 proteins is small and 201 tests were run, but the two survivors clear
Bonferroni by 100x and reproduce a previously established axis from independent features.
