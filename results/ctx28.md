
---

# CHECKPOINT 28 — 2026-09-08 — THE COIL RESULT WAS MISREAD. b_p HAS NOW SURVIVED EIGHT EXPLANATIONS.

## The claim that motivated a whole programme

CHECKPOINT 2 recorded the project's most-cited number: the Flory coil with `--coil_b fixed`
improves absolute dG MAE from **4.9650 to 3.9521**, and it was called "our best geometric lever on
b_p". A GPU arm (`gld_dg_coil_s42`) was submitted on the strength of it.

**It does not mean what it was taken to mean. Verified by the main session on the raw
`results/w0_dg.json` per-protein data, with 2K5H's reference corrected:**

| condition | MAE | mean(b_p) | **std(b_p)** |
|---|---|---|---|
| base | 5.0751 | **-5.0751** | **0.9967** |
| **coil_fixed_b** | **4.0622** | **-4.0622** | **1.1231** |
| coil | 5.8547 | -5.8547 | 1.0743 |
| noemb | 3.2416 | -3.2416 | 1.0234 |

**All 28 of 28 proteins are UNDER-predicted in every condition.** When every residual has the same
sign, `MAE` is identically `|mean(b_p)|` — **the metric was measuring MEAN BIAS, not dispersion.**

**And the dispersion went the WRONG WAY: std(b_p) rose 0.9967 -> 1.1231, 12.7% WORSE.**

A uniform per-protein shift is exactly what Pearson is invariant to (a fact this project already
established when it showed subtracting a constant changes pooled PCC by exactly zero). **A shift
cannot generate or remove per-protein dispersion, so it cannot be a b_p lever at all.**

## The trained arm confirms it, and adds a second proof

`gld_dg_coil_s42`, scored at epoch 14 against the ddg-loss control:

| | dG arm | control |
|---|---|---|
| dG MAE | **2.4003** | 1.3029 |
| std(b_p) | 0.9656 | 1.6030 |
| per-protein ddG PCC | **0.3151** | 0.7310 |
| a_p median | **0.0852** | 0.4975 |
| pooled ddG PCC | **0.2278** | 0.5910 |

std(b_p) fell 40%, which looks like a win. **It is the degenerate baseline again.** The trajectory
settles it:

    epoch 0 : std(b_p) = 0.9410,  a_p = 0.0253,  corr(b_p, true) = -0.9262
    epoch 14: std(b_p) = 0.9656,  a_p = 0.0852,  corr(b_p, true) = -0.7744

**std(b_p) is FLAT from epoch 0 — before the model has learned anything.** The "improvement" was
present at initialisation; it is a scale property of the dG loss, not an effect of the coil. And
0.9656 sits 4.9% from the known degenerate attractor `std(true WT dG) = 0.9208`, with
`std(pred WT) = 0.6350`, i.e. the arm predicts LESS spread than the labels contain.

**Meanwhile the model got worse on the very metric it was justified by: dG MAE 1.30 -> 2.40.**

## Consequence: b_p has now survived EIGHT explanations

label noise (dead by 200x) - chain length (r=+0.025) - `--dg_length_norm` (degenerate) - embedding
norm (r=-0.29) - distribution shift (r=+0.043) - the train/test mean gap (p=0.588) - a feature-based
corrector (LOPO fails) - **and now the reference state, which was the last cheap one.**

**FINDINGS.md Part IV must be rewritten.** The line "what remains untested is the reference state,
where the coil produced the only real movement" is now false: the coil produced a mean shift, and
the trained arm degrades everything.

## Honest caveat on the trained arm

There is **no matched control**: `ref_dg_seed42/` is an EMPTY directory, so no trained dG-loss run
WITHOUT the coil exists. The comparison is against a ddg-loss control, which differs in the loss as
well as the coil. **What makes the verdict safe is the epoch-0 evidence**, which is internal to the
run: a quantity flat from initialisation was not produced by training.

## THE SLOPE ARM, by contrast, is CONFIRMED and stronger than predicted

`gld_slope1.0_s42` at epoch 14:

    a_p  0.4975 -> 0.7396   (+0.242)
    s    0.6342 -> 0.9901   (+0.356)
    r    0.7955 -> 0.7925   (-0.003, FLAT)

**The mechanism is exactly as the identity a_p = r*s predicts: all of the gain came from s, none
from r.** The identity is exact in our data (max |a_OLS - r*s| = 2.0e-15 over 28 proteins). The
trajectory shows r pinned at 0.791-0.794 from epoch 4 across ten epochs while s climbs 0.67 -> 1.02.

**a_p is now at 93% of its ceiling** (a_p <= r = 0.79). The residual (1-r) = 0.208 is the
hydrophobic ranking deficit, which this lever provably cannot touch.

This golden run BEATS the earlier factorial cell at the same flag and seed (a_p 0.740 vs 0.535), so
the full-data regime matters and FINDINGS 3.2 should be read as a **lower bound**.

**Its honest cost:** pooled gain is only **+0.0116** (pooled PCC is dominated by b_p, not a_p), and
**std(b_p) gets WORSE** (1.6030 -> 2.3585). **The arm trades offset calibration for slope
calibration.** Still n=1 seed.

## Method note worth keeping

The agent could not read `a_p median` from the training logs and **said so rather than fabricating
it**: the logging was added to train.py at 00:43 while both jobs started at 19:23 the previous
evening and hold the pre-edit module in memory. It measured the trajectory by scoring saved
checkpoints on the test set instead — strictly better evidence than a validation-set log line.
