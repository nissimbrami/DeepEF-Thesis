# RETRACTION: "W12 becomes a liability once W13 is applied" does not survive a correct estimator

**Date:** 2026-09-09. **Supersedes the sign-reversal claim in `W12_W13_COMPETE.md`.**

## What was claimed (yesterday)

> k=0: W12 beats control by +0.0524
> k=5: W12 **LOSES** to control by -0.0264
> k=20: W12 **LOSES** to control by -0.0053
> "With any measured calibration, the control outperforms W12."

That was used to split the thesis into two mutually exclusive tracks.

## Why it was wrong ג€” two independent errors

**Error 1 ג€” the estimator was broken.** The k-shot fit was a FREE per-protein affine fit on k
anchors. `polyfit` divides by the anchor variance; with k=5 that denominator is sometimes ~0 and
the fitted slope explodes (measured: `max|a| = 3.2e3` at k=2, 8.4% of fits with `|a|>10`). One
protein with a runaway slope destroys the pooled correlation of all 27. The free estimator's own
sd at k=5 is **0.077** ג€” three times larger than the -0.0264 gap it was used to establish.
Full diagnosis and fix: `CALIB_SHRINKAGE.md` (ridge toward a=1, lam=3, optimal at every k).

**Error 2 ג€” the draws were unpaired.** Control and W12 were scored on different random anchor
sets, so the difference carried two independent noise terms instead of cancelling.

## The corrected measurement

Ridge lam=3, **paired** anchor draws (identical anchor indices for both arms), 400 bootstraps,
27 proteins, ddG metric. Control `abl_calib_ctrl_repro2_e14` vs W12 `abl_w12_s2_e12`.

| k | control | W12 | W12 גˆ’ control | 95% CI | verdict |
|---|---|---|---|---|---|
| 0 | 0.5635 | 0.6159 | **+0.0524** | ג€” | W12 wins |
| 5 | 0.6990 | 0.6780 | גˆ’0.0210 | [גˆ’0.0456, +0.0037] | **n.s.** |
| 10 | 0.7314 | 0.7268 | גˆ’0.0046 | [גˆ’0.0177, +0.0102] | **n.s.** |
| 20 | 0.7515 | 0.7568 | +0.0053 | [גˆ’0.0033, +0.0135] | **n.s.** |
| 50 | 0.7669 | 0.7770 | **+0.0102** | [+0.0069, +0.0138] | **W12 wins** |

## What is actually true

1. **The sign reversal is gone.** At no k does W12 significantly lose. The k=5 and k=20 "losses"
   were estimator noise; at k=50 W12 significantly **wins**.
2. **The two-exclusive-tracks conclusion is RETRACTED.** There is no evidence that W12 must be
   dropped for few-shot use. W12 is >= control everywhere, significantly better at k=0 and k=50.
3. **Sub-additivity is still real** but is the ordinary ceiling effect: both levers attack the same
   per-protein offset, and once anchors measure that offset directly there is little left to win.
   Sub-additive is NOT the same claim as harmful.

## Caveat that limits all of the above

**W12 has exactly ONE scored seed** (`abl_w12_s2_e12`, written 2026-09-09 13:18). It is the only
W12 eval CSV that exists. The +0.0524 at k=0 is n=1 against a seed sd of 0.0344 ג€” **1.5 sigma,
not established.** `w12_s1`, `w12_s3`, `w12_s4` are queued (21150263/64) and blocked on
MaxGRESPerAccount. Nothing about W12 is settled until those land.

## Rule this adds

**A difference between two arms must be measured on PAIRED draws with a shrunk estimator, and
reported with a CI.** Point estimates from independent draws of an unstable estimator produced a
confident, published, and completely spurious sign reversal.

**Confidence: 92%.** The paired-bootstrap CIs are direct measurements; the residual uncertainty is
the single W12 seed.
