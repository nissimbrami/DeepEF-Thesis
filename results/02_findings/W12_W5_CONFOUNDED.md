# W12+W5 looks like destructive interference — but it is CONFOUNDED and cannot be read

**Date:** 2026-09-10. `w12_w5_dg_s42` landed. The naive reading is dramatic; the honest one is
that the arm cannot answer the question it was built for.

## What it shows

| arm | pooled | mean r | median r | improved | worst |
|---|---|---|---|---|---|
| control | 0.5757 | 0.7273 | 0.7897 | — | — |
| W12 alone (2 seeds) | 0.6175 | 0.7479 | 0.7965 | **18/27** | −0.017 |
| **W12+W5** | 0.5969 | 0.7262 | 0.7628 | **8/27** | −0.077 |

Taken at face value: adding W5 to W12 **destroys the entire W12 gain** — 18/27 improved collapses
to 8/27, and the median goes from +0.0080 to −0.0125.

## Why that reading is not available

**The two arms differ in TWO things, not one.**

    W12 alone   --sidechain_features                   --loss_mode ddg
    W12+W5      --sidechain_features --burial_features --loss_mode dg

`w12_w5_dg_s42` was trained on the **dG objective**; `w12_s1/s2` on **ddG**. And every dG-trained
arm in the project has a negative median on the ranking channel:

| arm | loss | median delta | improved |
|---|---|---|---|
| W12 s1 | ddg | +0.0080 | 17/27 |
| W12 s2 | ddg | +0.0076 | 21/27 |
| slope+W5 s1 | ddg | +0.0043 | 16/27 |
| **W12+W5** | **dg** | **−0.0125** | 8/27 |
| W5 s42 | dg | −0.0073 | 12/27 |
| W5 s1 | dg | −0.0079 | 12/27 |
| W5 s3 | dg | −0.0130 | — |

**The objective, not W5, is the factor shared by every arm with a negative median.**

## But the confound is NOT itself established

Tested directly: per-arm median delta, dG arms vs ddG arms.

    dg  arms: n=5,  median-of-medians -0.0079, negative in 4/5
    ddg arms: n=59, median-of-medians -0.0015, negative in 37/59
    Mann-Whitney p = 0.5926

**Not significant.** With only five dG arms — four of which are W5 variants — the dG class is
almost entirely W5, so "dG hurts" and "W5 hurts" cannot be separated with what exists.

## The honest conclusion

1. **W12+W5 does not test the W12/W5 combination.** It confounds the block with the objective.
   Its dramatic-looking collapse is uninterpretable and must not be reported as interference.
2. **W5's rejection stands but its attribution weakens.** `W5_REJECTED_4SEEDS.md` concluded W5's
   negative median is a property of the burial block. All four W5 seeds are dG-trained, so the
   negative median may belong to the objective instead. **W5-on-ddG has never been run.**
3. **The cheap fix:** one arm, `--sidechain_features --burial_features --loss_mode ddg`. That
   removes the confound and simultaneously gives the first W5-on-ddG measurement.

## Two other arms landed and are clean

| arm | pooled | mean r | median r | improved | Wilcoxon |
|---|---|---|---|---|---|
| **Flory nu=0.588** | **0.0742** | **0.1922** | 0.3813 | 2/27 | **<0.0001** |
| **Flory + coil_edges** | **0.2190** | **0.1631** | 0.3393 | 1/27 | **<0.0001** |
| w7span4 s1 | 0.5938 | 0.7319 | 0.7799 | 17/27 | 0.3242 |
| w7span4 s42 | 0.5757 | 0.7295 | 0.7874 | 17/27 | 0.2687 |

**Flory is catastrophic, and now at two configurations** — worst per-protein regression −1.562,
significant at p<0.0001 in the wrong direction. This is a far stronger rejection than the earlier
nu sweep, and it is on the ranking channel where the metric rule cannot excuse it.

**w7span4 replicates as null** across two seeds (17/27 both times, p>0.26) — consistent with the
scoreboard.

**Confidence: 94%** on the confound (the design difference is a fact read from the submit lines);
**60%** that dG-vs-ddG is a real effect — that part is explicitly not established.
