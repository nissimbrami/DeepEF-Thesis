# The three "per-protein r" numbers are three different statistics — and one lever flips sign

**Date:** 2026-09-10. Raised by the planning agent: `0.8034`, `0.7445` and `0.7273` all circulate
as "per-protein r". The planner was right that this invalidates comparison unless resolved.
**Resolved below by direct measurement. It is not a bug; it is three statistics wearing one name.**

## 1. All four conventions, on the control run (27 proteins, ddG)

| convention | value |
|---|---|
| **mean** of within-protein r | **0.7262** |
| **median** of within-protein r | **0.7928** |
| n-weighted mean | 0.7180 |
| Fisher-z mean | 0.7520 |

Keeping 2K5H shifts each by about +0.005. Dropping WT rows changes nothing (they are single
rows with ddG=0). Spearman gives 0.6569 mean / 0.7132 median. dG and ddG give identical
per-protein r, as they must — within a protein they differ by a constant.

## 2. Where each circulating number comes from

- **`0.798` (the FINDINGS headline)** = the **MEDIAN**, reported as a mean. The control's median
  is 0.7928; several runs land at 0.7978-0.7999. **Already flagged in the audit.**
- **`0.8034`** = the **median** of a *different, better run* — `p3_slope1.0_s42_e13` gives 0.8036
  and `w12_s2_e12` gives 0.8046. Not the control, and not a mean.
- **`0.7445`** = the scoreboard's `r` column. Source: `scripts/arm_analyze.py:72`,
  `out['r_median'] = median(rs)` — **also a median**, but computed on a different run population.
  It also coincides with the Fisher-z mean of several runs, which is why it looked like a third
  definition.

**So the numbers are all real. None of them is a mean of the control except 0.7262.**

## 3. The conclusion that matters: does W12 survive every convention?

**Yes. Under all four, and never below +2.75 sd.**

| convention | control mean | control sd (n=6) | W12 (n=2) | gain | **W12 in sd** | W5 in sd |
|---|---|---|---|---|---|---|
| mean | 0.7273 | 0.0065 | 0.7479 | +0.0207 | **+3.19** | +1.00 |
| median | 0.7897 | 0.0024 | 0.7965 | +0.0069 | **+2.80** | **−4.15** |
| n-weighted | 0.7198 | 0.0077 | 0.7409 | +0.0211 | **+2.75** | +0.85 |
| Fisher-z | 0.7503 | 0.0049 | 0.7643 | +0.0140 | **+2.83** | −0.44 |

**The W12 finding is convention-independent.** The planner's concern was legitimate and is
answered: W12 adds ranking information under every reasonable definition.

## 4. A NEW finding the planner's question exposed: W5 flips sign

**W5 is +1.00 sd on the mean and −4.15 sd on the median.** That is not noise — it is a
qualitative difference, and it means:

> **W5 raises the average protein while LOWERING the typical protein.**

A lever can only do that by helping a few bad proteins a lot while mildly hurting the majority.
This is the same concentration signature as W12 (5 proteins carry 98.4%) and W13 (92.8% from two)
— but for W5 the majority effect is **negative**, which the mean hides entirely.

**W5's already-weak case is weaker than recorded.** Combined with `W5_DG_SEED2.md` (a_p 0.664 vs
0.507 across seeds — the published 0.667 was one favourable seed), W5 should not be reported as a
working lever.

## 5. Rule this adds

**Every per-protein number must state mean-or-median.** The project has now been bitten three
times by this exact ambiguity: the 0.798 headline, the 0.7534 oracle (wrong run), and now W5's
sign. `arm_analyze.py` emits `r_median`, `a_p_median`, `s_median` — all medians, none labelled as
such at the point of use.

**And: report both.** A lever whose mean and median disagree in sign is telling you the gain is
concentrated, which is the single most important structural fact about this dataset.

**Confidence: 97%** — every value recomputed directly from the CSVs; the code line defining the
scoreboard's `r` is quoted verbatim.
