# CORRECTION: W12 IS significant at n=27 — I tested it wrongly

**Date:** 2026-09-10. **This corrects my own claim in `W12_TWO_SEEDS.md` and `PLANNER_REVIEW.md`
that "the protein-level bootstrap CI includes zero (P=0.88)".** That was the wrong test.

## The two tests

**A) Paired t-test on per-protein ranking r** — 27 proteins, W12 (2 seeds) vs control (6 seeds),
each protein its own pair:

    mean delta  +0.02066
    t = 2.196, df = 26,  p = 0.0372            SIGNIFICANT
    95% t CI          [+0.00132, +0.04000]     excludes 0
    95% bootstrap CI  [+0.00572, +0.04106]     P(>0) = 1.000

**B) Bootstrap on the pooled correlation** — what I reported before, giving P=0.88.

## Why A is the right test and B is not

B resamples proteins and recomputes a **pooled** correlation. A pooled correlation is dominated by
**between-protein offsets** (`b_p`), which is exactly the quantity W12 is not trying to fix — and
which `FULL_ANALYSIS_2026_09_09.md` showed is driven by two outlier proteins carrying +0.120 on
their own. So B measures mostly offset noise and answers a different question.

A is paired (each protein compared to itself), targets the **ranking channel that no affine
calibration can move**, and has 26 df.

**For the claim "W12 adds information", A is the correct test. W12 clears p<0.05.**

## The honest caveat: it is underpowered, not insignificant

    Cohen's d = 0.423
    power at n=27      = 75%
    n for 80% power    = 44 proteins
    n for p<0.01       = 41 proteins

**So n=27 was NOT the binding constraint I claimed it was.** W12 cleared the bar at 75% power.
The correct statement is: *the result is significant but the study is underpowered; a replication
at n≈44 would be comfortable rather than marginal.*

Simulated probability of reaching p<0.05, resampling the observed effect:

| n | P(p<0.05) |
|---|---|
| 27 | 0.750 |
| 40 | 0.953 |
| 60 | 1.000 |

## What this changes

1. **W12 is the project's first significant lever on the information channel.** p=0.0372,
   both CIs exclude zero.
2. **"More test proteins" drops from urgent to desirable.** It would raise power from 75% to
   ~95% at n=40, but it is no longer the difference between a result and no result.
3. **Retract the framing "n=27 is the binding constraint".** The binding constraint was my
   choice of statistic.

## The lesson, which is the same one twice today

Both of today's errors were **the wrong statistic on the right data** — the pooled bootstrap here,
and mean-instead-of-median on W5. In both cases the data were fine and the test was not.

**Confidence: 95%.** Paired t and bootstrap agree; the power calculation is standard; the reason
B is inappropriate is a measured property of the pooled metric, not an assertion.
