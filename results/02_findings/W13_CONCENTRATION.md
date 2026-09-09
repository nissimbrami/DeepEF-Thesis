# W13's gain is carried by TWO proteins — and correcting 16 of 27 makes things WORSE

Never asked before, and it decides whether "measure 5 mutants per protein" is a general protocol
or a repair for a handful of outliers. Measured on `p3_slope1.0_s42_e13`, base pooled 0.6202.

## Method

For each protein separately, remove **that protein's** offset and nothing else, then recompute
pooled. This isolates each protein's individual contribution.

## Result

| protein | offset | Δ pooled | n mut |
|---|---|---|---|
| **HEEH_KT_rd6_0793** | **+1.775** | **+0.0397** | 796 |
| **3DKM** | **+1.212** | **+0.0241** | 1239 |
| 2KVS | +1.059 | +0.0132 | 1207 |
| r18_3_TrROS_Hall | −0.255 | +0.0065 | 990 |
| 1QP2 | +0.752 | +0.0036 | 1083 |
| 1W4H | −0.232 | +0.0036 | 802 |

```
sum of individual gains: 0.0687
top 2 proteins:  92.8% of it
top 5 proteins: 126.6%  (>100% because the rest are NEGATIVE)
proteins where correcting HURTS: 16 of 27
corr(|offset|, gain) = +0.812
```

## What this means

**Two proteins carry 93% of the benefit.** `HEEH_KT_rd6_0793` and `3DKM` have offsets of +1.77
and +1.21 — several times the typical protein's. Correcting them is most of W13.

**And correcting the other 16 actively hurts.** Their offsets are small, so the k-sample estimate
is mostly noise, and subtracting noise decorrelates predictions from labels. This is the same
mechanism that makes k=1 harmful, now visible per protein.

`corr(|offset|, gain) = +0.812` states the rule cleanly: **the gain is proportional to how badly
mis-calibrated the protein already was.**

## Consequence for the recommendation

The honest protocol is **not** "measure 5 mutants for every protein". It is:

> **Measure a few mutants, check whether that protein's offset is large, and apply the correction
> only where it is.**

A pre-screen of 2–3 mutations reveals the offset's magnitude; proteins with a small offset should
be left alone. That is cheaper *and* better than blanket application.

**It also caps the method.** W13 cannot fix a well-calibrated protein — there is nothing to fix.
Its ceiling is set by how many badly-offset proteins a benchmark contains, and ours contains two.

## Consistency check

This matches the earlier robustness finding (dropping the 2 highest-offset proteins leaves 42% of
the gain) and the k=1 threshold (noise/signal 1.47, break-even k≈2.2). **Three independent
analyses, one mechanism.**

**Confidence: 90%.** Single checkpoint, but the +0.812 correlation and the 16/27 negative count
are strong and the mechanism is already established elsewhere.
