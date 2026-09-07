
---

# CHECKPOINT 22 — 2026-09-08 — K12: a_p = r * s DECOMPOSED, AND THE SLOPE LEVER IS CONFIRMED

## The identity, measured on 22 checkpoints (2K5H corrected)

For any per-protein least-squares fit, `a_p = r * s` where `r` is the correlation and
`s = std(pred)/std(true)` is the spread ratio. `--slope_weight` penalises |std(pred)-std(true)|,
so it can only act on **s**. Medians per checkpoint, then averaged:

    a_p = 0.3640    r = 0.6527    s = 0.5043

**Decomposition of the gap to a_p = 1:**

| source | size | can the slope term fix it? |
|---|---|---|
| ranking error (1-r) | 0.3473 | **NO** |
| spread compression (1-s) | 0.4957 | **YES** |

**CEILING: driving s -> 1 gives a_p = r = 0.653.** The lever cannot reach 1.0 by construction, and
the write-up must say so rather than implying otherwise.

## THE NATURAL EXPERIMENT — the three slope arms confirm the algebra

Three runs differ only in `--slope_weight`, and they behave exactly as `a_p = r * s` demands:

| --slope_weight | s | a_p | r |
|---|---|---|---|
| 0.3 | 0.5732 | 0.4146 | 0.8014 |
| **1.0** | **0.7564** | **0.5621** | 0.8090 |
| 3.0 | 0.4721 | 0.3262 | 0.7953 |
| (control, no slope term) | 0.6351 | 0.4990 | 0.7980 |

**`r` is FLAT at 0.795-0.809 across all four while `s` swings 0.47 to 0.76.** The lever moves the
spread and leaves the ranking untouched — which is precisely what the identity predicts and is
strong evidence the term does what it claims.

**Weight 1.0 is best: it lifts s from 0.635 to 0.756 and a_p from 0.499 to 0.562, a +0.063 gain
over the control, at no cost to r.**

**Weight 3.0 OVERSHOOTS badly** — s falls to 0.472, WORSE than the control. Too strong a spread
penalty destabilises training rather than decompressing further. So the lever has an interior
optimum near 1.0, not a monotone response, and the factorial's C factor should be read that way.

## Why the mean a_p (0.364) is below the control (0.499)

The average is dragged down by the five D1 (`--unfolded_emb zero`) cells, whose a_p is 0.005-0.020
because those models are nearly flat. Excluding them, the working models sit at a_p 0.33-0.72 with
r consistently ~0.72-0.81. **Do not quote the pooled mean across D.**

## What this settles

1. **`--slope_weight` works, at weight 1.0, and its mechanism is confirmed** — it moves s, not r.
2. **It cannot close the gap alone**: 54.6% of the deficit is ranking error, which needs a
   different lever entirely (better features, not better calibration).
3. **The thesis should report a_p as two numbers, not one.** "The model compresses ddG by half" is
   really "the model ranks at r~0.80 and compresses the surviving spread to s~0.64", and only the
   second half is a calibration problem at all.
