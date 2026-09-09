# W12 — the first information lever ever scored, and it ranks **#1 of 53 on `r`**

`gld_w12_s2` at e12, the first side-chain arm to finish and the first width-changing arm this
project has ever been able to score at all. Canonical basis.

## The result

| | pooled | PP | a_p | r | s |
|---|---|---|---|---|---|
| **w12_s2_e12** | **0.6159** | **0.8046** | 0.3805 | **0.7485** | 0.5005 |
| control | 0.5635 | 0.7929 | 0.5211 | 0.7262 | 0.6983 |
| **Δ** | **+0.0524** | +0.0117 | −0.1406 | **+0.0224** | −0.1978 |

## Why `r` is the number that matters

Across 53 healthy runs — every lever, seed, epoch and loss mode this project has tried:

```
r: mean 0.7156   sd 0.0297
```

**Every lever moved `s`; none moved `r`.** And `r` is what correlates with pooled (+0.399 against
+0.281 for `s`). So the whole question was whether any lever could raise the ranking itself.

**W12 raises `r` by +0.0224 — three quarters of the entire historical sd — and ranks first of 53:**

| rank | run | r |
|---|---|---|
| **1** | **w12_s2_e12** | **0.7485** |
| 2 | p3_slope1.0_s42_e13 | 0.7445 |
| 3 | p3_a0_d1_s0_D0_coil_seed1 | 0.7412 |
| 4 | slope1.0_s2_e13 | 0.7382 |
| 5 | sigma_seed2_e10 | 0.7379 |

**1.11 sd above the population mean.** And per-protein PCC is 0.8046, also the highest recorded.

**This is the first lever whose mechanism is information rather than rescaling, and it is the
first to move the quantity that rescaling cannot touch.** That is exactly what the r/s
decomposition predicted would be required.

## What this is NOT yet

**n = 1.** The slope lever looked like +0.058 at n=1 and died at n=4 when seed 42 turned out to be
an outlier. **The same caution applies here and the counter-evidence is already in the data:**
`a_p` falls to 0.3805 and `s` to 0.5005 — the same "seeds 1/2/3 pattern" that the slope arm showed
outside seed 42. **W12's pooled gain of +0.0524 may be riding the same seed lottery.**

Four more W12 seeds are training (s1 at 13/15, s42 and dg_s42 at 10/15, slope_s42 at 8/15) plus
s3 and s4 from wave 10. **The verdict waits for them.**

## Pre-registered criterion, written now

**W12 is established if, across ≥3 seeds, mean `r` exceeds the control's 0.7262 by more than
0.0297 (one population sd) AND mean pooled exceeds the properly-averaged control.**
**It is rejected if mean `r` falls inside ±0.0297**, exactly as the slope lever was rejected.

**Confidence that W12 moved `r` in this run: 95%** (direct measurement).
**Confidence that the effect survives replication: 45%** — the base rate in this project is poor
and the a_p/s pattern hints at the same seed sensitivity.
