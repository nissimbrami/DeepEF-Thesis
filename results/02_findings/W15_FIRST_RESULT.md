# W15 first result: the second-strongest lever ever measured — and it is multiplicative

**Date:** 2026-09-10. `abl_w15_s42_e10.csv` landed. W15 had **never run** before today.
Reported per rule A3/§4: mean AND median, plus training health.

## The numbers

Control family, 6 same-config seeds: pooled 0.5757 ± 0.0314, mean r 0.7273 ± 0.0065,
median r 0.7897 ± 0.0024.

| arm | pooled | mean r | median r | in mean-r sd | improved | Wilcoxon |
|---|---|---|---|---|---|---|
| **W15 (n=1)** | **0.6186** | **0.7429** | **0.7966** | **+2.41** | 14/27 | 0.1855 |
| W12 (n=2, for scale) | 0.6175 | 0.7479 | 0.7965 | +3.19 | 18/27 | 0.0104 |
| control | 0.5757 | 0.7273 | 0.7897 | — | — | — |

**W15's mean AND median both exceed the control** (+0.0156 and +0.0069) — it does not have W5's
sign split. Its pooled (0.6186) is the highest of any lever arm measured, marginally above W12.

**Training health:** run TIMEOUT at epoch 11/15 (12h limit), scored at the val-selected epoch 10.
Loss was falling normally; `pub_w15b_s42` is resuming with a 24h limit to reach e14.

## Status: SCREEN, not a claim (rule A1)

**n=1. Wilcoxon p=0.1855.** This is a **screen result** and yields a ranking, not a claim.

By rule A1 it must not be multiplicity-corrected, and by A2 it needs **3 seeds minimum** before
anything is asserted. `gld_w15_s1`, `gld_w15_s2`, `gld_w15_dg_s42`, `gld_w15_w12_s42` are all
running now, so n=3+ arrives without new submissions.

**On the evidence so far W15 is the leading candidate after W12** — and it is the natural next
confirmation once the seeds land.

## Why this matters: it fits the multiplicative pattern

`B2D_EDGE_DESCRIPTORS.md` established that **every mechanism with signal in this project is
multiplicative**. W15 is built the same way — `scripts/w15_block.py` column 2 is
`inter = reach * env`, a residue property times a geometric one, plus a clash term
`relu(reach - free_space)` that is not even expressible as a product.

| lever | form | mean-r result |
|---|---|---|
| **W12** | geometric, state-dependent | **+3.19 sd, established** |
| **W15** | `reach * env` + clash | **+2.41 sd, screen** |
| W5 | `bur * hyd` | +1.00 sd mean, **−4.15 sd median** |
| W6 descriptors | `one_hot @ T` (linear) | **collapsed** |
| w7edge / w7span4 / u10bidir | linear/topological | null |

**The three multiplicative levers are the three with positive mean-r; the linear ones are null or
collapsed.** W5 is the partial exception and E8 explains it: its gain goes to *polar* destinations,
the opposite of the deficit it was built for.

## Confidence

**90%** that W15 is a genuine leading candidate; **not established** — n=1, Wilcoxon p=0.19, and
scored at e10 rather than e14. The three running seeds decide it.
