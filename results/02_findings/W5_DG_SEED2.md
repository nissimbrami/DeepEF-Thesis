# W5 (burial on dG): the a_p 0.667 headline is a SEED, not an effect

**Date:** 2026-09-09. Two seeds now scored. **The lever does not replicate.**

## a_p definition (this was itself an error source)

`a_p` in this project = **median over proteins of the slope of `pred ~ true`** (regressing
prediction on truth, within protein, then taking the median). Regressing the other direction
(`true ~ pred`) gives a completely different number — 0.741 vs 0.664 on the same file. Any table
quoting `a_p` without stating the direction is unusable.

## The measurement

| arm | a_p | gain over control |
|---|---|---|
| control `calib_ctrl_repro2_e14` | 0.496 | — |
| `gld_w5_dg_s42_e14` | **0.664** | **+0.168** |
| `w5_dg_s1_e14` | **0.507** | **+0.011** |
| mean of the two seeds | 0.586 | +0.090 |

**Seed spread = 0.157.** Seed 42 reproduces the published +0.168; seed 1 gives +0.011 — within
rounding of nothing. The two seeds differ from each other by more than seed 1's entire gain.

## Verdict

**The "best single-flag a_p 0.667" claim in `IDEA_LEDGER.md` is n=1 and does not replicate.**
It should read: *a_p 0.664 (seed 42) / 0.507 (seed 1); mean +0.090, spread 0.157; not established.*

A third seed (`gld_w5_dg_s3`, job 21137937, running 9h) will decide whether the mean gain is real
or whether seed 42 was simply a favourable draw — exactly the pattern that already retired the
slope lever (seed 42 was an outlier there too, at s=1.053).

## Pooled score, for completeness (ridge lam=3, 27 proteins, ddG)

| arm | k=0 | k=20 |
|---|---|---|
| control | 0.5635 | 0.7524 |
| w5_dg s42 | 0.5956 | 0.7401 |
| w5_dg s1 | 0.6060 | 0.7449 |

Both seeds beat control at k=0 and **both fall below it at k=20** — the same shape seen for W12:
a lever that improves zero-shot calibration is worth less once anchors measure the offset directly.

**Confidence: 94%** on non-replication (two independent seeds, one file each, direct measurement).
