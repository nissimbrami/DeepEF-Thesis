
---

# CHECKPOINT 18 — 2026-09-08 — FIRST FACTORIAL RESULTS: FACTOR D IS DECISIVE

## The 10 scored cells

| cell | pooled ddG PCC | a_p median |
|---|---|---|
| a1_d1_s0_**D0_coil** | **0.6175** | 0.5173 |
| a1_d0_s1_**D0_coil** | **0.6100** | 0.5942 |
| a0_d1_s1_**D0_coil** | **0.6064** | 0.3733 |
| a1_d0_s0_**D0_coil** | **0.5982** | 0.5736 |
| a1_d1_s1_**D0_coil** | **0.5827** | **0.7156** |
| a0_d1_s1_**D1_uemb** | 0.3121 | 0.0113 |
| a0_d0_s0_**D1_uemb** | 0.2472 | 0.0068 |
| a0_d1_s0_**D1_uemb** | 0.2179 | 0.0197 |
| a0_d0_s1_**D1_uemb** | 0.1001 | 0.0048 |
| a1_d1_s0_**D1_uemb** | **-0.0782** | 0.0136 |

## FACTOR D SEPARATES THE TWO GROUPS COMPLETELY

**Every D0 (coil) cell scores 0.58-0.62. Every D1 (`--unfolded_emb zero`) cell scores -0.08 to 0.31.**
There is no overlap. Five versus five, perfectly separated.

**And the a_p column is even starker: D0 cells have a_p 0.37-0.72, D1 cells have a_p 0.005-0.020 —
two orders of magnitude apart.** An a_p of 0.005 means the model is essentially FLAT: it barely
responds to mutation at all.

## What this means, and it is a reversal worth stating carefully

W0 chose factor D as `--unfolded_emb {full,zero}` on the ddG metric, because zeroing the embedding
in the unfolded pass collapsed var(E_u) by 67% and cut corr(E_u, wt_err) from 0.420 to 0.119. That
made it look like the cleanest way to remove the offset.

**Trained end to end, zeroing the unfolded embedding destroys the model.** It removes the offset by
removing the signal — exactly the caveat recorded in CHECKPOINT 2 for `noemb` on dG ("lowest MAE but
kills the correlation, 0.35 -> 0.05"), now confirmed under training rather than at inference.

One cell, `a1_d1_s0_D1_uemb`, is ANTI-correlated at **-0.0782**. The autopilot's own audit flagged it
(`audit: pooled out of [0,1]`), which is the audit working as designed.

**The coil arm is not merely better; it is the only arm that produces a working model.**

## Consequences

1. **The D1 half of the factorial is largely wasted compute.** 24 of the 48 cells use
   `--unfolded_emb zero`, and on this evidence they will all land in the 0.0-0.3 band. That is
   ~190 GPU-hours confirming a negative we can already see at n=5.
2. **Do not average across D.** Any main-effect estimate for A, B or C that pools D0 and D1 will be
   swamped by a factor-D effect roughly ten times larger than anything else.
3. The best cell so far, `a1_d1_s1_D0_coil`, has the HIGHEST a_p (0.7156) but not the highest pooled
   PCC (0.5827 vs 0.6175). **That is the calibration thesis in one line: the highest-slope model is
   not the highest-pooled model**, because pooled PCC is dominated by the offset while a_p measures
   the compression.
4. These numbers still carry the 2K5H reference-row bug (CHECKPOINT 17), which inflates the
   offset-removal gain by ~36%. Fix 2K5H before quoting any of this in the thesis.

## Caveats

Epochs differ across cells (e5 to e14) because each was scored at its own best epoch, so the
comparison is best-epoch to best-epoch, not equal-epoch. Two cells were scored early (e5, e6) and
may improve. All are seed 42 only — no seed replication yet.
