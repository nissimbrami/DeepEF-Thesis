
---

# CHECKPOINT 27 — 2026-09-08 — K10: THE D1 HALF IS SETTLED AT 14.4 SIGMA

## The measurement

Twelve cells scored (2K5H corrected throughout), split by factor D:

| arm | n | pooled ddG PCC | sd | range | a_p median |
|---|---|---|---|---|---|
| **D0 (coil)** | 6 | **0.5939** | **0.0091** | 0.5808 - 0.6100 | **0.5736** |
| **D1 (unfolded_emb zero)** | 6 | **0.1579** | 0.1171 | -0.0366 - 0.3056 | **0.0136** |

**The reference scale — five `abl_sigma` seeds of the SAME configuration:**

    mean 0.5798, sd 0.0303

**D0 - D1 = 0.4359 = 14.4 seed-sigmas.**

## Why this closes the question

D0's own spread is **sd = 0.0091**, three times TIGHTER than the seed noise, so the D0 arm is
remarkably reproducible. D1's spread is 0.1171 — an order of magnitude wider — which is itself
diagnostic: those models are not converging to a consistent solution, they are failing in varying
degrees. One is anti-correlated at -0.0366.

**And a_p separates by a factor of 42** (0.5736 vs 0.0136). An a_p of 0.014 means the model barely
responds to mutation at all — it is not a worse model, it is a flat one.

**No plausible number of additional seeds moves a 14.4-sigma difference.** Running the remaining
D1 cells would spend roughly 190 GPU-hours to add decimal places to a conclusion already established
at n=6 per arm.

## The scientific reading

This is the W0 decision reversed under training. W0 chose `--unfolded_emb zero` as factor D because
it collapsed var(E_u) by 67% and cut corr(E_u, wt_err) from 0.420 to 0.119 — the cleanest-looking
way to remove the offset on a frozen checkpoint.

**Trained end to end, it removes the offset by removing the signal.** That exact caveat was recorded
in CHECKPOINT 2 for `noemb` on dG ("lowest MAE but kills the correlation, 0.35 -> 0.05"); the
factorial has now confirmed it under training rather than at inference.

**This is also the metric rule in a new form.** W0 scored factor D on a frozen-checkpoint variance
decomposition, which is neither dG nor ddG performance — it measured whether the offset channel
could be silenced, not whether silencing it helps.

## Recommendation, for the user to decide

**Stop the D1 arm; redirect its ~190 GPU-hours.** Highest-value alternatives, in order:
1. **More seeds on the D0 cells** — turns single-seed cells into replicated ones, which is what the
   A/B/C main effects actually need, since their effects will be far smaller than D's.
2. **The severing experiment** (~3 GPU-h, built and dry-run green) — decides whether the whole
   reference-state programme is aimed correctly.
3. **A side-chain feature arm** — the hydrophobic ranking failure is the largest unexplained
   deficit and the only one with a clear architectural fix.

**Not acted on.** Cancelling queued cells is the user's call.
