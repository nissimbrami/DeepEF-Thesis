# POSTMORTEM — every failure re-planned, every success explained

**Standing method (Nissim, 2026-09-09):** a failure is not closed until we ask whether the IDEA
failed or the PLAN or the EXECUTION failed, and whether another route exists. A success is not
recorded until we can say WHY it worked.

---

# PART 1 — FAILURES, RE-EXAMINED

## 1.1 LORO descriptors — EXECUTION failed, idea untested

| | |
|---|---|
| what happened | RMSE frozen at 2.541, val ddG PCC ~0 at every epoch |
| recorded as | "descriptors do not help" |
| **actual cause** | descriptors centred but never SCALED: median L2 3.71 vs one-hot 1.000 = **13.8x energy** |
| **verdict** | **EXECUTION fault.** The hypothesis was never tested. |
| re-plan | `mordred_pca16_only` REPLACES one-hot instead of appending, removing the scale clash. Running now as `gld_loroWdesc2_s42`. |
| second route | residual chemistry: fit embedding->chemistry, keep only the residual, standardise to unit variance, and MEASURE the block-energy ratio before submitting (W12's 1.83x was fine; 16.8x was fatal). |

## 1.2 Flory coil — the METRIC was wrong, then the idea also failed

| | |
|---|---|
| what happened | "coil cuts dG error 20%" (MAE 4.965 -> 3.952) |
| **actual cause** | `n_under = 28/28` in every condition, so **MAE is identically \|mean(b_p)\|** — it measured a uniform BIAS. Pearson is exactly shift-invariant, so a bias reduction cannot change any correlation. Meanwhile std(b_p) **rose** 0.9967 -> 1.1231. |
| **verdict** | **PLAN fault first** (wrong metric), then a genuine negative on the right metric. |
| re-plan done | swept nu 0.5-0.65 x b on the right metric. **All 10 cells worse than no coil** (best 1.1036 vs 0.9972). |
| new finding | the coil **INDUCES** a length artifact: corr(b_p,N) goes -0.215 -> -0.507 monotonically in nu. The "correct" physics is the worse artifact. |
| second route STILL OPEN | (a) `--coil_edges` — the coil changes node distances but NOT graph edges, so it has only ever been half-applied. Never run once. Queued as 21144534. (b) trained-on-unfolded with nu=0.588 (21144462) — the sweep was frozen-checkpoint only, which is a different experiment. |

## 1.3 b_p prediction — 9 attempts, all failed, and the reason is now structural

Rejected: length (r=+0.025) · length-norm (degenerate) · embedding-norm (claimed -0.99, measured
-0.29) · distribution shift (r=+0.043) · train/test mean gap (p=0.588) · LOPO ridge on 22
features (R^2 negative on 8/10) · reference state (flat from epoch 0) · **mean-pooled ProtT5
(LOPO R^2 = -0.1011)**.

| | |
|---|---|
| **why they all fail** | b_p is **96.3% ONE GLOBAL CONSTANT** (-5.0749) plus a residual spread of std 1.0155. Predicting a per-protein constant from features at n=28 is close to unidentifiable. |
| **verdict** | **the whole APPROACH is wrong**, not any single attempt. |
| **re-plan** | stop predicting. **MEASURE** it, as Ofir does: 0.369 -> 0.753 with 3 mutations per protein. That is the single highest-value item left and it costs ZERO GPU. |

## 1.4 P0c rescore — pure EXECUTION bug, caught only by the artifact rule

`run_calib_eval.sh` takes `<train_log> <model_dir> <run_tag>`; I passed `<tag> <epoch>`.
It died with FileNotFoundError **and still printed `P0C_DONE`**.
**Only `ls` on the artifact exposed it.** Resubmitted correctly as 21144483.

## 1.5 The autopilot — 24 hours of nothing

```
grep -c sbatch scripts/autopilot.py  ->  0
state: S5_BUILD_INFO, cells_done 18 > cells_total 16, no exit transition
```
It re-decided the same D1 selection every 10 minutes since 2026-09-07.
**Replaced by `scripts/refill.sh`**, which does call sbatch, is idempotent, and refuses to add
work when the lane already holds >= 16 jobs.

## 1.6 Levers "rejected" at n=1 — NOT actually rejected

W7edge, W7span4, U10bidir each ran ONCE and landed inside the +/-0.060 seed band.
**Inside a noise band at n=1 is not a rejection — it is an untested arm.**
Second seeds queued: 21144536/37/38.

## 1.7 Ligands / metals / complexes — NOT refuted, UNMEASURABLE here

All 28 test proteins are single-chain, ligand-free, metal-free monomers:
`n_chains=1, n_het=0, n_metal=0, interchain_BSA=0`. **Zero variance ⇒ a constant column
contributes nothing to any gradient.** This is a DATASET limitation, not evidence against the
idea. W11 is queued anyway (21144535) to confirm it is inert rather than harmful.

---

# PART 2 — SUCCESSES, AND WHY THEY WORKED

## 2.1 `--slope_weight 1.0` — the only lever that beat the noise band (+0.058)

**Mechanism, exactly:** `a_p = r * s` where `s = std(pred)/std(true)` — verified to 2.0e-15.
The loss can only move `s`. `r` stays flat at 0.79 across every epoch, so **the lever rescales;
it never adds information.** That gives a hard ceiling: `a_p <= r`.

**And it OVERSHOOTS.** Tracking `s` across epochs:
```
e0 0.686 (under-dispersed)  e4 0.948  e8 1.089 (already past 1.0)  e10 1.053  e13 1.144
```
Perfect calibration is `s = 1.0`. Past that the model is **over**-dispersed and `a_p` keeps
climbing only because `s` inflates — **a defect reported as a gain.**
**Consequence:** the target is `s = 1.0`, not max `a_p`, and the optimum weight is BELOW 1.0.
Untested territory, now queued: `slope 0.5` x2 seeds, `slope 0.7`.

## 2.2 Requiring D0 — 3.9 sigma

D1 (zeroing the unfolded embedding) collapses the model: 0.1508 +/- 0.1243 vs D0 0.5886 +/- 0.0094.
**Why it matters:** it proves the unfolded embedding is load-bearing, which is why any
reference-state lever must be judged on dG, not ddG.

## 2.3 The METRIC RULE — the most productive idea in the project

*Score a lever on the metric it acts on.* ddG cancels anything identical between WT and mutant,
so reference-state and whole-protein levers must be scored on dG or std(b_p); `a_p` is
within-protein and IS ddG-measurable.
**It has caught five levers**, including our own model-selection metric.

## 2.4 The DEGENERATE BASELINE rule

`--dg_length_norm` looked like a 31% improvement and was **the model predicting nothing**
(it converged to std(true WT dG) = 0.9239). Every arm must report `std(pred WT)` alongside its
score, or the number is void.

## 2.5 The ARTIFACT rule

A DONE message is not evidence; only `ls` on the output file is. This caught P0c v1 and three
earlier scoring bugs that hid 12 finished cells.

---

# PART 3 — THE PATTERN

Five of the seven "failures" above were **execution or metric faults, not refuted ideas**.
The project's signature failure is: **code runs, prints success, feature never read.**

The three rules that catch it — metric rule, degenerate baseline, artifact check — have between
them caught more real defects than any positive result has produced gain.
