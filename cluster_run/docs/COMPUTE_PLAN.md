# COMPUTE_PLAN.md — using the cluster properly

**This document supersedes §6 of `ORIENTATION.md`.** Steps 0–4 there are unchanged. What changes
is everything after σ is known, because the experiment design in that section was written for a
single 8 GB laptop GPU and is the wrong design for this cluster.

---

## 1. The constraint moved, so the design must move

On one RTX 4070 the binding constraint was wall-clock: 42 min per epoch, ~10.5 h per run, so a
sequential ladder with a gate after each rung was the only affordable design. Levers were run one
at a time, and interactions between them were never measurable.

On this cluster the binding constraint is no longer time. It is **how many effects can be cleanly
attributed**. That is a statistics problem, not a compute problem, and it has a better answer than
a ladder.

**A sequential ladder gives main effects only, one at a time, each conditioned on the previous
winner. A factorial design gives main effects and interactions from the same number of runs.**

This matters concretely here. The one existing observation on the WT anchor is that anchor 1.0
plus designed-reweight ×3 reduced offset spread the most (0.29 → 0.19) **and** collapsed the
slope distribution the most (max 1.5 → 0.85). That is an interaction. A ladder cannot see it. A
factorial measures it directly.

---

## 2. What the hardware changes, beyond speed

Four things, in order of importance. Only the first is about time.

### 2.1 Mini-batch 64 becomes possible — this is a correctness fix, not a speedup

`train.py` line 40 hardcodes `MINI_BATCH_SIZE = 64`. Every reference number in this project was
produced at 64. The 8 GB laptop could not hold it, so local runs used 16, which is **4× more
optimizer steps on the same data** — a different optimisation regime.

The local reproduction landed at pooled 0.534 (against the 0.531 baseline) but per-protein 0.655
(against 0.691). **Mini-batch is the leading explanation for that per-protein shortfall.** On a
4090 or a 6000/L40s, 64 fits comfortably.

**Do not treat this as free.** Measure peak VRAM at 64 on the largest protein before committing,
and record the value used with every result.

### 2.2 Memory-bound levers become runnable at all

`--pooled_corr_weight` holds `pooled_window × pooled_cap × 2` graphs live at backward. At the
defaults (8 × 8) that is 128 graphs — impossible on 8 GB, routine on 48 GB.

That lever was recorded as negative, but it was run at those small window and cap values under
memory pressure. **One retry at a genuinely large window is justified**, because "no effect at
window 8" and "no effect" are different claims. Budget one arm for it, not a campaign.

The same applies to any auxiliary distogram head if the coil work reaches that point.

### 2.3 Statistical resolution

Lever effects are expected in the 0.01–0.05 range. Whether they are measurable depends entirely
on σ, which nobody has measured.

- With 1 seed you can resolve roughly 2σ. If σ ≈ 0.02, that is 0.04 — **most levers are invisible**.
- With 5 seeds the standard error is σ/√5, so you resolve roughly 0.02.
- With 5 seeds **and** 5-fold cross-validation instead of a single 10% validation split, tighter
  still, and every training protein is used.

**Parallelism buys resolution, and resolution is what makes the thesis defensible.** This is the
single most valuable thing the cluster provides.

### 2.4 Hyperparameters that were never touched

`gaussian_coef = -0.08`, `k_neighbors = 30`, `distance_cutoff = 12 Å`, `dropout = 0.2`,
`num_layers = 3`, learning-rate schedule. **None of these was ever swept**, because at 10 h per
run a sweep was unaffordable. Some are physically meaningful — the Gaussian coefficient sets the
interaction length scale, and the cutoff sets what counts as a contact.

This is a legitimate late-phase use of spare capacity. It is **not** a lever with a hypothesis
behind it, so it goes last and it is reported as a sweep, not as a finding.

---

## 3. Revised phase structure

Phases 0–2 are sequential because everything downstream depends on them. Phase 3 onward is
parallel.

### Phase 0 — environment and data (unchanged)

`ORIENTATION.md` §5 steps 0–1. Find Shahar's processed tensors on this cluster before downloading
anything; `calib_ctrl` was produced here.

### Phase 1 — reproduce the reference, exactly

```
python Megascale-fineTuning/train.py \
  --full_data --no_pretrain --no_freeze --loss_mode ddg \
  --pooled_corr_weight 0 --dg_length_norm none --affine_calib \
  --wt_anchor_weight 0 --designed_weight 1 \
  --val_frac 0.1 --epochs 15 --run_tag calib_ctrl_repro
```

**Change nothing.** Not the mini-batch (leave the hardcoded 64), not the split, not the epochs.
This run exists to prove the environment reproduces 0.606 / 0.711. A reproduction with a
modification is not a reproduction.

**Gate:** pooled ≈ 0.606, PP ≈ 0.711. Epoch-2 reproduction check ≥ 0.55 or stop early.

**Note for context:** the local run already reproduced the *thesis* baseline (0.534 vs 0.531,
ΔG objective, `pnas_train.py`, mini-batch 16). That validates the data and the pipeline. It does
**not** validate `calib_ctrl`, which is a different entrypoint, a different objective, and a
different split.

### Phase 2 — σ, five seeds, parallel

Five submissions of Phase 1's configuration with `--seed {42, 1, 2, 3, 4}`, distinct `--run_tag`.
On five GPUs this is one run's wall-clock.

**Report mean ± σ for pooled and PP.** σ sets the detection threshold for everything after, and
determines whether Phase 3 needs 3 or 5 seeds per cell.

**Second deliverable, free:** average the five seeds' predictions. Seed ensembling typically adds
0.02–0.03 and costs nothing since the runs already exist. Report it separately — it is a
deployment result, not a scientific finding.

### Phase 3 — the screening factorial

**This replaces the ladder.**

Four factors at two levels each:

| Factor | Low | High | Why |
|---|---|---|---|
| **A** — WT anchor | `--wt_anchor_weight 0` | `--wt_anchor_weight 1.0` | Attacks offset `b`; already shown to reduce std(b) 0.29 → 0.25 |
| **B** — designed reweight | `--designed_weight 1` | `--designed_weight 3` | Attacks the folds where slope collapses; 121 of 368 train proteins are designed |
| **C** — slope term | off | on, weight to be fixed in a pilot | Attacks `a`; **nothing else on the list touches slope**, and slope predicts per-protein performance best (+0.60) |
| **D** — coil | off | `--flory_unfolded --flory_nu 0.5` | 77–88% of across-protein ΔG variance lives in `E_unfolded`; the coil makes that state depend only on sequence separation |

2⁴ = 16 cells. At 3 seeds that is **48 runs**; at 5 seeds, 80. Choose based on Phase 2's σ: if
σ ≤ 0.01, three seeds suffice for screening; if σ ≥ 0.02, use five.

**What this yields that a ladder cannot:**

- Four main effects, each averaged over all levels of the others — more robust than a single
  conditional comparison.
- **All six two-way interactions.** In particular **A×B**, which is the one existing observation
  we cannot currently interpret, and **A×C**, which asks directly whether fixing the offset and
  fixing the slope are complementary or antagonistic.
- A defensible statistical statement per effect, not a difference of two numbers.

**Every cell reports:** pooled ΔΔG PCC, per-protein PCC, absolute wild-type ΔG PCC, `std(b)`, the
distribution of slope `a` (min, median, max, count below 0.3), and the same restricted to designed
folds. Slope is a primary outcome here, not a diagnostic — the anchor's known cost is slope
collapse, and a cell that improves pooled while destroying slope is not a win.

**Before the factorial, one small pilot:** factor C's weight is unknown. Run three weights at one
seed with everything else off, pick the one that moves `std(pred)/std(true)` toward 1 without
hurting pooled, and fix it. That is 3 runs and it prevents 16 cells from being run at a
meaningless setting.

### Phase 4 — refinement, driven by Phase 3

Sweep the factors that showed a main effect, at 5 seeds and finer levels: anchor at
{0.3, 1.0, 3.0}; designed at {3, 5}; slope weight around the pilot's choice.

**The deliverable is a trade-off curve**, not a maximum: `std(b)`, slope distribution, and PP as
functions of anchor weight. That curve is a result in its own right and it does not exist
anywhere.

Also in this phase, one arm each:

- **Objective arm.** Run the best configuration under `--loss_mode dg` as well as `ddg`. The coil
  changes `E_unfolded`, which enters ΔG directly but partially cancels in ΔΔG. Prediction: **the
  coil helps more under ΔG training.** Two arms test it.
- **`--pooled_corr_weight` at a large window**, now that memory allows. One arm, to close a
  question that was answered under a memory constraint.

### Phase 5 — mechanism, not score

For the coil arm, regenerate the branch-energy columns and test the prediction: after the coil,
`corr(E_u, GAT-only)` falls, `corr(E_u, GCN-only)` rises, `var(E_u)` shrinks. Today the spatial
branch carries 98–99% of ΔG and the chain branch −0.063 — backwards for a state in which only
chain-local terms should survive.

**Report it whichever way it comes out.** A mechanism test that fails is a finding; a score that
moves without a mechanism is not.

### Phase 6 — spare capacity

Only if Phases 3–5 are complete and capacity remains:

- Hyperparameter sweep: `gaussian_coef`, `k_neighbors`, `distance_cutoff`, `dropout`, `num_layers`
- 5-fold cross-validation of the final configuration for tighter error bars, using `--fold` for
  parallel folds across GPUs
- Structural features from the group's biologist, when available (burial **zeroed in the unfolded
  state**, normalised by a constant not by chain length; metal coordination; secondary structure;
  disulfides)

---

## 4. Scheduling

Assume a 15-epoch run at mini-batch 64 on a 4090 takes 3–4 h — roughly 3× faster than the local
10.5 h, from the larger batch raising utilisation off 37% and from the faster card. **Verify this
on the first run and re-plan from the measured number.**

| Phase | Runs | Concurrency | Wall-clock |
|---|---|---|---|
| 1 — reference | 1 | 1 | 4 h |
| 2 — σ | 5 | 5 | 4 h |
| 3 — pilot for C | 3 | 3 | 4 h |
| 3 — factorial, 3 seeds | 48 | 8 | ~24 h |
| 4 — refinement | ~30 | 8 | ~16 h |
| 5 — mechanism | reuses Phase 3/4 checkpoints | — | hours |

**Under a week of wall-clock for what was two months of sequential laptop time.**

**Queue discipline.** Do not submit 48 jobs at once. Submit in waves of 8, verify the first wave
completes and its numbers are sane, then continue. A malformed flag replicated across 48 jobs
wastes a day and annoys everyone sharing the partition.

**Node selection.** Prefer `cs-4090-*`. `cs-6000-*` and `ee-l40s-*` have more VRAM and are the
right choice for the memory-bound arms (`pooled_corr`, large batch). Avoid `cs-1080-*` and
`cs-2080-*` for training: too little VRAM for mini-batch 64, and mixing GPU types across seeds of
the same cell introduces a confound. **Keep all seeds of one cell on the same GPU type**, and
record the type with every result.

---

## 5. What does not change

- The reference is `calib_ctrl`, 0.606 / 0.711. Not 0.655.
- Selection on validation, never on test. The test set is touched once per run.
- Every number carries three qualifiers: pooled or per-protein · ΔG or ΔΔG · which split and
  which selection.
- Results come only from `run_calib_eval.sh` → `score_runs.py`, never from in-training
  `validate()`.
- `shaharec/DeepPEF` is read-only. No push to `nissimbrami/DeepEF-Thesis` without per-action
  approval.
- The goal is calibration — making `a_p` and `b_p` learnable — not a leaderboard number. The
  oracle 0.80 is not a target.

---

## 6. Expected outcome, revised

From the 0.606 reference: **pooled ΔΔG median ≈ 0.68, range 0.63–0.74.**

The cluster does not raise the median much, because the levers overlap — the anchor, the coil and
the designed reweight all act on the same calibration failure, and their effects do not add. What
it raises is **the probability of landing in the upper part of that range**, and more importantly
**the fraction of the result that can be attributed to a specific cause**.

That second thing is what the thesis is graded on. A factorial with interaction terms, five seeds,
a trade-off curve and a mechanism test is a stronger submission than a higher number with a
sequential ladder behind it.
