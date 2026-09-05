# DeepEF — Current State and Next Actions

**Last updated: 2026-09-05, evening.** This file records where the project stands and what happens next. It is
written to survive a context reset: if you are reading it cold, this plus `ORIENTATION.md` and
`COMPUTE_PLAN.md` is everything.

---

## 1. Where we are

### Confirmed by measurement, today

The local run from 2026-09-04 was rescored with the correct per-protein metric. It had been
misread as a failure at 0.28; it was not.

| Checkpoint | pooled ΔΔG PCC | per-protein (PP) | pooled absolute ΔG |
|---|---|---|---|
| epoch 8 | 0.5337 | 0.6548 | 0.2718 |
| **epoch 12** | **0.5500** | 0.6617 | 0.2683 |
| epoch 14 | 0.5490 | **0.6645** | 0.2811 |

Configuration: `pnas_train.py`, ΔG objective, random init (`--no_pretrained`), mini-batch 16,
15 epochs, held-out 28 proteins, in-training epoch selection.

**Four conclusions:**

1. **The thesis baseline is reproduced and slightly exceeded** — 0.550 pooled against 0.531.
2. **The "0.28" was a metric artifact.** The in-training `validate()` subtracts one global
   wild-type reference for all proteins, understating the real per-protein signal roughly 2×.
   Rule 3 (name the metric before reading the number) paid for itself.
3. **The ΔΔG objective is worth about +0.05, not +0.12.** From 0.550 to `calib_ctrl`'s 0.606.
   The larger figure came from the retracted, test-peeked 0.655. **Expect the first cluster run
   to land near 0.60, and treat that as success rather than shortfall.**
4. **Peak at epoch 12, plateau by 14.** Third independent dataset showing 12–14. The 15-epoch
   budget is confirmed.

### A prediction to check later, recorded now

The per-protein gap to `calib_ctrl` is only 0.6645 vs 0.711 — **0.026**. That is smaller than
mini-batch 16 vs 64 would suggest. **If the cluster run at mini-batch 64 does not close that gap,
mini-batch was not the explanation and the cause is elsewhere.** Write this down before the run,
not after.

### Decisions taken

- **No local data adapter.** `train.py` reads the old K50 tensor layout, which does not exist in
  WSL. `calib_ctrl` was produced on the BGU cluster where those tensors live. Building a local
  converter for a cluster-bound job is wasted work. **Consequence: `train.py` cannot run locally**,
  so preflight checks 1, 2, 5, 7 and the smoke test are cluster-only.
- **Reference is `calib_ctrl`: pooled 0.606 / PP 0.711**, via `train.py`. Not 0.655.
- **Epoch budget 15.** Confirmed three times independently.
- **Work inside WSL, not across the Windows boundary.** Writing through `\\wsl.localhost\`
  injected CRLF and broke a script; it is also the slow path.

### A second stale-premise incident, same evening

A five-seed sweep was launched to measure σ, using `ref_dg_seed*` — the ΔG objective on
`pnas_train.py`, mini-batch 16. **σ is needed for `calib_ctrl`**: `train.py`, ΔΔG objective,
`--val_frac 0.1`, mini-batch 64. Different objective, entrypoint and split. **σ does not transfer
between configurations**, so the sweep would have produced a number that could never be used. It
was stopped at seed 1, epoch 4.

Two incidents now share one pattern: **a clean run executed faithfully against a stale premise.**
Both would have been caught by the run card in `RUNBOOK.md` §1, specifically its fifth line —
*is the comparison number from the same entrypoint, objective, split and mini-batch?*

**`RUNBOOK.md` is the countermeasure. Read it at the start of every session and after every
compaction, before any command.**

Also recorded from that run: a **1h20m stall at step 76→77**, then normal speed resumed. A
reliability datum about this laptop under long jobs, and further justification for the cluster.

### An operational failure mode, five occurrences in one session

Shell variables and command substitutions are mangled when passed inside a `wsl -- bash -lc`
string: `$f`, `$D`, `$S`, `$(find ...)`, `< <(...)` all failed. The last caused a runaway that
executed markdown as shell and triggered a stray `git clone` into `cluster_run/DeepPEF/`
(inside Nissim's fork, not Shahar's; a clone can only pull, so no damage — but delete it).

**The fix is not care, it is not crossing the boundary.** Work inside one persistent `tmux`
session, or write a script into WSL with a quoted heredoc and run it by path. No `$var`, no
`$(...)`, no `< <(...)` inside a `wsl -- bash -lc` string, ever.

---

## 2. What is missing before training can start

Three items. Roughly one working day.

| # | Item | Status | Why it blocks |
|---|---|---|---|
| 1 | `--seed` patch to `train.py` | **done**, verified in isolation | `RANDOM_SEED = 42` hardcoded at line 33. Without it there is no σ |
| 2 | `calib_diag.py` | written; **end-to-end validation not yet run** | Measures `std(b)` and the slope distribution — **the actual objects of study**. Must reproduce the 0.77–0.81 offset-removed range on a committed eval CSV, or the instrument is wrong and nothing downstream can be trusted. **Highest-value outstanding item** |
| 3 | `monitor.py`, `preflight.py` | written, partly tested | Make a failed run cost one hour instead of ten |
| 4 | **Slope-term loss** | **not written — exists in no codebase** | The one factorial factor with no implementation. Without it the cluster factorial stalls waiting for code |
| 5 | The six cluster scripts | written, `bash -n` clean, untestable locally | — |
| 6 | `CHECKLIST.md` | **not written** | The gate that proves the package is complete before it ships |
| 7 | Package pushed to GitHub | **not done** | The cluster agent cannot start without it. **Critical path** |
| 8 | Cluster access verified | not done | Environment, QOS, partition, and where the data is. **Can be done in parallel, independently of the package** |

Items 2, 4, 6 and 7 are what stand between here and the first cluster run.

---

## 3. On the four levers — two exist, two do not

| Lever | Status |
|---|---|
| **WT anchor** | Exists: `--wt_anchor_weight` in `train.py`. Already run at 0.3 and 1.0 by Shahar; reduces `std(b)` 0.29 → 0.25 |
| **Designed reweight** | Exists: `--designed_weight`. Combined with the anchor it gave the largest offset reduction (0.19) **and** the largest slope collapse |
| **Slope term** | **Does not exist. Nobody has written it.** This is the thesis's own contribution |
| **Coil (Flory unfolded state)** | Implemented in the v5 bundle, but v5 is not synchronised with `train.py` |

**Do not enable any of them before σ is known.** Running a lever without a measured noise floor
is how this project already lost seven hours once.

**But the slope term can be written now**, in parallel with everything else, because it depends
on nothing: a loss term penalising `abs(std(pred_ddG) − std(true_ddG))` within each protein,
behind a flag defaulting to off, so the baseline stays bit-identical when the flag is absent.
Writing it now means all four factorial factors are ready when σ arrives.

---

## 4. The path from here

| When | Where | What |
|---|---|---|
| Today | local | `--seed` patch, `calib_diag.py`, `monitor.py`, `preflight.py`, assemble `cluster_run/`, push |
| Today, parallel | local | Write the slope-term loss behind a default-off flag |
| Day 1 | cluster | Environment and data check; reproduce `calib_ctrl` at mini-batch 64. **Gate: pooled ≈ 0.606, PP ≈ 0.711** |
| Day 2 | cluster | Five seeds in parallel. **Gate: σ reported.** Free bonus: average the five for a seed ensemble |
| Day 3 | cluster | Pilot the slope-term weight (3 runs, 1 seed) |
| Day 3–4 | cluster | **The screening factorial**: anchor × designed × slope × coil, 2⁴ = 16 cells, 3 or 5 seeds depending on σ |
| Day 5+ | cluster | Refinement sweeps, the two-arm ΔG/ΔΔG test, the coil mechanism check |

**Three to four days from now until levers are actually training**, and most of that is waiting
on results rather than working.

---

## 5. Standing reminders

- **Read `RUNBOOK.md` after every compaction, before any command.** Two runs have already been
  lost to stale premises; its run card is what stops a third.
- The design on the cluster is a **factorial, not a ladder**. See `COMPUTE_PLAN.md`. Running
  levers one at a time was a laptop-era compromise and it cannot measure interactions — and the
  one interaction we already have evidence for (anchor × designed) is the most interesting open
  question we have.
- Selection on validation, never on test. Test touched once per run as a reproduction check.
- Every number carries three qualifiers: pooled or per-protein · ΔG or ΔΔG · which split and
  which selection.
- Results from `run_calib_eval.sh` → `score_runs.py` only. Never from in-training `validate()`.
- `shaharec/DeepPEF` is read-only, always. No push to `nissimbrami/DeepEF-Thesis` without
  per-action approval.
- The goal is calibration — making `a_p` and `b_p` learnable — not a leaderboard number. The
  oracle 0.80 is not a target.
