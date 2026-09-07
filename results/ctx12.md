
---

# CHECKPOINT 12 — 2026-09-07 — WHY THE FACTORIAL SHOWED 0/48, AND THE FIX

## The diagnosis: the evals were STARVED, not broken

`eval CSVs: 0/48` had nothing to do with a bug. Measured:

- **8+ trainings COMPLETED today** (p3_a1_d0_s0_D0_coil_seed42 08:16:14,
  p3_a0_d0_s0_D1_uemb_seed42 07:48:59, p3_slope0.3_s42 14:41:20, and more).
- Their chained evals show **`Dependency=(null)`** — the dependency was SATISFIED. They were not
  waiting on training at all.
- They were pending on **`QOSMaxGRESPerUser`**: queued BEHIND 41 other trainings in the public
  5-GPU lane. **Short jobs stuck behind long ones.**

**And the eval script works.** `run_calib_eval.sh` already produced
`eval_results/abl_p3_slope3.0_s42_e8.csv` and the 10 earlier CSVs. The only error in its log is a
missing `validation/diag_unfolded_offset.py` (a cosmetic diagnostic PLOT), and the CSV is still
written afterwards. **There was never an eval bug — only queue starvation.**

This is the concrete reason the autopilot has sat at `S3: 0/48 cells scored` for days: it is
waiting for CSVs that were physically unable to start.

## The fix: run the evals on the golden lane

Evals are short and independent, so they do not belong behind the training queue. Submitted 10 of
them (two batches of 5) with `--partition=rtx6000 --qos keasar`, which draws on the account's
separate `MaxGRESPerAccount` pool:

batch 1: p3_a0_d0_s0_D1_uemb_seed42, p3_a0_d0_s1_D1_uemb_seed42, p3_a1_d0_s0_D0_coil_seed42,
         p3_a1_d0_s1_D0_coil_seed42, p3_a1_d1_s0_D0_coil_seed42
batch 2: p3_a1_d1_s1_D0_coil_seed42, p3_a0_d1_s1_D0_coil_seed42, p3_a0_d1_s0_D1_uemb_seed42,
         p3_slope0.3_s42, p3_slope1.0_s42

Each is guarded: skipped if a CSV already exists, if it is already queued, or if its log/model dir
is missing. **The public queue was not touched: 5 running before and after.**

## The golden lane is capped by the ACCOUNT, and a labmate holds it

`squeue -A keasar` shows **bo.hassonof holding all 8 golden cards** (`gres/gpu:rtx_6000:8`, 6h18m
elapsed). `MaxGRESPerAccount` is a SHARED cap, so one labmate's job consumes the whole group
allowance. 14 cards sit free on the rtx6000 partition that we cannot use for that reason.

**Nothing of theirs was touched.** Our golden jobs pend and start as his free up.

## --resume is now VERIFIED

`scripts/gate_resume.py`: **42 passed, 0 failed**, including two-stage refusal, optimizer and
scheduler state, backward compatibility with the 361 existing bare-state_dict checkpoints, and the
scheduler-state fixup ordering. `gate_g4_cpu` still prints `dG=-0.0030 width=1092`.
So a requeued golden job can now resume instead of restarting from epoch 0.

## Standing lesson

**A queued job is not a running job, and a finished training is not a result.** Track
`ls eval_results/abl_*.csv | wc -l`, never `sacct | grep COMPLETED`. Two different resource pools
exist and short jobs should never be queued behind long ones in the same pool.
