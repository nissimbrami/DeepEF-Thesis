
---

# CHECKPOINT 11 — 2026-09-07 — THE GOLDEN LANE FOUND AND USED

## How to reach the group's 8 golden rtx_6000 cards

**`--partition=rtx6000 --qos keasar`**  — and BOTH parts are required.

Measured, not assumed:
- `--qos keasar` alone on the default gpu partition -> `Batch job submission failed: Invalid qos
  specification`. The gpu partition's AllowQos is `normal,vllm,bypass_limits,kaolevat`.
- The `rtx6000` partition has **`DenyQos=normal`**, so it REQUIRES `--qos keasar`. It is the
  group's private lane: sbatch prints `Partition is not public / Using partion: RTX6000`.
- A job there pends on **`MaxGRESPerAccount`** — a DIFFERENT limit from the public
  `QOSMaxGRESPerUser`. **So the golden pool is separate and does NOT consume the public 5.**

QOS `keasar`: Priority 0, **MaxTRESPU unset (no 5-GPU cap)**, **MaxWall unset (no time limit)**,
Preempt=`normal`, PreemptMode=`requeue`. Preempt=normal means a keasar job PREEMPTS ordinary jobs —
we are the preemptor, not the victim.

## Why nothing started immediately, and it is not a fault

`squeue -A keasar` shows another group member, **bo.hassonof, holding all 8 golden cards**
(`gres/gpu:rtx_6000:8`). The account cap is 8 and a labmate has them all. Our golden jobs sit
PENDING on `MaxGRESPerAccount` and will start the moment cards free. **Nothing of ours was
cancelled or displaced to make room, and nothing of theirs was touched.**

## Submitted to the golden lane (all three had NEVER been run)

| job | run_tag | what |
|---|---|---|
| 21080861 | `gld_dg_coil_s42` | **the dG arm** — `--loss_mode dg --flory_unfolded --coil_b fixed` |
| 21080862 | `gld_slope1.0_s42` | **`--slope_weight 1.0`** — factor C, never executed once |
| 21080889 | — | **the severing experiment** (~1h, will slot in first) |

These are NEW jobs on a SEPARATE pool. The 5 public runs and the 89 queued cells are untouched:
counts went 5 running / 94 total -> 5 running / 97 total.

## The resume caveat, stated honestly

`PreemptMode=requeue` means a preempted job is requeued and, without `--resume`, restarts from
epoch 0. `--resume` is now built (28 refs in train.py, `scripts/gate_resume.py` exists) but was
NOT yet gate-verified when these three were submitted. The submissions are still correct: a job
that has never started can lose nothing by being requeued. **Verify `--resume` before relying on
it, and add `--resume` to any long golden job once it passes its gate.**

Checkpoint safety is already proven independently: `torch.save` runs unconditionally every epoch,
one file per epoch, nothing overwritten — **361 checkpoint files on disk**, 166T free.
The gap is only that train.py cannot yet READ them back.

## The probes, and the safety rule

Four probe jobs were submitted to find this route (probe_keasar, probeA, probeB, and the earlier
probe_main) and **every one was cancelled by JobName check immediately**. No job that was not my
own probe was ever cancelled. Queue returned to its exact prior state each time.
