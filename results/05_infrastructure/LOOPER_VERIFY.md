# Looper verification — the toy case first, then the real one

## What the looper IS

`ScheduleWakeup` is a harness timer. It re-invokes **me** with a prompt after N seconds. It does
not run scripts and it does not depend on the cluster. **This is what the user asked for: a loop
that calls me, checks I finished, and calls me again for the next task.**

It is NOT `scripts/taskloop.py` — that was my own wrong build (a loop that ran scripts itself),
and it was cancelled (job 21140854).

## Why the old autopilot could never do this

Two defects, both measured, not assumed:

```
grep -c sbatch scripts/autopilot.py   ->  0
```

**It contains no submit call anywhere.** It watches and decides; it can never queue work.

And its state file shows:
```
"state": "S5_BUILD_INFO",  "cells_done": 18,  "cells_total": 16
```
`done > total` with no exit transition, so since 2026-09-07 it has re-decided the same D1
selection every 10 minutes forever. Alive for 24h, useful for none of it.

**Replacement:** `scripts/refill.sh`, which DOES call sbatch, is idempotent, and refuses to add
work when the lane already holds >= 16 jobs. Verified: run twice, second run added nothing.

## TOY CASE (the small test first, as asked)

A 60-second wakeup fired and re-invoked me with the loop prompt. Observed behaviour:
- the wakeup fired on schedule
- I received the prompt verbatim and continued the task list without a human message
- the next task (P0b) was executed in that invocation

**PASS** — the mechanism re-invokes me unattended.

## REAL CASE (the full loop)

Each firing must do all of: read TASKS.md, pick the next open task, execute it with real code,
verify the ARTIFACT with `ls`, run the gate if model code was touched, refill the golden lane,
mark `[x]`, commit and push, report in Hebrew.

Evidence it is working across firings:

| firing | task executed | artifact verified |
|---|---|---|
| 1 | P0b — canonical epoch claim corrected | `OPEN_PROBLEMS.md` line 31 re-read |
| 2 | P0c submitted; **caught that v1 wrote nothing despite printing DONE** | `ls` showed MISSING |
| 3 | wave-3 + wave-4 submitted, 10 arms | `scontrol` per job |

**The P0c catch is the proof that the verification step is real.** `run_calib_eval.sh` takes
`<train_log> <model_dir> <run_tag>`; v1 passed `<tag> <epoch>`, died with FileNotFoundError, and
still printed `P0C_DONE`. Only the `ls` artifact check exposed it. Resubmitted correctly as 21144483.

## Failure modes and what covers them

| failure | cover |
|---|---|
| wakeup never fires | every task is also a queued SLURM job; work continues without me |
| I claim done without doing | ARTIFACT check with `ls`; a DONE line is not evidence |
| duplicate submissions | `refill.sh` skips anything queued or already scored |
| model code corrupted | `gate_g4_cpu.py` must print ALL PASS at width 1092, else REVERT |
| a running job killed | no `scancel` exists in any script I wrote |
