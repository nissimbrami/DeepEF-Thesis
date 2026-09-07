
---

# CHECKPOINT 6 — 2026-09-07 — THE GPU CAP: CORRECTED, MEASURED, AND NOT BYPASSABLE

## The previous claim was wrong

CHECKPOINT 3 recorded: "8 rtx_6000 and ZERO of every other card (MaxTRESPA on account keasar)"
and suggested an admin could raise it with `sacctmgr modify account keasar set MaxTRESPA=...`.
**That is not where the limit lives.** Measured:

```
sacctmgr show qos where name=normal   -> no GPU limit, DenyOnLimit only
sacctmgr show assoc where user=nissimb -> account keasar, no MaxTRES at all
scontrol show partition gpu            -> QoS=gpu-part
sacctmgr show qos where name=gpu-part  -> MaxTRESPU = gres/gpu=5
```

**The real cap is `gres/gpu=5`, per USER, from the `gpu-part` QOS attached to the `gpu`
partition.** It is 5, not 8. We hold exactly 5 and are pinned at the ceiling; the other 94 jobs
sit on `Reason=QOSMaxGRESPerUser`. The correct escalation, if we ever ask for one, is to raise
MaxTRESPU on the **gpu-part QOS**, not MaxTRESPA on the keasar account.

## `main` looked like a second lane. It is not.

`main` allows qos=normal, contains 149 rtx_6000, and carries NO GPU cap, so it looked like a way
to run beyond 5. Tested with a 5-minute probe job (no training, no checkpoint, nothing at risk),
submitted with `--partition=main --gres=gpu:rtx_6000:1`:

```
sbatch: GPU Parameter Set ! Using GPU Partition.
...
SubmitLine=... --partition=main ...
Partition=gpu                       <- REWRITTEN by the lua job_submit plugin
Reason=QOSMaxGRESPerUser
```

**The lua submit filter rewrites any GPU request onto the `gpu` partition**, where the 5-GPU cap
applies. So the cap cannot be dodged by partition choice, and this also explains the standing site
rule "never select a GPU by partition name" -- the filter overrides you, and if you ask by
partition instead of by `--gres=gpu:rtx_6000:1` you can land on a GTX 1080 and OOM.

The probe was cancelled immediately (verified by JobName before scancel). Queue returned to
5 running / 99 total, untouched.

## What follows from this

- Throughput is FIXED at 5 concurrent GPU jobs. 48 factorial trainings + their chained evals at
  ~6 h each is therefore ~5-7 days of wall clock, and nothing we control shortens it.
- **Every GPU-arm we still owe (the dG arm, --slope_weight) must WAIT or DISPLACE factorial cells.**
  There is no spare capacity, so this is the user's call, not an optimisation to make quietly.
- All remaining CPU-only work should be finished while the GPU queue drains. That is free time.
