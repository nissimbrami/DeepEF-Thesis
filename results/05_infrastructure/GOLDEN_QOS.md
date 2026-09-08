# Reaching the Keasar group's 8 "golden" rtx_6000 cards

Status: **THE ROUTE WORKS.** `--qos keasar` is accepted and escapes the 5-GPU public cap.
It was NOT usable at the moment of testing only because another group member was holding
all 8 cards. Nothing needs to be asked of Ofer.

Method: read-only inspection + one 3-minute probe job (submitted, observed, cancelled).
Probe was job 21080162, JobName `ZZPROBE_QOS_KEASAR_nissimb`, verified by name+user before
`scancel`. Nothing else was ever cancelled. Our 5 running / 90 pending jobs were untouched.

## The exact sbatch line that works

```bash
#SBATCH --partition=rtx6000
#SBATCH --qos=keasar
#SBATCH --account=keasar
#SBATCH --gres=gpu:rtx_6000:1
```

Submitting this was **accepted** (`Submitted batch job 21080162`) while 5 public jobs were
already running. `sbatch --test-only` scheduled it onto `cs-6000-03`.

Note the partition is **`rtx6000`**, not `gpu`. This is the whole trick, and it is the
opposite of what the original caveat assumed.

## 1. Why `--qos keasar` is rejected on `gpu` but accepted on `rtx6000`

The caveat was right about `gpu` and wrong about the cluster. The two partition families
gate QOS in **opposite directions**:

| Partition | Gate | Effect on `keasar` |
|---|---|---|
| `gpu` | `AllowQos=normal,vllm,bypass_limits,kaolevat` | **rejected** (allow-list, keasar absent) |
| `rtx6000` | `DenyQos=normal,eliyanac` | **accepted** (deny-list, keasar not denied) |

Same for `gtx1080`/`rtx2080`/`rtx3090`/`rtx4090`/`rtx_pro_6000` — all `DenyQos=normal`.
Those hardware partitions deny only `normal`; every named PI QOS passes. So `keasar` was
never meant for `gpu` at all — it is the key to the per-hardware partitions.

## 2. There is no Keasar node set. The 8 cards are a QOS *entitlement*.

No owned nodes, no features, no reservations. The formatted `sacctmgr` view hides it; the
raw `-P` dump shows the entitlement lives in **MaxTRESPA** (per-Account):

```
MaxTRESPA = gres/gpu:gtx_1080=0, gres/gpu:rtx_2080=0, gres/gpu:rtx_3090=0,
            gres/gpu:rtx_4090=0, gres/gpu:rtx_6000=8,
            gres/gpu:rtx_pro_6000=0, gres/gpu:tesla_p100=0, gres/gpu:titan_rtx=0
```

**8 rtx_6000 and every other GPU type set to 0.** That is precisely Ofer's "8 golden
rtx_6000 cards": 8 concurrent rtx_6000 GPUs for the *whole keasar account*, on any of the
133 rtx_6000 cards in the `rtx6000` partition. They are golden because nothing else in the
QOS is capped — no MaxTRESPU, no MaxJobs, no MaxWall.

Consequence to plan around: the 8 are shared **per account, not per user**.

## 3. Does it escape the 5-GPU cap? YES — proven by an existing job.

The public cap is not on the `normal` QOS (which has no limits). It is the **partition
QOS** of `gpu`:

- `gpu` → `QoS=gpu-part`, and `gpu-part` has `MaxTRESPU=gres/gpu=5`
- `rtx6000` → `QoS=N/A` — **no partition QOS, so no gres/gpu=5 cap at all**

That is exactly why we sit pinned at 5 running / 90 pending. Existence proof that the cap
does not apply on the keasar route, observed live:

```
user=bo.hassonof qos=keasar part=rtx6000 gres=gres/gpu:rtx_6000:8   (RUNNING 6h)
```

A single job holding **8** GPUs — impossible under `gres/gpu=5`. The escape is real.

## 4. Why the probe pended, and the one real constraint

The probe did not start. Reason code was decisive and is good news, not bad:

```
21080162  ZZPROBE_QOS_KEASAR_nissimb  rtx6000  keasar  PD  (MaxGRESPerAccount)
```

`MaxGRESPerAccount` = the account's 8 rtx_6000 were **fully consumed by
`bo.hassonof`'s 8-GPU `birder_training_franca` job**. Not a permission failure — the
opposite. It proves the entitlement is live and being enforced for us. Had we been
rejected, we would have seen `Invalid qos specification` at submit time; instead sbatch
accepted the job and the scheduler queued it against the group quota.

**So the 8 cards are currently 100% occupied by one other group member.** Capacity, not
access, is the blocker.

## 5. Preemption: we are the preemptor, never the victim

From `scontrol show config`: `PreemptType=preempt/qos`, `PreemptMode=REQUEUE`,
`PreemptExemptTime=00:00:00`.

- `keasar` has `Preempt=normal` → a keasar job **preempts** `normal` jobs.
- Scanning the Preempt column of all 42 QOS: the only row containing `keasar` is `keasar`
  itself. **No QOS lists `keasar` as preemptable.** Nothing can preempt us.
- `kaolevat` (Priority 10000) and `vllm` (999999) outrank us on priority, but priority is
  not preemption — neither declares `Preempt`, so neither can evict a keasar job.

`PreemptMode=requeue` on the keasar QOS describes what we do **to** victims, not to us.

This matters given train.py has no `--resume`: on this route preemption is not a risk we
need to engineer around. **But our own 5 `normal`-QOS jobs are preemptable** — including by
keasar jobs from our own account. Migrating work onto keasar removes that exposure.

## 6. Time limit

- `keasar` QOS `MaxWall`: **unset** (no limit).
- `rtx6000` partition `MaxTime`: **14-00:00:00**.

Effective ceiling **14 days** — vs. 7 days on `gpu`. An ~8-hour training fits with enormous
margin. Ask for what you need; there is no short-wall trap here.

## Bottom line

Nothing to send to Ofer — the access he asked about is configured and working. The reason
we are not using the golden cards is that **we never submitted to the `rtx6000` partition**;
all 95 of our jobs target `gpu`, where `keasar` is disallowed and `gpu-part` pins us at 5.

Practical caveats before moving real work:
1. The 8 are **per-account**. Right now another member holds all 8; we would queue behind
   him (`MaxGRESPerAccount`) rather than displace him.
2. `DenyOnLimit` is set: a request **exceeding** 8 rtx_6000 is rejected at submit, not
   queued. Keep concurrent keasar GPUs at or under 8.
3. Never select GPUs by partition name — the lua plugin rewrites requests. It confirmed
   `GPU Type Set, Using Requested GPU` only because `--gres=gpu:rtx_6000:1` was explicit.
4. `export WANDB_MODE=disabled` and a unique run tag, as always.

Verification note: `scripts/gate_g4_cpu.py` re-run after this investigation —
`dG=-0.0030 width=1092`, unchanged. No repo file was modified; this was read-only plus one
cancelled probe.
