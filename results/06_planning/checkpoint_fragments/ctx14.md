
---

# CHECKPOINT 14 — 2026-09-07 — THE SCORING BUG THAT HID 12 FINISHED CELLS

## The situation that made no sense

12 factorial cells had COMPLETED all 15 epochs, with 361 checkpoints on disk, and
`eval_results/abl_p3_a*.csv` was **empty**. `sacct` showed COMPLETED. The autopilot sat at
`S3: 0/48 cells scored` for days. It looked like nothing had progressed for 34 hours.

**The training had progressed. The SCORING was broken, in three independent ways.**

## Three bugs, each sufficient on its own

**1. `evaluate.py:58` had `DEVICE = 'cuda'` with the CPU fallback COMMENTED OUT.**
Scoring is a forward pass over 28 proteins; it never needed a GPU. Because it demanded one, every
eval had to queue for a card **behind 41 trainings in the same 5-GPU pool** — short jobs starved
behind long ones. Restored to `'cuda' if torch.cuda.is_available() else 'cpu'`.

**2. `train_utils.load_checkpoint` assumed a wrapper dict** and raised
`KeyError: 'model_state_dict'` on every per-epoch save, because `train.py` writes a **bare
state_dict**. Two formats exist in this tree and only one was handled. Now detects both, so the 361
existing checkpoints keep working.

**3. `evaluate.py`'s `torch.load` calls lacked `map_location`.** The preprocessed tensors were saved
on CUDA, so loading them on a CPU node raised. All 9 now pass `map_location=DEVICE`, a no-op on GPU.

## Why nobody noticed for days — the signature failure again

`run_calib_eval.sh` prints

    DONE: <tag> best epoch N scored on 28-test -> eval_results/abl_<tag>_eN.csv

**AFTER the crash, with no CSV on disk.** A success line that outlives the failure. This is the
project's signature mode, now seen a fifth time.

**The fix in `scripts/run_cpu_evals.sh`: verify the ARTIFACT with `ls`, never the message.**

## A fourth trap found while automating it

`run_calib_eval.sh` invokes **`python3`**, which resolves to the SYSTEM python with no pandas unless
the conda env's bin is FIRST on `PATH`. And `conda activate` **cannot be used** in a detached shell
(no `conda init`) — it fails with `CondaError: Run 'conda init' before 'conda activate'`. The runner
exports `PATH` directly and asserts `import pandas` before doing any work.

## Now running, and confirmed working

SLURM job **21085470** on the **`--partition=cpu`** partition, so it consumes **neither GPU pool**:

    python3 resolves to: /home/nissimb/.conda/envs/esm2_env_py38/bin/python3
    deps OK: pandas 2.0.3 torch 2.4.1+cu121
    === SCORING p3_a0_d0_s0_D1_uemb_seed42 ===
    Validation Epoch: 0: 3/28 [01:38<13:09, 31.58s/it]

~35 s per protein, ~16 min per cell, so **roughly 3 hours for all 12 — at zero GPU cost.**
Public lane stays 5/5; the 19 golden-lane jobs are untouched.

## The operational lesson

**Track the artifact, not the status.** `sacct | grep COMPLETED` said the work was done; only
`ls eval_results/abl_*.csv | wc -l` told the truth. And when a long-running job must survive an SSH
disconnect, submit it to SLURM — a `setsid nohup` on a login node kept dying silently, and its stale
log made it look like the new script was failing when the new script had never run.
