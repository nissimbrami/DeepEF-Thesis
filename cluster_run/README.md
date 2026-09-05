# DeepEF — Cluster Execution Package

<!-- VERSION MARKER -->
**Package version: 2026-09-05.** Assembled locally in WSL (`~/workspace/DeepPEF`).
Built against `shaharec/main` commit `0ef6d3d` (the upstream read-only reference) with the
offset-attack machinery already ported onto the local run branch `offset-attack-run` at commit
`d0d8c4a`. The `--seed` patch (`code/add_seed.patch`) and the shipped scripts target that
offset-attack `train.py`. This is a **snapshot inside the repository working tree**: the four
code artifacts under `code/` and the six scripts under `scripts/` were produced against, and
depend on, the surrounding `Megascale-fineTuning/` tree (`preflight.py` imports
`pnas_train.build_energy_model`; the scripts call `Megascale-fineTuning/train.py` and
`run_calib_eval.sh`). Run the package from inside a checkout of that repo, not in isolation.

**If you are an agent starting with no context: read `docs/HANDOVER.md` first (it is written for
exactly that), then `docs/RUNBOOK.md` (again after every compaction), then `docs/ORIENTATION.md`
in full. Do not run anything before you have read them.**

---

## What this is

A self-contained brief for reproducing and then extending the DeepEF calibration experiments on
the Ben-Gurion University CS SLURM cluster.

| File | What it is |
|---|---|
| `docs/HANDOVER.md` | **The zero-context entry point for the cluster agent.** What to receive from Nissim, the two repos and the read-only rule, the first sixty minutes step-by-step, the phase gates, and what to report. **Read first if you are the cluster operator.** |
| `docs/RUNBOOK.md` | **The preflight checklist. Re-read at the start of every session and after every context compaction, before any command.** The run card, the currency check, the six numbers and their qualifiers, the predictions register, and the operational mistakes already paid for |
| `docs/STATE.md` | The measured state: results to date, decisions taken, and what is still missing before training can start |
| `docs/ORIENTATION.md` | The context. What the project is, the research goal, the settled facts with their sources, the hard rules, and the procedure with its gates. **Read first.** |
| `docs/COMPUTE_PLAN.md` | **Supersedes ORIENTATION §6.** How to use this cluster's parallel capacity: mini-batch 64 as a correctness fix, a screening factorial instead of a sequential ladder, statistical resolution, scheduling |
| `docs/BUILD.md` | What to construct: environment check, data location, preflight assertions, live monitor, calibration diagnostic, seed patch, and the run scripts |
| `docs/STATUS.md` | Written by the local agent: which artifacts were tested, how, and what must be verified here first |

The scripts and code artifacts are now **shipped** (`code/` and `scripts/`), tested to the extent
possible off-cluster — see `docs/STATUS.md` for the per-artifact honesty table. Exactly three
things remain unknowable from outside this cluster and are the only real TODOs: **partition/QOS**,
**conda environment name**, and **where the K50 data lives**. The scripts carry these as
`${VAR:-default}` overrides. Verify each artifact on the real cluster (`00_env_check.sh` first),
then proceed.

---

## The three things that matter most

**The goal.** DeepEF's remaining error is per-protein calibration: `pred ≈ a_p · true + b_p`.
Removing `a_p` and `b_p` with an oracle lifts every checkpoint ever trained to 0.77–0.81. The
research question is whether they can be made **learnable during training** rather than corrected
afterwards. Not "beat method X", not "reach 0.80".

**The reference.** `calib_ctrl`: pooled ΔΔG PCC **0.606**, per-protein **0.711**, produced by
`Megascale-fineTuning/train.py` on this cluster. **Not 0.655** — that number was selected on the
test set and retracted by its author.

**The rules.** `github.com/shaharec/DeepPEF` is read-only, always. No push to
`nissimbrami/DeepEF-Thesis` without per-action approval. Every number carries three qualifiers:
pooled or per-protein, ΔG or ΔΔG, which split and which selection.

---

## Start here

0. `docs/HANDOVER.md` — **the cluster operator starts here**: prerequisites, the repos, the first sixty minutes, the phase gates
1. `docs/RUNBOOK.md` — **read this early, and again after every compaction.** It is short and it is what stops you measuring the wrong thing
2. `docs/ORIENTATION.md` — read fully; save sections 2, 3 and 7 to memory
3. `docs/STATE.md` — the measured state and the decisions already taken
4. `docs/COMPUTE_PLAN.md` — the experiment design; it replaces ORIENTATION §6
5. `docs/STATUS.md` — what is already tested and what is not
6. `docs/BUILD.md` — build B1, then B2
7. Report what you found before building anything further

**One thing to internalise before you start:** on the laptop this was a sequential ladder because
each run cost ten hours. Here it is a factorial design, because the constraint is no longer time
— it is how many effects you can attribute. Do not run the levers one at a time.

---

## SAFETY — read before touching any remote

These are absolute and non-negotiable. They are repeated in `docs/ORIENTATION.md §3`.

- **`github.com/shaharec/DeepPEF` is READ-ONLY, always.** You may `git clone`, `git fetch`, and
  read commits and files. You may **never** push, commit, tag, force-push, or open a pull request
  against it — not even when you started your checkout from `shaharec/main`. `git fetch shaharec`
  is allowed; `git push shaharec` is banned. The difference matters.
- **No push to `github.com/nissimbrami/DeepEF-Thesis`** (Nissim's own fork) **without explicit,
  per-action approval.** Ask every time. One approval is not standing approval.
- **Never overwrite a checkpoint directory.** Every run gets a unique `--run_tag`. The eval step
  derives its model directory from the tag; reusing a tag silently evaluates the wrong model.
- **Never store a token/PAT** in files, git remotes, or memory. If one is pasted to you, tell the
  user to revoke it and use a local env var for the single command only.

## Operational discipline — mistakes already paid for

Read `docs/RUNBOOK.md §7` for the full list; the three that have actually cost time:

- **One persistent session.** Do the whole run inside a single `tmux`/`screen` session on the
  cluster. Load the environment once. Never a fresh shell per command.
- **Never write files across a filesystem boundary.** Writing from Windows into WSL (or the
  reverse) injects CRLF (`\r`) and breaks a script on its first line with `$'\r': command not
  found`. Every shipped file here is LF-only; keep it that way (`file scripts/*.sh` must say
  "ASCII text", never "with CRLF"). Inside `wsl -- bash -lc`, never use `$var`, `$(...)`, or
  `< <(...)` — they are mangled and once triggered a runaway `git clone`. Write a script with a
  quoted heredoc and run it by path instead.
- **Poll with `tail -f`, not long `sleep`.** Submit training with `sbatch`, capture the job ID,
  poll `squeue`. Never run a multi-hour training job in the foreground of your login shell.

## Supervision — required, not optional

- **Attach `code/monitor.py` to every training job.** A run that starts is not a run that should
  finish. When it prints `STOP: <reason>`, act on it — `scancel` the job and read the reason.
  The stop rules and the epoch-2 reproduction gate (below 0.55 at epoch 2 → stop) are in
  `docs/COMPUTE_PLAN.md` and `docs/BUILD.md §B5`; the rationale is in `docs/RUNBOOK.md §4`.
- **Epoch-2 check is reproduction, not selection.** Epoch selection stays on validation, always.

---

Contact for decisions: Nissim Brami.
