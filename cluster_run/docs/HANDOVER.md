# HANDOVER.md — everything the cluster agent needs, from zero

**You are starting with no context. Your job is execution, not planning.** The research design is
already decided and written down. If you find yourself designing an experiment, stop — you have
misread something.

Read in this order: this file → `RUNBOOK.md` → `ORIENTATION.md` → `COMPUTE_PLAN.md` → `STATE.md`
→ `STATUS.md` → `BUILD.md`. Then run Part C below.

---

## Part A — what you must receive from Nissim before you can start

**None of these are in this repository, and none of them should ever be.** Ask for them, use
them, do not write them down anywhere.

| # | Item | Why | Received? |
|---|---|---|---|
| A1 | Cluster login hostname and your username | To reach the cluster at all | ☐ |
| A2 | SSH access — key or password, and any VPN/2FA requirement | Login | ☐ |
| A3 | Confirmation that your account holds the **`keasar` QOS** | Shahar's launchers require it; without it your jobs are rejected | ☐ |
| A4 | Storage location and quota for your work | 75 GB of tensors plus checkpoints; the home quota is usually too small | ☐ |
| A5 | GitHub access for `nissimbrami/DeepEF-Thesis` — read is enough | To clone the package | ☐ |
| A6 | **Where Shahar's processed K50 tensors live on this cluster**, if known | Decides whether Phase 0 is ten minutes or a day | ☐ |
| A7 | Whether a Weights & Biases account/key is available, or run with `WANDB_MODE=disabled` | Scripts log to wandb by default | ☐ |

**A6 is the highest-value question.** `calib_ctrl` was produced on this cluster, so those tensors
existed here under Shahar's account. If Nissim or Chen can point at the path, you skip an
83 GB download and a data-conversion step entirely. **Ask before you search.**

### Credentials — the rule, without exception

- **Never** put a token, key, password, or PAT into a file, a git remote, a commit, a log, or
  your own memory.
- If one is pasted to you, use it for the single command that needs it, then tell the user to
  **revoke it**.
- For git over HTTPS, prefer a read-only clone. If a token is unavoidable, pass it as an
  environment variable for one command and never write it to `.git/config`.
- After any clone that used a credential, run `git remote set-url origin <clean-url>` and confirm
  with `git remote -v` that no secret is embedded.

---

## Part B — the repositories, and the one rule that matters most

| Repository | Access | What it is |
|---|---|---|
| `github.com/nissimbrami/DeepEF-Thesis` | clone; **push only with per-action approval** | Nissim's fork. Contains `cluster_run/` — this package |
| `github.com/shaharec/DeepPEF` | **READ-ONLY, absolutely** | Shahar's upstream. `git fetch shaharec` is allowed. `git push shaharec` is banned, always, including tags and pull requests |

The package was built against `shaharec/main` at commit `0ef6d3d`, with the offset-attack
machinery on the local branch `offset-attack-run` at `d0d8c4a`. The version marker at the top of
`README.md` is authoritative; if it disagrees with this file, trust the marker.

---

## Part C — the first sixty minutes, in order

Do these literally. Do not reorder. Report after each numbered step.

**C1. Get on the cluster and confirm the basics.**
```
sinfo -o "%P %a %l %D %t %N"      # which partitions exist and are up
sacctmgr show assoc where user=$USER format=account,qos   # is keasar there?
```

**C2. Clone and check out.**
```
git clone https://github.com/nissimbrami/DeepEF-Thesis.git DeepPEF
cd DeepPEF
git remote add shaharec https://github.com/shaharec/DeepPEF.git
git fetch shaharec
git checkout -b cluster-run origin/offset-attack-run   # or the branch Nissim names
grep -c "wt_anchor_weight\|no_freeze\|val_frac\|designed_weight\|slope_weight" \
     Megascale-fineTuning/train.py
```
The grep must return **5 or more**. If it returns 0 you are on the wrong branch — stop and ask.

**C3. Start one persistent session and stay in it.**
```
tmux new -s deepef
module load anaconda
source activate esm2_env_py38
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Never open a fresh shell per command for the rest of this work.

**C4. Run the environment check.**
```
bash cluster_run/scripts/00_env_check.sh
```
Every line must read `PASS`. Report the output verbatim.

**C5. Find the data.**
```
bash cluster_run/scripts/01_data_check.sh
```
Must end `DATA: OK` with **368 protein directories** and **28 proteins in `mega_test.csv`**.
If it fails, report what it found before trying anything else. Only fall back to the Hugging Face
dataset (`nissimb/deepef-megascale`, ~83 GB, different layout) if Shahar's tensors truly are not
reachable — and then follow `BUILD.md` B3, including its six-point equivalence gate.

**C6. Validate the measurement instrument. This is the single most important step in the package.**

Find a committed evaluation CSV in Shahar's tree — under `eval_results/` or referenced by
`analysis/wt_dg_error/` — with columns `protein, deltaG, pred_deltaG, ddG, pred_ddG`. Then:
```
python cluster_run/code/calib_diag.py <that_csv>
```
**The offset-removed pooled PCC must land in 0.77–0.81.** That range holds across all 52
checkpoints ever evaluated in this project.

`calib_diag.py` is not a reporting utility. `std(b)` and the slope distribution are **the objects
this thesis studies**; PCC is secondary. **If this check misses that range, the instrument is
wrong and every number produced afterwards is meaningless. Stop and report.**

---

## Part D — the phases, with their gates

Full detail in `COMPUTE_PLAN.md`. This is the summary you execute against.

| Phase | Command | Gate — do not proceed unless met |
|---|---|---|
| **Smoke** | `scripts/02_smoke.sh` | All 8 preflight assertions `PASS`; one epoch completes |
| **1 — reference** | `scripts/03_calib_ctrl.sh` | pooled ≈ **0.606**, PP ≈ **0.711**. Epoch-2 reproduction check **≥ 0.55** or stop early |
| **2 — σ** | `scripts/04_seeds.sh` | Five seeds complete; report **mean ± σ** for pooled and PP |
| **3 — pilot** | slope-weight pilot, 3 runs, 1 seed | A weight chosen that moves `std(pred)/std(true)` toward 1 without hurting pooled |
| **3 — factorial** | anchor × designed × slope × coil, 2⁴ cells | All cells complete; main effects **and** the six two-way interactions reported |
| **4 — refinement** | `scripts/05_anchor_sweep.sh` and follow-ups | The trade-off curve: `std(b)`, slope distribution and PP versus anchor weight |

**Phase 1 must reproduce before anything else runs.** A modified reproduction is not a
reproduction: do not change the mini-batch, the split, or the epochs for that run.

**σ from Phase 2 decides Phase 3's seed count.** σ ≤ 0.01 → 3 seeds per cell; σ ≥ 0.02 → 5.

**Submit in waves of about 8 jobs**, verify the first wave's numbers are sane, then continue. Do
not queue 48 jobs at once.

**Keep all seeds of one factorial cell on the same GPU type**, and record the type with every
result. Prefer `cs-4090-*`; use `cs-6000-*` or `ee-l40s-*` for memory-heavy arms; avoid
`cs-1080-*` and `cs-2080-*` for training.

---

## Part E — what to report, and how

**Before every run over an hour**, paste the completed six-line run card from `RUNBOOK.md` §1
into the log and the report. **A report without a run card is not accepted.** Its fifth line —
*is the comparison number from the same entrypoint, objective, split and mini-batch?* — has
already caught two wasted runs in this project.

**After every run**, write `results/<phase>_<run_tag>.md` containing:

- the exact command line
- SLURM job ID, node, GPU type, wall-clock
- the metric line with all three qualifiers:
  `<entrypoint>, <objective> objective, <init>, mini_batch <n>, <split>, <selection>: pooled ΔΔG PCC = X, PP = Y, absolute ΔG PCC = Z`
- `std(b)` and the slope distribution from `calib_diag.py`
- the designed-fold breakdown
- anything that surprised you

**All numbers come from `run_calib_eval.sh` → `score_runs.py`.** Never from the in-training
`validate()` — it subtracts one global wild-type reference for every protein and is not a ΔΔG at
all. That mistake already cost a day.

**Attach `code/monitor.py` to every training job.** When it prints `STOP:`, act on it.

---

## Part F — when to stop and ask, rather than decide

Stop and report if:

- Any gate in Part D is missed
- `calib_diag.py` fails the 0.77–0.81 check
- The data check does not find 368 proteins and 28 test proteins
- You are about to push anything anywhere
- You are about to change the reference configuration for any reason
- A number disagrees with `STATE.md` by more than σ
- You are about to design an experiment rather than run one

Otherwise: **investigate, decide, and state the decision with its evidence.** You have the
repository and the cluster. Do not present menus for questions the code can answer.

---

## Part G — package completeness, verify on arrival

Confirm the package you received is intact before trusting it.

```
cluster_run/
├── README.md              (version marker at top; SAFETY, operational, supervision sections)
├── CHECKLIST.md           (the build gate, all lines PASS)
├── docs/
│   ├── HANDOVER.md        (this file)
│   ├── RUNBOOK.md         (read after every compaction)
│   ├── ORIENTATION.md     (the context; §2 settled facts, §3 rules)
│   ├── COMPUTE_PLAN.md    (experiment design; supersedes ORIENTATION §6)
│   ├── STATE.md           (measured state, decisions, predictions)
│   ├── STATUS.md          (what was tested, how, what is untestable off-cluster)
│   └── BUILD.md           (what to construct; only 3 real TODOs remain)
├── code/
│   ├── calib_diag.py      (the primary instrument)
│   ├── monitor.py         (stop rules)
│   ├── preflight.py       (8 pre-run assertions)
│   └── add_seed.patch     (--seed flag)
└── scripts/
    ├── 00_env_check.sh  01_data_check.sh  02_smoke.sh
    ├── 03_calib_ctrl.sh  04_seeds.sh  05_anchor_sweep.sh
```

- [ ] Every file above is present and non-empty
- [ ] `file cluster_run/scripts/*.sh` says "ASCII text", never "with CRLF line terminators"
- [ ] `bash -n` passes on all six scripts
- [ ] `python -m py_compile` passes on all three Python files
- [ ] `grep -rn "0.655"` appears only where it is described as retracted
- [ ] `grep -rniE "ghp_|token|password"` returns nothing
- [ ] `du -sh cluster_run/` is well under a megabyte — no data, checkpoints, or logs
- [ ] The version marker in `README.md` names a commit you can `git show`

**The cold-read test.** With only `cluster_run/` open, can you answer: what the research goal is ·
which number you are reproducing and why not 0.655 · which entrypoint · what the first three
commands are · what is tested and what is not? **If any answer needs outside knowledge, say so
before starting rather than compensating for it.**

---

## Part H — the one-paragraph summary, if you read nothing else

DeepEF is a neural free-energy function for proteins. Its remaining error is **per-protein
calibration**: `pred ≈ a_p · true + b_p`. Removing `a_p` and `b_p` with an oracle lifts every
checkpoint to 0.77–0.81, so the ranking information is there and the loss is calibration. **The
research question is whether `a_p` and `b_p` can be made learnable during training.** Your job is
to reproduce the reference (`calib_ctrl`, pooled 0.606 / per-protein 0.711), measure the seed
noise σ, and then run a factorial over four levers — WT anchor, designed-fold reweight, slope
term, and the Flory unfolded reference state — reporting offset spread and slope distribution
alongside every correlation. **Not a leaderboard number. The oracle 0.80 is a ceiling, never a
target.**
