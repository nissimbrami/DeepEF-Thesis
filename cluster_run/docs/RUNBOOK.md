# RUNBOOK.md — how to run without measuring the wrong thing

**Re-read this at the start of every session and after every context compaction, before any
command. It is short on purpose.**

This file exists because the same failure has now occurred twice: a long run was executed
faithfully against a plan whose premise had gone stale, and the result was unusable. Both times
the run itself was clean. Both times a thirty-second check beforehand would have caught it.

- **2026-09-04.** Ran `pnas_train.py`, ΔG objective, mini-batch 16, targeting 0.655 — a number
  that had been retracted as test-peeked. Seven hours. The run was fine; the target and the
  entrypoint were wrong.
- **2026-09-05.** Ran a five-seed sweep to measure σ, using the ΔG arm on `pnas_train.py`. σ is
  needed for `calib_ctrl` — `train.py`, ΔΔG objective, `--val_frac 0.1`, mini-batch 64. **σ does
  not transfer between configurations.** Forty GPU-hours for a number that would never be used.

The pattern is identical: correct execution of a stale premise. The countermeasure is below.

---

## 1. The run card — mandatory before any job over one hour

Write this out. Six lines. Do not skip it because the answers feel obvious; both failures above
felt obvious.

```
RUN CARD
  Question:     what am I trying to learn from this run?
  Entrypoint:   which script, which flags
  Metric:       pooled or per-protein · ΔG or ΔΔG · which split · which selection
  Compared to:  which number, and where does it come from (file:line, date)
  Same config?  is the comparison number from the SAME entrypoint, objective,
                split and mini-batch? If not — STOP.
  Cost:         expected wall-clock
```

**The fifth line is the one that matters.** Both failures die there.

- 09-04: comparison number 0.655 came from `train.py` + ΔΔG + freeze. The run was
  `pnas_train.py` + ΔG. **Not the same config → stop.**
- 09-05: σ was wanted for `calib_ctrl` (`train.py` + ΔΔG + val split + mb 64). The run was
  `pnas_train.py` + ΔG + mb 16. **Not the same config → stop.**

Paste the completed card into the run log and into the report.

---

## 2. Currency check — is the premise still true?

For the "Compared to" line, find the newest artifact in the repo that touches that quantity.
**If it is newer than your source, your premise is stale.**

Precedence, newest wins:

```
RESEARCH_LOG.md  >  sbatch_*.sh launchers  >  code comments  >  the thesis
```

The thesis is last not because it is unreliable but because it is the oldest. It is authoritative
for *definitions*; the log and launchers are authoritative for *current numbers*.

Worked example: `score_runs.py:22` contains the string `0.655`. It is real. It is also superseded
by `RESEARCH_LOG.md:386-390`, which retracts it, and by `sbatch_wtanchor.sh:7`, which names
`calib_ctrl` 0.606/0.711 as the reference. **Verifying that a claim exists in the code is not the
same as verifying that it is current.**

---

## 3. The four numbers, and why they are never comparable

| Number | Quantity | Split | Note |
|---|---|---|---|
| 0.82 | ΔG | **validation** | thesis Table 3.3 |
| 0.51 | ΔG | held-out | thesis Table 3.6 |
| 0.531 | ΔΔG pooled | held-out | thesis baseline, ΔG objective |
| 0.655 | ΔΔG pooled | held-out | **test-peeked, retracted** |
| **0.606 / 0.711** | **ΔΔG pooled / per-protein** | **held-out, val-selected** | **the reference** |
| 0.550 / 0.662 | ΔΔG pooled / PP | held-out | our local reproduction, ΔG objective, mb 16, epoch 12 |

Most of the confusion in this project has come from comparing two rows of this table. **Before
comparing any two numbers, check that all three qualifiers match.**

And: the in-training `validate()` in `pnas_train.py` subtracts `val_dg[0]` — one global wild-type
reference for every protein. **It is not a ΔΔG.** It understated the real signal by roughly 2×
and cost a day of misinterpretation. Results come only from `run_calib_eval.sh` →
`score_runs.py`.

---

## 4. During a run

- **Attach `monitor.py` to every training job.** A run that starts is not a run that should
  finish. Act on `STOP:`.
- **Epoch-2 reproduction check.** Once `epoch_2.pt` exists, evaluate it once against the held-out
  set. Shahar's honest runs sit near 0.64 there. Below 0.55, stop — the run will not recover, and
  finding out at epoch 2 costs one hour instead of ten.
- **This is reproduction, not selection.** Epoch selection stays on validation, always.
- **Watch for stalls.** The laptop showed a 1h20m hang mid-epoch on 09-05. If wall-clock per step
  exceeds twice the running median, investigate before assuming progress.

---

## 5. After a run

Report with all three qualifiers, or it is not a result:

```
<entrypoint>, <objective> objective, <init>, mini_batch <n>, <split>, <selection>:
pooled ΔΔG PCC = X, PP = Y, absolute ΔG PCC = Z
```

Then run `calib_diag.py` and report `std(b)` and the slope distribution. **Those are the objects
of study; PCC is secondary.** A run that improves pooled while collapsing slope is not a win, and
without both numbers that failure is invisible.

---

## 6. Predictions register — write them before, check them after

Recording a prediction before the run is what separates a finding from a rationalisation.

| Date | Prediction | Checked | Outcome |
|---|---|---|---|
| 09-05 | The PP gap to `calib_ctrl` is only 0.026 (0.6645 vs 0.711) — smaller than mini-batch 16 vs 64 would predict. **If the cluster run at 64 does not close it, mini-batch was not the explanation.** | pending | |
| 09-05 | The ΔΔG objective is worth about +0.05 (0.550 → 0.606), not the +0.12 implied by the retracted 0.655. **Expect the cluster reference near 0.60 and treat that as success.** | pending | |
| — | After the coil: `corr(E_u, GAT-only)` falls, `corr(E_u, GCN-only)` rises, `var(E_u)` shrinks. | pending | |

Add a row before every lever run. Fill in the outcome afterwards, including when it is wrong.

---

## 7. Working rules

- **One persistent session.** Never a new shell per command. Enter once, load the environment,
  work inside `tmux`.
- **Never write files across a filesystem boundary.** Writing from Windows into WSL injected CRLF
  and broke a script on its first line. Use a heredoc from inside the target system.
- **Poll with `tail -f` or short intervals**, not long `sleep`. A session slept 16 minutes waiting
  on 12 minutes of work.
- **Training via `sbatch`, never foreground.** Submit, capture the job ID, poll `squeue`.
- **Close every loop.** A diagnostic that exits non-zero gets fixed and re-run before you move on.
  The 09-04 rescore was abandoned at `exit 1` and the run looked like a failure for a day.
- **Investigate and decide; do not present menus.** You have the repository. When you hit a fork,
  read the code, decide, and state the decision with its evidence.

---

## 8. Absolute rules

- `github.com/shaharec/DeepPEF` is **READ-ONLY**. Clone, fetch and read only. Never push, commit,
  tag, or open a pull request there.
- No push to `github.com/nissimbrami/DeepEF-Thesis` without explicit per-action approval.
- Every run gets a unique `--run_tag`. Never overwrite a checkpoint directory.
- Selection on validation, never on test.
- The goal is calibration — making `a_p` and `b_p` learnable — not a leaderboard number. The
  oracle 0.80 is a ceiling, not a target, and must never be quoted as one.
