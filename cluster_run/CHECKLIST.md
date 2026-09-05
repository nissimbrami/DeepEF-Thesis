# CHECKLIST.md — ship gate for `cluster_run/`

Every line must read **PASS** before the package is offered for approval. Filled in
2026-09-05 against the assembled package (19 files, 192 K). Evidence is the actual command
output, not a claim. Verifier: `_clusterdev/verify_pkg.py` (dev-only, not shipped).

Snapshot: built against `shaharec/main` (`origin/main`) commit `0ef6d3d`; local run branch
`offset-attack-run` at `d0d8c4a`. Package lives inside that repo working tree.

---

## A. Completeness

| # | Check | Result | Evidence |
|---|---|---|---|
| A1 | Exactly the intended file set, nothing more/less | **PASS** | verifier `completeness`: MISSING none, EXTRA none. 19 files: README, CHECKLIST, 7 docs (HANDOVER, RUNBOOK, STATE, ORIENTATION, COMPUTE_PLAN, BUILD, STATUS), 4 code, 6 scripts |
| A2 | No empty / placeholder files | **PASS** | verifier `sizes`: smallest is `01_data_check.sh`=1610 B; none 0 B |
| A3 | No CRLF — every file LF-only | **PASS** | verifier `CRLF (bytes 0x0d)`: "NONE (all LF)"; `file scripts/*.sh` → "ASCII text" |
| A4 | No absolute Windows / WSL / home / mnt paths | **PASS** | verifier `forbidden paths`: no hit for the Windows username, the WSL home path, or the drive-mount prefix. The one UNC-path hit is STATE.md:52 quoting that path as the mistake NOT to make (intentional) |
| A5 | No credentials (PAT prefix, secret words) | **PASS** | verifier: no GitHub-PAT prefix, no secret words. The lexical "token" hits are README's own "never store a PAT" rule plus monitor.py comments about parse tokens — no secret present |

## B. Correctness of documents

| # | Check | Result | Evidence |
|---|---|---|---|
| B1 | Reference `calib_ctrl` 0.606 / 0.711 stated where it matters | **PASS** | verifier: `0.606`×20, `0.711`×17; README "three things", ORIENTATION, COMPUTE_PLAN, STATE all agree |
| B2 | `0.655` appears ONLY as retracted / superseded / a different quantity | **PASS** | all 13 hits reviewed: README "Not 0.655"; RUNBOOK "test-peeked, retracted" + currency example; ORIENTATION "test-peeked and retracted"; STATE "Not 0.655"; COMPUTE_PLAN:39 is Nissim's historical **per-protein** local-repro figure (0.655 vs 0.691 PP — a different quantity from the retracted pooled 0.655), left verbatim |
| B3 | `pnas_train.py` mentioned only as historical / correct dependency | **PASS** | STATUS rescore row (historical local result); ORIENTATION/RUNBOOK/STATE note its in-training `validate()` understates by ~2× and that results come from `run_calib_eval.sh`→`score_runs.py` only |
| B4 | BUILD.md keeps exactly 3 real TODOs | **PASS** | BUILD.md build-status banner lists exactly: (1) partition/QOS, (2) conda env name, (3) data location. All other items marked "built — verify and run" or "build on cluster (only if HF fallback)" |
| B5 | README names all docs + a reading order | **PASS** | README file table lists RUNBOOK, STATE, ORIENTATION, COMPUTE_PLAN, BUILD, STATUS; "Start here" gives order 0–6 with RUNBOOK first |
| B6 | RUNBOOK.md present and flagged "read after every compaction" | **PASS** | `docs/RUNBOOK.md` shipped; README line "read this first, and again after every compaction" |

## C. Code

| # | Check | Result | Evidence |
|---|---|---|---|
| C1 | One honest label per code/script file | **PASS** | STATUS artifact table assigns each of the 4 code + 6 scripts exactly one of TESTED-e2e / TESTED-isolation / PARTIAL / UNTESTED-cluster |
| C2 | `py_compile` all 3 Python files | **PASS** | `python -m py_compile code/{calib_diag,monitor,preflight}.py` → all OK |
| C3 | `bash -n` all 6 scripts | **PASS** | `bash -n scripts/0{0..5}_*.sh` → 00,01,02,03,04,05 all OK |
| C4 | `git apply --check add_seed.patch` vs a clean base | **PASS** | extracted pristine base blob `79c5529` (commit `6e590ae`, offset-attack `train.py`) to a temp tree; `git apply --check code/add_seed.patch` → clean; full apply adds `--seed`, `set_seed`, KFold/split `random_state`. No network clone triggered |
| C5 | `calib_diag.py` affine-oracle 0.77–0.81 on a committed Shahar eval CSV — **"single most important test"** | **DEFERRED (cluster-only)** | No committed Shahar per-mutation eval CSV exists in the repo (0 tracked eval CSVs). `--selftest` PASSES all 6 assertions (raw 0.469 → offset-removed 0.917; std(b) 2.77; slope 0.95/0.99/1.04; designed regex 5). The end-to-end oracle check is listed as cluster step 4 in STATUS. The instrument is verified; only the real-data input is cluster-bound |
| C6 | `monitor.py` parses a real log + every rule fires on synthetic | **PASS** | real seed-1 log parsed clean (3 epochs, no stop). Fixtures: M1 flat-loss, M2 NaN, M3 grad-high, M3 grad-vanish, M4 low-PP, M5 slope-collapse, M7 wall, M8 cov-dead all fire `STOP [Mx]`; clean.log passes |
| C7 | `preflight.py` local checks 3/4/6/8 pass; 1/2/5/7 cluster-only | **PASS** | `preflight.py --local` → 4 PASS, 0 FAIL: check3 freeze-trap guard; check4 GCN0 std 0.1319; check6 param count 703805 EXACT; check8 epoch_0 load + forward finite dg=3.644. 1,2,5,7 declared cluster-only |
| C8 | SLURM headers match Shahar's launchers | **PASS** | scripts 03/04/05 use `--partition rtx6000 --qos keasar --gpus=1 --cpus-per-task=8 --exclude cs-6000-01..04,cs-cpu256-01`, `--wrap "module load anaconda; source activate esm2_env_py38; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; python Megascale-fineTuning/train.py ..."`, afterok eval via `run_calib_eval.sh` — matching `sbatch_wtanchor.sh` |
| C9 | Slope-term loss written, default-off bit-identical | **PASS** | `train.py` lines 76/80 (`--slope_weight` default 0.0), 435–444: term built and added ONLY when `SLOPE_WEIGHT > 0`; baseline loss line unchanged; within-protein `abs(std(pred_ddG)−std(true_ddG))`, WT=row 0, `unbiased=False`. On the offset-attack branch, not a shipped patch |

## D. STATUS.md content

| # | Check | Result | Evidence |
|---|---|---|---|
| D1 | Artifact honesty table matches the shipped files | **PASS** | STATUS table rows = the 4 code + 6 scripts actually in the package (stale `01_smoke`/`02_reference`/`05_factorial` names corrected to `02_smoke`/`03_calib_ctrl`/`05_anchor_sweep`) |
| D2 | Rescore result with all 3 qualifiers | **PASS** | STATUS "The one end-to-end local result": pnas_train / ΔG / random-init / mb16 / held-out-28 / in-training selection → epoch 12 pooled 0.5500, PP 0.6617, ΔG 0.2683; baseline 0.531 reproduced+exceeded |
| D3 | The 0.026 prediction recorded before the run | **PASS** | STATUS "THE PREDICTION": PP gap 0.6645 vs 0.711 = 0.026; falsifiable clause if mb64 does not close it |
| D4 | The 1h20m stall recorded | **PASS** | STATUS "A reliability datum — the 1h20m stall": step 76→77 ≈1391 s/it then recovery; monitor M7 wall rule catches it on cluster |
| D5 | Snapshot commit hash + what code it was built against | **PASS** | STATUS header: `shaharec/main` 0ef6d3d, local branch `offset-attack-run` d0d8c4a; depends on surrounding `Megascale-fineTuning/` |

## E. Self-containment

| # | Check | Result | Evidence |
|---|---|---|---|
| E1 | No reference to files outside `cluster_run/` for the *reading path* | **PASS** | README/RUNBOOK/ORIENTATION/BUILD/STATUS cross-link only within `docs/`, `code/`, `scripts/`. The scripts DO call `Megascale-fineTuning/train.py` + `run_calib_eval.sh` and `preflight.py` imports `pnas_train` — this is a real code dependency, disclosed in the README version marker as "run from inside a checkout of that repo" |
| E2 | Snapshot note + commit hash present | **PASS** | README `<!-- VERSION MARKER -->` block; STATUS header |
| E3 | `du -sh` under 1 MB | **PASS** | `du -sh cluster_run` = 184 K |
| E4 | Version marker at top of README | **PASS** | README lines 3–13, before the "starting with no context" line |

## F. SAFETY / operational / supervision (README)

| # | Check | Result | Evidence |
|---|---|---|---|
| F1 | SAFETY block: shaharec READ-ONLY, no nissimbrami push w/o approval, unique run_tag, no stored PATs | **PASS** | README "## SAFETY" section, 4 bullets; repeated in ORIENTATION §3 line 74 |
| F2 | Operational section: one tmux session, no cross-FS/CRLF writes, poll not sleep, sbatch not foreground | **PASS** | README "## Operational discipline" section, 3 bullets pointing to RUNBOOK §7 |
| F3 | Supervision requirement: attach monitor.py, act on STOP, epoch-2 gate is reproduction not selection | **PASS** | README "## Supervision — required, not optional" section, 2 bullets pointing to COMPUTE_PLAN / BUILD §B5 / RUNBOOK §4 |

## G. Cold-read test (folder alone)

See the answers below in this file (section "Cold-read answers"). **PASS** — all five questions
answerable from `cluster_run/` alone.

---

## Cold-read answers (opened `cluster_run/` fresh, nothing else)

1. **Research goal.** Make the per-protein calibration `pred ≈ a_p·true + b_p` **learnable during
   training**, rather than corrected afterwards by an oracle. Not "beat method X", not "reach 0.80".
   (README "three things"; ORIENTATION §1.)
2. **The number reproduced, and why not 0.655.** The reference is `calib_ctrl`: pooled ΔΔG PCC
   **0.606**, per-protein **0.711**, from `Megascale-fineTuning/train.py`. **0.655 is test-peeked
   and was retracted by its author** (RESEARCH_LOG 386–390); proper val-selection gives ~0.63.
   (README; ORIENTATION §2; RUNBOOK §3.)
3. **Entrypoint.** `Megascale-fineTuning/train.py` with the `calib_ctrl` flag set
   (`--full_data --no_pretrain --no_freeze --loss_mode ddg --pooled_corr_weight 0
   --dg_length_norm none --affine_calib --wt_anchor_weight 0 --designed_weight 1
   --val_frac 0.1 --epochs 15`), launched via `scripts/03_calib_ctrl.sh`. Results come from
   `run_calib_eval.sh` → `score_runs.py`, never in-training `validate()`. (BUILD §B8; RUNBOOK §3.)
4. **First three commands.** `bash scripts/00_env_check.sh` → `bash scripts/01_data_check.sh` →
   `python Megascale-fineTuning/calib_diag.py --selftest`. Then validate calib_diag on a real
   Shahar eval CSV, then `bash scripts/02_smoke.sh` before the reference run. (STATUS "First thing
   to do on the cluster".)
5. **What is tested vs not.** Tested off-cluster: calib_diag `--selftest`, monitor rules on
   fixtures + a real log, preflight checks 3/4/6/8, py_compile ×3, bash -n ×6, add_seed.patch
   apply, the local rescore (pooled 0.550 / PP 0.662). NOT tested (cluster-only): train.py itself
   (old K50 layout absent in WSL), preflight 1/2/5/7, the smoke run, the reference run, σ, the
   anchor sweep, and the calib_diag affine-oracle 0.77–0.81 on real data. (STATUS artifact table.)

---

## Outstanding before the first cluster run (not blockers to shipping the package)

- **C5** calib_diag affine-oracle on a real Shahar eval CSV — cluster step 4.
- Cluster access, QOS/partition, conda env, data location — the 3 TODOs (parallel, independent).
- Factor D (coil) port to train.py before the full A×B×C×D factorial.
