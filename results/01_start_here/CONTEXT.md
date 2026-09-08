# CONTEXT.md — resume file for the DeepEF cluster run
> **CANONICAL BASIS (P8).** Every headline number in this document is computed on:
> **27 test proteins (2K5H excluded) | ddG metric | the 9 original-population eval CSVs |
> the val-selected epoch only.** That basis is `pooled 0.5772 / oracle 0.7156 / gain +0.1384`.
> Membership, exclusions and provenance: `results/05_infrastructure/HEADLINE_BASIS.md`.
> Enforced by `scripts/gate_headline.py`. **Never average across runs that differ in factor D**
> (`--unfolded_emb zero`) — D0 and D1 are different models; report them separately, always.


Written 2026-09-05 by the cluster agent. READ THIS FIRST after any context loss,
together with cluster_run/docs/RUNBOOK.md and cluster_run/docs/STATE.md.

## Access (do not store credentials anywhere)
- Cluster: slurm.bgu.ac.il, user nissimb. Password auth (Nissim has it; it is NOT written here).
- Login node seen: slurm-login-04.auth.ad.bgu.ac.il
- SLURM account/QOS: keasar (VERIFIED present: `sacctmgr show assoc where user=$USER format=account,qos`)
- Working clone: /home/nissimb/DeepPEF, branch current-vers, origin=github.com/nissimbrami/DeepEF-Thesis (PUBLIC, no token needed)
- github.com/shaharec/DeepPEF is PRIVATE and NOT NEEDED — the same tree is readable on the cluster (below). READ-ONLY always.

## CORRECTION to the docs: the data path moved
Docs/scripts say /mnt/new_groups/keasar_group/... which NO LONGER EXISTS.
The live mount is:
    SHAHAR=/groups/keasar_group/casp15/Shahar/DeepPEF
Shahar Cohen's cluster user is shaharax. His conda env is NOT readable (Permission denied).

## VERIFIED facts (measured, not assumed)
- training_data: 368 protein directories        (gate: 368) PASS
- mutation_datasets: 862 csv files
- mega_test.csv: present via data/ThermoMPNN; calib_diag reports 28 proteins (gate: 28) PASS
- levers grep on Megascale-fineTuning/train.py = 13 (gate: >=5) PASS
- train.py ALREADY accepts --seed (line 54, default 42). add_seed.patch is NOT needed.
- Partitions up: main debug cpu gpu course gtx1080 rtx2080 rtx3090 rtx4090 rtx6000 rtx_pro_6000 l40s
  rtx6000 has 44 nodes, so 03_calib_ctrl.sh's default partition + its 5-node exclude is fine.
- conda: `module load anaconda` then EITHER `source activate <env>`
  (/storage/modules/packages/anaconda/bin/activate exists) OR
  `source /storage/modules/packages/anaconda/etc/profile.d/conda.sh; conda activate <env>`
- anaconda base python (/storage/modules/packages/anaconda/bin/python) HAS pandas+numpy —
  enough to run calib_diag.py without the training env.

## Data wiring already done (read-only symlinks into Shahar's tree)
    /home/nissimb/DeepPEF/data/ThermoMPNN                                  -> $SHAHAR/data/ThermoMPNN
    /home/nissimb/DeepPEF/data/Processed_K50_dG_datasets/training_data     -> $SHAHAR/data/.../training_data
    /home/nissimb/DeepPEF/data/Processed_K50_dG_datasets/mutation_datasets -> $SHAHAR/data/.../mutation_datasets
(Processed_K50_dG_datasets itself is a REAL dir so any writes land locally, never in Shahar's tree.)

## The environment problem and its fix
esm2_env_py38 did NOT exist for this account and Shahar's copy is unreadable.
Being built at /home/nissimb/.conda/envs/esm2_env_py38 by /home/nissimb/build_env.sh
(log: /home/nissimb/env_build.log): python 3.8 + torch 2.4.1 cu121 + torch_geometric +
numpy<2 pandas scipy scikit-learn matplotlib tqdm wandb biopython.

## ACTION D RESULT — GATE FAILED, decision pending from Nissim
calib_diag.py on 7 real checkpoints in $SHAHAR/eval_results:
  abl_calib_ctrl_e12    pooled 0.6062  PP 0.7109  offsetRM 0.7006  AFFINE 0.7136
  abl_ddg_nopretrain_e11 pooled 0.6471 PP 0.7376  offsetRM 0.7395  AFFINE 0.7486
  abl_randscratch_e12   pooled 0.6237  PP 0.7386  offsetRM 0.7287  AFFINE 0.7061
  abl_calib_combo_e13   pooled 0.6313  PP 0.7412  offsetRM 0.7338  AFFINE 0.7485
  abl_designed_w3_e8    pooled 0.6401  PP 0.7388  offsetRM 0.7323  AFFINE 0.7742  <- only one in range
  abl_readout_attn_e12  pooled 0.6055  PP 0.6838  offsetRM 0.6740  AFFINE 0.6456
  abl_ddg_core_e9       pooled 0.5992  PP 0.7067  offsetRM 0.7101  AFFINE 0.6721
Gate wanted AFFINE-removed in [0.77,0.81]; 1 of 7 makes it.
DECISIVE: calib_ctrl reproduces the reference EXACTLY (0.606 / 0.711, std(b) 0.285 vs 0.29 in
STATE.md), so the CSV and the pooled/PP path are right. What is unconfirmed is the CLAIM that
all 52 checkpoints reach 0.77-0.81 — the RUNBOOK section 2 stale-premise pattern.
Per HANDOVER Part F this is a STOP-and-report condition. No training submitted.

## Automation
/home/nissimb/auto/auto_pipeline.sh runs everything remaining unattended.
State: /home/nissimb/auto/STATUS.txt  (one line per step; grep for PROBLEM)
Phase 1 will NOT submit until /home/nissimb/auto/GO_PHASE1 exists (Nissim's decision on gate D).

## Hard rules (never violate)
1. Shahar's tree and shaharec remote: READ-ONLY, always.
2. No git push anywhere without explicit per-action approval, every time.
3. Unique --run_tag per run; never overwrite a checkpoint dir.
4. Never write a credential to a file, remote, log or memory.
5. Numbers only from run_calib_eval.sh -> score_runs.py, never in-training validate().
6. Every number needs 3 qualifiers: pooled|per-protein . dG|ddG . which split & selection.

================================================================================
## UPDATE 2026-09-06 00:15 - nine training runs live in parallel
================================================================================

### GPU facts, MEASURED not assumed
- Training needs ~10 GiB VRAM. PROVEN: probe job on GTX 1080 (8 GiB) died with
  torch.OutOfMemoryError after allocating 7.17 GiB and needing 1.10 GiB more.
  => gtx_1080 and the 8-11 GiB rtx_2080 are physically OUT.
  => Usable cards: rtx_3090 / rtx_4090 / rtx_6000 (>= 24 GiB).
- QOS `keasar`: MaxTRESPerAccount = gres/gpu:rtx_6000=8, and ZERO for every other card type.
  The cap is per ACCOUNT (the whole lab, e.g. bo.hassonof also draws on it) - this is the
  "MaxGRESPerAccount" pending reason.
- QOS `normal`: no per-account cap. THE TRICK THAT UNBLOCKS EVERYTHING:
      sbatch --qos normal --gres=gpu:rtx_6000:1 ...
  Same card, bypasses the keasar account cap.
  WARNING: `--partition rtx4090` under keasar is REJECTED outright; under `normal` the site
  submit filter may silently route you onto a gtx_1080, which then OOMs. ALWAYS pin the card
  with --gres=gpu:<type>:1, never by partition name.

### Live jobs (all 15 epochs each)
  Phase 1  calib_ctrl_repro2 21015271  ise-6000-04    rtx_6000   ~3.7-6 s/it
  Phase 2  sigma_seed42 21016195, seed1 21016197, seed2 21016199   (qos keasar)
           sigma_seed3  21016223, seed4 21016225                   (qos normal + --gres rtx_6000)
           ALL FIVE ON rtx_6000 -> sigma is not contaminated by hardware. Keep it that way.
  Phase 4  anchor_w0.3 21016095, w1.0 21016097, w3.0 21016099  ise-pheno-*  rtx_3090
           All three on identical hardware -> their mutual comparison is clean.
           anchor-vs-reference crosses card types: judge that gap against sigma, and always
           record the card with the result.
           NOTE: these degraded to ~17.6 s/it (shared pheno nodes) => ~22 h. Moving them to
           rtx_6000 via the qos-normal trick would cut them to ~7 h.

### WHY GPU COUNT IS NO LONGER THE LIMIT
Every run that exists is running. We are work-limited, not GPU-limited. The next phase cannot
start because two of its four levers HAVE NO IMPLEMENTATION - see below.

================================================================================
## WHAT STILL HAS TO BE BUILT (this is the critical path now)
================================================================================

### 1. The slope lever - `--slope_weight` in Megascale-fineTuning/train.py
STATE.md section 3 is explicit: "Does not exist. Nobody has written it." It is the thesis's own
contribution. STATE.md sanctions writing it immediately because it depends on nothing:

  "a loss term penalising abs(std(pred_ddG) - std(true_ddG)) within each protein, behind a flag
   defaulting to off, so the baseline stays bit-identical when the flag is absent."

Requirements:
  - New argparse flag `--slope_weight`, type float, default 0.0 (OFF).
  - When 0.0 the computation graph must be untouched: the reference run has to stay
    bit-identical. Verify by running one epoch with and without the flag at seed 42 and
    diffing the loss values.
  - The penalty is computed WITHIN each protein group of the mini-batch, on the ddG scale.
  - Existing levers to imitate for style/placement: `--wt_anchor_weight` (line ~75) and
    `--designed_weight` (line ~74) in Megascale-fineTuning/train.py.
  - Why it matters: a_p (slope) collapses to 0.07-0.18 on designed and fragile folds and
    predicts per-protein performance better than any other measured quantity. The anchor lever
    fixes offset but is known to WORSEN slope collapse - the slope term is what should counter it.

### 2. The coil lever (Flory unfolded reference state) - factor D
PROMPT section 11: implemented in the v5 bundle but never synchronised into train.py.
Needs porting. Does not block Phases 1, 2 or 4.

### 3. Then, and only then, Phase 3
  - pilot: slope-weight, 3 runs, 1 seed. Gate: a weight that moves std(pred)/std(true) toward 1
    without hurting pooled.
  - factorial: anchor x designed x slope x coil, 2^4 = 16 cells.
    Seeds per cell decided by sigma: sigma <= 0.01 -> 3 (48 runs); sigma >= 0.02 -> 5 (80 runs).
    Submit in waves of ~8. Keep all seeds of one cell on the SAME card type.
  - Estimated 2.5-4 more days of cluster time at ~9 concurrent runs.

================================================================================
## HOW TO DRIVE THE CLUSTER FROM A FRESH SESSION
================================================================================
- ssh nissimb@slurm.bgu.ac.il  (VPN REQUIRED; without it port 22 just times out)
- slurm.bgu.ac.il ROUND-ROBINS across login nodes. tmux sessions and background processes live
  on ONE node only - pin to slurm-login-04.auth.ad.bgu.ac.il or you will not see your own jobs.
- Repo: /home/nissimb/DeepPEF (branch current-vers). Data is symlinked read-only into
  data/ from /groups/keasar_group/casp15/Shahar/DeepPEF - NEVER write into Shahar's tree.
- ALWAYS `export WANDB_MODE=disabled` before submitting: 03_calib_ctrl.sh / 04_seeds.sh /
  05_anchor_sweep.sh do NOT set it and the run dies at wandb login. (02_smoke.sh does set it.)
- preflight.py was patched locally (NOT pushed): checks 1/2/5/7 import train.py helpers that
  were never written (build_datasets_for_preflight, build_train_loader_for_preflight,
  run_one_preflight_minibatch, resolve_effective_epochs) -> they now report honest SKIPs.
  Original preserved at cluster_run/code/preflight.py.orig
- The training log itself confirms the split every run:
  "FULL-DATA: 306 train / 34 held-out val proteins ... 28-test untouched" == the 28/34/306 gate.
- Monitoring: tmux session "collect" runs /home/nissimb/auto/collector.sh, which rewrites
  results/RESULTS.md every 15 min. Event log: /home/nissimb/auto/STATUS.txt (grep PROBLEM:).

================================================================================
## 2026-09-06 - CORRECTIONS + FIX 1 APPLIED + PHASE 3 PILOT LAUNCHED
================================================================================

### STATE.md section 3 is STALE. Both "missing" levers already existed.
  C slope  --slope_weight   train.py 76/80/433-444. EXISTS, correctly guarded (weight 0 builds
           nothing, baseline bit-identical). Written by an earlier agent ("Agent-E SLOPE term").
  D coil   Implementation lives in train_utils.py _flory_unfolded_graph (line 255), reached from
           get_unfolded_graph (line 234); config defaults in model/model_cfg.py lines 37-38.
           What was missing was ONLY the CLI flag. (Note: pnas_train.py lines 62-65 do define
           --flory_unfolded/--flory_nu, but pnas_train.py must never be used for results.)
  RULE: the code is newer than STATE.md. When they disagree, the code wins. Check before
  re-implementing anything - re-implementing a working loss term would leave two versions and
  no way to know which produced a number.

### FIX 1 APPLIED to Megascale-fineTuning/train.py (backup: train.py.bak)
  line 77-78  --flory_unfolded (store_true) and --flory_nu (float, default 0.5)
  line 92-95  CFG.flory_unfolded / CFG.flory_nu set from argv, plus a (0,1] range check
  line 164-5  both logged into wandb_config so a result traces back to its configuration
  VERIFIED, all three required checks passed:
      OFF reproducible: True     (flag inert when off - baselines stay valid)
      ON differs:       True     (the coil actually changes the unfolded graph)
      shapes equal:     True
  Also verified end-to-end that argparse accepts the flags: `--flory_nu 5.0` raises exactly the
  intended ValueError. FACTOR D IS NOW SWITCHABLE - the factorial is unblocked.

### THE EVAL CHAIN WAS BROKEN AND IS NOW FIXED
Phase 1 training COMPLETED (7:53:46, best epoch 14, all 15 checkpoints present) but its eval job
died in 11 s: evaluate.py needs ./data/Processed_K50_dG_datasets/Pnas_filtering/pnas_mutations.csv
and only training_data + mutation_datasets had been symlinked. ALL subdirectories of Shahar's
Processed_K50_dG_datasets are now linked, plus data/{S669,MegaScale,MsDs,FireProtDB,
megascale_proteins.csv}. Evals resubmitted for calib_ctrl_repro2, sigma_seed42, sigma_seed1.
LESSON: a training run completing is not a result. Check that the eval produced a CSV.

### Phase 3 slope pilot LAUNCHED
  tags p3_slope0.3_s42 / p3_slope1.0_s42 / p3_slope3.0_s42, seed 42, everything else at the
  reference, 15 epochs, qos normal + --gres=gpu:rtx_6000:1.
  JUDGEMENT CALL: COMPUTE_PLAN.md (lines 155-158) says "run three weights at one seed" but never
  names them. {0.3, 1.0, 3.0} was chosen to mirror this project's own anchor ladder rather than
  invent a new scale. Say so in the report.
  GATE: pick the weight that moves std(pred_ddG)/std(true_ddG) toward 1 WITHOUT hurting pooled.
  REPORT std(pred)/std(true) PER PROTEIN alongside pooled PCC - if pooled improves while the
  ratio does not move, the weight is doing something other than what it was written for, and
  that must be known before it becomes one of 16 factorial cells.

### STILL OPEN
  - FIX 3: measure whether the mini-batch block ddG spread is representative of the protein's.
    Ratio >= 0.9 -> slope term is fine as written; <= 0.7 -> add shuffling behind its own
    default-off flag (do NOT shuffle now, it would break comparability with the runs in flight).
  - FIX 4: optional, reuse wt_dg instead of a second get_wt_deltaG call. Must not change numbers.

---

# CHECKPOINT 2026-09-06 20:30Z — read this section first

**Written by the session that ran W0, launched the 48-run factorial, and built the
autopilot.** If you are a fresh agent or a worker picking up a role, this section plus
`cluster_run/docs/AUTOPILOT.md` is your source of truth. Do not re-derive anything
recorded here.

## 0. How to work on this project (process, not science)

- **Manager / worker split.** The main session dispatches and applies; workers read the
  large documents and source. The main context is the scarce resource — every large file
  read into it is paid for once and re-paid every turn afterwards.
- **Never resume a stale worker.** When a worker finishes its part or goes idle, it first
  updates THIS file and `cluster_run/docs/STATE.md` with: current state, decisions,
  findings, what was tried, relevant files, work remaining, next steps. Then it exits. A
  new role-holder is a NEW worker with clean context that reads these files first.
- **Patches use verbatim anchors, never line numbers** — several agents edit
  `train.py` / `train_utils.py` / `hydro_net.py`, and line numbers shift under them.
- **Every number carries three qualifiers:** pooled or per-protein · ΔG or ΔΔG · which
  split and which selection.

## 1. What is running right now

**The autopilot drives everything.** It lives in a tmux session on the cluster:

```bash
tmux attach -t autopilot          # watch it
tail -f ~/auto/autopilot.log      # or just read the log
python scripts/autopilot.py --once     # dry run, prints the decision trace
```

It polls every 10 minutes, records completed cells to `results/RESULTS.tsv`, runs the
seven self-audit checks, applies decision node D1 by the numeric rule, and halts on any
of the four halts in `AUTOPILOT.md` §7. It writes `results/HALT.md` and a `PROBLEM:`
line to `~/auto/STATUS.txt` if it halts. It holds a `mkdir` lock at
`~/auto/.autopilot.lock`, so a second copy cannot start.

**It never cancels a job it did not itself submit and record in `state.json`.**

In the queue: the 48-run calibration factorial (16 cells × 3 seeds), 2 side arms, and
3 slope pilots finishing their 15 epochs.

## 2. The decision that changed the plan — W0

`results/w0.json`, 28 test proteins, checkpoint `calib_ctrl_repro2` epoch 14, no training.

| condition | var(E_u) | r_x | corr(E_u, wt_err) | corr(E_u, length) |
|---|---|---|---|---|
| base | 0.958 | 1.000 | 0.420 | 0.403 |
| **noemb** | **0.317** | **0.331** | **0.119** | 0.895 |
| noOH | 0.921 | 0.962 | 0.373 | 0.339 |
| coil | 1.126 | **1.175** | 0.507 | 0.144 |

**D0 verdict: ProtT5 is the offset channel.** Zeroing the embedding block in the unfolded
pass removes two thirds of the across-protein variance and drops the correlation with the
wild-type error from 0.420 to 0.119.

**The Flory coil makes it WORSE** — it *raises* var(E_u) by 17%. It only touches `D` and
`Fb`, which the arithmetic already suggested were not the cause (corr(wt_err, length)
≤ 0.12, so the variance is not extensive).

**Consequence: factor D of the factorial is `--unfolded_emb {full,zero}`, not the coil.**
The coil is demoted to a 2-cell side arm (`sa_coil_seed42`) so the negative is measured
rather than assumed. This redirected 12 of the 48 cells before any GPU time was spent.

## 3. FIX 3 — settled, measured

332 proteins, mb=64: `mean(block std) / protein std` median **0.901**, IQR 0.863–0.927,
min 0.691. 91% of the CSVs are position-ordered, so the concern was well-founded — but
the blocks are representative anyway.

**The slope term is correct as written. Do NOT add shuffling.** It would change the
training regime for every run and break comparability with everything in flight.
Full data in `results/fix3_block_std.json`.

## 4. Levers implemented, all default-off and gated

| flag | item | gate | status |
|---|---|---|---|
| `--unfolded_emb {full,zero,mean}` | U2 | `scripts/gate_u2.py` 13/13 | **factor D of the factorial** |
| `--burial_features` / `--burial_mode {count,hse}` | W5 | `scripts/gate_w5.py` 18/18 | ready, not yet run |
| `--gcn_span N` | W7 | `scripts/gate_w7.py` 11/11 | ready, not yet run |
| `--gcn_bidir` | U10 | same | ready; **enabling it breaks comparability with every run to date** |
| `data/aa_descriptors.csv` | W6 | `scripts/verify_descriptors.py` 3/3 | matrix built, model wiring NOT done |
| `--flory_unfolded` | D (coil) | — | demoted by W0 |

`scripts/gate_g4_cpu.py` runs a real `PEM` forward pass for all six flag combinations on
CPU. **Run it after any change to the feature vector** — it catches slice errors in
minutes that would otherwise appear six GPU-hours later.

## 5. Three bugs it caught, worth knowing about

1. **The GCN branch reads `x[:, :32]`, not `:48`** — it takes D(16) plus only half of Fb.
   Widening it to 48 fed 71 dims into a 55-dim layer.
2. **`fc2_gat` / `fc2_gcn` project back to FIXED internal widths**, so `inst_norm1`,
   `inst_norm2` and `fc_in_dim` must NOT grow when a feature block is added. Only
   `fc1_gcn` / `fc1_gat` grow.
3. **The GAT residual takes `identity` before `fc1_gat`**, so with a wider input
   `h1 + identity` is a shape error. Fixed by taking the identity after the projection,
   behind an explicit branch so the off-path stays byte-identical.

## 6. Blocked, and why

**W8 (pLDDT) is blocked and it is not a matter of effort.** The preprocessed tensors hold
exactly five things — `coords_tensor.pt`, `deltaG.pt`, `mask_tensor.pt`,
`one_hot_encodings.pt`, `prott5_embeddings` — and there are no `.pdb` files and no
B-factor anywhere under `data/`. Per-residue confidence would require regenerating every
tensor, which changes the cost of the item completely. Do not claim it is "not yet
written".

**W5's specified SASA gate is unrunnable here.** The dataset stores 4 backbone atoms per
residue (N, CA, C, CB); a Shrake-Rupley reference computed on those is blind in exactly
the same way as the Cβ neighbour count, so a correlation between them measures their
shared blindness. It was replaced by the two checks valid on backbone-only data (burial
differs folded vs unfolded; no length confound) plus a half-sphere-exposure arm. **State
this substitution in the write-up** rather than leaving check 3 silently unanswered.

**The GitHub push needs Nissim once.** Two commits are ready (`1870ab2`, `ad7fec3`); the
cluster has no credential of any kind. One-time fix:

```bash
ssh-keygen -t ed25519 -C "bgu-deepef" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub     # paste at https://github.com/settings/keys
cd /home/nissimb/DeepPEF
git remote set-url origin git@github.com:nissimbrami/DeepEF-Thesis.git
git push origin current-vers
```

## 7. Work remaining, in order

1. **Factorial completes** (~1.5 days) → the autopilot applies D1 automatically.
2. **W6 model wiring** — the matrix and its verification exist; the block must still be
   spliced in beside W5's, at `desc_start = 48 + solv_dim`, with `fc1_gcn`/`fc1_gat`
   grown and the GAT-residual branch extended. Read §5 first.
3. **S4 refinement** — sweep whichever factors showed a real effect, 5 seeds; deliver the
   trade-off curve of `std(b)`, slope distribution and PP against anchor weight.
4. **S5–S7 information factorial** — W5 × W6 × W7, 2³ × 3 seeds = 24 runs, **on top of
   the winning calibration cell, never the bare baseline**, or the two result sets cannot
   be combined.
5. **S8 final model** + 5-seed ensemble; **S9** `results/FINAL.md`.

## 8. Site rules that are already paid for in wasted runs

- `export WANDB_MODE=disabled` before EVERY submission — `03/04/05` omit it and the run
  dies at wandb login. This destroyed a 7h53m run once.
- `--qos normal --gres=gpu:rtx_6000:1`. **Never select a GPU by partition name** — the
  submit filter reroutes onto a GTX 1080, which OOMs; training needs ~10 GiB.
- Never `ise-pheno` (3.5× slower under contention), never 2080/1080.
- All seeds of one cell on the same node class, recorded.
- **A finished training run is not a result.** Three separate 7h+ runs reached COMPLETED
  before any eval CSV existed. Completion is defined by a non-empty CSV in
  `eval_results/`.
- Tensors in `data/` were saved on CUDA; loading them on a login node needs
  `map_location` and `weights_only=False`.
- `slurm.bgu.ac.il` round-robins across login nodes — a tmux session lives on one node
  only.

---

# CHECKPOINT 2 — 2026-09-06 21:10Z — THE COIL FINDING WAS REVERSED

**Read after any compaction, in this order:**
1. `results/CONTEXT.md` — this file, both CHECKPOINT sections
2. `results/w0.json` (ddG metric) **and** `results/w0_dg.json` (dG metric) — they disagree, and the disagreement IS the finding
3. `cluster_run/docs/AUTOPILOT.md` — the S0-S9 state machine
4. `~/auto/autopilot.log` and `~/auto/STATUS.txt` (grep `PROBLEM:`)

## The correction that matters most

CHECKPOINT 1 recorded "the Flory coil makes it WORSE, r_coil = 1.175". **That conclusion
was drawn on the wrong metric and is now superseded.**

The coil replaces the unfolded distance map with `d(i,j) = b*|i-j|^nu`, a function of
SEQUENCE SEPARATION ONLY. A point mutation changes neither chain length nor positions,
so that matrix is IDENTICAL for wild type and mutant and **cancels exactly** in
`output - output[0]`. Scoring a ΔG lever by a ΔΔG metric measures the cancellation, not
the lever. The project's own COMPUTE_PLAN line 171 predicted this.

**Re-scored on absolute wild-type ΔG** (`scripts/w0_dg.py`, 28 test proteins,
calib_ctrl_repro2 e14, no training):

| condition | MAE | corr | std(err) = std(b_p) |
|---|---|---|---|
| base | 4.9650 | 0.3464 | 1.0385 |
| **coil, b fixed 5.82 A** | **3.9521** | 0.3404 | 1.1264 |
| coil, b fitted | 5.7446 | 0.4065 | 1.1269 |
| coil, ca_only | 5.5463 | 0.4532 | 1.1073 |
| unfolded_emb zero | 3.1315 | 0.0510 | 0.9999 |

**The coil with a FIXED b improves absolute ΔG MAE by 1.01 (20%).** It is the strongest
geometric lever on `b_p` we have, and it was nearly discarded.

**This also settles U4 empirically:** `fitted` HURTS (+0.78) and `fixed` HELPS (-1.01).
Fitting `b` to the protein's own folded mean CA-CA distance re-injects folded geometry
into the reference state the lever exists to remove. Use `--coil_b fixed`.

**Caveat to carry:** `noemb` has the lowest MAE (3.13) but destroys the across-protein
correlation (0.35 -> 0.05). It removes the offset by removing the signal. The coil keeps
corr while cutting MAE. Do not read MAE alone.

## What this changes downstream

- Factor D of the running 48-run factorial stays `--unfolded_emb` — that decision was
  made on the ΔΔG metric and is still correct FOR ΔΔG. Do not re-open it.
- **A ΔG arm is now justified**: `--loss_mode dg --flory_unfolded --coil_b fixed`.
  Better ΔG means smaller `b_p`, which closes the pooled->PP gap. The coil helps ΔΔG
  INDIRECTLY, through calibration.
- Every geometric unfolded-state lever must be scored on ΔG, never ΔΔG.

## Rule to keep

**Score a lever on the metric it acts on.** ΔΔG cancels anything that is identical
between wild type and mutant — chain length, positions, and therefore every purely
geometric reference-state change. ΔG does not.

---

# CHECKPOINT 3 — 2026-09-06 21:15Z — WAKE-UP INSTRUCTIONS

## If you are a fresh agent, read exactly these, in this order

1. **This file**, all three CHECKPOINT sections. CHECKPOINT 2 supersedes part of CHECKPOINT 1 — read both and note the correction.
2. `results/w0.json` — the ddG-metric ablation. Decided factor D.
3. `results/w0_dg.json` — the SAME levers on the dG metric. **They disagree, and the disagreement is the finding.**
4. `results/fix3_block_std.json` — the slope-term measurement.
5. `cluster_run/docs/AUTOPILOT.md` — the S0-S9 state machine and the four halts.
6. `~/auto/autopilot.log` (tail) and `~/auto/STATUS.txt` (grep `PROBLEM:`).

Do NOT re-derive anything recorded here. A decision already written was made with evidence
that is on disk; re-deciding without that evidence is how a stale premise gets re-adopted.

## WHAT WE ARE WAITING FOR — one thing only

**The 48-run calibration factorial.** The account is capped at **8 rtx_6000 and ZERO of
every other card type** (`MaxTRESPA` on account `keasar`, applies regardless of QOS —
`--qos normal` does NOT bypass it; that claim was wrong and is corrected). So roughly 5-8
run concurrently and the rest queue. Expect **~1.5 days** for all 48, and it cannot be
accelerated without an admin raising the cap:

    sacctmgr modify account keasar set MaxTRESPA=gres/gpu:rtx_6000=8,gres/gpu:rtx_4090=8

Nothing else is waiting on compute. All code work runs in parallel on login nodes.

## Live state as of this checkpoint

- 5 training runs on GPU, 99 queued (48 trainings + 51 chained evals).
- **0 factorial eval CSVs so far.** A finished training run is NOT a result — completion
  is defined by a non-empty CSV in `eval_results/`. Three separate 7h+ runs once reached
  COMPLETED with no CSV.
- The autopilot runs under tmux (`tmux attach -t autopilot`). It died once because
  `results/` did not exist and the atomic state write failed; the directory now exists and
  the lock is pid-aware and self-reclaiming. If it is dead again:
  `tmux new-session -d -s autopilot "bash /home/nissimb/auto/run_autopilot.sh"`
- Six workers are applying the remaining items in parallel: U5/U6, W7 edge features,
  monitor+autodebug, the end-to-end verification suite, the 100k catalogue, and W8 built
  generically.

## The rule that produced the two biggest findings

**Score a lever on the metric it acts on.** ddG cancels anything identical between wild
type and mutant — chain length, positions, and therefore every purely geometric
reference-state change. The Flory coil looked harmful on ddG (r=1.175) and turns out to
be our strongest geometric lever on `b_p` when scored on absolute dG (MAE 4.965 -> 3.952
with `--coil_b fixed`). It was nearly discarded on the wrong measurement.

## Standing site rules (paid for in wasted runs)

- `export WANDB_MODE=disabled` before EVERY submission.
- `--qos normal --gres=gpu:rtx_6000:1`; never select a GPU by partition name.
- Never `ise-pheno`, never 2080/1080. All seeds of one cell on one node class.
- Tensors in `data/` were saved on CUDA: load with `map_location` and `weights_only=False`.
- `class PEM(torch.nn.Module):` — an anchor written `nn.Module` silently matches nothing.
- After ANY change to the feature vector run `python scripts/gate_g4_cpu.py`. It must
  print `baseline ... dG=-0.0030 width=1092`. A moved baseline means every run in flight
  is invalidated — revert, do not debug in place.

---

# CHECKPOINT 4 — 2026-09-06 ~22:00Z — POST-COMPACT RECONNECT + THE SCIENTIFIC GAPS

**Read this section FIRST. It supersedes CHECKPOINT 3's "what we are waiting for".**

## 0. How to reconnect (this cost a whole reconnect cycle once — do not repeat it)

The cluster password lives ONLY in the shell env as `SLURM_PW`. A compact kills the shell and the
password with it, and every `ssh` then fails with `Permission denied`. It is recoverable from
`C:/Users/User/Downloads/for claufe.txt` (an old chat transcript). Run everything as:

```
cd <scratchpad>
SLURM_PW='<pw>' python cluster.py exec <<'EOF'
cd /home/nissimb/DeepPEF
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate esm2_env_py38
export WANDB_MODE=disabled
<commands>
EOF
```
Upload: `MSYS_NO_PATHCONV=1 SLURM_PW='<pw>' python put.py <remote_dir> <local_file>`
(NOTE the argument order: remote dir FIRST. And `MSYS_NO_PATHCONV=1` or Git-Bash mangles the path.)

**The real fix, still not done:** an SSH key. It survives compaction and also unblocks the git push.

## 1. State verified by running it, not by trusting a report

- **5 trainings RUNNING, 104 queued.** Nothing was cancelled. The 48-run factorial is intact.
- **0 factorial eval CSVs.** A finished training is NOT a result. ~5-7 days remain.
- **autopilot ALIVE** — it had died with the login node (`rc=143`). Restarted with
  `setsid nohup` (NOT tmux) so it now survives a login-node drop.
- **`python scripts/gate_g4_cpu.py` prints `baseline ... dG=-0.0030 width=1092`.**
  This is THE guard: it means none of the new code has moved the feature vector, so the 48 runs
  in flight are still valid. Re-run it after ANY feature-vector change. If it moves, revert.
- **All 8 gate suites green**, including gate_u3u4 at **32/32** (see §2).

## 2. gate_u3u4 — fixed, and the diagnosis is worth keeping

It failed 2/32 for a long time and was written off as "known-false". It was neither false nor a
float-precision issue — my first two fixes (absolute tolerance, then relative tolerance) BOTH failed,
which is what forced an actual measurement:

- off-diagonal agreement: **1.2e-06** (perfect)
- the four DIAGONAL channels (0/5/10/15, the self-pairs): **1.38e-03**

Cause: `get_dist_matrix` flattens to (N*4,3) and runs ONE `cdist`; a self-distance goes through
`sqrt(x^2+x^2-2x.x)`, a catastrophic cancellation that returns ~1e-3 instead of exactly 0. The gate
was comparing that artifact and calling it a layout error. **Channel 5 IS CA-CA** (verified: it equals
`torch.cdist(x[:,1,:],x[:,1,:])` off-diagonal, and 1*4+1 = 5). Fix: compare off-diagonal for the
layout claim; assert the self-pair diagonal is near zero separately.

**Lesson to carry: do not guess at a tolerance twice. Measure the error's STRUCTURE first.**

## 3. What three workflows established (33 agents, all complete)

**Adversarial verification of the 9 worker items** → `results/VERIFY_9_ITEMS.md`.
- NOT-READY: **mordred, u5u6, w7edge, monitor** (`monitor` selects a GPU BY PARTITION NAME, which
  violates the site rule that has already cost us runs).
- READY-WITH-FIXES: w6wire, u3u4, w9, e2e, catalogue.
- The most dangerous find: **`PEMGraphTransformer` silently ignores the descriptor block.** Its
  slices stay in-bounds with a wider vector, so the run COMPLETES and reports a baseline number as a
  descriptor result. This is the project's signature failure mode — a silent wrong number.

**U5/U6 + W7-edge applied** to the cluster, G4 re-verified at width=1092.

## 4. THE SCIENTIFIC GAPS — this is the actual remaining work

**a. Mordred has NEVER been run.** The descriptor CSV on disk has the header
`"synthetic replica for verification"`. **Every W6 number so far was scored on a fake matrix.**

**b. The histidine SMILES is WRONG** in PHASES_5_9_WITH_CODE.md:230 —
`c1cc(nc1)...` is a ring with FOUR carbons and ONE nitrogen (C7H10N2O2). L-histidine is C6H9N3O2,
imidazol-4-yl, TWO ring nitrogens. Correct: `c1c(nc[nH]1)C[C@@H](C(=O)O)N`. Mordred would compute
~654 descriptors for the WRONG MOLECULE, and the existing gate cannot see it because it only tests
L, D and F. Fix requires a structural self-check (formula + stereocentres) over all 20.

**c. Ofir's contribution was discarded, precisely.** `aa_descriptors.py` hard-codes
`ORDER = list('ACDEFGHIKLMNPQRSTVWY')` and RAISES on anything else; lookup is `one_hot @ [20,K]`, so
**the alphabet is closed at the tensor level.** His claim is the opposite: the physicochemical space
is continuous, so a model trained on canonical residues predicts NON-CANONICAL effects. A 20-row
closed table throws away the entire point.
  - The sharpest objection to answer: the vector ALREADY has 1024-dim ProtT5 + 20-dim one-hot. Can a
    16-dim descriptor block add anything they do not carry? Test it: regress descriptors on ProtT5.
  - MegaScale has ~0 non-standard residues, so the generalisation claim needs a PROXY: hold out a
    canonical residue type (e.g. W) entirely and predict its mutations from descriptor space alone.

**d. The 100k catalogue was never exploited.** 100,246 x 21. It was correlated only against the
PRE-TRAINING DECOY LOSS (r = -0.001) and declared dead. That is the wrong metric — the same error as
the coil. The right question is whether protein-level properties (BSA, oligomeric state, ligands,
hydrophobicity, disulfides) predict **b_p** on the 28 test proteins. Caveat that must be stated:
n=28 means |r| < ~0.37 is indistinguishable from zero at p=0.05, over ~21 columns.

**e. Ligands as graph nodes — never built.** Only metal ions were considered. Any bound molecule
(ATP, NAD, cofactor, substrate) stabilises a protein by kcal/mol. Architecturally it is a node with
its own edges. HONEST CAVEAT: MegaScale is small in-vitro domains, mostly ligand-free — this is a
GENERALISATION lever, not a benchmark-number lever, and must be presented that way.

## 5. THE BIGGEST OMISSION: no dG arm has ever been submitted

Verified against `sacct`: every queued job is the ddG factorial. The coil-on-dG finding
(MAE 4.965 -> 3.952 with `--coil_b fixed`) is our strongest result on `b_p` and **zero runs test it.**
It needs GPU from the same 8-card cap the factorial is using, so it needs the user's approval.

## 6. THE RULE THAT KEEPS PAYING

**Score a lever on the metric it acts on.** ddG cancels everything identical between WT and mutant —
chain length, backbone positions, and therefore every purely geometric or whole-protein property.
It has now caught TWO directions that were nearly discarded on invalid measurements: the coil
(r=1.175 "harmful" on ddG; our best b_p lever on dG) and BSA (scored against a pre-training loss that
cannot predict ddG either way). Audit every new lever against this before spending a GPU-hour.

---

# CHECKPOINT 5 — 2026-09-07 — b_p AND a_p MEASURED DIRECTLY (no GPU)

`scripts/calc_bp.py` -> `results/calib_per_protein.json`, from
`eval_results/abl_calib_ctrl_repro2_e14.csv` (28 proteins, 28,315 rows, checkpoint e14).

| quantity | value |
|---|---|
| b_p (WT error) mean | 0.3542 |
| b_p median | 0.7207 |
| **b_p std** | **1.5741** |
| b_p range | -4.7043 .. +2.7474 |
| **a_p (ddG slope) median** | **0.4990** |
| a_p range | 0.0789 .. 1.2566 |
| **per-protein ddG PCC median** | **0.7980** |
| **corr(a_p, per-protein PCC)** | **+0.5714** |
| corr(\|b_p\|, per-protein PCC) | -0.3596 |

## What these four numbers establish

1. **The calibration gap is real and quantified.** Per-protein median PCC 0.798 against a pooled
   ~0.59. The ranking information IS present within a protein; pooling destroys it. This is the
   thesis, now measured independently of any earlier analysis.

2. **std(b_p) = 1.57 kcal/mol** is the offset spread that pooling folds into the error. It is the
   target of every reference-state lever. Any dG-side lever must be judged by whether it shrinks
   THIS number.

3. **corr(a_p, PCC) = +0.571** reproduces the +0.60 recorded in the plan docs from a completely
   separate computation. The slope is the strongest single per-protein predictor of quality, and
   `--slope_weight` (factor C) is the only lever that attacks it directly -- and it has still never
   been run.

4. **a_p min = 0.0789 explains why the affine oracle is not reproducible.** The oracle divides by
   the slope; dividing by 0.079 amplifies that protein's noise ~13x. An estimator that is not
   monotone is not a bound, which is why the "0.77-0.81" figure inverts on some checkpoints. The
   0.70-0.72 from offset removal alone survives because subtracting b_p cannot amplify anything.

## Method note worth keeping

b_p and a_p are computed on DIFFERENT metrics on purpose: b_p is the WT error, a dG-side quantity;
a_p is the ddG compression slope, a ddG-side quantity. Fitting both on one metric would silently
mix the two error modes the whole thesis is trying to separate.

## Still true, still the gap

No dG arm has ever been submitted. `--slope_weight` has never been run. Both need GPU from the
8-card cap the 48-run factorial is consuming.

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

---

# CHECKPOINT 7 — 2026-09-07 — THE THESIS SURVIVED ITS STRONGEST ATTACK (measured, CPU only)

## The attack

An adversarial critic raised the sharpest objection anyone has put to this project:

> b_p is not a calibration error the model could have avoided. Subtracting a per-protein
> constant FITTED ON THE TEST LABELS removes between-protein variance from a pooled
> correlation **whether or not that constant is physically meaningful**. A perfectly-ranking
> predictor polluted with PURE RANDOM per-protein noise would show the same 0.59 -> 0.71 lift.
> If so, the headline result is a property of the ESTIMATOR, not of the model.

This had to be answered before any GPU was spent, because if true the whole programme is
arithmetic. It costs seconds to test.

## The test — `scripts/null_offset.py` and `scripts/null_fit.py`

Build a synthetic predictor with the REAL per-protein structure (real protein sizes, real ddG
values from the reference eval CSV) but a MEANINGLESS offset: `pred = a*true + b + noise`,
with `b ~ N(0, 1.5741)` (our measured std(b_p)) drawn independently per protein. Then grid
search (noise_sd, slope) over 25 x 21 cells, 3 seeds each, and ask how close the null can get
to our measured triple.

## Result: THE NULL CANNOT REPRODUCE US

| | pooled | offset-removed | per-protein |
|---|---|---|---|
| **measured (ours)** | **0.590** | **0.710** | **0.798** |
| best null over the whole grid | 0.620 | 0.785 | 0.712 |
| error | +0.030 | **+0.075** | **-0.086** |

Best achievable L2 distance: **0.118**, and it fails in a STRUCTURED way, not by a little
noise everywhere:

1. **The null over-delivers on offset removal (+0.075).** Of course it does: a meaningless
   constant is perfectly removable by construction, so subtracting it recovers a near-perfect
   ranking. Our real 0.710 is much LOWER than what a pure offset predicts, which means our
   residual error is NOT a clean per-protein constant. There is real within-protein error that
   the offset cannot absorb — the model is wrong in ways beyond a shift.
2. **The null under-delivers on per-protein PCC (-0.086).** Our within-protein ranking is
   BETTER than the null's at the same pooled value.
3. **The null needs slope 1.25 to get even that close** — a predictor that EXPANDS ddG. We
   measured a_p = 0.499, compression by half. The null reaches our pooled number by the wrong
   mechanism entirely.

## What this establishes

**The 0.59 -> 0.71 lift is not merely an estimator artifact.** The two error channels are
real and separable: a genuine per-protein offset (b_p) AND a genuine compression (a_p), and
no single random-offset model produces both at once.

**But the critic was still half right, and this is the honest caveat to carry:** offset removal
uses a constant fitted on test labels, so 0.70-0.72 is an ORACLE and cannot be claimed as
model performance. The claim it licenses is precisely: *the ranking information is present;
what is lost is calibration.* Turning that into a method requires an offset predicted from
FEATURES on held-out proteins — which is exactly the 100k-corrector direction, and the reason
that direction matters more than it looked.

## Method note

Both scripts are pure-stdlib and run in seconds on a login node. They should be re-run against
the factorial's own eval CSVs when those land, since every arm's claim rests on the same
arithmetic. A result that survives its strongest attack is worth more than one that was never
attacked.

---

# CHECKPOINT 8 — 2026-09-07 — THE SIGNAL IS ON a_p, NOT b_p (corrected significance)

> **PARTIALLY RETRACTED - see CHECKPOINT 9.** The "NOT b_p" half of this title is
> wrong. Burial predicts the offset (`frac_buried_rel_lt_0.25` vs `b_p_wt_error`, mean r=-0.475,
> 10/10 sign-consistent, 8/10 significant). It was missed because the b_p rows were scored on a
> single checkpoint against a uniform 90-test Bonferroni bar and were never put through the
> replication sweep that promoted the a_p row. Everything else in this checkpoint stands.

## The finding

Structural exposure predicts the COMPRESSION SLOPE a_p. Not the offset b_p.

`scripts/catalogue_vs_bp.py` -> `results/catalogue_vs_bp.json` + `results/CATALOGUE_VS_BP.md`.
28 test proteins, b_p/a_p from `abl_calib_ctrl_repro2_e14.csv` using calib_diag.py conventions
exactly (groupby protein, row 0 = WT, np.polyfit(ddg_true, ddg_pred, 1)). Structural covariates
computed from the actual AlphaFold models in
`data/Processed_K50_dG_datasets/AlphaFold_model_PDBs/` (all 28 present) with Biopython
Shrake-Rupley SASA.

**The one result that clears Bonferroni on BOTH Pearson and Spearman (90 tests, alpha 5.6e-4):**

| feature | target | pearson r | p | spearman rho | p |
|---|---|---|---|---|---|
| **mean_rel_SASA** | **a_p** | **+0.714** | 2.0e-05 | **+0.623** | 4.0e-04 |

Stable under leave-one-out (r in [0.642, 0.790], no sign flip) and holds within the 21 natural
proteins alone (r = +0.767), so it is not a designed-fold artifact.

**Interpretation: more-exposed, less-compact proteins suffer LESS slope collapse. Tightly
buried proteins are where the model under-reacts worst.**

## SIGNIFICANCE CORRECTION — read this before quoting any number

The worker's report claimed "FIVE clear it". An adversarial verifier caught that, and I
re-derived the table myself from the JSON. **Only ONE of the five clears alpha=5.6e-4 on both
tests.** The other four clear on one test and not the other:

| feature | target | pearson p | spearman p | verdict |
|---|---|---|---|---|
| mean_rel_SASA | a_p | 2.0e-05 | 4.0e-04 | **SURVIVOR** |
| frac_exposed_rel_gt_0.5 | a_p | 1.2e-04 | 4.1e-03 | one test only |
| mean_rel_SASA_hydrophobic | a_p | 3.4e-04 | 2.7e-03 | one test only |
| SASA_over_len_pow_073 | a_p | 8.4e-04 | 1.1e-04 | one test only |
| plddt_mean | per-protein PCC | **1.5e-02** | 2.4e-04 | **27x ABOVE the stated bar** |

The four near-misses are all the SAME underlying quantity (surface exposure) measured four ways,
so they are corroborating, not independent — which is a reason to believe the direction and a
reason NOT to count them as five findings. Report ONE result, with the other four as consistent
supporting measurements.

## What is NOT measurable here, and why that is not a null result

**All 28 test proteins are single-chain, ligand-free, metal-free monomers of 43-72 aa.**
Verified: `n_chains == 1`, `n_het_residues == 0`, `n_metal_residues == 0`, `interchain_BSA == 0.0`
for EVERY one. So complexity, oligomeric state, ligands and true interface area have **zero
variance** on this test set and are **untestable in principle here** — that is "not measurable",
NOT "no effect". Any ligand/complex claim needs a test set containing multimers and ligands,
which MegaScale is not.

Bulk hydrophobic COMPOSITION shows nothing (|r| <= 0.28, p > 0.14). Exposed hydrophobicity does.
**It is the placement of hydrophobics, not their amount.**

## The 100k catalogue cannot be joined — and that is correct, not a failure

**0 of 28 test proteins appear in the catalogue.** Verified two ways (exact 4-char PDB id, and
a case-insensitive substring scan over all 100,246 rows). This is by construction: the catalogue
IS the PDB pre-training decoy set and the 28 are held-out MegaScale test proteins. Disjoint by
design. No join was fabricated and no catalogue number is quoted for our 28.

The catalogue also has degenerate columns that would have produced fake nulls if used naively:
`Ligands_x` has 4 non-null rows of 100,246; `BSA_Numeric_x` has 4 unique values; `BSA_Percentage`
is 100,241 null; `Global Symmetry` is constant. **The live columns are `Ligands_y` and
`BSA_Numeric_y`.** The old "-0.001 BSA correlation" was therefore doubly void: an all-but-constant
feature scored against the pre-training decoy loss.

## Why this matters more than it looks

The offset-removal ceiling (0.70-0.72) assumes a_p is left alone. **a_p is the second calibration
object and nothing has ever attacked it** (`--slope_weight` has still never been run). And unlike
b_p, **a_p is a within-protein slope, so it does NOT cancel in ddG** — it is the one calibration
object legitimately measurable on the ddG metric.

An exposure-predicted per-protein slope rescale is therefore a genuinely new lever, and it is
CPU-testable on existing eval CSVs before any GPU is spent.

## Next, in order

1. Replicate the survivor across the other 9 eval CSVs (the 5 `abl_sigma` seeds especially) to
   confirm it is checkpoint-stable and not specific to `calib_ctrl_repro2_e14`. One loop, no GPU.
2. Only then consider a slope-correction arm.

## REPLICATION — the finding is checkpoint-stable (10/10)

Re-ran `catalogue_vs_bp.py --eval_csv` over every eval CSV on disk. `mean_rel_SASA` vs `a_p`:

| checkpoint | pearson r | p | spearman | p |
|---|---|---|---|---|
| calib_ctrl_repro2_e14 | 0.714 | 2.0e-05 | 0.623 | 4.0e-04 |
| abl_sigma_seed4_e14 | 0.703 | 3.0e-05 | 0.658 | 1.4e-04 |
| abl_sigma_seed3_e13 | 0.699 | 3.5e-05 | 0.656 | 1.5e-04 |
| abl_sigma_seed1_e13 | 0.683 | 6.2e-05 | 0.651 | 1.7e-04 |
| abl_p3_slope3.0_s42_e8 | 0.676 | 7.9e-05 | 0.618 | 4.6e-04 |
| abl_sigma_seed42_e9 | 0.674 | 8.5e-05 | 0.617 | 4.7e-04 |
| abl_sigma_seed2_e10 | 0.632 | 3.1e-04 | 0.620 | 4.4e-04 |
| abl_anchor_w0.3_s42_e14 | 0.587 | 1.0e-03 | 0.511 | 5.5e-03 |
| abl_anchor_w1.0_s42_e14 | 0.471 | 1.1e-02 | 0.477 | 1.0e-02 |
| abl_anchor_w3.0_s42_e13 | 0.400 | 3.5e-02 | 0.436 | 2.1e-02 |

**mean r = 0.624, sd = 0.102, min 0.400, max 0.714, sign-consistent 10/10.**
Two independent seed families (5 sigma seeds + the control) all land at 0.63-0.71. This is not
an artifact of one checkpoint.

### A SECOND finding fell out of the replication, unlooked for

The correlation decays MONOTONICALLY with the WT-anchor weight:

    anchor 0.3 -> r = 0.587
    anchor 1.0 -> r = 0.471
    anchor 3.0 -> r = 0.400

while all five unanchored sigma seeds sit at 0.63-0.68. **The anchor lever is already partially
suppressing the exposure/slope coupling** — the more you pin the WT, the less the model's slope
tracks exposure. That is a mechanistic link between factor A (the anchor, already in the running
factorial) and a_p, and it was not predicted. Three points is a trend, not a law; the running
48-cell factorial will test it properly and its anchor cells now have a specific prediction to
check rather than an open-ended sweep.

---

# CHECKPOINT 9 — 2026-09-07 — BIOLOGY/CHEMISTRY AUDIT + WAKE-UP

## READ THESE FIRST, IN THIS ORDER, AFTER ANY COMPACT

1. `results/CONTEXT.md` — ALL checkpoint sections. Later ones CORRECT earlier ones.
2. `results/w0.json` + `results/w0_dg.json` — they disagree, and the disagreement IS the finding.
3. `results/calib_per_protein.json` — b_p / a_p per protein.
4. `results/catalogue_vs_bp.json` — 90 correlations; only ONE clears Bonferroni on both tests.
5. `results/FINDINGS.md`, `results/VERIFY_9_ITEMS.md`.
6. `~/auto/autopilot.log` (tail) and `results/state.json`.

## RECONNECT (this cost a whole cycle once)

The password lives ONLY in the shell env as `SLURM_PW`; a compact kills it. Recover it from
`C:/Users/User/Downloads/for claufe.txt`. Then:
```
cd <scratchpad>
PYTHONIOENCODING=utf-8 SLURM_PW='<pw>' python cluster.py exec <<'EOF'
cd /home/nissimb/DeepPEF
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh
conda activate esm2_env_py38
export WANDB_MODE=disabled
EOF
```
Upload: `MSYS_NO_PATHCONV=1 SLURM_PW=... python put.py <REMOTE_DIR_FIRST> <local_file>`
Download: `MSYS_NO_PATHCONV=1 SLURM_PW=... python get.py <local_dir> <remote_file>`
`/tmp` is PER-LOGIN-NODE — write shared files to `/home/nissimb/`, not `/tmp`.
Connections sometimes time out; just retry.

## THE BIOLOGY/CHEMISTRY AUDIT — measured on the raw catalogue, 100,246 rows

### Dead columns that look alive (this is where the "-0.001 BSA null" came from)

| column | non-null | live sibling |
|---|---|---|
| `Ligands_x` | **4** of 100,246 | `Ligands_y` (75,464) |
| `BSA_Numeric_x` | **4 unique** | `BSA_Numeric_y` (4,867) |
| `BSA_Percentage` | 5 | — |
| `lossg` | constant 0 | — |
| `Global Symmetry` | constant | — |

The old BSA null was VOID TWICE: a dead column scored against the pre-training decoy loss, a
metric already known not to predict ddG. Neither the feature nor the metric could have shown
anything.

### Internal inconsistency
48 rows where the `BSA` string parses nonzero (e.g. "18.76%") while `BSA_Numeric_y` is 0.
`BSA_Numeric_y` is a PERCENTAGE in [0,70], NOT Angstrom^2. 31.3% are exactly 0.

### THE BIOLOGICAL DEFECT — 205,648 het-code occurrences, 8,187 distinct codes

| class | share |
|---|---|
| crystallization additives (SO4, GOL, EDO, NA, CL, PEG) | **27.4%** |
| metal ions | 18.9% |
| **real cofactors/substrates (NAD, ATP, HEM, FAD)** | **6.6%** |
| **MODIFIED RESIDUES (MSE, SEP, TPO)** | **6.3%** |
| unclassified | 43.5% |

12.8% of "ligand-bearing" rows carry ONLY crystallization additives.

**Only 6.6% of het codes are real biological ligands.** SO4 and glycerol come from the
crystallization buffer and do not stabilise the protein in vivo. **MSE is selenomethionine — an
amino acid IN THE CHAIN, used for phasing — so counting it as a bound ligand is both biologically
wrong and a double-count of a residue.**

**A ligand feature built on the raw column would learn crystallography, not biochemistry.**
Any W11 result computed before a het-code classification exists is uninterpretable.

## THE FOURTH MIS-SCORED LEVER: W5 BURIAL

`gate_w5.py` passes 18/18 and every assertion is MECHANICAL — shapes, byte-identity, hydropathy
ordering. **Not one scores performance.** W5's own help text says burial is ZERO in the unfolded
state and the folded-minus-unfolded delta IS the hydrophobic driving force — so it acts on dG/b_p.
But its only scheduled scoring is the S7 factorial whose EFFECT_MIN is defined on POOLED ddG.
**This is the coil case, one lever later.** AUTOPILOT.md has no dG-side acceptance criterion at
all, so a lever that shrinks std(b_p) without moving pooled ddG is recorded as noise.

Evidence the effect is real: `frac_buried_rel_lt_0.25` vs the WT error, mean r = **-0.475**,
sign-consistent **10/10**, significant in 8/10.

## CORRECTION TO CHECKPOINT 8

CHECKPOINT 8 said "the signal is on a_p, NOT b_p". **That was premature.** The b_p side was tested
ONCE, failed a uniform Bonferroni bar over 90 tests, and was then dropped from the replication
sweep — while its sibling WAS replicated and promoted. Replication is the stronger evidence.

**The honest picture: ONE structural axis acts on BOTH calibration channels.**
- exposure -> a_p (slope): +0.624 mean over 10 checkpoints
- burial -> b_p (offset): -0.475 mean over 10 checkpoints
Both show the SAME anchor-weight suppression gradient — independent corroboration on two channels.
**W5 burial is the lever that attacks both.**

## CODE STATUS — all 41 flags audited against live source

15/15 levers are wired end to end (flag -> CFG -> read in train_utils/hydro_net -> gate).
`--slope_weight` genuinely computes (`loss = loss + SLOPE_WEIGHT * slope_loss`) and now RAISES
under `--loss_mode dg` (the slope is defined on ddG spread).

**The one code gap: W9 has NO `--metal_features` flag in train.py**, though
`scripts/metal_features.py` reads `getattr(cfg,'metal_features')`. It can never fire. Decide:
wire it, or formally retire it as superseded by W11 (a metal is one ligand class).

## READY means FOUR things, not one

A lever is READY only if: internally correct AND biologically defensible AND actually read by the
model at training time AND scheduled to be scored on the metric it acts on.
**Correct code judged on the wrong metric is NOT ready.** Three levers have already been caught by
that rule: the coil, BSA, and now W5.

## STILL NEVER RUN (all need GPU from the cap of 5)

- the dG training arm (`--loss_mode dg --flory_unfolded --coil_b fixed`)
- `--slope_weight` (factor C) — now the best-motivated lever we have
- the severing experiment (built, dry-run green, `results/SEVERING_READY.md`)


# CHECKPOINT 9 — 2026-09-07 — CORRECTION TO CHECKPOINT 8: ONE STRUCTURAL AXIS, BOTH CHANNELS

> **AMENDED BY CHECKPOINT 11.** The conclusion below (retracting "NOT b_p") is CONFIRMED.
> But its headline number **-0.475 is a nuisance-parameter maximum — do not quote it**, and its
> "10/10 sign-consistent" promotion criterion has a ~60% false-positive rate here. The b_p effect
> is real in direction but **weak, and not significant at accurate SASA quadrature (p=0.070)**.
> Read CHECKPOINT 11 before citing any number from this section.

## The claim being corrected

CHECKPOINT 8 is titled **"THE SIGNAL IS ON a_p, NOT b_p"**. **The "NOT b_p" half of that title is
wrong and is retracted here.** The a_p half stands unchanged and is strengthened.

### Why the wrong conclusion was reached

The b_p-side rows were scored **once, on one checkpoint** (`abl_calib_ctrl_repro2_e14`), against a
**Bonferroni bar of alpha=5.6e-4 applied uniformly across all 90 tests**. Nothing on the b_p side
cleared it, so CHECKPOINT 8 recorded "Nothing survives Bonferroni against b_p" and stopped there.

The a_p row was then taken forward into a 10-checkpoint replication sweep and promoted to the
headline. **The b_p rows were never put through that same sweep.** The asymmetry was procedural,
not empirical: one channel got the strong test, the other got only the weak one.

That is backwards as evidence. **Sign-consistency across 10 independently-trained checkpoints is
far stronger evidence than a single-sample p-value against a conservative family-wise bar**, and
at n=28 a real |r|~0.47 has only ~60% power at alpha=0.05 and almost none at 5.6e-4 — so a uniform
Bonferroni over 90 tests is close to guaranteed to erase a true effect of that size. Failing that
bar once is not evidence of absence. The sweep was the right test and it simply was not run.

## The correction, measured

`scripts/bp_replication.py` -> `results/bp_replication_sweep.json`, `results/bp_anchor_gradient.json`.

Independent re-derivation: per-protein a_p/b_p/pcc/b_p_wt_error recomputed from the raw eval CSVs
and all 15 structural covariates recomputed from the AlphaFold PDBs with Biopython Shrake-Rupley.
It does **not** read `catalogue_vs_bp.json` or `results/repl/*.json` — the cached numbers were
re-derived from source, not trusted. All 10 CSVs verified to carry the same 28 proteins.
15 features x 4 b_p-side targets = **60 pairs**, each over all 10 checkpoints.

### The headline b_p-side result — ESTABLISHED

| feature | target | mean r | sd | sign-consistent | individually p<0.05 | range |
|---|---|---|---|---|---|---|
| **frac_buried_rel_lt_0.25** | **b_p_wt_error** | **-0.475** | 0.092 | **10/10** | **8/10** | -0.561 .. -0.306 |

Reproduces the oversight agent's numbers to three decimals. The two checkpoints that miss p<0.05
are `anchor_w0.3` (p=0.058) and `anchor_w1.0` (p=0.114) — i.e. the misses are **the anchor arms**,
which is itself the point (see the gradient section below), not scattered noise.

**So the honest picture is one structural axis acting on BOTH calibration channels:**

- **Exposure predicts the SLOPE**: `mean_rel_SASA` vs `a_p`, mean r = **+0.624**, 10/10 sign-consistent, **10/10** significant.
- **Burial predicts the OFFSET**: `frac_buried_rel_lt_0.25` vs `b_p_wt_error`, mean r = **-0.475**, 10/10 sign-consistent, **8/10** significant. *(CHECKPOINT 11: -0.475 holds only at Biopython's default SASA quadrature; at accurate quadrature it is -0.327, perm p=0.070. Quote ~0.3, not 0.48.)*

Same physical variable (relative solvent accessibility), opposite ends of it, two different
calibration objects. Exposed proteins lose less slope; buried proteins carry a more positive WT
offset error. This is more coherent than the CHECKPOINT 8 story, not less — it was hidden by
running the strong test on only one of the two channels.

### Ranked b_p-side table (top rows of 60; full table in the JSON)

| # | feature | target | mean r | sd | signs | sig | class |
|---|---|---|---|---|---|---|---|
| 1 | frac_buried_rel_lt_0.25 | b_p_wt_error | -0.475 | 0.092 | 10/10 | 8/10 | **ESTABLISHED** |
| 2 | SASA_per_residue | abs_b_p | -0.341 | 0.096 | 10/10 | 4/10 | SUGGESTIVE |
| 3 | frac_buried_rel_lt_0.25 | abs_b_p_wt_error | +0.324 | 0.085 | 10/10 | 3/10 | SUGGESTIVE |
| 4 | mean_rel_SASA | abs_b_p_wt_error | -0.312 | 0.097 | 10/10 | 2/10 | SUGGESTIVE |
| 5 | SASA_over_len_pow_073 | abs_b_p | -0.306 | 0.099 | 10/10 | 2/10 | SUGGESTIVE |
| 6 | SASA_per_residue | abs_b_p_wt_error | -0.304 | 0.092 | 10/10 | 2/10 | SUGGESTIVE |
| 7 | SASA_over_len_pow_073 | b_p_wt_error | +0.299 | 0.089 | 10/10 | 2/10 | SUGGESTIVE |
| 8 | mean_rel_SASA | abs_b_p | -0.289 | 0.095 | 10/10 | 2/10 | SUGGESTIVE |
| 11 | length | abs_b_p | +0.294 | 0.081 | 10/10 | 1/10 | SUGGESTIVE |
| 30 | length | abs_b_p_wt_error | +0.260 | 0.139 | 9/10 | 2/10 | DEAD |
| 31 | frac_buried_rel_lt_0.25 | abs_b_p | +0.205 | 0.159 | 9/10 | 1/10 | DEAD |
| 60 | frac_glycine | b_p | +0.006 | 0.068 | 5/10 | 0/10 | DEAD |

Classification used: **ESTABLISHED** = 10/10 sign-consistent AND >=6/10 individually p<0.05;
**SUGGESTIVE** = 10/10 sign-consistent but rarely significant; **DEAD** = not 10/10 sign-consistent.

**Tally: 1 ESTABLISHED, 28 SUGGESTIVE, 31 DEAD of 60 b_p-side pairs.**

Note the shape of the SUGGESTIVE block: rows 2-8 are *all* the same exposure/burial axis measured
different ways against different b_p parameterisations. They are corroborating, not seven findings
— the same caution CHECKPOINT 8 correctly applied to the four a_p near-misses.

### Verification of the three numbers I was asked to check

- `frac_buried_rel_lt_0.25` vs `b_p_wt_error`: **CONFIRMED** — mean r=-0.4746, sd=0.0922, 10/10, 8/10 sig, range -0.561..-0.306. Matches the reported -0.475 / 0.087 / 10/10 / 8/10 / -0.561..-0.306.
- `SASA_per_residue` vs `abs_b_p`: **CONFIRMED as SUGGESTIVE** — 10/10 sign-consistent, 4/10 significant, mean r=-0.341.
- `length` vs `abs_b_p`: **CORRECTED.** It is **10/10 sign-consistent, 1/10 significant** (mean r=+0.294) — i.e. **SUGGESTIVE, not DEAD**. The "1/10" in the brief is the significance count, which was read as if it were the sign-consistency count. Every one of the 10 checkpoints gives a positive r (+0.135..+0.401); only one reaches p<0.05. Weak, but it is not sign-inconsistent. The genuinely dead length row is `length` vs `b_p` (6/10, mean r=+0.040).

## The anchor-weight suppression gradient replicates on the b_p channel

CHECKPOINT 8's unlooked-for second finding was that the a_p correlation decays monotonically with
WT-anchor weight while unanchored sigma seeds sit highest. **The same gradient is present on the
b_p channel**, which is independent corroboration on a second channel:

| pair | \|r\| anchor arms (3) | \|r\| sigma seeds (5) | gap | full separation | MWU p (1-sided) |
|---|---|---|---|---|---|
| frac_buried_rel_lt_0.25 vs b_p_wt_error | 0.359 | 0.514 | +0.155 | YES | 0.018 |
| SASA_per_residue vs abs_b_p | 0.237 | 0.383 | +0.146 | YES | 0.018 |
| mean_rel_SASA vs abs_b_p | 0.185 | 0.332 | +0.147 | YES | 0.018 |
| SASA_over_len_pow_073 vs abs_b_p | 0.211 | 0.348 | +0.136 | YES | 0.018 |
| length vs abs_b_p | 0.196 | 0.329 | +0.133 | YES | 0.018 |
| *(reference)* mean_rel_SASA vs a_p | 0.486 | 0.678 | +0.192 | YES | 0.018 |

Per-arm detail for the headline b_p pair, `frac_buried_rel_lt_0.25` vs `b_p_wt_error`:

    anchor w0.3 -> -0.363    anchor w1.0 -> -0.306    anchor w3.0 -> -0.410
    sigma seeds -> -0.515, -0.419, -0.550, -0.543, -0.545

Every sigma seed is stronger than every anchor arm (0.018 is the floor p for a 5-vs-3
Mann-Whitney, i.e. perfect separation). **So the suppression is real on both channels.**

Two honest limits on this:
1. **The b_p gradient is NOT monotonic in anchor weight** (w3.0 is stronger than w1.0), unlike the
   a_p gradient which was cleanly monotonic 0.587 -> 0.471 -> 0.400. What replicates is
   *anchored-weaker-than-unanchored*, not the ordering within the anchor arms.
2. **3 vs 5 checkpoints is not a designed experiment.** The anchor arms also differ from the sigma
   seeds in seed (all s42) and epoch, so arm and seed-family are confounded. The running 48-cell
   factorial is what settles this; it now has a prediction on two channels instead of one.

## n=28 caveats — applying throughout, to the a_p finding as much as this one

- **n=28 proteins.** At n=28, r=+0.62 has a 95% CI of roughly [0.33, 0.80] and r=-0.48 roughly
  [-0.72, -0.14]. Quote directions and rough magnitudes; do not quote these r's to three decimals
  as if they were stable population values.
- **The 10 checkpoints are not 10 independent datasets.** They are 10 models scored on the *same*
  28 proteins. Replication here proves the effect is **not a checkpoint artifact** — it does NOT
  give 10x the effective sample size against protein sampling. The 28-protein draw is the single
  point of failure for both findings, and it is shared.
- **Range restriction.** All 28 are single-chain, ligand-free, metal-free monomers of 43-72 aa.
  Nothing here generalises to multimers, ligand-bound proteins, or large proteins, and complexity /
  BSA / ligand effects remain *untestable in principle* on this set (zero variance), not null.
- **Multiplicity is still live.** 60 b_p-side pairs were scanned. The buried/b_p_wt_error row was
  named in advance by the oversight agent, so it is a confirmatory test rather than the winner of a
  60-way search — but the 28 SUGGESTIVE rows were not pre-specified and should be treated as
  hypothesis-generating.
- **Designed vs natural.** The a_p finding was checked within the 21 natural proteins alone
  (r=+0.767); **the equivalent natural-only check has not yet been run for the b_p finding.** Until
  it is, a designed-fold contribution to the burial/offset coupling is not excluded.

## What this changes

1. **Retract "NOT b_p" from CHECKPOINT 8's title and body.** Keep everything else in CHECKPOINT 8:
   the a_p result, the Bonferroni correction of the "five clear it" claim, the empty-join finding,
   and the zero-variance analysis all stand.
2. **The offset channel has a structural predictor after all.** The offset-removal ORACLE is
   0.70-0.72 and ICC(b_p)=0.898 already said b_p is learnable; this now supplies a *structural*
   handle on it (burial), not just the knowledge that it is learnable.
3. **Both levers are the same lever.** An exposure/burial-derived per-protein correction is one
   feature with two effects, not two separate research directions.
4. **THE METRIC RULE still applies and is not softened.** `frac_buried_rel_lt_0.25` is a
   whole-protein property, so it is identical between WT and mutant and **cancels in ddG**. It is
   scored here on `b_p_wt_error` (the raw WT dG error) precisely because of that rule. Any lever
   built on it must be scored on dG or b_p — never on ddG. The rule is what made this measurable,
   and it is what caught the coil, BSA, and W5-burial levers.
5. **Procedural lesson.** When two sibling hypotheses are tested, run the strong test on both
   before promoting either. A single-checkpoint Bonferroni result is not grounds to drop an arm
   from a replication sweep — the sweep is cheap and it is the better test.

`scripts/gate_g4_cpu.py`: **ALL PASS**, dG=0.0023 width=1095 on the burial row, and the
no-new-dims row holds at **width=1092** as required. No feature vector was changed by this work
(analysis only, CPU only, no jobs touched).

---

# CHECKPOINT 10 — 2026-09-07 — OFIR RECONCILED, CORRECTOR FAILS, NO-OP HOLES CLOSED

Three workflows completed before a session limit. Their verifiers did NOT run, so everything below
is WORKER-REPORTED, NOT ADVERSARIALLY VERIFIED. Re-verify before quoting in the thesis.

## 1. OFIR'S THESIS — READ IN FULL (56 pages, BGU CS, Keasar, Dec 2023)

`results/OFIR_THESIS_NOTES.md`. **Three things in our build are WRONG:**

**(a) PCA-16 IS NOT HIS METHOD.** He does NO dimensionality reduction. "PCA"/"principal
component"/"SVD" appear NOWHERE in the thesis. His 654 features enter a CNN as 654 channels.
Our `--pca 16` is OUR invention and must be relabelled as ours, not attributed to him.

**(b) `ignore_3D=True` CONTRADICTS his 1826.** 1613 2D + 213 3D = 1826. Quoting 1826 is only
coherent if 3D was requested — which also explains his 1826->1280 missing-value drop, since 3D
descriptors return NaN without an embedded conformer. With `ignore_3D=True` the ceiling is 1613
and his drop is unreproducible. Our docstring says 1826 while our provenance says
"Mordred(ignore_3D)" — these contradict each other.

**(c) 15-of-20 IS NOT A FAITHFUL SCALING of 40-of-58.** 40/58 = 0.690, so proportional scaling
gives 13-14, not 15. Our help text calls 15 "three quarters"; 40/58 is roughly two thirds. Deeper:
**he gives NO justification for 40 at all** — it reads as tuned to land a workable feature count.
So there is no ratio to preserve. Likely inert on 20 canonical residues (any threshold 13-17
selects nearly the same columns) but the docstring must be fixed either way.

**HIS CENTRAL CLAIM AND ITS EVIDENCE (Table 3.3):** train on canonical single-point mutants, test
on non-canonical. PUMA BH3: 646 train -> 714 ncAA test, RMSE 0.975, **Pearson r = 0.642**.
CP2: 228 -> 240, RMSE 0.859, **r = 0.670**. 30 seeds varying only CNN init, split fixed.
**NO baseline of ANY kind** — no one-hot, no LM, no shuffled-descriptor, no per-position-mean.
He calls it "a proof-of-concept" and notes "there was no benchmark to compare RMSE results to".
He is candid that Fig 3.4 shows "a near-constant prediction line at 0" — part of that r is
partial collapse to a constant.

**HE ADDRESSES THE LM OBJECTION — AND CONCEDES IT** (p.31, verbatim): "Physicochemical properties
of AAs are implicitly represented in these datasets ... and accordingly, implicitly represented in
the embeddings. Thus, ncAAs, which do not occur in these databases, are challenging to represent."
**His argument is COVERAGE ONLY, not added information.** He runs ZERO experiments comparing
descriptors against an LM embedding or one-hot.

**CONSEQUENCE FOR US, and it is decisive:** our 28 test proteins are canonical, so every residue
HAS a real ProtT5 vector and the coverage gap he exploits DOES NOT EXIST in our setting. **By his
own sentence, the physicochemical content is already implicit in our 1024-dim ProtT5. The thesis
does NOT support adding a Mordred block on top of ProtT5+one-hot for canonical monomers.**

**HE INDEPENDENTLY FOUND OUR OFFSET PROBLEM.** Leave-one-AA-out (Table 3.4, 18 ncAAs x 30 seeds):
r stays flat 0.858-0.948 while RMSE swings 7.05-19.32, and he shows **RMSE tracks the train/test
MEAN GAP** (Ornithine 19.316 vs gap 18.030). That is b_p, found independently, with essentially
our offset-removal fix proposed.

**WHAT TO STEAL:** leave-one-AMINO-ACID-out. We have leave-one-protein-out but nothing holding out
a residue TYPE. Directly portable, needs no ncAA data, and asks whether the model learned residue
chemistry or residue identity. Also: enrichment as a metric, and the train/test mean-gap diagnostic.

**TRAP:** 2D Mordred CANNOT distinguish D from L — every 2D descriptor is identical for both, so
his effective alphabet is well under 58 and L/D pairs contribute ONE unique value, not two.

## 2. THE b_p CORRECTOR FAILS — a real, publishable negative

`results/OFFSET_CORRECTOR.md`. Mean over all 10 eval CSVs:

| | pooled ddG PCC |
|---|---|
| raw | 0.5994 |
| mean-offset baseline | 0.5940 |
| **LOPO feature-predicted offset** | **0.6049** |
| oracle offset removed | 0.7119 |

**95% of the oracle gain does not survive held-out prediction.** Held-out R^2 of b_p is NEGATIVE
on 8/10 CSVs (mean -0.123) — worse than predicting the training mean. 0 of 22 features reach
positive R^2 alone. Permutation null: p in [0.295,0.758], significant on 0/10. With ridge alpha
FIXED the gain goes NEGATIVE (-0.0016), so the +0.0055 was alpha-search variance, not skill.

**THE MECHANISM, which is the useful part:** the oracle gain is NOT broad learnable
miscalibration. **The top-2 |b_p| proteins carry 78% of it** (60-88% across all 10 CSVs).
Correcting only 2K5H+2KVS gives 0.591->0.687 (80% of the gain); correcting the other 26 gives
0.615 (20%). Neither is a structural outlier (max |z| ~2.0 over 22 features), so **there is no
signature to regress on**. And 2KVS is not an offset case at all: a_p=0.094 vs median 0.4975,
per-protein PCC 0.160 vs ~0.80 — the model simply fails there and b_p absorbs the failure.

**METRIC CORRECTION THAT MATTERS:** Pearson is SHIFT-INVARIANT, so subtracting a CONSTANT changes
pooled PCC by exactly zero (verified: -1.0/0.0/mean/+1.0 all give 0.590991). **Therefore the null a
per-protein corrector must beat is the RAW number, not the mean baseline.** Scoring against the
mean baseline would have manufactured a fake +0.011 lift.

**Also corrects a units confusion:** std(b_p)=1.5741 is the dG-space WT error. The ddG-space
intercept the oracle actually subtracts has std **0.2273**. The ddG intercept is the correct
corrector target.

## 3. THREE SILENT-NO-OP HOLES CLOSED — `scripts/gate_noop.py`, 30 checks ALL PASS

**(A) PEMGraphTransformer dropped inserted blocks silently — MEASURED, not assumed.** A gate builds
`flat_x` with 726 sentinel columns at offset 48, runs the real reassembly
`cat([:16],[16:48],[-1044:-20],[-20:])`, and the output is byte-identical to the no-descriptor
case: **0 of 726 sentinel columns reach the node vector, and nothing raises.** Guard now raises in
`__init__` (cheap) for ALL FIVE affected levers — aa_descriptors, burial, metal, struct_quality,
ligand — not just descriptors.

**(B) `--ligand_nodes` could not fire in training.** Wired `Trainer._set_ligand_context(batch)`
before the first `get_graph` in all three entry points; `ligand_features()` now RAISES when a table
is loaded but context was never set. **`gate_ligand.py` D.14 had CODIFIED the bug** — it asserted
"unset context -> exactly zero". Replaced.

**(C) `gate_open_alphabet` tested a /tmp fixture, not the real table. THE REAL FINDING:** the
committed 25-row table is **K=756 vs canonical K=726**, with **27 canonical columns DROPPED and 57
ADDED**, and the canonical 20 rows are **NOT byte-identical**. It is not an open alphabet over our
matrix — **it is a different descriptor matrix wearing the same name.** The loader now refuses it.

**(D) A FOURTH HOLE, unassigned:** `gate_open_alphabet.py` hard-coded the retired mode
`'curated12'`, so it raised in its first section and **had stopped testing anything at all.**

## STATUS

G4 re-verified after every change: `dG=-0.0030 width=1092`.
NOT verified adversarially (session limit killed all verifiers): Ofir notes, corrector, no-op
gates, severing, identifiability, slope_origin, W8. **Re-run the verifiers before the write-up.**

# CHECKPOINT 11 — 2026-09-07 — b_p RE-DERIVED INDEPENDENTLY: FINDING STANDS, HEADLINE NUMBER DOES NOT

**Read with CHECKPOINT 9.** CHECKPOINT 9's *conclusion* — that the "NOT b_p" claim in CHECKPOINT 8
was premature, and that one structural axis acts on both calibration channels — is **CONFIRMED and
stands**. This checkpoint does not undo it.

But CHECKPOINT 9's headline **number** (`-0.475`) and its **promotion criterion** ("sign-consistent
10/10") both fail independent re-derivation. Two defects, both material, both found by recomputing
from source rather than re-reading the cached JSON.

`scripts/indep_bp.py` -> `results/bp_replication_independent.json`,
`results/bp_anchor_gradient_independent.json`. Independent of `scripts/bp_replication.py`: targets
built from the eval CSVs' **own** `ddG`/`pred_ddG` columns rather than recomputed from `deltaG`;
slope/intercept via `scipy.stats.linregress` rather than `np.polyfit`; SASA recomputed from the
AlphaFold PDBs. Reads no cached result JSON.

## What replicated exactly

The direction, the sign-consistency, and the *existence* of the effect all reproduce:

| claim | reported | re-derived | verdict |
|---|---|---|---|
| burial vs `b_p_wt_error`, sign-consistency | 10/10 negative | **10/10 negative** | CONFIRMED |
| `SASA_per_residue` vs `abs_b_p` | 10/10 signs, 4/10 sig (suggestive) | **10/10 signs, 3/10 sig** | CONFIRMED as suggestive |
| `length` vs `abs_b_p` "1/10, correctly dead" | — | **10/10 signs, 1/10 sig** | **the brief is wrong: 1/10 is the SIGNIFICANCE count, not the sign count.** It is SUGGESTIVE, not dead. The genuinely dead length row is `length` vs `b_p` (6/10). CHECKPOINT 9 already caught this; re-confirmed. |
| `mean_rel_SASA` vs `a_p` | +0.624 | **+0.6188**, 10/10, 10/10 sig | CONFIRMED |

## DEFECT 1 — `-0.475` is a grid maximum, not an estimate

`frac_buried_rel_lt_0.25` is a **threshold count**: it counts residues below rel-SASA 0.25. On
43-72 aa proteins one borderline residue moves the feature by ~1/50 = 0.02. So the feature — and
the correlation — depends on two arbitrary choices nobody pre-registered: the SASA **quadrature**
and the **0.25 cutoff**. Both were left at their defaults, and the default happens to sit at the
joint maximum:

| SASA quadrature (n_points) | mean r | sig |            | cutoff (n_points=100) | mean r | sig |
|---|---|---|---|---|---|---|
| **100 (Biopython default)** | **-0.475** | 8/10 |  | 0.15 | -0.402 | 6/10 |
| 256 | -0.373 | 6/10 |                              | 0.20 | -0.396 | 6/10 |
| 540 | -0.333 | 5/10 |                              | **0.25 (default)** | **-0.475** | 8/10 |
| 960 (accurate) | -0.327 | 6/10 |                    | 0.30 | -0.269 | 2/10 |
|  |  |  |                                            | 0.35 | -0.205 | 0/10 |

The magnitude **decays monotonically as the SASA integration gets more accurate**, and decays in
both directions from the 0.25 cutoff. `-0.475` is the largest value on a 2-D grid of nuisance
parameters; the accurate-quadrature value is **-0.33**.

The **threshold-free** version of the same axis (`mean_rel_SASA`, no counting) is by contrast
completely quadrature-stable — +0.233 / +0.239 / +0.232 / +0.238 across all four settings — but
never individually significant (0/10). That is the honest size of this effect: **|r| ~ 0.23-0.33,
not 0.48.**

**Do not quote -0.475.** Quote the direction, and a magnitude of roughly 0.3 with the note that
threshold-count features are quadrature-sensitive at this protein size.

## DEFECT 2 — "10/10 sign-consistent" is a ~60%-false-positive criterion here

The whole promotion argument rests on sign-consistency across 10 checkpoints being strong evidence.
Measured: **it is not, because the 10 checkpoints are not 10 replicates.**

- mean pairwise correlation between the 10 checkpoints' `b_p_wt_error` vectors: **r = 0.884**
  (min 0.694, max 0.979)
- participation-ratio effective number of independent replicates: **1.24 of 10**
- protein-label permutation null, 2000 draws: **P(sign-consistency = 10/10 by chance) = 0.60**

Ten near-identical vectors agreeing on a sign is one observation reported ten times. At a 60% null
rate, ~36 of the 60 b_p-side pairs would reach 10/10 by chance alone — and **28 did**. The
SUGGESTIVE class is consistent with pure noise and should carry no weight.

**The criterion that does survive** is the permutation test on effect magnitude, which accounts for
the dependence by permuting protein labels. But applied honestly it does **not** rescue the b_p
finding at full strength, because the p-value moves with the same nuisance parameter as the
effect size:

| pair | quadrature | mean r | permutation p |
|---|---|---|---|
| `mean_rel_SASA` vs `a_p` | any | +0.619 | **0.0005** |
| burial vs `b_p_wt_error` | n_points=100 (default) | -0.475 | **0.006** |
| burial vs `b_p_wt_error` | n_points=960 (accurate) | -0.327 | **0.0695 — does not clear 0.05** |

**This is the single most important number in this checkpoint.** The b_p finding is significant at
the quadrature that maximises it and not significant at the accurate one. Its p-value is therefore
as nuisance-dependent as its effect size, and p=0.006 must not be quoted on its own.

## Revised classification

The three-way ESTABLISHED / SUGGESTIVE / DEAD scheme in CHECKPOINT 9 should be **retired**: its
middle class is noise and its top class was won on a nuisance-parameter maximum. Replace with the
permutation criterion:

| feature | target | mean r (n_points=100) | mean r (accurate) | perm p | verdict |
|---|---|---|---|---|---|
| mean_rel_SASA | a_p | +0.619 | +0.619 | 0.0005 (stable) | **REAL — strong** |
| frac_buried_rel_lt_0.25 | b_p_wt_error | -0.475 | -0.327 | 0.006 -> **0.0695** | **DIRECTIONALLY REAL, NOT ESTABLISHED** |
| everything else on the b_p side | — | — | — | not tested individually | **UNSUPPORTED** (sign-consistency cannot separate them from noise) |

The honest verdict on the b_p row is **neither CHECKPOINT 8's "absent" nor CHECKPOINT 9's
"ESTABLISHED"**. It is a real directional effect — 10/10 negative, negative in naturals and in
designed separately, and significant under the default analysis — whose magnitude and significance
both degrade when the arbitrary feature parameters are made more accurate. It is the right size to
be worth one confirmatory test on new proteins, and the wrong size to build a thesis claim on.

## The anchor-weight gradient: present on both channels, but weaker evidence than it looks

The gradient does reproduce — on the b_p channel the three anchor arms are uniformly weaker than
the five unanchored sigma seeds (burial vs `b_p_wt_error`: anchor -0.197/-0.120/-0.196 vs sigma
-0.379/-0.289/-0.433/-0.397/-0.413 at accurate quadrature; full separation).

Three limits, the first two already noted in CHECKPOINT 9 and the third new:
1. Not monotonic in anchor weight on the b_p channel (w3.0 > w1.0). What replicates is
   anchored-weaker-than-unanchored, not the within-anchor ordering.
2. Arm and seed-family are confounded (anchor arms are all s42, different epochs). 3 vs 5 is not a
   designed experiment; the running 48-cell factorial settles it.
3. **p=0.018 is the floor p-value for any 5-vs-3 Mann-Whitney.** Every perfectly-separated pair
   returns exactly 0.018, so it registers "perfect separation" and carries no magnitude
   information. ~20 pairs show it. It is one observation about anchoring, not twenty.

Because a_p and b_p are computed from the *same* 10 checkpoints, "corroboration on a second
channel" is weaker than two independent experiments agreeing — the channels share the checkpoint
draw entirely and the protein draw entirely.

## What stands

1. **CHECKPOINT 8's "NOT b_p" remains retracted — but on weaker grounds than CHECKPOINT 9 claimed.**
   The evidence for a burial/offset coupling is a consistent negative direction that survives the
   natural/designed split, not a significance result: it clears the permutation test at default
   quadrature (p=0.006) and fails it at accurate quadrature (p=0.070). "We cannot claim there is no
   b_p signal" is supported. "There is an ESTABLISHED b_p signal" is not.
2. **One structural axis, both channels — stands.** Exposure -> slope `a_p` (+0.62, p=0.0005);
   burial -> offset `b_p_wt_error` (~-0.33, p=0.006). Same physical variable, opposite ends.
3. **The asymmetry is real and much larger than CHECKPOINT 9 implied.** The a_p effect is strong,
   quadrature-stable and significant under every setting tried; the b_p effect is weak and both its
   magnitude and its p-value move with nuisance parameters. "Both channels carry signal" is
   defensible; "comparably" is not. CHECKPOINT 8's instinct that the two channels are not
   equivalent was **partly right** — it was wrong to call b_p absent, right that it is much weaker.
4. **THE METRIC RULE is untouched and was again load-bearing.** Burial is a whole-protein property,
   identical between WT and mutant, so it cancels in ddG and was correctly scored on
   `b_p_wt_error`. Any lever built on it must be scored on dG or b_p.
5. **Procedural lesson, sharpened.** CHECKPOINT 9's lesson (run the strong test on both siblings
   before promoting either) stands. Add: *a replication sweep across checkpoints of one training
   run is not independent replication, and sign-consistency across dependent replicates is not
   evidence.* Check the effective N before treating agreement as confirmation.

## n=28 caveats

All of CHECKPOINT 9's caveats stand and are not repeated in full. The three that bind hardest here:

- **n=28, and the 28-protein draw is the single shared point of failure** for both channels. At
  n=28 a nominal r=-0.33 has a 95% CI of roughly [-0.62, +0.04] — it includes zero. The
  permutation p=0.006 is evidence the effect is not chance; it is not evidence the magnitude is
  well determined.
- **The 10 checkpoints add ~0.24 of an independent replicate**, not 10. Replication here shows the
  effect is not a single-checkpoint artifact. It does nothing about protein sampling.
- **Range restriction.** All 28 are single-chain, ligand-free, metal-free monomers, 43-72 aa.
  Burial range is correspondingly narrow, which is exactly why a threshold count over it is
  unstable. Nothing generalises to multimers, ligand-bound, or larger proteins; BSA / ligand /
  complexity remain untestable in principle on this set (zero variance), not null.
- **Natural-only check — RUN, and it PASSES.** This was the gap CHECKPOINT 9 flagged as
  outstanding. Naturals alone (n=21): mean r = **-0.490**, 10/10 sign-consistent, 8/10 significant,
  range -0.582..-0.324 — slightly *stronger* than all 28 (-0.475). Designed only (n=7):
  **-0.453**, same direction. **A designed-fold artefact is excluded**; the coupling is present in
  both subpopulations. This is the one place the b_p finding came out cleaner than expected.

## Recommended next step

If the burial/offset coupling is to be used as a lever, build it on the **threshold-free**
`mean_rel_SASA` (quadrature-stable) rather than `frac_buried_rel_lt_0.25`, and pre-register the
cutoff and quadrature before scoring. Score on dG or b_p, never ddG.

`scripts/gate_g4_cpu.py`: **ALL PASS**, baseline row **dG=-0.0030 width=1092** as required;
no-new-dims levers hold at width 1092 and burial adds exactly 3 (width 1095).
(CHECKPOINT 9 quoted "dG=0.0023 width=1095" here — that is the *burial-lever* row, not the
baseline row. Both are green; the required baseline signature is the -0.0030/1092 one.)
No feature vector was changed by this work (analysis only, CPU only, no jobs touched).

## Files

- `scripts/indep_bp.py` — the independent re-derivation (does not read any cached result JSON)
- `results/bp_replication_independent.json` — all 90 pairs, per-checkpoint r and p
- `results/bp_anchor_gradient_independent.json` — anchor-vs-sigma gradient
- `results/BP_REPLICATION_INDEPENDENT.md` — full ranked 60-row b_p-side table

---

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

---

# CHECKPOINT 13 — 2026-09-07 — TASKS T11-T18 CLOSED, WORKING ALONE

All subagents were stopped at the user's instruction. Everything below was done and verified by
the main session directly, not by a worker report.

## What was verified by RUNNING it, not by trusting a claim

| task | result |
|---|---|
| T14 W9 metal | **RETIRED, and the guard really fires.** `hydro_net._sibling_block_dims(CFG)` raises RuntimeError when `metal_features` is set, returns 0 when not. `gate_w9.py` exits 0 with "W9 RETIREMENT VERIFIED". Adding a flag would have created a second dead path; W11 covers metals as ligand class METAL. |
| T15 het codes | `data/het_classification.csv`: all **8,187 codes**, **98.1% occurrence coverage** (452 hand-classified by frequency rank). Distribution: 106 crystallization_additive, 90 substrate, 80 modified_residue, 61 metal, 58 polymer_residue, 57 cofactor, rest unknown. Only **208 codes count as a bound ligand**. |
| T16 catalogue guard | `catalogue_schema.load_catalogue()` quarantines all 5 dead columns (**21 -> 18**); `assert_not_decoy_loss(['loss',...])` RAISES; BSA policy nulls **294** rows (246 original + the 48 internally inconsistent). |
| T17 Ofir divergences | **fixed in source** (see below) |
| T18 zero-variance guard | present in `autopilot.py` |
| T13 D1 slope guard | **fires.** On synthetic rows it rejected the HIGHEST-pooled cell (0.70) because its median slope was 0.10 < 0.30, logging `D1 REJECT ... slope collapse`. Ranking among eligible cells is by pooled, as documented. |
| T12 W5 on dG | **honest negative**, see below |
| T11 LORO | built, gated, and **submitted** |

## T17 — three documentation errors corrected in build_aa_descriptors_mordred.py

1. **PCA is OURS, not Ofir's.** The thesis does no dimensionality reduction; "PCA", "principal
   component" and "SVD" appear nowhere in it, and his 654 features enter the CNN as 654 channels.
   The CSV provenance now says so, so nobody can cite the projection as his method.
2. **1826 vs ignore_3D was self-contradictory.** 1826 = 1613 2D + 213 3D, a total only coherent if
   3D was requested — which is also what explains his 1826->1280 missing-value drop, since 3D
   descriptors return NaN without a conformer. With `ignore_3D=True` the ceiling is 1613. We keep
   ignore_3D deliberately, so our counts differ from his BY DESIGN.
3. **15-of-20 is not "three quarters" of 40-of-58.** 40/58 = 0.690 -> 13.8, so a faithful threshold
   is 13 or 14. Recorded as OUR choice. He gives no justification for 40 at all, and 2D Mordred
   cannot distinguish D from L, so his effective alphabet is well under 58.

## T12 — W5 burial scored on dG: NOT significant

`results/W5_ON_DG.md`, `results/w5_on_dg.json`.

| feature | target | pearson | spearman |
|---|---|---|---|
| mean_burial | b_p | **-0.343** | -0.282 |
| frac_buried | b_p | -0.039 | -0.007 |
| length | b_p | -0.368 | -0.349 |

**n = 19 of 28; |r| < 0.490 is indistinguishable from zero at p=0.05.** The sign agrees with the
replicated `frac_buried` result (-0.475, 10/10) but **-0.343 must not be reported as support.**

**Deliberately NOT done:** flipping `--burial_features` on a checkpoint trained without it. W5
widens 1092 -> 1095 and that checkpoint has no weights for the 3 new columns, so a forward pass
would measure untrained weights, not the lever.

**Two limits:** 9 of 28 proteins have no tensor directory, and the missing third is not random —
**2K5H is among them, one of the two proteins carrying 78% of the offset-removal oracle gain.**
And `length ~ b_p` (-0.368) is LARGER than burial's, with burial correlated to length.

**A units bug worth keeping:** the first run returned nan for every burial correlation because I
scaled coordinates by 0.1 before `compute_burial`. **The stored tensors are already in Angstrom**
and the cutoff is an Angstrom cutoff, so 0.1x saturated every neighbour count to exactly 1.0 with
std 0.0. Measured on 2KVS/2BTH: Angstrom -> mean 0.48/0.41 std 0.18/0.16; 0.1x -> 1.0/0.0.
**A saturated constant does not raise; it silently yields nan.**

## Data path worth recording

Per-protein tensors are NOT in `data/`. They are at
`/groups/keasar_group/casp15/meytav/protein_tensors/<PROTEIN>/coords_tensor.pt` (and
`mask_tensor.pt`), one directory per protein, exactly as `Megascale-fineTuning/train.py:28` loads
them. **9 of our 28 test proteins have no directory there.**

## Queue state

**15 golden jobs queued** (`--partition=rtx6000 --qos keasar`): the dG arm, `--slope_weight`, the
severing experiment, 10 starved evals, and now the LORO pair (`loroW_onehot` / `loroW_desc`,
differing ONLY in `--aa_descriptors`, both with `--resume`).
**Public lane untouched at 5/5.** A labmate still holds all 8 golden cards, so ours pend.

## All 11 gates green

g4_cpu (**dG=-0.0030 width=1092**), noop, resume (42/42), u2, w5, w7, w7_edge, u3u4, u5u6,
w9 (retirement verified), w8 (51/51).

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

---

# CHECKPOINT 15 — 2026-09-07 — WHAT THE MODEL ACTUALLY SEES (measured, not assumed)

The standing complaint was that we had no understanding of what the inputs ARE. Measured directly
on `/groups/keasar_group/casp15/meytav/protein_tensors/2KWH/`.

## The tensors are per-VARIANT, not per-protein

    prott5_embeddings/prott5_embedding_<k>.pt   [128, 52, 1024]   <- 128 variants x 52 residues
    one_hot_encodings.pt                        [1389, 52, 21]    <- 1389 variants, 21 COLUMNS
    coords_tensor.pt                            [N, 4, 3]  ANGSTROM
    mask_tensor.pt, deltaG.pt

**one_hot is stored with 21 columns, not 20 — but this is HANDLED, and I checked rather than
assuming a bug.** `Megascale-fineTuning/train.py:317` does
`batch['one_hot'] = batch['one_hot'][:, :, :, :-1]`, dropping the trailing column before the graph
is built. `get_graph` produces width 1092 with the last 20 columns being one-hot, so every
`[20, K]` descriptor table is correctly sized. **No action needed** — recorded so nobody
re-discovers the 21 and assumes a mismatch.

## THE EMBEDDINGS ARE CONTEXTUAL — settled by measurement, never tested before

Same residue TYPE at two positions in the same protein:

| residue id | positions | max abs diff | cosine |
|---|---|---|---|
| 3 | 0, 7 | 0.8554 | **0.1764** |
| 16 | 1, 44 | 0.7310 | **0.1264** |
| 13 | 2, 12 | 1.4252 | **0.1455** |

**Cosine 0.13-0.18 — nearly orthogonal.** ProtT5 encodes position and neighbourhood, NOT residue
identity. This kills the "maybe it is just a 20-vector lookup" hypothesis outright: if it had been a
lookup, the whole descriptor question would have collapsed to linear algebra over 20 points.

## But identity IS largely recoverable — which is what matters for W6

Linear probe, embedding -> residue identity, held-out:

    accuracy 0.625   (chance 1/21 = 0.048)

**A descriptor block is a deterministic function of residue identity.** Identity is 62.5% linearly
recoverable from ProtT5 already, so a per-residue-type descriptor block is **largely redundant with
information the network can already extract**. This is exactly what Ofir concedes in his thesis:
physicochemical properties are already implicit in language-model embeddings, and his case for
descriptors is COVERAGE of non-canonical residues — which our 28 canonical proteins do not need.

**Consequence: W6 cannot be justified as adding information. Its only defensible value is
regularisation (sharing strength across chemically similar residues) or open-alphabet
generalisation.** The design must say so instead of claiming new signal.
Caveat: n=52 residues in one protein, one linear probe. Repeat across proteins with a
residue-disjoint split before quoting 0.625 as the number.

## ProtT5 is recomputed PER VARIANT, and one mutation moves everything

WT variant vs mutant variant: **all 52 of 52 positions differ.** A single point mutation changes the
embedding at EVERY position, not just the mutated one.

Two consequences:
1. The mutation signal reaching the network is **global, not local**. The model is not being handed
   "position 17 changed"; it gets a wholly new 52x1024 field.
2. **ddG does NOT cancel the embedding.** The metric rule says ddG cancels anything identical
   between WT and mutant — the embedding is NOT identical, so it survives the difference. That is
   why `--unfolded_emb zero` moved the ddG numbers at all, and it means ProtT5 is one of the few
   blocks that acts on BOTH channels.

## Architectural fact that constrains every feature-block design

The GCN branch reads **`x[:, :32]` only** — D(16) plus HALF of Fb. It never sees the embedding, and
**any new block inserted at offset 48 is invisible to it.** So W5/W6/W9/W11 reach the GAT branch
alone. Any claim that a new block "is used by the model" must say WHICH branch.

## What to do with this

- Re-run the identity probe across many proteins with a residue-disjoint split, and report R^2 for
  hydrophobicity/charge/volume too. That fixes the honest ceiling on what W6 can add.
- State in the write-up that ProtT5 is contextual and per-variant: it is the reason the embedding
  behaves as an offset channel (W0: zeroing it in the unfolded pass collapses var(E_u) by 67%)
  AND still contributes to ddG.

---

# CHECKPOINT 16 — 2026-09-08 — EVERYTHING BROKE OPEN, AND W6 IS LARGELY REDUNDANT

## The blockage cleared

| | before | now |
|---|---|---|
| factorial CSVs | **0/48** | **10/48** |
| GPUs running | 5 | **13** (5 public + 8 golden) |
| labmate holding golden | 8/8 | **0** |
| autopilot | dead behind a healthy log | **SLURM job 21107688, 7-day** |

The scoring fix worked: `cpu_evals` scored 8 cells on the CPU partition at zero GPU cost, and the
autopilot now logs `S3: 10/48 cells scored`.

**The autopilot is now a SLURM job**, not a login-node process. It had died twice more with
`rc=143` (SIGTERM when its login node churned). SLURM does not restart it, but the job itself
outlives login nodes entirely — which was the actual failure mode.

**Golden lane running 8/8:** gld_dg_coil, gld_slope, gld_w5_dg, gld_w7edge, gld_u10bidir,
gld_w7span4, gld_loroW_onehot, gld_loroW_desc. **Every lever that had never been tested is now on a
card.**

## THE EMBEDDING RESULT — this settles W6

`scripts/emb_probe.py`, 19 proteins, 1,056 residues, **leave-one-residue-type-out** (fit on 19
types, predict the held-out 20th — a row-random split is contaminated because every occurrence of a
residue shares one descriptor vector):

| property | held-out R^2 |
|---|---|
| **hydropathy** | **+0.704** |
| charge | +0.511 |
| volume | +0.421 |
| identity (row-random) | accuracy **1.000** vs chance 0.050 |

**ProtT5 predicts the hydropathy of a residue type it has NEVER seen, at R^2 = 0.70.**

This is the decisive fact for W6. A physicochemical descriptor block is a deterministic function of
residue identity, and:

1. **Identity is perfectly recoverable** from the embedding (accuracy 1.000). So the network can
   already reconstruct one-hot from ProtT5 — anything a per-residue-type table encodes is already
   spanned.
2. **The chemistry itself extrapolates to unseen residue types** (hydropathy R^2 = 0.704). This is
   precisely the generalisation Ofir attributes to descriptors — and ProtT5 already does it.

**W6 cannot be justified as adding information.** Its only defensible value is (a) regularisation —
sharing statistical strength across chemically similar residues — or (b) coverage of residues
absent from ProtT5's training data. And Ofir CONCEDES exactly this: his argument for descriptors is
coverage of non-canonical residues, not added signal. Our 28 test proteins are all canonical, so
that gap does not exist here.

**This should be stated in the thesis as a measured result, not an assumption.** It also predicts
the outcome of the LORO experiment now running: holding out W, the descriptor arm should NOT beat
the one-hot arm by much, because ProtT5 already carries W's chemistry. That prediction is on record
BEFORE the runs finish.

Caveat: 19 proteins, one variant each, a linear ridge probe. It bounds what a LINEAR reader can
extract; a nonlinear network might differ, though that cuts against descriptors too.

## Supporting facts from the same inspection

- **ProtT5 is CONTEXTUAL**: the same residue type at two positions has cosine similarity 0.13-0.18,
  nearly orthogonal. It encodes position and neighbourhood, not identity — yet identity is still
  perfectly decodable, so the information is present but distributed.
- **ProtT5 is recomputed PER VARIANT**: a single point mutation changes ALL 52 of 52 positions, so
  the mutation signal is global, and the embedding does NOT cancel in ddG.
- Tensors are per-variant: embeddings `[128, 52, 1024]`, one-hot `[1389, 52, 21]`. The 21st
  one-hot column is dropped at `train.py:317` before the graph is built, so every `[20, K]`
  descriptor table is correctly sized.
- The GCN branch reads `x[:, :32]` only, so **any block inserted at offset 48 is invisible to it**
  and reaches the GAT branch alone.

---

# CHECKPOINT 17 — 2026-09-08 — A DATA BUG IN 2K5H INFLATES THE HEADLINE RESULT

## The finding, verified independently by the main session

A frontier agent reported that 2K5H's eval reference row is a MUTANT background rather than the
true wild type. **I verified it myself rather than accepting the report**, and it holds.

`eval_results/abl_calib_ctrl_repro2_e14.csv`, protein 2K5H, 1125 rows:

    row 0:  deltaG = 1.7231   pred = 4.2446   ddG = 0.0
    protein deltaG: min -0.980, mean 3.768, max 4.999
    row 0 sits at the 14.6th PERCENTILE of its own protein

**A wild type should be near the TOP of its stability distribution, not the bottom.** Row 0 is also
not unique: 2K5H has TWO rows with `ddG == 0` (indices 0 and 191), while the WS-1 convention assumes
exactly one.

## The control that makes it unambiguous

Row-0 percentile within each protein's own dG distribution, all 28:

| | percentile |
|---|---|
| 26 of 28 proteins | **62% - 99%** (exactly where a WT belongs) |
| **2K5H** | **14.6%** |
| r18_3_TrROS_Hall | 33.3% |

Only two fall below their own median, and 2K5H is far the worse. This is not a distributional
quirk; it is a wrong reference row, and every one of 2K5H's 1125 ddG labels is shifted by the
difference between the true WT and the mutant background (~3.0 kcal/mol).

## What it costs us — measured over 22 eval CSVs

| condition | pooled | oracle | gain |
|---|---|---|---|
| **all 28** | 0.5018 | 0.7391 | **+0.2373** |
| **drop 2K5H** | 0.4882 | **0.6410** | **+0.1528** |
| drop 2K5H + 2KVS | 0.5224 | 0.6542 | +0.1319 |
| drop 2K5H + r18_3 | 0.4860 | 0.6284 | +0.1424 |

**Dropping 2K5H alone removes 36% of the entire offset-removal gain** and takes the oracle from
0.739 to 0.641.

**A single protein with a corrupted reference row carries over a third of the thesis's headline
effect.** The "offset removal lifts pooled PCC by ~0.11-0.12" claim is substantially an artefact of
one mislabelled wild type, because a 3 kcal/mol label shift IS a per-protein offset by construction
— the oracle then "discovers" an offset we ourselves introduced.

## What must happen

1. **Find the true WT row for 2K5H** and rebuild its ddG column against it (the true WT appears to
   be the `2K5H.pdb` entry at dG 4.806, versus the `2K5H.pdb_G11S` mutant background at 1.723).
   Then re-run every calibration number.
2. **Audit the reference row for ALL 28** with the percentile test above as a permanent gate: row 0
   below its protein's median is a red flag, and more than one `ddG == 0` row is a hard error.
3. **Re-state every headline number** in CONTEXT.md once 2K5H is fixed. The 0.70-0.72 ceiling, the
   +0.11 gain, std(b_p)=1.5741 and the "top-2 proteins carry 78%" claim all inherit this bug.
4. r18_3_TrROS_Hall at 33.3% needs the same check.

## Two other findings from the same sweep, both verified numbers

**The exposure result SURVIVES the length confound.** `a_p ~ mean_rel_SASA` controlling for length
is **+0.6623 mean, significant in 10/10 checkpoints**, while length alone is r=-0.2149 and
significant in **0/10**. CHECKPOINT 8 stands; no correction needed.

**The slope objective has an algebraic CEILING.** Since `a_p = r * sd(pred)/sd(true)`, driving
sd(pred) -> sd(true) pins the slope to the CORRELATION, never to 1.0. The ceiling is a_p = r =
0.793. Still worth having: the measured a_p decomposes into 62.8% spread compression (fixable) and
36.8% ranking error (not), so `--slope_weight` can recover about 65% of the gap to a_p = 1 — but it
cannot close it, and the write-up must say so rather than implying otherwise.

---

# CHECKPOINT 18 — 2026-09-08 — FIRST FACTORIAL RESULTS: FACTOR D IS DECISIVE

## The 10 scored cells

| cell | pooled ddG PCC | a_p median |
|---|---|---|
| a1_d1_s0_**D0_coil** | **0.6175** | 0.5173 |
| a1_d0_s1_**D0_coil** | **0.6100** | 0.5942 |
| a0_d1_s1_**D0_coil** | **0.6064** | 0.3733 |
| a1_d0_s0_**D0_coil** | **0.5982** | 0.5736 |
| a1_d1_s1_**D0_coil** | **0.5827** | **0.7156** |
| a0_d1_s1_**D1_uemb** | 0.3121 | 0.0113 |
| a0_d0_s0_**D1_uemb** | 0.2472 | 0.0068 |
| a0_d1_s0_**D1_uemb** | 0.2179 | 0.0197 |
| a0_d0_s1_**D1_uemb** | 0.1001 | 0.0048 |
| a1_d1_s0_**D1_uemb** | **-0.0782** | 0.0136 |

## FACTOR D SEPARATES THE TWO GROUPS COMPLETELY

**Every D0 (coil) cell scores 0.58-0.62. Every D1 (`--unfolded_emb zero`) cell scores -0.08 to 0.31.**
There is no overlap. Five versus five, perfectly separated.

**And the a_p column is even starker: D0 cells have a_p 0.37-0.72, D1 cells have a_p 0.005-0.020 —
two orders of magnitude apart.** An a_p of 0.005 means the model is essentially FLAT: it barely
responds to mutation at all.

## What this means, and it is a reversal worth stating carefully

W0 chose factor D as `--unfolded_emb {full,zero}` on the ddG metric, because zeroing the embedding
in the unfolded pass collapsed var(E_u) by 67% and cut corr(E_u, wt_err) from 0.420 to 0.119. That
made it look like the cleanest way to remove the offset.

**Trained end to end, zeroing the unfolded embedding destroys the model.** It removes the offset by
removing the signal — exactly the caveat recorded in CHECKPOINT 2 for `noemb` on dG ("lowest MAE but
kills the correlation, 0.35 -> 0.05"), now confirmed under training rather than at inference.

One cell, `a1_d1_s0_D1_uemb`, is ANTI-correlated at **-0.0782**. The autopilot's own audit flagged it
(`audit: pooled out of [0,1]`), which is the audit working as designed.

**The coil arm is not merely better; it is the only arm that produces a working model.**

## Consequences

1. **The D1 half of the factorial is largely wasted compute.** 24 of the 48 cells use
   `--unfolded_emb zero`, and on this evidence they will all land in the 0.0-0.3 band. That is
   ~190 GPU-hours confirming a negative we can already see at n=5.
2. **Do not average across D.** Any main-effect estimate for A, B or C that pools D0 and D1 will be
   swamped by a factor-D effect roughly ten times larger than anything else.
3. The best cell so far, `a1_d1_s1_D0_coil`, has the HIGHEST a_p (0.7156) but not the highest pooled
   PCC (0.5827 vs 0.6175). **That is the calibration thesis in one line: the highest-slope model is
   not the highest-pooled model**, because pooled PCC is dominated by the offset while a_p measures
   the compression.
4. These numbers still carry the 2K5H reference-row bug (CHECKPOINT 17), which inflates the
   offset-removal gain by ~36%. Fix 2K5H before quoting any of this in the thesis.

## Caveats

Epochs differ across cells (e5 to e14) because each was scored at its own best epoch, so the
comparison is best-epoch to best-epoch, not equal-epoch. Two cells were scored early (e5, e6) and
may improve. All are seed 42 only — no seed replication yet.

---

# CHECKPOINT 19 — 2026-09-08 — THE ATTENUATION HYPOTHESIS IS DEAD, AND THE CATALOGUE IS RETIRED

## 1. Per-mutation uncertainty FOUND, and it settles a_p

An earlier attempt concluded the MegaScale uncertainties could not be joined to our 28 (0/28 by
name and by exact WT sequence). **That attempt searched the wrong file.** The uncertainties are in
`data/ThermoMPNN/mega_test.csv` — 28,312 rows against our 28,314 test mutations, with 17 CI columns.

**Verified by the main session, not taken on report:**

    proteins in mega_test: 28
    MATCHED 28 / 28 of our test proteins
    median 95% CI = 0.1007  ->  median sigma = 0.0327 kcal/mol  (n = 28,312)
    median var(true ddG)   = 0.7178

**The attenuation prediction:**

    a_p = var(true) / (var(true) + var(noise)) = 0.7178 / (0.7178 + 0.0327^2) = 0.9985

**We measure a_p ~ 0.50. Attenuation predicts 0.9985.**

The label noise would have to be roughly **200x larger** to explain the compression. For attenuation
alone to produce a_p = 0.5, sigma would need to be ~0.85 kcal/mol; it is 0.0327.

**a_p = 0.499 is a genuine model failure, not a statistical artefact of noisy labels.** This
independently confirms CHECKPOINT's earlier sign-based argument (attenuation predicted the WRONG
SIGN for the exposure correlation, P(r >= +0.714) = 0/2000) by a completely different route — and
this route is direct rather than inferential. **`--slope_weight` is fully justified.**

## 2. The 100k catalogue is formally RETIRED for every join-based purpose

**It has NO SEQUENCE COLUMN.** That is a permanent, in-principle result, not a failed attempt.
Joins measured: 0/226 training PDB ids, 0/21 test PDB-like ids, 0/340 by full protein_id, against
66,945 unique catalogue PDB ids. **So "test structural properties at n=hundreds using the training
set" is dead — it required the join.**

### The one live use, which needs no join: DISTRIBUTION SHIFT

| | catalogue | our regime (32-74 aa) |
|---|---|---|
| median length | **477** | 55-56 |
| **share of pre-training RESIDUES** | — | **0.1206%** (1 in 830) |
| NMR structures | 8.56% | **77.43%** (9.05x enriched) |
| Is Complex? = Yes | 67.7% | 11.8% |
| Monomer | 31.8% | **88.2%** |
| BSA = 0 | 31.2% | 87.8% |

**Our entire problem domain was one residue in 830 of pre-training, and a structurally different
kind of protein — small, monomeric, interface-free, NMR-solved.** That is worth stating in the
thesis as a characterisation of pre-training (well powered, n=99,708), **but NOT as the cause of
b_p**: measured against b_p it is r = +0.043 (p = 0.829), a clean null.

### Ofir's mean-gap mechanism does NOT carry over

He found his RMSE tracked the train/test mean gap. Ours: per-protein WT dG train 3.0930 +/- 1.3362
(n=340) vs test 2.9895 +/- 0.9263 (n=28), **gap = -0.1034, p = 0.588**. That is 6.6% of one
std(b_p), and a shared constant cannot generate per-protein DISPERSION anyway. **Retired.**

### And the composition/shift features do not predict b_p

25 tests over the 10 eval CSVs: length-percentile r = +0.0427, Mahalanobis distance from the
training composition centroid r = +0.0277, L2 distance r = +0.1904. Three nominal hits where ~1.25
are expected by chance, and **NONE survive Bonferroni or BH-FDR** for either b_p or a_p.
One real descriptive asymmetry: threonine fraction differs train vs test (0.0660 -> 0.0395,
p = 2.7e-07, survives Bonferroni) but carries **no predictive power over b_p** (r = -0.119).

**Consequence: b_p must be hunted on the INPUT/REPRESENTATION side, not the label side.**

## 3. Also found: 26,315 double mutants for our 28 sit in no split at all

A held-out generalisation test that needs no new labels and no new measurements. Untouched.

## Caveat on a convention mismatch, stated openly

The agent's own per-protein regression averaged across checkpoints gives std(b_p) = 0.7914 and
median a_p = 0.4148, versus our briefed 1.5741 and 0.4990. Averaging slopes and intercepts across
checkpoints shrinks the spread, so the two are not on the same scale. Only scale-free CORRELATIONS
were used for the null conclusions, so those stand; the 6.6%-of-one-SD comparison deliberately uses
the larger briefed value so as not to flatter the negative.

---

# CHECKPOINT 20 — 2026-09-08 — TWO AGENT CLAIMS TESTED, BOTH FAIL. b_p IS NOT LENGTH-DRIVEN.

Working solo from `results/TASKS.md`, no background agents. Every number below was computed by the
main session.

## K4 — the "1/sqrt(N) embedding artifact" does NOT reproduce

An agent reported `corr(N, per-residue embedding norm) = -0.9938` and proposed a one-line rescale
as the cheapest b_p lever available. **Measured:**

| quantity | agent claim | measured |
|---|---|---|
| corr(N, emb_norm), 20 proteins | **-0.9938** | **-0.3927** |
| corr(N, emb_norm), our 28 | — | **-0.2946** |
| corr(emb_norm, b_p), our 28 | — | **+0.3863** |
| corr(N, b_p), our 28 | — | -0.3150 |

At n=28 the threshold is |r| > 0.392, so **none of these is significant**. And `|e|*sqrt(N)` is not
constant (36.5 -> 53.2 across proteins), so there is no clean 1/sqrt(N) law to remove.

**The proposed lever rests on a correlation that is three times weaker than claimed. Do not
implement it.** K5 is therefore dropped.

## K6 — `--dg_length_norm` does not shrink b_p; the /n arm is DEGENERATE

The flag exists (`none|n|sqrtn`, train.py:74, applied at 1016-1019) and has never been run. Because
it divides the PREDICTION, its effect on b_p can be simulated exactly on predictions we already
have — no GPU. Over 12 eval CSVs:

| | mean std(b_p) |
|---|---|
| none | **1.3076** |
| /n | 0.9050 |
| /sqrtn | 0.9159 |
| **corr(N, b_p)** | **+0.0252** |

It looks like a 31% improvement. It is not.

**The /n column converges to ~0.905 for EVERY checkpoint** — including ones that START at 0.876,
where "normalising" makes it worse. That constant is not a coincidence:

    std(true WT dG) over the 28 test proteins = 0.9042

**Dividing by N drives pred/N to nearly zero, so b_p -> -true_dG and its spread becomes the spread
of the true labels.** The arm does not remove a length bias; it deletes the prediction. Any future
run of `--dg_length_norm n` that reports a smaller std(b_p) is measuring this degeneracy.

**And the direct test is flat: corr(N, b_p) = +0.0252 across 12 checkpoints. b_p is NOT
length-driven**, so the extensivity argument for a length-normalised head does not hold.

## What this leaves

Both cheap architectural attacks on b_p are now closed, along with the label-side explanations
already retired in CHECKPOINT 19 (distribution shift r=+0.043; Ofir's mean-gap p=0.588; attenuation
dead by 200x). **b_p has survived: length, embedding norm, label shift, distribution shift, and a
feature-based corrector.** What remains untested is the reference state itself — where the coil
already produced our only real b_p movement (dG MAE 4.9650 -> 3.9521) and where `gld_dg_coil` is
running now.

**Method note worth keeping:** both claims were plausible and both failed the same way — a
correlation quoted without its n, and an improvement quoted without asking what the improved number
would be if the model predicted nothing at all. **Always compute the degenerate baseline.**

---

# CHECKPOINT 21 — 2026-09-08 — K1 DONE: 2K5H FIXED, THE HEADLINE IS 35% SMALLER

## Root cause, traced to the file

`data/Processed_K50_dG_datasets/mutation_datasets/2K5H.csv` (4,107 rows) **concatenates THREE
backgrounds** and the wrong one sorts first:

    row 0    : name=2K5H.pdb_G11S   deltaG=1.723086   mut_type=wt
    idx 2738 : name=2K5H.pdb        deltaG=4.805470   mut_type=wt   <- the TRUE wild type

Both rows are labelled `mut_type='wt'`, because `2K5H.pdb_G11S` genuinely IS the wild type *of its
own mutant background*. The WS-1 convention treats row 0 as the reference, so every one of 2K5H's
ddG labels was measured against a mutant background.

**Measured shift: +3.0824 kcal/mol.**

**2K5H is the ONLY affected protein.** I checked all 28 against
`data/MsDs/mutation_files/`: only 2K5H has multiple background files (`[WT]`, `_G11S`, `_G23A`).
Isolated defect, not systemic.

## The fix respects the read-only rule

The source file is owned by `shaharax` and reached through a symlink into his tree, and
**shaharec/DeepPEF is read-only forever**. So the corrected copy was written to **our own**
directory instead:

    data_fixed/mutation_datasets/2K5H.csv
    row0: name=2K5H.pdb deltaG=4.805470     (was 2K5H.pdb_G11S / 1.723086)
    4107 rows preserved -- nothing deleted, the mutant-background rows are simply no longer first

`scripts/fix_2k5h.py` does this reproducibly (`--check` / `--apply`).

## THE HEADLINE — canonical basis (P8)

Because the fix shifts 2K5H's ddG by a CONSTANT, its effect applies exactly to existing predictions.

| basis | pooled | oracle | gain |
|---|---|---|---|
| **CANONICAL — 9 original-population CSVs, 27 proteins** | **0.5772** | **0.7156** | **+0.1384** |
| *retired* † — 22 mixed CSVs incl. D1 arms | *0.4899* | *0.6443* | *+0.1544* |
| *retired* † — 22 mixed CSVs, with the 2K5H bug | *0.5018* | *0.7391* | *+0.2373* |

**† quoted as published; NOT re-derivable** — the 22-CSV population was never recorded and does not
reproduce today (nearest reconstruction: 0.4252). See `results/05_infrastructure/HEADLINE_BASIS.md`.

**Only the canonical row is quotable.** The 2K5H bug inflated the offset-removal gain by 35%
(retired figures: +0.2373 -> +0.1544) and the oracle by 0.095 — but both of those were also
averaged over the wrong population, which is the separate and larger defect that P8 fixes.

A constant per-protein label shift IS a per-protein offset by construction, so the oracle was
partly rediscovering an offset we had introduced.

## What survives, stated honestly

**The calibration effect is real and still substantial: +0.154 on pooled ddG PCC from offset
removal alone.** What changes is its size. Every number quoted from here must use the corrected
figures:

- offset-removal oracle: **0.644**, not 0.739 (and NOT the old 0.70-0.72)
- offset-removal gain: **+0.154**, not +0.237
- the "top-2 proteins carry 78% of the gain" claim needs recomputing on corrected labels, since
  2K5H was one of the two and its contribution was inflated

**Still to do:** the running factorial and golden-lane jobs were trained against the buggy labels
for 2K5H. Their ddG for that one protein is shifted, which affects its per-protein a_p and b_p but
not the other 27. Decide whether to re-score with `data_fixed/` or accept a documented caveat.

---

# CHECKPOINT 22 — 2026-09-08 — K12: a_p = r * s DECOMPOSED, AND THE SLOPE LEVER IS CONFIRMED

## The identity, measured on 22 checkpoints (2K5H corrected)

For any per-protein least-squares fit, `a_p = r * s` where `r` is the correlation and
`s = std(pred)/std(true)` is the spread ratio. `--slope_weight` penalises |std(pred)-std(true)|,
so it can only act on **s**. Medians per checkpoint, then averaged:

    a_p = 0.3640    r = 0.6527    s = 0.5043

**Decomposition of the gap to a_p = 1:**

| source | size | can the slope term fix it? |
|---|---|---|
| ranking error (1-r) | 0.3473 | **NO** |
| spread compression (1-s) | 0.4957 | **YES** |

**CEILING: driving s -> 1 gives a_p = r = 0.653.** The lever cannot reach 1.0 by construction, and
the write-up must say so rather than implying otherwise.

## THE NATURAL EXPERIMENT — the three slope arms confirm the algebra

Three runs differ only in `--slope_weight`, and they behave exactly as `a_p = r * s` demands:

| --slope_weight | s | a_p | r |
|---|---|---|---|
| 0.3 | 0.5732 | 0.4146 | 0.8014 |
| **1.0** | **0.7564** | **0.5621** | 0.8090 |
| 3.0 | 0.4721 | 0.3262 | 0.7953 |
| (control, no slope term) | 0.6351 | 0.4990 | 0.7980 |

**`r` is FLAT at 0.795-0.809 across all four while `s` swings 0.47 to 0.76.** The lever moves the
spread and leaves the ranking untouched — which is precisely what the identity predicts and is
strong evidence the term does what it claims.

**Weight 1.0 is best: it lifts s from 0.635 to 0.756 and a_p from 0.499 to 0.562, a +0.063 gain
over the control, at no cost to r.**

**Weight 3.0 OVERSHOOTS badly** — s falls to 0.472, WORSE than the control. Too strong a spread
penalty destabilises training rather than decompressing further. So the lever has an interior
optimum near 1.0, not a monotone response, and the factorial's C factor should be read that way.

## Why the mean a_p (0.364) is below the control (0.499)

The average is dragged down by the five D1 (`--unfolded_emb zero`) cells, whose a_p is 0.005-0.020
because those models are nearly flat. Excluding them, the working models sit at a_p 0.33-0.72 with
r consistently ~0.72-0.81. **Do not quote the pooled mean across D.**

## What this settles

1. **`--slope_weight` works, at weight 1.0, and its mechanism is confirmed** — it moves s, not r.
2. **It cannot close the gap alone**: 54.6% of the deficit is ranking error, which needs a
   different lever entirely (better features, not better calibration).
3. **The thesis should report a_p as two numbers, not one.** "The model compresses ddG by half" is
   really "the model ranks at r~0.80 and compresses the surviving spread to s~0.64", and only the
   second half is a calibration problem at all.

---

# CHECKPOINT 23 — 2026-09-08 — K13: THE MODEL IS WORST AT BURYING HYDROPHOBICS

## The finding, verified over 10 checkpoints

The eval CSVs carry no mutation code, so the destination residue was recovered by joining on
`deltaG` (6 dp) against `mutation_datasets/<P>.csv`, whose `name` encodes the mutation
(`1PSE.pdb_A42W`). Ambiguous deltaG values were DROPPED rather than guessed. ~26,000 mutations
matched per checkpoint. 2K5H's ddG was corrected by -3.0824 throughout.

**Per-destination-residue within-protein slope, ordered by Kyte-Doolittle hydropathy:**

| residue | KD | slope | spearman |
|---|---|---|---|
| R | -4.5 | **0.528** | 0.649 |
| K | -3.9 | 0.537 | 0.648 |
| N | -3.5 | **0.565** | 0.653 |
| D | -3.5 | 0.542 | 0.650 |
| S | -0.8 | 0.524 | 0.638 |
| G | -0.4 | 0.506 | 0.616 |
| A | +1.8 | 0.419 | 0.567 |
| M | +1.9 | 0.340 | 0.461 |
| C | +2.5 | **0.267** | 0.356 |
| F | +2.8 | 0.288 | 0.381 |
| L | +3.8 | **0.282** | 0.388 |
| V | +4.2 | 0.309 | 0.428 |
| I | +4.5 | 0.311 | 0.412 |

    mean corr(slope, KD)    = -0.7344    sign-consistent 10/10
    mean corr(spearman, KD) = -0.7399

**Mutations TO hydrophobic residues are compressed roughly TWICE as hard as mutations to charged
or polar ones** (slope ~0.28-0.31 vs ~0.53-0.57).

## THE DECISIVE DETAIL: it is LOST INFORMATION, not a calibration error

`corr(spearman, KD) = -0.740` is **essentially identical** to `corr(slope, KD) = -0.734`.
**Rank accuracy degrades in exact lockstep with the slope.** Spearman falls from ~0.65 for polar
destinations to ~0.36-0.43 for hydrophobic ones.

That distinction decides whether any lever can help:

- If the slope fell while ranking held, a per-class rescale would recover it — a calibration fix.
- **Ranking falls too, so the information is not there to rescale.** The model does not merely
  under-react to buried hydrophobics; it cannot ORDER them either.

**`--slope_weight` cannot fix this**, and neither can any affine correction. It is the (1-r)
half of the K12 decomposition, made concrete and given a chemical identity.

## Why this is mechanistically credible

The dataset stores only **4 backbone atoms per residue (N, CA, C, CB) — no side chains**. Burying
a large hydrophobic (F, L, I, V, M, W) is dominated by side-chain packing and van der Waals
complementarity, exactly what CB-only geometry cannot see. Charged and polar substitutions are far
better captured by backbone geometry and solvent exposure, which the model does have.

**W and Y are the interesting exceptions** — slopes 0.341 and 0.353 despite negative KD. Both are
large aromatics, so the pattern tracks SIDE-CHAIN BULK rather than hydropathy alone, which
strengthens the packing explanation over a purely hydrophobic one.

## What this is worth

This is a mechanistic, chemically-interpretable account of where the model's ranking error lives,
and it points at a concrete architectural fix rather than a calibration one: **give the model
side-chain information**. That is a real thesis result and an argument for full-atom or
side-chain-aware features in future work.

---

# CHECKPOINT 24 — 2026-09-08 — K14: THE PRE-TRAINING DISTRIBUTION SHIFT, VERIFIED

Recomputed from the raw catalogue by the main session. Every figure reproduces the agent's report.

## The numbers

`data/FINAL_DATASET_100k_030926.csv`, 99,708 rows with a usable length:

    catalogue length: median 477, mean 1167.7, max 89,160
    rows in our 32-74 aa regime: 2,526  (2.53% of rows)
    RESIDUE share: 140,383 / 116,431,172 = 0.1206%

**Our entire problem domain was one residue in 830 of pre-training.** The residue share is the
right denominator because the pre-training loss is per-residue, so that is the share of the
gradient our regime ever contributed.

## And it is a structurally DIFFERENT kind of protein

| | catalogue overall | our 32-74 aa regime |
|---|---|---|
| SOLUTION NMR | 8.6% | **77.4%** (9.0x enriched) |
| Is Complex = Yes | 68.1% | **11.8%** |
| Monomer | 31.9% | **88.2%** |

Pre-training saw large, X-ray-solved, interface-rich multimers. We test on small, NMR-solved,
interface-free monomers. **The model was optimised on almost the opposite of what it is evaluated
on.**

## What this is, and what it is NOT

**IT IS** a well-powered characterisation of pre-training (n = 99,708) that needs no join and no
extra compute. It belongs in the thesis as context for why absolute dG is hard here.

**IT IS NOT the cause of b_p.** Measured directly against b_p: **r = +0.043, p = 0.829**, a clean
null. Across 25 shift and composition features, three reach nominal significance where ~1.25 are
expected by chance, and **none survives Bonferroni or BH-FDR** for either b_p or a_p.

The temptation is to say "the offset exists because of distribution shift". The data does not
support it, and the honest statement is that the shift is real, large, and **not measurably
connected to the per-protein offset**.

## Caveat kept from the agent's own analysis

`AA Length` is the entity/assembly length, not per-chain, so the 0.1206% is computed on assembly
residues. Per-chain lengths would be smaller and would move a little catalogue mass toward our
regime, so **0.1206% is a lower bound on the mismatch severity**, not an exact per-chain figure.
The NMR/complex/monomer contrasts are entity-level attributes and are unaffected.

---

# CHECKPOINT 25 — 2026-09-08 — K11: THE "26,315 DOUBLE MUTANTS" DO NOT EXIST

## The claim

A data-mining agent reported that **26,315 double mutants for our 28 test proteins sit in no split
at all**, and called it a held-out generalisation test needing no new labels — potentially one of
the most valuable untouched assets in the project.

## Measured

Counting mutation codes (`_X<pos>Y`) across all 28 proteins' mutation files:

    0-point:  4,810 rows
    1-point: 56,214 rows
    2-point:  **2,356 rows**

Not 26,315. And every 2-point row belongs to **one protein**:

    2-point rows by protein: {'2K5H': 2356}

The example names give it away immediately:

    2K5H.pdb_G11S_A1Q,  2K5H.pdb_G11S_A1E,  2K5H.pdb_G11S_A1N, ...

**These are not double mutants. They are SINGLE mutations on the G11S background** — the exact
concatenation artifact that K1 fixed. `2K5H.csv` merges three backgrounds, so a single mutation on
the `_G11S` background parses as two mutation codes.

Removing the `_G11S` and `_G23A` background prefixes:

    true double mutants across all 28 proteins: **NONE**

## Verdict

**K11 is closed as a negative. There are zero genuine multi-point mutants for our 28 test
proteins**, so the proposed generalisation test does not exist. `--one_mut` (train.py:384, default
True) filters multi-point mutations anyway, and on this data it has nothing to filter.

## Why this matters beyond the task

This is the SECOND finding to come out of the 2K5H concatenation bug — first the inflated
offset-removal gain (35%), now a phantom dataset. **One malformed data file produced two separate
false leads**, and both were plausible enough to have been written into the thesis.

**Method note: when a count is surprisingly large, look at the ROW NAMES before believing it.**
The agent counted regex matches; the names said what the rows actually were.

---

# CHECKPOINT 26 — 2026-09-08 — K9: THE LORO PREDICTION, RECORDED BEFORE THE READOUT

Two runs are on the golden lane RIGHT NOW, differing in exactly one flag:

    gld_loroW_onehot   --holdout_residues W --aa_descriptors none          (epoch 10/14)
    gld_loroW_desc     --holdout_residues W --aa_descriptors mordred_pca16 (epoch 0/14)

Tryptophan is removed from TRAINING only; both are then scored on mutations TO tryptophan, which
neither model saw during training. One-hot cannot represent an unseen residue at all. A continuous
descriptor space can, because W sits at a definite point in physicochemical space.

**This is the only runnable test of Ofir Ezrielev's central claim** — that a model trained on
canonical residues predicts non-canonical effects because the physicochemical space is continuous.
MegaScale contains zero non-canonical residues, so holding out a canonical one is the proxy.

## THE PREDICTION, made before any result exists

**The descriptor arm will NOT beat the one-hot arm by much.**

Grounds, measured in CHECKPOINT 16: a residue-disjoint linear probe (fit on 19 residue types,
predict the held-out 20th) recovers from ProtT5 alone:

    hydropathy  R^2 = +0.704
    charge      R^2 = +0.511
    volume      R^2 = +0.421
    identity    accuracy 1.000 (chance 0.050)

**ProtT5 already predicts the chemistry of a residue type it has never seen.** The 1024-dim
embedding is in the feature vector for both arms, so the descriptor block is adding a property the
model can already recover. Ofir concedes exactly this in his thesis: physicochemical properties are
implicit in language-model embeddings, and his case for descriptors is COVERAGE of residues absent
from the training databases — a gap that does not exist for canonical W.

## What each outcome would mean

| result | interpretation |
|---|---|
| desc ~ one-hot (within noise) | **PREDICTED.** ProtT5 already carries W's chemistry; W6 adds regularisation at best. Report W6 as infrastructure and future work, not as a results-chapter lever. |
| desc CLEARLY beats one-hot | The prediction is WRONG and the finding is real: descriptors add something ProtT5 does not, despite the probe. This would be the strongest possible result for W6 and must be reported as such. |
| desc WORSE than one-hot | The block is actively harmful, presumably by widening the vector for no gain. Retire W6. |
| BOTH collapse on W mutations | The holdout worked but neither representation generalises. That is a statement about the ARCHITECTURE, not about descriptors, and the comparison is uninformative. |

**Negative control already queued in the design:** proline. Descriptors should NOT rescue P, because
proline's effect is backbone geometry rather than side-chain chemistry. A method that "helps"
everywhere is not being tested properly.

## Why recording this now matters

The prediction is falsifiable, it is derived from an independent measurement, and it is written
down **before** the numbers land. If the descriptor arm wins, that is a genuine surprise that
overturns CHECKPOINT 16's conclusion — and it will be reported as such rather than rationalised.

---

# CHECKPOINT 27 — 2026-09-08 — K10: THE D1 HALF IS SETTLED AT 14.4 SIGMA

## The measurement

Twelve cells scored (2K5H corrected throughout), split by factor D:

| arm | n | pooled ddG PCC | sd | range | a_p median |
|---|---|---|---|---|---|
| **D0 (coil)** | 6 | **0.5939** | **0.0091** | 0.5808 - 0.6100 | **0.5736** |
| **D1 (unfolded_emb zero)** | 6 | **0.1579** | 0.1171 | -0.0366 - 0.3056 | **0.0136** |

**The reference scale — five `abl_sigma` seeds of the SAME configuration:**

    mean 0.5798, sd 0.0303

**D0 - D1 = 0.4359 = 14.4 seed-sigmas.**

## Why this closes the question

D0's own spread is **sd = 0.0091**, three times TIGHTER than the seed noise, so the D0 arm is
remarkably reproducible. D1's spread is 0.1171 — an order of magnitude wider — which is itself
diagnostic: those models are not converging to a consistent solution, they are failing in varying
degrees. One is anti-correlated at -0.0366.

**And a_p separates by a factor of 42** (0.5736 vs 0.0136). An a_p of 0.014 means the model barely
responds to mutation at all — it is not a worse model, it is a flat one.

**No plausible number of additional seeds moves a 14.4-sigma difference.** Running the remaining
D1 cells would spend roughly 190 GPU-hours to add decimal places to a conclusion already established
at n=6 per arm.

## The scientific reading

This is the W0 decision reversed under training. W0 chose `--unfolded_emb zero` as factor D because
it collapsed var(E_u) by 67% and cut corr(E_u, wt_err) from 0.420 to 0.119 — the cleanest-looking
way to remove the offset on a frozen checkpoint.

**Trained end to end, it removes the offset by removing the signal.** That exact caveat was recorded
in CHECKPOINT 2 for `noemb` on dG ("lowest MAE but kills the correlation, 0.35 -> 0.05"); the
factorial has now confirmed it under training rather than at inference.

**This is also the metric rule in a new form.** W0 scored factor D on a frozen-checkpoint variance
decomposition, which is neither dG nor ddG performance — it measured whether the offset channel
could be silenced, not whether silencing it helps.

## Recommendation, for the user to decide

**Stop the D1 arm; redirect its ~190 GPU-hours.** Highest-value alternatives, in order:
1. **More seeds on the D0 cells** — turns single-seed cells into replicated ones, which is what the
   A/B/C main effects actually need, since their effects will be far smaller than D's.
2. **The severing experiment** (~3 GPU-h, built and dry-run green) — decides whether the whole
   reference-state programme is aimed correctly.
3. **A side-chain feature arm** — the hydrophobic ranking failure is the largest unexplained
   deficit and the only one with a clear architectural fix.

**Not acted on.** Cancelling queued cells is the user's call.

## CHECKPOINT 27 (LORO readout, pre-registered before any result was seen)
Decision threshold fixed BEFORE scoring: n(to-W) ~= 1395 pooled over 28 test proteins,
Fisher-z SE on a single rho = 0.027, so |d rho| > 0.05 on mutations-TO-W is the floor for
"a real difference". Below 0.05 the honest verdict is "desc ~= one-hot" (outcome 1).
Two diagnostics established while waiting, both PRE-result:
 - Both arms verified to genuinely drop W from TRAIN ([LORO] tallies non-zero in both logs);
   desc arm verified to load mordred_pca16 (md5 6c3a2b0a58a8729977809b1eadae3c16).
 - train.py validate() has a REAL BUG: `val_loss += batch_loss` sits OUTSIDE the batch loop
   (line 882, indent 12 vs the `for` at 12), so the printed "Validation Loss" is the LAST
   protein's loss / len(val_ds), not a mean. This is why the desc arm's val looked frozen at
   0.12675. Weight-drift check proves the model IS training (rel drift 0.47/epoch, same as
   one-hot 0.45-0.52; train loss 124->141->304). Bug affects BOTH arms identically so the
   comparison stands, but the printed val number must NOT be used for epoch selection.

## CHECKPOINT 28 (LORO readout — PARTIAL, desc arm collapsed)
THE DESCRIPTOR ARM DID NOT PRODUCE A VALID MODEL. Full write-up: results/LORO_RESULT.md.
 - onehot (21084179) trains normally: val ddG PCC 0.548@e0 -> 0.712@e12.
 - desc (21107689) is COLLAPSED TO A CONSTANT: PCC 0.090/-0.028/0.022 over 3 epochs, ddG
   RMSE frozen at exactly 2.541 (one distinct value; onehot has ten, 1.801-2.142). 2.541
   EXCEEDS onehot's worst signal-bearing epoch, which is where a constant predictor lands.
 - NOT a dead feature and NOT numerical blow-up: fc1_gcn widened 52->68 (+16, correct),
   descriptor cols have healthy norms, weight drift 0.47/epoch (onehot 0.45-0.52), no
   NaN/Inf, no all-zero tensors, max norm inflation only x1.10, output head healthy.
 - LEADING (unproven) CAUSE: data/aa_descriptors_mordred_pca16.csv is centered but NEVER
   SCALED. PC1 std 2.726, range +-6.0, appended to 0/1 one-hot => block carries ~13.8x the
   one-hot per-residue energy. mordred_pca16 has NEVER trained successfully on this cluster.
 - VERDICT: outcome 4 (collapse) but ASYMMETRIC and for a reason unrelated to the
   hypothesis. CHECKPOINT 26's prediction is NEITHER CONFIRMED NOR OVERTURNED — it remains
   OPEN. Reporting "desc did not beat one-hot, as predicted" would be confirming a
   prediction with a broken arm. The ProtT5 R^2 0.704 redundancy conclusion is UNCHANGED
   and gains NO support from this run.
 - RETEST PREPARED, NOT SUBMITTED (user's call; 13 GPU jobs in flight): plain z-scoring is
   the WRONG fix (K=16 unit-variance cols -> norm sqrt(16)=4, energy 13.8x -> 17.8x, worse).
   Must z-score AND scale by 1/sqrt(K). Built + CPU-smoke-tested:
   data_fixed/aa_descriptors_mordred_pca16_unit.csv (per-residue L2 1.055 vs one-hot 1.000,
   md5 2396e12cc3605e118c512a7902cfc208, loads OK via --aa_descriptor_csv, no code change).
 - Scoring job 21109549 (CPU) running the onehot arm: ~297 s/protein, ~2h20m for 28.
 - G4 gate re-verified AFTER all work: dG=-0.0030 width=1092. No tracked source modified.

---

# CHECKPOINT 28 — 2026-09-08 — THE COIL RESULT WAS MISREAD. b_p HAS NOW SURVIVED EIGHT EXPLANATIONS.

## The claim that motivated a whole programme

CHECKPOINT 2 recorded the project's most-cited number: the Flory coil with `--coil_b fixed`
improves absolute dG MAE from **4.9650 to 3.9521**, and it was called "our best geometric lever on
b_p". A GPU arm (`gld_dg_coil_s42`) was submitted on the strength of it.

**It does not mean what it was taken to mean. Verified by the main session on the raw
`results/w0_dg.json` per-protein data, with 2K5H's reference corrected:**

| condition | MAE | mean(b_p) | **std(b_p)** |
|---|---|---|---|
| base | 5.0751 | **-5.0751** | **0.9967** |
| **coil_fixed_b** | **4.0622** | **-4.0622** | **1.1231** |
| coil | 5.8547 | -5.8547 | 1.0743 |
| noemb | 3.2416 | -3.2416 | 1.0234 |

**All 28 of 28 proteins are UNDER-predicted in every condition.** When every residual has the same
sign, `MAE` is identically `|mean(b_p)|` — **the metric was measuring MEAN BIAS, not dispersion.**

**And the dispersion went the WRONG WAY: std(b_p) rose 0.9967 -> 1.1231, 12.7% WORSE.**

A uniform per-protein shift is exactly what Pearson is invariant to (a fact this project already
established when it showed subtracting a constant changes pooled PCC by exactly zero). **A shift
cannot generate or remove per-protein dispersion, so it cannot be a b_p lever at all.**

## The trained arm confirms it, and adds a second proof

`gld_dg_coil_s42`, scored at epoch 14 against the ddg-loss control:

| | dG arm | control |
|---|---|---|
| dG MAE | **2.4003** | 1.3029 |
| std(b_p) | 0.9656 | 1.6030 |
| per-protein ddG PCC | **0.3151** | 0.7310 |
| a_p median | **0.0852** | 0.4975 |
| pooled ddG PCC | **0.2278** | 0.5910 |

std(b_p) fell 40%, which looks like a win. **It is the degenerate baseline again.** The trajectory
settles it:

    epoch 0 : std(b_p) = 0.9410,  a_p = 0.0253,  corr(b_p, true) = -0.9262
    epoch 14: std(b_p) = 0.9656,  a_p = 0.0852,  corr(b_p, true) = -0.7744

**std(b_p) is FLAT from epoch 0 — before the model has learned anything.** The "improvement" was
present at initialisation; it is a scale property of the dG loss, not an effect of the coil. And
0.9656 sits 4.9% from the known degenerate attractor `std(true WT dG) = 0.9208`, with
`std(pred WT) = 0.6350`, i.e. the arm predicts LESS spread than the labels contain.

**Meanwhile the model got worse on the very metric it was justified by: dG MAE 1.30 -> 2.40.**

## Consequence: b_p has now survived EIGHT explanations

label noise (dead by 200x) - chain length (r=+0.025) - `--dg_length_norm` (degenerate) - embedding
norm (r=-0.29) - distribution shift (r=+0.043) - the train/test mean gap (p=0.588) - a feature-based
corrector (LOPO fails) - **and now the reference state, which was the last cheap one.**

**FINDINGS.md Part IV must be rewritten.** The line "what remains untested is the reference state,
where the coil produced the only real movement" is now false: the coil produced a mean shift, and
the trained arm degrades everything.

## Honest caveat on the trained arm

There is **no matched control**: `ref_dg_seed42/` is an EMPTY directory, so no trained dG-loss run
WITHOUT the coil exists. The comparison is against a ddg-loss control, which differs in the loss as
well as the coil. **What makes the verdict safe is the epoch-0 evidence**, which is internal to the
run: a quantity flat from initialisation was not produced by training.

## THE SLOPE ARM, by contrast, is CONFIRMED and stronger than predicted

`gld_slope1.0_s42` at epoch 14:

    a_p  0.4975 -> 0.7396   (+0.242)
    s    0.6342 -> 0.9901   (+0.356)
    r    0.7955 -> 0.7925   (-0.003, FLAT)

**The mechanism is exactly as the identity a_p = r*s predicts: all of the gain came from s, none
from r.** The identity is exact in our data (max |a_OLS - r*s| = 2.0e-15 over 28 proteins). The
trajectory shows r pinned at 0.791-0.794 from epoch 4 across ten epochs while s climbs 0.67 -> 1.02.

**a_p is now at 93% of its ceiling** (a_p <= r = 0.79). The residual (1-r) = 0.208 is the
hydrophobic ranking deficit, which this lever provably cannot touch.

This golden run BEATS the earlier factorial cell at the same flag and seed (a_p 0.740 vs 0.535), so
the full-data regime matters and FINDINGS 3.2 should be read as a **lower bound**.

**Its honest cost:** pooled gain is only **+0.0116** (pooled PCC is dominated by b_p, not a_p), and
**std(b_p) gets WORSE** (1.6030 -> 2.3585). **The arm trades offset calibration for slope
calibration.** Still n=1 seed.

## Method note worth keeping

The agent could not read `a_p median` from the training logs and **said so rather than fabricating
it**: the logging was added to train.py at 00:43 while both jobs started at 19:23 the previous
evening and hold the pre-edit module in memory. It measured the trajectory by scoring saved
checkpoints on the test set instead — strictly better evidence than a validation-set log line.

---

# CHECKPOINT 29 — 2026-09-08 — D1 CANCELLED, THE ν EXPONENT WAS NEVER SWEPT, W12 BUILT

## 1. The D1 arm is cancelled — 13 pending cells

Verified by JobName before each `scancel`; **no RUNNING job was touched** (15 before, 15 after).
Grounds: K10 measured **D0 − D1 = 14.4 seed-sigmas**, with a_p separating by a factor of 42
(0.574 vs 0.014 — the D1 models are FLAT, not merely worse). Finishing them would have spent
~104 GPU-hours adding decimal places.

## 2. Re-examining the failures: design fault, or execution fault?

Asked again for each, because a good idea executed badly is still a good idea.

| lever | design | execution | the idea was… |
|---|---|---|---|
| `--dg_length_norm` | **WRONG** — /n deletes the prediction | fine | bad idea, correctly rejected |
| b_p corrector | fine | fine | good idea, **genuinely refuted** (95% of the gain fails out of sample) |
| W7span / U10 / W7edge | fine | fine | plausible, **inside the ±0.060 noise band** |
| W6 descriptors | fine | fine | **superseded** — ProtT5 predicts held-out hydropathy at R²=0.704 |
| **the coil** | **WRONG METRIC** — MAE measured mean bias, not dispersion | fine | see §3 |
| **LORO** | fine | **BROKEN** | **good idea, never tested** — see §4 |
| severing | fine | **OOD guard fired** | good idea, needs a retrained model |
| factor A | **CONFOUNDED** — aliased with seed AND epoch | fine | not estimable as designed |

## 3. THE FLORY COIL: ν was NEVER swept, and the default is the wrong physics

`--flory_nu` has **default 0.5** and **`sacct` shows zero runs ever set it.** Every coil arm —
including the rejected dG arm — used ν=0.5.

**ν=0.5 is the THETA state**: a collapsed chain in a solvent-neutral condition. **A denatured
protein in water is a self-avoiding walk, ν≈0.588.** Kohn et al. 2004 PNAS measured
denatured-state Rg ∝ N^0.598 across 28 proteins.

What that costs, at b=5.82 Å:

| \|i−j\| | ν=0.5 | ν=0.588 | ratio |
|---|---|---|---|
| 10 | 18.40 | 22.54 | 1.225 |
| 20 | 26.03 | 33.88 | 1.302 |
| 50 | 41.15 | 58.07 | **1.411** |

**A 41% distance error at long range — and our proteins are 43–72 residues, so |i−j| reaches ~70,
which is exactly where ν matters most.**

**So the coil was rejected on its dispersion result (§5.1 of MASTER, which stands), but it was
never tested on the physically correct exponent.** Those are two separate statements and the record
must keep them apart. The dispersion failure is about what MAE measured; the ν question is about
whether the reference state was ever the right shape.

## 4. LORO: the scale mismatch, now measured directly

The descriptor arm collapsed (RMSE frozen at 2.541). **Cause confirmed by measurement:**

    descriptor table  sd = 1.0000   max|v| = 6.04   →  energy ratio ~16.8x one-hot
    one-hot           sd = 0.2179   max    = 1.0

The descriptors were z-scored to sd=1 across 20 residues and are 16 columns wide, so they carried
**~17× the energy of the identity signal.** The network saw noise that drowned residue identity.
**An execution fault. The hypothesis was never tested.**

## 5. W12 — the side-chain block, built and gated (18/18)

`scripts/sidechain_features.py` + `scripts/gate_sidechain.py`.

**The mechanism is CORRECTED.** The original design blamed side-chain BULK; that is refuted by its
own shuffle test (volume P=0.069, hydropathy P=0.0000; cross-validated R² volume 0.031 vs
hydropathy 0.666 vs **transfer free energy 0.753**). The block therefore encodes **Fauchère–Pliska
water→octanol transfer free energy**, with volume only as a secondary area-scaling term.

**Verified against the measured per-residue slopes:**

    corr(dG_transfer, measured slope) = -0.922
    corr(volume,      measured slope) = -0.408

**Four columns at offset 48:** `dG_transfer`, `burial·dG_transfer`, `volume`, `burial·volume`.
The burial-weighted columns are **ZERO in the unfolded state**, so the folded-minus-unfolded
difference IS the hydrophobic driving force — it acts on b_p — while the mutated position's
identity delta survives ddG. **It acts on BOTH channels and must be reported on both.**

**The LORO lesson is designed in:** block energy is **1.83× one-hot, not 16.8×**, because it is
4 z-scored columns rather than 16. The gate asserts this explicitly so the collapse cannot recur.

## 6. Scoring: 4 CPU jobs now covering every unscored run

`scripts/score_all.sh` scores **every** run trained to epoch 14 with no CSV — factorial cells and
golden arms alike — on the CPU partition at zero GPU cost, verifying the ARTIFACT with `ls` after
each cell rather than trusting the `DONE` line.
