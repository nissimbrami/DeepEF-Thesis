# CONTEXT.md — resume file for the DeepEF cluster run
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
