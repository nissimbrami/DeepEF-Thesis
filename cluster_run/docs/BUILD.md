# BUILD.md — what to construct before running

**Read `ORIENTATION.md` first.** This file tells you what to build. It deliberately does not
contain finished scripts: paths, partition names, environment names and data locations on this
cluster cannot be verified from outside it, and a script written blind would fail on its first
line. **Build each item, verify it against the real cluster, then proceed.**

Put everything under `cluster_run/` in the repository working tree, on a local branch. Do not
push without approval.

---

## Build status (updated 2026-09-05 by the local agent)

Most items below were **already built and tested to the extent possible off-cluster** (see
`STATUS.md` for the per-artifact honesty table). Where an item is marked **built — verify and
run**, the file is shipped in `code/` or `scripts/`; you do not write it, you inspect it, adapt
the three cluster-specific unknowns, and run it. Where it says **build on cluster**, it is a
task that genuinely cannot be produced from outside (needs `sinfo`, real data, or a GPU).

**Exactly three things are unknown from outside this cluster; they are the only real TODOs:**

1. **Partition / QOS** — Shahar used `--partition rtx6000 --qos keasar`. Confirm what this
   account may submit to (see `00_env_check.sh`, which is written to test exactly this).
2. **Conda env name** — Shahar used `source activate esm2_env_py38`. Confirm it exists / matches.
3. **Data location** — where Shahar's processed K50 tensors live on THIS cluster (`calib_ctrl`
   was produced here, so they exist here). See `01_data_check.sh`.

Everything else is decided. The scripts carry these three as `${VAR:-default}` overrides.

**Two operational cautions, learned the hard way while building this package off-cluster:**

- **Line endings.** Every shipped file is LF-only. If you edit one on a machine that inserts
  CRLF (`\r`), bash will fail with obscure `$'\r': command not found` errors. Check with
  `file scripts/*.sh` (must say "ASCII text", never "with CRLF line terminators").
- **One shell, one session.** Do the whole run inside a single persistent `tmux`/`screen`
  session on the cluster. Do not tail a job with a long `sleep`; use `tail -f` on the log and
  the `monitor.py` stop rules. Submit training with `sbatch`, never run it in the foreground of
  your login shell.

---

## Repository setup

```
git clone https://github.com/nissimbrami/DeepEF-Thesis.git DeepPEF
cd DeepPEF
git remote add shaharec https://github.com/shaharec/DeepPEF.git
git fetch shaharec
git checkout -b cluster-run shaharec/main
```

Start from `shaharec/main`, not from Nissim's `main`. Nissim's fork is an older snapshot that
lacks `train.py`'s calibration levers. Verify immediately:

```
grep -c "wt_anchor_weight\|no_freeze\|val_frac\|designed_weight" Megascale-fineTuning/train.py
```

Expect four or more matches. If zero, you are on the wrong branch.

---

## B1 — `00_env_check.sh` — **built — verify and run** (`scripts/00_env_check.sh`)

**Purpose:** fail in ten seconds rather than ten hours. The script is shipped; it tests exactly
the three unknowns (partition/QOS, conda env, data path). Run it first and read its output.

Must verify and print, one line each with `PASS` or `FAIL`:

- `sinfo` runs; list the partitions you can submit to and their idle node counts
- whether `--qos keasar` is accepted (submit a trivial `--wrap "hostname"` job and check it is
  not rejected)
- `module load anaconda && source activate esm2_env_py38` succeeds
- `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` inside a GPU
  allocation, not on the login node
- free space on the filesystem that will hold `./data` and `Megascale-fineTuning/models/`

**Decide and record:** which partition and GPU type you will use. Shahar used
`--partition rtx6000 --qos keasar --gpus=1 --cpus-per-task=8` and excluded
`cs-6000-01..04, cs-cpu256-01`. The `cs-4090-*` nodes are faster; if you switch, write down that
you switched, because it changes wall-clock estimates but must not change results.

**Gate:** every line `PASS`.

---

## B2 — `01_data_check.sh` — **built — verify and run** (`scripts/01_data_check.sh`)

**Purpose:** find the data before converting anything. The script is shipped; point it at the
candidate roots below.

`train.py` requires:

```
./data/Processed_K50_dG_datasets/training_data/<protein>/
      coords_tensor.pt  deltaG.pt  mask_tensor.pt  one_hot_encodings.pt
      prott5_embeddings/prott5_embedding_<i>.pt
./data/Processed_K50_dG_datasets/mutation_datasets/<protein>.csv
./data/Processed_K50_dG_datasets/Pnas_filtering/train_proteins.csv
./data/Processed_K50_dG_datasets/Pnas_filtering/pnas_mutations.csv
./data/ThermoMPNN/mega_test.csv
```

Search, in this order, and stop at the first hit:

1. **Shahar's account or any shared project directory on this cluster.** `calib_ctrl` was
   produced here, so these tensors existed here. Look for `training_data` directories containing
   `coords_tensor.pt`. If found and readable, symlink and you are finished — no conversion.
2. **Nissim's own cluster storage**, if he has run here before.
3. **Hugging Face** `nissimb/deepef-megascale` (~83 GB) — different layout, see B3.

Print: number of protein directories found, number of proteins listed in `mega_test.csv`
(expect 28), and total size. End with `DATA: OK` or `DATA: FAIL <reason>`.

**Gate:** `DATA: OK`, 368 protein directories, 28 test proteins.

---

## B3 — `msds_adapter.py` — **build on cluster, only if step B2 fell through to Hugging Face**

Not shipped: it depends on the exact HF layout, which cannot be confirmed off-cluster. Only
needed if `01_data_check.sh` does NOT find Shahar's tensors and you fall back to the HF dataset.
The HF dataset uses a different file layout. Write a **subclass** of `train.py`'s
`AllProteinValidationDataset` that overrides **only file I/O**:

| `train.py` expects | HF/MsDs layout provides |
|---|---|
| `coords_tensor.pt` | `coords.pt` (or `crd_backbone.pt`) |
| `mask_tensor.pt` | `mask.pt` |
| `deltaG.pt` | `deltaG.pt` — same |
| directory of `prott5_embedding_<i>.pt`, sorted by integer suffix, `vstack`ed | single `emb.pt`, a list, one entry per variant |
| `one_hot_encodings.pt` | absent — compute with `get_one_hot(seq)` from the mutation CSV's `aa_seq` column |
| mutations under `mutation_datasets/` | mutations under `mutation_files/` |

**Do not touch** the filtering, the train/test split logic, or the returned dictionary. Those
determine what the model sees, and changing them invalidates the 0.606 target.

**Ordering hazard:** the original loader sorts embedding files by their integer suffix. `emb.pt`
is stored in mutation-CSV order. Verify on one protein that the two orderings agree before
trusting the adapter.

### `equivalence_check.py` — the gate on B3

Load three proteins — one small, one large, one designed (`HEEH_*`) — through both the adapter
and `new_dataset.MSDataset(one_mut=True, dg_ml=True)`, and assert:

1. identical mutation count after filtering
2. `delta_g` equal
3. `coords` shape and values equal
4. embeddings shape `[n_mut, L, 1024]` and `torch.allclose`
5. one-hot equal
6. row 0 is the wild type in both

Print each as `PASS` or `FAIL`.

**Gate: all six pass. If one fails, that discrepancy is the finding — report it and stop. Do not
patch around it.**

---

## B4 — `preflight.py` — **built — verify and run** (`code/preflight.py`)

Shipped and importable. Checks 3, 4, 6, 8 were exercised off-cluster (param count 703,805 exact;
GCN0 weight std 0.13; forward finite); checks 1, 2, 5, 7 need `train.py` + K50 data and run only
on the cluster. Called at the top of a training run before the first optimiser step. Prints eight
lines, each `PASS` or `FAIL`, and raises on any failure.

1. **Counts.** 28 test, ~34 validation, ~306 train. Any other split means the wrong filter or
   the wrong data root.
2. **Wild-type row.** `batch['delta_g'][0, 0]` present and finite for every training protein.
   Both the ΔΔG objective and the WT anchor subtract it. Print how many proteins were checked.
3. **Freeze/pretrain trap.** `assert not (no_pretrain and freeze)`. Random initialisation with
   the head-only freeze trains `fc1`/`fc2` on random GNN features and produces a flat, low
   curve. This has already cost this project one wasted run.
4. **Random init took.** Print the standard deviation of the first GCN layer's weight. Random
   initialisation gives roughly 0.05–0.3. A trained checkpoint looks different.
5. **Objective wiring.** On the first minibatch print `l1_loss`, `ddg_loss`, `wt_anchor_loss`.
   Under `--loss_mode ddg --wt_anchor_weight 0`, `ddg_loss` must be nonzero, `l1_loss` must not
   be driving `data_loss`, and `wt_anchor_loss` must be zero.
6. **Parameter count** ≈ 703,805, within 5%.
7. **Epochs took.** Print the epoch count actually in effect and assert it equals what was
   passed.
8. **Checkpoint round-trip.** After epoch 0 writes `kf_all_epoch_0.pt`, load it in a subprocess
   and run one forward pass.

---

## B5 — `monitor.py` — **built — verify and run** (`code/monitor.py`)

Shipped. All stop rules were fired against fixture logs off-cluster and it parsed real training
logs cleanly. Runs alongside a training job. Tails the log, parses the per-epoch validation line, appends one
JSON object per epoch to `results/<run_tag>_monitor.jsonl`, and enforces stop rules.

| Signal | Healthy | Stop when |
|---|---|---|
| training `ddg_loss` | falls at least 10% by epoch 2 | flat or rising through epoch 2 |
| NaN or Inf in loss, gradients or predictions | never | first occurrence |
| gradient norm after clipping | 0.1 – 30 | above 100, or below 1e-6 for a whole epoch |
| validation ΔΔG PCC-PP | ≥ 0.45 at epoch 2, rising | below 0.30 at end of epoch 2 |
| median over proteins of `std(pred_ddG)/std(true_ddG)` | 0.4 – 1.0 | below 0.15 (slope collapse) |
| `std(b)` on validation | around 0.3, stable | above 0.6 or rising every epoch — **log, do not stop**; this is the quantity under study |
| epoch wall-clock | consistent | more than twice the first epoch |
| `cov(): degrees of freedom <= 0` warnings | rare | above 20% of minibatches — the ranking term is inert |
| validation minus training PP gap | below 0.15 | above 0.3 early |

On a stop condition: print `STOP: <reason>`, `scancel` the job, and write the reason into the
report file.

**Also implement the epoch-2 reproduction check.** When `kf_all_epoch_2.pt` appears, run
`evaluate.py` on it against the 28-protein test set and print the pooled PCC. Shahar's honest
runs are near 0.64 at epoch 2. **Below 0.55, stop.** This is a reproduction check, not epoch
selection — selection stays on validation.

---

## B6 — `calib_diag.py` — **built — verify and run** (`code/calib_diag.py`)

Shipped. `--selftest` passes off-cluster (synthetic raw 0.47 → offset-removed 0.92). The
end-to-end validation — feeding it a REAL Shahar eval CSV and confirming the affine-oracle lands
in 0.77–0.81 — is deferred to the cluster, because no committed Shahar eval CSV exists locally.
Do that validation first thing (see STATUS.md). Input: an evaluation CSV with columns
`protein, deltaG, pred_deltaG, ddG, pred_ddG`.

For each protein, fit `pred_ddG ≈ a·ddG + b`. Output:

- `std(b)` across proteins — the offset spread, the primary quantity of the thesis
- the distribution of `a` — min, median, max, and the count below 0.3
- pooled PCC, per-protein PCC (mean over `groupby('protein')` with `len(g) ≥ 3`)
- pooled PCC after removing `b` per protein — the oracle ceiling, expect 0.77–0.81
- the same table restricted to designed folds, matched by name
  (`HHH|HEEH|EEHEE|EHEE|EHHE|HHHH`, `*_TrROS_*`, `v2_*`)

This runs after every evaluation. It is what makes results comparable across runs.

---

## B7 — `--seed` patch to `train.py` — **built — verify and run** (`code/add_seed.patch`)

Shipped as a small reviewable patch. `git apply --check code/add_seed.patch` was confirmed to
apply cleanly against the offset-attack `train.py` base. `RANDOM_SEED = 42` is hardcoded at line 33 with no flag. Add `--seed`, defaulting to 42, that
sets `random.seed`, `np.random.seed`, `torch.manual_seed` and `torch.cuda.manual_seed_all`, and
that also feeds the `KFold` and any `train_test_split` call so the validation split moves with
the seed.

**Verify:** two runs at the same seed produce identical first-epoch loss; two runs at different
seeds do not.

Keep this as a small, reviewable patch. It is the only change to `train.py` in this phase.

---

## B8 — the run scripts — **built — verify and run** (`scripts/02_smoke.sh` … `05_anchor_sweep.sh`)

All four run scripts are shipped and `bash -n`-clean. They are modelled on
`Megascale-fineTuning/sbatch_calib.sh` and `sbatch_wtanchor.sh`, which already
target this cluster. Each script submits training, then chains
`run_calib_eval.sh <train_log> <model_dir> <run_tag>` with `--dependency afterok:<jobid>`.

**Checkpoint directory naming**, needed for the eval step: `MODEL_NAME` is built at `train.py`
lines 46–47 and 103–104 as `PEM_full_trained-<base>` (under `--no_freeze`), then `kf`, then
`_<run_tag>`. Checkpoints are `kf_all_epoch_<N>.pt`. Derive the directory rather than hardcoding
it, and assert it exists before submitting the eval job.

**`02_smoke.sh`** — reference flags, `--epochs 1`, `preflight.py` enabled. Not `--max_folds 1`:
`--full_data` bypasses k-fold, so that flag does nothing.

**`03_calib_ctrl.sh`** —

```
python Megascale-fineTuning/train.py \
  --full_data --no_pretrain --no_freeze --loss_mode ddg \
  --pooled_corr_weight 0 --dg_length_norm none --affine_calib \
  --wt_anchor_weight 0 --designed_weight 1 \
  --val_frac 0.1 --epochs 15 --run_tag calib_ctrl_repro
```

with `monitor.py` attached.

**`04_seeds.sh`** — the same, five seeds, distinct `--run_tag` per seed.

**`05_anchor_sweep.sh`** — the same with `--wt_anchor_weight` in {0.3, 1.0, 3.0}.

---

## Order of operations

```
B1 env  →  B2 data  →  [B3 adapter + equivalence, only if needed]
        →  B7 seed patch  →  B4 preflight  →  B5 monitor  →  B6 diag
        →  02 smoke  →  03 reference  →  04 seeds  →  05 anchor sweep
```

Build B4–B6 before the first long run, not after. They are what makes a failed run cost one hour
instead of ten.

---

## Reporting

After each step write `results/<step>_<run_tag>.md`:

- exact command line
- SLURM job ID, node, wall-clock
- metric table with all three qualifiers: pooled ΔΔG · per-protein ΔΔG · absolute wild-type ΔG
- `std(b)` and the slope distribution from `calib_diag.py`
- for step 04: mean ± σ across seeds
- for step 05: one row per anchor weight, so the trade-off is visible
- anything that surprised you

Commit locally. **Do not push without explicit approval.**
