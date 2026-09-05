# STATUS.md — what was built, what was tested, what is untestable locally

**Date built: 2026-09-05, in WSL (`~/workspace/DeepPEF`). Nothing pushed anywhere.**
**Snapshot:** built against `shaharec/main` (`origin/main`) commit `0ef6d3d`; local run branch
`offset-attack-run` at commit `d0d8c4a`, which carries the ported offset-attack machinery
(`--loss_mode`, `--no_freeze`, `--wt_anchor_weight`, `--designed_weight`, `--slope_weight`) that
the scripts and `code/add_seed.patch` target. The package lives inside that repo working tree and
depends on the surrounding `Megascale-fineTuning/` (see README version marker).

This package is the local half of the DeepEF cluster runbook (ORIENTATION Steps 0–5). `train.py` cannot run in WSL
(it reads the old K50 tensor layout `./data/Processed_K50_dG_datasets/training_data`, which does
not exist here; `calib_ctrl` was produced on the BGU cluster where those tensors live). So the
long training run and half the preflight checks are **cluster-only** by construction, not by
omission. Every file below carries one of three honest labels in its header.

## The three honest labels
- **TESTED end-to-end** — exercised on real data locally, output verified.
- **TESTED in isolation only** — logic proven on a fixture / synthetic input / py_compile; the
  real-data path is cluster-only.
- **UNTESTED, requires cluster** — cannot be run at all in WSL; provided for the cluster operator.

## Artifact honesty table

| Artifact | What it does | Label | Evidence |
|---|---|---|---|
| `code/calib_diag.py` | the PRIMARY instrument: std(b), slope dist, offset-removed and affine-oracle PCC, designed subset | **TESTED in isolation only** | `--selftest` PASSES (synthetic: raw 0.47 → offset-removed 0.92, std(b) 2.77, slope 0.99). The 0.77–0.81 affine-oracle validation needs a REAL Shahar eval CSV (cluster-only, see below) |
| `code/monitor.py` | parse a training log, fire the health rules M1–M9, exit 2 = STOP | **TESTED in isolation only** | each rule fired on a hand-made fixture log; parsed 2 real logs (rescore, seed-1) cleanly |
| `code/preflight.py` | 8 pre-run assertions on the training setup | **PARTIALLY TESTED** | checks 3,4,6,8 exercised locally (param count = 703,805 EXACT; GCN0 weight std 0.1319; forward finite dg=3.644); checks 1,2,5,7 need train.py + K50 data = cluster-only |
| `code/add_seed.patch` applied to `train.py` | `--seed` flag (was hardcoded RANDOM_SEED=42 at line 33) | **TESTED in isolation only** | `git apply --check` PASSES against the offset-attack `train.py` base blob `79c5529` (commit `6e590ae`); seed 42 vs seed 1 give different outputs |
| `--slope_weight` in `train.py` (Agent-E) | factor C: within-protein `abs(std(pred_ddG) − std(true_ddG))`, default 0.0 | **TESTED in isolation only** | present in the local `train.py` (lines 76/80/435–444); guarded so weight 0.0 is bit-identical to baseline (nothing added to the loss / autograd graph); weight 0.5 changes loss. **NOT a shipped patch — it lives on the offset-attack branch this package is built against** |
| `scripts/00_env_check.sh` | verify conda env, torch+CUDA, partition/QOS, data location on the cluster | **UNTESTED, requires cluster** | `bash -n` clean; no cluster access from WSL |
| `scripts/01_data_check.sh` | find K50 tensors + the eval_results CSV; count proteins | **UNTESTED, requires cluster** | `bash -n` clean; needs cluster filesystem |
| `scripts/02_smoke.sh` | preflight + 1-epoch full_data smoke + monitor, reference flags | **UNTESTED, requires cluster** | `bash -n` clean; needs train.py + K50 data |
| `scripts/03_calib_ctrl.sh` | reproduce `calib_ctrl` (Phase 1), afterok eval chain, GATE 0.606/0.711 | **UNTESTED, requires cluster** | `bash -n` clean; this is the 3–4h reference run |
| `scripts/04_seeds.sh` | Phase 2: 5 seeds {42,1,2,3,4} to measure σ | **UNTESTED, requires cluster** | `bash -n` clean |
| `scripts/05_anchor_sweep.sh` | Phase 3/4: `--wt_anchor_weight` in {0.3,1.0,3.0}, trade-off curve | **UNTESTED, requires cluster** | `bash -n` clean |

## The one end-to-end local result (the rescore)
The 2026-09-04 local run was rescored offline with the correct per-protein metric on the 28
held-out proteins (28,314 mutations). Reported with all three qualifiers:

> `pnas_train.py`, ΔG objective, random init, mini_batch 16, held-out 28, in-training epoch
> selection: epoch 12 pooled ΔΔG PCC **0.5500**, per-protein **0.6617**, absolute ΔG **0.2683**;
> epoch 14 pooled 0.5490, PP **0.6645**. **The thesis baseline 0.531 is reproduced and slightly
> exceeded.** The earlier "0.28" reading was a metric artifact of in-training `validate()`
> (one global WT reference), not a real failure.

## THE PREDICTION, recorded BEFORE the reference run (do not edit after)
Local reproduction (pnas_train, dg, random init, **mini_batch 16**) reached PP **0.6645** (epoch 14).
The reference calib_ctrl PP is **0.711**. The gap is **0.026** — and 0.026 is SMALLER than a
16-vs-64 mini-batch change would be expected to produce.

**Prediction:** the cluster reference at mini_batch 64 will close most of that 0.026. **If it does
NOT** (i.e. PP stays near 0.66-0.68 at mb64), then mini-batch was NOT the explanation for the
per-protein shortfall, and the cause is elsewhere (objective? split? affine_calib interaction?).
Either outcome is informative; the point is that this is written down now, not rationalised later.

## A reliability datum — the 1h20m stall
The stopped seed-1 sweep (kept as checkpoints, ΔG arm) showed a **1h20m hang mid-epoch at step
76→77** (≈1391 s/it), then normal speed resumed for the rest of the epoch. Nothing crashed; the
loss curve was continuous across the gap. Recorded as a laptop-under-long-jobs reliability fact
and one more reason the real runs belong on the cluster. On the cluster, `monitor.py`'s
epoch-wall-clock rule (stop if a step exceeds ~2× the running median) is what catches this.

## CRITICAL GAP — factor D (coil) is NOT in train.py yet
The COMPUTE_PLAN Phase-3 factorial is A × B × C × D = 2⁴ = 16 cells. But:
- **A** `--wt_anchor_weight` — EXISTS in train.py.
- **B** `--designed_weight` — EXISTS in train.py.
- **C** `--slope_weight` — EXISTS on the offset-attack branch (Agent-E), default-off bit-identical,
  isolation-tested. Not yet exercised on real data.
- **D** `--flory_unfolded` / `--flory_nu` — **DOES NOT EXIST in train.py.** It lives only in the
  DeepPEF_v5 bundle (`hydro_net_v5.py` / `train_utils` flory_reference), which is not synced to
  train.py. **The D-high cells cannot run until the coil is ported to train.py and its equivalence
  proven (flory off == current tridiagonal mask, bit-identical).**

The shipped `05_anchor_sweep.sh` covers the WT-anchor sweep (Phase 3/4), not the full factorial.
The A×B×C×D screening factorial is a cluster-side build once σ is known and D is ported; it is
flagged here, not silently dropped.

## Three genuinely-unknown cluster items (leave as TODO for the operator)
1. **Partition / QOS availability** — reference sbatch uses `--partition rtx6000 --qos keasar`.
   COMPUTE_PLAN prefers `cs-4090-*` nodes; `cs-6000-*` / `ee-l40s-*` for memory-bound arms. Confirm
   what this account can submit to.
2. **Conda env name** — reference uses `source activate esm2_env_py38`. Confirm it exists / matches.
3. **Data location** — find Shahar processed tensors on THIS cluster before downloading anything
   (`calib_ctrl` was produced here). The real per-mutation eval CSV that validates calib_diag lives
   at `/mnt/new_groups/keasar_group/casp15/Shahar/DeepPEF/eval_results/` per `analyze_best.py`.

## First thing to do on the cluster
1. `bash scripts/00_env_check.sh`
2. `bash scripts/01_data_check.sh`
3. `python Megascale-fineTuning/calib_diag.py --selftest`  (proves the instrument on that machine)
4. Point calib_diag at a real Shahar eval CSV and confirm affine-oracle in **0.77-0.81** — THIS is
   the deferred end-to-end validation of the primary instrument.
5. `bash scripts/02_smoke.sh` (preflight checks + 1-epoch smoke) BEFORE the reference run.
