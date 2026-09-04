# Offset-Attack Build + Run Plan (code-anchored, verified against WSL tree)

Repo: `~/workspace/DeepPEF`  branch `offset-attack-run`  (origin = shaharec/DeepPEF — READ-ONLY, never push).
venv: `~/pytorch_env` (torch 2.6.0+cu124, scipy 1.17.1), RTX 4070 8GB. Always `export WANDB_MODE=disabled`.

> This plan was written by INSPECTING the actual ported code. Several claims in the "LOCKED
> 2026-09-04" plan do NOT match this tree; the discrepancies + blockers are called out explicitly.
> The PLANNER did not edit any model code — build steps are instructions for the executor.

---

## 0. GROUND TRUTH: what is actually in the tree (verified)

There are THREE distinct trainers; the LOCKED plan conflated their flags. Verified facts:

| Script | Data root | Model | Offset levers present | Runnable NOW? |
|---|---|---|---|---|
| `Megascale-fineTuning/pnas_train.py` | `data/MsDs/` (369 dirs, 28-test all present) | `build_energy_model` | `--loss_mode {dg,ddg,joint}`, `--ddg_weight`, `--freeze_layers`, `get_wt_deltaG`, unfolded-graph via `get_unfolded_graph` | **NO — ImportError** |
| `Megascale-fineTuning/train.py` | `data/Processed_K50_dG_datasets/training_data` | `PEM` (imports OK) | `--loss_mode`, `--ddg_weight`, `--wt_anchor_weight`, `--no_pretrain`, `--no_freeze`, `--full_data`, `--val_frac`, `--affine_calib`, `--dg_length_norm`, `--fold`, `--designed_weight`, `--pooled_corr_weight`, `--readout` | **NO — data dir empty** |
| `Megascale-fineTuning/pnas_train_sm.py` | `data/MsDs/` | `PEM` (imports OK) | GNN subtract-mut (`f_type='subtract_mut'`, [L,20] scores). Has `--seed/--emb_type/--use_knn_gat/--ranking_weight/--lr_min/--weight_decay/--huber_delta`. NO unfolded state, NO loss_mode/freeze/wt_anchor | YES (but WRONG objective) |

**The LOCKED-plan REFERENCE flags (`--loss_type huber_rank`, `--ranking_weight`, `--use_knn_gat`,
`--cosine_lr`, `--lr_min`, `--weight_decay`, `--emb_type`, `--seed`, `--no_pretrained`,
`--mini_batch_size`) exist ONLY in `pnas_train_sm.py`** (the subtract-mut GNN-SM script), which is
the wrong architecture for the offset attack (it has no folded/unfolded energy difference, so there
is no coil lever and no wt-anchor). They do NOT exist in `pnas_train.py` or `train.py`.

### BLOCKERS (must be fixed by the executor before any run) — see §8 for detail
- **B-IMPORT**: `pnas_train.py:16` does `from model.hydro_net import build_energy_model`, but
  `build_energy_model` is NOT defined in `model/hydro_net.py` (only class `PEM`). `pnas_train.py`
  **crashes on import**. This is the primary blocker for the MsDs offset-attack tree.
- **B-K50DATA**: `train.py --full_data` reads `data/Processed_K50_dG_datasets/training_data`, which
  does NOT exist (only `__MACOSX/` is there; `mutation_datasets/` has 862). So the `wt_anchor`
  regime in `sbatch_wtanchor.sh` cannot run here. `train.py` is NOT usable on this machine unless
  the K50 training tensors are restored.
- **B-EVALMISSING**: `validation/score_runs.py:19` does `import benchmark_metrics as M` but
  `validation/benchmark_metrics.py` DOES NOT EXIST (not in tree, not on origin/main). `score_runs.py`
  will `ModuleNotFoundError`. Also `run_calib_eval.sh` (referenced by `sbatch_wtanchor.sh`) is MISSING.
- **B-PRETRAIN**: pretrained core `res/trianed_models-light_attention/43_final_model.pt` is ABSENT.
  `pnas_train.py` hardcodes `PRETRAINED=True` (line 80) with NO override flag → it will crash in
  `load_checkpoint` / `torch.load`. `train.py` HAS `--no_pretrain` to bypass. `pnas_train_sm.py`
  never loads a pretrained core (random init).
- **B-MINIBATCH (OOM)**: neither `pnas_train.py` nor `train.py` exposes `--mini_batch_size`; both
  hardcode `MINI_BATCH_SIZE = 64`. Memory's OOM lesson requires 16. Must be reduced in-code.

### Consequence for this run
The ONLY tree that is (a) offset-attack-capable (folded/unfolded dG, loss_mode, freeze, wt-anchor,
coil hook) AND (b) data-complete on this machine is **`pnas_train.py` on `data/MsDs/`**. It requires
B-IMPORT + B-PRETRAIN + B-MINIBATCH fixed first. The `train.py` wt-anchor arm needs K50 tensors
(B-K50DATA) that are absent — Rung 5 as written in the LOCKED plan is BLOCKED here; a `pnas_train.py`
`--wt_anchor_weight` equivalent must be ADDED (it currently only exists in `train.py`).

---

## 1. REFERENCE base_dg (the 0.531 point)

`score_runs.py` BASELINE string (authoritative, line 22):
`"baseline 0.531/0.500/1.085/0.691 | freeze-best ddg_nopretrain 0.655 PCC / 0.738 PP"`
=> pooled-ddG PCC 0.531, SCC 0.500, RMSE 1.085, PCC-PP 0.691. This is the **MsDs / pnas_train.py**
regime (dg objective). The `sbatch_wtanchor.sh` calib_ctrl 0.606/PP0.711 is a DIFFERENT regime
(train.py K50 full_data) that is BLOCKED here (B-K50DATA).

**Once B-IMPORT/B-PRETRAIN/B-MINIBATCH are fixed**, the reference is (per seed in 42,1,2,3,4):

```bash
cd ~/workspace/DeepPEF
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python Megascale-fineTuning/pnas_train.py \
  --loss_mode dg --one_mut --dg_ml --dataset_type pnas \
  --emb_projection mlp --model_arch pem \
  --model_name abl_base_dg_seed${SEED} \
  --trained_model_path NONE   # requires B-PRETRAIN fix (add --no_pretrained OR restore 43_final_model.pt)
```
Notes / must-verify before trusting the number:
- **Seeds**: `pnas_train.py` has NO `--seed` arg and calls no `set_seed()`. To run 5 seeds
  (42,1,2,3,4) the executor must ADD a `--seed` flag + `set_seed()` (copy the 7-line `set_seed`
  from `pnas_train_sm.py:65`). Without it, "5 seeds" is not reproducible.
- **mini_batch 16**: reduce `MINI_BATCH_SIZE = 64` → `16` at `pnas_train.py:75` (OOM rule). This is a
  file edit, not a flag.
- `--one_mut --dg_ml` are ESSENTIAL filters (memory: unfiltered data HURT 0.41 vs 0.53).
- `pnas_train.py` two-stage schedule is FIXED in code: `EPOCHS_FREEZE=20` then `EPOCHS_NO_FREEZE=60`
  (lines 70-71). `--epochs` is parsed but NOT used in `run_training()`. For the frozen-arm the useful
  checkpoints are the freeze-stage epochs (0..19); saved as `models/<MODEL_NAME>/epoch_<e>.pt`.

---

## 2. Rungs 1–3 on `pnas_train.py` (MsDs) — exact commands (post-fix)

All share: `--one_mut --dg_ml --dataset_type pnas --emb_projection mlp --model_arch pem`,
`MINI_BATCH_SIZE=16` (code), `--seed` (added), WANDB off, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Rung 1 — ddg_only** (direct mutation-aware ddG objective):
```bash
python Megascale-fineTuning/pnas_train.py --loss_mode ddg \
  --one_mut --dg_ml --dataset_type pnas --emb_projection mlp --model_arch pem \
  --seed ${SEED} --model_name abl_ddgonly_seed${SEED}   # + pretrain fix
```
Selection metric auto-switches to ddG-Pearson for ddg/joint (`validate()` line 511) — good.

**Rung 2 — freeze_only** (dg objective, frozen backbone; only fc1/fc2/LA train):
```bash
python Megascale-fineTuning/pnas_train.py --loss_mode dg --freeze_layers \
  --one_mut --dg_ml --dataset_type pnas --emb_projection mlp --model_arch pem \
  --seed ${SEED} --model_name abl_freezeonly_seed${SEED}   # + pretrain fix
```
IMPORTANT (freeze semantics): `--freeze_layers` only freezes STAGE 1 (the 20-epoch freeze phase);
STAGE 2 (60 epochs) always unfreezes (`run_training()` sets FREEZE_LAYERS=False before stage 2 — see
`train.py:496`; `pnas_train.py` mirrors this). So "freeze-best" = pick the BEST of the STAGE-1 epoch
checkpoints (`epoch_0.pt` .. `epoch_19.pt`). **Eval EVERY frozen epoch and take our best** (§6).

**Rung 3 — ddg_freeze** (both):
```bash
python Megascale-fineTuning/pnas_train.py --loss_mode ddg --freeze_layers \
  --one_mut --dg_ml --dataset_type pnas --emb_projection mlp --model_arch pem \
  --seed ${SEED} --model_name abl_ddgfreeze_seed${SEED}   # + pretrain fix
```
Target: memory says "freeze-best ddg_nopretrain 0.655 PCC / 0.738 PP" — i.e. Rung 3 with random init
is the 0.655 headline. NOTE that requires `--no_pretrained` (B-PRETRAIN); with the pretrained core
restored instead, the number will differ — decide ONE pretraining regime and hold it fixed across all
rungs so comparisons are valid.

---

## 3. Rung 4 — coil (analytic Flory random-coil unfolded reference)

### What to port (executor, flag-gated, DEFAULT OFF = bit-exact baseline)
Target function: `train_utils.get_unfolded_graph` (lines 223-241). It currently zeroes all off-
tridiagonal contacts via `zero_except_udiagonal(D)` (line 302). The coil replaces that stark
tridiagonal mask with an analytic ideal-coil distance `d = b*|i-j|^nu`.

Reference implementation to adapt: `/mnt/c/Users/I763940/DeepPEF/DeepPEF_v5/training/train_utils.py:256`
`def flory_reference(...)`. **Port a SIMPLIFIED version matching Shahar's `get_unfolded_graph`
exactly** (no burial, no RBF bank, no AFRC — those v5 helpers do not exist in this tree):

```
# in train_utils.py get_unfolded_graph, add a flag branch (CFG.flory_unfolded, default False):
#   nu   = float(getattr(CFG, 'flory_nu', 0.5))   # (0,1]; 0.5 ideal chain, ~0.588 SAW
#   N    = x.shape[0]; ca = x[:,1,:]
#   b    = ||ca[1:]-ca[:-1]||.mean()  (clamp min 1e-3)   # ~3.8 A CA-CA => memory's b~5.8 is per-atom-pair; use measured
#   sep  = |i-j|                                          # [N,N]
#   d    = b * (sep+1e-6)^nu                              # [N,N] analytic coil distance
#   D    = d[...,None].expand(N,N,16)                     # broadcast across 16 atom channels
#   D    = relu(exp(gaussian_coef * D**2))                # SAME kernel as folded path (line 208/226)
#   ... then IDENTICAL to get_unfolded_graph from the mask step down (NO zero_except_udiagonal) ...
```
Add to `model/model_cfg.py`: `flory_unfolded = False`, `flory_nu = 0.5`. Gate in
`get_unfolded_graph`: `if getattr(CFG,'flory_unfolded',False): <coil path> else: <existing tridiagonal path>`.
Default False must reproduce the baseline BIT-FOR-BIT (regression-check: run base_dg with the flag off,
confirm identical val curve). Since `pnas_train.py`, `train.py`, AND `pnas_train_sm.py` all import the
SAME `train_utils.get_unfolded_graph`, one edit covers the tree — but only pnas_train.py/train.py use
the unfolded graph (pnas_train_sm.py has no unfolded state).

Executor must ADD a `--flory_unfolded` (+ optional `--flory_nu`) flag to `pnas_train.py` that sets
`CFG.flory_unfolded`/`CFG.flory_nu` (mirror how `--emb_projection` sets `CFG.emb_projection`).

### Mechanism check (split Eu/Ef per protein; do this in the eval/validate path)
The offset hypothesis: the tridiagonal unfolded reference gives a per-protein-constant unfolded energy
`Eu` (dominated by chain length), so `dG = Eu - Ef` inherits a length/offset term uncorrelated with
per-mutation signal. The coil should make `Eu` sequence/structure-aware. Predicted signatures if coil
attacks the offset:
- `corr(Eu, GAT-pathway output) ↓` (unfolded energy stops tracking the global attention readout),
- `corr(Ef via GCN) ↑` or unchanged (folded contact signal preserved),
- `var(Eu) ↓` across mutations of a protein (unfolded ref becomes a stabler per-protein anchor),
- per-protein offset `b` (intercept of pred_dG vs exp_dG regression) correlates LESS with chain length.

Instrument: in `pnas_train.py get_deltaG` we already return `(dG, u_energy, f_energy)`. Have the
executor dump per-variant `Eu`, `Ef`, `pred_dG`, `exp_dG`, `protein`, `len` to a diagnostic CSV during
`validate()` (a 6-col extension of the existing `val_df`), for BOTH `--flory_unfolded` off and on.
Then compute the four correlations above off-line. This is the cheap decisive test from
`shahar_talk_insights.md` (scatter `b` vs contact-order/length).

### Run commands (post-build; both objectives)
```bash
# dg arm
python Megascale-fineTuning/pnas_train.py --loss_mode dg  --flory_unfolded --flory_nu 0.5 \
  --one_mut --dg_ml --dataset_type pnas --emb_projection mlp --model_arch pem \
  --seed ${SEED} --model_name abl_coil_dg_seed${SEED}
# ddg arm
python Megascale-fineTuning/pnas_train.py --loss_mode ddg --flory_unfolded --flory_nu 0.5 \
  --one_mut --dg_ml --dataset_type pnas --emb_projection mlp --model_arch pem \
  --seed ${SEED} --model_name abl_coil_ddg_seed${SEED}
```

---

## 4. Rung 5 — wt_anchor

`--wt_anchor_weight` currently exists ONLY in `train.py` (line 73), whose data is BLOCKED (B-K50DATA).
Two options for the executor:

**(a) Preferred — port `--wt_anchor_weight` into `pnas_train.py`** (MsDs, runnable). The mechanism is
identical to `train.py:405-406`: `wt_anchor_loss = L1(pred_dG(WT), exp_dG(WT))`, added as
`loss += WT_ANCHOR_WEIGHT * wt_anchor_loss`. `pnas_train.py` already has `get_wt_deltaG` (line 537,
returns pred WT dG in-graph) and `batch['delta_g'][0,0]` is the exp WT dG — so the port is ~8 lines in
the `train()` loop (mirror train.py:392-416) + a `--wt_anchor_weight` flag. Only meaningful together
with `--loss_mode ddg` (in ddg mode the absolute scale is free, so the anchor pins the per-protein
baseline = the offset attack).

Run (both weights, both arms; ddg is the meaningful one — dg arm is a control):
```bash
for W in 0.3 1.0; do
 python Megascale-fineTuning/pnas_train.py --loss_mode ddg --wt_anchor_weight $W \
   --one_mut --dg_ml --dataset_type pnas --emb_projection mlp --model_arch pem \
   --seed ${SEED} --model_name abl_wtanchor_ddg_w${W}_seed${SEED}
 python Megascale-fineTuning/pnas_train.py --loss_mode dg  --wt_anchor_weight $W \
   --one_mut --dg_ml --dataset_type pnas --emb_projection mlp --model_arch pem \
   --seed ${SEED} --model_name abl_wtanchor_dg_w${W}_seed${SEED}
done
```

**(b) Fallback — restore K50 `training_data`** and use `train.py` exactly as `sbatch_wtanchor.sh`:
```bash
python Megascale-fineTuning/train.py --full_data --no_pretrain --no_freeze --loss_mode ddg \
  --pooled_corr_weight 0 --dg_length_norm none --affine_calib \
  --wt_anchor_weight {0.3|1.0} --designed_weight 1.0 --val_frac 0.1 --epochs 15 --run_tag wtanchor_w{03|10}
```
(Still needs MINI_BATCH_SIZE 64→16 in train.py:40 for the 8GB card; baseline to beat here is
calib_ctrl 0.606/PP0.711, a DIFFERENT number than the 0.531 MsDs reference — do not mix scoreboards.)

---

## 5. BOTH ARMS rule

Every lever (Rungs 1,2,3,4,5) is run under **`--loss_mode dg` AND `--loss_mode ddg`**. The `dg` arm is
the calibration-sensitive control (absolute dG supervised → offset can't hide); the `ddg` arm is where
the offset attack matters (absolute scale free). Rung 2 (freeze) and Rung 4 (coil) obviously combine
with each objective; keep the `_dg_`/`_ddg_` tag in `--model_name` so `score_runs.py` groups them.

---

## 6. Eval command + scoreboard

### Eval producer (writes eval CSVs)
`Megascale-fineTuning/evaluate.py` loads a checkpoint (`--trained_model_path`) and writes
`./<model_name>.csv` with columns `protein,deltaG,pred_deltaG,ddG,pred_ddG` (per-protein ddG computed
correctly: `protein_df['ddG'] = deltaG - deltaG.iloc[0]` per protein, line 426-427). **CAVEAT:**
`evaluate.py.__main__` (line 545) hardcodes `Processed_K50_dG_datasets` as the data root and uses the
OLD `AllProteinValidationDataset` — it is NOT wired to MsDs and has NO `--loss_mode`. For the MsDs
scoreboard the executor must either (i) point evaluate.py at MsDs + MSDataset, or (ii) reuse
`pnas_train.py validate()` to dump the CSV (it already builds `val_df` with `protein,deltaG,pred_deltaG`;
extend with per-protein ddG). Name outputs `eval_results/abl_<tag>_e<N>.csv` so score_runs picks them up.

Per frozen epoch (Rung 2/3): loop the STAGE-1 checkpoints and emit one CSV each:
```bash
mkdir -p eval_results
for e in $(seq 0 19); do
  python Megascale-fineTuning/evaluate.py \
    --model_name eval_results/abl_<tag>_e${e} \
    --trained_model_path Megascale-fineTuning/models/<MODEL_NAME>/epoch_${e}.pt \
    --one_mut --dg_ml --dataset_type pnas   # + MsDs wiring fix
done
```

### Scorer
```bash
python validation/score_runs.py <tag1> <tag2> ...     # prints per-epoch PCC/SCC/RMSE/PCC-PP + BEST
```
Prints, per tag, the epoch curve `epoch  PCC  SCC  RMSE  PCC-PP` and `BEST PCC epN=.. | BEST PCC-PP
epN=..`. Metric defs: pooled `M.pearson(df.ddG, df.pred_ddG)`; PCC-PP = mean over proteins with
`len(group)>=3` of per-protein `M.pearson`. **BLOCKER B-EVALMISSING**: `validation/benchmark_metrics.py`
providing `pearson/spearman/rmse` is ABSENT — executor must create it (3 trivial wrappers over
`scipy.stats.pearsonr/spearmanr` + RMSE) or score_runs crashes on import.

### Scoreboard columns (record to a results file per run)
`run, arm(dg|ddg), seed, absdG_heldout(pooled dG PCC on 28-test), PCC-PP(per-protein ddG),
pooled_ddG(pooled ddG PCC), b(mean per-protein intercept of pred_dG vs exp_dG)`.
- `absdG_heldout` = `pc_corr` from validate() (pooled dG Pearson).
- `pooled_ddG` = `ddg_pearson_corr` (validate line 485) OR score_runs pooled PCC.
- `PCC-PP` = score_runs PCC-PP (per-protein, len>=3).
- `b` = fit `pred_dG = a*exp_dG + b` per protein, report mean b (the calibration offset under attack).

---

## 7. Suggested execution order (cheapest decisive first)
1. Fix B-IMPORT, B-PRETRAIN (decide regime), B-MINIBATCH, B-EVALMISSING; add `--seed`+set_seed to pnas_train.py.
2. Smoke: `--debug` 1-epoch base_dg on MsDs to confirm no import/OOM/shape errors.
3. base_dg reference (seed 42 first) → confirm ~0.531 pooled ddG before spending 5 seeds.
4. Build coil (Rung 4) + wt_anchor port (Rung 5a); regression-check defaults OFF == baseline.
5. Run the ladder R1..R5 × {dg,ddg} × seeds; eval every frozen epoch for R2/R3; score_runs; fill scoreboard.
6. Run the coil MECHANISM diagnostic (Eu/Ef/b vs length) — the decisive cheap experiment.

---

## 8. v5 bugs B1–B5 relevance to THIS tree (audit_verification.md)
- **B1 (split-CFG: hydro_net imports base model_cfg while trainer mutates a v5 CFG)**: N/A here.
  All three trainers import the SAME `model.model_cfg.CFG` and mutate it in place (e.g.
  `CFG.model_arch=args.model_arch`, `CFG.emb_projection`). NO split-CFG. But NOTE the coil port MUST
  set `CFG.flory_unfolded` on the SAME `model.model_cfg.CFG` object that `train_utils` reads (it does:
  `train_utils.py:9 from model.model_cfg import CFG`) — so no split-CFG risk if done that way.
- **B2 (ensemble re-runs module-level argparse on import → crash)**: PARTIALLY relevant. `pnas_train.py`
  and `pnas_train_sm.py` parse args at MODULE LEVEL. Any wrapper that `import`s them (e.g. an ensemble
  driver) will trigger `parse_args()`. If the executor builds an ensemble/eval wrapper, it MUST inject
  `sys.argv` before import (the v5 rule). `train.py` uses `parse_known_args()` (line 74) so it tolerates
  extra argv, but still parses at import.
- **B3 (denoise head absent from baseline state_dict / shape mismatch on load)**: relevant to
  checkpoint loading. `evaluate.py`/`train.py` load via `load_checkpoint` then fall back to
  `load_state_dict(torch.load(...))`. If the coil/wt_anchor levers are value-only (they are — no new
  params), state_dicts stay shape-compatible across levers. KEEP levers param-free so any checkpoint
  loads into any lever config. (The coil changes only the unfolded distance VALUES, adds no weights.)
- **B4 (ReduceLROnPlateau `verbose=` removed in new torch)**: **ACTIVE here.** `train.py:307` (and the
  old evaluate.py:307) construct `ReduceLROnPlateau(..., verbose=True)`. torch 2.6 still accepts
  `verbose` but emits a deprecation warning; if it errors on this build, drop `verbose=True`. Verify
  in the smoke run. `pnas_train.py` uses ReduceLROnPlateau too — check its constructor for `verbose=`.
- **B5 (get_deltaG arity change / callers)**: relevant. `pnas_train.py get_deltaG` returns a 3-tuple
  `(dG,u_energy,f_energy)`; `train.py get_deltaG` also 3-tuple. The v5 4-tuple (denoise) is NOT in this
  tree. The coil/wt_anchor ports must NOT change get_deltaG's arity (keep 3-tuple) or every caller
  (train loop, validate, get_wt_deltaG, fit_affine) breaks. wt_anchor uses the existing `get_wt_deltaG`
  3-tuple — safe.

**Additional tree-specific bug (NOT in B1-B5), FYI:** `pnas_train.py validate()` computes the
in-training ddG metric against a SINGLE global WT (`val_dg[0]`, lines 463-466) rather than per-protein
groupby. This under-reports/mis-reports the in-training ddG PCC. The AUTHORITATIVE scoreboard therefore
must come from `score_runs.py` (proper per-protein groupby), NOT the wandb `val_ddg_pc_corr`. `train.py`
validate() (lines 499-513) DOES do it per-protein correctly — another reason its metric plumbing is
cleaner, if K50 data is ever restored.

---

## 9. One-line blocker summary for the executor
Fix in order: (1) add `build_energy_model` to `model/hydro_net.py` OR switch pnas_train.py import to
`PEM`; (2) decide pretrain regime (add `--no_pretrained` to pnas_train.py OR restore
`43_final_model.pt`); (3) `MINI_BATCH_SIZE 64→16` in pnas_train.py:75 (and train.py:40 if used);
(4) create `validation/benchmark_metrics.py` (pearson/spearman/rmse); (5) add `--seed`+set_seed to
pnas_train.py; (6) port coil into `get_unfolded_graph` (+CFG flags, +`--flory_unfolded`); (7) port
`--wt_anchor_weight` into pnas_train.py; (8) wire evaluate.py (or pnas_train validate CSV) to MsDs and
emit `eval_results/abl_<tag>_e<N>.csv`. K50 `train.py` wt-anchor arm is BLOCKED (missing training_data).
```

---

## 10. EXECUTED EDITS + FINAL COMMANDS (executor, 2026-09-04, commit 96a316c on offset-attack-run)

### 10.1 Which §8/§9 blockers were STALE vs REAL (verified on the live WSL tree)
- **B-IMPORT — ALREADY RESOLVED (stale).** `model/hydro_net.py:731` DOES define `build_energy_model`.
  `python -m py_compile Megascale-fineTuning/pnas_train.py` succeeds; `--help` runs the whole module.
- **B-EVALMISSING — ALREADY RESOLVED (stale).** `validation/benchmark_metrics.py` EXISTS (5965 bytes,
  committed 883252c) and `score_runs.py:19 import benchmark_metrics` resolves. No action needed.
- **B-PRETRAIN — FIXED.** Added `--no_pretrained` -> `PRETRAINED = not args.no_pretrained`
  (pnas_train.py). With the flag, no checkpoint is loaded (43_final_model.pt is absent -> would crash).
  Default (flag absent) keeps PRETRAINED=True = original behavior.
- **B-MINIBATCH — FIXED.** Added `--mini_batch_size` (default 16) -> `MINI_BATCH_SIZE`. Was hardcoded 64.
- **--seed — FIXED.** Added `--seed` (default 42) + `set_seed()` (random/np/torch + cudnn deterministic),
  called at module load right after CFG wiring. Reproducible 5-seed runs now possible.
- **B4 ReduceLROnPlateau `verbose=` — FIXED.** Dropped `verbose=True` from the scheduler ctor (torch 2.6).
- **Coil (Rung 4) — BUILT.** `CFG.flory_unfolded=False`, `CFG.flory_nu=0.5` in model/model_cfg.py;
  `--flory_unfolded/--flory_nu` in pnas_train.py push to that SAME `model.model_cfg.CFG` object that
  `train_utils` reads (no split-CFG). `get_unfolded_graph` gates on `CFG.flory_unfolded`: default OFF is
  **bit-identical** to the tridiagonal baseline (verified `torch.allclose` exact); ON uses analytic coil
  `d(i,j)=b*|i-j|^nu`, `b`=mean CA-CA neighbor distance, shape-identical `[N,76]`, value-only (no params).
  `flory_nu` guarded to (0,1]. No burial/RBF/AFRC (those v5 helpers do not exist here).
- **--dump_energies (mechanism split) — BUILT.** In `validate()`, per-protein mean `Eu`,`Ef`,`dG`
  (+`L`,`n_variants`) dumped to `Megascale-fineTuning/models/<MODEL_NAME>/energies_epoch_<e>.csv`.
  Mirrors analysis/wt_dg_error/task6 columns. Guarded by the flag; normal path untouched.

### 10.2 STILL REMAINING for the VERIFIER (not blockers I can fix without a run/decision)
- **wt_anchor (Rung 5):** `--wt_anchor_weight` lives in `Megascale-fineTuning/train.py:73` (WS-1, loss at
  :406/:416) whose K50 `training_data` is ABSENT (B-K50DATA). It is NOT in pnas_train.py. Two choices for
  the verifier/runner: (a) port the ~8-line anchor into pnas_train.py's train loop (get_wt_deltaG already
  exists; exp WT dG = `batch['delta_g'][0,0]`), or (b) restore K50 tensors and use train.py. Per task 4 I
  only VERIFIED it exists; I did NOT reimplement it. DECIDE before Rung 5.
- **evaluate.py MsDs wiring:** `evaluate.py.__main__` hardcodes Processed_K50_dG_datasets. For the MsDs
  scoreboard, either point evaluate.py at MsDs+MSDataset, or reuse `pnas_train.py validate()` val_df CSV.
  Confirm which producer feeds `eval_results/abl_<tag>_e<N>.csv` before scoring.
- **Two-stage schedule:** EPOCHS_FREEZE=20 then EPOCHS_NO_FREEZE=60 are hardcoded; `--epochs` is parsed
  but NOT used by run_training(). Frozen-arm best = best of stage-1 checkpoints epoch_0..epoch_19.

### 10.3 FINAL EXACT REFERENCE COMMAND (base_dg, from-scratch, mini_batch 16, seeded)
```bash
cd ~/workspace/DeepPEF && source ~/pytorch_env/bin/activate
export WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for SEED in 42 1 2 3 4; do
  python Megascale-fineTuning/pnas_train.py \
    --loss_mode dg --no_pretrained --mini_batch_size 16 --seed ${SEED} \
    --one_mut --dg_ml --dataset_type pnas --emb_projection mlp --model_arch pem \
    --model_name abl_base_dg_seed${SEED}
done
```
GATE: seed σ. If σ>=0.02, narrow to 3 levers @5 seeds instead of 8 @1.

### 10.4 THE LADDER — final commands (all share the base flags; add per-rung flags)
BASE = `--no_pretrained --mini_batch_size 16 --seed ${SEED} --one_mut --dg_ml --dataset_type pnas --emb_projection mlp --model_arch pem`
```bash
# Rung 1 ddg_only          (attribute the +0.124):   BASE --loss_mode ddg  --model_name abl_ddgonly_${ARM}_seed${SEED}
# Rung 2 freeze_only:                                 BASE --loss_mode dg  --freeze_layers --model_name abl_freezeonly_${ARM}_seed${SEED}
# Rung 3 ddg_freeze:                                  BASE --loss_mode ddg --freeze_layers --model_name abl_ddgfreeze_${ARM}_seed${SEED}
# Rung 4 coil (both arms, +mechanism dump):
python Megascale-fineTuning/pnas_train.py $BASE --loss_mode dg  --flory_unfolded --flory_nu 0.5 --dump_energies --model_name abl_coil_dg_seed${SEED}
python Megascale-fineTuning/pnas_train.py $BASE --loss_mode ddg --flory_unfolded --flory_nu 0.5 --dump_energies --model_name abl_coil_ddg_seed${SEED}
# (baseline-arm energy dump for the Eu/Ef split, coil OFF:)
python Megascale-fineTuning/pnas_train.py $BASE --loss_mode dg  --dump_energies --model_name abl_base_dg_dump_seed${SEED}
# Rung 5 wt_anchor: BLOCKED until anchor is ported into pnas_train.py OR K50 restored (see 10.2).
```
BOTH ARMS rule: run each rung under `--loss_mode dg` AND `--loss_mode ddg` (tag `${ARM}` = dg|ddg).
Coil hypothesis: helps MORE under dg (under ddg the unfolded term partly cancels mut-vs-wt).

### 10.5 EVAL + SCOREBOARD
- Producer: `Megascale-fineTuning/evaluate.py` -> `eval_results/abl_<tag>_e<N>.csv`
  (cols protein,deltaG,pred_deltaG,ddG,pred_ddG) — first wire it to MsDs (10.2). For frozen rungs,
  loop stage-1 checkpoints `Megascale-fineTuning/models/<MODEL_NAME>/epoch_${e}.pt`, e in 0..19.
- Scorer: `python validation/score_runs.py <tag> ...` (benchmark_metrics.py present; pooled =
  pearson(ddG,pred_ddG); PCC-PP = mean per-protein pearson, len(group)>=3).
- Scoreboard row: `run, arm(dg|ddg), seed, absdG_heldout(pooled dG PCC on 28-test = validate pc_corr),
  PCC-PP(per-protein ddG), pooled_ddG(pooled ddG PCC), b(mean per-protein intercept pred_dG vs exp_dG)`.
- Mechanism check (Rung 4, from energies_epoch_<e>.csv): var(dG) split into Eu vs Ef, corr(Eu,wt_err),
  and coil-vs-baseline Δ in those; predicted var(Eu)↓ under coil (attacks the 77-88% offset channel).

