# DeepPEF v5 — The Ultimate Version

A self-contained, flag-gated evolution of the proven DeepPEF dG pipeline (baseline PCC 0.5259),
adding verified, dG-focused improvements toward PCC ≥ 0.70. Every change is toggleable so each
can be ablated one at a time.

## What's new vs the working code
| Feature | Flag | What it does | Where |
|---|---|---|---|
| Length/mass-balance norm | `--length_norm` | energy per-residue (intensive) instead of a length-biased sum | `model/hydro_net_v5.py get_energy` |
| Serial fusion (default on in ladder) | `--serial_fusion` | inject ProtT5 into GNN message-passing | already in `hydro_net_v5.py` |
| Correlation loss | `--corr_weight 0.1 [--corr_type pearson|ccc]` | trains the PCC trend directly | `training/losses_v5.py` + train loop |
| dual_esmif embeddings | `--emb_type dual_esmif` | ProtT5(1024)+ESM-IF1(512) structural prior | `training/new_dataset.py` |
| Seed ensemble | `training/ensemble_v5.py` | average dG across seeds | `training/ensemble_v5.py` |

## Proposal levers A–F (physics-grounded, flag-gated, ablatable)
These are the thesis-proposal research directions. Every lever's **default reproduces the baseline
bit-for-bit** (verified by the CPU smoke test + a state_dict regression check). See
`docs/RESEARCH_PROPOSAL.md` for physics motivation and `docs/TEACHING_*.md` for teaching material.

| Lever | Flag(s) | Default (= baseline) | What it does |
|-------|---------|----------------------|--------------|
| **A. Energy decomposition** | `--energy_terms K` | `K=1` | `fc2` emits K per-residue energy terms, summed to the total. K=1 = today's scalar. Per-term totals cached in `model.last_term_energies`. |
| **B. RBF distance bank** | `--rbf_centers M` `--rbf_min` `--rbf_max` | `M=0` | Replace the single Gaussian distance kernel with M RBFs summed back to width 16 (dimension-preserving). M=0 = single Gaussian = today. |
| **C. Burial / solvation** | `--use_burial` `--burial_radius` | off, `burial_dim=0` | Insert a per-residue CB neighbor-density scalar BEFORE emb: `[D\|Fb\|burial\|emb\|onehot]`. Fed to both GNN towers; widths grow by 1. |
| **D. Flory unfolded reference** | `--flory_unfolded` `--flory_nu` | off | Model the unfolded state as a random coil (`d~\|i−j\|^ν`) instead of tridiagonal. Value-only; shapes unchanged. |
| **E. Decoys + denoising head** | `--denoise_weight w` `--denoise_sigma` `--denoise_prob` | `w=0` | Train-only aux MLP predicts injected coordinate noise on the folded structure. Head absent when w=0. |
| **F. Edge features + connectivity** | `--gcn_span S` `--use_edge_features` | `S=1`, off | (i) GCN offsets 1..S both directions (S=1 = i,i+1 = today). (ii) GATv2 consumes a 41-dim edge feature `[onehot_src(20)\|onehot_dst(20)\|dist(1)]`. |

Composability: A/E are orthogonal to widths; B is dimension-preserving; D is value-only; C (node
width) and F (edges) compose because all widths derive from `model_cfg_v5.CFG`.

## Standalone package + smoke test (no GPU / no real data)
`DeepPEF_v5/` runs independently. Paths resolve via (in priority order): `--data_root` CLI arg →
`DEEPPEF_DATA_ROOT` env var → `./data/...` default (byte-identical to the historical layout). See
`config_paths.py`.

```bash
# install (CPU is fine for the smoke test)
pip install -r DeepPEF_v5/requirements.txt

# run the whole pipeline on synthetic data — baseline + EVERY lever — on CPU, no data needed
python DeepPEF_v5/tests/smoke_test.py     # exit 0 = all configs passed (CI-friendly)

# point training at a custom data location
python DeepPEF_v5/training/pnas_train_v5.py --data_root /path/to/data ...   # or: export DEEPPEF_DATA_ROOT=...
```
The smoke test generates a tiny synthetic dataset, forces CPU + `WANDB_MODE=disabled` + `--debug`,
then runs 1 train step + 1 validation + `get_deltaG` for the baseline AND each lever (A–F) and
asserts finite outputs + correct energy-vector shapes. It catches every dimension-contract bug.

## What's kept (proven — unchanged)
Energy-difference dG (`dG = E_unfolded − E_folded`), PNAS filtering (`--one_mut --dg_ml`, dG clamp
[-1,5]), Huber+ranking loss, k-NN GAT (k=30), `--mini_batch_size 16` (OOM-safe), WANDB disabled.

## Files
```
model/hydro_net_v5.py     — PEM + length_norm (one surgical change to get_energy)
model/model_cfg_v5.py     — config
training/pnas_train_v5.py — training with --length_norm and --corr_weight wired in
training/losses_v5.py     — pearson_loss, ccc_loss (verified numerically)
training/new_dataset.py   — dataset (unchanged, supports dual_esmif)
training/train_utils.py   — graph construction (unchanged)
training/ensemble_v5.py   — seed-ensemble evaluation
run_v5_evolution.sh       — the rung ladder (run from repo root)
docs/V5_PLAN.md           — full design rationale + verification protocol
```

## Run (on GPU machine, from repo root)
```bash
export WANDB_MODE=disabled
bash DeepPEF_v5/run_v5_evolution.sh
```
This runs the ladder: baseline → +length_norm → +serial_fusion → +corr_loss → (dual_esmif) → ensemble.
Compare `best_model.pt` PCC across `Megascale-fineTuning/models/v5_rung*`.

## Evolution logic (memory lesson: change ONE thing at a time)
Each rung inherits the winner of the previous rung and adds exactly one flag. If a rung REGRESSES,
drop that change and continue from the prior winner. Do not stack losers.

## Confidence
- Code correctness: HIGH (compiles; length_norm math + correlation losses unit-verified).
- PCC gains: HYPOTHESES from literature/other models — only the experiments prove real gains.
- Grounding: all changes are diffs against the real code, read and verified 3x (see V5_PLAN.md).

## End-to-end verification (done) + bugs fixed
Verified with 4 agents + manual checks against the PROVEN baseline (pnas_train.py, hydro_net.py):
- CUDA/device flow: BYTE-IDENTICAL to baseline — `DEVICE='cuda'`, `model.to(device)`, all 4
  `get_deltaG` tensor `.to(device)` calls, `delta_g.to(device)` in train+validate, and
  `torch.cuda.empty_cache()` in validate() all match. Only intended diffs: `length_norm` +
  opt-in `corr_loss`.
- Graph construction (train_utils.py) and dataset (new_dataset.py): byte-identical to originals.
  get_one_hot = 20 dims; unfolded keeps diagonal + i±1; k-NN GAT k=30, 12 Å, CA=atom idx 1.
- length_norm: math verified (divide summed energy by residue count); default off = exact baseline.
- correlation losses: pearson_loss=1−r, ccc_loss=1−CCC, differentiable, guards correct.

Two real bugs FOUND and FIXED during verification:
1. Split-CFG bug: `hydro_net_v5.py` imported `model.model_cfg` (base, emb_input_dim=1024) while
   `pnas_train_v5.py` mutated the separate `model_cfg_v5.CFG`. Harmless for prott5 (1024) but would
   SILENTLY MISALIGN embeddings for `dual_esmif` (1536). FIX: hydro_net_v5 now imports
   `model.model_cfg_v5` so model + train script share ONE CFG object.
2. ensemble_v5.py: importing pnas_train_v5 re-ran its module-level `parse_args()` (import-time
   crash), never set `USE_KNN` (wrong GAT topology vs training), and omitted constructor args
   (load_state_dict size-mismatch risk). FIX: ensemble now injects matching train argv via sys.argv
   before import, exposes `--use_knn_gat/--use_learned_aa/--emb_projection`, and constructs PEM with
   the same args used at training. Pass the SAME flags used to train the seeds.
