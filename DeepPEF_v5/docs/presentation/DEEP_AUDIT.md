# DeepPEF_v5 — Deep Teaching-Materials Audit

> Purpose: verify the 3 teaching docs (TEACHING_LEVERS_AF.md, TEACHING_VISUAL_REPRESENTATIONS.md,
> TEACHING_FLOW_AND_COMPATIBILITY.md) fully and correctly cover the project — original architecture,
> the v5 levers, the biology, the NN, the full flow (no gaps), method definitions, why-it-works,
> and compatibility. Each agent appends its findings below with: (1) what the CODE actually does,
> (2) whether the DOCS cover it, (3) GAPS/ERRORS found, (4) recommended additions.

---

## Agent 1 — Original project baseline (root repo: hydro_net.py, model_cfg.py, train.py, data_loader.py)
> Auditor: Agent 1. Scope = ORIGINAL baseline before any v5 levers. All claims cite `file:line`.
> Files read in full: `model/hydro_net.py`, `model/model_cfg.py`, `model/data_loader.py`,
> `model/net.py`, `train.py`, `train_utils.py`, `Megascale-fineTuning/pnas_train.py`,
> `Megascale-fineTuning/new_dataset.py`.

### 0. Two distinct pipelines exist in the original repo

There are TWO separate models/training loops, and it is critical for teaching not to conflate them:

- **`model/net.py :: ProteinEnergyNet`** — the OLD "physics-style" hand-rolled GNN (custom
  bonded/non-bonded conv1d message passing, `net.py:19-288`). It is a legacy/first-generation model.
  `train.py` imports `params` from it (`train.py:5`) but builds `PEM` from `hydro_net.py`, NOT
  `ProteinEnergyNet`. So `ProteinEnergyNet` is effectively dead code in the current training path.
- **`model/hydro_net.py :: PEM`** — the ACTUAL model used by both `train.py` (pretraining/EBM) and
  `Megascale-fineTuning/pnas_train.py` (the ddG fine-tuning that reached PCC 0.5259). This is THE
  model to teach.

The `PEM.get_energy` sums features linearly `E = torch.sum(Fh, dim=(1,2))` (`hydro_net.py:525`),
whereas the legacy `ProteinEnergyNet.get_energy` and `PEMSM.get_energy` sum SQUARES
(`net.py:141`, `hydro_net.py:662`). The v5 length_norm lever modifies exactly this `PEM.get_energy`
linear sum, so the linear-sum form is the correct baseline to present.

---

### 1. Exact PEM architecture (`model/hydro_net.py`)

**Constructor** `PEM.__init__(layers, gaussian_coef, dropout_rate=0.2, light_attention=False, emb_projection="none", gat_cutoff=None, serial_fusion=False, serial_fusion_dim=64, use_learned_aa=False, aa_emb_dim=64)` — `hydro_net.py:296-298`.

Note: `serial_fusion` and `use_learned_aa` are ALREADY present as constructor args in the "original"
root `hydro_net.py` (default OFF). They are v2-era levers that predate v5; the true baseline runs
with all of them OFF (see §5). With all flags off, `aa_dim = 20` (`hydro_net.py:329`).

**Twin GNN towers (both run every forward):**
- **GCN tower**: `layers` copies of `GCN(gcn_dim_in, 64, gcn_dim_out, dropout_rate)` — `hydro_net.py:343`.
  Each `GCN` = `GCNConv(dim_in,64) -> relu -> GCNConv(64,dim_out) -> InstanceNorm1d` with dropout on
  input (`hydro_net.py:701-720`).
- **GAT tower**: `layers` copies of `GAT(gat_dim_in, 64, gat_dim_out, heads=8, dropout_rate)` — `hydro_net.py:347`.
  Each `GAT` = `GATv2Conv(dim_in,64,heads=8) -> elu -> GATv2Conv(64*8, dim_out, heads=1) -> InstanceNorm1d`
  with input dropout (`hydro_net.py:676-698`). Uses `GATv2Conv` (not v1).
- `CFG.num_layers = 3` (`model_cfg.py:28`) → 3 GCN + 3 GAT sub-layers.

**Internal dims (baseline, no projection/fusion):**
- `gnn_internal_dim = 16 + aa_dim + proj_extra + serial_extra` = `16 + 20 + 0 + 0 = 36`
  (`hydro_net.py:339`). Both GCN and GAT operate at width 36, hidden 64.
- Per-tower input projection FCs: GCN input `fc1_gcn: (16+16+aa_dim)=52 -> 64`, `fc2_gcn: 64 -> 36`
  (`hydro_net.py:355-357`). GAT input `fc1_gat: (16+aa_dim)=36 -> 64`, `fc2_gat: 64 -> 36`
  (`hydro_net.py:360-362`). (GCN gets the extra 16 "bonded" dims; GAT does not.)

**Residual + norm structure:**
- `forward_gcn` (`hydro_net.py:502-515`): `fc1_gcn->relu->fc2_gcn`, reshape, `inst_norm1`, set
  `identity`, then for each GCN layer `x = gcn_layer(...) + identity` (residual to the post-FC input).
- `forward_gat` (`hydro_net.py:486-500`): `identity = raw x` (before fc), `fc1_gat->relu->fc2_gat`,
  `inst_norm1`, then each GAT layer `x = h1 + identity`. NB: GAT's residual is to the RAW pre-FC input
  (`identity=x` at line 488), a subtle asymmetry vs GCN (`identity` after norm at line 511).
- `inst_norm1 = Normalization_layer(gnn_internal_dim=36, affine=True)` shared by both towers
  (`hydro_net.py:365`); `inst_norm2 = Normalization_layer(gcn_dim_out+gat_dim_out=72)` (`hydro_net.py:366`).
- `Normalization_layer` = transpose→`nn.InstanceNorm1d(affine=True)`→transpose (`hydro_net.py:722-740`).

**Fusion + head:**
- After towers: `x = cat(x1, x2)` → 72-dim, reshape, `inst_norm2` (`hydro_net.py:452-455`).
- When `emb_projection == "none"` (baseline), the RAW PLM embedding is concatenated AFTER the GNN:
  `x = cat(x, x_emb_features)` (`hydro_net.py:458-459`). So the final head input dim
  `fc_in_dim = gcn_dim_out + gat_dim_out + post_gnn_emb = 36 + 36 + 1024 = 1096` (`hydro_net.py:335,368`).
  (`post_gnn_emb = CFG.emb_input_dim = 1024` when no projector — `hydro_net.py:335`, `model_cfg.py:38`.)
- **Light attention** (`light_attention=True` in both train scripts): applied on the 1096-dim vector
  before FC, `LightAttention(embeddings_dim=fc_in_dim)` (`hydro_net.py:394-395`, `742-773`). It is a
  Conv1d feature branch × softmax(Conv1d attention branch), kernel_size=9, elementwise gate
  (`hydro_net.py:766-772`). No dimensionality change.
- Head: `fc1: 1096 -> 128 -> relu`, `fc2: 128 -> 1` (`hydro_net.py:369-370, 468-477`). Output reshaped
  to `[B, N, 1]`.
- `f_type='Default'` → returns `get_energy(x)` = `torch.sum(x, dim=(1,2))` → scalar energy per graph
  (`hydro_net.py:481-482, 517-527`). `f_type='A_inference'` returns per-residue `[B,N,1]`
  (`hydro_net.py:483-484`). `f_type='subtract_mut'` uses a separate `fc2_sm: 128 -> 20` head
  (`hydro_net.py:372-373, 472-475`) — this is the GNN-SM path (the MEMORY notes it FAILED at 0.42).

**Edge construction** (`get_edge_index`, `hydro_net.py:529-574`):
- GCN edges = sequential chain `(i, i+1)` within each batch element (`hydro_net.py:543-549`). This is
  the "backbone bond" topology.
- GAT edges = fully connected (all i≠j) by default (`hydro_net.py:561-568`); OR a CA-distance cutoff
  when `ca_coords` + `self.gat_cutoff` are given (`hydro_net.py:552-559`). The fine-tuning k-NN GAT
  path sets `gat_cutoff` dynamically from the k-th nearest neighbor distance
  (`pnas_train.py:684-700`). Edges cached for the shape-only (fully-connected) case (`hydro_net.py:537-541,570-572`).

---

### 2. dG/ddG thermodynamic logic (baseline)

**Two graphs per protein state — folded vs unfolded** (`train_utils.py:196-241`):
- `get_graph` (folded): distance matrix `D = get_dist_matrix(x)` [N,N,16] → Gaussian kernel
  `D = relu(exp(gaussian_coef * D**2))` (`train_utils.py:207-208`), zero masked residues
  (`train_utils.py:210-212`), bonded features `Fb = get_bonded_features(D)` [N,32] (`train_utils.py:214`,
  `243-254`), then `D = D.sum(dim=1)` [N,16] normalized, and `Fh = cat([D(16), Fb(32), emb(1024), one_hot(20)])`
  (`train_utils.py:216-219`).
- `get_unfolded_graph` is IDENTICAL except it calls `zero_except_udiagonal(D)` (`train_utils.py:232`),
  which zeros all pairwise distances except the diagonal + immediate sequence neighbors
  (`train_utils.py:302-311`). So the unfolded graph keeps ONLY the linear-chain contacts; the folded
  graph keeps ALL 3D contacts. Coordinates are the SAME — only the contact map is stripped.
- `gaussian_coef` default `-0.08` (`model_cfg.py:32`, re-asserted `train.py:836`).

**Energy → dG → ddG** (`pnas_train.py:668-706`):
- For each mutation, build folded + unfolded graphs from the SAME WT coords but the mutation's own
  one-hot and ProtT5 embedding (`pnas_train.py:675-682`).
- `minibatch_energy = model(all_graph_minibatch)`; split into folded/unfolded halves
  (`pnas_train.py:702-704`).
- **`dG = unfolded_energy - folded_energy`** (`pnas_train.py:706`, returned as `output`). Lower folded
  energy ⇒ larger positive dG ⇒ more stable.
- **ddG** computed post-hoc in `validate`: `ddg = (val_dg - val_dg[0])` where index 0 is the wildtype
  (`pnas_train.py:621-625`). The dataset always inserts index 0 (WT) as the reference
  (`pnas_train.py:346-347`, `new_dataset.py:221,290`).

**Per-mutation signal (the KEY point, matches MEMORY):** structure/coords are fixed WT; what changes
per mutation is (a) the 20-dim one-hot and (b) the 1024-dim ProtT5 embedding of the mutant sequence.
The ProtT5 embedding is the dominant mutation signal (concatenated raw post-GNN, `hydro_net.py:459`).

---

### 3. Training loops, losses, optimizer, schedule

There are TWO training procedures. The one that produced PCC 0.5259 is `pnas_train.py`.

**(A) ddG fine-tuning — `Megascale-fineTuning/pnas_train.py` (THE 0.5259 run):**
- Loss (`Trainer.__init__`, `pnas_train.py:439-443`): `nn.HuberLoss(delta=HUBER_DELTA)` when
  `--use_huber_loss` (via `--loss_type huber_rank`), else `nn.L1Loss()`. Best config uses
  `huber_rank` → Huber(δ=1.0) + ranking (`pnas_train.py:91-96`).
- Total per-minibatch loss (`pnas_train.py:502-518`):
  `loss = primary_loss + reg_loss + energy_reg + rank_loss` where:
  - `primary_loss = criterion(output=dG_pred, delta_g)` (`pnas_train.py:503`).
  - `reg_loss = REG_LAMBDA * MSE(weights, 0)`; `REG_LAMBDA = 0` in best config so this is 0
    (`pnas_train.py:154, 506-509`).
  - `energy_reg = E_REG_LAMBDA * MSE(cat(u_energy,f_energy), 0)`, `E_REG_LAMBDA = 0.001`
    (`pnas_train.py:155, 510-511`) — pulls raw energies toward zero to keep the extensive sum bounded.
  - `rank_loss = RANKING_LAMBDA * ranking_loss(output, delta_g, margin)` (`pnas_train.py:514-516`).
    `ranking_loss` = pairwise margin loss over ≤128 random pairs:
    `max(0, -sign(target_i - target_j) * (pred_i - pred_j) + margin)` (`pnas_train.py:242-267`).
    Best config: `ranking_weight=0.1`, `margin=0.1` (`pnas_train.py:64,50,96`).
- Optimizer: `optim.Adam(params, lr=LR=1e-4, weight_decay=WEIGHT_DECAY)` (`pnas_train.py:445, 152`).
  Best config `--weight_decay 1e-5`.
- LR schedule: `--cosine_lr` → `CosineAnnealingLR(T_max=args.epochs, eta_min=lr_min=1e-6)`
  (`pnas_train.py:447-449`); else `ReduceLROnPlateau(mode='max', factor=0.1, patience=5)`
  (`pnas_train.py:451-452`, stepped on `pc_corr`).
- Gradient clipping: `clip_grad_norm_(..., max_norm=CFG.max_grad_norm=10.0)` (`pnas_train.py:529,548`,
  `model_cfg.py:58`). Optional grad accumulation (`grad_accum`, default 1, `pnas_train.py:520-531`).
- Inner loop iterates mutations in chunks of `MINI_BATCH_SIZE` (`pnas_train.py:498`). Each chunk builds
  `2*mini_batch_size` graphs (folded+unfolded), one forward pass. Metrics: Pearson/Spearman/RMSE on
  dG AND ddG (`pnas_train.py:626-665`).
- Freeze control (`handle_freez_layers`, `pnas_train.py:458-472`): if `--freeze_layers`, only
  `fc1`, `fc2`, and `LA` train; best config does NOT freeze (`epochs_freeze 15 epochs_unfreeze 0`,
  from_scratch, so full model trains).

**(B) EBM pretraining — root `train.py` (InfoNCE + DSM, native/decoy discrimination):**
- Produces 9 graph representations per protein (`get_noised_proteins`, `train.py:189-283`):
  Xjf (folded native), Xju (unfolded native), Xd (shuffled-seq decoy), Xcd (decoy STRUCTURE),
  Xdu (unfolded decoy), Xcy1..4 (cyclic sequence permutations). Feature dim asserted = 1092
  (16+32+1024+20) — `train.py:412`, `_build_graph_features` (`train.py:170-186`).
- `lossd` = InfoNCE / Boltzmann contrastive ranking (`lossd_fucntion`, `train.py:690-715`):
  primary term forces Ejf (native folded) to be the lowest energy among {Ejf,Exd,Ecd,Eju,Ecy1..4}:
  `L = E_pos/τ + logsumexp(-E_i/τ)`; secondary term forces Eju < Ecd. `τ = CFG.tau = 1.0`
  (`model_cfg.py:61`).
- `lossg` = denoising score matching (FD-DSM) on the 16 distance dims only
  (`denoising_score_matching`, `train.py:602-662`): finite-difference `v·∇E ≈ (E(x+εv)-E(x-εv))/2ε`
  matched to DSM target `v·(-noise/σ²)`, `σ = CFG.sigma = 0.5` (`model_cfg.py:22`). Replaces the older
  autograd gradient penalty (`gradient_penalty`, `train.py:591-600`).
- Gradient balancing: DSM gradient is rescaled by `dsm_alpha = clamp(grad_norm_d/grad_norm_g, 0.01, 10)`
  so DSM ≈ InfoNCE gradient magnitude (`train.py:439-460`).
- Optimizer `Adam(lr=CFG.lr=1e-4)`; schedule `StepLR(step_size=2, gamma=0.9)` (`train.py:791-793`).
  AMP autocast + GradScaler, TF32 enabled (`train.py:127-162`).
- This is the pretraining that WOULD produce a checkpoint like `43_final_model.pt` — but per MEMORY
  that checkpoint is unavailable, so the 0.5259 result was reached WITH `--no_pretrained` (from scratch).

---

### 4. Node feature layout and slicing (CRITICAL for teaching)

Feature vector per residue is built by `get_graph`/`_build_graph_features` in this order:
`[ D_sum(16) | Fb(32) | emb(E=1024) | one_hot(20) ]` (`train_utils.py:219`, `train.py:186`).

`PEM.forward` slices it (`hydro_net.py:417-422`) using indices set at `hydro_net.py:380-383`:
- `x_dist   = x[:, :16]`                → 16 non-bonded distance sums (`non_bonded_index = 16`).
- `x_bonded = x[:, 16:32]`              → first 16 of the 32 bonded features (GCN-only).
  NB: only the FIRST 16 bonded dims are used; the layout has 32 bonded dims but the slice is
  `[16:32]` (`hydro_net.py:420`). The 2nd 16 bonded dims are effectively unused by the model — a
  deliberate "matching the original behavior" choice noted in the code comment (`hydro_net.py:417-418`).
- `x_emb_features = x[:, llm_index:one_hot_index]` = `x[:, -(E+20):-20]` → the E-dim PLM embedding
  (`hydro_net.py:421`, `llm_index = -(emb_input_dim+20) = -1044`, `hydro_net.py:383`).
- `x_onehot = x[:, -20:]` → 20-dim one-hot (`hydro_net.py:422`, `one_hot_index = -20`).

Because `llm_index` is derived from `CFG.emb_input_dim`, changing embedding type (e.g. dual_esmif=1536)
shifts the slice correctly ONLY if the model reads the same CFG the trainer mutated — this is exactly
the split-CFG bug the MEMORY records for v5; the baseline (prott5, 1024) is unaffected.

`get_bonded_features` (`train_utils.py:243-254`): `f1 = D[i, i+1]` (bond to next), `f2 = D[i+1, i]`
(bond to prev), each [N,16], padded and concatenated → [N,32].

---

### 5. Key hyperparameters and WHY

Best-known config (from MEMORY, matches argparse defaults + shortcuts):
```
--no_pretrained --loss_type huber_rank --ranking_weight 0.1 --use_knn_gat --one_mut --dg_ml
--cosine_lr --lr_min 1e-6 --weight_decay 1e-5 --mini_batch_size 16 --seed 42  (WANDB_MODE=disabled)
```
- **`mini_batch_size=16` (OOM lesson):** the argparse DEFAULT is 64 (`pnas_train.py:71`), but each unit
  builds `2×mini_batch` graphs (folded+unfolded) in one forward on fully/knn-connected GATs. 64→128
  graphs/step→~7.7 GB→OOM on an 8 GB RTX 4070; 16→32 graphs→~2 GB. MUST pass `--mini_batch_size 16`.
  Also `torch.cuda.empty_cache()` runs inside the validation inner loop (`pnas_train.py:615`).
- **`seed=42`** everywhere for determinism: `set_seed` sets torch/cuda/numpy/random +
  `cudnn.deterministic=True, benchmark=False` (`pnas_train.py:104-113`); `CFG.seed=42` (`model_cfg.py:16`).
- **`num_layers=3`, `dropout_rate=0.2`** (`model_cfg.py:28-29`); note `train.py:835` overrides dropout to
  0.3 for the EBM pretraining run only. `pnas_train.py` uses `DROP_OUT=0.2` (`pnas_train.py:153,784`).
- **`gaussian_coef=-0.08`** (`model_cfg.py:32`) — width of the distance→contact kernel.
- **`lr=1e-4`, `weight_decay=1e-5`, `max_grad_norm=10`** (`pnas_train.py:152`, `model_cfg.py:58`).
- **`E_REG_LAMBDA=0.001`** keeps the extensive energy sum from drifting (`pnas_train.py:155`).
- Serial fusion / learned-AA / emb_projection all DEFAULT OFF in the baseline
  (`model_cfg.py:40-43`, `pnas_train.py:72` default `none`), so the true baseline is:
  raw 1024-dim ProtT5 concatenated post-GNN, static 20-dim one-hot, no projection.

---

### 6. Dataset filtering (PNAS / one_mut / dg_ml) — biological meaning

Filtering applied in `MSDataset.load_protein_data` (`new_dataset.py:242-308`) and mirrored in
`AllProteinValidationDataset` (`pnas_train.py:367-418`):
- **ins/del removal** (`new_dataset.py:247,260`): drop insertions/deletions — the model uses a fixed WT
  backbone of fixed length, so indels (which change length/register) are structurally incompatible.
- **`--dg_ml` (dG_ml, clamp to [-1, 5])** (`new_dataset.py:270-273`, `pnas_train.py:381-384`): clamps
  the deltaG label to the range the K50 assay can actually resolve. Below ~-1 and above ~5 kcal/mol the
  measurement saturates (fully unfolded / fully folded), so values outside are noise; clamping stops the
  regressor from chasing unmeasurable extremes. Applied to the tensor via `torch.where`.
- **`--one_mut` (one_mut)** (`new_dataset.py:282-283`, `pnas_train.py:392-393`): drop multi-site
  mutations (`mut_type` containing ':'). The per-mutation signal is a single-residue change; multi-site
  entries mix effects and the ProtT5 delta is less cleanly attributable.
- **`ds_type='pnas'`** (`new_dataset.py:285-287`, `pnas_train.py:395-397`): keep only mutations whose
  `name` is in the curated PNAS mutation list (`pnas_mutations.csv`). PNAS curation selects
  high-quality, single-domain, well-measured stability data — the MEMORY records that using the
  UNFILTERED full megascale set HURT PCC (0.41 vs 0.53): quality > quantity for noisy ddG labels.
- **Homolog / test-leakage removal:** TM (ThermoMPNN) test proteins are held out
  (`new_dataset.py:52-56`, `pnas_train.py:282-285`), and training homologs listed in `mega_train.csv`
  are removed (`new_dataset.py:77-81`). `remove_homologs` in `pnas_train.py:296-299` keeps only proteins
  in `train_proteins.csv`. This prevents train/test leakage against the ThermoMPNN benchmark.
- WT reference is force-kept as index 0 for ddG computation (`new_dataset.py:221,290`,
  `pnas_train.py:346-347`).

Data files: coords/deltaG/mask/one_hot/ProtT5 tensors per protein under `data/MsDs/training_data`,
mutation CSVs under `data/MsDs/mutation_files` (`pnas_train.py:782-783`, `new_dataset.py:26-27`).

---

### 7. Original design intent / purpose

The core thesis is an **energy-based model (EBM)**: a learned scalar "energy" where the native folded
state is the minimum. Two training regimes express this:
1. **Pretraining (`train.py`):** teach the energy landscape by native-vs-decoy discrimination (InfoNCE
   contrastive ranking) plus denoising score matching so the native structure sits at a low-energy,
   low-gradient basin. Decoys = shuffled sequences, wrong structures, and cyclic permutations.
2. **Fine-tuning (`pnas_train.py`):** reuse the SAME `PEM` energy to predict thermodynamic stability by
   defining `dG = E_unfolded - E_folded` per mutation, regressing to experimental deltaG (Huber+rank),
   and deriving ddG relative to the WT reference. The extensive (summed) energy is intentionally the
   quantity being differenced between folded/unfolded and between mutant/WT, so absolute scale cancels.

Design choices that matter for the v5 story: energy is an EXTENSIVE linear sum over residues
(`hydro_net.py:525`) — this is what v5's `length_norm` normalizes; the PLM embedding is concatenated
raw AFTER the GNN (`hydro_net.py:459`) — this is what v5's `serial_fusion` changes by injecting PLM
INTO message passing; and the loss is Huber+margin-rank — v5 adds correlation (Pearson/CCC) loss on top.
Establishing these three baseline facts (linear-sum energy, post-GNN raw PLM concat, Huber+rank loss)
is essential before teaching the v5 levers.

**One caveat for the teaching docs:** do not describe the baseline GNN as "GCN OR GAT" — it runs BOTH
towers in parallel and concatenates them (`hydro_net.py:449-452`). And do not attribute square-energy
`Σ Fh²` to `PEM`; that is the legacy `ProteinEnergyNet`/`PEMSM` form. `PEM` uses linear `Σ Fh`.


---

## Agent 2 — v5 model + training core (hydro_net_v5.py, pnas_train_v5.py, model_cfg_v5.py)

> All citations are `file:line`. Files: `training/pnas_train_v5.py` (T), `model/hydro_net_v5.py` (M),
> `model/model_cfg_v5.py` (C), `training/train_utils.py` (U), `training/losses_v5.py` (L).
> Verified: every lever default reproduces the baseline shapes/energy. Bugs/omissions in §11.

### 1. CLI flags + defaults (argparse) — baseline reproduction confirmed

Parser at `T:37`, `args = parser.parse_args()` at `T:128` (MODULE LEVEL — importing this module
runs parse_args(); see §11.5). Full flag list with defaults:

| Flag | Default | Notes |
|------|---------|-------|
| `--debug` | False | `T:38` |
| `--epochs` | 50 | `T:39` (real runs pass 15) |
| `--model_name` | `PEM_fine_tuned` | `T:40` |
| `--dataset_type` | `pnas` | `T:41` |
| `--unstable_mut` | False | `T:42` |
| `--one_mut` | False | `T:43` (best config sets it) |
| `--freeze_layers` | False | `T:44` |
| `--trained_model_path` | `.../43_final_model.pt` | `T:45` |
| `--dg_ml` | False | `T:46` clamps dG to [-1,5] |
| `--seed` | 42 | `T:48` |
| `--cosine_lr` | False | `T:49` |
| `--lr_min` | 1e-6 | `T:50` |
| `--grad_accum` | 1 | `T:51` |
| `--weight_decay` | 0 | `T:52` |
| `--use_huber_loss` | False | `T:54` |
| `--huber_delta` | 1.0 | `T:55` |
| `--use_ranking_loss` | False | `T:56` |
| `--ranking_lambda` | 0.1 | `T:57` |
| `--ranking_margin` | 0.1 | `T:58` |
| `--use_knn` | False | `T:60` |
| `--knn_k` | 30 | `T:61` |
| `--from_scratch` | False | `T:63` |
| `--emb_type` | `prott5` | `T:65-67` (prott5=1024, esmif_enc=512, dual_esmif=1536, saprot=1280, saprot_pm=1280, dual_saprot_pm=2304) |
| `--loss_type` | `l1` | `T:69-71` (l1/huber/huber_rank) |
| `--ranking_weight` | 0.1 | `T:72` |
| `--use_knn_gat` | False | `T:73` shortcut → use_knn + knn_k=30 (`T:148-150`) |
| `--no_pretrained` | False | `T:74` alias for from_scratch (`T:136-137`) |
| `--epochs_freeze` | None | `T:76` (parsed but NOT used — see §11.4) |
| `--epochs_unfreeze` | None | `T:77` (parsed but NOT used — see §11.4) |
| `--mini_batch_size` | **64** | `T:79` — DANGER default; runs MUST pass 16 (OOM lesson) |
| `--emb_projection` | `none` | `T:80-81` |
| `--serial_fusion` | False | `T:83` |
| `--serial_fusion_dim` | 64 | `T:85` |
| `--use_learned_aa` | False | `T:87` |
| `--aa_emb_dim` | 64 | `T:89` |
| `--length_norm` | False | `T:92` (v5) |
| `--corr_weight` | 0.0 | `T:94` (v5) |
| `--corr_type` | `pearson` | `T:96` (pearson/ccc) |
| `--data_root` | None | `T:99` |
| `--energy_terms` (A) | 1 | `T:103` |
| `--rbf_centers` (B) | 0 | `T:105` |
| `--rbf_min` (B) | 0.0 | `T:107` |
| `--rbf_max` (B) | 20.0 | `T:108` |
| `--use_burial` (C) | False | `T:109` |
| `--burial_radius` (C) | 10.0 | `T:111` |
| `--flory_unfolded` (D) | False | `T:113` |
| `--flory_nu` (D) | 0.5 | `T:115` |
| `--denoise_weight` (E) | 0.0 | `T:117` |
| `--denoise_sigma` (E) | 0.3 | `T:119` |
| `--denoise_prob` (E) | 0.5 | `T:121` |
| `--gcn_span` (F) | 1 | `T:123` |
| `--use_edge_features` (F) | False | `T:125` |

CFG mutation block: `T:198-215` pushes lever CLI values onto the single `CFG` object BOTH the
model and graph builder import. Note `CFG.burial_dim = 1 if args.use_burial else 0` at `T:203`
(there is NO `--burial_dim` CLI flag; the width is derived). **All A–F defaults reproduce baseline**
— the comment blocks in `C:68-111` state this and the code confirms it (per-lever sections below).

### 2. Lever A — energy_terms (fc2 shape, get_energy summation)

- `self.energy_terms = max(int(getattr(CFG,'energy_terms',1)),1)` at `M:391`.
- `self.fc2 = nn.Linear(128, self.energy_terms)` at `M:392` — output width = K (baseline K=1).
- forward: `x = self.fc2(x)` → `x.reshape(B, N, self.energy_terms)` at `M:543-545`.
- `get_energy` (`M:614-635`): `self.last_term_energies = torch.sum(Fh, dim=1).detach()` (`M:628`,
  per-term totals [B,K] for interpretability), then `E = torch.sum(Fh, dim=(1,2))` (`M:630`) — sums
  over residues AND terms; **for K=1 this is bit-identical to the historical `sum(dim=(1,2))`**.
- Length-norm interplay: `if self.length_norm: E = E / max(n_res,1)` at `M:631-633` (applied AFTER
  the per-term sum, so composes cleanly with K>1).

### 3. Lever B — rbf_centers (where applied)

- Applied in the GRAPH BUILDER, NOT the model. `apply_distance_kernel(D, gaussian_coef)` at
  `U:298-322`. `M<=0` → `torch.relu(torch.exp(gaussian_coef * D**2))` (`U:310-311`) = exact
  baseline single Gaussian. `M>0` → `torch.linspace(rbf_min,rbf_max,M)` centers, SUM of M RBFs
  `relu(exp(coef*(D-c)^2))` (`U:318-322`), dimension-preserving (same last-dim width 16, so no
  downstream dim change).
- Called from: folded `get_graph` at `U:210`; unfolded `get_unfolded_graph` at `U:238`; Flory path
  `flory_reference` at `U:287`. Fail-fast on inverted range at `U:315-317`.

### 4. Lever C — use_burial / burial_dim (node slicing, burial_start, fc widths)

- Model: `self.burial_dim = int(getattr(CFG,'burial_dim',0)) if getattr(CFG,'use_burial',False)
  else 0` at `M:349`; `b = self.burial_dim` at `M:350`.
- Width propagation: `gnn_internal_dim = 16 + aa_dim + proj_extra + serial_extra + b` (`M:354`);
  GCN raw fc-in `gcn_fc_in = 16+16+aa_dim+proj_extra+serial_extra+b` (`M:373`); GAT raw fc-in
  `gat_fc_in = 16+aa_dim+proj_extra+serial_extra+b` (`M:378`). So b=0 ⇒ baseline widths exactly.
- `self.burial_start = CFG.dist_dim*3` = 48 at `M:423`. Slice in forward:
  `x_burial = x[:, self.burial_start : self.burial_start+self.burial_dim]` at `M:465`, guarded by
  `if self.burial_dim > 0` (`M:464`). Concatenated into BOTH towers at `M:487-489`.
- Fail-fast: width mismatch raises "[Lever C]…" at `M:468-472`.
- Graph builder inserts burial BEFORE emb: layout `[D(16)|Fb(32)|burial(b)|emb(E)|one_hot(20)]` in
  `maybe_concat_burial` (`U:341-360`, cat at `U:359`); `compute_burial` counts CB (atom idx 3)
  neighbors within radius, `counts/max(N,1)`, returns [N,1] (`U:325-338`). Fail-fast `U:353-358`.
- CONSISTENCY: model reads emb from the RIGHT (`llm_index`), one_hot always last, so inserting
  burial after the 48 dist+bonded cols does NOT disturb the from-the-right emb/one_hot slices.

### 5. Lever D — flory_unfolded (unfolded graph branch)

- Branch in `get_unfolded_graph`: `if getattr(CFG,'flory_unfolded',False): return
  flory_reference(...)` at `U:235-236`. Baseline path (`U:237-253`) zeros off-tridiagonal via
  `zero_except_udiagonal` (`U:244`, defn `U:422-431`).
- `flory_reference` (`U:256-296`): builds coil distance `d_coil = b*|i-j|^nu` (`U:283`), b = mean
  CA-CA neighbor distance (`U:277-282`), broadcast across 16 channels (`U:286`), then applies the
  SAME kernel/mask/reduce as folded (`U:287-295`). Output SHAPE identical to baseline. Fail-fast
  nu∈(0,1] at `U:268-270`. Entirely a training-graph change; NO model change.

### 6. Lever E — denoise_weight (head, get_deltaG noise, loss, train-only) + ARITY

- Head construction: `self.denoise_weight = float(getattr(CFG,'denoise_weight',0.0))` (`M:401`);
  `if >0: self.denoise_head = nn.Sequential(Linear(128,128),ReLU,Linear(128,4*3))` (`M:402-404`);
  else `None` (`M:405-406`). `self.last_denoise_pred=None` (`M:407`). In forward, when head present:
  `dn = self.denoise_head(x); self.last_denoise_pred = dn.reshape(B,N,4,3)` at `M:531-533`; else the
  cache is set to None (`M:534-535`).
- Noise injection (TRAIN ONLY): in `get_deltaG` at `T:784-799`. `denoise_active` requires
  `train and denoise_head is not None and denoise_weight>0 and rand()<denoise_prob` (`T:790-794`).
  Noise added to FOLDED coords only: `coords_folded = coords_folded + noise; denoise_target=noise`
  (`T:797-799`). Validation calls `get_deltaG(batch, j)` with `train=False` (default) at `T:708`
  ⇒ no perturbation, exact baseline coords.
- Aux loss: computed on FOLDED half only at `T:840-846`
  (`pred_folded = last_denoise_pred[:n_all//2]`, `F.mse_loss(pred_folded, target)`), added as
  `dn_term = CFG.denoise_weight * denoise_loss` to the TRAIN loss at `T:617-622`. NEVER in validate.

- **⚠ ARITY — CRITICAL:** `get_deltaG` returns **FOUR** values:
  `return unfolded_energy - folded_energy, unfolded_energy, folded_energy, denoise_loss` (`T:848`).
  - TRAIN unpacks 4: `output,u_energy,f_energy,denoise_loss = self.get_deltaG(batch, j, train=True)`
    at `T:588`.
  - VALIDATE unpacks 4 (discards 4th): `output,u_energy,f_energy,_ = self.get_deltaG(batch, j)`
    at `T:708`.
  Both unpack sites match the 4-tuple. **This differs from the baseline pnas_train.py which returns
  a 3-tuple `(dG, u_energy, f_energy)`.** Any teaching doc that says v5 get_deltaG returns 3 values is
  WRONG (see §11.1). ddG sign convention: `output = unfolded − folded` (`T:848`), same as baseline.

### 7. Lever F — gcn_span / use_edge_features (span logic, build_edge_attr, GAT edge_dim)

- `self.gcn_span = max(int(getattr(CFG,'gcn_span',1)),1)` at `M:410`.
- `get_edge_index` (`M:637-702`): S<=1 → directed (i,i+1) only (`M:656-660`) = baseline bit-exact.
  S>1 → for s in 1..S add forward (i→i+s) AND backward (i+s→i) edges (`M:661-677`).
- Edge features: `self.use_edge_features = bool(getattr(CFG,'use_edge_features',False))` (`M:363`);
  `edge_dim = 41 if use_edge_features else None` (`M:364`); passed into every `GAT(...,edge_dim=)`
  (`M:365`) → `GATv2Conv(..., edge_dim=edge_dim)` for BOTH conv layers (`M:811-812`).
- `build_edge_attr` (`M:585-612`): `[onehot_src(20)|onehot_dst(20)|ca_dist(1)]` = 41 dims (`M:605`).
  If `ca_coords is None` the dist term is zeros (`M:603-604`). Fail-fast width≠41 at `M:608-611`.
- forward passes edge_attr: built at `M:507-508` (only if use_edge_features), threaded through
  `forward_gat(..., edge_attr=edge_attr)` (`M:509,552,565`) → `gat_layer(..., edge_attr=)` (`M:565`)
  → `self.gat1/gat2(h, edge_index, edge_attr=edge_attr)` (`M:820,822`).
- CA supply: in `get_deltaG`, when `use_edge_features` but NOT knn, ca_coords are still provided
  (fully-connected topology unchanged) at `T:828-834`. Under knn, cutoff+ca set at `T:813-827`.

### 8. forward() feature slicing + index constants

Layout: `[dist:16 | bonded:32 | (burial:b) | emb:E | one_hot:20]` (comment `M:457`). Reshape
`x -> [B*N, features]` at `M:456`. Slices (`M:459-465`):
- `x_dist = x[:, :16]` (`non_bonded_index=16`, `M:419,459`).
- `x_bonded = x[:, 16:32]` — **only first 16 of the 32 bonded cols** (bonded-to-previous), matching
  original (`M:460`; comment `M:457-458`).
- `x_emb_features = x[:, llm_index : one_hot_index]` where `one_hot_index=-20` (`M:417`) and
  `llm_index = -(CFG.emb_input_dim+20)` (`M:420`). Dynamic ⇒ correct for any emb dim (1024/1536/…).
- `x_onehot = x[:, -20:]` (always last, `M:462`).
- `x_burial = x[:, 48:48+b]` when b>0 (`M:465`).
- Index constants: `one_hot_index=-20` (`M:417`), `bonded_index=48` (`M:418`, set but UNUSED in
  slicing), `non_bonded_index=16` (`M:419`), `llm_index` dynamic (`M:420`), `burial_start=48`
  (`M:423`).
- aa handling: learned_aa → `aa_indices = x_onehot.argmax(-1); x_aa = self.aa_embedding(...)`
  (`M:477-479`); else `x_aa = x_onehot` (20-dim) (`M:480-481`).
- GCN input = `cat(x_dist, x_bonded, x_aa)` (`M:484`); GAT input = `cat(x_dist, x_aa)` (`M:486`);
  burial appended to both (`M:487-489`); proj/serial appended (`M:492-501`).

### 9. Fail-fast / validation checks (what they guard)

- `T:180-197`: pre-training lever value guards (A≥1, B≥0 & range, C radius>0, D nu∈(0,1], E weight≥0
  & prob∈[0,1] & sigma>0, F span≥1) — each raises a "[Lever X]…" message BEFORE any training work.
- `M:468-472`: burial slice width ≠ burial_dim (short/misaligned node vector) — Lever C.
- `M:608-611`: edge_attr width ≠ 41 (one_hot width must be 20) — Lever F.
- `U:315-317`: RBF range inverted/empty — Lever B.
- `U:353-358`: computed burial width ≠ CFG.burial_dim (builder/model disagree on layout) — Lever C.
- `U:268-270`: Flory nu∈(0,1] — Lever D.
- Guard against 1-mutation proteins: `if batch['delta_g'].size(1) == 1: continue` (`T:583-584`).

### 10. Light attention, GCN tower, GAT tower — exact dims

- `gnn_internal_dim = 16 + aa_dim + proj_extra + serial_extra + b` (`M:354`). Default (no levers,
  no projection): 16 + 20 = **36**. NOTE: `CFG.emb_projection` default is "mlp" (`C:34`) but training
  sets it from `--emb_projection` default `none` (`T:174`), so proj_extra=0 in normal runs (see §11.3).
- GCN: `GCN(gcn_dim_in=36, gcn_dim_h=64, gcn_dim_out=36)` ×layers (`M:355-358`); internals
  `GCNConv(36→64)`, `GCNConv(64→36)`, InstanceNorm, input dropout (`M:833-850`).
- GAT: `GAT(gat_dim_in=36, gat_dim_h=64, gat_dim_out=36, heads=8, edge_dim)` ×layers (`M:359-365`);
  `GATv2Conv(36→64, heads=8)` then `GATv2Conv(64*8=512→36, heads=1)` (`M:811-812`).
- fc towers: `fc1_gcn: Linear(gcn_fc_in=48+aa_dim... →64)`, `fc2_gcn: Linear(64→36)` (`M:374-375`);
  `fc1_gat: Linear(gat_fc_in=16+aa_dim... →64)`, `fc2_gat: Linear(64→36)` (`M:379-380`).
- Concat GCN+GAT outputs → `inst_norm2(gcn_out+gat_out = 72)` (`M:384,511-514`).
- Post-GNN emb concat: `post_gnn_emb = 0 if projector else CFG.emb_input_dim` (`M:344`); raw emb
  re-appended only when NO projector at `M:517-518`.
- `fc_in_dim = gcn_dim_out + gat_dim_out + post_gnn_emb` (`M:386`) = 36+36+1024 = **1096** (prott5,
  no projection). `self.fc1 = Linear(fc_in_dim, 128)` (`M:387`); `self.fc2 = Linear(128, K)` (`M:392`).
- LightAttention: `if light_attention: self.LA = LightAttention(embeddings_dim=fc_in_dim=1096)`
  (`M:434-435`); applied on the [B,N,fc_in_dim] tensor (swap to channels-first) at `M:520-525`. Class
  at `M:872-903`: two `Conv1d(dim,dim,k=9,pad=4)`, `o = feature_conv(x)`, `attn = attention_conv(x)`,
  returns `o * softmax(attn)` (`M:896-902`). NOTE the `output_dim=11` default arg is IGNORED (the
  constructor only uses `embeddings_dim`); it is a same-width gated output, not an 11-class head.
- Extra heads: `fc2_sm = Linear(128,20)` for the subtract-mut branch (`M:396,538-541`) — UNUSED by
  pnas_train_v5 (never passes `f_type='subtract_mut'`); denoise head (§6).

### 11. Things the teaching docs might get wrong / omit

11.1 **get_deltaG arity = 4, not 3.** v5 added `denoise_loss` as a 4th return (`T:848`). Baseline
   returns 3. Both train (`T:588`) and validate (`T:708`) unpack 4. A doc copied from baseline
   describing a 3-tuple is WRONG for v5.

11.2 **`--mini_batch_size` default is 64** (`T:79`), the exact value that caused historical OOM.
   Baseline hardcoded 16. Any "just run pnas_train_v5.py" instruction WITHOUT `--mini_batch_size 16`
   will OOM on 8 GB. Footgun the docs must call out (matches project MEMORY.md lesson).

11.3 **CFG.emb_projection default = "mlp"** (`C:34`) but the TRAINING default is "none" (argparse
   `T:80`, applied `T:174`). The effective run uses raw post-GNN concat (no projection). If a doc
   quotes CFG's "mlp" as the operative default it is misleading — the CLI overrides it.

11.4 **`--epochs_freeze` / `--epochs_unfreeze` are parsed but NEVER used.** `run_training` uses only
   `args.epochs` (`T:877`) for single-stage training. No freeze/unfreeze schedule logic consumes
   these flags; `FREEZE_LAYERS` comes from the separate `--freeze_layers` flag (`T:229`). The
   module-level `EPOCHS_FREEZE=20 / EPOCHS_NO_FREEZE=60` constants (`T:227-228`) are also dead.

11.5 **Module-level argparse (`T:128`).** Importing `pnas_train_v5` runs `parse_args()`. Any script
   importing it (e.g. ensemble) must inject matching `sys.argv` first (documented in MEMORY.md).

11.6 **serial_fusion / use_learned_aa / length_norm details:**
   - serial_fusion (`M:326-331,498-501`): MLPProjection(emb_input_dim → 2*dim → dim=64) added into
     BOTH GNN towers (message-passing). Default off ⇒ serial_extra=0.
   - use_learned_aa (`M:334-338,477-479`): nn.Embedding(20, aa_emb_dim=64) replaces the 20-dim
     one-hot; changes aa_dim 20→64 everywhere in the width formulas. Default off.
   - length_norm (`M:308,631-633`): the ONLY change to get_energy — divide total by residue count.
     Default off ⇒ extensive sum (baseline).

11.7 **Correlation loss interplay (`T:607-615`, `L:12-44`).** `--corr_weight>0` adds
   `corr_weight * (pearson_loss | ccc_loss)` to the primary loss. `pearson_loss = 1 - r` (`L:12-27`),
   `ccc_loss = 1 - CCC` (`L:30-44`); both return 0 for <2 elements (`L:18-19,36-37`). Added to the
   TRAIN loss alongside primary+reg+energy_reg+rank (`T:615`) and BEFORE the denoise term
   (`T:617-621`). Validation loss (`T:710-713`) uses ONLY criterion + energy_reg — corr loss does
   NOT affect the reported val number; a doc implying otherwise is wrong.

11.8 **energy_reg is ALWAYS on** (`E_REG_LAMBDA=0.001`, `T:244`; used in train `T:600` and val `T:713`)
   while `REG_LAMBDA=0` (`T:243`) makes the parameter-L2 `reg_loss` a no-op by default. Not a v5
   change but easily misdescribed.

11.9 **k-NN GAT mutates `self.model.gat_cutoff` per mini-batch** (`T:827`) from a k-NN-derived cutoff,
   not a fixed 12Å. The constructor `gat_cutoff=12.0 if args.use_knn_gat else None` (`T:866`) is
   overwritten at runtime. Distance edges built in `get_edge_index` at `M:680-687`.

11.10 **Where the levers live:** B (kernel) and D (unfolded reference) are ENTIRELY in the graph
   builder (`train_utils.py`) — they never touch `hydro_net_v5.py`. A (energy_terms), E (denoise),
   F (span/edge_feat) touch the MODEL. C (burial) touches BOTH (builder inserts the feature, model
   slices it). A doc that files all six as "model changes" is inaccurate.

11.11 **Bonded feature width nuance:** the builder emits 32 bonded cols (`get_bonded_features`,
   `U:363-374`: f1=above-diag, f2=below-diag, each 16 → 32) but the model consumes ONLY the first 16
   in the GCN (`x_bonded = x[:,16:32]`, `M:460`). The second 16 bonded cols are effectively unused.
   Baseline behavior preserved, but worth noting for completeness.

11.12 **Single-CFG import (v5 fix).** `hydro_net_v5` imports `from model.model_cfg_v5 import CFG`
   (`M:11`) — the SAME object `pnas_train_v5` mutates (`T:23`), so lever widths and emb_input_dim
   propagate correctly (this is the split-CFG bug the MEMORY records as FIXED).

<!-- AGENT 2 DONE -->


---

## Agent 3 — Data pipeline + losses (new_dataset.py, train_utils.py, losses_v5.py, ensemble_v5.py)

> Auditor: Agent 3. Scope = DATA PIPELINE + LOSSES. Files read in full:
> `training/new_dataset.py` (D), `training/train_utils.py` (U), `training/losses_v5.py` (L),
> `training/ensemble_v5.py` (E), `config_paths.py` (P). Loss wiring cross-checked in
> `training/pnas_train_v5.py` (T). All claims cite `file:line`. NO code was modified.

### 1. The dataset — what Megascale/PNAS data IS (biology) + exact filters

**What the data is (biology).** The training data is the **Tsuboyama et al. 2023 (PNAS/Science)
"mega-scale" folding-stability dataset**, measured by the **K50 cDNA-display proteolysis assay**.
Small protein domains (typically ~40–72 residues) are exposed to a protease (trypsin/chymotrypsin);
folded proteins resist cleavage, unfolded ones are cut. By titrating protease concentration you get
**K50** = the protease concentration at the unfolding midpoint, from which a **folding free energy dG
(kcal/mol)** is inferred for the wild type and for each single-point mutant. **ddG = dG_mut − dG_wt**
is the change in stability upon mutation. This is why the model regresses a per-mutation `deltaG`
tensor and computes ddG relative to the WT reference (index 0). Each protein folder stores
`coords.pt`, `deltaG.pt`, `mask.pt`, `emb.pt` (ProtT5), and the per-mutation CSV lives under
`mutation_files/<protein>.csv` (`P:40-47`, `D:16-21`).

**Data-root resolution.** All paths derive from a single root (`set_data_root` > `DEEPPEF_DATA_ROOT`
env > `./data`, `P:25-35`). Derived: `tensor_root=data/MsDs/training_data` (`P:40-42`),
`mut_root=data/MsDs/mutation_files` (`P:45-47`), PNAS whitelists under
`Processed_K50_dG_datasets/Pnas_filtering/` (`train_proteins.csv`, `pnas_mutations.csv`, `P:50-59`),
ThermoMPNN splits under `ThermoMPNN/mega_test.csv` & `mega_train.csv` (`P:62-69`).

**The exact filters (order matters), in `MSDataset.load_protein_data` (train) `D:244-310`, mirrored
in `load_test_protein_data` `D:175-241`:**

1. **ins/del removal** (`D:248-263` train; `D:180-195` test): rows whose `mut_type` matches the
   regex `'ins|del'` are dropped from BOTH the mutation dataframe and `delta_g_tensor` (built via a
   boolean `ins_del_mask`, `D:258-263`). Reason: the model uses a FIXED WT backbone of fixed length,
   so insertions/deletions change chain length/register and are structurally incompatible.
2. **dg_ml clamp to [-1, 5]** (`D:271-275` train; `D:203-207` test): if `self.dG_ml` (module default
   `DG_ML=True`, `D:34`), `delta_g_tensor` is clamped via two `torch.where` calls to the interval
   `[-1.0, 5.0]` kcal/mol. Biology: the K50 proteolysis assay only *resolves* dG within roughly this
   window — below ~−1 the protein is effectively fully unfolded and above ~5 fully folded (the
   protease readout saturates), so values outside are unreliable noise. Clamping stops the regressor
   from chasing unmeasurable extremes. NOTE: this is a **clamp (saturation), not a mask** — clamped
   rows are still kept as training targets pinned at the boundary value.
3. **unstable-mut removal (OPT, default OFF)** (`D:280-281` train; `D:211-212` test): only if
   `not self.unstable_mut` do we drop rows where `ddG_ML == '-'`. Module default `UNSTABLE_MUT=True`
   (`D:36`) ⇒ this branch is SKIPPED by default, i.e. unstable mutations are KEPT.
4. **one_mut removal** (`D:284-285` train; `D:215-216` test): if `self.one_mut`, drop rows whose
   `mut_type` contains ':' (multi-site mutations). The per-mutation signal is a single-residue change;
   multi-site entries mix effects and the ProtT5 delta is less cleanly attributable.
5. **PNAS whitelist** (`D:287-289` train; `D:218-220` test): if `ds_type=='pnas'` (train) or
   `ds_type in ('pnas','deepef1')` (test), drop every mutation whose `name` is NOT in the curated
   whitelist — train uses `self.pnas_mutations` = `pnas_mutations.csv` (`D:61,289`); test uses
   `self.test_mutations` = the ThermoMPNN test CSV (`D:62,220`). This is the curation that MEMORY
   records as CRITICAL (unfiltered full-megascale HURT PCC 0.41 vs 0.53: quality > quantity).
6. **ThermoMPNN test-protein hold-out** (`D:53-60`): at construction, `tm_proteins` = unique
   `WT_name` (`.pdb` stripped) from `mega_test.csv` (`D:54-55`); `self.test_protein` = intersection
   (`D:57`), `self.protein_dirs` = complement (`D:58`). So test proteins never enter the training
   protein list. `__len__` returns `len(protein_dirs)` (train) or `len(test_protein)` (test) `D:67-71`.
7. **Homolog removal** (`remove_homologs`, `D:79-83`): TRAIN ONLY (`if self.train`, `D:81`). Reads
   `mega_train.csv`'s `WT_name` (`.pdb` stripped) and removes any of those from `protein_dirs`
   (`D:82-83`). Prevents train/test leakage against the ThermoMPNN benchmark. Called at end of
   `__init__` (`D:65`). NB: the variable is confusingly named `pnas_proteins` but it reads the
   ThermoMPNN *train* CSV (`P:67-69`), not the PNAS whitelist.

**Filtering is set-based, and WT is force-kept at index 0.** Filters are accumulated by SUBTRACTING
index sets from `indexes = set(mutations.index)` (`D:278-289`), then **`indexes.add(0)`** guarantees
row 0 (the wild type) survives every filter (`D:292` train; `D:223` test). `indexes` is then
listified and used to slice `mutations`, `delta_g_tensor`, `one_hot_tensor`, `embedding_tensor`
(`D:293-298`). **Caveat for teaching:** `set` ordering + `list(set)` means the returned row ORDER is
not the CSV order; downstream ddG uses `val_dg - val_dg[0]`, and index-0 = WT is the only positional
guarantee. (Agent 2 §2 confirms ddG is `val_dg - val_dg[0]`.)

**`__getitem__` return** (`D:73-77` → `load_protein_data`/`load_test_protein_data`): a **dict**
(`D:300-308` train; `D:231-239` test) with keys:
- `'name'`: protein id (str),
- `'mutations'`: list of `mut_type` strings,
- `'prott5'`: embedding tensor `[n_muts, seq_len, emb_dim]` (emb_dim depends on EMB_TYPE, see below),
- `'coords'`: `[seq_len, 3, 3]` backbone N/CA/C (CB added later via `add_cb`; note coords are in
  nanometers and are ×10 to Angstrom in the trainer, `T:325`),
- `'one_hot'`: `[n_muts, seq_len, 20]` (stacked `get_one_hot` over each mutant `aa_seq`, `D:269`),
- `'delta_g'`: `[n_muts]` clamped dG targets,
- `'masks'`: valid-residue mask.

**Embedding selection (`_load_and_concat_embeddings`, `D:85-173`).** Module default `EMB_TYPE='prott5'`
(`D:38`). The raw `emb.pt` is a LIST of per-mutation embeddings; the loader keeps only entries whose
`shape[0]==seq_len` and stacks them (`D:170-173`). Other modes: `esmif_enc` (ESM-IF1 encoder 512-dim,
WT-broadcast to all muts, `D:92-104`), `dual_esmif` (ProtT5 1024 ⊕ ESM-IF1 512 = 1536, `D:106-125`),
`saprot`/`saprot_pm` (1280, `D:127-147`), `dual_saprot_pm` (1024+1280=2304, `D:149-168`). Missing
feature files fall back to ProtT5 or zero-pad. **GAP for docs:** the `esmif_enc`/`dual_esmif` modes
broadcast the SAME WT structural embedding to every mutation (`.expand`, `D:104,123`) — so for those
modes the per-mutation signal comes ONLY from one-hot + (for dual) the mutant ProtT5, not from ESM-IF.

### 2. Graph construction step-by-step (`train_utils.py`) — exact shapes

Two builders: `get_graph` (folded, `U:196-223`) and `get_unfolded_graph` (unfolded, `U:225-253`).
Both take `x` coords `[N, 4, 3]` (4 atoms N,CA,C,CB after `add_cb`), `one_hot [N,20]`, `emb [N,E]`,
`mask [N]`, and `gaussian_coef` (default `CFG.gaussian_coef`).

1. **Distance matrix — `get_dist_matrix(Xd)` (`U:377-391`).** Input `[N, N_atoms=4, coords=3]`.
   Reshapes to `[N*4, 3]`, computes full pairwise Euclidean distance `torch.cdist(.,.,p=2)`
   `[N*4, N*4]` (`U:387`), reshapes to `[N, 4, N, 4]`, swaps axes 1↔2 → `[N, N, 4, 4]`, flattens the
   last two → **`D = [N, N, 16]`** (16 = all 4×4 atom–atom distances per residue pair) (`U:388-390`).
2. **Distance kernel (Lever B) — `apply_distance_kernel(D, coef)` (`U:298-322`).** Baseline
   (`CFG.rbf_centers<=0`): `D = relu(exp(coef * D**2))` (`U:310-311`) — a single Gaussian *contact*
   kernel: distance 0 → 1, large distance → 0, so it turns raw distances into soft contact strengths.
   Lever B (`M>0`): a bank of M RBFs `sum_c relu(exp(coef*(D-c)**2))` over `linspace(rbf_min,rbf_max,M)`
   centers, **summed** so the last-dim width stays 16 (dimension-preserving; `U:318-322`). Fail-fast
   on inverted range (`U:315-317`). Shape unchanged: `[N,N,16]`.
3. **Mask zeroing (`U:210-214` folded; `U:239-242` unfolded).** `mask_index = where(mask==0)`; then
   `D[mask_index,:,:]=0` and `D[:,mask_index,:]=0` — zero out both rows and columns of masked
   residues (they contribute no contacts).
4. **Unfolded ONLY: keep chain-local contacts — `zero_except_udiagonal(D)` (`U:244`, defn
   `U:422-431`).** Saves the main diagonal `D[i,i]` and the two off-diagonals `D[i,i+1]` (f1) and
   `D[i+1,i]` (f2), zeros everything else, then writes those three bands back (`U:425-430`). Result:
   the unfolded graph retains ONLY the linear-chain (tridiagonal) contacts — no long-range 3D
   structure. Folded keeps ALL contacts. Coordinates are IDENTICAL between the two states; only the
   contact map is stripped. (Lever D `flory_unfolded` replaces this with a polymer random-coil
   reference — see Agent 2 §5 / `U:256-296`.)
5. **Bonded features — `get_bonded_features(D)` (`U:363-374`).** Extracts the two off-diagonals:
   `f1 = D[i,i+1]` (contact to NEXT residue), `f2 = D[i+1,i]` (contact to PREVIOUS), each `[N-1,16]`.
   Pads f1 with a zero row at the BOTTOM and f2 with a zero row at the TOP (`U:369-371`), then
   concatenates → **`Fb = [N, 32]`**. This is computed on the *kernelized* D (so bonded features are
   contact strengths of adjacent residues). NB the model consumes only the first 16 of these 32
   (Agent 2 §11.11).
6. **Sum over neighbors + normalize (`U:216-220` folded; `U:247-250` unfolded).**
   `D = D.sum(dim=1)` collapses the neighbor axis → **`D = [N, 16]`** (each residue's total contact
   strength per atom-pair channel). Then `D = F.normalize(D, p=2, dim=0)` — L2-normalize each of the
   16 channels ACROSS residues (dim=0 = the N axis). `emb = F.normalize(emb, p=2, dim=0)` likewise
   L2-normalizes each embedding channel across residues (`U:219-220`).
7. **Final concat — `maybe_concat_burial(x, D, Fb, emb, one_hot, mask)` (`U:341-360`).** Baseline
   (no burial): `Fh = cat([D(16), Fb(32), emb(E), one_hot(20)], dim=1)` (`U:360`) → **`Fh =
   [N, 16+32+E+20]` = `[N, 68+E]`** (prott5: 1092). Lever C inserts a burial scalar BEFORE emb:
   `[D(16)|Fb(32)|burial(b)|emb(E)|one_hot(20)]` (`U:359`), with one_hot ALWAYS last so the model's
   from-the-right slices are unaffected (fail-fast width check `U:353-358`).

**Final layout (baseline):** `[ D_sum:16 | Fb:32 | emb:E | one_hot:20 ]`, shape `[N, 68+E]`.

### 3. `add_cb` — synthesizing CB (incl. glycine) (`U:441-466`)

Input `crd_coords [N, 3, 3]` (N, CA, C). Computes vectors `CAmN = N-CA`, `CAmC = C-CA`, and their
cross product `ANxAC = cross(CAmN, CAmC)` (`U:451-457`). Stacks these three vectors columnwise into a
per-residue basis `A` (`U:459`), multiplies by fixed coefficients
`c = tensor([0.5507, 0.5354, -0.5691]) / 100` (`U:460`) to get displacement `b = (A @ c)` reshaped
`[N,3]`, and sets **`CB = CA - b`** (`U:461-462`). The new CB is concatenated as a 4th atom →
`[N, 4, 3]` (`U:465`). **Glycine handling: there is NONE special** — CB is synthesized geometrically
from the N/CA/C backbone for EVERY residue including glycine (which has no real CB). This is a
deliberate "virtual CB" so all residues have a uniform 4-atom representation; the distance matrix and
`compute_burial` (which uses atom index 3 = CB, `U:332`) therefore always have a CB even for Gly.
Worth flagging in teaching docs (glycine's CB is a fabricated placeholder, not measured).

### 4. Loss definitions WITH FORMULAS

**v5 correlation losses (`losses_v5.py`).** Motivation (`L:1-7`): the model is graded on Pearson
correlation of predicted vs true dG, but Huber/L1 only penalize point-wise error; two predictions can
share a Huber loss but differ wildly in PCC. These losses reward getting the ORDERING/trend right.

- **`pearson_loss(pred, target, eps=1e-8)` (`L:12-27`)** returns **`1 − r`** where r is Pearson's:
  ```
  pred_c = pred − mean(pred);  target_c = target − mean(target)
  r = Σ(pred_c · target_c) / sqrt( Σ pred_c² · Σ target_c² + eps )
  loss = 1 − r          # range [0,2]; 0 = perfect positive correlation
  ```
  Returns `0.0` if `numel < 2` (`L:18-19`); eps guards zero variance (`L:25-26`).

- **`ccc_loss(pred, target, eps=1e-8)` (`L:30-44`)** returns **`1 − CCC`** (Lin's Concordance
  Correlation Coefficient). **CCC** measures agreement combining correlation AND shift/scale match:
  ```
  CCC = 2·cov(pred,target) / ( var(pred) + var(target) + (mean(pred) − mean(target))² )
  ```
  where `var` is the BIASED (population) variance `var(unbiased=False)` and
  `cov = mean((pred−mp)·(target−mt))` (`L:41-43`). It equals Pearson r scaled by a bias-penalty term:
  it drops if the predictions are shifted (mean offset) or scaled differently from the target, so it
  is STRICTER than Pearson (rewards absolute agreement of dG, not just trend). Returns `0.0` if
  `numel < 2` (`L:36-37`).

**Where the primary/Huber/ranking/energy_reg losses live (`pnas_train_v5.py`).** The v5 corr losses
are a small ADD-ON; the full loss stack is assembled in the train inner loop:
- **Total (`T:615`):** `loss = primary_loss + reg_loss + energy_reg + rank_loss + corr_loss`, then
  (Lever E) `loss += CFG.denoise_weight * denoise_loss` (`T:617-621`).
- **`primary_loss = self.criterion(output, delta_g)` (`T:592`)** — `criterion` is `nn.HuberLoss(delta)`
  when `--use_huber_loss` (set by `--loss_type huber`/`huber_rank`, `T:140-144`) else `nn.L1Loss`.
  Huber: quadratic for |err|<δ, linear beyond (robust to outliers), δ=`huber_delta` default 1.0
  (`T:55`). Best config uses `huber_rank`.
- **`reg_loss = REG_LAMBDA * Σ MSE(param, 0)` (`T:595-598`)** — parameter-L2; `REG_LAMBDA=0` default
  ⇒ NO-OP (see Agent 2 §11.8).
- **`energy_reg = E_REG_LAMBDA * MSE(cat(u_energy, f_energy), 0)` (`T:599-600`)** — `E_REG_LAMBDA=0.001`,
  ALWAYS ON; pulls the raw folded/unfolded energies toward 0 to keep the extensive sum bounded. Also
  applied in VALIDATION loss (`T:712-713`).
- **`rank_loss = RANKING_LAMBDA * ranking_loss(output, delta_g, margin)` (`T:603-605`)**, gated by
  `USE_RANKING`. **`ranking_loss` (`T:331-356`)**: samples ≤128 random index pairs (i,j) from the
  mini-batch (`T:340-346`), and applies a **margin ranking loss**:
  ```
  sign = sign(target_i − target_j)
  loss = mean( clamp( −sign·(pred_i − pred_j) + margin, min=0 ) )
  ```
  i.e. if target_i>target_j the model is penalized unless pred_i exceeds pred_j by at least `margin`
  (`T:352-355`). Default `RANKING_LAMBDA=0.1`, `margin=0.1`.
- **`corr_loss` (`T:607-613`):** gated by `args.corr_weight > 0`; `corr_weight * ccc_loss` if
  `--corr_type ccc` else `corr_weight * pearson_loss`. Default `corr_weight=0.0` ⇒ OFF (exact baseline).
- **VALIDATION loss (`T:710-713`)** uses ONLY `criterion + energy_reg` — corr/rank/denoise do NOT
  enter the reported val number (matches Agent 2 §11.7).

### 5. `ensemble_v5.py` — checkpoint reload, mirrored flags, argv injection, config_paths

Purpose (`E:1-5`): load `best_model.pt` from several seed dirs, average the per-mutation predicted dG,
report the ensembled PCC (+0.02–0.04 typical).

- **argv injection (`E:58-97`).** Because `pnas_train_v5` runs `argparse` at MODULE LEVEL (importing
  it triggers `parse_args()`; Agent 2 §11.5), the ensemble BUILDS a `train_argv` list mirroring the
  training run (`E:64-71`: dataset_type pnas, no_pretrained, one_mut, dg_ml, huber_rank,
  ranking_weight 0.1, mini_batch_size 16, emb_type, emb_projection, model_name), conditionally appends
  the architecture flags (`E:72-90`), then does `saved_argv=sys.argv; sys.argv=train_argv; import
  pnas_train_v5 as T; finally sys.argv=saved_argv` (`E:92-97`). This makes `T.DEVICE`, `T.USE_KNN`,
  `T.KNN_K`, `CFG.emb_input_dim`, and `T.args` consistent with training. This is the exact fix MEMORY
  records for the earlier ensemble crash.
- **Which flags it mirrors (`E:38-55`).** Shape/topology-relevant flags that MUST match the checkpoint:
  `--length_norm`, `--serial_fusion`, `--use_knn_gat`, `--use_learned_aa`, `--emb_type`,
  `--emb_projection`, and shape-changing levers A/C/E/F (`--energy_terms`, `--use_burial`,
  `--burial_radius`, `--denoise_weight`, `--gcn_span`, `--use_edge_features`). Levers **B (rbf) and D
  (flory) are deliberately OMITTED** (`E:46-48`) because they are shape-preserving/value-only — they
  only change how graphs are built at TRAIN time and do NOT affect `state_dict` shapes.
- **Model reconstruction + reload (`E:111-128`).** For each dir: `ckpt = <dir>/best_model.pt`
  (`E:112`); build `PEM(...)` with the SAME args as training — `layers/gaussian_coef/dropout_rate`
  from `CFG`, `light_attention=True`, `emb_projection=T.args.emb_projection`, `gat_cutoff=12.0`,
  `serial_fusion/use_learned_aa/length_norm` from the mirrored args (`E:115-121`) — then
  `model.load_state_dict(torch.load(ckpt, map_location=T.DEVICE))` (`E:122`). It reuses
  `T.Trainer(model, test_ds, test_ds).validate(0, test=True)` to get a `val_df` (`E:123-124`), and
  collects `val_df['pred_deltaG']` per model (`E:125-126`).
- **Ensembling + PCC (`E:130-132`).** `mean_pred = mean(vstack(preds_per_model), axis=0)`;
  `pcc = np.corrcoef(all_true, mean_pred)[0,1]`. **NB:** it averages *dG* predictions — PCC is on dG.
  `all_true` is taken from the first model's `val_df['deltaG']` (`E:127-128`).
- **config_paths usage (`E:101-107`).** `tensor_root()`/`mut_root()` drive the test `MSDataset(...,
  train=False)` DataLoader (batch_size=1, shuffle=False). So the ensemble evaluates on the SAME
  ThermoMPNN test hold-out defined in §1.
- **Gotcha for docs:** `load_state_dict` is called WITHOUT `strict=False` (`E:122`), so any lever
  mismatch between mirrored flags and the checkpoint raises a size-mismatch error — this is the
  intended fail-fast, but a doc must warn that the ensemble flags have to EXACTLY match training.

### 6. Noise-injection functions (used by Lever E and pretraining)

- **`add_gaussian_noise(X_native, sigma)` (`U:393-398`):** `X_decoy = X_native + randn(shape)*sigma`
  — isotropic Gaussian coordinate noise. Lever E (denoise) adds Gaussian noise to the FOLDED coords
  during training and asks a `denoise_head` to predict the noise (Agent 2 §6; injection at
  `T:784-799`, `denoise_sigma` default 0.3, `denoise_prob` 0.5). This is a denoising-score-matching-
  style auxiliary task on structure.
- **`Add_random_step(X, h=CFG.h)` (`U:468-479`):** returns a symmetric pair `X1=X+h·v`, `X2=X−h·v`
  with `v=randn(shape)` — the finite-difference perturbation used by the pretraining DSM gradient
  estimate (`v·∇E ≈ (E(X+εv)−E(X−εv))/2ε`).
- **`mix_A_acid(seq_one_hot, emb, mask, val_type, device)` (`U:121-136`):** SEQUENCE noise, not
  coordinate noise — builds a "decoy" by either fully permuting residues (`val_type in {'robust',
  'train'}`, `U:123-127`) or swapping just two positions (else branch, `U:128-135`), shuffling the
  one-hot, mask, AND embedding consistently. Used in `get_noised_proteins` to make shuffled-sequence
  decoys (`U:518`) for the native-vs-decoy pretraining objective.
- `get_noised_proteins` (`U:488-551`) assembles the full pretraining graph set (folded/unfolded
  native, shuffled-seq decoy, decoy STRUCTURE, unfolded decoy, 2 cyclic permutations) — this is the
  EBM pretraining path, NOT the ddG fine-tuning path (fine-tuning uses `get_deltaG` in the trainer).

### 7. GAPS the teaching docs may have about the DATA

1. **The K50 assay biology is not obvious from code.** Docs should state plainly: dG comes from a
   **proteolysis (protease-susceptibility) mega-scale assay** (Tsuboyama 2023), K50 = protease
   midpoint concentration; folded ⇒ protease-resistant. This is the ground truth behind `deltaG.pt`.
2. **Why clamp to [-1, 5]** (`D:270-275`): it's the assay's *dynamic range* — outside it the readout
   saturates (fully unfolded / fully folded) and dG is unmeasurable. It's a SATURATION clamp, not a
   drop; boundary rows are kept pinned at ±the limit.
3. **"Decoy" definition (for the pretraining story).** A decoy is a *deliberately wrong* protein
   representation the EBM must assign HIGHER energy than the native: (a) shuffled sequence
   (`mix_A_acid`, `U:121-136`), (b) a wrong 3-D structure (`crd_decoy`, `U:498`), and (c) cyclic
   sequence permutations (`U:526-527`). The unfolded state is a related "reference", not a decoy.
   Docs should not conflate "decoy" (pretraining, native-vs-decoy) with "unfolded" (fine-tuning dG).
4. **Virtual CB for glycine (`U:441-466`).** All residues, including Gly, get a *synthesized* CB from
   backbone geometry. Burial (`compute_burial`, `U:325-338`) and 4-atom distances therefore always use
   a CB even where none exists biologically — a modeling simplification worth stating.
5. **Filter default subtleties.** `unstable_mut` defaults to KEEPING unstable mutations
   (`UNSTABLE_MUT=True`, `D:36`; the drop only fires when the flag is False, `D:280`). `dg_ml` and
   `one_mut` module defaults are True (`D:33-34`) but the trainer sets them from CLI. The best-config
   filters `--one_mut --dg_ml --dataset_type pnas` are ESSENTIAL (MEMORY: unfiltered data → 0.41).
6. **Row order is not CSV order.** Because filtering uses Python `set` subtraction then `list(...)`
   (`D:278-293`), the returned mutation order is nondeterministic-looking; only **index 0 = WT** is a
   positional guarantee, and ddG is always computed relative to it.
7. **Embedding dim depends on EMB_TYPE** (`D:85-173`): prott5=1024, esmif_enc=512, dual_esmif=1536,
   saprot(_pm)=1280, dual_saprot_pm=2304. The final graph feature width is `68 + E`, and the model's
   `llm_index`/`emb_input_dim` must match (the split-CFG bug MEMORY records). For esmif_enc/dual_esmif
   the ESM-IF part is WT-broadcast (same for all mutations), so it carries NO per-mutation signal.
8. **coords are nanometers → Angstrom.** The trainer multiplies `batch['coords'] *= NANO_TO_ANGSTROM`
   (=0.1) at `T:325`; note the dataset stores nm and downstream distance kernels/`add_cb` coefficients
   assume the Angstrom scale. Docs should note the unit conversion so `gaussian_coef=-0.08` is
   interpreted in the correct units.

<!-- AGENT 3 DONE -->


---

## Agent 4 — Research intent (RESEARCH_PROPOSAL.md, V5_PLAN.md, V5_README.md) vs what's built

> Auditor: Agent 4. Scope = the research/intent docs (RESEARCH_PROPOSAL.md, V5_PLAN.md,
> V5_README.md, TEACHING_PRESENTATION.md, PRESENTATION_AUDIT.md) vs the 3 teaching docs
> (TEACHING_LEVERS_AF.md, TEACHING_VISUAL_REPRESENTATIONS.md, TEACHING_FLOW_AND_COMPATIBILITY.md).
> Citations are `file:line` or `## section heading`. Research only; nothing was modified except
> this section.

### 1. The scientific goal / "why" of the whole project (thesis-level motivation)

The unifying thesis is that DeepEF is a **physics-grounded energy-based model (EBM)**, not a generic
regressor: the network learns a scalar *energy* whose difference between two states of the SAME
sequence is a folding free energy, and every v5 change is justified by **protein thermodynamics /
polymer physics** rather than "generic ML tricks."

Key passages (RESEARCH_PROPOSAL.md):
- Framing as levers, not tricks — `RESEARCH_PROPOSAL.md:22-23`:
  > "The six levers each target a specific weakness in this pipeline, motivated by protein
  > thermodynamics and polymer physics rather than generic ML tricks."
- The model-in-one-paragraph (energy = contrast of folded vs unfolded graph) — `RESEARCH_PROPOSAL.md:9-20`:
  > "DeepEF predicts protein folding free energy by contrasting a **folded** graph (real 3D
  > contacts) against an **unfolded** reference graph (chain-local contacts only). … A two-tower GNN
  > (GCN + GATv2) with light attention maps the graph to a per-residue energy; the protein energy is
  > their sum." with `dG = E_unfolded - E_folded`, `ddG = dG_mutant - dG_wildtype`.
- Baseline-preservation as a scientific control — `RESEARCH_PROPOSAL.md:4-7`:
  > "Every lever is **flag-gated** and **ablatable**; the default value of every flag reproduces the
  > proven baseline … PCC ≈ 0.5259 **bit-for-bit**."
- The "dimension contract" as the correctness backbone — `RESEARCH_PROPOSAL.md:25-36`:
  > "every width is driven from ONE value on `model_cfg_v5.CFG` … when a lever's flag is at its
  > default, the produced tensors and the energy math are identical to the historical baseline."

The deeper EBM intent (native-as-energy-minimum, pretraining via contrastive + score-matching) is
best stated in Agent 1's audit (`DEEP_AUDIT.md` §7, `train.py` InfoNCE + FD-DSM) and echoed by
Lever E's denoising rationale (`RESEARCH_PROPOSAL.md:46`): the auxiliary head "teaches the energy
surface to have a minimum at the native structure." **This EBM/native-minimum framing is the
thesis-level "why" and is under-covered by the teaching docs (see §5).**

### 2. Proposal levers A–F → stated scientific hypothesis (the "why it helps")

From the `## The six levers` table (`RESEARCH_PROPOSAL.md:40-47`) and `V5_README.md:21-28`:

- **A. Energy decomposition (`--energy_terms K`)** — Hypothesis: a real folding free energy is a
  *sum of physically distinct contributions* (H-bonds, vdW, electrostatics, solvation, backbone
  strain). Splitting the scalar into K interpretable per-residue terms lets the head specialize each
  channel to a physical component and enables interpretability (`RESEARCH_PROPOSAL.md:42`;
  TEACHING_LEVERS_AF.md Part A(a)). K=1 = today's single scalar.
- **B. RBF distance bank (`--rbf_centers M`)** — Hypothesis: a single Gaussian kernel can only encode
  "close = strong, far = weak" by magnitude; a bank of M RBFs is a *soft histogram of contact
  distances* that distinguishes an H-bond (~3 Å) from a vdW contact (~6 Å) — the SchNet/DimeNet
  trick. Dimension-preserving (summed back to width 16) (`RESEARCH_PROPOSAL.md:43`).
- **C. Burial / solvation (`--use_burial`)** — Hypothesis: buried vs exposed residues contribute
  differently to stability (mutating a buried hydrophobic core residue is often catastrophic, the
  same swap on the surface harmless). Adds a per-residue CB neighbor-density scalar
  (`RESEARCH_PROPOSAL.md:44`).
- **D. Flory unfolded reference (`--flory_unfolded --flory_nu`)** — Hypothesis: the baseline unfolded
  state (tridiagonal / stick-straight chain) is physically too rigid; a real unfolded protein is a
  *random coil* whose separation scales `⟨r_ij⟩ ∝ |i−j|^ν` (Flory polymer theory, ν≈0.5 ideal /
  ~0.59 good solvent). Value-only change (`RESEARCH_PROPOSAL.md:45`).
- **E. Decoys + denoising head (`--denoise_weight w`)** — Hypothesis: an EBM should have its energy
  *minimum at the native structure*; an aux MLP that predicts injected coordinate noise (a
  denoising / score-matching objective) sculpts that minimum. Train-only aux loss
  (`RESEARCH_PROPOSAL.md:46`).
- **F. Edge features + extended connectivity (`--gcn_span S`, `--use_edge_features`)** — Hypothesis:
  (i) secondary structure has longer reach than i,i+1 (an α-helix H-bonds i→i+4), so GCN offsets 1..S
  help; (ii) chemistry is *pairwise* (a Lys→Asp salt bridge depends on BOTH identities AND distance),
  so a 41-dim edge feature `[onehot_src(20)|onehot_dst(20)|dist(1)]` lets GATv2 reason about explicit
  pairwise chemistry (`RESEARCH_PROPOSAL.md:47`).

The 3 teaching docs cover all six lever *hypotheses* well (TEACHING_LEVERS_AF.md Parts A–F each have
an explicit "(a) Physics / biology intuition" subsection that matches the proposal). **This mapping
is the strongest-covered part of the teaching material.**

### 3. What V5_PLAN / V5_README claim is implemented — done vs aspirational

Note a naming split: the proposal/README speak of **six levers A–F** (physics research directions),
while V5_PLAN and the top of V5_README speak of **five "v5 improvements"** (length_norm,
serial_fusion, correlation loss, dual_esmif, seed ensemble). **These are two different, overlapping
lists.** A reader can conflate them. The teaching docs (TEACHING_LEVERS_AF, FLOW, VISUAL) teach the
A–F set; TEACHING_PRESENTATION Slide 10 teaches the *five-improvement* set. Both sets are legitimate,
but no single doc explicitly states "there are TWO lists (5 v5 improvements + 6 A–F levers) and they
partially overlap." This is a mild organizational gap (see §6).

Claims that appear DONE (corroborated by Agent 1/Agent 2 line-level audits above):
- length_norm — implemented as the single get_energy change (`hydro_net_v5.py:631-633`, Agent 2 §2/§11.6). DONE.
- serial_fusion — constructor + forward present (`hydro_net_v5.py:326-331,498-501`, Agent 2 §11.6). DONE (code exists; note default is OFF in argparse, ladder turns it ON at Rung 2).
- correlation loss — `losses_v5.py` pearson_loss/ccc_loss wired at `T:607-615` (Agent 2 §11.7). DONE.
- Levers A–F — all six implemented and defaulted to baseline, verified line-by-line by Agent 2 (§2–§7) and the smoke test. DONE.
- Ensemble — `training/ensemble_v5.py` exists; two real bugs were found and fixed (V5_README.md:97-106). DONE but fragile (must re-supply flags).

Claims that read as ASPIRATIONAL / conditional (not fully turnkey):
- **dual_esmif route.** V5_PLAN.md:37-42 and V5_README.md:13 present it as a tier, but it is gated
  on a per-protein `esmif_enc.pt` that must be generated first: `V5_PLAN.md:40-41` — "needs
  `esmif_enc.pt` per protein (generation script referenced: generate_proteinmpnn_features.py
  exists)." The ladder rung is explicitly conditional: `V5_README.md:74` "(dual_esmif)" and
  TEACHING_FLOW_AND_COMPATIBILITY.md:265 "swap `--emb_type dual_esmif` (only if esmif_enc.pt
  exists)." So dual_esmif is *implemented in code paths* but *not runnable out-of-the-box* without a
  feature-generation step. The teaching docs mention the ESM-IF1 512-dim block but do NOT flag the
  "you must first generate esmif_enc.pt" prerequisite prominently.
- **Smoke test / "verified" claims.** V5_README.md:33-50 and RESEARCH_PROPOSAL.md:58-64 claim a CPU
  smoke test runs baseline + every lever and asserts shapes. Agent 2/Agent 3 should confirm the
  test file actually exercises A–F (I did not re-run it); treat "CI-friendly, exit 0 = all passed"
  as a claim to verify, not audited here.
- **Target PCC ≥ 0.70.** V5_PLAN.md:3 states "Target: ≥0.70"; every doc is careful elsewhere to
  label PCC gains as *hypotheses* (TEACHING_PRESENTATION Slide 10 header "All PCC numbers are
  hypotheses"; V5_README.md:82-84). The 0.70 target is aspirational and clearly labeled as such.

### 4. Stated baseline (PCC 0.5259) and the ablation-ladder methodology

Baseline: **PCC ≈ 0.5259, dG, from scratch (no pretraining), ProtT5** — stated consistently in
`RESEARCH_PROPOSAL.md:7`, `V5_PLAN.md:3`, `V5_README.md:3`, and the exact command in
`TEACHING_FLOW_AND_COMPATIBILITY.md:9-18` and the project MEMORY "Best Known Config."

Ladder methodology — the ladder is stated in three places with the SAME structure but the two
different lever-sets:
- V5_PLAN.md:48-58 and V5_README.md:74 — the **5-improvement** ladder:
  Rung 0 baseline → 1 +length_norm → 2 +serial_fusion → 3 +corr_loss → 4 dual_esmif → 5 5-seed ensemble.
- TEACHING_FLOW_AND_COMPATIBILITY.md:247-278 — same 5-improvement ladder with the fixed COMMON flag
  string (`## 4. Ablation Methodology`).
- TEACHING_LEVERS_AF.md Part 8 / "Golden rules" (`:518-521`) — states the *discipline* (one lever
  per rung, inherit the winner, drop regressions) but for the A–F set.

The methodology rationale ("change ONE thing at a time; don't stack losers; a regressing rung is
dropped") is well covered: `V5_PLAN.md:57-58`, `V5_README.md:77-79`,
`TEACHING_FLOW_AND_COMPATIBILITY.md:280-301` (with the concrete GNN-SM 0.42 and full_megascale 0.4098
failure examples). This is a STRONG point of the teaching docs.

### 5. Topics the research docs EMPHASIZE that the 3 teaching docs UNDER-cover

These are the actionable gaps for teaching completeness. Ordered by severity.

5.1 **EBM pretraining theory (InfoNCE + Denoising Score Matching) — MISSING from all 3 teaching
   docs.** The original design intent (Agent 1 §7; `train.py` `lossd_fucntion` InfoNCE at
   `train.py:690-715`, FD-DSM `denoising_score_matching` at `train.py:602-662`, τ=1.0, σ=0.5) is the
   thesis-level "why this is an energy-based model." The 3 teaching docs describe ONLY the
   fine-tuning (dG regression) pipeline. Lever E's denoising head is taught as a stability trick
   (TEACHING_LEVERS_AF Part E) but is NOT connected to the score-matching / native-minimum EBM theory
   or to the `train.py` pretraining regime. TEACHING_PRESENTATION mentions "from scratch, no
   pretraining" (Slide 9) and the priority hierarchy "Inverse-folding prior >> data >> architecture"
   (Slide 12) but never explains what pretraining *is*. **Recommend: a short section on the EBM /
   contrastive (InfoNCE) + DSM pretraining objective and why native = energy minimum.**

5.2 **Evaluation-metric definitions (what PCC / Spearman / RMSE actually MEAN) — under-covered.**
   All docs report "PCC 0.5259" and mention Spearman/RMSE, but none of the 3 teaching docs defines
   Pearson vs Spearman (linear trend vs rank), why correlation (not RMSE) is the grading metric, or
   why that motivates the correlation-loss lever. TEACHING_VISUAL 7d lists `pc_corr (PCC)` as "the
   grading metric" but with no definition. **Recommend: a 3-line metrics primer** (Pearson = linear
   correlation of predicted vs true dG; Spearman = rank correlation; RMSE = magnitude error; the
   correlation loss aligns training with the PCC metric — this last point IS made in
   TEACHING_PRESENTATION Slide 10(3) but not in the LEVERS/FLOW/VISUAL trio).

5.3 **Loss definitions — partially covered, one gap.** Huber and pairwise ranking are defined
   (TEACHING_PRESENTATION Slide 9; TEACHING_FLOW §7-loss block `:97-107`). But the *correlation
   loss* math (`pearson_loss = 1 − r`, `ccc_loss = 1 − CCC`, Agent 2 §11.7 / `losses_v5.py:12-44`)
   and the *energy_reg* term (E_REG_LAMBDA=0.001, always on) are only lightly touched in the A–F
   trio. TEACHING_FLOW lists them in the loss box but does not give the formulas. **Recommend: give
   the pearson/ccc formula and note energy_reg is always-on (not a v5 change).**

5.4 **Dataset biology / filtering rationale — thin in the A–F teaching docs.** The proposal and
   Agent 1 §6 give rich biological justification for PNAS / one_mut / dg_ml (K50 assay saturates
   outside [-1,5]; indels break the fixed-length backbone; quality>quantity, full_megascale HURT to
   0.41). TEACHING_FLOW_AND_COMPATIBILITY.md:43-54 lists the filters but with terse one-liners and no
   biology; TEACHING_LEVERS_AF and TEACHING_VISUAL barely mention dataset curation.
   TEACHING_PRESENTATION Slide 9 mentions the clamp/one_mut but not *why the assay saturates*.
   **Recommend: a short "dataset biology" note (why the clamp, why single mutations, why curation
   beats volume).**

5.5 **Denoising / score-matching theory depth — shallow.** Lever E is taught operationally (inject
   noise, predict noise) in TEACHING_LEVERS_AF Part E and TEACHING_PRESENTATION, but the *why* (score
   matching estimates ∇log p, the noise-prediction target equals `-noise/σ²`, native = low-gradient
   basin — Agent 1 §3B / `train.py:602-662`) is not explained. Given this is central to the "physics
   energy function" thesis, the operational-only treatment under-serves the motivation.

5.6 **Limitations / future work — scattered, not consolidated.** The proposal implies the four
   standing bottlenecks (no inverse-folding pretraining ≈ 0.20 PCC gap; fixed WT structure for all
   mutants; extensive-sum energy; no explicit mutation-delta feature — see project MEMORY
   "Bottlenecks"). TEACHING_PRESENTATION Slide 8 covers the extensive-sum limitation well and Slide
   2/6 note the fixed-WT-structure assumption, but the "no true inverse-folding pretraining" and "no
   emb_mut−emb_wt delta feature" limitations are not called out as limitations in the A–F trio.
   **Recommend: a consolidated Limitations/Future-Work slide.**

5.7 **The "why physics" meta-motivation is implicit, not stated, in the A–F trio.** The proposal's
   thesis sentence (`RESEARCH_PROPOSAL.md:22-23`, "motivated by protein thermodynamics and polymer
   physics rather than generic ML tricks") never appears verbatim in the teaching docs. Each lever's
   physics is explained locally, but the overarching "this is a physics-grounded energy function, and
   THAT is the contribution" framing is absent from LEVERS/FLOW/VISUAL. TEACHING_PRESENTATION comes
   closest (Slide 8 "the scientific heart") but frames it around length_norm specifically.

### 6. Contradictions between the research docs and the teaching docs

6.1 **Node-vector width arithmetic: 1092 vs 1096 vs 1112 — potential confusion, resolvable but not
   reconciled in one place.** The teaching docs use THREE different totals for the same pipeline:
   - TEACHING_PRESENTATION Slide 4 / Slide 12 and TEACHING_LEVERS_AF Part 0 and TEACHING_VISUAL §2a
     say the **node feature vector** = 16+32+1024+20 = **1092** (1604 with ESM-IF1). CORRECT for the
     builder output.
   - TEACHING_PRESENTATION Slide 7 says the head is "FC 1112→128→1" and TEACHING_VISUAL §5 shows
     `fc_in_dim = 88 + 1024 = 1112`. This is the **model's internal head input** (GCN 52 + GAT 36 =
     88, +1024 emb), a DIFFERENT quantity from the 1092 node vector.
   - Agent 1 §1 computes `fc_in_dim = 36 + 36 + 1024 = 1096` (using GCN_out=36, GAT_out=36), while
     TEACHING_VISUAL §5 uses 52 + 36 = 88 → 1112 (GCN_out=52). **This is a genuine numeric
     discrepancy between Agent 1's audit (1096) and TEACHING_VISUAL (1112)** driven by whether the
     GCN tower OUTPUT width is 36 or 52. Agent 2 §10 (`M:491`) states `fc_in_dim = 36+36+1024 =
     1096`. So **TEACHING_VISUAL's 1112 and its `x_gcn` output "[B*N, 52]" (VISUAL §7c) appear to be
     WRONG** — the GCN tower's `fc2_gcn` maps 64→36 (Agent 1 §1, `hydro_net.py:357`; Agent 2 §10,
     `M:375`), so the concatenation is 36+36=72 not 52+36=88, and fc_in_dim = 72+1024 = 1096, not
     1112. TEACHING_PRESENTATION Slide 6 correctly says "concat → 72-dim" (Agent 1 §1) while
     TEACHING_VISUAL says 88. **CONTRADICTION between the two teaching docs and against the
     line-level model audit; VISUAL should be corrected to 36/72/1096.** (This mirrors the
     PRESENTATION_AUDIT Slide 7 note that the 52-dim grouping "could confuse.")

6.2 **get_deltaG arity: teaching docs show a 3-tuple; v5 code returns a 4-tuple.** TEACHING_VISUAL §3c
   (`:311-313`) and §6 (`:493`) show `return unfolded_energy - folded_energy, unfolded_energy,
   folded_energy` (3 values). Agent 2 §6/§11.1 proves v5 `get_deltaG` returns **FOUR** values
   (adds `denoise_loss`, `pnas_train_v5.py:848`). **CONTRADICTION: the teaching doc reflects the
   baseline 3-tuple, not the v5 4-tuple.** Minor (pedagogically the 4th is the denoise aux), but
   technically wrong for v5.

6.3 **mini_batch_size default framing is consistent (good).** All docs correctly warn 64→OOM, use 16.
   TEACHING_VISUAL §Notation says "default 16" (`:17`) which is slightly loose (the *argparse* default
   is 64; 16 is the *proven-run* value, per Agent 2 §11.2 / MEMORY). Not a contradiction with the
   research docs, but "default 16" is imprecise vs "argparse default 64, always pass 16."

6.4 **serial_fusion "default on" wording.** V5_PLAN.md:26 ("Serial fusion ON by default") and
   V5_README.md:11 ("default on in ladder") vs argparse `--serial_fusion` default **False**
   (Agent 2 §1, `T:83`). The reconciliation is that it is on *in the ladder script at Rung 2*, not in
   the argparse default. RESEARCH_PROPOSAL.md:40 correctly lists serial-fusion-adjacent behavior as
   baseline-off. The teaching docs (TEACHING_FLOW ladder Rung 2) handle this correctly, but the
   V5_PLAN/V5_README "on by default" phrasing is a mild internal inconsistency with the code default.

6.5 **"predicts ddG" vs "predicts dG" — RESOLVED, no contradiction.** Both PRESENTATION_AUDIT
   (Slide 1-2) and TEACHING_PRESENTATION (Slide 1 "KEY FRAMING") correctly state the model predicts
   dG and derives ddG at validation. All teaching docs are consistent on this. (Noted here to confirm
   the previously-flagged pptx issue is fixed in the teaching materials.)

6.6 **Unfolded reference "i±1 vs i±2" — RESOLVED.** PRESENTATION_AUDIT Slide 5 flags the old pptx
   said i±2; TEACHING_PRESENTATION Slide 5 ("Correction #4") and TEACHING_VISUAL §3b correctly show
   i±1 (tridiagonal). Consistent. No contradiction remaining.

6.7 **one_hot 20 vs 21 — RESOLVED.** All teaching docs correctly state 20 (TEACHING_PRESENTATION
   Slide 3 "Correction #5"; TEACHING_VISUAL uses 20 throughout). Consistent with code (Agent 1 §4).

### 7. Summary for the orchestrator (Agent 4 verdict)

- **Lever A–F hypotheses:** fully and accurately taught (TEACHING_LEVERS_AF). Best-covered area.
- **Baseline + ladder methodology:** fully and accurately taught (TEACHING_FLOW §4). Strong.
- **Biggest teaching GAPS (add these):** (1) EBM pretraining theory — InfoNCE + DSM / native-minimum
  (§5.1, §5.5); (2) evaluation-metric definitions PCC/Spearman/RMSE (§5.2); (3) dataset biology
  rationale (§5.4); (4) consolidated limitations/future-work (§5.6); (5) the explicit "physics-
  grounded energy function, not ML tricks" thesis framing (§5.7); (6) correlation/energy_reg loss
  formulas (§5.3).
- **Concrete ERRORS to fix in teaching docs:** (a) TEACHING_VISUAL_REPRESENTATIONS.md GCN-output/
  fc_in_dim arithmetic (52/88/1112 → should be 36/72/1096, §6.1); (b) TEACHING_VISUAL get_deltaG
  3-tuple → 4-tuple for v5 (§6.2); (c) "default 16" imprecision (§6.3).
- **Naming clarity:** state explicitly that there are TWO overlapping lists — the 5 "v5 improvements"
  (length_norm, serial_fusion, corr loss, dual_esmif, ensemble) and the 6 physics levers A–F — and
  which doc teaches which (§3).
- **Aspirational-not-turnkey:** dual_esmif needs `esmif_enc.pt` generated first; PCC ≥ 0.70 is a
  labeled target; smoke-test "all pass" is an unverified claim here (§3).

<!-- AGENT 4 DONE -->


---

## Agent 5 — The 3 teaching docs: coverage/accuracy vs code, gaps, errors

> Auditor: Agent 5. Scope = COVERAGE + ERROR audit of the 3 teaching docs against the code as
> documented by Agents 1–3. Docs abbreviated: **LAF** = TEACHING_LEVERS_AF.md, **VIS** =
> TEACHING_VISUAL_REPRESENTATIONS.md, **FLOW** = TEACHING_FLOW_AND_COMPATIBILITY.md. Citations are
> `DOC:line` for the teaching docs and `Agent N §X` for prior findings. Every ERROR carries exact
> line numbers.

### Coverage + Error Table (15 topics)

| # | Topic | Status | Where covered (doc:section/line) | Error? |
|---|-------|--------|----------------------------------|--------|
| 1 | Biology of proteins / folding / mutations | **COVERED** | LAF Part 0 (`:18-56`); VIS §1a (`:34-52`), §6 (`:467-480`); FLOW mental model (`:406-417`) | none |
| 2 | dG / ddG thermodynamic logic | **COVERED** | LAF Part 0 (`:27-44`); VIS §3c (`:306-327`), §6 (`:511-519`); FLOW (`:28-30, 91-95`) | **ERROR (E2)** — VIS `:312` shows dG readout as 3-tuple |
| 3 | Node feature layout + slicing (arithmetic) | **PARTIAL / ERROR** | LAF Part 0 (`:46-56`); VIS §2 (`:164-251`), §5 (`:383-435`), §7c (`:564-588`); FLOW §2 (`:142-154`) | **ERROR (E1)** — VIS 52/88/1112 vs correct 36/72/1096 |
| 4 | GCN vs GAT towers run in parallel | **COVERED** | LAF Part 0 (`:37-39`); VIS §5 (`:375-450`); FLOW (`:84-85`) | none — no doc says "OR" (see verification below) |
| 5 | get_deltaG arity | **PARTIAL / ERROR** | VIS §3c (`:309-314`), §6 (`:486-493`) | **ERROR (E2)** — VIS shows 3-tuple; v5 returns 4-tuple |
| 6 | Each lever A–F (physics / shape / flag / default=baseline) | **COVERED** | LAF Parts A–F (`:66-369`), cheat sheet (`:504-521`); FLOW §3 (`:193-243`), decision tree (`:354-402`) | none |
| 7 | Loss functions (Huber, ranking, energy_reg, corr/pearson/ccc, denoise) — FORMULAS | **PARTIAL** | LAF Part E worked example (`:285-296`); FLOW loss box (`:97-107`); VIS (`:528-530`) | none factually wrong, but **formulas MISSING** (see gaps) |
| 8 | Evaluation metrics (PCC / Spearman / RMSE definitions) | **MISSING** | VIS §7d names `pc_corr (PCC)` "grading metric" (`:599`); FLOW (`:112`) | none (absence, not error) |
| 9 | Dataset biology (K50 assay, Tsuboyama, clamp [-1,5], filters) | **PARTIAL** | FLOW §1 filter list (`:43-54`); LAF/VIS barely mention | none wrong; biology thin (K50/Tsuboyama name MISSING) |
| 10 | EBM pretraining (InfoNCE + DSM) | **MISSING** | (not in any of the 3 docs) | none (absence) — see gaps |
| 11 | mini_batch_size OOM / seed / reproducibility | **COVERED** | FLOW §5 (`:305-322`) | **minor imprecision (E3)** — VIS `:17` "default 16" |
| 12 | Compatibility / composability of levers | **COVERED** | LAF Part 7 (`:373-423`); FLOW §2–3 (`:130-243`), §5 reload rule (`:324-350`) | none |
| 13 | Ablation ladder methodology | **COVERED** | FLOW §4 (`:247-301`); LAF golden rules (`:518-521`) | none |
| 14 | add_cb / glycine handling | **PARTIAL** | VIS §1a (`:36-38`, only "synthesized by add_cb()") | none wrong; glycine-has-no-real-CB caveat MISSING (Agent 3 §3) |
| 15 | Limitations / future work | **MISSING** | (not in any of the 3 docs) | none (absence) |

### ERRORS — pinpointed to exact lines

**E1 — VIS node/tower arithmetic is WRONG: 52 / 88 / 1112 should be 36 / 72 / 1096.**
This is the error the task flagged, and it is real. The root cause: VIS conflates the GCN tower's
*input* width (52) with its *output* width (which is 36). `fc2_gcn` maps `64 → 36` (Agent 1 §1
`hydro_net.py:357`; Agent 2 §10 `M:375`), so the GCN tower OUTPUT is 36, not 52. Concatenation is
therefore `36 + 36 = 72` (Agent 1 §1 `hydro_net.py:449-452`; Agent 2 §10 `M:384,511-514`), and
`fc_in_dim = 72 + 1024 = 1096` (Agent 1 §1 `hydro_net.py:335,368` "= 36 + 36 + 1024 = 1096";
Agent 2 §10 `M:491` "= 36+36+1024 = **1096**"). VIS instead uses 52 as the GCN *output*, giving
88 and 1112. Exact offending lines in **TEACHING_VISUAL_REPRESENTATIONS.md**:

- `:410` — `│  x1: [B*N, 52]` — labels the GCN tower **output** as 52 (should be 36). *This is the
  seed of the whole error* — 52 is the GCN input (`:245`), not its output.
- `:413` — `concat  x = [x1 | x2]   (dim = 52+36 = 88)` — should be `36+36 = 72`.
- `:415,417` — `reshape [B, N, 88]` / `reshape [B*N, 88]` — should be 72.
- `:420` — `x = [x | x_emb_features]   (dim = 88 + 1024 = 1112)` — should be `72 + 1024 = 1096`.
- `:424` — `fc_in_dim = 88 + 1024 = 1112` — should be `72 + 1024 = 1096`.
- `:578` — shapes cheat-sheet `| \`x1\` (GCN out) | \`[B*N, 52]\` |` — should be `[B*N, 36]`.
- `:580` — `| \`concat(x1,x2)\` | \`[B*N, 88]\` |` — should be `[B*N, 72]`.
- `:581` — `after \`inst_norm2\` | \`[B, N, 88]\` → \`[B*N, 88]\`` — should be 72.
- `:582` — `| + raw emb | \`[B*N, 1112]\` | \`88 + 1024\` (= \`fc_in_dim\`) |` — should be
  `[B*N, 1096]`, `72 + 1024`.
- `:583` — `after LightAttention | \`[B*N, 1112]\`` — should be 1096.
- `:584` — `fc1 out` row precedes; the LightAttention `embeddings_dim=fc_in_dim` is 1096, not 1112
  (Agent 2 §10 `M:493`).
- **NOTE (not an error):** VIS `:245` `x_gcn = ... = 52` and `:396` `fc1_gcn: 52→64` and `:574`
  `x_gcn | [B*N, 52]` are all CORRECT — 52 is the legitimate GCN *input* width
  `[dist16|bonded16|aa20]`. The mistake is only where 52 is carried through as the GCN *output*.
- Cross-doc consistency: VIS is the *only* doc with this error. LAF never states 88/1112. Per Agent 4
  §6.1, TEACHING_PRESENTATION Slide 6 correctly says "concat → 72-dim." So the fix is isolated to VIS.

**E2 — VIS get_deltaG shown as a 3-tuple; v5 returns a 4-tuple.**
Agent 2 §6/§11.1 proves v5 `get_deltaG` returns FOUR values —
`return unfolded_energy - folded_energy, unfolded_energy, folded_energy, denoise_loss`
(`pnas_train_v5.py:848`) — and both train (`T:588`) and validate (`T:708`) unpack 4. The teaching
doc reflects the *baseline* `pnas_train.py` 3-tuple. Exact offending lines in
**TEACHING_VISUAL_REPRESENTATIONS.md**:

- `:312` — `return unfolded_energy - folded_energy, unfolded_energy, folded_energy` — only 3 values;
  v5 appends `, denoise_loss` (the 4th). The comment on `:9-12` explicitly says these diagrams are
  "traced directly from … `pnas_train_v5.py` — `Trainer.get_deltaG`," so the 3-tuple is a factual
  mismatch with the v5 code it claims to trace.
- `:486-493` — the §6 mini-batch code block ends at `dG = unfolded_energy - folded_energy` (`:493`)
  and never shows the 4th (`denoise_loss`) return element. Pedagogically fine (the 4th is the Lever E
  aux term and is train-only), but technically incomplete for v5.
- Sign convention `dG = unfolded − folded` on `:312-313` and `:493` is CORRECT (matches Agent 2 §6
  `T:848`); only the arity is wrong.

**E3 — VIS "mini-batch size … (default 16)" is imprecise (minor).**
- `:17` — `\`M\` = mini-batch size = number of mutations processed together (default 16)`. The
  *argparse* default is **64** (Agent 2 §1 `T:79`, §11.2); 16 is the proven-run value that must be
  passed to avoid OOM. FLOW §5 (`:312`) states this correctly ("Default is 64 … pass 16"). Not a
  contradiction of the physics, but "default 16" understates the OOM footgun. Recommend "argparse
  default 64; always pass 16."

### Topic-4 verification (GCN vs GAT — does any doc say "OR"?)

Checked all three docs for any "GCN OR GAT" framing (the caveat Agent 1 §7 warns against). **None do.**
All three correctly present the towers as running in PARALLEL and concatenating:
- LAF `:37-39` "Two graph neural networks run in parallel — a **GCN** … and a **GATv2** … Their
  outputs feed a small head."
- VIS `:375-377` "The model runs **two independent towers in parallel**, concatenates their outputs";
  `:440-442` `x1 = forward_gcn(...)`, `x2 = forward_gat(...)`, `x = cat((x1,x2))`.
- FLOW `:84` "split → GCN branch … + GAT branch … → concat."
- One benign "OR" appears at VIS `:455-459` describing the *GAT edge topology* ("fully connected OR
  CA-distance cutoff") — this is the correct fully-connected-vs-kNN choice, NOT a GCN-vs-GAT "or." No
  error.

### GAPS (MISSING / thin) — for teaching completeness (corroborates Agent 4 §5)

- **G1 — Loss FORMULAS missing (Topic 7).** No teaching doc gives `pearson_loss = 1 − r`,
  `ccc_loss = 1 − CCC`, the Huber piecewise definition, the margin-ranking formula, or notes
  `energy_reg` (E_REG_LAMBDA=0.001) is ALWAYS on. FLOW `:97-107` lists them as a loss stack but
  without math. Formulas exist in Agent 3 §4 (`losses_v5.py:12-44`) — should be lifted in.
- **G2 — Evaluation metrics undefined (Topic 8).** PCC/Spearman/RMSE are named but never defined
  (VIS `:599`, FLOW `:112`). No doc explains Pearson (linear trend) vs Spearman (rank) vs RMSE
  (magnitude), nor why correlation is the grading metric (which is what motivates the corr-loss lever).
- **G3 — EBM pretraining absent (Topic 10).** InfoNCE contrastive + FD-DSM native-vs-decoy pretraining
  (Agent 1 §3B, §7; `train.py:602-662, 690-715`, τ=1.0, σ=0.5) — the thesis-level "why this is an
  energy-based model" — is in NONE of the 3 docs. Lever E's denoising head (LAF Part E) is taught as a
  stability trick but never linked to score-matching / native-minimum theory.
- **G4 — Dataset biology thin (Topic 9).** The K50 cDNA-display **proteolysis** assay and the
  **Tsuboyama 2023** mega-scale dataset (Agent 3 §1) are NOT named in any teaching doc; the `[-1,5]`
  clamp is listed (FLOW `:47`) but the *assay-saturation* reason is not given there.
- **G5 — Glycine CB caveat missing (Topic 14).** VIS `:36-38` says CB is "synthesized geometrically by
  add_cb()" but does not state that glycine has **no real CB** and gets a fabricated virtual one used
  by burial + distances (Agent 3 §3, `U:441-466`).
- **G6 — Limitations/Future-work not consolidated (Topic 15).** No inverse-folding-pretraining gap,
  fixed-WT-structure assumption, extensive-sum energy, or missing emb_mut−emb_wt delta feature are
  gathered as limitations in the A–F trio (Agent 4 §5.6).

### Agent 5 verdict

- **Best covered:** Topics 1, 4, 6, 12, 13 (biology intuition, parallel towers, lever A–F
  physics/shape/flag/default, composability, ablation ladder) — accurate and complete.
- **Two hard ERRORS to fix, both isolated to TEACHING_VISUAL_REPRESENTATIONS.md:**
  (E1) the 52/88/1112 arithmetic → correct to **36/72/1096** at lines
  `410, 413, 415, 417, 420, 424, 578, 580, 581, 582, 583`; and
  (E2) the get_deltaG 3-tuple at line `312` (and the incomplete §6 block ending `:493`) → v5 returns a
  **4-tuple** with `denoise_loss`.
- **One minor imprecision:** VIS `:17` "default 16" → argparse default is 64.
- **Six content GAPS** (G1–G6): loss formulas, metric definitions, EBM pretraining theory, dataset
  biology (K50/Tsuboyama), glycine-CB caveat, and a consolidated limitations section.

<!-- AGENT 5 DONE -->


---

## CONSOLIDATED GAP LIST (orchestrator — synthesized from Agents 1–6)

### PART 1 — HARD ERRORS (factually wrong vs code; MUST fix)

| ID | Where | Error | Correct value | Source |
|----|-------|-------|---------------|--------|
| **E1** | `TEACHING_VISUAL_REPRESENTATIONS.md` lines 410, 413, 415, 417, 420, 424, 578, 580, 581, 582, 583 | GCN output = 52, concat = 88, `fc_in_dim` = 1112 | GCN out = **36**, GAT out = 36, concat = **72**, `fc_in_dim` = **1096**. (52 is the GCN *input* only; `fc2_gcn` maps 64→36) | Agent 1 §1, Agent 2 §10, Agent 5 E1 |
| **E2** | `TEACHING_VISUAL_REPRESENTATIONS.md` line 312 (and §6 block ~493) | `get_deltaG` shown as 3-tuple `(dG, u, f)` | v5 returns **4-tuple** `(dG, u_energy, f_energy, denoise_loss)` (`pnas_train_v5.py:848`) | Agent 2 §6/§11.1, Agent 5 E2 |
| **E3** | `TEACHING_VISUAL_REPRESENTATIONS.md` line 17 | "mini-batch size = ... default 16" | argparse **default is 64**; **16** is the mandatory proven-run value (OOM lesson). Flow doc line 312 states it correctly. | Agent 2 §11.2, Agent 5 E3 |

Presentation artifacts (figures/PDF/HTML) are CLEAN of E1/E2/E3 — the errors are isolated to the ONE markdown doc (Agent 6).

### PART 2 — CONTENT GAPS (missing topics you explicitly asked about)

| ID | Topic asked | Status in 3 teaching docs | Fix source |
|----|-------------|---------------------------|-----------|
| **G1** | Loss definitions (Huber, ranking, energy-reg, Pearson, CCC, denoise MSE) — FORMULAS | MISSING formulas (levers named, not defined) | `GAP_FILL_CONTENT.md §2` |
| **G2** | Evaluation metrics (PCC / Spearman / RMSE) definitions + why PCC is the score | MISSING | `GAP_FILL_CONTENT.md §1` |
| **G3** | Dataset biology (Tsuboyama 2023, K50 proteolysis assay, why clamp [-1,5], why 1-mut + PNAS curation, ThermoMPNN) | THIN / names absent | `GAP_FILL_CONTENT.md §3` + Agent 3 §1 |
| **G4** | EBM pretraining theory (InfoNCE contrastive + Denoising Score Matching, native = energy minimum) | MISSING entirely (present in root `train.py`, links to Lever E) | `GAP_FILL_CONTENT.md §4` + Agent 1 §3B |
| **G5** | "Why physics-grounded, not ML tricks" thesis framing | MISSING as a motivation section | `GAP_FILL_CONTENT.md §5` + Agent 4 §1 |
| **G6** | Limitations / future work | MISSING (no consolidated section) | Agent 4 §5 |
| **G7** | `add_cb` / glycine virtual-CB caveat | MISSING (Gly gets a fabricated CB) | Agent 3 §3 |
| **G8** | ESM-IF / dual_esmif broadcast WT embedding (no per-mut signal from ESM-IF) | MISSING nuance | Agent 3 §1 |

### PART 3 — CORRECT & COMPLETE (verified accurate, no action)

Biology of folding/mutations; dG/ddG thermodynamics; folded-vs-unfolded graph; node layout `[D16|Fb32|(burial)|emb|onehot20]` + slicing; **both GCN+GAT towers run in parallel** (no doc says "OR"); all six levers A–F physics + shape + flag + default-reproduces-baseline; composability/dimension-contract; ablation-ladder methodology; mini_batch OOM lesson (Flow doc); single-CFG fix. Verified in Agents 1, 2, 3, 5.

### RECOMMENDED FIX PLAN
1. Patch E1/E2/E3 in `TEACHING_VISUAL_REPRESENTATIONS.md` (exact lines above).
2. Insert `GAP_FILL_CONTENT.md` sections into the docs: G2+G1 → new "Metrics & Losses" section; G3 → expand dataset section; G4 → new "Pretraining (EBM)" section; G5 → intro/motivation; G6 → closing "Limitations"; G7/G8 → footnotes.
3. Regenerate `DeepPEF_v5_Teaching.html` (`build_html.py`) and `DeepPEF_v5_Slides.pdf` so both formats carry the corrections.
4. (Optional cosmetic) bump fig4 DPI / font for legibility (Agent 6 §5).

---

## Agent 6 — Presentation artifacts (figures/PDF/HTML) audit

Scope: `make_figures.py` (fig1–fig5 generator), `DeepPEF_v5_Slides.tex` (Beamer deck),
`build_html.py` (portable HTML wrapper), and the 5 rendered PNGs in `figures/`. Verified against
`model/hydro_net_v5.py` (M), `model/model_cfg_v5.py` (C), `training/pnas_train_v5.py` (T).

### 1. E1/E2/E3 error scan of the presentation scripts — RESULT: CLEAN

- **E1 (wrong dims 52 / 88 / 1112).** NOT present in `make_figures.py` nor the `.tex`. Neither the
  fig generator nor the deck ever prints raw GCN/concat/fc widths. The only numeric widths shown are
  correct or intentionally symbolic:
  - fig2 node layout uses `D 16`, `Fb 32`, `emb E=1024`, `one_hot 20` (`make_figures.py:97-100,110-111`)
    — matches CFG `dist_dim=16`, bonded `2*dist_dim=32`, `emb_input_dim=1024`, one-hot 20 (`C:38,76,78`).
  - fig3 shows `fc1(128)` (`make_figures.py:145`) which matches `self.fc1 = nn.Linear(fc_in_dim,128)`
    (`M:387`); it does NOT quote `fc_in_dim`, so the wrong 1112 never appears.
  - For the record the CORRECT values (per code) are: `gnn_internal_dim = 16 + aa_dim(20) = 36`
    (`M:353-354`), `gcn_dim_out = gat_dim_out = 36` (`M:357,361`), concat `36+36 = 72`,
    `fc_in_dim = 72 + post_gnn_emb(1024) = 1096` (`M:344,386`). 52 is only the GCN INPUT contract in
    the teaching-doc arithmetic; the figures avoid the whole trap by not stating it. **No fix needed.**
- **E2 (get_deltaG 3-tuple).** NOT present. Neither script mentions `get_deltaG` or its return arity.
  The deck's Verification slide (`.tex:69-79`) speaks only of smoke-test/regression, no tuple. Code
  truth: `get_deltaG` returns a **4-tuple** `(output, u_energy, f_energy, denoise_loss)` (`T:777` def;
  `T:588` train unpacks 4; `T:708` validate unpacks 4; `smoke_test.py:205` unpacks 4). **No fix needed.**
- **E3 (mini_batch_size default 64 vs proven 16).** NOT present as a wrong number in the artifacts —
  but it is also NOT mentioned at all. The `.tex` gives no run command with `--mini_batch_size`; the
  only command shown is `python DeepPEF_v5/tests/smoke_test.py` (`.tex:78`). So the deck neither
  states the wrong default nor warns about the 16-vs-64 OOM footgun. Code truth: argparse
  `default=64` (`T:79`); proven run uses `16` (ensemble injects `'16'` at `ensemble_v5.py:68`).
  **Not an error in the slides; it is a GAP (see §6, G-mb).**

### 2. Lever A–F labels vs code — ACCURATE

Checked every lever caption in fig4 (`make_figures.py:178-257`) and the `.tex` lever table
(`.tex:53-59`) against CFG + model:

- **A energy decomposition** — fig4 "fc2: 128 -> K per-residue terms", "K=1 == today's single scalar"
  (`:185,192`); `.tex` "fc2 emits K per-residue energy terms; K=1 = scalar" (`:54`). Matches
  `self.fc2 = nn.Linear(128, self.energy_terms)`, `energy_terms=max(...,1)` (`M:391-392`), CFG
  `energy_terms=1` (`C:89`). CORRECT.
- **B RBF bank** — fig4 "M RBFs summed -> width 16 (dim-preserving)" (`:202`); `.tex` "M RBFs summed
  back to width 16" (`:55`). Matches CFG `rbf_centers=0` single Gaussian, sum back to `dist_dim=16`
  (`C:79-84`). CORRECT.
- **C burial** — fig4 "buried vs surface" (`:214`) and fig2 inserts `burial b=1` BEFORE emb (`:111`);
  `.tex` "CB neighbor-density scalar inserted before emb" (`:56`). Matches CFG `use_burial=False`,
  `burial_dim=0` when off / width 1 when on, inserted before emb (`C:90-95`; layout comment `M:346-350`).
  CORRECT.
- **D Flory** — fig4 "coil |i-j|^0.5", "random-coil distances (value-only)" (`:220,224`); `.tex`
  "random-coil unfolded reference d~|i-j|^nu (value-only)" (`:57`). Matches CFG `flory_unfolded=False`,
  `flory_nu=0.5` (`C:96-100`). CORRECT.
- **E denoise** — fig4 "head predicts injected noise (train only)" (`:238`) and fig3 "denoise head
  (Lever E, aux)" (`:156`); `.tex` "aux head predicts injected coord noise (train only)" (`:58`).
  Matches denoise head `nn.Linear(128,128)->ReLU->Linear(128,4*3)`, active only when
  `denoise_weight>0` (`M:401-406`), train-only loss (`T:588` vs `T:708`). CORRECT.
- **F edges** — fig4 "span S both dirs; edge_attr=[oh|oh|dist]=41" (`:252`) and fig3 "F: span S" /
  "F: edge_attr 41-d" (`:160-161`); `.tex` "span-S GCN edges + 41-d GAT edge features" (`:59`).
  Matches CFG `gcn_span=1`, `use_edge_features` (`C:107-111`), `edge_dim = 41` (`M:364`), build order
  `[onehot_src(20)|onehot_dst(20)|dist(1)]` (`M:585`, comment `C:109`). CORRECT.

fig5 composability kinds (`make_figures.py:279-280`): A=head, B=dim-preserve, C=node width,
D=value-only, E=aux head, F=edges — all consistent with the above. CORRECT.

### 3. fig2 node-layout widths + ordering — CORRECT

Baseline row `[D 16 | Fb 32 | emb E=1024 | one_hot 20]` (`make_figures.py:97-100`) and Lever-C row
insert `burial b=1` between `Fb 32` and `emb E` (`:110-111`) exactly match the code layout
`[dist:16 | bonded:32 | (burial:b) | emb:E | one_hot:20]` (M layout comment §8 of Agent-4 notes; CFG
`C:75-78`). The caption "one_hot always stays LAST (model slices it from the right)" (`:120`) matches
`x_onehot = x[:, -20:]` slicing; "Burial is inserted BEFORE emb" (`:121`) matches the width comment
`M:346-350`. Ordering and widths are accurate.

### 4. fig3 architecture — ACCURATE

Two parallel towers drawn (GCN top, GATv2 bottom) both fed from "build graph" and both feeding
"concat + light attn + fc1(128)" (`make_figures.py:142-153`). Matches the real two-tower design:
`GCN_layers` + `GAT_layers`, concat then LightAttention then `fc1(...,128)` (`M:358,365,369-370,387,435`).
`edge_attr 41-d` annotation on the GAT tower (`:161`) matches `edge_dim=41` (`M:364`). Denoise head
branch off the concat/fc block (`:156-157`) matches the head hanging off the 128-dim pre-energy
features (`M:398-406`). "fc2 -> K terms / sum -> energy" (`:146`) matches `fc2 -> K` + `get_energy` sum
(`M:392,543,614`). The `[L,4,3]` coords box (`:138`) matches 4 backbone atoms. Accurate.

Minor nuance (not an error): the light-attention box says `fc1(128)` but LightAttention actually runs
BEFORE fc1 on the concatenated `fc_in_dim` features (`M:435` sets `LightAttention(embeddings_dim=fc_in_dim)`);
the box groups "concat + light attn + fc1" together, which is a fair schematic compression.

### 5. Rendered PNG visual/labeling problems

- **fig4 is rendered at low effective resolution / cramped.** The saved `fig4_levers_grid.png`
  displays with tiny sub-panel titles and near-unreadable italic captions (e.g. panel D
  "random-coil distances (value-only)", panel F "span S both dirs; edge_attr=[oh|oh|dist]=41").
  In the 2x3 grid at the delivered pixel size the caption text and axis legends are very small. The
  numbers/labels are CORRECT (see §2); this is a legibility issue, not a factual one. Consider a
  higher dpi or fewer words per caption.
- **fig1–fig3, fig5:** clean, no overlap, labels legible, text matches code. fig1 arrow "same
  sequence, two states" and the boxed `dG`/`ddG` formulas render correctly (`:77-82`).
- **fig1 wording nuance (not an error):** UNFOLDED labelled "random coil / chain-local" (`:74`)
  conflates the baseline unfolded reference (chain-local / tridiagonal zeroing) with the opt-in Lever D
  random-coil reference. Acceptable as an intuition-level simplification, but a purist could note the
  baseline is "chain-local", and "random coil" is specifically Lever D (`C:96-100`).
- No misspellings that change meaning; the deck body has the usual project typos only inside code
  identifiers it quotes verbatim (none introduced by the figures).

### 6. Does the Beamer deck cover the same GAPS the prior agents flagged? — NO (same gaps persist)

The deck is a visual/mechanical overview (levers, dimension contract, verification). It does NOT
close the content gaps Agents 1–5 recorded:

- **G-loss:** No loss formula. Deck never shows the Huber+ranking training objective or the
  `denoise_weight * MSE` aux term (Lever E) as an equation. (`.tex` mentions denoise only as a box.)
- **G-metrics:** No definition of PCC / Spearman / RMSE — the ablation ladder slide says "measures PCC
  one lever at a time" (`.tex:75`) without defining PCC or the eval protocol.
- **G-pretraining:** No EBM / native-vs-decoy pretraining theory; the deck presents the model as if
  trained directly on ddG (consistent with the v5 no-pretrained run, but the theory gap remains).
- **G-dataset:** No dataset biology — K50 / Tsuboyama / Megascale, the `--one_mut --dg_ml` PNAS
  curation, or why filtered data beats full Megascale. Nothing on protein/mutation counts.
- **G-mb (mini_batch):** No `--mini_batch_size 16` run command and no OOM warning (see §1 E3). A viewer
  running the pipeline from the deck alone would hit the argparse default 64 → OOM.
- **G-glycine:** Burial (Lever C) CB-density caveat for glycine (no CB) not mentioned in fig4/deck.

### Agent 6 verdict

- **E1/E2/E3 are NOT present in any presentation artifact** (`make_figures.py`, `.tex`, `build_html.py`,
  or the 5 PNGs). The figures deliberately avoid stating the derived widths (52/88/1112), the
  get_deltaG arity, and any run command, so none of the three doc-level errors leaked into the deck.
- **All lever A–F labels/descriptions are accurate** vs code, in both the figures and the `.tex` table.
- **fig2 node widths/order and fig3 architecture are correct** (16|32|burial|emb|20; two towers;
  edge_attr=41; denoise aux head; K energy terms).
- **Only issues found are: (a) fig4 legibility** (small captions at delivered resolution — cosmetic),
  and **(b) content GAPS** — the deck inherits the same missing loss formulas, metric definitions,
  pretraining theory, dataset biology, mini_batch/OOM warning, and glycine-CB caveat noted for the
  teaching docs. No numeric or arity corrections are required in the presentation code.

<!-- AGENT 6 DONE -->
