# DeepPEF v5 — Teaching Flow & Compatibility Guide

> A pedagogical walk-through of how a single mutation becomes a ddG prediction, why
> the code keeps every tensor width consistent, and how the six proposal levers (A–F)
> can be combined without breaking anything.
>
> Read this alongside `V5_PLAN.md`, `V5_README.md`, and `TEACHING_PRESENTATION.md`.

**Baseline best config (PCC ≈ 0.5259, all levers OFF):**

```bash
export WANDB_MODE=disabled
python DeepPEF_v5/training/pnas_train_v5.py \
  --dataset_type pnas --no_pretrained --one_mut --dg_ml \
  --loss_type huber_rank --ranking_weight 0.1 --use_knn_gat \
  --cosine_lr --lr_min 1e-6 --weight_decay 1e-5 \
  --mini_batch_size 16 --epochs 15 --seed 42
```

Every lever below defaults to the value that reproduces this baseline **bit-for-bit**.

---

## 1. End-to-End Training Flow

The pipeline turns a folder of per-protein tensors into a Pearson correlation (PCC)
between predicted and experimental stability. The key idea: the model never sees "ddG"
directly — it predicts a per-mutation **dG = E_unfolded − E_folded**, and correlation is
measured on those dG values.

```
                          RAW PER-PROTEIN TENSORS
                          (data/MsDs/training_data/<protein>/)
   ┌──────────────────────────────────────────────────────────────────────┐
   │ coords.pt   [L, 4, 3]     backbone atoms N, CA, C, CB                  │
   │ deltaG.pt   [n_mut]       experimental stability, one per mutation     │
   │ mask.pt     [L]           1 = valid residue, 0 = masked                │
   │ emb.pt      list of ProtT5 [L, 1024] tensors (one per mutant sequence) │
   │ mutation_files/<protein>.csv   mut_type, name, aa_seq, ddG_ML          │
   └──────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
   (1) MSDataset (training/new_dataset.py)  — ONE protein per __getitem__
   ┌──────────────────────────────────────────────────────────────────────┐
   │  • drop ins/del mutations                                              │
   │  • dg_ml     → clamp deltaG to [-1, 5]        (--dg_ml)                 │
   │  • one_mut   → drop multi-site ':' mutations  (--one_mut)              │
   │  • ds_type=pnas → keep only PNAS-whitelisted mutation names            │
   │  • remove ThermoMPNN test proteins; remove train homologs              │
   │  • one_hot   → get_one_hot(aa_seq) → [n_mut, L, 20]                     │
   │  • ALWAYS keep index 0 (wildtype reference for ddG)                    │
   │  returns dict: name, prott5[n_mut,L,E], coords[L,4,3], one_hot,        │
   │                delta_g[n_mut], masks[L]                                 │
   └──────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
   (2) DataLoader(batch_size=1, shuffle=True)
       → the "batch" is a whole protein; mutations are the inner loop
                                     │
                                     ▼
   (3) Trainer.train()  (training/pnas_train_v5.py)
       for epoch in range(epochs):                       # 15
         for batch in train_ds:                          # one protein
           for j in range(0, n_mut, mini_batch_size):    # step of 16
                                     │
                                     ▼
   (4) Trainer.get_deltaG(batch, j)  — the physics core
   ┌──────────────────────────────────────────────────────────────────────┐
   │  slice 16 mutations: one_hot[j:j+16], prott5[j:j+16]                    │
   │                                                                        │
   │  for each mutation m in the mini-batch:                                │
   │    folded_graph[m]   = get_graph(coords, one_hot[m], emb[m], mask)      │
   │    unfolded_graph[m] = get_unfolded_graph(coords, one_hot[m], emb[m])   │
   │                                                                        │
   │  all_graphs = cat([folded(16), unfolded(16)])  → 32 graphs / step       │
   │  (if --use_knn_gat: compute CA k-NN cutoff, pass ca_coords)             │
   └──────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
   (5) PEM model (model/hydro_net_v5.py)  — one energy scalar per graph
   ┌──────────────────────────────────────────────────────────────────────┐
   │  node feature layout (per residue):                                    │
   │    [ D:dist(16) | Fb:bonded(32) | (burial) | emb:E | one_hot(20) ]      │
   │  split → GCN branch (dist+bonded+aa) + GAT branch (dist+aa)             │
   │  → 3 GNN layers → concat → InstanceNorm → (+raw emb) → LightAttention   │
   │  → fc1(→128) → fc2(→1) per residue → get_energy = Σ per-residue         │
   │  (length_norm: divide by residue count → intensive energy)             │
   └──────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
   (6) Energy difference
       folded_energy   = energies[:16]
       unfolded_energy = energies[16:]
       output (dG)     = unfolded_energy − folded_energy      # [16]
                                     │
                                     ▼
   (7) Loss (training only)
   ┌──────────────────────────────────────────────────────────────────────┐
   │  primary   = Huber(output, delta_g)          (--loss_type huber_rank)  │
   │  reg_loss  = REG_LAMBDA * ||params||          (0 by default)            │
   │  energy_reg= E_REG_LAMBDA * ||energies||^2    (0.001, keeps E bounded)  │
   │  rank_loss = 0.1 * margin_ranking(output, delta_g)  (--ranking_weight)  │
   │  corr_loss = w * (1 - Pearson) or CCC          (--corr_weight, opt-in)  │
   │  denoise   = w * MSE(pred_noise, true_noise)   (Lever E, train only)    │
   │  loss = primary + reg + energy_reg + rank + corr (+ denoise)           │
   │  → backward → clip_grad_norm(10) → Adam.step → cosine LR step          │
   └──────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
   (8) Trainer.validate(epoch)  — no_grad, empty_cache each mini-batch
       collect all (delta_g, output) over the ThermoMPNN test proteins
       PCC = corrcoef(val_dg, val_dg_pred);  also Spearman, RMSE, ddG PCC
       save best_model.pt when PCC improves
                                     │
                                     ▼
                    BEST VALIDATION PCC  (the number we grade on)
```

### Why this shape of loop?

- **One protein = one DataLoader batch** because a protein graph is large; `batch_size=1`
  is set in `CFG` and in `run_training()`. The *real* batching happens over **mutations**.
- **mini_batch_size = 16** mutations → 32 graphs (16 folded + 16 unfolded) per forward
  pass. This is the single most important memory knob (see §5).
- **Index 0 is always kept** by the dataset (`indexes.add(0)`), so a wildtype reference
  is always present; ddG metrics subtract the first value.

---

## 2. The Dimension Contract (One Source of Truth)

**Rule: every tensor width in the pipeline is derived from a single object,
`model.model_cfg_v5.CFG`. The graph builder and the model read the *same* numbers.**

There are two consumers that must agree on widths:

| Consumer | File | What it produces / expects |
|----------|------|----------------------------|
| Graph builder | `training/train_utils.py` (`get_graph`, `get_unfolded_graph`) | node features `[L, 16 + 32 + E + 20]` |
| Model | `model/hydro_net_v5.py` (`PEM`) | slices those exact widths back out |

The node feature layout is fixed and both sides slice it identically:

```
        ┌─────────┬──────────────┬──────────┬─────────────┬──────────────┐
 node = │ D dist  │ Fb bonded    │ (burial) │ emb E        │ one_hot 20   │
        │ 16      │ 32 (2×16)    │ 0 or 1   │ 1024/1536/…  │ 20           │
        └─────────┴──────────────┴──────────┴─────────────┴──────────────┘
 slice indices in PEM.forward:
   x_dist       = x[:, :16]                     # non_bonded_index = 16
   x_bonded     = x[:, 16:32]                   # first 16 of the 32 bonded cols
   x_emb        = x[:, llm_index:one_hot_index] # llm_index = -(E + 20)
   x_onehot     = x[:, -20:]                    # one_hot_index = -20
```

### How the contract is enforced at runtime

In `pnas_train_v5.py`, right after `parse_args()`, the CLI values are **pushed onto the
one `CFG` object** before the model or graph builder are ever constructed:

```python
CFG.emb_input_dim = EMB_DIMS[args.emb_type]   # e.g. 1024 (prott5) or 1536 (dual_esmif)
CFG.energy_terms  = args.energy_terms         # A
CFG.rbf_centers   = args.rbf_centers          # B
CFG.use_burial    = args.use_burial           # C  → CFG.burial_dim = 1 if on else 0
CFG.flory_unfolded= args.flory_unfolded       # D
CFG.denoise_weight= args.denoise_weight       # E
CFG.gcn_span      = args.gcn_span             # F
CFG.use_edge_features = args.use_edge_features# F
```

Both `hydro_net_v5.py` and `train_utils.py` do `from model.model_cfg_v5 import CFG`
— the *same* module object. So when the model computes, e.g.,
`llm_index = -(CFG.emb_input_dim + 20)`, it automatically matches the width the graph
builder concatenated. No hard-coded `1024` on the model side.

### Why this matters for compatibility

This is exactly the class of bug that bit v5 before (recorded in project memory):

> **Split-CFG bug:** `hydro_net_v5` originally imported `model.model_cfg` (base CFG,
> `emb_input_dim=1024`) while training mutated `model_cfg_v5.CFG`. Harmless for ProtT5
> (1024), but for `dual_esmif` (1536) the model sliced the embedding at the wrong offset
> — **silently misaligned**, no crash, just wrong numbers. Fixed by importing the *one*
> CFG everywhere.

The lesson: **a width that lives in two places will eventually disagree.** Keeping widths
in one `CFG` object means a lever can change a dimension in exactly one spot and every
downstream index recomputes correctly.

---

## 3. Lever Compatibility Matrix

The six proposal levers differ in **what part of the tensor contract they touch**. That
is the key to knowing which combine safely.

| Lever | Flag | Changes node width? | Changes edges? | Value-only? | Changes checkpoint shapes? |
|-------|------|:---:|:---:|:---:|:---:|
| **A** energy decomposition | `--energy_terms K` | No | No | No (adds a head) | **Yes** (final head `128→K`) |
| **B** RBF distance bank | `--rbf_centers M` | No (dim-preserving) | No | **Yes** (D reweighted) | No |
| **C** burial node feature | `--use_burial` | **Yes** (+1 col) | No | No | **Yes** (fc input widths +1) |
| **D** Flory unfolded ref | `--flory_unfolded` | No | No | **Yes** (unfolded D only) | No |
| **E** denoising aux head | `--denoise_weight w` | No | No | No (aux loss) | **Yes** (extra denoise head) |
| **F** extended conn. + edges | `--gcn_span S`, `--use_edge_features` | No | **Yes** | No | Maybe (edge-feature GAT weights) |

### Reading the matrix — the four "kinds" of change

- **A, E are orthogonal to widths.** They add an *output head* (A: K energy terms;
  E: a coordinate-noise predictor). They do not touch node feature widths or edges, so
  they compose with everything. They *do* add parameters, so a checkpoint trained with
  them must be reloaded with the same flags (see §5 ensemble rule).
- **B is dimension-preserving.** It replaces the single Gaussian kernel `exp(coef·D²)`
  with a bank of M RBF kernels, then **sums back to the same `dist_dim=16`**. The node
  layout is byte-identical; only the *values* in the distance block change. Safe to add
  on top of anything.
- **D is value-only.** It changes how the **unfolded** graph's distances are built
  (random-coil `|i−j|^ν` scaling instead of the tridiagonal zeroing in
  `zero_except_udiagonal`). The folded graph, all widths, and all edges are untouched.
- **C changes node width.** Turning on burial inserts a per-residue CB-neighbor-density
  scalar (`burial_dim=1`) into the node vector *before* the embedding block. That shifts
  the fc input widths, so a burial checkpoint is a different shape than a baseline one.
- **F changes edges only.** `--gcn_span S` adds sequence-offset edges (±2…±S) to the GCN
  beyond the baseline `(i, i+1)`. `--use_edge_features` feeds GATv2 a 41-dim edge_attr
  `[onehot_src(20) | onehot_dst(20) | dist(1)]`. Node widths are unchanged; only the
  graph topology / edge tensor changes (and, with edge features, the GAT weight shapes).

### Composition guidance

```
   value-only / dim-preserving levers  →  freely stackable, cheap to try:
        B (RBF)   D (Flory)   +  A (energy_terms)  +  E (denoise)
   ────────────────────────────────────────────────────────────────
   width- or edge-changing levers      →  compose fine, but each changes
        C (burial: +1 node col)             checkpoint shape or topology,
        F (gcn_span / edge features)        so keep flags identical at
        A/E (extra heads)                   reload / ensemble time.
```

There are **no mutually-exclusive levers** — the CFG contract is designed so any subset
composes. The only discipline required is: *whatever flags changed the model's parameter
shapes (A, C, E, and F-with-edge-features) must be re-supplied when you load that
checkpoint.*

---

## 4. Ablation Methodology (the Evolution Ladder)

`run_v5_evolution.sh` is an **evolution ladder**: each rung takes the *winner of the
previous rung* and adds exactly **one** change.

```
  RUNG 0  baseline                         (reproduce ~0.5259)
     │        COMMON flags only
     ▼
  RUNG 1  + --length_norm                  intensive energy
     │
     ▼
  RUNG 2  + --serial_fusion                PLM into message-passing
     │        (kept: length_norm)
     ▼
  RUNG 3  + --corr_weight 0.1 --corr_type pearson
     │        (kept: length_norm, serial_fusion)
     ▼
  RUNG 4  swap --emb_type dual_esmif       (only if esmif_enc.pt exists)
     │
     ▼
  RUNG 5  5-seed ensemble of best rung     seeds 42,1,2,3,4
```

The shared `COMMON` string is fixed for every rung:

```
--dataset_type pnas --no_pretrained --one_mut --dg_ml
--loss_type huber_rank --ranking_weight 0.1 --use_knn_gat
--cosine_lr --lr_min 1e-6 --weight_decay 1e-5 --mini_batch_size 16
--epochs 15 --seed 42
```

### Why one change at a time?

This is a **hard-won project lesson**, not a stylistic preference. From the project
memory of failed experiments:

> *"My predictions repeatedly failed because … I changed too many things at once
> (new architecture + new data + new output)."*

Concrete failures caused by bundling changes:

- **GNN-SM** (subtract-mut) dropped to PCC 0.42 vs 0.5259 — a new architecture *and* a
  new output head at once, so the cause was unattributable until isolated.
- **`--full_megascale`** (more data) *hurt* (0.4098) because it also removed the PNAS
  quality filters — two coupled changes masking each other.

If a rung regresses, one-change-at-a-time means you know **exactly** which lever did it
and can stop climbing that branch. If you change three levers and PCC moves, you have
learned nothing about which one mattered.

**Rule of the ladder:** the levers A–F should be added the same way — take the current
best config and toggle a *single* lever, compare the best-model PCC, keep it only if it
helps, then move to the next lever from the new baseline.

---

## 5. Reproducibility & Gotchas

### Non-negotiables

| Setting | Why |
|---------|-----|
| `export WANDB_MODE=disabled` | The ladder runs headless; W&B init otherwise blocks / prompts. Baked into `run_v5_evolution.sh`. |
| `--mini_batch_size 16` | **The OOM lesson.** Default is 64. With 64 → 128 graphs/step → ~7.7 GB → OOM on an 8 GB RTX 4070. With 16 → 32 graphs/step → ~2 GB. Commit `df7e831` literally exists to reduce 64→16. |
| `--seed 42` | `set_seed` fixes torch/cuda/numpy/random and sets `cudnn.deterministic=True`, `benchmark=False`. Reproducible runs; also the 20% val split uses the seed. |
| cache cleanup | The script `rm -f ./data/MsDs/mutation_files/.cache.csv` and `rm -rf ./data/MsDs/training_data/.cache` — stale caches caused `FileNotFoundError`. |
| `torch.cuda.empty_cache()` in validate | Called every mini-batch in `validate()` (plus `gc.collect()`) to keep validation memory flat. |

### CPU fallback for smoke tests

`DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'` — the same code runs on CPU
for a quick shape/plumbing check. Use `config_paths.set_data_root(...)` (or
`--data_root` / `$DEEPPEF_DATA_ROOT`) to point at a tiny dataset so a smoke test finishes
fast without touching CUDA. `--debug` further trims to 5 proteins and skips W&B/saves.

### The ensemble / reload rule (mirror shape-changing flags)

A checkpoint stores parameter *shapes*. `load_state_dict` fails (or silently misaligns)
if the model you build has different shapes than the one you trained.

> **You MUST re-supply every shape-changing flag when loading or ensembling.**
> Shape-changing levers: **A** (`--energy_terms`), **C** (`--use_burial`),
> **E** (`--denoise_weight`), **F** (`--use_edge_features`) — plus the training-time
> structural flags `--serial_fusion`, `--use_learned_aa`, `--emb_type`, `--use_knn_gat`.

This mirrors a real fixed bug (project memory):

> **ensemble_v5 bug:** importing `pnas_train_v5` re-ran its module-level `parse_args()`
> and it also omitted constructor args, so the rebuilt `PEM` had a different topology
> (`USE_KNN` unset) and shapes → `load_state_dict` mismatch. Fix: inject matching argv
> before import and build `PEM` with the *same* args as training.
>
> **Gotcha:** `pnas_train_v5.py` calls `argparse` at **module level** (line ~128).
> Importing the module *runs* the parser. If you import it from another script, guard
> `sys.argv` first.

### Value-only vs shape-only levers at reload time

- **B (RBF), D (Flory)** change only *values*, not parameter shapes — a checkpoint is
  reload-compatible even if you toggle them, though results will differ. (You still want
  to keep them consistent for a fair ensemble.)
- **A, C, E, F(edge-features)** change parameter *shapes* — flags are mandatory at reload.

---

## 6. A Learner's Decision Tree

Pick a lever by the *scientific question* you are asking, not by expected PCC.

```
 What do you want to improve?
 │
 ├─ "I want the model to EXPLAIN its energy"  (interpretability)
 │      → Lever A  --energy_terms K
 │        splits the single scalar into K per-residue terms that sum to the total;
 │        inspect per-term contributions. Orthogonal to widths.
 │
 ├─ "Distances feel too coarse / one Gaussian is crude"  (better distances)
 │      → Lever B  --rbf_centers M   (+ --rbf_min/--rbf_max)
 │        replaces one Gaussian with an M-kernel RBF bank, summed back to 16 dims.
 │        Dimension-preserving → safe to add anytime.
 │
 ├─ "Buried vs exposed residues behave differently"  (solvation / burial)
 │      → Lever C  --use_burial   (+ --burial_radius)
 │        adds a CB neighbor-density node feature. NOTE: changes node width →
 │        checkpoint shape changes; re-supply the flag at reload.
 │
 ├─ "My unfolded state is unphysical (just tridiagonal)"  (unfolded physics)
 │      → Lever D  --flory_unfolded   (+ --flory_nu 0.5)
 │        uses a random-coil |i-j|^nu reference for the UNFOLDED graph only.
 │        Value-only → reload-safe.
 │
 ├─ "I want the native structure to sit at an energy minimum"  (native minimum)
 │      → Lever E  --denoise_weight w   (+ --denoise_sigma, --denoise_prob)
 │        aux head predicts injected coordinate noise; trains the energy surface to
 │        favor the true fold. Train-only loss; adds a head (reload flag needed).
 │
 └─ "Residues far apart in sequence but close in space need to talk"  (long-range context)
        → Lever F  --gcn_span S   and/or   --use_edge_features
          gcn_span connects ±2..S sequence offsets; edge features give GATv2 a
          41-dim [onehot_src|onehot_dst|dist] edge_attr. Changes EDGES (and, with
          edge features, GAT weight shapes) — keep flags consistent at reload.
```

### Quick reference — lever → mechanism → contract impact

| Goal | Lever | Mechanism | Contract impact |
|------|-------|-----------|-----------------|
| Interpretability | A | K per-residue energy terms summed | +head params |
| Better distances | B | RBF kernel bank → summed to 16 | none (dim-preserving) |
| Solvation | C | CB neighbor-density node feature | +1 node column |
| Unfolded physics | D | Flory `|i−j|^ν` unfolded reference | unfolded values only |
| Native minimum | E | denoising MSE aux loss (train only) | +head params |
| Long-range context | F | extended GCN span / GAT edge_attr | edges (± GAT weights) |

---

## Appendix: One-Screen Mental Model

```
 dataset filters (pnas, one_mut, dg_ml)   →   clean per-protein mutations
 graph builder (train_utils, reads CFG)   →   node = [D|Fb|(burial)|emb|onehot]
 PEM model (hydro_net_v5, reads SAME CFG)  →   per-residue energy → Σ (÷L if length_norm)
 folded − unfolded                          →   predicted dG per mutation
 Huber + rank (+ corr + denoise)            →   optimize
 corrcoef(dG_true, dG_pred) on test         →   PCC  (the score)

 Levers A–F all default to the baseline shape; each is one CFG field the graph
 builder AND the model read from the SAME object, so widths never drift.
```
