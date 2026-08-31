# DeepPEF — End-to-End Teaching Deck

**A slide-by-slide walkthrough of the Protein Energy Model (PEM): from biology → graph → GNN → energy → dG → ddG, plus the v5 improvements.**

> How to read this document: each `## Slide N` is one slide. Present them in order.
> Every architectural claim is traced to a source file and line so you can defend it to your professor.
> **Honesty note:** all *PCC-gain* numbers for v5 methods are **hypotheses** from the literature and analogous
> models. They are only *proven* by running the ablation experiments. Do not present them as measured results.

---

## Slide 1 — The Problem: Protein Stability

**What is a protein?** A chain of amino acids (residues) that folds into a specific 3D shape. The shape
determines the function. If the protein does not fold, or unfolds too easily, it does not work.

**What is stability?** A folded protein sits in an equilibrium between two states:

```
        FOLDED  <====>  UNFOLDED
       (working)       (random coil, non-functional)
```

**dG — folding free energy** (units: kcal/mol):

```
    dG = G_unfolded − G_folded
```

- `dG > 0`  → the folded state is favored → the protein is **stable**.
- Larger `dG` → **more stable**.

**ddG — change in stability upon mutation:**

```
    ddG = dG_mutant − dG_wildtype
```

- `ddG < 0` → the mutation **destabilizes** the protein.
- `ddG > 0` → the mutation **stabilizes** it.

**Why it matters:** protein/enzyme engineering, drug design, understanding disease-causing mutations.
Measuring dG/ddG in a wet lab is slow and expensive; predicting it computationally is the goal.

**KEY FRAMING (correction #1 from the audit):**
> **DeepPEF predicts dG (a folding free energy). It does NOT predict ddG directly.**
> ddG is *derived* at validation time as `dG_mut − dG_wt` (see Slide 9).
> Source: `Megascale-fineTuning/pnas_train.py:624-625`.

---

## Slide 2 — The Core Idea in One Picture

We compute the energy of the SAME protein sequence in two graph "states" and subtract:

```
   For one sequence:

   ┌─────────────────────┐            ┌─────────────────────┐
   │   FOLDED graph      │            │  UNFOLDED graph     │
   │  (all 3D contacts)  │            │ (linear chain only) │
   └──────────┬──────────┘            └──────────┬──────────┘
              │  PEM model                       │  PEM model
              ▼                                  ▼
         E_folded                           E_unfolded
              └───────────────┬──────────────────┘
                              ▼
                  dG = E_unfolded − E_folded
```

`dG` is a *learned* proxy energy, trained so that its value correlates with the experimental dG label.

**Per mutation, what changes?** (correction: structure is FIXED)
- **Same** wildtype 3D coordinates (we do NOT re-fold the mutant).
- **Different** amino-acid identity at the mutated position → different **one-hot** (20-dim).
- **Different** **PLM embedding** (ProtT5 of the mutant sequence) — *this is the main mutation signal*.

Source: `pnas_train.py get_deltaG:668-706`, `train_utils.py get_graph/get_unfolded_graph`.

---

## Slide 3 — Input Data (what one protein folder contains)

For each protein we store tensors (`new_dataset.py`):

| Tensor | Shape | Meaning |
|---|---|---|
| `coords.pt` | `[L, 4, 3]` | 3D coords of 4 backbone atoms per residue: **N, CA, C, CB** |
| `mask.pt` | `[L]` | 1 = valid residue, 0 = ignore (padding / missing) |
| `one_hot_encodings.pt` | `[n_muts, L, 20]` | amino-acid identity per residue, **20 AA** |
| `emb.pt` (ProtT5) | `[n_muts, L, 1024]` | protein-language-model embedding per residue |
| `esmif_enc.pt` (optional) | `[L, 512]` | ESM-IF1 inverse-folding structural features (v5 dual mode) |
| `deltaG.pt` | `[n_muts]` | the **label** — experimental dG for each variant |

- `L` = sequence length (number of residues). `n_muts` = number of measured variants (row 0 = wildtype).
- **CB is computed** from N, CA, C when missing (`train_utils.add_cb`), so glycine also gets a virtual CB.

**Correction #5 — one-hot is 20, not 21.** Some stored files carry a 21st padding column; the model's
`get_one_hot` (train_utils.py:313) produces exactly **20** dims, and `normalize_batch` (pnas_train.py:235)
trims one_hot with `[..., :-1]`. **The model uses 20.**

**Tiny worked example (a 3-residue peptide "A-C-D"):**
```
one_hot = [[1,0,0,...],   # A → index 0
           [0,1,0,...],   # C → index 1
           [0,0,1,...]]   # D → index 2      shape [3, 20]
```

---

## Slide 4 — The Graph Representation (nodes, edges, distances)

We turn the protein into a **graph** where each **residue is a node**.

**Node features** are built in `train_utils.get_graph` (line 196). Final per-residue vector layout:

```
   [ 16 distance | 32 bonded | PLM embedding (1024) | 20 one-hot ]  =  1092 dims
     └── geometry ──┘         └──── chemistry / identity ────┘
```
(With ESM-IF1 dual embeddings the PLM block is 1024+512=1536 → total **1604**.)

**Where do the 16 "distance" numbers come from?**
Each residue has **4 atoms** (N, CA, C, CB). Between two residues i and j there are **4×4 = 16**
inter-atomic distances. So the raw pairwise object is `D[i, j, :]` of length 16
(`get_dist_matrix`, train_utils.py:257).

```
   residue i atoms: N  CA  C  CB
   residue j atoms: N  CA  C  CB
   16 distances = every atom-of-i to every atom-of-j
   e.g.  d(Ni,Nj), d(Ni,CAj), ... , d(CBi,CBj)
```

**Gaussian kernel** turns raw distances (Å) into "contact strength" in [0,1]:
```
   contact = ReLU( exp( gaussian_coef · D² ) ),   gaussian_coef = −0.08
```
(train_utils.py:208; CFG.gaussian_coef = −0.08 in model_cfg.py:32.)

**Tiny worked example (one atom pair):**
```
   D = 3 Å  → exp(−0.08 · 9)  = exp(−0.72) ≈ 0.487   (close → strong contact)
   D = 8 Å  → exp(−0.08 · 64) = exp(−5.12) ≈ 0.006   (far  → almost no contact)
   D = 15 Å → exp(−0.08 · 225)≈ 1e−8               (effectively zero)
```
So the kernel makes nearby atoms matter and distant ones vanish — a smooth, differentiable "contact map".

**Aggregation:** for each residue we **sum the kernelized contacts over all partners** and normalize,
giving a **16-dim** distance feature per residue (`D.sum(dim=1)`, train_utils.py:216). The **32 bonded**
features are the kernelized distances to the immediate sequence neighbors i−1 and i+1
(`get_bonded_features`: two 16-dim vectors → 32), encoding the local backbone geometry.

---

## Slide 5 — Folded vs Unfolded: the Thermodynamic Trick

Both graphs use the **same coordinates and same node features layout**. The only difference is **which
edges/contacts survive** in the distance block.

**FOLDED graph** (`get_graph`): keep **all** pairwise contacts → the residue "feels" its full 3D
environment (helices, sheets, long-range contacts).

**UNFOLDED graph** (`get_unfolded_graph` → `zero_except_udiagonal`, train_utils.py:302): zero out
everything **except**:
- the **diagonal** (a residue with itself), and
- the **immediate ±1 neighbors** (i−1 and i+1).

```
   Contact matrix kept (X = kept, . = zeroed):

   FOLDED                     UNFOLDED (linear chain)
        1 2 3 4 5                  1 2 3 4 5
      1 X X X X X                1 X X . . .
      2 X X X X X                2 X X X . .
      3 X X X X X                3 . X X X .
      4 X X X X X                4 . . X X X
      5 X X X X X                5 . . . X X
```

**Correction #4 — it is i±1, NOT i±2.** The code keeps `f1 = D[i, i+1]`, `f2 = D[i+1, i]`, and the
diagonal. Only the tridiagonal band survives. This models a **random-coil / linear chain**: connected
along the backbone but with no folded 3D contacts.

**Interpretation:**
```
   dG = E_unfolded − E_folded
```
measures the energetic "reward" the model assigns for forming real 3D contacts (folded) versus the
bare linear chain (unfolded). A destabilizing mutation should shrink that reward.

---

## Slide 6 — The Two GNN Branches (GCN + GAT), and WHY

Inside `PEM.forward` (hydro_net.py:398) the node features are split and fed to **two parallel graph
networks** that see **different edges** and **different feature subsets**:

**Branch A — GCN (Graph Convolutional Network)** — *local backbone*
- Edges: **sequential only**, `(i, i+1)` — a simple chain (`get_edge_index`, hydro_net.py:543-549).
- Input features: `dist(16) + bonded(32→first 16 used) + aa(20)`.
- WHY: captures **local, along-the-chain** patterns (secondary structure, neighbor context).

**Branch B — GAT (Graph Attention Network, GATv2)** — *long-range 3D contacts*
- Edges: either **fully-connected** `O(L²)` (every residue to every other) **or** **k-NN / distance-cutoff**.
- Input features: `dist(16) + aa(20)`.
- WHY: **attention** lets each residue weigh its spatial neighbors adaptively — good for tertiary contacts.

**k-NN GAT (the proven, better setting):**
- Build edges from **CA atom** positions (**CA = atom index 1**; order is N=0, CA=1, C=2, CB=3).
- Connect residues whose CA-CA distance is within a cutoff derived from the **k=30** nearest neighbors,
  capped at **12 Å** (`pnas_train.py:684-700`, `gat_cutoff=12.0`).
- WHY better: fully-connected treats a 10 Å and a 60 Å pair identically; k-NN focuses attention on the
  local 3D neighborhood, which is where the physics is. (Our own result: ~+0.043 PCC.)

The two branch outputs are **concatenated** (hydro_net.py:452) → InstanceNorm.

---

## Slide 7 — Light Attention + FC Head → per-residue energy → SUM

After the GNN branches concat and normalize, the model appends the **raw PLM embedding** (default mode,
`emb_projection="none"`) and passes through **Light Attention** then a small FC head.

**Light Attention** (`LightAttention`, hydro_net.py:742; Stark et al. 2022):
```
   o          = Conv1d_feature(x)         # kernel_size = 9
   attention  = Conv1d_attention(x)       # kernel_size = 9
   output     = o * softmax(attention)    # feature-wise soft gating along the sequence
```
It lets the model up-weight the residues that matter (e.g., near the mutation) using a 9-wide 1D
convolution over the sequence.

**FC head → per-residue scalar:**
```
   x → fc1 (fc_in_dim → 128) → ReLU → fc2 (128 → 1)     # hydro_net.py:369-370, 468-477
   → per-residue energy of shape [B, L, 1]
```

**Energy readout (`get_energy`, hydro_net.py:517-527):**
```
   E = torch.sum(Fh, dim=(1,2))      # SUM over all residues → one scalar per graph
```

**Tiny worked example (L = 4 residues):**
```
   per-residue energies:  [0.8, 1.2, 0.5, 0.9]
   E = 0.8 + 1.2 + 0.5 + 0.9 = 3.4
```
This `E` is computed for the folded graph and the unfolded graph; `dG = E_unfolded − E_folded`.

---

## Slide 8 — THE KEY LIMITATION: the Energy is an EXTENSIVE SUM

**This is the scientific heart of the v5 work (correction #2).**

`E = sum over residues` means the energy is **extensive** — it **grows with protein length**. A 300-residue
protein will produce a larger `|E|` than a 60-residue protein *simply because it has more terms to add*,
independent of how stable it actually is.

```
   protein length L → 60      per-residue mean ≈ 0.9  →  E ≈  54
   protein length L → 300     per-residue mean ≈ 0.9  →  E ≈ 270
   (same intrinsic stability, 5× the energy, purely from length)
```

**Why this hurts ddG:**
- Real dG/ddG are roughly **intensive** properties of the fold, not proportional to raw length.
- Length becomes a confounding variable that the model must "cancel out" instead of learning chemistry.
- Cross-protein comparison (which the PCC metric rewards) is polluted by a length signal.

**The v5 fix — `length_norm` (per-residue / intensive energy):**
```
   Baseline:   E = sum_i  e_i                         (extensive, length-biased)
   v5:         E = (1/L)  sum_i  e_i   =  mean_i e_i   (intensive, length-invariant)
```
This is a one-line surgical change in `hydro_net_v5.py get_energy`. dG is still `E_unfolded − E_folded`,
so nothing else in the pipeline changes. It removes the length confound so the model can focus on the
actual stability signal. (See Slide 10 for confidence.)

---

## Slide 9 — Training Loop and How ddG is Derived

**Training target = dG.** We never train on ddG directly.

**Loop structure** (`pnas_train.py Trainer.train:474`):
- One protein at a time (`batch_size = 1`); mutations processed in **mini-batches of 16**
  (`--mini_batch_size 16`). **Correction #3: the argparse default is 64, but 64 causes OOM; the proven
  runs use 16.** With 16 mutations we build 16 folded + 16 unfolded = **32 graphs** per forward pass.
- For each mini-batch: build folded & unfolded graphs → model → `dG = E_unfolded − E_folded`
  (`get_deltaG:668-706`).

**Loss** (`pnas_train.py:503-518`):
```
   loss = Huber(dG_pred, dG_true)                       # primary, robust regression
        + ranking_lambda · pairwise_ranking(dG_pred, dG_true)   # order/trend, λ = 0.1
        + reg_loss + energy_reg                          # small regularizers
```
- **Huber** (delta=1.0): like MSE near 0, like L1 for outliers → robust to noisy labels.
- **Pairwise ranking** (`ranking_loss:242`): if `dG_true_i > dG_true_j`, push `dG_pred_i > dG_pred_j`
  by a margin. Directly trains the *ordering* that correlation cares about.

**Optimizer / schedule:** Adam, `lr = 1e-4`, **cosine annealing** down to `1e-6`, weight_decay `1e-5`,
gradient clipping (max_norm 10), **15 epochs**, `WANDB_MODE=disabled`.

**Data:** PNAS-filtered **single** mutations (`--one_mut`), dG clamped to **[-1, 5]** (`--dg_ml`),
~**340 train** proteins / ~**28 test** proteins. Baseline **PCC = 0.5259** (from scratch, no pretraining).

**Deriving ddG at validation** (`pnas_train.py:621-625`):
```
   dg_wt      = dG_pred[0]              # row 0 is the wildtype
   ddg_pred   = dG_pred − dg_wt
   ddg_true   = dG_true − dg_wt_true
```
**Metric:** Pearson correlation (PCC) between predicted and true dG (and separately for ddG), plus
Spearman and RMSE.

**Tiny worked ddG example:**
```
   dG_pred(WT)   = 3.0     dG_true(WT)   = 3.2
   dG_pred(mut)  = 2.4     dG_true(mut)  = 2.5
   ddG_pred = 2.4 − 3.0 = −0.6   (predicted destabilizing)
   ddG_true = 2.5 − 3.2 = −0.7   (measured destabilizing)  → good agreement
```

---

## Slide 10 — The 5 v5 Methods: what, where, and confidence

Each method is **flag-gated** so it can be ablated one at a time. **All PCC numbers are hypotheses.**

**1) `length_norm` — intensive energy (Slide 8 fix)**
- **What:** `E = mean_i e_i` instead of `sum_i e_i`.
- **Where:** `hydro_net_v5.py get_energy` (replaces the extensive sum used in `hydro_net.py:525`).
- **Why it should help:** removes the protein-length confound from cross-protein dG comparison.
- **Confidence:** MEDIUM–HIGH that it *helps or is neutral* (it is a principled, low-risk change).
  Hypothesized gain **+0.01 to +0.03 PCC** (estimate only).

**2) `serial_fusion` — inject PLM into GNN message-passing**
- **What:** project the ProtT5 embedding down (MLP → `serial_fusion_dim`, default 64) and **concatenate it
  into the GNN input** so it participates in message-passing.
- **Where:** `PEM.__init__` builds `serial_projector`; `PEM.forward` concatenates into `x_gcn`/`x_gat`
  (hydro_net.py:316-322, 442-446).
- **Why:** by default (`emb_projection="none"`) the PLM is only added **after** the GNN (hydro_net.py:458)
  — so the graph layers never "see" the mutation signal during message-passing. Serial fusion fixes that.
- **Confidence:** MEDIUM. Analogous to ESM-GearNet (ICLR 2024). Hypothesized **+0.02 to +0.05 PCC**.

**3) Correlation loss (`--corr_weight`, `--corr_type pearson|ccc`)**
- **What:** add a differentiable Pearson / CCC term so we optimize the **PCC trend directly**, not only
  point-wise error.
- **Where:** `training/losses_v5.py` (pearson_loss, ccc_loss), wired into the train loop.
- **Why:** Huber/L1 minimize magnitude error; the actual metric is correlation. Optimizing it directly
  aligns training with evaluation.
- **Confidence:** MEDIUM. Hypothesized **+0.01 to +0.04 PCC** (risk: instability on tiny mini-batches).

**4) `dual_esmif` embeddings (ProtT5 1024 + ESM-IF1 512 = 1536)**
- **What:** concatenate an **inverse-folding** structural embedding (ESM-IF1 encoder) to the PLM block.
- **Where:** `new_dataset.py _load_and_concat_embeddings` (dual_esmif branch); node vector becomes 1604-dim.
- **Why:** ESM-IF1 encodes "which residue fits this structural pocket" — a strong stability prior.
  Inverse-folding pretraining is the single biggest lever in the literature (ThermoMPNN).
- **Confidence:** MEDIUM–HIGH *directionally* (this is the known big lever), but our concat is a lighter
  form than full pretraining. Hypothesized **+0.05 to +0.15 PCC** (wide band; experiment decides).

**5) Seed ensemble**
- **What:** train several seeds, **average dG** predictions.
- **Where:** `training/ensemble_v5.py`.
- **Why:** variance reduction; almost always a small, reliable gain.
- **Confidence:** HIGH that it helps a little. Hypothesized **+0.02 to +0.04 PCC**. Do it **last**.

**Ladder discipline (memory lesson):** change **ONE** thing per rung; each rung inherits the previous
winner; if a rung regresses, drop it and continue from the prior winner. Do not stack losers.

---

## Slide 11 — End-to-End Flow (ASCII recap)

```
  ┌────────────────────────────────────────────────────────────────────────────┐
  │  INPUT (per mutation): coords[L,4,3] (FIXED WT) · one_hot[L,20] (mutant)     │
  │                        ProtT5 emb[L,1024] (mutant) · mask[L] · label dG      │
  └───────────────┬─────────────────────────────────────┬──────────────────────┘
                  │ build FOLDED graph                   │ build UNFOLDED graph
                  │ (all contacts)                       │ (tridiagonal: diag + i±1)
                  ▼                                       ▼
        node features [L, 1092]                 node features [L, 1092]
        = 16 dist | 32 bonded | 1024 PLM | 20 AA   (1604 if dual_esmif)
                  │                                       │
                  ▼   ── PEM.forward (shared weights) ──  ▼
     ┌───────────────────────────┐          ┌───────────────────────────┐
     │ split → GCN (i,i+1 edges)  │          │ split → GCN                │
     │        GAT (kNN/12Å, CA)   │          │        GAT                 │
     │ concat → InstanceNorm      │          │ concat → InstanceNorm      │
     │ + raw PLM  (or serial      │          │ + raw PLM                  │
     │   fusion INTO GNN in v5)   │          │                            │
     │ LightAttention (k=9)       │          │ LightAttention             │
     │ FC 1096→128→1 (per residue)│          │ FC → per residue           │
     │ E = SUM  (v5: MEAN)        │          │ E = SUM  (v5: MEAN)        │
     └─────────────┬─────────────┘          └─────────────┬─────────────┘
              E_folded                               E_unfolded
                   └──────────────────┬─────────────────────┘
                                      ▼
                        dG_pred = E_unfolded − E_folded
                                      │
              ┌───────────────────────┴───────────────────────┐
              │ TRAIN: Huber(dG_pred,dG) + λ·ranking(dG)        │
              │        (+ v5 corr loss)  ·  minibatch = 16      │
              │ VALIDATE: ddG = dG − dG(WT_row0) ; report PCC   │
              └────────────────────────────────────────────────┘
```

---

## Slide 12 — Summary Numbers & Honest Confidence

**Verified architecture facts (defend these):**
- Model predicts **dG**; **ddG derived at validation** = `dG_mut − dG_wt`.
- Node vector = **16 dist + 32 bonded + 1024 PLM + 20 one-hot = 1092** (1604 with ESM-IF1).
- Two branches: **GCN** on sequential `(i,i+1)` edges + **GATv2** fully-connected OR **k-NN (k=30, 12 Å,
  CA = atom index 1)**; concat → InstanceNorm → **Light Attention (Conv1d kernel=9, feature·softmax)** →
  FC `1096→128→1`.
- Energy readout = **extensive SUM** (`E = torch.sum(Fh, dim=(1,2))`, hydro_net.py:525) — **length-biased**;
  v5 replaces it with a per-residue **MEAN** (intensive).
- one-hot uses **20** AA; unfolded keeps **i±1** (tridiagonal); **mini_batch_size = 16** (64 → OOM);
  structure **fixed** to WT per mutation (only one-hot + PLM change).
- Training: Adam, cosine LR **1e-4 → 1e-6**, **Huber + pairwise ranking** on dG, **15 epochs**,
  PNAS-filtered single mutations, dG clamp **[-1, 5]**, ~**340 train / 28 test**, baseline **PCC 0.5259**.

**The 5 v5 methods (one flag each):**
`length_norm` · `serial_fusion` · correlation loss · `dual_esmif` · seed ensemble.

**Honest confidence statement (say this to your professor):**
> "The code changes are verified against the real pipeline and the math (length_norm, correlation losses)
> is unit-checked. The **PCC-gain figures are hypotheses** drawn from the literature and analogous models
> (ThermoMPNN, ESM-GearNet), **not measured results**. Each improvement is flag-gated and added one at a
> time on a ladder, so every claimed gain is confirmed — or rejected — by its own controlled ablation.
> The realistic target is **PCC ≈ 0.58–0.62 without inverse-folding pretraining**, and **0.63–0.72 with a
> strong structural/inverse-folding prior** such as dual_esmif."

**Priority hierarchy (what matters most):**
```
   Inverse-folding prior (dual_esmif)  >>  more/better data  >>  architecture (serial_fusion)  >  tricks (length_norm, corr loss, ensemble)
```
```
