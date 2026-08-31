# DeepPEF v5 — Visual Representations of the Model

> A pictures-first walkthrough of how a protein becomes a graph, how the graph
> becomes an energy, and how that energy becomes a ddG prediction.
>
> Every diagram here is traced directly from the code:
> - `DeepPEF_v5/training/train_utils.py` — `get_graph`, `get_unfolded_graph`,
>   `get_dist_matrix`, `get_bonded_features`, `zero_except_udiagonal`
> - `DeepPEF_v5/model/hydro_net_v5.py` — `PEM.forward`, `forward_gcn`,
>   `forward_gat`, `get_edge_index`, `get_energy`, `GCN`, `GAT`, `LightAttention`
> - `DeepPEF_v5/training/pnas_train_v5.py` — `Trainer.get_deltaG`
>
> Notation used throughout:
> - `L` = number of residues (protein length, `n_nodes` / `N` in code)
> - `A` = number of backbone atoms per residue = **4** (N, CA, C, CB)
> - `E` = PLM embedding width (ProtT5 = **1024**; dual_esmif = 1536, etc.)
> - `M` = mini-batch size = number of mutations processed together (argparse default 64; **always pass `--mini_batch_size 16`** — the proven value that avoids OOM on 8 GB GPUs)

---

## Table of Contents
1. [From 3D structure to graph](#1-from-3d-structure-to-graph)
2. [The node feature vector](#2-the-node-feature-vector)
3. [Folded vs unfolded graph](#3-folded-vs-unfolded-graph)
4. [Message passing](#4-message-passing)
5. [Two parallel GNN towers (GCN + GAT)](#5-two-parallel-gnn-towers-gcn--gat)
6. [How a mutation flows through](#6-how-a-mutation-flows-through)
7. [Shapes cheat-sheet](#7-shapes-cheat-sheet)

---

## 1. From 3D structure to graph

### 1a. The protein backbone

A protein is a chain of residues. Each residue contributes **4 backbone atoms**:
`N`, `CA`, `C`, and `CB` (the `CB` for glycine is synthesized geometrically by
`add_cb()` in `train_utils.py`). We draw three residues of the chain:

```
        residue i-1            residue i              residue i+1
      ┌───────────────┐    ┌───────────────┐     ┌───────────────┐
      │  N   CA   C    │    │  N   CA   C    │     │  N   CA   C    │
      │  •───•────•    │────│  •───•────•    │─────│  •───•────•    │
      │       \        │    │       \        │     │       \        │
      │        • CB    │    │        • CB    │     │        • CB    │
      └───────────────┘    └───────────────┘     └───────────────┘
             node i-1            node i                node i+1

  In the GRAPH:  one residue  =  one NODE.
  The 4 atoms live "inside" the node and are used to build its features.
```

Coordinates arrive as a tensor `X` of shape `[L, 4, 3]`
(L residues × 4 atoms × xyz). See `get_dist_matrix(Xd)`:

```python
N_residu, N_atoms, coords_size = Xd.shape        # L, 4, 3
```

### 1b. Building the raw distance matrix `D[L, L, 16]`

For **every pair of residues** `(i, j)` we measure **all 4×4 = 16 atom-to-atom
distances**. This is the heart of `get_dist_matrix`:

```python
Xd = Xd.reshape(N_residu*N_atoms, coords_size)   # [L*4, 3]  -> flatten atoms
D  = torch.cdist(Xd, Xd, p=2)                     # [L*4, L*4] Euclidean distances
D  = D.reshape(N_residu, N_atoms, N_residu, N_atoms)   # [L,4,L,4]
D  = torch.swapaxes(D, 1, 2)                      # [L,L,4,4]
D  = D.reshape(N_residu, N_residu, N_atoms*N_atoms)    # [L,L,16]
```

Picture the 16 numbers stored for one residue pair `(i, j)` as a flattened 4×4
block of atom-atom distances:

```
        D[i, j]  =  16 distances between residue i's atoms and residue j's atoms

                        residue j atoms
                     N     CA     C     CB
                  ┌──────────────────────────┐
   residue     N  │ d00   d01   d02   d03     │
   i           CA │ d10   d11   d12   d13     │   flatten (row-major)
   atoms       C  │ d20   d21   d22   d23     │ ───────────────────►  [d00 d01 ... d33]
               CB │ d30   d31   d32   d33     │                        16 values
                  └──────────────────────────┘
```

So the full raw tensor is:

```
        j = 0     1     2    ...   L-1
      ┌─────┬─────┬─────┬─────┬─────┐
 i=0  │[16] │[16] │[16] │ ... │[16] │      Each cell [16] is the 4x4 atom-atom
      ├─────┼─────┼─────┼─────┼─────┤      distance block for that residue pair.
 i=1  │[16] │[16] │[16] │ ... │[16] │
      ├─────┼─────┼─────┼─────┼─────┤      Shape overall: [L, L, 16]
 ...  │ ... │ ... │ ... │ ... │ ... │
      ├─────┼─────┼─────┼─────┼─────┤      Diagonal cells (i==j) are a residue's
 L-1  │[16] │[16] │[16] │ ... │[16] │      OWN internal atom geometry.
      └─────┴─────┴─────┴─────┴─────┘
```

### 1c. Gaussian kernel: turn distance into "contact strength"

Raw distances are unbounded and hard for a network to use. We convert each
distance `d` into a soft **contact weight** in `(0, 1]` using a Gaussian.
In `get_graph`:

```python
D = torch.relu(torch.exp(gaussian_coef * D**2))   # gaussian_coef ≈ -0.08
```

The kernel `exp(coef * d²)` (with `coef < 0`) is 1 when atoms touch and decays
toward 0 as they move apart. The `relu` just clamps away tiny negatives.

```
  weight = exp(coef * d^2),  coef = -0.08
  1.0 ┤●
      │ ●
      │   ●
  0.5 ┤      ●
      │          ●
      │              ●  ●
  0.0 ┤                     ●  ●  ●  ●  ●  ●
      └───────────────────────────────────────►  distance d (Å)
      0    2    4    6    8   10   12   14
      close contact  →  strong weight   |   far apart → ~0
```

Masked residues (padding / missing) are zeroed out **before** any summing:

```python
mask_index = torch.where(mask == 0)
D[mask_index[0], :, :] = 0     # zero out rows for masked residues
D[:, mask_index[0], :] = 0     # zero out cols for masked residues
```

### 1d. Sum over neighbors → per-residue distance feature `D[L, 16]`

Now collapse the neighbor axis. For each residue `i` we **sum its 16-vector
contact weights over all partners `j`**, giving a 16-dim "how contacted is each
of my atom-channels" summary, then L2-normalize:

```python
D = D.sum(dim=1)             # [L, L, 16] -> [L, 16]   (sum over neighbors j)
D = F.normalize(D, p=2, dim=0)
```

```
   [L, L, 16]   --- sum over j (neighbors) --->   [L, 16]
   full pairwise contacts                          per-residue contact profile

   Residue i's 16-vector = Σ_j (Gaussian contact of i's atoms with j's atoms)
   Buried, densely-packed residues -> large values.  Exposed loop residues -> small.
```

This 16-dim block is the **`D` (distance) part** of the node feature. It is the
model's compressed picture of 3D packing around each residue.

---

## 2. The node feature vector

After `get_graph` finishes, every node (residue) carries one long feature
vector. The concatenation order is fixed in code:

```python
Fh = torch.cat([D, Fb, emb, one_hot], dim=1)     # train_utils.get_graph
#                16   32   E     20
```

### 2a. The labeled layout `[16 | 32 | E | 20]`

```
  ONE NODE'S FEATURE VECTOR  (width = 16 + 32 + E + 20 = 68 + E)
                                                    (E=1024 → width 1092)

  index:  0        16              48                48+E          48+E+20
          ┌────────┬───────────────┬─────────────────┬─────────────┐
          │   D    │      Fb       │      emb         │  one_hot    │
          │ dist   │   bonded      │   ProtT5 PLM     │ amino acid  │
          │  16    │      32       │    E (1024)      │     20      │
          └────────┴───────────────┴─────────────────┴─────────────┘
             ▲          ▲                  ▲                ▲
             │          │                  │                │
   summed contact   distance to      per-residue        which of the
   profile over     the previous &   language-model     20 amino acids
   all neighbors    next residue     embedding of the   this residue is
   (§1d)            (§2b)            (mutant) sequence   (one-hot)
```

### 2b. Where the 32 "bonded" features come from

`get_bonded_features(D)` pulls out the distance blocks to the **immediate
sequence neighbors** — residue `i` to `i+1` (above diagonal) and `i` to `i-1`
(below diagonal) — and concatenates them (16 + 16 = 32):

```python
f1, f2 = D[n_range[:-1], n_range[1:]], D[n_range[1:], n_range[:-1]]   # i->i+1, i->i-1
# pad each with a zero row at the appropriate end (chain termini), then:
Fb = torch.cat([f1, f2], dim=-1)     # [L, 32]
```

```
   Fb[i]  =  [  16 dists to NEXT residue (i+1)  |  16 dists to PREV residue (i-1)  ]
                       forward bond                        backward bond
   (first residue has no prev → zeros; last residue has no next → zeros)
```

These encode the **local chain geometry** (the covalent backbone), separate from
the long-range contacts captured in the summed `D`.

### 2c. How `PEM.forward` slices the vector back apart

The dataset packs `[D | Fb | emb | one_hot]`. Inside the model, `forward()`
reshapes to `[B*N, features]` and re-splits it using index constants
(`non_bonded_index = 16`, `one_hot_index = -20`,
`llm_index = -(emb_input_dim + 20)`):

```python
x_dist         = x[:, :self.non_bonded_index]                 # [:, :16]      -> D
x_bonded       = x[:, self.non_bonded_index:self.non_bonded_index*2]  # [:, 16:32] -> first 16 of Fb
x_emb_features = x[:, self.llm_index:self.one_hot_index]      # [:, -(E+20):-20] -> emb (E)
x_onehot       = x[:, self.one_hot_index:]                    # [:, -20:]     -> one_hot
```

```
  full vector  [ D:16 | Fb:32 | emb:E | one_hot:20 ]
                  │       │        │         │
   x_dist ────────┘       │        │         │      x[:, :16]
   x_bonded ──────────────┘        │         │      x[:, 16:32]   (GCN uses only
                (NOTE: the model reads only the                    the first 16 of
                 first 16 of the 32 bonded cols —                  the 32 bonded)
                 the "bond to previous" block)
   x_emb_features ─────────────────┘         │      x[:, -(E+20):-20]
   x_onehot ─────────────────────────────────┘      x[:, -20:]
```

The two GNN input tensors are then assembled (default config, no learned-AA /
no projection / no serial fusion):

```python
x_gcn = torch.cat((x_dist, x_bonded, x_aa), dim=-1)   # 16 + 16 + 20 = 52
x_gat = torch.cat((x_dist,           x_aa), dim=-1)   # 16 +      20 = 36
```

The raw `emb` (E-dim) is **held aside** and concatenated back **after** the GNN
towers (see §5), unless `--serial_fusion` / `--emb_projection` fold it in early.

---

## 3. Folded vs unfolded graph

The whole model is a comparison of **two graphs built from the same
coordinates**. The only difference is which contacts are allowed to exist.

### 3a. Folded: full contact matrix

`get_graph` keeps **all** pairwise contacts (the full `[L, L, 16]` before the
sum). A compact protein has many off-diagonal contacts:

```
   FOLDED contact map (● = contact kept)          j →
                    0   1   2   3   4   5   6
                 0  ●   ●   ●   .   ●   .   ●
                 1  ●   ●   ●   ●   .   ●   .
             i   2  ●   ●   ●   ●   ●   .   ●
             ↓   3  .   ●   ●   ●   ●   ●   .
                 4  ●   .   ●   ●   ●   ●   ●
                 5  .   ●   .   ●   ●   ●   ●
                 6  ●   .   ●   .   ●   ●   ●
                    long-range contacts present  →  3D structure "seen"
```

### 3b. Unfolded: tridiagonal only

`get_unfolded_graph` calls `zero_except_udiagonal(D)`, which **erases every
contact except the main diagonal and its two immediate off-diagonals**:

```python
def zero_except_udiagonal(D):
    f1, f2 = D[n_range[:-1], n_range[1:]], D[n_range[1:], n_range[:-1]]
    diag   = D[n_range, n_range]
    D[:, :, :] = 0                          # wipe everything
    D[n_range[:-1], n_range[1:]] = f1       # restore i -> i+1
    D[n_range[1:], n_range[:-1]] = f2       # restore i -> i-1
    D[n_range, n_range]          = diag     # restore i -> i (self)
    return D
```

```
   UNFOLDED contact map (only self + sequential neighbors)   j →
                    0   1   2   3   4   5   6
                 0  ●   ●   .   .   .   .   .
                 1  ●   ●   ●   .   .   .   .
             i   2  .   ●   ●   ●   .   .   .
             ↓   3  .   .   ●   ●   ●   .   .
                 4  .   .   .   ●   ●   ●   .
                 5  .   .   .   .   ●   ●   ●
                 6  .   .   .   .   .   ●   ●
                    ONLY the linear chain survives → "an extended coil"
```

### 3c. Why the difference is the signal

Both graphs run through the **same network weights** and produce a scalar
energy. `get_deltaG` (in `pnas_train_v5.py`) computes:

```python
return unfolded_energy - folded_energy, unfolded_energy, folded_energy, denoise_loss
#      dG  =  E_unfolded  -  E_folded
# The 4th value (denoise_loss) is None unless Lever E is on (train only); see §Lever E.
```

```
   E_folded      = energy when 3D contacts are visible   (stable, low)
   E_unfolded    = energy of the bare chain              (reference)
   dG = E_unfolded - E_folded  ≈  "reward" the model assigns to folding

   A residue network that packs into many good contacts lowers E_folded,
   which RAISES dG  →  the model learns that good 3D packing = more stable.
```

The unfolded graph is the model's **built-in reference state**: it strips away
everything except the covalent chain so that `dG` isolates the contribution of
3D structure.

---

## 4. Message passing

A GNN layer updates each node by mixing in information from its neighbors. Below,
node `i` (residue 3) aggregates from the nodes connected to it, and this repeats
layer by layer so information travels further along the graph each round.

```
   LAYER 0 (input features)          LAYER 1 (after 1 hop)        LAYER 2 (after 2 hops)

        (n1)                              (n1)                         (n1)
          \                                 \  \                         \
   (n0)—(  i  )—(n2)   ── aggregate ──►  (n0)—( i' )—(n2)  ──►   (n0)—( i'' )—(n2)
          /  (edges from graph)             /  /                         /
        (n5)                              (n5)                         (n5)

   i'  = UPDATE( i ,  AGGREGATE{ neighbor features } )
   Each layer widens node i's "receptive field" by one more hop.
```

Concretely, one node's update inside a GNN layer looks like:

```
                         neighbor messages
             ┌───────────────┴───────────────┐
   feat(n0)  feat(n1)  feat(n2)  feat(n5) ... (only true graph neighbors)
      │         │         │         │
      └────► aggregate (GCN: normalized mean  |  GAT: attention-weighted sum) ──┐
                                                                                 ▼
   feat(i) ───────────────────────────────────────────► combine ─────► feat'(i)
                                     (linear + nonlinearity, then residual add)
```

- **GCN** aggregates over **sequential** edges only (`i → i+1`), so it refines
  the local chain signal.
- **GAT** aggregates over the **distance / fully-connected** edges and learns
  *how much* to weight each neighbor via attention — this is where long-range 3D
  contacts get mixed in.

Both towers use a **residual connection** (`x = h1 + identity`) so the update is
a correction to the input rather than a full replacement (see `forward_gcn` /
`forward_gat`).

---

## 5. Two parallel GNN towers (GCN + GAT)

The model runs **two independent towers in parallel**, concatenates their
outputs, re-attaches the raw PLM embedding, applies light attention, and finally
maps to a per-residue energy. Here is the full flow of `PEM.forward` (default
config: ProtT5 E=1024, one-hot AA, light attention on, k-NN GAT via
`gat_cutoff`).

```
                      NODE FEATURES  x = [D:16 | Fb:32 | emb:1024 | onehot:20]
                                            │
              ┌─────────────────────────────┼───────────────────────────┐
              │  split (PEM.forward)         │                           │
              ▼                              ▼                           ▼
        x_dist(16)+x_bonded(16)        x_dist(16)+x_aa(20)          x_emb_features(1024)
        +x_aa(20)  = 52 dims           = 36 dims                    (held aside)
              │                              │                           │
   ┌──────────▼──────────┐        ┌──────────▼──────────┐               │
   │   GCN TOWER          │        │   GAT TOWER          │              │
   │   forward_gcn(...)   │        │   forward_gat(...)   │              │
   │                      │        │                      │             │
   │ fc1_gcn: 52→64       │        │ fc1_gat: 36→64       │             │
   │   ReLU               │        │   ReLU               │             │
   │ fc2_gcn: 64→36       │        │ fc2_gat: 64→36       │             │
   │ inst_norm1           │        │ inst_norm1           │             │
   │ ┌── GCN layer × L ──┐│        │ ┌── GAT layer × L ──┐│             │
   │ │ GCNConv 36→64      ││        │ │ GATv2 36→64×8 heads││            │
   │ │ ReLU               ││        │ │ ELU                ││            │
   │ │ GCNConv 64→36      ││        │ │ GATv2 512→36 head=1││            │
   │ │ inst_norm          ││        │ │ inst_norm          ││            │
   │ │ + residual (id)    ││        │ │ + residual (idty)  ││            │
   │ └────────────────────┘│        │ └────────────────────┘│           │
   │  edges = SEQUENTIAL    │        │  edges = DIST / FULL   │           │
   │  (i, i+1)              │        │  (CA cutoff or all-vs-all)         │
   └──────────┬─────────────┘        └──────────┬───────────┘           │
              │  x1: [B*N, 36]                   │  x2: [B*N, 36]        │
              └──────────────┬───────────────────┘                       │
                             ▼                                            │
                 concat  x = [x1 | x2]   (dim = 36+36 = 72)               │
                             │                                            │
                     reshape [B, N, 72]                                   │
                     inst_norm2                                           │
                     reshape [B*N, 72]                                    │
                             │                                            │
                     concat raw emb ◄─────────────────────────────────────┘
                     x = [x | x_emb_features]   (dim = 72 + 1024 = 1096)
                             │
                     ┌───────▼────────┐
                     │ LightAttention │   (Conv1d feature × softmax(Conv1d attn))
                     │  over residues │    fc_in_dim = 72 + 1024 = 1096
                     └───────┬────────┘
                             │
                     fc1: 1096 → 128
                       ReLU
                     fc2: 128  → 1                (per-residue scalar)
                             │
                     reshape [B, N, 1]
                             │
                     get_energy: E = Σ over residues   → [B]
                     (÷ residue count if --length_norm)
```

Key code anchors:

```python
x1 = self.forward_gcn(x_gcn, edge_index_gcn, B, N)   # sequential edges
x2 = self.forward_gat(x_gat, edge_index_gat, B, N)   # distance / full edges
x  = torch.cat((x1, x2), dim=-1)                     # merge towers
x  = self.inst_norm2(x.reshape(B, N, -1)).reshape(B*N, -1)
if self.emb_projector is None:
    x = torch.cat((x, x_emb_features), dim=-1)       # re-attach raw PLM
# light attention (if enabled) ...
x = F.relu(self.fc1(x))
x = self.fc2(x).reshape(B, N, 1)                     # per-residue energy
return self.get_energy(x)                            # sum -> [B]
```

### Edge construction (`get_edge_index`)

```
   GCN edges (sequential):        GAT edges (fully connected OR CA-distance cutoff):

     0 → 1 → 2 → 3 → ... → L-1      every i ↔ j with i≠j        (gat_cutoff = None)
                                    OR only pairs with          (gat_cutoff set,
     a simple chain                 ‖CA_i − CA_j‖ < cutoff       via ca_coords)
```

`get_deltaG` sets `gat_cutoff` on the fly from a k-NN radius when `--use_knn_gat`
is on, so the GAT sees a **spatially local** neighborhood instead of all-vs-all.

---

## 6. How a mutation flows through

A point mutation (e.g. `V23A` — valine → alanine at position 23) changes the
**sequence**, not the crystal structure the model is given. So per mutation:

```
                        WILD-TYPE  vs  MUTANT  (position 23: V → A)

  coordinates X [L,4,3]  ──────────  SAME  ──────────►  (structure is reused)

  one_hot  [L,20]        row 23 = V(1-hot)  ─────────►  row 23 = A(1-hot)  ◄─ CHANGES
  emb (ProtT5) [L,E]     PLM of WT seq      ─────────►  PLM of MUTANT seq  ◄─ CHANGES
                                                            (the KEY signal)
```

In `get_deltaG`, one protein's mutations are stacked into a mini-batch of size
`M`. For **each** mutation the code builds BOTH a folded and an unfolded graph
from the shared coordinates but with that mutation's own `one_hot` + `prott5`:

```python
folded   = [get_graph(coords, one_hot[j], prott5[j], mask) for j in range(M)]
unfolded = [get_unfolded_graph(coords, one_hot[j], prott5[j], mask) for j in range(M)]
all_graph = torch.cat([folded, unfolded], dim=0)      # [2M, L, 68+E]
energy    = self.model(all_graph, ca_coords=ca_coords)  # [2M]
folded_energy   = energy[:M]
unfolded_energy = energy[M:]
dG = unfolded_energy - folded_energy                  # [M]  one dG per mutation
```

The pipeline per mutation:

```
   mutation j
      │
      ├── build folded graph   (coords + mut one_hot + mut ProtT5, ALL contacts)  ─┐
      └── build unfolded graph (coords + mut one_hot + mut ProtT5, tridiagonal)   ─┤
                                                                                    ▼
                                     PEM.forward (shared weights)
                                                                                    │
                          E_folded[j]        E_unfolded[j]  ◄──────────────────────┘
                                    \        /
                                     dG[j] = E_unfolded[j] − E_folded[j]
```

Then **ddG** is the shift relative to the wild-type entry (row 0 of the batch —
`indexes.insert(0, 0)` guarantees the WT is present as the first sample):

```
   ddG(mutant)  =  dG_mutant  −  dG_wildtype

   ddG > 0  →  mutation DESTABILIZES  (unfolding is easier)
   ddG < 0  →  mutation STABILIZES
```

Because coordinates are identical for WT and mutant, the **entire ddG signal
comes from how the mutant's one-hot + ProtT5 embedding reshape the per-residue
energies** through message passing. This is exactly why (per project memory)
GNN-SM failed at PCC 0.42: it lacked the mutant-specific ProtT5 embedding and
had to rely on the 20-dim one-hot alone — too weak a signal. The energy-diff
design here reaches PCC ~0.5259.

Training then pushes predicted `dG` toward the measured `deltaG` label using an
L1/Huber loss (+ optional ranking, energy regularization, and the v5
correlation loss), and grades on Pearson/Spearman correlation.

---

## 7. Shapes cheat-sheet

Protein of length `L`, `A=4` atoms/residue, PLM width `E` (1024 for ProtT5),
mini-batch `M` mutations. In the model, `B = 2M` (folded + unfolded stacked) and
`N = L`.

### 7a. Graph construction (`train_utils.py`, per single graph)

| Tensor | Shape | Meaning |
|---|---|---|
| `X` (coords) | `[L, 4, 3]` | Backbone atom xyz (N, CA, C, CB) |
| `Xd` flattened | `[L*4, 3]` | Atoms flattened for `cdist` |
| `cdist(Xd,Xd)` | `[L*4, L*4]` | All atom-atom Euclidean distances |
| `D` reshaped | `[L, L, 16]` | 16 atom-atom distances per residue pair |
| `D` after Gaussian | `[L, L, 16]` | Soft contact weights `exp(coef·d²)` |
| `Fb` (bonded) | `[L, 32]` | Dist blocks to i+1 and i−1 (16+16) |
| `D` summed | `[L, 16]` | Per-residue contact profile (Σ over j) |
| `emb` (ProtT5) | `[L, E]` | Per-residue PLM embedding |
| `one_hot` | `[L, 20]` | Amino-acid identity |
| `Fh` = `get_graph(...)` | `[L, 68+E]` | Final node features `[16│32│E│20]` |

### 7b. Mini-batch assembly (`get_deltaG`)

| Tensor | Shape | Meaning |
|---|---|---|
| `folded_graph_minibatch` | `[M, L, 68+E]` | One folded graph per mutation |
| `unfolded_graph_minibatch` | `[M, L, 68+E]` | One unfolded graph per mutation |
| `all_graph_minibatch` | `[2M, L, 68+E]` | Folded + unfolded stacked → model input |
| `ca_coords` (if k-NN) | `[2M, L, 3]` | CA atoms for distance-cutoff GAT edges |

### 7c. Forward pass through `PEM` (`B = 2M`, `N = L`)

| Tensor / step | Shape | Meaning |
|---|---|---|
| `x` (input) | `[B, N, 68+E]` | Batched node features |
| `x` reshaped | `[B*N, 68+E]` | Flatten nodes for slicing |
| `x_dist` | `[B*N, 16]` | Summed distance block `D` |
| `x_bonded` | `[B*N, 16]` | First 16 of bonded `Fb` (bond-to-prev) |
| `x_emb_features` | `[B*N, E]` | Raw PLM embedding (held aside) |
| `x_onehot` / `x_aa` | `[B*N, 20]` | AA identity (or learned emb if enabled) |
| `x_gcn` | `[B*N, 52]` | `[dist│bonded│aa]` GCN input |
| `x_gat` | `[B*N, 36]` | `[dist│aa]` GAT input |
| `edge_index_gcn` | `[2, B·(N−1)]` | Sequential edges (i, i+1) |
| `edge_index_gat` | `[2, ~B·N·(N−1)]` | Full or distance-cutoff edges |
| `x1` (GCN out) | `[B*N, 36]` | GCN tower output (`fc2_gcn` maps 64→36; + residual) |
| `x2` (GAT out) | `[B*N, 36]` | GAT tower output (+ residual) |
| `concat(x1,x2)` | `[B*N, 72]` | Merged towers |
| after `inst_norm2` | `[B, N, 72]` → `[B*N, 72]` | Instance-normalized |
| + raw emb | `[B*N, 1096]` | `72 + 1024` (= `fc_in_dim`) |
| after LightAttention | `[B*N, 1096]` | Conv1d feature × softmax(attn) |
| `fc1` out | `[B*N, 128]` | Hidden |
| `fc2` out | `[B*N, 1]` | Per-residue scalar |
| reshaped | `[B, N, 1]` | Per-residue energies |
| `get_energy` → `E` | `[B]` = `[2M]` | Protein energy (Σ residues; ÷L if length_norm) |

### 7d. Energy → ddG (`get_deltaG` / `Trainer`)

| Quantity | Shape | Meaning |
|---|---|---|
| `folded_energy` | `[M]` | `energy[:M]` |
| `unfolded_energy` | `[M]` | `energy[M:]` |
| `dG` = `unfolded − folded` | `[M]` | Predicted folding free energy per mutation |
| `delta_g` (label) | `[M]` | Measured Megascale/PNAS dG |
| `ddG_true` = `dG − dG[0]` | `[M]` | Experimental ddG (relative to WT) |
| `ddG_pred` | `[M]` | Predicted ddG (relative to WT) |
| `pc_corr` (PCC) | scalar | Pearson correlation — the grading metric (~0.5259) |

---

## One-paragraph recap

**Coordinates → distances (16 per residue pair) → Gaussian contact weights →
per-residue features `[D:16 | Fb:32 | emb:E | one_hot:20]`.** Build two graphs
from identical coordinates — a **folded** one with all contacts and an
**unfolded** one with only the sequential chain. Push both through a shared
**GCN (sequential) + GAT (spatial) parallel network**, re-attach the PLM
embedding, apply light attention, and sum per-residue outputs to a scalar
**energy**. `dG = E_unfolded − E_folded`; `ddG = dG_mut − dG_wt`. A mutation
changes only the one-hot and the ProtT5 embedding, and that change — after
message passing — is the entire ddG signal the model learns to predict.
