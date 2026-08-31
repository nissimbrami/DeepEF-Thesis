# DeepPEF_v5 — The Six Levers (A–F): A Teaching Guide

**A beginner-friendly walkthrough of the six flag-gated proposal levers that turn the proven
DeepPEF dG pipeline into an experiment platform. Every lever defaults to OFF, and OFF reproduces
the baseline bit-for-bit.**

> How to read this document: Part 0 gives the whole-model intuition. Parts A–F each explain one
> lever with (a) the physics, (b) the tensor/shape change, (c) an ASCII worked example, (d) the
> exact CLI flag, and (e) why the default reproduces the baseline. Part 7 explains why the levers
> compose cleanly. Part 8 is a self-test.
>
> **Honesty note:** this is a teaching document about *what the code does* and *why the physics
> motivates it*. It does NOT claim measured accuracy gains — those come only from running the
> ablation ladder. Treat every "should help" as a hypothesis to be tested.

---

## Part 0 — The Whole Model in One Paragraph

A protein is a chain of amino-acid residues that folds into a 3D shape. It lives in a tug-of-war:

```
        FOLDED  <====>  UNFOLDED
       (working)       (random coil)
```

DeepPEF learns a *proxy energy* `E` for a protein in a given shape, then computes the folding free
energy as a **difference of two states of the SAME sequence**:

```
   dG = E_unfolded − E_folded
```

We build a **graph** where every residue is a node. Node features encode the local geometry
(pairwise atom distances passed through a Gaussian "contact strength" kernel), the chemistry
(amino-acid identity), and a **protein-language-model embedding** (ProtT5) that captures sequence
context. Two graph neural networks run in parallel — a **GCN** on sequential backbone edges (local
context) and a **GATv2** on spatial-neighbor edges (long-range 3D contacts). Their outputs feed a
small head that emits **one energy scalar per residue**, and the total energy is the **sum** over
residues. The FOLDED graph keeps all 3D contacts; the UNFOLDED graph keeps only the linear
backbone (each residue sees just itself and its ±1 neighbors). So `dG` literally measures the
**energetic reward for forming real 3D contacts** versus a bare chain. A destabilizing mutation
should shrink that reward. For mutations the structure is held fixed (WT coordinates); only the
amino-acid one-hot and the ProtT5 embedding change, and `ddG = dG_mutant − dG_wildtype`.

**The node-feature layout you must memorize** (per residue, left to right):

```
   [ D (16 distance) | Fb (32 bonded) | emb (E) | one_hot (20) ]
     └── geometry ──┘   └─ neighbors ┘  └─ chemistry / identity ┘
```

`D=16` because each residue has **4 backbone atoms** (N, CA, C, CB) and there are `4×4=16`
inter-atomic distance channels between two residues. `Fb=32` is the kernelized distance to the two
sequence neighbors i−1 and i+1 (two 16-vectors). `emb` is ProtT5 (`E=1024`, or `1536` if ESM-IF1 is
concatenated). `one_hot` is the 20 amino-acid identity.

**The dimension contract (why v5 never breaks):** the graph builder and the model read the SAME
numbers from one `CFG` object (`model/model_cfg_v5.py`). Flip a flag, `CFG` updates, and both the
tensor builder and the model's slice indices update together. Widths never drift.

The six levers below each nudge ONE part of this pipeline. All default OFF.

---

## Part A — Energy Decomposition (`--energy_terms K`)

### (a) Physics / biology intuition
A real folding free energy is a **sum of physically distinct contributions**: hydrogen bonds, van
der Waals packing, electrostatics, hydrophobic burial, backbone strain, and so on. The baseline
squashes all of that into a single learned scalar per residue. Lever A lets the head emit **K
separate per-residue "energy terms"** that are summed into the total. The model is free to
specialize each channel toward a different physical component, and — because the terms are stored —
you can inspect them for interpretability ("term 3 lit up near the buried core").

### (b) What changes in the tensors/shapes
Only the **final head width** changes.

```
   Baseline (K=1):   fc2: 128 → 1     per-residue energy  [B, L, 1]
   Lever A (K>1):    fc2: 128 → K     per-residue terms   [B, L, K]  → sum over K → [B, L, 1]
```

The total energy readout is unchanged in spirit: sum over residues (and over the K terms). The
GNN, the node features, the graph construction — untouched.

### (c) Worked mini-example (L = 3 residues, K = 3)
```
   per-residue, per-term energies  E_terms  [3 residues × 3 terms]:

              term1  term2  term3     row-sum (per-residue energy e_i)
   res1       0.20   0.50   0.10   →   0.80
   res2       0.30   0.70   0.20   →   1.20
   res3       0.10   0.30   0.10   →   0.50
                                        ────
   total energy E = 0.80 + 1.20 + 0.50 = 2.50

   Now set K=1 (baseline): the head directly emits the row-sum column [0.80, 1.20, 0.50],
   and E = 2.50 — IDENTICAL. K just factors each e_i into named pieces before summing.
```

### (d) Exact CLI flag
```
   --energy_terms K        # K=1 is baseline; K=3 or 5 to decompose
```

### (e) Why default = OFF reproduces baseline exactly
`energy_terms = 1` (default in `CFG`). With K=1 the head is `128 → 1`, producing one scalar per
residue, and `sum_terms` over a length-1 axis is the identity. The output tensor and the energy
math are byte-identical to the baseline single-scalar head.

---

## Part B — RBF Kernel Bank (`--rbf_centers M`)

### (a) Physics / biology intuition
The baseline turns a raw distance `D` (Å) into a contact strength with **one** Gaussian:
`exp(coef · D²)`. That is a single smooth bump — it can say "close = strong, far = weak" but cannot
distinguish a 3 Å hydrogen-bond contact from a 6 Å van-der-Waals contact except by magnitude. A
**bank of M radial basis functions (RBFs)** centered on a grid of distances acts like a **soft
histogram of contact distances**: one basis fires for ~2–4 Å, another for ~5–7 Å, etc. This is a
much richer distance encoding (the same trick SchNet/DimeNet use).

### (b) What changes in the tensors/shapes
**Dimension-preserving.** The distance block stays width 16. We expand each distance into M basis
responses, then **sum them back** to the original 16-wide channel so nothing downstream changes.

```
   Baseline:   D  --exp(coef·D²)-->  contact       (width 16 preserved)
   Lever B:    D  --[rbf_1 .. rbf_M]-->  sum_M  ->  contact   (width 16 preserved)
```

### (c) Worked mini-example (one distance, M = 4 centers on {2,5,8,12} Å)
```
   Suppose D = 4.5 Å. Each RBF is a Gaussian bump centered at c_m:
       rbf_m(D) = exp( -(D - c_m)^2 / (2σ^2) )

              center c_m     rbf_m(4.5)   ← how strongly this "distance bin" fires
   rbf_1        2.0            0.29        (a bit — 4.5 is near-ish 2)
   rbf_2        5.0            0.88        (strong — 4.5 ≈ 5)
   rbf_3        8.0            0.04
   rbf_4       12.0          ~0.00

   Instead of one number (single Gaussian), the distance is now a PROFILE across bins.
   To stay width-preserving, the M responses are recombined (summed/projected) back into
   the same 16-wide distance channel the rest of the network expects:

       [rbf_1 rbf_2 rbf_3 rbf_4]  --sum-->  single enriched contact value  (feeds the 16-vec)

   With M=0 the bank is skipped and we fall straight back to exp(coef·D²).
```

### (d) Exact CLI flag
```
   --rbf_centers M         # M=0 is baseline (single Gaussian); M=16 or 32 for a bank
   # (grid spans [CFG.rbf_min, CFG.rbf_max] = [0, 20] Å)
```

### (e) Why default = OFF reproduces baseline exactly
`rbf_centers = 0` (default). The code branches to the original single Gaussian
`exp(gaussian_coef · D²)` and never allocates the bank. Because the bank is **summed back to width
16**, even when ON it cannot change any downstream shape — it only changes the *values* inside the
existing 16 distance channels. OFF = original values.

---

## Part C — Burial / Solvation Feature (`--use_burial`)

### (a) Physics / biology intuition
Whether a residue is **buried in the hydrophobic core** or **exposed on the surface** hugely
affects how a mutation changes stability. Mutating a buried hydrophobic residue to a charged one is
often catastrophic; the same swap on the surface may be harmless. The baseline gives the model no
explicit "how buried am I" signal. Lever C adds a per-residue **CB neighbor-density** scalar: count
how many other residues' CB atoms sit within a radius (`burial_radius`, default 10 Å). High density
= buried; low = surface.

### (b) What changes in the tensors/shapes
One scalar is **inserted into the node vector BEFORE the embedding block**. The layout grows by
`burial_dim = 1`.

```
   Baseline:   [ D(16) | Fb(32) |            emb(E) | one_hot(20) ]
   Lever C:    [ D(16) | Fb(32) | burial(1) | emb(E) | one_hot(20) ]
                                  ↑ new width-1 slot, before emb
```

Because the layout is defined once in `CFG` (`burial_dim`), the model's slice indices for `emb` and
`one_hot` shift automatically — the dimension contract keeps builder and model in sync.

### (c) Worked mini-example (burial_radius = 10 Å, L = 5)
```
   Count CB neighbors within 10 Å of each residue's CB (excluding self):

   residue:   1    2    3    4    5
   #neighbors 8    9    2    3    7          ← raw counts
   burial     hi   hi   lo   lo   hi         ← interpretation (core vs surface)

   Node vector for residue 1 becomes:
     [ 16 dist ... | 32 bonded ... | 0.80(burial) | 1024 emb ... | 20 one_hot ]
   (the scalar is normalized before insertion)
```

### (d) Exact CLI flag
```
   --use_burial            # off by default; sets burial_dim=1 and inserts the scalar
```

### (e) Why default = OFF reproduces baseline exactly
`use_burial = False` and `burial_dim = 0` (defaults). With width 0 the node vector is
`[D | Fb | emb | one_hot]` — the exact baseline layout with no inserted slot — and no burial
computation runs. The model's `emb`/`one_hot` slice indices are computed from `CFG` widths, so with
`burial_dim=0` they land exactly where the baseline expects.

---

## Part D — Flory Unfolded Reference (`--flory_unfolded --flory_nu 0.5`)

### (a) Physics / biology intuition
The baseline "unfolded" state is crude: it zeros ALL contacts except the backbone tridiagonal band
(a residue sees only itself and i±1). That is a *stick-straight* chain, which is physically too
rigid. Real unfolded proteins are **random coils** whose average spatial separation between
residues i and j grows as a **power law of sequence separation**: `⟨r_ij⟩ ∝ |i − j|^ν`. This is
**Flory polymer theory** (ν ≈ 0.5 for an ideal chain, ~0.59 in good solvent). Lever D makes the
unfolded reference physically grounded instead of a bare line.

### (b) What changes in the tensors/shapes
**Value-only.** No shape changes at all — only the *distance values* used to build the UNFOLDED
graph change. The FOLDED graph is untouched.

```
   Baseline unfolded:  keep tridiagonal band only (diag + i±1), everything else zeroed.
   Lever D unfolded:   distances follow random-coil scaling |i-j|^nu for all pairs.
```

### (c) Worked mini-example (5 residues; contact matrix, X = kept)
```
   FOLDED (all contacts)      BASELINE UNFOLDED         FLORY UNFOLDED (|i-j|^0.5)
      1 2 3 4 5                  1 2 3 4 5                  effective distance grows
    1 X X X X X                1 X X . . .                 smoothly with |i-j|:
    2 X X X X X                2 X X X . .
    3 X X X X X                3 . X X X .                 |i-j|=1 -> r ∝ 1.00
    4 X X X X X                4 . . X X X                 |i-j|=2 -> r ∝ 1.41
    5 X X X X X                5 . . . X X                 |i-j|=3 -> r ∝ 1.73
                                                           |i-j|=4 -> r ∝ 2.00
   (baseline: hard cutoff)    (only ±1 survive)            (soft power-law separations)
```

The Flory distances then go through the same Gaussian/RBF contact kernel, so distant residues get
small-but-nonzero contact strength instead of a hard zero — a smoother, more realistic reference.

### (d) Exact CLI flag
```
   --flory_unfolded --flory_nu 0.5     # nu=0.5 ideal chain; ~0.59 for good-solvent coil
```

### (e) Why default = OFF reproduces baseline exactly
`flory_unfolded = False` (default). The unfolded builder calls the original
`zero_except_udiagonal` (keep diag + i±1, zero the rest). Since the FOLDED path never used Flory at
all, and the unfolded path falls back to the identical tridiagonal masking, `dG = E_unfolded −
E_folded` is computed from the exact baseline graphs.

---

## Part E — Decoys + Denoising Head (`--denoise_weight w`)

### (a) Physics / biology intuition
We want the learned energy surface to have a **minimum at the native structure** — that is what
makes it a real energy function. A clean way to teach that: **jitter the coordinates with random
noise, then ask an auxiliary head to predict the noise you added** (a denoising / score-matching
objective). To predict the displacement, the model must learn the local shape of the energy well,
which pushes the native structure toward being an energy minimum. This is an **auxiliary training
signal only** — it does not change the dG prediction path.

### (b) What changes in the tensors/shapes
A small **MLP head** is attached to the existing 128-dim pre-energy features. It predicts a
per-residue 3-vector (the injected coordinate noise). Nothing in the main energy head changes; the
denoising loss is a **weighted add-on to the TRAIN loss only** (never validation).

```
   Main path (unchanged):   128-dim feats → fc2 → per-residue energy → dG
   Aux path (Lever E):      128-dim feats → denoise_MLP → predicted noise [L, 3]
                            loss += denoise_weight * MSE(predicted_noise, injected_noise)
```

### (c) Worked mini-example
```
   1. Take native coords X_native.
   2. With prob denoise_prob (0.5), inject Gaussian noise of scale denoise_sigma (0.3 Å):
          X_noisy = X_native + eps,   eps ~ N(0, 0.3^2)
   3. Run the network on X_noisy; the aux head outputs eps_hat.
   4. Auxiliary loss:  L_denoise = || eps_hat - eps ||^2
   5. Total train loss:  L = L_dG  +  w * L_denoise        (w = denoise_weight)

   Learning to recover eps forces the features to encode "which way is downhill toward native",
   i.e. it sculpts an energy minimum at the true structure.
```

### (d) Exact CLI flag
```
   --denoise_weight w      # w=0 is baseline (head absent). w=0.1..1.0 to enable
   # (uses CFG.denoise_sigma=0.3, CFG.denoise_prob=0.5)
```

### (e) Why default = OFF reproduces baseline exactly
`denoise_weight = 0.0` (default). With `w=0` the denoising head is **not constructed**, no noise is
injected, and the `w * L_denoise` term is exactly 0, so it never enters the gradient. The train
loss, the forward pass, and validation are all identical to the baseline. (Note it is a
**train-only** signal by design — validation dG is never affected even when ON.)

---

## Part F — Edge Features + Extended Connectivity (`--gcn_span S`, `--use_edge_features`)

### (a) Physics / biology intuition
Two ideas bundled:
- **Extended GCN connectivity (`--gcn_span S`):** the baseline GCN only connects each residue to
  its immediate sequence neighbor `(i, i+1)`. Secondary structure has longer reach — an α-helix
  hydrogen-bonds i to i+4. Connecting sequence offsets `1..S` lets the local branch see more of the
  helical/strand context along the chain.
- **Edge features (`--use_edge_features`):** the baseline GATv2 attends using node features only; an
  edge is just "these two residues are connected." Real chemistry is **pairwise** (a
  Lys→Asp salt bridge depends on BOTH identities AND their distance). Lever F gives each GAT edge a
  **41-dim feature vector** `[onehot_src(20) | onehot_dst(20) | dist(1)]`, so attention can reason
  about the explicit chemistry and distance of each pair.

### (b) What changes in the tensors/shapes
- Connectivity: the GCN **edge_index** gains more edges (offsets `2..S` both directions). Node
  feature widths are unchanged.
- Edge features: a new **edge_attr** tensor of shape `[num_edges, 41]` is passed to GATv2. The node
  vectors are unchanged; this adds information **on the edges**, not the nodes.

```
   GCN edges (S=1, baseline):   i —— i+1                 (chain)
   GCN edges (S=3):             i —— i+1, i —— i+2, i —— i+3   (both directions)

   GAT edge_attr (per edge, when --use_edge_features):
       [ one_hot(src) 20 | one_hot(dst) 20 | distance 1 ]  = 41 dims
```

### (c) Worked mini-example (gcn_span S = 3, residues 1..5)
```
   Baseline GCN edges (S=1):
        1—2   2—3   3—4   4—5

   Extended GCN edges (S=3): add offset-2 and offset-3 links (both directions):
        1—2 1—3 1—4
        2—3 2—4 2—5
        3—4 3—5
        4—5
   → residue 1 now directly hears from 2,3,4 (helix-range context), not just 2.

   One GAT edge, residue 8 (Lys, K, idx 8) → residue 24 (Asp, D, idx 2), 4.2 Å apart:
        onehot_src = [..1 at pos 8..]   (20)
        onehot_dst = [..1 at pos 2..]   (20)
        dist       = [4.2]              (1)
        edge_attr  = concat  → length 41   ← "a K–D pair at 4.2 Å" (a likely salt bridge)
```

### (d) Exact CLI flags
```
   --gcn_span S            # S=1 is baseline (i,i+1 only); S=3 or 4 for helix reach
   --use_edge_features     # off by default; feeds 41-dim edge_attr to GATv2
```

### (e) Why default = OFF reproduces baseline exactly
`gcn_span = 1` and `use_edge_features = False` (defaults). With `S=1` the GCN edge_index is exactly
the sequential `(i, i+1)` chain the baseline builds. With edge features off, GATv2 is called with
no `edge_attr`, attending on node features only — the original behavior. Both node feature widths
stay the same, so nothing downstream shifts.

---

## Part 7 — Composability: Why the Levers Stack Cleanly

The levers were designed to be **orthogonal** so you can run a clean ablation ladder (one lever per
rung). Here is the reasoning per group:

### A and E are orthogonal to feature widths
```
   Lever A changes the HEAD OUTPUT width (128→K), then sums back to a scalar.
   Lever E adds an AUXILIARY HEAD off the 128-dim features + a train-only loss term.
   Neither touches the node-feature layout [D | Fb | (burial) | emb | one_hot],
   the graph edges, or the dG readout path. So they can't collide with B/C/D/F —
   or with each other. A reshapes the exit; E bolts on a side door.
```

### B is dimension-preserving
```
   Lever B expands each distance into M RBF responses, then SUMS them back to width 16.
       D (16) → [rbf_1..rbf_M per channel] → sum_M → D' (still 16)
   Because the output width is invariant, B changes only the VALUES in the distance
   block, never any shape. It composes with everything: no index in any other lever
   ever needs to know B is on.
```

### D is value-only
```
   Lever D only rewrites the UNFOLDED distance VALUES (Flory |i-j|^nu) before the
   contact kernel. No new dimensions, no head changes, no edge changes. It even
   composes with B: the Flory distances flow through the (possibly RBF) kernel just
   like real distances. Value-only = maximally composable.
```

### C (node width) and F (edges) compose because they live on different objects
```
   Lever C widens the NODE vector by +1 (burial), inserted before emb. Because the
   layout width lives in CFG (burial_dim), the model's emb/one_hot slice indices shift
   automatically — the "dimension contract."

   Lever F changes EDGES: more GCN edges (gcn_span) and a 41-dim edge_attr for GAT.
   Edges and edge_attr are a DIFFERENT tensor from the node matrix.

   Nodes (C) and edges (F) are separate inputs to a GNN, so widening nodes and
   enriching edges never interfere — the message-passing takes both independently.
   You can run C+F together and each still reduces to baseline when its own flag is off.
```

### The master rule
Because **all widths are read from one `CFG` object**, and each lever either (i) preserves width
(B, D), (ii) changes an isolated head (A, E), or (iii) changes a distinct tensor whose width is
declared in `CFG` (C on nodes, F on edges), the builder and model can never silently disagree. Turn
any subset of flags on; the OFF ones vanish and the ON ones slot in without reshaping their
neighbors.

---

## Part 8 — Learner Exercises (with Answers)

**Q1.** With `--energy_terms 5`, a residue's five term-energies are `[0.1, 0.2, 0.0, 0.3, 0.1]`.
What single per-residue energy does the baseline (`K=1`) head need to output to match, and what is
this residue's contribution to total energy?

<details><summary>Answer</summary>

Sum the terms: `0.1 + 0.2 + 0.0 + 0.3 + 0.1 = 0.7`. The `K=1` head would output `0.7` directly.
Its contribution to total energy is `0.7`. Lever A only *factors* the same scalar into named
pieces before summing — the total is unchanged, which is exactly why `K=1` is the baseline.
</details>

**Q2.** Lever B replaces one Gaussian with a bank of M RBFs. If the bank produced a 16×M tensor and
we fed *that* straight into the GNN, what contract would break — and how does the actual design
avoid it?

<details><summary>Answer</summary>

Feeding 16×M would change the distance-block width from 16 to 16·M, shifting every downstream slice
index (`emb`, `one_hot`) and breaking the dimension contract. The actual design **sums the M
responses back to width 16** (dimension-preserving), so only the *values* change, never the shape.
That is why B composes with every other lever.
</details>

**Q3.** Why is the burial scalar (Lever C) inserted *before* the embedding block rather than
appended at the very end of the node vector? What machinery keeps `emb` and `one_hot` correctly
located after the insertion?

<details><summary>Answer</summary>

Placement is a design choice: burial is a geometric/structural feature, so it sits next to the
other geometry (D, Fb) ahead of the sequence-derived `emb`/`one_hot`. What keeps everything aligned
is the **dimension contract**: `burial_dim` is declared in `CFG`, and the model computes its `emb`
and `one_hot` slice indices from the `CFG` widths. With `burial_dim=1` the indices shift by one; with
`burial_dim=0` (default) they land exactly at the baseline positions.
</details>

**Q4.** Lever E's denoising loss is added only to *training*, never validation. Why is that the
correct design, and would turning it on change the reported dG/ddG PCC purely by existing?

<details><summary>Answer</summary>

Validation must measure the quantity we actually care about (dG / derived ddG); mixing in a
denoising term would pollute the metric and make runs incomparable. Denoising is an **auxiliary
objective** that shapes the energy well during learning; at inference we only run the main energy
head. Turning it on changes the *learned weights* (and thus predictions), but it never adds a term
to the validation computation — the dG/ddG path is identical whether E is on or off.
</details>

**Q5.** You want to test "helix-range local context AND explicit pairwise chemistry" in one run.
Which flags do you set, which tensor does each modify, and why don't they collide?

<details><summary>Answer</summary>

Set `--gcn_span 4` (helix i→i+4 reach) and `--use_edge_features` (41-dim `[onehot_src|onehot_dst|
dist]` on GAT edges). `--gcn_span` adds edges to the **GCN edge_index**; `--use_edge_features` adds
an **edge_attr** tensor to GATv2. Both live on the *edge* side of the graph, separate from the node
feature matrix, and separate from each other's branch. Neither changes node widths, so they compose
freely and each still collapses to baseline when its own flag is off.
</details>

**Q6 (bonus).** The Flory reference (D) uses `|i-j|^ν`. If you set `ν = 0`, what unfolded geometry
do you get, and why is that *not* the same as the baseline tridiagonal unfolded state?

<details><summary>Answer</summary>

With `ν = 0`, `|i-j|^0 = 1` for all pairs — every pair sits at the *same* effective separation, i.e.
a collapsed/uniform blob, not a straight chain. The baseline unfolded state instead **keeps only
the ±1 backbone contacts and hard-zeros everything else**. So `ν=0` still lets all pairs contribute
(through the contact kernel), whereas the baseline masks all non-adjacent pairs to zero. They are
different reference states — which is exactly why D is a distinct, opt-in lever rather than a
special case of the baseline.
</details>

---

## One-Screen Cheat Sheet

```
   Lever  Flag                         Touches            Default(OFF)      Shape impact
   ─────  ───────────────────────────  ─────────────────  ────────────────  ──────────────
   A      --energy_terms K             final head         K=1               head 128→K, sum→scalar
   B      --rbf_centers M              distance kernel    M=0 (1 Gaussian)  NONE (sum back to 16)
   C      --use_burial                 node vector        off (burial_dim0) +1 node dim (before emb)
   D      --flory_unfolded --flory_nu  unfolded distances off                NONE (value-only)
   E      --denoise_weight w           aux head + train   w=0.0             +aux head, train loss only
   F      --gcn_span S                 GCN edges          S=1               more GCN edges
          --use_edge_features          GAT edges          off               +41-dim edge_attr
```

**Golden rules:** (1) OFF = exact baseline for every lever. (2) Change ONE lever per ladder rung.
(3) A rung inherits the previous winner; if a rung regresses, drop it and continue from the prior
winner. (4) All widths come from `CFG` — that single source of truth is why the levers never make
the builder and model disagree.

---

<a id="evaluation-metrics"></a>
## Evaluation Metrics

We evaluate ddG / dG predictions against experimental values with three metrics.
Two measure *trend/ranking* (correlation), one measures *absolute error* (RMSE).
Let `y` be the true stability values and `p` the predicted values over `n` mutations.

### Pearson correlation coefficient (PCC)

```
r = cov(p, y) / (σ_p · σ_y)
  = Σ_i (p_i - p̄)(y_i - ȳ) / sqrt( Σ_i (p_i - p̄)²  ·  Σ_i (y_i - ȳ)² )
```

`r ∈ [-1, +1]`: `+1` perfectly linear, `0` no linear relationship, `-1` anti-correlated.
It is **shift- and scale-invariant** — adding a constant or multiplying all predictions by a
positive factor does not change `r`. So PCC answers "does the model get the *pattern* right?"

**Why PCC is our primary grading metric.** For ranking mutation candidates, the *relative
ordering* and linear trend matter far more than the exact kcal/mol value. Our energy model
outputs energy in relative units (not calibrated to absolute kcal/mol), so a shift/scale-invariant
metric is the fair way to score it. The project's historical best on the PNAS-filtered Megascale
setup is a PCC of **~0.5259**, the reference bar the v5 improvements are measured against.

### Spearman rank correlation (ρ)

Spearman is Pearson computed on the *ranks* of the values: `ρ = Pearson(R(p), R(y))`. For distinct
ranks with no ties, `ρ = 1 - 6·Σ d_i² / (n(n² - 1))`, `d_i = R(p_i) - R(y_i)`. Same range as
Pearson but for *monotonic* (not necessarily linear) agreement. It is robust to outliers because a
large numeric error only moves an item by its rank position. For prioritizing which variants to
test experimentally, correct *ordering* is exactly the quantity of interest.

### RMSE (Root-Mean-Square Error)

```
RMSE = sqrt( (1/n) · Σ_i (p_i - y_i)² )
```

Same units as the target (kcal/mol *when calibrated*; here our energy is relative units, so read
RMSE as a magnitude-of-error diagnostic). Unlike the correlations, RMSE penalizes *absolute*
deviations and squaring makes it sensitive to outliers. A model can have excellent PCC (right
trend) but poor RMSE (offset/scale) — that is why we report all three.

---

<a id="loss-functions"></a>
## Loss Functions

Losses are the *training* objective; metrics are the *evaluation* objective. The fine-tuning loss
in `pnas_train_v5.py` is a weighted sum:

```
loss = primary_loss + reg_loss + energy_reg + rank_loss + corr_loss  (+ denoise term, Lever E)
```

### Huber loss (δ = 1)

```
L_δ(e) = 0.5 · e²              if |e| ≤ δ
       = δ · (|e| - 0.5·δ)     if |e| >  δ            (e = p - y)
```

Near zero it behaves like **MSE** (smooth, fine-grained convergence); far from zero like **L1**
(constant slope `δ`, gradient saturates so a single mislabeled mutation cannot dominate). This
robustness matters because experimental ddG labels contain measurement noise.

### Margin ranking loss

```
L_rank(i, j) = max( 0, -s · Δp + margin ),   s = sign(y_i - y_j),  Δp = p_i - p_j
```

Random pairs are sampled from the mini-batch (up to 128), scaled by `RANKING_LAMBDA` (default 0.1).
It says "if `i` is truly more stable than `j`, `p_i` should exceed `p_j` by at least `margin`."
Correct-and-separated pairs clamp to zero; wrong/too-close pairs get pushed apart. Because our
grading metric is a correlation (an ordering metric), optimizing pairwise order aligns training
with evaluation.

### Energy regularization (`0.001 · ‖E‖²`)

```
energy_reg = E_REG_LAMBDA · mean(E²),   E_REG_LAMBDA = 0.001
```

The energy is *extensive* (`E = Σ_residues Σ_terms Fh`), so it grows with length and can drift to
large magnitudes only weakly constrained by the ddG *difference* signal. The tiny L2 penalty anchors
the absolute energy scale near zero without distorting the difference that carries the ddG signal.

### Correlation losses (v5, opt-in via `--corr_weight`)

**Pearson loss** `L_pearson = 1 - r` — directly rewards the linear trend PCC measures;
shift/scale-invariant. Implemented in `losses_v5.py` with centering + `eps`; returns 0 for batches
< 2 or zero-variance.

**CCC loss (Lin's Concordance)** `CCC = 2·cov(p,y) / (σ_p² + σ_y² + (μ_p-μ_y)²)`, `L_ccc = 1 - CCC`.
Equivalently `CCC = r · C_b`, where `C_b ∈ (0,1]` is a bias-correction factor = 1 only when means
**and** variances match. So CCC additionally penalizes a **location shift** (`(μ_p-μ_y)²`) and a
**scale mismatch** (`σ_p²+σ_y²`). It demands agreement about the **1:1 line**, not just a linear
relationship: `CCC ≤ |r|` always. Use Pearson loss when only the trend matters; CCC when absolute
location/scale should agree with experiment.

### Denoising MSE (Lever E)

`denoise_MSE = mean( (prediction_noised - target)² )` — perturb structural/distance features with
Gaussian noise and train the model to stay consistent with the clean structure. This is the
fine-tuning-time echo of the pretraining Denoising Score Matching objective (see EBM Pretraining);
gated by its lever flag so OFF = exact baseline. Carried through `get_deltaG` as `denoise_loss`.

---

<a id="dataset-biology"></a>
## Dataset Biology

### Tsuboyama et al. (2023) mega-scale stability dataset

Training/validation data derive from Tsuboyama et al. (*Nature*, 2023), which measured folding
stability for ~hundreds of thousands of protein domains and point mutants via a high-throughput
**cDNA-display proteolysis** assay — far beyond one-protein-at-a-time calorimetry.

### K50 cDNA-display proteolysis assay (how dG is obtained)

A folded protein hides its backbone and **resists** protease cleavage; an unfolded one is rapidly
digested. Variants are exposed to a protease across a *range of concentrations*. **K50** is the
protease concentration at the *midpoint* (half the molecules cleaved). More stable ⇒ higher K50.
A thermodynamic model linking the proteolysis midpoint to the folded↔unfolded equilibrium converts
K50 into a **folding free energy dG**; then `ddG = dG_mut - dG_wt`. Intuition: **K50 is a stability
thermometer read through protease resistance.**

### Why clamp dG to [-1, 5] kcal/mol

`--dg_ml` clamps target dG to **[-1, 5] kcal/mol** (`threshold = [-1.0, 5.0]` in the dataset code).
This is the **assay's dynamic range**: below the low end a protein is always unfolded/cut (assay
saturates), above the high end always folded/never cut (saturates the other way). Values outside
the window are extrapolations, not measurements; clamping keeps labels in the measurable region.

### Single mutations + PNAS curation (quality > quantity)

`--one_mut` keeps only **single-point** mutations (clean, attributable effect; multi-mutants
confound epistasis). **PNAS-filtering** selects high-quality, well-measured single-site variants.
The project's own experiments showed *adding* full unfiltered Megascale data **hurt** performance
vs the curated subset: for noisy-label ddG regression, **data quality beats data quantity.**

### ThermoMPNN and held-out test proteins

**ThermoMPNN** is the benchmark competitor: a ddG predictor fine-tuned on **ProteinMPNN**
inverse-folding representations, carrying a strong learned structure→sequence prior (inverse-folding
pretraining is the single biggest known lever in this field). ThermoMPNN benchmark proteins — and
their homologs — are **held out** of DeepPEF training to prevent *test leakage* and keep the
comparison a fair, out-of-sample test.

---

<a id="ebm-pretraining"></a>
## EBM Pretraining Theory

Implemented in the root `train.py` (pretraining stage); the conceptual foundation fine-tuning
builds on.

### Native structure = energy minimum

An **energy-based model** assigns a scalar energy `E(x)` to a configuration. We *define* correctness
by low energy: the native folded structure should have the **lowest** energy; wrong structures,
wrong sequences, and unfolded chains should be higher. Training *shapes the energy landscape* so its
minimum sits at the native state — a physics-flavored inductive bias (proteins fold to their
free-energy minimum).

### InfoNCE / Boltzmann contrastive loss

Turn each energy into an unnormalized Boltzmann probability `exp(-E/τ)` and make the native the
"correct class" among negatives:

```
L = -log[ exp(-E_native/τ) / Σ_k exp(-E_k/τ) ] = E_native/τ + log Σ_k exp(-E_k/τ)
```

In `train.py` (`lossd_fucntion`): (1) native folded energy must be lowest among decoy sequence,
decoy structure, native unfolded, and cyclic-permutation graphs; (2) native *unfolded* must beat a
*decoy structure*. Negatives cover distinct failure modes — shuffled sequence (right fold, wrong
seq), decoy structure (right seq, wrong fold), cyclic permutations (respect sequence order). The
`logsumexp` saturates to ~0 once native is well below all negatives, focusing effort on confused
cases.

### Denoising Score Matching (DSM)

Contrastive loss sets *where* the minimum is; DSM shapes the *slope around it* — teaching the energy
**gradient** to point back to native. With `noise ~ N(0, σ²)` on distance features, along direction
`v`: `v · ∇E ≈ v · (-noise / σ²)`. The implementation (`denoising_score_matching`) uses a
finite-difference estimate `v·∇E ≈ (E(x+εv) - E(x-εv)) / (2ε)` to avoid Hessian issues through norm
layers. Result: an energy surface with a smooth basin whose gradients push perturbed structures back
toward native — a learned, differentiable force field.

### Connection to Lever E

Lever E is the fine-tuning-time reflection of DSM: perturb the structure, require consistency with
the clean structure via a denoising MSE. This preserves the well-shaped basin learned in pretraining
and prevents ddG fine-tuning from flattening/distorting the landscape. DSM builds the force field;
Lever E keeps it intact while learning mutation effects.

---

<a id="why-physics-grounded"></a>
## Why Physics-Grounded (thesis framing)

The central bet is **architectural**: build **thermodynamics into the model** and let the network
fill in only what physics does not pin down. Three design choices encode this bias:

1. **Folded − unfolded energy difference.** Stability is *by definition* `dG = E_unfolded -
   E_folded`. The model computes two energies from the *same* coordinates (folded graph = full
   contacts; unfolded graph = linear chain) and takes the difference — mirroring the thermodynamic
   cycle. A mutation changes only the sequence embedding + one-hot (coordinates fixed), so
   `ddG = dG_mut - dG_wt` isolates the sequence-driven change.
2. **Extensive (summed) energy.** Energy = **sum of per-residue contributions**, exactly as physical
   energies are additive. Interpretable (per-residue terms) and physically motivated; the length
   dependence it introduces can be made intensive via v5's optional `length_norm` lever.
3. **Flory / unfolded-state reference.** The unfolded state is an explicit **reference state** (in
   the spirit of Flory's random-coil model). Stability is always a comparison to the unfolded
   ensemble, and the architecture bakes that comparison in.

**Why make this bet?** Encoding the folded/unfolded difference, additive energy, and Flory reference
shrinks the hypothesis space to physically plausible functions → better data efficiency, more
interpretable outputs, more trustworthy generalization. The honest trade-off: these constraints can
cap raw accuracy if a purely data-driven competitor (e.g. inverse-folding-pretrained ThermoMPNN) has
a stronger prior. The research question is whether a physics-grounded energy function can approach or
match such models while remaining interpretable and mechanistically faithful.

---

<a id="limitations"></a>
## Limitations & Future Work

- **No inverse-folding pretraining in the baseline** — the largest known lever in this field
  (ProteinMPNN-style structure→sequence pretraining) accounts for the bulk of the gap to ThermoMPNN;
  DeepPEF's EBM pretraining is a different (physics-grounded) prior.
- **Fixed WT structure for all mutations** — coordinates stay fixed per protein, so the model cannot
  see mutation-induced local structural relaxation.
- **Extensive energy** confounds signal with length unless `length_norm` is used.
- **Relative-unit energy** — output is not calibrated to absolute kcal/mol, so RMSE is a diagnostic,
  not a calibrated error.
- **Glycine virtual-CB caveat** — `add_cb` synthesizes a virtual CB for *all* residues including
  glycine (which has no real CB); burial/geometry features treat it uniformly.
- **ESM-IF broadcast nuance** — when `dual_esmif` is used, per-mutation embeddings must align to the
  1536-dim layout; the CFG single-source contract is what keeps this from silently misaligning.

Future directions the levers explore: energy decomposition (A) for interpretability, RBF kernels (B)
and burial (C) for richer geometry, Flory reference (D) and denoising (E) for a better physical
reference/force field, and extended connectivity + edge features (F) for longer-range structure.
