# WHAT WE SEE — an honest inventory of the model's actual inputs

Everything below was measured by inspection on the cluster, not recalled. Reproduction
scripts are `scratch_wws3.py` … `scratch_wws11.py` in `/home/nissimb/DeepPEF/`.
Gate after this work: `scripts/gate_g4_cpu.py` → **ALL PASS, dG=-0.0030 width=1092**.

Dataset for every statistic: the **28 test proteins**, 43–72 residues, **1563 residues pooled**.

```
['1GYZ','1PSE','1QKH','1QP2','1TUC','1W4H','2BTH','2K1B','2K28','2K5H','2KVS','2KWH',
 '2KXD','2L33','2LQK','2WXC','3DKM','4C26','6EWS','6EWT','6EWU','HEEH_KT_rd6_0746',
 'HEEH_KT_rd6_0793','HHH_rd1_0142','HHH_rd1_0244','r11_1081_TrROS_Hall',
 'r12_757_TrROS_Hall','r18_3_TrROS_Hall']
```

---

## 0. The five findings that matter

1. **The Gaussian kernel is saturated.** Coordinates are multiplied by `NANO_TO_ANGSTROM = 0.1`
   *before* `get_graph`, so `exp(-0.08·d_model²) = exp(-0.0008·d_Å²)`. Across the entire physical
   range 0–30 Å the kernel moves only from 1.000 to **0.487**. At contact distance (3.8 Å) it is
   **0.9885**. The distance channels are a near-constant 1.0 with a small ripple on top.
2. **D(16) has effective rank 1.016.** The 16 channels are mutually correlated at median
   **0.994**; PC1 explains **99.22 %**. Sixteen columns carry **one** number.
3. **That one number is protein length.** `corr(N, per-protein mean of D) = −0.9960`, and
   `corr(mean D, 1/√N) = +0.9989`. D is a length readout, not a structure readout.
4. **Columns 32:48 are read by nothing.** Proven by ablation: `+100` on those columns moves
   the energy by **exactly `+0.000000e+00`**. 16 of 1092 dims are dead weight.
5. **With default flags, folded and unfolded differ in only 16 of 1092 dims** — all inside D.
   `Fb`, `emb`, `one_hot` are **bit-identical** between the two graphs. dG is therefore driven
   by one effective scalar that is mostly a length proxy.

---

## 1. Block-by-block inventory

Layout (`train_utils.get_graph`, `_blocks` order):
`[ D(16) | Fb(32) | «new blocks at offset 48» | emb(1024) | one_hot(20) ]` → **1092**.

### Provenance of the raw tensors (per protein directory)

| file | shape | notes |
|---|---|---|
| `coords_tensor.pt` | `[N,4,3]` | N, CA, C **and CB** — CB is stored, not recomputed. Multiplied by **0.1** in `normalize_batch`. |
| `one_hot_encodings.pt` | `[M,N,21]` | M = mutation rows. The **21st column is dropped** at `train.py:317` (`[:,:,:,:-1]`). Row 0 = WT. |
| `mask_tensor.pt` | `[N]` | all 1.0 for the 28 (no missing residues). |
| `prott5_embeddings/*.pt` | `[128,N,1024]` chunks | one row per mutation, concatenated → `[M,N,1024]`. |
| `deltaG.pt` | `[M]` | the label. |

### D(16) — `x[:, 0:16]`

Channel `c = atom_i*4 + atom_j` over `[N, CA, C, CB]`. Pipeline:
`cdist → exp(g·d²) → mask → sum over j → F.normalize(p=2, dim=0)`.

The **diagonal artefact is real but tiny and channel-dependent**: self-distances are
`~5e-5` for the four true-diagonal channels 0/5/10/15, but the *cross-atom* diagonals are
genuine intra-residue distances (N–CA `0.1466`, N–C `0.2483`, CA–CB `0.1514` model units
= 1.47/2.48/1.51 Å — correct bond geometry).

Pooled over all 28 proteins:

| ch | pair | mean | std | min | max | CV |
|---|---|---|---|---|---|---|
| 0 | N-N | 0.132910 | 0.015785 | 0.0 | 0.162365 | 0.1188 |
| 5 | CA-CA | 0.132899 | 0.015880 | 0.0 | 0.162921 | 0.1195 |
| 10 | C-C | 0.132905 | 0.015824 | 0.0 | 0.162992 | 0.1191 |
| 15 | CB-CB | 0.132873 | 0.016091 | 0.0 | 0.164280 | 0.1211 |

All 16 means agree to **4 decimal places** (0.13287–0.13291). That is the point: they are
the same number.

```
16×16 correlation, off-diagonal: min 0.98126  median 0.99398  max 0.99994
eigenvalues: [15.876, 0.064, 0.053, 0.004, 0.003, 0.0002, 0, 0, ...]
PC1 = 99.22 %   PC1+PC2 = 99.62 %   effective rank = 1.016
```

**Nothing is near-constant in the "zero variance" sense** (min std 0.0158, none below 1e-2),
but the block is *rank-deficient*, which is the operationally identical problem: 16 columns,
one degree of freedom.

Why D ≈ 1/√N: with the kernel pinned near 1, `sum_j exp(g d_ij²) ≈ 0.95·N` for every residue,
so the column is nearly constant at `0.95N`; `F.normalize(dim=0)` then divides by
`√(N·(0.95N)²) = 0.95N√N`, leaving every entry ≈ `1/√N`. Confirmed: `corr = +0.9989`.

### Fb(32) — `x[:, 16:48]` — **what these actually are**

From `get_bonded_features(D)`, read literally: it takes the **Gaussian-transformed** `D`
(N,N,16) and extracts two off-diagonals.

```python
f1, f2 = D[n[:-1], n[1:]], D[n[1:], n[:-1]]   # (i,i+1) and (i,i-1)
f1 = cat([f1, zero_row])   # forward,  LAST row zero-padded
f2 = cat([zero_row, f2])   # backward, FIRST row zero-padded
Fb = cat([f1, f2], dim=-1) # N,32
```

So Fb is **not** 32 distinct bonded/geometric features. It is:

- `x[:, 16:32]` — the 16 atom-pair Gaussian values of the **forward** peptide bond `(i, i+1)`,
  last row zero.
- `x[:, 32:48]` — the same 16 for the **backward** bond `(i, i−1)`, first row zero.

Because consecutive-residue geometry is nearly rigid, the 16 channels *within* each half are
duplicates:

```
fwd half 16×16 off-diag corr: min 0.999172  median 0.999899  max 1.000000
                 eigenvalues: [15.99777, 0.00137, ...]   effective rank = 1.000
bwd half: identical statistics                            effective rank = 1.000
whole 32-block: PC1 57.37 %, effective rank 1.958
```

**Fb(32) carries exactly two numbers**: forward-bond geometry and backward-bond geometry.

A subtlety worth stating because it is easy to misread: `corr(fwd[ai,aj], bwd[aj,ai])` pooled
is only **0.147**, which looks like the two halves are independent. They are not — the
**zero padding dominates the variance**. Restricted to interior residues the correlation rises
to **0.351**, and:

| col | std (all) | std (interior) | ratio | corr with terminus flag |
|---|---|---|---|---|
| 16 | 0.148825 | 0.051075 | 0.343 | **−0.7055** |
| 24 | 0.149794 | 0.051378 | 0.343 | **−0.7047** |
| 32 | 0.148825 | 0.072110 | 0.485 | −0.6126 |
| 40 | 0.146829 | 0.071200 | 0.485 | −0.6126 |

**Roughly two-thirds of Fb's apparent variance is the zero pad**, i.e. the feature mostly
encodes "am I the first/last residue". Fb's headline std (0.148) is the largest of any block
and it is largely an artefact.

### emb(1024) — `x[:, 48:1072]`

ProtT5 per-residue, then `F.normalize(p=2, dim=0)` — **normalized down the residue axis**,
not per residue. Consequence: **every column has L2 norm exactly 1.0** (measured min = max =
1.000000), so per-residue norms are not unit and vary (3.11–6.39, mean 4.12 on 1GYZ).

```
after normalize : min −0.558148  max 0.609691  mean −0.003497  std 0.130142
exact zeros     : 0 / 60416 (0.0000 %)   |v| < 1e-6 : 1 (0.0017 %)
raw ProtT5      : min −0.8795  max 0.9837  mean −0.00425  std 0.17694
raw per-residue L2 mean 5.5946
```

**Not sparse.** Dense, zero-centred, roughly Gaussian.

### one_hot(20) — `x[:, 1072:1092]`

Values `{0,1}`, row sums exactly 1. Alphabet is `AA_MAP`, alphabetical:
`ACDEFGHIKLMNPQRSTVWY`. Decoded 1GYZ sequence:
`WIARINAAVRAYGLNYSTFINGLKKAGIELDRKILADMAVRDPQAFEQVVNKVKEALQV` — correct.

**Confirming this is what a mutation changes:**

| row | mut | one-hot residues changed | emb residues changed | ‖Δemb‖ | ‖Δone-hot‖ |
|---|---|---|---|---|---|
| 5 | W1Q | `[0]` | **59/59** | 6.3518 | 1.4142 |
| 6 | W1E | `[0]` | 59/59 | 6.9263 | 1.4142 |
| 20 | W1Y | `[0]` | 59/59 | 4.5310 | 1.4142 |
| 100 | I5T | `[5]` | 59/59 | 6.3403 | 1.4142 |

Two facts follow, and they are the whole ddG story:

- **One-hot changes 2 numbers out of 59×1092.** It is a local, sparse, ‖·‖ = √2 edit.
- **The embedding changes everywhere.** ProtT5 is re-run on the mutant sequence, so context
  propagates: per-residue |Δemb|₁ for W1Q decays 98.4 → 75.8 → 38.9 → 21.4 … but never to
  zero (still ~7–15 at the far terminus). ‖Δemb‖ is **4.5× larger** than ‖Δone-hot‖.
- **Coordinates do NOT change.** The mutant is scored on WT coordinates. So D and Fb are
  *bit-identical* between WT and mutant.

> **ddG is carried almost entirely by the embedding**, with a 2-number one-hot assist.
> D and Fb cancel exactly. This is the metric rule in its sharpest form.

---

## 2. Which blocks are informative? Variance over 1563 residues

```
block      dims     mean std      min std      max std   mean range
D            16     0.015902     0.015778     0.016125     0.163194
Fb           32     0.147999     0.146421     0.149794     0.991203
emb        1024     0.130924     0.013900     0.133844     0.901873
one_hot      20     0.203328     0.000000     0.318446     0.950000
```

### How many dimensions are dead?

```
std <= 0        :  1 / 1092   D 0/16  Fb 0/32  emb 0/1024  one_hot 1/20
std <= 1e-6     :  1 / 1092   (identical)
std <= 1e-2     :  1 / 1092   (identical)
```

**By the literal zero-variance test, only ONE dimension is dead** — a one-hot column for a
residue type absent from all 28 proteins. That is the honest answer to the question as posed,
and it is *not* the interesting answer.

### The real answer: rank, not variance

Variance per dimension is the wrong instrument, because it cannot see duplication. Effective
rank can:

| block | dims | effective rank | genuine d.o.f. | wasted dims |
|---|---|---|---|---|
| D | 16 | **1.016** | ~1 | ~15 |
| Fb | 32 | **1.958** | ~2 | ~30 |
| one_hot | 20 | 19 (rank-20 minus 1 unused) | 19 | 1 |
| emb | 1024 | high (dense) | many | — |
| **structural total** | **48** | **~3** | **3** | **~45** |

Plus the 16 provably unread columns (§3). Net:

> **The 48 structural dimensions carry about 3 numbers.** Of 1092 inputs, ~45 are redundant
> copies and 16 more are read by nothing at all. The model's structural view of a protein is
> approximately: *(a length proxy, forward-bond geometry, backward-bond geometry)*.

This is the mechanistic explanation for the measured facts in the brief. Structure barely
enters, so per-protein dG is poorly determined (pooled PCC ~0.59), while ddG — which is carried
by the embedding — does much better (0.798). It also explains why the coil (U3–U6) is the best
b_p lever: it is the only lever that changes the one channel family that actually differs
between folded and unfolded.

---

## 3. What does the GCN see? — confirmed from code and by ablation

`model/hydro_net.py` `PEM.forward`, with all levers off:

```python
x_gcn = torch.cat((x[:, :self.non_bonded_index + self.non_bonded_index],  # 0:32
                   x[:, self.one_hot_index:]), dim=-1)                    # -20:
x_gat = torch.cat((x[:, :self.non_bonded_index],                          # 0:16
                   x[:, self.one_hot_index:]), dim=-1)                    # -20:
x_emb_features = x[:, self.llm_index:self.one_hot_index]                  # -1044:-20
```

Constructed constants (dumped from a live `PEM`):

```
non_bonded_index 16   bonded_index 48   one_hot_index -20   llm_index -1044
solv_start 48  solv_dim 0   desc_start 48  desc_dim 0   lig_start 48  lig_dim 0
fc1_gcn (64,52)   fc1_gat (64,36)   fc2_gcn (36,64)   fc2_gat (36,64)   fc1 (128,1096)
```

**A correction to the brief.** The GCN does **not** read `x[:, :32]` only — it reads
`x[:, 0:32]` **concatenated with `x[:, -20:]`**, width **52**. The GAT reads `x[:, 0:16]` +
`x[:, -20:]`, width **36**. Both branches see the one-hot; neither sees the embedding.

| branch | input | width | sees emb? |
|---|---|---|---|
| GCN (chain-local edges) | `x[:,0:32]` ++ `x[:,-20:]` | 52 | **no** |
| GAT (k-NN / fully-connected edges) | `x[:,0:16]` ++ `x[:,-20:]` | 36 | **no** |
| residual path | `x[:,-1044:-20]` | 1024 | concatenated **after** both GNNs |

The embedding **bypasses message passing entirely**: it is concatenated at
`x = torch.cat((x, x_emb_features), dim=-1)` → `fc1 (128, 1096)`, where 1096 = 72 + 1024.
ProtT5 is never propagated along an edge. It is a per-residue lookup.

### The dead 16 — proven, not argued

`x[:, 32:48]` (the backward half of Fb) appears in **no** slice: GCN stops at 32, GAT stops at
16, the emb window starts at 48. Ablation on an untrained net, `+100` per block:

```
cols   0:16   D             delta E = +3.332525e-03
cols  16:32   Fb fwd half   delta E = +1.136661e-04
cols  32:48   Fb bwd half   delta E = +0.000000e+00   <-- DEAD
cols 48:1072  emb           delta E = +9.797035e-01
cols 1072:1092 one_hot      delta E = +4.595131e-03
```

Exactly zero. `get_bonded_features` computes 32 columns; the network consumes 16. **The
backward bond is computed and discarded.**

Note also the magnitudes: the embedding term is **~300× larger** than D and **~200×** larger
than one-hot. The model is dominated by the residual embedding path.

### The architectural fact and its consequence for every feature-block design

> **A block inserted at offset 48 is invisible to both GNNs *unless* `fc1_gcn`/`fc1_gat` are
> widened *and* the forward pass is taught to splice it in.**

The code already does this correctly — `_extra` in `forward()` explicitly re-slices W5/W6/W11
blocks and concatenates them into `x_gcn`/`x_gat`, which is why `fc1_gcn` is
`52 + _sd + _dd + _ld` and not 52. **This splice is mandatory, not optional.** Omit it and the
block lands between 48 and −1044, where the GCN slice has already stopped and the emb window
has not yet started — a silent no-op with every shape check passing. That is precisely the
failure mode `PEMGraphTransformer` now raises on.

Three design rules follow:

1. **Any new block needs three edits, not one**: `_blocks` in `get_graph`
   *and* `get_unfolded_graph`; `fc1_gcn`/`fc1_gat` widths; the `_extra` splice in `forward`.
   `fc2_*`, `inst_norm1`, `inst_norm2`, `fc_in_dim` must not move.
2. **The two branches see different information**, so a block reaches the GCN and the GAT with
   different context (GCN also has the forward bond; GAT has only D). A feature that needs
   bonded context is better used by the GCN branch.
3. **A block that only needs to be a per-residue lookup does not need the GNN at all** — but
   there is currently no path for that except the emb concatenation, which is not exposed.

---

## 4. The energy head, and where the per-protein offset enters

### Effective input

```
h_i = relu( fc1( [ gcn_out_i(36) ; gat_out_i(36) ; emb_i(1024) ] ) )   fc1: 1096 -> 128
e_i = fc2(h_i)                                                          fc2: 128 -> 1
E   = sum_i e_i                                                         (readout='sum')
dG  = E_folded - E_unfolded
```

`get_energy` is a **plain sum over residues** — the energy is *extensive*.

### Where the offset enters — reasoning from the architecture

Four structural reasons, in order of force:

**(a) Extensivity times a near-constant per-residue term.** `E = Σᵢ eᵢ ≈ N·⟨e⟩`. Any component
of `eᵢ` that is roughly constant within a protein becomes a term proportional to **N**. Over the
28, N spans **43–72 (a 52 % spread)**. A per-residue bias `β` contributes `βN` to E, and to dG
it contributes `β(N_f − N_u) = 0` only if the bias is identical in both states — which for the
dominant emb path it **is** (emb is bit-identical folded vs unfolded under default flags). So
the constant part cancels in dG but **not** its interaction with the 16 D columns that do differ.

**(b) D is a length readout, and D is the *only* thing that differs between the two graphs.**
Measured: dims differing folded-vs-unfolded = `[0..15]`, exactly D; `Fb`, `emb`, `one_hot` are
identical to machine precision. And `corr(N, mean D) = −0.9960`. Therefore **dG is a function of
a length-dominated scalar**. Directly measured:

```
per-protein sum(D_f − D_u):  min −0.21203  max +0.42384   corr with N = −0.5742
```

This is the offset's most plausible entry point. It is not a subtle statistical artefact — the
reference state differs from the folded state in one length-correlated direction, and the
network must convert that into an absolute free energy.

**(c) The 1/√N normalization couples every protein's D to its own length.**
`F.normalize(p=2, dim=0)` normalizes **within a protein, down the residue axis**. Its divisor is
a function of N. So the same local geometry in a 43-mer and a 72-mer produces *different* D
values. The feature is not comparable across proteins — which is the definition of a
per-protein offset in the input.

**(d) The between-protein / within-protein split confirms D is where it lives.**

```
block      between-protein std   within-protein std   B²/(B²+W²)
D               0.011366             0.010127           0.5573
Fb              0.013561             0.147359           0.0084
emb             0.031049             0.128071           0.0597
one_hot         0.030672             0.183443           0.0291
```

**D is the only block whose variance is majority between-protein (55.7 %).** Fb, emb and
one-hot are overwhelmingly within-protein. A per-protein offset needs a per-protein-constant
carrier, and D is the only candidate. Correspondingly:

```
corr(N, per-protein block mean):  D −0.9960   Fb +0.5878   emb +0.2317   one_hot −0.0969
```

### Conclusion

> The per-protein offset **b_p** most plausibly enters through **D(16) → the 1/√N
> normalization → the extensive sum**. D is (i) the only block that differs between folded and
> unfolded, (ii) 99.2 % one direction, (iii) correlated −0.996 with N, and (iv) the only block
> whose variance is majority between-protein. Fb, emb and one-hot cannot carry it: they are
> within-protein by construction and cancel identically in dG.

This is consistent with, and *mechanistically explains*, the reported measurements: `ICC(b_p) =
0.898` (the offset is a stable protein property), the b_p corrector failing out of sample (it is
fitting N, which does not generalize), and the coil being the best b_p lever (U3–U6 act on the
unfolded D — the one channel family that actually moves). `a_p` behaves differently because
exposure varies *within* a protein, which is where emb and one-hot live.

---

## 5. Implications for feature-block design (what this changes)

1. **Do not add another block that duplicates D.** Any new distance-derived, Gaussian-kernelled
   feature will land in the same saturated, rank-1 regime. Fixing `gaussian_coef` or the 0.1
   scaling would recover more information than any new block — the kernel currently discriminates
   3.8 Å from 8 Å by 0.038.
2. **Reclaim the dead 16.** `x[:, 32:48]` is computed and discarded. Widening the GCN slice to
   `x[:, :48]` and `fc1_gcn` to `(64, 68)` costs 16×64 = 1024 parameters and is the cheapest
   real information gain available — but it is **not** bit-identical, so it needs its own arm
   and a gate assertion that the width changed *and* dG moved.
3. **Score reference-state levers on dG or b_p, never ddG.** Confirmed at the tensor level:
   coordinates are shared between WT and mutant, so D and Fb are bit-identical across a
   mutation and cancel *exactly* in ddG. Only emb and one-hot survive a ddG.
4. **Any anti-silent-no-op gate must assert on VALUES, not shapes.** The 32:48 result is the
   proof: a block can be present, correctly shaped, correctly ordered, and consume zero
   influence. The gate must perturb the block and assert `|ΔE| > 0`.

### The gate a new block must pass

```python
# 1. width grew by exactly K, and only via fc1_*
assert x.shape[-1] == 1092 + K
assert m.fc1_gcn.in_features == 52 + K and m.fc1_gat.in_features == 36 + K
assert m.fc2_gcn.out_features == 36 and m.fc2_gat.out_features == 36
assert m.inst_norm1.inst_norm.num_features == 36
assert m.inst_norm2.inst_norm.num_features == 72
assert m.fc1.in_features == 1096
# 2. right-anchored slices did not move
assert torch.equal(x[:, -20:], one_hot)
assert torch.equal(x[:, -1044:-20], F.normalize(emb, p=2, dim=0))
# 3. ANTI-SILENT-NO-OP: perturbing the block must move the energy
z = x.clone(); z[:, :, 48:48+K] += 100.0
assert abs(float(m(z) - m(x))) > 1e-6, "block inserted but read by nothing"
# 4. baseline unchanged with the lever off
#    scripts/gate_g4_cpu.py -> dG=-0.0030 width=1092
```

Item 3 is the one that would have caught the dead 16.
