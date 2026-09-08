# DESIGN_CHEMISTRY.md — How chemistry enters DeepEF

Status: written design, implementation-ready. All numbers below were measured on the
cluster from the committed CSVs, not recalled. Gate baseline at time of writing:
`scripts/gate_g4_cpu.py` -> **G4-CPU: ALL PASS, dG=-0.0030, width=1092**.

---

## 1. WHAT WE ACTUALLY SEE

### 1.1 Provenance of the matrix

`data/aa_descriptors_mordred.csv` — a **20 x 726** float matrix, row-keyed by one-letter
code, row order `ACDEFGHIKLMNPQRSTVWY` (= `train_utils.AA_MAP`, so **row i is exactly
one-hot index i**). The header block records the whole pipeline:

| stage | count |
|---|---|
| Mordred descriptors registered | 1613 |
| after dropping any column with a missing value | 1278 (-335) |
| after `nunique >= 15` of 20 | **726** (-552) |

Inputs are PubChem canonical isomeric SMILES for **neutral free L-amino acids** (not
zwitterionic), `smiles_sha256_16=b3f4304c226f896f`, `Calculator(descriptors, ignore_3D=True)`.
Histidine was corrected against the PHASES table (which encoded a pyridine-type ring,
C7H10N2O2, instead of imidazol-4-yl C6H9N3O2); an RDKit formula + stereocentre gate
enforces 20/20. Normalisation is a **z-score down each column across the 20 residues**
(ddof=0) — verified live: column means are 1.35e-10 and every column std is exactly
1.0000. So "scales" is a settled question: **every one of the 726 columns is already unit
variance and zero mean.** There is no scale heterogeneity left to fix; a raw Mordred
matrix would have spanned molecular weight (~10^2) to fractional indices (~10^-2), and
that is already gone.

One deviation to state in the thesis: Ofir kept descriptors with `nunique >= 40` across
**58** residues. With 20 residues, 40 is unsatisfiable (max possible = 20), so the
threshold was rescaled 40/58 = 0.690 -> 15/20 = 0.750. This is a defensible rescaling, but
it is *our* choice and must be reported as such.

### 1.2 What the 726 columns are

They are Mordred's 2D descriptor families, and the naming makes the redundancy visible on
sight: `SpAbs_A, SpMax_A, SpDiam_A, SpAD_A, SpMAD_A, LogEE_A, VE1_A, VE2_A, VE3_A,
VR1_A...` are all functionals of the *same* adjacency-matrix eigenspectrum; `ATS0dv..ATS4dv,
ATS0d..ATS5d, ATS0s..ATS6s, ATS0Z..ATS6Z, ATS0m..ATS6m, ATS0v..` are Moreau-Broto
autocorrelations of the same molecular graph at lags 0-6, weighted by six different atomic
properties (valence-electron count, degree, intrinsic state, atomic number, mass, volume).
These are not 726 independent measurements of a molecule. They are on the order of a dozen
underlying graph-theoretic quantities, each read out through many near-equivalent
functionals.

### 1.3 Correlation structure — measured

Over the 726*725/2 = 263,175 column pairs:

- mean |r| = **0.423**, median |r| = **0.377**
- fraction with |r| > 0.90 = **8.3%**
- fraction with |r| > 0.99 = **0.7%**

Average-linkage clustering on 1-|r| gives **278** clusters at |r|>0.95, **194** at
|r|>0.90, **109** at |r|>0.80. So even at a loose 0.8 threshold the 726 columns collapse
to about a hundred groups — and that is still an overcount, because clustering on
correlation cannot see nonlinear dependence between, say, `SpMax_A` and `SpDiam_A`.

### 1.4 The eigenvalue spectrum — the number that decides everything

SVD of the column-centred 20x726 matrix. **Rank = 19 exactly** (singular values beyond 19
are at machine zero). This is not a property of chemistry; it is arithmetic: 20 points,
centred, live in a 19-dimensional affine space. Variance fractions:

```
PC1  0.4627   PC6  0.0431   PC11 0.0101   PC16 0.0035
PC2  0.1775   PC7  0.0313   PC12 0.0083   PC17 0.0020
PC3  0.0814   PC8  0.0244   PC13 0.0067   PC18 0.0017
PC4  0.0524   PC9  0.0178   PC14 0.0062   PC19 0.0002
PC5  0.0517   PC10 0.0145   PC15 0.0046
```

Cumulative: PC1..3 = 0.7216, PC1..7 = **0.9001**, PC1..10 = 0.9568, PC1..15 = 0.9819,
PC1..16 = **0.9962**.

Two scalar summaries of effective dimension:

- **participation ratio (sum L)^2 / sum(L^2) = 3.82**
- **exp(spectral entropy) = 6.45**

And the components are chemically legible — this matters for the thesis, because it shows
the matrix is not noise:

| component | r with residue volume | r with Kyte-Doolittle |
|---|---|---|
| PC1 (46.3% var) | **+0.966** | -0.152 |
| PC2 (17.8% var) | +0.172 | **+0.695** |
| PC3 (8.1% var) | -0.012 | -0.196 |

**PC1 is size. PC2 is hydrophobicity.** Together 64% of the variance. That is the same
two axes every amino-acid property scale since Kyte-Doolittle 1982 has recovered.

### 1.5 What "726 columns on 20 points" means, said plainly

The matrix has 20*726 = 14,520 entries but at most 20*19 = 380 degrees of freedom, and
effectively ~4-6. **726 is not a dimensionality, it is a redundancy factor of ~38-190x.**
Any statement of the form "the descriptor block adds 726 features" is false. It adds at
most 19 linearly independent numbers per residue, of which ~7 carry 90% of the variance.

There is no estimation problem here in the usual sense — the matrix is a fixed lookup
table, not data we fit — but there is a **conditioning** problem, and §4 shows it is the
whole story.

---

## 2. THE HONEST PROBLEM — the block adds *zero* information

State this in the thesis without hedging.

The descriptor block is computed as `Desc = one_hot @ T` where `T` is the fixed [20, K]
table (`aa_descriptors.residue_descriptors`, a matmul). Therefore `Desc` is a
**deterministic linear function of the one-hot.** Concatenating it produces

```
[ one_hot | one_hot @ T ]  =  one_hot @ [ I | T ]
```

which spans exactly the same column space as `one_hot` alone. For the first linear layer
`fc1_gcn` / `fc1_gat`, whose relevant slice acts as `one_hot @ W`:

> For **every** weight `W_desc` on the descriptor block there exists `W_oh = T @ W_desc`
> on the one-hot giving **identical outputs for all inputs**. The function classes are
> equal. A model with a one-hot and a free first layer **cannot** be improved by adding
> descriptors, at any width, for any K, for any descriptor set. Not "unlikely to be" —
> *cannot be*, as a matter of linear algebra.

This holds for the full nonlinear network too, because the one-hot enters only through
that first Linear: the map `one_hot -> fc1 preactivation` is linear, and everything after
it is shared. Descriptors do not enlarge the hypothesis space by one function.

So any honest case for W6 rests on exactly two things, and it must be argued on those and
scored accordingly:

**(a) Regularisation / inductive bias.** The parameterisation changes even though the
function class does not. Under one-hot, the 20 residue vectors are free parameters and
mutually orthogonal — evidence about leucine tells the model *nothing* about isoleucine.
Under descriptors, they are tied: `w_L` and `w_I` are forced to be close because the rows
are close (measured: |L-I| = 21.3 vs alphabet mean 37.5). With ~10 residues of a given
type per protein and 28 test proteins, that sharing is the only mechanism by which W6 can
help. **This is a variance-reduction claim, and it can be falsified.**

**(b) Generalisation to unseen residues.** A residue with no one-hot column still has a
SMILES and therefore a descriptor row. This is Ofir's real contribution, and
`aa_descriptors.py` already implements the open alphabet (label-keyed gather, raises on
unknown labels rather than returning zeros). **But it is worth nothing on our benchmark:
all 28 test proteins are single-chain, ligand-free, metal-free monomers, 43-72 aa, and
every residue is canonical.** Ofir himself concedes physicochemical properties are already
implicit in LM embeddings, so his case narrows to coverage of non-canonical residues — a
case our evaluation set cannot express. MSE (selenomethionine) in the het codes is an
amino acid *in the chain*, not a ligand, and is exactly the kind of thing this would
serve — but it is not in the 28.

The correct thesis sentence: *W6 cannot add information; it can only add bias. We test the
bias and report the result, and we implement the open alphabet because it is the honest
generalisation of the idea, while stating that our benchmark cannot reward it.*

---

## 3. THE THREE DESIGNS

Three ways to use `T`, all currently reachable from `--aa_descriptors`.

### 3.1 Arm A — CONCAT (`mordred726`, `mordred_pca16`)

`[ ... | Desc(K) | emb(1024) | one_hot(20) ]`. Both blocks present.

By §2 this is a strict no-op on function class. Worse, it is a *harmful*
reparameterisation: the model now has two redundant routes to the same quantity, the
one-hot route is perfectly conditioned (§4) and the descriptor route is not, and weight
decay splits the burden between them by norm rather than by usefulness. Gradient descent
will overwhelmingly use the one-hot route because it is better conditioned, so the
descriptor columns act mainly as **additional parameters that receive gradient and
contribute nothing** — pure variance. This arm is the one most likely to be run, because
it is the default-shaped "add a feature" move, and it is the **least defensible**.

Note `mordred726` widens 1092 -> 1818, a 66% widening of the `fc1_gcn` / `fc1_gat` input,
to carry at most 19 independent numbers. That is indefensible on its face and should be
run only as a negative control.

### 3.2 Arm B — REPLACE (`mordred_pca16_only`)

One-hot **zeroed** (not deleted), descriptors carry residue identity alone.

This is the only arm where the regularisation claim (a) is actually *tested*, because it
is the only one where the descriptor geometry is load-bearing. If chemical similarity is a
good prior, replace ~= baseline or better with 16 columns instead of 20. If it is a bad
prior, replace is worse — and that is a **real, publishable negative result**, unlike
concat's guaranteed null.

The implementation detail here is already right and is worth a thesis sentence:
`train_utils._onehot_block` zeroes rather than deletes, because `hydro_net` slices `emb`
and `one_hot` **right-anchored** (`self.one_hot_index = -20`,
`x[:, llm_index:one_hot_index]`). Deleting 20 columns would silently re-point `x[:, -20:]`
into the tail of ProtT5 and slide the 1024-wide LLM window left into `Fb`. Every shape
check would still pass and every number would be wrong. Zeroing costs nothing — a constant
column is a bias the following Linear already has.

Caveat that must be stated: replace is **not** information-preserving in the way concat
is. The 20x16 PCA table has rank 16 < 19, so it merges the alphabet: residues that differ
only along PC17-PC19 become indistinguishable. Measured, PC17-19 carry
0.0020+0.0017+0.0002 = **0.39%** of variance, and no pair of the 20 collides at 16
components (minimum pairwise distance 1.81 in PCA space, well clear of zero). So the merge
is real but tiny. Using the full 726 (rank 19) instead would be exactly
information-preserving — and, per §4, far worse conditioned. **This is the trade to state:
pca16 loses 0.39% of the alphabet's resolution to buy a 4.5x better-conditioned mutation
delta.**

### 3.3 Arm C — DESCRIPTOR-INITIALISED EMBEDDING (not yet built)

Keep the one-hot path and its exact geometry, but **initialise** the 20 rows of the
first-layer residue weights from the descriptor table, then let them fine-tune:

```
W_res[0:20, :]  <-  alpha * (T_pca16 @ R)     R: [16, 64] random; alpha set so the
                                              init's weight std matches the default
```

where `W_res` is the one-hot slice of `fc1_gcn` / `fc1_gat`. Chemically similar residues
*start* with similar weights; gradient descent may separate them if the data demands it.

This is the design that matches the actual claim. The claim is "chemical similarity is a
good **prior**". A prior belongs in the initialisation (or in a penalty), **not** in the
function class — and §2 proved it cannot be in the function class. Arm C puts it exactly
where it belongs:

- **zero** added width (1092 stays 1092), zero added parameters, zero added FLOPs;
- the well-conditioned one-hot geometry of §4 is retained in full;
- the prior is *escapable* — unlike replace, which imposes the descriptor metric for the
  entire run and cannot recover if the metric is wrong;
- it degrades gracefully: with enough data it converges toward the one-hot baseline, so
  its downside is bounded, whereas replace's downside is not.

A soft variant, if a stronger prior is wanted: keep the init and add
`lambda * ||W_res - W_res_init||^2` (an L2-to-descriptors pull rather than L2-to-zero).
One extra scalar `lambda`; `lambda=0` recovers Arm C, `lambda -> inf` recovers Arm B's
rigidity.

### 3.4 Verdict

**Arm C is optimal. Arm B is the informative experiment. Arm A should be run once as a
negative control and then dropped.**

Ranking, with the reason each alternative is worse:

| arm | width | can it help? | why not optimal |
|---|---|---|---|
| A concat | 1108 / 1818 | **provably not** (§2) | adds parameters and an ill-conditioned redundant route; guaranteed null at best, added variance at worst |
| B replace | 1092 (OH zeroed) | yes, via bias | imposes the prior irrevocably; inherits the 132:1 delta conditioning of §4; loses 0.39% alphabet resolution |
| **C init** | **1092** | **yes, via bias** | — chosen: same prior, escapable, no width, no parameters, keeps cond=1 geometry |

The honest expected outcome for all three is *no significant dG improvement*, because
ProtT5 already encodes residue chemistry and the network already has a free 20-row
embedding. The value of the section is that it **says which experiment can distinguish
them, and predicts the answer in advance.**

---

## 4. THE CRUX — where the mutation actually enters

ddG is a difference of two forward passes over graphs identical except at **one residue's
one-hot row**. So the descriptor block changes in exactly one row too, and the model sees
a K-dimensional delta at one position. The question nobody has analysed: **is a 726-dim
delta better or worse conditioned than a 20-dim one-hot delta?** Measured over all
20*19/2 = 190 residue pairs:

| representation | dim | mean ‖delta‖ | min | max | max/min | CV | delta rank | eff. dim (PR) |
|---|---|---|---|---|---|---|---|---|
| one_hot | 20 | 1.4142 | 1.4142 | 1.4142 | **1.00** | **0.000** | 19 | **19.00** |
| pca16 | 16 | 5.568 | 1.814 | 12.194 | 6.72 | 0.294 | 16 | 3.79 |
| mordred726 | 726 | 37.530 | 12.359 | 81.980 | **6.63** | 0.292 | **19** | **3.82** |

And the Gram matrix `G = D^T D / n` of those 190 deltas — the quantity that sets the
per-direction curvature of a linear head on this block:

| representation | lambda_max | lambda_min | **cond(G)** | sqrt(cond) = gradient anisotropy |
|---|---|---|---|---|
| one_hot | 0.105 | 0.105 | **1.0** | **1.00** |
| pca16 | 15.644 | 0.118 | 132.2 | 11.50 |
| mordred726 | 707.1 | 0.260 | **2724.5** | **52.20** |

**The answer is: strictly worse, and by a large factor.**

Four things follow, and they are the core of this section.

**(i) The one-hot mutation delta is perfectly conditioned — it is the best possible
representation of a mutation, and this is not an accident.** Every one-hot delta is
`e_mut - e_wt`, always of norm sqrt(2), always orthogonal to the deltas of unrelated
pairs. The Gram is a scaled identity, cond = **1.0** exactly. Every mutation type gets
identical gradient magnitude, and the 190 mutation directions span a 19-dimensional space
*isotropically* (PR = 19.00, the maximum possible). No representation can beat cond = 1.
**The one-hot is already optimal in the conditioning sense, and any chemical block we add
can only degrade it.** That is the sentence this chapter has been missing.

**(ii) Descriptors collapse the mutation space from 19 effective directions to 3.8.**
Both descriptor arms have participation ratio ~= 3.8 while retaining nominal rank 16-19.
Because PC1 (volume, 46% var) and PC2 (hydrophobicity, 18%) dominate, the Gram's top
eigenvalue is 707 against a floor of 0.26. Gradient descent with a single learning rate
therefore moves ~52x faster along "got bigger / got greasier" than along the 17 remaining
chemical directions. **The model will learn a size-and-polarity rule quickly and be
effectively blind to everything else** — and the fine distinctions (D vs N, E vs Q, F vs
Y, measured as the three closest pairs at 13.6, 12.4, 14.3 against an alphabet mean of
37.5) are exactly the ones compressed into the slow directions. Those are precisely the
mutations where ddG prediction is hard and interesting.

**(iii) A 6.6x spread in ||delta|| across mutation types is an implicit, unrequested
reweighting of the loss.** Under one-hot, G->W and I->V present the same input magnitude,
and the model is free to learn that one matters more. Under descriptors, G->W arrives with
||delta|| = 82.0 and I->V with 17.2. A head of fixed norm produces a ~5x larger ddG for
G->W *before seeing any data*. This is a prior that conservative mutations have small
ddG — often true, which is exactly what makes it dangerous: it will look like it is
working while it is really encoding the same size/polarity heuristic, and it will fail
precisely on the stabilising conservative mutations that a ddG predictor exists to catch.
This also interacts with the a_p/b_p decomposition: an amplitude prior of this kind is a
**within-protein slope** effect, i.e. it lands on a_p, which is ddG-measurable.

**(iv) Why this kills Arm A specifically.** Under concat the model has both routes to the
same function: cond = 1 via one-hot, cond = 2724 via descriptors. Gradient descent is a
conditioning-greedy algorithm; it will fit through the one-hot route and leave the
descriptor weights near their initialisation, doing nothing but consuming weight decay.
Concat is therefore not merely a null — it is *predictably* a null, and we can say so
before running it.

**(v) Why this makes Arm C optimal, restated from the conditioning side.** Arm C touches
only the *initialisation* of `W_res`. The forward geometry the optimiser sees is still the
one-hot's cond = 1, isotropic, 19-effective-dimension geometry. We get the chemical prior
without importing the 132:1 (pca16) or 2724:1 (mordred726) anisotropy that would carry it.
**That is the entire argument in one line: put the chemistry in the starting point, not in
the coordinate system.**

If Arm B is nonetheless run, the mitigation implied by the table is to **whiten** the
block — use `U[:, :16]` rather than `U[:, :16] * S[:16]`. The committed CSV deliberately
does *not* whiten (header: `pca_scaling=NATURAL COMPONENT SCALE PRESERVED`, per-component
std PC1 2.726 -> PC16 0.237, ratio 11.4986). That choice preserves the eigenvalue
structure for interpretability, and it is the right default for *reporting* — but it is
the wrong one for *optimisation*, and whitening would take cond(G) from 132 to ~1 at the
cost of amplifying the 0.39%-variance noise directions. State the trade; do not silently
pick one.

---

## 5. IMPLEMENTATION

### 5.1 Layout and offsets (as built)

```
Fh = [ D(16) | Fb(32) | S(solv_dim) | Desc(K) | <siblings> | emb(1024) | one_hot(20) ]
                                      ^
                                      desc_start = solv_start + solv_dim = 48 + solv_dim
```

`model/hydro_net.py`: `self.desc_start = self.solv_start + self.solv_dim` (l.449),
`self.lig_start = self.desc_start + self.desc_dim + _sibling_block_dims(CFG)` (l.458).
Right-anchored indices `self.one_hot_index = -20` (l.481) and
`x_emb_features = x[:, self.llm_index:self.one_hot_index]` (l.549) are unmoved by
construction.

Layers that grow — **only** these two (l.462, l.465):

```
fc1_gcn: Linear(52 + solv + K + lig + proj_extra, 64)   # 52 = 32 dist + 20 one-hot
fc1_gat: Linear(36 + solv + K + lig + proj_extra, 64)   # 36 = 16 dist + 20 one-hot
```

Layers that must **not** change (l.468-472): `inst_norm1 = Normalization_layer(36 +
proj_extra)`, `inst_norm2 = Normalization_layer(2*(36+proj_extra))`,
`fc_in_dim = 2*(36+proj_extra) + post_gnn_emb`, `fc1 = Linear(fc_in_dim, 128)`.

The GCN branch reads `x[:, :32]` plus explicitly appended blocks (l.536-544); the
descriptor block is appended into `_extra` for **both** branches, so it reaches the GCN —
this is why W6 must extend that branch rather than bypass it.

`PEMGraphTransformer` slices **left-anchored** (l.850-855) and would silently ignore any
inserted block; it now raises. Keep it raising.

Widths: `mordred_pca16` 1092 -> **1108**; `mordred726` 1092 -> **1818**;
`mordred_pca16_only` **1092** (one-hot zeroed in place, not deleted).

### 5.2 Arm C — the one change to make

Arm C is the only new code. It is ~15 lines in `model/hydro_net.py` `PEM.__init__`, after
`fc1_gcn` / `fc1_gat` are constructed, gated on a new flag `--aa_desc_init {none,pca16}`
(default `none` => byte-identical):

1. Load `T = residue_descriptors_table('mordred_pca16')` -> `[20, 16]`, z-scored per §1.1.
2. Draw `R ~ N(0, 1)` of shape `[16, 64]`, seeded from `CFG.seed` so the run is reproducible.
3. `W_init = T @ R` -> `[20, 64]`; rescale `W_init *= (w_std / W_init.std())`, where
   `w_std` is the std of the *existing* initialised one-hot slice, so the init's overall
   scale is unchanged and only its **20-row correlation structure** differs.
4. Write into the one-hot slice of both first layers, transposed to `nn.Linear`'s
   `[out, in]` convention:
   - `fc1_gcn.weight[:, 32:52] = W_init.T`
   - `fc1_gat.weight[:, 16:36] = W_init.T`

   Those column ranges are the one-hot slice given `_sd = _dd = _ld = proj_extra = 0`;
   derive them from `self.one_hot_index` and the branch's own concat order rather than
   hard-coding, so they stay correct when a sibling block is enabled.
5. Leave biases untouched. Nothing else changes; no width change; no new parameters.

Do **not** combine Arm C with Arm A or B in the same run — C presumes the one-hot is
present and carrying identity.

### 5.3 What the gate must assert

Extend `scripts/gate_g4_cpu.py` with a `descriptors` section. Every assertion is
mechanical:

1. **Default is byte-identical.** `--aa_descriptors none` and `--aa_desc_init none`:
   `dG == -0.0030`, `width == 1092`. Compare a full forward against a stored baseline
   tensor with `torch.equal`, not `allclose`.
2. **Widths are exact.** `mordred_pca16` -> 1108; `mordred726` -> 1818;
   `mordred_pca16_only` -> **1092**. Assert `==`, not `>=`.
3. **Anti-silent-no-op — the assertion that actually matters.** For each non-`none` arm,
   assert the forward output **differs** from baseline:
   `assert not torch.equal(out_arm, out_base)`, plus
   `assert (out_arm - out_base).abs().max() > 1e-6`.
   A descriptor block that is inserted but never read would otherwise pass every width and
   finiteness check while changing nothing. This is exactly the `PEMGraphTransformer`
   left-anchored-slice failure mode, and it is the one that costs months.
4. **The block is non-constant and correctly placed.** Extract
   `x[:, desc_start : desc_start + K]` from a real `get_graph` call on a real sequence and
   assert: (a) `x_desc.std(dim=0).min() > 0` — it varies across residues, so it is neither
   zero nor constant; (b) for two positions with the **same** residue letter the rows are
   `torch.equal`; (c) for two positions with **different** letters they are not;
   (d) `torch.allclose(x_desc, one_hot @ T)` — it is the table, at the offset we claim.
5. **Right-anchored slices are unmoved.** For every arm, assert `x[:, -20:]` equals the
   one-hot (or is exactly zero under `mordred_pca16_only`) and that
   `x[:, llm_index:one_hot_index]` is `torch.equal` to the baseline's ProtT5 block. This is
   what catches a left-shifted LLM window.
6. **Replace really replaces.** Under `mordred_pca16_only`, assert
   `x[:, -20:].abs().max() == 0` **and** that the forward differs from baseline — zeroed
   *and* consequential.
7. **Frozen layers stayed frozen.** Assert `inst_norm1.weight.shape`,
   `inst_norm2.weight.shape` and `fc1.in_features` are **identical** across all arms.
8. **Arm C changes init only.** With `--aa_desc_init pca16`: assert `width == 1092`;
   assert total `sum(p.numel())` is **unchanged** vs baseline; assert the forward
   **differs** at step 0; assert the one-hot weight slice has rank <= 16. Then assert the
   *rest* of the parameters are `torch.equal` to the baseline init — only the one-hot slice
   moved.
9. **Provenance is enforced.** Assert the loaded CSV's header `final_shape` and column
   count match the mode name (already implemented via `descriptor_provenance`; keep it, and
   keep the retired names `pca16` / `curated12` / `pca16_only` **raising**, since
   `curated12` had come to load a 726-column matrix — two runs with identical logged config
   and different tensors is the worst failure mode in the repo).

After any change, `scripts/gate_g4_cpu.py` must still print **`G4-CPU: ALL PASS`**,
**`dG=-0.0030`**, **`width=1092`** on the default path.

### 5.4 Cost

| item | cost |
|---|---|
| Arm C implementation | ~15 lines + flag + gate section; no new tensors at train time |
| Arm C runtime | **zero** — init-only; width 1092 unchanged; no parameters added |
| Arm B runtime | zero added width; `fc1_*` input unchanged (one-hot zeroed in place) |
| Arm A runtime | `fc1_*` input +16 (pca16) or +726 (mordred726); the latter is +66% on the first layer |
| CSV build | already done, offline; `rdkit` / `mordred` never imported on the training path |
| Experiments | 4 runs (baseline, A-pca16, B, C) x seeds. Score on ddG/a_p, see §6 |

---

## 6. HOW IT IS SCORED

Per the metric rule: ddG cancels anything identical between WT and mutant. A descriptor
block is **not** such a thing — it changes in exactly the mutated row (§4), so W6 **is**
ddG-measurable and is the rare lever that does not have to fall back to dG. Report:

- **Primary: ddG PCC per-protein** (baseline 0.798), since the effect is within-protein.
- **Secondary: a_p.** §4(iii) shows the descriptor ||delta|| spread acts as an amplitude
  prior, which is a within-protein slope effect and therefore lands on a_p (median 0.4990;
  a_p correlates +0.5714 with per-protein PCC). If W6 does anything at all, this is where.
- **Not b_p.** The block is per-residue-type and averages to a near-constant over a whole
  protein, so it is a weak whole-protein lever; b_p belongs to the coil (dG MAE 4.9650 ->
  3.9521), not here. Do not claim W6 as a b_p lever.
- Report dG as well, so the arms stay comparable to every other lever on the board.

**Pre-registered prediction, to be written before the runs:** Arm A null (proved, §2 and
§4iv); Arm B neutral-to-slightly-worse (prior imposed irrevocably, cond 132); Arm C
neutral-to-slightly-better, with the smallest variance across seeds. If Arm A shows a
"gain", it is seed noise and must be checked against the seed spread (std(b_p) = 1.5741;
ICC(b_p) = 0.898) before it goes anywhere near the thesis.

---

## 7. THE PARAGRAPH FOR THE THESIS

> Physicochemical descriptors cannot add information to a model that already has a free
> per-residue embedding: the block is a deterministic linear function of the one-hot, so
> the two parameterisations span identical function classes. We verified that the matrix is
> rank 19 on 20 residues with participation ratio 3.82 — 726 Mordred columns carry at most
> 19 and effectively ~4 independent numbers, with PC1 = volume (r = 0.966) and PC2 =
> hydrophobicity (r = 0.695). We then analysed where a mutation actually enters, since ddG
> is a difference of two passes differing in one residue. The one-hot mutation delta has a
> perfectly conditioned Gram (cond = 1.0; all 190 pairwise deltas of equal norm; 19
> effective directions), whereas the descriptor delta has cond = 132 (PCA-16) or 2725 (full
> 726), collapsing to 3.8 effective directions with a 6.6x spread in delta magnitude across
> mutation types. Descriptors therefore make the mutation signal *worse* conditioned and
> impose an unrequested prior that conservative mutations have small effects. We conclude
> that the chemistry belongs in the **initialisation** of the residue embedding rather than
> in the feature vector, and we implement all three arms so that the claim is testable
> rather than asserted. We retain the open-alphabet lookup because it is the correct
> generalisation of Ofir's contribution, while noting that our 28-protein benchmark — all
> canonical, single-chain, ligand-free monomers — cannot reward it.
