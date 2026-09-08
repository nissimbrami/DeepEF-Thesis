# DESIGN: Attacking b_p Architecturally

Target: the per-protein offset b_p in absolute dG.
Prior work: reference-state levers (the coil) move it; the post-hoc corrector FAILED
out of sample (LOPO 0.6049 vs raw 0.5994 vs oracle 0.7119).
This document designs the third option: change the ARCHITECTURE or the LOSS so the
offset cannot form.

All numbers below were measured in this session, on the cluster, from the live tree.
Where a number contradicts the brief I say so explicitly and show the measurement.

---

## 0. WHAT WE ACTUALLY SEE

### 0.1 b_p is not a fitted intercept. It is one number per protein.

`scripts/calc_bp.py` defines it as the wild-type residual:

```
b_p = pred_deltaG(WT) - deltaG(WT)          # WT = the row with ddG == 0
```

One scalar per protein, from one row of the eval CSV. Not an OLS intercept over the
variants, not a slope. `a_p` (the ddG compression slope) is a different fit on a
different metric. This matters for everything below: b_p lives entirely in the
ABSOLUTE dG of a single structure.

### 0.2 The measured object, n=28, checkpoint `abl_calib_ctrl_repro2_e14`

```
std(b_p)              = 1.5741      (matches the brief exactly)
mean(b_p)             = 0.3542
range                 = -4.704 (2KVS) .. +2.747 (r12_757_TrROS_Hall)
std(true  WT dG)      = 0.9042
std(pred  WT dG)      = 1.3530
corr(true, pred)      = +0.0696     p = 0.725
```

**The predicted WT dG has MORE spread than the truth and is uncorrelated with it.**

### 0.3 Not a one-checkpoint accident

Across all 10 eval CSVs:

| checkpoint | r(true WT dG, pred WT dG) | p |
|---|---|---|
| abl_anchor_w0.3_s42_e14 | -0.1188 | 0.547 |
| abl_anchor_w1.0_s42_e14 | -0.0628 | 0.751 |
| abl_anchor_w3.0_s42_e13 | -0.0213 | 0.915 |
| abl_calib_ctrl_repro2_e14 | +0.0696 | 0.725 |
| abl_p3_slope3.0_s42_e8 | -0.1052 | 0.594 |
| abl_sigma_seed1_e13 | -0.0676 | 0.733 |
| abl_sigma_seed2_e10 | +0.0652 | 0.742 |
| abl_sigma_seed3_e13 | +0.0171 | 0.931 |
| abl_sigma_seed42_e9 | -0.0474 | 0.811 |
| abl_sigma_seed4_e14 | +0.0272 | 0.891 |
| **mean** | **-0.0244** | — |

Ten checkpoints, five seeds, three anchor weights, one slope arm. Not one reaches
|r| = 0.12. **The model has zero cross-protein signal on absolute stability.**

### 0.4 The variance decomposition, which settles what b_p *is*

Averaging b_p over 6 checkpoints to isolate the stable protein effect:

```
var(b_p) = var(pred) + var(true) - 2*cov(pred, true)
2.0243   = 1.1971    + 0.8479    - 2*(0.0104)
```

**cov(pred, true) = 0.0104, i.e. zero.** b_p is not a bias sitting on top of a correct
prediction. It is the sum of two independent variances. The model contributes its own
uncorrelated 1.20 of variance and fails to capture the 0.85 of real protein-to-protein
variation.

This is the most important fact in this document, and it kills the whole "correct the
offset" family. You cannot subtract a bias that carries no information about the
target. It also explains, exactly, why the corrector failed out of sample: LOPO 0.6049
vs oracle 0.7119. The oracle uses the held-out protein's own b_p, which encodes the
true dG (r = -0.639). No test-time-observable feature can reproduce that, because the
model's own prediction is independent of it.

### 0.5 b_p is highly reproducible, and that is the good news

```
mean pairwise corr of b_p vectors across 10 checkpoints = +0.8844
between-protein var / total var                          = 0.8375   (~ ICC 0.898 in the brief)
```

b_p is a real, stable, structural property of the (protein, architecture) pair -- not
training noise. It is deterministic and therefore attackable. It is just not attackable
by *regression on b_p*, because it is uncorrelated with anything observable at test
time.

### 0.6 What predicts the stable b_p (avg over 6 ckpts), n=28

| feature | r | p |
|---|---|---|
| **true WT dG** (NOT observable at test time) | **-0.6393** | **0.0003** |
| frac_buried_rel_lt_0.25 | -0.5337 | 0.0034 |
| SASA_over_len_pow_073 | +0.3609 | 0.0592 |
| SASA_per_residue | +0.3536 | 0.0649 |
| frac_charged | +0.3529 | 0.0655 |
| mean_rel_SASA | +0.3221 | 0.0946 |
| length | -0.2612 | 0.1795 |
| plddt / hydropathy / glycine / proline | abs(r) < 0.15 | ns |

Only two things clear p < 0.01, and one of them is the label itself.

### 0.7 CORRECTION TO THE BRIEF: b_p does NOT correlate with length

The brief states "b_p correlates with length (it does: r=-0.368 measured)" and builds
option (b) on it. Measured across all 10 checkpoints:

| checkpoint | r(b_p, length) | p |
|---|---|---|
| abl_anchor_w0.3_s42_e14 | **+0.0932** | 0.637 |
| abl_anchor_w1.0_s42_e14 | **+0.1604** | 0.415 |
| abl_anchor_w3.0_s42_e13 | **+0.0904** | 0.647 |
| abl_calib_ctrl_repro2_e14 | -0.3150 | 0.103 |
| abl_p3_slope3.0_s42_e8 | -0.1840 | 0.349 |
| abl_sigma_seed1_e13 | -0.2812 | 0.147 |
| abl_sigma_seed2_e10 | -0.1666 | 0.397 |
| abl_sigma_seed3_e13 | -0.3184 | 0.099 |
| abl_sigma_seed42_e9 | -0.2235 | 0.253 |
| abl_sigma_seed4_e14 | -0.2257 | 0.248 |
| **mean** | **-0.1371** | — |

**Sign flips: 7/10 negative, 3/10 positive. Not one checkpoint reaches p < 0.05.**
Contrast the burial -> b_p finding, which is 10/10 consistent. The -0.368 in the brief
is within the single-checkpoint noise band of a statistic whose across-checkpoint mean
is -0.137.

The mechanical reason is in the data: all 28 test proteins span **43-72 aa**. A 1.67x
range. There is not enough length leverage in this test set to drive a cross-protein
offset, whatever the theory says.

Option (b) is therefore **DROP**, not on cost but because its premise is false.
Section 2(b) gives a second, independent reason.

---

## 1. WHY DOES AN OFFSET FORM AT ALL?

The brief's framing: "a per-protein bias that appears in BOTH passes cancels in the
difference; an offset survives only if the bias is DIFFERENT between the two passes."
That framing is correct, and following it to the code produces a surprise.

### 1.1 What is actually different between the two graphs: 16 columns out of 1092

Measured by building both graphs on the same input and diffing column by column
(`train_utils.get_graph` vs `train_utils.get_unfolded_graph`, N=60):

```
total width                                    1092
columns that DIFFER folded vs unfolded:        [0..15]
n differing columns:                           16
Fb   (cols 16:48)   identical:  True
emb  (cols 48:1072) identical:  True
oh   (cols -20:)    identical:  True
```

Only the 16 D channels differ. This follows from the source. Both paths run:

```python
D  = get_dist_matrix(x); D = relu(exp(gaussian_coef*D**2))
# ... unfolded ONLY:  D = zero_except_udiagonal(D)
Fb = get_bonded_features(D)          # cols 16:48
D  = D.sum(dim=1); D = F.normalize(D, p=2, dim=0)
emb = F.normalize(_unfolded_emb(emb), p=2, dim=0)
```

`get_bonded_features` reads only `D[i, i+1]` and `D[i+1, i]` -- the tridiagonal.
`zero_except_udiagonal` **preserves the tridiagonal exactly**. So Fb is bit-identical
by construction. With `--unfolded_emb full` (the default), emb is bit-identical too.
one_hot never depended on structure.

**The unfolded state differs from the folded state in 1.5% of the input width.**

### 1.2 And even those 16 columns carry no scale difference

`F.normalize(D, p=2, dim=0)` normalizes each of the 16 channels to unit L2 norm **over
residues**. Verified:

```
D folded colnorm: [1.0000, 1.0000, 1.0000, 1.0000]
D unfold colnorm: [1.0000, 1.0000, 1.0000, 1.0000]
```

Before normalization the raw block sums differ a lot (folded 96.4 vs unfolded 58.1 at
N=60, ratio 0.60). After normalization that entire magnitude difference is destroyed.
**dG is driven purely by a 16-dimensional DIRECTION change on a unit sphere.** The
number of contacts a protein makes -- the most obvious physical correlate of stability
-- is normalized away before the network ever sees it.

### 1.3 The asymmetric-bias question, answered

Where does an asymmetric bias come from? Not from an input asymmetry, because there
almost isn't one. It comes from the network being NONLINEAR on the 1076 columns that
ARE identical.

`E = sum_i fc2(relu(fc1(x_i)))`. Write the folded and unfolded node features as
`x_i^f = (d_i^f, c_i)` and `x_i^u = (d_i^u, c_i)` where `c_i` is the 1076-dim common
part. Then

```
dG = sum_i [ f(d_i^u, c_i) - f(d_i^f, c_i) ]
```

If f were linear in its first argument with a coefficient independent of `c_i`, the
common part would cancel identically and dG would depend only on `d^u - d^f`. It is
not. The path from input to energy contains `relu` in fc1_gcn/fc1_gat and fc1, GAT
softmax attention, and two `InstanceNorm1d` layers (`inst_norm1`, `inst_norm2`) which
re-standardize over residues per graph. Every one of these makes the local slope
`df/dd` a function of `c_i`.

So:

> **The offset is not an asymmetric bias that failed to cancel. It is the common
> 1076-dimensional part -- ProtT5 embedding, one-hot, bonded features -- LEAKING into
> the difference through the network's nonlinearity, modulated by a 16-dimensional
> structural perturbation.**

The ProtT5 embedding is a fold-family fingerprint. It is bit-identical in both passes,
so a linear model would cancel it perfectly. Instead it acts as a per-protein gain on
the structural difference. That is exactly a per-protein multiplicative offset, and it
explains why b_p is reproducible (ICC 0.898 -- a deterministic function of the
protein's embedding) yet uncorrelated with true dG (nothing ever tied that gain to
stability).

This is also the mechanism the U2 `--unfolded_emb` lever was built to break, and the
W0 ablation already confirmed it from the other side: zeroing emb in the unfolded pass
drops var(E_u) to 0.331 of baseline and corr(E_u, wt_err) from 0.420 to 0.119. U2
works because it forcibly breaks the symmetry of the leak, not because emb was
"different" between passes -- it wasn't.

### 1.4 The scale catastrophe, measured

At random init (untrained PEM, CFG defaults, GAT cutoff 12A):

| N | E_f | E_u | dG | dG/N | abs(E_f)/N |
|---|---|---|---|---|---|
| 43 | -2.789 | -2.788 | +0.001 | 0.0000 | 0.0649 |
| 50 | -3.251 | -3.259 | -0.008 | -0.0002 | 0.0650 |
| 60 | -3.910 | -3.902 | +0.008 | 0.0001 | 0.0652 |
| 72 | -4.678 | -4.678 | +0.000 | 0.0000 | 0.0650 |

```
corr(N, E_f) = -1.0000        E is PERFECTLY extensive
corr(N, dG)  = +0.2626        dG is not
```

And the live gate agrees: `scripts/gate_g4_cpu.py` -> `dG=-0.0030 width=1092`.

Two things follow.

**(i) dG is a catastrophic cancellation.** `abs(E_f)/N = 0.0650` exactly, for every
length. E is a clean extensive quantity. But `dG = E_u - E_f` is ~0.003, while
`abs(E_f)` is ~3.9. **dG is 3 orders of magnitude smaller than the quantities being
differenced.** The model must produce a 0.1%-level difference between two large
extensive sums whose common part is 98.5% identical by construction. Any per-protein
drift in that common part -- exactly what the embedding leak produces -- lands directly
in dG at full size.

**(ii) The length argument for option (b) is already dead here.** `E_f` is perfectly
extensive but `dG` is not: `corr(N, E_f) = -1.0000` while `corr(N, dG) = +0.263`. The
extensivity cancels in the difference *at initialization*, before any training.
Dividing dG by N or sqrt(N) does not remove a length-driven offset, because the
length-driven part has already cancelled. It would divide the *signal* by a
protein-specific constant instead. See 2(b).

---

## 2. THE FOUR OPTIONS, SPECIFIED

### (a) Learned per-protein bias head

**Idea.** Predict b_p from protein-level features and subtract:
`dG_final = (E_u - E_f) - b_head(z_p)`.

**Concrete form.** Mean-pool the folded per-residue latents `h` (already available via
`f_type='features'`, `[B,N,128]`) to `z_p` in R^128; optionally concatenate observables
(length, mean_rel_SASA, frac_buried); `b_head = nn.Linear(128+k, 1)`. Trained jointly
on the dG loss.

**What it would condition on -- and why that is fatal.** Two separate reasons:

1. **The target is not learnable from observables.** Section 0.6: the only feature
   passing p < 0.01 is `frac_buried_rel_lt_0.25` (r = -0.534, R^2 = 0.285). The oracle
   needs r ~ -0.64 to the *true* dG. This is precisely the ceiling the post-hoc
   corrector already hit and failed at (LOPO 0.6049 vs raw 0.5994 -- a gain of 0.0055,
   i.e. nothing). Moving the same regression inside the network changes nothing about
   the information available to it.
2. **It is degenerate under the training loss.** `b_head` conditions on the folded
   protein only, so it is CONSTANT across the variants of a protein. Under
   `--loss_mode dg` the gradient can freely trade `b_head` against a global shift of
   `E_f`; under `--loss_mode ddg` it cancels exactly and receives ZERO gradient. There
   is no regime where it is both identifiable and useful.

**Verdict: DROP.** It is the failed corrector, re-implemented with more parameters and
a new way to be unidentifiable.

### (b) Length-normalised energy head (`--dg_length_norm`)

**Status.** The flag exists and is wired: `Trainer._dg_norm` in
`Megascale-fineTuning/train.py:1007`, applied in `get_deltaG` and `get_wt_deltaG`,
`choices=['none','n','sqrtn']`, default `'none'`. Zero implementation cost.

**Would it remove the offset? No, for two independent measured reasons.**

1. **b_p does not correlate with length** (Section 0.7): mean r = -0.137 over 10
   checkpoints, sign flips 7/10 vs 3/10, no p < 0.05. Range is 43-72 aa.
2. **The extensive part already cancels.** Section 1.4: `corr(N, E_f) = -1.0000` but
   `corr(N, dG) = +0.263`. Differencing removes extensivity before normalization could
   act on it.

**What it would actually do.** Divide a quantity that is already length-independent by
a per-protein constant N or sqrt(N). Over 43-72 aa that is a multiplicative rescale of
up to 1.67x (1.29x for sqrtn) applied *differently to each protein* -- i.e. it would
**inject** a per-protein scale error into dG, and simultaneously rescale every
protein's ddG by the same factor, corrupting a_p as collateral damage.

**Verdict: DROP.** The brief's premise for this option does not survive measurement.
Cheap is not the same as harmless: this one is actively harmful.

### (c) Siamese constraint tying the two passes

**Idea.** Force the two passes to share whatever is common so their biases must cancel.

**Why it is already true, and therefore a no-op.** The two passes ALREADY share every
weight -- there is one `PEM`, called once on `torch.cat([folded, unfolded], dim=0)`
(`train.py:958-964`). And Section 1.1 shows the inputs are already identical in 1076 of
1092 columns. The architecture is maximally Siamese already. There is no untied
parameter left to tie.

**The one non-trivial reading**, worth stating because it is nearly right: add a
penalty forcing the per-residue *latents* to agree except on the structural channels,
e.g. `L_siam = || h_i^u - h_i^f ||^2` over residues. But `dG = sum_i (e_i^u - e_i^f)`
and `e_i = fc2(h_i)`. Driving `h^u -> h^f` drives `dG -> 0` for every protein. The
penalty's minimum is the trivial model. You would have to exempt exactly the subspace
carrying the signal -- and identifying that subspace IS the original problem.

**Verdict: DROP.** Already satisfied in the trivial sense; degenerate in the
non-trivial sense.

### (d) Direct ddG head

**Status.** Already built: `--loss_mode ddg_head`, `Trainer.get_ddg_head`
(`train.py:971`), `f_type='features'` returning `h [B,N,128]`, and a `self.ddg_head`
module. No unfolded state, no energy.

**It removes the offset by construction** -- there is no absolute dG for an offset to
live in.

**What is lost: everything the thesis is named after.** DeepEF is an *energy function*.
`get_ddg_head` never calls `get_unfolded_graph`. It computes `ddg_head(h_wt, h_mut)`
from two FOLDED structures. There is no `E_u`, no `E_f`, no `dG = E_u - E_f`, no
reference state, no transferable energy. The coil work (U3-U6), the U2 embedding
lever, and the entire reference-state research programme become dead code. It is a ddG
regressor with a graph encoder -- a fine model, and not this thesis.

**Verdict: MAYBE, as a documented control arm.** See Section 4.

### (e) THE OPTION THE BRIEF DID NOT LIST -- and the one I recommend

Every option above accepts the framing "there is an offset; remove it." Section 0.4
says that framing is wrong: `cov(pred, true) = 0.0104`. There is no bias to remove. The
model has **no cross-protein dG signal to be offset in the first place.**

And Section 1.2 says why, mechanically:

> `F.normalize(D, p=2, dim=0)` erases the magnitude of the contact block in BOTH
> passes, before the network sees it. The folded/unfolded raw block sums differ by 40%
> (96.4 vs 58.1 at N=60); after normalization both are exactly 1.0.

**The number of contacts a protein makes is deleted from the input.** Absolute
stability is, to first order, contact count and buried hydrophobic surface. The model
is structurally blind to the leading term of the quantity it is asked to predict. That
is not an offset. That is a missing input.

This also explains the two facts nothing else explains together:
- **why b_p is reproducible (ICC 0.898)**: it is a deterministic function of the
  embedding leak (Section 1.3), fixed per protein;
- **why b_p is uncorrelated with true dG (cov ~ 0)**: the channel that carries true dG
  information was normalized to a constant before the network could use it.

And it is consistent with the one robust structural finding: burial -> b_p, r = -0.475,
10/10 checkpoints. Burial is the contact-count proxy. It predicts the error precisely
because the model cannot see it.

**Design: `--dg_scale_channels` -- restore the erased magnitude as an explicit block.**

Give back what `F.normalize` deleted, as its own input block, computed identically in
both passes so it is honest, and *differing* between them because the underlying
contact map genuinely differs.

**The block: 4 columns, per residue.**

For each residue i, from the SAME `D` tensor already computed in `get_graph` /
`get_unfolded_graph`, taken **after** the Gaussian kernel and masking but **before**
`D.sum(dim=1)` and `F.normalize`:

| col | quantity | definition |
|---|---|---|
| 0 | `log1p(contact_mass_i)` | `log1p(D[i,:,CA_CA].sum())`, CA-CA is channel 5 |
| 1 | `log1p(total_mass_i)` | `log1p(D[i].sum())` over all 16 channels |
| 2 | `n_contacts_i / 50.0` | `(D[i,:,5] > 0.5).sum() / 50.0` |
| 3 | `log1p(N_valid)/5.0` | chain length, broadcast to every residue |

Columns 0-2 are per-residue and DIFFER between the two passes (the unfolded path zeroes
off-tridiagonal entries, so contact mass collapses to ~2 neighbours). Column 3 is
identical in both passes and cancels to first order -- it is there so the network can
form an intensive quantity (per-residue contact density) internally rather than having
one imposed by `--dg_length_norm`. `log1p` keeps the block O(1) without destroying the
ratio information the way L2 normalization does.

Crucially this is **not** normalized over residues. That is the entire point.

**Tensor shapes and insertion offset.**

Node feature vector today, width 1092:
```
[ D(16) | Fb(32) | emb(1024) | one_hot(20) ]
   0-15   16-47     48-1071     1072-1091
```
With the block on, width 1096:
```
[ D(16) | Fb(32) | SCALE(4) | emb(1024) | one_hot(20) ]
   0-15   16-47     48-51      52-1075     1076-1095
```

The block goes BETWEEN Fb and emb, so the right-anchored `emb` and `one_hot` slices
(`self.llm_index = -(1024+20)`, `self.one_hot_index = -20`) never move. It is a new
sibling of the W5 solvation block and must take a slot in the SAME ordering that
`train_utils.get_graph` uses for `_blocks`:

```python
_blocks = [D, Fb] + ([] if _S    is None else [_S])     \
                  + ([] if _SCL  is None else [_SCL])   \   # <-- NEW, immediately after W5
                  + ([] if _Dsc  is None else [_Dsc])   \
                  + ([] if _L    is None else [_L])     \
                  + [emb, _oh]
```

Placing it after W5 and before W6 means `desc_start` and `lig_start` both shift by 4
when it is on. Both are computed, never hard-coded:
- `PEM.__init__`: `self.scale_dim = 4 if CFG.dg_scale_channels else 0`;
  `self.scale_start = self.solv_start + self.solv_dim` (= 48 + 0 or 48 + 3);
  then `self.desc_start = self.scale_start + self.scale_dim`.
- `self.lig_start` already reads `_sibling_block_dims(CFG)`, which is the ONE authority
  for this arithmetic (`model/hydro_net.py:470`) -- **`_sibling_block_dims` must be
  updated to include `scale_dim`, or W11 will silently read the wrong columns and never
  crash.** This is the single highest-risk line in the change.

**Which layers grow, and which must not.**

Both branches read the block, because both are structurally blind:
```python
self.fc1_gcn = nn.Linear(52 + _sd + _scl + _dd + _ld + proj_extra, 64)   # 68 -> 72
self.fc1_gat = nn.Linear(36 + _sd + _scl + _dd + _ld + proj_extra, 64)   # 52 -> 56
```
(Verified live at CFG defaults with `emb_projection="mlp"`, `proj_extra=16`:
`fc1_gcn.in_features = 68`, `fc1_gat.in_features = 52`.)

**Unchanged:** `fc2_gcn`, `fc2_gat`, `inst_norm1` (36+proj), `inst_norm2`
(2*(36+proj)), `fc_in_dim`, `fc1`, `fc2`. Only `fc1_*` grow.

`PEM.forward` extends the existing `_extra` list, in the same order as `_blocks`:
```python
if getattr(self, 'scale_dim', 0):
    _extra.append(x[:, self.scale_start:self.scale_start + self.scale_dim])
```
inserted immediately after the `solv_dim` append and before the `desc_dim` append.
Because `_extra` is non-empty, `forward_gat` takes the already-existing "input wider
than baseline" branch that projects before the residual -- no new branch is needed.

`PEMGraphTransformer` slices LEFT-anchored and already raises on any inserted block.
Nothing to do; the guard covers this.

**Flag.**
```python
_p.add_argument('--dg_scale_channels', action='store_true',
                help='restore the contact-magnitude information that F.normalize(D, p=2, dim=0) '
                     'erases: 4 unnormalised columns [log1p contact mass, log1p total mass, '
                     'n_contacts/50, log1p(N)/5] inserted between Fb and emb. Default off = '
                     'bit-identical to baseline.')
CFG.dg_scale_channels = _a.dg_scale_channels
```
Default off, bit-identical when off, following the house pattern exactly.

**Why this is a b_p lever and not an a_p lever.** Columns 0-2 differ between the folded
and unfolded passes, so they survive `E_u - E_f`. They are near-constant across the
point mutants of a single protein (a single substitution barely moves total contact
mass), so they largely cancel in ddG. That is the correct signature for a b_p lever
under THE METRIC RULE: it must be scored on dG or b_p, and it is.

---

## 3. RANKING: P(work) x cost

| # | option | P(work) | cost | expected value | verdict |
|---|---|---|---|---|---|
| **(e)** | **`--dg_scale_channels`** | **~0.45** | **~40 lines + 1 run** | **highest** | **DO-IT** |
| (d) | `--loss_mode ddg_head` | ~0.85 on ddG; 0 on b_p | already built | high but off-target | MAYBE (control arm) |
| (b) | `--dg_length_norm` | ~0.02 | zero | negative | DROP |
| (a) | learned b_p head | ~0.05 | ~80 lines | negative | DROP |
| (c) | Siamese constraint | ~0.0 | ~30 lines | zero | DROP |

**(e) is the single best one.** It is the only option that addresses the measured cause
-- `cov(pred, true) = 0.0104`, a MISSING INPUT -- rather than the assumed cause (a bias
to subtract). It is the only one that is cheap, gated, default-off, metric-legal, and
consistent with the one 10/10-reproducible structural finding in the project (burial ->
b_p). And it preserves the energy interpretation completely: it adds inputs, not a
correction term. dG is still `E_u - E_f` from the same network on two graphs.

P(work) = 0.45 and not higher because the leak of Section 1.3 is a second, independent
mechanism that this does not touch. Restoring the erased magnitude is necessary; it may
not be sufficient. The honest expected outcome is that
`corr(true WT dG, pred WT dG)` moves off zero -- say to 0.3-0.5 -- and std(b_p) drops
from 1.57 toward ~1.2. **That is the number to watch, and it is a better primary
endpoint than std(b_p) itself**, because it is what "the model knows something about
absolute stability" actually means. A lever that shrinks std(b_p) while leaving corr at
zero has only shrunk the predictions toward their mean.

**Stacking note.** (e) and U2 `--unfolded_emb zero` attack the two different mechanisms
of Section 1 -- missing magnitude and embedding leak. They are the natural 2x2
factorial and should be run as one, not sequentially.

---

## 4. THE HONEST CAVEAT: can the thesis accept (d)?

**No, not as the main result. Yes, as a labelled control arm.**

DeepEF is an energy function. The claim that gives the thesis its scientific content is
that a network computes a transferable energy `E(structure, sequence)` such that
`dG = E_unfolded - E_folded` is a physical free energy of folding. Every artifact of
the project depends on that: the reference-state programme (U2-U6, the Flory coil), the
whole notion of an unfolded state, the transferability argument to unseen folds, and
the pre-training on the 100k PDB decoy set -- which trains an ENERGY, and which
`--loss_mode ddg_head` cannot consume, because that path never builds an unfolded graph
and never produces a scalar energy.

`--loss_mode ddg_head` removes b_p by removing dG. Adopting it as the main result would
mean the thesis reports a ddG regressor and calls it an energy function. It would also
make the best measured result in the project -- the coil, dG MAE 4.9650 -> 3.9521 --
unreportable, because there would be no dG to have an MAE on.

**What it IS legitimately good for:** an upper bound. Run it, report its per-protein
ddG PCC next to the energy model's 0.798, and state plainly: *"a direct ddG head, which
abandons the energy interpretation, achieves X; the energy formulation costs us
X - 0.798 in ddG accuracy and buys the reference state, the transferability claim, and
an absolute dG."* That is a strong, honest paragraph and it makes the thesis better.
Reporting it as the main model is a different thesis.

There is also a cleaner way to say the same thing, which the data supports: **b_p and
a_p are different problems and the thesis should stop treating them as one.** a_p is
within-protein, ddG-measurable, and already good (per-protein PCC 0.798). b_p is
cross-protein, dG-only, and currently at r = -0.024 -- the model has *no* cross-protein
signal. Option (d) is an excellent a_p architecture and a vacuous b_p one. Option (e)
is the b_p one.

---

## 5. THE GATE

`scripts/gate_dg_scale.py`, run before any training. `scripts/gate_g4_cpu.py` must
still print `dG=-0.0030 width=1092` on the baseline row.

### The anti-silent-no-op check (the one that matters most)

Every failure mode of this lever is silent. A wrong offset feeds the wrong columns into
`fc1`; an all-zero block trains happily; a block identical in both passes cancels in
`E_u - E_f` and does nothing while looking fine. The gate must make each of those a
hard error.

```
A. OFF is bit-identical
A.1  CFG.dg_scale_channels=False -> get_graph width == 1092
A.2  torch.equal(graph_off, graph_baseline)  for both folded and unfolded
A.3  gate_g4_cpu.py baseline row still prints dG=-0.0030 width=1092

B. Widths
B.1  ON  -> width == 1096 exactly (1092 + 4)
B.2  fc1_gcn.in_features == 68 + 4 == 72
B.3  fc1_gat.in_features == 52 + 4 == 56
B.4  fc2_gcn.in_features, fc2_gat.in_features UNCHANGED
B.5  inst_norm1, inst_norm2, fc1.in_features UNCHANGED
B.6  with --burial_features also on: width == 1099, scale_start == 51

C. Placement -- the block is where the model reads it
C.1  torch.allclose(graph[:, PEM.scale_start : PEM.scale_start+4], block_built_directly)
C.2  emb slice x[:, -1044:-20] identical ON vs OFF        (right-anchor unmoved)
C.3  one_hot slice x[:, -20:] identical ON vs OFF          (right-anchor unmoved)
C.4  _sibling_block_dims(CFG) includes scale_dim: with --dg_scale_channels
     AND --ligand_nodes, ligand_features.ligand_start == PEM.lig_start.
     ***This is the assertion that catches the W11 silent-corruption bug.***

D. ANTI-SILENT-NO-OP  (each is a way the lever does nothing while passing)
D.1  the block is not all-zero:     block.abs().sum() > 0            folded AND unfolded
D.2  the block is not constant:     block.std(dim=0) > 1e-6 for cols 0,1,2
D.3  THE CORE ONE -- the block DIFFERS between the two passes:
       (blk_f[:, :3] - blk_u[:, :3]).abs().max() > 0.1
     If this fails the block cancels in E_u - E_f and the lever is a no-op.
D.4  col 3 (length) IS identical between passes:
       torch.equal(blk_f[:, 3], blk_u[:, 3])
D.5  the block is NOT L2-normalised over residues -- the whole point:
       abs(blk[:, 1].norm() - 1.0) > 0.01
D.6  monotone in contact count: build a compact structure and an extended one at the
     same N; the compact one has strictly larger contact mass.
       blk_compact[:, 1].mean() > blk_extended[:, 1].mean()
     (The analogue of the W7 rbf_expand k(2A) > k(8A) > k(15A) assertion that caught
     the flat-kernel bug: it proves the channel encodes what it claims.)
D.7  gradient reaches it: dG.backward(); fc1_gcn.weight.grad[:, scale_cols_in_gcn_slice]
     .abs().sum() > 0, and likewise for fc1_gat.
D.8  finiteness at the edges: N=1 and an all-zero mask produce finite output
     (log1p of 0 is 0; assert no NaN/Inf).

E. Metric legality
E.1  on a WT/mutant pair of the SAME protein, the block changes by < 5% of its
     folded/unfolded difference -- confirming this is a b_p lever (cross-protein,
     dG-scored) and not a disguised a_p lever.
```

**Post-run assertion, on the eval CSV, not the gate:** the primary endpoint is
`corr(true WT dG, pred WT dG)` over the 28, which is `-0.024` on average today. The
lever is a success only if that moves decisively off zero. Report std(b_p) alongside,
never alone.

---

## 6. WHAT I WOULD NOT CLAIM

- I did not train anything. P(work) = 0.45 is a judgement, not a measurement.
- The `_sibling_block_dims` interaction with W11 is the one place this design can
  silently corrupt an existing lever. Gate check C.4 exists for exactly that and is not
  optional.
- Section 1.3's leak mechanism is an argument from the code's nonlinearities plus the
  W0 ablation, not a direct measurement of the leak itself. A clean test exists and is
  cheap: freeze the trained model, run both passes with `emb` replaced by a per-protein
  constant, and see how much of std(b_p) survives. Worth doing before the training run.
- The brief's r = -0.368 for length is reproducible on ONE checkpoint
  (`abl_calib_ctrl_repro2_e14`, where I measure -0.315). It does not survive the other
  nine. I would not build a lever on it, and Section 2(b) explains the mechanical
  reason it cannot be real over a 43-72 aa range.
