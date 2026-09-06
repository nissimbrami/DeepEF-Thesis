# MASTER IMPLEMENTATION PLAN — code level

**Verified against `nissimbrami/DeepEF-Thesis`, branch `current-vers`.** Every line number and
variable name below was read from that branch. Where I could not verify something, it is marked
`[VERIFY]` and the agent must check before applying.

**How to use this.** Each work item has: what it changes, the code, the flag, the verification that
must pass before it is used, and what result would falsify it. **Nothing enters a factorial before
its verification block passes.**

---

# PART I — THE REGISTRY

Everything considered in this project, with its disposition. Do not re-propose a closed item.

## I.1 The diagnosis

`pred ≈ a_p · true + b_p` per protein. Removing both with an oracle was believed to give
0.77–0.81; **that is not reproducible** (7/7 of Shahar's checkpoints and 5/5 of ours fail, and the
estimator is non-monotone). **The defensible ceiling is offset-removal alone: 0.70–0.72, stable
across twelve checkpoints from two systems.**

- `b_p` = the model's error on wild-type ΔG. Constant within a protein, so it vanishes from
  per-protein ranking and poisons pooling.
- `a_p` = compression. Collapses to 0.07–0.18 on designed folds; **no designed fold exceeds 0.646.**
- Measured: `corr(a_p, per-protein PCC) = +0.60` — slope predicts performance better than anything
  else.

**Where the offset is manufactured:** 77–88% of across-protein ΔG variance sits in `E_unfolded`,
`corr(E_u, wt_err) = 0.865`, `var(E_u)` is 6–10× `var(E_f)`, and `corr(wt_err, length) ≤ 0.12`.
**So it is neither geometric nor extensive — it is sequence content.**

## I.2 Reference numbers

| Quantity | Value | Note |
|---|---|---|
| `calib_ctrl` reference | pooled **0.606** / PP **0.711** | via `train.py` |
| Our reproduction | pooled **0.591** / PP **0.731** / std(b) 0.223 | PP exceeds reference |
| σ (5 seeds) | **0.0072** pooled, **0.0037** PP | → 3 seeds per cell suffices |
| Thesis baseline | 0.531 | ΔG objective |
| 0.655 | **retracted** | test-peeked; author states val-selection gives ~0.63 |

## I.3 Lever registry

| # | Lever | Attacks | Status |
|---|---|---|---|
| A | `--wt_anchor_weight` | `b` | exists, run: std(b) 0.29→0.25 |
| B | `--designed_weight` | `a` | exists; **A+B gave the largest offset drop (0.19) and the largest slope collapse — an interaction** |
| C | `--slope_weight` | `a` directly | exists (`train.py` 76, 80, 433–444), **never run**. Nissim's contribution |
| D | coil | `b` at source | implemented; CLI flag added by FIX 1 |
| 5 | burial / solvation | PP | **not built** — W5 below |
| 6 | Mordred descriptors | PP | **not built** — W6 below |
| 7 | pairwise edges + span | PP | **not built** — W7 below |
| 8 | pLDDT | PP | **not built** — W8 below |
| 9 | metal coordination | generality | **not built** — W9 below |

## I.4 Closed on evidence — do not re-propose

Interfaces / BSA (**corr −0.001 over 100,246 structures**) · complexity (0.2 of 18) ·
pooled-correlation loss · length normalisation · learned readout · post-hoc offset correction ·
more MegaScale · FireProtDB / ProTherm (S669 leakage) · ESM-2 fusion · graph-transformer backbone ·
contact-order head (R = 0.40, and it cancels exactly in `output − output[0]`) · near-native decoys
(no data for fine-tuning proteins) · pretraining as a headline lever (0.65→0.82 is a **validation**
figure) · SaProt as a one-flag swap (generator emits `[1,1280]`, not `[L,1280]`).

## I.5 Code defects, all confirmed by reading the branch

| ID | Defect | Effect |
|---|---|---|
| U1 | Unfolded state = folded coords with non-local contacts deleted | Not an unfolded chain |
| U2 | **ProtT5 feeds the unfolded pass** | Unfolded state knows its fold family |
| U3 | Coil broadcasts one Cα distance over all 16 atom-pair channels | States trivially separable |
| U4 | Coil's `b` fitted from the folded structure | Re-injects folded geometry |
| U5 | GAT k-NN topology built from folded coords for **both** states | Unfolded keeps folded contact graph |
| U6 | `ca_coords` expanded across both halves | Phase 7 and coil cancel each other |
| U7 | Burial cancels unless zeroed unfolded; and `/N` normalisation | Feature inert + length confound |
| U8 | RBF bank summed, not concatenated | Measured: 5 Å and 15 Å both → 4.649, model blind to distance |
| U9 | `energy_terms=K` then sum ≡ `K=1` | Algebraic no-op |
| U10 | `gcn_span=2` makes span-1 bidirectional too | Confounds two changes |

---

# PART II — THE UNFOLDED STATE: diagnosis, decision procedure, and code

This is the part that was specified least well before. Here it is in full.

## II.1 The problem, stated exactly

`ΔG = E_unfolded − E_folded`, and `E_unfolded` carries 77–88% of the across-protein variance —
i.e. **the offset is manufactured there.** Four channels feed it:

| Channel | Width | Varies between proteins? | Fixed by coil? |
|---|---|---|---|
| `D` distance | 16 | yes (folded geometry) | **yes** |
| `Fb` bonded | 32 | yes | **yes** |
| `one_hot` | 20 | yes (composition) | no |
| `emb` ProtT5 | 1024 | yes (fold family) | **no** |

**Length explains ≤ 0.12 of `wt_err`.** So the variance is not extensive — not the sum over
residues, not the size. **It is sequence content, and only `one_hot` and `emb` carry that.**

**Therefore the coil alone may not fix the offset.** It repairs the two channels that were probably
not the cause and leaves the two that probably were. **This must be tested before spending twelve
factorial cells on it.**

## II.2 The decision experiment — W0, no training, one hour

Take an existing checkpoint. Compute `E_unfolded` per protein under four ablations of the unfolded
pass only. Compare `var(E_u)` across proteins and `corr(E_u, wt_err)`.

```python
# scripts/w0_unfolded_channel_ablation.py
# For each of the 28 test proteins, WT variant only, one forward pass each.
#   base   : unfolded graph as-is
#   noemb  : ProtT5 block zeroed in the unfolded graph only
#   noOH   : one-hot zeroed in the unfolded graph only
#   coil   : CFG.flory_unfolded = True
# Report per condition: var(E_u) across proteins, corr(E_u, wt_err), corr(E_u, length)
```

**Decision table — commit to it before running:**

| Result | Meaning | Action |
|---|---|---|
| `noemb` collapses `var(E_u)` by >50% | The offset is the language-model channel | **U2 fix becomes lever D′ and replaces the coil in the factorial** |
| `noOH` collapses it | Composition and the extensive sum | Coil is right; add per-length normalisation of `E_u` |
| `coil` collapses it | Geometry after all | Coil stays as factor D, as planned |
| Nothing collapses it | The variance is in the learned weights, not the input | Escalate — the whole offset story needs re-examination |

**This is one hour of compute and it decides how 12 of 48 factorial cells are spent.** It is the
single highest-value action available.

## II.3 The fixes, as code

### W0-fix / U2 — make the unfolded state fold-blind

**Rationale.** An unfolded chain has no fold. It should not know which family it came from.
Fold-family memorisation is precisely the diagnosed failure (WT ΔG correlation 0.86
in-distribution, ~0.07 out-of-distribution, monotone in homology to the training set).

In `train_utils.get_unfolded_graph` and `_flory_unfolded_graph`:

```python
UNFOLDED_EMB = getattr(CFG, 'unfolded_emb', 'full')   # 'full' | 'zero' | 'mean'

def _unfolded_emb(emb):
    if UNFOLDED_EMB == 'zero':
        return torch.zeros_like(emb)
    if UNFOLDED_EMB == 'mean':
        # keep global scale, remove per-residue identity
        return emb.mean(dim=0, keepdim=True).expand_as(emb)
    return emb                                        # 'full' = current behaviour
```

then use `_unfolded_emb(emb)` in place of `emb` in the unfolded concatenation only.

Flag: `--unfolded_emb {full,zero,mean}`, default `full` (bit-identical to today).

**`mean` is the interesting middle case:** it keeps whatever global scale ProtT5 contributes while
removing per-residue identity. If `zero` hurts but `mean` helps, the embedding was carrying scale,
not just identity.

### U5 — coil-consistent edge topology

The k-NN edge set is built from folded Cα distances and reused for the unfolded half, so the
"unfolded" graph keeps the folded **contact topology** even after node features are zeroed.
**Without this fix the coil is half-applied**, and that alone may explain a weak coil result.

```python
# in the model's edge construction, when building the UNFOLDED half:
if getattr(CFG, 'flory_unfolded', False) and getattr(CFG, 'coil_edges', False):
    # chain-local only: a random coil has no defined long-range contacts
    idx = torch.arange(N, device=dev)
    src = torch.cat([idx[:-1], idx[1:]])
    dst = torch.cat([idx[1:],  idx[:-1]])
    edge_index_unfolded = torch.stack([src, dst])
else:
    edge_index_unfolded = edge_index_folded          # current behaviour
```

Flag: `--coil_edges`, default off. **Only meaningful with `--flory_unfolded`; assert that.**

### U3 / U4 — the coil's two open design choices, as arms not silent defaults

**U3:** the coil expands one Cα distance across all 16 atom-pair channels, so in the coil state all
16 become identical. After the row-sum, folded and unfolded live on different sub-manifolds and are
**trivially separable** — the network can detect which state it is in rather than computing an
energy.

```python
COIL_CHANNELS = getattr(CFG, 'coil_channels', 'broadcast')  # 'broadcast'|'ca_only'|'offset'
# broadcast : current — same d in all 16
# ca_only   : Ca-Ca channel gets d, the other 15 are zero
# offset    : each channel = d + fixed intra-residue offset measured from the folded set
```

**U4:** `b` is fitted from the protein's own folded mean Cα–Cα distance, which re-injects folded
geometry into the reference state.

```python
COIL_B = getattr(CFG, 'coil_b', 'fitted')    # 'fitted' | 'fixed'
b = protein_mean_ca_ca if COIL_B == 'fitted' else 5.82   # Å, experimentally calibrated
```

**Run both as a 2×2 sub-arm of factor D, four cells, one seed each** — this settles two design
questions for four runs instead of leaving them as untested defaults.

### U6 — per-half `ca_coords`

```python
# was: ca_coords = ca_pos.unsqueeze(0).expand(all_graph.size(0), -1, -1)
ca_f = ca_pos.unsqueeze(0).expand(B, -1, -1)
ca_u = coil_ca_positions(seq_len, b, nu) if CFG.flory_unfolded else ca_f
ca_coords = torch.cat([ca_f, ca_u], dim=0)
```

**Blocking:** do not run Phase 7 together with the coil until this lands. They cancel.

---

# PART III — THE INFORMATION LEVERS, as code

## W5 — burial and solvation

**External support:** DeepDDG, trained on 5700 curated mutations and beating eleven methods,
found the **solvent accessible surface area of the mutated residue to be its single most important
input**, concluding that buried hydrophobic area is the major determinant of stability. Our model
has no equivalent feature at all.

```python
# train_utils.py
_KD = torch.tensor([1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,
                    1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3])  # AA_MAP order
_KD_NORM = (_KD + 4.5) / 9.0
_BURIAL_CAP, _BURIAL_R = 30.0, 10.0

def compute_burial(x, mask, radius=_BURIAL_R, cap=_BURIAL_CAP):
    """Cbeta neighbour-count burial → [N,1] in [0,1]. Normalised by a CONSTANT, never by N:
    burial is local and does not scale with chain length."""
    cb = x[:, 3, :]
    d  = torch.cdist(cb, cb)
    v  = (mask > 0).float()
    w  = (d < radius).float() * v.unsqueeze(0) * v.unsqueeze(1)
    w  = w - torch.diag_embed(torch.diagonal(w))
    return (w.sum(1, keepdim=True) / cap).clamp(0, 1) * v.unsqueeze(1)

def solvation_features(x, one_hot, mask, folded=True):
    """[N,3] = burial, hydrophobicity, burial*hydrophobicity.
    Burial is ZERO in the unfolded state — an extended chain buries nothing, and that
    difference IS the hydrophobic driving force. A previous implementation used the same
    coordinates in both states, so the column cancelled exactly in E_u - E_f."""
    hyd = one_hot @ _KD_NORM.to(one_hot.device).unsqueeze(1)
    bur = compute_burial(x, mask) if folded else torch.zeros_like(hyd)
    return torch.cat([bur, hyd, bur * hyd], dim=1)
```

The **product term matters**: a linear layer cannot construct a product from its factors, and
buried × hydrophobic is the term that carries the driving force.

**Slicing.** The block sits between `Fb` and `emb`, so left-anchored indices shift and
right-anchored ones do not:

```python
self.solv_dim   = 3 if CFG.burial_features else 0
self.solv_start = 48                      # after D(16) + Fb(32)
self.llm_index  = -(CFG.emb_input_dim + 20)   # unchanged
self.one_hot_index = -20                      # unchanged
gcn_fc_in = 16 + 16 + 20 + self.solv_dim
gat_fc_in = 16 + 20 + self.solv_dim
```

Flag `--burial_features`. **Predicted PP +0.02 to +0.05.**

## W6 — Mordred descriptors

Ofir Ezrielev's method: chiral SMILES → Mordred (1826 descriptors) → drop those with missing
values (1280) → keep those with ≥40 unique values across the AA set (654) → normalise. **His
central result is that a model trained on canonical residues predicts non-canonical effects,
because the physicochemical space is continuous.**

**Deviation, stated:** with 20 residues rather than 58 the maximum possible unique count is 20, so
the threshold scales to **15** — a descriptor must distinguish three quarters of the alphabet.

Build once, offline, to `data/aa_descriptors.csv` with a provenance header. Two arms, `pca16` and
`curated12`, plus a third `pca16_only` that removes one-hot entirely.

**Default is to append, not replace.** The literature is consistent: chemical encodings alone are
"as effective as one-hot", while the **ensemble of one-hot and chemical encodings improves
accuracy** — they are complementary. One-hot gives exact identity; descriptors give metric
structure.

**Verification that decides everything, before training:**

```python
# nearest neighbours in descriptor space must be chemically sensible
# REQUIRED: L → {I,V,M};  D → {E,N};  F → {Y,W}
# If aspartate comes out near leucine, the matrix is wrong and no training will fix it.
```

Flag `--aa_descriptors {none,pca16,curated12,pca16_only}`. **Predicted PP +0.01 to +0.03.** The
larger prize is that the alphabet stops being closed — any residue with a SMILES string gets a
vector, so phosphoserine and non-canonical residues become representable instead of becoming a row
of zeros.

## W7 — pairwise edges and reachable secondary structure

```python
def rbf_expand(d, n=16, lo=0.0, hi=20.0):
    """CONCATENATE the bank. Never sum it: a previous implementation summed the M responses
    back to the original width, and a 5 A contact and a 15 A non-contact both returned 4.649 —
    the kernel went flat past 5 A and the model became blind to distance."""
    c = torch.linspace(lo, hi, n, device=d.device)
    w = (hi - lo) / n
    return torch.exp(-((d.unsqueeze(-1) - c) ** 2) / (2 * w ** 2))
```

Edge attributes: source one-hot (20) + destination one-hot (20) + 16 RBF = **56**, passed as
`edge_dim` to `GATv2Conv`.

**Assertion before any use:** `k(2Å) > k(8Å) > k(15Å)`, strictly. One line, catches U8 completely.

**Extended connectivity.** The GCN edge set is `(i, i+1)` only; with three layers the chain reach
is three residues. **An α-helix is defined by i→i+4 and a β-sheet by i→i+2, so secondary structure
is currently unrepresentable.** Extend to `|i−j| ≤ 4` — and **make span-1 bidirectional in the
control arm too**, or the comparison confounds two changes (U10).

Flags `--edge_features`, `--rbf_centers 16`, `--gcn_span 4`. **Predicted PP +0.01 to +0.03.**

## W8 — structure quality

Resolution was the strongest annotation in the 100k catalogue (+0.158). For AlphaFold structures
the analogue is pLDDT, available and unused. One node feature, normalised.

**And report every metric split by pLDDT tercile.** If error concentrates in the low-confidence
tercile, that is a limitation of the *input structures*, not the model — worth a paragraph
regardless of whether the feature helps. Flag `--struct_quality`. **PP +0.00 to +0.02.**

## W9 — metal coordination

**Not the same thing as interface BSA**, which was dropped on measurement: a dative bond from a
side chain to a metal ion is a different physical term from buried surface between chains.

Two forms. *Feature:* per-residue first-shell flag, metal identity, coordination number.
*Graph, and more principled:* **the metal ion becomes a node** with its own type and typed
coordination edges — which is what the graph formalism is for.

**Honest caveat:** MegaScale is small soluble domains, mostly without metal sites. **Generality
lever, not a benchmark lever. PP ~0 here.** Needs Hadar's annotations — ask now for a list and a
date.

---

# PART IV — VERIFICATION PROTOCOL

Every work item passes all four gates before it enters a factorial. **No exceptions.**

**G1 — inert when off.** With the flag at its default, the node tensor is byte-identical to the
current build on a fixed input. `torch.equal`, not `allclose`.

**G2 — it does what it is named for.** The specific assertion per item:

| Item | Assertion |
|---|---|
| W5 burial | The burial column **differs** between folded and unfolded graphs of the same protein |
| W5 burial | `corr(compute_burial, Shrake-Rupley SASA)` with `|r| > 0.7` over ≥200 residues |
| W5 burial | `corr(mean burial, chain length)` across proteins ≈ 0 — no length confound |
| W6 descriptors | L's nearest neighbours ⊂ {I,V,M}; D's ⊂ {E,N}; F's ⊂ {Y,W} |
| W7 RBF | `k(2Å) > k(8Å) > k(15Å)` strictly |
| W7 + coil | The unfolded half's edge distances **differ** from the folded half's |
| U2 | With `--unfolded_emb zero`, the unfolded graph's embedding block is all zeros and the folded one is not |
| U5 | With `--coil_edges`, the unfolded edge count equals `2(N−1)`, not the k-NN count |

**G3 — no silent interaction.** Enumerate which flags are known to conflict and assert:
`--edge_features` with `--flory_unfolded` requires the U6 fix; `--coil_edges` requires
`--flory_unfolded`; `--slope_weight` with `--loss_mode dg` is undefined (the slope term is defined
on ΔΔG) — raise rather than run.

**G4 — one epoch on GPU** with `preflight.py` active, all eight assertions passing.

**Continuous control during runs:** `monitor.py` attached, stop rules armed, epoch-2 reproduction
check, `calib_diag.py` on every evaluation reporting `std(b)` and the slope distribution.

---

# PART V — EXECUTION ORDER

| # | Item | Cost | Gate |
|---|---|---|---|
| 1 | **W0 channel ablation** | 1 h, no training | Decides how factor D is spent |
| 2 | U5 + U6 fixes | small | G1, G2 |
| 3 | Calibration factorial 2⁴ × 3 seeds | 48 runs, ~1.5 d | A/B/C/D main effects + 6 interactions |
| 4 | U3/U4 coil sub-arms 2×2 | 4 runs | Settles two design questions |
| 5 | W5 burial | build + 8 runs | G1–G4 |
| 6 | W6 descriptors | build + 8 runs | G2 nearest-neighbour check is decisive |
| 7 | W8 pLDDT | 6 runs | — |
| 8 | Information factorial W5×W6×W7, 2³ × 3 | 24 runs, ~18 h | Main effects + interactions |
| 9 | W9 metals | when annotations arrive | — |

**Everything after step 3 runs on top of the winning calibration configuration**, never the bare
baseline, or the two result sets cannot be combined.

**Budget:** ~115 runs, ~690 GPU-hours, **4–5 days wall-clock at 8 concurrent jobs**, realistically
**8–12 calendar days** with queueing, failures and analysis. RTX 6000 ≈ 6 h per 15-epoch run;
**never `ise-pheno`** (3.5× slower under contention) and never mix node classes within a cell.

**If the queue tightens:** a 2⁴⁻¹ half-fraction gives all four main effects and unconfounded
two-factor interactions in 8 cells instead of 16 — 24 runs instead of 48, at the cost of aliasing
three-way interactions we have no reason to expect.

---

# PART VI — EXPECTED OUTCOME

| Stage | pooled | PP | ceiling |
|---|---|---|---|
| Now | 0.591 | 0.731 | 0.711 |
| Calibration only | 0.65 – 0.68 | 0.735 | 0.711 |
| **Plus information levers** | **0.70 – 0.75** | **0.76 – 0.79** | **0.75 – 0.78** |

**The two sets do not add.** Calibration closes the pooled→PP gap; information raises PP and
therefore raises the ceiling calibration converges toward. **P(pooled > 0.75) ≈ 35%**, dominated by
whether burial and descriptors land at the top of their ranges.

**And the number is the appendix.** The thesis is: the affine oracle is not a valid ceiling and
here is why; the real ceiling is 0.70–0.72 and here is the measurement; the slope term follows from
that diagnosis; calibration is capped by construction and information breaks the cap; and the
dominant force in protein folding was not represented in the model at all.
