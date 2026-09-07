# PHASES 5–9 — Information Levers, with Code

**Verified against `nissimbrami/DeepEF-Thesis`, branch `current-vers`.** Every line number and
variable name below was read from that branch, not assumed. Every predicted effect is tied to a
published result or an in-house measurement.

---

## Part 0 — Why these phases exist, and the arithmetic that fixes an inconsistency

Removing the per-protein offset lifts pooled ΔΔG to **0.70–0.72**, measured across twelve
checkpoints from two independent systems. That is the ceiling of every calibration lever, because
none of them adds information — the anchor, coil, slope term and designed reweight redistribute
what the network already computes.

Our per-protein correlation is already **0.731**, above that ceiling.

**The correction to an earlier estimate.** I previously wrote "0.68–0.72 after everything", which
is the same range I had given for calibration alone. That was wrong and it made the biology look
worthless. The error was treating the ceiling as fixed. It is not:

> **The ceiling tracks PP.** Calibration closes the pooled→PP gap. Information raises PP itself,
> and therefore raises the ceiling that calibration converges toward.

| Stage | pooled | PP | ceiling |
|---|---|---|---|
| Now | 0.591 | 0.731 | 0.711 |
| Calibration only | 0.65 – 0.68 | 0.735 | 0.711 |
| **Plus Phases 5–9** | **0.70 – 0.75** | **0.76 – 0.79** | **0.75 – 0.78** |

**P(pooled > 0.75) rises from ~20% to ~35%**, and it depends almost entirely on burial and
descriptors landing at the top of their ranges.

---

## Part 1 — Exactly what the model sees now (read from the branch)

`train_utils.get_graph`, lines 196–221:

```python
D  = get_dist_matrix(x)                       # [N,N,16]
D  = torch.relu(torch.exp(gaussian_coef*D**2))
Fb = get_bonded_features(D)                   # [N,32]
D  = D.sum(dim=1)                             # [N,16]  ← the row-sum that destroys pair identity
D  = F.normalize(D, p=2, dim=0)
emb = F.normalize(emb, p=2, dim=0)
Fh = torch.cat([D, Fb, emb, one_hot], dim=1)  # [N, 16+32+1024+20]
```

`model/hydro_net.py` `PEM`, lines 359–399:

```python
self.one_hot_index    = -20
self.non_bonded_index = 16
self.llm_index        = -(CFG.emb_input_dim + 20)

x_gcn = torch.cat((x[:, :32], x[:, -20:]), dim=-1)   # 52 dims
x_gat = torch.cat((x[:, :16], x[:, -20:]), dim=-1)   # 36 dims
x_emb_features = x[:, self.llm_index:self.one_hot_index]   # 1024, used only after the GNN
```

**Three consequences that define the phases:**

1. **No solvation term of any kind.** No SASA, no burial, no hydrophobicity. `constants.py`
   defines a hydrophobic amino-acid list that no code reads.
2. **The alphabet is 20 symbols, mutually equidistant.** `get_one_hot` writes an all-zero row for
   anything outside the twenty — not an "unknown" token, the absence of one.
3. **`D.sum(dim=1)` destroys pair identity** before the network sees anything: each residue knows
   *how much* contact it has, never *with whom*.

And structurally: since one structure serves every variant, `D` and `Fb` are **identical between
mutant and wild type** and cancel exactly in ΔΔG. **All mutation signal flows through `one_hot`
and `emb` alone** — 20 of the 68 non-embedding dimensions. Holes 2 and 3 sit on that channel.

---

## Part 2 — Every biological and chemical item we discussed, and its disposition

| Item | Where it lands | Basis |
|---|---|---|
| **Hydrophobicity** | Phase 5, feature 2 | Dominant folding force; a 20-entry lookup, free |
| **Burial / SASA** | Phase 5, features 1 and 3 | **DeepDDG: SASA of the mutated residue is the single most important feature**, and they conclude buried hydrophobic area is the major determinant of stability |
| **ΔASA folded vs unfolded** | Phase 5, the zeroing rule | This difference *is* the hydrophobic driving force. A previous implementation computed burial from the same coordinates in both states, so it cancelled exactly in `E_u − E_f` |
| **Residues beyond the 20** | Phase 6 | Ofir: Mordred on chiral SMILES for 58 residues, 1826 → 1280 → 654 by two filters; a model trained on canonical residues predicts non-canonical effects because the space is continuous |
| **Chemical distance between residues** | Phase 6 | One-hot asserts all pairs equidistant. Literature is consistent: chemical encodings "provide additional information that a limited-size model cannot learn from one-hot alone", and the ensemble of one-hot **plus** chemical beats one-hot |
| **Energy as a sum over pairs** | Phase 7 | Every physical energy function is a sum over (identity, identity, distance) triples |
| **Secondary structure** | Phase 7, extended connectivity | An α-helix is defined by i→i+4; the chain branch reaches only i→i+1, so secondary structure is *unrepresentable* |
| **Structure quality** | Phase 8 | Resolution was the strongest annotation in the 100k catalogue at +0.158 |
| **Metal coordination** | Phase 9 | Dative bonds to side chains — a different physical term from the hydrophobic effect, and not the same thing as interface burial |
| **Interfaces / BSA** | **Dropped** | **Measured over 100,246 structures: corr = −0.001** |
| **Complexity (complex vs monomer)** | **Dropped** | Mean loss differs by 0.2 out of 18 |

**On BSA and complexity, measured rather than assumed.** I ran the correlations over the whole
catalogue: BSA −0.001, length +0.070, resolution +0.158, complex-vs-monomer 0.2/18. And the
decisive caveat: that file records the **pretraining** decoy loss, which has already been measured
as not helping ΔΔG. **Do not build an interface phase on it.** Its one useful signal is
Phase 8.

---

# PHASE 5 — Solvation and burial

**Highest expected value, lowest cost, strongest external support.**

DeepDDG trained on 5700 curated mutations and reached Pearson 0.48–0.56 on three independent test
sets, beating eleven other methods, and their feature analysis found **the solvent accessible
surface area of the mutated residue to be the most important input.** Our model has no equivalent
feature at all.

## 5.1 Code — `train_utils.py`

```python
# Kyte-Doolittle hydropathy, normalised to [0,1]. Index order matches AA_MAP.
_KD = torch.tensor([1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,
                    1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3])
_KD_NORM = (_KD + 4.5) / 9.0                     # [0,1]
_BURIAL_MAX = 30.0                               # constant, NOT N — see 5.2
_BURIAL_RADIUS = 10.0

def compute_burial(x, mask, radius=_BURIAL_RADIUS, cap=_BURIAL_MAX):
    """Cbeta neighbour-count burial, [N,1] in [0,1].
    x: [N,4,3] with atom order (N, CA, C, CB). Glycine has no CB; the loader
    stores CA there, which is the standard substitute."""
    cb = x[:, 3, :]                                        # [N,3]
    d  = torch.cdist(cb, cb)                               # [N,N]
    valid = (mask > 0).float()
    within = ((d < radius).float() * valid.unsqueeze(0) * valid.unsqueeze(1))
    within = within - torch.diag(torch.diag(within))       # exclude self
    counts = within.sum(dim=1, keepdim=True)               # [N,1]
    return (counts / cap).clamp(0.0, 1.0) * valid.unsqueeze(1)

def solvation_features(x, one_hot, mask, folded=True):
    """[N,3]: burial, hydrophobicity, burial*hydrophobicity.
    In the UNFOLDED state burial is 0 by construction: an extended chain buries
    nothing. That difference IS the hydrophobic driving force."""
    hyd = (one_hot @ _KD_NORM.to(one_hot.device).unsqueeze(1))   # [N,1]
    if folded:
        bur = compute_burial(x, mask)
    else:
        bur = torch.zeros_like(hyd)
    return torch.cat([bur, hyd, bur * hyd], dim=1)               # [N,3]
```

Then in `get_graph`, replacing line 219:

```python
if getattr(CFG, 'burial_features', False):
    S  = solvation_features(x, one_hot, mask, folded=True)
    Fh = torch.cat([D, Fb, S, emb, one_hot], dim=1)      # +3 dims
else:
    Fh = torch.cat([D, Fb, emb, one_hot], dim=1)          # unchanged
```

and the identical block in `get_unfolded_graph` and `_flory_unfolded_graph` with
`folded=False`.

## 5.2 Two traps, both previously hit

**Burial must be zero in the unfolded state.** An earlier implementation called the same helper
with the same coordinates in both passes, so the column was bit-identical and cancelled in
`E_u − E_f`. The feature could not express the thing it was named for.

**Normalise by a constant, never by `N`.** An earlier version divided the neighbour count by chain
length. Burial is a *local* quantity — neighbours within 10 Å do not scale with protein length.
Dividing by `N` injects a per-protein length confound into the one feature meant to fix a
per-protein problem.

## 5.3 `hydro_net.py` — the slicing must move

The new block sits between `Fb` and `emb`, so absolute indices from the left change and the
right-anchored ones do not. In `PEM.__init__`:

```python
self.solv_dim = 3 if getattr(CFG, 'burial_features', False) else 0
self.solv_start = 48                                            # after D(16)+Fb(32)
self.llm_index = -(CFG.emb_input_dim + 20)                      # unchanged, right-anchored
self.one_hot_index = -20                                        # unchanged
gcn_fc_in = 16 + 16 + 20 + self.solv_dim
gat_fc_in = 16 + 20 + self.solv_dim
```

and in `forward`:

```python
solv = x[:, self.solv_start:self.solv_start+self.solv_dim] if self.solv_dim else None
x_gcn = torch.cat([x[:, :32], solv, x[:, -20:]] if solv is not None
                  else [x[:, :32], x[:, -20:]], dim=-1)
x_gat = torch.cat([x[:, :16], solv, x[:, -20:]] if solv is not None
                  else [x[:, :16], x[:, -20:]], dim=-1)
```

Flag: `--burial_features` in `train.py`, then `CFG.burial_features = _a.burial_features`.

## 5.4 Verification, before any training

1. **Off is bit-identical.** `torch.equal(Fh_off_before, Fh_off_after)` on a fixed input.
2. **The burial column differs between folded and unfolded** for the same protein. If it does not,
   the cancellation bug is back and the lever is inert.
3. **The proxy measures burial.** Correlate `compute_burial` against Shrake-Rupley SASA
   (`Bio.PDB.SASA`) over ≥200 residues; require |r| > 0.7. If it fails, the neighbour count is
   measuring something else.
4. **No length confound.** corr(mean burial, chain length) across proteins must be near zero.

## 5.5 Prediction and falsification

**PP +0.02 to +0.05; pooled +0.02 to +0.04 through the PP gain.** Confidence high — the physics is
unambiguous, the feature is genuinely absent, and DeepDDG's ablation puts exactly this feature
first among its inputs.

**Falsified if** PP moves less than σ (0.004) with check 2 passing. That would mean contact counts
already encode burial implicitly — a real and reportable finding.

---

# PHASE 6 — Physicochemical residue descriptors

Ofir's method, adapted to 20 residues.

## 6.1 Build the descriptor matrix, once, offline

```python
# scripts/build_aa_descriptors.py  → data/aa_descriptors.csv (committed artifact)
from rdkit import Chem
from mordred import Calculator, descriptors
import pandas as pd, numpy as np

SMILES = {  # chiral SMILES, L-amino acids, neutral form
 'A':'C[C@@H](C(=O)O)N', 'C':'C([C@@H](C(=O)O)N)S', 'D':'C([C@@H](C(=O)O)N)C(=O)O',
 'E':'C(CC(=O)O)[C@@H](C(=O)O)N', 'F':'c1ccc(cc1)C[C@@H](C(=O)O)N',
 'G':'C(C(=O)O)N', 'H':'c1cc(nc1)C[C@@H](C(=O)O)N', 'I':'CC[C@H](C)[C@@H](C(=O)O)N',
 'K':'C(CCN)C[C@@H](C(=O)O)N', 'L':'CC(C)C[C@@H](C(=O)O)N',
 'M':'CSCC[C@@H](C(=O)O)N', 'N':'C([C@@H](C(=O)O)N)C(=O)N',
 'P':'C1C[C@H](NC1)C(=O)O', 'Q':'C(CC(=O)N)[C@@H](C(=O)O)N',
 'R':'C(C[C@@H](C(=O)O)N)CNC(=N)N', 'S':'C([C@@H](C(=O)O)N)O',
 'T':'C[C@H]([C@@H](C(=O)O)N)O', 'V':'CC(C)[C@@H](C(=O)O)N',
 'W':'c1ccc2c(c1)c(c[nH]2)C[C@@H](C(=O)O)N',
 'Y':'c1cc(ccc1C[C@@H](C(=O)O)N)O'}

calc = Calculator(descriptors, ignore_3D=True)
df = calc.pandas([Chem.MolFromSmiles(s) for s in SMILES.values()])
df.index = list(SMILES.keys())

df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(axis=1)                                   # Ofir filter 1
df = df.loc[:, df.nunique() >= 15]                       # Ofir filter 2, scaled 40→15 for n=20
df = (df - df.mean()) / df.std().replace(0, 1)           # normalise across AAs
df.to_csv('data/aa_descriptors.csv')
```

**One deliberate deviation, stated.** Ofir kept descriptors with ≥40 unique values across 58
residues. With 20 residues the maximum possible is 20, so the threshold is scaled to 15 —
i.e. a descriptor must distinguish at least three quarters of the alphabet.

## 6.2 Two arms, both run

- **`pca16`** — PCA to 16 components on the 20×D matrix; report variance explained.
- **`curated12`** — molecular weight, van der Waals volume, TPSA, logP, formal charge, HBD, HBA,
  aromatic rings, rotatable bonds, and three topological indices. Interpretable, defensible in
  writing.

A third arm, **`pca16_only`**, replaces one-hot entirely. That is the arm that demonstrates the
alphabet is no longer needed — the claim that generalises beyond the twenty.

**Default is to keep one-hot and append descriptors.** The literature is consistent on this:
chemical encodings are "as effective as one-hot" alone but the **ensemble of one-hot and chemical
encodings improves accuracy**, because they carry complementary information. One-hot gives exact
identity; descriptors give metric structure.

## 6.3 Integration

Same pattern as Phase 5: a block between `Fb` and `emb`, `solv_start` becomes
`48 + solv_dim`, and both `fc1` widths grow. Flag
`--aa_descriptors {none,pca16,curated12,pca16_only}`.

> **CORRECTION (descriptor-matrix naming trap, applied this session).** The arm names in
> the text above are RETIRED. `data/aa_descriptors.csv` was overwritten with the 20x726
> Mordred matrix while `aa_descriptors.py` still mapped the mode `curated12` to it, so
> anything reported as "curated12" after that rebuild was a **726-column Mordred run, not
> a 12-descriptor curated run** -- two runs with identical logged config used different
> matrices. `pca16` and `pca16_only` both pointed at `data/aa_descriptors_pca16.csv`,
> a name that did not say which pipeline built it.
>
> The modes are renamed so a name states its matrix, and each points at a file whose
> name matches:
>
> | mode | file | K | effect on the feature vector |
> |---|---|---|---|
> | `none` | -- | 0 | 1092 (baseline, byte-identical) |
> | `mordred726` | `data/aa_descriptors_mordred.csv` | 726 | 1092 -> 1818 |
> | `mordred_pca16` | `data/aa_descriptors_mordred_pca16.csv` | 16 | 1092 -> 1108 |
> | `mordred_pca16_only` | `data/aa_descriptors_mordred_pca16.csv` | 16 | replaces one-hot |
>
> `mordred_pca16` is the sane default. `curated12` as previously wired widened the vector
> to 1818, which is almost certainly not what the plan intended by "12 descriptors".
>
> The old names are **not aliases** -- `curated12`, `pca16` and `pca16_only` now RAISE,
> from argparse and again from the loader, so no old invocation is silently
> reinterpreted. The loader additionally asserts the CSV's own `#` provenance header and
> its column count against what the mode expects, and `train.py`'s run config records
> `aa_desc_csv`, `aa_desc_md5`, `aa_desc_k` and `aa_desc_provenance` so a result traces
> to exact bytes. Gate: `scripts/gate_desc_provenance.py`.
>
> Chemical sanity of the committed matrix was verified before training
> (`scripts/chem_sanity.py`, 20/20 checks): F -> Y,H,W; L -> V,I; D -> N,E; N -> D,Q;
> K -> ...,R. The plan's own required checks (L->{I,V,M}, D->{E,N}, F->{Y,W}) all hold.
> The histidine SMILES in the PHASE 6 table was WRONG (a pyridine-type ring, C7H10N2O2);
> the committed matrix used the corrected imidazole (C6H9N3O2) -- confirmed by
> de-z-scoring the surviving `MW` column against 19 known amino-acid masses (max residual
> 0.07 Da), which places H at **155.168 Da** vs 155.157 correct / 154.169 for the typo.



## 6.4 Verification — before training, and this one is decisive

The descriptor space must be chemically sensible or nothing downstream can work:

```python
import pandas as pd, numpy as np
D = pd.read_csv('data/aa_descriptors.csv', index_col=0)
from scipy.spatial.distance import squareform, pdist
dist = pd.DataFrame(squareform(pdist(D.values)), index=D.index, columns=D.index)
for aa in ['L','D','F']:
    print(aa, '→', dist[aa].drop(aa).nsmallest(3).index.tolist())
```

**Required:** L's nearest neighbours are among {I, V, M}; D's are among {E, N}; F's are among
{Y, W}. **If aspartate comes out near leucine, the matrix is wrong and no training will fix it.**

## 6.5 Prediction and falsification

**PP +0.01 to +0.03.** Confidence medium — ProtT5 encodes biochemical similarity implicitly, so
some signal is redundant. But **ProtT5 enters after message passing** (`x_emb_features` is used
only at the light-attention stage) whereas descriptors enter **before**, reaching the graph
convolution where one-hot is currently the sole chemistry.

**The larger prize is not the number.** It is that the alphabet stops being closed: any residue
with a SMILES string gets a vector, so phosphoserine and non-canonical residues become
representable instead of becoming a row of zeros. **Report that as the contribution; report the
ΔΔG delta as secondary.**

**Falsified if** `pca16_only` collapses — that would mean discrete identity carries information
the continuous space cannot, which is itself worth stating.

---

# PHASE 7 — Pairwise edges and reachable secondary structure

## 7.1 Edge attributes

56 dims per edge: source one-hot (20), destination one-hot (20), Cα distance as a **bank of 16
radial basis functions**. `GATv2Conv` takes `edge_dim` directly.

**Two failures already measured, do not repeat them.**

**The RBF bank must be concatenated, never summed.** A previous implementation summed the M
responses back to the original width. Measured: a 5 Å contact and a 15 Å non-contact both returned
4.649 — flat beyond 5 Å, the model blind to distance. The baseline single Gaussian spans
0.835 → 0.000 over the same range.

**The unfolded pass must not receive folded coordinates.** A previous implementation expanded the
true Cα coordinates across both halves of the concatenated graph, injecting folded geometry into
the unfolded state — which directly contradicts the coil lever. Build `ca_coords` per half.

```python
def rbf_expand(d, n=16, lo=0.0, hi=20.0):
    centers = torch.linspace(lo, hi, n, device=d.device)
    width = (hi - lo) / n
    return torch.exp(-((d.unsqueeze(-1) - centers) ** 2) / (2 * width ** 2))   # CONCATENATE
```

**Assertion before any training:** `k(2Å) > k(8Å) > k(15Å)`, strictly. One line, and it catches
the summed-bank failure completely.

## 7.2 Extended chain connectivity

The GCN edge set is `(i, i+1)` only, so with three layers the reach along the chain is three
residues. **An α-helix is defined by i→i+4 and a β-sheet by i→i+2: secondary structure is
currently unrepresentable.** Extend to `|i−j| ≤ 4`.

**Confound to avoid:** at span 1 the edges are directed forward only; a naive extension makes all
offsets bidirectional, changing two things at once. **Make span-1 bidirectional in the control arm
too.**

## 7.3 Prediction and falsification

**PP +0.01 to +0.03.** Confidence medium: the physics is right, but a project record concluded the
bottleneck is data coverage rather than model capacity, and this is a capacity change.

**Falsified if** neither edge features nor extended connectivity moves PP beyond σ — which
supports the capacity conclusion and is worth reporting as such.

---

# PHASE 8 — Structure quality

Resolution correlated **+0.158** with the pretraining loss, the strongest annotation in the
catalogue. For AlphaFold structures the analogue is pLDDT, available and unused.

**Build:** per-residue pLDDT (or B-factor), normalised to [0,1], one node feature.
Flag `--struct_quality`.

**Also report every metric split by pLDDT tercile.** If error concentrates in the low-confidence
tercile, that is a limitation of the *input structures*, not the model — and distinguishing those
is worth a paragraph in the discussion regardless of whether the feature helps.

**PP +0.00 to +0.02** as a feature; **high value as an analysis.**

---

# PHASE 9 — Metal coordination

**Not previously planned, and it should be.** Interface BSA was dropped on measurement, but metal
coordination is a **different physical term**: a dative bond from a side chain to a metal ion, not
buried surface between chains.

**Build, two options.**

*Feature form:* per residue, a flag for membership in a first coordination shell, the metal
identity as a small one-hot, and the coordination number.

*Graph form, and more principled:* **the metal ion becomes a node** with its own node type, and
coordination bonds become typed edges. That is what the graph formalism is for, and it is more
natural than compressing the site into a per-residue scalar.

**The honest caveat.** MegaScale is small soluble domains of 30–80 residues, most without metal
sites. **This is a generalisation lever for natural proteins, not a benchmark lever.** Sell it as
generality, not accuracy. **PP +0.00 to +0.02 on this benchmark**, potentially much more on
natural proteins.

**Prerequisite:** the per-residue annotations from the group's biologist. Ask now for a list and a
date so this can be scheduled rather than hoped for.

---

# Ordering and execution

| Order | Phase | Cost | Predicted PP | Why here |
|---|---|---|---|---|
| 1 | **5 burial** | low | **+0.02 to +0.05** | Dominant force, entirely absent, external ablation ranks it first |
| 2 | **6 descriptors** | medium | +0.01 to +0.03 | Opens the alphabet permanently; contribution outlives the number |
| 3 | **8 pLDDT** | very low | +0.00 to +0.02 | Nearly free; the tercile analysis alone justifies it |
| 4 | **7 pairwise** | high | +0.01 to +0.03 | Most principled, most expensive, capacity is the least-supported bottleneck |
| 5 | **9 metals** | medium | ~0 here | Generality, and it needs external annotation |
| — | interface / BSA | — | **~0** | **Measured. Do not build** |

**Run 5, 6 and 7 as a second factorial**, 2³ = 8 cells at three seeds = 24 runs, roughly a day at
eight concurrent jobs. They may interact: descriptors and pairwise both act on the mutation
channel; burial and pairwise both act on contact representation.

**Every phase runs on top of the winning calibration configuration**, never on the bare baseline,
or the two result sets cannot be combined.

---

# What makes this a thesis

**One.** The affine oracle is not a valid ceiling. Seven of Shahar's checkpoints and five of ours
fail to reach the 0.77–0.81 range it was said to guarantee, and the estimator is **not monotone** —
removing two parameters yields less than removing one. Cause identified: dividing by `a_p`
explodes when the slope is collapsed, and no designed fold exceeds slope 0.646.

**Two.** The defensible ceiling is offset-removal alone: **0.70–0.72, stable across twelve
checkpoints from two systems.** A measured bound replacing a quoted one.

**Three.** The slope term follows from that diagnosis and is the contribution: the oracle's
instability is itself the evidence that slope, not offset alone, is the failure.

**Four.** Calibration is capped by construction and the cap was measured. Phases 5–9 break it with
information rather than normalisation — and Phase 5 closes a hole that is not subtle: **the
dominant force in protein folding was not represented in the model at all**, while the
best-established feature-importance result in this exact task puts it first.

**Five.** Directions closed on evidence before spending GPU time, including interfaces and BSA
closed by direct measurement over 100,246 structures.

**That is a methods chapter, a diagnosis, a contribution, and a negative-results section. The
correlation number is the appendix.**
