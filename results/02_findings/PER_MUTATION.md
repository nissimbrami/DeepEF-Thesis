# LINE 4 — Below the protein: per-mutation error analysis

**n = 28,172 single point mutations** across 28 test proteins, x 10 trained checkpoints.
Significance notes: at the per-protein level n=28 and |r| < 0.374 is indistinguishable from zero
at p=0.05. At the per-mutation level n is 10^4 and almost any effect is "significant", so every
claim below is additionally required to be **sign-consistent across all 10 checkpoints**, which is
the real evidence standard used here.

---

## 0. Provenance: how mutation identity was recovered (this did not exist before)

The eval CSVs (`eval_results/abl_*.csv`) contain only `protein, deltaG, pred_deltaG, ddG, pred_ddG`.
There is **no mutation column**. Identity was reconstructed by replaying the loader's exact filter
chain from `Megascale-fineTuning/evaluate.py::load_test_protein_data`:

1. drop `mut_type` containing `ins|del`, `reset_index(drop=True)`
2. attach `deltaG.pt` positionally
3. drop rows with `ddG_ML == '-'` (UNSTABLE_MUT=False, the eval default)
4. drop rows whose `name` is not in `data/ThermoMPNN/mega_test.csv` (DS_TYPE='pnas')
5. `indexes = list(set(...))` — a set of small ints, which CPython iterates in ascending order,
   so the order is deterministic and ascending

**Verification (the artifact, not the success message):** the reconstruction produces 28,314 rows,
matching the CSVs exactly, and after per-protein alignment the **maximum absolute difference between
reconstructed `dg` and the CSV `deltaG` is 0.0 across all 28,314 rows** in every protein. Protein
*order* differs between recon and CSV because the eval DataLoader uses `shuffle=True`, so the join
is done per protein, not positionally over the whole file.

Composition: 28,172 single point mutants + 142 wt replicates, **zero multi-mutants**.
Per-residue relative SASA computed with Shrake-Rupley over the AlphaFold PDBs (same MAXASA table as
`scripts/bp_replication.py`); available for 28/28 proteins and all 28,172 mutants.

**Sign convention, verified independently** against the `Stabilizing_mut` column:
mean ddG where `Stabilizing_mut=True` is **+1.474**, where False is **-1.153**.
=> **positive ddG = stabilising**. All signs below use this.

**Caveat found along the way:** `ddG` is referenced to **row 0, a single wt replicate**, not the wt
mean. Each protein has 5 wt replicates (2K5H has 7). Replicate SD is typically 0.05-0.31 kcal/mol,
but **2K5H has wt SD = 1.498** (range 1.56-4.81). So 2K5H's entire ddG column carries an arbitrary
~1 kcal/mol reference offset. This is a pre-existing property of the eval, not introduced here, and
it is a plausible partial contributor to that protein's b_p.

---

## 1. Per mutation-type: the mean error is almost entirely compression

380 substitution types possible, **353 observed**. Ranked mean signed error (`pred_ddG - true_ddG`,
averaged over 10 checkpoints; positive = model predicts too high a ddG = **under-predicts
destabilisation**).

### Most under-predicted destabilisation (worst offenders), n>=30

| substitution | n | mean signed err | true ddG | pred ddG | t | sign-consistent |
|---|---|---|---|---|---|---|
| F>E | 35 | **+1.441** | -2.206 | -0.766 | 6.69 | 10/10 |
| F>A | 36 | +1.420 | -1.904 | -0.495 | 7.06 | 10/10 |
| F>Q | 37 | +1.412 | -2.090 | -0.684 | 6.99 | 10/10 |
| F>R | 37 | +1.399 | -2.076 | -0.670 | 6.93 | 10/10 |
| F>G | 33 | +1.389 | -2.316 | -0.973 | 6.51 | 10/10 |
| F>N | 37 | +1.371 | -2.196 | -0.870 | 6.88 | 10/10 |
| F>D | 31 | +1.364 | -2.334 | -0.998 | 5.66 | 10/10 |
| F>S | 37 | +1.340 | -2.004 | -0.681 | 6.59 | 10/10 |
| F>K | 37 | +1.336 | -2.126 | -0.800 | 6.15 | 10/10 |
| F>T | 34 | +1.329 | -1.876 | -0.564 | 6.44 | 10/10 |
| I>G | 89 | +0.986 | -1.935 | -1.145 | 8.01 | 10/10 |
| Y>G | 37 | +0.978 | -1.732 | -0.833 | 4.47 | 10/10 |

### Marginals

**By SOURCE residue** (the dominant axis — spread 0.098 to 1.185):

| from | n | mean err | true ddG | from | n | mean err | true ddG |
|---|---|---|---|---|---|---|---|
| K | 2724 | +0.098 | -0.239 | A | 2110 | +0.505 | -0.717 |
| E | 3293 | +0.130 | -0.211 | L | 2625 | +0.522 | -1.131 |
| S | 1288 | +0.175 | -0.226 | I | 1694 | +0.576 | -1.207 |
| Q | 868 | +0.214 | -0.270 | H | 460 | +0.739 | -0.862 |
| N | 1370 | +0.304 | -0.420 | Y | 766 | +0.810 | -1.247 |
| R | 1767 | +0.318 | -0.516 | W | 507 | +0.840 | -1.595 |
| V | 2097 | +0.344 | -0.996 | **F** | **692** | **+1.185** | **-1.709** |

**By DESTINATION residue** (much flatter — 0.162 to 0.735; P is the outlier at +0.735,
mut-to-proline true ddG -1.593). All 20 destinations and all 19 sources are **10/10 sign-consistent**.

### But this is a trap: the mean-error table is ~90% just global compression

If `pred = a*true` globally, then `mean_err = (a-1)*mean_true` mechanically — the table above would
be an artefact of which residues happen to be destabilising, telling us nothing chemical.

- `corr(mean_true, mean_err)` over the 19 source residues: **r = -0.906, p = 9.6e-8**, implied
  global a = 0.450
- over the 20 destination residues: **r = -0.919, p = 1.0e-8**, implied a = 0.670
- residual SD after removing compression collapses from 0.273 -> **0.116** (source) and
  0.107 -> **0.042** (destination)

**So there is no large mean-error chemistry beyond compression.** The "F is badly predicted" headline
is mostly "F mutations are very destabilising and everything gets halved." The residuals that do
survive are small: sources H (+0.223) and F (+0.204) under-destabilised beyond compression, V (-0.245)
and L (-0.142) over-destabilised.

**This is exactly the metric rule in a new place.** Mean signed error is the wrong statistic for a
chemistry question, because it is dominated by the slope. The right statistic is the slope itself.

---

## 2. Position in sequence: nothing. Burial: yes, but modestly

Slopes below are `pred_ddG ~ true_ddG` fitted **within-protein-centred**, which strips b_p so the
number is a clean within-protein slope.

**Normalised sequence position (0=N-term, 1=C-term), 10 bins:** slope wanders 0.38-0.53 with **no
monotone trend**; `corr(seq_frac, signed err)` over 28,172 mutations = **-0.005, p = 0.42**. Position
along the chain is a **clean null**.

**Distance from nearest terminus** does matter, but as a *tolerance* effect, not a model defect: the
terminal residue itself (n=1006) has mean true ddG -0.128 vs -0.695 for the interior, i.e. termini
really are tolerant, and the model's slope there is low (0.270) mostly because there is almost no
signal to fit.

**Relative SASA (burial):**

| rel SASA | n | mean true ddG | mean err | slope | PCC |
|---|---|---|---|---|---|
| 0.00-0.05 | 1750 | -1.702 | +0.827 | 0.532 | 0.616 |
| 0.05-0.10 | 1259 | -1.461 | +0.884 | 0.464 | 0.688 |
| 0.10-0.20 | 2815 | -1.073 | +0.629 | 0.456 | 0.635 |
| 0.30-0.40 | 3537 | -0.595 | +0.370 | 0.469 | 0.715 |
| 0.50-0.70 | 5818 | -0.293 | +0.158 | 0.386 | 0.586 |
| 0.70-1.00 | 3403 | -0.228 | +0.119 | 0.304 | 0.450 |

`corr(rel_sasa, signed err)` = **-0.289** over 28,172 mutations. Slope core (<0.1) 0.427 vs
exposed (>0.5) 0.304, **10/10 checkpoints negative gap**. So burial raises the slope somewhat.
Note this is the *opposite direction* to the known per-protein result (`mean_rel_SASA` vs a_p,
r=+0.714): more exposed *proteins* have higher a_p, but within a protein more exposed *positions*
have lower slope. These are different objects and do not conflict, but the sign flip is worth flagging.

---

## 3. Stabilising vs destabilising: badly asymmetric

**Imbalance in the data:**

| class | n | share |
|---|---|---|
| destabilising (ddG < 0) | 22,049 | **78.27%** |
| stabilising (ddG > 0) | 6,122 | **21.73%** |
| strongly destabilising (< -0.5) | 13,566 | 48.15% |
| strongly stabilising (> +0.5) | 1,921 | **6.82%** |
| ddG > +1 | 1,049 | 3.72% |

True ddG skew -0.06; **predicted ddG skew -1.46**. The model's output distribution is far more
one-sidedly negative than the truth.

**PCC by half (mean over 10 checkpoints):**

| | pooled | stabilising | destabilising |
|---|---|---|---|
| PCC (raw) | 0.599 | 0.570 | 0.535 |
| **PCC (within-protein)** | — | **0.425** | **0.678** |
| **slope (within-protein)** | — | **0.268** | **0.423** |
| mean signed err | +0.387 | **-0.579** | **+0.655** |

Pooled PCC on the two halves looks deceptively similar (0.570 vs 0.535) because between-protein
offset variance inflates both. **Within protein — the regime that matters — the model is much worse
on stabilising mutations: PCC 0.425 vs 0.678, slope 0.268 vs 0.423.** The signed errors are equal and
opposite (-0.579 stabilising, +0.655 destabilising): the model is pulling everything toward the
middle, which is exactly the "only learned that mutations destabilise" failure the task asked about,
in its quantitative form.

**It is not useless for design, though.** Ranking the top-10 predictions per protein gives 55.4%
true-stabilising precision against a 21.7% base rate = **2.55x enrichment** (top-50: 2.30x,
top-100: 2.15x). So it finds stabilising mutations well above chance while systematically
under-stating how stabilising they are.

---

## 4. THE BIG QUESTION: yes — a_p is driven by one chemical class

### The finding

Fit the within-protein slope separately by **destination residue class**
(hydrophobic = AVILMFWCY vs polar/charged = the rest):

| destination | mean slope over 10 ckpts |
|---|---|
| C 0.212, F 0.247, L 0.251, V 0.276, M 0.281, W 0.282, I 0.284, Y 0.291, A 0.332 | **hydrophobic mean 0.334** |
| T 0.385, G 0.415, P 0.415, S 0.420, R 0.422, H 0.425, E 0.425, D 0.428, Q 0.429, K 0.436, N 0.445 | **polar mean 0.497** |

- `corr(TO-slope, Kyte-Doolittle hydropathy)` over 20 residues: **r = -0.855, p = 1.6e-6**;
  across the 10 checkpoints mean **r = -0.833, 10/10 negative** (individual values -0.71 to -0.87)
- polar-minus-hydrophobic slope gap: mean **+0.149, 10/10 positive**
- t-test hydrophobic vs polar slopes: **t = -13.4, p < 1e-9**

**The model compresses mutations-to-hydrophobic ~49% harder than mutations-to-polar.**

### It is not burial in disguise

Mean rel SASA is **0.3898 for TO-hydrophobic vs 0.3904 for TO-polar** — essentially identical. And
the gap holds inside every burial stratum:

| burial | n hyd | slope hyd | n pol | slope pol | gap |
|---|---|---|---|---|---|
| core <0.1 | 1393 | 0.370 | 1616 | 0.480 | +0.110 |
| 0.1-0.25 | 1966 | 0.313 | 2423 | 0.419 | +0.106 |
| 0.25-0.5 | 4373 | 0.330 | 5187 | 0.465 | +0.135 |
| exposed >0.5 | 4350 | 0.284 | 5102 | 0.403 | +0.119 |

### It is universal, not a few proteins

Per protein, `a_polar - a_hydrophobic` is **positive in 26/28 proteins** (binomial p = 3.0e-6),
mean gap +0.203. Across the 10 checkpoints: 26-28 of 28 proteins positive **every time**.
Median a_p on class-restricted subsets: **all 0.409, TO-hydrophobic 0.251, TO-polar 0.439**;
TO-polar > TO-hydrophobic in **10/10 checkpoints**.

### The honest limit: it does NOT explain the between-protein spread of a_p

This is where the finding stops. Composition does not predict which protein has a low a_p:

| covariate | corr with a_p (n=28) | 10-ckpt |
|---|---|---|
| `frac_toh` (share of TO-hydrophobic mutations) | **-0.243, p=0.21 — ns** | 10/10 sign |
| `frac_from_arom` | -0.405, p=0.033 | 10/10 sign |
| `mean_rel_SASA` | **+0.708** (the known result, replicated) | 10/10 sign |
| `frac_stab` | +0.018 — ns | 5/10 |

OLS `a_p ~ frac_toh + frac_from_arom + frac_stab`: **R2 = 0.206, adjusted R2 = 0.106** (n=28, k=3).
And `corr(a_p, gap) = +0.741` while `corr(a_p, a_hyd/a_pol) = +0.059, p=0.77`, with ratio
`a_hyd/a_pol` median **0.655**: the defect is **multiplicative**, a roughly constant ~65% factor
riding on top of whatever a_p the protein already has. So it is a *universal* multiplicative defect,
not the *source* of the between-protein spread. `mean_rel_SASA` remains the only real predictor of a_p.

### And it is NOT fixable by rescaling — it is lost information

The obvious fix is to multiply TO-hydrophobic predictions by a class factor. It fails:

| | mean over 10 ckpts | beats raw |
|---|---|---|
| raw pooled PCC | 0.5987 | — |
| **LOPO class rescale** (honest, held-out) | **0.5917** | **0/10** |
| global per-class affine ORACLE (fitted on test) | 0.5999 | 10/10, +0.0012 |
| per-protein per-class affine ORACLE | 0.8349 | (upper bound only) |

Even the *oracle* single global class rescale buys **+0.0012 PCC**. Consistent with the known
shift-invariance result, a pure scale fix cannot help pooled Pearson much.

The reason is that the class defect is **not** pure scale. Within-protein Spearman rho, which is
scale-invariant, is also degraded:

| | mean rho over 10 ckpts x 28 proteins |
|---|---|
| all mutations | 0.651 |
| **TO-hydrophobic** | **0.490** |
| **TO-polar/charged** | **0.737** |

And the per-residue ranking is a **perfect class separation with zero overlap** — all 9 hydrophobic
destinations rank below all 11 polar/charged:

`F 0.414 < L 0.424 < W 0.447 < C 0.481 < I 0.492 < Y 0.506 < V 0.515 < M 0.515 < A 0.639`
**|** `H 0.652 < T 0.678 < K 0.707 < Q 0.708 < R 0.714 < S 0.720 < G 0.721 < E 0.721 < P 0.726 < N 0.740 < D 0.749`

Two controls rule out the "hydrophobic mutations just have less ddG variance" explanation
(TO-hyd ddG SD 1.056 vs TO-pol 1.266):

- **decile-stratified** on true ddG: rho_hyd 0.125 vs rho_pol 0.169, **10/10**
- **histogram-matched resampling** of TO-polar down to the TO-hydrophobic ddG distribution:
  rho_hyd 0.490 vs rho_pol(matched) **0.642**, gap **+0.152, 10/10**

---

## Bottom line

1. **Mean signed error per mutation type is ~90% global compression** (r=-0.91 with mean true ddG).
   The metric rule bites again: for a chemistry question the slope, not the mean error, is the
   statistic. Reporting the F>E table as "the model mispredicts aromatics" would have been wrong.
2. **Sequence position is a clean null** (r=-0.005). Burial matters modestly and, notably, in the
   opposite within-protein direction to the known between-protein `mean_rel_SASA` effect.
3. **The model is much worse on stabilising mutations** — within-protein PCC 0.425 vs 0.678, slope
   0.268 vs 0.423, on a dataset that is 78% destabilising and only 6.8% strongly stabilising.
   Still 2.55x enrichment at top-10, so useful but badly under-scaled for design.
4. **a_p IS driven by a chemical class, and it is a real, universal, replicated defect:
   mutations to hydrophobic residues are compressed ~49% harder than mutations to polar residues
   (slope 0.334 vs 0.497, r=-0.83 with hydropathy, 10/10 checkpoints, 26/28 proteins, independent
   of burial).** But it is (a) multiplicative and universal, so it does not explain the
   *between-protein spread* of a_p, and (b) accompanied by a matched loss of *rank* accuracy
   (rho 0.490 vs 0.737, surviving distribution-matched controls), so it is **lost information about
   hydrophobic packing, not a miscalibrated scale**. A rescaling correction is therefore dead on
   arrival (LOPO 0/10). The fix has to be on the *input/representation* side — giving the model
   what it needs to see hydrophobic burial and packing — not on the output calibration side.

This is the fourth calibration lever the metric rule has disciplined, and the first one located
*below* the protein.

## Generalisation caveat

All of this is measured on 28 single-chain, ligand-free, metal-free monomers of 43-72 aa with
canonical residues only. The class defect is 10/10 checkpoint-replicated and 26-28/28
protein-replicated *within that benchmark*, which makes it solid here, but whether the
hydrophobic-compression defect persists on larger, multi-chain, or ligand-bound proteins is a
generalisation claim this benchmark cannot test.

## Files

- `results/PER_MUTATION.md` (this file)
- `scratch_nb/build_master.py` — identity reconstruction + verified join (max |dG diff| = 0)
- `scratch_nb/master_mut.pkl`, `scratch_nb/single_mut.pkl` — per-mutation tables, 10 checkpoints
- `scratch_nb/q1.py` `q1b.py` `q1c.py` `q2.py` `q3.py` `q4.py` `q4b.py` `q4c.py` — analyses
- `scratch_nb/q1_subst.csv` `q1_dest.csv` `q1_src.csv` `q1c_to_slopes.csv` `q1c_from_slopes.csv`
  `q1b_slopes.csv` `q4_ap.csv` — ranked tables

Gate: `scripts/gate_g4_cpu.py` -> **G4-CPU: ALL PASS**, baseline line `dG=-0.0030 width=1092`.
No repo code was modified; all work is under `scratch_nb/`.
