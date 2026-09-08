# Hadar's 100k Catalogue — What It Actually Contains, and What It Is Good For

`data/FINAL_DATASET_100k_030926.csv` — 100,246 rows x 21 columns.
All numbers below were measured, not recalled. Gate re-run after inspection: `dG=-0.0030 width=1092` (unchanged; this report writes no code into the model path).

---

## 0. The one-paragraph answer

This file is **not a feature source and not joinable to our 28**. It is the *census of the pre-training
distribution*, and read as a census it produces a single hard result that we did not have before:

> The pre-training objective is a **strong function of chain length**, it **saturates above ~250 aa**,
> and our entire test set sits in the pre-asymptotic regime where the model is measurably worst.
> The cell matching our 28 (monomeric, 43–72 aa, ligand-free) is **1.50% of rows** but only
> **0.08% of residues**, and its decoy loss is **-14.82 vs -18.16** for everything else — **Cohen d = 0.96, t = 27.7**.

That is a distribution-shift result requiring **no join at all**, and it is the first mechanistic account
of `b_p` we have that is independent of the 28. It explains why `b_p` is a large, protein-specific,
**reference-state** offset with ICC 0.898 (highly reproducible across checkpoints — i.e. a property of the
*trained function*, not of noise), and why the `b_p` corrector fitted on the 28 fails out of sample:
within our 43–72 aa window there is no length variation left to fit against.

**Verdict: MAYBE.** One measurement is thesis-grade and free (Section 4.A). One model change is real but
expensive and out of scope for the M.Sc. (Section 4.B). Everything that would make `b_p` a per-residue
input feature is a **DROP**, and Section 4.D says honestly why.

---

## 1. WHAT WE SEE — column by column

Counts are over N = 100,246 rows.

### 1.1 Identity / bookkeeping

| Column | Non-null | Unique | What it actually is |
|---|---|---|---|
| `rank` | 100,246 (100%) | 100,246 | **Spearman(rank, loss) = -1.0000, exactly.** `rank` is nothing but the sort order of `loss`, descending. Range 2..100,247 with N=100,246 → exactly **one rank is missing** (one row was dropped after ranking). Carries zero information beyond `loss`. |
| `protein_id` | 100,246 | 100,246 | `{PDB}_{entity}_{chain}`, e.g. `3F3Q_1_A`. SCOP-style domains appear as `2CRV_d2crva1`. One row = one **chain/domain**, not one PDB. |
| `PDB_ID_and_Entity` | 100,246 | 100,246 | Same string as `protein_id`, upper-cased. **Pure duplicate.** |
| `split` | 100,246 | 2 | `training` 100,086 / `validation` **160**. |

**66,945 distinct PDB entries** back the 100,246 rows (mean 1.50 chains/entry, max 77).

> **Finding 1.1 — the validation split is metadata-blind.**
> All 160 validation rows have `Method`, `AA Length`, `Oligomeric State`, `Chain Composition`,
> `Is Complex?` = `'Unknown'` and `Resolution` = NaN. Their ids carry a numeric prefix
> (`20#2Y1B_1_A`, `10#2FK5_1_A`) that broke the metadata join. Of the 538 `Unknown` rows,
> 160 are exactly the whole validation split and 378 are training.
> **Consequence: any stratified analysis must be run on the training split only.** The 160
> validation rows can contribute `loss` but no covariate. Their mean loss is -15.68 vs -18.11
> for training — this is *not* a generalisation gap, it is the same length effect (validation
> was deliberately enriched for short chains), and it is uninterpretable without the lengths.

### 1.2 The loss columns — decoded exactly

| Column | Non-null | Distribution |
|---|---|---|
| `loss` | 100,246 | mean -18.11, sd 3.46, min -20.36, max +18.29; 99.3% negative |
| `lossd` | 100,246 | mean -38.21, sd 9.00, min -72.25, max +9.98; 99.7% negative |
| `lossg` | 100,246 | **constant 0** — dead (confirmed) |
| `lossc` | 100,246 | mean +20.10, sd 6.44, min 0.12, max 67.29 — **alive, and decodable** |

> **Finding 1.2 — `loss = lossd + lossc`, exactly.**
> OLS of `loss` on `(lossd, lossc, 1)` returns coefficients `[1.000000, 1.000000, 0.000000]`,
> R² = 1.0, max |residual| = **1e-6**. So `loss` is fully redundant. There are **two** free
> loss numbers per protein, not four.

> **Finding 1.3 — `lossc` is the energy-magnitude regulariser, and it is invertible.**
> From `evaluate_train.py:232-237` (the code version that generated this CSV; current
> `train.py:686` has since hard-zeroed `lossc`):
> ```
> lossc = reg_alpha * mean( [Ejf, Eju, Exd, Ecd, Exdu, Ecy1..Ecy4]^2 )      # reg_alpha = 0.1
> ```
> Therefore **`RMS|E| = sqrt(lossc / 0.1)`** is the *raw energy scale the pre-trained network
> assigns to that protein*, in the network's own energy units, recoverable for all 100,246 rows.
> This is the single most useful quantity in the file and it was previously written off as
> "pre-training decoy loss".
>
> Note this also dates the file: `lossg` is 0 because these are `with_grad=False` validation-path
> evaluations (`lossg` is a gradient penalty), and `lossc` is non-zero because the CSV predates
> the commit that removed it. **The CSV is not reproducible with today's `train.py`.**

`lossd` is the InfoNCE decoy-ranking term, `-log[ exp(-E_Ejf/τ) / Σ exp(-E_i/τ) ]` over 9 states
(native folded/unfolded, decoy sequence, decoy structure, decoy unfolded, 4 cyclic permutations),
τ = 1.0, plus a secondary `Eju < Ecd` term. Textbook InfoNCE is ≥ 0; the observed values are 99.7%
**negative**, so the generating version used a different (unnormalised log-ratio) form. **Use `lossd`
only ordinally** — more negative = better decoy discrimination. Its cardinal scale is not defined.

`corr(lossc, lossd) = -0.953`: the two terms are almost collinear because both track the same
underlying energy magnitude. Effectively **one** degree of freedom.

### 1.3 Structural metadata

**`Method`** — 10 values, 100,246 non-null:

| | n | mean loss |
|---|---|---|
| X-RAY DIFFRACTION | 88,879 (88.7%) | -18.42 |
| SOLUTION NMR | 8,545 (8.5%) | -15.67 |
| ELECTRON MICROSCOPY | 2,161 (2.2%) | -15.43 |
| Unknown | 538 | -16.99 |
| ELECTRON CRYSTALLOGRAPHY | 39 | -17.08 |
| SOLID-STATE NMR | 37 | -10.86 |
| SOLUTION SCATTERING | 25 | -7.77 |
| FIBER DIFFRACTION | 12 | -10.90 |
| NEUTRON DIFFRACTION | 8 | -19.16 |
| INFRARED SPECTROSCOPY | 2 | -2.37 |

The apparent "NMR is harder" effect is **almost entirely length confounding** — see Finding 2.3.

**`Resolution (Å)`** — 91,084 non-null (90.9%), 2,452 unique. Mean 2.33, median 2.10, p25 1.80,
p75 2.55, min 0.48, max 70.0. The 9,162 nulls are exactly the methods that have no resolution
(NMR 8,545 + Unknown 538 + a few). **max = 70.0 Å is an EM entry** — if resolution is ever used,
clip at 5 Å. Spearman(Resolution, loss) = **+0.186**, the strongest single covariate → worse-resolved
structures are harder, as expected, but the effect is modest.

**`AA Length`** — stored as **string** (`'Unknown'` for 538 rows), so it must be coerced.
99,708 numeric.

> **Finding 1.4 — `AA Length` is the ASSEMBLY length, not the chain length.**
> Median 477, p75 920, p99 19,110, **max 89,160**. No protein chain is 89,160 aa. Grouping by
> chain count confirms it: median `AA Length` / n_chains is **~200 aa for every oligomeric state
> from 1 to 25 chains** (216, 229, 194, 220, 177, 205, 196, 212, 229, 156, ...). The column counts
> **every residue in the biological assembly**, while `protein_id` names a **single chain**.
>
> This is a booby-trap: the row `4V8P_45_BX` has `AA Length` = 43,352 but is one chain of a
> ribosome. **Any per-chain analysis must restrict to `Oligomeric State == 'Monomer'`**, where
> assembly length = chain length. Monomers-only: n = 31,852, median **216**, p25 124, p75 342,
> min 20, max 1,520 — a believable single-chain distribution.

**`Oligomeric State`** — 104 values, all of the form `Monomer` / `Complex (k chains)` / `Unknown`:

| | n | share |
|---|---|---|
| Monomer | 31,852 | 31.8% |
| Complex (2 chains) | 29,253 | 29.2% |
| Complex (3–4) | 21,073 | 21.0% |
| Complex (5–8) | 9,944 | 9.9% |
| Complex (9–16) | 3,787 | 3.8% |
| Complex (17–64) | 2,356 | 2.4% |
| Complex (>64) | 1,443 | 1.4% |
| Unknown | 538 | 0.5% |

The tail is real biology: 182 rows at 86 chains, 147 at 162 chains, 41 at 184 (ribosomes),
1 at **480 chains** (a viral capsid). The 104 "values" are just 102 distinct integer chain counts
plus `Monomer` and `Unknown` — **parse `\((\d+) chains\)` to an integer; do not one-hot 104 levels.**

**`Is Complex?`** — 3 values, `Yes` 67,856 / `No` 31,852 / `Unknown` 538. Exactly
`Oligomeric State != 'Monomer'`. **Redundant.**

**`Chain Composition`** — 6 values. This one is *not* redundant; it names the **partner type**:

| | n | mean loss |
|---|---|---|
| homomeric protein | 66,336 | -18.51 |
| heteromeric protein | 22,775 | -17.18 |
| protein/NA (nucleic acid) | 5,978 | -17.18 |
| protein/oligosaccharide | 4,591 | -18.23 |
| Unknown | 538 | -16.99 |
| protein/NA/oligosaccharide | 28 | -19.02 |

**5,978 + 28 = 6,006 rows (6.0%) are protein–nucleic-acid complexes.** That is where the `A/C/G/U/DA/DC/DG/DT`
"ligands" come from — they are *chain residues of a partner polymer*, not small molecules.

**`Global Symmetry`** — constant `'Asymmetric'`, all 100,246 rows. Dead (confirmed). Worth noting
*why* it is suspicious rather than merely uninformative: a set with 480-chain capsids and 162-chain
assemblies containing **zero** C/D/I/O symmetry is a **failed extraction**, not a fact about the PDB.

### 1.4 The ligand columns

**`Ligands_x`** — 4 non-null / 100,246. Dead (confirmed).
**`Ligands_y`** — 75,464 non-null, **8,186 distinct het codes**, 212 literal `'Error'`.
75,252 rows (75.1%) carry ≥1 het code. Top codes:

`SO4` 12,140 · `GOL` 9,920 · `MG` 9,620 · `ZN` 9,360 · `MSE` 9,072 · `CL` 7,743 · `CA` 6,555 ·
`NAG` 4,973 · `EDO` 4,875 · `NA` 3,872 · `PO4` 3,504 · `U/A/C/G` ~3,100 each · `DT/DG/DC/DA` ~2,650 each ·
`ACT` 2,540 · **`HEM` 2,081** · `MN` 1,702 · `PEG` 1,649 · `K` 1,646 · `BMA` 1,596 · **`FAD` 1,585** ·
`MAN` 1,578 · **`ADP` 1,409** · `UNK` 1,287 · `MPD` 1,059 · **`GDP` 1,028** · `FE` 1,020

> **Finding 1.5 — the raw 75% "has ligand" figure is an illusion.**
> Classifying every row by whether it has a het code that is *not* a nucleic-acid chain residue,
> a crystallisation additive, a bare metal ion, or a modified amino acid:
>
> | class | n | share |
> |---|---|---|
> | **real cofactor / substrate** | 43,798 | **43.7%** |
> | junk-only (additives, metals, NA residues, modres) | 31,454 | 31.4% |
> | no het record at all | 24,994 | 24.9% |
>
> So the honest ligand-presence rate is **43.7%, not 75%** — still high, and still 43.7 points
> above our 28 (which are 0%). `MSE` (9,072 occurrences, the single largest "ligand" after the
> additives) is **selenomethionine — an amino acid inside the chain**, a phasing artefact, not a ligand.

### 1.5 The BSA columns — one of these is alive

`BSA_Percentage` (5 non-null) and `BSA_Numeric_x` (4 distinct values recycled over 100,246 rows)
are dead, as documented. `BSA` is a string-with-`%` version with 233 `'Error'` values.

> **Finding 1.6 — `BSA_Numeric_y` is ALIVE and well-behaved, and the brief's "BSA is dead" is
> half wrong.**
> 100,000 non-null (99.75%), 4,867 distinct values, units **percent**, range 0.00–70.08.
> 31,277 rows (31.2%) are exactly 0. It is internally consistent with `Oligomeric State` to a
> degree that rules out corruption:
>
> | n_chains | n | median BSA% |
> |---|---|---|
> | 1 | 31,766 | **0.00** |
> | 2 | 29,189 | 12.44 |
> | 3–4 | 21,019 | 19.68 |
> | 5–8 | 9,919 | 23.35 |
> | 9–16 | 3,780 | 27.72 |
> | 17–64 | 2,352 | 28.00 |
> | >64 | 1,441 | 32.56 |
>
> Monotone in chain count, and **30,536 of 31,852 monomers have BSA exactly 0** (as they must).
> The 1,230 monomers with BSA > 0 are crystallographic-contact leakage.
>
> **This does not resurrect BSA as a feature for us** — all 28 test proteins have `interchain_BSA = 0`,
> so the column is *constant on our test set* and cannot correlate with anything (this is the
> "-0.001 BSA null" already on record, and it stands). What `BSA_Numeric_y` is good for is
> **describing the pre-training distribution**, which is exactly Section 2. The correct statement
> for the thesis is "`BSA_Numeric_x`/`BSA_Percentage` are dead merge artefacts; `BSA_Numeric_y`
> is a valid interface measurement that is *constant zero on our test set*" — three different
> failure modes that must not be collapsed into one sentence.

---

## 2. WHAT CAN BE EXTRACTED — the distribution-shift result

The join is empty by construction: **0 of 28** test proteins appear, under exact PDB-4 match and
under case-insensitive substring scan of both id columns. Nothing per-protein can be transferred.
What *can* be extracted is the shape of the distribution the network was trained on.

### 2.1 The mismatch, quantified

Marginals of the pre-training set vs. our 28:

| property | pre-training | our 28 | 
|---|---|---|
| monomeric | 31.8% | **100%** |
| 43 ≤ L ≤ 72 aa | 1.84% | **100%** |
| no real ligand | 57.6% | **100%** |
| BSA = 0 | 31.2% | **100%** |
| X-ray | 88.7% | mixed NMR/design |

Jointly:

| cell | n | P | rarity |
|---|---|---|---|
| Monomer | 31,852 | 0.3177 | 1 in 3 |
| Monomer & 43–72 aa | 1,588 | 0.01584 | 1 in 63 |
| + no real ligand | 1,504 | 0.01500 | 1 in 67 |
| + BSA = 0 | **1,500** | **0.01496** | **1 in 67** |

Independence product = 0.00337 vs observed 0.01496 → **ratio 4.45**. The properties are strongly
positively associated (small ⇒ monomeric ⇒ ligand-free), which is *good* news: the true cell is
4.45× larger than an independence argument would predict. It is still 1.5%.

> **Finding 2.1 — the per-residue mismatch is 20× worse than the per-row mismatch.**
> The cell is **1.584% of rows** but its residues are **93,686 of 116,431,172 = 0.0805% of all
> residues in the catalogue**. Because `AA Length` is assembly length and the loss is evaluated
> per structure, a 43,352-residue ribosome chain and a 43-residue designed helix bundle each count
> as **one** training example — but the GNN's message passing, the ProtT5 embedding statistics, and
> every per-residue normalisation see them in a **1000:1** ratio.
> **Our test regime is 1 residue in 1,250 of pre-training.**

### 2.2 The loss gap — the model is worst exactly where we test

Mean decoy loss by chain length (more negative = better; **monomers only**, so length is real):

| length | n | loss | lossd | lossc |
|---|---|---|---|---|
| ≤30 | 470 | **-5.68** | -8.64 | 2.95 |
| 30–43 | 572 | **-9.94** | -16.06 | 6.12 |
| **43–50** | 373 | **-13.05** | -21.31 | 8.26 |
| **50–60** | 402 | **-14.09** | -23.94 | 9.84 |
| **60–72** | 787 | **-16.30** | -29.92 | 13.61 |
| 72–85 | 1,065 | -16.87 | -31.80 | 14.93 |
| 85–100 | 1,517 | -17.88 | -35.90 | 18.03 |
| 100–120 | 2,307 | -18.34 | -37.90 | 19.56 |
| 120–150 | 3,278 | -18.68 | -39.34 | 20.66 |
| 150–200 | 4,092 | -18.95 | -39.96 | 21.01 |
| 200–300 | 6,320 | **-19.12** | -40.96 | 21.84 |
| 300–500 | 8,353 | -19.08 | -41.99 | 22.91 |
| 500–1000 | 2,246 | -18.71 | -40.82 | 22.11 |

The curve rises steeply to ~200 aa and is **flat thereafter**. The bolded rows are our test regime.

Cell vs rest (monomer & 43–72 & ligand-free & BSA 0, n = 1,500):

| | n | loss | median | sd | lossd | lossc |
|---|---|---|---|---|---|---|
| cell | 1,500 | **-14.82** | -16.13 | 4.64 | -26.04 | 11.22 |
| rest | 98,746 | **-18.16** | -19.26 | 3.42 | -38.40 | 20.24 |

**Cohen d = +0.964, Welch t = 27.7.** The cell's mean rank-percentile is **0.189** (0.5 = typical):
our test regime sits in the **worst fifth** of the pre-training set. The cell's loss sd is also
36% larger (4.64 vs 3.42) — the model is not just worse there, it is **less consistent** there.

### 2.3 It is length, not method, and not ligands

The obvious objection is that short proteins are NMR structures and NMR is the real problem.
It is not. Restricting to 43–150 aa and splitting by method gives **near-identical curves**:

| length | SOLUTION NMR | X-RAY |
|---|---|---|
| 43–60 | -12.87 (n=733) | -12.75 (n=161) |
| 60–72 | -15.88 (n=621) | -14.50 (n=286) |
| 72–100 | -16.94 (n=1,982) | -16.59 (n=958) |
| 100–150 | -17.49 (n=2,841) | -18.17 (n=3,921) |

> **Finding 2.3 — ANCOVA `loss ~ log10(L) + is_NMR` on 43–300 aa (n = 28,957):**
> `β_log10L = -4.211`, `β_NMR = +0.485`.
> Over the 43→300 aa range the length term moves the loss by 3.56 units; the method term moves it
> by 0.49. **Length is 7.3× the method effect.** The ligand and BSA conditions barely move the cell
> mean at all (-14.86 with ligands allowed → -14.82 without; -14.80 for rows with no het record).
> **The mismatch that matters is length. Only length.**

### 2.4 The energy scale is not calibrated at short length — the `b_p` mechanism

Decoding `RMS|E| = sqrt(lossc/0.1)` (Finding 1.3):

| length | mean RMS\|E\| | RMS/L |
|---|---|---|
| ≤43 | 6.37 | 0.201 |
| **43–72** | **9.85** | 0.173 |
| 72–100 | 12.30 | 0.147 |
| 100–150 | 13.51 | 0.111 |
| 150–250 | 13.80 | 0.070 |
| 250–500 | 14.33 | 0.039 |
| 500–1000 | 14.35 | 0.020 |
| >1000 | 14.01 | 0.008 |

The energy magnitude **saturates at ≈14.8** above 250 aa — the network has learned a bounded
output scale, essentially independent of protein size, which is the correct behaviour for a
*contrastive ranking* objective (it only ever needed energy **differences** to have the right sign)
and the **wrong** behaviour for a model that must emit an absolute ΔG.

On monomers with L ≤ 1000 the scale follows a clean power law:

```
log10( RMS|E| ) = 0.2006 * log10(L) + 0.6735       r = 0.563,  n = 31,782
```

Sanity check: predicted at L = 57 → **10.61**; observed median for the cell → **10.69**. **0.8% error.**

> **Finding 2.4 — a mechanism for `b_p` that does not use the 28 at all.**
> Pre-training taught the network an energy scale calibrated on 200–500 aa proteins and **never
> asked it to be right in absolute terms**. At 43–72 aa the scale is at ~67% of its asymptote and
> still moving. `dG = E_unfolded - E_folded` is a difference of two such mis-scaled energies, so
> a **protein-specific additive offset** in `dG` is exactly what this predicts — which is what
> `b_p` is: std 1.5741, **ICC 0.898** across checkpoints (a stable property of the learned
> function, not noise), reproducing across 10/10 checkpoints.
>
> This is the first account of `b_p` sourced entirely from the pre-training distribution rather
> than fitted on the 28, and it is consistent with three things we already measured and could not
> previously explain together:
> - the `b_p` corrector **fails out of sample** — because within 43–72 aa there is no length
>   variation left to fit against (Section 3.1 proves this);
> - **`a_p` is fine** (median 0.499, per-protein PCC 0.798) — `a_p` is a *within-protein slope*
>   and a multiplicative mis-scaling largely cancels in it;
> - the **coil is our best `b_p` lever** (dG MAE 4.9650 → 3.9521) — the coil changes the
>   *unfolded reference state*, which is exactly the term whose absolute scale pre-training
>   never constrained.

---

## 3. CAN THIS FILE IMPROVE THE MODEL? — the honest answer

### 3.1 The negative result that must be stated

The tempting move is: turn the length/energy-scale law into a per-protein correction and feed it in.
**I tested it and it does not work.** Four catalogue-derived predictors, evaluated on the 28
(r_crit at p=0.05, n=28 is **0.3739**):

| predictor | → b_p | → abs_b_p | → a_p | → pcc | → b_p_wt_error |
|---|---|---|---|---|---|
| length | +0.085 | +0.363 | -0.284 | -0.292 | -0.315 |
| log10 length | +0.066 | +0.360 | -0.276 | -0.278 | -0.292 |
| predicted RMS scale | +0.070 | +0.361 | -0.277 | -0.281 | -0.297 |
| saturation deficit | -0.070 | -0.361 | +0.277 | +0.281 | +0.297 |

**Nothing clears r_crit.** The best is `abs_b_p` at r = +0.363 (p = 0.059) — and note it is the same
correlation four times over, because all four predictors are monotone functions of length.

**Why it fails, precisely:** our 28 span **43–72 aa**, a 1.67× range at the *bottom* of a curve whose
interesting variation is 20–250 aa. On the fitted law the predicted RMS scale varies only from
9.71 (L=43) to 10.85 (L=72) — an **11% spread**, against a `b_p` spread of ±1.57. The catalogue's
length signal is real and large *across the catalogue*, and **near-constant across our test set**.

> **This is the crucial distinction the report must not blur: the length effect explains why
> `b_p` EXISTS (a between-regime effect, measured on 100k) but cannot predict which protein gets
> which `b_p` (a within-regime effect, needing variation we do not have).**
> Any "insert a length column into the node features" proposal is a **DROP**. It would add a
> feature that is 100% constant within each protein — so it is invisible to `ddG` by the metric
> rule — and near-constant *across* our 28, so it cannot move `b_p` either. It is dead on both metrics.

### 3.2 What is actually worth doing

**A. Report the distribution shift as a thesis result. — DO-IT, zero cost, zero risk.**

Sections 2.1–2.4 are a complete, defensible, quantitative chapter subsection that requires no
training run, no join, and no new code in the model path. It converts "the model has an
unexplained per-protein offset" into "the model has a per-protein offset **because** it was
pre-trained on a distribution in which our test regime is 0.08% of residues, with a contrastive
objective that never constrained absolute energy scale, and it is measurably worst exactly there
(d = 0.96)." That is the difference between a limitation and a **finding**.

Concretely, add to the thesis: Finding 2.1 (0.0805% of residues), the monomer loss-vs-length table
in 2.2, the ANCOVA in 2.3 ruling out the method confound, and the power law + saturation in 2.4 as
the `b_p` mechanism. State 3.1 as a negative control in the same breath — it is what makes the
claim honest rather than convenient.

**B. Length-stratified re-pre-training. — MAYBE, correct but out of scope.**

The mechanistically right fix is to remove the shift at its source: resample the pre-training set
so short monomers carry their share of the gradient. Feasibility is real —

| pool | n | upsample to reach 20% of a 100k epoch |
|---|---|---|
| monomer 20–72 aa | 2,604 | 7.7× |
| monomer 20–100 aa | 5,186 | 3.9× |
| monomer 20–150 aa | 10,771 | 1.9× |

The 20–150 aa pool needs only **1.9×**, which is a mild reweight, not a fabrication. Concretely:
add a per-example sampling weight `w_i = 1` for `L > 150` and `w_i = 1.9` for monomeric `L ≤ 150`
in the pre-training `DataLoader`'s `WeightedRandomSampler`, keeping epoch size fixed.

**Why it is still a MAYBE:** it requires re-running pre-training on GPU (explicitly out of bounds
here), it changes the checkpoint every downstream measurement is anchored to (ICC 0.898, the 10-checkpoint
replications, all `b_p` numbers), and the fine-tuning stage may already absorb part of it. **Recommend:
write it up as the identified root cause and the principled fix, and scope it as future work.**
If a single GPU run were ever affordable, the prediction to test is sharp and falsifiable:
*length-balanced pre-training should reduce `std(b_p)` below 1.5741 while leaving `a_p` (median 0.4990)
and per-protein PCC (0.798) essentially unchanged*, because the shift is a reference-state effect.

**C. Fix `catalogue_schema.py`. — DO-IT, small, and it prevents a real error.**

Three corrections, all justified above:
1. `lossc` is **not** meaningless — document the exact decode `RMS|E| = sqrt(lossc/0.1)` from
   `evaluate_train.py:232-237`, and note the CSV predates the commit that zeroed it.
2. Document `loss = lossd + lossc` exactly (R²=1, residual 1e-6) so nobody treats them as
   three independent signals; and `Spearman(rank, loss) = -1.0` so nobody analyses `rank`.
3. **Add a hard guard on `AA Length`**: it is assembly length, not chain length (Finding 1.4).
   `load_catalogue()` should expose `n_chains` (parsed from `Oligomeric State`) and
   `chain_length = AA Length` **defined only where `Oligomeric State == 'Monomer'`, NaN elsewhere**.
   This is the highest-value change in the whole report: without it the next person to touch this
   file will silently correlate against ribosome assembly sizes. Also record that the 160
   validation rows are metadata-blind (Finding 1.1), and split the BSA verdict three ways
   (Finding 1.6) instead of calling all four BSA columns dead.

**D. Everything else. — DROP.**

- **Length / n_chains / BSA / ligand-count as node features:** dead by the metric rule (constant
  within a protein ⇒ cancels in `ddG`) *and* dead on `b_p` (near-constant across the 28, per 3.1).
- **Pre-training loss as a per-protein prior:** requires the join. The join is 0/28.
- **Ligand features from `Ligands_y`:** all 28 are ligand-free; the column is constant on the test set.
- **Anything using `rank`, `lossg`, `Global Symmetry`, `Ligands_x`, `BSA_Numeric_x`, `BSA_Percentage`,
  `Is Complex?`, `PDB_ID_and_Entity`:** dead or duplicate, as established.

---

## 4. Reproduction

All figures above come from `data/FINAL_DATASET_100k_030926.csv` on the cluster, read with
`pd.read_csv(..., low_memory=False)`. Two coercions are mandatory and are the source of most
mistakes with this file:

```python
L  = pd.to_numeric(df['AA Length'], errors='coerce')          # 'Unknown' -> NaN (538 rows)
nch = df['Oligomeric State'].str.extract(r'\((\d+) chains\)')[0].astype(float)
nch = pd.Series(np.where(df['Oligomeric State']=='Monomer', 1.0, nch), index=df.index)
chain_len = L.where(df['Oligomeric State']=='Monomer')        # assembly length is NOT chain length
rms_E = np.sqrt(df['lossc'] / 0.1)                            # reg_alpha = 0.1
```

Ligand classification treats as **non-ligand**: nucleic-acid chain residues (`A C G U DA DC DG DT`),
crystallisation additives (`SO4 GOL PO4 EDO PEG MES TRS ACT ACY CL NA CIT MPD DMS FMT IOD BR NO3
IMD BME EPE 1PE PG4 PGE TLA SCN CAC MRD BEN DTT`), bare metals (`ZN MG CA MN FE FE2 NI CU CU1 CO
CD K HG CS SR BA PT AU AG W MO V`), modified residues (`MSE SEP TPO PTR CSO CME CSD KCX LLP MLY
CGU HYP PCA SAC ALY`), and placeholders (`UNK UNX NH2 ACE`).

Gate after this work: **`dG=-0.0030 width=1092`** — unchanged, as this report adds no code to the
model path.
