# Catalogue integrity: defects, measurements, and the guards that now prevent them

Subject file: `data/FINAL_DATASET_100k_030926.csv` — 100,246 rows x 21 columns.

Code: `scripts/catalogue_schema.py` (guarded loader), `scripts/gate_catalogue_integrity.py`
(self-test, 24 checks), `data/het_classification.csv` (8,187 het codes),
`scripts/build_het_classification.py` (builder).

Every number below was re-measured on the raw file for this document. Nothing is inherited.

---

## 0. SCOPE — what this catalogue may and may not be used for

**This catalogue IS the PDB pre-training decoy set. It does NOT join our 28 test proteins:
0/28, verified two ways.**

| verification | method | result |
|---|---|---|
| way 1 | exact 4-char PDB code matched against `protein_id` and `PDB_ID_and_Entity` prefixes | **0 / 28** |
| way 2 | case-insensitive substring scan of both id columns over all 100,246 rows | **0 / 28** |

The 28 test proteins are `1GYZ 1PSE 1QKH 1QP2 1TUC 1W4H 2BTH 2K1B 2K28 2K5H 2KVS 2KWH 2KXD
2L33 2LQK 2WXC 3DKM 4C26 6EWS 6EWT 6EWU` plus seven designed folds (`HEEH_KT_rd6_0746`,
`HEEH_KT_rd6_0793`, `HHH_rd1_0142`, `HHH_rd1_0244`, `r11_1081_TrROS_Hall`,
`r12_757_TrROS_Hall`, `r18_3_TrROS_Hall`). The designed folds have no PDB entry by
construction; none of the 21 PDB-like names appears anywhere in the catalogue.

This is **expected and correct** — the two sets are disjoint by design. The consequences are
absolute:

- The catalogue can **never** be used to score, calibrate, or validate our 28 test proteins.
  A join appearing in future would mean *leakage*, not good luck. The self-test asserts the
  join stays empty (`scope: catalogue does NOT join our 28`).
- Its only legitimate use is **building features for a future ligand-bearing benchmark** —
  a benchmark our 28 monomers cannot supply, since all 28 are single-chain, ligand-free,
  metal-free monomers of 43–72 aa. There, ligands, complexes and interface area have **zero
  variance**: "not measurable", NOT "no effect".

---

## 1. DEAD COLUMNS that look alive

Five columns are present and inviting, and none of them carries usable data. A column can
be dead in **three** distinct ways, and this matters — a single null-count rule misses one
of them (see §1.1, which is a bug this work actually hit).

| column | measured | kind of death | live sibling |
|---|---|---|---|
| `Ligands_x` | nonnull=**4** / 100,246 | sparse | **`Ligands_y`** (75,464 nonnull, 8,187 codes) |
| `BSA_Percentage` | nonnull=**5** / 100,246 | sparse | **`BSA_Numeric_y`** |
| `BSA_Numeric_x` | nonnull=100,246, nuniq=**4** | **degenerate** | **`BSA_Numeric_y`** (4,867 values) |
| `lossg` | constant **0** | constant | none (see §4) |
| `Global Symmetry` | constant **'Asymmetric'** | constant | `Oligomeric State` (104), `Chain Composition` (6) |

`BSA_Numeric_x` recycles exactly four values — `0.0, 18.76, 26.99, 35.95` — across all
100,246 rows. It is a broken merge artefact, not a measurement. `Ligands_x` has literally
four values in the whole file (`ZN`; `CL, GCO, GOL, MG`; …).

### 1.1 The trap that this class of defect sets — hit during this work

The obvious guard is "dead == mostly null or constant". That guard **passes
`BSA_Numeric_x` as alive**: it is 100% non-null and not constant. It is the *healthiest-
looking* column of the five and one of the deadest. The first version of the loader written
here used exactly that blanket rule, and the self-test caught it (15/23 failing). The fix
was to declare deadness *per column and by kind* in `catalogue_schema.DEADNESS`:

```python
DEADNESS = {
    "Ligands_x":       ("sparse",     16),   # max non-null rows tolerated
    "BSA_Percentage":  ("sparse",     16),
    "lossg":           ("constant",    1),   # max distinct values tolerated
    "Global Symmetry": ("constant",    1),
    "BSA_Numeric_x":   ("degenerate",  8),   # max distinct values tolerated
}
```

A self-test check (`dead: every dead column has a kind`) asserts the `degenerate` kind stays
declared, because that is the one a naive rewrite would drop.

### GUARD

`load_catalogue()` physically **drops** all five columns, and `GuardedCatalogue` raises
`DeadColumnError` on any attempt to read one — via `df["Ligands_x"]`, via `df.Ligands_x`,
and via a list-select `df[["BSA_Numeric_y", "Ligands_x"]]`. The message **names the live
sibling**:

```
'Ligands_x' is a QUARANTINED DEAD COLUMN and was removed by
catalogue_schema.load_catalogue(). Ligands_x has only 4 non-null rows out of 100,246
(a silent NaN for everyone else). USE THE LIVE SIBLING: 'Ligands_y' (75,464 non-null,
8,187 distinct het codes, 205,648 occurrences).
```

`load_catalogue(strict=True)` additionally **re-measures deadness on every load** and
refuses to return data if a quarantined column has come back to life — so a future rebuild
of the CSV that genuinely fixes `Ligands_x` forces a deliberate decision instead of
silently changing results.

---

## 2. `BSA_Numeric_y` is a PERCENTAGE, not Å²

Measured: range **[0.00, 70.08]**, 0 values above 100, 0 below 0, 246 NaN, and
**31,277 rows (31.3%) exactly 0**.

The name says "Numeric" and the neighbouring column is called "BSA", which invites reading
it as buried surface area in Å² — where 70 would be a rounding error rather than a large
interface. Any threshold, normalisation, or unit conversion built on that misreading is
silently wrong by a factor of ~10³.

### GUARD

`BSA_UNITS = "percent"` and `BSA_RANGE = (0.0, 100.0)` are module constants; the loader
asserts the observed range lies inside `[0,100]` and attaches `attrs["bsa_units"]`. If the
file is ever regenerated in Å², the loader **refuses to return data** rather than
rescaling silently — verified by feeding it a copy with BSA multiplied by 100
(`bsa: Angstrom^2 magnitudes refused`).

---

## 3. 48 rows that disagree with themselves

Measured: **48 rows** where the free-text `BSA` string parses to a nonzero percentage
(e.g. `"18.76%"`, `"30.58%"`, `"34.96%"`) while `BSA_Numeric_y` is exactly `0.0`.

The decisive measurement is what happens **everywhere else**:

> over the 99,721 rows where both representations exist and do not conflict,
> **max |string − numeric| = 0.000000000** — exact agreement, to the last digit.

### POLICY: treat as MISSING (NaN), not as zero — and why

Exact agreement everywhere else establishes that `BSA_Numeric_y` is a *mechanical copy* of
the string. That makes these 48 a **failed extraction**, not a measured zero. A row whose
own string says `18.76%` is not a row with no buried surface.

Keeping the `0` would inject 48 false "fully exposed" monomers directly into the **left
tail** of any BSA feature — exactly the tail a burial or interface lever is most sensitive
to, and exactly where 48 fabricated points do the most damage. Dropping the rows discards
real information in the other 19 columns. Setting them to `NaN` states what is actually
known — the percentage is unrecoverable from this file — and every downstream estimator
already handles NaN.

### GUARD

`bsa_policy="missing"` is the **default**. The loader exposes `BSA_inconsistent` (a boolean
column) and `BSA_parsed_pct` so the disagreement is inspectable rather than buried.
`bsa_policy="drop"` is available for analyses that cannot carry NaN. `bsa_policy="keep_zero"`
exists **only** to reproduce the historical bug and emits a `RuntimeWarning` saying so. The
self-test asserts all 48 are NaN and none is 0, and that the exact-agreement premise still
holds — if the premise ever breaks, the policy must be re-justified rather than inherited.

---

## 4. The decoy-loss columns cannot answer a ddG question

`loss`, `lossd`, `lossg`, `lossc` are **decoy-discrimination** losses on PDB structures.

### The historical "-0.001 BSA null", reproduced exactly

| correlation | measured | verdict |
|---|---|---|
| `r(BSA_Numeric_x, loss)` | **−0.001181** | the historical null — **void twice over** |
| `r(BSA_Numeric_y, loss)` | **+0.1059** | wrong metric, so still uninformative |
| `r(BSA_Numeric_y, lossd)` | +0.1266 | wrong metric |
| `r(BSA_Numeric_y, lossc)` | −0.1199 | wrong metric |

The famous "-0.001" reproduces **to the digit** — and it came from `BSA_Numeric_x`, the
**dead** column. So it was void twice: a dead column, scored against a metric that cannot
answer the question. The live column on the same metric gives +0.106, two orders of
magnitude larger. Neither number means anything for ddG; the contrast simply proves the
null was an artefact of reading a dead column, not a finding about BSA.

### THE METRIC RULE (enforced in code)

> `ddG = dG_mut − dG_wt` cancels anything identical between WT and mutant. A reference-state
> or whole-protein lever must therefore be scored on **dG** or on the per-protein offset
> **b_p** — never on pooled ddG, and never against the pre-training decoy loss. `a_p` is
> within-protein and does **not** cancel, so it is a legitimate target.

The decoy objective is already known not to predict ddG, which makes a correlation against
it uninformative **in both directions**: a null does not mean the feature is dead, and a hit
would not mean it is alive. This rule has already caught three levers — the Flory coil, BSA,
and W5 burial.

### GUARD

`assert_not_decoy_loss()` and `safe_correlate()` raise `MetricRuleError` for any of the four
loss columns, with the full rule in the message. `safe_correlate()` also refuses dead
columns. Legitimate pairs still compute normally.

---

## 5. Ligands: 63.4% of het occurrences are not bound ligands at all

Measured from `Ligands_y`: **205,648 het-code occurrences over 8,187 distinct codes**.

`data/het_classification.csv` classifies every code into seven categories using PDB
chemical-component conventions:

| category | occurrences | share | counts as a bound ligand? |
|---|---|---|---|
| crystallization_additive | 60,238 | 29.29% | **no** |
| metal | 40,874 | 19.88% | yes |
| unknown | 27,862 | 13.55% | **no** |
| polymer_residue | 27,326 | 13.29% | **no** |
| substrate | 22,294 | 10.84% | yes |
| modified_residue | 14,939 | 7.26% | **no** |
| cofactor | 12,115 | 5.89% | yes |

**Coverage, stated honestly.** 456 codes are hand-curated, covering **99.81%** of the
top-200 codes by frequency. Over the whole file there are two defensible coverage numbers
and they mean different things, so both are given:

- **177,786 / 205,648 = 86.45%** of occurrences receive a *chemically meaningful* category.
  **This is the number to quote.**
- **180,350 / 205,648 = 87.70%** of occurrences are *curated* in the wider sense — the
  above plus 2,564 occurrences of codes that are deliberately uninterpretable: `UNK`
  (1,287), `UNX` (565), `UNL` (500) are the PDB's own "unknown atom/ligand/residue"
  placeholders, and `ERROR` (212) is a literal error string in the source column. We know
  what these are; they simply carry no chemistry.

The uncurated remainder is a long tail of 7,700+ rare codes, each appearing a handful of
times; it is labelled `unknown` with `curated=no`, never guessed at. `unknown` does **not**
count as a bound ligand in either sense — an unidentified code is not evidence of binding,
and neither is an explicit `UNK`.

### Why this is a biology problem, not a bookkeeping problem

- **Crystallization additives (29.3%)** — SO4, GOL, EDO, PEG, MPD, ACT, PO4, CL. These come
  from the crystallization buffer and the cryoprotectant. They do not stabilise the protein
  *in vivo*. `SO4` alone is the single most frequent code in the file (12,140).
- **Modified residues (7.3%)** — MSE, SEP, TPO, PTR, CSO, KCX. `MSE` is **selenomethionine:
  an amino acid IN the polypeptide chain**, incorporated for MAD/SAD phasing. It is the 5th
  most frequent code (9,072). Counting it as a bound ligand is simply wrong.
- **Polymer residues (13.3%)** — A, U, C, G, DA, DT, DC, DG, PSU, 5MU. These are
  **nucleotides IN a DNA/RNA chain**, i.e. monomers of a covalent polymer, exactly the same
  category error as MSE. They occupy 8 of the top 20 slots. This category was added on top
  of the original scheme precisely because folding them into "substrate" would repeat the
  MSE mistake for a *larger* share of the data.

**A ligand feature built on the raw `Ligands_y` column would be learning crystallography and
polymer bookkeeping, not biochemistry.** In a 3,000-row sample the raw het count is 5,932
while the biologically-defensible bound-ligand count is 2,054, and **564 rows carry only
crystallization additives**.

### GUARD

`classify_ligands(df)` returns per-category counts plus `n_bound_ligand`
(= metal + cofactor + substrate) and an `additives_only` flag. The self-test asserts
SO4/GOL/EDO/PEG/MPD/ACT are additives, MSE/SEP/TPO/PTR are modified residues, A/U/DA/DT are
polymer residues, all of them excluded from `n_bound_ligand`, and that HEM/FAD/NAD/FMN *are*
counted — so the filter cannot be quietly inverted either.

---

## 6. The self-test

`python scripts/gate_catalogue_integrity.py` — **24 checks, ALL PASS**, exit 0.

It was verified to actually fail by mutating the code four ways and confirming each
mutation is caught, then restoring:

| adversarial mutation | result |
|---|---|
| A. remove `Ligands_x` from the quarantine | **4/24 FAILED** |
| B. flip the default BSA policy back to `keep_zero` | **1/24 FAILED** — `48 flagged rows are still numeric` |
| C. empty `DECOY_LOSS_COLUMNS` so the loss is allowed through | **1/24 FAILED** — `safe_correlate computed BSA vs decoy loss - the exact void-twice-over mistake` |
| D. relabel `MSE` as a bound substrate | **1/24 FAILED** — `MSE must be modified_residue` |
| restored | **ALL 24 PASS** |

A guard that has never been seen to fail is not a verified guard. These four were.

---

## 7. Effect on the model: none

These changes are **analysis-side only**. No file under the feature-vector or training path
was touched. Confirmed by re-running the feature-vector gate after all changes:

```
G4-CPU: ALL PASS
baseline    forward ok, dG finite    PASS dG=-0.0030 width=1092
```

`dG=-0.0030 width=1092` — unchanged, as required.

---

## Summary table

| # | defect | measurement | guard |
|---|---|---|---|
| 1 | 5 dead columns look alive | `Ligands_x` 4 nonnull; `BSA_Numeric_x` 4 uniq/100,246; `BSA_Percentage` 5 nonnull; `lossg`≡0; `Global Symmetry`≡'Asymmetric' | dropped + `DeadColumnError` naming the live sibling; deadness re-measured per kind on every load |
| 2 | `BSA_Numeric_y` misread as Å² | range [0.00, 70.08], 31.3% exact 0 | documented as percent, asserted in [0,100], Å² magnitudes refused |
| 3 | 48 self-inconsistent rows | string nonzero vs numeric 0; exact agreement (0.000000000) on the other 99,721 | default policy NaN, not 0; `keep_zero` warns; premise re-asserted |
| 4 | decoy loss used for ddG | `r(BSA_Numeric_x, loss) = −0.001181` reproduces the historical null | `MetricRuleError` on all four loss columns, rule in the message |
| 5 | het codes ≠ ligands | 63.4% of 205,648 occurrences are additive/modres/polymer/unknown | 7-category table, `n_bound_ligand` excludes non-ligands; 86.45% of occurrences chemically classified (99.81% of top-200) |
| 6 | guards could silently rot | 4 adversarial mutations all caught | 24-check self-test, exit 1 on failure |
