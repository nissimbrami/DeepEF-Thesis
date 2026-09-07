# Item 40 — The 100k Structure Catalogue: First Read

`FINAL_DATASET_100k_030926.csv` (17.2 MB, 100,246 rows x 21 columns), supplied by Nissim, opened here for the first time.

## THE CAVEAT THAT GOVERNS EVERY NUMBER BELOW

The `loss` column in this file is the **pretraining decoy loss**. That objective has already been measured and found not to help ddG. Therefore:

> **A correlation in this file is evidence about what makes a structure easy or hard for the pretraining decoy task. It is NOT evidence about ddG.**

Nothing in this report is a recommendation to add a feature to the ddG model. The catalogue's honest use is (a) diagnosing the pretraining corpus and (b) generating structure-level hypotheses that would still need independent testing on a ddG target. Every finding below carries this caveat; I do not repeat it each time.

---

## 1. The real schema

| # | Column | dtype | non-null | missing | n unique | Verdict |
|---|--------|-------|---------:|--------:|---------:|---------|
| 1 | `rank` | int64 | 100,246 | 0 | 100,246 | **Redundant** — Spearman with `loss` is exactly -1.0000 |
| 2 | `protein_id` | str | 100,246 | 0 | 100,246 | Primary key |
| 3 | `split` | str | 100,246 | 0 | 2 | training 100,086 / validation 160 |
| 4 | `loss` | float64 | 100,246 | 0 | 96,925 | Total decoy loss |
| 5 | `lossd` | float64 | 100,246 | 0 | 99,105 | Decoy term |
| 6 | `lossg` | int64 | 100,246 | 0 | **1** | **Dead — constant 0** |
| 7 | `lossc` | float64 | 100,246 | 0 | 99,536 | Contrastive term |
| 8 | `PDB_ID_and_Entity` | str | 100,246 | 0 | 100,246 | Duplicate of `protein_id` |
| 9 | `Method` | str | 100,246 | 0 | 10 | Experimental method |
| 10 | `Resolution (Å)` | float64 | 91,084 | 9,162 | 2,452 | **The useful quality column** |
| 11 | `AA Length` | str | 100,246 | 0 | 3,261 | 538 are the string `"Unknown"` |
| 12 | `Oligomeric State` | str | 100,246 | 0 | 104 | Monomer 31,852 / Complex(2) 29,253 / … |
| 13 | `Is Complex?` | str | 100,246 | 0 | 3 | Yes 67,856 / No 31,852 / Unknown 538 |
| 14 | `Ligands_x` | str | **4** | 100,242 | 4 | **Dead — merge artifact** |
| 15 | `Global Symmetry` | str | 100,246 | 0 | **1** | **Dead — constant "Asymmetric"** |
| 16 | `Chain Composition` | str | 100,246 | 0 | 6 | homomeric 66,336 / heteromeric 22,775 / … |
| 17 | `BSA_Percentage` | str | **5** | 100,241 | 4 | **Dead — merge artifact** |
| 18 | `BSA` | str | 100,246 | 0 | 4,869 | **Live BSA**, `"12.34%"` strings; 233 unparseable (`"Error"`) |
| 19 | `BSA_Numeric_x` | float64 | 100,246 | 0 | **4** | **POISONED — 0.00 in 100,243 of 100,246 rows.** See section 3 |
| 20 | `Ligands_y` | str | 75,464 | 24,782 | 21,650 | Live ligand list, comma-separated |
| 21 | `BSA_Numeric_y` | float64 | 100,000 | 246 | 4,867 | Live numeric BSA, agrees with `BSA` |

**Five of 21 columns carry no information.** `lossg` and `Global Symmetry` are constants; `Ligands_x`, `BSA_Percentage` and `BSA_Numeric_x` survive from a botched merge and are populated in 5 rows or fewer; `rank` is a monotone restatement of `loss`; `PDB_ID_and_Entity` duplicates `protein_id`. The genuinely independent annotation set is seven columns: Method, Resolution, AA Length, Oligomeric State / Is Complex?, Chain Composition, BSA, Ligands_y.

**Loss decomposition verified:** `loss == lossd + lossc + lossg` to a max absolute deviation of 1e-6 across all 100,246 rows. Since `lossg` is identically 0, the loss is exactly `lossd + lossc`. Note that `lossd` (mean -38.2) and `lossc` (mean +20.1) pull in opposite directions and are individually about 3x the magnitude of their sum — the aggregate `loss` is a difference of two large opposed terms, so correlations against `loss` alone can hide cancelling effects. I report all three throughout.

**Two parsing traps for anyone using this file.** `AA Length` is dtype `str`, not int (538 `"Unknown"` values force object dtype — a naive `df['AA Length'].mean()` fails and a naive sort is lexicographic). `BSA` is a percent *string* requiring `.str.replace('%','')` before coercion.

---

## 2. The four headline correlations — three reproduce, one does not

Pearson against total `loss`, computed on all available rows:

| Claim | Established value | **Reproduced** | Holds? |
|---|---|---|---|
| Resolution — strongest annotation | +0.158 | **+0.1582** (n=91,084, p about 0) | YES, exact |
| AA Length | +0.070 | **+0.0698** (n=99,708) | YES, exact |
| Complex vs monomer, "0.2 out of 18" | 0.2 | **0.2028** (-18.051 vs -18.254) | YES, exact |
| Interface BSA | -0.001 | **-0.0012 … but see section 3** | **Reproduces only on a dead column** |

Three of the four are confirmed to the reported precision. The complex-vs-monomer gap of 0.2028 is Cohen's d = 0.059 against a loss sd of 3.46 — negligible, and correctly CLOSED.

Spearman diverges sharply from Pearson for AA Length (+0.070 Pearson vs **-0.066** Spearman — the sign flips). Length's linear correlation is driven by a heavy right tail, not by a monotone trend. Any claim about length should be stated as rank-based or not at all.

---

## 3. The BSA result is an artifact — the finding should be reopened

**The -0.001 BSA correlation was computed against `BSA_Numeric_x`, which is 0.00 in 100,243 of 100,246 rows.** It is a correlation with a near-constant. It cannot be anything but zero, and it is not evidence that BSA is uninformative.

Correlating the same target against the columns that actually contain the data:

| Column | Live? | n | Pearson vs loss |
|---|---|---:|---:|
| `BSA_Numeric_x` | **No — 4 real values** | 100,246 | **-0.0012** (the reported result) |
| `BSA_Numeric_y` | Yes | 100,000 | **+0.1059** |
| `BSA` parsed from percent strings | Yes | 100,013 | **+0.1074** |
| `BSA` restricted to BSA > 0 | Yes | 68,865 | **+0.1467** |
| `BSA` within `Is Complex? == Yes` | Yes | 67,789 | **+0.1465** |

The real BSA correlation is **+0.107**, rising to **+0.147** once the 31,148 monomers (structurally forced to BSA = 0) are excluded. That is comparable in magnitude to resolution's +0.158 — i.e. BSA is roughly as strong as the annotation currently believed to be the strongest, not null.

**Recommendation: reopen the "interface BSA correlates -0.001" finding as not established.** The correct statement is that interface BSA correlates **+0.15 with the pretraining decoy loss among complexes**. Whether that means anything for ddG is a separate and untested question — larger interfaces co-occur with larger, harder assemblies, so this may be a size proxy rather than an interface effect (the OLS in section 4 partly addresses this: BSA survives length control).

One caution against over-reading this. The fix does not overturn the *conclusion* that BSA is not worth pursuing for ddG. It overturns the *evidence*. The conclusion may still be right for the reason stated in the governing caveat — this is a pretraining correlation.

---

## 4. W8 substitute: resolution DOES support a structure-level tercile analysis

W8 (pLDDT) is blocked because the preprocessed tensors carry no per-residue confidence and there are no PDB files to recompute it from. The question posed was whether this catalogue carries a per-structure quality column that could support a **structure-level** tercile analysis instead. It does, and the analysis is clean.

`Resolution (Å)` is present for 91,084 of 100,246 structures (90.9%). Its missingness is not random but is fully explained and benign — it is missing exactly where resolution is undefined:

| Method | n | Resolution present | Mean res |
|---|---:|---:|---:|
| X-RAY DIFFRACTION | 88,879 | **88,879 (100%)** | 2.18 |
| SOLUTION NMR | 8,545 | 0 (undefined for NMR) | — |
| ELECTRON MICROSCOPY | 2,161 | 2,144 | 8.34 |
| Unknown | 538 | 0 | — |
| all others | 123 | 61 | — |

**Restricting to X-ray gives a complete-case cohort of 88,879 structures with zero missingness** — a better-powered stratification than most W-items get. Resolution there ranges 0.48 to about 4 Å with median 2.10.

Note that raw resolution has a max of 70.0 Å, from low-resolution EM entries. Any use of this column must filter by method; a global tercile split would put EM structures in the worst tercile for reasons unrelated to crystallographic quality.

### Tercile result (X-ray only, n = 88,879)

| Tercile | n | Resolution range | Mean loss | Median length |
|---|---:|---|---:|---:|
| T1 best | 31,637 | 0.48 - 1.90 | **-18.72** | 374 |
| T2 mid | 27,700 | 1.90 - 2.35 | -18.42 | 517 |
| T3 worst | 29,542 | 2.35 - ~4 | **-17.99** | 816 |

T3 minus T1 = **+0.733**, or 0.246 sd of the X-ray loss distribution; Mann-Whitney p about 0. Monotone across all three terciles. Better-resolved structures have systematically lower decoy loss.

### The length confound, and why the result survives it

Resolution correlates +0.32 (Spearman +0.41) with chain length — bigger assemblies diffract worse — and T3's median length is 816 vs T1's 374. So the naive tercile gap is partly a length effect. Stratifying resolution terciles **within** length quartiles:

Mean loss, length-quartile x resolution-tercile:

| | R1 best | R2 | R3 worst | **R3 - R1** |
|---|---:|---:|---:|---:|
| L1 (shortest) | -18.412 | -18.041 | -17.701 | **+0.711** |
| L2 | -18.936 | -18.622 | -18.093 | **+0.844** |
| L3 | -18.960 | -18.657 | -18.204 | **+0.756** |
| L4 (longest) | -18.928 | -18.754 | -17.904 | **+1.024** |

**The resolution effect is present, monotone, and of full magnitude inside every length quartile** — it is not a length proxy. Cell counts are 3,421 to 12,653, all well-powered.

OLS on standardized predictors (X-ray, complete cases, n = 88,838):

```
loss ~ z(resolution) + z(log10 length) + z(BSA%)
  resolution    beta = +0.551   t = +51.3
  log10 length  beta = -0.584   t = -48.5
  BSA%          beta = +0.733   t = +67.0
  R2 = 0.0685
```

All three are independently significant with the same sign, and **BSA has the largest standardized coefficient of the three** — further confirmation that section 3's finding is real and not a length artifact. But note R2 = 0.069: together these three annotations explain under 7% of decoy-loss variance. The effects are statistically overwhelming (n = 89k) and practically small. That combination is exactly what one should expect from metadata proxies, and it is the honest headline: **real, reproducible, monotone, and small.**

### The paragraph this was worth

Resolution is a legitimate structure-level substitute for the blocked per-residue pLDDT analysis, and the substitution is favorable in one respect and unfavorable in another. Favorably: coverage is 100% within X-ray, the stratification is clean, the effect survives the obvious confound, and n = 89k gives power that a per-residue analysis on the preprocessed tensors could not have matched. Unfavorably, and decisively: **resolution is a per-structure scalar, so it can only support structure-level weighting or filtering — it cannot become a per-residue input feature, which is what W8 was actually for.** pLDDT's value was that it varies along the chain and could flag which *residues* are unreliable; resolution assigns one number to all of them. So this does not unblock W8; it answers a different, coarser question. What it genuinely enables is a **corpus-curation experiment**: retrain pretraining on the best resolution tercile only, or resolution-weight the decoy loss, and see whether the resulting representation transfers better. That is a real, runnable experiment nobody has proposed. It is also, per the governing caveat, an experiment about pretraining — and pretraining has already been measured as not helping ddG, which caps the expected payoff of the whole line.

---

## 5. Genuinely unexamined findings

Each carries the governing caveat: these are pretraining-decoy-loss effects.

**(a) The 160 validation rows have NO annotations at all — and the cause is a fixable bug.** Every metadata column is `"Unknown"` (or `"Error"` for BSA) for all 160 validation rows; resolution is null for all of them. The reason is visible in the IDs: validation `protein_id`s carry a numeric prefix — `20#2Y1B_1_A`, `10#2FK5_1_A`, `40#3KRA_2_B` — where training IDs are bare (`3F3Q_1_A`). **The annotation join never stripped the `NN#` prefix, so all 160 failed to match.** This is the single most actionable item in the file: any attempt to use these annotations at evaluation time will silently see 100% missing data on the validation set. A regex strip of `^\d+#` should recover all 160. Worth noting the validation split also has a much higher mean loss (-15.68 vs -18.11) and far heavier tails — it is a deliberately harder set, so it is not interchangeable with a random training slice.

**(b) Method is a much larger effect than any continuous annotation, and is confounded with everything.** Mean loss by method: X-ray -18.42 (n=88,879), Solution NMR -15.67 (n=8,545), EM -15.43 (n=2,161). The NMR/EM penalty is **about 2.8 to 3.0 loss units — roughly 4x the entire resolution tercile spread** and about 0.85 sd. NMR and EM structures are systematically much harder for the decoy task. Nobody has looked at this. It is a plausible confound sitting underneath the resolution finding: since resolution is *undefined* for NMR, any analysis that drops missing-resolution rows silently drops the hardest 11% of the corpus. This is a reason to be more confident in the X-ray-only numbers in section 4 (which are immune) and less confident in any whole-corpus statistic.

**(c) `Chain Composition` separates cleanly and is unexamined.** Homomeric protein -18.51 (n=66,336) vs heteromeric -17.18 (n=22,775) vs protein/NA -17.18 (n=5,978). The homomeric/heteromeric gap is **1.34 units**, about 6x the complex-vs-monomer gap that was investigated and closed. The far more informative question was adjacent to the one asked: *whether* a structure is a complex barely matters (0.20), but *what kind* matters substantially (1.34). Nucleic-acid-containing structures (5,978 protein/NA) are also worth flagging as a possible contamination stratum — the model is being pretrained on 6% structures whose environment includes DNA/RNA it cannot represent.

**(d) Ligand count is non-monotone.** Pearson with loss is -0.0009 — apparently null — but the relationship is an inverted U: 0 ligands -17.49, 1 ligand -18.16, 2-3 -18.61, 4-7 -18.21, 8+ -16.64. Loss is *lowest* (best) in the middle. Reporting the Pearson alone would have called this dead; it is not, it is just non-linear. Mechanistically plausible — apo structures and heavily-liganded ones are both atypical. Low priority, but a clean example of why the linear screen used on this file is inadequate.

**(e) The 378 non-validation `Unknown` rows are junk.** Beyond the 160 validation rows, 378 training rows have Method=Unknown *and* AA Length=Unknown *and* Is Complex?=Unknown simultaneously — total annotation failure, mean loss -16.99. These 538 rows should be excluded from any annotation analysis rather than treated as a category.

**(f) `lossd` and `lossc` should be analyzed separately, and they disagree in sign.** Against resolution: `lossd` +0.140 but `lossc` **-0.109**. Against BSA: `lossd` +0.128, `lossc` **-0.120**. The two loss terms respond to annotations in *opposite directions*, and the aggregate `loss` correlation is a partial cancellation of two larger, opposed effects. Every headline correlation in this file understates the underlying structure. Anyone continuing this analysis should use `lossd` and `lossc` as separate targets.

---

## 6. Summary

- **Schema:** 100,246 x 21. Five columns carry no information; two more are duplicates. Seven annotations are genuinely independent. `loss = lossd + lossc` exactly.
- **Headlines:** resolution +0.158, length +0.070, complex-vs-monomer 0.2028 — all three reproduce exactly. The BSA -0.001 does **not** hold; it was computed on a column that is constant in 99.997% of rows. Real value is +0.107, or +0.147 among complexes. **That finding should be reopened.**
- **W8 substitute:** resolution supports a clean structure-level tercile analysis on 88,879 X-ray structures with zero missingness. T3-T1 = +0.73 (0.25 sd), monotone, and it survives length stratification in every quartile. It does **not** unblock W8 — resolution is per-structure and cannot become a per-residue feature — but it enables a corpus-curation experiment (train on the best tercile, or resolution-weight the loss) that nobody has proposed.
- **Effect sizes are small:** the three continuous annotations jointly explain R2 = 0.069 of decoy-loss variance.
- **Most actionable defect:** the validation split's annotations are 100% missing because of an unstripped `NN#` ID prefix.
- **Largest unexamined effect:** Method (NMR/EM about 2.9 units worse than X-ray, about 4x the resolution effect).
- **All of the above concerns the pretraining decoy loss, which has already been measured as not helping ddG.** No claim here is evidence about ddG.

## Provenance

Analysis run locally against `C:\Users\User\Downloads\FINAL_DATASET_100k_030926.csv`; pandas 3.0.5 / numpy 2.1.3 / scipy 1.14.1. No model code, config, or repo file was modified. `scripts/gate_g4_cpu.py` was run after the analysis and returned `G4-CPU: ALL PASS` with `baseline PASS dG=-0.0030 width=1092` — unchanged, as expected for a read-only item.
