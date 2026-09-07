# Catalogue vs per-protein calibration (b_p, a_p) — the RIGHT metric

Eval CSV: `eval_results/abl_calib_ctrl_repro2_e14.csv` (reference control, 28 test proteins,
28,314 mutations). b_p / a_p computed with `cluster_run/code/calib_diag.py` conventions
exactly: group by protein, **row 0 = WT**, ddG = deltaG − deltaG[0], `np.polyfit(ddg_true,
ddg_pred, 1)` → (a_p, b_p). Also reported: `b_p_wt_error` = pred_deltaG[0] − deltaG[0], the
model's raw WT dG error (the dG-space offset the coil lever moved).

Artifacts: `results/catalogue_vs_bp.json`, script `scripts/catalogue_vs_bp.py`.

## 0. Why the previous check was wrong

A prior check correlated BSA against the **pre-training decoy loss** and got −0.001 →
"no signal". That is the wrong metric, exactly like the Flory coil. The decoy loss is
decoy discrimination on PDB structures and is already known not to predict ddG, so neither
a strong nor a weak correlation there carries information about our ddG performance.
Per THE METRIC RULE, a whole-protein property is identical between WT and mutant and
cancels in ddG — it can only be scored on **b_p / a_p / per-protein PCC**. That is what
is done here.

## 1. The catalogue join is EMPTY — 0 / 28

`data/FINAL_DATASET_100k_030926.csv`, 100,246 × 21. Columns present:

| role asked for | columns that exist |
|---|---|
| BSA / interface | `BSA`, `BSA_Numeric_y` (usable); `BSA_Percentage`, `BSA_Numeric_x`, — **degenerate**: 100,241/100,246 null and only 4 unique values |
| complex / oligomer / chain count | `Oligomeric State` (104 vals), `Is Complex?`, `Chain Composition`, `Global Symmetry` (constant = "Asymmetric") |
| ligand / HETATM / cofactor / metal | `Ligands_y` (21,650 vals, usable); `Ligands_x` — **degenerate**, 100,242 null |
| hydrophobicity | **none** |
| resolution | `Resolution (Å)` (9,162 null) |
| pLDDT | **none** |
| EC number | **none** |
| organism | **none** |
| length | `AA Length` (stored as object/string, not numeric) |

Plus pre-training loss columns `loss`, `lossd`, `lossg` (constant 0), `lossc`, and ids
`rank`, `protein_id`, `PDB_ID_and_Entity`, `split` (100,086 training / 160 validation).

**Join result: 0 of 28 matched.** This is not a formatting artifact. Tried: exact 4-char
PDB id (with the `NN#` prefix on validation rows stripped) against both `protein_id` and
`PDB_ID_and_Entity`, and a case-insensitive **substring scan of both id columns over all
100,246 rows**. Zero hits for every one of the 28. 21 of the 28 are PDB-like 4-char names
(`2K28`, `1PSE`, `6EWS`, …) and none appears anywhere in the catalogue; the other 7 are
designed folds (`HHH_rd1_0244`, `r12_757_TrROS_Hall`, …) which have no PDB entry by
construction. This is **expected and correct**: the catalogue is the pre-training decoy
set (PDB), the 28 are held-out MegaScale test proteins. The two sets are disjoint by
design. **The join was not fabricated and no catalogue number is quoted for our 28.**

## 2. What was done instead

Rather than declaring the item blocked, the covariates the catalogue *would* have supplied
were computed **directly on our own 28 test proteins** from their actual AlphaFold models
(`data/Processed_K50_dG_datasets/AlphaFold_model_PDBs/`, all 28 present) with Biopython
Shrake-Rupley SASA, plus `plddt.csv` (24/28 covered):

buried/exposed fraction and relative SASA (the within-protein analogue of BSA), total and
per-residue SASA, a compactness term `SASA/len^0.73`, Kyte-Doolittle hydropathy, exposed
hydrophobic fraction, charged/Gly/Pro fractions, length, chain count, HETATM/metal counts,
inter-chain BSA, and mean/min pLDDT.

**Important limitation, stated plainly:** on these 28, `n_chains` = 1, `n_het_residues` = 0,
`n_metal_residues` = 0 and `interchain_BSA` = 0.0 for **every** protein. The MegaScale test
set is 28 small (43–72 aa) single-chain, ligand-free monomers. So *complexity, ligands and
true inter-chain interface area have zero variance here and are unmeasurable in principle
on this test set* — no correlation can be computed, and none is reported.

## 3. Results — n = 28, |r| < 0.374 is indistinguishable from 0 at p = 0.05

90 feature × target tests were run, so the honest bar is Bonferroni: α = 0.05/90 = 5.6e-4.
**Five associations clear it**, and all involve **surface exposure vs the slope a_p**:

| feature | target | n | Pearson r (p) | Spearman ρ (p) | LOO range | natural-only (n=21) |
|---|---|---|---|---|---|---|
| `mean_rel_SASA` | **a_p** | 28 | **+0.714** (2.0e-5) | +0.623 (4.0e-4) | +0.64…+0.79 | +0.767 (4.9e-5) |
| `frac_exposed_rel_gt_0.5` | **a_p** | 28 | **+0.664** (1.2e-4) | +0.526 (0.004) | +0.54…+0.74 | +0.661 (0.001) |
| `mean_rel_SASA_hydrophobic` | **a_p** | 28 | **+0.628** (3.4e-4) | +0.545 (0.003) | +0.57…+0.69 | +0.609 (0.003) |
| `SASA_over_len_pow_073` | **a_p** | 28 | **+0.595** (8.4e-4) | +0.665 (1.1e-4) | +0.57…+0.64 | +0.622 (0.003) |
| `plddt_mean` | **per-protein PCC** | 24 | −0.490 (0.015) | **−0.682** (2.4e-4) | −0.54…−0.45 | −0.489 (0.025) |

Below Bonferroni but above the n=28 noise floor (suggestive only):

| feature | target | Pearson r (p) | Spearman ρ (p) |
|---|---|---|---|
| `frac_buried_rel_lt_0.25` | b_p_wt_error | −0.561 (0.0019) | −0.443 (0.018) |
| `SASA_per_residue` | a_p | +0.536 (0.0033) | +0.466 (0.012) |
| `plddt_mean` | a_p | −0.534 (0.0072) | −0.291 (0.17) |
| `frac_charged` | b_p_wt_error | +0.419 (0.026) | +0.466 (0.012) |
| `length` | \|b_p\| | +0.363 (0.057) | +0.428 (0.023) |

The five Bonferroni-surviving hits are stable: leave-one-out never flips a sign or crosses
the noise floor, and each holds within the 21 natural proteins alone, so none is an
artifact of the designed-fold subgroup. (Designed folds do have lower mean a_p, 0.389 vs
0.564, but that difference is itself not significant, p = 0.13.)

## 4. Verdict per property asked

- **BSA / interface.** True inter-chain BSA: **unmeasurable** — all 28 are single-chain
  monomers, interchain_BSA ≡ 0. But its within-protein analogue, **relative solvent
  exposure, is the single strongest signal found anywhere in this scan**, and it acts on
  **a_p, not b_p**: `mean_rel_SASA` vs a_p, r = +0.714, p = 2e-5, surviving Bonferroni
  over 90 tests. More-exposed, less-compact proteins have **less slope collapse**; tightly
  buried proteins are where the model under-reacts worst. Note this is exactly the coil
  lesson repeating — a surface/geometry quantity that looked like nothing against the decoy
  loss is a real effect once scored against the right calibration object.
- **Complexity (oligomer / chain count).** **No signal measurable** — zero variance on this
  test set (all monomers). Not "no effect": untestable here. Would need a test set with
  multimers.
- **Ligands / cofactors / metals.** **No signal measurable** — zero variance (no HETATM in
  any of the 28). Untestable here.
- **Hydrophobicity.** Bulk composition (`mean_hydropathy_KD`, `frac_hydrophobic`) shows
  **nothing** (|r| ≤ 0.28, all p > 0.14 — below the n=28 noise floor). But *exposed*
  hydrophobicity does: `mean_rel_SASA_hydrophobic` vs a_p, r = +0.628, p = 3.4e-4,
  Bonferroni-surviving. It is the **placement** of hydrophobics (exposed vs buried), not
  their amount, that tracks calibration.
- **b_p specifically.** Nothing survives Bonferroni against b_p or \|b_p\|. The best
  candidate is `frac_buried_rel_lt_0.25` vs `b_p_wt_error` (r = −0.561, p = 0.0019): more
  buried protein → more negative WT dG error. Suggestive, worth a follow-up, **not
  established** at this n.

## 5. The honest n = 28 caveat

n = 28 (n = 24 wherever pLDDT is involved). The p = 0.05 two-sided threshold is
**|r| = 0.374**; anything below that is statistically indistinguishable from zero and is
reported here only as "not measurable", never as evidence of absence. Because 90 tests were
run, the single-test p = 0.05 bar is far too permissive — the defensible bar is
Bonferroni α = 5.6e-4, i.e. roughly **|r| ≳ 0.60**. Only the five associations in the first
table of §3 clear it. Everything else in this document, including every b_p result, is
suggestive at best. With 28 points a correlation of 0.5 has a 95% CI of roughly
[0.16, 0.74] — wide enough that effect *sizes* here should not be quoted as precise.

Two structural caveats beyond sample size: (1) the catalogue contributed **no data** to the
correlations — it could not, the join is empty — so these are self-contained measurements on
our own test proteins, and the catalogue's own BSA/ligand/oligomer columns remain
**untested against our metric** and untestable until a test set overlapping PDB exists;
(2) the features come from AlphaFold models, not experimental structures, so exposure is
model-derived — which is also why the `plddt_mean` vs PCC result (ρ = −0.682) should be
read carefully: it says we do *better* on proteins AlphaFold is *less* confident about,
which is more plausibly a proxy for flexible/exposed folds (consistent with the a_p story)
than a claim about model quality.
