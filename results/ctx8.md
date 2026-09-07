
---

# CHECKPOINT 8 — 2026-09-07 — THE SIGNAL IS ON a_p, NOT b_p (corrected significance)

## The finding

Structural exposure predicts the COMPRESSION SLOPE a_p. Not the offset b_p.

`scripts/catalogue_vs_bp.py` -> `results/catalogue_vs_bp.json` + `results/CATALOGUE_VS_BP.md`.
28 test proteins, b_p/a_p from `abl_calib_ctrl_repro2_e14.csv` using calib_diag.py conventions
exactly (groupby protein, row 0 = WT, np.polyfit(ddg_true, ddg_pred, 1)). Structural covariates
computed from the actual AlphaFold models in
`data/Processed_K50_dG_datasets/AlphaFold_model_PDBs/` (all 28 present) with Biopython
Shrake-Rupley SASA.

**The one result that clears Bonferroni on BOTH Pearson and Spearman (90 tests, alpha 5.6e-4):**

| feature | target | pearson r | p | spearman rho | p |
|---|---|---|---|---|---|
| **mean_rel_SASA** | **a_p** | **+0.714** | 2.0e-05 | **+0.623** | 4.0e-04 |

Stable under leave-one-out (r in [0.642, 0.790], no sign flip) and holds within the 21 natural
proteins alone (r = +0.767), so it is not a designed-fold artifact.

**Interpretation: more-exposed, less-compact proteins suffer LESS slope collapse. Tightly
buried proteins are where the model under-reacts worst.**

## SIGNIFICANCE CORRECTION — read this before quoting any number

The worker's report claimed "FIVE clear it". An adversarial verifier caught that, and I
re-derived the table myself from the JSON. **Only ONE of the five clears alpha=5.6e-4 on both
tests.** The other four clear on one test and not the other:

| feature | target | pearson p | spearman p | verdict |
|---|---|---|---|---|
| mean_rel_SASA | a_p | 2.0e-05 | 4.0e-04 | **SURVIVOR** |
| frac_exposed_rel_gt_0.5 | a_p | 1.2e-04 | 4.1e-03 | one test only |
| mean_rel_SASA_hydrophobic | a_p | 3.4e-04 | 2.7e-03 | one test only |
| SASA_over_len_pow_073 | a_p | 8.4e-04 | 1.1e-04 | one test only |
| plddt_mean | per-protein PCC | **1.5e-02** | 2.4e-04 | **27x ABOVE the stated bar** |

The four near-misses are all the SAME underlying quantity (surface exposure) measured four ways,
so they are corroborating, not independent — which is a reason to believe the direction and a
reason NOT to count them as five findings. Report ONE result, with the other four as consistent
supporting measurements.

## What is NOT measurable here, and why that is not a null result

**All 28 test proteins are single-chain, ligand-free, metal-free monomers of 43-72 aa.**
Verified: `n_chains == 1`, `n_het_residues == 0`, `n_metal_residues == 0`, `interchain_BSA == 0.0`
for EVERY one. So complexity, oligomeric state, ligands and true interface area have **zero
variance** on this test set and are **untestable in principle here** — that is "not measurable",
NOT "no effect". Any ligand/complex claim needs a test set containing multimers and ligands,
which MegaScale is not.

Bulk hydrophobic COMPOSITION shows nothing (|r| <= 0.28, p > 0.14). Exposed hydrophobicity does.
**It is the placement of hydrophobics, not their amount.**

## The 100k catalogue cannot be joined — and that is correct, not a failure

**0 of 28 test proteins appear in the catalogue.** Verified two ways (exact 4-char PDB id, and
a case-insensitive substring scan over all 100,246 rows). This is by construction: the catalogue
IS the PDB pre-training decoy set and the 28 are held-out MegaScale test proteins. Disjoint by
design. No join was fabricated and no catalogue number is quoted for our 28.

The catalogue also has degenerate columns that would have produced fake nulls if used naively:
`Ligands_x` has 4 non-null rows of 100,246; `BSA_Numeric_x` has 4 unique values; `BSA_Percentage`
is 100,241 null; `Global Symmetry` is constant. **The live columns are `Ligands_y` and
`BSA_Numeric_y`.** The old "-0.001 BSA correlation" was therefore doubly void: an all-but-constant
feature scored against the pre-training decoy loss.

## Why this matters more than it looks

The offset-removal ceiling (0.70-0.72) assumes a_p is left alone. **a_p is the second calibration
object and nothing has ever attacked it** (`--slope_weight` has still never been run). And unlike
b_p, **a_p is a within-protein slope, so it does NOT cancel in ddG** — it is the one calibration
object legitimately measurable on the ddG metric.

An exposure-predicted per-protein slope rescale is therefore a genuinely new lever, and it is
CPU-testable on existing eval CSVs before any GPU is spent.

## Next, in order

1. Replicate the survivor across the other 9 eval CSVs (the 5 `abl_sigma` seeds especially) to
   confirm it is checkpoint-stable and not specific to `calib_ctrl_repro2_e14`. One loop, no GPU.
2. Only then consider a slope-correction arm.

## REPLICATION — the finding is checkpoint-stable (10/10)

Re-ran `catalogue_vs_bp.py --eval_csv` over every eval CSV on disk. `mean_rel_SASA` vs `a_p`:

| checkpoint | pearson r | p | spearman | p |
|---|---|---|---|---|
| calib_ctrl_repro2_e14 | 0.714 | 2.0e-05 | 0.623 | 4.0e-04 |
| abl_sigma_seed4_e14 | 0.703 | 3.0e-05 | 0.658 | 1.4e-04 |
| abl_sigma_seed3_e13 | 0.699 | 3.5e-05 | 0.656 | 1.5e-04 |
| abl_sigma_seed1_e13 | 0.683 | 6.2e-05 | 0.651 | 1.7e-04 |
| abl_p3_slope3.0_s42_e8 | 0.676 | 7.9e-05 | 0.618 | 4.6e-04 |
| abl_sigma_seed42_e9 | 0.674 | 8.5e-05 | 0.617 | 4.7e-04 |
| abl_sigma_seed2_e10 | 0.632 | 3.1e-04 | 0.620 | 4.4e-04 |
| abl_anchor_w0.3_s42_e14 | 0.587 | 1.0e-03 | 0.511 | 5.5e-03 |
| abl_anchor_w1.0_s42_e14 | 0.471 | 1.1e-02 | 0.477 | 1.0e-02 |
| abl_anchor_w3.0_s42_e13 | 0.400 | 3.5e-02 | 0.436 | 2.1e-02 |

**mean r = 0.624, sd = 0.102, min 0.400, max 0.714, sign-consistent 10/10.**
Two independent seed families (5 sigma seeds + the control) all land at 0.63-0.71. This is not
an artifact of one checkpoint.

### A SECOND finding fell out of the replication, unlooked for

The correlation decays MONOTONICALLY with the WT-anchor weight:

    anchor 0.3 -> r = 0.587
    anchor 1.0 -> r = 0.471
    anchor 3.0 -> r = 0.400

while all five unanchored sigma seeds sit at 0.63-0.68. **The anchor lever is already partially
suppressing the exposure/slope coupling** — the more you pin the WT, the less the model's slope
tracks exposure. That is a mechanistic link between factor A (the anchor, already in the running
factorial) and a_p, and it was not predicted. Three points is a trend, not a law; the running
48-cell factorial will test it properly and its anchor cells now have a specific prediction to
check rather than an open-ended sweep.
