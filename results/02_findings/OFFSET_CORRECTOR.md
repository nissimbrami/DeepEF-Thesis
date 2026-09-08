# Predicting the per-protein offset b_p: the corrector is a NEGATIVE result

**Bottom line.** The 0.71 offset-removed ceiling is an ORACLE and it stays an oracle. Structure-only features do NOT predict b_p on held-out proteins: the leave-one-protein-out held-out R^2 of b_p itself is **negative on 8 of 10 evaluation CSVs** (mean R^2 = -0.123), meaning the feature model predicts b_p *worse than the training mean does*. The pooled-PCC gain it produces is +0.0055 on average -- about 5% of the oracle gain of +0.1125. A matched permutation null (see below) puts the observed gain at p >= 0.05 on 10 of 10 CSVs. We report this as a negative result rather than dressing it up.

The single sharpest statement of the result: of the 22 structural features, **0 achieve positive held-out R^2** when used alone to predict b_p. Not one structural covariate predicts the offset out of sample, and no combination of them does either.

## The three numbers, side by side

Mean over all 10 evaluation CSVs (28 proteins, 28,314 mutation pairs each):

| quantity | pooled ddG PCC | gain vs raw | is it a method? |
|---|---|---|---|
| raw (no correction) | 0.5994 | -- | yes (baseline) |
| mean-offset baseline (no features) | 0.5940 | -0.0054 | yes, trivially |
| **LOPO-predicted offset removed** | **0.6049** | **+0.0055** | **yes -- this is the method** |
| oracle offset removed | 0.7119 | +0.1125 | **NO -- fitted on test labels** |

The honest reading: the oracle buys +0.1125 pooled PCC. A real, held-out corrector buys +0.0055. **95% of the oracle gain does not survive contact with held-out prediction.**

### An important technical point about the "mean baseline"

Pearson correlation is invariant to adding a constant. Subtracting the *same* number from every protein therefore changes pooled PCC by **exactly zero** (verified: subtracting -1.0, 0.0, mean(b_p), or +1.0 all give pooled = 0.590991 on the primary CSV). The mean-baseline row above differs from raw only because leave-one-out makes the subtracted mean very slightly protein-dependent, and it comes out marginally *worse* than raw on 10/10 CSVs. **So the null that a per-protein corrector must beat is the RAW number, not the mean baseline.** Reporting a lift over the mean baseline would overstate the result; we do not do that.

## Per-CSV detail

| eval CSV | raw | mean-base | LOPO ridge | oracle | d(ridge-raw) | held-out R^2 of b_p |
|---|---|---|---|---|---|---|
| abl_anchor_w0.3_s42_e14.csv | 0.6157 | 0.6100 | 0.6287 | 0.7294 | +0.0130 | -0.012 |
| abl_anchor_w1.0_s42_e14.csv | 0.6018 | 0.5965 | 0.6088 | 0.7066 | +0.0070 | +0.103 |
| abl_anchor_w3.0_s42_e13.csv | 0.5972 | 0.5931 | 0.6315 | 0.6834 | +0.0343 | +0.351 |
| abl_calib_ctrl_repro2_e14.csv | 0.5910 | 0.5853 | 0.5820 | 0.7113 | -0.0090 | -0.436 |
| abl_p3_slope3.0_s42_e8.csv | 0.5624 | 0.5560 | 0.5722 | 0.6974 | +0.0098 | -0.159 |
| abl_sigma_seed1_e13.csv | 0.5948 | 0.5899 | 0.5817 | 0.7005 | -0.0132 | -0.297 |
| abl_sigma_seed2_e10.csv | 0.6575 | 0.6528 | 0.6529 | 0.7503 | -0.0047 | -0.111 |
| abl_sigma_seed3_e13.csv | 0.5799 | 0.5734 | 0.5864 | 0.7178 | +0.0065 | -0.227 |
| abl_sigma_seed42_e9.csv | 0.5929 | 0.5879 | 0.5967 | 0.7016 | +0.0038 | -0.269 |
| abl_sigma_seed4_e14.csv | 0.6008 | 0.5951 | 0.6080 | 0.7204 | +0.0072 | -0.177 |

Ridge beats raw on 7/10 CSVs; held-out R^2 > 0 on only **2/10**. On the primary CSV (`abl_calib_ctrl_repro2_e14.csv`) the corrector *loses*: 0.5820 vs 0.5910 raw, with R^2 = -0.436 and corr(b_p, predicted b_p) = 0.0369 -- i.e. essentially no relationship at all.

## Why it fails: the predictions miss exactly the proteins that matter

The oracle gain is driven by the two extreme-offset proteins. The LOPO model predicts both at essentially zero, and gets the sign wrong on several mid-range proteins:

| protein | true b_p | ridge prediction | mean prediction |
|---|---|---|---|
| 2K5H | -0.7199 | +0.0059 | +0.0022 |
| r18_3_TrROS_Hall | -0.1887 | -0.3524 | -0.0175 |
| r12_757_TrROS_Hall | -0.1856 | -0.2324 | -0.0176 |
| 2BTH | -0.0476 | -0.0910 | -0.0227 |
| HHH_rd1_0142 | -0.0473 | -0.0569 | -0.0227 |
| HEEH_KT_rd6_0793 | +0.1736 | -0.0162 | -0.0309 |
| 3DKM | +0.2426 | -0.0421 | -0.0335 |
| 2KVS | +0.7680 | -0.0531 | -0.0529 |

`2K5H` (b_p = -0.7199) and `2KVS` (b_p = +0.7680) are the two proteins whose offsets the oracle most needs to remove. The ridge predicts +0.0059 and -0.0531 for them -- both effectively zero, and 2KVS with the wrong sign.

This is decisive, and it is measurable. On the primary CSV, applying the oracle correction to **only the two largest-|b_p| proteins** (`2KVS`, `2K5H`) and leaving the other 26 uncorrected already recovers **80% of the entire oracle gain** (0.5910 -> 0.6873 of the full 0.5910 -> 0.7113). Correcting all 26 *others* and leaving those two alone recovers only 20% (-> 0.6152).

This replicates across all 10 CSVs: the top-2 |b_p| proteins carry a mean of **78% of the oracle gain** (range 60-88%), with `2K5H` in the top pair every time.

So the "0.71 ceiling" is not a broad, systematic per-protein miscalibration that a feature model could learn -- it is dominated by two outlier proteins out of 28. A smooth structural regression cannot capture them, which is exactly what the negative held-out R^2 reports. **The oracle gain is the outliers.**

### The outliers are not structural outliers -- and one is not an offset at all

Checked directly: neither dominant protein is an outlier in feature space. The largest absolute z-score across all 22 features is only +2.03 for `2K5H` (`contact_density`) and +1.88 for `2KVS` (`frac_buried_rel_lt_0.25`). Both sit inside the structural distribution, so there is no structural signature for a regression to latch onto. That is the mechanism behind the negative R^2, not a modelling mistake.

More importantly, `2KVS` is not really a "calibration offset" case at all: its slope is a_p = 0.094 against a median of 0.4975, and its per-protein PCC is 0.160 against a median of 0.80. The model essentially fails on that protein outright; the large fitted b_p is absorbing that failure. Subtracting an offset is the wrong repair for it, and no offset predictor -- however good -- would be the right fix. (`2K5H` is the opposite case: a_p = 0.499 and PCC = 0.861, a genuinely well-ranked protein carrying a real offset.)

|b_p| is also mildly related to the number of mutations measured per protein (r = 0.372, p = 0.051, n = 28), so part of the spread is estimation noise in b_p itself rather than a physical property waiting to be predicted.

### The small positive gain is an artefact of the alpha search, not skill

The main table selects the ridge penalty by an inner leave-one-out on each training fold. Repeating the whole experiment with the penalty **fixed** at alpha=30 (the value the permutation null uses, so the two are exactly comparable) removes that extra degree of freedom, and the corrector gets *worse than useless*:

| | mean d(ridge-raw) | mean held-out R^2 | R^2 > 0 |
|---|---|---|---|
| inner-CV alpha (main table) | +0.0055 | -0.123 | 2/10 |
| fixed alpha = 30 | **-0.0016** | **-0.092** | **0/10** |

With the penalty fixed, the mean pooled-PCC change is **negative**: subtracting the predicted offsets makes the pooled correlation slightly worse on average. The small positive number in the main table is variance introduced by the alpha search, not evidence of a learned structure->offset relationship.

## Permutation null: the small gain is not significant

Shuffling the b_p labels against the feature rows destroys any true feature->b_p mapping while preserving the b_p distribution and the entire LOPO machinery. Real model and null both use a fixed alpha=30 so the comparison is exactly matched. 500 permutations per CSV. (The LOPO ridge here is a closed-form solve verified identical to the sklearn path to 2.5e-16.)

| eval CSV | d(ridge-raw) | null mean d | null 95th pct | p |
|---|---|---|---|---|
| abl_anchor_w0.3_s42_e14.csv | -0.0052 | -0.0043 | +0.0131 | 0.555 |
| abl_anchor_w1.0_s42_e14.csv | -0.0114 | -0.0042 | +0.0121 | 0.758 |
| abl_anchor_w3.0_s42_e13.csv | -0.0052 | -0.0029 | +0.0110 | 0.593 |
| abl_calib_ctrl_repro2_e14.csv | -0.0002 | -0.0036 | +0.0162 | 0.393 |
| abl_p3_slope3.0_s42_e8.csv | +0.0048 | -0.0041 | +0.0184 | 0.295 |
| abl_sigma_seed1_e13.csv | -0.0008 | -0.0031 | +0.0148 | 0.427 |
| abl_sigma_seed2_e10.csv | -0.0067 | -0.0039 | +0.0105 | 0.621 |
| abl_sigma_seed3_e13.csv | +0.0038 | -0.0043 | +0.0185 | 0.297 |
| abl_sigma_seed42_e9.csv | +0.0012 | -0.0030 | +0.0139 | 0.367 |
| abl_sigma_seed4_e14.csv | +0.0038 | -0.0036 | +0.0164 | 0.307 |

CSVs with p < 0.05: **0/10**. The observed gains sit inside the permutation null: the corrector is indistinguishable from shuffled labels.

## Sensitivity analyses

### pLDDT (24 of 28 proteins)

`plddt.csv` covers only 24 of the 28 test proteins. The 4 missing ones (`HHH_rd1_0244`, `HHH_rd1_0142`, `HEEH_KT_rd6_0746`, `HEEH_KT_rd6_0793`) are all de-novo designed, and their AlphaFold PDB B-factor columns are all exactly 0.00, so there is no pLDDT to recover -- imputing would fabricate data. pLDDT is therefore excluded from the main 28-protein model and tested here on the 24-protein subset.

| eval CSV | raw | LOPO without pLDDT | LOPO with pLDDT | oracle |
|---|---|---|---|---|
| abl_anchor_w0.3_s42_e14.csv | 0.6434 | 0.6381 (R2 -0.116) | 0.6370 (R2 -0.143) | 0.7530 |
| abl_anchor_w1.0_s42_e14.csv | 0.6291 | 0.6158 (R2 -0.218) | 0.6132 (R2 -0.261) | 0.7299 |
| abl_anchor_w3.0_s42_e13.csv | 0.6187 | 0.6112 (R2 -0.167) | 0.6097 (R2 -0.216) | 0.7010 |
| abl_calib_ctrl_repro2_e14.csv | 0.6143 | 0.6160 (R2 -0.053) | 0.6170 (R2 -0.059) | 0.7331 |
| abl_p3_slope3.0_s42_e8.csv | 0.5803 | 0.5868 (R2 -0.018) | 0.5877 (R2 -0.025) | 0.7154 |
| abl_sigma_seed1_e13.csv | 0.6110 | 0.6144 (R2 -0.058) | 0.6151 (R2 -0.063) | 0.7183 |
| abl_sigma_seed2_e10.csv | 0.6838 | 0.6760 (R2 -0.117) | 0.6741 (R2 -0.159) | 0.7696 |
| abl_sigma_seed3_e13.csv | 0.5955 | 0.6025 (R2 -0.021) | 0.6035 (R2 -0.028) | 0.7310 |
| abl_sigma_seed42_e9.csv | 0.6167 | 0.6193 (R2 -0.054) | 0.6190 (R2 -0.079) | 0.7214 |
| abl_sigma_seed4_e14.csv | 0.6204 | 0.6268 (R2 -0.005) | 0.6269 (R2 -0.016) | 0.7399 |

Mean change in held-out R^2 from adding pLDDT: **-0.022**. Adding pLDDT does not rescue the corrector.

### Smaller feature sets (n=28 demands ruthless parsimony)

| feature set | features | mean raw | mean LOPO | mean held-out R^2 | beats raw |
|---|---|---|---|---|---|
| sasa_len | `mean_rel_SASA`, `length` | 0.5994 | 0.5908 | -0.123 | 0/10 |
| sasa_only | `mean_rel_SASA` | 0.5994 | 0.5924 | -0.102 | 0/10 |
| compact4 | `mean_rel_SASA`, `length`, `rel_contact_order`, `frac_hydrophobic` | 0.5994 | 0.5951 | -0.108 | 0/10 |
| ss3 | `frac_helix`, `frac_strand`, `frac_coil` | 0.5994 | 0.5980 | -0.055 | 3/10 |

### Univariate LOPO on the primary CSV

Best single features by held-out R^2 (all should be read against the fact that R^2 <= 0 means "worse than predicting the training mean"):

| feature | held-out R^2 | pooled PCC after correction |
|---|---|---|
| `frac_helix` | -0.0184 | 0.5902 |
| `frac_strand` | -0.0247 | 0.5893 |
| `rel_contact_order` | -0.0474 | 0.5880 |
| `frac_charged` | -0.0549 | 0.5871 |
| `frac_coil` | -0.0639 | 0.5873 |
| `abs_contact_order` | -0.0713 | 0.5850 |
| `mean_rel_SASA_hydrophobic` | -0.0766 | 0.5834 |
| `contact_density` | -0.0786 | 0.5826 |

0 of 22 single features achieve positive held-out R^2.

## Method

- **Target.** `b_p` = the per-protein ddG-space intercept, `np.polyfit(ddg_true, ddg_pred, 1)[1]`, with row 0 of each protein group as WT -- identical to `calib_diag.per_protein_fit`, reused via `scripts/catalogue_vs_bp.recover_bp`. Note this is the **ddG-space** intercept (std 0.2273), not `b_p_wt_error`, the dG-space WT error (std 1.6030, matching the recorded std(b_p)=1.5741). The ddG intercept is the quantity the oracle actually subtracts, so it is the quantity a corrector must predict.
- **Features (22).** Computed from the AlphaFold models of the 28 test proteins via `scripts/catalogue_vs_bp.structural_features` (imported, not duplicated) plus radius of gyration, contact order and CA-geometry secondary-structure fractions: `length`, `mean_hydropathy_KD`, `frac_hydrophobic`, `frac_charged`, `frac_glycine`, `frac_proline`, `total_SASA`, `SASA_per_residue`, `mean_rel_SASA`, `frac_buried_rel_lt_0.25`, `frac_exposed_rel_gt_0.5`, `mean_rel_SASA_hydrophobic`, `SASA_over_len_pow_073`, `radius_gyration`, `Rg_over_len_pow_038`, `n_contacts_CA8`, `contact_density`, `rel_contact_order`, `abs_contact_order`, `frac_helix`, `frac_strand`, `frac_coil`.
- **Dropped (6).** `n_chains`, `n_het_residues`, `n_metal_residues`, `interchain_BSA`, `plddt_mean`, `plddt_min` -- zero variance or unavailable. As recorded, all 28 test proteins are single-chain, ligand-free, metal-free monomers, so chain count, HET count, metal count and interchain BSA have **zero variance here**: not measurable, not "no effect".
- **Protocol.** Leave-one-protein-out ridge. For each held-out protein the standardiser and the ridge are fit on the other 27 only, and the regularisation strength is chosen by an **inner** leave-one-out over those 27. The held-out protein's own b_p never enters any fit at any level. Its predicted offset is then subtracted from its ddG predictions and pooled PCC is recomputed over all pairs.
- **Guard against the signature failure mode** (code runs, reports a number, feature never read): verified that shuffling the feature matrix changes the predictions (max|delta| = 0.456) and that zeroing the features changes them (max|delta| = 0.249), with the zero-feature model collapsing **exactly** onto the LOPO training mean (max|delta| = 0.00000000). The features are demonstrably read.
- **G4 gate.** `python scripts/gate_g4_cpu.py` -> `baseline ... dG=-0.0030 width=1092`, ALL PASS. This work is analysis-only and touches no model code.

## Honest caveats

1. **n=28 with 22 features is a severe overfitting regime** (~1.3 proteins per feature). Strong ridge regularisation and a nested inner loop were used, and the model still fails to generalise. With n=28 the standard error on any correlation is large: |r| < 0.374 is indistinguishable from zero at p=0.05.
2. **A negative held-out R^2 is the honest headline.** Mean R^2 = -0.123 across CSVs means the feature model is, on average, worse than a constant. The occasional positive pooled-PCC delta is not evidence of skill; it is what a noisy near-zero predictor does to a shift-invariant metric.
3. **This does not prove b_p is unpredictable in principle** -- only that these 22 structure-only covariates, on these 28 small single-domain monomers, do not predict it. A larger and more diverse protein set, or features derived from the model's own internal state rather than from structure, remain open.
4. **What this costs the thesis.** The offset-removal number (0.70-0.72) must be labelled in the text as an **oracle upper bound / diagnostic of headroom**, never as achieved performance. The achievable pooled PCC with a real corrector is 0.6049 with the alpha search and 0.5978 with the penalty fixed -- either way indistinguishable from the raw 0.5994. The defensible sentence is: "removing a per-protein offset fitted on the test labels raises pooled PCC to 0.71, but that offset cannot be predicted from structure on held-out proteins, so 0.71 is an upper bound rather than an achieved result."

## Reproduce

```bash
python scripts/offset_corrector.py --repo /home/nissimb/DeepPEF   # main table
python scripts/perm_null.py       /home/nissimb/DeepPEF 200       # permutation null
python scripts/sensitivity.py     /home/nissimb/DeepPEF           # pLDDT / small sets
python scripts/outlier_decomp.py  /home/nissimb/DeepPEF           # oracle-gain decomposition
python scripts/make_report.py     /home/nissimb/DeepPEF           # regenerate this file
```

Artifacts: `results/offset_corrector.json`, `results/offset_corrector_perm.json`, `results/offset_corrector_sensitivity.json`, `results/offset_corrector_outliers.json`, `results/offset_corrector_fixedalpha.json`.
