
---

# CHECKPOINT 19 — 2026-09-08 — THE ATTENUATION HYPOTHESIS IS DEAD, AND THE CATALOGUE IS RETIRED

## 1. Per-mutation uncertainty FOUND, and it settles a_p

An earlier attempt concluded the MegaScale uncertainties could not be joined to our 28 (0/28 by
name and by exact WT sequence). **That attempt searched the wrong file.** The uncertainties are in
`data/ThermoMPNN/mega_test.csv` — 28,312 rows against our 28,314 test mutations, with 17 CI columns.

**Verified by the main session, not taken on report:**

    proteins in mega_test: 28
    MATCHED 28 / 28 of our test proteins
    median 95% CI = 0.1007  ->  median sigma = 0.0327 kcal/mol  (n = 28,312)
    median var(true ddG)   = 0.7178

**The attenuation prediction:**

    a_p = var(true) / (var(true) + var(noise)) = 0.7178 / (0.7178 + 0.0327^2) = 0.9985

**We measure a_p ~ 0.50. Attenuation predicts 0.9985.**

The label noise would have to be roughly **200x larger** to explain the compression. For attenuation
alone to produce a_p = 0.5, sigma would need to be ~0.85 kcal/mol; it is 0.0327.

**a_p = 0.499 is a genuine model failure, not a statistical artefact of noisy labels.** This
independently confirms CHECKPOINT's earlier sign-based argument (attenuation predicted the WRONG
SIGN for the exposure correlation, P(r >= +0.714) = 0/2000) by a completely different route — and
this route is direct rather than inferential. **`--slope_weight` is fully justified.**

## 2. The 100k catalogue is formally RETIRED for every join-based purpose

**It has NO SEQUENCE COLUMN.** That is a permanent, in-principle result, not a failed attempt.
Joins measured: 0/226 training PDB ids, 0/21 test PDB-like ids, 0/340 by full protein_id, against
66,945 unique catalogue PDB ids. **So "test structural properties at n=hundreds using the training
set" is dead — it required the join.**

### The one live use, which needs no join: DISTRIBUTION SHIFT

| | catalogue | our regime (32-74 aa) |
|---|---|---|
| median length | **477** | 55-56 |
| **share of pre-training RESIDUES** | — | **0.1206%** (1 in 830) |
| NMR structures | 8.56% | **77.43%** (9.05x enriched) |
| Is Complex? = Yes | 67.7% | 11.8% |
| Monomer | 31.8% | **88.2%** |
| BSA = 0 | 31.2% | 87.8% |

**Our entire problem domain was one residue in 830 of pre-training, and a structurally different
kind of protein — small, monomeric, interface-free, NMR-solved.** That is worth stating in the
thesis as a characterisation of pre-training (well powered, n=99,708), **but NOT as the cause of
b_p**: measured against b_p it is r = +0.043 (p = 0.829), a clean null.

### Ofir's mean-gap mechanism does NOT carry over

He found his RMSE tracked the train/test mean gap. Ours: per-protein WT dG train 3.0930 +/- 1.3362
(n=340) vs test 2.9895 +/- 0.9263 (n=28), **gap = -0.1034, p = 0.588**. That is 6.6% of one
std(b_p), and a shared constant cannot generate per-protein DISPERSION anyway. **Retired.**

### And the composition/shift features do not predict b_p

25 tests over the 10 eval CSVs: length-percentile r = +0.0427, Mahalanobis distance from the
training composition centroid r = +0.0277, L2 distance r = +0.1904. Three nominal hits where ~1.25
are expected by chance, and **NONE survive Bonferroni or BH-FDR** for either b_p or a_p.
One real descriptive asymmetry: threonine fraction differs train vs test (0.0660 -> 0.0395,
p = 2.7e-07, survives Bonferroni) but carries **no predictive power over b_p** (r = -0.119).

**Consequence: b_p must be hunted on the INPUT/REPRESENTATION side, not the label side.**

## 3. Also found: 26,315 double mutants for our 28 sit in no split at all

A held-out generalisation test that needs no new labels and no new measurements. Untouched.

## Caveat on a convention mismatch, stated openly

The agent's own per-protein regression averaged across checkpoints gives std(b_p) = 0.7914 and
median a_p = 0.4148, versus our briefed 1.5741 and 0.4990. Averaging slopes and intercepts across
checkpoints shrinks the spread, so the two are not on the same scale. Only scale-free CORRELATIONS
were used for the null conclusions, so those stand; the 6.6%-of-one-SD comparison deliberately uses
the larger briefed value so as not to flatter the negative.
