
# CHECKPOINT 11 — 2026-09-07 — b_p RE-DERIVED INDEPENDENTLY: FINDING STANDS, HEADLINE NUMBER DOES NOT

**Read with CHECKPOINT 9.** CHECKPOINT 9's *conclusion* — that the "NOT b_p" claim in CHECKPOINT 8
was premature, and that one structural axis acts on both calibration channels — is **CONFIRMED and
stands**. This checkpoint does not undo it.

But CHECKPOINT 9's headline **number** (`-0.475`) and its **promotion criterion** ("sign-consistent
10/10") both fail independent re-derivation. Two defects, both material, both found by recomputing
from source rather than re-reading the cached JSON.

`scripts/indep_bp.py` -> `results/bp_replication_independent.json`,
`results/bp_anchor_gradient_independent.json`. Independent of `scripts/bp_replication.py`: targets
built from the eval CSVs' **own** `ddG`/`pred_ddG` columns rather than recomputed from `deltaG`;
slope/intercept via `scipy.stats.linregress` rather than `np.polyfit`; SASA recomputed from the
AlphaFold PDBs. Reads no cached result JSON.

## What replicated exactly

The direction, the sign-consistency, and the *existence* of the effect all reproduce:

| claim | reported | re-derived | verdict |
|---|---|---|---|
| burial vs `b_p_wt_error`, sign-consistency | 10/10 negative | **10/10 negative** | CONFIRMED |
| `SASA_per_residue` vs `abs_b_p` | 10/10 signs, 4/10 sig (suggestive) | **10/10 signs, 3/10 sig** | CONFIRMED as suggestive |
| `length` vs `abs_b_p` "1/10, correctly dead" | — | **10/10 signs, 1/10 sig** | **the brief is wrong: 1/10 is the SIGNIFICANCE count, not the sign count.** It is SUGGESTIVE, not dead. The genuinely dead length row is `length` vs `b_p` (6/10). CHECKPOINT 9 already caught this; re-confirmed. |
| `mean_rel_SASA` vs `a_p` | +0.624 | **+0.6188**, 10/10, 10/10 sig | CONFIRMED |

## DEFECT 1 — `-0.475` is a grid maximum, not an estimate

`frac_buried_rel_lt_0.25` is a **threshold count**: it counts residues below rel-SASA 0.25. On
43-72 aa proteins one borderline residue moves the feature by ~1/50 = 0.02. So the feature — and
the correlation — depends on two arbitrary choices nobody pre-registered: the SASA **quadrature**
and the **0.25 cutoff**. Both were left at their defaults, and the default happens to sit at the
joint maximum:

| SASA quadrature (n_points) | mean r | sig |            | cutoff (n_points=100) | mean r | sig |
|---|---|---|---|---|---|---|
| **100 (Biopython default)** | **-0.475** | 8/10 |  | 0.15 | -0.402 | 6/10 |
| 256 | -0.373 | 6/10 |                              | 0.20 | -0.396 | 6/10 |
| 540 | -0.333 | 5/10 |                              | **0.25 (default)** | **-0.475** | 8/10 |
| 960 (accurate) | -0.327 | 6/10 |                    | 0.30 | -0.269 | 2/10 |
|  |  |  |                                            | 0.35 | -0.205 | 0/10 |

The magnitude **decays monotonically as the SASA integration gets more accurate**, and decays in
both directions from the 0.25 cutoff. `-0.475` is the largest value on a 2-D grid of nuisance
parameters; the accurate-quadrature value is **-0.33**.

The **threshold-free** version of the same axis (`mean_rel_SASA`, no counting) is by contrast
completely quadrature-stable — +0.233 / +0.239 / +0.232 / +0.238 across all four settings — but
never individually significant (0/10). That is the honest size of this effect: **|r| ~ 0.23-0.33,
not 0.48.**

**Do not quote -0.475.** Quote the direction, and a magnitude of roughly 0.3 with the note that
threshold-count features are quadrature-sensitive at this protein size.

## DEFECT 2 — "10/10 sign-consistent" is a ~60%-false-positive criterion here

The whole promotion argument rests on sign-consistency across 10 checkpoints being strong evidence.
Measured: **it is not, because the 10 checkpoints are not 10 replicates.**

- mean pairwise correlation between the 10 checkpoints' `b_p_wt_error` vectors: **r = 0.884**
  (min 0.694, max 0.979)
- participation-ratio effective number of independent replicates: **1.24 of 10**
- protein-label permutation null, 2000 draws: **P(sign-consistency = 10/10 by chance) = 0.60**

Ten near-identical vectors agreeing on a sign is one observation reported ten times. At a 60% null
rate, ~36 of the 60 b_p-side pairs would reach 10/10 by chance alone — and **28 did**. The
SUGGESTIVE class is consistent with pure noise and should carry no weight.

**The criterion that does survive** is the permutation test on effect magnitude, which accounts for
the dependence by permuting protein labels. But applied honestly it does **not** rescue the b_p
finding at full strength, because the p-value moves with the same nuisance parameter as the
effect size:

| pair | quadrature | mean r | permutation p |
|---|---|---|---|
| `mean_rel_SASA` vs `a_p` | any | +0.619 | **0.0005** |
| burial vs `b_p_wt_error` | n_points=100 (default) | -0.475 | **0.006** |
| burial vs `b_p_wt_error` | n_points=960 (accurate) | -0.327 | **0.0695 — does not clear 0.05** |

**This is the single most important number in this checkpoint.** The b_p finding is significant at
the quadrature that maximises it and not significant at the accurate one. Its p-value is therefore
as nuisance-dependent as its effect size, and p=0.006 must not be quoted on its own.

## Revised classification

The three-way ESTABLISHED / SUGGESTIVE / DEAD scheme in CHECKPOINT 9 should be **retired**: its
middle class is noise and its top class was won on a nuisance-parameter maximum. Replace with the
permutation criterion:

| feature | target | mean r (n_points=100) | mean r (accurate) | perm p | verdict |
|---|---|---|---|---|---|
| mean_rel_SASA | a_p | +0.619 | +0.619 | 0.0005 (stable) | **REAL — strong** |
| frac_buried_rel_lt_0.25 | b_p_wt_error | -0.475 | -0.327 | 0.006 -> **0.0695** | **DIRECTIONALLY REAL, NOT ESTABLISHED** |
| everything else on the b_p side | — | — | — | not tested individually | **UNSUPPORTED** (sign-consistency cannot separate them from noise) |

The honest verdict on the b_p row is **neither CHECKPOINT 8's "absent" nor CHECKPOINT 9's
"ESTABLISHED"**. It is a real directional effect — 10/10 negative, negative in naturals and in
designed separately, and significant under the default analysis — whose magnitude and significance
both degrade when the arbitrary feature parameters are made more accurate. It is the right size to
be worth one confirmatory test on new proteins, and the wrong size to build a thesis claim on.

## The anchor-weight gradient: present on both channels, but weaker evidence than it looks

The gradient does reproduce — on the b_p channel the three anchor arms are uniformly weaker than
the five unanchored sigma seeds (burial vs `b_p_wt_error`: anchor -0.197/-0.120/-0.196 vs sigma
-0.379/-0.289/-0.433/-0.397/-0.413 at accurate quadrature; full separation).

Three limits, the first two already noted in CHECKPOINT 9 and the third new:
1. Not monotonic in anchor weight on the b_p channel (w3.0 > w1.0). What replicates is
   anchored-weaker-than-unanchored, not the within-anchor ordering.
2. Arm and seed-family are confounded (anchor arms are all s42, different epochs). 3 vs 5 is not a
   designed experiment; the running 48-cell factorial settles it.
3. **p=0.018 is the floor p-value for any 5-vs-3 Mann-Whitney.** Every perfectly-separated pair
   returns exactly 0.018, so it registers "perfect separation" and carries no magnitude
   information. ~20 pairs show it. It is one observation about anchoring, not twenty.

Because a_p and b_p are computed from the *same* 10 checkpoints, "corroboration on a second
channel" is weaker than two independent experiments agreeing — the channels share the checkpoint
draw entirely and the protein draw entirely.

## What stands

1. **CHECKPOINT 8's "NOT b_p" remains retracted — but on weaker grounds than CHECKPOINT 9 claimed.**
   The evidence for a burial/offset coupling is a consistent negative direction that survives the
   natural/designed split, not a significance result: it clears the permutation test at default
   quadrature (p=0.006) and fails it at accurate quadrature (p=0.070). "We cannot claim there is no
   b_p signal" is supported. "There is an ESTABLISHED b_p signal" is not.
2. **One structural axis, both channels — stands.** Exposure -> slope `a_p` (+0.62, p=0.0005);
   burial -> offset `b_p_wt_error` (~-0.33, p=0.006). Same physical variable, opposite ends.
3. **The asymmetry is real and much larger than CHECKPOINT 9 implied.** The a_p effect is strong,
   quadrature-stable and significant under every setting tried; the b_p effect is weak and both its
   magnitude and its p-value move with nuisance parameters. "Both channels carry signal" is
   defensible; "comparably" is not. CHECKPOINT 8's instinct that the two channels are not
   equivalent was **partly right** — it was wrong to call b_p absent, right that it is much weaker.
4. **THE METRIC RULE is untouched and was again load-bearing.** Burial is a whole-protein property,
   identical between WT and mutant, so it cancels in ddG and was correctly scored on
   `b_p_wt_error`. Any lever built on it must be scored on dG or b_p.
5. **Procedural lesson, sharpened.** CHECKPOINT 9's lesson (run the strong test on both siblings
   before promoting either) stands. Add: *a replication sweep across checkpoints of one training
   run is not independent replication, and sign-consistency across dependent replicates is not
   evidence.* Check the effective N before treating agreement as confirmation.

## n=28 caveats

All of CHECKPOINT 9's caveats stand and are not repeated in full. The three that bind hardest here:

- **n=28, and the 28-protein draw is the single shared point of failure** for both channels. At
  n=28 a nominal r=-0.33 has a 95% CI of roughly [-0.62, +0.04] — it includes zero. The
  permutation p=0.006 is evidence the effect is not chance; it is not evidence the magnitude is
  well determined.
- **The 10 checkpoints add ~0.24 of an independent replicate**, not 10. Replication here shows the
  effect is not a single-checkpoint artifact. It does nothing about protein sampling.
- **Range restriction.** All 28 are single-chain, ligand-free, metal-free monomers, 43-72 aa.
  Burial range is correspondingly narrow, which is exactly why a threshold count over it is
  unstable. Nothing generalises to multimers, ligand-bound, or larger proteins; BSA / ligand /
  complexity remain untestable in principle on this set (zero variance), not null.
- **Natural-only check — RUN, and it PASSES.** This was the gap CHECKPOINT 9 flagged as
  outstanding. Naturals alone (n=21): mean r = **-0.490**, 10/10 sign-consistent, 8/10 significant,
  range -0.582..-0.324 — slightly *stronger* than all 28 (-0.475). Designed only (n=7):
  **-0.453**, same direction. **A designed-fold artefact is excluded**; the coupling is present in
  both subpopulations. This is the one place the b_p finding came out cleaner than expected.

## Recommended next step

If the burial/offset coupling is to be used as a lever, build it on the **threshold-free**
`mean_rel_SASA` (quadrature-stable) rather than `frac_buried_rel_lt_0.25`, and pre-register the
cutoff and quadrature before scoring. Score on dG or b_p, never ddG.

`scripts/gate_g4_cpu.py`: **ALL PASS**, baseline row **dG=-0.0030 width=1092** as required;
no-new-dims levers hold at width 1092 and burial adds exactly 3 (width 1095).
(CHECKPOINT 9 quoted "dG=0.0023 width=1095" here — that is the *burial-lever* row, not the
baseline row. Both are green; the required baseline signature is the -0.0030/1092 one.)
No feature vector was changed by this work (analysis only, CPU only, no jobs touched).

## Files

- `scripts/indep_bp.py` — the independent re-derivation (does not read any cached result JSON)
- `results/bp_replication_independent.json` — all 90 pairs, per-checkpoint r and p
- `results/bp_anchor_gradient_independent.json` — anchor-vs-sigma gradient
- `results/BP_REPLICATION_INDEPENDENT.md` — full ranked 60-row b_p-side table

---
