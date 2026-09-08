# SLOPE_ORIGIN — is `a_p` a loss artefact or a model failure?

**Verdict: NOT a loss artefact. Attenuation is refuted by sign, not merely by magnitude.**

The regression-dilution hypothesis is arithmetically consistent with the *median* `a_p`
(it takes a plausible σ_noise ≈ 0.59 kcal/mol), but it makes a second, unavoidable
prediction about the *cross-protein pattern* of `a_p` — and that prediction comes out
**backwards** in our data. `a_p` is at least partly a real model failure, and
`--slope_weight` is therefore a legitimate lever, not a fight against label noise.

All numbers below were produced by running
`scripts/slope_origin2.py` and `scripts/slope_sim.py` on the cluster
(CPU only, no GPU, no training). Raw output: `results/slope_origin.json`,
`results/slope_origin_sim.json`.
Gate after all work: `python scripts/gate_g4_cpu.py` → `baseline ... dG=-0.0030 width=1092`, ALL PASS.

---

## 1. The attenuation relation, and the noise level it demands

For ordinary least squares of a response on a predictor measured with additive,
independent error, the fitted slope is attenuated toward zero by the **reliability ratio**:

```
             var(true)                        1
a_hat  =  ─────────────────────  =  ───────────────────
          var(true) + var(noise)     1 + var(noise)/var(true)
```

Solving for the noise-to-signal variance ratio λ = var(noise)/var(true):

```
λ = (1 - a) / a
```

Our measured median `a_p` = **0.4975** (28 test proteins, control eval;
mean 0.5203, min 0.0789, max 1.2566 — reproduces the known 0.4990 / 0.0789–1.2566).

```
λ = (1 - 0.4975) / 0.4975 = 1.0100
```

So **the label noise variance would have to equal the true signal variance** — exactly
the critic's reading. Converting to an absolute scale using the observed spread of ddG:

| quantity | value |
|---|---|
| median var(observed ddG) | 0.6900 (sd **0.8307** kcal/mol) |
| ⇒ implied var(noise) | 0.3467 (**σ_noise = 0.5888 kcal/mol**) |
| ⇒ implied var(true) | 0.3433 (sd_true = 0.5859 kcal/mol) |

Per protein, the σ_noise required to explain that protein's own `a_p` by attenuation alone
(26 of 28 proteins have 0 < a_p < 1): **median 0.7186, range 0.3053 – 1.1211 kcal/mol**.

**This is the part of the critic's case that survives.** σ ≈ 0.6–0.7 kcal/mol is *not* an
absurd number for a high-throughput proteolysis assay. Attenuation cannot be dismissed on
plausibility grounds. It has to be killed on its second prediction — which is what follows.

---

## 2. Are the Tsuboyama per-mutation uncertainties present in our data?

**Partially — and critically, NOT for the 28 test proteins.** Stated plainly, because this
constrains what could be tested.

I scanned **81,717 CSV files** under `data/` (23 distinct header schemas). Five files carry
genuine Tsuboyama uncertainty columns (`deltaG_95CI`, `deltaG_t_95CI_high/low`,
`fitting_error_t`, and the chymotrypsin equivalents):

- `data/Processed_K50_dG_datasets/Pnas_filtering/pnas_mutations.csv` (244,400 rows, 42 cols)
- `data/Processed_K50_dG_datasets/Pnas_filtering/pnas_mutatns.csv`
- `data/Processed_K50_dG_datasets/Pnas_filtering/mega_val.csv`
- `data/ThermoMPNN/mega_test.csv`, `data/ThermoMPNN/mega_val.csv`

The label files we actually train and evaluate on do **not** carry them. Both of our
label schemas —
`name,deltaG,aa_seq,mut_type,ddG_ML,Stabilizing_mut` (862 files) and
`name,deltaG,aa_seq_full,aa_seq,mut_type,WT_name,dG_ML,ddG_ML,...` (479 files) —
have **no error/σ/CI column**. The uncertainty was dropped upstream when `ddG_ML` was
materialised.

**Can the CIs be joined back onto our 28?** No. I tried three joins and verified the join
machinery works:

| join key | result |
|---|---|
| `WT_name` / `protein_name`, direct | 0 / 28 |
| PDB-style stem (160 PDB-like names in the PNAS table) | 0 / 28 |
| **exact WT `aa_seq`** (naming-independent) | **0 / 28** |
| 30-residue containment (tolerates trimming) | 1 / 28 (`1TUC`, ambiguous multi-hit) |
| **positive control**: `1ENH`, `1VII` by `aa_seq` | **found — join logic is sound** |

The positive control matters: the same code finds `1ENH` and `1VII` immediately, so the
0/28 is a real held-out split, not a formatting failure. **Our 28 test proteins are disjoint
from every file that carries an uncertainty column.** Per-mutation σ therefore cannot be
used directly, and I proceed with the sensitivity/discrimination analysis instead — as
instructed, without inventing numbers.

---

## 3. THE DISCRIMINATING TEST — attenuation predicts the wrong sign

This is the decisive result.

Attenuation makes `a_p` a function of **label noise only**. It knows nothing about
structure. So for pure attenuation to produce the measured `r(a_p, mean_rel_SASA) = +0.714`
(replicated 10/10), **exposure would have to predict lower label noise** — more-exposed
proteins would need cleaner labels.

### 3a. Does any available noise proxy track exposure? Yes — in the wrong direction.

| relation | r | p | Spearman ρ |
|---|---|---|---|
| `a_p` ~ `mean_rel_SASA` | **+0.7139** | 2.0e-05 | +0.6229 |
| `n_mutations` ~ `mean_rel_SASA` | **−0.6850** | 5.8e-05 | −0.6939 |
| `sd_true_ddG` ~ `mean_rel_SASA` | −0.3880 | 0.041 | −0.5161 |
| `var_true_ddG` ~ `mean_rel_SASA` | −0.3712 | 0.052 | −0.5161 |
| `length` ~ `mean_rel_SASA` | −0.6662 | 1.1e-04 | −0.6280 |

Exposed proteins are **smaller**, have **fewer** mutations measured, and have a **narrower**
ddG spread. Fewer mutations and narrower spread both mean a *noisier*, less
well-determined per-protein fit — i.e. exposure predicts **more** effective noise, not less.
Attenuation would drive `a_p` **down** for exposed proteins. We observe it going **up**.

### 3b. The required noise itself has the wrong sign

Invert the attenuation relation per protein and ask what σ_noise each protein would need:

```
required σ_noise  ~  mean_rel_SASA :   r = −0.5297,  p = 0.0054,  n = 26
```

Attenuation-only requires exposed proteins to have **systematically lower** label noise
(r = −0.53). There is no mechanism in the Tsuboyama assay by which a solvent-exposed
monomer's ddG is measured half a kcal/mol more precisely than a buried one's; if anything
the assay's dynamic-range limits bite hardest at the extremes of stability, which tracks
size, not exposure. This is an ad-hoc requirement invented purely to rescue the hypothesis.

### 3c. Forward simulation: the attenuation null cannot produce the correlation

I simulated the attenuation null directly on the **real** per-protein true-ddG vectors:
one **global, structure-independent** σ_noise = 0.5888 kcal/mol (the value item 1 says is
needed), labels drawn as `observed = true + N(0, σ)`, `a_p` refitted exactly as
`calc_bp.py` does, 2000 trials.

```
simulated r(a_p, mean_rel_SASA) under pure attenuation, 2000 trials
  mean    -0.3871
  sd       0.0177
  95% CI  [-0.4219, -0.3528]
  max     -0.3305
  P(r >= observed +0.7139) = 0 / 2000 = 0.00000
```

**Pure attenuation predicts r ≈ −0.39. We measure +0.71.** Not merely too small — the
**opposite sign**, and the observed value lies ~60 standard deviations outside the null.
The null never once reached the observed value.

The sign flip is mechanical and worth stating: under attenuation, `a_p` is dragged toward
zero *most* where var(true) is smallest. Exposed proteins have the smallest ddG spread
(§3a), so attenuation compresses *them* hardest — producing a negative `a_p`–SASA
correlation. Reality does the reverse.

**Conclusion: attenuation is not the whole story, and cannot be. `a_p` carries a real,
structure-dependent model failure.**

---

## 4. The trivial controls everyone skipped

Both obvious statistical confounds for a fitted slope, on the control eval (N = 28):

| confound | r | p | ρ | verdict |
|---|---|---|---|---|
| `a_p` ~ **n_mutations per protein** | **−0.2754** | 0.156 | −0.1489 | not significant |
| `a_p` ~ **spread of true ddG** (sd) | **−0.2516** | 0.197 | −0.3153 | not significant |
| `a_p` ~ var(true ddG) | −0.2457 | 0.208 | −0.3153 | not significant |
| `a_p` ~ range(true ddG) | −0.2583 | 0.184 | −0.3317 | not significant |
| `a_p` ~ length | −0.2841 | 0.143 | −0.1266 | not significant |
| `a_p` ~ per-protein PCC | +0.5714 | 0.0015 | +0.4061 | significant (known) |

Both named confounds are **negative, weak, and non-significant** — and both point the
*wrong way* to explain a positive SASA correlation. (`corr(a_p, PCC) = +0.5714` reproduces
the recorded value exactly, confirming the pipeline.)

### Partial correlations — the structural signal survives every control

```
r(a_p, mean_rel_SASA | n_mutations)    = +0.7500
r(a_p, mean_rel_SASA | sd_true_ddG)    = +0.6909
r(a_p, mean_rel_SASA | var_true_ddG)   = +0.6918
r(a_p, mean_rel_SASA | length)         = +0.7337
```

Controlling for mutation count **strengthens** the association (+0.714 → +0.750), because
the confound was masking it. Nothing here is a counting artefact.

### Replication across all 10 eval CSVs

| eval | median a_p | r(a_p,SASA) | r(a_p,n) | r(a_p,sd) |
|---|---|---|---|---|
| anchor_w0.3_s42_e14 | 0.580 | +0.587 | −0.113 | −0.321 |
| anchor_w1.0_s42_e14 | 0.589 | +0.471 | −0.044 | −0.296 |
| anchor_w3.0_s42_e13 | 0.153 | +0.400 | −0.121 | −0.216 |
| calib_ctrl_repro2_e14 | 0.498 | +0.714 | −0.275 | −0.252 |
| p3_slope3.0_s42_e8 | 0.325 | +0.676 | −0.236 | −0.339 |
| sigma_seed1_e13 | 0.407 | +0.683 | −0.222 | −0.323 |
| sigma_seed2_e10 | 0.430 | +0.632 | −0.212 | −0.282 |
| sigma_seed3_e13 | 0.339 | +0.699 | −0.286 | −0.305 |
| sigma_seed42_e9 | 0.437 | +0.674 | −0.210 | −0.327 |
| sigma_seed4_e14 | 0.327 | +0.703 | −0.239 | −0.279 |
| **mean** | | **+0.6239 (10/10 positive)** | **−0.196 (0/10 positive)** | **−0.294 (0/10 positive)** |

The structural correlation replicates 10/10 (mean +0.624, matching the recorded value).
Both confounds are sign-consistently **negative** 10/10. They are not the explanation.

**The single most damaging fact for the attenuation hypothesis is in this table:**
`median a_p` moves from **0.580 → 0.153** (a 3.8×) purely as a function of
`--anchor_weight` (0.3 → 3.0), and to 0.325 under `--slope_weight 3.0`. The training labels
were **identical** across all ten runs. Label noise is a fixed property of the dataset;
it cannot produce a 3.8-fold swing in the attenuation factor. **`a_p` responds to
optimisation-side knobs, so it is not determined by label noise.** Attenuation predicts
`a_p` should be pinned near var(true)/(var(true)+var(noise)) regardless of loss weighting.

---

## 5. What this changes about `--slope_weight`

**Before:** if `a_p ≈ 0.5` were regression dilution, `--slope_weight` would be cosmetic —
forcing the slope to 1 would inflate variance without adding information, trading a
well-calibrated shrunken estimator for a noisy unbiased one. Under a pure-attenuation
world the *correct* fix is an errors-in-variables / Deming loss, and no architectural lever
could help.

**After, what the evidence supports:**

1. **`--slope_weight` targets a real deficiency.** `a_p` is structure-dependent
   (partial r ≈ +0.69 to +0.75 after every control) and moves 3.8× with loss weighting.
   Both are impossible under noise-driven attenuation.

2. **But `--slope_weight` is not a free win, because part of the compression *is* shrinkage.**
   The attenuation arithmetic in §1 is not refuted in magnitude — only in its cross-protein
   *pattern*. A defensible reading: a structure-independent floor near σ_noise ≈ 0.59
   contributes some shrinkage to every protein, and the model adds a
   **structure-dependent** failure on top that varies with exposure. `--slope_weight` should
   be expected to recover the second component, not the first. Pushing global `a_p` all the
   way to 1.0 would over-correct the genuinely noisy part and should degrade PCC.
   Note `--anchor_weight 3.0` drove median `a_p` to 0.153 while *also* weakening
   r(a_p,SASA) to +0.400 — evidence these knobs move the two components together, and a
   caution against reading `a_p → 1` as the objective.

3. **The right target is the SASA-dependence, not the median.** The failure mode is that
   the model compresses **exposed** proteins' ddG far more than buried ones. The success
   criterion for any slope lever should be **flattening r(a_p, mean_rel_SASA) toward 0**,
   with median `a_p` as a secondary readout. A run that lifts median `a_p` while leaving
   r(a_p,SASA) at +0.7 has not fixed the failure — it has rescaled it.

4. **The errors-in-variables loss is demoted, not discarded.** It is not the primary fix;
   it addresses at most the structure-independent floor. It cannot explain — and so cannot
   repair — the exposure-dependent component, which is the larger and more interesting
   signal.

5. **Recommended follow-up (no GPU consumed here):** the per-mutation Tsuboyama CIs exist
   in `Pnas_filtering/pnas_mutations.csv` but are disjoint from our 28. The clean
   experiment is to compute `a_p` vs. measured σ on the ~160 PDB-like proteins that *do*
   carry CIs and *are* in the training/validation split. If `a_p` tracks measured σ there
   while still tracking SASA independently, both components are confirmed and separable.
   That requires an eval pass over those proteins — **write the command and stop**, per
   the standing rule.

---

## Provenance

- Analysis: `scripts/slope_origin2.py`, `scripts/slope_sim.py` (both new, read-only w.r.t. the model).
- `a_p` refit with the same OLS as `calc_bp.py` (`fit(ddG, pred_ddG)`); reproduces
  median 0.4975, range 0.0789–1.2566, `corr(a_p, PCC)=+0.5714`.
- `mean_rel_SASA` and `length` reused verbatim from the cached `results/catalogue_vs_bp.json`
  `merged_table` (Shrake–Rupley, Tien et al. 2013 max-ASA) — not re-derived.
- p-values: two-sided t on the correlation via the regularized incomplete beta.
- Simulation seed 20260907, 2000 trials.
- No feature-vector code touched. `gate_g4_cpu.py` after all work:
  `baseline forward ok, dG finite PASS dG=-0.0030 width=1092`, **G4-CPU: ALL PASS**.
