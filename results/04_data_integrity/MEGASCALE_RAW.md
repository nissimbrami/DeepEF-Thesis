# MegaScale Raw Data — Inventory and Unused Value

Scope: what is actually on disk under `data/`, whether per-mutation uncertainty can be
attached to our 28 test proteins, whether multi-point mutants exist for them, and what
fraction of MegaScale we train on. All numbers computed on the cluster; n and the n=28
significance threshold (|r| < 0.374 indistinguishable from 0 at p=0.05) stated where used.

Gate after this work: `scripts/gate_g4_cpu.py` -> **ALL PASS** (read-only analysis; no code changed).

---

## 1. Real file inventory

`data/` in the repo is mostly **symlinks** into a shared group tree. A local listing is
misleading — `du` reports 512 B for entries that are actually tens of GB.

Real root: `/groups/keasar_group/casp15/Shahar/DeepPEF/data`

| Path | Rows / count | Notes |
|---|---|---|
| `Processed_K50_dG_datasets/mutation_datasets/*.csv` | **862 files, 733,945 rows** | THE per-protein MegaScale source. Cols: `name,deltaG,aa_seq,mut_type,ddG_ML,Stabilizing_mut` |
| `Processed_K50_dG_datasets/training_data/<P>/` | 368 dirs | `coords_tensor.pt`, `deltaG.pt`, `mask_tensor.pt`, `one_hot_encodings.pt`, `prott5_embeddings/` |
| `ThermoMPNN/mega_test.csv` | 28,312 x **41** | **17 CI columns.** This IS our test set (28/28 proteins) |
| `ThermoMPNN/mega_train.csv` | 216,919 x 7 | no CI columns |
| `ThermoMPNN/mega_val.csv` | 27,481 x 41 | 17 CI columns |
| `Processed_K50_dG_datasets/Pnas_filtering/pnas_mutations.csv` | 244,400 x 41 | 270 proteins, CI cols, **disjoint from our 28** |
| `Pnas_filtering/mega_train.csv`, `mega_val.csv` | 216,919 / 27,481 | same-size twins of the ThermoMPNN pair |
| `foldx_all.csv`, `foldx.csv`, `rosetta*.csv` | 73 MB / 99 MB / up to 137 MB | third-party predictor baselines, untouched |
| `blusom62.csv`, `blusom62_values.csv` | 24 MB / 15 MB | untouched |
| `S669/` | `all_data.pt` 404 MB, proT5 embs 3 x 192 MB | separate benchmark |
| `MsDs/`, `FireProtDB/`, `3DRobot/`, `casp12_data_*`, `data2/` (42,382 entries) | — | not MegaScale |

CI columns in the 41-col schema: `log10_K50_{t,c}_95CI{,_high,_low}`,
`fitting_error_{t,c}`, `deltaG_{t,c}_95CI{,_high,_low}`, `deltaG_95CI{,_high,_low}`.

---

## 2. KEY QUESTION: per-mutation uncertainty for our 28 — SOLVED, YES

The earlier attempt failed because it searched the **wrong file**. `pnas_mutations.csv`
genuinely does not contain our 28 — confirmed four independent ways:

| Join attempt vs `pnas_mutations.csv` (270 proteins) | Hits |
|---|---|
| exact `protein_name` / `WT_name` / `name` | 0/28 |
| PDB-id normalisation (strip `.pdb`, first `_` token, upcase) | 0/28 |
| exact WT sequence | 0/28 |
| WT seq substring, either direction | 0/28 |
| **nearest-neighbour difflib ratio over all 270 WTs** | **max 0.741; zero >= 0.95** |

Best near-miss: `1TUC` vs `1PWT` at 0.741. The 28 are a **genuinely held-out split** —
that negative is now proven, not assumed.

**But `ThermoMPNN/mega_test.csv` is a different file and it contains all 28:**

- 28/28 by name (`WT_name` with `.pdb` stripped), 28/28 by exact WT sequence.
- 28,312 rows vs our eval CSVs' 28,314; per-protein counts match exactly for 27/28
  (only `2K5H` differs: 1,123 vs 1,125).
- **100% non-null** on `deltaG`, `deltaG_95CI`, `fitting_error_t`, `ddG_ML`.

Row-level join on `protein + round(deltaG, 5)`:

| rounding | matched | ambiguous keys |
|---|---|---|
| 6 dp | 26,476 / 28,314 (93.5%) | 7 |
| **5 dp** | **28,021 / 28,314 (99.0%)** | 64 |
| 4 dp | 28,277 (99.9%) | 662 |
| 3 dp | 28,314 (100%) | 5,475 |

Use 5 dp and drop ambiguous keys -> **27,895 rows (98.5%) carrying a per-mutation sigma**.

### The errors-in-variables answer: a_p is NOT attenuation

sigma = `deltaG_95CI` / (2 x 1.96). Median sigma = **0.0372 kcal/mol**; std(deltaG) = 1.3275.
Label noise is **0.30% of the variance in deltaG** (max 0.86% on any single protein).

Attenuation-corrected slope a_eiv = a_ols x var(x) / (var(x) - mean(sigma^2)), all 10 checkpoints:

| checkpoint | a_ols median | a_eiv median |
|---|---|---|
| abl_calib_ctrl_repro2_e14 | 0.4969 | 0.4976 |
| abl_anchor_w1.0_s42_e14 | 0.5896 | 0.5922 |
| abl_anchor_w0.3_s42_e14 | 0.5825 | 0.5854 |
| abl_anchor_w3.0_s42_e13 | 0.1536 | 0.1543 |
| abl_p3_slope3.0_s42_e8 | 0.3258 | 0.3273 |
| abl_sigma_seed{1,2,3,4,42} | 0.3266–0.4371 | 0.3285–0.4407 |
| **mean over 10** | **0.4088** | **0.4106** |

**Correction factor 1.0015 — a 0.15% change against an a_p median of 0.4990.**

> **Verdict: measurement error in the labels explains essentially none of a_p ~= 0.50.
> The regression-dilution hypothesis is dead.** a_p is a real property of the model
> (consistent with exposure -> a_p r=+0.714 on 10/10 checkpoints), not an artifact of
> noisy targets. This closes the question the a_p/b_p work has been circling.

Secondary unlock: 27,895 per-mutation sigmas enable inverse-variance-weighted metrics.
Given lambda=0.003, expect these to move pooled PCC by <0.01 — a robustness line, not a lever.

---

## 3. MULTI-POINT MUTANTS: 26,315 exist for our 28 and are in NO split

`mutation_datasets/*.csv` `mut_type` uses `:` for multi-point (e.g. `R15Q:T36Q`).
All are **exactly double** mutants (0 triples). For our 28, raw 63,380 rows vs 28,314 evaluated:

| category | rows (28 proteins) |
|---|---|
| single | 31,873 |
| **multi (double)** | **26,315** |
| insertion | 3,366 |
| deletion | 1,676 |
| wt | 150 |
| **total** | **63,380** |
| currently evaluated | 28,314 (44.7%) |

20 of the 28 have doubles; largest are `3DKM` 7,448, `1QP2` 3,600, `4C26` 2,285,
`2K28` 1,884, `2KVS` 1,601, `1QKH` 1,483. Eight have none.
All 26,315 have **finite measured `deltaG`** (0 NaN).

Confirmed these never entered any split: `mega_test.csv` multi rows = **0**,
`mega_train.csv` multi rows = **0**, ins/del = 0 in both. The loader
`data_loaders/dg_data_loader.py:34` drops only ins/del:

    mutations = mutations[~mutations['mut_type'].str.contains('ins|del')]

so doubles were removed further upstream, when the ThermoMPNN split was built.
Tensors exist for all 28 (`training_data/<P>/coords_tensor.pt`, e.g. 3DKM (72,4,3)),
so **this test needs no new labels and no new preprocessing.**

### Additivity baseline (the number a model must beat)

89.9% of doubles (23,652) decompose into two singles that are themselves measured, so
ddG_additive = ddG(mut1) + ddG(mut2) is computable:

- **r(additive, observed) = 0.7108** (weighted, n=23,652)
- **mean epistasis = +1.235 kcal/mol**, weighted sd **1.311**

Per-protein r ranges 0.34 (`2WXC`) to 0.91 (`6EWT`, `2KXD`). The large positive mean
epistasis says doubles are systematically **more stable than additive** — a real, strong
signal, and a held-out generalisation test with a well-defined naive baseline. This is
the single largest block of unused labelled data we have.

---

## 4. What fraction of MegaScale do we train on

Across all 862 proteins in `mutation_datasets/`:

| category | rows | share |
|---|---|---|
| single | 451,514 | 61.5% |
| **multi (double)** | **210,118** | **28.6%** |
| insertion | 46,805 | 6.4% |
| deletion | 23,359 | 3.2% |
| wt | 2,149 | 0.3% |
| **total** | **733,945** | 100% |

The train/val/test split totals 216,919 + 27,481 + 28,312 = **272,712 rows = 37.2%**.

**We use 37.2% of what is on disk; 62.8% (461,233 rows) is excluded.** The excluded part is
210,118 doubles + 46,805 insertions + 23,359 deletions + the singles belonging to the
862 - 368 = 494 proteins that have a `mutation_datasets` CSV but **no tensor directory**
in `training_data/`. Doubles alone are 210,118 rows — a ~77% increase over the 272,712
currently split, all with measured deltaG.

Usability of the excluded part:
- **Doubles (210k): usable now** for the 368 tensor-backed proteins; no new labels needed.
- **Ins/del (70k): not usable** without an architecture change — they alter sequence
  length, and the loader explicitly drops them.
- **494 proteins with no tensors: needs preprocessing** (coords + prott5), not a data gap.

---

## Recommended next actions, ranked

1. **Report the EIV result.** a_ols 0.4088 -> a_eiv 0.4106 (factor 1.0015) across 10
   checkpoints kills the attenuation explanation of a_p. Thesis-grade negative, already
   computed, costs nothing further.
2. **Double-mutant generalisation test.** 26,315 held-out doubles on the 28, tensors
   present, baseline r=0.711 / epistasis +1.235 already established. Zero new labels.
3. **Sigma-weighted metrics** as a robustness paragraph — expect <0.01 movement (lambda=0.003).
4. Scale-up option: 210,118 doubles over 368 tensor-backed proteins as extra training signal.

## Caveats

- The `deltaG`-value join is a value join, not a key join; 419 of 28,314 rows (1.5%) are
  dropped as ambiguous at 5 dp. `mut_type` exists in `mega_test.csv` and would give an
  exact key if the eval CSVs were regenerated to carry it.
- `ddG_ML` in `mutation_datasets/*.csv` is string-typed with `-` as a sentinel for
  unreliable values (2,033 of 9,037 for 3DKM). **Always `pd.to_numeric(..., errors='coerce')`.**
  `deltaG` is fully numeric. The additivity analysis uses `deltaG` minus WT, not `ddG_ML`.
- The 2-row discrepancy (28,314 eval vs 28,312 mega_test, from 2K5H) is unexplained; small,
  but worth resolving before quoting exact per-protein n.
- Epistasis mean uses a WT reference taken as the mean of the `wt` rows per protein; a
  different reference convention shifts the mean but not r.
