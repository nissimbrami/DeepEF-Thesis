# Is `b_p` learnable? — identifiability of the per-protein calibration parameters

**Verdict: YES. `b_p` is overwhelmingly a deterministic function of the protein, not optimisation
noise. ICC(1,1) = 0.898 as a strict lower bound; 0.972 once the epoch confound is controlled.
The critic's ceiling concern is real in principle but quantitatively small here: run-to-run noise
costs at most ~0.013 pooled PCC of the offset-correction headroom.**

CPU only, no GPU, no new training. Everything below is computed from eval CSVs already on disk.
`scripts/gate_g4_cpu.py` was run before and after: `baseline ... dG=-0.0030 width=1092` — unmoved.

---

## 1. The experiment

`eval_results/` holds five evaluations of the **same `abl_sigma` configuration at different seeds**:

| seed | file | epoch | pooled ddG PCC | offset-removed PCC |
|---|---|---|---|---|
| 1  | `abl_sigma_seed1_e13.csv`  | 13 | 0.5948 | 0.7005 |
| 2  | `abl_sigma_seed2_e10.csv`  | 10 | 0.6575 | 0.7503 |
| 3  | `abl_sigma_seed3_e13.csv`  | 13 | 0.5799 | 0.7178 |
| 4  | `abl_sigma_seed4_e14.csv`  | 14 | 0.6008 | 0.7204 |
| 42 | `abl_sigma_seed42_e9.csv`  |  9 | 0.5929 | 0.7016 |

All five carry the identical 28 proteins and identical 28,314 mutations (verified: same protein
set, same sorted `deltaG` multiset). Row order differs between files, but the WT convention holds
in every one — row 0 of each protein group has `ddG == 0` in 28/28 groups in all 5 CSVs, so the
`calib_diag.py` "row 0 = WT" reference is valid per file.

This is a genuine same-data, same-config, different-seed replicate set: exactly the experiment the
question needs, already run.

---

## 2. A definitional correction that matters

**There are two different `b_p` objects in this codebase, and the headline `std(b_p) = 1.5741` is
only one of them.**

| symbol | definition | source | spread |
|---|---|---|---|
| **`b_p` (canonical)** | `pred_deltaG[WT] - deltaG[WT]` — the **WT error on absolute dG** | `scripts/calc_bp.py` | **std 1.5741** |
| `b_ddg` | intercept of `polyfit(ddG_true, ddG_pred, 1)` | `calib_diag.py` (`std_b`) | std ~0.175 |

The brief said to follow `calib_diag.py` conventions. That is correct for `a_p` (both scripts
compute the ddG slope identically), but `calib_diag.py`'s intercept is **not** the 1.5741 object —
it is ~9x smaller. Running `calc_bp.py` unmodified reproduces every measured fact exactly
(`std 1.5741`, `a_p` median `0.4990`, per-protein ddG PCC median `0.7980`,
`corr(a_p, PCC) = +0.5714`), confirming which definition owns the number.

This is also what the METRIC RULE demands: `b_p` is a reference-state quantity and must be scored
on dG. **Both are reported below**; the canonical dG-side `b_p` carries the conclusion.

---

## 3. Variance decomposition (ICC)

One-way random-effects `ICC(1,1)` on a balanced 28 proteins x 5 seeds matrix:
`ICC = (MSB - MSW) / (MSB + (k-1)*MSW)`, where MSB is between-protein and MSW is within-protein
(across-seed) mean square. ICC is the fraction of variance attributable to the protein.

| quantity | across-protein SD | within-protein across-seed SD | SD protein | SD noise | **ICC(1,1)** |
|---|---|---|---|---|---|
| **`b_p` (WT error, dG)** | 1.3919 | 0.4405 | 1.3807 | 0.4658 | **0.8978** |
| `b_ddg` (ddG intercept) | 0.1755 | 0.0412 | 0.1678 | 0.0626 | 0.8780 |
| **`a_p` (ddG slope)** | 0.1960 | 0.0730 | 0.1932 | 0.0827 | **0.8450** |

Assumption-light corroboration — mean pairwise across-seed correlation of the per-protein vector:
`b_p` **0.964** (min 0.937), `a_p` **0.962** (min 0.942), `b_ddg` 0.883 (min 0.709).
Two independently seeded runs rank the proteins' offsets nearly identically.

Bootstrap 95% CI over proteins: `b_p` **[0.776, 0.941]**, `a_p` [0.734, 0.884].
Leave-one-seed-out `b_p` ICC ranges 0.873-0.961 — no single seed drives the result.

---

## 4. The epoch confound — why 0.898 is a LOWER bound

**Stated honestly: these five checkpoints are not epoch-matched (e9-e14).** `ICC(1,1)` charges
*all* of that epoch drift to the "noise" term. So the across-seed variance here is an **upper bound
on the true seed-noise share**, and the reported ICC is a **lower bound on the protein-attributable
share**. The drift is real and visible: the per-seed mean `b_p` correlates with epoch at
**r = -0.805**, and seed42 (the earliest, e9) is the clear outlier (column mean +0.515 vs about
-0.2 for the others).

Two controls separate epoch drift from seed noise:

| estimator | what it removes | `b_p` | `a_p` |
|---|---|---|---|
| `ICC(1,1)` one-way | nothing — epoch counted as noise | **0.8978** *(lower bound)* | 0.8450 |
| `ICC(C,1)` two-way consistency | per-seed additive shift (about the epoch effect) | 0.9515 | 0.9243 |
| `ICC(1,1)` on the **epoch-matched pair** (seed1_e13 vs seed3_e13) | epoch entirely — pure seed contrast | **0.9722** | 0.8060 |

Dropping seed42_e9 alone lifts `b_p` ICC to 0.961. Converging evidence: **the true seed-only ICC
for `b_p` is about 0.95-0.97, and 0.898 is conservative.**

(`a_p`'s epoch-matched value, 0.806, is slightly *below* its one-way value — with a single pair and
28 proteins that estimate is noisy, so `a_p` is best quoted as ICC about 0.85 with the two-way 0.92
as the upper end. The `a_p` conclusion is unchanged either way.)

---

## 5. Consequence — the ceiling this sets

With `std(b_p) = 1.5741`, total offset variance = **2.4778**.

- Protein-attributable (**reachable by any perfect feature-based corrector**):
  `0.8978 * 2.4778` = **2.2246** (89.8%)
- Irreducible run-to-run noise: **0.2532**, i.e. an SD of **0.503** kcal/mol of `b_p` that
  *no* feature can ever predict — because it is not a property of the protein.

Translated into the metric the thesis is judged on. Averaged over the five seeds, raw pooled ddG
PCC = **0.6052** and the offset-removal oracle = **0.7181** (gain **+0.1130**). A corrector can
only capture the identifiable share of the offset variance, so the gain scales in
variance-explained terms as `r^2 -> r_raw^2 + ICC*(r_oracle^2 - r_raw^2)`:

> **Best achievable pooled PCC from offset correction is about 0.705**
> (vs. 0.718 for the unattainable oracle — **only 0.013 PCC is lost to seed noise**).

Using the canonical dG-side ICC of 0.898 instead gives 0.707; the epoch-matched 0.97 gives about
0.714. **The bound is insensitive to which of these ICCs is used.**

### What this means for the b_p levers

The critic's mechanism is correct — unidentifiable variance *does* set a hard ceiling — but the
measured magnitude does not threaten the programme. Roughly **90% (likely 95%+) of `b_p` is a
stable, protein-determined quantity**, and the ceiling it imposes sits ~0.013 PCC below the oracle
target, not meaningfully below it. **If a `b_p` lever underperforms, identifiability is not the
explanation; the feature set is.** The same holds for `a_p` (ICC 0.845), which remains a legitimate
within-protein target.

---

## 6. Guard against the signature failure mode

*(code runs, reports a number, feature never read)*

| control | expected | measured (`b_p`) |
|---|---|---|
| Shuffle protein identity within each seed column (200x) | ICC -> 0 | **-0.005** (p95 +0.097) |
| Pure-noise matrix, same marginal scale (200x) | ICC -> 0 | **+0.004** |
| Identical columns (perfectly reproducible object) | ICC = 1 | **1.0000** |
| Real data | >> shuffle p95 | **0.8978** PASS |

`GUARD: PASS`. The statistic collapses when per-protein identity is destroyed and saturates when
reproducibility is perfect, so 0.898 is genuinely reading the protein-attributable structure and is
not an artefact of the arithmetic.

---

## 7. Reproduce

```
python scripts/icc2.py        # per-seed a_p / b_p / b_ddg fits + ICC -> results/identifiability_raw.json
python scripts/icc3.py        # epoch controls + achievable-PCC bound -> results/identifiability_epoch.json
python scripts/icc_guard.py   # negative/positive controls           -> results/identifiability_guard.json
python scripts/mkreport.py    # assembles                            -> results/identifiability.json
```

## 8. Caveats

- 28 proteins x 5 seeds. The bootstrap CI on `b_p` ICC ([0.776, 0.941]) reflects that; the point
  estimate is well-determined but not razor-sharp.
- Seeds are not epoch-matched (e9-e14); handled in section 4, and stated as a lower bound
  throughout.
- The achievable-PCC bound assumes offset removal acts as a variance subtraction and that a
  corrector captures the identifiable share uniformly. It is an *upper* bound on what a corrector
  can deliver — a real feature set will fall short of it.
- All 28 proteins are single-chain, ligand-free monomers (43-72 aa), so this ICC is measured on
  that regime only.
