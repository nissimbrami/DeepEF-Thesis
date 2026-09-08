# THE CANONICAL HEADLINE BASIS
### DeepEF · P8 · resolved 2026-09-08 · enforced by `scripts/gate_headline.py`

---

## 1. The declaration

> **27 test proteins (2K5H excluded) · ddG metric · the 9 original-population eval CSVs ·
> the val-selected epoch only.**

    pooled 0.5772 ± 0.0316    oracle 0.7156 ± 0.0474    gain +0.1384    (n = 9 runs)

`±` is the standard deviation **across the nine runs**, not a confidence interval on any one of
them. The offset-removal oracle subtracts, from each protein's predictions, the mean residual
fitted on **that protein's own test labels** — it is an oracle, not a method, and licenses exactly
one claim: *the ranking information is present; what is lost is calibration.*

**Best single run on this basis: pooled 0.6382** (`abl_sigma_seed2_e10.csv`). Quote the mean when
describing the model, the best run only when explicitly naming it as the best of nine seeds.

---

## 2. The nine CSVs, exhaustively

| eval CSV | pooled | oracle |
|---|---|---|
| `abl_calib_ctrl_repro2_e14.csv` | 0.5635 | 0.7364 |
| `abl_sigma_seed1_e13.csv` | 0.5607 | 0.7244 |
| `abl_sigma_seed2_e10.csv` | **0.6382** | **0.7534** |
| `abl_sigma_seed3_e13.csv` | 0.5564 | 0.7287 |
| `abl_sigma_seed4_e14.csv` | 0.5759 | 0.7278 |
| `abl_sigma_seed42_e9.csv` | 0.5592 | 0.7274 |
| `abl_anchor_w0.3_s42_e14.csv` | 0.6105 | 0.7384 |
| `abl_anchor_w1.0_s42_e14.csv` | 0.5942 | 0.7112 |
| `abl_anchor_w3.0_s42_e13.csv` | 0.5364 | 0.5929 |
| **mean (n=9)** | **0.5772** | **0.7156** |

**What makes these nine a population.** Their SLURM job names all begin `DeepEF_`. They share
**one training set** and **one recipe**, differing only in seed or in the single lever named in
the tag. That shared training set is the entire justification for averaging over them. Every
exclusion below breaks exactly that property.

---

## 3. What is excluded, and why each exclusion is forced

| excluded | why |
|---|---|
| `p3_*` factorial cells (17) | A crossed design, not a population. 8 of the 17 are D1. |
| `gld_*` golden-lane arms (16) | A different lane; a different lever per arm. |
| `abl_loroW_onehot_s42_e12.csv` | **A different training set** — see below. |
| non-val-selected epochs of the two trajectory runs | Quoting the best of six scored epochs is test-set peeking (P0). |

### 3.1 The one trap — read this before regenerating the number

A glob over `eval_results/abl_*.csv` that excludes `p3_` and `gld_` prefixes returns **ten** files,
not nine. The extra file is `abl_loroW_onehot_s42_e12.csv`.

It looks original-population because **its filename is missing the `gld_` prefix its siblings
carry**. It is not. Its SLURM job is `gld_loroW_onehot` (job 21084179), and line 1 of its log reads:

    [LORO] holding out of TRAINING (both directions): W

Every tryptophan mutation was held out of training. **It is a different training set**, and pooling
it with the nine would be averaging across exactly the thing that makes the population a population.

Including it moves the headline by only **+0.0009 pooled** — small enough to hide, which is
precisely why membership in `scripts/gate_headline.py` is **listed by hand and never globbed**.

---

## 4. Why the record disagreed with itself — and what was actually wrong

The audit's finding is not the one P8 anticipated. **0.5772 was already the canonical number,**
computed on exactly the right nine CSVs. The defect was never bad arithmetic. It was that
**the record never wrote down which nine**, so two averages over other populations stood beside it
as apparent equals, and later documents picked whichever they met first.

| basis | n | pooled | oracle | gain | status |
|---|---|---|---|---|---|
| **9 original-population CSVs** | **9** | **0.5772 ± 0.0316** | **0.7156** | **+0.1384** | **CANONICAL** |
| + `loroW` (naive 10-file glob) | 10 | 0.5767 ± 0.0299 | 0.7165 | +0.1398 | wrong — different training set |
| 22 mixed CSVs (p3 17 + sigma 5) | 22 | 0.4252 ± 0.2251 | 0.6014 | +0.1762 | *retired* — reconstruction, see §4.1 |
| 20 non-D1 checkpoints, three lanes | 20 | 0.5646 ± 0.0898 † | 0.6347 † | +0.0702 † | *retired* — **does not reproduce**, see §4.1 |
| all D0 CSVs | 35 | 0.5175 ± 0.1425 | 0.6704 | +0.1529 | mixed lanes — not a population |
| all 43 CSVs | 43 | 0.4493 ± 0.1996 | 0.6171 | +0.1679 | meaningless |
| p3 cells, D0 only | 9 | 0.5841 ± 0.0237 | 0.7218 | +0.1378 | valid *within* the factorial only |
| p3 cells, D1 only | 8 | 0.1508 ± 0.1243 | 0.3841 | +0.2333 | valid *within* the factorial only |
| gld arms only | 16 | 0.4430 ± 0.1851 | 0.6127 | +0.1696 | per-arm, never averaged |

**Read the standard deviations, not the means.** The canonical basis has σ = 0.0316; the 22-CSV
average has σ = 0.2251, seven times larger. That inflation *is* the mixed population showing
itself — the "average" is describing a bimodal mixture of working and broken models, and a
correlation averaged over a mixed population is not a measurement of anything.

### 4.1 The retired figures cannot be fully re-derived — and that is the finding

Every row above marked **CANONICAL** or unmarked was recomputed today from `eval_results/`. The two
*retired* rows were not, and could not be:

- **The 22-CSV average.** The closest reconstruction available today (the 17 `p3_*` cells plus the
  5 `sigma_*` seeds = 22 files) gives **0.4252**, not the published **0.4899**. So the published
  figure was computed over *some other* set of 22 CSVs, and which 22 was never recorded.
- **The 20-checkpoint average.** `0.5646 ± 0.0898 / 0.6347 / +0.0702` is quoted here **as
  published** (marked †), reproduced from `THESIS_RESULTS.md` and **not independently verified**.
  It does not reproduce from the current CSV population: the nearest defensible reconstruction —
  one CSV per D0 run, val-selected epoch — is n=25, pooled 0.5665 ± 0.0756, oracle 0.7081. The
  original 20 checkpoints included models that are no longer in `eval_results/` in this form.

**This is the defect in its sharpest form: a number whose basis is unrecorded cannot even be
audited afterwards.** Both figures are retired not because they were shown to be wrong, but
because they can no longer be shown to be anything at all. That is a sufficient reason to retire a
number, and it is the reason §2 lists all nine canonical files by name.

---

## 5. THE RULE — never average across factor D

**Factor D** is `--unfolded_emb zero`. D1 replaces the unfolded-state embedding with zeros and
**destroys the model**.

    D0 pooled ddG PCC:   0.176 … 0.638
    D1 pooled ddG PCC:  −0.074 … 0.317

Within the p3 factorial, where D is the only deliberately varied factor:

    D0  0.5841 ± 0.0237  (n=9)
    D1  0.1508 ± 0.1243  (n=8)
    difference = 0.4333, i.e. 14.4 × the ±0.030 seed-noise sigma

> **RULE. Never average a correlation across runs that differ in factor D.
> Report D0 and D1 SEPARATELY, always.**

D0 and D1 are not two settings of one model; they are two different models. Any main effect for
another factor estimated across both is swamped by a D effect roughly ten times larger. The
retired 22-CSV headline broke this rule, which is why it read 0.4252 instead of 0.58.

**Stating the σ units precisely:** the 14.4 figure is in units of the **seed-noise band**
(0.4333 / 0.030), which is the meaningful yardstick for this project. As a conventional pooled-SD
t-statistic the same contrast is 5.0σ within the factorial. Both describe the same fact — D is
decisive — but they are different units and must not be quoted interchangeably.

---

## 6. How to regenerate, and what enforces it

    python scripts/gate_headline.py          # recomputes from the CSVs; exit 0 = record agrees

The gate does three things:

1. **Recomputes** pooled / oracle / gain from the nine CSVs and fails if any drifts by >5e-5, or if
   any CSV does not carry exactly 27 proteins after dropping 2K5H.
2. **Scans** every `.md` in `results/01_start_here/` and fails if a retired figure is quoted for a
   canonical quantity without a retirement marker.
3. **Confirms** the canonical figures actually appear in the record.

**A note on the scanner's design.** It keys on the *quantity*, not the bare number. A naive numeric
grep for `0.7156` false-flags two legitimate lines in CONTEXT.md and FINDINGS.md, where 0.7156 is
the **a_p of a p3 factorial cell** — an unrelated quantity that merely collides numerically. Any
future gate over this record must match on quantity, not digits.

---

## 7. Standing caveats on the canonical number

- **2K5H is dropped, not fixed.** Its reference row is a mutant background (`_G11S`, a +3.0824
  kcal/mol shift on all 1,125 labels). A corrected file exists at
  `data_fixed/mutation_datasets/2K5H.csv`, but `pred_ddG` was also computed against the buggy
  reference, so a column subtraction would be exact only for the labels. The 27-protein basis needs
  no approximation, which is why it is canonical. Re-scoring against `data_fixed/` is **P7**.
- **The oracle is an oracle.** It fits on test labels. It bounds what solving b_p could buy; it is
  not a result.
- **n = 9 runs, 27 proteins.** At n=28 proteins, |r| < 0.374 is indistinguishable from zero. The
  ±0.060 seed-noise band applies to every comparison drawn against this basis.
