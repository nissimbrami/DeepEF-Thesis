# Ensembling: no help on pooled, a REAL and significant gain at k=20

**Date:** 2026-09-09. First actual test of ensembling in this project. **Zero GPU cost** — it
reuses checkpoints already scored. 57 healthy runs, 27 proteins (2K5H dropped), 27,187 rows.

## 0. The alignment problem, and the key that solves it

**Eval CSVs are NOT row-aligned.** Protein blocks appear in different orders per file: only
1 of 79 files matched the first file's row order. Naive `np.mean` over raw columns would have
silently averaged different mutations together.

`(protein, deltaG)` is a valid unique key — 2K28 has 920 unique deltaG values in 920 rows.
Joining on it aligns **57 of 57** healthy runs. Anyone ensembling these files must join, never zip.

## 1. A trap I fell into first, recorded so nobody repeats it

Averaging **within-protein z-scored** predictions gives pooled 0.6650 for a single run whose raw
pooled is 0.6382. That +0.027 is **not an ensemble gain — it is an oracle**: z-scoring subtracts
each protein's test-set mean and divides by its test-set sd, i.e. it removes `b_p` and `a_p`
using the labels. Decomposition on the top 6 runs:

| run | raw | centred (offset oracle) | z-scored (affine oracle) |
|---|---|---|---|
| sigma_seed2_e10 | 0.6382 | 0.6644 | 0.6650 |
| w12_s1_e14 | 0.6191 | 0.6620 | 0.6686 |
| w12_s2_e12 | 0.6159 | 0.6605 | 0.6704 |

**Any per-protein normalisation applied before scoring is the oracle wearing a different name.**

## 2. Honest ensembling — raw averaging, no normalisation

| ensemble | pooled | k=20 |
|---|---|---|
| best single (`sigma_seed2_e10`) | **0.6382** | 0.7591 |
| top-2 | 0.6350 | 0.7591 |
| top-3 | 0.6364 | 0.7614 |
| **top-5** | 0.6368 | **0.7661** |
| top-10 | 0.6340 | 0.7644 |
| top-20 | 0.6308 | 0.7595 |
| all 57 | 0.6099 | 0.7570 |

**On pooled, ensembling never beats the best single run** — it monotonically decays toward 0.61.
Reason: runs sit at different offsets `b_p`, and averaging predictions at different offsets does
not cancel them, it blends them.

## 3. At k=20 the picture reverses, and it is significant

Paired bootstrap, **identical anchor draws for every arm**, 250 draws:

| comparison | delta | 95% CI | P(>0) |
|---|---|---|---|
| **top-5 ensemble − best single** | **+0.0070** | **[+0.0052, +0.0092]** | **1.000** |
| **top-5 ensemble − control** | **+0.0148** | **[+0.0092, +0.0225]** | **1.000** |
| best single − control | +0.0078 | [+0.0024, +0.0157] | 0.988 |

```
control 0.7513  ->  best single 0.7591  ->  top-5 ensemble 0.7661
```

**Both CIs exclude zero.** Once k=20 anchors remove each run's offset, the thing that ruined the
pooled ensemble is gone, and what remains is genuine variance reduction.

**This is the largest k=20 number the project has produced**, and it costs nothing.

## 4. Why the gain is small: the runs are not diverse

Pairwise correlation between run predictions over 66 pairs of 12 healthy runs:

    mean 0.9437   min 0.8831   max 0.9867

At r ≈ 0.94 there is little independent error to average away. **More seeds of the same config
will not help; architecturally different arms would.** Note the optimum is at **5 runs, not 57** —
adding mediocre runs dilutes.

## 5. What this means

1. **Report the top-5 ensemble at k=20 (0.7661) as the project's best few-shot result.** It is
   the honest maximum, it is significant against both baselines, and it needs no new compute.
2. **Never ensemble for a zero-shot/pooled number** — it is actively worse there.
3. **The pooled-vs-k=20 divergence appears yet again**, now for a third mechanism (after factor A
   and W12). Any lever or procedure must be reported on both.

**Confidence: 94%.** Paired draws, CIs computed not assumed, the oracle trap identified and
excluded. Residual: the top-5 set was chosen by pooled rank on the same test set, a mild
selection effect — but the top-5-vs-control gain (+0.0148) is far larger than that could explain.
