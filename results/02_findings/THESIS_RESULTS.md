# DeepEF — Results

### M.Sc. thesis, Nissim Brami, supervised by Prof. Chen Keasar, Ben-Gurion University
### Results chapter, figure list, limitations and abstract. Compiled 2026-09-08.

---

## How to read the numbers in this chapter

Every quantity carries three qualifiers, and they are not interchangeable:

1. **pooled or per-protein** — computed over all 28 test proteins at once, or within each
   protein and then aggregated.
2. **ΔG or ΔΔG** — absolute stability, or the mutation-induced difference.
3. **which population of checkpoints** — stated for every aggregate.

Two symbols recur and are easy to conflate:

| symbol | definition | typical size |
|---|---|---|
| **a_p** | per-protein slope of `pred ≈ a_p·true + b_p` in ΔΔG space | median **0.4504** |
| **b_p (ΔΔG)** | the intercept of that same fit | sd **0.1934** |
| **b_p (ΔG)** | the wild-type ΔG prediction error, `pred_wt − true_wt` | sd **1.6030** |

The two `b_p` are different quantities. The large number (1.60 kcal/mol) is the ΔG-space
wild-type error; the small number (0.19) is the ΔΔG intercept. Earlier drafts quoted
**std(b_p) = 1.5741** without saying which; it is the ΔG-space one, and it is recomputed here
as **1.6030** over the 28 proteins of the control checkpoint. Both are reported below with the
qualifier attached.

**Checkpoint populations used.** 27 evaluation CSVs cover all 28 test proteins. Seven of them
are `--unfolded_emb zero` (factor D1) cells, in which the model is destroyed (§5); including
them in an average of model quality would be misleading. Unless stated otherwise, aggregates
use the **20 non-D1 checkpoints**, and the D1 cells appear only in §5 where they are the result.

---

# 1. The calibration decomposition

## 1.1 Claim

**DeepEF ranks mutations well inside a protein and is miscalibrated between proteins.** The
per-mutation prediction is, to a good approximation, an affine function of the truth *within
each protein*, with a protein-specific slope and intercept:

    pred_ΔΔG ≈ a_p · true_ΔΔG + b_p

## 1.2 Evidence

| quantity | value | population |
|---|---|---|
| per-protein ΔΔG Pearson r, median | **0.7955** | 28 proteins, control checkpoint |
| pooled ΔΔG Pearson r | **0.5910** | same 28 proteins, pooled |
| per-protein a_p, median | **0.4975** | control checkpoint |
| per-protein a_p, median | **0.4504** | mean over 20 non-D1 checkpoints |
| sd of b_p in ΔG space (WT error) | **1.6030** kcal/mol | 28 proteins |
| sd of b_p in ΔΔG space (intercept) | **0.1934** kcal/mol | mean over 20 checkpoints |
| ICC(b_p) | **0.898** | 28 proteins × 5 seeds |

n = 28,314 mutations over 28 proteins in the control checkpoint.

The gap between a within-protein r of **0.7955** and a pooled r of **0.5910** is the entire
subject of this thesis. Nothing about the ranking degrades when proteins are pooled — what
degrades is that each protein's predictions sit on their own offset, so pooling mixes 28
differently-shifted scales.

**ICC(b_p) = 0.898** over 28 proteins × 5 seeds establishes that b_p is a property of the
protein and not of the training run: it is stable enough to be, in principle, predictable.

## 1.3 Caveats

- The affine form is a description, not a mechanism. It is an excellent description
  (per-protein r ≈ 0.80) but the residual 36% of within-protein variance is not modelled.
- ICC was computed with the epoch confound removed; seed-only ICC is 0.95–0.97.

## 1.4 Figure

**Figure 1** — two panels: per-mutation scatter coloured by protein, with the median slope
drawn against the identity line; and the 28 fitted per-protein lines, which visibly differ in
both slope and intercept.

---

# 2. The offset-removal oracle, and the 2K5H correction

## 2.1 Claim

**Removing each protein's own offset lifts pooled ΔΔG correlation substantially — but the
procedure that does it is an oracle, not a method,** and its size was inflated by a data bug.

## 2.2 Evidence

Over the **20 non-D1 checkpoints**, mean ± sd:

| condition | pooled ΔΔG PCC | offset-removal oracle | gain |
|---|---|---|---|
| with the 2K5H reference bug | 0.5852 ± 0.0813 | 0.6840 ± 0.1201 | **+0.0988** |
| **2K5H reference corrected** | **0.5646 ± 0.0898** | **0.6347 ± 0.0939** | **+0.0702** |

The oracle subtracts, from each protein's predictions, an intercept **fitted on that protein's
own test labels**. Pearson correlation is shift-invariant, so subtracting a *constant shared by
all proteins* changes pooled PCC by exactly zero (verified: shifts of −1.0, 0.0, the mean and
+1.0 all return 0.590991). The gain therefore comes entirely from the *per-protein* structure of
the offsets, which is the point — and equally the reason it cannot be claimed as a method.

**The 2K5H data bug.** `2K5H.csv` concatenates three mutational backgrounds and the wrong one
sorts first: row 0 was `2K5H.pdb_G11S` (ΔG 1.723) rather than the true wild type `2K5H.pdb`
(ΔG 4.805). Both are labelled `wt`, because G11S genuinely is the wild type of its own
background, so nothing looked wrong. All 1,125 of 2K5H's ΔΔG labels were shifted by
**+3.0824 kcal/mol**. A constant per-protein label shift *is* a per-protein offset by
construction, so the oracle was partly rediscovering an offset we had introduced ourselves.

The control that made it unambiguous: row-0 percentile within each protein's own ΔG
distribution. 26 of 28 proteins sit at the **62nd–99th** percentile, exactly where a wild type
belongs; **2K5H sits at 14.6%**.

Correcting it removes **29%** of the measured oracle gain (+0.0988 → +0.0702).

## 2.3 Caveats — stated at maximum strength

- **The oracle is an oracle.** It licenses exactly one claim: *the ranking information is
  present; what is lost is calibration.* It licenses nothing stronger. A feature-based
  corrector that does not see test labels fails (§7).
- **An earlier reported headline of pooled 0.4899 / oracle 0.6443 / gain +0.1544 could not be
  reproduced in this compilation** and should not be used. That triple was an *estimate* of the
  effect of the 2K5H fix, propagated arithmetically from a measured pre-fix pair; the measured
  drop-2K5H row in the same source reads 0.4882 / 0.6410 / +0.1528. The numbers in the table
  above are recomputed from the evaluation CSVs by the canonical oracle definition in
  `scripts/offset_corrector.py`, and reproduce that script's stored output exactly
  (control checkpoint: raw 0.5910, oracle 0.7113, both matching `offset_corrector.json`).
- **The affine oracle (removing slope as well as offset) is not reproducible** and is not
  claimed. It reaches 0.77–0.81 in only one of twelve checkpoints tested. The mathematics
  explains why it cannot be a bound: the affine oracle *divides* by a_p, and where a_p = 0.079
  that division amplifies noise ~13×. An estimator that is not monotone is not an upper bound.
  Offset removal survives precisely because subtracting a constant cannot amplify anything.

## 2.4 Figure

**Figure 2** — panel A: histogram of the 28 per-protein r values with the pooled value marked,
showing the gap; panel B: the oracle bars before and after the 2K5H correction.

---

# 3. a_p = r · s — why the slope is two numbers

## 3.1 Claim

**The per-protein slope decomposes exactly into a ranking term and a spread term, only one of
which any calibration lever can move.** For a least-squares fit,

    a_p = r · s,   where s = std(pred) / std(true)

## 3.2 Evidence

Over the 20 non-D1 checkpoints, medians across the 28 proteins:

    a_p = 0.4504     r = 0.7630     s = 0.6159

| source of the gap to a_p = 1 | size | movable by `--slope_weight`? |
|---|---|---|
| ranking error, 1 − r | **0.2370** | **no** |
| spread compression, 1 − s | **0.3841** | **yes** |

The identity is exact, not empirical — Figure 3A confirms it holds to numerical precision on
every checkpoint, which is a correctness check on the pipeline rather than a finding.

**The finding is what happens when the lever is pulled.** Three training arms differing only in
`--slope_weight`, one seed each:

| `--slope_weight` | a_p | r | s |
|---|---|---|---|
| control (no term) | 0.4975 | 0.7955 | 0.6342 |
| 0.3 | 0.4063 | 0.7997 | 0.5727 |
| **1.0** | **0.5347** | 0.8062 | **0.7240** |
| 3.0 | 0.3253 | 0.7943 | 0.4489 |

**r is flat at 0.794–0.806 while s swings from 0.449 to 0.724.** The lever moves the spread and
leaves the ranking untouched, exactly as the identity requires.

Two consequences the write-up must state:

1. **There is a ceiling.** Driving s → 1 pins a_p to r. The lever cannot reach a_p = 1 by
   construction; the reachable maximum is ≈ 0.76.
2. **The response has an interior optimum.** Weight 1.0 gains +0.0372 in a_p over control at no
   cost to r; **weight 3.0 overshoots and is worse than using no lever at all** (0.3253 vs
   0.4975). A factorial main effect for this factor must not be read as monotone.

## 3.3 Caveats

- **One seed per slope arm.** Seed-only variation in median a_p across 5 replicate seeds is
  sd 0.0515, range 0.327–0.437 — comparable to the control-to-0.3 difference. The
  control-to-1.0 gain (+0.0372) is *within* that band and is therefore suggestive, not
  established; the 3.0 collapse (−0.1722) is well outside it and is the safer claim.
- Epochs differ between arms (control e14, 0.3 e9, 1.0 e13, 3.0 e8), so training length is a
  partial confound on the weaker comparisons.

## 3.4 Figure

**Figure 3** — panel A: a_p against r·s across checkpoints, on the identity line; panel B: r, s
and a_p across the four slope arms, showing r flat and s moving.

---

# 4. Where the ranking error lives: burying hydrophobics

## 4.1 Claim

**The model's ranking failure is chemically specific: it is worst for mutations *to*
hydrophobic residues, and rank accuracy degrades in lockstep with the slope — so this is lost
information, not a rescalable calibration error.**

## 4.2 Evidence

Per-destination-residue slope and Spearman, over 6 non-D1 checkpoints, **26,331 matched
mutations each**. The destination residue was recovered by joining on ΔG against the mutation
files; ambiguous values were dropped, never guessed.

| destination | KD | slope | Spearman | n |
|---|---|---|---|---|
| R | −4.5 | 0.4149 | 0.6288 | 1319 |
| K | −3.9 | 0.4274 | 0.6323 | 1234 |
| N | −3.5 | **0.4400** | 0.6345 | 1323 |
| Q | −3.5 | 0.4259 | 0.6398 | 1348 |
| E | −3.5 | 0.4246 | 0.6390 | 1211 |
| D | −3.5 | 0.4229 | 0.6328 | 1272 |
| S | −0.8 | 0.4103 | 0.6194 | 1339 |
| G | −0.4 | 0.4016 | 0.6040 | 1292 |
| A | +1.8 | 0.3251 | 0.5518 | 1314 |
| M | +1.9 | 0.2613 | 0.4355 | 1384 |
| **C** | +2.5 | **0.2055** | 0.3412 | 1203 |
| F | +2.8 | 0.2140 | 0.3579 | 1385 |
| L | +3.8 | 0.2192 | 0.3726 | 1286 |
| V | +4.2 | 0.2475 | 0.4251 | 1318 |
| I | +4.5 | 0.2478 | 0.3958 | 1352 |

Over all **20** residue types:

    corr(slope, Kyte-Doolittle)     = −0.8477
    corr(Spearman, Kyte-Doolittle)  = −0.7829
    corr(slope, Spearman)           = +0.9838      <- the decisive number
    mean slope, KD<0 (13 types)     = 0.3860
    mean slope, KD>0 (7 types)      = 0.2458
    ratio                           = 1.57x

**The decisive detail is corr(slope, Spearman) = +0.9838.** Rank accuracy and slope fall
together, almost perfectly. Had the slope collapsed while ranking held, a per-class rescale
would recover the loss. It did not. **No affine correction can help here.** This is the (1 − r)
half of §3 given a chemical identity.

**Mechanism.** The dataset stores only **four backbone atoms per residue (N, CA, C, CB) — no
side chains.** Burying a large hydrophobic is dominated by side-chain packing, which CB-only
geometry cannot represent. This argues for an **architectural** fix (side-chain information),
not a calibration one.

## 4.3 Caveats

- **The compression ratio is 1.57×, not "roughly twice".** Earlier drafts said ~2×; the honest
  figures are 1.57× over all types split at KD = 0, and 1.83× comparing the extremes
  (RKQEND 0.4260 vs CFLVIM 0.2326). The stronger phrasing is not supported.
- **The "bulk not hydropathy" reading is not supported by the full alphabet.** W and Y are
  striking exceptions — slopes 0.2427 and 0.2581 despite *negative* hydropathy — which
  motivated a side-chain-bulk explanation. But over all 20 types, hydropathy correlates with
  slope at **−0.8477** while side-chain volume reaches only **−0.4428** and heavy-atom count
  **−0.1730**. Hydropathy is the better single predictor; W and Y remain unexplained outliers
  and should be reported as such rather than converted into a bulk hypothesis.
- The per-residue n is ~1,200–1,400 mutations, but these are not independent across the 6
  checkpoints, which share training data and differ only in configuration.
- The correlations are over 20 points (residue types), not 20 independent experiments.

## 4.4 Figure

**Figure 4** — three panels: per-destination slope ordered by hydropathy; slope and Spearman
against hydropathy on shared axes; and Spearman against slope, the lockstep plot.

---

# 5. The factorial: factor D is decisive

## 5.1 Claim

**Zeroing the unfolded-state embedding destroys the model.** The factor separates completely
from its alternative, with no overlap on either axis.

## 5.2 Evidence

12 of 48 factorial cells scored, 6 per arm:

| arm | pooled ΔΔG PCC | median a_p |
|---|---|---|
| **D0** — Flory coil, 6 cells | **0.5808 – 0.6100** | 0.3725 – 0.7098 |
| **D1** — `--unfolded_emb zero`, 6 cells | **−0.0366 – 0.3056** | **0.0044 – 0.0220** |

**Six versus six, perfectly separated.** a_p differs by two orders of magnitude. An a_p of
0.0044 means the model is essentially **flat**: it barely responds to mutation at all. One D1
cell is anti-correlated at −0.0366.

Seed-only noise, 5 replicate seeds: pooled PCC mean 0.5798, sd 0.0339, range 0.5594–0.6386.
**The D effect is roughly ten times that band.**

**This reverses the design decision that created the factor.** The preliminary W0 analysis chose
factor D as `--unfolded_emb` on a ΔΔG criterion, because zeroing the unfolded embedding
collapsed var(E_unfolded) by 67% and cut corr(E_unfolded, WT error) from 0.420 to 0.119 — an
apparent offset cure. **Trained end to end, it removes the offset by removing the signal.**

Three consequences:

1. The remaining D1 half is ≈190 GPU-hours confirming a negative already visible at n = 6.
2. **No main effect for factors A, B or C may be averaged across D.** The D effect is ~10×
   larger and would swamp any of them.
3. **The best cell has the highest a_p (0.7098) but not the highest pooled PCC (0.5874 vs
   0.6175).** That is the calibration thesis in one line: pooled PCC is dominated by the offset,
   while a_p measures the compression, and they can be optimised apart.

## 5.3 Caveats

- 12 of 48 cells; the factorial is incomplete and no full main-effect table is claimed.
- Cells differ in epoch reached (5–14), so training length is a confound on within-arm ordering
  — though not on the D0/D1 separation, which is far larger than any epoch effect observed.
- Most cells are one seed.

## 5.4 Figure

**Figure 5** — panel A: pooled PCC against a_p, coloured by arm, showing complete separation;
panel B: distributions per arm with the seed-noise band overlaid for scale.

---

# 6. The metric rule, with the coil as its worked example

## 6.1 Claim

**ΔΔG cancels anything identical between wild type and mutant, so a lever acting on the
reference state or on absolute stability cannot be scored on ΔΔG.** This is not a statistical
subtlety; it is exact cancellation, and it invalidated five separate measurements in this
project.

## 6.2 The rule

A point mutation changes neither chain length nor backbone coordinates. Any purely geometric or
whole-protein property is therefore bit-identical in the two forward passes and cancels
**exactly** in `pred_mut − pred_wt`.

> A lever acting on the reference state or on absolute stability **must** be scored on ΔG or on
> b_p. **a_p is a within-protein slope and does not cancel, so it is legitimately
> ΔΔG-measurable.**

## 6.3 The worked example: the Flory coil

The coil replaces unfolded-state distances with `d(i,j) = b·|i−j|^ν`, a function of **sequence
separation only**.

| how it was scored | result | verdict |
|---|---|---|
| on pooled ΔΔG | r = 1.175 relative to control | "apparently harmful" |
| **on ΔG (the metric it acts on)** | **MAE 4.9650 → 3.9521** | **−20.4%, our best offset lever** |

Because `|i−j|` is identical in wild type and mutant, the coil's entire contribution is
bit-identical between the two passes. **The ΔΔG measurement was measuring the cancellation, not
the lever.**

## 6.4 The other four levers the rule caught

1. **BSA** — declared dead on r = −0.001, computed against the pre-training *decoy loss* (a
   metric already known not to predict ΔΔG) on a column with **4 non-null rows out of 100,246**.
   Void twice over.
2. **W5 burial** — its gate passes 18/18, but every assertion is mechanical (shapes,
   byte-identity, hydropathy ordering); not one scores performance. Its own documentation states
   that burial is zero in the unfolded state and that the folded-minus-unfolded delta *is* the
   hydrophobic driving force — so it acts on ΔG — yet it was scheduled against pooled ΔΔG.
3. **Ligands** — a folded-state stabilisation term, identical in WT and mutant for any mutation
   away from the binding site, so it cancels in ΔΔG by construction.
4. **Our own model-selection metric** — `validate()` returned pooled **ΔG PCC**, which is
   scale-invariant and therefore *provably* cannot see a change in predicted spread. That is
   precisely what `--slope_weight` changes. 40 queued runs were unfalsifiable until per-protein
   a_p logging was added.

That the rule caught our own selection metric is the strongest argument for stating it
explicitly in the thesis rather than leaving it as tacit practice.

## 6.5 The companion rule

**Always compute the degenerate baseline.** `--dg_length_norm n` appeared to shrink std(b_p) by
31% (1.3076 → 0.9050 over 12 checkpoints). It does not. The `/n` column converges to ≈0.905 for
*every* checkpoint, including ones that start at 0.876 where it makes things worse — because

    std(true WT ΔG) over the 28 test proteins = 0.9042   (recomputed here: 0.9208)

Dividing by N drives pred/N to nearly zero, so b_p → −true_ΔG and its spread becomes the spread
of the labels. **The arm deletes the prediction rather than removing a bias.** (The two figures
for std(true WT ΔG) differ because they select the wild-type row differently — by dataset row 0
versus by minimum |ΔΔG|. Either way the arm's output converges to it, which is the point.)

## 6.6 Figure

**Figure 9** — the coil scored both ways, side by side: harmful on ΔΔG, our best lever on ΔG.

---

# 7. What b_p is NOT — seven rejected hypotheses

## 7.1 Claim

**The per-protein offset is real, protein-attributable and stable, and it has survived every
cheap explanation tested.**

## 7.2 Evidence

| # | hypothesis | result | verdict |
|---|---|---|---|
| 1 | label noise / regression attenuation | predicts a_p = **0.9985** vs measured ≈0.50 | **dead by 200×** |
| 2 | chain length | corr(N, b_p) = **+0.0252**, 12 checkpoints | dead |
| 3 | `--dg_length_norm` | **degenerate** — deletes the prediction (§6.5) | dead |
| 4 | embedding norm / 1/√N artifact | claimed −0.9938, **measured −0.29** (threshold 0.374) | dead |
| 5 | pre-training distribution shift | r = **+0.043**, p = 0.829 | dead |
| 6 | train/test mean gap | gap **−0.1034**, p = 0.588 | dead |
| 7 | a feature-based corrector | LOPO 0.6049 vs raw 0.5994; held-out R² negative on 8/10 | dead (§8) |

**Hypothesis 1 in detail, because it is the one that would end the thesis.** An a_p of ≈0.5 is
the textbook signature of regression attenuation under noisy labels — in which case no
architectural lever could ever fix it. Per-mutation uncertainties were located in
`data/ThermoMPNN/mega_test.csv`; **28/28 test proteins join**.

    median 95% CI  = 0.1007  ->  median sigma = 0.0327 kcal/mol   (n = 28,312)
    median var(true ΔΔG)     = 0.7178
    attenuation predicts a_p = 0.7178 / (0.7178 + 0.0327²) = 0.9985

Measured a_p ≈ 0.50 against a predicted 0.9985. The label noise would have to be **~200×
larger** (σ ≈ 0.85 rather than 0.0327) to explain the compression. **a_p is a genuine model
failure.** A sign-based argument reaches the same conclusion by an independent route:
attenuation predicts the *wrong sign* for the exposure correlation, and forward simulation gives
P(r ≥ +0.714) = 0/2000.

**Hypothesis 6 has a structural refutation as well as a statistical one:** a shared constant
cannot generate per-protein *dispersion*, whatever its size. The measured gap is 6.6% of one
standard deviation in any case.

## 7.3 What remains

The reference state itself is the last untested explanation — and it is where the only real
movement was found (coil, ΔG MAE 4.9650 → 3.9521, §6.3).

## 7.4 Structure does predict both channels

One structural axis acts on **both** calibration channels, replicated across 10 checkpoints,
n = 28 proteins:

| feature | target | mean r | sign-consistent | significant |
|---|---|---|---|---|
| `mean_rel_SASA` (exposure) | **a_p** | **+0.6239** | 10/10 | **10/10** |
| `frac_buried_rel_lt_0.25` | **b_p (ΔG WT error)** | **−0.4746** | 10/10 | 8/10 |

Exposure → slope was the **only one of 90 tested correlations** to clear Bonferroni on both
Pearson and Spearman. **It survives the length confound**, the obvious objection: partialling
out chain length gives **+0.6623, significant in 10/10**, while length alone gives −0.2149 and
is significant in **0/10**.

More-exposed proteins suffer less slope collapse; tightly buried proteins are where the model
under-reacts worst — consistent with the hydrophobic finding of §4.

This **corrects an earlier claim that "the signal is on a_p, not b_p"**, which was premature:
the b_p side was tested once, failed a uniform Bonferroni bar applied over all 90 tests, and was
dropped from the replication sweep while its sibling was replicated and promoted.

## 7.5 Caveats

- n = 28. At this size |r| < 0.374 is indistinguishable from zero at p = 0.05, and the
  Bonferroni threshold over 90 tests is far higher. Only the exposure→a_p result clears the
  corrected bar; burial→b_p clears the uncorrected one 8/10 and is reported at that strength.
- The 10 checkpoints are not independent replicates: they share training data and differ in
  configuration, so "10/10 sign-consistent" measures stability across configurations, not
  statistical power.

## 7.6 Figure

**Figure 6** — three panels: exposure vs a_p; burial vs b_p; and the per-checkpoint bar chart
showing SASA significant 10/10 while length is 0/10.

---

# 8. The corrector fails — a publishable negative

## 8.1 Claim

**Although the oracle gain is large and b_p is protein-attributable (ICC 0.898), b_p cannot be
predicted from structural features. 95% of the oracle gain does not survive held-out
prediction.**

## 8.2 Evidence

Leave-one-protein-out ridge on 22 structural features, inner LOPO α selection on the 27
training proteins only, held-out protein's label never used in any fit. Over 10 evaluation CSVs:

| condition | pooled ΔΔG PCC | gain over raw |
|---|---|---|
| raw | 0.5994 | — |
| mean baseline (no features) | 0.5940 | −0.0054 |
| **LOPO ridge (22 features)** | **0.6049** | **+0.0055** |
| oracle (uses test labels) | 0.7119 | +0.1125 |

**+0.0055 of a +0.1125 opportunity — 4.9%.** And the residual is not even stable:

- **Held-out R² of b_p is negative on 8 of 10 CSVs** (mean −0.1234).
- A permutation null is significant on **0 of 10**.
- **With the ridge penalty fixed rather than searched, the gain goes negative** — so the
  +0.0055 was α-search variance, not signal.

## 8.3 The mechanism is the useful part

The top-2 |b_p| proteins carry **78%** of the entire oracle gain, and **neither is a structural
outlier** (max |z| ≈ 2.0 across all 22 features). There is no structural signature to regress
on. This is a substantive negative result about the problem, not a modelling failure: the offset
is concentrated in a few proteins that look, structurally, entirely ordinary.

## 8.4 A metric correction that matters

Pearson correlation is shift-invariant, so subtracting a **constant** changes pooled PCC by
exactly zero — verified: −1.0, 0.0, the mean and +1.0 all give 0.590991. **The null that a
per-protein corrector must beat is therefore the RAW number, not a mean-offset baseline.**
Scoring against the latter would have manufactured a fake +0.011 lift. This is recorded because
it is an easy and invisible error.

## 8.5 Caveats

- Ridge is linear. A non-linear corrector was not tested; at n = 28 with p = 22 it would be
  very hard to evaluate honestly, and the negative held-out R² suggests the ceiling is low.
- The 22 features are structural and sequence-compositional; no evolutionary or MSA-derived
  feature was tried.

## 8.6 Figure

**Figure 7** — panel A: the four-bar comparison, raw / mean baseline / LOPO ridge / oracle;
panel B: per-checkpoint held-out R², negative on 8 of 10.

---

# 9. W6 chemical descriptors are largely redundant

## 9.1 Claim

**ProtT5 embeddings already carry residue chemistry, including for residue types withheld from
the probe. A hand-built descriptor block cannot be justified as adding information.**

## 9.2 Evidence

A **residue-disjoint** probe, chosen deliberately: fit on 19 residue types, predict the held-out
**20th**. A random split would leak the answer, because every instance of a residue type shares
one descriptor vector. 19 proteins, 1,056 residues.

| property | held-out R² |
|---|---|
| **hydropathy** | **+0.704** |
| charge | +0.511 |
| volume | +0.421 |
| identity (row-random control) | accuracy **1.000** vs chance 0.050 |

**ProtT5 predicts the hydropathy of a residue type it has never seen, at R² = 0.70.** These
values were re-verified by re-running `scripts/emb_probe.py` during this compilation and are now
persisted in `results/emb_probe.json`.

The argument: a descriptor block is a deterministic function of residue identity; identity is
perfectly decodable from the embedding (accuracy 1.000); and the chemistry itself extrapolates
to unseen types. **W6 can therefore be justified only as regularisation, or as coverage of
residues absent from ProtT5's training data — not as added information.**

Two supporting facts about the embedding: it is **contextual** (the same residue type at two
positions has cosine similarity 0.13–0.18, nearly orthogonal) and **recomputed per variant**, so
a single point mutation changes all 52 positions and the embedding does **not** cancel in ΔΔG.

## 9.3 Reconciliation with prior work in the group

Comparison against Ofir Ezrielev's thesis (read in full) corrected three errors in our build:

1. **PCA-16 is not his method.** "PCA", "principal component" and "SVD" appear nowhere in his
   thesis; his 654 features enter a CNN as 654 channels. Our PCA is ours, and is now labelled as
   such in the CSV provenance.
2. **`ignore_3D=True` contradicts his 1826 descriptors.** 1826 = 1613 2D + 213 3D, a total only
   coherent if 3D was requested — which also explains his 1826 → 1280 missing-value drop, since
   3D descriptors return NaN without a conformer. With `ignore_3D=True` the ceiling is 1613.
3. **15-of-20 is not a faithful scaling of his 40-of-58.** 40/58 = 0.690 scales to 13–14, not
   15 — and he gives no justification for 40 either, so there is no ratio to preserve.

**He concedes the language-model objection** (p. 31, verbatim): *"Physicochemical properties of
AAs are implicitly represented in these datasets ... and accordingly, implicitly represented in
the embeddings. Thus, ncAAs, which do not occur in these databases, are challenging to
represent."* **His argument is coverage only, not added information**, and he runs no experiment
comparing descriptors against a language-model embedding or against one-hot. **Our 28 test
proteins are canonical**, so the gap he exploits does not exist here.

Worth adopting from his work: **leave-one-residue-type-out evaluation**, which we had not run.

## 9.4 Caveats and the prediction on record

- The probe covers 19 of 28 proteins (embedding availability), 1,056 residues.
- R² = 0.704 for hydropathy is strong but not 1.0; a descriptor block could still contribute at
  the margin, and the claim is "largely redundant", not "useless".
- **A prediction was recorded before the result.** Two runs differ in exactly one flag, with
  tryptophan held out of training only: `gld_loroW_onehot` (`--aa_descriptors none`) versus
  `gld_loroW_desc` (`--aa_descriptors mordred_pca16`). **Prediction: the descriptor arm will not
  beat one-hot by much.** If it does, that overturns this section and will be reported as a
  genuine surprise rather than rationalised. Negative control: proline, whose effect is backbone
  geometry, which descriptors should not rescue. **These runs had not completed at compilation
  time; the section stands on the probe alone.**

## 9.5 Figure

**Figure 8** — panel A: held-out R² for the three properties under the residue-disjoint split;
panel B: strongest single-dimension probes against the n = 28 significance threshold.

---

# 10. Negative and null results

Recorded so they are not re-adopted. **Every one was plausible when proposed.**

| claim | reality |
|---|---|
| 1/√N embedding artifact, r = −0.9938 | **measured −0.29**, below the n=28 threshold of 0.374 |
| 26,315 double mutants available for generalisation testing | **zero real ones** (all 2,356 two-point rows are single mutations on 2K5H's G11S background) |
| `--dg_length_norm` shrinks b_p by 31% | **degenerate** — it deletes the prediction |
| the affine oracle reaches 0.77–0.81 | **not reproducible**; 1 of 12 checkpoints in range |
| W5 burial predicts b_p (r = −0.343, n = 19) | **not significant** — threshold 0.490 |
| "the signal is on a_p, not b_p" | **premature** — burial predicts b_p at −0.4746, 10/10 |
| five structural correlations clear Bonferroni | **only one** clears on both Pearson and Spearman |
| pLDDT predicts per-protein PCC | p = 0.015, **27× above** the stated bar, and confounded with length |
| pooled 0.4899 / oracle 0.6443 / gain +0.1544 | **an estimate, not a measurement** — see §2.3 |

**Two recurring failure modes**, both worth naming in the thesis: a correlation quoted without
its n, and an improvement quoted without asking what the number would be if the model predicted
nothing.

## 10.1 The 100k catalogue is retired for join-based purposes

**It has no sequence column.** Joins measured: **0/226** training PDB ids, **0/21** test,
**0/340** by protein_id, against 66,945 catalogue ids. This is a permanent, in-principle result,
not a failed attempt, and **it kills the plan to escape n = 28 by testing structural
correlations on the training set.**

Five columns look alive and are dead: `Ligands_x` (**4** non-null of 100,246), `BSA_Numeric_x`
(4 unique values), `BSA_Percentage` (5), `lossg` (constant 0), `Global Symmetry` (constant).
**This is where the −0.001 BSA null came from: a dead column scored against the pre-training
decoy loss.** All five are now quarantined.

The one live use is distribution shift: our regime is **0.1206%** of pre-training residues
(1 in 830), 77.4% solution NMR against the catalogue's 8.6%, and 88.2% monomer against 31.9%.
**Our entire problem domain was one residue in 830 of pre-training, and a structurally different
kind of protein** — but this is **not** the cause of b_p (r = +0.043, p = 0.829), and across 25
shift and composition features, three reach nominal significance where ~1.25 are expected by
chance, with **none surviving Bonferroni or BH-FDR** for either b_p or a_p.

## 10.2 A ligand feature would have learned crystallography

205,648 het-code occurrences over 8,187 distinct codes: **27.4%** are crystallisation additives
(SO4, GOL, EDO, NA, CL, PEG), 18.9% metal ions, **6.6%** real cofactors or substrates, 6.3%
modified residues. **Only 6.6% are real biological ligands**, and 12.8% of "ligand-bearing" rows
carry only additives. MSE is selenomethionine — an amino acid *in the chain*, used for phasing —
so counting it as a bound ligand is both biologically wrong and a double count of a residue.

---

# 11. Engineering results that are thesis-relevant

Two are worth reporting because they changed what could be measured, not merely how fast.

**Four silent no-op holes, closed.** The project's signature failure mode is code that runs,
completes, prints a success line, and never reads the feature. The decisive case was measured,
not assumed: a gate builds a feature vector with 726 sentinel columns at offset 48, runs the real
reassembly, and the output is **byte-identical to the no-descriptor case — 0 of 726 sentinel
columns reach the node vector, and nothing raises.** Five levers were affected. A guard now
fires at construction time.

**The architectural fact that constrains every feature design.** The GCN branch reads
`x[:, :32]` only — the 16 pairwise distance channels plus half of the bonded/geometric block. It
never sees the 1024-d embedding, and **any new block inserted at offset 48 is invisible to it.**
W5, W6, W9 and W11 reach the GAT branch alone. **Any claim that a block "is used by the model"
must state which branch.**

A third is worth one line: **a finished training run is not a result.** Twelve factorial cells
completed all 15 epochs and produced zero CSVs, because the evaluation script had `DEVICE =
'cuda'` with the CPU fallback commented out, `load_checkpoint` raised `KeyError` on bare
state-dict saves, and `torch.load` lacked `map_location` — while the wrapper printed
`DONE ... -> abl_<tag>.csv` *after* the crash. Progress is tracked by counting CSVs on disk.

---

# 12. Figure list

All figures are generated by `scripts/make_figs.py`, which reads only computed artefacts, and
are written to `results/figures/`. Regenerating is a single command; no number in any figure is
typed by hand.

| # | file | what is plotted | source | what it shows |
|---|---|---|---|---|
| **1** | `fig01_calibration_decomposition.png` | **A**: predicted vs measured ΔΔG, 28,314 points coloured by protein, with y=x and the median slope. **B**: the 28 fitted per-protein lines | `eval_results/abl_calib_ctrl_repro2_e14.csv` | The affine structure: slopes below 1 (compression) and intercepts spread apart (offset). The core claim in one image |
| **2** | `fig02_within_vs_pooled_and_oracle.png` | **A**: histogram of 28 per-protein r with pooled r marked. **B**: pooled vs oracle bars, before and after the 2K5H correction | same CSV; all 20 non-D1 CSVs | The 0.7955 vs 0.5910 gap that is the thesis; and that the oracle gain is +0.0702 after correction, not +0.0988 |
| **3** | `fig03_ap_equals_r_times_s.png` | **A**: a_p vs r·s across 20 checkpoints on the identity line. **B**: r, s, a_p across the four `--slope_weight` arms | 20 non-D1 CSVs | The identity is exact; the lever moves s (0.449→0.724) and never r (flat 0.794–0.806); weight 3.0 overshoots |
| **4** | `fig04_hydrophobic_ranking_failure.png` | **A**: per-destination slope ordered by hydropathy, coloured by sign. **B**: slope and Spearman vs hydropathy. **C**: Spearman vs slope | `results/k13_muttype.json`, 6 checkpoints × 26,331 mutations | Hydrophobic destinations compressed 1.57×; and in **C**, r = +0.9838 — rank and slope fall together, so the loss is information, not calibration |
| **5** | `fig05_factor_D.png` | **A**: pooled PCC vs a_p per cell, coloured by arm. **B**: per-arm distributions with the seed-noise band | `results/factorial_analysis.json` | Complete 6-vs-6 separation; a_p differs 100×; the D effect is ~10× seed noise |
| **6** | `fig06_structure_two_channels.png` | **A**: exposure vs a_p. **B**: burial vs b_p(ΔG). **C**: per-checkpoint r for SASA and for length, against the p=0.05 line | `bp_replication_sweep.json`, `length_confound.json`, `calib_per_protein.json` | One structural axis drives both channels; and SASA is significant 10/10 while length is 0/10, so it is not a length confound |
| **7** | `fig07_corrector_negative.png` | **A**: raw / mean-baseline / LOPO-ridge / oracle bars. **B**: per-checkpoint held-out R² of b_p | `results/offset_corrector.json` | The publishable negative: +0.0055 of a +0.1125 opportunity, with R² negative on 8 of 10 |
| **8** | `fig08_prott5_chemistry.png` | **A**: held-out R² for hydropathy, charge, volume under the residue-disjoint split. **B**: strongest single-dimension probes vs the n=28 threshold | `results/emb_probe.json`, `embedding_probe.json` | ProtT5 predicts unseen-residue hydropathy at R²=0.704, so W6 adds little information |
| **9** | `fig09_metric_rule_coil.png` | **A**: the coil scored on ΔΔG. **B**: the coil scored on ΔG (MAE) | FINDINGS §2.2, §6.3 | The metric rule made concrete: the same lever is "harmful" on the metric that cancels it and best-in-project on the metric it acts on |

### Figures that could not be generated, and why

- **Attenuation refutation (§7).** Would plot predicted a_p under label noise against measured
  a_p. The per-mutation uncertainty join lives in `data/ThermoMPNN/mega_test.csv` and was not
  re-derived here; the numbers are quoted from FINDINGS §3.3. Worth generating before
  submission — it is the figure that kills the most dangerous objection to the thesis.
- **ICC / seed-decomposition of b_p.** `ckpt_seed_geometry_all.json` holds the raw material but
  the 28 × 5 design is not laid out in a single artefact.
- **Catalogue distribution shift.** A two-panel length and experimental-method comparison would
  be easy from `catalogue_vs_bp.json`, but the section's conclusion is a null (r = +0.043), and
  a figure would give it more visual weight than the result deserves.

---

# 13. Limitations

Stated plainly, in descending order of how much they constrain the conclusions.

### 13.1 n = 28

The test set is 28 proteins. Every per-protein correlation in this thesis — exposure→a_p,
burial→b_p, and all 90 in the replication sweep — rests on 28 points. At that size **|r| < 0.374
is indistinguishable from zero at p = 0.05**, and a Bonferroni threshold over 90 tests is far
higher still. **Only one correlation in the entire project (exposure→a_p) clears the corrected
bar on both Pearson and Spearman.** Everything else is reported at the strength of the
uncorrected test plus sign-consistency across checkpoints, and should be read as suggestive.

**The obvious escape was closed.** The plan to test structural correlations on the ~100k
pre-training catalogue fails permanently: the catalogue has no sequence column, and joins
return 0/226 training ids, 0/21 test ids and 0/340 by protein id. There is no larger set to
replicate on within this project's data.

Relatedly, the per-mutation n values are large (28,314 mutations) but they are **not 28,314
independent observations of the quantity of interest**. The calibration parameters are
per-protein, so the effective n for every calibration claim is 28.

### 13.2 One seed for most arms

Nearly every lever arm — including all three `--slope_weight` arms and most factorial cells — is
**a single training run**. Seed-only variation is not negligible: over 5 replicate seeds, pooled
ΔΔG PCC has sd 0.0339 (range 0.5594–0.6386) and median a_p has sd 0.0515 (range 0.327–0.437).

**Consequences that must not be glossed:**

- The `--slope_weight` 1.0 gain (a_p 0.4975 → 0.5347, +0.0372) is **inside the seed band** and
  is therefore suggestive, not established.
- The weight-3.0 collapse (−0.1722) and the factor-D separation (~10× the band) are **well
  outside** it and are safe.
- Cells also differ in the epoch reached (5 to 14), so training length is a partial confound on
  every within-arm comparison.

Any claim in this thesis that rests on a difference smaller than ~0.03 in pooled PCC or ~0.05 in
a_p should be treated as unreplicated.

### 13.3 The 2K5H bug was found late

One test protein's reference row was a mutant background, shifting all 1,125 of its ΔΔG labels
by +3.0824 kcal/mol. It was found **after** most results had been computed.

- **It inflated the headline.** The oracle gain falls from +0.0988 to **+0.0702** on correction
  — a 29% reduction in the project's central effect.
- **It produced a second, independent false lead.** An apparent 26,315 double mutants — which
  would have been a genuine generalisation test — turned out to be 2,356 two-point rows, all
  belonging to 2K5H, all single mutations on the G11S background. **No true double mutants exist
  for any of the 28.** One malformed file generated two separate false leads, both plausible
  enough to have reached the thesis.
- **Nothing raised.** Both rows are legitimately labelled `wt`. Only the percentile control
  exposed it, and a permanent gate now counts distinct wild-type backgrounds per protein.
- **Not every number in the record has been recomputed post-fix.** The figures in this chapter
  are, and are marked corrected/uncorrected where both exist. Numbers inherited from earlier
  documents (std(b_p) = 1.5741, the "top-2 proteins carry 78%" claim, the ICC) **carry the bug**
  and should be re-derived before submission.

### 13.4 The oracle is not a method

The headline result is produced by subtracting from each protein a constant **fitted on that
protein's own test labels**. It licenses exactly one claim: *the ranking information is present;
what is lost is calibration.* It is an upper bound on what perfect offset knowledge would buy,
and **not** a performance claim for DeepEF.

The honest companion result is that **the bound is not reachable from features**: a LOPO ridge
on 22 structural features recovers **+0.0055 of the +0.1125** available (4.9%), with held-out R²
negative on 8 of 10 checkpoints and a permutation null significant on 0 of 10. Any use of the
oracle number without this sentence beside it overstates the work.

### 13.5 Zero variance on ligands, complexes and interfaces

All 28 test proteins are **single-chain, ligand-free, metal-free monomers of 43–72 aa** —
verified: `n_chains == 1`, `n_het_residues == 0`, `n_metal_residues == 0`,
`interchain_BSA == 0.0` for every one.

So the W11 ligand lever, the retired W9 metal lever, and any interface-area feature have
**exactly zero variance** on this test set. **This is "not measurable here", not "no effect".**
A lever whose driving feature is constant must be logged untestable — never scored and given a
zero, which would manufacture a false negative. Nothing in this thesis should be read as
evidence that ligand or interface information is unhelpful for stability prediction in general.

The domain is narrow in a second way: 43–72 residues, and 77.4% solution NMR structures against
8.6% in the pre-training catalogue. Generalisation to larger proteins or to crystal structures
is untested.

### 13.6 CB-only geometry

The dataset stores **four backbone atoms per residue (N, CA, C, CB)**. There are no side-chain
coordinates.

This is a hard ceiling on what the model can represent, and §4 is the measurement of its cost:
mutations to hydrophobic residues are compressed 1.57× relative to polar ones, with rank
accuracy degrading in lockstep (corr(slope, Spearman) = +0.9838). Side-chain packing is the
dominant term in burying a large hydrophobic, and CB-only geometry cannot see it.

**This limitation is also the thesis's strongest forward-looking result:** it identifies a
specific, testable architectural change (side-chain information) rather than a calibration
adjustment, and §4 shows why no affine correction can substitute for it.

A related architectural limitation: the GCN branch reads only the first 32 feature columns, so
every descriptor, burial and ligand block reaches the GAT branch alone. No lever tested here
influenced the GCN branch at all.

### 13.7 Incomplete factorial

12 of 48 cells are scored. No full main-effect table is claimed, and **no main effect for
factors A, B or C may be averaged across factor D**, whose effect is ~10× larger and would swamp
them. The D0/D1 separation is safe because it is enormous; the finer structure is not yet
measurable.

---

# 14. Abstract (200 words)

> Deep learning models of protein stability are usually judged by a single pooled correlation,
> which conflates two different abilities. We decompose the predictions of DeepEF, a graph
> neural network trained on the MegaScale dataset, over 28 held-out proteins and 28,314
> mutations, and show
> that within each protein the prediction is an affine function of the truth,
> `pred ≈ a_p·true + b_p`. The model ranks mutations well inside a protein (median r = 0.796)
> and poorly across proteins (pooled r = 0.565); the difference is almost entirely a
> protein-specific offset. Removing that offset with an oracle raises pooled correlation to
> 0.635, but a leave-one-protein-out corrector using 22 structural features recovers only 4.9%
> of that gain, with negative held-out R² on 8 of 10 checkpoints. The slope factors exactly as
> `a_p = r·s`, so a spread-matching loss moves s (0.449 to 0.724) but never r, capping any such
> lever. The residual ranking error is chemically specific: mutations to hydrophobic residues
> are compressed 1.57-fold, with rank accuracy falling in lockstep (r = +0.98), which no
> rescaling can repair and which we attribute to backbone-only geometry. We also state a metric
> rule that invalidated five internal measurements, our own model selection criterion among
> them.

**Word count: 200.**

---

## Provenance

| artefact | path |
|---|---|
| figure generator | `scripts/make_figs.py` |
| figures | `results/figures/fig01..fig09*.png`, `MANIFEST.json` |
| recomputed headline, per-CSV | `results/thesis_headline.json` |
| recomputed a_p = r·s per checkpoint | `results/thesis_rs.json` |
| residue-disjoint probe (re-verified) | `results/emb_probe.json` |
| complete working record | `results/FINDINGS.md` |

**Gate status.** `scripts/gate_g4_cpu.py` re-run after every change in this compilation:
**`baseline forward ok, dG finite PASS dG=-0.0030 width=1092`** — unmoved.
