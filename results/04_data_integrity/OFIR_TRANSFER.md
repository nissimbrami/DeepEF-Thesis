# OFIR_TRANSFER.md — transferable METHOD from Ofir Ezrielev's thesis, computed on DeepEF data

Source: `C:/Users/User/Downloads/Thesis Ofir Ezrielev.pdf`, read directly (56 pp).
All numbers below were computed on the cluster this session; scripts and artifacts named per section.
Statistical rule used throughout: at n=28, |r| < 0.392 is indistinguishable from zero at p=0.05
(n=18 -> 0.500; n=10 -> 0.632). Every claim states n and threshold.

Required gate re-run after all work: `scripts/gate_g4_cpu.py` -> `baseline ... PASS dG=-0.0030 width=1092`, G4-CPU: ALL PASS.

---

## HEADLINE

**Ofir's leave-one-AA-out offset diagnosis (p.34) replicates on our data at r=+0.9895 (n=28),
and his proposed remedy — MEASURE the offset from a handful of samples rather than PREDICT it —
lifts pooled dG PCC from 0.369 to 0.753 with 3 measured mutations per protein
(oracle 0.817), averaged over 11 checkpoints, with all three negative controls passing.**

This is the single most valuable thing in the thesis for us. Our LOPO b_p corrector failed
(0.6049 vs 0.5994 raw) because it tried to *regress* b_p on protein-level features. Ofir never
does that. He measures it.

---

## 1. ENRICHMENT — exact definition, and the first computation of it on DeepEF

### 1.1 The formula (thesis p.21, section 2.3, third metric)

> "The third metric is Enrichment. It measures whether the model succeeds in predicting which
> samples are in the highest (or lowest, depending on the context) X% true value group."

Enrichment = |argmax_{floor(X%N)}(y) INTERSECT argmax_{floor(X%N)}(yhat)| / ( N * (floor(X%N)/N)^2 )

with N the size of the target set. Equivalently **enrichment = hit_rate / chance_rate**, where
chance = k/N. Chance = 1.0; the theoretical maximum is N/k (= 20 at X=5%, 6.67 at X=15%).

His argument for why it matters (p.17 section 1.4.4, restated p.31-32 section 4):

> "unless performance is very good ... these values are typically considered in the context of
> performance estimation of other methods. ... we are not aware of similar studies, and any
> accepted benchmark, with which we can compare our results. Contrarily, enrichment, the third
> metric, has an innate meaning, the increased probability of picking desired entities
> (e.g., stable proteins) using the prediction method, compared with the probability of picking
> them by chance." (p.17)

> "The meaning of the more common measures, RMSE and correlation ... is only relative. ...
> Enrichment, on the other hand, has an objective meaning, the relative prospective of success
> using the predictions, compared with a random sample of the available chemical space." (p.32)

**This directly applies to us: we have no external benchmark on these 28 proteins either.**

### 1.2 Implementation validated against its own null

`scripts/enrich.py`, `scripts/enrich2.py`, `scripts/enrich3.py`.
Random-shuffle null over 200 permutations x 28 proteins returned
**E5top = 1.006, E5bot = 0.998** against a theoretical 1.000. The estimator is correct.

### 1.3 RESULT — per-protein enrichment (the design use case: rank mutations within one protein)

10 eval CSVs x 28 proteins, non-WT rows (28,285 of 28,314), mean over protein-checkpoint cells:

| X%  | end | k    | hit rate | chance | **enrichment** | screened per hit |
|-----|-----|------|----------|--------|----------------|------------------|
| 5%  | bot (most destabilizing) | 50.0 | 51.4% | 5.0% | **10.37** | 1.9 |
| 5%  | top (most stabilizing)   | 50.0 | 23.2% | 5.0% | **4.69**  | 4.3 |
| 10% | bot | 100.5 | 61.9% | 10.0% | 6.22 | 1.6 |
| 10% | top | 100.5 | 31.2% | 10.0% | 3.13 | 3.2 |
| 15% | bot | 151.0 | 66.4% | 14.9% | 4.45 | 1.5 |
| 15% | top | 151.0 | 37.1% | 14.9% | 2.48 | 2.7 |

97.1% of protein-checkpoint cells beat chance on E5top; 100% on E5bot; **zero total misses.**

**Budget translation** (what an experimentalist actually asks): to obtain 10 true top-5%
stabilizing mutants you synthesize **43 model-ranked candidates vs 202 at random — 4.7x saving.**
For the destabilizing tail the saving is 10.4x.

**The model is more than twice as good at finding destabilizing mutations as stabilizing ones**
(10.37 vs 4.69). Nobody here had measured this. For a stability-design use case, which needs the
*stabilizing* tail, that asymmetry is the headline weakness — and it is invisible in PCC.

### 1.4 RESULT — enrichment is NOT redundant with PCC, and PCC is the wrong selection metric

Across the 28 proteins (n=28, threshold 0.392):

| target | vs a_p | vs b_p | vs per-protein PCC | vs n_mut | vs ddG skew |
|--------|--------|--------|--------------------|----------|-------------|
| E5top  | **+0.490** | -0.053 | +0.431 | -0.397 | -0.375 |
| E5bot  | +0.267 | +0.163 | **+0.710** | -0.196 | **-0.567** |
| E15top | +0.269 | +0.138 | **+0.461** | **-0.488** | +0.373 |
| E15bot | **+0.569** | +0.070 | **+0.938** | -0.280 | -0.372 |

Two things fall out:

- **a_p -> E5top r=+0.490 (n=28, >0.392).** The slope lever drives top-tail design utility.
  This is a **fifth lever-relevant hit for a_p**, alongside the coil, BSA, W5 burial and ligands.
  b_p is flat everywhere (|r| <= 0.163), exactly as the metric rule predicts — within a protein
  b_p is a constant added to every mutant, so ddG ranks are invariant to it. That flat column is
  a correctness check on the pipeline, not a finding.
- **ddG skew -> E5bot r=-0.567.** This is precisely Ofir's own explanation of why CP2's top-5%
  and bottom-5% enrichment differ so wildly (2.333 vs 10.092, Table 3.3 p.27): "There is a peak
  in the distribution near the top values, without a clear tail ... This contrasts with the tail
  of few high values in the PUMA BH3 dataset" (pp.32-33). His distribution-shape diagnosis
  reproduces on our data as a quantified correlation.

**And the decisive one — checkpoint selection (n=10 checkpoints, threshold 0.632):**

    corr(pooled ddG PCC, mean per-protein E5top) = +0.0021
    corr(pooled ddG PCC, mean per-protein E5bot) = +0.2852

    best checkpoint by PCC:   abl_sigma_seed2_e10
    best checkpoint by E5top: abl_sigma_seed3_e13   <- a DIFFERENT checkpoint
    best checkpoint by E5bot: abl_sigma_seed2_e10

**PCC carries essentially zero information about top-tail design utility across checkpoints.**
Spread is real and comparable in both (E5top 3.91-5.02, 1.28x; PCC 0.5623-0.6574). We have been
selecting checkpoints on a metric uncorrelated with the use case we care about. Recommendation:
report E5top/E5bot alongside PCC in `validation/score_runs.py` and in the thesis results table.

### 1.5 RESULT — enrichment exposes the offset damage that ddG hides

Pooled over all 28 proteins mixed (the "screen a library across proteins" setting), mean of 10 ckpts:

| pct | dG raw top | dG raw bot | offset-removed top | offset-removed bot | full oracle top | full oracle bot |
|-----|-----------|-----------|--------------------|--------------------|-----------------|-----------------|
| 5%  | 5.116 | 4.059 | 4.953 | **9.554** | 7.941 | 8.787 |
| 15% | 2.716 | 2.242 | 2.742 | **3.907** | 4.236 | 4.215 |

**Removing b_p alone multiplies pooled dG bottom-5% enrichment by 2.35x (4.06 -> 9.55).**
Artifacts: `results/enrichment_pooled.csv`, `enrichment_perprotein.csv`, `enrichment_offset.csv`,
`enrichment_bp.csv`, `enrichment_hitrate.csv`.

---

## 2. SEQUENCE ABLATIONS — permutation and collapse, run on a frozen DeepEF checkpoint

### 2.1 What he did

**Permutation (p.24, section 3.1.1):**
> "The results presented in Table 3.1 clearly indicate that the machine learning models learned
> something. But what did they learn? Only the consequences of the changing molecular composition
> or also the importance of the sequence, the sequential order of the AAs."

Table 3.2 (p.24), CNN on X4pSX4K: correlation 0.953 intact -> 0.870-0.887 permuted (91-93%
retained); enrichment 5.575 -> 4.631-4.899 (83-88% retained). His verdict (p.34):
> "Random sequence permutations of the X4pSX4K training set ... reduce the prediction performance,
> by only small factor. This suggests that much of the learned information, about this set of
> peptide variants, is simply composition dependent"

**Collapse (p.25, section 3.1.2):** replace the per-position feature matrix by its mean along the
sequence axis — "we obtain a mean AA composition". Lasso stayed flat; SVR and LGBM *improved*.
That is his corroborating evidence for the composition claim.

### 2.2 Our analogue, and why it is a fair one

DeepEF is a graph energy model, not a 1D CNN, so "permute the sequence" has to be defined
carefully. `scripts/seqabl.py` patches `train_utils.get_graph` / `get_unfolded_graph` in memory
(no file edited, nothing pushed) and permutes the **residue axis of the identity channels
(one_hot + prott5) while holding coords and mask fixed**. Composition is preserved exactly;
the sequence<->structure correspondence is destroyed. One permutation is drawn per protein and
reused for every mutant of that protein, so WT and mutant see the *same* scramble — otherwise
ddG would be pure permutation noise rather than a measurement.

Run as SLURM CPU job 21085641 (no GPU), 7 modes, `results/seqabl/*.csv`, 28,314 rows each.
**Every mode printed `PATCH_CALLS 28314`**, i.e. the patch demonstrably fired on every graph
build — the guard against the signature failure of a feature that is never read.

**Reproduction gate.** Mode `none` vs the shipped `abl_calib_ctrl_repro2_e14.csv`, compared by
(protein, within-protein index) key, not by row position (the dataloader shuffles proteins):
`deltaG` and `ddG` match at **0.000e+00**. `pred_deltaG` differs because the shipped CSV has the
`affine.json` sidecar (a=0.6975855, b=3.9830644) applied and my runner does not. Recovering the
map empirically from my raw predictions gives **a=0.697576, b=3.983037, corr=0.99999878** — the
sidecar to 5 decimals. Residual 1.5e-2 is float32 accumulation. Since the affine is a pure
monotone rescale it cannot change any enrichment or correlation below. Gate PASSES.
(`scripts/gate_none.py`, `scripts/gate_none2.py`.)

### 2.3 RESULT

| mode | pooled ddG PCC | per-prot ddG PCC | dG PCC | ddG RMSE | E5top | E5bot | median a_p | sd(pred_ddG) |
|------|---------------|------------------|--------|----------|-------|-------|-----------|--------------|
| **none** (intact)   | 0.5912 | 0.7310 | 0.4499 | 1.0465 | 4.99 | 10.51 | 0.714 | 1.068 |
| perm (oh+emb) s42   | 0.6047 | 0.6668 | 0.3953 | 1.0135 | 4.77 | 9.69 | 0.554 | 0.861 |
| perm (oh+emb) s7    | 0.5945 | 0.6673 | 0.3623 | 1.0166 | 4.74 | 9.34 | 0.593 | 0.848 |
| perm emb only       | 0.6088 | 0.6709 | 0.4135 | 1.0085 | 4.83 | 9.68 | 0.566 | 0.883 |
| perm one_hot only   | 0.5997 | 0.7266 | 0.4568 | 1.0328 | 4.73 | 10.56 | 0.719 | 1.062 |
| **collapse (emb -> mean)** | **0.4085** | **0.5297** | 0.3864 | **1.9092** | **3.23** | **7.24** | 0.714 | 2.053 |
| collapse one_hot -> mean | 0.6025 | 0.7265 | 0.4561 | 1.0353 | 4.82 | 10.47 | 0.730 | 1.083 |

Retained fraction vs intact: permutation keeps **101-103% of pooled ddG PCC, 95-97% of E5top,
89-100% of E5bot, and RMSE actually improves 1-4%.**

**Our model survives sequence permutation even more completely than Ofir's did.** His CNN lost
9% of correlation and 17% of enrichment; ours loses nothing measurable on pooled PCC and ~4% on
E5top, across two independent permutation seeds. Per-protein ddG PCC is the one metric that does
move (0.731 -> 0.667, 91% retained) — the same order as his 91%.

Reading the columns against each other localizes the finding precisely:

- **Permuting one_hot alone is nearly free** (0.7266 vs 0.7310 per-protein PCC; E5bot 10.56 vs
  10.51 — a *rise*). So is collapsing one_hot to mean composition (0.7265). The 20-dim identity
  channel is contributing almost nothing positional.
- **Permuting prott5 alone reproduces nearly the entire permutation effect** (0.6709, vs 0.6668
  for permuting both). The positional information that exists lives in the ProtT5 embedding.
- **Collapse of prott5 is the only intervention that genuinely breaks the model**: per-protein PCC
  0.7310 -> 0.5297, E5top 4.99 -> 3.23, E5bot 10.51 -> 7.24, and RMSE +82% with sd(pred_ddG)
  nearly doubling to 2.05. Destroying the per-residue embedding hurts; merely *reordering* it
  does not.

**Interpretation.** The model reads *which* residues are present and what the fold's geometry is,
but is close to indifferent to *which residue sits at which position*. That is Ofir's
composition-dependence conclusion, reproduced independently on a different architecture, a
different data type (dG/ddG vs retention time) and a different scale (28,285 vs ~120 test points).
It is also consistent with our measured a_p ~ 0.499 median: a model that cannot fully localize
mutations to their structural context will systematically under-respond to them.

Note the direction of the a_p column: permutation drops median a_p 0.714 -> 0.554-0.593, while
collapse leaves it at 0.714. Slope loss tracks *reordering*, magnitude loss tracks *destruction* —
they are separable failure modes and only the first is about sequence position.

**Caveat, stated plainly.** This is a permutation of the *input at inference* on a frozen
checkpoint. Ofir also retrained on permuted data ("Permuted Model"), which we did not — that
needs GPU training and was out of scope. Inference-time permutation is the weaker of his two
tests, so the honest claim is "the trained model does not use sequence order at inference",
not "sequence order is unlearnable here".

---

## 3. THE LEAVE-ONE-AA-OUT OFFSET DIAGNOSIS — our b_p problem, found independently

### 3.1 What his Table 3.4 (pp.29-30) actually shows

18 held-out amino acids. Correlation stays in a narrow band **0.858-0.948** while RMSE swings
**7.047-19.316 (2.74x)**. The table's last column is "Absolute Delta Train and Test Means".
His observation (p.34):

> "An interesting finding in this dataset is that the RMSE of prediction for proteins with the new
> AA is similar to the difference in means of the two distributions – the proteins with the new AA,
> and the proteins that were used in training."

I recomputed his table (`loao_check.py`, local, n=18, threshold 0.500):

    corr(RMSE, |Delta mean|)        = +0.9883
    corr(correlation, |Delta mean|) = -0.5078
    corr(enrichment, |Delta mean|)  = -0.5427
    RMSE range 7.05-19.32 = 2.74x
    after removing the offset in quadrature, sqrt(RMSE^2 - dMean^2): 6.82-11.67 = 1.71x
    corr(de-offset RMSE, |Delta mean|) = +0.2337   (now below the 0.500 threshold)

So his qualitative remark is a near-deterministic identity, and removing the mean gap collapses
most of the spread. **His RMSE variation is almost entirely an offset phenomenon, exactly like ours.**
Note his correlation and enrichment are only mildly degraded (-0.51, -0.54, both at the n=18
threshold) — i.e. the offset wrecks RMSE while largely sparing rank metrics. Same structure as ours.

### 3.2 The same identity on OUR data — it replicates almost exactly

`scripts/fewshot.py`, per-protein on dG, n=28 proteins, threshold 0.392:

    corr(RMSE_dG, |mean gap|) = +0.9895        [Ofir, n=18: +0.9883]
    RMSE_dG range 0.345-3.135 (9.08x)
    after removing offset:     0.333-1.002 (3.01x)
    the offset accounts for 80.3% of total dG MSE
    corr(residual RMSE, |mean gap|) = +0.5011

**+0.9895 vs his +0.9883.** Two different theses, different architectures, different targets,
same identity. This is strong external corroboration that our b_p is not a DeepEF quirk but a
generic property of this class of model, and it belongs in the thesis as such.

### 3.3 How he handled it — and why our corrector failed

His prescription, p.34, immediately after the identity:

> "One may speculate that with a small number of proteins with the new AA that are experimentally
> evaluated (that may not significantly change the model's predictions when used in training),
> the RMSE can be used to calibrate the predictions of the protein."

**He does not predict the offset from features. He measures it from a few samples.** Our LOPO
corrector regressed b_p on protein-level descriptors and failed out of sample (0.6049 vs 0.5994
raw, oracle 0.7119). That failure is now explained: b_p carries ICC 0.898, i.e. it is a stable
per-protein property, but it is evidently not a *function of the protein-level features we have*.
Measuring it sidesteps the question entirely.

### 3.4 RESULT — few-shot measured calibration, with controls

For each held-out protein, draw k of its mutations, set b_p = mean(pred_dG - true_dG) over those
k, subtract it from the rest, and **score only the rest** (the k calibration points are excluded).
`scripts/fewshot.py` (60 draws) and `scripts/fewshot_vfy.py` (30 draws, 11 checkpoints, + controls).

Mean over 11 checkpoints, pooled dG PCC:

| k (measured mutations per protein) | pooled dG PCC | pooled dG E5bot |
|-----------------------------------|---------------|-----------------|
| RAW (k=0)                         | 0.369 / 0.393 | 4.06 |
| **k=1**                           | **0.680**     | 6.95 |
| **k=3**                           | **0.753**     | 8.11 |
| k=5                               | 0.794         | 8.30 |
| k=10                              | 0.796 / 0.813 | 8.64 |
| k=20                              | 0.822         | 8.82 |
| k=100                             | 0.829         | 8.92 |
| ORACLE (all mutations)            | 0.817 / 0.831 | 8.92 |

(two figures where the 60-draw and 30-draw runs are both reported)

**One single measured mutation per protein recovers 0.393 -> 0.686. Three recover 0.772.
Ten reach 0.813 against an oracle of 0.831 — 98% of the achievable gain from 10 measurements.**
Consistent across all 11 checkpoints (per-checkpoint table in `results/fewshot_calib.csv`).

**Controls (mean over 11 checkpoints) — all three pass:**

| quantity | value | required | verdict |
|----------|-------|----------|---------|
| raw | 0.369 | — | — |
| k=3 measured | **0.753** | > raw | PASS |
| random constant, same magnitude | 0.266 | <= raw | PASS (hurts) |
| swapped: protein q's offset on protein p | 0.245 | << raw | PASS (hurts badly) |
| ddG PCC under k=3 | 0.557 | == untouched ddG PCC | PASS (untouched = 0.557) |

The last row is the metric-rule check: b_p cancels in ddG, so few-shot offset correction **must**
be a no-op on ddG. It is, to three decimals. A gain there would have meant a leak.

Because the gain is on dG and not ddG, this is a **reference-state / whole-protein lever**, scored
correctly per the metric rule.

### 3.5 What this means practically

This is not a modelling change and needs no retraining. It is a deployment protocol:
**to use DeepEF on a new protein, measure 3-10 of its mutations first and subtract the mean
residual.** That converts a pooled dG PCC of ~0.39 into ~0.75-0.81. It also honestly reframes the
b_p story for the thesis: the offset is not fixable from features (LOPO failed, and we now have
Ofir's independent replication of the identity to show the offset is real and dominant), but it
is trivially fixable with a handful of measurements, and 3 measurements per protein is a
negligible experimental ask next to the 1,000-mutation libraries these proteins already have.

---

## 4. OTHER TRANSFERABLE METHOD

**4.1 Batch normalization as a deliberate WT-signal suppressor (p.17, section 1.4.3).** Not a generic
training-stability remark — he uses it specifically against the offset problem:

> "When training models for the PUMA BH3 and CP2 datasets, batch normalization has an additional
> goal. There is a very high dependence between samples in these datasets, with most feature
> values being shared across most of a training batch's samples. Batch normalization brings these
> values closer to 0 compared to values that are not shared among many samples. This reduces the
> effect of values that are similar to the protein WT compared to what we aim to learn – the
> effect of the mutation."

Our setting has exactly that structure: ~1,010 mutants per protein sharing one backbone, so
nearly every graph feature is constant within a protein's batch. Whitening per batch would
suppress the shared WT component and amplify the mutation-specific one — an architectural attack
on b_p (and possibly on a_p ~ 0.499) that we have not tried. This is a concrete, cheap,
GPU-training-scale experiment: batch-norm the graph features with protein-homogeneous batches.
Flagged as untested — no number attached.

**4.2 Discarding degenerate runs by an explicit criterion (p.30, footnotes i-iv).** Four of his
LOAO cells had a repeat where "the model reached a local minimum and outputs identical values for
all the test set peptides"; these were discarded and the discard is documented per-cell. We have
361 checkpoints and no such stated criterion. A constant-prediction detector (sd(pred_ddG) below
a threshold) run over all of them is cheap and would tell us whether any reported cell is
contaminated by a collapsed run. Our ablation table already shows the diagnostic value of that
column: `collapse` doubled sd(pred_ddG) to 2.05 while `none` sat at 1.07.

**4.3 Report metrics as mean +/- SD over repeats, never single numbers.** Every table in the
thesis (3.1-3.4) is 30 repeats with a documented seed list in Appendix 1, and the SD is what makes
his enrichment differences interpretable — e.g. Fluorophenylalanine's enrichment 5.451 +/- **2.687**
is visibly not distinguishable from the others, whereas 4-nitrophenylalanine's 4.213 +/- 0.318 is.
Our per-protein enrichment numbers should carry the same across-checkpoint SD.

**4.4 Explicit anti-benchmark framing.** He states plainly that no benchmark exists for his task
and lets that justify the choice of an absolute metric. Our 28 proteins have the same problem.
Adopting his framing — "PCC is relative, enrichment is absolute" — gives the thesis a defensible
answer to "is 0.59 good?", which is otherwise unanswerable.

**4.5 What is NOT transferable (checked, negative).** His feature funnel (1826 Mordred -> 1280 ->
654 by "at least 40 unique values", p.20, Fig 2.1) is a small-data device for representing
non-canonical amino acids that have no LM embedding. We have ProtT5 and only canonical residues.
He concedes the point himself (p.31): physicochemical properties "are implicitly represented in
these datasets ... and accordingly, implicitly represented in the embeddings". Our section 2 result
sharpens this into a measurement: collapsing one_hot to mean composition costs us nothing
(0.7265 vs 0.7310), so the identity channel is already near-redundant with the embedding. Adding
more hand-built per-residue chemistry to that channel should be expected to do nothing, and that
expectation is now backed by a number rather than an argument.

---

## ARTIFACTS

Cluster, `/home/nissimb/DeepPEF/`:
- `scripts/enrich.py`, `scripts/enrich2.py`, `scripts/enrich3.py` — enrichment metric + analyses
- `scripts/seqabl.py`, `scripts/seqabl_job.sh`, `scripts/seqabl_an.py` — permutation/collapse ablation
- `scripts/gate_none.py`, `scripts/gate_none2.py` — reproduction gate for the ablation runner
- `scripts/fewshot.py`, `scripts/fewshot_vfy.py` — Ofir identity + few-shot calibration + controls
- `results/enrichment_{pooled,perprotein,offset,bp,hitrate}.csv`
- `results/seqabl/abl_{none,perm,perm_s7,perm_emb,perm_oh,collapse,collapse_oh}_*.csv` (7 x 28,314 rows)
- `results/seqabl_summary.csv`, `results/ofir_identity.csv`, `results/fewshot_calib.csv`

Local: `loao_check.py` (recomputation of his Table 3.4), `results/OFIR_TRANSFER.md` (this file).

SLURM CPU jobs (no GPU): 21085641 (ablation), 21085746 (few-shot), 21085798 (controls).
Nothing was edited in tracked source, nothing pushed, no GPU job submitted, nothing cancelled.
`scripts/gate_g4_cpu.py` re-run after all work: `baseline ... PASS dG=-0.0030 width=1092`, ALL PASS.
