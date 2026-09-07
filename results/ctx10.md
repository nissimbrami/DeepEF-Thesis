
---

# CHECKPOINT 10 — 2026-09-07 — OFIR RECONCILED, CORRECTOR FAILS, NO-OP HOLES CLOSED

Three workflows completed before a session limit. Their verifiers did NOT run, so everything below
is WORKER-REPORTED, NOT ADVERSARIALLY VERIFIED. Re-verify before quoting in the thesis.

## 1. OFIR'S THESIS — READ IN FULL (56 pages, BGU CS, Keasar, Dec 2023)

`results/OFIR_THESIS_NOTES.md`. **Three things in our build are WRONG:**

**(a) PCA-16 IS NOT HIS METHOD.** He does NO dimensionality reduction. "PCA"/"principal
component"/"SVD" appear NOWHERE in the thesis. His 654 features enter a CNN as 654 channels.
Our `--pca 16` is OUR invention and must be relabelled as ours, not attributed to him.

**(b) `ignore_3D=True` CONTRADICTS his 1826.** 1613 2D + 213 3D = 1826. Quoting 1826 is only
coherent if 3D was requested — which also explains his 1826->1280 missing-value drop, since 3D
descriptors return NaN without an embedded conformer. With `ignore_3D=True` the ceiling is 1613
and his drop is unreproducible. Our docstring says 1826 while our provenance says
"Mordred(ignore_3D)" — these contradict each other.

**(c) 15-of-20 IS NOT A FAITHFUL SCALING of 40-of-58.** 40/58 = 0.690, so proportional scaling
gives 13-14, not 15. Our help text calls 15 "three quarters"; 40/58 is roughly two thirds. Deeper:
**he gives NO justification for 40 at all** — it reads as tuned to land a workable feature count.
So there is no ratio to preserve. Likely inert on 20 canonical residues (any threshold 13-17
selects nearly the same columns) but the docstring must be fixed either way.

**HIS CENTRAL CLAIM AND ITS EVIDENCE (Table 3.3):** train on canonical single-point mutants, test
on non-canonical. PUMA BH3: 646 train -> 714 ncAA test, RMSE 0.975, **Pearson r = 0.642**.
CP2: 228 -> 240, RMSE 0.859, **r = 0.670**. 30 seeds varying only CNN init, split fixed.
**NO baseline of ANY kind** — no one-hot, no LM, no shuffled-descriptor, no per-position-mean.
He calls it "a proof-of-concept" and notes "there was no benchmark to compare RMSE results to".
He is candid that Fig 3.4 shows "a near-constant prediction line at 0" — part of that r is
partial collapse to a constant.

**HE ADDRESSES THE LM OBJECTION — AND CONCEDES IT** (p.31, verbatim): "Physicochemical properties
of AAs are implicitly represented in these datasets ... and accordingly, implicitly represented in
the embeddings. Thus, ncAAs, which do not occur in these databases, are challenging to represent."
**His argument is COVERAGE ONLY, not added information.** He runs ZERO experiments comparing
descriptors against an LM embedding or one-hot.

**CONSEQUENCE FOR US, and it is decisive:** our 28 test proteins are canonical, so every residue
HAS a real ProtT5 vector and the coverage gap he exploits DOES NOT EXIST in our setting. **By his
own sentence, the physicochemical content is already implicit in our 1024-dim ProtT5. The thesis
does NOT support adding a Mordred block on top of ProtT5+one-hot for canonical monomers.**

**HE INDEPENDENTLY FOUND OUR OFFSET PROBLEM.** Leave-one-AA-out (Table 3.4, 18 ncAAs x 30 seeds):
r stays flat 0.858-0.948 while RMSE swings 7.05-19.32, and he shows **RMSE tracks the train/test
MEAN GAP** (Ornithine 19.316 vs gap 18.030). That is b_p, found independently, with essentially
our offset-removal fix proposed.

**WHAT TO STEAL:** leave-one-AMINO-ACID-out. We have leave-one-protein-out but nothing holding out
a residue TYPE. Directly portable, needs no ncAA data, and asks whether the model learned residue
chemistry or residue identity. Also: enrichment as a metric, and the train/test mean-gap diagnostic.

**TRAP:** 2D Mordred CANNOT distinguish D from L — every 2D descriptor is identical for both, so
his effective alphabet is well under 58 and L/D pairs contribute ONE unique value, not two.

## 2. THE b_p CORRECTOR FAILS — a real, publishable negative

`results/OFFSET_CORRECTOR.md`. Mean over all 10 eval CSVs:

| | pooled ddG PCC |
|---|---|
| raw | 0.5994 |
| mean-offset baseline | 0.5940 |
| **LOPO feature-predicted offset** | **0.6049** |
| oracle offset removed | 0.7119 |

**95% of the oracle gain does not survive held-out prediction.** Held-out R^2 of b_p is NEGATIVE
on 8/10 CSVs (mean -0.123) — worse than predicting the training mean. 0 of 22 features reach
positive R^2 alone. Permutation null: p in [0.295,0.758], significant on 0/10. With ridge alpha
FIXED the gain goes NEGATIVE (-0.0016), so the +0.0055 was alpha-search variance, not skill.

**THE MECHANISM, which is the useful part:** the oracle gain is NOT broad learnable
miscalibration. **The top-2 |b_p| proteins carry 78% of it** (60-88% across all 10 CSVs).
Correcting only 2K5H+2KVS gives 0.591->0.687 (80% of the gain); correcting the other 26 gives
0.615 (20%). Neither is a structural outlier (max |z| ~2.0 over 22 features), so **there is no
signature to regress on**. And 2KVS is not an offset case at all: a_p=0.094 vs median 0.4975,
per-protein PCC 0.160 vs ~0.80 — the model simply fails there and b_p absorbs the failure.

**METRIC CORRECTION THAT MATTERS:** Pearson is SHIFT-INVARIANT, so subtracting a CONSTANT changes
pooled PCC by exactly zero (verified: -1.0/0.0/mean/+1.0 all give 0.590991). **Therefore the null a
per-protein corrector must beat is the RAW number, not the mean baseline.** Scoring against the
mean baseline would have manufactured a fake +0.011 lift.

**Also corrects a units confusion:** std(b_p)=1.5741 is the dG-space WT error. The ddG-space
intercept the oracle actually subtracts has std **0.2273**. The ddG intercept is the correct
corrector target.

## 3. THREE SILENT-NO-OP HOLES CLOSED — `scripts/gate_noop.py`, 30 checks ALL PASS

**(A) PEMGraphTransformer dropped inserted blocks silently — MEASURED, not assumed.** A gate builds
`flat_x` with 726 sentinel columns at offset 48, runs the real reassembly
`cat([:16],[16:48],[-1044:-20],[-20:])`, and the output is byte-identical to the no-descriptor
case: **0 of 726 sentinel columns reach the node vector, and nothing raises.** Guard now raises in
`__init__` (cheap) for ALL FIVE affected levers — aa_descriptors, burial, metal, struct_quality,
ligand — not just descriptors.

**(B) `--ligand_nodes` could not fire in training.** Wired `Trainer._set_ligand_context(batch)`
before the first `get_graph` in all three entry points; `ligand_features()` now RAISES when a table
is loaded but context was never set. **`gate_ligand.py` D.14 had CODIFIED the bug** — it asserted
"unset context -> exactly zero". Replaced.

**(C) `gate_open_alphabet` tested a /tmp fixture, not the real table. THE REAL FINDING:** the
committed 25-row table is **K=756 vs canonical K=726**, with **27 canonical columns DROPPED and 57
ADDED**, and the canonical 20 rows are **NOT byte-identical**. It is not an open alphabet over our
matrix — **it is a different descriptor matrix wearing the same name.** The loader now refuses it.

**(D) A FOURTH HOLE, unassigned:** `gate_open_alphabet.py` hard-coded the retired mode
`'curated12'`, so it raised in its first section and **had stopped testing anything at all.**

## STATUS

G4 re-verified after every change: `dG=-0.0030 width=1092`.
NOT verified adversarially (session limit killed all verifiers): Ofir notes, corrector, no-op
gates, severing, identifiability, slope_origin, W8. **Re-run the verifiers before the write-up.**
