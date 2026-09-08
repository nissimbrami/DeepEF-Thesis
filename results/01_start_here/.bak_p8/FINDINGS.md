# DeepEF — COMPLETE FINDINGS OF RECORD
### M.Sc. thesis, Nissim Brami, supervised by Prof. Chen Keasar, BGU
### State as of 2026-09-08. Every number computed or re-derived by the main session.

**Reading rule — three qualifiers on every number: pooled-or-per-protein / dG-or-ddG / which split.**
Agent-reported figures that did not reproduce are listed in §14 rather than quietly dropped.

---

# PART I — THE PROBLEM

## 1.1 What DeepEF is

A graph neural network over protein structure. It predicts a folding free energy as a difference of
two forward passes over the same network:

    dG = E_unfolded - E_folded
    ddG = dG_mutant - dG_wild-type

Benchmark: **MegaScale** (Tsuboyama et al. 2023), ~776,000 measured stabilities on small domains,
40-72 residues, by cDNA-display proteolysis. **28 held-out test proteins**, 28,314 test mutations.

## 1.2 The central finding — the calibration decomposition

Per protein, the prediction is an affine function of the truth:

    pred ~= a_p * true + b_p

| quantity | value | what it is |
|---|---|---|
| **std(b_p)** | **1.5741** | the per-protein OFFSET, a dG-space wild-type error |
| **a_p median** | **0.4990** | the per-protein SLOPE — the model compresses ddG by half |
| **per-protein ddG PCC** | **0.798** | ranking quality WITHIN a protein |
| **pooled ddG PCC** | **~0.59** | ranking quality ACROSS all proteins together |
| corr(a_p, per-protein PCC) | **+0.5714** | the slope is the strongest per-protein quality predictor |
| ICC(b_p) | **0.898** | b_p is protein-attributable, not seed noise |

**The model ranks well within a protein and is miscalibrated between proteins.** The gap between
0.798 and 0.59 is the entire thesis.

## 1.3 The corrected headline

Removing each protein's own offset (an oracle, see the caveat) lifts pooled ddG PCC:

| | pooled | offset-removal oracle | gain |
|---|---|---|---|
| **corrected, 22 eval CSVs** | **0.4899** | **0.6443** | **+0.1544** |
| previously reported | 0.5018 | 0.7391 | +0.2373 |

**The oracle is 0.644 — NOT 0.739, and NOT the older 0.70-0.72 figure.** The effect is real and
substantial; its size was inflated 35% by the 2K5H data bug (§12.1).

**THE ORACLE IS AN ORACLE.** It subtracts a constant fitted on TEST labels. It licenses exactly one
claim — *the ranking information is present, what is lost is calibration* — and nothing stronger.
A feature-based corrector FAILED out of sample (§5.7).

## 1.4 The affine oracle (0.77-0.81) is not reproducible

The earlier claim that removing BOTH offset and slope reaches 0.77-0.81 does not hold. Seven of the
original checkpoints give 0.646-0.774 (one in range); five of ours give 0.488-0.658 (none).
**The mathematics explains it:** the affine oracle DIVIDES by the slope, and when a_p = 0.079 that
division amplifies noise ~13x. An estimator that is not monotone is not a bound. Offset removal
survives because subtracting a constant cannot amplify anything.

---

# PART II — THE RULE THAT ORGANISES EVERYTHING

## 2.1 The metric rule

**ddG cancels anything identical between wild type and mutant.** A point mutation changes neither
chain length nor backbone positions, so every purely geometric or whole-protein property cancels
EXACTLY in `pred_mut - pred_wt`. Therefore:

> **A lever acting on the reference state or on absolute stability MUST be scored on dG or on b_p.
> a_p is a WITHIN-protein slope and does NOT cancel, so it IS legitimately ddG-measurable.**

## 2.2 It has caught FIVE levers

1. **The Flory coil.** Scored r=1.175 on ddG — apparently harmful. On dG: **MAE 4.9650 -> 3.9521**,
   our single best b_p lever. The coil replaces unfolded distances with `d(i,j) = b*|i-j|^nu`,
   a function of sequence separation ONLY, which is bit-identical between WT and mutant and cancels
   exactly. **We were measuring the cancellation, not the lever.**
2. **BSA.** Declared dead on r=-0.001 — against the PRE-TRAINING DECOY LOSS, a metric already known
   not to predict ddG, computed on a column with **4 non-null rows out of 100,246**. Void twice over.
3. **W5 burial.** Its gate passes 18/18 and every assertion is mechanical (shapes, byte-identity,
   hydropathy ordering); not one scores performance. Its own help text says burial is zero unfolded
   and the folded-minus-unfolded delta IS the hydrophobic driving force — so it acts on dG — yet it
   was scheduled against pooled ddG.
4. **Ligands.** A folded-state stabilisation term, identical in WT and mutant for any mutation away
   from the site, so it cancels in ddG by construction.
5. **Our own model-selection metric.** `validate()` returns pooled dG PCC, which is scale-invariant
   and therefore **provably cannot see a change in predicted spread** — exactly what
   `--slope_weight` does. 40 queued runs were unfalsifiable until per-protein a_p logging was added.

## 2.3 The second rule, learned the hard way

**Always compute the degenerate baseline.** `--dg_length_norm n` looked like a 31% improvement in
std(b_p) and was the model predicting nothing (§5.3). And: **a correlation without its n is not a
result** — the "1/sqrt(N) artifact" was quoted at -0.9938 and measured at -0.29 (§14).

---

# PART III — WHAT WE LEARNED ABOUT a_p (THE SLOPE)

## 3.1 a_p = r * s — the slope is TWO numbers

For any least-squares fit, `a_p = r * s` where r is the correlation and `s = std(pred)/std(true)`.
Over 22 checkpoints:

    a_p = 0.364    r = 0.653    s = 0.504

| source of the gap to a_p = 1 | size | fixable by --slope_weight? |
|---|---|---|
| ranking error (1-r) | 0.347 | **NO** |
| spread compression (1-s) | 0.496 | **YES** |

**Ceiling: driving s -> 1 pins a_p to r.** The lever cannot reach 1.0 by construction, and the
write-up must say so.

## 3.2 The three slope arms confirm the mechanism

| --slope_weight | s | a_p | r |
|---|---|---|---|
| control (no term) | 0.635 | 0.499 | 0.798 |
| 0.3 | 0.573 | 0.415 | 0.801 |
| **1.0** | **0.756** | **0.562** | 0.809 |
| 3.0 | 0.472 | 0.326 | 0.795 |

**r is FLAT at 0.795-0.809 while s swings 0.47 to 0.76.** The lever moves the spread and leaves the
ranking untouched, exactly as the identity demands. **Weight 1.0 gives +0.063 over control at no
cost to r. Weight 3.0 OVERSHOOTS and is worse than no lever at all**, so the response has an
interior optimum rather than being monotone — factor C must be read that way.

## 3.3 a_p is NOT a statistical artefact — attenuation is dead by 200x

The obvious objection: a_p ~ 0.5 is the textbook signature of regression attenuation under noisy
labels, in which case no architectural lever could ever fix it.

Per-mutation uncertainties were found in `data/ThermoMPNN/mega_test.csv` (an earlier attempt
searched the wrong file and reported 0/28). **28/28 of our test proteins join.**

    median 95% CI = 0.1007  ->  median sigma = 0.0327 kcal/mol  (n = 28,312)
    median var(true ddG)   = 0.7178
    attenuation predicts a_p = 0.7178/(0.7178 + 0.0327^2) = 0.9985

**Measured a_p ~ 0.50 against a predicted 0.9985.** The label noise would need to be ~200x larger
(sigma ~0.85 instead of 0.0327). **a_p is a genuine model failure.** This confirms by a direct route
what a sign-based argument showed inferentially: attenuation predicts the WRONG SIGN for the
exposure correlation, with forward simulation giving P(r >= +0.714) = 0/2000.

## 3.4 WHERE the ranking error lives: burying hydrophobics

Per-destination-residue slope, 10 checkpoints, ~26,000 matched mutations each (destination residue
recovered by joining on deltaG against the mutation files; ambiguous values dropped, not guessed):

| destination | KD | slope | spearman |
|---|---|---|---|
| R | -4.5 | 0.528 | 0.649 |
| K | -3.9 | 0.537 | 0.648 |
| N | -3.5 | **0.565** | 0.653 |
| D | -3.5 | 0.542 | 0.650 |
| S | -0.8 | 0.524 | 0.638 |
| G | -0.4 | 0.506 | 0.616 |
| A | +1.8 | 0.419 | 0.567 |
| M | +1.9 | 0.340 | 0.461 |
| **C** | +2.5 | **0.267** | 0.356 |
| F | +2.8 | 0.288 | 0.381 |
| **L** | +3.8 | **0.282** | 0.388 |
| V | +4.2 | 0.309 | 0.428 |
| I | +4.5 | 0.311 | 0.412 |

    corr(slope, Kyte-Doolittle)    = -0.734   sign-consistent 10/10
    corr(spearman, Kyte-Doolittle) = -0.740

**Mutations TO hydrophobic residues are compressed roughly TWICE as hard.** And the decisive
detail: **those two correlations are essentially identical**, so rank accuracy degrades in exact
lockstep with the slope (Spearman 0.65 for polar destinations, 0.36-0.43 for hydrophobic).

**This is LOST INFORMATION, not a rescalable calibration error.** If the slope had fallen while
ranking held, a per-class rescale would recover it. It did not. No affine correction can help, and
this is the (1-r) half of §3.1 given a chemical identity.

**Mechanism:** the dataset stores only **4 backbone atoms per residue (N, CA, C, CB) — no side
chains**. Burying a large hydrophobic is dominated by side-chain packing, which CB-only geometry
cannot see. **W and Y are the telling exceptions** (slopes 0.341/0.353 despite negative hydropathy),
so the pattern tracks **side-chain BULK** rather than hydropathy alone — which strengthens the
packing explanation. **This argues for an architectural fix (side-chain information), not a
calibration one.**

## 3.5 Exposure predicts the slope

`mean_rel_SASA` vs a_p: **r = +0.714**, replicated across all 10 eval CSVs (mean 0.624,
sign-consistent 10/10). The only one of 90 tested correlations to clear Bonferroni on BOTH Pearson
and Spearman.

**It survives the length confound**, which was the obvious objection: partialling out chain length
gives **+0.6623, significant in 10/10 checkpoints**, while length alone is -0.2149 and significant
in **0/10**.

**More-exposed proteins suffer LESS slope collapse; tightly buried proteins are where the model
under-reacts worst** — consistent with §3.4.

## 3.6 The anchor suppresses the coupling

The exposure-slope correlation decays monotonically with the WT-anchor weight:

    anchor 0.3 -> r = 0.587      anchor 1.0 -> r = 0.471      anchor 3.0 -> r = 0.400
    unanchored sigma seeds       -> r = 0.63-0.68

**Factor A of the running factorial is already suppressing the exposure/slope coupling**, giving
its anchor cells a specific prediction to test rather than an open-ended sweep. Three points is a
trend, not a law.

---

# PART IV — WHAT WE LEARNED ABOUT b_p (THE OFFSET)

## 4.1 b_p is real, separable, and learnable

A pure random-offset null CANNOT reproduce our numbers. Grid-searching (noise, slope) over 525
cells, the closest the null gets to our triple (pooled 0.590 / removed 0.710 / per-protein 0.798)
is **L2 = 0.118**, and it fails structurally:

- it **over-delivers** on offset removal (+0.075) — a meaningless constant is perfectly removable
- it **under-delivers** on per-protein PCC (-0.086)
- it needs slope **1.25** (expansion) where we measure **0.499** (compression)

**ICC(b_p) = 0.898** over 28 proteins x 5 seeds (seed-only ~0.95-0.97 once the epoch confound is
removed). Only SD 0.503 kcal/mol is unpredictable noise, so the best achievable pooled PCC from
offset correction is ~0.705 against the 0.718 oracle — **b_p is genuinely learnable in principle**.

## 4.2 What b_p is NOT — seven hypotheses tested and rejected

| hypothesis | result |
|---|---|
| label noise / attenuation | **DEAD by 200x** (§3.3) |
| chain length | corr(N, b_p) = **+0.0252** over 12 checkpoints |
| `--dg_length_norm` | **DEGENERATE** (§5.3) |
| embedding norm / 1/sqrt(N) artifact | claimed -0.9938, **measured -0.29** — below the n=28 threshold of 0.392 |
| pre-training distribution shift | r = **+0.043**, p = 0.829 |
| train/test mean gap (the Ofir mechanism) | gap **-0.1034**, p = 0.588 — 6.6% of one SD, and a shared constant cannot generate per-protein DISPERSION anyway |
| a feature-based corrector | LOPO 0.6049 vs raw 0.5994; held-out R2 NEGATIVE on 8/10 |

**b_p has survived every cheap explanation.** What remains untested is the reference state itself —
where the coil produced the only real movement (dG MAE 4.9650 -> 3.9521), and `gld_dg_coil` is
running now.

## 4.3 Burial predicts the offset

`frac_buried_rel_lt_0.25` vs the WT error: **mean r = -0.475, sign-consistent 10/10**, significant
in 8/10. This corrects an earlier claim that "the signal is on a_p, NOT b_p", which was premature:
the b_p side was tested once, failed a uniform Bonferroni bar over 90 tests, and was dropped from
the replication sweep while its sibling was replicated and promoted.

**The honest picture: ONE structural axis acts on BOTH calibration channels.** Exposure -> a_p
(+0.624), burial -> b_p (-0.475), both showing the same anchor-weight suppression.

## 4.4 The corrector fails — a publishable negative

Over 10 eval CSVs, leave-one-protein-out ridge on 22 structural features:

    raw pooled                    0.5994
    LOPO feature-predicted offset 0.6049      <- +0.0055
    oracle offset removed         0.7119      <- +0.1125

**95% of the oracle gain does not survive held-out prediction.** Held-out R2 of b_p is NEGATIVE on
8/10 CSVs; a permutation null is significant on 0/10; with the ridge penalty fixed the gain goes
negative, so the +0.0055 was alpha-search variance.

**The mechanism is the useful part:** the top-2 |b_p| proteins carry **78%** of the oracle gain, and
**neither is a structural outlier** (max |z| ~2.0 over 22 features), so there is no signature to
regress on.

**A metric correction that matters:** Pearson is SHIFT-INVARIANT, so subtracting a CONSTANT changes
pooled PCC by exactly zero (verified: -1.0/0.0/mean/+1.0 all give 0.590991). **The null a
per-protein corrector must beat is the RAW number, not a mean-offset baseline** — scoring against
the latter would have manufactured a fake +0.011 lift.

---

# PART V — LEVERS: BUILT, GATED, AND WHAT EACH IS WORTH

## 5.1 Inventory — 12 levers, all wired end to end and gated

| lever | flag | acts on | status |
|---|---|---|---|
| U2 unfolded embedding | `--unfolded_emb {full,zero,mean}` | b_p | **factor D — decisive, see §7** |
| U3/U4 Flory coil | `--flory_unfolded --coil_b {fitted,fixed}` | **dG / b_p** | best b_p lever; `gld_dg_coil` running |
| U5/U6 coil edges + per-half ca_coords | `--coil_edges` | dG | on a card |
| W5 burial | `--burial_features --burial_mode {count,hse}` | **dG** | `gld_w5_dg` running |
| W6 descriptors | `--aa_descriptors {none,mordred726,mordred_pca16,...}` | a_p | **largely redundant, §8** |
| W7 chain span | `--gcn_span N` | a_p | `gld_w7span4` running |
| W7-edge | `--edge_features` | a_p | `gld_w7edge` running |
| U10 bidirectional | `--gcn_bidir` | a_p | `gld_u10bidir` running |
| W9 metals | — | dG | **RETIRED**, superseded by W11 |
| W11 ligands | `--ligand_nodes --ligand_annotations` | **dG / b_p** | zero variance on our 28 |
| C slope | `--slope_weight` | **a_p** | **works at 1.0, §3.2** |
| LORO | `--holdout_residues` | a_p | both arms on cards |

## 5.2 Gates — 15 suites, all green

`gate_g4_cpu` is the guard that protects everything in flight: it must print
**`baseline ... dG=-0.0030 width=1092`** after ANY change. It has been re-run after every edit in
this project and has never moved.

Others: `gate_slope` (21/21), `gate_loro` (13/13), `gate_refrow`, `gate_noop` (30 checks),
`gate_resume` (42/42), `gate_u2`, `gate_w5` (18/18), `gate_w7`, `gate_w7_edge`, `gate_u3u4` (32/32),
`gate_u5u6`, `gate_w9` (retirement verified), `gate_w8` (51/51).

## 5.3 --dg_length_norm is DEGENERATE

Over 12 eval CSVs: std(b_p) none **1.3076**, /n **0.9050**, /sqrtn **0.9159** — an apparent 31%
improvement.

**It is not.** The /n column converges to ~0.905 for EVERY checkpoint, including ones that START at
0.876 where it makes things worse. The reason:

    std(true WT dG) over the 28 test proteins = 0.9042

**Dividing by N drives pred/N to nearly zero, so b_p -> -true_dG and its spread becomes the spread
of the labels.** The arm deletes the prediction rather than removing a bias.

## 5.4 gate_u3u4: the diagnosis worth keeping

It failed 2/32 for a long time and was written off as "known-false". It was neither. Two tolerance
fixes both failed, which forced an actual measurement:

    off-diagonal agreement: 1.2e-06 (perfect)
    the four DIAGONAL channels (0/5/10/15, the self-pairs): 1.38e-03

`get_dist_matrix` flattens to (N*4,3) and runs ONE cdist; a self-distance goes through
`sqrt(x^2+x^2-2x.x)`, a catastrophic cancellation returning ~1e-3 instead of exactly 0. **The gate
was comparing that artifact and calling it a layout error.** Channel 5 IS CA-CA (1*4+1=5).

**Do not guess at a tolerance twice — measure the error's STRUCTURE first.**

## 5.5 Four silent no-op holes, closed

The project's signature failure mode: code runs, completes, prints a success line, and the feature
was never read. `scripts/gate_noop.py`, 30 checks, all pass.

1. **PEMGraphTransformer dropped inserted blocks silently — MEASURED, not assumed.** A gate builds
   `flat_x` with 726 sentinel columns at offset 48, runs the real reassembly, and the output is
   byte-identical to the no-descriptor case: **0 of 726 sentinel columns reach the node vector, and
   nothing raises.** The guard now fires in `__init__` for all five affected levers.
2. **`--ligand_nodes` could not fire in training at all** — and `gate_ligand` D.14 had CODIFIED the
   bug by asserting the zeros were correct behaviour.
3. **`gate_open_alphabet` tested a /tmp fixture**, and the real committed 25-row table is K=756 vs
   canonical 726 with 27 columns dropped and 57 added — a different matrix wearing the same name.
4. That same gate hard-coded a retired mode name and **had stopped testing anything at all.**

## 5.6 The scoring pipeline: three bugs hid 12 finished cells

12 factorial cells completed all 15 epochs and produced ZERO CSVs. Three independent faults:

1. `evaluate.py:58` had `DEVICE = 'cuda'` with the CPU fallback **commented out**, so every eval had
   to queue for a card behind 41 trainings in the same 5-GPU pool.
2. `load_checkpoint` assumed a wrapper dict and raised `KeyError: 'model_state_dict'` on every
   per-epoch bare state_dict save.
3. `torch.load` calls lacked `map_location`; the tensors were saved on CUDA.

**And it stayed hidden because `run_calib_eval.sh` prints `DONE ... -> abl_<tag>.csv` AFTER the
crash, with no CSV on disk.** A success line that outlives the failure.
Scoring now runs on the **CPU partition at zero GPU cost**.

## 5.7 --resume, built and verified

`gate_resume`: **42 passed, 0 failed** — two-stage refusal, optimizer and scheduler state, backward
compatibility with the 361 existing bare-state_dict checkpoints, and the scheduler-state fixup
ordering. Needed because the golden lane is `PreemptMode=requeue`.

---

# PART VI — WHAT THE MODEL ACTUALLY SEES

## 6.1 The feature vector

    [ D(16) | Fb(32) | <new blocks inserted here> | emb(1024) | one_hot(20) ]  = 1092

- **D(16)**: pairwise distance channels, `atom_i*4 + atom_j` over N, CA, C, CB. **CA-CA is channel 5.**
- **Fb(32)**: bonded/geometric features.
- **emb(1024)**: ProtT5, per residue.
- **one_hot(20)**: residue identity — what the mutation actually changes.

New blocks go BETWEEN Fb and emb so the right-anchored emb and one_hot slices never move. Only
`fc1_gcn`/`fc1_gat` grow; `fc2_*`, `inst_norm1`, `inst_norm2` and `fc_in_dim` must NOT.

## 6.2 The architectural fact that constrains every feature design

**The GCN branch reads `x[:, :32]` only** — D(16) plus HALF of Fb. It never sees the embedding, and
**any new block inserted at offset 48 is invisible to it.** W5/W6/W9/W11 reach the GAT branch alone.
Any claim that a block "is used by the model" must say WHICH branch.

## 6.3 The tensors are per-VARIANT

    prott5_embeddings/prott5_embedding_<k>.pt   [128, 52, 1024]   128 variants x 52 residues
    one_hot_encodings.pt                        [1389, 52, 21]    21 columns
    coords_tensor.pt                            [N, 4, 3]  ANGSTROM (already scaled)

The stored one-hot's 21st column is dropped at `train.py:317` before the graph is built, so every
`[20, K]` descriptor table is correctly sized — **checked rather than assumed.**

Per-protein tensors live at `/groups/keasar_group/casp15/meytav/protein_tensors/<P>/`, NOT in
`data/`.

---

# PART VII — THE FACTORIAL: FACTOR D IS DECISIVE

11 of 48 cells scored:

| arm | pooled ddG PCC | a_p |
|---|---|---|
| **D0 (coil)** — 5 cells | **0.58-0.62** | 0.37-0.72 |
| **D1 (`--unfolded_emb zero`)** — 5 cells | **-0.08 to 0.31** | **0.005-0.020** |

**No overlap. Five versus five, perfectly separated.** a_p differs by two orders of magnitude —
**an a_p of 0.005 means the model is FLAT and barely responds to mutation at all.** One D1 cell is
ANTI-correlated at -0.0782, which the autopilot's own audit flagged.

**This reverses the W0 decision.** W0 chose factor D as `--unfolded_emb` on the ddG metric because
zeroing the unfolded embedding collapsed var(E_u) by 67% and cut corr(E_u, wt_err) from 0.420 to
0.119. **Trained end to end, it destroys the model** — it removes the offset by removing the signal,
exactly the caveat recorded for `noemb` on dG (lowest MAE but correlation 0.35 -> 0.05).

**Three consequences:**
1. The D1 half is ~190 GPU-hours confirming a negative already visible at n=5.
2. **No main effect for A, B or C may be averaged across D** — the D effect is ~10x larger.
3. **The best cell has the HIGHEST a_p (0.7156) but NOT the highest pooled PCC (0.5827 vs 0.6175).**
   That is the calibration thesis in one line: pooled PCC is dominated by the offset, a_p measures
   the compression.

---

# PART VIII — CHEMISTRY: WHY W6 IS LARGELY REDUNDANT

## 8.1 ProtT5 already extrapolates chemistry

ProtT5 is **contextual** (same residue type at two positions: cosine **0.13-0.18**, nearly
orthogonal) and **recomputed per variant** (a single point mutation changes ALL 52 positions, so the
embedding does NOT cancel in ddG and acts on both channels).

Residue-disjoint probe — fit on 19 residue types, predict the held-out **20th**, 19 proteins,
1,056 residues:

| property | held-out R2 |
|---|---|
| **hydropathy** | **+0.704** |
| charge | +0.511 |
| volume | +0.421 |
| identity (row-random) | accuracy **1.000** vs chance 0.050 |

**ProtT5 predicts the hydropathy of a residue type it has NEVER seen, at R2 = 0.70.**

A descriptor block is a deterministic function of residue identity; identity is perfectly decodable
from the embedding, and the chemistry itself extrapolates. **W6 cannot be justified as adding
information — only regularisation, or coverage of residues absent from ProtT5's training data.**

## 8.2 Reconciliation with Ofir Ezrielev's thesis (56 pages, read in full)

**Three things in our build were WRONG and are now corrected in the source:**

1. **PCA-16 is NOT his method.** He does no dimensionality reduction of any kind — "PCA",
   "principal component" and "SVD" appear NOWHERE in the thesis; his 654 features enter a CNN as 654
   channels. **Our PCA is ours** and is now labelled as such in the CSV provenance.
2. **`ignore_3D=True` contradicts his 1826.** 1826 = 1613 2D + 213 3D, a total only coherent if 3D
   was requested — which also explains his 1826 -> 1280 missing-value drop, since 3D descriptors
   return NaN without a conformer. With `ignore_3D=True` the ceiling is 1613.
3. **15-of-20 is not a faithful scaling of 40-of-58.** 40/58 = 0.690 scales to 13-14, not 15. And
   he gives **no justification for 40 at all**, so there is no ratio to preserve. Also, 2D Mordred
   cannot distinguish D from L, so his effective alphabet is well under 58.

**HE CONCEDES THE LANGUAGE-MODEL OBJECTION** (p.31, verbatim): *"Physicochemical properties of AAs
are implicitly represented in these datasets ... and accordingly, implicitly represented in the
embeddings. Thus, ncAAs, which do not occur in these databases, are challenging to represent."*
**His argument is COVERAGE ONLY, not added information**, and he runs zero experiments comparing
descriptors to an LM embedding or to one-hot. **Our 28 test proteins are canonical, so the gap he
exploits does not exist here.**

**He found our offset problem independently:** his leave-one-amino-acid-out RMSE tracks the
train/test MEAN GAP while r stays flat (0.858-0.948). Worth stealing: **leave-one-residue-type-out**,
which we had never run and is now on a card.

**Histidine SMILES bug:** the plan document had `c1cc(nc1)...` — a ring with FOUR carbons and ONE
nitrogen (C7H10N2O2). L-histidine is imidazol-4-yl, C6H9N3O2, TWO ring nitrogens. Mordred would have
computed ~654 descriptors for the wrong molecule, invisibly, because the gate tested only L, D and F.

## 8.3 The LORO test — prediction on record

Two runs differ in exactly one flag, with tryptophan held out of TRAINING only:

    gld_loroW_onehot   --aa_descriptors none
    gld_loroW_desc     --aa_descriptors mordred_pca16

**PREDICTION, recorded before any result: the descriptor arm will NOT beat one-hot by much**,
because ProtT5 already carries W's chemistry (§8.1). If it DOES win, that overturns §8.1 and will
be reported as a genuine surprise rather than rationalised. Negative control: proline, whose effect
is backbone geometry and which descriptors should NOT rescue.

---

# PART IX — BIOLOGY: WHAT IS AND IS NOT MEASURABLE HERE

## 9.1 The hard constraint

**All 28 test proteins are single-chain, ligand-free, metal-free monomers of 43-72 aa.** Verified:
`n_chains == 1`, `n_het_residues == 0`, `n_metal_residues == 0`, `interchain_BSA == 0.0` for every
one.

**So ligands, complexes and true interface area have ZERO VARIANCE on this test set.** That is
"not measurable here", **NOT "no effect"** — and a lever whose driving feature is constant must be
logged untestable, never scored and given a zero, or we manufacture a false negative.

## 9.2 The het-code composition — why a naive ligand feature would learn crystallography

205,648 het-code occurrences over 8,187 distinct codes in the 100k catalogue:

| class | share |
|---|---|
| **crystallization additives** (SO4, GOL, EDO, NA, CL, PEG) | **27.4%** |
| metal ions | 18.9% |
| **real cofactors/substrates** (NAD, ATP, HEM, FAD) | **6.6%** |
| **modified residues** (MSE, SEP, TPO) | **6.3%** |
| unclassified | 43.5% |

**Only 6.6% are real biological ligands.** 12.8% of "ligand-bearing" rows carry ONLY additives. And
**MSE is selenomethionine — an amino acid IN THE CHAIN, used for phasing** — so counting it as a
bound ligand is both biologically wrong and a double-count of a residue.

`data/het_classification.csv` now classifies all 8,187 codes at **98.1% occurrence coverage**;
only 208 count as a bound ligand.

## 9.3 W9 metals: retired, not half-wired

`metal_features.py` was fully written and read `CFG.metal_features`, but **no `--metal_features`
flag ever existed in train.py**, so it could never fire. Rather than add a second dead path, W9 is
formally retired — W11 `--ligand_nodes` generalises it, since a metal is one ligand class — and
`hydro_net._sibling_block_dims()` now RAISES if the retired flag is set, with the reason.

---

# PART X — THE 100k CATALOGUE

## 10.1 Retired for every join-based purpose

**It has NO SEQUENCE COLUMN.** That is a permanent, in-principle result, not a failed attempt.
Joins measured: **0/226** training PDB ids, **0/21** test, **0/340** by protein_id, against 66,945
catalogue ids. **This kills the plan to escape n=28 by testing structural correlations on the
training set.**

## 10.2 Dead columns that look alive

| column | non-null | live sibling |
|---|---|---|
| `Ligands_x` | **4** of 100,246 | `Ligands_y` (75,464) |
| `BSA_Numeric_x` | **4 unique** | `BSA_Numeric_y` (4,867) |
| `BSA_Percentage` | 5 | — |
| `lossg` | constant 0 | — |
| `Global Symmetry` | constant | — |

**This is where the -0.001 BSA null came from: a dead column scored against the pre-training decoy
loss.** `scripts/catalogue_schema.py` now quarantines all five (21 -> 18 columns), refuses to
correlate anything against the decoy-loss columns for a ddG question, and applies a BSA policy
(294 nulls: 246 original + 48 internally inconsistent rows where the % string is nonzero while the
numeric is 0). `BSA_Numeric_y` is a PERCENTAGE in [0,70], not Angstrom^2.

## 10.3 The one live use: distribution shift

| | catalogue | our 32-74 aa regime |
|---|---|---|
| median length | 477 | 55-56 |
| **share of pre-training RESIDUES** | — | **0.1206%** (1 in 830) |
| SOLUTION NMR | 8.6% | **77.4%** |
| Is Complex = Yes | 68.1% | **11.8%** |
| Monomer | 31.9% | **88.2%** |

**Our entire problem domain was one residue in 830 of pre-training, and a structurally different
kind of protein.** The residue share is the right denominator because the pre-training loss is
per-residue.

**But it is NOT the cause of b_p:** r = **+0.043**, p = 0.829. Across 25 shift and composition
features, three reach nominal significance where ~1.25 are expected by chance, and **none survives
Bonferroni or BH-FDR** for either b_p or a_p.

Caveat: `AA Length` is entity/assembly length, not per-chain, so 0.1206% is a **lower bound** on the
mismatch severity.

---

# PART XI — INFRASTRUCTURE AND RESOURCES

## 11.1 The two GPU lanes

| | public | golden |
|---|---|---|
| flags | `--qos normal` | `--partition=rtx6000 --qos keasar` |
| cap | **`gres/gpu=5` per USER** (`gpu-part` QOS) | **`gres/gpu:rtx_6000=8` per ACCOUNT** (MaxTRESPA) |
| MaxWall | — | unset |
| preemption | — | `Preempt=normal`, `PreemptMode=requeue` |

**Both parts of the golden invocation are required**: `--qos keasar` alone is rejected on the gpu
partition ("Invalid qos specification"), and the `rtx6000` partition has `DenyQos=normal` so it
REQUIRES that QOS. Jobs there pend on `MaxGRESPerAccount`, a different limit — **the pools are
genuinely separate.**

**A lua job_submit plugin rewrites any GPU request onto the `gpu` partition** (it prints
`GPU Parameter Set ! Using GPU Partition.`), which is why the site rule says never to select a GPU
by partition name — ask with `--gres=gpu:rtx_6000:1` or you can land on a GTX 1080 and OOM.

The account cap is SHARED: when a labmate held all 8, we got zero regardless of 14 free cards.

## 11.2 The autopilot

Runs the S0-S9 state machine, records completed cells, applies decision node D1, and halts on any of
four halts. **Now a SLURM CPU job with a 7-day limit** rather than a login-node process — it had
died repeatedly with `rc=143` (SIGTERM from login-node churn).

**Two bugs found and fixed:**
1. `load_state` only fell back to a default when the file was ABSENT. A file containing `{}` — what
   an interrupted write leaves — was loaded verbatim and `main()` died on `KeyError: 'state'`.
2. **The wrapper broke its loop on `rc=1`, which is ALSO what Python returns for an uncaught
   exception**, so any crash killed the autopilot permanently. It now stops only on `rc=3` (a halt
   is a human decision) and counts fast crashes instead of dying silently.

**D1 has a slope guard** that rejects a cell which raises pooled while collapsing the slope below
0.3 median — verified by construction: it rejected a synthetic cell with the HIGHEST pooled (0.70)
because its slope was 0.10, and logged the rejection loudly.

## 11.3 Standing site rules, each paid for in wasted runs

- `export WANDB_MODE=disabled` before EVERY submission — one omission destroyed a 7h53m run.
- `--gres=gpu:rtx_6000:1`, **never select a GPU by partition name**.
- Never `ise-pheno` (3.5x slower under contention), never 2080/1080.
- All seeds of one cell on the same node class.
- **A finished training run is NOT a result.** Track `ls eval_results/abl_*.csv | wc -l`, never
  `sacct | grep COMPLETED`.
- Tensors in `data/` were saved on CUDA: load with `map_location` and `weights_only=False`.
- `class PEM(torch.nn.Module):` — an anchor written `nn.Module` silently matches nothing.
- `/tmp` is per-login-node; use `/home/nissimb/` for anything shared.
- `run_calib_eval.sh` invokes `python3`, which resolves to the SYSTEM python without pandas unless
  the env's bin is first on PATH — and `conda activate` fails in a detached shell with no conda init.

---

# PART XII — DATA INTEGRITY

## 12.1 2K5H's reference row was a MUTANT

`data/Processed_K50_dG_datasets/mutation_datasets/2K5H.csv` (4,107 rows) **concatenates THREE
backgrounds** and the wrong one sorts first:

    row 0    : name=2K5H.pdb_G11S   deltaG=1.723086   mut_type=wt
    idx 2738 : name=2K5H.pdb        deltaG=4.805470   mut_type=wt   <- the TRUE wild type

Both are labelled `wt` because G11S genuinely is the wild type *of its own background*, so nothing
looked wrong. **Shift: +3.0824 kcal/mol** applied to all 1,125 of its ddG labels.

**The control that makes it unambiguous:** row-0 percentile within each protein's own dG
distribution — 26 of 28 sit at the **62nd-99th percentile**; 2K5H sits at **14.6%**.

**Cost, measured over 22 CSVs: dropping 2K5H alone removes 36% of the entire offset-removal gain.**
A constant per-protein label shift IS a per-protein offset, so the oracle was rediscovering one we
had introduced.

**Fixed in `data_fixed/mutation_datasets/2K5H.csv`** — the source is owned by another user and
reached through a symlink, and that tree is read-only, so the corrected copy went into our own
space with all 4,107 rows preserved.

**`gate_refrow.py`** now counts **distinct wild-type backgrounds** per protein, which is the
decisive test — the percentile alone false-flagged `r18_3_TrROS_Hall`, which is CLEAN (single
background, row 0 is the true WT; a low percentile is legitimate for a designed protein with many
stabilising mutations).

## 12.2 The same bug produced a second false lead

An agent reported **26,315 double mutants** for our 28 as an untouched generalisation test.
Measured: **2,356 two-point rows, ALL belonging to 2K5H**, with names like `2K5H.pdb_G11S_A1Q` —
**single mutations on the G11S background.** Removing the background prefixes leaves **NO true
double mutants** for any of the 28.

**One malformed file produced two separate false leads, both plausible enough to have reached the
thesis.**

---

# PART XIII — WHAT IS RUNNING NOW

## 13.1 GPU (13 cards) as of 2026-09-08

**Golden lane, 8/8 — every lever that had NEVER been run:**

| job | what it tests | epoch |
|---|---|---|
| `gld_dg_coil` | **the dG arm** — the coil on the metric it acts on | 12/14 |
| `gld_slope1.0` | **`--slope_weight`** — the only lever attacking a_p | 12/14 |
| `loroW_onehot` | LORO baseline | 11/14 |
| `gld_u10bidir` | bidirectional GCN edges | 11/14 |
| `gld_w7span4` | chain span 4 | 10/14 |
| `gld_w5_dg` | burial scored on dG | 9/14 |
| `gld_w7edge` | edge features | 8/14 |
| `loroW_desc` | LORO with descriptors | 1/14 |

**Public lane, 5/5:** factorial cells. **Pending: 84** (48 blocked on the public cap, 36 dependencies).
**CPU: 2** — the autopilot and the scorer.

## 13.2 Results so far

**11 of 48 factorial CSVs.** 17 cells have completed all 15 epochs; 8 more are partially trained
(epochs 1-11) with checkpoints on disk.

## 13.3 The throughput opportunity, NOT yet acted on

The 48 pending factorial jobs were submitted with `--qos normal`, so **they cannot see the golden
lane at all**, and they were submitted **without `--resume`**, so the 8 partially-trained cells
(~50 epochs already done) would restart from zero.

Resubmitting the PENDING ones (never the running ones) with `--partition=rtx6000 --qos keasar
--resume` would cut the remaining ~49 hours to roughly **18**. **Not done — awaiting approval**,
because it requires cancelling queued jobs and verifying that resume genuinely continues rather
than printing that it does.

---

# PART XIV — CLAIMS THAT FAILED VERIFICATION

Recorded so they are not re-adopted. **Every one was plausible.**

| claim | reality |
|---|---|
| 1/sqrt(N) embedding artifact, r = -0.9938 | **measured -0.29**, below the n=28 threshold of 0.392 |
| 26,315 double mutants available | **zero real ones** (§12.2) |
| `--dg_length_norm` shrinks b_p by 31% | **degenerate** — it deletes the prediction (§5.3) |
| The affine oracle reaches 0.77-0.81 | **not reproducible**; offset removal alone gives 0.644 corrected |
| W5 burial predicts b_p (r=-0.343, n=19) | **not significant** — threshold 0.490 |
| "The signal is on a_p, NOT b_p" | **premature** — burial predicts b_p at -0.475, 10/10 (§4.3) |
| Five structural correlations clear Bonferroni | **only ONE** clears on both Pearson and Spearman |
| pLDDT predicts per-protein PCC | p=0.015, **27x above** the stated bar, and confounded with length |

**Two recurring failure modes:** a correlation quoted without its n, and an improvement quoted
without asking what the number would be if the model predicted nothing.

---

# PART XV — OPEN QUESTIONS

1. **Does the coil reduce b_p under training?** `gld_dg_coil` is at epoch 12/14. This is the last
   untested explanation for b_p (§4.2).
2. **Does `--slope_weight` at 1.0 hold up over 15 epochs and multiple seeds?** §3.2 is one seed.
3. **Will the descriptor arm beat one-hot on held-out tryptophan?** Prediction recorded (§8.3).
4. **Is the D1 half worth finishing?** ~190 GPU-hours to confirm a negative visible at n=5 (§7).
5. **Would side-chain information fix the hydrophobic ranking failure?** §3.4 says the deficit is
   information the model cannot see; this is the strongest architectural lead in the project.
6. **Can b_p be predicted from features at all?** The corrector failed (§4.4), and the mechanism —
   two proteins carrying 78% with no structural signature — suggests the answer may be no.
7. **The severing experiment** is built and dry-run green but not submitted: it would measure
   `cov(E_u, E_f)`, never reported, and decide whether the 77-88%-of-variance-in-E_u premise is
   reading a shared per-protein network bias rather than the reference state.
