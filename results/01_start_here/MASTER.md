# DeepEF — MASTER RECORD
### M.Sc. thesis · Nissim Brami · supervised by Prof. Chen Keasar · Ben-Gurion University
### Complete state as of 2026-09-08. Written to survive a full context wipe.

> **CANONICAL BASIS (P8).** Every headline number in this document is computed on:
> **27 test proteins (2K5H excluded) | ddG metric | the 9 original-population eval CSVs |
> the val-selected epoch only.** That basis is `pooled 0.5772 / oracle 0.7156 / gain +0.1384`.
> Membership, exclusions and provenance: `results/05_infrastructure/HEADLINE_BASIS.md`.
> Enforced by `scripts/gate_headline.py`. **Never average across runs that differ in factor D**
> (`--unfolded_emb zero`) — D0 and D1 are different models; report them separately, always.


> **How to read every number here.** Three qualifiers, always: *pooled-or-per-protein* /
> *dG-or-ddG* / *which split*. A number without its basis is not a measurement — that mistake was
> made in this project and is corrected in §1.4.
>
> **Scoring basis for this document unless stated otherwise:** 27 test proteins (2K5H excluded,
> see §11.1), ddG metric, per-protein fits.

---

# 1. THE PROBLEM AND THE SCORE

## 1.1 What DeepEF is

A graph neural network over protein structure that predicts folding free energy as a **difference
of two forward passes over the same network**:

    dG  = E_unfolded - E_folded
    ddG = dG_mutant - dG_wild-type

**Benchmark:** MegaScale (Tsuboyama et al. 2023) — ~776,000 measured stabilities on small domains,
40–72 residues, by cDNA-display proteolysis. **28 held-out test proteins, 28,314 test mutations.**

**Architecture:** two branches over the residue graph — a GCN over chain-local edges and a GAT over
k-NN edges — feeding a per-residue energy head summed over residues (E is **extensive**).

## 1.2 The central finding — the calibration decomposition

Per protein, the prediction is an affine function of the truth:

    pred ≈ a_p · true + b_p

| quantity | value | what it is |
|---|---|---|
| **per-protein ddG PCC** | **0.798** | ranking quality WITHIN a protein |
| **pooled ddG PCC** | **~0.59** | ranking quality ACROSS all proteins together |
| **a_p** (median) | **0.4990** | the SLOPE — the model compresses ddG by half |
| **std(b_p)** | **1.603** (dG-space) | the per-protein OFFSET |
| corr(a_p, per-protein PCC) | **+0.5714** | the slope is the strongest per-protein quality predictor |
| ICC(b_p) | **0.898** | b_p is protein-attributable, not seed noise |

**The model ranks well within a protein and is miscalibrated between proteins. The gap between
0.798 and 0.59 is the entire thesis.**

## 1.3 THE SCOREBOARD — every run, one basis

| rank | run | pooled | oracle | a_p | r | s |
|---|---|---|---|---|---|---|
| 1 | `sigma_seed2_e10` | **0.6382** | 0.7534 | 0.410 | 0.791 | 0.598 |
| 2 | `p3_slope1.0_s42_e13` | **0.6210** | 0.7468 | 0.507 | 0.803 | 0.756 |
| 3 | `anchor_w0.3_s42_e14` | 0.6105 | 0.7384 | 0.542 | 0.774 | 0.830 |
| 4 | `p3_a1_d0_s1_D0_coil_seed42` | 0.6028 | 0.7185 | 0.571 | 0.751 | 0.808 |
| 5 | **`gld_w5_dg_s42_e14`** | 0.5956 | 0.7212 | **0.664** | 0.779 | **0.870** |
| 6 | `anchor_w1.0_s42_e14` | 0.5942 | 0.7112 | 0.574 | 0.734 | 0.849 |
| — | **control** `calib_ctrl_repro2_e14` | **0.5635** | 0.7364 | 0.496 | 0.793 | 0.635 |
| … | `gld_slope1.0_s42_e13` | 0.5810 | 0.7228 | **0.740** | 0.790 | **1.042** |
| ✗ | `gld_dg_coil_s42_e14` | **0.2212** | 0.4470 | 0.091 | 0.509 | 0.211 |
| ✗ | `p3_a1_d1_s0_D1_uemb_seed42` | **−0.0741** | 0.3461 | 0.011 | 0.199 | 0.121 |

**Best pooled ddG PCC = 0.6382** · **control = 0.5635** · **best gain = +0.075**
**Best oracle = 0.7534** (the ceiling if b_p were solved) · **best a_p = 0.740**

**SEED NOISE BAND = ±0.060** (five `abl_sigma` seeds, identical config, pooled 0.556–0.638).
**Any lever under +0.060 is not distinguishable from noise.** This is the bar everything must clear.

## 1.4 A correction to my own headline

I earlier published "corrected headline 0.4899 / 0.6443 / +0.1544". **That was averaged over 22
mixed CSVs including D1 arms scoring as low as −0.08.** Averaging a correlation across a mixed
population is not a measurement — it is an artifact of which runs happened to be scored that day.

**On the 9 original-population CSVs with 2K5H dropped:**

    pooled 0.5772   oracle 0.7156   gain +0.1384

Three different populations gave 0.4899 (22 mixed CSVs, D1 arms included), 0.5646 (20 non-D1
checkpoints, three lanes) and 0.5772. **The last is canonical; the first two are retired.**
The basis is now declared and machine-enforced: `results/05_infrastructure/HEADLINE_BASIS.md`
and `scripts/gate_headline.py`.

## 1.5 The affine oracle (0.77–0.81) is NOT reproducible

The earlier claim that removing offset AND slope reaches 0.77–0.81 does not hold: seven original
checkpoints give 0.646–0.774 (one in range), five of ours 0.488–0.658 (none).

**The mathematics explains it.** The affine oracle **divides** by the slope; when a_p = 0.079 that
amplifies noise ~13×. An estimator that is not monotone is not a bound. Offset removal survives
because subtracting a constant cannot amplify anything.

---

# 2. THE RULE THAT ORGANISES EVERYTHING

## 2.1 The metric rule

**ddG cancels anything identical between wild type and mutant.** A point mutation changes neither
chain length nor backbone positions, so every purely geometric or whole-protein property cancels
EXACTLY in `pred_mut − pred_wt`.

> **A lever acting on the reference state or on absolute stability MUST be scored on dG or b_p.
> a_p is a WITHIN-protein slope and does NOT cancel, so it IS legitimately ddG-measurable.**

## 2.2 It has caught FIVE levers

1. **The Flory coil** — r=1.175 on ddG (apparently harmful) vs dG MAE 4.9650→3.9521. The coil sets
   `d(i,j) = b·|i−j|^ν`, a function of sequence separation ONLY, bit-identical between WT and
   mutant. **We were measuring the cancellation, not the lever.** (Later overturned again — §5.1.)
2. **BSA** — killed on r=−0.001 against the pre-training decoy loss, a metric already known not to
   predict ddG, computed on a column with **4 non-null rows of 100,246**. Void twice over.
3. **W5 burial** — its gate passes 18/18 with every assertion mechanical (shapes, byte-identity,
   hydropathy order); not one scores performance. It acts on dG, yet was scheduled against pooled ddG.
4. **Ligands** — a folded-state term, identical in WT and mutant away from the site.
5. **Our own model-selection metric** — `validate()` returns pooled dG PCC, which is scale-invariant
   and **provably cannot see** a change in predicted spread. 40 queued runs were unfalsifiable until
   per-protein a_p logging was added.

## 2.3 The second rule

**Always compute the degenerate baseline.** `--dg_length_norm n` looked like a 31% improvement and
was the model predicting nothing (§6.3). **And a correlation without its n is not a result** — the
"1/√N artifact" was quoted at −0.9938 and measured at −0.29.

---

# 3. WHAT WE LEARNED ABOUT a_p (THE SLOPE)

## 3.1 a_p = r · s — the slope is TWO numbers

For any least-squares fit, `a_p = r · s` where `s = std(pred)/std(true)`. **Exact in our data to
2.0×10⁻¹⁵ over 28 proteins.** Over 22 checkpoints: **a_p 0.364 = r 0.653 × s 0.504**.

| source of the gap to a_p = 1 | size | fixable by `--slope_weight`? |
|---|---|---|
| ranking error (1−r) | 0.347 | **NO** |
| spread compression (1−s) | 0.496 | **YES** |

**Ceiling: driving s→1 pins a_p to r.** The lever cannot reach 1.0 by construction.

## 3.2 `--slope_weight` — the ONLY proven lever

`gld_slope1.0_s42`, epoch 14:

    a_p  0.4975 → 0.7396   (+0.242)
    s    0.6342 → 0.9901   (+0.356)
    r    0.7955 → 0.7925   (−0.003, FLAT)

**All of the gain came from s, none from r** — exactly as the identity demands. Trajectory: r pinned
at 0.791–0.794 from epoch 4 across ten epochs while s climbs 0.67→1.02. **a_p is at 93% of its
ceiling.**

**The weight sweep has an interior optimum:**

| `--slope_weight` | s | a_p | r |
|---|---|---|---|
| control | 0.635 | 0.499 | 0.798 |
| 0.3 | 0.573 | 0.415 | 0.801 |
| **1.0** | **0.756** | **0.562** | 0.809 |
| 3.0 | 0.472 | 0.326 | 0.795 |

**Weight 3.0 OVERSHOOTS and is worse than no lever at all.**

**Its honest cost:** pooled gain only **+0.0116** on that arm (pooled PCC is dominated by b_p), and
**std(b_p) WORSENS 1.603 → 2.359**. **The arm trades offset calibration for slope calibration.**
Currently **n=1** — seed replication is now running (§12).

## 3.3 a_p is NOT a statistical artifact — attenuation dead by 200×

The obvious objection: a_p ≈ 0.5 is the textbook signature of regression attenuation under noisy
labels, in which case no architectural lever could fix it.

Per-mutation uncertainties found in `data/ThermoMPNN/mega_test.csv` (an earlier attempt searched the
wrong file and reported 0/28). **28/28 of our test proteins join.**

    median 95% CI = 0.1007  →  σ = 0.0327 kcal/mol   (n = 28,312)
    median var(true ddG) = 0.7178
    attenuation predicts a_p = 0.7178/(0.7178 + 0.0327²) = 0.9985

**Measured a_p ≈ 0.50 against a predicted 0.9985.** Noise would need to be **~200× larger**.
**a_p is a genuine model failure.**

## 3.4 WHERE the ranking error lives: burying hydrophobics

Per-destination-residue slope, 10 checkpoints, ~26,000 matched mutations each:

| destination | KD | slope | spearman |
|---|---|---|---|
| N | −3.5 | **0.565** | 0.653 |
| K | −3.9 | 0.537 | 0.648 |
| R | −4.5 | 0.528 | 0.649 |
| G | −0.4 | 0.506 | 0.616 |
| A | +1.8 | 0.419 | 0.567 |
| M | +1.9 | 0.340 | 0.461 |
| **C** | +2.5 | **0.267** | 0.356 |
| **L** | +3.8 | **0.282** | 0.388 |
| I | +4.5 | 0.311 | 0.412 |

    corr(slope, Kyte-Doolittle)    = −0.734   sign-consistent 10/10
    corr(spearman, Kyte-Doolittle) = −0.740

**Mutations TO hydrophobic residues are compressed roughly twice as hard.** And the decisive detail:
**those two correlations are essentially identical**, so **rank accuracy degrades in lockstep**.

**This is LOST INFORMATION, not a rescalable calibration error.** No affine correction can help.
It is the (1−r) half of §3.1 given a chemical identity.

**⚠ MECHANISM CORRECTED.** The original claim was side-chain *bulk*. **That is refuted by its own
shuffle test:** `corr(slope, volume) = −0.372` fails (P=0.069) while `corr(slope, KD) = −0.765`
passes (P=0.0000). Cross-validated R²: **volume 0.031 vs hydropathy 0.666 vs transfer free energy
0.753.** **The mechanism is hydrophobic transfer free energy, not steric packing.**

Still consistent with the data limitation: only **4 backbone atoms per residue (N, CA, C, CB)** are
stored — no side chains.

## 3.5 Exposure predicts the slope

`mean_rel_SASA` vs a_p: **r = +0.714**, replicated across all 10 eval CSVs (mean 0.624,
sign-consistent 10/10). The only one of 90 tested correlations to clear Bonferroni on BOTH Pearson
and Spearman.

**It survives the length confound:** partialling out chain length gives **+0.6623, significant in
10/10**, while length alone is −0.2149 and significant in **0/10**.

## 3.6 The anchor suppresses the coupling

    anchor 0.3 → r = 0.587    anchor 1.0 → r = 0.471    anchor 3.0 → r = 0.400
    unanchored sigma seeds    → r = 0.63–0.68

**Factor A is already suppressing the exposure/slope coupling.** Three points is a trend, not a law.

---

# 4. WHAT WE LEARNED ABOUT b_p (THE OFFSET)

## 4.1 b_p is real, separable, and learnable

**A pure random-offset null CANNOT reproduce our numbers.** Grid-searching (noise, slope) over 525
cells, the closest it gets to our triple is **L2 = 0.118**, failing structurally: it **over-delivers**
on offset removal (+0.075, since a meaningless constant is perfectly removable), **under-delivers**
on per-protein PCC (−0.086), and needs slope **1.25** (expansion) where we measure **0.499**.

**ICC(b_p) = 0.898** over 28 proteins × 5 seeds (seed-only ~0.95–0.97). Only SD 0.503 kcal/mol is
unpredictable noise, so **b_p is genuinely learnable in principle** — best achievable pooled PCC
from offset correction ≈ 0.705 against the 0.718 oracle.

## 4.2 What b_p is NOT — EIGHT hypotheses tested and rejected

| hypothesis | result |
|---|---|
| label noise / attenuation | **DEAD by 200×** (§3.3) |
| chain length | corr(N, b_p) = **+0.0252** over 12 checkpoints |
| `--dg_length_norm` | **DEGENERATE** (§6.3) |
| embedding norm / 1/√N | claimed −0.9938, **measured −0.29** (n=28 threshold 0.392) |
| pre-training distribution shift | r = **+0.043**, p = 0.829 |
| train/test mean gap (Ofir's mechanism) | gap **−0.1034**, p = 0.588 |
| a feature-based corrector | LOPO 0.6049 vs raw 0.5994; held-out R² NEGATIVE on 8/10 |
| **the reference state (the coil)** | **REJECTED — §5.1** |

**b_p has survived every explanation tested. This is the single largest open question (§13).**

## 4.3 Burial predicts the offset

`frac_buried_rel_lt_0.25` vs the WT error: **mean r = −0.475, sign-consistent 10/10**, significant
8/10. This corrects an earlier "the signal is on a_p, NOT b_p" claim, which was premature.

**One structural axis acts on BOTH channels:** exposure → a_p (+0.624), burial → b_p (−0.475).

## 4.4 The corrector fails — a publishable negative

    raw pooled                    0.5994
    LOPO feature-predicted offset 0.6049      (+0.0055)
    oracle offset removed         0.7119      (+0.1125)

**95% of the oracle gain does not survive held-out prediction.** Held-out R² NEGATIVE on 8/10;
permutation null significant on 0/10; with the ridge penalty fixed the gain goes negative.

**The mechanism is the useful part:** the **top-2 |b_p| proteins carry 78%** of the oracle gain, and
**neither is a structural outlier** — there is no signature to regress on.

**A metric correction that matters:** Pearson is **shift-invariant**, so subtracting a CONSTANT
changes pooled PCC by exactly zero. **The null a per-protein corrector must beat is the RAW number**,
not a mean-offset baseline — the latter would manufacture a fake +0.011.

---

# 5. THE TWO BIGGEST REVERSALS

## 5.1 The coil result was MISREAD — and the dG arm is rejected

The most-cited number in the project was "coil improves dG MAE 4.9650 → 3.9521, our best b_p lever".

**Verified on the raw per-protein data:**

| condition | MAE | mean(b_p) | **std(b_p)** |
|---|---|---|---|
| base | 5.0751 | **−5.0751** | **0.9967** |
| **coil_fixed_b** | **4.0622** | **−4.0622** | **1.1231** |

**ALL 28 of 28 proteins are under-predicted in every condition.** When every residual shares a sign,
`MAE ≡ |mean(b_p)|` — **the metric was measuring MEAN BIAS, not dispersion.** And **dispersion went
the WRONG WAY: 0.9967 → 1.1231, +12.7% worse.**

**A uniform shift is exactly what Pearson is invariant to.** It cannot be a b_p lever at all.

**The trained arm confirms it.** `gld_dg_coil_s42` at epoch 14 vs control: dG MAE **2.40 vs 1.30**,
per-protein PCC **0.315 vs 0.731**, a_p **0.085 vs 0.498**, pooled **0.221 vs 0.564**.

std(b_p) fell 40%, which looks like a win — **it is the degenerate baseline again:**

    epoch  0: std(b_p) = 0.9410,  a_p = 0.0253
    epoch 14: std(b_p) = 0.9656,  a_p = 0.0852

**FLAT from epoch 0, before the model learned anything**, sitting 4.9% from the known attractor
`std(true WT dG) = 0.9208`.

**Caveat:** `ref_dg_seed42/` is EMPTY, so there is no matched dG-loss control without the coil. The
verdict rests on the **epoch-0 evidence, which is internal to the run.**

## 5.2 Factor D is decisive — 14.4σ

| arm | n | pooled | sd | a_p |
|---|---|---|---|---|
| **D0 (coil)** | 6 | **0.5939** | **0.0091** | **0.5736** |
| **D1 (`--unfolded_emb zero`)** | 6 | **0.1579** | 0.1171 | **0.0136** |
| seed reference | 5 | 0.5798 | 0.0303 | — |

**D0 − D1 = 0.4359 = 14.4 seed-sigmas.** D0's own spread is **3× tighter** than seed noise; D1's is
an order of magnitude wider, and one cell is anti-correlated. **a_p separates by a factor of 42** —
an a_p of 0.014 means the model is **flat**, not merely worse.

**This reverses the W0 decision.** W0 chose D1 on a frozen-checkpoint variance decomposition —
neither dG nor ddG performance. **Trained end to end it removes the offset by removing the signal.**

---

# 6. LEVERS: BUILT, GATED, AND WHAT EACH IS WORTH

## 6.1 Inventory — 12 levers, all wired and gated

| lever | flag | acts on | verdict |
|---|---|---|---|
| U2 unfolded embedding | `--unfolded_emb` | b_p | **D1 destroys the model** |
| U3/U4 Flory coil | `--flory_unfolded --coil_b` | dG | **REJECTED** (§5.1) |
| U5/U6 coil edges | `--coil_edges` | dG | untested alone |
| W5 burial | `--burial_features --burial_mode` | dG | **best a_p of any single flag (0.664)** |
| W6 descriptors | `--aa_descriptors` | a_p | **largely redundant** (§8) |
| W7 chain span | `--gcn_span N` | a_p | inside seed band |
| W7-edge | `--edge_features` | a_p | inside seed band |
| U10 bidirectional | `--gcn_bidir` | a_p | inside seed band |
| W9 metals | — | dG | **RETIRED**, superseded by W11 |
| W11 ligands | `--ligand_nodes` | dG | zero variance on our 28 |
| **C slope** | **`--slope_weight`** | **a_p** | ✅ **PROVEN at 1.0** |
| LORO | `--holdout_residues` | a_p | **arm collapsed — never tested** |

## 6.2 Gates — 15 suites, all green

**`gate_g4_cpu` is the guard protecting everything in flight**: it must print
**`baseline … dG=-0.0030 width=1092`** after ANY change. Re-run after every edit; **it has never
moved.**

Others: `gate_slope` (21/21), `gate_loro` (13/13), `gate_refrow`, `gate_noop` (30 checks),
`gate_resume` (42/42), `gate_u2`, `gate_w5` (18/18), `gate_w7`, `gate_w7_edge`, `gate_u3u4` (32/32),
`gate_u5u6`, `gate_w9`, `gate_w8` (51/51).

## 6.3 `--dg_length_norm` is DEGENERATE

std(b_p): none **1.3076**, /n **0.9050**, /sqrtn **0.9159** — an apparent 31% improvement.

**It is not.** /n converges to ~0.905 for EVERY checkpoint, including ones that START at 0.876.
The reason: **`std(true WT dG) = 0.9042`**. Dividing by N drives pred/N→0, so **b_p → −true_dG** and
its spread becomes the spread of the labels. **It deletes the prediction.**

## 6.4 `gate_u3u4`: the diagnosis worth keeping

It failed 2/32 and was written off as "known-false". Two tolerance fixes both failed, forcing a
measurement:

    off-diagonal agreement: 1.2e-06 (perfect)
    the four DIAGONAL channels (self-pairs): 1.38e-03

`get_dist_matrix` runs ONE cdist over (N·4,3); a self-distance goes through
`sqrt(x²+x²−2x·x)`, a catastrophic cancellation returning ~1e-3 instead of 0. **The gate was
comparing that artifact and calling it a layout error.** Channel 5 IS CA–CA (1·4+1).

## 6.5 Four silent no-op holes, closed

The signature failure mode: **code runs, prints a success line, feature never read.**
`gate_noop.py`, 30 checks.

1. **PEMGraphTransformer dropped inserted blocks silently — MEASURED.** 726 sentinel columns at
   offset 48 survive the reassembly with **0 reaching the node vector, and nothing raises.**
2. **`--ligand_nodes` could not fire in training** — and `gate_ligand` D.14 had **codified the bug**.
3. **`gate_open_alphabet` tested a /tmp fixture**; the real 25-row table is K=756 vs canonical 726.
4. That same gate hard-coded a retired mode name and **had stopped testing anything at all.**

## 6.6 The scoring pipeline: three bugs hid 12 finished cells

1. `evaluate.py:58` had `DEVICE='cuda'` with the CPU fallback **commented out** — every eval queued
   behind 41 trainings in the same 5-GPU pool.
2. `load_checkpoint` assumed a wrapper dict → `KeyError: 'model_state_dict'` on bare saves.
3. `torch.load` lacked `map_location`; tensors were saved on CUDA.

**It stayed hidden because `run_calib_eval.sh` prints `DONE … → abl_<tag>.csv` AFTER the crash.**
Scoring now runs on the **CPU partition at zero GPU cost**.

---

# 7. WHAT THE MODEL ACTUALLY SEES

## 7.1 The feature vector

    [ D(16) | Fb(32) | <new blocks here> | emb(1024) | one_hot(20) ]  = 1092

**D(16)**: pairwise distances, `atom_i·4 + atom_j` over N, CA, C, CB — **CA–CA is channel 5**.
**Fb(32)**: bonded/geometric. **emb(1024)**: ProtT5. **one_hot(20)**: what the mutation changes.

New blocks go BETWEEN Fb and emb so right-anchored slices never move. Only `fc1_gcn`/`fc1_gat` grow.

## 7.2 The architectural fact that constrains every feature design

**The GCN branch reads `x[:, :32]` only** — D(16) plus HALF of Fb. It never sees the embedding, and
**any block inserted at offset 48 is invisible to it.** W5/W6/W9/W11 reach the GAT branch alone.

## 7.3 The tensors are per-VARIANT

    prott5_embeddings/prott5_embedding_<k>.pt   [128, 52, 1024]
    one_hot_encodings.pt                        [1389, 52, 21]
    coords_tensor.pt                            [N, 4, 3]  ANGSTROM

The 21st one-hot column is dropped at `train.py:317` — **checked, not assumed.** Per-protein tensors
live at `/groups/keasar_group/casp15/meytav/protein_tensors/<P>/`, **not** in `data/`.

---

# 8. CHEMISTRY: WHY W6 IS LARGELY REDUNDANT

## 8.1 ProtT5 already extrapolates chemistry

ProtT5 is **contextual** (same residue at two positions: cosine **0.13–0.18**, nearly orthogonal)
and **recomputed per variant** (one mutation changes ALL 52 positions, so it does NOT cancel in ddG).

**Residue-disjoint probe** — fit on 19 residue types, predict the held-out **20th**:

| property | held-out R² |
|---|---|
| **hydropathy** | **+0.704** |
| charge | +0.511 |
| volume | +0.421 |
| identity | accuracy **1.000** (chance 0.050) |

**ProtT5 predicts the hydropathy of a residue type it has NEVER seen.** A descriptor block is a
deterministic function of identity, so **W6 cannot add information** — only regularisation or
coverage of residues absent from ProtT5's training data.

## 8.2 Reconciliation with Ofir Ezrielev's thesis (56 pages, read in full)

**Three things in our build were WRONG and are corrected in source:**

1. **PCA-16 is NOT his method.** He does no dimensionality reduction — "PCA"/"SVD" appear NOWHERE.
   **Our PCA is ours.**
2. **`ignore_3D=True` contradicts his 1826** (= 1613 2D + 213 3D, coherent only if 3D was requested).
3. **15-of-20 is not a faithful scaling of 40-of-58** (0.690 → 13–14), and **he gives no
   justification for 40 at all.**

**HE CONCEDES THE LANGUAGE-MODEL OBJECTION** (p.31): *"Physicochemical properties of AAs are
implicitly represented in these datasets … and accordingly, implicitly represented in the
embeddings."* **His argument is COVERAGE ONLY.** Our 28 are canonical, so that gap does not exist here.

**He found our offset problem independently:** his leave-one-amino-acid-out RMSE tracks the
train/test mean gap while r stays flat.

**Histidine SMILES bug:** the plan doc had a **4-carbon, 1-nitrogen ring** (C7H10N2O2) instead of
imidazole (C6H9N3O2). Invisible to a gate testing only L, D and F.

## 8.3 The LORO test NEVER RAN

`gld_loroW_desc` **collapsed to a constant** — RMSE frozen at exactly **2.541** for ~11 epochs, PCC
oscillating about zero. **Cause: descriptors were centred but never scaled (~13.8× the one-hot
energy).** Technical failure, unrelated to the hypothesis.

**Reporting "desc did not beat one-hot, as predicted" would be confirming a prediction with a broken
arm.** Re-running now with `mordred_pca16_only`, which REPLACES one-hot instead of appending.

---

# 9. BIOLOGY: WHAT IS AND IS NOT MEASURABLE

## 9.1 The hard constraint

**All 28 test proteins are single-chain, ligand-free, metal-free monomers of 43–72 aa.**
`n_chains == 1`, `n_het_residues == 0`, `n_metal_residues == 0`, `interchain_BSA == 0.0` for **every
one**.

**Ligands, complexes and interface area have ZERO VARIANCE here.** That is **"not measurable"**, NOT
"no effect".

## 9.2 The het-code composition

205,648 occurrences over 8,187 codes:

| class | share |
|---|---|
| **crystallization additives** (SO4, GOL, EDO, NA, CL) | **27.4%** |
| metal ions | 18.9% |
| **real cofactors/substrates** | **6.6%** |
| **modified residues** (MSE, SEP, TPO) | **6.3%** |

**Only 6.6% are real biological ligands.** And **MSE is selenomethionine — an amino acid IN THE
CHAIN** — so counting it as a ligand is both wrong and a double-count.

**A naive ligand feature would learn crystallography, not biochemistry.**

---

# 10. THE 100k CATALOGUE

## 10.1 Retired for every join-based purpose

**It has NO SEQUENCE COLUMN** — a permanent, in-principle result. Joins: **0/226** training PDB ids,
**0/21** test, **0/340** by protein_id, against 66,945 catalogue ids. **This kills the plan to escape
n=28 via the training set.**

## 10.2 Dead columns that look alive

`Ligands_x` **4 of 100,246** · `BSA_Numeric_x` **4 unique** · `BSA_Percentage` 5 · `lossg` constant ·
`Global Symmetry` constant. **This is where the −0.001 BSA null came from.**

## 10.3 The one live use: distribution shift

| | catalogue | our 32–74 aa regime |
|---|---|---|
| median length | 477 | 55 |
| **share of pre-training RESIDUES** | — | **0.1206%** (1 in 830) |
| SOLUTION NMR | 8.6% | **77.4%** |
| Monomer | 31.9% | **88.2%** |

**Our entire domain was one residue in 830 of pre-training.** But **NOT the cause of b_p**:
r = **+0.043**, p = 0.829; nothing survives Bonferroni over 25 features.

---

# 11. DATA INTEGRITY

## 11.1 2K5H's reference row was a MUTANT

`2K5H.csv` (4,107 rows) **concatenates THREE backgrounds**; the wrong one sorts first:

    row 0    : 2K5H.pdb_G11S   dG=1.723086   mut_type=wt
    idx 2738 : 2K5H.pdb        dG=4.805470   mut_type=wt   ← the TRUE wild type

Both are labelled `wt` because G11S genuinely is the WT *of its own background*. **Shift: +3.0824
kcal/mol** on all 1,125 labels.

**The control:** row-0 percentile within each protein — 26 of 28 sit at the **62nd–99th**; 2K5H at
**14.6%**.

**Corrected copy in `data_fixed/`** (the source tree is read-only). **`gate_refrow.py`** now counts
distinct WT backgrounds, which is the decisive test — the percentile alone false-flagged
`r18_3_TrROS_Hall`, which is **CLEAN**.

## 11.2 The same bug produced a second false lead

"**26,315 double mutants**" are all `2K5H.pdb_G11S_<mut>` — single mutations on a background.
**There are ZERO true double mutants.** **One malformed file produced two false leads.**

## 11.3 Everything else is clean

Labels **bit-identical** to the independent ThermoMPNN benchmark (28,312 rows). No second malformed
file. `gate_refrow` failures are **100% attributable to 2K5H alone**.

---

# 12. INFRASTRUCTURE

## 12.1 The two GPU lanes

| | public | golden |
|---|---|---|
| flags | `--qos normal` | **`--partition=rtx6000 --qos keasar`** |
| cap | **5 per USER** (`gpu-part` QOS) | **8 per ACCOUNT** (MaxTRESPA) |
| preemption | — | `PreemptMode=requeue` |

**Both parts of the golden invocation are required.** `--qos keasar` alone is rejected; the
`rtx6000` partition has `DenyQos=normal`. **The pools are genuinely separate.**

**A lua job_submit plugin rewrites any GPU request onto `gpu`** — which is why the site rule says
never to select a GPU by partition name.

## 12.2 The autopilot

Runs the S0–S9 state machine. **Now a SLURM CPU job (7-day limit)** rather than a login-node process
— it died repeatedly with `rc=143`.

**Two bugs fixed:** `load_state` only fell back when the file was ABSENT, so `{}` → `KeyError`; and
**the wrapper broke its loop on `rc=1`, which is ALSO what Python returns for an uncaught
exception**, so any crash killed it permanently.

## 12.3 Standing site rules, each paid for in wasted runs

- `export WANDB_MODE=disabled` before EVERY submission — one omission destroyed a 7h53m run.
- `--gres=gpu:rtx_6000:1`, **never select a GPU by partition name**.
- **A finished training run is NOT a result.** Track `ls eval_results/abl_*.csv | wc -l`.
- Tensors were saved on CUDA: use `map_location`, `weights_only=False`.
- `/tmp` is per-login-node; use `/home/nissimb/`.
- `run_calib_eval.sh` calls `python3`, which resolves to the SYSTEM python without pandas unless the
  env's bin is first on PATH — and `conda activate` fails in a detached shell.

## 12.4 Resume — what is and is not recoverable

**Verified by reading the checkpoint files, not by assumption:**

| checkpoint format | contents | resumable? |
|---|---|---|
| **wrapper** | model + **optimizer** + **scheduler** + epoch | ✅ **100% faithful** |
| **bare state_dict** | 80 tensors only | ❌ **`--resume` REFUSES** |

`--resume` refuses a bare checkpoint rather than silently producing a different trajectory
(`--resume_allow_partial_state` overrides, but **Adam moments and the LR schedule reset, so the run
is NOT equivalent**).

**Of the partially-trained cells: 3 are wrapper-format (epochs 1–3, so little is saved) and 2 are
bare at epoch 9.** Resume is therefore **not the throughput lever here** — filling idle cards is.

## 12.5 Job history — not everything ran

    COMPLETED 30 | PENDING 15 | RUNNING 5 | CANCELLED 1 | PREEMPTED 1

**52 factorial trainings submitted, only 31 reached epoch 14.**

---

# 13. WHERE WE STAND, AND WHAT IS OPEN

## 13.1 Answered — 17 conceptual questions

**Only TWO produced a gain:** `--slope_weight` (+0.058 pooled, a_p +0.244) and **requiring D0**
(14.4σ). **Eleven were rejections** — each closed a direction that would have cost GPU-months.

## 13.2 Open

| # | question | why it matters |
|---|---|---|
| **Q1** | **What IS b_p?** | 8 explanations rejected; the oracle says **+0.14 is available** |
| Q2 | Does side-chain / transfer-free-energy information fix the ranking deficit? | §3.4 — the largest deficit, with a clear architectural fix |
| Q3 | Does `--slope_weight` hold across seeds? | the only proven lever, **n=1** — **running now** |
| Q4 | Does the descriptor hypothesis survive a working test? | the arm **collapsed technically** — **re-running now** |
| Q5 | Can the slope arm's b_p cost be avoided? | it trades offset for slope (std(b_p) 1.60→2.36) |
| Q6 | Is factor A estimable? | **perfectly aliased with seed AND epoch** — needs a clean design |
| Q7 | Do W5 and slope ADD? | both move a_p by different routes — **running now** |

## 13.3 What is running (all 8 golden cards, saturated)

| job | tests | priority |
|---|---|---|
| `gld_slope1.0_s1/s2/s3` | **seed-replicate the only proven lever** | P1 |
| `gld_loroWdesc2_s42` | the LORO test that never ran, scale clash removed | P2 |
| `gld_w5_dg_s1/s2` | replicate the best single-flag a_p (0.664) | P3 |
| `gld_slope_w5_s42` | **do the two a_p levers add?** | P4 |
| `gld_slope_anchor_s42` | slope + anchor combination | P4 |

Plus **5 public cards** on factorial cells and **2 CPU jobs** (autopilot + scorer for 14 unscored
D0 cells).

## 13.4 The honest summary

**We improved the score from 0.5635 to 0.6382 (+0.075), and we know exactly why:** the slope lever
works and is understood mechanistically (`a_p = r·s`, all gain through s).

**We know what the ceiling is (0.7534) and that it is gated by b_p, which has resisted eight
explanations.**

**And we removed more than we added:** eleven rejections, two data-integrity bugs, five mis-scored
levers, four silent no-ops, and three scoring bugs that had hidden twelve finished runs. **The
negative results are the more valuable half of this work.**

---

# 14. DID IT FAIL, OR DID IT NOT RUN? — the triage table

**Read this before trusting any negative result.** A lever that failed *because the science is
wrong* is settled. A lever that failed *because the execution broke* is still open. They look
identical in a results table and mean opposite things.

| lever / experiment | design correct? | execution correct? | verdict | re-run? | files |
|---|---|---|---|---|---|
| **`--slope_weight 1.0`** | ✅ | ✅ | ✅ **WORKS** — a_p 0.496→0.740, all via s | replicating seeds now | `02_findings/SLOPE_ARM.md`, `SLOPE_OBJECTIVE.md` |
| **Factor D (D0 vs D1)** | ✅ | ✅ | ✅ **SETTLED 14.4σ** — D1 destroys the model | **no** | `02_findings/FACTORIAL_ANALYSIS.md` |
| **W5 burial on dG** | ✅ | ✅ | 🟡 best single-flag a_p (0.664), **n=1** | replicating now | `02_findings/W5_ON_DG.md` |
| **The Flory coil / dG arm** | ❌ **design flaw** | ✅ | ❌ **REJECTED** — the motivating MAE measured *mean bias*, not dispersion; std(b_p) rose | **no** | `02_findings/DG_ARM.md` |
| **`--dg_length_norm`** | ❌ **degenerate** | ✅ | ❌ arm deletes the prediction | **no** | §6.3 |
| **LORO descriptors** | ✅ | ❌ **BROKE** — collapsed to a constant, RMSE frozen at 2.541 | ⚠️ **NEVER TESTED** | **YES — running now** | `02_findings/LORO_RESULT.md` |
| **W7span / U10 / W7edge** | ✅ | ✅ | ❌ all inside the ±0.060 seed band | not a priority | `02_findings/INFO_LEVERS.md` |
| **b_p feature corrector** | ✅ | ✅ | ❌ 95% of the gain does not survive held-out | **no** | `02_findings/OFFSET_CORRECTOR.md` |
| **Severing experiment** | ✅ | ⚠️ **OOD guard fired** | ⚠️ **UNINTERPRETABLE** — f_resid invalid; corr(E_u,E_f)=0.543 still informative | needs a *retrained* severed model | `02_findings/SEVERING_RESULT.md` |
| **Factor A (anchor)** | ❌ **confounded** | ✅ | ⚠️ **NOT ESTIMABLE** — perfectly aliased with seed AND epoch | needs a clean design | `02_findings/FACTORIAL_ANALYSIS.md` |
| **Ligands (W11)** | ✅ | — | ⚠️ **UNTESTABLE HERE** — zero variance on 28 monomers | needs a ligand-bearing set | `03_levers/LIGAND_SPEC.md` |
| **W6 descriptors (theory)** | ✅ | ✅ | ❌ **redundant** — ProtT5 predicts held-out hydropathy at R²=0.704 | **no** | §8.1 |
| **Side-chain "bulk" mechanism** | ❌ **wrong mechanism** | ✅ | ❌ refuted by its own shuffle test (volume R² 0.031 vs hydropathy 0.666) | mechanism is transfer free energy | `02_findings/SIDECHAIN_DESIGN.md` |

## 14.1 The three that are NOT settled

**These are the only negatives that could still turn positive:**

1. **LORO descriptors** — the arm collapsed technically (descriptors centred but never scaled, ~13.8×
   the one-hot energy). **Re-running now** as `gld_loroWdesc2_s42` with `mordred_pca16_only`, which
   *replaces* one-hot instead of appending, removing the scale clash.
2. **The severing experiment** — the OOD guard fired, so `f_resid` must not be used. Needs a
   *retrained* severed model, not a frozen-checkpoint ablation. **Script ready:**
   `scripts/severing.py`, spec in `03_levers/SEVERING_READY.md`.
3. **Factor A** — aliased with seed and epoch, so its apparent effect is not attributable. Needs a
   design where anchor weight varies with seed and epoch held fixed. **No file yet — this one still
   needs planning.**

## 14.2 What is ready to run, and where

| work | ready? | where |
|---|---|---|
| Seed replication of the slope arm | ✅ **running** (3 seeds) | `scripts/fill_golden.sh` |
| LORO re-run with the scale fix | ✅ **running** | `scripts/fill_golden.sh` |
| W5-on-dG replication | ✅ **running** (2 seeds) | `scripts/fill_golden.sh` |
| slope + W5 / slope + anchor combinations | ✅ **running** | `scripts/fill_golden.sh` |
| Scoring the 14 unscored D0 cells | ✅ **running** (CPU) | `scripts/run_cpu_evals.sh` |
| Severing, retrained | 📄 script ready, **not submitted** | `scripts/severing.py` |
| Side-chain / transfer-free-energy feature | 📄 **design only**, not implemented | `02_findings/SIDECHAIN_DESIGN.md` |
| Clean factor-A design | ❌ **nothing yet** | — |
| Re-score 2K5H against the true WT row | ❌ only the constant-shift approximation exists | `data_fixed/`, `scripts/fix_2k5h.py` |

## 14.3 The one-line summary

**Settled and positive:** the slope lever, and requiring D0.
**Settled and negative:** the coil, length-norm, the corrector, the information levers, W6.
**NOT settled — do not quote as negatives:** LORO, severing, factor A.
