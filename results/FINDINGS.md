# DeepEF — FINDINGS OF RECORD

Every number here was computed or re-derived by the main session. Agent-reported figures that did
not reproduce are listed under "Claims that failed verification" rather than quietly dropped.

**Three qualifiers on every number: pooled-or-per-protein / dG-or-ddG / which split.**

---

## 1. THE CALIBRATION DECOMPOSITION

Per protein, `pred ~= a_p * true + b_p`.

| quantity | value | metric |
|---|---|---|
| std(b_p) | 1.5741 | dG-space WT error, 28 test proteins |
| a_p median | 0.4990 | ddG, control checkpoint |
| per-protein ddG PCC median | 0.798 | ddG |
| pooled ddG PCC | ~0.59 | ddG |

**The model ranks well within a protein and is miscalibrated between proteins.** That gap is the thesis.

### CORRECTED headline (after the 2K5H fix, see section 6)

| | pooled | offset-removal oracle | gain |
|---|---|---|---|
| **corrected, 22 CSVs** | **0.4899** | **0.6443** | **+0.1544** |
| (previously reported) | 0.5018 | 0.7391 | +0.2373 |

**The oracle is 0.644, NOT 0.739, and NOT the old 0.70-0.72.** The effect is real and substantial;
its size was inflated 35% by a data bug.

**The oracle is an ORACLE.** It subtracts a per-protein constant fitted on TEST labels. It licenses
"the ranking information is present, what is lost is calibration" and nothing stronger. A
feature-based corrector FAILED out of sample (LOPO 0.6049 vs raw 0.5994).

---

## 2. a_p = r * s — THE SLOPE IS TWO NUMBERS, NOT ONE

For any least-squares fit, `a_p = r * s` with `s = std(pred)/std(true)`. Over 22 checkpoints:

    a_p = 0.364    r = 0.653    s = 0.504

| source of the gap to a_p=1 | size | fixable by --slope_weight? |
|---|---|---|
| ranking error (1-r) | 0.347 | **NO** |
| spread compression (1-s) | 0.496 | **YES** |

**Ceiling: driving s to 1 gives a_p = r.** The lever cannot reach 1.0 by construction.

### The three slope arms confirm the mechanism

| --slope_weight | s | a_p | r |
|---|---|---|---|
| control | 0.635 | 0.499 | 0.798 |
| 0.3 | 0.573 | 0.415 | 0.801 |
| **1.0** | **0.756** | **0.562** | 0.809 |
| 3.0 | 0.472 | 0.326 | 0.795 |

**r is flat while s swings** — exactly as the identity requires. Weight 1.0 gives +0.063 over
control at no cost to r. **Weight 3.0 overshoots and is worse than no lever**, so the response has
an interior optimum rather than being monotone.

**Report a_p as two numbers:** the model ranks at r~0.80 and compresses the surviving spread to
s~0.64. Only the second half is a calibration problem.

---

## 3. WHERE THE RANKING ERROR LIVES: BURYING HYDROPHOBICS

Per-destination-residue slope, 10 checkpoints, ~26,000 matched mutations each:

| destination | slope | spearman |
|---|---|---|
| charged/polar (R,K,N,D,E,Q) | 0.53-0.57 | ~0.65 |
| hydrophobic (C,L,F,V,I,M) | **0.27-0.31** | **0.36-0.43** |

    corr(slope, Kyte-Doolittle)    = -0.734   sign-consistent 10/10
    corr(spearman, Kyte-Doolittle) = -0.740

**Ranking degrades in LOCKSTEP with slope, so this is LOST INFORMATION, not a rescalable
calibration error.** No affine correction can recover it.

**Mechanism:** the dataset stores only 4 backbone atoms (N, CA, C, CB) and NO side chains. Burying
a large hydrophobic is dominated by side-chain packing, which CB-only geometry cannot see. W and Y
are the telling exceptions (slopes 0.34/0.35 despite negative hydropathy), so the pattern tracks
**side-chain bulk**, not hydropathy alone.

**This argues for an architectural fix — side-chain information — not a calibration one.**

---

## 4. THE METRIC RULE (the most reusable result)

**ddG cancels anything identical between wild type and mutant.** A reference-state or whole-protein
lever must be scored on dG or b_p; a_p is within-protein and IS ddG-measurable.

**It has now caught FIVE levers:**

1. **The Flory coil** — r=1.175 "harmful" on ddG; **dG MAE 4.9650 -> 3.9521**, our best b_p lever.
2. **BSA** — killed on r=-0.001 against the pre-training decoy loss, a metric that cannot predict
   ddG, computed on a column with 4 non-null rows of 100,246.
3. **W5 burial** — gate passes 18/18 with every assertion mechanical; scheduled against pooled ddG
   while acting on dG.
4. **Ligands** — a folded-state term that cancels in ddG for any mutation away from the site.
5. **Our own model-selection metric** — validate() returns pooled dG PCC, which is scale-invariant
   and **provably cannot see what --slope_weight does.** 40 queued runs were unfalsifiable until
   per-protein a_p logging was added.

---

## 5. WHAT b_p IS NOT

Each tested and rejected by the main session:

| hypothesis | result |
|---|---|
| label noise (attenuation) | **DEAD by 200x**: sigma=0.0327 kcal/mol over 28,312 mutations predicts a_p=0.9985 vs measured 0.50 |
| chain length | corr(N, b_p) = **+0.0252** across 12 checkpoints |
| --dg_length_norm | **DEGENERATE**: /n converges to 0.905 = std(true WT dG), i.e. it deletes the prediction |
| embedding norm / 1/sqrt(N) | claimed -0.9938, **measured -0.29** (below the n=28 threshold of 0.392) |
| distribution shift | r = **+0.043**, p=0.829 |
| the train/test mean gap (Ofir mechanism) | gap -0.1034, **p=0.588** |
| a feature-based corrector | LOPO 0.6049 vs raw 0.5994; held-out R2 negative on 8/10 |

**b_p has survived every cheap explanation. What remains untested is the reference state**, where
the coil produced the only real movement, and gld_dg_coil is running now.

---

## 6. DATA INTEGRITY

**2K5H's reference row was a MUTANT.** 2K5H.csv concatenates three backgrounds and _G11S sorts
first; the true WT sits at index 2738. **Shift +3.0824 kcal/mol.** A constant per-protein label
shift IS a per-protein offset, so the oracle was rediscovering one we introduced. Corrected copy in
data_fixed/ (the source tree is read-only). gate_refrow.py now counts distinct WT backgrounds and
names the defect.

**The same bug produced a second false lead:** the "26,315 double mutants" are all
2K5H.pdb_G11S_<mut>, single mutations on a background. **There are ZERO true double mutants** for
any of the 28.

---

## 7. EMBEDDINGS: W6 IS LARGELY REDUNDANT

ProtT5 is **contextual** (same residue at two positions: cosine 0.13-0.18) and **recomputed per
variant** (one mutation changes all positions, so it does NOT cancel in ddG).

Residue-disjoint probe, fit on 19 residue types, predict the **20th**:

    hydropathy R2 = +0.704    charge R2 = +0.511    volume R2 = +0.421
    identity accuracy = 1.000 (chance 0.050)

**ProtT5 already predicts the chemistry of a residue type it has never seen.** A descriptor block is
a deterministic function of identity, so **W6 cannot be justified as adding information** — only as
regularisation or open-alphabet coverage. The thesis it comes from concedes this itself: the case
for descriptors is COVERAGE of residues absent from the training databases, and our 28 are all
canonical.

---

## 8. FIRST FACTORIAL RESULTS — FACTOR D IS DECISIVE

10 of 48 cells scored:

| arm | pooled ddG PCC | a_p |
|---|---|---|
| **D0 (coil)**, 5 cells | **0.58-0.62** | 0.37-0.72 |
| **D1 (unfolded_emb zero)**, 5 cells | **-0.08 to 0.31** | **0.005-0.020** |

**No overlap.** a_p differs by two orders of magnitude; D1 models are essentially flat. W0 chose D1
on the ddG metric because it collapsed var(E_u); **trained end to end it destroys the model**,
removing the offset by removing the signal. One D1 cell is anti-correlated (-0.078).

**The best cell has the highest a_p (0.716) but NOT the highest pooled PCC (0.583 vs 0.618)** —
the calibration thesis in one line, because pooled PCC is dominated by the offset while a_p
measures the compression.

---

## 9. CLAIMS THAT FAILED VERIFICATION

Recorded so they are not re-adopted:

- 1/sqrt(N) embedding artifact: **-0.29, not -0.9938**
- 26,315 double mutants: **zero real ones**
- --dg_length_norm shrinks b_p: **degenerate**
- The affine oracle 0.77-0.81: **not reproducible**; offset removal alone gives 0.644 corrected
- W5 burial vs b_p at n=19: r=-0.343, **not significant** (threshold 0.490)

**Two recurring failure modes:** a correlation quoted without its n, and an improvement quoted
without asking what the number would be if the model predicted nothing. **Always compute the
degenerate baseline.**
