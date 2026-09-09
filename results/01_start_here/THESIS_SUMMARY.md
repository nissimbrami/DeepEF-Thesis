# DeepEF — thesis summary

M.Sc., Nissim Brami, supervised by Prof. Chen Keasar, Ben-Gurion University.
Every number below states its basis. Canonical basis unless noted: **27 test proteins
(2K5H excluded), ddG metric, validation-selected epoch.**

---

## 1. The headline

| quantity | value |
|---|---|
| per-protein ddG PCC (median) | **0.8034** |
| pooled ddG PCC | **0.6382** |
| control (no lever) | 0.5635 / 0.7929 |
| offset-removal oracle | **0.7534** |
| seed noise band | **±0.060** |

**The problem was never the per-protein ranking.** It was 0.79 at the control and 0.80 at best —
a difference of 0.0105, six times smaller than the seed band. **The entire gap is *pooled*:
calibration between proteins.**

Two things beat the noise band, out of everything tested: `--slope_weight 1.0` (+0.058) and
requiring factor D0 (3.9σ). **Eleven hypotheses were rejected.**

---

## 2. The largest result: measure the offset, do not predict it

`pred ≈ a_p·true + b_p`. The per-protein offset `b_p` is the whole remaining error.

**Ten attempts to predict it failed** — length, length-normalisation, embedding norm, distribution
shift, train/test mean gap, LOPO ridge on 22 features, the reference state, mean-pooled ProtT5
(LOPO R² = −0.101), and structural features in ddG space (R² = −0.073, −0.004, −0.232, all
negative, all correlations below the n=27 threshold of 0.381).

**Measuring it works:**

| k measured mutations | pooled | % of recoverable gain |
|---|---|---|
| 0 | 0.6210 | — |
| **1** | 0.5644 | **−45%** ⚠ |
| 3 | 0.6643 | 34% |
| **5** | **0.6967** | **60%** |
| 8 | 0.7146 | 74% |
| 20 | **0.7334** | 89% |
| oracle | 0.7468 | 100% |

**k=1 is actively harmful** on all three checkpoints, and the mechanism is measured: within-protein
error sd 0.6801 against between-protein offset sd 0.4615 gives noise/signal 1.47, so one sample is
noisier than the signal. Noise falls as 1/√k, break-even at **k = 2.2** — exactly where the table
turns positive. **Never use k < 3; k = 5 is the practical recommendation.**

**This is FEW-SHOT, not zero-shot**, and must be reported as such. The scenario is a wet lab
measuring a handful of mutants before a campaign. **Both numbers belong in the thesis:
0.6382 zero-shot and 0.7334 with 20 measured.**

Robustness, stated honestly: dropping the two highest-offset proteins leaves 42% of the gain.
The effect is real but concentrated.

---

## 3. Corrections to the record

**`b_p` is 96.3% one global constant — in dG space only.** In **ddG space**, which is where the
headline is reported, the global constant is only **38–55%**, and proteins are mostly
*over*-predicted (20/27, 22/27, 26/27), not under. Both numbers are right; they measure different
quantities. Every such statement must now name its space.

**The slope lever overshoots.** `a_p = r·s` exactly (max error 8.88e-16). Tracking `s`:
e0 0.686 → e4 0.948 → **e8 1.089** → e13 1.144. Perfect calibration is s = 1.0; the arm crosses it
between e4 and e8 and keeps going. **`a_p` rising past s = 1 is a defect reported as a gain.**
The canonical epoch is **e10** (a_p 0.782), not the e13 maximum (0.857) that was quoted.

**Slope and the measured offset do NOT compose.** At k=20 the arms converge and the ordering
scrambles: the best-slope arm finishes last, the worst ties for first.

---

## 4. Rejections, and which were really execution faults

**Five of seven "failed ideas" were execution or metric faults, not refuted hypotheses.**

| idea | what actually happened |
|---|---|
| LORO descriptors | **scale bug** — descriptors carried 13.8× the one-hot energy; the hypothesis was never tested |
| Flory coil | **wrong metric** — MAE ≡ \|global\| because n_under = 28/28, so it measured bias. On std(b_p) all 10 ν×b cells are worse than no coil, and the coil *induces* a length artifact (corr(b_p,N) −0.215 → −0.507 in ν) |
| b_p prediction | the **approach** was wrong, not any attempt |
| W7edge / W7span / U10bidir | **n=1 inside a noise band is not a rejection** — second seeds now queued |
| ligands / metals / complexes | **not refuted — unmeasurable.** Fetched all 21 PDB-coded proteins from RCSB: 0 ligands, 0 metals, **21/21 monomeric**. A constant-zero column contributes nothing to any gradient. A property of the benchmark, not of the idea |

**The arithmetic that killed a whole family:** any per-residue descriptor table is `one_hot @ T`,
a rank-1 projection of a block `fc1` already receives. That explains LORO, W6, and two of W12's
four columns.

---

## 5. What is queued

**43 golden-lane jobs.** The levers that survived the redundancy test:

- **W15 side-chain packing** — built from FASPR reconstructions of all 368 proteins.
  **Passes the test that killed the others**: one-hot alone explains only 0.443 / 0.488 / 0.245 of
  sc_sasa / contacts / clash, so 33–76% is geometry residue identity cannot express.
  Gate 10/10, including that the block differs between WT and mutant and the difference is local
  to the mutated position.
- **HSE direction-aware burial** — already in the code, never once run.
- **`--pooled_corr_weight`** — a loss on the pooled cross-protein correlation, i.e. the exact
  reported metric. **It was 0 in every run ever done.**
- **`--readout attention/gated`** — the only lever touching the sum that manufactures `b_p`.
- **W12 side-chain chemistry, K20 unfolded ensemble, K21 distogram head.**

---

## 6. Honest expectation

`P(pooled > 0.75) ≈ 25%`. MegaScale is a proteolysis assay with ~0.3–0.5 kcal/mol resolution and
censoring above ~4.5, so **a perfect model cannot exceed the noise in its own labels.** Our
measured oracle of 0.7534 sits at that boundary — independent support for the estimate.

---

## 7. What the thesis rests on

**Five findings, none of them a correlation coefficient:**

1. The per-protein offset is **not predictable** from structure (ten methods) but **is measurable**
   — ~5 mutations recover 60% of the recoverable gain at no GPU cost.
2. A **data bug** (2K5H's reference row is a mutant) accounted for a large part of the apparent gain.
3. **Calibration levers overlap rather than compose.**
4. The compression is **desolvation on burial, not bulk** — volume LORO R² 0.031 vs hydropathy
   0.666 — with the arithmetic showing why no per-residue descriptor can fix it.
5. **Most recorded failures were execution faults.** The three rules that caught them — score a
   lever on the metric it acts on, always compute the degenerate baseline, verify an artifact and
   never a DONE message — found more real defects than any positive result produced gain.

**The number is the appendix.**
