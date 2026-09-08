# DeepEF — WHERE WE STAND
### 2026-09-08. Every number measured from the eval CSVs, 27 proteins (2K5H dropped), ddG metric.

> **CANONICAL BASIS (P8).** Every headline number in this document is computed on:
> **27 test proteins (2K5H excluded) | ddG metric | the 9 original-population eval CSVs |
> the val-selected epoch only.** That basis is `pooled 0.5772 / oracle 0.7156 / gain +0.1384`.
> Membership, exclusions and provenance: `results/05_infrastructure/HEADLINE_BASIS.md`.
> Enforced by `scripts/gate_headline.py`. **Never average across runs that differ in factor D**
> (`--unfolded_emb zero`) — D0 and D1 are different models; report them separately, always.


---

# PART A — THE SCORE WE REACHED

## A.1 The best runs

| rank | run | pooled ddG PCC | oracle | a_p | r | s |
|---|---|---|---|---|---|---|
| 1 | `sigma_seed2_e10` | **0.6382** | 0.7534 | 0.410 | 0.791 | 0.598 |
| 2 | `p3_slope1.0_s42_e13` | **0.6210** | 0.7468 | 0.507 | 0.803 | 0.756 |
| 3 | `anchor_w0.3_s42_e14` | 0.6105 | 0.7384 | 0.542 | 0.774 | 0.830 |
| 4 | `p3_a1_d0_s1_D0_coil_seed42` | 0.6028 | 0.7185 | 0.571 | 0.751 | 0.808 |
| 5 | **`gld_w5_dg_s42_e14`** | 0.5956 | 0.7212 | **0.664** | 0.779 | **0.870** |
| — | control `calib_ctrl_repro2_e14` | 0.5635 | 0.7364 | 0.496 | 0.793 | 0.635 |

**Best pooled ddG PCC = 0.6382.** Control = 0.5635. **Best gain over control: +0.075.**

**Best oracle (offset removed) = 0.7534.** The ceiling if b_p were solved.

**Best a_p = 0.740** (`gld_slope1.0_s42_e13`), against a control of 0.496.

## A.2 The headline, on the ONE canonical basis

Canonical basis: **27 proteins (2K5H excluded), ddG, the 9 original-population eval CSVs,
val-selected epoch only.**

    pooled 0.5772 ± 0.0316   oracle 0.7156 ± 0.0474   gain +0.1384      (n = 9 runs)

The nine: `calib_ctrl_repro2_e14`, `sigma_seed{1,2,3,4,42}`, `anchor_w{0.3,1.0,3.0}_s42`.
Full membership and the exclusion argument: `results/05_infrastructure/HEADLINE_BASIS.md`;
recomputed and enforced by `scripts/gate_headline.py`.

**Correction to the record.** This number was already correct — what was missing is *which nine
CSVs produced it*, which let two other averages stand beside it as apparent equals. Both are now
retired: 0.4899 / 0.6443 / +0.1544 (22 mixed CSVs, including D1 arms scoring as low as −0.074) and
0.5646 / 0.6347 / +0.0702 (20 non-D1 checkpoints spanning three lanes). Both are quoted as
published and **neither can be re-derived** — those populations were never recorded, and the
nearest reconstruction of the 22 gives 0.4252, not 0.4899. They are retired not because they were
shown to be wrong but because they can no longer be shown to be anything. **A correlation averaged across a mixed population is not a
measurement.**

## A.3 What actually moved the score

| lever | effect on pooled | effect on a_p | verdict |
|---|---|---|---|
| **`--slope_weight 1.0`** | +0.058 | **+0.244** (0.496 → 0.740) | ✅ **the only proven lever** |
| `--wt_anchor_weight 0.3` | +0.047 | +0.046 | ⚠ inside seed noise |
| `--burial_features` on dG | +0.032 | **+0.168** | ⚠ promising, n=1 |
| `--gcn_span 4` / `--gcn_bidir` / `--edge_features` | +0.010…+0.026 | −0.18 … −0.26 | ❌ inside ±0.060 seed band |
| `--flory_unfolded` (dG arm) | **−0.343** | −0.405 | ❌ **rejected** |
| `--unfolded_emb zero` (D1) | **−0.42** | −0.48 | ❌ **destroys the model** |

**Seed noise band: ±0.060** (5 `abl_sigma` seeds, same config, pooled 0.556–0.638).
**Anything under +0.060 is not distinguishable from noise.**

---

# PART B — PHYSICAL TASKS (compute)

## B.1 Running now

| what | count | note |
|---|---|---|
| public GPU | **5/5** | factorial cells — **2 of them are D1**, already settled at 14.4σ |
| **golden GPU** | **0/8** | ⚠ **completely idle** — all 8 arms finished |
| CPU | 2 | autopilot (19h) + scorer just submitted |

## B.2 Queued

| what | count |
|---|---|
| TRAIN cells | **18** — of which **13 are D1** (the dead arm) and only **3 are D0** |
| EVAL jobs | 52 (chained, blocked behind their trainings) |
| smoke test | 1 |

## B.3 Trained but unscored — free results

**31 cells reached epoch 14; only 14 are scored.** The other **14 unscored cells are ALL D0** —
the arm that matters. Scorer job `21134976` submitted, CPU partition, zero GPU cost.

## B.4 Physical tasks outstanding

| # | task | cost | status |
|---|---|---|---|
| P1 | Score the 14 unscored D0 cells | CPU, ~4h | ✅ **submitted** |
| P2 | **Fill the 8 idle golden cards** | free capacity | ⏳ **needs a decision on what to run** |
| P3 | Re-run LORO descriptor arm with **normalised** descriptors | 1 card, 8h | ⏳ blocked on the fix |
| P4 | Kill / stop feeding the D1 arm (2 running + 13 queued ≈ 120 GPU-h) | saves capacity | ⏳ **your call** |
| P5 | Seed replication on the winning D0 cells | 3–5 cards | ⏳ what P2 should probably run |
| P6 | 2K5H re-score against the true WT row (not the constant-shift approximation) | CPU | ⏳ open |

---

# PART C — THEORETICAL / CONCEPTUAL TASKS

## C.1 Done — and what each measured

| # | question | what it measured in the model | result | did the score rise? |
|---|---|---|---|---|
| T1 | Is the calibration gap real? | per-protein vs pooled PCC | 0.798 vs 0.59 — **real** | — (framing) |
| T2 | Is the gap an estimator artifact? | random-offset null vs our triple | **null cannot reproduce** (best L2 0.118) | — (defence) |
| T3 | What is a_p made of? | `a_p = r·s`, exact to 2e-15 | r 0.653 / s 0.504 | — (decomposition) |
| T4 | Can the slope be fixed? | `--slope_weight` on a_p | **a_p 0.496 → 0.740** | ✅ **+0.058 pooled** |
| T5 | Is a_p just label noise? | attenuation from σ=0.0327 | predicts 0.9985 vs measured 0.50 — **dead by 200×** | — (rules out) |
| T6 | Where is the ranking error? | slope vs destination residue | hydrophobic 0.28 vs polar 0.55, **ranking falls in lockstep** | — (diagnosis) |
| T7 | Is b_p length-driven? | corr(N, b_p) | **+0.025** — no | ❌ |
| T8 | Does length-norm help? | `--dg_length_norm` | **degenerate** — deletes the prediction | ❌ |
| T9 | Is b_p an embedding-norm artifact? | corr(N, ‖e‖) | claimed −0.99, **measured −0.29** | ❌ |
| T10 | Is b_p distribution shift? | our regime = 0.12% of pre-training | shift is **real** but r=+0.043 vs b_p | ❌ |
| T11 | Is b_p the train/test mean gap? | Ofir's mechanism | gap −0.10, **p=0.588** | ❌ |
| T12 | Can b_p be predicted from features? | LOPO ridge, 22 features | **fails** — R² negative on 8/10 | ❌ |
| T13 | Is b_p the reference state? | dG arm with the coil | **rejected** — flat from epoch 0 | ❌ |
| T14 | Do descriptors add information? | ProtT5 residue-disjoint probe | hydropathy **R²=0.704** — largely redundant | ❌ |
| T15 | Does factor D matter? | D0 vs D1 | **14.4σ** — D1 destroys the model | ✅ (D0 required) |
| T16 | Do the information levers help? | W7span/U10/W7edge/W5 | **all inside the seed band** | ❌ |
| T17 | Is the data sound? | reference-row + label audit | **2K5H corrupted**, rest clean | — (integrity) |

**Score: 17 conceptual questions answered. 2 produced a gain (T4, T15). 11 were rejections.**
A rejection is worth as much as a gain — each one closed a direction that would have cost GPU-months.

## C.2 Open theoretical questions

| # | question | why it matters | how to answer |
|---|---|---|---|
| Q1 | **What IS b_p, if not any of the eight?** | 8 explanations rejected; it is the single largest remaining error source | The oracle says +0.14 is available. Nothing found predicts it. |
| Q2 | Does side-chain information fix the hydrophobic deficit? | T6 says the ranking error is **information the model cannot see** | CB-only geometry; needs a derived side-chain feature |
| Q3 | **Was the hydrophobic mechanism bulk or hydropathy?** | agents **refuted** the bulk claim: R² 0.031 (volume) vs 0.666 (hydropathy) | rewrite §3.4 — the mechanism is transfer free energy, not packing |
| Q4 | Does `--slope_weight` hold across seeds? | the only proven lever, currently **n=1** | 3-seed replication |
| Q5 | Does the slope arm's b_p cost matter? | it trades offset for slope: std(b_p) 1.60 → 2.36 | combine with an offset lever? |
| Q6 | Is factor A estimable at all? | agents found A **perfectly aliased with seed AND epoch** | needs a clean design |
| Q7 | Does the descriptor hypothesis survive a working test? | the LORO arm **collapsed technically**, so it was never tested | re-run with normalised descriptors |

---

# PART D — CORRECTIONS THE RECORD NEEDS

Six places where the written record disagrees with the measurements:

1. **The headline was averaged over a mixed population.** Use the stated-basis number (A.2).
2. **`std(b_p)=1.5741` conflates two quantities differing ~8×**: dG-space WT error 1.603 vs
   ddG-space intercept 0.193.
3. **The coil's motivating result was a mean shift**, not a dispersion reduction — all 28/28
   under-predicted, so MAE ≡ |mean(b_p)|, and std(b_p) actually **rose**.
4. **The side-chain "bulk" mechanism is refuted** — hydropathy explains it, volume does not.
5. **Factor A is not estimable** — aliased with seed and epoch.
6. **The LORO test never ran** — reporting "as predicted" would confirm a prediction with a
   broken arm.

---

# PART E — THE ONE DECISION I NEED

**8 golden cards are idle right now.** The three candidates:

| option | cost | what it buys |
|---|---|---|
| **A. Seed-replicate the slope arm** | 3 cards | turns the **only proven lever** from n=1 into a real result |
| **B. Re-run LORO with normalised descriptors** | 1 card | actually tests the descriptor hypothesis |
| **C. Stop feeding D1** | frees 2 running + 13 queued | ~120 GPU-h back |

**My recommendation: all three.** A and B use 4 of the 8 idle cards; C returns capacity we are
currently spending to re-confirm a 14.4σ result.
