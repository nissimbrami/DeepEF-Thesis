# THE FLORY COIL — COMPLETE VERDICT

### Reconciling "the coil cuts dG error by 20%" with "the coil is rejected"
### Nissim Brami · M.Sc. thesis · BGU · supervised by Prof. Chen Keasar
### Written 2026-09-08. Every number below was re-derived from raw per-protein data for this document.

---

## 0. PLAIN LANGUAGE — WHAT THE 20% ACTUALLY WAS

The project record contains two statements that sound like a contradiction:

> **(a)** "The Flory coil reduces dG error by 20%" — mean absolute error 4.9650 → 3.9521 kcal/mol.
> **(b)** "The Flory coil is rejected" — the trained arm degrades every quantity we care about.

**Both are true. They are not in conflict, because they measure different things.**

Here is the whole story in one paragraph. Our model **under-predicts the stability of all 28 test
proteins** — not most of them, *all* of them, in every condition we have ever tested. Every
prediction is too low by roughly 5 kcal/mol. When every error shares the same sign, mean absolute
error stops being a measure of *scatter* and becomes a measure of *how far the whole cloud sits from
the truth* — the average bias. Adding the Flory coil made the reference state (the unfolded chain)
more expansive, which pushed every prediction up by about 1 kcal/mol at once. The cloud moved
closer to the truth, so MAE fell 20%. **But the cloud did not get tighter — it got 12.7% wider.**

This matters because of what the thesis is actually trying to fix. The gap between our within-protein
ranking (PCC 0.798) and our across-protein ranking (~0.59) is caused by `b_p`, the **per-protein
offset** — the fact that each protein's predictions sit at a different wrong level. Fixing `b_p`
means pulling the proteins *toward each other*, i.e. **reducing the spread** `std(b_p)`. Moving them
all in the same direction together does nothing for that. And Pearson correlation is mathematically
**invariant to a uniform shift**: subtract any constant from every prediction and the correlation
changes by exactly zero. We verified this numerically to 1×10⁻¹⁶.

**So the 20% was real arithmetic on a metric that could not see the thing we were trying to fix.**
The decisive demonstration: simply subtracting one constant — the mean bias, a number with no
physics in it at all — cuts the same MAE from 5.0751 to **0.7162, an 85.9% reduction**, four times
better than the coil, while leaving every correlation *bit-identical*. A metric that a single
constant can improve by 86% was never evidence for a structural hypothesis.

**The verdict: the coil was a bias correction wearing the costume of a variance correction.**
It is rejected as a `b_p` lever — on the frozen checkpoint, across a ten-cell ν×b sweep, and in the
one arm we trained end to end.

---

## 1. THE TWO CLAIMS, PRECISELY STATED

| | claim (a) — "20% better" | claim (b) — "rejected" |
|---|---|---|
| **metric** | dG MAE | std(b_p), a_p, per-protein PCC, pooled PCC |
| **what it measures** | mean **bias** (distance of the cloud from truth) | **dispersion** (width of the cloud) + model quality |
| **setting** | frozen checkpoint, input transformation only | trained end to end, 15 epochs |
| **result** | 4.9650 → 3.9521 (−20.4%) | std(b_p) **rose**; the trained arm collapsed |
| **status** | **arithmetically correct** | **the decision-relevant result** |

They are not in conflict because **MAE and std(b_p) are different functionals of the same
residual vector**, and under the sign-degeneracy documented in §2 the first one collapses onto a
quantity the thesis does not care about.

---

## 2. ALL 28/28 PROTEINS ARE UNDER-PREDICTED — SO MAE ≡ |mean(b_p)|

Re-derived from `results/08_data/w0_dg.json` (frozen `calib_ctrl_repro2` epoch 14, the metric is
ABSOLUTE wild-type dG, n=28). Define the per-protein offset `b_p = pred − true`.

**2K5H-corrected basis** (its reference row was a mutant; +3.0824 kcal/mol — MASTER §11.1):

| condition | MAE | mean(b_p) | **std(b_p)** | n under-predicted | dG PCC |
|---|---|---|---|---|---|
| **base (no coil)** | 5.0751 | −5.0751 | **0.9967** | **28 / 28** | 0.4109 |
| coil, b fitted | 5.8547 | −5.8547 | 1.0743 | **28 / 28** | 0.4703 |
| **coil, b fixed** | **4.0622** | −4.0622 | **1.1231** | **28 / 28** | 0.3556 |
| coil, CA only | 5.6563 | −5.6563 | 1.0602 | **28 / 28** | 0.5075 |
| no unfolded embedding | 3.2416 | −3.2416 | 1.0235 | **28 / 28** | 0.0367 |

Raw (uncorrected 2K5H) basis, which is where the famous numbers come from:

| condition | MAE | mean(b_p) | std(b_p) | n under |
|---|---|---|---|---|
| base | **4.9650** | −4.9650 | 1.0385 | 28 / 28 |
| **coil, b fixed** | **3.9521** | −3.9521 | 1.1264 | 28 / 28 |

### The identity, verified exactly

For every one of the five conditions, on both bases:

    max | MAE − |mean(b_p)| |  =  0.000e+00        all residuals negative = True

This is not an approximation — it is **exact to the last bit**, and it is forced. If every `b_p < 0`
then `mean(|b_p|) = |mean(b_p)|` identically. **MAE was measuring mean bias and nothing else.**

### And the dispersion moved the WRONG WAY

    std(b_p):  0.9967  →  1.1231      +12.68%   WORSE      (2K5H corrected)
    std(b_p):  1.0385  →  1.1264      + 8.47%   WORSE      (raw)

**The quantity the coil was recruited to reduce is the one quantity it made worse.**
Confirmed on both bases, so this is not an artifact of the 2K5H correction.

### Why std(b_p) rose: the shift is non-uniform but uninformative

The per-protein shift `b_p(coil) − b_p(base)` is **not** a constant:

    mean = +1.0129    std = 0.6514    range −0.0855 (3DKM) … +2.4772 (2K1B)

So the coil *does* inject per-protein variation. It is simply **the wrong variation** — it is
uncorrelated with chain length (`corr = −0.3213`, below the n=28 significance threshold of 0.374)
and it adds variance rather than cancelling the existing offset. A coil that helped would have
shifted the *most* under-predicted proteins the most; instead the correlation between the base
offset and the shift is weak and the spread compounds. Hence a bigger `std(b_p)` despite a smaller
mean bias.

---

## 3. WHY A UNIFORM SHIFT CANNOT HELP POOLED PCC — PROVEN NUMERICALLY

**Pearson's r is invariant under `x → x + k`** because it is computed on mean-centred deviations:
adding `k` shifts the mean by exactly `k`, so every `(x_i − x̄)` is unchanged.

Verified directly on our data (2K5H corrected):

| condition | shift k | pooled dG PCC | Δ from k=0 |
|---|---|---|---|
| base | 0.0 | 0.410932119023543 | — |
| base | +1.0 | 0.410932119023543 | +0.000e+00 |
| base | −4.0622 | 0.410932119023543 | +0.000e+00 |
| base | +100.0 | 0.410932119023543 | −1.665e−16 |
| base | −1000.0 | 0.410932119023543 | −5.551e−17 |
| coil_fixed_b | ±(0, 1, −4.06, 100, −1000) | 0.355558008032720 | ≤ 1.110e−16 |

**Zero to floating-point round-off across four orders of magnitude in k.**

### The demonstration that settles the argument

De-bias each condition by its *own* mean — subtract a single scalar, no physics, no structure:

| condition | MAE before | MAE after | std(b_p) before → after | PCC before → after |
|---|---|---|---|---|
| **base** | 5.0751 | **0.7162** | 0.9967 → **0.9967** | 0.410932 → **0.410932** |
| coil, fitted | 5.8547 | 0.8619 | 1.0743 → 1.0743 | 0.470280 → 0.470280 |
| **coil, b fixed** | 4.0622 | 0.8756 | 1.1231 → **1.1231** | 0.355558 → **0.355558** |
| coil, CA only | 5.6563 | 0.8795 | 1.0602 → 1.0602 | 0.507545 → 0.507545 |
| no unfolded emb | 3.2416 | 0.8012 | 1.0235 → 1.0235 | 0.036658 → 0.036658 |

> **One constant beats the coil by 4× on MAE (0.7162 vs 4.0622) while changing std(b_p) and PCC by
> exactly nothing.** The 20% was 20% of a quantity that a single scalar removes 86% of.
> After de-biasing, the coil is *worse than base on both remaining metrics* — 0.8756 vs 0.7162 MAE
> and 1.1231 vs 0.9967 std(b_p).

This is the same trap flagged in MASTER §4.4: **the null a per-protein corrector must beat is the
RAW number**, never a mean-offset baseline, because the latter manufactures a fake improvement.

---

## 4. THE TRAINED ARM — `gld_dg_coil_s42`

The frozen result was an input transformation. The real test is whether *training* on dG with the
coil shrinks `std(b_p)`. Job 21080861:

    --full_data --no_pretrain --no_freeze --loss_mode dg --flory_unfolded --coil_b fixed
    --pooled_corr_weight 0 --dg_length_norm none --affine_calib --val_frac 0.1 --epochs 15 --seed 42

### Epoch 14 vs the control (raw-2K5H basis, matching `DG_ARM.md`)

| quantity | control `calib_ctrl_repro2` e14 | **dG coil arm e14** | change |
|---|---|---|---|
| **dG MAE** | 1.3029 | **2.4003** | **+1.10 WORSE** |
| **per-protein ddG PCC** | 0.7310 | **0.3151** | **−0.416** |
| **a_p** (median) | 0.4975 | **0.0852** | **−0.412** |
| r (median) | 0.7955 | 0.4951 | −0.300 |
| s (median) | 0.6342 | 0.2098 | −0.424 |
| **pooled ddG PCC** | 0.5910 | **0.2278** | **−0.363** |
| std(b_p) | 1.6030 | 0.9656 | −39.8% *(see below)* |

My independent recomputation from the six eval CSVs reproduces the control at **pooled 0.5635 /
a_p 0.4960 / r 0.7929** on the canonical 27-protein 2K5H-dropped basis and **0.5910 / 0.4975 /
0.7955** on the raw 28-protein basis — matching the scoreboard exactly. The arm scores **pooled
0.2212** canonically.

**`a_p = 0.085` means the model is essentially flat**: it has stopped tracking ddG within a protein
at all. The arm is *worse on dG MAE — the very metric it was built to optimise — than a control
trained on ddG.*

### The 40% std(b_p) drop is the degenerate baseline, not a win

The project has seen this exact signature before (`--dg_length_norm`, MASTER §6.3). As a prediction
is deleted, `b_p → −true_dG`, so `std(b_p) → std(true WT dG)` and `corr(b_p, true) → −1`:

| run | std(pred WT dG) | std(b_p) | corr(b_p, true WT) |
|---|---|---|---|
| control e14 | 1.3778 | 1.6030 | −0.5146 |
| **dG coil arm e14** | **0.6350** | **0.9656** | **−0.7744** |
| *degenerate attractor* | *0* | ***0.9208*** | *−1* |

**0.9656 sits 4.87% from the attractor 0.9208**, and the arm predicts *less* spread across proteins
(0.6350) than the labels actually contain (0.9208). The offset did not become better calibrated —
**the prediction contracted toward a near-constant and `b_p` inherited the spread of the labels.**

### The epoch-0 evidence — the part that makes the verdict safe

Six full 28-protein evaluations, checkpoints verified on disk:

| epoch | dG MAE | **std(b_p)** | per-prot PCC | a_p | r | s | pooled | std(pred WT) | **corr(b_p,true)** |
|---|---|---|---|---|---|---|---|---|---|
| **0** | 2.2615 | **0.9410** | 0.3205 | 0.0253 | 0.4333 | 0.0689 | 0.1547 | 0.3582 | **−0.9262** |
| 4 | 2.4088 | 0.9556 | 0.2245 | 0.0483 | 0.3138 | 0.1854 | 0.1257 | 0.6273 | −0.7771 |
| 8 | 2.3735 | 0.9168 | 0.2626 | 0.0774 | 0.4628 | 0.1836 | 0.1529 | 0.5983 | −0.7880 |
| 12 | 2.3451 | 0.9722 | 0.3136 | 0.0827 | 0.5385 | 0.1953 | 0.1953 | 0.6490 | −0.7662 |
| 13 | 2.3820 | 0.9732 | 0.3383 | 0.1019 | 0.5842 | 0.2213 | 0.2416 | 0.6502 | −0.7656 |
| **14** | 2.4003 | **0.9656** | 0.3151 | 0.0852 | 0.4951 | 0.2098 | 0.2278 | 0.6350 | −0.7744 |

> **`std(b_p)` is FLAT at 0.92–0.97 from epoch 0 to epoch 14. Training never moved it.**

At **epoch 0 — before the model has learned anything —** `std(b_p)` is already **0.9410** and
`corr(b_p, true) = −0.9262`, essentially the pure degenerate limit. The 40% "improvement" over the
control **was present at initialisation**. It is a property of the dG loss's output scale, **not an
effect of the coil, and not something training achieved.**

This is what makes the verdict robust despite the missing control (§6): *a quantity that is flat
from initialisation was not produced by training.* That argument is internal to the run and needs
no comparison arm.

Epoch 13 agrees with epoch 14 throughout, so this is not epoch-selection noise. (Per MASTER, this
arm was deliberately scored across epochs for a trajectory; it is not a val-selected headline.)

---

## 5. THE ν SWEEP — THE LAST PHYSICS DEFENCE, NOW CLOSED

**Status change.** The coil's remaining defence was that it had been given the wrong physics: every
run ever done used **ν = 0.5**, the *theta state* (an ideal random walk), whereas a denatured
protein in water is a **self-avoiding walk at ν ≈ 0.588** (Kohn et al. 2004 PNAS: R_g ∝ N^0.598).
At |i−j| = 50 that is a 41% distance error, and our chains run to ~70 residues.

**That sweep has now been run** (job 21137926, CPU, 3m12s, COMPLETED; frozen `calib_ctrl_repro2`
e14; artifacts `results/08_data/nu_sweep.json` and `nu_sweep_raw2k5h.json`). Re-derived
independently for this document from the raw per-protein predictions:

| condition | MAE | **std(b_p)** | n under | std(pred WT) |
|---|---|---|---|---|
| **base (no coil)** | 5.0749 | **0.9972** | 28/28 | 0.9125 |
| ν=0.620 b=4.97 | 3.9763 | **1.1036** ← best cell | 28/28 | 1.0026 |
| **ν=0.588 b=4.97** *(correct physics)* | 3.9949 | 1.1043 | 28/28 | 1.0102 |
| ν=0.550 b=5.82 | 4.0091 | 1.1066 | 28/28 | 1.0146 |
| ν=0.588 b=5.82 | 3.9931 | 1.1085 | 28/28 | 1.0097 |
| ν=0.650 b=4.97 | 3.9748 | 1.1118 | 28/28 | 1.0075 |
| ν=0.550 b=4.97 | 4.0355 | 1.1160 | 28/28 | 1.0336 |
| ν=0.620 b=5.82 | 4.0011 | 1.1208 | 28/28 | 1.0200 |
| **ν=0.500 b=5.82** *(every run to date)* | 4.0621 | 1.1233 | 28/28 | 1.0467 |
| ν=0.650 b=5.82 | 4.0315 | 1.1416 | 28/28 | 1.0424 |
| ν=0.500 b=4.97 | 4.1091 | 1.1452 | 28/28 | 1.0819 |

**All ten ν×b cells are worse than no coil at all on std(b_p).** Best cell 1.1036 vs base 0.9972 —
**+10.7% worse**. Correct physics does not rescue it.

Three things worth keeping:
1. **ν=0.500/b=5.82 reproduces `coil_fixed_b` to four decimals (4.0621 / 1.1233)** — proof that ν is
   genuinely read and not silently defaulted. Gate: `scripts/gate_coil_nu.py`.
2. **The ν direction is real but tiny and in the wrong place.** 0.5 → 0.62 improves std(b_p) by
   0.0416 — real and monotone, but it is only *recovering part of the damage the coil itself caused*.
   It never returns to baseline. **ν was a genuine bug; fixing it does not make the coil useful.**
3. **`n_under = 28/28` in all fifteen conditions**, so the MAE ≡ |mean(b_p)| identity holds
   throughout the sweep. MAE calls the best ν cell a **21.7%** win (5.0749 → 3.9748). It is not one.

**P1 is CLOSED. Do not re-run the ν sweep.**

---

## 6. WHAT REMAINS GENUINELY OPEN

Honesty requires separating "refuted" from "untested".

### 6.1 `--coil_edges` (U5/U6) has never been run — the coil may be only half-applied

I scanned the batch scripts of every job in the recent `sacct` window via
`scontrol write batch_script`: **no job has ever passed `--coil_edges`.** The flag is implemented
and gated (`scripts/gate_u5u6.py`: off-path byte-identity; and it *raises* rather than silently
no-ops if passed without `--flory_unfolded`), but never exercised in training or inference.

**Why this matters.** `--flory_unfolded --coil_b fixed` changes the unfolded-state *distances*
(node/geometry features) but leaves the unfolded graph's **edge topology** as the folded k-NN
topology. A true Flory coil should also have coil-appropriate *connectivity*. So every result in
this document tests a **partially applied** coil.

**Caveat on expectations.** The sign of the evidence is not encouraging: the distance half made
dispersion worse at every ν, and `coil_ca_only` (a different partial application) also came in
above base at 1.0602. But "untested" is not "refuted", and this is the last structurally distinct
version of the reference-state hypothesis. Given the project's signature failure mode, if it is run
the first check must be that the edges are actually *read*, not a DONE line.

### 6.2 No matched dG-loss control exists

`Megascale-fineTuning/models/ref_dg_seed42/` is **empty (0 entries, verified)**. There is no trained
`--loss_mode dg` run *without* the coil, so §4 compares against a **ddG-loss** control — differing in
loss *and* in coil. The verdict rests on the **epoch-0 evidence**, which is internal to the run.
A dG-loss/no-coil control would still separate *"the coil did nothing"* from *"the dG loss did it
all"*, and remains the cheapest single follow-up.

### 6.3 What is NOT open

- The MAE result (§2) — the sign-degeneracy is exact, on both 2K5H bases.
- Shift-invariance (§3) — a theorem, verified to 1e-16.
- ν (§5) — measured across ten cells, closed.
- The trained arm (§4) — three independent signatures (epoch-0 flatness, attractor proximity,
  collapse of a_p to 0.085) all agree.

---

## 7. VERDICT

> **The Flory coil is REJECTED as a `b_p` lever, and the "20% improvement" is real arithmetic on a
> metric that could not detect the failure.**
>
> All 28 of 28 proteins are under-predicted in every condition, so **MAE ≡ |mean(b_p)| exactly** —
> it measured **mean bias**, not dispersion. The coil moved all 28 predictions closer to the truth
> *together* without moving them closer to *each other*: **std(b_p) rose 0.9967 → 1.1231 (+12.7%)**.
> Pearson is exactly invariant to the uniform component, so the improvement was **unable in
> principle** to raise pooled PCC. A single constant — no physics — cuts the same MAE by **85.9%**
> while leaving every correlation bit-identical.
>
> Trained end to end, the arm collapses: dG MAE **2.40 vs 1.30**, per-protein PCC **0.315 vs 0.731**,
> a_p **0.085 vs 0.498**, pooled **0.221 vs 0.564**. Its apparent 40% `std(b_p)` reduction is the
> degenerate baseline — **flat from epoch 0 (0.9410 at e0, 0.9656 at e14)**, sitting **4.9%** from
> the attractor `std(true WT dG) = 0.9208`.
>
> The ν exponent was a genuine bug — every run used the theta value 0.5 instead of the
> self-avoiding-walk 0.588 — but **fixing it does not help: all ten ν×b cells are worse than no coil
> on std(b_p).**
>
> **Still open:** `--coil_edges` (U5) has never been run, so the coil has only ever been
> **half-applied**; and no matched dG-loss control exists.

**This is a publishable negative.** `b_p` has now survived **eight** explanations, of which the
reference state was the last cheap one — and the oracle still says **+0.14 pooled PCC** is available.

### The transferable lesson

**When every residual shares a sign, MAE silently becomes a bias metric.** Always report
`n_under-predicted` alongside it, always report the dispersion you actually care about, and always
compute the degenerate baseline — here, *"what does subtracting one constant achieve?"* Had that
one question been asked at the start, the coil would have been correctly classified in an afternoon
instead of becoming the project's most-cited number.

---

### Provenance

| item | source |
|---|---|
| frozen sweep | `results/08_data/w0_dg.json` (28 proteins, `calib_ctrl_repro2` e14) |
| ν sweep | `results/08_data/nu_sweep.json`, `nu_sweep_raw2k5h.json`; job 21137926 COMPLETED, cpu, 3m12s |
| trained arm | `eval_results/abl_gld_dg_coil_s42_e{0,4,8,12,13,14}.csv` — six evaluations, verified on disk |
| control | `eval_results/abl_calib_ctrl_repro2_e14.csv` |
| re-derivation for this document | `scripts/flory_rederive.py`, `scripts/flory_arm.py` |
| 2K5H correction | +3.0824 kcal/mol, MASTER §11.1 |
| gates | `gate_u3u4` (32/32), `gate_u5u6`, `gate_coil_nu` |
| **G4 guard after all work** | **`baseline … dG=-0.0030 width=1092` — UNCHANGED** |

**Related:** MASTER §5.1 (the reversal), §4.2 (eight rejected `b_p` hypotheses), §6.3
(`--dg_length_norm`, the same degeneracy), §2 (the metric rule);
`results/02_findings/DG_ARM.md`; `results/02_findings/NU_SWEEP.md`;
`results/01_start_here/OPEN_PROBLEMS.md` P1 (now closed) and P6.
