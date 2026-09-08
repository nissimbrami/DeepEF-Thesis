
---

# CHECKPOINT 29 — 2026-09-08 — D1 CANCELLED, THE ν EXPONENT WAS NEVER SWEPT, W12 BUILT

## 1. The D1 arm is cancelled — 13 pending cells

Verified by JobName before each `scancel`; **no RUNNING job was touched** (15 before, 15 after).
Grounds: K10 measured **D0 − D1 = 14.4 seed-sigmas**, with a_p separating by a factor of 42
(0.574 vs 0.014 — the D1 models are FLAT, not merely worse). Finishing them would have spent
~104 GPU-hours adding decimal places.

## 2. Re-examining the failures: design fault, or execution fault?

Asked again for each, because a good idea executed badly is still a good idea.

| lever | design | execution | the idea was… |
|---|---|---|---|
| `--dg_length_norm` | **WRONG** — /n deletes the prediction | fine | bad idea, correctly rejected |
| b_p corrector | fine | fine | good idea, **genuinely refuted** (95% of the gain fails out of sample) |
| W7span / U10 / W7edge | fine | fine | plausible, **inside the ±0.060 noise band** |
| W6 descriptors | fine | fine | **superseded** — ProtT5 predicts held-out hydropathy at R²=0.704 |
| **the coil** | **WRONG METRIC** — MAE measured mean bias, not dispersion | fine | see §3 |
| **LORO** | fine | **BROKEN** | **good idea, never tested** — see §4 |
| severing | fine | **OOD guard fired** | good idea, needs a retrained model |
| factor A | **CONFOUNDED** — aliased with seed AND epoch | fine | not estimable as designed |

## 3. THE FLORY COIL: ν was NEVER swept, and the default is the wrong physics

`--flory_nu` has **default 0.5** and **`sacct` shows zero runs ever set it.** Every coil arm —
including the rejected dG arm — used ν=0.5.

**ν=0.5 is the THETA state**: a collapsed chain in a solvent-neutral condition. **A denatured
protein in water is a self-avoiding walk, ν≈0.588.** Kohn et al. 2004 PNAS measured
denatured-state Rg ∝ N^0.598 across 28 proteins.

What that costs, at b=5.82 Å:

| \|i−j\| | ν=0.5 | ν=0.588 | ratio |
|---|---|---|---|
| 10 | 18.40 | 22.54 | 1.225 |
| 20 | 26.03 | 33.88 | 1.302 |
| 50 | 41.15 | 58.07 | **1.411** |

**A 41% distance error at long range — and our proteins are 43–72 residues, so |i−j| reaches ~70,
which is exactly where ν matters most.**

**So the coil was rejected on its dispersion result (§5.1 of MASTER, which stands), but it was
never tested on the physically correct exponent.** Those are two separate statements and the record
must keep them apart. The dispersion failure is about what MAE measured; the ν question is about
whether the reference state was ever the right shape.

## 4. LORO: the scale mismatch, now measured directly

The descriptor arm collapsed (RMSE frozen at 2.541). **Cause confirmed by measurement:**

    descriptor table  sd = 1.0000   max|v| = 6.04   →  energy ratio ~16.8x one-hot
    one-hot           sd = 0.2179   max    = 1.0

The descriptors were z-scored to sd=1 across 20 residues and are 16 columns wide, so they carried
**~17× the energy of the identity signal.** The network saw noise that drowned residue identity.
**An execution fault. The hypothesis was never tested.**

## 5. W12 — the side-chain block, built and gated (18/18)

`scripts/sidechain_features.py` + `scripts/gate_sidechain.py`.

**The mechanism is CORRECTED.** The original design blamed side-chain BULK; that is refuted by its
own shuffle test (volume P=0.069, hydropathy P=0.0000; cross-validated R² volume 0.031 vs
hydropathy 0.666 vs **transfer free energy 0.753**). The block therefore encodes **Fauchère–Pliska
water→octanol transfer free energy**, with volume only as a secondary area-scaling term.

**Verified against the measured per-residue slopes:**

    corr(dG_transfer, measured slope) = -0.922
    corr(volume,      measured slope) = -0.408

**Four columns at offset 48:** `dG_transfer`, `burial·dG_transfer`, `volume`, `burial·volume`.
The burial-weighted columns are **ZERO in the unfolded state**, so the folded-minus-unfolded
difference IS the hydrophobic driving force — it acts on b_p — while the mutated position's
identity delta survives ddG. **It acts on BOTH channels and must be reported on both.**

**The LORO lesson is designed in:** block energy is **1.83× one-hot, not 16.8×**, because it is
4 z-scored columns rather than 16. The gate asserts this explicitly so the collapse cannot recur.

## 6. Scoring: 4 CPU jobs now covering every unscored run

`scripts/score_all.sh` scores **every** run trained to epoch 14 with no CSV — factorial cells and
golden arms alike — on the CPU partition at zero GPU cost, verifying the ARTIFACT with `ls` after
each cell rather than trusting the `DONE` line.
