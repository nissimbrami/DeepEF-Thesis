# DeepEF — THE IDEA LEDGER
### Every idea ever proposed, and its true status. 2026-09-08.
### This is the document that answers "did we try X?"

> **THE EVIDENCE RULE.** An idea counts as **TESTED** only if a scored CSV exists in
> `eval_results/`. A flag in `train.py` is NOT evidence it ran. A finished training run is NOT
> evidence it was scored. A gate passing is NOT evidence it helps — gates check mechanics
> (shapes, byte-identity, ordering), never performance.
>
> **The authoritative list of what ran** is 44 CSVs → **34 unique run tags**:
> ```
> ls eval_results/abl_*.csv | sed 's|.*/abl_||;s|_e[0-9]*\.csv||' | sort -u
> ```
> Verified 2026-09-08. Every "TESTED" row below names a tag from that list. Every row that
> cannot name one says so in plain words.

> **HOW TO READ THE RESULT COLUMN.** Three qualifiers, always: *pooled-or-per-protein* /
> *dG-or-ddG* / *which split*. Basis unless stated: **27 test proteins (2K5H excluded), ddG
> metric, val-selected epoch.** Control = `calib_ctrl_repro2_e14` **pooled 0.5635**.
> **SEED NOISE BAND = ±0.060** (5 identical-config seeds, pooled 0.556–0.638).
> **Anything under +0.060 is not distinguishable from noise.**

> **THE METRIC RULE, which governs half this table.** ddG cancels anything identical between
> WT and mutant. A point mutation changes neither chain length nor backbone geometry, so every
> whole-protein or reference-state property cancels **exactly** in `pred_mut − pred_wt`. Such
> levers **must** be scored on dG or std(b_p). `a_p` is a *within*-protein slope and IS
> legitimately ddG-measurable. This rule has caught **five** levers from being mis-scored.

---

## THE ONE-SCREEN ANSWER

| # | The idea, in the user's words | Flag | Built | Tested? | Verdict |
|---|---|---|---|---|---|
| 1 | Make the model stop squashing the range | `--slope_weight` | ✅ gated 21/21 | ✅ **4 tags** | ✅ **THE ONE WIN** |
| 2 | Hydrophobicity & burial | `--burial_features` | ✅ gated 18/18 | ✅ `gld_w5_dg_s42` | ✅ **best single-flag a_p 0.667** |
| 3 | Anchor the wild type | `--wt_anchor_weight` | ✅ | ✅ **3 tags** | ⚠️ +0.047 = **inside noise**, unattributable |
| 4 | Give the unfolded state a real coil (Flory) | `--flory_unfolded` | ✅ gated 32/32 | ✅ `gld_dg_coil_s42` | ❌ **REJECTED — destroys model** |
| 5 | …and use the *right* coil exponent ν | `--flory_nu` | ✅ gated | ✅ **ν sweep, 10 cells** | ❌ **CLOSED** — right physics, still worse |
| 6 | Chemistry descriptors instead of one-hot | `--aa_descriptors` | ✅ gated | ⚠️ **arm COLLAPSED twice** | ⛔ **STILL UNTESTED** |
| 7 | Non-canonical residues (open alphabet) | `--aa_descriptors` (25-row) | ⚠️ table only | ❌ **never run** | ⛔ **not measurable on 28 canonical** |
| 8 | Bonding / edge features | `--edge_features` | ✅ gated | ✅ `gld_w7edge_s42` | ➖ +0.017 = **inside noise** |
| 9 | Longer chain reach | `--gcn_span` | ✅ gated | ✅ `gld_w7span4_s42` | ➖ +0.012 = **inside noise** |
| 10 | Edges should point both ways | `--gcn_bidir` | ✅ gated | ✅ `gld_u10bidir_s42` | ➖ +0.026 = **inside noise** |
| 11 | Side-chain transfer free energy | `--sidechain_features` | ✅ **gated 18/18** | ❌ **queued, never trained** | 🟡 **READY — the best untested idea** |
| 12 | Metals | *(none — retired)* | ⚠️ module only | ❌ never | ⛔ **0 metals in all 28** |
| 13 | Ligands (ATP, haem, cofactors) | `--ligand_nodes` | ✅ gated | ❌ never | ⛔ **format cannot hold one** |
| 14 | Complexes / buried surface area | **NONE** | ❌ | ❌ never | ⛔ **all 28 are monomers** |
| 15 | Leave-one-residue-out (LORO) | `--holdout_residues` | ✅ gated 13/13 | ⚠️ one arm only | ⛔ **comparison never completed** |
| 16 | Unfolded embedding → zero | `--unfolded_emb` | ✅ gated | ✅ **12 factorial cells** | ❌ **DESTROYS MODEL, 14.4σ** |
| 17 | Coil edges for the unfolded graph | `--coil_edges` | ✅ gated | ❌ never alone | 🟡 ready, low priority |
| 18 | Divide dG by chain length | `--dg_length_norm` | ✅ | ✅ (frozen-ckpt) | ❌ **DEGENERATE — deletes prediction** |
| 19 | Fit a global affine calibration | `--affine_calib` | ✅ | ✅ (oracle analysis) | ❌ **cannot change PCC — by construction** |
| 20 | Push pooled correlation directly | `--pooled_corr_weight` | ✅ | ❌ never | 🟡 ready, untried |
| 21 | Oversample designed folds | `--designed_weight` | ✅ | ❌ never | 🟡 ready, untried |

**Legend:** ✅ real effect / ➖ inside the noise band / ❌ rejected or impossible /
⚠️ broken or partial / ⛔ not measurable on this dataset / 🟡 built and ready, never run.

**The headline count: 21 ideas. 2 helped. 6 were rejected on evidence. 5 are not measurable on
these 28 proteins at all. 5 are built and never run. 3 are broken or unfinished.**

---

# PART I — THE IDEAS THAT WORKED

## 1. "Stop the model squashing everything toward the middle" — the slope term

| | |
|---|---|
| **Idea** | Predictions span half the range of the truth. Add a loss term that punishes compressed spread. |
| **Flag** | `--slope_weight` (lever C) |
| **Built?** | ✅ Wired end to end. **`scripts/gate_slope.py`, 21/21 checks.** |
| **Tested?** | ✅ **YES — 4 scored tags**: `p3_slope1.0_s42_e13`, `p3_slope0.3_s42_e9`, `p3_slope3.0_s42_e8`, `gld_slope1.0_s42` (6 epochs). |
| **Result** | **pooled 0.6210 vs control 0.5635 = +0.0575.** `a_p` **0.4975 → 0.7396**. |

**Why it works, and the exact accounting.** For any least-squares fit `a_p = r · s`, where
`s = std(pred)/std(true)`. **This holds in our data to 2.0×10⁻¹⁵.** The lever moves `s` and only `s`:

    a_p  0.4975 → 0.7396   (+0.242)
    s    0.6342 → 0.9901   (+0.356)   ← all the gain is here
    r    0.7955 → 0.7925   (−0.003)   ← FLAT across ten epochs

**The ceiling is therefore `a_p = r ≈ 0.79`, and we are at 93% of it.** The lever cannot reach
1.0 by construction, because the ranking half `(1−r)` is untouched.

**The weight sweep has an interior optimum — more is worse:**

| `--slope_weight` | s | a_p | pooled |
|---|---|---|---|
| control | 0.635 | 0.499 | 0.5635 |
| 0.3 | 0.573 | 0.415 | 0.5676 |
| **1.0** | **0.756** | **0.562** | **0.6210** |
| 3.0 | 0.472 | 0.326 | **0.5362** ← worse than no lever |

**Its honest cost:** `std(b_p)` **worsens 1.603 → 2.359**. The arm trades offset calibration for
slope calibration. **Seed replication (`gld_slope1.0_s1/s2/s3`) is RUNNING now** — at n=1 the
+0.0575 sits marginally below the ±0.060 band, so this is not yet closed.

---

## 2. "Hydrophobicity and burial" — W5

| | |
|---|---|
| **Idea** | Tell the model which residues are buried and how greasy they are. |
| **Flag** | `--burial_features` (+ `--burial_mode count\|hse`), 3 columns |
| **Built?** | ✅ **`scripts/gate_w5.py`, 18/18.** Width 1092→1095, confirmed in G4. |
| **Tested?** | ✅ **YES — `gld_w5_dg_s42_e14`** (`eval_results/abl_gld_w5_dg_s42_e14.csv`) |
| **Result** | **`a_p` = 0.6666 — the highest of any single flag.** pooled 0.5956 (+0.032, inside band). |

**The honest reading.** W5's pooled gain is **inside the noise band**; its `a_p` gain is **not**.
This is exactly the split the metric rule predicts: burial is a **folded-state** quantity, so it
acts on `dG`/`b_p`, and scoring it on pooled ddG alone would repeat a known mistake — **W5's own
gate passes 18/18 with every assertion mechanical, and not one scores performance.**

**Replication `gld_w5_dg_s1/s2` is RUNNING; `s3` pending.** The combination arm
`gld_slope_w5_s42` (do the two `a_p` levers add?) is **RUNNING now** — this is open question Q7.

---

# PART II — THE IDEAS THAT WERE REJECTED ON EVIDENCE

## 3. "Give the unfolded state a real polymer coil" — the Flory reference state

| | |
|---|---|
| **Idea** | The unfolded state is not the folded structure. Model it as a random coil: `d(i,j) = b·\|i−j\|^ν`. |
| **Flag** | `--flory_unfolded --coil_b` (U3/U4) |
| **Built?** | ✅ **`scripts/gate_u3u4.py`, 32/32.** |
| **Tested?** | ✅ **YES — `gld_dg_coil_s42`, 6 scored epochs.** |
| **Result** | ❌ **REJECTED. pooled 0.2212 vs control 0.5635. `a_p` 0.084 vs 0.498 — the model is nearly FLAT.** |

**This is the project's biggest reversal, and it is worth understanding.** The most-cited number
for weeks was *"the coil improves dG MAE 4.9650 → 3.9521"*. That number is real and it is
**meaningless**, for a precise reason:

> **All 28 of 28 proteins are under-predicted in every condition.** When every residual shares a
> sign, **MAE ≡ |mean(b_p)|** — it measures *bias*, a uniform shift. **Pearson correlation is
> exactly invariant to a uniform shift.** So the coil's headline improvement could not have
> improved any correlation score, ever.

Meanwhile the quantity that matters — the *dispersion* `std(b_p)` — went the **wrong way**:
**0.9967 → 1.1231, +12.7% worse.** The coil moved all 28 predictions closer to the truth
*together*, without moving them closer to *each other*.

**The trained arm confirms the rejection**, and the degenerate check catches the trap: `std(b_p)`
fell 40%, which looks like a win, but it is **flat from epoch 0** (0.9410 at e0 vs 0.9656 at e14),
before the model learned anything, sitting 4.9% from the known attractor `std(true WT dG)=0.9208`.

**Caveat on the record:** `ref_dg_seed42/` is empty, so there is no matched dG-loss control
*without* the coil. The verdict rests on the epoch-0 evidence, which is internal to the run.

---

## 4. "But we used the WRONG ν — a real chain is a self-avoiding walk" — ν = 0.588

| | |
|---|---|
| **Idea** | ν=0.5 is the *theta state*. A denatured protein in water is a self-avoiding walk, **ν≈0.588** (Kohn 2004 PNAS, Rg ∝ N^0.598). At \|i−j\|=50 that is a **41% distance error**. |
| **Flag** | `--flory_nu` (default 0.5) |
| **Built?** | ✅ `scripts/gate_coil_nu.py` — asserts ν is *read*, not silently defaulted. |
| **Tested?** | ✅ **YES — full sweep, job 21137926, frozen checkpoint, 10 ν×b cells.** `results/02_findings/NU_SWEEP.md`, `results/08_data/nu_sweep.json`. |
| **Result** | ❌ **CLOSED. Every one of the ten cells is WORSE than no coil at all.** |

**This was a legitimate objection, properly tested, and it lost.** Scored on `std(b_p)`, not MAE:

| condition | MAE | **std(b_p)** | std(pred WT) |
|---|---|---|---|
| **base (no coil)** | 5.0749 | **0.9972** | 0.9125 |
| ν=0.500 *(every run to date)* | 4.0621 | 1.1233 | 1.0467 |
| **ν=0.588 (correct physics)** | 3.9949 | 1.1043 | 1.0102 |
| ν=0.620 ← best cell | 3.9763 | **1.1036** | 1.0026 |

**Best cell 1.1036 vs base 0.9972 = +10.7% worse.** The ν direction is real, monotone, and tiny
(0.5→0.62 recovers 0.0416) — but it is **recovering part of the damage the coil itself caused**
and never gets back to baseline. **ν was a genuine bug; fixing it does not make the coil useful.**

**Do not re-run this sweep.** And note the two claims were correctly kept apart: the coil's
motivating MAE result was a bias artifact *(independent of ν)*, and the reference state had
*never been given the physically correct shape* — that second claim is now tested and dead too.

---

## 5. "Zero the unfolded embedding so the reference state is protein-independent" — factor D

| | |
|---|---|
| **Flag** | `--unfolded_emb zero` (U2) |
| **Tested?** | ✅ **YES — 12 factorial cells, both arms.** |
| **Result** | ❌ **DESTROYS THE MODEL — 14.4 seed-sigmas.** |

| arm | n | pooled | sd | a_p |
|---|---|---|---|---|
| **D0 (coil)** | 6 | **0.5939** | 0.0091 | **0.5736** |
| **D1 (`--unfolded_emb zero`)** | 6 | **0.1579** | 0.1171 | **0.0136** |

**`a_p` separates by a factor of 42. An `a_p` of 0.014 means the model is FLAT, not merely worse.**
One D1 cell is *anti*-correlated (pooled −0.074). **It removes the offset by removing the signal.**

**This reversed the earlier W0 decision**, which had chosen D1 on a *frozen-checkpoint variance
decomposition* — neither dG nor ddG performance. Trained end to end, the choice inverts.

> **STANDING RULE, now machine-enforced: never average across runs that differ in factor D.**
> D0 spans 0.536–0.638; D1 spans −0.074–0.317. Report them **separately, always.** Violating this
> once produced a published headline of 0.4899 that is now retired.

---

## 6. "Divide dG by chain length to remove the size effect"

| | |
|---|---|
| **Flag** | `--dg_length_norm n\|sqrtn` |
| **Result** | ❌ **DEGENERATE. It deletes the prediction.** |

`std(b_p)`: none **1.3076** → /n **0.9050** — an apparent **31% improvement**, and it is not real.
`/n` converges to ~0.905 for **every** checkpoint, including ones that *start* at 0.876. The
reason: **`std(true WT dG) = 0.9042`.** Dividing by N drives `pred/N → 0`, so `b_p → −true_dG` and
its spread becomes simply the spread of the labels.

> **This is the origin of the second standing rule: ALWAYS compute the degenerate baseline.**
> A lever that looks like a 31% win can be the model predicting nothing.

---

## 7. "Just fit a calibration line afterwards"

| | |
|---|---|
| **Flag** | `--affine_calib` |
| **Result** | ❌ **Cannot change PCC — by construction.** |

**Pearson is invariant to affine transformation.** Fitting a global `(a,b)` fixes RMSE and changes
correlation by **exactly zero**. Worse, the *affine oracle* (per-protein `a` and `b` removed) is
**not reproducible as a bound**: it **divides** by the slope, so at `a_p = 0.079` it amplifies
noise ~13×. An estimator that is not monotone is not a bound. **Offset-only removal survives**
(subtracting a constant cannot amplify anything) and that is the 0.7156 oracle we quote.

**A corollary that has bitten us:** since Pearson is shift-invariant, **the null a per-protein
corrector must beat is the RAW pooled number**, not a mean-offset baseline — the latter
manufactures a fake +0.011.

---

## 8. The b_p offset correctors — eight hypotheses, all dead

`b_p` is the per-protein offset, and **the oracle says +0.14 pooled PCC is available if it were
solved.** It is real (a random-offset null cannot reproduce our triple; best L2 = 0.118) and
learnable in principle (**ICC = 0.898** over 28 proteins × 5 seeds).

| hypothesis | result |
|---|---|
| label noise / regression attenuation | ❌ **DEAD by 200×** — noise would need to be 200× larger |
| chain length | ❌ corr(N, b_p) = **+0.0252** over 12 checkpoints |
| `--dg_length_norm` | ❌ **DEGENERATE** (§6) |
| embedding norm / 1/√N artifact | ❌ claimed −0.9938, **measured −0.29** (n=28 threshold 0.392) |
| pre-training distribution shift | ❌ r = **+0.043**, p = 0.829 |
| train/test mean gap (Ofir's mechanism) | ❌ gap −0.1034, p = 0.588 |
| a feature-based corrector | ❌ **95% of the oracle gain fails out of sample**; held-out R² NEGATIVE on 8/10 |
| the reference state (the coil) | ❌ **REJECTED** (§3, §4) |

**On attenuation specifically** — the obvious objection is that `a_p ≈ 0.5` is the textbook
signature of noisy labels, which no architecture could fix. Per-mutation uncertainties (found in
`data/ThermoMPNN/mega_test.csv`; an earlier attempt searched the wrong file and reported 0/28):
median σ = 0.0327 kcal/mol against median var(true ddG) = 0.7178, so attenuation predicts
**a_p = 0.9985** against a measured 0.50. **`a_p` is a genuine model failure, not a statistical
artifact.** This is what licenses the whole slope programme.

**The strongest surviving clue:** the **top-2 |b_p| proteins carry 78%** of the oracle gain, and
**neither is a structural outlier** (max |z| ≈ 2.0 across 22 features). `b_p` is not a smooth
function of structure — it is concentrated in a few proteins for a reason not yet identified.

---

# PART III — THE IDEAS THAT ARE NOT MEASURABLE ON THIS DATASET

> **This is the most important section for thesis writing, because these are NOT failures.**
> The distinction is *"no experiment on these 28 proteins can return an answer either way"* —
> which is a statement about our benchmark, not about the chemistry.

**The hard constraint, verified by opening the stored tensors for all 28 test proteins — not
taken from any report:**

    coords_tensor.pt is [N, 4, 3] for all 28 without exception
    N = 43 to 72 residues | 4 = backbone N, CA, C, CB | 3 = x, y, z

    n_chains == 1 · n_het_residues == 0 · n_metal_residues == 0 · interchain_BSA == 0.0

> **There is no fourth thing.** No slot a zinc ion could occupy, no row an ATP molecule could
> live in, no second chain. **No PDB or mmCIF file exists anywhere in the pipeline** — HETATM
> records were discarded before this project began and are not recoverable.
>
> These are not 28 proteins that happen to have no ligand. They are 28 proteins **stored in a
> format with no way to express one.**

## 12. Metals — `NONE` (retired)

Module `scripts/metal_features.py` is fully written and `gate_w9.py` passes. **But there is no
`--metal_features` argument in `train.py`**, so `CFG.metal_features` is never set and the module
**can never activate**. Deliberately **RETIRED** on 2026-09-07, superseded by W11, and documented
in `results/03_levers/W9_RETIRED.md`. **Zero metal residues in all 28 test proteins.**

## 13. Ligands — `--ligand_nodes`, built and gated, never run

10-dim contact block, hetero-nodes for ATP/NAD/haem/cofactors. **Fully built, `gate_ligand.py`
passes.** It has **never been trained**, and training it would be uninformative: a feature that is
**constant across every example carries no gradient signal and cancels identically in ddG** (it is
a folded-state term, identical in WT and mutant away from the site).

**Two traps this idea already sprang, both now closed:**
1. `--ligand_nodes` **could not fire in training** at all — and `gate_ligand` D.14 had **codified
   the bug**, asserting the broken behaviour was correct.
2. The 100k catalogue's `Ligands_x` column has **4 non-null rows out of 100,246**. An earlier BSA
   result of r = −0.001 was computed on that column, against a metric already known not to predict
   ddG. **Void twice over.**

**And a warning for any future dataset:** of 205,648 het-code occurrences over 8,187 codes,
**27.4% are crystallization additives** (SO4, GOL, EDO, NA, CL) and only **6.6% are real
biological ligands**. **MSE is selenomethionine — an amino acid IN THE CHAIN**, so counting it as
a ligand is both wrong and a double-count. **A naive ligand feature would learn crystallography,
not biochemistry.**

## 14. Complexes / buried surface area — **NO FLAG EXISTS**

Never built, and correctly so. **`interchain_BSA == 0.0` for all 28**; all are monomers. The
catalogue columns that looked like evidence are dead: `BSA_Numeric_x` has **4 unique values**,
`BSA_Percentage` has 5.

## 7. Non-canonical residues (open alphabet) — the one place descriptors could win

A 25-row descriptor table exists (`data/aa_descriptors_open25.csv`). **Never run.** And on our
data it **cannot** win: **all 28 test proteins are canonical**, so the coverage gap descriptors
exist to fill **does not exist here**. This matters because coverage is the *only* surviving
argument for descriptors at all (§15).

**Two build faults found and fixed:** `gate_open_alphabet` was testing a `/tmp` fixture rather
than the real 25-row table (K=756 vs canonical 726), and it had hard-coded a retired mode name, so
it **had stopped testing anything at all.**

---

# PART IV — THE IDEAS THAT ARE BUILT AND HAVE NEVER RUN

## 11. 🟡 Side-chain transfer free energy — W12. **The best untested idea in the project.**

| | |
|---|---|
| **Idea** | The model only sees 4 backbone atoms (N, CA, C, CB) — **no side chains**. Give it the chemistry of the side chain being created: its **transfer free energy**. |
| **Flag** | **`--sidechain_features`** — 4 columns: `dG_transfer`, `burial×dG_transfer`, `volume`, `burial×volume` |
| **Built?** | ✅ **YES, and fully wired — verified line by line today.** |
| **Tested?** | ❌ **NO. Five jobs are PENDING in the queue** (`gld_w12_s42/s1/s2`, `gld_w12_dg_s42`, `gld_w12_slope_s42`). **Zero CSVs.** |

**Why this is the best remaining idea — it targets the single largest measured deficit.** The
model is worst at burying hydrophobics:

| destination | KD | slope | spearman |
|---|---|---|---|
| N | −3.5 | **0.565** | 0.653 |
| K | −3.9 | 0.537 | 0.648 |
| A | +1.8 | 0.419 | 0.567 |
| **C** | +2.5 | **0.267** | 0.356 |
| **L** | +3.8 | **0.282** | 0.388 |
| I | +4.5 | 0.311 | 0.412 |

    corr(slope,    Kyte-Doolittle) = −0.734    sign-consistent 10/10
    corr(spearman, Kyte-Doolittle) = −0.740

**Those two correlations are essentially identical, so rank accuracy degrades in LOCKSTEP with
the slope. This is LOST INFORMATION, not a rescalable calibration error — no affine correction
can help.** It is the `(1−r)` half of `a_p = r·s`, given a chemical identity.

**⚠ The mechanism was corrected, and the correction is what makes W12 the right fix.** The
original claim was side-chain **bulk**. That is refuted by its own shuffle test:

| | cross-validated R² | corr with measured slope |
|---|---|---|
| volume (bulk) | **0.031** | −0.408 (P=0.069, **fails**) |
| hydropathy | 0.666 | −0.765 |
| **transfer free energy** | **0.753** | **−0.922** |

**The mechanism is hydrophobic transfer free energy, not steric packing.** W12's leading column
is exactly that quantity.

**Verified today, not assumed** (`gate_sidechain.py` **18/18 PASS**):
- Block width **4**; total width **1092 → 1096**, and `dG=-0.0030 width=1092` still holds with the flag OFF.
- **Unfolded burial columns are exactly ZERO** → the folded-minus-unfolded difference **IS** the hydrophobic driving force.
- Identity columns identical in both states (sequence, not structure) → they survive ddG at the mutated position.
- Chemistry sane: most hydrophobic **W**, most hydrophilic **R**; I/L/F all > +1.5 kcal/mol; D/E/K/R all negative.
- **Block energy ratio 1.83× one-hot** — versus the **16.8×** that collapsed the LORO descriptor arm (§15). *This is the check whose absence cost 8 GPU-hours.*
- Wiring confirmed present in `train_utils.py` (folded + **both** unfolded paths) and `model/hydro_net.py` (`fc1_gcn`/`fc1_gat` sizing + the PEM guard).

> **⚠ CORRECTION TO THE RECORD.** `OPEN_PROBLEMS.md` **P3** states *"There is no
> `--sidechain_features` flag in `train.py`, so the block cannot fire."* **That is now STALE.**
> The wiring was completed **2026-09-08 21:04**. **P3's 2-hour wiring task is DONE; only the
> 8 GPU-hours remain.**

**READY TO RUN? ✅ Yes — nothing is missing.** The jobs are queued and blocked only on golden-lane
capacity. **Scoring instruction, which matters:** it acts on **both** channels, so report **ddG
pooled/per-protein AND std(b_p)**. Scoring it on ddG alone would repeat the W5 mistake.

## 17. 🟡 Coil edges — `--coil_edges` (U5/U6)
Built, `gate_u5u6.py` passes. **Never run alone** — only ever inside the rejected dG-coil arm.
Low priority: it shares the reference-state mechanism that §3–§5 closed.

## 20. 🟡 Pooled correlation loss — `--pooled_corr_weight`
Built (weight + window accumulation). **Never run — zero CSVs.** Attacks the cross-protein
objective directly, which is precisely where pooled PCC lives. **Untried and cheap.**

## 21. 🟡 Oversample designed folds — `--designed_weight`
Built (WeightedRandomSampler over HHH/HEEH/EEHEE/EHEE/TrROS/v2_). **Never run — zero CSVs.**

---

# PART V — THE IDEAS THAT ARE BROKEN OR UNFINISHED

## 6 & 15. Chemistry descriptors, and the LORO test — **STILL UNTESTED AFTER TWO ATTEMPTS**

| | |
|---|---|
| **Idea** | One-hot asserts all twenty residues are equidistant, which is chemically false. Give the alphabet a real metric — Mordred descriptors — and it should generalise to a residue it never saw. |
| **Flag** | `--aa_descriptors none\|mordred726\|mordred_pca16\|mordred_pca16_only`; **`--holdout_residues`** (LORO, `gate_loro.py` 13/13) |
| **Tested?** | ⚠️ **The one-hot control ran (`loroW_onehot_s42_e12`, pooled 0.5716). The descriptor arm has COLLAPSED TWICE.** |

**Attempt 1 — `gld_loroW_desc`: collapsed.** ddG RMSE frozen at exactly **2.541** for ~11 epochs,
PCC oscillating about zero. **Cause, measured directly:** descriptors were z-scored to unit
variance *per column* and concatenated **alongside** one-hot —

    descriptors: sd = 1.0000, max|v| = 6.04, 16 cols  →  ~16.8× the energy of one-hot
    one-hot:     sd = 0.2179, max = 1.0,     20 cols

**The network optimised against what was loud, and residue identity — the thing the LORO
experiment is about — was drowned.** An execution fault, not a refutation.

**Attempt 2 — `gld_loroWdesc2_s42`: RUNNING NOW, AND COLLAPSING THE SAME WAY.** I checked its log
today (job 21136228). The intended fix **is** correctly applied —
`[W6] mode=mordred_pca16_only K=16`, which **replaces** one-hot rather than appending, and
`[LORO] holding out of TRAINING (both directions): W` confirms the holdout is real:

    ddG PCC=0.085  ddG PCC-PP=0.028  ddG RMSE=2.541  a_p median=nan
    ddG PCC=0.050  ddG PCC-PP=0.016  ddG RMSE=2.541  a_p median=nan

> **⚠ NEW FINDING, not in MASTER.md or OPEN_PROBLEMS.md.** **RMSE is frozen at the identical
> 2.541 and `a_p median = nan`.** The scale-clash explanation predicted this run would recover.
> **It has not.** Only 2 epochs are in, so this is early — but the signature is exact, and
> `a_p = nan` means the predictions have **zero variance within a protein**: the model is emitting
> a constant. **The scale clash was therefore not the whole cause**, and P2's plan needs revisiting.

**What is missing before this idea can be called tested:** a descriptor arm that **trains at all**.
Diagnose why `mordred_pca16_only` still degenerates (16 PCA columns replacing 20 one-hot columns
may simply be **rank-deficient for identity** — PCA delivered 16 against a rank cap of 19).
Then compare Spearman on **mutations to tryptophan only** against the `loroW_onehot` control.

**The prediction is on record and must be honoured either way:** descriptors should **NOT** beat
one-hot, because **ProtT5 already predicts the hydropathy of a held-out residue type at R² = 0.704**
(§below). **If they DO win, that overturns the redundancy conclusion and must be reported as a
surprise, not rationalised.**

### Why descriptors were expected to be redundant — the probe that made the call

ProtT5 is **contextual** (the same residue at two positions has cosine **0.13–0.18**, nearly
orthogonal) and **recomputed per variant** (one mutation changes all 52 positions, so it does
**not** cancel in ddG). A **residue-disjoint probe** — fit on 19 residue types, predict the
held-out 20th:

| property | held-out R² |
|---|---|
| **hydropathy** | **+0.704** |
| charge | +0.511 |
| volume | +0.421 |
| identity | accuracy **1.000** (chance 0.050) |

**ProtT5 predicts the hydropathy of a residue type it has NEVER seen.** A descriptor block is a
deterministic function of identity, so **W6 cannot add information** — only regularisation, or
**coverage of residues absent from ProtT5's training data**. Our 28 are canonical, so that gap
does not exist here (§7).

### Reconciliation with Ofir Ezrielev's thesis (56 pages, read in full)

Three things in our build were **wrong** and are corrected in source:
1. **PCA-16 is NOT his method** — "PCA"/"SVD" appear **nowhere** in his thesis. **Our PCA is ours.**
2. **`ignore_3D=True` contradicts his 1826 descriptors** (= 1613 2D + 213 3D).
3. **15-of-20 is not a faithful scaling of 40-of-58**, and he gives **no justification for 40**.

**He concedes the language-model objection himself** (p.31): *"Physicochemical properties of AAs
are implicitly represented in these datasets … and accordingly, implicitly represented in the
embeddings."* **His argument is COVERAGE ONLY.** He also **found our offset problem
independently**: his leave-one-amino-acid-out RMSE tracks the train/test mean gap while r stays flat.

**A bug worth recording:** the plan doc's **histidine SMILES was a 4-carbon, 1-nitrogen ring**
(C7H10N2O2) instead of imidazole (C6H9N3O2) — invisible to a gate testing only L, D and F.

---

## 3. ⚠️ The wild-type anchor — real signal, but **not attributable**

| | |
|---|---|
| **Idea** | Pin the wild-type prediction so per-protein offsets cannot drift. |
| **Flag** | `--wt_anchor_weight` (factor A) |
| **Tested?** | ✅ 3 tags: `anchor_w0.3_s42_e14` **0.6105**, `anchor_w1.0_s42_e14` 0.5942, `anchor_w3.0_s42_e13` 0.5364. |

**Best is +0.047 over control — inside the ±0.060 band.** And there is a deeper defect:

> **In the factorial as run, factor A is PERFECTLY ALIASED with both seed and epoch.** The a0
> cells are seeds {1,42} at epochs {5,13}; the a1 cells are all seed 42 at epochs 12–14.
> `corr(A, epoch) = +0.62`, `corr(epoch, a_p) = +0.55`. **Any apparent A effect could be the
> anchor, the seed, or simply training longer.**

There **is** a signal worth resolving: the exposure→`a_p` coupling decays monotonically with
anchor weight (0.3 → r=0.587, 1.0 → 0.471, 3.0 → 0.400, versus 0.63–0.68 unanchored). Three
points is a trend, not a law.

**What is missing:** a clean 1-factor design — `{0, 0.3, 1.0, 3.0} × seeds {1,2,3}`, everything
else fixed, all scored at the val-selected epoch. **12 runs, 96 GPU-hours.** State the power limit
*before* running: with 3 seeds the SE is ~0.035, so **only an effect above ~0.07 is detectable**.
Primary metric **std(b_p)**, because the anchor is a `b_p` lever. `gld_slope_anchor_s42` is
running now, but that is a *combination* arm and does not resolve the aliasing.

---

## 8, 9, 10. Bonding, chain span, bidirectional edges — all real, all inside the noise

| idea | flag | tag | pooled | vs control |
|---|---|---|---|---|
| Bonding / edge attributes (src+dst one-hot + 16 RBF = 56 dims) | `--edge_features` | `gld_w7edge_s42_e14` | 0.5808 | **+0.017** |
| Longer chain reach (helix i→i+4, sheet) | `--gcn_span 4` | `gld_w7span4_s42_e12` | 0.5757 | **+0.012** |
| Edges point both ways | `--gcn_bidir` | `gld_u10bidir_s42_e14` | 0.5893 | **+0.026** |

**All three are BUILT, GATED, TESTED, and all three land inside the ±0.060 seed band.** These are
honest negatives at n=1 each: **not "they don't work", but "any effect is smaller than seed
noise, and we did not spend the seeds to resolve it."**

**A structural reason to expect little from block-style features here, worth stating in the
thesis:** **the GCN branch reads `x[:, :32]` only** — D(16) plus half of Fb. It never sees the
embedding, and **any block inserted at offset 48 is invisible to it.** W5/W6/W9/W11/W12 reach the
**GAT branch alone.** Say so in the write-up rather than claiming "the model uses it".

---

# PART VI — THE TRAPS THAT PRODUCED FALSE IDEAS

These are not ideas; they are the reasons some "ideas" existed at all.

**1. The signature failure mode: code runs, prints a success line, feature never read.**
`gate_noop.py` (30 checks) exists solely for this. Four silent no-op holes were found and closed —
including **PEMGraphTransformer silently dropping inserted blocks**: 726 sentinel columns at
offset 48 survived reassembly with **0 reaching the node vector, and nothing raised.**
**Verify ARTIFACTS with `ls`, never a DONE message.**

**2. 2K5H's reference row was a MUTANT.** `2K5H.csv` concatenates three backgrounds and the wrong
one sorts first: row 0 is `2K5H.pdb_G11S` (dG 1.723) instead of `2K5H.pdb` (dG 4.805) at index
2738. Both are labelled `wt` because G11S genuinely is the WT *of its own background*.
**A +3.0824 kcal/mol shift on all 1,125 labels.** Corrected copy in **`data_fixed/`** (the source
tree is read-only); `gate_refrow.py` now counts distinct WT backgrounds.

**The same bug produced a second false lead:** the "**26,315 double mutants**" are all
`2K5H.pdb_G11S_<mut>` — single mutations on a background. **There are ZERO true double mutants.**
**One malformed file produced two false leads.**

**3. Escaping n=28 via the 100k catalogue is impossible.** It has **no sequence column** — joins
are **0/226** training PDB ids, **0/21** test, **0/340** by protein_id. Its one live use is
measuring distribution shift: our 32–74 aa regime was **0.1206% of pre-training residues (1 in
830)**, 77.4% NMR vs 8.6%. **But that is NOT the cause of b_p** (r = +0.043, p = 0.829).

**4. Three scoring bugs hid 12 finished cells**, because `run_calib_eval.sh` prints
`DONE … → abl_<tag>.csv` **after** the crash. **A finished training run is not a result.**

**5. Epoch selection is clean, and here is the proof.** `run_calib_eval.sh` line 15 selects the
best epoch by **validation** ddG-PCC parsed from the training log, then scores test **once**.
**We did not peek.** Two runs have six scored epochs each — `gld_slope1.0_s42` and
`gld_dg_coil_s42` — **deliberately, to measure a trajectory** (does `r` stay flat while `s` rises).
Both are declared in `results/trajectory_runs.txt` and enforced by `gate_epoch_selection.py`.
**If anyone ever quotes the best of those six as a headline, that IS test-set peeking.**

---

# WHAT TO DO NEXT — ranked by (value × readiness)

| | action | cost | why |
|---|---|---|---|
| **1** | **Score the W12 arms** (`gld_w12_*`, 5 queued) | 8 GPU-h each | **Built, gated 18/18, zero CSVs.** Targets the largest measured deficit (lost information, r not s) with the corrected mechanism (dG_transfer R²=0.753 vs volume 0.031). **The single best untested idea.** |
| **2** | **Finish the slope seed replication** (`gld_slope1.0_s1/s2/s3`, running) | free | The only proven lever is **n=1** and its +0.0575 sits just under the ±0.060 band. |
| **3** | **Diagnose the LORO descriptor collapse** | CPU | It is collapsing *again* at RMSE 2.541 with `a_p=nan`. The scale-clash fix did not work; suspect rank deficiency (16 PCA cols replacing 20 one-hot). |
| **4** | **Run `--pooled_corr_weight`** | 8 GPU-h | Built, never run, attacks pooled PCC directly. |
| **5** | **Characterise the two b_p outlier proteins** | CPU | 78% of the +0.14 oracle gain, and neither is a structural outlier. |
| **6** | Re-score 2K5H properly from `data_fixed/` | CPU | Current corrections shift a column, but `pred_ddG` was also computed against the buggy reference. |
| **7** | Clean factor-A design | 96 GPU-h | Only after capacity frees. |

**Do not re-run:** the ν sweep (closed), the coil dG arm (rejected), `--dg_length_norm`
(degenerate), any D1/`--unfolded_emb zero` arm (14.4σ dead), ligands/metals/complexes/BSA on
these 28 (zero variance — the data format cannot express them).

---

*Every "TESTED" row names a run tag backed by a CSV in `eval_results/`. Every untested row says so
in plain words. `scripts/gate_g4_cpu.py` re-run at the end of this work:*
**`baseline … PASS dG=-0.0030 width=1092`, ALL PASS — unchanged.**
