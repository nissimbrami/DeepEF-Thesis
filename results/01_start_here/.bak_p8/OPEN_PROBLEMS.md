# DeepEF — OPEN PROBLEMS: a detailed plan for every unsolved item
### 2026-09-08. One section per problem. Each is specified to the level a person could execute without asking a question.

---

## P0 — THE EPOCH-SELECTION AUDIT (do this first; it validates or invalidates every number)

**The question.** Is our headline `0.6382` the epoch chosen on the VALIDATION set, or the best of 15
epochs on the TEST set? If the latter, we did exactly what we criticised in the prior work.

**The answer, verified in code.** `Megascale-fineTuning/run_calib_eval.sh` line 15 reads
`# 1) best epoch by VAL overall ddG-PCC`, and selects `BEST_E` by parsing `ddG PCC=` lines out of
the **training log**, which `validate()` emits from `self.val_ds`. The test set is then scored
**once**, at that epoch. **We did not peek.**

**The residual risk, and it is real.** Two runs have SIX scored epochs each:

    gld_slope1.0_s42   e0, e4, e8, e12, e13, e14
    gld_dg_coil_s42    e0, e4, e8, e12, e13, e14

They were scored at multiple epochs deliberately, to measure a *trajectory* (does `r` stay flat
while `s` rises). That is a legitimate use. **But if anyone ever quotes the best of those six as a
headline, that IS test-set peeking.** Every other run has exactly one scored epoch.

**The fix, in full.**
1. Add `scripts/gate_epoch_selection.py`. For every `eval_results/abl_*.csv`, parse the `_e<N>`
   suffix, group by run tag, and FAIL if any tag has more than one scored epoch **unless** the tag
   appears in an explicit allowlist file `results/trajectory_runs.txt`.
2. Seed that allowlist with the two trajectory runs above and a one-line reason each.
3. In the allowlist, record which epoch is the **canonical** one (the val-selected epoch) for each.
   For both runs that is `e14` (the val log's argmax), and any table must quote that one.
4. Add to `scripts/loop_check.sh` so it runs every round.
5. Re-audit `results/01_start_here/` and grep every quoted score against the canonical epoch.

**Cost:** ~1 hour, CPU only. **This gates the credibility of every other number, so it is P0.**

---

## P1 — THE FLORY ν EXPONENT WAS NEVER SWEPT

**What is unresolved.** `--flory_nu` has `default=0.5` and `sacct` shows **zero runs ever set it**.
Every coil arm, including the rejected dG arm, used ν=0.5.

**Why that is the wrong physics.** ν=0.5 is the **theta state** — a chain in a solvent where
monomer–monomer and monomer–solvent interactions cancel, so it behaves as an ideal random walk. A
denatured protein in water is not in a theta solvent; it is a **self-avoiding walk with ν≈0.588**.
Kohn et al. (2004) *PNAS* measured denatured-state R_g ∝ N^0.598 across 28 proteins — the
canonical experimental result for exactly this quantity.

**What it costs, computed at b = 5.82 Å:**

| \|i−j\| | ν=0.5 | ν=0.588 | error |
|---|---|---|---|
| 10 | 18.40 Å | 22.54 Å | 22% |
| 20 | 26.03 Å | 33.88 Å | 30% |
| 50 | 41.15 Å | 58.07 Å | **41%** |

Our proteins are 43–72 residues, so |i−j| reaches ~70 — **the regime where ν matters most.**

**Why this does not resurrect the coil's rejected result.** Two separate claims must stay apart:
(a) the coil's *motivating* MAE improvement was a **mean-bias shift**, not a dispersion reduction —
all 28/28 proteins under-predicted, so MAE ≡ |mean(b_p)|, and std(b_p) actually **rose** 0.9967 →
1.1231. That rejection stands and is independent of ν.
(b) the reference state was **never given the physically correct shape**. That is untested.

**The plan.**
1. **Free inference sweep first.** Extend `scripts/w0_dg.py` to loop ν ∈ {0.5, 0.55, 0.588, 0.62,
   0.65} × b ∈ {4.97, 5.82} on a **frozen checkpoint**. The coil is an input transformation, so
   this needs no training and costs minutes. b=4.97 Å is the Kohn-calibrated value; 5.82 Å is ours.
2. **Report std(b_p), NOT MAE.** MAE is the metric that misled us. The success criterion is a
   *dispersion* reduction: std(b_p) below the 0.9967 baseline.
3. **Add the degenerate check.** Report std(pred WT dG) alongside. If it collapses toward zero, the
   arm is deleting the prediction (the `--dg_length_norm` failure) and the number is void.
4. **Only if a ν beats baseline on dispersion**, train one arm at that ν with `--coil_b fixed`, on
   the golden lane, 15 epochs, seed 42.
5. Gate: `scripts/gate_coil_nu.py` asserting ν is read (not silently defaulted), that d(i,j) scales
   as |i−j|^ν to 1e-6, and that ν=0.5 reproduces the current behaviour byte-identically.

**Cost:** minutes for the sweep, 8 GPU-hours only if the sweep justifies it.

---

## P2 — LORO WAS NEVER ACTUALLY TESTED

**What happened.** `gld_loroW_desc` collapsed to a constant: ddG RMSE frozen at exactly **2.541**
for ~11 epochs, PCC oscillating about zero.

**The cause, measured directly:**

    descriptor table:  sd = 1.0000, max|v| = 6.04, 16 columns  →  ~16.8× the energy of one-hot
    one-hot:           sd = 0.2179, max = 1.0,     20 columns

The descriptors were z-scored to unit variance *per column* and concatenated alongside one-hot, so
they carried roughly **17× the signal energy of residue identity**. The network optimised against
what was loud, and residue identity — the thing the LORO experiment is about — was drowned.

**This is an execution fault, not a refutation.** The hypothesis (a continuous descriptor space can
represent a residue one-hot never saw) has **not been tested**.

**The plan.**
1. A re-run is already on the golden lane as `gld_loroWdesc2_s42` using
   `--aa_descriptors mordred_pca16_only`, which **replaces** one-hot rather than appending — this
   removes the competition rather than rebalancing it.
2. **Add a scale gate** (`scripts/gate_descriptor_scale.py`): assert the descriptor block's mean
   row energy is within 3× of the one-hot block's. FAIL otherwise. This is the check whose absence
   cost 8 GPU-hours.
3. If the `_only` arm also collapses, the next variant is explicit down-scaling: divide the table by
   `sqrt(K/20)` so total block energy matches one-hot by construction.
4. **The prediction is on record** (CHECKPOINT 26): the descriptor arm should NOT beat one-hot,
   because ProtT5 already predicts held-out-residue hydropathy at R²=0.704. If it DOES win, that
   overturns the redundancy conclusion and must be reported as a surprise, not rationalised.
5. Report Spearman on **mutations to tryptophan only**, per arm, using `scripts/k13_muttype.py`'s
   join. Also report overall pooled PCC to confirm the holdout did not simply damage both models.

**Cost:** already running; the gate is ~30 minutes.

---

## P3 — W12 SIDE-CHAIN BLOCK IS BUILT BUT NOT WIRED

**Status.** `scripts/sidechain_features.py` and `scripts/gate_sidechain.py` exist and the gate
passes **18/18**. **There is no `--sidechain_features` flag in `train.py`**, so the block cannot fire.

**Why it matters.** This addresses the largest measured deficit: the model compresses mutations to
hydrophobic residues twice as hard (slope 0.27–0.31 vs 0.53–0.57), and **Spearman degrades in
lockstep** (corr with KD −0.734 and −0.740), so it is **lost information** that no calibration can
recover. Verified: `corr(dG_transfer, measured slope) = −0.922` versus volume's −0.408.

**The wiring, precisely.**
1. `train.py`: add `--sidechain_features` (`action='store_true'`, default off) and set
   `CFG.sidechain_features`.
2. `train_utils.get_graph` / `get_unfolded_graph`: insert the 4-column block at
   `48 + solv_dim + desc_dim`, passing `folded=True` in `get_graph` and `folded=False` in
   `get_unfolded_graph` — the burial-weighted columns MUST be zero in the unfolded pass, because
   the folded-minus-unfolded difference *is* the hydrophobic driving force.
3. `model/hydro_net.py`: grow `fc1_gcn` and `fc1_gat` by 4. **Do NOT touch** `fc2_*`,
   `inst_norm1`, `inst_norm2` or `fc_in_dim` — they project to fixed internal widths.
4. `PEMGraphTransformer.__init__`: add `sidechain_features` to the guard that already raises for
   the other five block levers, since it slices left-anchored and would silently ignore the block.
5. **Note the GCN branch reads `x[:, :32]` only**, so this block reaches the GAT branch alone. Say
   so in the write-up rather than claiming "the model uses it".
6. Run `scripts/gate_g4_cpu.py` — must still print `dG=-0.0030 width=1092` with the flag OFF, and
   `width=1096` with it on.

**Scoring.** It acts on BOTH channels: the mutated position's identity delta survives ddG, while
the burial-weighted term is a folded-state quantity acting on b_p. **Report ddG pooled/PP AND
std(b_p).** Scoring it on ddG alone would repeat the W5 mistake.

**Cost:** ~2 hours wiring, 8 GPU-hours per arm.

---

## P4 — FACTOR A IS NOT ESTIMABLE

**The defect.** In the factorial as run, factor A (`--wt_anchor_weight`) is **perfectly aliased with
both seed and epoch**: the a0 cells are seeds {1,42} at epochs {5,13} while the a1 cells are all
seed 42 at epochs 12–14, giving corr(A, epoch) = +0.62 and corr(epoch, a_p) = +0.55. **Any apparent
A effect is unattributable** — it could be the anchor, the seed, or simply training longer.

**Why it matters.** The anchor is the one lever aimed directly at b_p, and there is a measured
signal worth resolving: the exposure→a_p correlation decays monotonically with anchor weight
(0.3 → 0.587, 1.0 → 0.471, 3.0 → 0.400), suggesting the anchor suppresses the coupling.

**The plan.**
1. A clean 1-factor design: `--wt_anchor_weight ∈ {0, 0.3, 1.0, 3.0}` × seeds {1,2,3},
   **everything else fixed**, all 15 epochs, all scored at the val-selected epoch. 12 runs.
2. **Balance is the point:** every anchor level gets the same three seeds and the same epoch budget.
   Nothing else may vary.
3. Report against the ±0.060 seed band. With 3 seeds per level the standard error is ~0.035, so
   only an effect above ~0.07 is detectable — **state that power limit before running**, so a null
   is interpretable.
4. Primary metric is **std(b_p)**, because the anchor is a b_p lever; pooled ddG PCC is secondary.
5. Add `scripts/gate_factorial_design.py` that checks any factorial for aliasing between a factor
   and seed/epoch before the runs are launched. This defect should have been caught by design.

**Cost:** 12 × 8 = 96 GPU-hours. **Do not start until P0 and the golden lane free up.**

---

## P5 — WHAT IS b_p? (the largest open scientific question)

**Status.** Eight explanations tested and rejected: label noise (dead by 200×), chain length
(r=+0.025), `--dg_length_norm` (degenerate), embedding norm (−0.29 not −0.99), distribution shift
(r=+0.043), the train/test mean gap (p=0.588), a feature-based corrector (95% of the gain fails out
of sample), and the reference state (the coil arm degrades everything).

**And yet the oracle says +0.14 of pooled PCC is available.** b_p is real (a random-offset null
cannot reproduce our triple, best L2 = 0.118) and learnable in principle (ICC = 0.898).

**The strongest remaining clue.** The **top-2 |b_p| proteins carry 78%** of the oracle gain and
**neither is a structural outlier** (max |z| ≈ 2.0 across 22 features). So b_p is not a smooth
function of structure — it is concentrated in a few proteins for a reason we have not identified.

**The plan.**
1. **Characterise the two outliers exhaustively.** Not 22 aggregate features — look at their
   per-residue predictions, their pLDDT profile, their mutation composition, and whether their WT
   dG is unusual relative to their own mutant distribution.
2. **Test the "model failure" hypothesis directly.** 2KVS has a_p = 0.094 and per-protein PCC =
   0.160 — the model essentially fails on it, and b_p absorbs the failure. Compute, across all 28,
   the relationship between per-protein PCC and |b_p|. If low-PCC proteins carry high |b_p|, then
   **b_p is a symptom, not a cause**, and correcting it is treating the wrong thing.
3. **The severing experiment retrained** (P6) tests whether b_p lives in E_u or E_f at all.
4. **Report the negative honestly if it stays negative.** "The offset is real, learnable in
   principle, and not predictable from any structural feature we tested" is a legitimate thesis
   result, and more useful than an overfit corrector.

**Cost:** CPU only for items 1–2.

---

## P6 — THE SEVERING EXPERIMENT IS UNINTERPRETABLE AS RUN

**What happened.** `f_resid = 0.8512` but the **OOD guard fired** (C3's E_u range collapsed to 0.21×
of C0), so per its own design the number **must not** be used to cancel the reference-state work.

**What survived.** `corr(E_u, E_f) = 0.5430` with 95% CI [0.294, 0.736]. **45% of the two energies'
combined variance is SHARED and cancels identically in E_u − E_f.** That is the first direct
measurement of a quantity that had never been reported, and it cuts against the premise that
var(E_u) dominance licenses a causal reading.

**The plan.**
1. The frozen-checkpoint ablation cannot answer this: zeroing the unfolded embedding pushes a
   trained network far out of distribution, so the variance collapse measures OOD degeneracy rather
   than reference-state dominance.
2. **Train a severed model**: `--unfolded_emb zero --flory_unfolded --coil_b fixed --coil_edges`,
   15 epochs, so the network *learns* with a protein-independent reference state.
3. Compare its std(b_p) against the D0 baseline. **If b_p does not shrink in a model trained that
   way, the reference-state programme is closed for good.**
4. Report cov(E_u, E_f) and corr(E_u, E_f) for the trained model too — the frozen-checkpoint value
   of 0.543 may itself be an artefact of the frozen state.
5. Keep all four OOD triggers active and print the verdict line first.

**Caveat to carry:** D1 (`--unfolded_emb zero`) alone destroys the model (14.4σ), so a severed model
may simply be a bad model. **The comparison must be std(b_p) at matched per-protein PCC**, or the
result is confounded.

**Cost:** 8 GPU-hours.

---

## P7 — THE 2K5H FIX IS AN APPROXIMATION

**Status.** 2K5H's reference row was a mutant background (`_G11S`, dG 1.723) instead of the true WT
(`2K5H.pdb`, dG 4.805) — a **+3.0824 kcal/mol** shift on all 1,125 of its labels. A corrected file
exists at `data_fixed/mutation_datasets/2K5H.csv`.

**What is approximate.** Every corrected number so far was produced by subtracting the constant from
the ddG column of existing eval CSVs. **But `pred_ddG` was also computed against the buggy
reference**, so the correction is only exact for the labels, not the predictions.

**The plan.**
1. Point the eval loader at `data_fixed/` and **re-score** rather than shifting a column.
2. Re-derive every affected number: the headline, std(b_p), the "top-2 carry 78%" claim (2K5H was
   one of the two), and every per-protein table.
3. Until then, **prefer the drop-2K5H basis** (27 proteins), which requires no approximation and is
   what this document and MASTER.md use.
4. `scripts/gate_refrow.py` already prevents recurrence by counting distinct WT backgrounds.

**Cost:** CPU, ~2 hours.

---

## P8 — THE HEADLINE HAS BEEN QUOTED ON THREE DIFFERENT BASES

**The defect.** The same "corrected headline" has appeared as 0.4899, 0.5646 and 0.5772 — because
each was averaged over a **different population** of eval CSVs. **Averaging a correlation across a
mixed population is not a measurement**, and one of those averages included D1 arms scoring as low
as −0.08.

**The plan.**
1. **Declare one canonical basis** and write it at the top of every document: *27 proteins (2K5H
   excluded), ddG metric, the 9 original-population eval CSVs, val-selected epoch only.*
2. Grep every quoted score in `results/01_start_here/` and re-derive it on that basis.
3. Add `scripts/gate_headline.py` that recomputes the canonical number and FAILS if any document
   quotes a different one.
4. Never average across runs that differ in factor D. Report D0 and D1 separately, always.

**Cost:** ~2 hours, CPU only. **Do this with P0 — they are the same audit.**

---

## Priority order

| | problem | cost | why this order |
|---|---|---|---|
| **1** | **P0** epoch-selection audit | 1 h CPU | validates or invalidates everything else |
| **2** | **P8** one canonical basis | 2 h CPU | same audit; the record currently disagrees with itself |
| **3** | **P1** ν sweep (inference only) | minutes | free, and the physics says we used the wrong branch |
| **4** | **P3** wire W12 | 2 h + 8 GPU-h | addresses the largest measured deficit |
| **5** | **P7** re-score 2K5H properly | 2 h CPU | removes an approximation from every number |
| **6** | **P5** characterise the two b_p outliers | CPU | the largest open question, cheapest next step |
| **7** | **P2** LORO gate | 30 min | the re-run is already going; the gate prevents recurrence |
| **8** | **P6** severing retrained | 8 GPU-h | decides the reference-state programme |
| **9** | **P4** clean factor-A design | 96 GPU-h | most expensive; wait for capacity |
