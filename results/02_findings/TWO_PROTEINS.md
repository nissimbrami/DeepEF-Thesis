# LINE 3 — The two proteins that carry 78% of the oracle gain

**Bottom line: they are two completely different things, and one of them is a data bug.**

`2K5H` is not a model failure at all. Its eval reference row is a **mutant background**, not the
wild type, which shifts every one of its 1125 ddG labels by **+3.0123 kcal/mol**. Fixing the
reference row removes **46% of the entire oracle gain** (+0.1125 -> +0.0611) and drops 2K5H out of
the top-2 |b_p| list on 10/10 checkpoints.

`2KVS` is a genuine, replicated model failure on clean data. It is *not* an offset case and it is
*not* ill-conditioned: its true ddG spread is above average (rank 22/28, z = +0.78). The model
loses the burial channel on this protein specifically — `corr(rel_SASA, pred) = -0.054` versus
`+0.367` on the other 26, while the truth still has `corr(rel_SASA, true) = +0.615`.

After both are handled honestly, the 0.59 -> 0.71 story becomes **0.5953 -> 0.6564**, and even that
residual is 59% carried by the new top-2. **The headline number is not robust.**

---

## 0. Provenance — how the per-mutation identities were recovered

The 10 eval CSVs (`eval_results/abl_*.csv`) carry only `protein, deltaG, pred_deltaG, ddG,
pred_ddG` — no mutation identity. I reconstructed the exact row order by replaying the selection in
`Megascale-fineTuning/evaluate.py::load_test_protein_data` (drop `ins|del`, drop `ddG_ML == '-'`,
intersect `name` with `data/ThermoMPNN/mega_test.csv`, then `list(set(...))`).

Verification is exact on **all 28 proteins**: row counts match to the row and `max|deltaG_recon -
deltaG_eval| < 4.7e-07` (float32 round-trip). This is the artifact, not a success message:
`twoprot/permut.csv` holds 28,172 rows with `protein, mut, wt, pos, mu, ddG, pred, rel_SASA,
helix, ext, buried, err`.

---

## 1. Complete characterisation

### Structure — neither is a structural outlier (confirms the prior finding)

| feature | 2K5H | z | 2KVS | z |
|---|---|---|---|---|
| length L | 62 | +0.63 | 67 | +1.15 |
| mean_rel_SASA | 0.3615 | -0.67 | 0.3431 | -1.14 |
| frac_buried | 0.3387 | +0.12 | 0.4328 | **+1.88** |
| frac_helix | 0.4194 | -1.17 | 0.9552 | +1.37 |
| frac_ext | 0.5484 | +1.25 | 0.0448 | -1.30 |
| radius of gyration | 10.04 | -0.39 | 10.53 | +0.11 |
| contact order | 0.2795 | +0.66 | 0.1603 | -1.12 |
| pLDDT mean | 87.14 | -0.64 | 94.20 | +1.02 |
| pLDDT min | 52.02 | -0.91 | 68.20 | +0.40 |
| frac hydrophobic | 0.4194 | +0.55 | 0.4478 | +1.18 |
| frac charged | 0.2903 | -0.38 | 0.1940 | -1.36 |
| net charge | -2 | -0.45 | -3 | -0.72 |
| frac Gly | 0.1129 | +1.26 | 0.0299 | -0.96 |
| **max abs z (14 features)** | **1.26** (frac_G) | | **1.88** (frac_buried) | |

Sequences (both single-chain, ligand-free, metal-free monomers):

```
2K5H (62aa) AVSDRLIGRKGVVMEAISPQNSGLVKVDGETWRATSGTVLDVGEEVSVKAIEGVKLVVEKLE
2KVS (67aa) TFYNFIMGFQNDNTPFGILAEHVSEDKAFPRLEERHQVIRAYVMSNYTDHQLIETTNRAISLYMANL
```

2KVS is the most buried and most helical protein in the set, but at |z| = 1.88 that is still
inside the cohort. **At n = 28 nothing here reaches |z| = 2.** The prior finding stands.

### The ddG distributions — this is where they separate from everyone else

| | n | true ddG mean | sd | frac < -0.5 | frac > +0.5 | pred mean | pred sd | mean err | MAE |
|---|---|---|---|---|---|---|---|---|---|
| **2K5H** | 1118 | **+2.044** | 1.586 | 0.122 | **0.832** | +0.298 | 0.920 | **-1.746** | 1.848 |
| **2KVS** | 1202 | -1.200 | 1.148 | 0.632 | **0.000** | **+0.659** | 0.672 | **+1.859** | 1.880 |
| other 26 | 25852 | -0.781 | 1.018 | 0.490 | 0.038 | -0.414 | 0.692 | +0.367 | 0.600 |

Both have MAE ~1.87 against a cohort MAE of 0.600 — 3.1x worse — but for **opposite reasons**.
2K5H's ddG column says 83% of its mutations are *stabilizing* (cohort: 3.8%). 2KVS's says none
are, yet the model predicts a *positive* mean.

---

## 2. WHY the model fails — two different answers

### 2K5H — a data bug in the reference row, not a model failure

`data/Processed_K50_dG_datasets/mutation_datasets/2K5H.csv` contains **15 rows labelled
`mut_type == 'wt'`, at three distinct dG levels**:

```
2K5H.pdb_G11S       dG 1.7231   <- eval row 0
2K5H.pdb_G11S_wtm   dG 1.8069
2K5H.pdb_G23A       dG 1.5646
2K5H.pdb_G23A_wt*   dG 1.58-1.60
2K5H.pdb            dG 4.8055   <- the ACTUAL wild type
2K5H.pdb_wtm/h/y/e  dG 4.55-4.75
```

The file pools three constructs — the true WT and two *stabilized mutant backgrounds*
(`G11S`, `G23A`) — all tagged `wt`. `evaluate.py` computes `ddG = deltaG - deltaG.iloc[0]`, and
row 0 lands on `2K5H.pdb_G11S` at dG 1.723, which is **3.08 kcal/mol less stable than the true
WT**. The WT dG is 2.80 kcal/mol *below* the protein's own median mutant dG, and 85.2% of its
mutants score as more stable than "WT" — physically absurd for a natural protein.

The check against the source label is decisive:

| protein | corr(source ddG_ML, our ddG) | mean offset (ours - source) |
|---|---|---|
| **2K5H** | **0.9967** | **+3.0123** |
| 2KVS | 1.0000 | +0.0023 |
| all 27 others | 1.0000 | -0.647 .. +0.634 |

Correlation 0.9967 with a +3.01 constant offset: this is a **pure shift**, and 2K5H is the only
protein of 28 outside +-0.65 and the only one where r != 1.0000. `b_p(2K5H)` is not a calibration
property of the model — it is the pipeline's mis-chosen origin.

**This is not an isolated file.** Auditing all 862 mutation files:

- 367 have >= 2 `wt`-labelled rows
- **37 (10.1%) have row 0 sitting on a mutant background**
- 37 (10.1%) have a WT dG spread > 1.0 kcal/mol; 29 (7.9%) > 2.0
- worst: `5GU9` 11.03, `1UBQ` 9.70, `2KT8` 8.63, `2LSS` 7.90, `1SIF` 7.85 kcal/mol

Within the 28 test proteins, **13 of 28** have >= 2 WT dG clusters; 2K5H is the only one where the
clusters are far enough apart (1.56 .. 4.81) to matter, but the *mechanism* is present in half the
benchmark.

### 2KVS — a genuine model failure, and the narrow-range hypothesis is falsified

The task asked whether `a_p = 0.094` reflects a tiny true ddG spread. **It does not.**

- true ddG sd = **1.1474**, rank **22/28** (z = **+0.78**) — above average, not narrow
- true ddG IQR = 1.8845, the **2nd largest of 28**; range -4.11 .. +0.49
- pred ddG sd = 0.6716 — the predictor is *not* stuck; it varies, just with the wrong thing
- `a_p = 0.0939, SE = 0.0166`: **t = +5.64 vs 0** (significantly non-zero) and **t = -24.25 vs the
  cohort median 0.4975** (significantly flatter). Not an ill-conditioned fit.
- corr(true_sd, a_p) over 28 proteins = **-0.2516, p = 0.197** — below the |r| = 0.374 threshold,
  i.e. spread does not predict slope at all.

So **2KVS should not be excluded or down-weighted**; it is a real failure on real data.

**What actually breaks.** The model's normal signal on this benchmark is burial. On the other 26
proteins `corr(rel_SASA, pred) = +0.367` against `corr(rel_SASA, true) = +0.452`. On 2KVS the truth
is *stronger* than usual (`+0.615`) but the model's prediction is **`-0.054` — the burial channel
is gone**. What the prediction tracks instead is sequence position (`corr(pos, pred) = -0.411`,
while `corr(pos, true) = +0.196`), which is anti-correlated with the truth.

| driver | 2KVS corr(pred) | 2KVS corr(true) | other 26 corr(pred) | other 26 corr(true) |
|---|---|---|---|---|
| rel_SASA | **-0.054** | **+0.615** | +0.367 | +0.452 |
| d(hydropathy) | +0.350 | +0.390 | +0.393 | +0.331 |
| d(volume) | +0.289 | +0.352 | +0.193 | +0.136 |
| position | **-0.411** | +0.196 | — | — |

Per-mutation stratification confirms the channel is inverted, not merely weak:

```
2KVS   buried  n= 538  a=0.241  PCC=0.355  meanErr=+2.719   <- worst where burial matters most
       exposed n= 664  a=0.138  PCC=0.156  meanErr=+1.162
       helix   n=1155  a=0.108  PCC=0.185
other  buried  n=8243  a=0.458  PCC=0.593  meanErr=+0.646
26     exposed n=17609 a=0.360  PCC=0.576  meanErr=+0.237
```

The errors are **not uniform**: 41.8% of 2KVS's 67 positions have a per-position slope `a < 0.2`
and 20.9% have `a < 0` (the model moves the *wrong way*). Cohort-normal 2K5H, by contrast, has
only 19.4% and 9.7%. 2KVS's failure is broad but concentrated at buried positions:
`V23 (rel=0.000) a=-0.158`, `F29 (rel=0.009) a=-0.074`, `T14 (rel=0.294) a=-0.084` — all core
residues where the model is flat or inverted.

2KVS is a monomeric, 95.5%-helical, unusually buried (frac_buried z = +1.88), low-contact-order
(z = -1.12) bundle with the fewest charged residues in the set (z = -1.36). The most plausible
reading is that its long-helix/low-contact-order topology is under-represented in training, so the
burial term learned elsewhere does not transfer — but **with n = 28 this cannot be tested here**;
it is a generalisation claim, not a measured result.

The failure replicates on **10/10 checkpoints**: `a_p` in 0.019 .. 0.207 (median 0.100, cohort
median 0.4975) and per-protein PCC 0.121 .. 0.510 (median 0.216, cohort median ~0.80). It is not a
single-checkpoint artefact.

### 2K5H is otherwise a perfectly ordinary, well-fit protein

Once the offset is set aside: `a_p = 0.4990` (cohort median 0.4975 — dead on) and per-protein
PCC = **0.8610** (cohort median 0.798 — above average). **2K5H is one of the model's better
proteins.** Its entire contribution to the calibration story was the mis-chosen origin.

---

## 3. Robustness — how much of 0.59 -> 0.71 survives?

### As published (reference bug in place), means over 10 CSVs

| set | raw pooled PCC | oracle | gain | gain retained |
|---|---|---|---|---|
| all 28 | 0.5994 | 0.7119 | **+0.1125** | 100% |
| drop 2K5H (n=27) | 0.5731 | 0.6391 | +0.0659 | 59% |
| drop 2KVS (n=27) | 0.6504 | 0.7367 | +0.0863 | 77% |
| **drop both (n=26)** | **0.6293** | **0.6685** | **+0.0392** | **35%** |

Dropping just two of 28 proteins raises the raw baseline by +0.0299 and destroys **65% of the
oracle gain**. Correcting only those two recovers 0.5994 -> 0.6873 (78% of the gain, mean over 10
CSVs, range 60-88%); correcting the other 26 recovers only -> 0.6152.

### With the 2K5H reference row FIXED (the honest number)

Referencing both `deltaG` and `pred_deltaG` to `2K5H.pdb` (eval-order row 2) instead of row 0.
Only 2K5H changes; the other 27 already reference a true WT construct.

| quantity | reference bug | reference fixed | change |
|---|---|---|---|
| raw pooled PCC | 0.5994 | 0.5953 | -0.0041 |
| oracle pooled PCC | 0.7119 | **0.6564** | **-0.0554** |
| **oracle gain** | **+0.1125** | **+0.0611** | **-46%** |
| std(b_p) | 0.1893 | 0.1348 | -29% |
| b_p(2K5H) | -0.6891 | **-0.1020** | -85% |
| b_p(2KVS) | +0.3761 | +0.3761 | 0 |
| share of gain in top-2 | 78.0% | 59.2% | |

The raw number barely moves (-0.0041) exactly as Pearson shift-invariance predicts. **The oracle is
what collapses** — because the oracle's job is to remove per-protein offsets, and one of the two
largest offsets was never a model property.

After the fix 2K5H leaves the top-2 on **10/10 CSVs**, replaced by `3DKM` or `r18_3_TrROS_Hall`;
2KVS remains in the top-2 on 6/10.

| CSV | top-2 as published | top-2 after fix |
|---|---|---|
| abl_anchor_w0.3_s42_e14 | 2K5H, 3DKM | 3DKM, r18_3_TrROS_Hall |
| abl_anchor_w1.0_s42_e14 | 2K5H, 3DKM | 3DKM, r18_3_TrROS_Hall |
| abl_anchor_w3.0_s42_e13 | 2K5H, 2K28 | 2K28, 2K1B |
| abl_calib_ctrl_repro2_e14 | 2KVS, 2K5H | 2KVS, 3DKM |
| abl_p3_slope3.0_s42_e8 | 2KVS, 2K5H | 2KVS, r18_3_TrROS_Hall |
| abl_sigma_seed1_e13 | 2KVS, 2K5H | 2KVS, 3DKM |
| abl_sigma_seed2_e10 | 2K5H, 3DKM | 3DKM, r18_3_TrROS_Hall |
| abl_sigma_seed3_e13 | 2KVS, 2K5H | 2KVS, 3DKM |
| abl_sigma_seed42_e9 | 2K5H, 2KVS | 2KVS, 3DKM |
| abl_sigma_seed4_e14 | 2K5H, 2KVS | 2KVS, r18_3_TrROS_Hall |

### Per-CSV, as published (part 4: not a single-checkpoint artefact)

| CSV | raw all | oracle all | raw drop-both | oracle drop-both | only-2 | only-26 |
|---|---|---|---|---|---|---|
| abl_anchor_w0.3_s42_e14 | 0.6157 | 0.7294 | 0.6313 | 0.6727 | 0.7003 | 0.6459 |
| abl_anchor_w1.0_s42_e14 | 0.6018 | 0.7066 | 0.6044 | 0.6441 | 0.6785 | 0.6309 |
| abl_anchor_w3.0_s42_e13 | 0.5972 | 0.6834 | 0.5632 | 0.6173 | 0.6448 | 0.6342 |
| abl_calib_ctrl_repro2_e14 | 0.5910 | 0.7113 | 0.6440 | 0.6782 | 0.6873 | 0.6152 |
| abl_p3_slope3.0_s42_e8 | 0.5624 | 0.6974 | 0.6229 | 0.6633 | 0.6691 | 0.5909 |
| abl_sigma_seed1_e13 | 0.5948 | 0.7005 | 0.6427 | 0.6675 | 0.6828 | 0.6123 |
| abl_sigma_seed2_e10 | 0.6575 | 0.7503 | 0.6589 | 0.6972 | 0.7230 | 0.6859 |
| abl_sigma_seed3_e13 | 0.5799 | 0.7178 | 0.6532 | 0.6943 | 0.6884 | 0.6085 |
| abl_sigma_seed42_e9 | 0.5929 | 0.7016 | 0.6215 | 0.6639 | 0.6711 | 0.6234 |
| abl_sigma_seed4_e14 | 0.6008 | 0.7204 | 0.6510 | 0.6860 | 0.6959 | 0.6253 |
| **mean** | **0.5994** | **0.7119** | **0.6293** | **0.6685** | **0.6873** | **0.6152** |

The pattern holds on 10/10. The drop-both raw beats the all-28 raw on 10/10.

---

## 4. Plain statement

**The result is carried by two proteins, and one of them is a data bug.**

Of the +0.1125 oracle gain:
- **~46 percentage points of it are the 2K5H reference-row bug** — a pipeline artefact, removable
  today, nothing to do with the model.
- **~19 more points are 2KVS**, a real but single-protein failure where the model loses its burial
  channel.
- **What is left is +0.0611**, and 59% of *that* still sits in the new top-2 (`3DKM`,
  `r18_3_TrROS_Hall`, `2KVS`).

The thesis cannot claim "offset removal lifts pooled PCC from 0.59 to 0.71." The defensible
statement is **0.5953 -> 0.6564**, and it must be reported alongside the fact that removing the two
most extreme proteins leaves +0.0392.

This does **not** overturn the calibration story — `std(b_p)` after the fix is still 0.1348, the
oracle still gains +0.0611, ICC(b_p) = 0.898 was measured on b_p across seeds and is unaffected by
a constant per-protein shift, and the `mean_rel_SASA` <-> `a_p` (r = +0.714) and `frac_buried`
<-> `b_p` findings are untouched by 2K5H's origin. It does mean the *magnitude* was inflated
roughly two-fold.

It also **strengthens** the corrector negative result. The corrector was being asked to predict
`b_p(2K5H) = -0.72` from structure. No structural feature could ever have predicted it, because it
was not structural — it was a mis-chosen row. The corrector's failure was correct behaviour.

### Recommended actions

1. **Fix the reference row.** In `evaluate.py::load_test_protein_data`, replace
   `deltaG.iloc[0]` with the row whose source `name` is the bare `<PROTEIN>.pdb` construct, or
   whose `ddG_ML` is nearest zero among `mut_type == 'wt'` rows. Affects 13/28 test proteins
   materially only for 2K5H, but 37/862 files repo-wide. **This changes the headline number** and
   must be done before the thesis is written.
2. **Re-run every calibration number** downstream of `b_p` on the fixed reference. The b_p
   corrector, the anchor levers, and `std(b_p) = 1.5741` in dG-space all inherit this origin.
3. **Keep 2KVS in.** It is a real failure on clean, above-average-spread data, replicated 10/10.
   Excluding it would be exclusion-by-inconvenience. Report it as the model's known failure mode:
   loss of the burial channel on high-burial, low-contact-order, all-helical folds.
4. **Report drop-both as a robustness row** in the thesis, not as the headline.

### Caveats

- The fix uses the source `ddG_ML` / construct name as ground truth for which row is WT. If the
  MegaScale release itself mislabels 2K5H, the fix inherits that. The +3.01 offset with r = 0.9967
  against an independent column makes this unlikely.
- Fixing the reference row **re-references the predictions too**, so it is a re-scoring of existing
  predictions, not a re-training. Whether a model *trained* with the corrected 2K5H labels behaves
  differently is unmeasured and needs a GPU run.
- The "under-represented topology" explanation for 2KVS is a hypothesis. With n = 28 (|r| < 0.374
  is indistinguishable from zero at p = 0.05) it cannot be tested on this benchmark. It is a
  generalisation claim requiring a larger held-out set.
- All numbers here are re-scorings of the 10 existing eval CSVs. No model was retrained.

---

## Files

| path (on cluster, `/home/nissimb/DeepPEF/`) | contents |
|---|---|
| `twoprot/recon.py` | exact eval-row reconstruction + alignment proof |
| `twoprot/analyse.py` | robustness / oracle-gain across 10 CSVs |
| `twoprot/charac.py` | structural + per-mutation + positional characterisation |
| `twoprot/wtref.py` | WT-reference-row diagnostic, median re-referencing |
| `twoprot/wtsrc.py` | source-data trace; 28-protein offset audit |
| `twoprot/fixref.py` | **the decisive fix-and-recompute** |
| `twoprot/kvs.py` | 2KVS failure diagnosis; 862-file WT audit |
| `twoprot/permut.csv` | 28,172 rows: mutation identity + SASA/SS + pred/true/err |
| `twoprot/robustness.csv`, `twoprot/fixref.csv` | the tables above |
| `twoprot/wt_audit_all.csv` | WT-row audit over all 862 mutation files |
| `twoprot/structural.csv`, `twoprot/pos_2K5H.csv`, `twoprot/pos_2KVS.csv` | supporting |

**Gate:** `scripts/gate_g4_cpu.py` -> `baseline ... PASS dG=-0.0030 width=1092`, `G4-CPU: ALL PASS`.
No repo file was modified; all work is read-only analysis in a new `twoprot/` directory.
