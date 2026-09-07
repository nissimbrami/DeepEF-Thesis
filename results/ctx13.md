
---

# CHECKPOINT 13 — 2026-09-07 — TASKS T11-T18 CLOSED, WORKING ALONE

All subagents were stopped at the user's instruction. Everything below was done and verified by
the main session directly, not by a worker report.

## What was verified by RUNNING it, not by trusting a claim

| task | result |
|---|---|
| T14 W9 metal | **RETIRED, and the guard really fires.** `hydro_net._sibling_block_dims(CFG)` raises RuntimeError when `metal_features` is set, returns 0 when not. `gate_w9.py` exits 0 with "W9 RETIREMENT VERIFIED". Adding a flag would have created a second dead path; W11 covers metals as ligand class METAL. |
| T15 het codes | `data/het_classification.csv`: all **8,187 codes**, **98.1% occurrence coverage** (452 hand-classified by frequency rank). Distribution: 106 crystallization_additive, 90 substrate, 80 modified_residue, 61 metal, 58 polymer_residue, 57 cofactor, rest unknown. Only **208 codes count as a bound ligand**. |
| T16 catalogue guard | `catalogue_schema.load_catalogue()` quarantines all 5 dead columns (**21 -> 18**); `assert_not_decoy_loss(['loss',...])` RAISES; BSA policy nulls **294** rows (246 original + the 48 internally inconsistent). |
| T17 Ofir divergences | **fixed in source** (see below) |
| T18 zero-variance guard | present in `autopilot.py` |
| T13 D1 slope guard | **fires.** On synthetic rows it rejected the HIGHEST-pooled cell (0.70) because its median slope was 0.10 < 0.30, logging `D1 REJECT ... slope collapse`. Ranking among eligible cells is by pooled, as documented. |
| T12 W5 on dG | **honest negative**, see below |
| T11 LORO | built, gated, and **submitted** |

## T17 — three documentation errors corrected in build_aa_descriptors_mordred.py

1. **PCA is OURS, not Ofir's.** The thesis does no dimensionality reduction; "PCA", "principal
   component" and "SVD" appear nowhere in it, and his 654 features enter the CNN as 654 channels.
   The CSV provenance now says so, so nobody can cite the projection as his method.
2. **1826 vs ignore_3D was self-contradictory.** 1826 = 1613 2D + 213 3D, a total only coherent if
   3D was requested — which is also what explains his 1826->1280 missing-value drop, since 3D
   descriptors return NaN without a conformer. With `ignore_3D=True` the ceiling is 1613. We keep
   ignore_3D deliberately, so our counts differ from his BY DESIGN.
3. **15-of-20 is not "three quarters" of 40-of-58.** 40/58 = 0.690 -> 13.8, so a faithful threshold
   is 13 or 14. Recorded as OUR choice. He gives no justification for 40 at all, and 2D Mordred
   cannot distinguish D from L, so his effective alphabet is well under 58.

## T12 — W5 burial scored on dG: NOT significant

`results/W5_ON_DG.md`, `results/w5_on_dg.json`.

| feature | target | pearson | spearman |
|---|---|---|---|
| mean_burial | b_p | **-0.343** | -0.282 |
| frac_buried | b_p | -0.039 | -0.007 |
| length | b_p | -0.368 | -0.349 |

**n = 19 of 28; |r| < 0.490 is indistinguishable from zero at p=0.05.** The sign agrees with the
replicated `frac_buried` result (-0.475, 10/10) but **-0.343 must not be reported as support.**

**Deliberately NOT done:** flipping `--burial_features` on a checkpoint trained without it. W5
widens 1092 -> 1095 and that checkpoint has no weights for the 3 new columns, so a forward pass
would measure untrained weights, not the lever.

**Two limits:** 9 of 28 proteins have no tensor directory, and the missing third is not random —
**2K5H is among them, one of the two proteins carrying 78% of the offset-removal oracle gain.**
And `length ~ b_p` (-0.368) is LARGER than burial's, with burial correlated to length.

**A units bug worth keeping:** the first run returned nan for every burial correlation because I
scaled coordinates by 0.1 before `compute_burial`. **The stored tensors are already in Angstrom**
and the cutoff is an Angstrom cutoff, so 0.1x saturated every neighbour count to exactly 1.0 with
std 0.0. Measured on 2KVS/2BTH: Angstrom -> mean 0.48/0.41 std 0.18/0.16; 0.1x -> 1.0/0.0.
**A saturated constant does not raise; it silently yields nan.**

## Data path worth recording

Per-protein tensors are NOT in `data/`. They are at
`/groups/keasar_group/casp15/meytav/protein_tensors/<PROTEIN>/coords_tensor.pt` (and
`mask_tensor.pt`), one directory per protein, exactly as `Megascale-fineTuning/train.py:28` loads
them. **9 of our 28 test proteins have no directory there.**

## Queue state

**15 golden jobs queued** (`--partition=rtx6000 --qos keasar`): the dG arm, `--slope_weight`, the
severing experiment, 10 starved evals, and now the LORO pair (`loroW_onehot` / `loroW_desc`,
differing ONLY in `--aa_descriptors`, both with `--resume`).
**Public lane untouched at 5/5.** A labmate still holds all 8 golden cards, so ours pend.

## All 11 gates green

g4_cpu (**dG=-0.0030 width=1092**), noop, resume (42/42), u2, w5, w7, w7_edge, u3u4, u5u6,
w9 (retirement verified), w8 (51/51).
