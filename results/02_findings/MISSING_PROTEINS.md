# LINE 2 — Recovery of the 9 "missing" test proteins

**Status: FULLY RECOVERED. 28/28 test proteins now have usable coordinate tensors.**
Gate after change: `G4-CPU: ALL PASS`, baseline `dG=-0.0030 width=1092`.

---

## 1. What was actually wrong

The brief said 9 proteins had "NO directory under `protein_tensors`". The truth is
slightly different, and the difference matters:

| root | dirs | of our 28 | with `coords_tensor.pt` |
|---|---|---|---|
| `/groups/keasar_group/casp15/meytav/protein_tensors` | 20 | 20 | **19** |
| `.../meytav/test_protein_tensors` | 0 (empty) | 0 | 0 |
| `.../meytav/MutationFineTuning/test_protein_tensors` | 75 | **27** | **27** |

Two separate faults produced the n=19:

1. **8 proteins had no directory at all** in `protein_tensors`:
   `1W4H, 2K1B, 2K28, 2KXD, 2L33, 6EWT, r11_1081_TrROS_Hall, r12_757_TrROS_Hall`.
2. **2K5H had a directory but no coordinates.** It contains *only*
   `prott5_embeddings/` — no `coords_tensor.pt`, no `mask_tensor.pt`.

This is why the brief's list was wrong about 2K5H: an `os.path.isdir` check says it is
present, and only the `os.path.exists(coords)` check that `w5_dg.py` actually performs
says it is not. **Any future resolver must key on the coords file, not the directory** —
`scripts/tensor_paths.py` does exactly this.

Note also that `protein_tensors` is *not* how the training pipeline loads structures.
`model/data_loader.py` reads `crd_backbone.pt` / `mask.pt` from `CFG.data_path + set_type`.
The string `protein_tensors` appears in exactly one file in the repo — `scripts/w5_dg.py`.
So this path was an analysis-side choice, never the pipeline's own resolution, and the
"trace how train.py resolves a name" step has no answer: train.py never touches this tree.

## 2. Where they live — the name to path map

All 9 are in `MutationFineTuning/test_protein_tensors`, under **identical names**. No case
folding, no chain suffixes, no PDB-vs-internal id mapping was needed. The recovery is a
root swap, not a renaming.

Canonical resolver: **`/home/nissimb/DeepPEF/scripts/tensor_paths.py`** (`resolve(name)`,
`coords_mask(name)`, `TEST_28`). Search order:

1. `/groups/keasar_group/casp15/meytav/MutationFineTuning/test_protein_tensors` -> 27/28
2. `/groups/keasar_group/casp15/meytav/protein_tensors` -> `r18_3_TrROS_Hall` only

`r18_3_TrROS_Hall` is the one protein the new root lacks, so **the union of both roots is
required**; neither alone reaches 28. `coords_mask()` raises `FileNotFoundError` rather
than returning None, so a future analysis fails loudly instead of silently shrinking n —
the failure mode that hid this for as long as it did.

### Provenance check (this is what makes the union legitimate)

18 proteins carry `coords_tensor.pt` in **both** roots. All **18/18 are bit-identical**
(`torch.equal` on float tensors, shapes equal). The two roots are the same preprocessing,
so mixing them introduces no batch effect. The 9 recovered tensors are all well-formed:
shape `(L,4,3)`, full masks, lengths 44-71 (consistent with the known 43-72 aa range),
coordinate ranges +/-13-23 A (Angstrom, matching the existing scale convention).

## 3. W5-on-dG at full n=28 — an honest negative

`scripts/w5_dg28.py` -> `results/w5_on_dg_n28.json`. All 28 resolved, none missing.

| feature | target | Pearson | Spearman | p |
|---|---|---|---|---|
| mean_burial | b_p | **-0.265** | -0.194 | 0.174 |
| mean_burial | abs_b_p | +0.233 | 0.142 | 0.233 |
| frac_buried | b_p | -0.060 | -0.003 | 0.760 |
| frac_buried | abs_b_p | -0.090 | -0.171 | 0.647 |
| length | b_p | -0.303 | -0.252 | 0.117 |
| length | abs_b_p | +0.264 | 0.153 | 0.175 |

At n=28 the p=0.05 threshold is **|r| = 0.392**.

**`mean_burial ~ b_p` does NOT cross the threshold. It got *weaker*, not stronger:
-0.343 at n=19 -> -0.265 at n=28, p=0.17.**

Sanity check: re-running on the n=19 subset reproduces **-0.343 exactly**, confirming the
resolver reproduces the historical computation and the only change is the added sample.

This kills the most attractive escape hatch. The n=19 result was not a real effect
starved of power — the recovered third pulled it *toward* zero. The honest reading is
that **restoring the sample did not rescue W5 as a b_p lever; it removed the excuse.**
W5 burial does not predict the WT offset on this benchmark, at full n, on the metric it
acts on. Combined with the fact that W5's only scheduled scoring is a pooled-ddG
factorial (the wrong metric per the metric rule), there is now no measurement anywhere
that supports W5 as a calibration lever.

Note this is *not* in tension with the known `frac_buried ~ b_p` mean r = -0.475 across
10 checkpoints: that is a different burial statistic averaged over checkpoints, whereas
`frac_buried` on this single checkpoint is -0.060. The 10-checkpoint replication remains
the stronger evidence; what n=28 shows is that `mean_burial` on this checkpoint is not it.

## 4. The missing set was BIASED — a caveat that belongs in the thesis

This is the most consequential finding of Line 2. The missing 9 were **not** a random
third. Mann-Whitney, present (n=19) vs missing (n=9):

| quantity | present | missing | MW p |
|---|---|---|---|
| **b_p** | -0.019 | **+1.143** | **0.052** |
| abs_b_p | 1.102 | 1.587 | 0.081 |
| **a_p** | 0.451 | **0.668** | **0.029** |
| **per-protein ddG PCC** | 0.692 | **0.813** | **0.037** |
| length | 55.3 | 56.3 | 0.806 |
| mean_burial | 0.449 | 0.440 | 0.658 |
| designed | 5/19 (26%) | 2/9 (22%) | — |

The missing set was **not** biased by length, burial, or designed-vs-natural — the
obvious structural suspects are all null. It was biased by **model behaviour**:

- **b_p: the missing proteins are almost all positively offset** (+1.14 vs -0.02, p=0.052).
  The n=19 sample was very nearly b_p-centred *by construction of the missing set*, which
  artificially flattered any statistic computed on the offset distribution.
- **a_p is significantly higher in the missing set** (0.668 vs 0.451, p=0.029). The
  missing proteins were the *less compressed* ones. Since `corr(a_p, per-protein PCC) =
  +0.571`, they are also the ones the model handles best — confirmed directly: their
  per-protein PCC is 0.813 vs 0.692 (p=0.037).

**Consequence:** every structure-based analysis run at n=19 was run on a sample
systematically depleted of the model's *good*, *high-slope*, *positively-offset*
proteins. Any a_p-related or b_p-related correlation estimated at n=19 should be
regarded as estimated on a biased subsample and re-run through `tensor_paths.py`.

### Impact on the offset-removal oracle

The top-6 |b_p| carriers, with recovery status:

| rank | protein | abs b_p | status |
|---|---|---|---|
| 1 | 2KVS | 4.704 | present |
| 2 | r12_757_TrROS_Hall | 2.747 | **was missing** |
| 3 | 2K5H | 2.522 | **was missing** |
| 4 | r18_3_TrROS_Hall | 2.500 | present |
| 5 | 1TUC | 2.022 | present |
| 6 | 2KXD | 1.935 | **was missing** |

The brief's premise — "2K5H is one of the TWO proteins carrying 78% of the oracle gain"
— needs correcting on this checkpoint: the top-2 are **2KVS and r12_757_TrROS_Hall**, and
2K5H is #3. But the direction of the concern was right and is now sharper: **three of the
top six offset carriers were invisible to every structural analysis**, including the #2.
The b_p-corrector post-mortem ("neither of the top-2 is a structural outlier") was
therefore conducted without structural access to one of the two proteins it was about.
That conclusion should be re-derived at n=28 before it is written up as settled.

## 5. What this cost, and what it now enables

- **Cost:** every structural analysis to date ran at n=19 (threshold 0.490 instead of
  0.392) on a sample biased toward low-a_p, near-zero-b_p, worse-fit proteins. The
  headline casualty is the W5-on-dG test, which could not clear its threshold — and now
  we know it would not have cleared at n=28 either, so no true positive was lost there.
- **Enabled:** the threshold drops 0.490 -> 0.392, a 20% reduction in the effect size
  needed for significance, and the sample is now unbiased in a_p and b_p. Re-running
  `mean_rel_SASA ~ a_p` (r=+0.714 at n=19) and `frac_buried ~ b_p` (mean r=-0.475) at
  n=28 is now the highest-value next step, since both were estimated on the biased
  subsample and both are load-bearing for the "one structural axis, both calibration
  channels" claim.

## 6. Caveats

- All numbers are from a single checkpoint, `abl_calib_ctrl_repro2_e14.csv`. The
  10-checkpoint replications should be redone at n=28 before any claim is finalised.
- The bias tests are n=19 vs n=9. Mann-Whitney at those sizes is underpowered; p=0.029
  and p=0.037 are nominal and uncorrected for the six quantities tested (Bonferroni
  alpha=0.0083 would retain none). Treat the *pattern* — three model-behaviour quantities
  differ, three structural quantities do not — as the finding, not any single p-value.
- Why `MutationFineTuning/test_protein_tensors` and `protein_tensors` diverge is not
  established. The former is a 75-protein superset that looks like the real test tensor
  store; the latter looks like a partial copy. `transfer_files.py` /
  `transfer_training_proteins.sh` in that directory are the place to look if provenance
  needs documenting further.
- This is a claim about this 28-protein benchmark. Whether burial predicts b_p in general
  is not answerable here.
