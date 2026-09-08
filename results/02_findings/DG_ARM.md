# THE dG ARM — `gld_dg_coil_s42`

### The last untested explanation for `b_p`. Verdict: **REJECTED.**
### Nissim Brami, M.Sc. thesis, BGU. Scored 2026-09-08 on the 28-protein / 28,314-mutation test set.

    Job 21080861   --full_data --no_pretrain --no_freeze --loss_mode dg --flory_unfolded
                   --coil_b fixed --pooled_corr_weight 0 --dg_length_norm none --affine_calib
                   --val_frac 0.1 --epochs 15 --seed 42
    Flags read from `scontrol write batch_script`, not from a submission script (none is on disk).
    Scored RAW: no `affine.json` was written, `DEEPEF_AFFINE` deliberately unset.

---

## 1. The question

Seven hypotheses for `b_p` have been tested and rejected (FINDINGS §4.2). The eighth — the
**reference state itself** — was the only one left, and the Flory coil was its instrument. The
motivating result was a FROZEN-checkpoint rescoring: dG MAE **4.9650 -> 3.9521** with `--coil_b
fixed`, a 20% cut, "our strongest geometric lever on `b_p`".

**The question this run had to answer: does TRAINING on dG with the coil actually SHRINK
`std(b_p)`?**

## 2. Headline numbers (epoch 14)

| quantity | control `calib_ctrl_repro2` e14 | **dG arm e14** | change |
|---|---|---|---|
| **dG MAE** | 1.3029 | **2.4003** | **+1.10 WORSE** |
| **std(b_p)** | 1.6030 | **0.9656** | −0.637 (−39.8%) |
| **per-protein ddG PCC** | 0.7310 | **0.3151** | **−0.416** |
| a_p median | 0.4975 | **0.0852** | −0.412 |
| pooled ddG PCC | 0.5910 | **0.2278** | **−0.363** |
| r median | 0.7955 | 0.4951 | −0.300 |
| s median | 0.6342 | 0.2098 | −0.424 |

Epoch 13 agrees (MAE 2.3820, std(b_p) 0.9732, ppPCC 0.3383, a_p 0.1019), so this is not
epoch-selection noise.

**Read the first two rows together.** `std(b_p)` fell by 40% — the largest movement any lever has
produced on the offset — while *every other quantity in the table collapsed*. That combination is
the signature the project already has a name for.

## 3. Why the 40% is not a win: the degenerate baseline

FINDINGS §5.3 records `--dg_length_norm`, which "shrank" `std(b_p)` by 31% and was the model
predicting nothing. The tell: as the prediction is deleted, `b_p -> -true_dG`, so

    std(b_p)  ->  std(true WT dG) = 0.9208        corr(b_p, true) -> -1

| run | std(pred WT dG) | std(b_p) | corr(b_p, true WT) |
|---|---|---|---|
| control | 1.3778 | 1.6030 | −0.5146 |
| **dG arm e14** | **0.6350** | **0.9656** | **−0.7744** |
| *degenerate attractor* | *0* | *0.9208* | *−1* |

**std(b_p) = 0.9656 is within 4.9% of the attractor 0.9208, and the dG arm now predicts LESS
spread across proteins (0.6350) than the labels actually contain (0.9208).** The offset did not
become better calibrated; the prediction contracted toward a near-constant, and `b_p` inherited the
spread of the labels. This is the `--dg_length_norm` failure in a milder, partial form.

*(Note: `corr(pred WT, true WT)` is a poor degeneracy test here — the healthy control itself scores
only 0.0696. The model never predicted absolute WT dG well. The live signature is spread collapse.)*

## 4. The trajectory kills it outright

Scored at epochs 0, 4, 8, 12, 13, 14 (six full 28-protein evaluations, checkpoints on disk):

| epoch | dG MAE | std(b_p) | per-prot PCC | a_p | r | s | pooled | std(pred WT) | corr(b_p,true) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 2.2615 | **0.9410** | 0.3205 | 0.0253 | 0.4333 | 0.0689 | 0.1547 | 0.3582 | −0.9262 |
| 4 | 2.4088 | 0.9556 | 0.2245 | 0.0483 | 0.3138 | 0.1854 | 0.1257 | 0.6273 | −0.7771 |
| 8 | 2.3735 | 0.9168 | 0.2626 | 0.0774 | 0.4628 | 0.1836 | 0.1529 | 0.5983 | −0.7880 |
| 12 | 2.3451 | 0.9722 | 0.3136 | 0.0827 | 0.5385 | 0.1953 | 0.1953 | 0.6490 | −0.7662 |
| 13 | 2.3820 | 0.9732 | 0.3383 | 0.1019 | 0.5842 | 0.2213 | 0.2416 | 0.6502 | −0.7656 |
| 14 | 2.4003 | 0.9656 | 0.3151 | 0.0852 | 0.4951 | 0.2098 | 0.2278 | 0.6350 | −0.7744 |

**`std(b_p)` is FLAT at 0.92-0.97 from epoch 0 to epoch 14. Training never moved it.** At epoch 0 —
before the model has learned anything — `std(b_p)` is already 0.9410 and `corr(b_p, true) = −0.9262`,
i.e. essentially the pure degenerate limit. The 40% "improvement" over the control was present at
initialisation and is a property of the dG loss's output scale, **not an effect of the coil and not
something training achieved.**

## 5. The frozen result that motivated this arm was misread

Recomputing from `results/w0_dg.json` (28 proteins, `calib_ctrl_repro2` e14, no training):

| condition | MAE | mean(b_p) | **std(b_p)** | under-predicted |
|---|---|---|---|---|
| base | 4.9650 | −4.9650 | **1.0576** | 28/28 |
| **coil, b fixed** | **3.9521** | −3.9521 | **1.1471** | 28/28 |
| coil, fitted | 5.7446 | −5.7446 | 1.1476 | 28/28 |
| noemb | 3.1315 | −3.1315 | 1.0182 | 28/28 |

**Every protein is under-predicted in every condition, so `MAE = |mean(b_p)|` identically.** The
celebrated 4.9650 -> 3.9521 was a **pure mean-bias shift** — and `std(b_p)`, the quantity that
actually matters, went *UP* 1.0576 -> 1.1471 (+8.5%).

**The lever never shrank the dispersion, not even in the frozen test that motivated the arm.** A
uniform shift of all 28 proteins is exactly the correction that Pearson is invariant to (FINDINGS
§4.4) and that cannot generate per-protein dispersion. The arm was chasing a bias term.

## 6. Verdict

> **The dG arm is REJECTED. Training on dG with the Flory coil does NOT shrink `std(b_p)`.**
>
> The apparent 40% reduction vs the control is a **scale artefact of the dG loss present at epoch 0**,
> sitting 4.9% from the degenerate attractor, bought at the price of the model itself:
> per-protein PCC 0.731 -> 0.315, a_p 0.498 -> 0.085, pooled 0.591 -> 0.228, and dG MAE 1.30 -> 2.40
> — the arm is *worse on the very metric it was scored on to justify it*.

**`b_p` has now survived EIGHT explanations.** The reference state was the last cheap one. Combined
with the failure of the feature-based corrector (§4.4) and the finding that the top-2 |b_p| proteins
carry 78% of the oracle gain with no structural signature, the honest reading is that **`b_p` may
not be predictable from anything the model currently sees** — which is a publishable negative and
the correct thing for the thesis to say.

### Caveat recorded honestly
There is **no matched control** for this arm: `ref_dg_seed42/` is an empty directory, so no trained
`--loss_mode dg` run *without* the coil exists. The comparison above is against a `ddg`-loss control,
which differs in loss as well as in coil. **The epoch-0 evidence is what makes the verdict safe** —
it is internal to this run and needs no control: a quantity flat from initialisation was not
produced by training. A dG-loss/no-coil control would still be worth one card to separate "the coil
did nothing" from "the dG loss did it all", and is the single cheapest follow-up.

### Artifacts
`eval_results/abl_gld_dg_coil_s42_e{0,4,8,12,13,14}.csv` — six evaluations, 28,315 rows each,
verified on disk. Scripts: `scripts/arm_analyze.py`, `scripts/degen_check.py`, `scripts/score_one.sh`.
G4 gate after all work: `dG=-0.0030 width=1092` — unchanged.
