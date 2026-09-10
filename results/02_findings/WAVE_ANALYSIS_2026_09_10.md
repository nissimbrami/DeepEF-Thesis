# Full analysis of everything that finished: 12 arms, and only the failures are significant

**Date:** 2026-09-10. Every arm scored since the last wave, on the **ranking channel**
(per-protein r, which no affine calibration can move), paired against the 6-seed control family.

Control (6 seeds): pooled **0.5757**, mean r **0.7273**, median r **0.7897**.

| arm | pooled | mean r | median r | improved | worst | Wilcoxon |
|---|---|---|---|---|---|---|
| slope 0.7 | 0.5879 | 0.7387 | 0.7931 | 17/27 | -0.033 | 0.0262 |
| w7span4 s1 | 0.5938 | 0.7319 | 0.7799 | 17/27 | -0.042 | 0.3242 |
| w7edge s1 | 0.5564 | 0.7319 | 0.7871 | 18/27 | -0.068 | 0.1286 |
| w7edge s42 | 0.5808 | 0.7307 | 0.7999 | 15/27 | -0.018 | 0.6617 |
| slope 0.5 s42 | 0.5898 | 0.7303 | 0.7846 | 12/27 | -0.032 | 0.8408 |
| W12+W5 (dg) | 0.5969 | 0.7262 | 0.7628 | 8/27 | -0.077 | 0.3012 |
| pooled_corr 0.3 | 0.5455 | 0.7251 | 0.7853 | 17/27 | -0.086 | 0.6109 |
| u10bidir s1 | 0.5610 | 0.7231 | 0.7714 | 8/27 | -0.044 | 0.0521 |
| slope 0.5 s1 | 0.5800 | 0.7216 | 0.7777 | 9/27 | -0.050 | 0.0619 |
| **Flory nu=0.588** | **0.0742** | **0.1922** | 0.3813 | 2/27 | **-1.562** | **<0.0001** |
| **Flory + coil_edges** | **0.2190** | **0.1631** | 0.3393 | 1/27 | **-1.481** | **<0.0001** |
| **LORO descriptors** | **0.0056** | **-0.0137** | -0.0249 | **0/27** | **-1.025** | **<0.0001** |

## Multiplicity: only the failures survive

Twelve arms tested. Benjamini-Hochberg at FDR 0.05:

| rank | arm | p | BH critical | |
|---|---|---|---|---|
| 1 | Flory nu=0.588 | <0.0001 | 0.0042 | **SURVIVES** |
| 2 | Flory + coil_edges | <0.0001 | 0.0083 | **SURVIVES** |
| 3 | LORO descriptors | <0.0001 | 0.0125 | **SURVIVES** |
| 4 | slope 0.7 | 0.0262 | 0.0167 | - |
| 5+ | everything else | >=0.05 | - | - |

**All three survivors are catastrophic negatives. Not one positive result in this wave survives
correction for the number of arms tested.**

`slope 0.7` (p=0.0262, mean r 0.7387 - the highest in the table) is the near-miss. It fails BH at
rank 4 and **it is a single seed**. A candidate for replication, not a finding.

## What this wave establishes

1. **Flory is dead at two independent configurations.** nu=0.588 gives mean r **0.1922**,
   +coil_edges **0.1631**, against a control of 0.7273; worst per-protein regression -1.562.
   Far stronger than the earlier nu sweep, and on the ranking channel where the metric rule
   offers no excuse. **The coil lever should be closed.**
2. **The descriptor arm is a collapse, not a result** (mean r -0.0137, 0/27 proteins). Cause
   still unknown - the scale hypothesis was eliminated today (`LORO_STILL_UNTESTED.md`).
3. **The information levers are all null.** w7edge, w7span4, u10bidir, the pooled-corr and the
   slope sweeps all sit inside noise on both mean and median. This confirms FINDINGS section 16
   on a sharper metric than the one that produced it.
4. **W12 remains the only lever that has ever cleared significance** (Wilcoxon p=0.0104,
   `W12_ROBUSTNESS.md`). Nothing in this wave of twelve came close.

## A bug I introduced and then found

`w12_s3`, `w12_s4`, `readatt_dg_s42`, `readgate_dg_s42` failed with `ARTIFACT MISSING` while the
log printed `DONE ... -> abl_w12_s3_e14.csv`.

**Cause: I patched only one of two checkpoint-loader call sites.** `evaluate.py:515` is inside
`run_training()` (dead code); the live path is **line 589** inside `run_validation_metrics()`,
which is what `__main__` calls. I fixed the dead one.

**This is exactly the three-part bug this project has already been bitten by, and I repeated it.**
Both sites are now patched (grep for the bare loader returns **0**), `gate_g4_cpu: ALL PASS`, and
the four arms are re-queued.

**Rule restated:** after editing `evaluate.py`, grep *every* call site and confirm which one
`__main__` reaches. A patch that looks applied but is not in the executing path is this project's
signature failure.

**Confidence: 96%** - values recomputed from the CSVs, BH correction is arithmetic, loader bug
read from the traceback line number.
