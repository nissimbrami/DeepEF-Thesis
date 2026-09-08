
---

# CHECKPOINT 17 — 2026-09-08 — A DATA BUG IN 2K5H INFLATES THE HEADLINE RESULT

## The finding, verified independently by the main session

A frontier agent reported that 2K5H's eval reference row is a MUTANT background rather than the
true wild type. **I verified it myself rather than accepting the report**, and it holds.

`eval_results/abl_calib_ctrl_repro2_e14.csv`, protein 2K5H, 1125 rows:

    row 0:  deltaG = 1.7231   pred = 4.2446   ddG = 0.0
    protein deltaG: min -0.980, mean 3.768, max 4.999
    row 0 sits at the 14.6th PERCENTILE of its own protein

**A wild type should be near the TOP of its stability distribution, not the bottom.** Row 0 is also
not unique: 2K5H has TWO rows with `ddG == 0` (indices 0 and 191), while the WS-1 convention assumes
exactly one.

## The control that makes it unambiguous

Row-0 percentile within each protein's own dG distribution, all 28:

| | percentile |
|---|---|
| 26 of 28 proteins | **62% - 99%** (exactly where a WT belongs) |
| **2K5H** | **14.6%** |
| r18_3_TrROS_Hall | 33.3% |

Only two fall below their own median, and 2K5H is far the worse. This is not a distributional
quirk; it is a wrong reference row, and every one of 2K5H's 1125 ddG labels is shifted by the
difference between the true WT and the mutant background (~3.0 kcal/mol).

## What it costs us — measured over 22 eval CSVs

| condition | pooled | oracle | gain |
|---|---|---|---|
| **all 28** | 0.5018 | 0.7391 | **+0.2373** |
| **drop 2K5H** | 0.4882 | **0.6410** | **+0.1528** |
| drop 2K5H + 2KVS | 0.5224 | 0.6542 | +0.1319 |
| drop 2K5H + r18_3 | 0.4860 | 0.6284 | +0.1424 |

**Dropping 2K5H alone removes 36% of the entire offset-removal gain** and takes the oracle from
0.739 to 0.641.

**A single protein with a corrupted reference row carries over a third of the thesis's headline
effect.** The "offset removal lifts pooled PCC by ~0.11-0.12" claim is substantially an artefact of
one mislabelled wild type, because a 3 kcal/mol label shift IS a per-protein offset by construction
— the oracle then "discovers" an offset we ourselves introduced.

## What must happen

1. **Find the true WT row for 2K5H** and rebuild its ddG column against it (the true WT appears to
   be the `2K5H.pdb` entry at dG 4.806, versus the `2K5H.pdb_G11S` mutant background at 1.723).
   Then re-run every calibration number.
2. **Audit the reference row for ALL 28** with the percentile test above as a permanent gate: row 0
   below its protein's median is a red flag, and more than one `ddG == 0` row is a hard error.
3. **Re-state every headline number** in CONTEXT.md once 2K5H is fixed. The 0.70-0.72 ceiling, the
   +0.11 gain, std(b_p)=1.5741 and the "top-2 proteins carry 78%" claim all inherit this bug.
4. r18_3_TrROS_Hall at 33.3% needs the same check.

## Two other findings from the same sweep, both verified numbers

**The exposure result SURVIVES the length confound.** `a_p ~ mean_rel_SASA` controlling for length
is **+0.6623 mean, significant in 10/10 checkpoints**, while length alone is r=-0.2149 and
significant in **0/10**. CHECKPOINT 8 stands; no correction needed.

**The slope objective has an algebraic CEILING.** Since `a_p = r * sd(pred)/sd(true)`, driving
sd(pred) -> sd(true) pins the slope to the CORRELATION, never to 1.0. The ceiling is a_p = r =
0.793. Still worth having: the measured a_p decomposes into 62.8% spread compression (fixable) and
36.8% ranking error (not), so `--slope_weight` can recover about 65% of the gap to a_p = 1 — but it
cannot close it, and the write-up must say so rather than implying otherwise.
