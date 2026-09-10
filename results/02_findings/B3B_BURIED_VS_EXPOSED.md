# B3b — W5 does help at buried positions, but the median says it helps a TAIL, not the typical mutation

**Date:** 2026-09-10. Checklist item **B3b**: *"decompose per protein — does W5 help exactly at
BURIED positions? Prediction: gap 0.095 buried vs 0.024 exposed."*

## What we had learned before this item

W5 was rejected at four seeds: per-protein mean +0.84 control-sd but **median −4.97 control-sd**,
and the sign flip replicated in every seed (`W5_REJECTED_4SEEDS.md`).

**What was not good:** the rejection summarised a lever with one number, and the review's point was
that mean-vs-median disagreement means two opposite effects were averaged together. If W5 helps
exactly where the model's known deficit is — burying hydrophobics (`FINDINGS` §3.4) — then the
mechanism holds and the rejection is an artifact of summarising.

## PLAN

**Q:** Is W5's gain concentrated at buried positions?
**Falsifier:** if the gain is NOT larger at buried positions, the desolvation mechanism fails and
the rejection stands as a plain null.
**Metric:** per-mutation `|error_control| − |error_W5|` (positive = W5 better), split by CB
neighbour-count burial, pooled over 4 W5 seeds.

## EXECUTE

19 of 27 proteins joined (mutation-name position recovered by regex; joined to the eval CSVs on
`deltaG` rounded to 5 dp — **the eval CSVs store float32, the mutation files float64, so an exact
join returns 4 rows out of 1703**). 76,416 mutation×seed pairs.

| stratum | n | **mean gain** | **median gain** |
|---|---|---|---|
| **BURIED** (burial ≥ 0.50) | 59,212 | **+0.03697** | **−0.00406** |
| **EXPOSED** (burial < 0.25) | 448 | +0.01051 | +0.00122 |

    buried - exposed = +0.02647    Welch t = 4.933,  p = 1.08e-06
    corr(burial, gain) = +0.0648   (n = 76,416)

| destination | n | mean gain |
|---|---|---|
| to hydrophobic | 27,084 | +0.02366 |
| to polar | 29,736 | +0.04700 |
| buried + hydrophobic | 20,980 | +0.02414 |

## VERIFY — the prediction is HALF confirmed, and the other half is the finding

**Confirmed on the mean:** W5's gain is significantly larger at buried positions
(+0.0265, p = 1.1e-06). The direction the review predicted is real.

**But three things cut against the mechanism story:**

1. **The buried MEDIAN is negative (−0.004) while its mean is positive (+0.037).** The same
   mean/median split that characterises W5 overall reappears *inside* the buried stratum.
   **W5 rescues a tail of badly-predicted buried mutations and slightly hurts the typical one.**
2. **corr(burial, gain) = +0.065** — significant only because n = 76,416. It explains
   **0.4% of the variance**. Burial is not what selects where W5 helps.
3. **It helps POLAR destinations more than hydrophobic ones** (+0.047 vs +0.024) — the opposite
   of a desolvation mechanism, which should act on burying greasy residues.

**The exposed stratum is also tiny (448 of 76,416).** These are 40–72-residue domains that are
mostly buried, so "buried vs exposed" is a weak contrast on this dataset.

## Verdict

**The mechanism story does NOT hold as stated.** W5's gain is larger at buried positions, but it
is (a) a tail effect even there, (b) explains under half a percent of the variance, and (c) is
*stronger for polar destinations*, which contradicts desolvation.

**The rejection stands, and its reason is now better characterised:** W5 is not a broken
desolvation term — it is a lever that rescues a small tail of hard mutations at a cost to the
majority, at every stratum. That is the same shape as `W5_REJECTED_4SEEDS.md`, now measured at
the mutation level rather than the protein level.

**B3a still binds:** all four seeds are `--loss_mode dg`, so the objective/block confound
(`W12_W5_CONFOUNDED.md`) is untouched by this analysis. `pub_w5_ddg_s42` remains the deciding arm.

## DONE

**Cost: ~30 min, no GPU.** Prediction tested and half-refuted; a real join bug found (float32 vs
float64) that would silently return 4 rows for anyone repeating this.

**Confidence: 92%** — n is large and the burial definition matches `compute_burial`, but 8 of 27
proteins failed to join and the exposed stratum is small.
