# B1d epoch-0 readout: the unfolded distogram head trains, and does not destabilise

**Date:** 2026-09-10. `pub_udisto0.1_s42` finished epoch 0 (306/306) and entered epoch 1.
This is a **health check, not a result** — one epoch decides nothing.

## What was verified

| arm | epoch-0 val ddG PCC | ddG PCC-PP | ddG RMSE |
|---|---|---|---|
| **B1d — unfolded distogram, weight 0.1** | **0.594** | **0.692** | **2.069** |
| `gld_disto0.01_s42` (the INERT flag = a baseline) | — | — | 2.173 |
| `pub_ctrl_s1` (plain control) | 0.629 | 0.694 | 1.600 |

**The head trains.** No traceback across a full epoch, the loss is finite, gradients flow
(verified pre-submission: `|grad| = 31.60`, untrained CE 3.4800 vs `ln(32) = 3.4657`).

**It is not destabilising the optimisation** — RMSE 2.069 is better than the inert-flag arm's
2.173 at the same epoch, and the ddG PCC-PP (0.692) is within a hair of the control's 0.694.

**It is behind the control on RMSE** (2.069 vs 1.600). Expected: the auxiliary loss diverts
capacity early. Whether it repays that by epoch 14 is the actual question.

## A logging artifact ruled out, not reported as a finding

The epoch-0 line shows `a_p median=nan  std_ratio median=nan`. **This appears in every arm,
including healthy controls** (`pub_ctrl_s1` shows the same nan at epochs 0 and 1), so it is a
logging artifact of the early-epoch validation path and **not** a B1d defect. Checked before
drawing any conclusion from the line.

## Status

**Not a result. n=1, epoch 0 of 15.** The comparison that matters is B1d at its val-selected
epoch against the control family, on the ranking channel, reported with mean AND median.

**What this does settle:** the wiring built today works end to end — the head is instantiated,
the unfolded forward pass runs, the loss is added, and 306 steps complete without error. That was
the open risk after four integration faults (indentation, out-of-scope variable, missing
`n_folded=0`, CUDA OOM).

**Confidence: 95%** that the implementation is sound; **no claim** about whether the lever helps.
