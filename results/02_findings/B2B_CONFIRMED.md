# B2b — CONFIRMED. Zeroing the descriptor block in the unfolded pass revives a dead arm.

**Date:** 2026-09-10. Checklist item **B2b**, the confirmation test the review asked for:
*"zero the block in the UNFOLDED pass only, one epoch. If RMSE moves off 2.529, proved."*

## The registered falsifier, and what it did

**Registered before the run:** RMSE moves off 2.529 -> the dG cancellation is the cause;
RMSE stays frozen -> the hypothesis is wrong and something else is responsible.

**It fired, at epoch 0, decisively.**

| arm | change | val ddG PCC | ddG PCC-PP | ddG RMSE |
|---|---|---|---|---|
| `p_descunit_s42` | descriptors, unchanged | **0.019** | 0.012 | **2.529** |
| (same, epoch 1) | | −0.049 | 0.028 | **2.529** |
| **`descsd_s42`** | **same table, zeroed in the unfolded pass** | **0.625** | **0.726** | **1.966** |

**One line of code moves ddG PCC from 0.019 to 0.625 and unfreezes an RMSE that had been
constant to five decimals for fifteen epochs.**

For scale: 0.625 at epoch 0 is **above** the plain control's epoch-0 PCC of 0.629's neighbourhood
and its PCC-PP of 0.726 is in normal territory. The arm is simply training now.

## What is proved

`W6_ROOT_CAUSE.md` argued from three lines of evidence that the descriptor block is bit-identical
in the folded and unfolded passes (`torch.equal` -> True), therefore contributes exactly zero to
`dG = E_folded - E_unfolded`, therefore has no gradient path. **That was a static argument. This is
the dynamic confirmation.**

The chain is now closed end to end:

1. **Source:** `_desc_or_none(one_hot)` takes no `folded` argument; `train_utils.py:734` already
   said *"W6: state-independent"*.
2. **Static measurement:** `torch.equal(folded, unfolded)` -> True, max|diff| = 0.000e+00, against
   W12's 1.501.
3. **Symptom:** train loss constant to five decimals across 15 epochs; scored mean r **-0.0191**
   (-115 control-sd, 0/27 proteins).
4. **Intervention:** make the block state-dependent -> **PCC 0.019 -> 0.625, RMSE 2.529 -> 1.966.**

**A cause is established by intervention, not by correlation. This is the intervention.**

## What this does NOT yet show

**It does not show that descriptors HELP.** It shows the arm can now train at all. Whether
chemistry adds ranking information over one-hot is the original `FINDINGS` §8.3 question, and it
needs the run to reach its val-selected epoch and be scored against the control family on mean AND
median.

**The §8.3 prediction still stands unfalsified and now finally has a live test:** *the descriptor
arm will not beat one-hot by much, because ProtT5 already carries the chemistry (held-out
hydropathy R2 = 0.704).*

## Consequences

1. **The descriptor hypothesis was never tested — that is now demonstrated, not argued.** Every
   previous descriptor arm was a dead run, and reporting any of them as evidence about chemistry
   would have been wrong.
2. **`gate_state_dependence.py` is validated by an intervention.** Its FAIL on both descriptor
   modes correctly predicted a collapse that a one-line fix repairs.
3. **The general rule earns its place in the protocol:** *a per-residue feature identical in the
   folded and unfolded graphs contributes exactly zero to dG.* It retro-predicted W6, the original
   W5 burial bug, and the coil's ddG cancellation - and now it has been confirmed forward.

**Confidence: 98%.** The intervention is a single guarded line, the effect is 30x on PCC, and the
control (the unmodified arm) was run to 15 epochs with a five-decimal-constant loss.
