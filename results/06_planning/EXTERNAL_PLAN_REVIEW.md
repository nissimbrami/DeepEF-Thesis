# Review of the external plan "from 0.638 toward 0.75"

Checked against the code and the cluster, not against my memory. Confidence stated per item.

## What the plan gets RIGHT, and it is the most valuable thing in it

**The descriptor arithmetic.** "Any per-residue descriptor table is `one_hot @ T` — a rank-1
projection of a block `fc1` already receives."

This is correct and it retroactively explains three separate failures: LORO collapsing, W6
descriptors adding nothing, and — uncomfortably — **two of my own W12 columns**. `dG_transfer`
and `volume` are pure functions of residue type, so they carry no information `fc1` lacks. **Only
W12's two burial-weighted columns are genuinely new.** Confidence: **90%**.

The consequence is right too: **the fix must be geometric and state-dependent.**

## The plan's central proposal ALREADY EXISTS — I found it in our code

The plan allocates **weeks 1–2** to building "W14 direction burial: how many neighbours lie in the
CB hemisphere — a residue can be surrounded yet point outward."

`train_utils.py:270` is `compute_hse`, half-sphere exposure:

```python
up = cb - ca                                   # the CB direction
same_side = ((diff * up).sum(-1) > 0)          # neighbours in the CB hemisphere
return (near * same_side).sum(1) / cap
```

**That is the same feature, already implemented, already exposed as `--burial_mode hse`,
already needing only CA and CB.** And a `scontrol` scan plus an `eval_results` scan confirm:
**it has never been run, and was not queued.** Confidence: **95%**.

**Submitted now:** `21144973` (dG), `21144974` (dG seed 1), `21144975` (ddG), `21144976` (× W12).
All four verified by `scontrol` to carry `hse`. **Two weeks of proposed work became four
submissions.**

## Where the plan is WRONG

**"Calibration is finished. No further calibration lever is worth a GPU hour."**

It builds this on my W13b result — which I gave **80% confidence** and explicitly flagged as
resting on **four arms**. My wider check over **35 runs** contradicts the strong reading: the
oracle spread does **not** collapse (sd 0.0237 → 0.0261), and `corr(zero-shot, oracle) = +0.548`.
**The model still matters after calibration.**

More decisively: **`--pooled_corr_weight` has never been enabled in this project.** Every arm ever
run passed `--pooled_corr_weight 0`. It is a loss on the pooled cross-protein correlation — the
exact metric being reported. And `--readout {attention,gated}` has never been run either; it is
the only lever that touches the **sum** that manufactures `b_p`.

Both are running now (`21144770`–`74`). **Closing the calibration direction before those return
would be a mistake.** Confidence the plan is premature here: **75%**.

## Claims I could not confirm

| claim | finding |
|---|---|
| "`saprot_pm` exists, generator emits `[1,1280]`" | **NOT FOUND.** No file matching `*saprot*` anywhere in the tree. W16 is not "mostly already built" — it does not exist | 90% |
| FASPR / SCWRL4 available for W15 | **NOT INSTALLED.** Not on PATH, not in home. W15 needs an install first | 95% |
| coordinates are N, CA, C, CB | **CONFIRMED** — `train_utils.py:256`, and glycine has no CB | 95% |

## Where I agree on the numbers

The ceiling argument is sound and honestly stated: MegaScale is a proteolysis assay with
~0.3–0.5 kcal/mol resolution and censoring above ~4.5, so **a perfect model cannot exceed the
noise in its own labels**. `P(pooled > 0.75) ≈ 25%` is a fair estimate. Our measured oracle
(0.7534) sits right at that boundary, which is independent support.

I also agree with §7: **the thesis rests on the five findings, not on the coefficient.** The
strongest of them is that the offset is not predictable but *is* measurable — and the new budget
result says **k=5 recovers 81%**, not k=20.

## Revised sequencing

1. **Now (already queued):** HSE ×4, pooled_corr ×3, readout ×2, distogram ×4, ensemble ×3
2. **When they return:** read out, then decide whether calibration is genuinely closed
3. **Only then:** W15 side-chain reconstruction — the largest bet, but it needs FASPR installed
4. **Drop W16** unless SaProt is actually obtainable; it is not in this tree
