# ROOT CAUSE FOUND: the descriptor block cancels in dG by construction

**Date:** 2026-09-10. Two tasks, both under 15 minutes, run to completion before anything else.
They close the descriptor question and clear Flory of the same charge.

---

## TASK 1 — training curves. The falsifier was registered in advance:
*"a flat or diverging curve is an optimisation failure and the rejection is void."*

| arm | val RMSE trajectory | train loss | verdict |
|---|---|---|---|
| Flory nu=0.588 | 1.450 -> 1.086 (monotone) | 0.0252 -> 0.0116 | **HEALTHY** |
| Flory + coil_edges | 2.313 -> 1.628 (monotone) | 0.0836 -> 0.0608 | **HEALTHY** |
| LORO descriptors | **2.541 x15, identical** | **0.12675... x15** | **NEVER TRAINED** |
| descunit (unit table) | **2.529 x15, identical** | **0.12744... x15** | **NEVER TRAINED** |

**Flory is cleared of the optimisation charge — both arms trained normally.** Its rejection
stands on its merits: mean r 0.1922 and 0.1631 against a control of 0.7273.

**The descriptor arms never trained at all.** The loss is constant to the **fifth decimal** across
all 15 epochs. That is not a lever losing; it is a gradient of zero.

---

## TASK 2 — the remaining suspect, tested directly

The hypothesis: **the descriptor block is identical in the folded and unfolded passes, so it
cancels exactly in `dG = E_folded - E_unfolded`.**

**It is confirmed in the code and in the numbers.**

`train_utils.py:361` — the function takes **no `folded` argument at all**:

```python
def _desc_or_none(one_hot):
    """W6: the [N,K] descriptor block, or None when --aa_descriptors none (default)."""
    return desc_or_none(one_hot, CFG)
```

and `train_utils.py:734` already says it out loud: `_Dsc = _desc_or_none(one_hot)  # W6: state-independent`.

Measured, same one_hot, folded vs unfolded call:

```
W6 descriptors : torch.equal(folded, unfolded) = True    max|diff| = 0.000e+00
W12 side-chain : torch.equal(folded, unfolded) = False   max|diff| = 1.501e+00
```

**The 16 descriptor columns are bit-identical between the two states, so they contribute EXACTLY
ZERO to dG.** The network is handed 16 dimensions that cannot move its output, and whose only
effect on the gradient is subtractive noise.

**This explains both symptoms at once** — why it does not learn (no usable gradient path) and why
RMSE 2.53 is *worse* than predicting a constant (1.30).

**And it explains the contrast with W12**, which is geometric, differs by 1.501 between states,
and is the only lever that has ever cleared significance.

---

## The rule this generalises to — it predicts failures before they cost GPU

> **Any per-residue feature that is identical in the folded and unfolded graphs contributes
> exactly zero to dG by construction. It cannot help, and it can actively destabilise training.**

**This rule retro-predicts three things we paid for:**

1. **W6 descriptors** — a pure function of residue identity, identical in both states. Collapsed.
2. **The original W5 burial failure** — same shape, and it was fixed precisely by zeroing burial
   in the unfolded pass (`train_utils.py:733`: *"W5: burial is ZERO unfolded"*).
3. **The Flory coil's ddG measurement** — `d = b*|i-j|^nu` reads only sequence separation, so it
   is identical between wild type and mutant and cancels in ddG. Already recorded as metric-rule
   catch #1.

**The same defect, three times, across three levers.** The rule below would have caught all three
on paper, before any code was written.

## What this changes

1. **The descriptor hypothesis is NOT rejected — it was never tested.** No run has ever presented
   the network with a state-dependent chemistry signal.
2. **To test it honestly, the block must differ between states.** The obvious route is the one W5
   already uses: weight the descriptor columns by burial, so they are non-zero folded and zero
   unfolded. That makes the chemistry a *desolvation* term rather than an identity lookup — which
   is also the mechanism §3.4 says the model is missing.
3. **Do not run another `--aa_descriptors` arm as currently wired.** It is guaranteed to collapse.

## Pre-flight test this adds

Before any new feature block is submitted:

```python
assert not torch.equal(block_folded, block_unfolded), \
    "block is state-independent: it cancels in dG by construction"
```

**Five lines, zero GPU, and it would have prevented every collapse above.**

**Confidence: 98%** — `torch.equal` is exact, the code comment states the property independently,
and the loss being constant to five decimals corroborates it.
