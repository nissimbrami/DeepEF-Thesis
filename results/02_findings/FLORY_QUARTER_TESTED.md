# Flory was rejected on a quarter of the idea — and the code says so in its own docstring

**Date:** 2026-09-10. **This narrows my own rejection of Flory.** The arms trained healthily and
scored terribly, so the *configuration* is dead. **The idea was never implemented.**

## What IFUM do, and what we did

IFUM do two things:
1. give the unfolded state a physical form, **and**
2. **train the network to predict the unfolded ensemble**, via an auxiliary loss (weight 100).

Their ablation, 0.78 -> 0.70, **removes both together.** No number in the paper isolates the map
swap. We implemented only the map swap and rejected their idea on that basis.

## Three independent proofs that only the map was swapped

**1. The docstring says it.** `train_utils.py:684`:

> *"...apply the SAME Gaussian kernel and masking as the baseline path, then reduce identically.
> The output tensor shape is byte-identical to the baseline unfolded graph
> (**value-only lever, no new parameters**)."*

**2. The map cannot read the sequence.** The whole computation:

```python
idx  = torch.arange(N)
sep  = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()   # |i-j| only
b    = _coil_bond_length(ca, N, ...)                 # one scalar per protein
d_coil = b * torch.pow(sep + 1e-6, nu)
```

**`one_hot` is never read.** The coil map is a function of `|i-j|` and one scalar. It is therefore
**identical for every variant of a protein** and cancels exactly in ddG — the metric rule, again.

**3. Zero learnable parameters** in the unfolded path. Nothing is trained to predict anything.

## And the learned half was explicitly ruled out in advance

`train.py:99`, the distogram help text:

> *"weight of a distogram auxiliary loss on the **FOLDED state only** ... **NOT applied to the
> unfolded state**: the coil map is an analytic function of |i-j| alone, so predicting it teaches
> nothing."*

That reasoning is **correct about our own coil and wrong as a reading of IFUM.** Predicting *our*
analytic map teaches nothing — but IFUM do not predict an analytic map, they predict a **sampled
ensemble**. The argument silently substitutes our implementation for their method, and by doing so
closed the only half that had never been tried.

## Revised verdict

| claim | status |
|---|---|
| "the analytic coil map as an input swap hurts badly" | **ESTABLISHED** — two configs, healthy training, mean r 0.1922 / 0.1631 vs 0.7273 |
| "the Flory/IFUM reference-state idea does not work" | **NOT ESTABLISHED** — never implemented |

**Rejection restated properly:** *rejected in the configuration "analytic |i-j| coil map
substituted into the unfolded input, no auxiliary loss, no learned reference".*

## What an honest test requires

The unfolded state must be **learned or sampled**, not substituted:

1. **A sampled coil ensemble** — k conformations per protein, cached; the unfolded energy is an
   average over samples rather than one analytic map. `scripts_new/unfolded_ensemble.py` exists
   (199 lines) and `ens3mean`/`ens8mean`/`ens8lse` are running now — **this is the closest thing
   to IFUM's half currently in flight.**
2. **A distogram head on the unfolded state**, predicting a *sampled* distance distribution.
   Currently blocked by the help-text argument above, which should be revised.
3. **Weight:** IFUM's 100 is calibrated to their loss scale. Cross-entropy here is ~ln(32)=3.5
   against an MSE of order 1, so 100 would swamp the primary objective. **Start at 0.1.** Copying
   100 and reporting the failure as the idea's would repeat exactly the error above.

**Confidence: 97%** — the docstring, the source of the map, and the parameter count are all read
directly; the IFUM ablation confounding is stated in their own paper.
