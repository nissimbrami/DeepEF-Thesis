# K9 — LORO prediction, ON RECORD BEFORE READOUT

**Written 2026-09-08 22:15 while `gld_loroWdesc2_s42` (21136228) is still TRAINING at 2h10m.**
No epoch of this arm has been scored. This file exists so the prediction cannot be adjusted
after the fact — a result that confirms a prediction written afterwards is worth nothing.

## The experiment

`gld_loroWdesc2_s42` re-runs the leave-one-residue-out descriptor test with **normalised**
descriptors (`mordred_pca16_only`, which REPLACES one-hot rather than appending to it).
The first attempt collapsed technically: descriptors were centred but never scaled, carrying
**~13.8× the one-hot block energy**, and the arm froze at RMSE 2.541 with val ddG PCC ≈ 0 across
every epoch. **That arm never tested the hypothesis** — it tested a scale bug.

The comparator is `loroW_onehot_s42`, which trained normally (val ddG PCC 0.548 → 0.712 over e0–e12).

## THE PREDICTION

**The descriptor arm will NOT beat one-hot by a meaningful margin.**

Concretely, before seeing any number:

1. **Pooled ddG PCC within ±0.060 of the one-hot arm** — i.e. inside the seed noise band, hence
   indistinguishable from no change.
2. **It will not exceed one-hot by more than +0.060.** If it does, this prediction is WRONG and
   the descriptor hypothesis is live.
3. **Most likely outcome: slightly worse than one-hot**, because 16 PCA components must
   reconstruct what 20 orthogonal indicator columns give exactly.

## THE REASONING (measured, not intuition)

**ProtT5 already encodes the chemistry the descriptors carry.** A residue-disjoint probe — fit on
19 residue types, predict the held-out 20th — recovers:

```
hydropathy  R^2 = 0.704
charge      R^2 = 0.511
volume      R^2 = 0.421
residue identity: 100% accuracy
```

The embedding is 1024 dims of every input vector. Adding 16 descriptor columns that are ~70%
predictable from those 1024 is **largely redundant information**, and redundant information does
not improve a model that already fits its training set.

**The one thing that could falsify this:** LORO holds out a residue TYPE, so at test time the
model sees a residue it never trained on. One-hot gives an all-zeros-but-one column it has no
learned weight for; descriptors place the unseen residue in a *continuous* space where it sits
near chemically similar seen residues. **That is a real mechanism and it is why the experiment is
worth running.** But the probe says the embedding already provides that interpolation, so the
descriptor block would be adding a second copy of it.

## FALSIFIER

If `gld_loroWdesc2_s42` beats `loroW_onehot_s42` on pooled ddG PCC by **more than +0.060** at the
val-selected epoch on the canonical basis (27 proteins, 2K5H dropped), **this prediction is wrong**,
the redundancy argument is wrong, and the descriptor programme deserves real GPU time.

## What must NOT happen at readout

- Do not compare at a hand-picked epoch. Use the val-selected epoch for both arms (P0a showed
  epoch-picking inflated a_p by +0.075 elsewhere).
- Do not compare across different scoring bases.
- Report the number even if it contradicts this file. **An honest negative beats a fabricated
  positive**, and so does an honest surprise.
