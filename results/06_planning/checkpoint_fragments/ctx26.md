
---

# CHECKPOINT 26 — 2026-09-08 — K9: THE LORO PREDICTION, RECORDED BEFORE THE READOUT

Two runs are on the golden lane RIGHT NOW, differing in exactly one flag:

    gld_loroW_onehot   --holdout_residues W --aa_descriptors none          (epoch 10/14)
    gld_loroW_desc     --holdout_residues W --aa_descriptors mordred_pca16 (epoch 0/14)

Tryptophan is removed from TRAINING only; both are then scored on mutations TO tryptophan, which
neither model saw during training. One-hot cannot represent an unseen residue at all. A continuous
descriptor space can, because W sits at a definite point in physicochemical space.

**This is the only runnable test of Ofir Ezrielev's central claim** — that a model trained on
canonical residues predicts non-canonical effects because the physicochemical space is continuous.
MegaScale contains zero non-canonical residues, so holding out a canonical one is the proxy.

## THE PREDICTION, made before any result exists

**The descriptor arm will NOT beat the one-hot arm by much.**

Grounds, measured in CHECKPOINT 16: a residue-disjoint linear probe (fit on 19 residue types,
predict the held-out 20th) recovers from ProtT5 alone:

    hydropathy  R^2 = +0.704
    charge      R^2 = +0.511
    volume      R^2 = +0.421
    identity    accuracy 1.000 (chance 0.050)

**ProtT5 already predicts the chemistry of a residue type it has never seen.** The 1024-dim
embedding is in the feature vector for both arms, so the descriptor block is adding a property the
model can already recover. Ofir concedes exactly this in his thesis: physicochemical properties are
implicit in language-model embeddings, and his case for descriptors is COVERAGE of residues absent
from the training databases — a gap that does not exist for canonical W.

## What each outcome would mean

| result | interpretation |
|---|---|
| desc ~ one-hot (within noise) | **PREDICTED.** ProtT5 already carries W's chemistry; W6 adds regularisation at best. Report W6 as infrastructure and future work, not as a results-chapter lever. |
| desc CLEARLY beats one-hot | The prediction is WRONG and the finding is real: descriptors add something ProtT5 does not, despite the probe. This would be the strongest possible result for W6 and must be reported as such. |
| desc WORSE than one-hot | The block is actively harmful, presumably by widening the vector for no gain. Retire W6. |
| BOTH collapse on W mutations | The holdout worked but neither representation generalises. That is a statement about the ARCHITECTURE, not about descriptors, and the comparison is uninformative. |

**Negative control already queued in the design:** proline. Descriptors should NOT rescue P, because
proline's effect is backbone geometry rather than side-chain chemistry. A method that "helps"
everywhere is not being tested properly.

## Why recording this now matters

The prediction is falsifiable, it is derived from an independent measurement, and it is written
down **before** the numbers land. If the descriptor arm wins, that is a genuine surprise that
overturns CHECKPOINT 16's conclusion — and it will be reported as such rather than rationalised.
