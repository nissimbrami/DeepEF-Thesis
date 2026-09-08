
---

# CHECKPOINT 15 — 2026-09-07 — WHAT THE MODEL ACTUALLY SEES (measured, not assumed)

The standing complaint was that we had no understanding of what the inputs ARE. Measured directly
on `/groups/keasar_group/casp15/meytav/protein_tensors/2KWH/`.

## The tensors are per-VARIANT, not per-protein

    prott5_embeddings/prott5_embedding_<k>.pt   [128, 52, 1024]   <- 128 variants x 52 residues
    one_hot_encodings.pt                        [1389, 52, 21]    <- 1389 variants, 21 COLUMNS
    coords_tensor.pt                            [N, 4, 3]  ANGSTROM
    mask_tensor.pt, deltaG.pt

**one_hot is stored with 21 columns, not 20 — but this is HANDLED, and I checked rather than
assuming a bug.** `Megascale-fineTuning/train.py:317` does
`batch['one_hot'] = batch['one_hot'][:, :, :, :-1]`, dropping the trailing column before the graph
is built. `get_graph` produces width 1092 with the last 20 columns being one-hot, so every
`[20, K]` descriptor table is correctly sized. **No action needed** — recorded so nobody
re-discovers the 21 and assumes a mismatch.

## THE EMBEDDINGS ARE CONTEXTUAL — settled by measurement, never tested before

Same residue TYPE at two positions in the same protein:

| residue id | positions | max abs diff | cosine |
|---|---|---|---|
| 3 | 0, 7 | 0.8554 | **0.1764** |
| 16 | 1, 44 | 0.7310 | **0.1264** |
| 13 | 2, 12 | 1.4252 | **0.1455** |

**Cosine 0.13-0.18 — nearly orthogonal.** ProtT5 encodes position and neighbourhood, NOT residue
identity. This kills the "maybe it is just a 20-vector lookup" hypothesis outright: if it had been a
lookup, the whole descriptor question would have collapsed to linear algebra over 20 points.

## But identity IS largely recoverable — which is what matters for W6

Linear probe, embedding -> residue identity, held-out:

    accuracy 0.625   (chance 1/21 = 0.048)

**A descriptor block is a deterministic function of residue identity.** Identity is 62.5% linearly
recoverable from ProtT5 already, so a per-residue-type descriptor block is **largely redundant with
information the network can already extract**. This is exactly what Ofir concedes in his thesis:
physicochemical properties are already implicit in language-model embeddings, and his case for
descriptors is COVERAGE of non-canonical residues — which our 28 canonical proteins do not need.

**Consequence: W6 cannot be justified as adding information. Its only defensible value is
regularisation (sharing strength across chemically similar residues) or open-alphabet
generalisation.** The design must say so instead of claiming new signal.
Caveat: n=52 residues in one protein, one linear probe. Repeat across proteins with a
residue-disjoint split before quoting 0.625 as the number.

## ProtT5 is recomputed PER VARIANT, and one mutation moves everything

WT variant vs mutant variant: **all 52 of 52 positions differ.** A single point mutation changes the
embedding at EVERY position, not just the mutated one.

Two consequences:
1. The mutation signal reaching the network is **global, not local**. The model is not being handed
   "position 17 changed"; it gets a wholly new 52x1024 field.
2. **ddG does NOT cancel the embedding.** The metric rule says ddG cancels anything identical
   between WT and mutant — the embedding is NOT identical, so it survives the difference. That is
   why `--unfolded_emb zero` moved the ddG numbers at all, and it means ProtT5 is one of the few
   blocks that acts on BOTH channels.

## Architectural fact that constrains every feature-block design

The GCN branch reads **`x[:, :32]` only** — D(16) plus HALF of Fb. It never sees the embedding, and
**any new block inserted at offset 48 is invisible to it.** So W5/W6/W9/W11 reach the GAT branch
alone. Any claim that a new block "is used by the model" must say WHICH branch.

## What to do with this

- Re-run the identity probe across many proteins with a residue-disjoint split, and report R^2 for
  hydrophobicity/charge/volume too. That fixes the honest ceiling on what W6 can add.
- State in the write-up that ProtT5 is contextual and per-variant: it is the reason the embedding
  behaves as an offset channel (W0: zeroing it in the unfolded pass collapses var(E_u) by 67%)
  AND still contributes to ddG.
