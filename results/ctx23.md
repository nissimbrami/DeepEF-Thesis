
---

# CHECKPOINT 23 — 2026-09-08 — K13: THE MODEL IS WORST AT BURYING HYDROPHOBICS

## The finding, verified over 10 checkpoints

The eval CSVs carry no mutation code, so the destination residue was recovered by joining on
`deltaG` (6 dp) against `mutation_datasets/<P>.csv`, whose `name` encodes the mutation
(`1PSE.pdb_A42W`). Ambiguous deltaG values were DROPPED rather than guessed. ~26,000 mutations
matched per checkpoint. 2K5H's ddG was corrected by -3.0824 throughout.

**Per-destination-residue within-protein slope, ordered by Kyte-Doolittle hydropathy:**

| residue | KD | slope | spearman |
|---|---|---|---|
| R | -4.5 | **0.528** | 0.649 |
| K | -3.9 | 0.537 | 0.648 |
| N | -3.5 | **0.565** | 0.653 |
| D | -3.5 | 0.542 | 0.650 |
| S | -0.8 | 0.524 | 0.638 |
| G | -0.4 | 0.506 | 0.616 |
| A | +1.8 | 0.419 | 0.567 |
| M | +1.9 | 0.340 | 0.461 |
| C | +2.5 | **0.267** | 0.356 |
| F | +2.8 | 0.288 | 0.381 |
| L | +3.8 | **0.282** | 0.388 |
| V | +4.2 | 0.309 | 0.428 |
| I | +4.5 | 0.311 | 0.412 |

    mean corr(slope, KD)    = -0.7344    sign-consistent 10/10
    mean corr(spearman, KD) = -0.7399

**Mutations TO hydrophobic residues are compressed roughly TWICE as hard as mutations to charged
or polar ones** (slope ~0.28-0.31 vs ~0.53-0.57).

## THE DECISIVE DETAIL: it is LOST INFORMATION, not a calibration error

`corr(spearman, KD) = -0.740` is **essentially identical** to `corr(slope, KD) = -0.734`.
**Rank accuracy degrades in exact lockstep with the slope.** Spearman falls from ~0.65 for polar
destinations to ~0.36-0.43 for hydrophobic ones.

That distinction decides whether any lever can help:

- If the slope fell while ranking held, a per-class rescale would recover it — a calibration fix.
- **Ranking falls too, so the information is not there to rescale.** The model does not merely
  under-react to buried hydrophobics; it cannot ORDER them either.

**`--slope_weight` cannot fix this**, and neither can any affine correction. It is the (1-r)
half of the K12 decomposition, made concrete and given a chemical identity.

## Why this is mechanistically credible

The dataset stores only **4 backbone atoms per residue (N, CA, C, CB) — no side chains**. Burying
a large hydrophobic (F, L, I, V, M, W) is dominated by side-chain packing and van der Waals
complementarity, exactly what CB-only geometry cannot see. Charged and polar substitutions are far
better captured by backbone geometry and solvent exposure, which the model does have.

**W and Y are the interesting exceptions** — slopes 0.341 and 0.353 despite negative KD. Both are
large aromatics, so the pattern tracks SIDE-CHAIN BULK rather than hydropathy alone, which
strengthens the packing explanation over a purely hydrophobic one.

## What this is worth

This is a mechanistic, chemically-interpretable account of where the model's ranking error lives,
and it points at a concrete architectural fix rather than a calibration one: **give the model
side-chain information**. That is a real thesis result and an argument for full-atom or
side-chain-aware features in future work.
