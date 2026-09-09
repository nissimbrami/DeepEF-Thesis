# Ligands, metals and complexes — the question is CLOSED, verified against the PDB itself

## What was done

Every one of our 21 PDB-coded test proteins was fetched directly from the RCSB REST API and
annotated for oligomeric state, non-polymer entities (ligands), and bound metals — with
crystallisation additives excluded (SO4, GOL, EDO, PEG, …), because of the 205,648 het-code
occurrences in the 100k catalogue only 6.6% are real cofactors while 27.4% are additives.

The other 7 of our 28 are designed sequences (`HHH_rd1_0244`, `r12_757_TrROS_Hall`, …) with no
PDB entry by construction.

## Result — unanimous

| quantity | count over 21 |
|---|---|
| real ligands (additives excluded) | **0** |
| bound metals | **0** |
| multi-chain deposited entries | **0** |
| **monomeric, single-chain** | **21 / 21** |

19 of 21 are **solution NMR** structures; 1TUC is the only X-ray entry. That explains the
uniformity: NMR structures of small domains are almost by definition single, ligand-free chains.

## Verdict

**W11 ligands, W9 metals, and the complex/BSA idea are UNMEASURABLE on this benchmark.**

A feature column that is identically zero for all 28 test proteins contributes nothing to any
gradient — the model cannot learn from it and we cannot measure it. This is **a property of the
benchmark, not evidence against the idea.** On a metalloprotein or protein-complex dataset these
levers could well matter; here they cannot be tested at all.

## Why this took three attempts to settle, and what was wrong each time

1. **First claim: "zero variance, unmeasurable."** Taken from an agent report, never verified.
   Right conclusion, no evidence.
2. **Then: "the catalogue has 67,856 complexes, so there IS variance."** Correct about the
   catalogue — and irrelevant, because **0 of our 28 test proteins and 0 of 247 PDB-like training
   proteins appear in it.** Catalogue median length 477aa (p5 = 94) against our 42–72aa: two
   different populations, not a subset.
3. **Now: fetched our own 21 structures from the PDB.** 21/21 monomeric, ligand-free, metal-free.

**The conclusion never changed; only now is it evidence.** Confidence: **95%** — this is the
primary source, not a join, not a report.

## What this closes

- `--ligand_nodes` (W11, 659 lines) — queued as 21144535 purely to confirm it is **inert rather
  than harmful**; it cannot show a gain.
- W9 metals — stays retired.
- Complex/BSA — no flag needs to be built.

**GPU time saved by not pursuing these: substantial. The honest thesis sentence is that our
benchmark contains no complexes, so this class of feature is out of scope here.**

`results/08_data/pdb_annotations.json` holds the full per-protein record.
