
---

# CHECKPOINT 24 — 2026-09-08 — K14: THE PRE-TRAINING DISTRIBUTION SHIFT, VERIFIED

Recomputed from the raw catalogue by the main session. Every figure reproduces the agent's report.

## The numbers

`data/FINAL_DATASET_100k_030926.csv`, 99,708 rows with a usable length:

    catalogue length: median 477, mean 1167.7, max 89,160
    rows in our 32-74 aa regime: 2,526  (2.53% of rows)
    RESIDUE share: 140,383 / 116,431,172 = 0.1206%

**Our entire problem domain was one residue in 830 of pre-training.** The residue share is the
right denominator because the pre-training loss is per-residue, so that is the share of the
gradient our regime ever contributed.

## And it is a structurally DIFFERENT kind of protein

| | catalogue overall | our 32-74 aa regime |
|---|---|---|
| SOLUTION NMR | 8.6% | **77.4%** (9.0x enriched) |
| Is Complex = Yes | 68.1% | **11.8%** |
| Monomer | 31.9% | **88.2%** |

Pre-training saw large, X-ray-solved, interface-rich multimers. We test on small, NMR-solved,
interface-free monomers. **The model was optimised on almost the opposite of what it is evaluated
on.**

## What this is, and what it is NOT

**IT IS** a well-powered characterisation of pre-training (n = 99,708) that needs no join and no
extra compute. It belongs in the thesis as context for why absolute dG is hard here.

**IT IS NOT the cause of b_p.** Measured directly against b_p: **r = +0.043, p = 0.829**, a clean
null. Across 25 shift and composition features, three reach nominal significance where ~1.25 are
expected by chance, and **none survives Bonferroni or BH-FDR** for either b_p or a_p.

The temptation is to say "the offset exists because of distribution shift". The data does not
support it, and the honest statement is that the shift is real, large, and **not measurably
connected to the per-protein offset**.

## Caveat kept from the agent's own analysis

`AA Length` is the entity/assembly length, not per-chain, so the 0.1206% is computed on assembly
residues. Per-chain lengths would be smaller and would move a little catalogue mass toward our
regime, so **0.1206% is a lower bound on the mismatch severity**, not an exact per-chain figure.
The NMR/complex/monomer contrasts are entity-level attributes and are unaffected.
