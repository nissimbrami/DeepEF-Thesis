# Re-checking the "zero variance" claim myself — I was wrong, and so was the framing

I had recorded that ligands / metals / complexes are unmeasurable because "all 28 test proteins
are monomers with zero variance". **I took that from an agent report and never verified it.**
Nissim pushed back. He was right. Here is what the data actually says.

## What the catalogue really contains

`data/FINAL_DATASET_100k_030926.csv` — **100,246 rows, 21 columns**, fully populated:

| column | coverage | values |
|---|---|---|
| `Is Complex?` | **100%** | **Yes 67,856** · No 31,852 · Unknown 538 |
| `Oligomeric State` | **100%** | Monomer 31,852 · 2-chain 29,253 · 4-chain 13,723 · 3-chain 7,350 · 6-chain 5,049 · 8-chain 3,117 |
| `Chain Composition` | **100%** | homomeric 66,336 · heteromeric 22,775 · protein/NA 5,978 · protein/oligosaccharide 4,591 |
| `Ligands_x` | **0.004%** | only **4 rows** populated |
| `BSA_Percentage` | **0.005%** | only **5 rows** populated |

**Complex state is NOT zero-variance — it is a 68/32 split across 100k structures.**
That part of my earlier statement was simply false.

## But the decisive fact is different, and it is worse

**Our 28 test proteins do not appear in the catalogue at all.**

```
distinct 4-char PDB prefixes in catalogue: 66,809
our 28 proteins found:                     0
```

21 of our 28 are real PDB codes (2K5H, 6EWT, 1W4H, 1GYZ, 1QP2 …) and **not one of them is there.**
The reason is size:

```
catalogue AA Length:  median 477   p5 94   p95 3320
entries <= 75aa:      3,105  (3.11% of the catalogue)
our 28 proteins:      42-72aa, median 56
```

**The catalogue is a corpus of large structures. Our proteins are tiny designed and NMR domains
that sit below its 5th percentile.** They are not a subset of it; they are a different population.

## What this changes

**Corrected claim:** it is not that complex-state has no variance in nature — it has plenty.
It is that **the annotation cannot be joined to our test set**, because our test set is not in the
annotated corpus. Two separate failures, and I had merged them into one wrong sentence.

**Ligands and BSA are genuinely unusable even for the 100k**: 4 and 5 populated rows respectively.
That is not a join problem, that is an empty column.

## The routes that are actually open

1. **Annotate our 28 directly from the PDB.** 21 have real PDB IDs. Fetching oligomeric state,
   HETATM records and interface area for 21 structures is a small, bounded job — not a
   100k-scale problem. **This is the honest way to test the idea and it was never attempted.**
2. **Check the training set, not just the test set.** The lever is trained on MegaScale proteins;
   if THOSE vary in complex state, the feature can learn something even if the 28 do not vary.
   **I never checked the training population at all.**
3. **Accept it as out of scope, explicitly.** If (1) shows all 21 are monomeric NMR/designed
   domains, then the honest thesis sentence is "our benchmark contains no complexes, so this
   class of feature is untestable here" — a statement about the BENCHMARK, not about the idea.

## Method note

This is the failure mode I had already written into memory: **an agent's conclusion recorded as
fact without an independent check.** The claim survived several documents because it sounded
reasonable. Nissim caught it by asking where the data actually was.
