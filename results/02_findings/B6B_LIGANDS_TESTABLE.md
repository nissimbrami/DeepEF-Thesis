# B6b — the ligand direction IS testable here. S669 is on disk, preprocessed, with 18 ligand proteins.

**Date:** 2026-09-10. Checklist item **B6b**: *"decide explicitly: evaluate on ProTherm/FireProtDB,
or record the direction as untestable here."*

**Half of this item was a decision. The other half was a measurable fact, and it is now measured.**

## What we had learned before this item

W11 ligands and W9 metals were recorded as **zero variance**: all 27–28 MegaScale test proteins
are single-chain, ligand-free, metal-free monomers. `FINDINGS` §9.1 correctly states this licenses
only *"not measurable here"*, never *"no effect"*.

**What was not good:** "not measurable **here**" was never followed by checking what *is* on disk.
`EXTERNAL_DATA_VERDICT.md` ruled out FireProtDB (it points back at MegaScale, our training source)
and S669 — but that verdict was about **b_p correction**, a different question from ligands.

## EXECUTE — what is actually available

| dataset | status |
|---|---|
| **S669** | **on disk, fully preprocessed**: `all_coords.pt`, `all_masks.pt`, `all_mutations.pt`, `all_ids.pt` — 669 mutations, **94 distinct PDBs** |
| FireProtDB | on disk as a 406 MB raw zip, not extracted |
| ProThermDB | not present |

**S669 is not a data-acquisition project. It is already in the form the model consumes.**

## The measurement — joined against the project's own het classifier

Joined the 94 S669 PDBs to `FINAL_DATASET_100k_030926.csv` (`protein_id`), classified het codes
with `data/het_classification.csv` (8,187 codes, 208 counting as a genuinely bound ligand).

    S669 PDBs matched to the catalogue : 62 of 94
    with ANY het code                  : 35 of 62
    with a REAL bound ligand           : 18 of 62
    with a METAL                       : 13 of 62

    our 27 MegaScale test proteins     : 0 of 27

Examples, with their actual ligands:

| PDB | ligands |
|---|---|
| 1A7V | **HEM** (heme) |
| 1FRD | **FES** (iron-sulfur cluster) |
| 1BNL | **ZN** |
| 1IR3 | ANP, **MG**, PTR |
| 1JLV | **GSH** (glutathione) |
| 1F8I | GLV, **MG**, SIN |

**The variance W11 and W9 need exists on a dataset already sitting on this cluster in tensor form.**

## What is decided by fact, and what is still Nissim's decision

**FACT — settled here:** the ligand/metal direction is **testable**, not untestable. 18 ligand-
bearing and 13 metal-bearing proteins against 0 in the current test set. Writing "untestable" in
the thesis would now be wrong.

**DECISION — still Nissim's, and I am not making it:** whether to evaluate on S669 at all. The
constraint is **leakage**, and it is a real one: S669 is a standard benchmark and its proteins may
overlap the training distribution. `EXTERNAL_DATA_VERDICT.md` ruled S669 out for the b_p question
on population grounds. **That leakage argument binds only if S669 is used as a scored test set.**
Using it purely to ask *"does turning on `--ligand_nodes` change predictions on proteins that
actually have ligands?"* is a **variance check, not a benchmark claim**, and leakage does not
invalidate a variance check.

**Two options, both defensible:**

1. **Variance check only** (cheap, no leakage exposure): run the existing model on the 18
   ligand-bearing S669 proteins with `--ligand_nodes` on and off, and report whether the feature
   changes anything at all. This tests that W11 is *wired and non-inert* — which, given that five
   silent no-ops have now been found, is worth knowing regardless.
2. **Full evaluation** (a thesis-scope change): adopt S669 as a secondary test set, which requires
   defending the leakage question explicitly.

**My recommendation is option 1**, because it is the only one that needs no scope decision, and
because `gate_state_dependence.py` already caught one inert feature this week. **But it is not
mine to choose, and I have not submitted it.**

## DONE

**Cost: ~30 min, no GPU.** The factual half is closed; the decision is stated with its options and
their costs rather than left as an open question.

**Confidence: 93%** — the het classification is the project's own curated table at 98.1%
occurrence coverage; the residual is that 32 of 94 S669 PDBs did not match the catalogue, so
18/62 is a **lower bound** on the ligand-bearing count.
