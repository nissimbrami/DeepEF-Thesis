
---

# CHECKPOINT 21 — 2026-09-08 — K1 DONE: 2K5H FIXED, THE HEADLINE IS 35% SMALLER

## Root cause, traced to the file

`data/Processed_K50_dG_datasets/mutation_datasets/2K5H.csv` (4,107 rows) **concatenates THREE
backgrounds** and the wrong one sorts first:

    row 0    : name=2K5H.pdb_G11S   deltaG=1.723086   mut_type=wt
    idx 2738 : name=2K5H.pdb        deltaG=4.805470   mut_type=wt   <- the TRUE wild type

Both rows are labelled `mut_type='wt'`, because `2K5H.pdb_G11S` genuinely IS the wild type *of its
own mutant background*. The WS-1 convention treats row 0 as the reference, so every one of 2K5H's
ddG labels was measured against a mutant background.

**Measured shift: +3.0824 kcal/mol.**

**2K5H is the ONLY affected protein.** I checked all 28 against
`data/MsDs/mutation_files/`: only 2K5H has multiple background files (`[WT]`, `_G11S`, `_G23A`).
Isolated defect, not systemic.

## The fix respects the read-only rule

The source file is owned by `shaharax` and reached through a symlink into his tree, and
**shaharec/DeepPEF is read-only forever**. So the corrected copy was written to **our own**
directory instead:

    data_fixed/mutation_datasets/2K5H.csv
    row0: name=2K5H.pdb deltaG=4.805470     (was 2K5H.pdb_G11S / 1.723086)
    4107 rows preserved -- nothing deleted, the mutant-background rows are simply no longer first

`scripts/fix_2k5h.py` does this reproducibly (`--check` / `--apply`).

## THE CORRECTED HEADLINE — 22 eval CSVs

Because the fix shifts 2K5H's ddG by a CONSTANT, its effect applies exactly to existing predictions.

| | pooled | oracle | gain |
|---|---|---|---|
| with the bug | 0.5018 | **0.7391** | **+0.2373** |
| **corrected** | 0.4899 | **0.6443** | **+0.1544** |

**The bug inflated the offset-removal gain by 35%** (+0.2373 -> +0.1544) and the oracle by 0.095.

A constant per-protein label shift IS a per-protein offset by construction, so the oracle was
partly rediscovering an offset we had introduced.

## What survives, stated honestly

**The calibration effect is real and still substantial: +0.154 on pooled ddG PCC from offset
removal alone.** What changes is its size. Every number quoted from here must use the corrected
figures:

- offset-removal oracle: **0.644**, not 0.739 (and NOT the old 0.70-0.72)
- offset-removal gain: **+0.154**, not +0.237
- the "top-2 proteins carry 78% of the gain" claim needs recomputing on corrected labels, since
  2K5H was one of the two and its contribution was inflated

**Still to do:** the running factorial and golden-lane jobs were trained against the buggy labels
for 2K5H. Their ddG for that one protein is shifted, which affects its per-protein a_p and b_p but
not the other 27. Decide whether to re-score with `data_fixed/` or accept a documented caveat.
