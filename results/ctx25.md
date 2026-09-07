
---

# CHECKPOINT 25 — 2026-09-08 — K11: THE "26,315 DOUBLE MUTANTS" DO NOT EXIST

## The claim

A data-mining agent reported that **26,315 double mutants for our 28 test proteins sit in no split
at all**, and called it a held-out generalisation test needing no new labels — potentially one of
the most valuable untouched assets in the project.

## Measured

Counting mutation codes (`_X<pos>Y`) across all 28 proteins' mutation files:

    0-point:  4,810 rows
    1-point: 56,214 rows
    2-point:  **2,356 rows**

Not 26,315. And every 2-point row belongs to **one protein**:

    2-point rows by protein: {'2K5H': 2356}

The example names give it away immediately:

    2K5H.pdb_G11S_A1Q,  2K5H.pdb_G11S_A1E,  2K5H.pdb_G11S_A1N, ...

**These are not double mutants. They are SINGLE mutations on the G11S background** — the exact
concatenation artifact that K1 fixed. `2K5H.csv` merges three backgrounds, so a single mutation on
the `_G11S` background parses as two mutation codes.

Removing the `_G11S` and `_G23A` background prefixes:

    true double mutants across all 28 proteins: **NONE**

## Verdict

**K11 is closed as a negative. There are zero genuine multi-point mutants for our 28 test
proteins**, so the proposed generalisation test does not exist. `--one_mut` (train.py:384, default
True) filters multi-point mutations anyway, and on this data it has nothing to filter.

## Why this matters beyond the task

This is the SECOND finding to come out of the 2K5H concatenation bug — first the inflated
offset-removal gain (35%), now a phantom dataset. **One malformed data file produced two separate
false leads**, and both were plausible enough to have been written into the thesis.

**Method note: when a count is surprisingly large, look at the ROW NAMES before believing it.**
The agent counted regex matches; the names said what the rows actually were.
