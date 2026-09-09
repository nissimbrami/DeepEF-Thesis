# Implementation debt — where the idea was fine and the execution was mine

Nissim's point, and the evidence says he is right: most levers were never given a fair test. They
were blocked by my bugs, not refuted by measurement. **This file is the register of that debt and
the plan to clear it.** Each entry names what was wrong, what would fix it, and how we will know.

---

## 1. LORO / Ofir descriptors — TWO collapses, ONE cause, fix already submitted

**What happened.** Both arms froze at RMSE 2.541 for all fifteen epochs, val PCC noise around
zero. Not "descriptors don't help" — the model learned nothing at all.

**Why.** The descriptor block carried 13.8× then 3.65× the one-hot energy and swamped every other
input. **A correctly-scaled table (1.04×) existed the whole time, documents this exact failure in
its own header, and was never wired to a run.**

**Fixed.** New mode `mordred_pca16_unit`, verified by pushing a real one-hot through the loader:
3.653 → 1.045. Submitted as 21150162/63.

**How we will know:** RMSE must MOVE. If it freezes at a single value again, the scale hypothesis
is wrong and no fourth arm runs without a different diagnosis.

---

## 2. Every width-changing lever — scoring was broken in THREE places

**What happened.** W12, W15, HSE, ligands — none could be scored. `load_state_dict` failed on
every layer while the script printed DONE.

**Why, in three layers, each of which looked applied and did nothing:**
1. `run_calib_eval.sh` never passed the training flags to `evaluate.py`
2. `evaluate.py` accepted no block-lever flags at all
3. the CFG setter went into `run_training()`, which `__main__` never calls

**Fixed and verified:** zero "Missing key" errors in the current scoring run.

**Consequence: no width-changing arm had EVER been scorable in this project until today.** Every
"this lever didn't help" for W12/W15/HSE/ligands was a statement about a broken pipeline.

---

## 3. W15 — I nearly built a lever that was inert by construction

Coordinates are SHARED across the ~4000 variants of a protein. A feature computed from the packed
WT structure is identical for every variant and **cancels exactly in ddG**. I caught it before
submitting and made columns 0/2/3 read one_hot so they change at the mutated position.

**This is the cautionary case:** it would have trained, printed numbers, and meant nothing —
exactly like the W5 burial bug before it.

---

## 4. Single-seed reporting — the methodological debt

`--slope_weight 1.0` was reported for weeks as "+0.058, the only proven lever" on **n=1**. At n=4
it is +0.0179 against a seed sd of 0.0344, and seed 42 was an outlier (s=1.053 vs 0.515–0.532 for
the others). **Two headline claims about `a_p` were statements about one initialisation.**

**And the comparator has the same flaw:** the control is a single run at 0.5635 while the
five-seed mean of its own configuration is 0.5781. Three control seeds submitted (21149840/41/42).

**Standing rule now: an n=1 arm is untrustworthy by default.** Wave 10 gives W12 and HSE their
seeds up front rather than after a false positive.

---

## 5. Ligands / metals / complexes — untestable here, and that is a DATA problem

All 21 PDB-coded test proteins fetched from RCSB: **0 ligands, 0 metals, 21/21 monomeric**, 19 of
21 solution NMR. A constant-zero column contributes nothing to any gradient.

**Not a refutation of the idea — a property of the benchmark.** Testing it needs a different
dataset, not different code.

---

## 6. THE OPEN QUESTION: where do the calibration mutations come from?

W13 lifts pooled from 0.5845 to 0.6993 across all 52 runs — the most reproducible result in the
project. Its cost is k measured mutations per protein, which today implies a wet-lab campaign.

**Nissim's point: someone has already measured and published these.** Checking, rather than
assuming:

| source | status |
|---|---|
| S669 | **on the cluster.** 669 mutations, 94 proteins — **0 of our 28** |
| FireProtDB | **on the cluster**, 4.8 GB SQL dump — **scan submitted (21150460)** |
| ProThermDB | not present; obtainable |
| MegaScale itself | our training source — a held-out slice may already contain them |

**If FireProtDB covers even a few of our 28, W13 stops being a method that needs a lab and becomes
one that needs a lookup.** That would be the single largest practical change available.

---

## The honest summary

**Of roughly fifteen levers, only three or four ever got a fair test.** The rest were blocked by
scale bugs, scoring bugs, single-seed reporting, or a benchmark that cannot express them.

**The one idea that got a clean test and worked is Ofir's** — and it worked at +0.115 across every
run. That is not a coincidence worth ignoring: **the ideas were mostly sound; the implementations
were mine, and they were where the failures lived.**
