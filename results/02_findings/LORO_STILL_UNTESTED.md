> **ROOT CAUSE FOUND 2026-09-10.** The block is bit-identical between the folded and
> unfolded passes (`torch.equal` = True, max|diff| = 0.000e+00), so it cancels exactly in
> dG and has no gradient path. Train loss is constant to five decimals across all 15
> epochs. See `W6_ROOT_CAUSE.md` and `scripts/gate_state_dependence.py`.

# The LORO descriptor arm scored — and it collapsed AGAIN. The hypothesis remains untested.

**Date:** 2026-09-10. `loroW_desc_s42_e14` finally produced a CSV (after today's `evaluate.py`
fix). **The number is not a result.**

## What it scored

| arm | pooled | mean r | median r |
|---|---|---|---|
| LORO one-hot (e14) | 0.5773 | 0.7235 | 0.7895 |
| **LORO descriptors** | **0.0056** | **−0.0137** | **−0.0249** |
| loroWdesc2 (known broken) | −0.0026 | −0.0153 | +0.0045 |

Descriptors lose to one-hot on **0 of 27** proteins, mean −0.7373, p<0.0001.

**That is not a lever losing. `mean r = −0.014` means the predictions are independent of the
labels** — the model learned nothing at all. It is the same signature as the arm already recorded
as broken.

## The proof, from the training log

`logs/gld_loroW_desc_21107689.out`:

```
RMSE=2.541   RMSE=2.541   RMSE=2.541   ... (all 15 epochs, identical)
```

**Validation RMSE is frozen at 2.541 for every epoch.** The optimiser never moved. This is the
exact failure already diagnosed for K9.

## Root cause: the same scale bug, still in the executing path

The arm loaded `mordred_pca16`. Measured energy of every descriptor table on disk:

| table | mean\|x\| | vs one-hot |
|---|---|---|
| `data/aa_descriptors_mordred_pca16.csv` (**what ran**) | 0.634 | **12.69x** |
| `data/aa_descriptors.csv` | 0.785 | 15.69x |
| `data_fixed/aa_descriptors_mordred_pca16_z.csv` | 0.814 | 16.27x |
| **`data_fixed/aa_descriptors_mordred_pca16_unit.csv`** (**the fix**) | 0.203 | **4.07x** |

A block entering at 12.69x the energy of the one-hot columns beside it swamps the input and the
network diverges to a constant.

**The unit-normalised table exists and was built precisely to fix this — and once again it was not
the table the run used.** This is the third time the pattern recurs: *the fix exists, looks
applied, and is not in the executing path.*

**Note also the `_unit` table is 4.07x, not the 1.04x recorded earlier.** That earlier figure does
not reproduce against the file now on disk, so even the fix needs re-checking before it is trusted.

## Status of the hypothesis

**`FINDINGS.md` §8.3 records a prediction made before any result:** *the descriptor arm will not
beat one-hot by much, because ProtT5 already carries the chemistry (held-out hydropathy R²=0.704).*

**That prediction is still untested.** Reporting "descriptors lose" from this run would be
confirming a prediction with a broken arm — exactly the error §14 exists to prevent.

## What is actually needed

One arm with `--aa_descriptors mordred_pca16_unit`, **plus a pre-flight assertion** that the loaded
table's mean|x| is within ~2x of the one-hot block. `gld_descunit_s42/s1` are queued and
`pub_p_descunit_s42` is running now — **the first honest attempt at this test.**

The falsifier is registered: **if val RMSE moves off 2.541, the arm is alive; if not, it collapsed
again.**

**Confidence: 97%** — the frozen RMSE is read directly from the training log and the table energies
are measured from the files on disk.


---

# UPDATE, same day: the FALSIFIER FIRED — the scale bug was NOT the root cause

`pub_p_descunit_s42` is running with the corrected table, and the registered falsifier was
*"if val RMSE moves off 2.541, the arm is alive"*.

```
descriptor matrix: mode=mordred_pca16_unit  K=16  md5=2396e12c
RMSE=2.529  RMSE=2.529  RMSE=2.529 ... (7 epochs, identical)
```

**The correct table IS loading — md5 confirms it — and the arm collapses anyway.**

## What this rules out

| candidate cause | verdict |
|---|---|
| wrong table loaded | **ruled out** — md5 verified in the log |
| table scale (12.69x energy) | **ruled out** — unit table is 4.07x and still collapses |
| numerical pathology in the table | **ruled out** — cond 1.0, uniform col std 0.2500, 0 NaN, rank 16/16 |
| degenerate "predict the mean" | **ruled out** — that baseline is RMSE 1.30, not 2.53 |
| the architecture silently dropping the block | **ruled out** — `hydro_net.py:835` raises at construction for exactly this, and it did not fire |

**So a run that loads a clean, correctly-scaled, full-rank table into an architecture that
provably reads it still fails to train at all.** RMSE 2.529 is worse than predicting a constant
(1.30), which means the descriptor block is actively destabilising the optimisation rather than
being ignored.

## Status

**The descriptor hypothesis is STILL untested, and the cause is now unknown** — the two obvious
explanations (wrong table, wrong scale) are both eliminated by measurement. This is a stronger
and more useful negative than "the table was mis-scaled", because it says the problem is in how
the block enters training, not in the data.

**Do not report any descriptor result until an arm shows val RMSE moving.** The falsifier stays
registered for `gld_descunit_s42/s1`.

**Confidence: 95%** on the eliminations (each is a direct measurement); the remaining cause is
explicitly unknown.
