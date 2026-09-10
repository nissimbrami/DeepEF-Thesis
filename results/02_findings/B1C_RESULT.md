# B1c — the coil is not a SCALE problem. It is a SPARSITY problem.

**Date:** 2026-09-10. Checklist item **B1c**, done end to end under `PROTOCOL.md`.

## What we had learned before this item

Flory scored mean r **0.1922** (nu=0.588) and **0.1631** (+coil_edges) against a control of
0.7273 — near-random. B1a established both arms **trained healthily** (RMSE 1.450->1.086 and
2.313->1.628), so it is not an optimisation failure.

**What was not good:** the arm was rejected without ever checking whether the coil distances even
land inside the Gaussian kernel's useful range. **The direction the review asked for:** if the
block magnitude after the kernel is far from the baseline's, it is a scale problem — fixed by
recalibrating `b`, not by another nu sweep.

## PLAN (registered before execution)

**Falsifier:** a ratio far from 1 means the coil left the kernel's useful range.
**If X then Y:** ratio far from 1 -> scale problem, recalibrate `b`; ratio near 1 -> the
rejection is not about scale and `b` recalibration is not the fix.

## EXECUTE

Kernel `relu(exp(-0.08*d^2))` (`train_utils.py:390` / `:652`), N=60, b=3.8 A.

| block | fraction nonzero | sum | mean\|x\| |
|---|---|---|---|
| baseline tridiagonal | 0.0333 | 37.17 | 1.05e-02 |
| coil nu=0.500 | 0.3893 | 53.84 | 1.52e-02 |
| coil nu=0.588 | 0.2797 | 47.77 | 1.35e-02 |

    RATIO coil(nu=0.500) / baseline = 1.4485
    RATIO coil(nu=0.588) / baseline = 1.2851

## VERIFY — the falsifier did NOT fire

**The ratio is 1.45 and 1.29 — close to 1, not orders of magnitude.** Total block magnitude is
essentially preserved. **This is not a scale problem, and recalibrating `b` will not fix it.**
That closes the route the review proposed, on measurement.

**But the measurement found the real difference: SPARSITY.**

    baseline : 3.3% of pairs nonzero, each worth 0.3150
    coil 0.5 : 38.9% of pairs nonzero (12x denser), mean value 0.0391
    coil .588: 28.0% of pairs nonzero (8x denser), mean value 0.0483

Per-separation, what the network actually receives:

| \|i-j\| | baseline k | coil nu=0.5 k |
|---|---|---|
| 1 | 0.314995 | 0.314995 |
| 2 | **0.000000** | 0.099222 |
| 3 | **0.000000** | 0.031254 |
| 5 | **0.000000** | 0.003101 |
| 10 | **0.000000** | 0.000010 |

**The baseline unfolded state is a hard tridiagonal mask: only immediate neighbours exist.** The
coil replaces it with a smooth decay that gives every separation a small nonzero value.

**The same contact mass, smeared over 12x more pairs.** Each individual signal is ~8x weaker
(0.039 vs 0.315), and the sharp "only neighbours" structure — which is the entire information
content of the baseline unfolded reference — is destroyed.

## What this means

1. **`b` recalibration is ruled out as the fix** (measured, not argued).
2. **The failure mode is that the coil is too DENSE, not too large.** A physically smooth coil is
   a worse *reference* than a hard mask, because the folded-minus-unfolded difference relies on
   the unfolded state being sharply structured.
3. **The next configuration to try is a truncated coil**: `d = b*|i-j|^nu` with contacts zeroed
   beyond a cutoff, restoring sparsity while keeping the physical distance profile. That is a
   different lever from another nu value, and it is the first Flory variant that would not repeat
   the tested configuration.
4. It also explains the "16 identical channels" mismatch recorded in `FLORY_QUARTER_TESTED.md`:
   smearing across separations and broadcasting across atom-pair channels both reduce the
   information the block carries.

## DONE

**Cost: ~15 min, no GPU.** The review's proposed fix (recalibrate `b`) is closed by measurement;
the actual mechanism is identified and gives a new, untested configuration.

**Confidence: 95%** — arithmetic on the kernel and the coil formula, both read from source. The
residual uncertainty is that this is computed for a typical N=60 and b=3.8 A rather than per
protein, which affects the exact ratio but not the 12x sparsity difference.
