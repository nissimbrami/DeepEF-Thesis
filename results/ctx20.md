
---

# CHECKPOINT 20 — 2026-09-08 — TWO AGENT CLAIMS TESTED, BOTH FAIL. b_p IS NOT LENGTH-DRIVEN.

Working solo from `results/TASKS.md`, no background agents. Every number below was computed by the
main session.

## K4 — the "1/sqrt(N) embedding artifact" does NOT reproduce

An agent reported `corr(N, per-residue embedding norm) = -0.9938` and proposed a one-line rescale
as the cheapest b_p lever available. **Measured:**

| quantity | agent claim | measured |
|---|---|---|
| corr(N, emb_norm), 20 proteins | **-0.9938** | **-0.3927** |
| corr(N, emb_norm), our 28 | — | **-0.2946** |
| corr(emb_norm, b_p), our 28 | — | **+0.3863** |
| corr(N, b_p), our 28 | — | -0.3150 |

At n=28 the threshold is |r| > 0.392, so **none of these is significant**. And `|e|*sqrt(N)` is not
constant (36.5 -> 53.2 across proteins), so there is no clean 1/sqrt(N) law to remove.

**The proposed lever rests on a correlation that is three times weaker than claimed. Do not
implement it.** K5 is therefore dropped.

## K6 — `--dg_length_norm` does not shrink b_p; the /n arm is DEGENERATE

The flag exists (`none|n|sqrtn`, train.py:74, applied at 1016-1019) and has never been run. Because
it divides the PREDICTION, its effect on b_p can be simulated exactly on predictions we already
have — no GPU. Over 12 eval CSVs:

| | mean std(b_p) |
|---|---|
| none | **1.3076** |
| /n | 0.9050 |
| /sqrtn | 0.9159 |
| **corr(N, b_p)** | **+0.0252** |

It looks like a 31% improvement. It is not.

**The /n column converges to ~0.905 for EVERY checkpoint** — including ones that START at 0.876,
where "normalising" makes it worse. That constant is not a coincidence:

    std(true WT dG) over the 28 test proteins = 0.9042

**Dividing by N drives pred/N to nearly zero, so b_p -> -true_dG and its spread becomes the spread
of the true labels.** The arm does not remove a length bias; it deletes the prediction. Any future
run of `--dg_length_norm n` that reports a smaller std(b_p) is measuring this degeneracy.

**And the direct test is flat: corr(N, b_p) = +0.0252 across 12 checkpoints. b_p is NOT
length-driven**, so the extensivity argument for a length-normalised head does not hold.

## What this leaves

Both cheap architectural attacks on b_p are now closed, along with the label-side explanations
already retired in CHECKPOINT 19 (distribution shift r=+0.043; Ofir's mean-gap p=0.588; attenuation
dead by 200x). **b_p has survived: length, embedding norm, label shift, distribution shift, and a
feature-based corrector.** What remains untested is the reference state itself — where the coil
already produced our only real b_p movement (dG MAE 4.9650 -> 3.9521) and where `gld_dg_coil` is
running now.

**Method note worth keeping:** both claims were plausible and both failed the same way — a
correlation quoted without its n, and an improvement quoted without asking what the improved number
would be if the model predicted nothing at all. **Always compute the degenerate baseline.**
