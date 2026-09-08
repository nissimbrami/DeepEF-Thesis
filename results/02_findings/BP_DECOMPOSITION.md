# T-E3 — b_p is 96.3 % one global constant. MAE was measuring that constant.

**CPU, minutes, no training.** Decomposes the per-protein offset `b_p = pred − true` (dG space,
28 proteins) into a single global constant plus a genuine per-protein spread.

## Result

| condition | MAE | global | spread | % of mean-square b_p that is the global constant | corr(b_p, N) |
|---|---|---|---|---|---|
| **base** | 5.0749 | **−5.0749** | **1.0155** | **96.3 %** | −0.215 |
| ν=0.500 b=5.82 | 4.0621 | −4.0621 | 1.1439 | 92.9 % | **−0.378** |
| ν=0.588 b=4.97 | 3.9949 | −3.9949 | 1.1246 | 92.9 % | **−0.456** |
| ν=0.620 b=4.97 | 3.9763 | −3.9763 | 1.1239 | 92.8 % | **−0.507** |
| noemb | 3.2414 | −3.2414 | 1.0427 | 90.9 % | −0.103 |

## What it means

**`MAE ≡ |global|` exactly, to four decimals, in every condition.** Look at columns 2 and 3 —
they are the same number. This is not a coincidence: with `n_under = 28/28`, MAE is *arithmetically
identical* to the global offset. **Every MAE we ever quoted was measuring one constant** — the same
constant for all 28 proteins, which any correlation is invariant to.

So `b_p = −5.0749 + (spread, std 1.0155)`, and **96.3 % of its mean square is that free constant.**
**The entire thesis target is the remaining 1.0155.**

> The coil reduced MAE 5.07 → 3.99 by shrinking a constant that was worth nothing,
> while making the spread that *is* worth something **worse** (1.0155 → 1.1246).

## The unexpected finding: the coil INDUCES a length artifact

`corr(b_p, N)` was **−0.215** at baseline — previously measured as ≈ 0 and correctly rejected as an
explanation of `b_p` (T7). But look at what the coil does to it:

```
base        −0.215
ν = 0.500   −0.378
ν = 0.588   −0.456
ν = 0.620   −0.507   ← monotone in ν
```

**The coil makes b_p length-dependent, and more so the larger ν gets.** This is a mechanism, not
noise: `d(i,j) = b·|i−j|^ν` grows with separation, so longer chains accumulate systematically
larger coil distances, and the error scales with N. **The "correct" physics (ν = 0.588) is the
worse artifact.**

**This retires T-E4 before it was run.** T-E4 asked whether the coil failure was a per-length
normalisation bug rather than a shape error. The answer is **yes, and here it is measured**:
the coil is mis-normalised per length. A length-normalised coil is the only version worth
retrying — and `--dg_length_norm` is already known to be degenerate, so that path is closed
unless a *non-degenerate* normalisation is designed first.

## Consequences

1. **Never quote MAE on dG again.** It is `|global|` and it is invariant-irrelevant. Report
   `std(b_p)` — this is now enforced by measurement, not convention.
2. **The reference-state programme should be judged on the 1.0155 spread only.** Every arm to date
   has raised it. **T-E1 (ensemble) must clear std(b_p) < 0.9972 or the programme ends.**
3. **T-E4 is CLOSED** — answered here at zero GPU cost.
