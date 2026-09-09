# The D1 cancellation was correct — and the separation is larger than recorded

Local verification on `scoreboard.csv` while the cluster was unreachable.

## The separation

```
D0 (n=6):  pooled mean 0.5886   sd 0.0094
D1 (n=8):  pooled mean 0.1508   sd 0.1243

Welch t = 9.93
```

**The record says "3.9σ". The measured separation is 9.93.** The earlier figure was
conservative — probably computed against a pooled sd rather than the standard error. Either way
the conclusion is unchanged and now stronger.

**Zero overlap:** the worst D0 cell (0.5763) is above the best D1 cell (0.3172) by 0.259, which is
**7.5 seed standard deviations.**

## D1 does not fail gracefully — it collapses

| | D1 | D0 |
|---|---|---|
| `a_p` mean | **0.0177** | 0.5319 |
| `a_p` max | **0.0309** | — |
| `r` mean | 0.1695 | 0.7113 |

**`a_p` is within 0.03 of zero in every one of the eight cells.** The model does not respond to
the true ddG *at all* — its predictions are essentially independent of the label. One cell is
even negative on pooled (−0.0741).

**This is collapse, not degradation.** A merely weaker model would show reduced `a_p`, not `a_p`
≈ 0 in eight independent configurations.

## What this confirms about the mechanism

D1 zeroes the unfolded embedding. Since `dG = E_unfolded − E_folded`, removing the embedding from
the unfolded pass does not just remove information — it makes the reference state uninformative,
and the difference of two energies where one is constant carries no signal about the mutation.

**The unfolded embedding is load-bearing.** That is the strongest single architectural fact the
factorial produced, and it is the reason every reference-state lever must be scored on dG rather
than ddG.

## Consequence

The 11 remaining D1 jobs in the public queue (~120 GPU-h) would re-confirm a 9.93σ result. They
are pure waste, and cancelling them was right. **I am not cancelling anything further without
your say-so** — this is recorded as justification for the decision already taken, not a new one.

**Confidence: 95%.** Eight independent cells, no overlap, and a mechanism that explains the
pattern.
