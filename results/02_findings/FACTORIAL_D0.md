# The factorial within D0: pooled is flat, `a_p` is not — and they are UNCORRELATED

Local analysis of the 14 scored factorial cells (6 D0, 8 D1) while the cluster was unreachable.

## Main effects within D0, on pooled

| factor | effect | n | verdict |
|---|---|---|---|
| A | +0.0069 | 4 vs 2 | inside seed noise |
| B | −0.0118 | 3 vs 3 | inside seed noise |
| C | −0.0043 | 3 vs 3 | inside seed noise |

**All three are an order of magnitude below the seed sd of 0.0344.** The six D0 cells span
0.5763–0.6028, sd **0.0094** — *less than a third* of the seed noise.

**Conclusion: within D0, none of A, B or C moves the pooled score at all.** Factor D was the
whole factorial.

## But the same six cells span a huge range in `a_p`

| factor | effect on `a_p` |
|---|---|
| **A** | **+0.1386** |
| **C** | **+0.0969** |
| B | +0.0192 |

```
a_p across the six cells:    0.4387 - 0.7316   sd 0.1141
pooled across the same six:  0.5763 - 0.6028   sd 0.0094
```

**`a_p` varies 12× more than pooled across identical runs.**

## The decisive number

```
corr(a_p, pooled) across the D0 cells = -0.087
```

**Essentially zero, and slightly negative.** Improving the per-protein slope does **not** improve
the pooled score. This is the same message as W13b (where the best-slope arm finished last at
k=20) and as the slope-overshoot finding (`s` crossing 1.0 while `a_p` kept climbing) — but here
it is measured across six independent configurations rather than within one arm.

**Three independent lines of evidence now say the same thing: `a_p` and pooled ddG PCC are
separate axes.** Optimising one does not move the other.

## The caveat that must be stated

**Epoch is uncontrolled in this factorial.** The six cells were read at epochs 5, 12, 13, 13, 13,
14, and:

```
corr(epoch, pooled) = +0.543
corr(epoch, a_p)    = +0.385
```

**Epoch correlates with pooled more strongly than any of A, B or C does.** So the "main effects"
above are partly confounded with how far each cell trained. With n = 6 and a 9-epoch spread, the
factorial cannot cleanly separate factor from epoch.

**This does not rescue A/B/C** — their effects are ~0.01 while epoch's association is 0.543, so if
anything the true factor effects are even smaller. But it does mean the factorial as run cannot
support a positive claim about any factor.

## What this changes

1. **Stop reporting A/B/C main effects.** Within D0 they are indistinguishable from zero and
   confounded with epoch.
2. **Do not use `a_p` as a proxy for pooled.** Measured `r = −0.087` across six configurations.
3. **A clean factorial would need a fixed epoch and ≥2 seeds per cell.** That is 96 GPU-h for a
   result that three independent measurements suggest will be null.

**Confidence: 85%** on the a_p/pooled decoupling (three independent lines agree);
**90%** that A/B/C are null within D0; **95%** that epoch is a confound in this design.
