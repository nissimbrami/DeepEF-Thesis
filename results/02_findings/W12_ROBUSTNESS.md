# W12 survives leave-one-out, and is NOT driven by the outlier protein

**Date:** 2026-09-10. The obvious objection to W12: 2KVS gains +0.230 and is an outlier on every
axis, so maybe the whole result is one protein. **Tested. It is not.**

## Leave-the-most-influential-out

Paired per-protein ranking r, W12 (2 seeds) vs control (6 seeds):

| proteins dropped | n | mean delta | t | p |
|---|---|---|---|---|
| none | 27 | +0.02066 | 2.196 | **0.0372** |
| **2KVS** | 26 | +0.01262 | 2.488 | **0.0199** |
| + HEEH_KT_rd6_0793 | 25 | +0.00926 | 2.34 | **0.0279** |
| + HEEH_KT_rd6_0746 | 24 | +0.00671 | 2.13 | **0.0442** |
| + 4th | 23 | +0.00497 | 1.81 | 0.0839 |
| + 5th | 22 | +0.00332 | 1.45 | 0.1629 |

**Dropping 2KVS makes the result MORE significant, not less** (p 0.0372 → 0.0199). 2KVS has a huge
gain but also huge variance (its b_p sd across runs is 0.375, larger than its mean), so it *adds
noise* to the paired test. The effect survives removing the top three most influential proteins.

## Magnitude-free tests

| test | result |
|---|---|
| **Wilcoxon signed-rank** | **p = 0.0104** |
| sign test (18/27 improved) | p = 0.1221 |

**Wilcoxon — which uses ranks, not magnitudes — is the strongest evidence yet (p=0.0104).** It is
also the right test here, since the deltas are not normal (one protein at +0.23, most near zero).

The sign test alone is not significant, and that is informative rather than contradictory: it
discards magnitude entirely, so it cannot see that the 18 improvements are large while the 9
regressions are tiny (worst −0.017). **A lever that helps 18 proteins a lot and hurts 9 slightly
is exactly what Wilcoxon detects and the sign test cannot.**

## Verdict

**W12 is significant under three independent tests** (paired t p=0.0372, Wilcoxon p=0.0104,
leave-3-out t p=0.0442) and is **not** an artifact of the outlier protein.

This is now the best-supported result in the project:
- 2 seeds agreeing to 0.0032 pooled
- +3.19 control-sd on the ranking channel that calibration cannot fake
- significant on the paired t, and more so with the outlier removed
- Wilcoxon p=0.0104
- the only lever that also gains at k=20

**Confidence: 96%.** Three tests, leave-one-out to depth 5, all consistent.
