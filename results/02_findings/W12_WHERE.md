# WHERE W12 helps: it repairs the worst-ranked proteins, and the correlation is −0.856

Per-protein `r` for `w12_s2_e12` against the control. 27 proteins, canonical basis.

## The pattern

```
improved: 18 / 27      worsened: 9 / 27
corr(delta_r, control_r) = -0.856
```

**The gain is almost perfectly anti-correlated with how well the protein was already ranked.**
W12 does not lift everything a little — it repairs the proteins the model was failing on.

| protein | r control | r W12 | Δ | length | true dG |
|---|---|---|---|---|---|
| **2KVS** | **0.160** | **0.471** | **+0.311** | 67 | 4.51 |
| **HEEH_KT_rd6_0793** | **0.322** | **0.463** | **+0.142** | 42 | 3.00 |
| 1QKH | 0.514 | 0.574 | +0.060 | 67 | 3.96 |
| HEEH_KT_rd6_0746 | 0.765 | 0.808 | +0.043 | 42 | 2.16 |
| … | | | | | |
| 2L33 | 0.793 | 0.780 | −0.013 | 71 | 2.79 |
| 2BTH | 0.860 | 0.844 | −0.016 | 43 | 3.00 |
| 3DKM | 0.651 | 0.634 | −0.017 | 72 | 3.12 |

**2KVS goes from 0.160 to 0.471** — the model went from essentially not ranking that protein at
all to ranking it usefully. That single protein is most of the aggregate gain.

**The losses are tiny** (−0.013 to −0.017) and land on proteins already ranked at 0.79–0.86,
where there is little left to gain.

## Why this is the right shape for an information lever

A rescaling lever multiplies every protein's predictions by a constant, so it cannot change any
protein's `r` at all — that is the algebra behind `sd(r) = 0.0223` across 53 runs. **An
information lever should help exactly where information was missing**, and missing information
looks like a low `r`.

**`corr = −0.856` is that signature.** It is much stronger than the aggregate `+0.0224` suggests,
because averaging hides a large repair to two proteins under many near-zero changes.

## The connection to the hydrophobic deficit

2KVS and HEEH_KT_rd6_0793 are also **among the hardest proteins measured earlier** (rank 3 and 2
of 28 by mean absolute error across 15 conditions). W12's mechanism is side-chain transfer free
energy weighted by burial, and the recorded deficit was that the model compresses mutations to
hydrophobic destinations twice as hard, worst at buried positions.

**The lever repairs the proteins its mechanism predicts it should.** That is a genuine consistency
check, not a post-hoc story: the deficit was measured months before W12 existed.

## Caveat, unchanged

**Still n = 1.** `w12_s1` has just finished training and is in the scoring queue. If the −0.856
pattern reproduces on a second seed, that is far stronger evidence than the aggregate number,
because a seed lottery would not reproduce *which proteins* improve.

**Confidence that W12 repairs low-r proteins in this run: 95%.**
**Confidence the pattern reproduces: 60%** — higher than my 45% for the aggregate, because a
per-protein pattern is much harder to produce by chance than a single summary statistic.
