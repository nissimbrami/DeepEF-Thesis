# The three distogram arms confirm the B1d no-op, quantitatively

**Date:** 2026-09-10. `disto0.01/0.1/1.0_s42` scored. `B1D_DISTOGRAM_NOOP.md` claimed the flag was
inert (parsed, stored, never read by a loss). **These scores test that claim.**

| arm | pooled | mean r | median r | in mean-r sd |
|---|---|---|---|---|
| disto 0.01 | 0.5696 | 0.7268 | 0.7890 | −0.07 |
| disto 0.1 | 0.5562 | 0.7223 | 0.7842 | −0.77 |
| disto 1.0 | 0.5692 | 0.7245 | 0.7850 | −0.43 |
| control family | 0.5757 ± 0.0314 | 0.7273 ± 0.0065 | 0.7897 ± 0.0024 | — |

**A 100x sweep of the weight produces a pooled spread of 0.0134 = 0.43 control-sd, and a mean-r
spread of 0.0045 — smaller than the control family's own seed sd of 0.0065.**

**Three arms whose only declared difference spans two orders of magnitude are indistinguishable
from three replicate seeds of the control.** That is exactly what an inert flag predicts, and it
independently confirms the source-level finding (`DistogramHead` absent from the live model,
`--distogram_weight` read only for logging).

**All three sit slightly BELOW the control mean** (−0.07, −0.77, −0.43 sd), consistent with noise
around a baseline rather than any effect.

**Consequence:** these are not distogram results and must never be reported as such. They are
three extra seed-42 baselines. The real test is `pub_udisto0.1_s42`, built today, which applies
the head to the **unfolded** state — running now.

**Confidence: 98%.** The source evidence and the empirical spread agree.
