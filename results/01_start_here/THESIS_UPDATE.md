# Amendment to THESIS_SUMMARY — after seed replication

**This supersedes §1 and §7 of `THESIS_SUMMARY.md`.** Written after the three-seed replication of
the slope lever landed. 68 scored runs on the canonical basis.

## The correction

`THESIS_SUMMARY.md` states: *"Two things beat the noise band: `--slope_weight 1.0` (+0.058) and
requiring factor D0."* **The first half is now refuted by measurement.**

| seed | epoch (val-selected) | pooled | PP |
|---|---|---|---|
| s1 | e9 (val argmax 0.8240) | 0.5853 | 0.7809 |
| s2 | e13 (val argmax 0.8010) | 0.6242 | 0.7920 |
| s42 | e10 | 0.5798 | 0.7878 |
| **mean** | | **0.5964 ± 0.0242** | |
| control | e14 | 0.5635 | **0.7929** |

```
gain      +0.0329
seed sd    0.0344     -> INSIDE the noise band
```

**And per-protein PCC is worse than control in all three seeds.**

## What survives

| claim | status after replication |
|---|---|
| requiring D0 | **holds**, t = 9.93, zero overlap — but it is a *constraint*, not a gain |
| `--slope_weight 1.0` | **inside the seed band at n=3** |
| `--wt_anchor_weight 0.3` | n=1, and its weight sweep is monotone — the optimum was never bracketed |
| W13 measured offset | **+0.112, real, but FEW-SHOT** |
| eleven other hypotheses | rejected |

**Honest position: no zero-shot lever tested in this project has produced a reproducible gain in
pooled ddG PCC.**

## Why, in one number

Across the 29 healthy runs:

```
sd(r) = 0.0223      r = within-protein ranking quality
sd(s) = 0.2441      s = dispersion ratio,  a_p = r * s
```

**Every lever moves `s` and none moves `r`** — and `r` is what correlates with pooled
(+0.399 vs +0.281 for `s`). Four independent lines agree:

1. this replication
2. `corr(a_p, pooled) = −0.087` across the D0 factorial
3. `sd(s)/sd(r) = 11×` over 29 runs
4. W13b — the best-slope arm finished **last** at k=20

## What this means for the thesis, and it is not bad news

The thesis does not rest on a coefficient. It rests on five findings that are all still standing:

1. **The per-protein offset is not predictable** (ten methods, all negative held-out R²) **but is
   measurable** — five labelled mutations recover 60% of the recoverable gain at zero GPU cost.
2. **A data bug** (2K5H's reference row is a mutant) contaminated every earlier number.
3. **Calibration levers overlap rather than compose**, and now: they do not reproduce across seeds.
4. **The compression is desolvation on burial, not bulk** — with the arithmetic showing why no
   per-residue descriptor table can fix it (`one_hot @ T` is rank-1).
5. **Most recorded failures were execution faults**, caught by three rules: score on the metric the
   lever acts on, always compute the degenerate baseline, and verify an artifact rather than a
   DONE message.

**Adding a sixth: a single-seed result is not a result.** The project reported a lever as proven
for weeks on n=1. **That is the most transferable lesson here.**

## What is still unread

12 arms are built, gated and queued but have **never run** — W15 packing (5), K21 distogram (4),
K20 ensemble (3) — plus HSE, pooled_corr and readout. These are the levers that add *information*
rather than rescale it, which is exactly what the `r`/`s` decomposition says is needed.
**They are the only remaining chance of a zero-shot gain, and they are starved behind four older
jobs holding the golden cards.**
