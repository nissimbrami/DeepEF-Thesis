# The slope lever, across three seeds — the project's decisive measurement

`--slope_weight 1.0`, three independent seeds, each read at its **own validation-selected epoch**
(s1 → e9, val argmax 0.8240; s2 → e13, val argmax 0.8010; s42 → e10). Canonical basis.

## The result

| run | pooled | PP | a_p | r | s |
|---|---|---|---|---|---|
| slope1.0_s1_e9 | 0.5853 | 0.7809 | 0.3903 | 0.7244 | 0.5322 |
| slope1.0_s2_e13 | 0.6242 | 0.7920 | 0.3952 | 0.7382 | 0.5246 |
| gld_slope1.0_s42_e10 | 0.5798 | 0.7878 | **0.7821** | 0.7250 | **1.0530** |
| **control** | 0.5635 | **0.7929** | 0.5211 | 0.7262 | 0.6983 |

```
slope arm:  mean 0.5964   sd 0.0242   n=3
control:    0.5635
gain:       +0.0329        seed sd 0.0344
```

## THE VERDICT: not established

**+0.0329 against a seed sd of 0.0344 — the gain is inside the noise band.**

The lever that was reported all along as "+0.058, the only proven lever" **does not survive seed
replication.** With n=3 the honest statement is: **the slope lever's effect on pooled ddG PCC is
indistinguishable from zero.**

**And per-protein PCC is WORSE than the control in all three seeds** (0.7809, 0.7920, 0.7878 vs
0.7929). The lever costs ranking quality while buying nothing measurable on pooled.

## The anomaly, and what it means

`a_p` is **0.78 for seed 42 but 0.39 for seeds 1 and 2** — a factor of two, from the same
configuration. Since every run was read at its own val-selected epoch, **this is a seed effect,
not an epoch artifact.**

Look at `s`: 1.053 for s42 versus 0.532 and 0.525 for the others. **Seed 42 crossed s = 1.0 and
overshot; the other two did not even reach it.** The same loss weight, applied to the same
architecture, lands on opposite sides of perfect calibration depending on initialisation.

**That is not a lever with a tunable strength — it is a lever whose effect is not reproducible.**
The earlier finding that "s crosses 1.0 between e4 and e8" was a property of seed 42 alone.

## Consistency with everything else measured tonight

- `r` is 0.724, 0.738, 0.725 against a control of 0.726 — **unchanged**, as in all 29 runs
  (sd(r) = 0.0223 vs sd(s) = 0.2441)
- `corr(a_p, pooled) = −0.087` across the D0 factorial — a_p and pooled are separate axes
- W13b: at k=20 the best-slope arm finished **last**

**Four independent lines now agree: moving `s` does not move the score.**

## What this changes for the thesis

The project has claimed one proven lever (slope) and one settled requirement (D0). **After seed
replication only D0 survives** — and D0 is a negative result (zeroing the unfolded embedding
destroys the model, t = 9.93).

**Honest position: no lever tested in this project has produced a reproducible gain in pooled
ddG PCC.** The measured contributions are:

| | status |
|---|---|
| requiring D0 | confirmed, t = 9.93 — but it is a *constraint*, not a gain |
| `--slope_weight 1.0` | **inside the seed band at n=3** |
| `--wt_anchor_weight 0.3` | n=1, and the weight sweep never bracketed its optimum |
| eleven other hypotheses | rejected |
| **W13 measured offset** | **+0.112, but few-shot, not zero-shot** |

**Confidence: 90%.** Three seeds, each at its own val-selected epoch, on the canonical basis.
The one caveat is that n=3 is small — but it is exactly the n that was demanded, and it answers
the question that was asked.
