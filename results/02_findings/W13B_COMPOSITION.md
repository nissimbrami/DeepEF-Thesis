# W13b — the measured offset does NOT compose with the slope lever

Zero GPU. 27 proteins, 2K5H dropped, ddG, 20 random draws per k.

## Result

| arm | a_p | std(b_p) | k=0 | k=3 | k=10 | **k=20** | gain |
|---|---|---|---|---|---|---|---|
| control | 0.521 | 0.516 | 0.5635 | 0.6496 | 0.7053 | **0.7227** | +0.159 |
| slope (canonical e10) | **0.782** | 0.577 | 0.5798 | 0.6672 | 0.7034 | **0.7106** | +0.131 |
| slope (e13) | 0.591 | 0.462 | 0.6210 | 0.6776 | 0.7233 | **0.7337** | +0.113 |
| best pooled (sigma_seed2) | 0.448 | 0.418 | **0.6382** | 0.6498 | 0.7171 | **0.7336** | +0.095 |

## The finding: the arms CONVERGE once the offset is measured

Zero-shot the arms span **0.5635 → 0.6382** (a range of 0.075).
At k=20 they span **0.7106 → 0.7337** (a range of 0.023) — and the ordering scrambles.

**The arm with the best slope (a_p = 0.782) ends up LAST at k=20.** The arm with the worst
`a_p` (0.448) ties for first. Whatever the slope lever bought is largely re-bought by measuring
the offset, so the two do not add.

**Interpretation:** `--slope_weight` improves the zero-shot number partly by *incidentally*
reducing the damage the offset does. Once the offset is measured directly, that contribution is
redundant. **They are not independent levers acting on independent terms, as I expected — they
overlap.**

## But the ceiling is NOT a flat constant — I checked, and my first reading was wrong

The four-arm table above suggested every arm lands at the same ceiling. Across **all 35 scored
runs with pooled > 0.4** that is not what happens:

```
zero-shot pooled : mean 0.5830  sd 0.0237  range 0.5362-0.6382
oracle (offset removed) : mean 0.7212  sd 0.0261  range 0.5929-0.7534
spread: sd 0.0237 -> 0.0261   (it does NOT collapse; it slightly widens)
corr(zero-shot, oracle) = +0.548
```

**The oracle spread is as large as the zero-shot spread.** So the model still matters after the
offset is removed — a better arm still has a better ceiling. The convergence in the four-arm table
is real but local: those four arms happen to sit close together in oracle terms (0.688–0.713).

**Correction to what the table alone would suggest:** it is NOT true that "any model reaches the
same place once calibrated". The correlation between zero-shot and oracle is only +0.548, which
means **zero-shot ranking is a poor predictor of post-calibration ranking** — but not a useless one.

## Consequences for the thesis

1. **Do not report slope + W13 as additive.** Measured: they overlap.
2. **Model selection should be done on the ORACLE, not on zero-shot pooled**, if the intended
   deployment measures a few mutations. The two rankings disagree (r = +0.548), and the arm we
   would pick zero-shot is not the arm that wins at k=20.
3. The best k=20 result so far is **0.7337**, from `p3_slope1.0_s42_e13`.

## Method note

My first pass read the four-arm table as "the ceiling is a constant". Checking all 35 runs showed
that was an artifact of picking four similar arms. **The wider check contradicted the narrow one,
and the wider one is right.**
