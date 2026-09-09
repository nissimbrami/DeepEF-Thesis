# `hydrophobic_failure.py` cannot run on our eval CSVs — and the reason is structural

## What it needs

Line 67: `parsed = df["mut"].apply(parse_mut)` — it groups mutations by the residue they
introduce, to test whether the model compresses hydrophobic destinations harder.

## Why it cannot get it

Our eval schema is:

```
protein, deltaG, pred_deltaG, ddG, pred_ddG
```

**There is no `mut` column, and no variant identifier of any kind.** Three routes were tested:

| route | result |
|---|---|
| join on a `mut` column | **absent from every eval CSV** |
| recover the mutation from `one_hot` vs WT | **works** — variant 5 of 2PTL decodes cleanly as `E1Q` |
| join by row order (row *i* = variant *i*) | **FAILS** — 2K28 has **920 CSV rows vs 2,838 variants** |

The mutation IS recoverable from the tensors. What is missing is the **link** between an eval row
and a variant index: the evaluation writes a filtered subset (920 of 2,838) and does not record
which subset. **The join key does not exist anywhere.**

## Verdict

**Not fixable by patching the script.** It needs either

1. `evaluate.py` modified to emit a variant index or mutation code alongside each prediction, and
   every eval CSV regenerated — ~52 files, hours of CPU; or
2. the per-mutation analysis recomputed inside the evaluation loop, where the variant index is
   still in scope.

**Neither is a small fix, and neither is the highest-value thing available**, so this is recorded
as blocked rather than attempted.

## What is NOT lost

The finding this script was written to re-derive is **already measured and on record**: the model
compresses mutations to hydrophobic destinations about twice as hard (slope 0.27–0.31 vs
0.53–0.57), the ranking degrades in lockstep (corr with Kyte-Doolittle −0.734 and −0.740), and the
deficit is worst at buried positions (gap 0.095 buried vs 0.024 exposed, permutation p < 0.002).

That earlier analysis ran **inside** the evaluation, where the mutation identity was available.
`hydrophobic_failure.py` was an attempt to redo it from the outside, and the outside does not have
the data.

**Confidence this is genuinely blocked, not a missed workaround: 90%** — three independent join
routes were tested against real files.

## Recommendation

If the per-destination-residue breakdown is wanted again for the thesis, add the variant index to
`evaluate.py` **once**, then every future analysis of this kind becomes trivial. That is a
15-minute change with a large payoff, but it invalidates nothing already measured.
