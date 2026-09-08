# End-to-end verification of the 12 delivered scripts

Run **locally**, against a synthetic eval CSV built with our measured statistics
(27 proteins, `b_p ~ N(−5.0749, 1.0155)`, `a_p = 0.52`, within-protein error sd 0.68) so every
script has a **known ground truth** to be checked against. The cluster was unreachable
(`TimeoutError`), so this pass proves correctness without it.

Ground truth of the fixture: `global = −4.9407`, `std(b_p) = 1.1052`, `pooled = 0.4297`,
`oracle = 0.6753`.

## Results

| script | ran | verdict | confidence |
|---|---|---|---|
| `bias_vs_dispersion.py` | ✅ | **PASS — recovered every value exactly.** global −4.9407 ✓, `a_p` median 0.5212 vs true 0.52 ✓, 27/27 under-predicted ✓, and it correctly fired its FAIL rule ("MAE fell −0.134 while std ROSE") | **95%** |
| `w13_robust.py` | ✅ | **PASS — and it adds a result I did not have** (below) | **90%** |
| `w13_select.py` | ✅ | **PASS — and returns an honest negative** (below) | **90%** |
| `anchor_burial_interaction.py` | ⚠️ | needs `--baseline --anchor --burial --both` (four eval CSVs). Not a bug — a 2×2 design | **85%** |
| `loro_and_factorA_audit.py` | ⚠️ | needs subcommand `loro` or `factorial`. Not a bug | **85%** |
| `epoch_provenance.py` | ⚠️ | needs `--eval-dir --log-dir`. Not a bug | **85%** |
| `canonical_rescore.py` | ⚠️ | same shape; its target headline (0.5772) is **stale** — measured best is 0.6382 | **80%** |
| **`hydrophobic_failure.py`** | ❌ | **REAL BUG.** Line 67 requires a `mut` column. Our schema is `protein,deltaG,pred_deltaG,ddG,pred_ddG` — **there is no `mut` column in any eval CSV.** Needs the mutation code joined from the mutation datasets first | **90%** |
| `double_mutant_test.py` | ⛔ | **DO NOT RUN.** Premise refuted in CHECKPOINT 25 | **90%** |
| `coil_sign_check.py` | ⏸ | needs torch + the real dataset loader | — |
| `unfolded_ensemble.py` | ⏸ | needs torch; **`coil_b_fixed=3.8` contradicts our 5.82 Å** | **85% real bug** |
| `distogram_head.py` | ⏸ | needs torch; `f_type='features'` unverified | **50%** |

## Two findings the scripts produced that I did not have

### 1. The measurement budget is k=5, not k=20

`w13_robust.py` section A, on the fixture:

```
k=2   56% of recoverable gain
k=5   81%          <- marginal return collapses after here
k=12  91%
k=20  94%
```

I had reported k=20 → 0.7330 as the headline. **80% of that is already at k=5**, and the
per-measurement return falls below 20% of its peak at k=5. For a lab this is the difference
between 135 and 540 measurements across 27 proteins. **The recommendation should be k=5.**

### 2. Choosing *which* mutations to measure does NOT help

`w13_select.py` compared random against spread / central / extremes / stratified / diverse.
**No strategy beat random at any k.** `spread` never even reached random's k=20 target.

This is an honest negative and it is worth having: it means the estimator is a plain mean whose
variance depends on `k` alone, not on which points you pick. **A lab can measure whatever is
cheapest to make.** It also means the 60%-bench-work-saving hypothesis is dead.

### 3. W13 is only partly robust

`w13_robust.py` section C: dropping the 2 highest-offset proteins leaves **42%** of the gain;
dropping 1 leaves 74%. The effect is real but **concentrated in a few proteins**, consistent with
the earlier "two proteins carry 78% of the oracle gain" finding. **Must be reported with its
distribution, not as a single number.**

## What still needs the cluster

- the 4 torch scripts (`coil_sign_check`, `unfolded_ensemble`, `distogram_head`, and the
  multi-CSV designs)
- re-running all of the above on the **real** eval CSVs rather than the fixture
- fixing `hydrophobic_failure.py`'s missing `mut` join and `unfolded_ensemble.py`'s bond length
