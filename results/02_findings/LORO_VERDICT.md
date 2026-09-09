# K9 — the LORO descriptor verdict. The prediction holds, but the arm still did not train.

**The prediction was written and committed while the arm was still training**
(`LORO_PREDICTION.md`): *descriptors will not beat one-hot by more than +0.060.*

## The result

| run | pooled | PP | a_p | r | s | oracle |
|---|---|---|---|---|---|---|
| loroW_onehot_s42_e12 | 0.5716 | 0.7917 | 0.6887 | 0.7226 | 0.9268 | 0.6790 |
| loroW_onehot_s42_e14 | 0.5773 | 0.7895 | 0.5611 | 0.7235 | 0.7543 | 0.6793 |
| **loroWdesc2_s42_e11** | **−0.0026** | **0.0045** | **0.0006** | **−0.0153** | 0.2517 | 0.0126 |

```
one-hot mean (2 epochs):  0.5744
descriptors:             -0.0026
difference:              -0.5770
```

**PREDICTION HOLDS** — but by −0.577, not by a narrow margin. That is not "descriptors don't
help"; **it is a model that learned nothing at all.** `r = −0.015`, `a_p = 0.0006`: the
predictions are statistically independent of the labels.

## The diagnosis: this arm never trained, for the SECOND time

```
val ddG PCC by epoch:
  0.085  0.050  0.002 -0.137  0.095 -0.074 -0.024 -0.018
  0.051  0.066 -0.076  0.193 -0.154 -0.043 -0.036

RMSE: 2.541 2.541 2.541 2.541 2.541 2.541 2.541 2.541   (ONE distinct value in 15 epochs)
```

**RMSE is frozen at 2.541 across every epoch.** The loss did not move once. Meanwhile the one-hot
arm climbs cleanly: 0.548 → 0.641 → 0.674 → 0.695 → … → 0.712.

**The val PCC series is pure noise around zero**, and the "val argmax" at e11 (0.193) is simply
the largest random fluctuation — which is why the val-selected epoch scores −0.0026 on test.

## This is the same failure as the first attempt, not a fix of it

The first descriptor arm collapsed because the descriptors were centred but never scaled, carrying
~13.8× the one-hot block energy. `mordred_pca16_only` was supposed to fix that by **replacing**
one-hot instead of appending to it.

**It did not.** The identical signature — RMSE frozen at exactly 2.541 — appears in both.

**So the descriptor hypothesis has still never been tested.** Two arms, two technical collapses,
zero measurements of the actual question.

## What the prediction got right, and what it got wrong

**Right:** descriptors do not beat one-hot. Confirmed, and the reasoning was sound — ProtT5
already predicts held-out-residue hydropathy at R² = 0.704, so the information is largely
redundant.

**Wrong in emphasis:** the prediction anticipated a *close* result inside the noise band. The
actual outcome is a **non-result** — the arm is uninformative, not merely unhelpful. **A
prediction that is confirmed by a broken experiment is not confirmed at all.**

## Honest status

**The descriptor / Ofir-thesis idea remains UNTESTED after two attempts.** The blocker is
technical: something about how the descriptor block enters the model prevents training entirely,
and it survived the fix that was supposed to address it.

**Not recommending a third attempt without first diagnosing why the loss cannot move** — that is a
CPU-scale question (check gradients reach the descriptor path, check the block is not all-constant
after the PCA, check for NaN masking) and should be answered before spending another 12 GPU-hours.

**Confidence: 95%** that the arm did not train (RMSE constant to four decimals over 15 epochs);
**90%** that the descriptor hypothesis is still untested.

---

# DIAGNOSIS — the fix existed, and the arm was launched with the wrong file

Two descriptor tables sit in the tree, both clean (20×16, no NaN, no zero-variance columns):

| file | median row L2 | vs one-hot |
|---|---|---|
| `data/aa_descriptors_mordred_pca16.csv` | 3.653 | **3.65×** |
| `data_fixed/aa_descriptors_mordred_pca16_unit.csv` | 1.045 | **1.04×** |

**`mordred_pca16_only` loads the file its name states — the 3.65× one.**

The unit-normalised table, built specifically to fix the scale clash, **was never used.** It sits
in `data_fixed/` with a median L2 of 1.045 — the correct scale — and no arm has ever loaded it.

## So the second collapse has the same cause as the first

The first arm died at **13.8×** the one-hot energy. `mordred_pca16_only` was supposed to fix this
by replacing one-hot rather than appending. It is **3.65×** — better, still wrong, still enough to
freeze the loss at RMSE 2.541 for fifteen epochs.

**The fix was built and then not wired to the run.** That is the same failure mode as the three
scoring bugs: the correction existed, looked applied, and was not in the path that executed.

## What a third attempt would need

Not a code change — a **flag pointing at the right file**. `mordred_pca16_only` hard-codes its
table by name, so the unit variant needs either its own mode or a `--aa_descriptors_file`
override.

**Cost: one 12-hour GPU arm. Expected outcome, stated in advance:** with the block at 1.04× the
one-hot energy the arm should now *train* — so we would finally get a real measurement of the
descriptor hypothesis instead of a third technical collapse.

**But the prediction on the hypothesis itself does not change:** ProtT5 already predicts
held-out-residue hydropathy at R² = 0.704, so descriptors are largely redundant and should land
inside the seed band. **The value of a third attempt is that it would produce a measurement rather
than a non-result.**

**Confidence: 90%** that the scale is the cause (two arms, two frozen losses, both at >3× one-hot,
and a correctly-scaled table exists unused); **not recommending the run without your approval**,
since it costs a golden card while W15, K21 and K20 have still never started.
