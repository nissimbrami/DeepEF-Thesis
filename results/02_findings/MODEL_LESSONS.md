# What we learned about the MODEL — the 19 lessons that survive when a lever falls

From Part F of the 2026-09-10 review. **Not "did the lever work" but "what does this tell us about
the model".** This is the part that keeps its value when an arm is rejected.

## On the structure of the error

**1. Training directly on ddG: 0.531 -> 0.591.** Absolute scale was never the bottleneck for
ranking. **The objective simply did not match the metric.** Same model, same data, different loss.

**2. The offset is 96.3% one global constant.** The model is uniformly biased, with a small
per-protein spread on top. **And the uniform part is irrelevant to any correlation** — Pearson is
shift-invariant. What looked like a size-5 problem is really a size-1 problem.

**3. The offset is unpredictable: 9 methods, 22 features, zero positive out-of-sample R².**
The per-protein error is **not a function of structural properties the model can see.** It is not
a missing feature — it is not there. (Now ten methods: `T4_INTERFACE_NULL.md` adds interface/SASA.)

**4. Two proteins carry 78% of the offset-removal gain.** The effect is **concentrated, not
distributed.** Any headline built on it is fragile.

## On the compression

**5. The compression is real, and was refuted BY SIGN.** Attenuation predicts r = −0.39; measured
**+0.71**; 0 of 2000 simulations reached the measured value. **A genuine model failure, not a
measurement artifact.** A sign refutation cannot be rescued by re-tuning a parameter, which is why
it is the strongest methodological result in the project.

**6. The compression is concentrated at buried positions and tracks hydrophobicity.** r = −0.765;
slope gap 0.095 buried vs 0.024 exposed. **The failure is specifically desolvation** — the model
does not understand the cost of burying a greasy residue.

**7. The volume hypothesis is refuted.** Volume alone gives R² = 0.031, and W/Y sit *inside* the
hydrophobic pack, not outside. **It is hydrophobicity, not physical size.**

**8. `slope_weight` raises the slope 0.499 -> 0.775 and does NOT raise pooled.** **Fixing the
compression does not fix the score.** They are two partly-overlapping problems, not one.

## On what does add information

**9. W12: +3.19σ across four conventions, on the ranking channel.** **Geometric, state-dependent
side-chain information adds real ranking ability** — something no calibration can fake.

**10. corr(Δr, control_r) = −0.856.** **The model's failure is concentrated in specific proteins,
and it is fixable.** W12 does not lift everything slightly; it repairs what fails. 2KVS goes
0.160 -> 0.471.

**11. W5 flips sign between mean and median.** **A feature can help the tail and hurt the majority
at the same time.** One summary number hides it.

**12. W13: measuring 20 mutations gives +0.11.** **The ranking information is present in the
model.** What is missing is a per-protein anchor — unpredictable, but measurable.

**13. At k=1 measurement HURTS.** Measured signal-to-noise floor: within-protein error 0.68 vs
between-protein signal 0.46. **An estimator that is too noisy is worse than no estimator.**

## On what does not

**14. Every connectivity lever is null** — edges, span4, bidirectional, fully-connected.
**Added capacity is not the bottleneck.** (Now sharpened: `T3_KERNEL_AND_SPAN.md` shows the
Gaussian kernel saturates past 12 Å, so span-4 edges carry almost no distance signal. The nulls
are real, but they are nulls *for this kernel width* — the actionable lever is `gaussian_coef`.)

**15. Pre-training: 0.82 in validation, 0.07 out of distribution.** **Memorisation of fold
families.** Excellent inside the distribution, near-random outside it.

**16. The oracle ceiling is not reproducible, and the estimator is not monotone.** **The metric
the project leaned on did not measure what it was thought to.** Removing two parameters gives less
than removing one, because dividing by a collapsed slope amplifies noise.

**17. The 2K5H bug explained 36% of the apparent improvement.** **The evaluation set itself was
faulty**, and every conclusion built on it was partly an artifact.

## On how the architecture behaves

**18. Descriptors collapse to a frozen RMSE, worse than predicting a constant.** **The difference
structure `E_folded − E_unfolded` filters what can enter at all.** A feature identical in both
states does not merely fail to help — it injects subtractive noise and destabilises the
optimisation. (Root cause proved: `W6_ROOT_CAUSE.md`; now enforced by `gate_state_dependence.py`.)

**19. Flory takes the model to near-random in two configurations.** Either the model is **fragile
to input scale**, or the reference state matters far more than assumed. **The loss curve decides
between them** — and it has now been read: **both arms trained healthily**, so this is not an
optimisation failure. Combined with lesson 18's rule, the coil map also cancels in ddG because it
reads only `|i−j|`. See `FLORY_QUARTER_TESTED.md`.

---

**Why this file exists:** twelve arms produced no surviving positive result, but the list above is
what the project actually knows. **A rejected lever still teaches something about the model, and
that is the part worth writing down.**
