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

---

# ADDENDUM — 2026-09-10. What the checklist run changed.

Six lessons are sharpened and three are new. Each cites the measurement that changed it.

## Sharpened

**9 (W12) is now MECHANISTIC, not just statistical.** `E8_W12_MECHANISM.md`: the gain is 10x
larger in tightly packed proteins (`corr(gain, packing_frac) = +0.4126`, predicted in advance by
T4) and **8.5x larger for hydrophobic destinations** (+0.02894 vs +0.00341), with the hydrophobic
median positive and the polar median negative. **W12 repairs the §3.4 hydrophobic-burial deficit,
which is what it was built to do.** But `E7_W12_CANONICAL.md`: pooled is only **+1.33 sd** — an
information gain **without a headline number**.

**11 (W5's sign flip) now has a mechanism, and it is the wrong one.** `B3B_BURIED_VS_EXPOSED.md`:
W5's gain is larger at buried positions (+0.0265, p=1.1e-06) — but corr(burial, gain) = +0.065
explains **0.4% of the variance**, and it helps **POLAR destinations more than hydrophobic**
(+0.047 vs +0.024). **The opposite of desolvation.** Direct contrast with lesson 9: both levers
were built to attack burial; only W12's measured gain matches its intended mechanism.

**14 (connectivity levers are null) has a CAUSE.** `T3_KERNEL_AND_SPAN.md`: the Gaussian kernel
`exp(-0.08 d²)` is strictly monotone but **saturates** — k(15 Å) = 1.5e-08, below float32 eps.
Residues at |i−j| ≤ 4 sit 5–13 Å apart, where the kernel has decayed 2–4 orders of magnitude.
**The nulls are real but they are nulls FOR THIS KERNEL WIDTH.** The actionable lever is
`gaussian_coef`, not `gcn_span`. Also: the claim that `span4` confounds range with directionality
is **wrong** — `--gcn_bidir` is gated independently and `u10bidir` is the control, also null.

**18 (descriptors collapse) has a PROVED root cause.** `W6_ROOT_CAUSE.md`: the block is
**bit-identical** in the folded and unfolded passes (`torch.equal` → True, max|diff| = 0.000e+00),
so it contributes exactly zero to `dG = E_f − E_u` and has no gradient path. Train loss is constant
to five decimals for 15 epochs. Confirmed empirically: `descunit` scores mean r **−0.0191**
(−115 control-sd, 0/27 proteins). **Now enforced by `scripts/gate_state_dependence.py`** — W5, W12
and W15 pass; both descriptor modes FAIL.

**19 (Flory) is settled on the optimisation question and NARROWED on the idea.** Both arms
**trained healthily** (RMSE 1.450→1.086 and 2.313→1.628), so it is not fragility. `B1C_RESULT.md`:
the block magnitude ratio is 1.45 — **not a scale problem**, so recalibrating `b` is ruled out.
The real difference is **sparsity**: the baseline is a hard tridiagonal mask (3.3% of pairs nonzero
at 0.3150); the coil smears the same mass over 39% of pairs at 0.0391 each — **12x denser, 8x
weaker per contact.** And `FLORY_QUARTER_TESTED.md`: only the input map was swapped
(*"value-only lever, no new parameters"*), so **a quarter of IFUM's method was tested.**

**16 (the oracle is not reproducible) has a second instance.** `E7_W12_CANONICAL.md`: recomputing
the declared 9-CSV canonical population gives oracle **0.7327 ± 0.0126** against the recorded
**0.7156 ± 0.0474** — a **4x sd discrepancy**, not rounding. Flagged, not fixed.

## New

**20. Every mechanism with signal in this project is MULTIPLICATIVE.**
`B2D_EDGE_DESCRIPTORS.md` + `W15_FIRST_RESULT.md`:

| lever | form | mean-r |
|---|---|---|
| W12 | geometric, state-dependent | **+3.19 sd, established** |
| W15 | `reach × env` + clash | **+2.41 sd, screen (n=1)** |
| W5 | `bur × hyd` | +1.00 mean / **−4.15 median** |
| W6 | `one_hot @ T` (linear) | **collapsed** |
| w7edge / span4 / u10bidir | linear / topological | null |

**The three multiplicative levers are the three with positive mean-r; every linear one is null or
collapsed.** Measured, not asserted: on an edge, hydropathy *difference* and volume *sum* are fit
at **R² = 1.0000** from the existing `edge_attr` (so `lin_edge` absorbs any table `T`), while the
*product* `h_src × h_dst` reaches only **R² = 0.1970**.

**21. A flag that parses is not a flag that runs — five times now.**
`--distogram_weight` was parsed, stored and logged but **never read by any loss**
(`B1D_DISTOGRAM_NOOP.md`). Proof at two levels: `DistogramHead` absent from the live model
(grep = 0 in three files), and empirically **a 100x weight sweep produced a pooled spread of
0.0134 = 0.43 control-sd**, smaller than the seed noise — three arms indistinguishable from three
replicate baselines. **Cost: ~32 GPU-hours training a baseline under four names.**
This joins the four in §5.5, the three-part scoring bug, the `evaluate.py` graph flags, and W6.

**22. Interface size does not predict the offset — but PACKING predicts the slope.**
`T4_INTERFACE_NULL.md`: no interface or SASA feature reaches significance against `b_p` over 201
tests (best `n_hbond` −0.446, p=0.020, vs a Bonferroni threshold of 0.00025). **`b_p` is now
unpredictable by ten independent methods.** But **`packing_frac` vs `a_p` = −0.662 (p=0.0002)** and
`void_vol_per_res` = +0.652 **both survive Bonferroni by ~100x**. **Tightly packed proteins compress
hardest** — the same axis as §3.5 from independent features, and the structural signature of the
§3.4 deficit. This is what made lesson 9's mechanism a *prediction* rather than a story.
