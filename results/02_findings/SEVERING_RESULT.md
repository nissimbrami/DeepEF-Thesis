# SEVERING EXPERIMENT — RESULT

Run: SLURM job `21109361`, **CPU partition** (golden lane was full at 8/8 RUNNING; no job was
displaced and no PENDING job was cancelled). 28 proteins x 4 conditions = 112 forward passes,
frozen checkpoint, no training. Log: `logs/severing_21109361.out`.
Artifacts: `results/severing.json`, `results/severing_per_protein.csv`.

Checkpoint: `…kf_p3_a0_d0_s0_D0_coil_seed42/kf_all_epoch_9.pt` (all levers off — the correct
in-distribution baseline for C0). No `affine.json`, so MAE is raw; `f_resid` is a variance ratio
and is unaffected by an affine map.

---

## VERDICT LINE (first, as required)

```
VERDICT: UNINTERPRETABLE_OOD    f_resid = 0.8512
GUARD FIRED: range_collapse — C3 range(E_u) = 0.21x of C0 (4.187 -> 0.898), below 0.25
```

**The script's own verdict is `UNINTERPRETABLE_OOD`, and I am reporting it as such.** The
headline `f_resid` must not be used to cancel the reference-state work. Section 4 below explains
why the result is nevertheless informative, and section 5 why it is *not* the direction the
premise needed.

---

## 1. The number that had never been reported

| cond | cov(E_u,E_f) | **corr(E_u,E_f)** | var(E_u) | var(E_f) | var(dG) |
|---|---|---|---|---|---|
| **C0 base** | **0.2412** | **0.5430** | 0.8520 | 0.2316 | 0.6012 |
| C1 uemb=zero | 0.2043 | 0.7007 | 0.3670 | 0.2316 | 0.1900 |
| C2 flory+coil_b | 0.1247 | 0.3143 | 0.6793 | 0.2316 | 0.6616 |
| C3 full severing | 0.0816 | 0.6405 | 0.0700 | 0.2316 | 0.1385 |

**`cov(E_u, E_f) = 0.2412`, `corr(E_u, E_f) = 0.5430` at C0.** Bootstrap 95% CI on the
correlation `[0.294, 0.736]`, `P(corr > 0) = 0.9996` — comfortably past the n=28 noise floor of
|r| < 0.374.

`2*cov/(var_Eu + var_Ef) = 0.4452`: **45% of the two energies' combined variance is shared
between them** and cancels identically in `E_u - E_f`. The shared per-protein component the
premise could not rule out is real and is large.

The variance identity `var(dG) = var(E_u) + var(E_f) - 2cov` was verified independently from the
per-protein CSV in all four conditions, agreeing to **1e-16**.

## 2. The project's own headline, reproduced at C0

| project claim | this run at C0 |
|---|---|
| 77–88% of variance sits in E_u | `var_Eu/(var_Eu+var_Ef)` = **0.7862** — reproduced |
| corr(E_u, wt_err) = 0.865 | **0.4527** — same sign, **half the magnitude** |

The variance-share number reproduces almost exactly. The 0.865 correlation does **not**; this
checkpoint gives 0.4527. So the decomposition is confirmed as a description, and it is confirmed
to be exactly the ambiguous statistic the design flagged: a large `var(E_u)` share coexisting
with a large `cov(E_u,E_f)` is precisely the shared-bias signature.

## 3. Severing, condition by condition

| cond | var(wt_err) | MAE | std(E_u) | rng(E_u) | corr(pred, true) |
|---|---|---|---|---|---|
| C0 | 0.9964 | 5.0338 | 0.9400 | 4.187 | 0.3012 |
| C1 | 1.0098 | 3.5709 | 0.6170 | 2.224 | -0.0028 |
| C2 | 1.0242 | 4.1538 | 0.8393 | 3.542 | 0.3093 |
| C3 | 0.8481 | 2.6986 | 0.2695 | 0.898 | 0.1605 |

`f_resid = var(wt_err|C3)/var(wt_err|C0) = 0.8512`, bootstrap 95% CI **[0.486, 1.761]**.

`E_f` is bit-identical across all four conditions (`max|dE_f| = 0.000e+00`) — the levers are
unfolded-only, as designed, and the script's runtime lever assertions passed (C3 unfolded GAT
edge set collapses to the 2(N-1) chain).

## 4. Why the guard fired — and why it fired in the *harmless* direction

The guard exists because a degenerating `E_u` shrinks `var(wt_err)` and makes `f_resid` look
falsely wonderful. `range_collapse` did fire: C3's `E_u` range is 0.21x of C0's, and `var(E_u)`
falls 0.852 -> 0.070, a 92% collapse. This is the same effect W0 already measured (67% collapse
of `var(E_u)` under `unfolded_emb=zero`, FINDINGS Part VII); the D1 factorial arm independently
showed this lever destroys the trained model (a_p 0.005–0.020). The checkpoint is genuinely OOD
in C1/C2/C3. **The verdict stands as UNINTERPRETABLE_OOD.**

But note the direction. The failure mode the guard protects against is *`f_resid` falsely LOW*.
Here `E_u` collapsed by 92% and `f_resid` still came out **0.8512** — high. The OOD artefact
pushes `f_resid` **down**, and it did not get there. The bias in this measurement runs *against*
the conclusion drawn in section 5, which is what makes the run informative despite the verdict.

## 5. THE DEGENERATE BASELINE — the decisive control

Severing is supposed to be bounded by total severing. If `E_u` is replaced by a **constant**
(all reference-state information destroyed, the strongest possible severing), then
`wt_err = const - E_f - dG_true`, and:

```
var(wt_err | E_u = const)  = 1.0142
f_resid floor              = 1.0142 / 0.9964 = 1.0179
observed f_resid at C3     = 0.8512
```

**C3 sits essentially at the total-severing floor.** Severing the reference state three ways
removes ~15% of the across-protein error variance, and destroying it *completely* removes
**none** (slightly negative, 1.018). The across-protein error does not live in `E_u`.

Corroborating reads, all pointing the same way:

- `corr(wt_err|C0, wt_err|C3) = 0.6208` — the error largely **survives** severing, protein by protein.
- `wt_err` at C0 is barely explained by the energies at all: R² = 0.205 from `E_u`, 0.003 from
  `E_f`, 0.256 from both. It **is** explained by the target: R² = **0.4514 from `true` alone**.
- The C0 error is dominated by a **whole-dataset offset**, not a per-protein reference state:
  MAE 5.0338 with bias −5.0338 (every one of 28 proteins predicted low), while
  `std(wt_err) = 0.9982`. Mean-centred, C3 removes only 8% of the spread (ratio 0.9226).

## 6. Interpretation

Against the pre-registered bands (which apply only when the guard passes — it did not, so this
is stated as a direction of evidence, not a licence):

- `f_resid = 0.8512` is in the **> 0.7 "stop and redirect"** band, and its bootstrap CI puts
  `P(f_resid < 0.3) = 0.0002` — the "proceed" band is essentially excluded.
- The mechanism the band interpretation predicts for that outcome is present and measured:
  `corr(E_u,E_f) = 0.5430`, 45% of combined energy variance shared and cancelling in the
  difference. The design named this "the direct corroborating read", and it moved consistently
  with `f_resid`.
- The degenerate baseline (§5) is stronger evidence than `f_resid` itself, because it is immune
  to the OOD objection: it is an *arithmetic* bound computed from C0's own `E_f` and the true dG,
  involving no OOD forward pass at all. It says total reference-state severing cannot reduce the
  across-protein error variance. That is not an OOD artefact; it is algebra on in-distribution
  numbers.

**Bottom line.** The premise — that 77–88% of across-protein dG variance sitting in `E_u` implies
the reference state manufactures the offset — is **not supported**. The variance share reproduces
(0.7862), but it coexists with a large shared component (`corr(E_u,E_f) = 0.543`) that cancels in
the difference, and the error survives total severing. The C0 error is overwhelmingly a single
global bias (−5.03 kcal/mol on every protein) plus a component correlated with the *target*
(R² 0.45), not a per-protein reference-state artefact.

**This does not by itself cancel the ~150 GPU-hours, and I am not claiming it does** — the
formal verdict is `UNINTERPRETABLE_OOD` and the honest next step named in the design is a
*retrained* severed model, levers on from step 0. But the evidence available now points away
from the reference-state programme, and it does so in the direction the OOD bias runs *against*.

## 7. Recommended next step

Retrain one cell with C3's levers on from step 0 (one run, not 150 hours) and re-measure
`f_resid` in-distribution. That is the only measurement that can convert this into a verdict.
Before spending anything further on the reference state, the global −5.03 offset and the
`corr(E_u,E_f) = 0.543` shared bias are the larger and cheaper targets.

## 8. Safety

- Golden lane untouched: 8/8 RUNNING at submission; job went to `--partition=cpu --qos normal`.
- No job cancelled, no `git push`, nothing written under `shaharec/` or `shaharax`.
- `scripts/gate_g4_cpu.py` after the run: **`baseline … PASS dG=-0.0030 width=1092`**, G4-CPU ALL
  PASS. Unchanged. (`severing.py` is read-only and touches no feature-vector code.)
