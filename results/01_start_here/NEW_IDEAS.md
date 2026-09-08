# Ideas not previously in TASKS.md

Added 2026-09-08 after the ν sweep closed and the IFUM comparison was checked against our code.
**Every architectural claim below was verified by reading `train_utils.py` / `model/hydro_net.py`,
not assumed.** Ordered by (measured justification) ÷ (cost).

---

## Correction to the IFUM analysis: our pair structure is NOT collapsed

The second agent's strongest architectural claim was that `D.sum(dim=1)` erases pair structure
before the network sees anything, so we have no pair representation at all. **That is half wrong,
and the half that is wrong changes the recommendation.**

`train_utils.py:606-611` actually does:

```python
D = zero_except_udiagonal(D)         # N,N,16
Fb = get_bonded_features(D)          # N,32   <-- PAIR INFO SURVIVES HERE
D  = D.sum(dim=1)                    # N,16   <-- only THIS branch is summed
```

**`get_bonded_features` reads the full `[N,N,16]` tensor and produces the 32-wide `Fb` block that
is the first 32 columns of the feature vector — the only columns the GCN branch reads.** So pair
geometry is not destroyed; it is *summarised* into `Fb` before reduction. A distogram head does
not have to reconstruct pair structure from scratch: **it can attach to the tensor that already
exists at line 607, before the sum.** This makes T-E2 considerably cheaper than argued — and it
means the honest framing is "the pair tensor is discarded after summarising", not "absent".

---

## T-E1 — Unfolded ENSEMBLE (mean-field vs true free energy)

**The real difference between us and IFUM**, and the one thing the ν sweep did *not* test.
Our unfolded state is **one deterministic distance map**. The physical unfolded state is an
**ensemble**. `E_u` should be a thermodynamic average over conformations, not a point estimate.

**Verified cheap:** `_flory_unfolded_graph` builds `sep = |i−j|` and `b` from CA coords only —
**it never reads `one_hot`.** The coil map is therefore *identical for every variant of a protein*,
so k conformations can be sampled **once per protein and cached**, and reused across all ~1000
mutants. Cost is k forward passes on the unfolded half only, not k full trainings.

**Sample coordinates, never distances.** Drawing each `d(i,j)` independently produces a matrix
that corresponds to no 3-D structure and violates the triangle inequality. Generate a chain with
the right step statistics, then compute its distance matrix — geometric consistency is free.

**Run BOTH reductions, because the difference is the physics:**
- `E_u = mean_k E_k` — mean field
- `E_u = −log Σ_k exp(−E_k)` — the true free energy

**Their difference IS the conformational entropy**, which is exactly what dominates the unfolded
state. Testing only the mean would miss the term we care about.

**Metric:** std(b_p), not MAE — per NU_SWEEP.md. **Degenerate check:** report std(pred WT); if it
collapses toward 0.9239 the arm is deleting the prediction.
**Falsifier stated in advance:** if k=1 → k=8 does not reduce std(b_p) below 0.9972, the
reference-state programme is finished and we stop paying for it.

---

## T-E2 — Distogram head on the FOLDED state only

Framed as **pair-information preservation**, not as an IFUM port.

**Do not put a distogram head on the unfolded state.** The coil map is an analytic function of
`|i−j|` alone (verified above) — predicting it is nearly free and teaches nothing. On the folded
state the target is real structure, so the task is real.

**Justification stronger than IFUM's:** the sum at line 610 discards the pair tensor after `Fb`
summarises it. An auxiliary head at line 607 forces the node representations to retain what the
sum throws away. Attach `MLP([h_i, h_j, |i−j|]) → distance bins`, auxiliary loss, small weight.

**Only after T-E1.** Higher cost, weaker prior.

---

## T-E3 — ONE arm that isolates bias from dispersion  ★ cheapest real test here

The sweep produced the cleanest fact of the day: **`n_under = 28/28` in all 15 conditions.**
Every protein under-predicted, always. That is not noise — it is a **systematic sign error
in the reference state**, and it means `b_p` has a large *common* component plus a *spread*.

**MAE only ever saw the common part; the thesis only cares about the spread.**

**Test:** subtract the global mean offset from all predictions, then recompute everything.
Pure post-processing — **zero GPU, minutes on CPU**. It answers: what fraction of `b_p` is one
global constant (free to remove, worth nothing) versus genuinely per-protein (the real target)?
**This should run before any further reference-state work**, because if the spread is small the
whole programme is mis-aimed.

---

## T-E4 — Is the coil failure a SIGN error, not a shape error?

Follows directly from 28/28. The coil makes `E_u` too *low* for every protein, so `dG = E_u − E_f`
is systematically small. We have swept ν (shape) and b (scale) and both fail. **We have never
checked the sign/offset convention of the coil energy itself.**

**Cheap diagnostic:** correlate the per-protein coil-vs-base `ΔE_u` against N, mean burial, and
contact order. If it tracks N, the coil is mis-normalised per length — a bug, not physics, and
one that would masquerade as "the coil idea fails". CPU, ~1 h.

---

## T-E5 — Double mutants as the held-out generalisation test (was K11)

**26,315 double mutants for our 28 proteins sit in NO split** — not train, not val, not test.
They are free, already-measured data. A model calibrated on singles that transfers to doubles is
a far stronger thesis claim than any within-split number, and **it is the only truly held-out
data we have.** No training required: score the existing checkpoints.

**Why it belongs here:** if `b_p` is a per-protein constant, it must be *the same constant* for
doubles. This is the sharpest available test of what `b_p` actually is — and eight explanations
have already failed.
