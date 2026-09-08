# ν sweep — the Flory exponent is NOT the problem

**Job 21137926, CPU, frozen checkpoint `calib_ctrl_repro2/kf_all_epoch_14.pt`.
28 test proteins, 2K5H reference corrected to 4.80547. Raw-2K5H run confirms every sign.**

## The question

The coil sets `d(i,j) = b·|i−j|^ν`. Every run ever done used **ν = 0.5**, the *theta state* —
a chain whose self-repulsion exactly cancels solvent attraction. A denatured protein in water is
a **self-avoiding walk at ν ≈ 0.588** (Kohn 2004 PNAS: Rg ∝ N^0.598 over 28 proteins). At
|i−j| = 50 that is a **41 % distance error**. The hypothesis: the coil failed because it was
given the wrong physics, not because the idea is wrong.

## The result

**Scored on std(b_p) — the between-protein offset spread the thesis targets.**
`std(true WT dG) = 0.9239` is the degenerate attractor: any arm collapsing toward it is deleting
the prediction, not improving it.

| condition | MAE | std(b_p) | corr | std(pred WT) |
|---|---|---|---|---|
| **base (no coil)** | 5.0749 | **0.9972** | 0.4104 | 0.9125 |
| ν=0.500 b=5.82 *(every run to date)* | 4.0621 | 1.1233 | 0.3554 | 1.0467 |
| ν=0.550 b=5.82 | 4.0091 | 1.1066 | 0.3512 | 1.0146 |
| **ν=0.588 b=4.97** *(correct physics)* | 3.9949 | **1.1043** | 0.3507 | 1.0102 |
| ν=0.620 b=4.97 | 3.9763 | **1.1036** ← best | 0.3459 | 1.0026 |
| ν=0.650 b=4.97 | 3.9748 | 1.1118 | 0.3399 | 1.0075 |

## The verdict: rejected, and the rejection is clean

**Every one of the ten ν×b cells is WORSE than no coil at all on std(b_p).**
Best cell 1.1036 vs base 0.9972 — **+10.7 % worse**. Correct physics (0.588) does not rescue it.

**The ν direction is real but tiny and in the wrong place.** Going 0.5 → 0.62 improves std(b_p)
by 0.0416 — real, monotone, but it is *recovering part of the damage the coil itself caused*.
It never gets back to baseline. **ν was a genuine bug; fixing it does not make the coil useful.**

### And this is why MAE misled us for weeks

MAE says the coil is a **20 % win** (5.07 → 3.99). It is not.

`n_under = 28/28` in **every single condition**. All 28 proteins are under-predicted, so
**MAE ≡ |mean(b_p)|** — it measures *bias*, a uniform shift. **Pearson correlation is exactly
invariant to a uniform shift**, so a pure bias reduction cannot improve any correlation score.
Meanwhile the quantity that *does* matter, the dispersion std(b_p), got **worse**.

> **The coil moved all 28 predictions closer to the truth together, without moving them closer
> to each other.** That is the whole story, and it is why "−20 % error" was never a real gain.

## Status

**CLOSED.** ν is measured, not assumed. Do not re-run the ν sweep. The remaining open question
is the *ensemble* (T-E1), which is a different mechanism: not the coil's shape, but that a single
deterministic map replaces a thermodynamic average.
