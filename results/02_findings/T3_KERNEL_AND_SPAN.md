# T3 — the distance kernel is monotone, and the `span4` confound claim is WRONG

**Date:** 2026-09-10. Task T3 of `CHECKLIST.md`, run under `PROTOCOL.md`.

## PLAN (registered before execution)

**Q:** Is the RBF/Gaussian distance kernel monotone in the runs the W7 arms actually used?
**Falsifier:** if `k(2A) > k(8A) > k(15A)` fails, the model is distance-blind and both W7 nulls
are uninterpretable.
**If X then Y:** monotone -> the W7 nulls stand on merit; flat -> the nulls are void.

## EXECUTE

The kernel, read from `train_utils.py:390` and `:652` (identical in the folded and unfolded paths):

```python
D = torch.relu(torch.exp(gaussian_coef * D**2))    # gaussian_coef = -0.08
```

| d (A) | k(d) | |
|---|---|---|
| 2 | 7.261e-01 | |
| 5 | 1.353e-01 | |
| 8 | 5.976e-03 | 0.82% of k(2) |
| 10 | 3.355e-04 | |
| 12 | 9.930e-06 | |
| 15 | 1.523e-08 | **below float32 eps (1.2e-07)** |
| 20 | 1.266e-14 | below float32 eps |

## VERIFY — the falsifier did NOT fire

**The kernel is strictly decreasing over 2-20 A.** The concern that contact at 5 A and non-contact
at 15 A both return 4.649 does **not** reproduce against the kernel these runs used.

**But it saturates.** Past ~12 A the value is at the edge of float32 resolution, and past 15 A it
is numerically zero. So the model resolves distance well inside ~10 A and not at all beyond it.

**Consequence for `--gcn_span 4`:** the flag adds chain edges at |i-j| <= 4. In a folded domain
those residues are typically 5-13 A apart, i.e. in the region where the kernel has already decayed
by two to four orders of magnitude. **The null result is consistent with the added edges carrying
almost no distance signal** — the arm is not broken, it is attenuated by the kernel width.

**This is a real, actionable finding: the lever to test is `gaussian_coef`, not `gcn_span`.**
A coefficient of -0.01 would put k(15 A) at 0.105 instead of 1.5e-08.

## CORRECTION to the review — `span4` is NOT confounded

The review states that `span4` "changes both range and directionality at once, so its null is
unattributable." **Read against the source, that is wrong.** `model/hydro_net.py:720-722`:

> *"U10: at span 1 the baseline is directed. Making all offsets bidirectional while extending the
> span would change TWO things at once, so `--gcn_bidir` is its own flag and can be set at span 1
> to give an honest control arm."*

and the branch confirms it:

```python
src_parts.append(a); dst_parts.append(b)
if bidir:                                   # <- gated INDEPENDENTLY of span
    src_parts.append(b); dst_parts.append(a)
```

**`span` and `bidir` are separate flags.** `gld_w7span4_*` ran with `--gcn_span 4` and no
`--gcn_bidir`, so only the range changed. The confound was anticipated and designed out.

**The requested "bidirectional span-1 control" already exists and has already run:** that is
`u10bidir` (`--gcn_bidir` at span 1), which is also null (8/27 improved, Wilcoxon p=0.0521).

## DONE

| arm | mean r | median r | training health |
|---|---|---|---|
| w7span4 s1 | 0.7319 | 0.7799 | trained (loss falling) |
| w7span4 s42 | 0.7295 | 0.7874 | trained |
| u10bidir s1 | 0.7231 | 0.7714 | trained |
| u10bidir s42 | 0.7358 | 0.7882 | trained |
| control | 0.7273 | 0.7897 | — |

**The W7/U10 nulls stand.** They are genuine nulls, not measurement artifacts — but they are nulls
*for the kernel width in use*, which is the finding worth carrying forward.

**Cost: ~15 minutes, no GPU. Confidence: 97%** — the kernel is arithmetic on a formula read from
the source; the flag independence is read from the branch itself.
