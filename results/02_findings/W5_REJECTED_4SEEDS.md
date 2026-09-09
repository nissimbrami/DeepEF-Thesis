# W5 (burial on dG) is REJECTED at four seeds — it helps the average protein by hurting most of them

**Date:** 2026-09-10. `w5_dg_s3_e12` landed, giving four independent seeds. This closes W5.

## 1. The four seeds, on every convention

| arm | pooled | per-prot MEAN | per-prot MEDIAN | a_p |
|---|---|---|---|---|
| control (6 seeds) | 0.5757 ± 0.0314 | 0.7273 ± 0.0065 | 0.7897 ± 0.0024 | 0.398 ± 0.062 |
| w5 s42 | 0.5956 | 0.7299 | 0.7790 | 0.664 |
| w5 s1 | 0.6060 | 0.7338 | 0.7788 | 0.507 |
| w5 s2 | 0.5948 | 0.7374 | 0.7807 | 0.543 |
| w5 s3 | 0.5884 | 0.7295 | 0.7715 | 0.561 |

| metric | W5 mean (4 seeds) | gain | in control-sd |
|---|---|---|---|
| pooled | 0.5962 | +0.0205 | +0.65 |
| per-protein **mean** | 0.7327 | +0.0054 | **+0.84** |
| per-protein **MEDIAN** | 0.7775 | −0.0122 | **−4.97** |
| a_p | 0.5689 | +0.1705 | +2.74 |

**The sign flip replicates across all four seeds.** It is not an artifact of one run: every single
seed has a median below the control's, and the W5 seed sd on the median is only 0.0041.

## 2. The mechanism, measured at n=27 proteins

Against the 6-seed control mean, per protein:

    improved: 12 of 27          mean delta +0.0054     MEDIAN delta -0.0067
    corr(control r, W5 delta) = -0.815

| protein | control r | W5 r | delta |
|---|---|---|---|
| 2KVS | 0.231 | 0.442 | **+0.211** |
| HEEH_KT_rd6_0793 | 0.375 | 0.490 | **+0.115** |
| 1QKH | 0.532 | 0.615 | +0.083 |
| ... | | | |
| 2K28 | 0.838 | 0.790 | −0.048 |
| 3DKM | 0.641 | 0.564 | −0.077 |
| 2KXD | 0.811 | 0.733 | −0.078 |

**The top 2 proteins contribute +0.3264 while the other 25 sum to −0.1800.**

`corr(control r, delta) = -0.815` says it plainly: **W5 helps exactly the proteins the baseline
already fails on, and damages the ones it handles well.** The mean is positive only because two
rescues outweigh fifteen regressions.

## 3. Why this rejects W5 but NOT W12 — the same mechanism, opposite verdict

| | improved | mean | **median** | corr(ctrl r, delta) | worst hurt |
|---|---|---|---|---|---|
| **W5** | 12/27 | +0.0054 | **−0.0067** | −0.815 | −0.078 |
| **W12** | **18/27** | **+0.0207** | **+0.0080** | −0.827 | **−0.017** |

Both levers concentrate on hard proteins — the correlations are nearly identical (−0.815 vs
−0.827). **The difference is the cost.** W12 improves a majority (18/27), its median is positive,
and its worst regression is −0.017. W5 improves a minority (12/27), its median is negative, and it
costs up to −0.078 on a protein the baseline scored 0.811.

**W12 rescues hard proteins nearly for free. W5 trades good proteins for hard ones.**

## 4. Verdict

**W5 is rejected.** Combined with `W5_DG_SEED2.md` (the published a_p 0.667 was one favourable
seed; four seeds give 0.664/0.507/0.543/0.561), W5 should not be reported as a working lever.

Its one honest positive is **a_p +2.74 sd** — it does decompress the predictions. But it buys that
by degrading the typical protein's ranking, and `a_p` is not worth ranking quality.

**Note this also settles the planner's chemistry proposal.** W5 *is* `torch.cat([bur, hyd,
bur*hyd])` — the hydrophobicity x burial product. The product term works exactly as the algebra
predicted it could (it is not expressible as `one_hot @ T`), and it still fails on the metric that
matters. **The idea was right; the result is negative.** That is a genuine tested refutation, and
worth more than the argument it replaces.

## 5. Rule reinforced

**Report mean AND median for every per-protein number.** W5 passed for weeks on the mean alone.
A lever whose mean and median disagree in sign is not a lever — it is a redistribution.

**Confidence: 96%** — four independent seeds, the flip present in every one, mechanism measured at
n=27 with corr −0.815.
