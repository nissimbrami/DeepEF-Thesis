# W12 (side-chain readout): two seeds, and the gain is real but NOT general

**Date:** 2026-09-09. `w12_s1_e14` landed 14:26, joining `w12_s2_e12`. First multi-seed read.
All numbers: 27 proteins (2K5H dropped), ddG, k=20 uses ridge lam=3 (see `CALIB_SHRINKAGE.md`).

## 1. The two seeds agree to 0.0032

| arm | pooled | k=20 | a_p | per-protein mean r |
|---|---|---|---|---|
| control `calib_ctrl_repro2_e14` | 0.5635 | 0.7540 | 0.496 | 0.7262 |
| `w12_s1_e14` | **0.6191** | 0.7631 | 0.411 | 0.7473 |
| `w12_s2_e12` | **0.6159** | 0.7593 | 0.350 | 0.7485 |
| **W12 mean** | **0.6175** | **0.7612** | 0.381 | **0.7479** |

**Seed spread 0.0032** — an order of magnitude tighter than the 0.0344 seed sd measured on the
sigma family. Gain over control **+0.0540 = 1.57 sigma**.

**This is the largest replicated pooled gain in the project.** Compare, on the same basis:
- slope1.0, 4 seeds: mean 0.5915, **+0.0280 = 0.81 sigma**, seed sd **0.0407**
- w5_dg, 3 seeds: mean 0.5988, **+0.0353 = 1.03 sigma**, seed sd 0.0062

W12 is the only lever whose replicate mean clears 1.5 sigma.

## 2. It also survives at k=20 — unlike every other lever

| lever | k=20 vs control |
|---|---|
| **W12** (2 seeds) | **+0.0072** |
| slope1.0 (4 seeds) | −0.0014 |
| w5_dg (3 seeds) | −0.0069 |

W12 is the **only** lever that does not lose ground once anchors are available. This directly
**contradicts** the retracted `W12_W13_COMPETE.md` claim that W12 becomes a liability under
calibration, and is consistent with `W12_W13_RETRACTION.md`.

**And it raises within-protein ranking:** per-protein mean r 0.7262 -> 0.7479 (**+0.0217**).
That is the one channel `a_p` correction cannot fake — W12 is adding information, not rescaling.
Note `a_p` FALLS (0.496 -> 0.381) while both pooled and r RISE, which is why a_p must never be
used as the sole score.

## 3. The load-bearing caveat: 98.4% of the gain is 5 proteins

Bootstrapping over **proteins** (n=27, the true replication unit) rather than over rows:

| arm | delta vs control | 95% CI | P(delta>0) |
|---|---|---|---|
| w12_s1 | +0.0518 | **[−0.0163, +0.1367]** | 0.877 |
| w12_s2 | +0.0487 | **[−0.0139, +0.1322]** | 0.869 |

**Both CIs include zero.** Only 17 of 27 proteins improve, and:

| protein | control r | W12 r | delta |
|---|---|---|---|
| 2KVS | 0.160 | 0.461 | **+0.300** |
| HEEH_KT_rd6_0793 | 0.321 | 0.471 | **+0.150** |
| 1QKH | 0.514 | 0.578 | +0.064 |
| HEEH_KT_rd6_0746 | 0.764 | 0.804 | +0.040 |
| HHH_rd1_0142 | 0.788 | 0.813 | +0.024 |
| ... | | | |
| 2BTH | 0.860 | 0.843 | −0.017 |

**The top 5 carry 98.4% of the total gain, and the top 2 carry 79%.** Both are proteins the
control handles badly (r = 0.16 and 0.32). W12 rescues the model's worst cases and does
essentially nothing — very slightly negative — on the proteins that already work.

**This is the same shape as W13** (92.8% from two proteins) and the same shape as the offset
oracle (78% from two proteins). It is the project's recurring structure: **gains concentrate in
a handful of hard proteins, so n=27 cannot resolve them.**

## 4. Verdict

**W12 is the strongest lever in the project on every metric that matters** (pooled, k=20,
per-protein r), it replicates across two seeds to 0.0032, and it is the only lever that survives
calibration. **But at n=27 proteins the effect is not statistically separable from zero**
(P(delta>0) = 0.88, CI includes 0), because it is carried by 5 proteins.

**Honest statement for the thesis:** *W12 raises pooled ddG PCC by +0.054 (2 seeds, spread 0.003)
and per-protein r by +0.022, concentrated in the proteins the baseline handles worst. On 27 test
proteins this is 1.57 sigma against seed noise but not significant under a protein-level
bootstrap.* Do not claim significance; do claim it is the best-replicated lever we have.

**What would settle it:** more test proteins, not more seeds. The seed spread is already 0.003;
the protein-level spread is what the CI is made of.

**Confidence: 95%** — two independent seeds, direct measurement, and the CI is computed rather
than assumed.
