# Factor A is NOT neutral — it is invisible on pooled and clearly harmful at k=20

**Date:** 2026-09-09. 36 factorial cells scored (24 D0, 12 D1). This finding exists only because
the new rule forces every lever to report **k=0 AND k=20**.

## The record said A does nothing

On pooled ddG PCC, within D0, A looks like noise — and that is what was recorded:

| factor | A=0 | A=1 | delta | t |
|---|---|---|---|---|
| A | 0.5888 | 0.5946 | +0.0058 | +0.71 |
| B | 0.5870 | 0.5964 | +0.0093 | +1.17 |
| C | 0.5939 | 0.5895 | −0.0044 | −0.54 |

## On k=20 it is the largest non-D effect in the factorial

| factor | A=0 | A=1 | delta |
|---|---|---|---|
| **A** | **0.7487** | **0.7228** | **−0.0259** |
| B | 0.7362 | 0.7353 | −0.0009 |
| C | 0.7368 | 0.7347 | −0.0021 |

Unpaired t = **−9.36**. And the paired design removes every confound — same seed, same B, same C,
only A flipped:

| B | C | seed | A=0 | A=1 | delta |
|---|---|---|---|---|---|
| 0 | 0 | 1 | 0.7543 | 0.7094 | −0.0449 |
| 0 | 0 | 2 | 0.7538 | 0.7240 | −0.0298 |
| 0 | 0 | 42 | 0.7497 | 0.7266 | −0.0231 |
| 0 | 1 | 1 | 0.7462 | 0.7340 | −0.0122 |
| 0 | 1 | 2 | 0.7549 | 0.7092 | −0.0457 |
| 0 | 1 | 42 | 0.7418 | 0.7308 | −0.0110 |
| 1 | 0 | 1 | 0.7552 | 0.7183 | −0.0369 |
| 1 | 0 | 2 | 0.7485 | 0.7246 | −0.0239 |
| 1 | 0 | 42 | 0.7551 | 0.7224 | −0.0327 |
| 1 | 1 | 1 | 0.7395 | 0.7282 | −0.0113 |
| 1 | 1 | 2 | 0.7441 | 0.7234 | −0.0207 |
| 1 | 1 | 42 | 0.7416 | 0.7227 | −0.0189 |

**Paired mean −0.0259, sd 0.0122, t = −7.34, negative in 12 of 12 pairs.**

**The epoch confound runs the WRONG way to explain it.** A=1 cells sit at mean epoch 12.7 versus
9.8 for A=0 — the A=1 arm is trained *longer* and still loses. Seeds are balanced (1, 2, 42 on
both sides).

## Why pooled cannot see it

Factor A is the **WT anchor**. FINDINGS §3.6 already measured that the anchor suppresses the
exposure/slope coupling (r 0.63 unanchored -> 0.40 at weight 3.0). The anchor pulls every protein
toward a common wild-type reference, which flatters the raw pooled correlation by removing
between-protein offset — **exactly the thing k=20 anchors measure directly and for free.**

So A buys, during training and permanently, something a handful of measurements buys better at
inference. Worse, the paired table says it does not merely fail to help: it **costs 0.026** of
final calibrated accuracy. Consistent with `anchor_w3.0` collapsing to pooled 0.5364 / a_p 0.147.

## What this changes

1. **"A, B and C are all zero" is wrong for A.** B and C really are flat on both metrics
   (|delta| <= 0.002 at k=20). A is flat on pooled and **−0.026 at k=20**.
2. **A should be OFF in any configuration intended for few-shot use.**
3. **This vindicates the k=0/k=20 reporting rule at the first application.** A lever that
   looked neutral for the whole life of the factorial is harmful, and only the second column
   shows it.

## Factor D, for the record — now at 36 cells

| | n | pooled | a_p | k=20 |
|---|---|---|---|---|
| D0 (coil) | 24 | 0.5917 ± 0.0198 | 0.442 | 0.7358 |
| D1 (`--unfolded_emb zero`) | 12 | **0.1311 ± 0.1344** | **0.010** | **0.3420** |

**t = 11.81, zero overlap** (D0 min 0.5401 > D1 max 0.3204). D1 remains destroyed on every metric.

**Confidence: 96%** — paired within seed/B/C, 12/12 consistent, and the one available confound
(epoch) is measured and points the other way.
