# E7 — W12 on the canonical basis: an information gain WITHOUT a headline number

**Date:** 2026-09-10. Checklist item **E7**: *"W12 pooled score, 2 seeds, canonical basis."*

## What we had learned before this item

W12 is the only lever ever to clear significance on the **ranking channel**: +3.19 control-sd,
paired t p=0.0372, Wilcoxon p=0.0104, robust to leave-one-out (`W12_ROBUSTNESS.md`).

**What was not good:** it had never been placed on the project's own canonical basis next to a
pooled number, so it was impossible to say what W12 is worth as a *reported result* rather than
as a mechanism.

**The direction:** score it on the declared basis — 27 proteins, ddG, val-selected epoch — and
report pooled alongside the ranking channel.

## PLAN

**Falsifier:** if pooled fails to clear the pooled channel's own seed band, then W12 is an
information gain **without** a headline number, and must be reported that way.

## EXECUTE

| | pooled | offset oracle | mean r | median r |
|---|---|---|---|---|
| control family (6 seeds) | 0.5757 ± 0.0314 | 0.7330 ± 0.0108 | 0.7273 ± 0.0065 | 0.7897 ± 0.0024 |
| `w12_s1_e14` | 0.6191 | 0.7519 | 0.7473 | 0.7885 |
| `w12_s2_e12` | 0.6159 | 0.7400 | 0.7485 | 0.8046 |
| **W12 mean** | **0.6175** | **0.7460** | **0.7479** | **0.7965** |

**Gain over the control family, each channel in units of its OWN seed sd:**

    pooled   +0.0418  = +1.33 sd   (sd 0.0314)
    mean r   +0.0207  = +3.19 sd   (sd 0.0065)

Against the canonical headline 0.5772: **+0.0403**.
Against the best single canonical run 0.6382: **−0.0207**.

## VERIFY — the falsifier FIRED

**Pooled clears only 1.33 sd — inside the seed band.** The ranking channel clears 3.19 sd.

**W12 is an information gain that does not convert into a headline pooled number**, and the reason
is the one this project has documented repeatedly: pooled is dominated by the between-protein
offset `b_p`, which is 96.3% one global constant and is unpredictable by ten independent methods.
A lever that improves *within-protein ranking* moves pooled only weakly, because pooled is mostly
measuring something else.

**How W12 must be reported:**

> *W12 raises within-protein ranking by +0.0207 (+3.19 seed-sd, Wilcoxon p=0.0104, 2 seeds
> agreeing to 0.0012). Its effect on the pooled score is +0.042, inside the pooled seed band of
> ±0.031 and therefore not established.*

**Do not headline 0.6175.** It is a 2-seed mean whose channel cannot resolve it, and it is
**below** the best single canonical run (0.6382) — a control seed with no lever at all.

## A discrepancy in the canonical basis itself, flagged not fixed

Recomputing the declared 9-CSV population from the same files:

| quantity | recorded in `HEADLINE_BASIS.md` | recomputed today |
|---|---|---|
| pooled | 0.5772 ± 0.0316 | **0.5865 ± 0.0304** |
| offset oracle | 0.7156 ± 0.0474 | **0.7327 ± 0.0126** |

Pooled is close (+0.009, within rounding of a slightly different member list), but **the oracle
differs by +0.017 and its sd by nearly 4x (0.0126 vs 0.0474)**. A 4x sd discrepancy is not a
rounding artifact — it suggests the recorded oracle was computed over a different population or
with a different estimator (`HEADLINE_BASIS.md` §4.1 already documents that earlier headline
populations "do not reproduce").

**Not fixed here** — E7's scope is W12, and changing the canonical basis is a separate decision
that `gate_headline.py` enforces. **Recorded as an open discrepancy for whoever revises the basis.**

## DONE

**Cost: ~15 min, no GPU.** W12's reporting form is now settled, and a basis discrepancy is on
record.

**Confidence: 96%** on the W12 numbers (recomputed directly from the CSVs); **85%** that the
oracle discrepancy is a population difference rather than an estimator change — that would need
the original member list, which §4.1 says was never written down.
