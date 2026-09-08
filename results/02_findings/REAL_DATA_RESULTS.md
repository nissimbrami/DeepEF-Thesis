# The four analysis files, read on REAL data — one result contradicts the record

Job 21144901, canonical basis (27 proteins, 2K5H dropped, ddG), 40 random draws per k.

## 1. W13 budget — the k=5 recommendation is CONFIRMED on real data

`p3_slope1.0_s42_e13` (best arm):

| k | pooled | sd | % of recoverable gain | marginal per measurement |
|---|---|---|---|---|
| 1 | 0.5644 | 0.042 | **−45.0%** | −0.0566 |
| 2 | 0.6362 | 0.028 | 12.1% | +0.0718 |
| 3 | 0.6643 | 0.025 | 34.4% | +0.0281 |
| **5** | **0.6967** | 0.015 | **60.2%** | +0.0162 |
| 8 | 0.7146 | 0.010 | 74.4% | +0.0060 |
| 12 | 0.7247 | 0.006 | 82.4% | +0.0025 |
| 20 | **0.7334** | 0.003 | 89.3% | +0.0011 |

**The marginal return collapses by a factor of 4 between k=2 and k=5, and by 15× by k=8.**
k=5 for 60% of the gain, k=8 for 74%. Beyond k=12 each extra measurement buys ~0.001.

**k=1 is actively harmful on all three checkpoints** — −0.057, −0.027, and −0.118. Confirmed
across independent arms, so it is structural, not noise. **Never use k<3.**

## 2. THE CONTRADICTION — b_p is NOT 96.3% one global constant on ddG

I recorded: *"b_p is 96.3% one global constant (−5.0749), MAE ≡ |mean| exactly, n_under = 28/28."*

Measured on the real eval CSVs:

| run | global | std(b_p) | MAE | frac variance global | n_under/n_over | MAE≡bias? |
|---|---|---|---|---|---|---|
| slope1.0_e13 | 0.3582 | 0.4529 | 0.4249 | **38.5%** | 7 / 20 | **False** |
| control | 0.4141 | 0.5064 | 0.4594 | **40.1%** | 5 / 22 | **False** |
| sigma_seed2 | 0.4518 | 0.4106 | 0.4669 | **54.8%** | 1 / 26 | **False** |

**38–55%, not 96.3%. And proteins are NOT all under-predicted — they are mostly OVER-predicted
(20/27, 22/27, 26/27).**

**Both numbers are right; they describe different quantities.** My 96.3% came from
`nu_sweep.json`, which is **dG-space WT error** — there the offset is −5.07 kcal/mol and all 28
proteins are under-predicted. These files measure **ddG-space intercept**, where the offset is
~0.4 and the sign flips. This is the same conflation the record already flagged once: *"std(b_p)
= 1.5741 conflates two quantities differing ~8×: dG-space WT error 1.603 vs ddG-space intercept
0.193."*

**Correction required:** every statement of the form "b_p is 96.3% one global constant" must say
**in dG space**. In ddG space — the space the headline is reported in — the global constant is
only 38–55% of it, so **more than half the offset is genuinely per-protein and therefore
potentially learnable.** That is a materially more optimistic picture than the record gives.

**Confidence: 90%** (three independent checkpoints agree; the two spaces are separately verified).

## 3. Per-protein PCC confirmed

`pp_median` 0.8034 / 0.7929 / 0.7906 — matches the headline exactly. **Confidence 95%.**

## 4. a_p spread is wide and 4–5 proteins are collapsed

`a_p` median 0.507, min 0.111, max **1.326** — some proteins are *over*-dispersed. And
**4–5 of 27 proteins have a_p < 0.3**, i.e. the model barely responds to their true ddG at all.
The `sigma_seed2` arm — our best pooled score — has the **worst** a_p median (0.410) and the most
collapsed proteins (5). **The best headline arm is the least well calibrated per protein.**

## Consequence for the thesis

The claim "the offset cannot be predicted, only measured" rests on nine failed prediction
attempts. Those attempts targeted the **dG-space** offset, which really is ~96% one constant and
therefore nearly unlearnable. **The ddG-space offset is only 38–55% global.** Whether the
remaining 45–62% is predictable was never tested in that space. **That is a real open question
the record had closed prematurely.**

---

## 5. ATTEMPT #10 — is the ddG-space offset predictable? NO.

The contradiction above opened a genuine question, so I tested it rather than speculating.
LOPO ridge on 7 label-free structural features (length, Rg, Rg/N^0.6, mean and sd of CA contacts,
contact order, median |CB-CA|), n=27:

| run | std(b_p) | LOPO R2 | corr |
|---|---|---|---|
| slope1.0_e13 | 0.4615 | **−0.0728** | +0.222 |
| control | 0.5160 | **−0.0038** | +0.318 |
| sigma_seed2 | 0.4184 | **−0.2323** | +0.056 |

All three R2 are NEGATIVE — worse than predicting the global mean — and every correlation is
below the n=27 significance threshold of 0.381.

**So the ddG-space offset is ALSO not predictable from structure.** The optimism in §2 does not
survive contact with a measurement. The record's conclusion stands, but now for the right reason:
it is not that the offset is a single constant, it is that **the per-protein part is real,
substantial (std 0.42-0.52), and still not a function of anything we can compute without labels.**

**This is attempt #10 and the cleanest one**, because it targets the quantity the headline is
actually reported in. **Confidence: 90%.**
