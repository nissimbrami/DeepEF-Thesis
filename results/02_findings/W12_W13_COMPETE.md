> **RETRACTED 2026-09-09.** The k=5/k=20 sign reversal below is an artifact of a free
> per-protein affine fit (unstable at small k) scored on UNPAIRED draws. Corrected paired
> ridge measurement: no significant loss at any k. See `W12_W13_RETRACTION.md`.

# W12 and W13 do NOT stack — they compete, and W12 becomes a LIABILITY once W13 is applied

Direct test rather than inference: apply W13 to the control and to W12, and see whether the gains
add. 25 random draws per k, canonical basis.

## The measurement

| | k=0 | k=5 | k=20 |
|---|---|---|---|
| control | 0.5635 | **0.6830** | **0.7213** |
| W12 | **0.6159** | 0.6566 | 0.7160 |

```
W12 alone (k=0):        +0.0524
W13 alone on control:   +0.1578
if additive, expect:    +0.2101
ACTUAL W12 + W13:       +0.1525
```

**Sub-additive by 0.058 — almost exactly W12's entire standalone gain is lost.**

## Worse than sub-additive: W12 REVERSES sign once W13 is on

```
k=0:   W12 beats control by +0.0524
k=5:   W12 LOSES to control by -0.0264
k=20:  W12 LOSES to control by -0.0053
```

**With any measured calibration, the control outperforms W12.** W12's advantage exists only in the
pure zero-shot setting.

## Why — the mechanism was already visible

```
corr(W12's per-protein r-gain, |offset| that W13 removes) = +0.744

W12 helps most: 2KVS, HEEH_KT_rd6_0793, 1QKH, HEEH_KT_rd6_0746, HHH_rd1_0142
W13 helps most: 2KVS, HEEH_KT_rd6_0793, 3DKM, 1QP2, HHH_rd1_0142
overlap: 3 of 5
```

**Both levers target the same proteins** — the badly-calibrated ones. W13 removes their offset
outright; W12 improves their ranking, which *also* shrinks the offset. Once W13 has removed it,
there is nothing left for W12 to recover, and W12's cost (`a_p` 0.38 vs control 0.52, `s` 0.50 vs
0.70) becomes pure loss.

## Consequence for the thesis — this forces a choice

**The two headline results are mutually exclusive in practice:**

| deployment | best option | score |
|---|---|---|
| **zero-shot**, no measurements | **W12** | 0.6159 |
| **few-shot**, k≥5 measurements | **control + W13** | 0.6830 – 0.7213 |

**Do not report "W12 + W13" as a combined pipeline.** Report them as alternatives selected by
whether calibration data is available.

**And it explains an earlier result.** W13b found that the best-slope arm finished *last* at k=20
while the worst tied for first. That was the same phenomenon: any lever that partially corrects
calibration is redundant once calibration is measured directly, and its side-costs remain.

**General rule this establishes: a lever that improves calibration cannot be evaluated in
isolation from W13.** Every arm in the queue needs its k=0 *and* k=20 number, or the comparison
is incomplete.

**Confidence: 90%.** Direct measurement, n=1 for W12 so the magnitudes may shift, but the
sign-reversal at k=5 and k=20 and the +0.744 mechanism correlation are mutually consistent.
