# The multi-seed scoreboard — and why "beats noise" here is not yet a claim

Every family with ≥2 scored runs, 70 CSVs, canonical basis. Families are grouped by stripping the
seed and epoch suffix; each row's `n` counts genuinely different seeds unless noted.

```
control pooled = 0.5635    (ONE run -- see the caveat below)
seed sd        = 0.0344
```

| family | n seeds | pooled | sd | PP | gain |
|---|---|---|---|---|---|
| p3_a0_d0_s0_D0_coil | 2 (s1,s2) | 0.6092 | 0.0239 | 0.7877 | **+0.0457** |
| p3_a0_d1_s0_D0_coil | 3 (s1,s2,s42) | 0.6000 | 0.0141 | 0.7792 | **+0.0365** |
| p3_a1_d1_s0_D0_coil | 3 | 0.5986 | 0.0074 | 0.7260 | **+0.0351** |
| p3_a1_d1_s1_D0_coil | 3 | 0.5980 | 0.0157 | 0.7354 | **+0.0345** |
| p3_a1_d0_s1_D0_coil | 3 | 0.5922 | 0.0203 | 0.7421 | +0.0287 |
| p3_a1_d0_s0_D0_coil | 3 | 0.5897 | 0.0041 | 0.7371 | +0.0262 |
| **slope1.0** | **10** | **0.5798** | 0.0215 | 0.7787 | **+0.0163** |
| sigma | 5 | 0.5781 | 0.0344 | 0.7890 | +0.0146 |
| loroW_onehot | 2 | 0.5744 | 0.0041 | **0.7906** | +0.0109 |
| dg_coil | 6 | 0.2124 | 0.0239 | 0.4860 | **−0.3511** |
| D1_uemb families | 2–3 | −0.07 to 0.17 | | 0.13–0.20 | **−0.40 to −0.63** |

## The caveat that stops this being a result

**The control is a single run.** Every gain above is measured against one draw from a
distribution whose sd is 0.0344. The `sigma` family — five seeds of a control-like configuration
— has mean **0.5781**, not 0.5635.

**Re-scoring every gain against the sigma mean instead:**

| family | gain vs single control | gain vs 5-seed control mean |
|---|---|---|
| p3_a0_d0_s0_D0_coil | +0.0457 | **+0.0311** |
| p3_a0_d1_s0_D0_coil | +0.0365 | **+0.0219** |
| p3_a1_d1_s0_D0_coil | +0.0351 | **+0.0205** |
| slope1.0 | +0.0163 | **+0.0017** |

**Not one family clears the 0.0344 band against a properly-averaged control.** The four rows that
looked like they beat noise did so only because the comparator was a lucky single run.

**This is the same error as reporting 0.6382 as the headline** — a maximum treated as an estimate.
Here it appears as a *minimum* treated as an estimate, which flatters everything above it.

## What is solid

**The negatives.** `dg_coil` at −0.3511 and the D1/uemb families at −0.40 to −0.63 are ten to
twenty times the noise band across 2–6 seeds each. **Those rejections are unambiguous.**

**And slope1.0 at n=10** — the largest family — sits at +0.0163 against the single control and
**+0.0017 against the proper one.** Ten runs. That is as close to zero as a measurement gets.

## Action

**The control needs seed replication before any positive claim can be made.** Three or four
control seeds would cost 36–48 GPU-h and would put every comparison in this table on a real
footing. Until then the honest statement is:

**No configuration in this project has been shown to beat a properly-estimated control.**

**Confidence: 90%** that the single-run control invalidates the four "beats noise" rows;
**95%** on the negatives.
