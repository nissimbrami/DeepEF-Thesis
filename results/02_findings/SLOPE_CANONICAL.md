# P0a — the slope arm at its CANONICAL epoch. The headline a_p was peeked.

`gld_slope1.0_s42`, canonical basis (27 proteins, 2K5H dropped, ddG).
Canonical epoch **e10**, selected on VALIDATION per `run_calib_eval.sh:15`. Seven epochs of this
run were scored on TEST to trace a trajectory — legitimate as a trajectory, **fatal if the best
one is quoted as a result.**

| epoch | pooled | oracle | a_p | r | s |
|---|---|---|---|---|---|
| e0 | 0.5712 | 0.6417 | 0.4764 | 0.6635 | 0.6861 |
| e4 | 0.5899 | 0.6749 | 0.7000 | 0.7216 | 0.9479 |
| e8 | 0.5837 | 0.6831 | 0.8092 | 0.7287 | 1.0893 |
| **e10 ← canonical** | **0.5798** | **0.6815** | **0.7821** | **0.7250** | **1.0530** |
| e12 | 0.5710 | 0.6802 | 0.7864 | 0.7274 | 1.0557 |
| e13 | 0.5810 | 0.6860 | **0.8567** | 0.7301 | 1.1438 |
| e14 | 0.5760 | 0.6849 | 0.8401 | 0.7302 | 1.1212 |

## The correction

**The record quotes a_p = 0.857 (e13). The quotable number is a_p = 0.782 (e10).**
e13 is the maximum over seven epochs scored on TEST — selecting it is peeking, and it inflates
a_p by **+0.075**, which is larger than the entire pooled gain the slope lever ever produced.

Every table quoting 0.857, 0.8401, or "a_p 0.496 → 0.857" must read **0.496 → 0.782**.
The lever still works — this is a correction of magnitude, not of direction.

## What e10 reveals that e13 hid: the slope lever OVERSHOOTS

`a_p = r·s`, exactly. Track `s = std(pred)/std(true)`:

```
e0   s = 0.686   under-dispersed  (the original defect)
e4   s = 0.948   almost calibrated
e8   s = 1.089   ALREADY OVERSHOT
e10  s = 1.053   canonical — still over 1
e13  s = 1.144   overshooting further
```

**Perfect calibration is s = 1.0.** The arm crosses it between e4 and e8 and keeps going. Past
that point the model is *over*-dispersed: predictions spread wider than the truth. `a_p` keeps
rising only because `a_p = r·s` and `s` keeps inflating — **a_p rising past s = 1 is a defect
being reported as a gain.**

And `r` is flat throughout (0.72–0.73): the lever never adds information, it only rescales.
This is the measured ceiling `a_p ≤ r` in action.

**Consequence:** the right target is not "maximise a_p". It is **s = 1.0**, which lands between
e4 and e8 — *earlier* than the epoch validation picked. Pooled is essentially flat across all
seven epochs (0.571–0.590, span 0.019, well inside the ±0.060 seed band), so pooled cannot
discriminate here. **s can.** The three seed replicates now running (`gld_slope1.0_s1/s2/s3`)
should be read on **s**, not on a_p.
