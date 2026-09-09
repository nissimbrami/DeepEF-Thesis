# `r` is nearly constant across every healthy run — all the variation is in `s`

Local analysis of all 43 scored runs. `a_p = r·s` exactly, where `r` is the within-protein
ranking quality and `s = std(pred)/std(true)` is the dispersion ratio.

## The measurement, over the 29 runs with pooled > 0.4

```
r:  mean 0.7185   sd 0.0223   range 0.6387 - 0.7445
s:  mean 0.7409   sd 0.2441   range 0.2240 - 1.1734

sd(s) / sd(r) = 11.0x
```

**Across 29 independent configurations — different levers, seeds, epochs, loss modes — `r` moves
by ±0.022 while `s` moves by ±0.244.** Every lever this project has tested changes the *dispersion*
of the predictions and leaves the *ranking* essentially untouched.

**`r` behaves like a property of the model class, not of any configuration.** Nothing we have
tried has improved the model's ability to rank mutations within a protein.

## The compression signature

Seven of the 29 runs sit above the median `r` and below the median `s` — they rank well and
compress hard. **That group includes our headline run:**

| run | pooled | r | s | a_p |
|---|---|---|---|---|
| **sigma_seed2_e10** | **0.6382** | 0.738 | **0.607** | 0.448 |
| p3_a0_d0_s0_D0_coil_seed1 | 0.5917 | 0.734 | 0.600 | 0.440 |
| gld_u10bidir_s42 | 0.5893 | 0.736 | **0.346** | 0.255 |
| gld_w7edge_s42 | 0.5808 | 0.731 | 0.508 | 0.371 |

**Our best-scoring run is one of the most compressed.** It ranks no better than the others
(r = 0.738 against a mean of 0.719) — it simply landed a favourable seed.

## What predicts pooled

```
corr(r, pooled) = +0.399
corr(s, pooled) = +0.281
```

**Neither is strong**, and `r` — the thing no lever moves — matters more than `s`, the thing every
lever moves. **That is the whole difficulty of this project in two numbers.**

## Correction to a claim I made earlier

I reported `corr(pooled, oracle) = +0.992` over all runs as evidence they track together.
**That was inflated by the collapsed D1 runs**, where both quantities go to zero simultaneously
and manufacture a correlation out of a shared failure. Restricted to healthy runs it is **+0.515**
— moderate, not near-perfect.

**Rule this establishes: never compute a correlation across a population that includes collapsed
runs.** It is the same error as the mixed-population headline (0.4899) caught earlier.

## Consequence for the thesis

The project's stated aim is to fix calibration. **Calibration is `s`, and `s` is the one thing our
levers can move — but moving it buys little (`corr(s, pooled) = +0.281`).** The quantity that
matters more, `r`, has not moved by more than ±0.022 across every arm ever run.

**A better score requires improving `r`, i.e. the ranking itself — which means new information,
not recalibration.** That is precisely the case for W15 packing and W12 transfer free energy,
the two levers that add information rather than rescale it. **Both are queued and unread.**

**Confidence: 90%** on the r/s decomposition (29 runs, exact algebraic identity);
**95%** that the +0.992 figure was an artifact of including collapsed runs.
