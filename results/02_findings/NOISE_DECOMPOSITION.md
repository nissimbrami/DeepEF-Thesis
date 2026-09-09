# Seed noise and epoch noise are NOT the same thing — and our headline depends on which

Measured locally on `scoreboard.csv` (43 scored runs) while the cluster was unreachable.

## The two variances

**Epoch-to-epoch, one seed, six scored epochs:**

```
gld_slope1.0_s42   e0 0.5712  e4 0.5899  e8 0.5837  e12 0.5710  e13 0.5810  e14 0.5760
                   sd 0.0074   range 0.0189

gld_dg_coil_s42    e0 0.1758  e4 0.2163  e8 0.2002  e12 0.2130  e13 0.2482  e14 0.2212
                   sd 0.0239   range 0.0724
```

**Seed-to-seed, one configuration, five seeds:**

```
sigma  seed2 0.6382  seed4 0.5759  seed1 0.5607  seed42 0.5592  seed3 0.5564
       sd 0.0344   range 0.0818
```

**Seed noise is 4.6× epoch noise** (0.0344 vs 0.0074).

## Why this matters for our headline

**Our best pooled score, 0.6382, is `sigma_seed2`.** The other four seeds of the *same
configuration* give 0.5564, 0.5592, 0.5607, 0.5759 — a mean of 0.5781.

**0.6382 is 1.7 standard deviations above its own family mean, and it is the maximum of five
draws.** Reporting it as "best achieved" is defensible only if stated as the maximum over seeds.
**The honest central estimate for that configuration is 0.578 ± 0.034**, which is *below* the
control-beating threshold that the slope lever clears.

## The rule this establishes

The ±0.060 band quoted throughout the project is a **seed** band. It has been used two ways, and
only one is right:

| use | verdict |
|---|---|
| "this lever's +0.03 gain is inside the noise band, so it is not established" | **correct** — that is a seed-scale comparison |
| "this epoch difference is inside the band, so it does not matter" | **wrong** — epoch sd is 0.0074, so a 0.02 epoch difference is ~3σ |

**And the converse matters more:** picking the best epoch inflates a score by ~0.019 at most,
while picking the best seed inflates it by up to 0.082. **The epoch audit caught the smaller of
the two problems.** Seed selection is the larger exposure and has not been audited.

## What should be reported

For any configuration with multiple seeds, report **mean ± sd across seeds at the val-selected
epoch**, not the maximum. On that basis:

| configuration | seeds | mean pooled | sd |
|---|---|---|---|
| sigma (control-like) | 5 | 0.5781 | 0.0344 |
| gld_slope1.0 | 1 seed × 6 epochs | 0.5788 | 0.0074 (epochs only) |

**Note that these two are statistically indistinguishable** — 0.5781 vs 0.5788. The slope arm's
advantage rests on the factorial cells (`p3_slope1.0_s42_e13` at 0.6210), not on the golden
replicate, and the three golden seed replicates have not finished scoring.

**Confidence: 90%.** Direct computation; the only caveat is that the sigma family mixes epochs
(e9–e14) with seeds, so its 0.0344 is a slight overestimate of pure seed noise.

## Action

**The three `gld_slope1.0_s1/s2/s3` replicates are the decisive measurement** and are still
training. When they land, `--slope_weight 1.0` either survives as a real lever or joins the
eleven rejections. **Nothing else in the queue answers this.**
