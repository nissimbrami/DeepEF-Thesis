# E6 — `slope 0.7` confirmation: criterion registered BEFORE any result

**Date:** 2026-09-10. Checklist item **E6**. Written and committed **before** the four new seeds
finish, so the test cannot be chosen after seeing the data.

## What we had learned before this item

In the 12-arm wave, `slope 0.7` had the **highest per-protein mean r in the table (0.7387)** and
Wilcoxon p = 0.0262 — and I recorded it as a miss because it failed Benjamini-Hochberg at rank 4
(critical 0.0167).

**What was not good:** applying a multiplicity correction to it at all. Per rule **A1**, a
one-seed arm is a **SCREEN**: it yields a ranking, never a claim, and BH does not apply to it.
Calling it a miss was a category error. The correct verdict was *"leading candidate, promote to
confirmation"*.

**The direction:** confirm it properly — 5 seeds, one pre-registered test at 0.05, no
multiplicity correction.

## The arms

| seed | status |
|---|---|
| 42 | already scored — `abl_slope0.7_s42_e14.csv` |
| 1, 2, 3, 4 | submitted 2026-09-10: jobs 21175198–21175201 |

Flags reproduce the scored arm exactly:

```
--full_data --no_pretrain --no_freeze --pooled_corr_weight 0 --dg_length_norm none
--affine_calib --val_frac 0.1 --epochs 15 --loss_mode ddg --slope_weight 0.7
```

## THE PRE-REGISTERED TEST — one test, decided now

**Primary metric:** per-protein **mean** r (the ranking channel, which no affine calibration can
move), paired by protein against the 6-seed control family, 27 proteins.

**Test:** paired t-test on the 27 per-protein deltas. **One test. No multiplicity correction.**

**ESTABLISHED if BOTH hold:**
1. mean over the 5 seeds exceeds the control family mean by **more than 2 control-sd = 0.0130**
   (control mean 0.7273, sd 0.0065), i.e. **mean r > 0.7403**; and
2. paired **p < 0.05**.

**REJECTED otherwise.**

**Also reported, not used for the decision** (rule A3/§4): per-protein **median**, the
training-health line for every seed, and the seed-to-seed spread.

## My prediction, recorded in advance

**I expect REJECTED.** The single scored seed gives median 0.7931 against a control of 0.7897 —
a gap of 0.0034 against a median sd of 0.0024, i.e. **1.4σ**. The mean looks better than the
median, which across this project has repeatedly signalled a tail effect rather than a real lever
(W5 is the clearest case: +1.00σ mean, −4.15σ median).

**Registering the prediction so that a positive result is a genuine surprise and a negative one
is not a rationalisation.**

## Power check (rule A2)

Control SE 0.0027; a 5-seed arm SE ≈ 0.0065/√5 = 0.0029; difference SE ≈ 0.0040. The threshold
0.0130 is **3.3 SE**, so the design can resolve the effect it is asked to resolve — unlike the
n=1 screen, where 0.011 was 1.6σ and undetectable by construction.

**Confidence in the design: 95%.** The criterion is arithmetic on measured quantities; the only
judgement is the 2-sd threshold, chosen to match the "3 seeds minimum, 5 for a headline" rule.
