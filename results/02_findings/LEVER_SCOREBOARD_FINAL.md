# The information-channel scoreboard: W12 is the ONLY significant lever

**Date:** 2026-09-10. Every multi-seed lever, tested on **per-protein ranking r** — the channel
no affine calibration can move. Paired against the 6-run control family, 27 proteins.

## The table

| lever | seeds | mean | **median** | improved | paired t | **Wilcoxon** | corr(ctrl r, delta) |
|---|---|---|---|---|---|---|---|
| **W12** | 2 | **+0.0207** | **+0.0080** | **18/27** | **0.0372** | **0.0104** | −0.827 |
| slope+W5 | 2 | +0.0139 | −0.0006 | 13/27 | 0.1128 | 0.3483 | −0.847 |
| slope1.0 | 4 | +0.0055 | +0.0010 | 15/27 | 0.2644 | 0.4846 | −0.762 |
| W5 alone | 4 | +0.0054 | −0.0067 | 12/27 | 0.6452 | 0.8593 | −0.815 |
| u10bidir | 2 | +0.0022 | −0.0009 | 13/27 | 0.5152 | 0.9906 | −0.704 |
| slope0.5 | 2 | −0.0012 | −0.0015 | 11/27 | 0.7130 | 0.4996 | −0.531 |

**W12 is the only lever significant on either test. Nothing else clears p<0.10.**

## A correction to my own reading

Looking at per-seed sd units, `slope+W5` appeared strong (+2.06 and +2.24 control-sd on the mean).
**The paired test says otherwise: p=0.1128, median −0.0006, improves only 13 of 27.** The
control-sd framing flattered it because it compares a 2-seed mean against a 6-seed band without
pairing by protein. **The paired per-protein test is the correct one and it is not significant.**

## The structural fact this table makes unavoidable

`corr(control r, delta)` is **negative for every single lever**, from −0.53 to −0.85. Every lever
we have built helps the proteins the baseline handles worst and does little or nothing — often
slightly negative — for the rest.

**That is not six findings. It is one:** the model has a *class* of proteins it fails on, and any
added information helps there and nowhere else. The levers differ only in **how much collateral
damage** they do:

| | improved | worst regression |
|---|---|---|
| **W12** | **18/27** | **−0.017** |
| W5 | 12/27 | −0.078 |

**W12 is distinguished not by a bigger gain on the hard proteins but by a smaller cost on the
easy ones.**

## What this means for the remaining arms

The queued arms (W15, HSE, distogram, ensemble) attack the same deficit. The scoreboard predicts
they will show the same negative correlation. **The question to ask of each is not "does the mean
rise" but "what is the median, and what is the worst regression".**

**Confidence: 95%** — paired tests on 27 proteins, two independent statistics per lever, and the
correlation sign is consistent across all six.
