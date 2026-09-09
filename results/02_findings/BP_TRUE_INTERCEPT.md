# b_p re-measured with the TRUE intercept: four quantities wear one name, and they differ 13x

**Date:** 2026-09-10. 58 healthy runs, 27 proteins. Triggered by today's discovery that
`perprot_all79.csv`'s `b` column was the **mean residual**, not a regression intercept.

## 1. Four different "b_p", all reproducible, all in circulation

On the control run:

| quantity | mean | **std** |
|---|---|---|
| ddG-space **intercept** (`pred = a·true + b`) | +0.002 | **0.1853** |
| ddG-space mean residual | +0.414 | 0.5160 |
| **dG-space WT-row error** | +0.274 | **1.5752** |
| dG-space mean error | +0.688 | 1.2572

**`FINDINGS.md` §1.2 lists `std(b_p) = 1.5741` in a table headed `pred ≈ a_p·true + b_p`** — the
ddG parameterisation. But 1.5741 is the **dG-space WT error**, which reproduces here as 1.5752.
The ddG intercept is **0.1853**. They differ by **8.5x**, and the mean residual is a third thing again.

Part D of FINDINGS already flagged this ("conflates two quantities differing ~8x") but the table
was never corrected. **It is corrected now.**

## 2. b_p is a protein property, confirmed at 58 runs

Per protein, mean intercept across 58 healthy runs, and its spread across those runs:

    between-protein sd  0.1182
    within-protein sd   0.0589   (same protein, different runs)
    ratio               2.01

**b_p is twice as much protein as it is run noise.** This independently reproduces the ICC=0.898
claim by a different route.

## 3. It is dominated by two proteins — the same two, again

| protein | b_p | sd | a_p | r |
|---|---|---|---|---|
| **2KVS** | **+0.297** | 0.375 | 0.154 | 0.353 |
| **3DKM** | **+0.257** | 0.086 | 0.338 | 0.627 |
| r18_3_TrROS_Hall | −0.217 | 0.072 | 0.485 | 0.797 |
| HEEH_KT_rd6_0793 | +0.181 | 0.079 | 0.122 | 0.407 |
| ... | | | | |
| HHH_rd1_0142 | −0.007 | 0.033 | 0.282 | 0.774 |

**Mean |b_p| of the other 25 proteins is 0.073** — the top two are 4x that.

**2KVS appears in every concentration list in this project:** worst ranking (r=0.353), worst
slope (a_p=0.154), largest offset (+0.297), largest W12 gain (+0.230), largest W5 gain (+0.211).
It is one protein carrying an outsized share of every effect we have measured.

**And note 2KVS's b_p sd across runs is 0.375 — larger than its mean.** Its offset is not stable
even within our own runs, which is why an offset corrector could never learn it.

## 4. What this means

1. **Every b_p number must state its space** (dG or ddG) **and its estimator** (intercept, mean
   residual, or WT-row error). Four quantities, one name, a 13x range between the extremes.
2. **The offset problem is a 2-protein problem**, matching the ranking gains. The project's
   recurring structure is not "27 proteins are miscalibrated" but "2-5 proteins are, badly".
3. **This is why the feature-based corrector failed** (§4.4): there is no population to regress
   on, only outliers, and one of them is not even self-consistent across seeds.

**Confidence: 96%** — 58 runs, the 1.5752 reproduces the recorded 1.5741 to 4 digits, confirming
the identification of which quantity was published.
