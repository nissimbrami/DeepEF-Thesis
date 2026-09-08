# W5 BURIAL SCORED ON dG — T12

## Why this had to be done

`gate_w5.py` passes 18/18 and **every one of those assertions is mechanical** — tensor shapes,
byte-identity when off, hydropathy ordering (I most hydrophobic, R least). **Not one scores
performance.** Meanwhile W5's own help text states that burial is ZERO in the unfolded state and
that the folded-minus-unfolded delta IS the hydrophobic driving force — so W5 acts on **dG / b_p**.
Yet its only scheduled scoring is the S7 information factorial, whose `EFFECT_MIN` is defined on
**pooled ddG**. That is the Flory-coil error one lever later.

## The trap this deliberately does NOT fall into

W5 widens the feature vector 1092 -> 1095. A checkpoint trained WITHOUT W5 has **no weights** for
those 3 columns. Flipping the flag on and running a forward pass would feed untrained weights and
measure noise, not the lever — a number that looks like a result and means nothing. **So that was
not done.**

What can be measured with no training is the honest b_p-side question:
**does the burial signal, computed on the WT structures, predict the WT error b_p?**
If it does, feeding burial to the model is motivated. If it does not, the ddG scoring is the least
of W5's problems.

## Result — `results/w5_on_dg.json`, checkpoint calib_ctrl_repro2 e14

| feature | target | pearson | spearman |
|---|---|---|---|
| **mean_burial** | **b_p** | **-0.343** | -0.282 |
| mean_burial | abs_b_p | 0.185 | 0.047 |
| frac_buried | b_p | -0.039 | -0.007 |
| length | b_p | -0.368 | -0.349 |

**n = 19 of 28.** At n=19, `|r| < 0.490` is indistinguishable from zero at p=0.05.

## The honest conclusion

**Not significant.** `mean_burial` vs `b_p` at -0.343 does **not** clear the bar. The SIGN agrees
with the replicated `frac_buried` vs b_p result (mean r = -0.475, 10/10 sign-consistent), and more
burial means a more negative WT error in both — but on this subset it cannot be called a finding.

**Do not report -0.343 as support.** It is consistent with the earlier result, not confirmation of it.

## Two caveats that limit this measurement

1. **9 of 28 proteins have no tensor directory** under
   `/groups/keasar_group/casp15/meytav/protein_tensors` (2K28, 2KXD, r12_757_TrROS_Hall, 2K1B,
   2K5H, 1W4H and 3 more). Dropping a third of the test set costs most of the power — and it is
   not a random third. Note **2K5H is among the missing**, and it is one of the two proteins that
   carry 78% of the entire offset-removal oracle gain.
2. `length` vs `b_p` is -0.368, **larger in magnitude than burial's -0.343**, and burial correlates
   with length. Neither survives at n=19, so this cannot be disentangled here.

## A units bug worth keeping

The first run returned `nan` for every burial correlation. Cause: I scaled coordinates by 0.1
("model units") before `compute_burial`. **The stored tensors are already in Angstrom and
`compute_burial`'s cutoff is an Angstrom cutoff**, so the 0.1x scaling saturated every neighbour
count to exactly 1.0 with std 0.0 — zero variance, hence nan. Measured on 2KVS and 2BTH:
Angstrom gives mean 0.48/0.41 with std 0.18/0.16; 0.1x gives 1.0 with std 0.0.
**A saturated constant does not raise; it silently produces nan or a meaningless zero.**

## What would actually settle W5

Training runs with `--burial_features` under `--loss_mode dg`, scored on dG MAE and std(b_p) rather
than pooled ddG — comparable to the coil's 4.9650 -> 3.9521. That needs GPU.
And regardless of outcome, `AUTOPILOT.md` must gain a dG-side acceptance criterion, because today a
lever that shrinks std(b_p) without moving pooled ddG is recorded as noise.
