# SIDECHAIN_DESIGN — the hydrophobic ranking failure: mechanism test and architectural fix

### DeepEF, M.Sc. thesis, Nissim Brami. Written 2026-09-08.
### Companion to `results/FINDINGS.md` §3.4. **This document CORRECTS §3.4 and §6.2.**

Everything below was computed on the 10 eval CSVs in `eval_results/abl_*.csv`
(~26,300 matched mutations each, 28 test proteins) by
`scripts/sidechain_mech.py`, `sidechain_mech2.py`, `sidechain_mech3.py`,
`sidechain_mech4.py`, with raw output in `results/sidechain_mech{,2,3,4}.json`
and `results/sidechain_mech{,2,3,4}.out`.

---

# PART 0 — SUMMARY, INCLUDING WHAT WAS REFUTED

FINDINGS §3.4 asserted a mechanism: *only 4 backbone atoms (N, CA, C, CB) are stored, so
side-chain packing is invisible; W and Y are the exceptions that show the pattern tracks
side-chain **BULK***. That was an assumption stated as a mechanism. It has now been tested,
and it is **wrong as stated** and **right in a narrower, more useful form**.

| claim | verdict | number |
|---|---|---|
| slope tracks hydropathy | **HOLDS** | r = −0.765, sign-consistent 10/10, shuffle p < 0.0002 |
| slope tracks side-chain BULK **as the primary axis** | **REFUTED** | volume alone LORO R² = **0.031** (i.e. nothing) |
| W and Y are "exceptions" | **REFUTED — the premise was an artifact** | on within-protein slopes W=0.184, Y=0.189, sitting **inside** the hydrophobic pack (mean 0.176), not outside it |
| bulk matters at all | **HOLDS, as a SECOND term** | adding VOL to KD moves LORO R² 0.666 → **0.801** (+0.135) |
| the deficit is worst at BURIED positions | **HOLDS, and this is the real packing signal** | polar-minus-hydrophobic slope gap 0.095 buried vs 0.024 exposed; within-protein permutation **p < 0.002** |
| a side-chain descriptor block would fix it | **REFUTED ON ARITHMETIC** | any per-residue scalar table is `one_hot @ T` — a rank-1 projection of a block `fc1` already receives |

**The one-sentence result.** The compression axis is **desolvation on burial**, not bulk:
hydropathy explains it, volume explains it only as a modifier, and the effect is
concentrated at buried positions — but **no per-residue descriptor can fix it**, because
every such descriptor is a linear function of the one-hot the model already reads. The
fix must add a **geometric, structure-dependent, state-dependent** quantity — one that
differs between the folded and unfolded passes — and the only such quantity available
from N/CA/C/CB is a **CB-direction pseudo-side-chain packing term**. That is designed
in Part IV.

---

# PART I — Q1: WHAT ACTUALLY EXPLAINS THE PER-DESTINATION SLOPE

## 1.1 A methodological correction that changes the numbers

FINDINGS §3.4's per-destination slopes (0.267–0.565) were fitted on mutations **pooled
across proteins**. Pooling lets the between-protein offset `b_p` enter as signal, so those
slopes are inflated and are not comparable with the per-protein `a_p ≈ 0.499`.

Every number in this document is fitted on **within-protein centred** residuals: each
protein's own mean is removed from both `ddG_true` and `ddG_pred` before the fit. The
global within-protein slope is then **0.2626** (pearson 0.5033, spearman 0.4434), and the
per-destination slopes fall to 0.14–0.29. **The ordering is unchanged; the magnitudes are
not.** Use the centred numbers.

## 1.2 The per-destination table (within-protein centred, mean of 10 checkpoints)

| res | n | KD | vol Å³ | dG_tr | slope | pearson | spearman |
|---|---|---|---|---|---|---|---|
| W | 1395 | −0.9 | 227.8 | **+2.25** | 0.1836 | 0.3955 | 0.3170 |
| Y | 1363 | −1.3 | 193.6 | **+0.96** | 0.1888 | 0.4138 | 0.3430 |
| F | 1385 | +2.8 | 189.9 | +1.79 | **0.1604** | 0.3435 | 0.2887 |
| R | 1319 | −4.5 | 173.4 | −1.01 | 0.2737 | 0.5377 | 0.4954 |
| K | 1234 | −3.9 | 168.6 | −0.99 | 0.2860 | 0.5267 | 0.4896 |
| I | 1352 | +4.5 | 166.7 | +1.80 | 0.1843 | 0.3687 | 0.3205 |
| L | 1286 | +3.8 | 166.7 | +1.70 | 0.1624 | 0.3437 | 0.2710 |
| M | 1384 | +1.9 | 162.9 | +1.23 | 0.1865 | 0.3731 | 0.3258 |
| H | 1388 | −3.2 | 153.2 | +0.13 | 0.2771 | 0.5415 | 0.4810 |
| Q | 1348 | −3.5 | 143.8 | −0.22 | 0.2838 | 0.5408 | 0.4943 |
| V | 1318 | +4.2 | 140.0 | +1.22 | 0.1792 | 0.3724 | 0.3289 |
| E | 1211 | −3.5 | 138.4 | −0.64 | 0.2823 | 0.5525 | 0.5248 |
| T | 1346 | −0.7 | 116.1 | +0.26 | 0.2553 | 0.4936 | 0.4541 |
| N | 1321 | −3.5 | 114.1 | −0.60 | **0.2914** | 0.5499 | 0.5141 |
| P | 1259 | −1.6 | 112.7 | +0.72 | 0.2755 | 0.5016 | 0.5326 |
| D | 1272 | −3.5 | 111.1 | −0.77 | 0.2835 | 0.5525 | 0.5462 |
| C | 1203 | +2.5 | 108.5 | +1.54 | **0.1427** | 0.3386 | 0.3149 |
| S | 1339 | −0.8 | 89.0 | −0.04 | 0.2765 | 0.5462 | 0.4923 |
| A | 1314 | +1.8 | 88.6 | +0.31 | 0.2175 | 0.4767 | 0.4379 |
| G | 1292 | −0.4 | 60.1 | 0.00 | 0.2733 | 0.5042 | 0.5064 |

**Read the table sorted by volume and the bulk story dies on sight.** The three smallest
residues G (60 Å³), A (89), S (89) have slopes 0.273 / 0.218 / 0.277 — indistinguishable
from the largest polar residues R (173 Å³, 0.274) and K (169, 0.286). Meanwhile C, at
108 Å³ the *fourth-smallest* residue in the set, has the **worst slope of all twenty**
(0.1427). Volume does not order this column; chemistry does.

## 1.3 The head-to-head: volume vs hydropathy

Marginal correlations across all 10 checkpoints, target = per-destination slope:

```
corr(slope, KD hydropathy)          -0.7654   sign-consistent 10/10   shuffle p = 0.0000
corr(slope, residue volume)         -0.3724   sign-consistent 10/10   shuffle p = 0.0690
corr(slope, SIDE-CHAIN volume)      -0.3724   (identical: a constant offset from VOL)
corr(slope, atoms beyond CB)        -0.1628
```

Note that `corr(KD, volume) = +0.047` — the two predictors are **nearly orthogonal** on
this alphabet, so the comparison is clean and the partials are stable:

```
PARTIAL vol | KD   = -0.5616        PARTIAL KD | vol  = -0.8115
```

Controlling for volume makes the hydropathy correlation **stronger** (−0.765 → −0.812);
controlling for hydropathy makes the volume correlation stronger too (−0.372 → −0.562),
which says both carry signal — but they are not equals, and the marginal volume
correlation does not survive its own **degenerate baseline**: 2000 destination-label
shuffles give **P(|r_vol_null| ≥ 0.418) = 0.069**, while KD gives **P = 0.0000**.

**Volume alone is indistinguishable from a label shuffle.**

## 1.4 The clean test: volume WITHIN chemical class

Volume and hydropathy are only orthogonal across the *whole* alphabet. The decisive test
holds chemistry fixed and asks whether bulk still orders the slope inside a class. If
packing is the mechanism, W (228 Å³) must be far worse than A (89 Å³) among hydrophobics,
and R (173 Å³) far worse than S (89 Å³) among polars.

```
hydrophobic AVILMFWCY  n=9   r(slope, vol) = -0.1093  [-0.288, +0.007]   sign<0: 8/10
   A(89,0.218) C(108,0.143) V(140,0.179) M(163,0.186) I(167,0.184)
   L(167,0.162) F(190,0.160) Y(194,0.189) W(228,0.184)

polar DEKRNQSTHG       n=10  r(slope, vol) = +0.1542  [-0.103, +0.287]   sign<0: 1/10
   G(60,0.273) S(89,0.277) D(111,0.284) N(114,0.291) T(116,0.255)
   E(138,0.282) Q(144,0.284) H(153,0.277) K(169,0.286) R(173,0.274)
```

**The polar row is flat to three decimal places across a 2.9-fold volume range** (60 → 173
Å³, slopes 0.273 → 0.274). The hydrophobic row is flat too, and W — the single largest
residue, the one the packing story most needs to fail — has a slope (0.184) *above* the
class mean (0.176). The sign of the within-class correlation **flips between classes**.

> **VERDICT: side-chain bulk is NOT the primary mechanism.** The class means are
> hydrophobic 0.1784 vs polar 0.2783 — a 36% gap — while volume inside each class
> explains nothing. The alphabet splits by chemistry, not by size.

## 1.5 The honest predictive comparison (leave-one-residue-out)

At n = 20 an in-sample R² always rises with parameters, so the models were compared by
**LORO** — leave-one-residue-out cross-validated R², which can and does go to zero:

| model | k | R² | adj R² | **LORO R²** |
|---|---|---|---|---|
| KD | 1 | 0.7155 | 0.6997 | 0.6659 |
| **dG_tr (Fauchère–Pliska)** | 1 | 0.7987 | 0.7875 | **0.7527** |
| **VOL** | 1 | 0.1749 | 0.1291 | **0.0305** |
| POLARSA | 1 | 0.5479 | 0.5228 | 0.4561 |
| BETA propensity | 1 | 0.6009 | 0.5787 | 0.5332 |
| **KD + VOL** | 2 | 0.8590 | 0.8424 | **0.8011** |
| dG_tr + VOL | 2 | 0.8013 | 0.7779 | 0.7072 |
| KD + dG_tr | 2 | 0.8417 | 0.8231 | 0.7942 |
| KD + dG_tr + VOL | 3 | 0.8799 | 0.8574 | **0.8067** |

Three things follow, and they are the deliverable:

1. **Volume alone is worthless: LORO R² = 0.0305.** A predictor that explains 3% of the
   out-of-sample variance of the thing it was proposed to explain is not the mechanism.
2. **Volume as a second term is worth a lot: KD → KD+VOL lifts LORO 0.666 → 0.801,
   +0.135.** So bulk *is* real, as a **modifier of a hydropathy effect**, not as the
   effect. The right statement is: *the model under-reacts to desolvation, and it
   under-reacts more when the residue being desolvated is large.*
3. **The best single table is not KD.** `dG_tr`, the Fauchère–Pliska octanol–water
   side-chain transfer free energy, beats it on LORO (0.753 vs 0.666) and on SSE
   (ratio 0.707 — 29% less squared error) despite a slightly lower Pearson r. Why is in
   the next section, and it matters for the design.

## 1.6 The W/Y "exception" dissolves — and that identifies the axis

FINDINGS §3.4 called W and Y "the telling exceptions … slopes 0.341/0.353 despite negative
hydropathy", and used them as the evidence for bulk. Two corrections:

* On within-protein-centred slopes, **W = 0.1836 and Y = 0.1888 sit inside the hydrophobic
  pack** (AVILMFC mean 0.1761), nowhere near the polar group (0.2783). They are not
  exceptions to the *pattern*. They are exceptions to **Kyte–Doolittle**.
* The residuals of `slope ~ KD` name the culprits exactly. The two largest negative
  residuals in the whole alphabet are **Y (−0.0566)** and **W (−0.0558)** — KD
  over-predicts their slope because KD scores them as polar (KD_W = −0.9, KD_Y = −1.3).

Kyte–Doolittle is a *bulk-transfer index* built partly from interior-exposure statistics;
it penalises W and Y for their indole NH and phenolic OH. **Fauchère–Pliska measures the
actual octanol–water transfer free energy of the side chain, and it puts W at +2.25 — the
most hydrophobic residue in the alphabet — and Y at +0.96.** Swapping tables removes
exactly those two residuals:

```
SSE(slope ~ KD)   = 0.015278
SSE(slope ~ dGtr) = 0.010809     ratio 0.707
```

> **The axis is DESOLVATION FREE ENERGY, in kcal/mol, not a hydropathy index.** That is a
> substantive finding: it puts the compression axis in **the same physical units as the
> label**, which a dimensionless index never was, and it explains the "exceptions" without
> invoking bulk at all.

---

# PART II — Q2: DOES IT DEPEND ON BURIAL? (YES — AND THIS IS THE PACKING SIGNAL)

Burial computed with the **same machinery as `scripts/catalogue_vs_bp.py`**: Shrake–Rupley
on the AlphaFold model, normalised by the Tien et al. 2013 theoretical max residue SASA,
on the **wild-type residue at the mutated position**. Buckets: buried rel < 0.25, mid
0.25–0.50, exposed > 0.50. The PDB residue at each position was required to **equal the
source residue of the mutation code**, or the row was dropped — **0 rows were dropped**,
so the numbering lines up exactly.

## 2.1 The marginal burial effect, with its range-restriction control

```
bucket      n      slope   pearson  spearman  var_true
buried    8627    0.2810    0.4860    0.4921    1.2804
mid       8921    0.2389    0.4404    0.3753    0.6597
exposed   8781    0.1980    0.3556    0.2923    0.3865
```

`var_true` spans 3.3× across buckets. **An OLS slope is invariant to the variance of y but
a correlation is not**, so the pearson/spearman columns are attenuated at exposed positions
by range restriction and must not be read as a ranking result on their own. The **slope**
column is the safe one, and it says buried positions are fitted *better* (0.281) than
exposed ones (0.198) — the opposite of a naive packing prediction, because buried mutations
simply have larger true effects to predict.

**So the marginal burial effect is not the packing signal.** The interaction is.

## 2.2 The interaction — the packing prediction, stated and tested

The packing story makes a sharper prediction than "buried is worse": it predicts that the
**hydrophobic penalty should be concentrated at buried positions**, because burying a large
hydrophobic is where side-chain packing dominates. Formally, the gap between polar and
big-hydrophobic slopes should grow with burial.

```
bucket    class                 n    slope   pearson  var_true
buried    big-phobic FILMWY  2674   0.1925    0.4087    0.9312
buried    small     AGSVTCP  2937   0.2760    0.4760    1.2090
buried    polar     DEKRNQH  3016   0.2873    0.4713    1.3162
mid       big-phobic FILMWY  2746   0.1725    0.3231    0.4700
mid       small     AGSVTCP  3077   0.2541    0.4689    0.8211
mid       polar     DEKRNQH  3098   0.2433    0.4605    0.6167
exposed   big-phobic FILMWY  2745   0.1538    0.2733    0.2819
exposed   small     AGSVTCP  3057   0.2260    0.4153    0.5581
exposed   polar     DEKRNQH  2979   0.1773    0.3243    0.2965
```

**(polar − big-phobic) slope gap:**

```
buried    0.0947
mid       0.0708
exposed   0.0235          buried − exposed = 0.0712     monotone in 7/10 checkpoints
```

**The gap is 4.0× larger at buried positions than at exposed ones.** At exposed positions
the hydrophobic penalty nearly vanishes (0.024), which is exactly right physically: an
exposed hydrophobic is not being packed against anything, so there is nothing for a
CB-only representation to miss.

### The degenerate baseline for this claim

The obvious objection is that burial and destination-residue composition are correlated
within a protein, so the interaction could be compositional. Tested by **shuffling the
burial label within each protein** — which preserves each protein's burial distribution,
its destination mix, and its overall slope, and destroys only the position↔burial pairing.
500 draws:

```
observed (ck0) = +0.1495     null mean -0.0187   sd 0.0307   P(null >= obs) = 0.0000
```

> **The burial × chemistry interaction is real: p < 0.002 against a within-protein
> permutation null, and it is the ONE place the original packing story survives.** The
> model's hydrophobic deficit is a *buried*-hydrophobic deficit.

## 2.3 The delta features — a null worth recording

Binning by the *change* in property (destination minus source) rather than the destination
alone:

```
d_volume bin        n     slope   spearman        d_KD bin         n     slope   spearman
[-120,-60)       2883    0.2645     0.5137        [-6,-3)       3913    0.2733     0.5331
[ -60,-20)       7142    0.2818     0.4992        [-3,-1)       3215    0.2188     0.4366
[ -20, 20)       4766    0.2762     0.4170        [-1, 1)       5817    0.2038     0.3518
[  20, 60)       7264    0.2276     0.3683        [ 1, 3)       3736    0.2264     0.3768
[  60,120)       3670    0.2194     0.3829        [ 3, 6)       4041    0.1905     0.3296
```

And at the individual-mutation level, the signed prediction error correlates with
`d_volume` at **+0.001 / +0.011** on two checkpoints (with −0.140 on the anchor-3.0
outlier) versus `d_KD` at **−0.052 / −0.054** (−0.340 on the same outlier).

**`corr(error, d_volume) ≈ 0.00`.** The model's per-mutation error is not a function of how
much bulk the mutation adds. This is the cleanest single refutation of bulk-as-mechanism in
the whole analysis, and it is a genuine negative: it says a "delta-volume" feature would
be predicting noise.

---

# PART III — THE ARITHMETIC THAT KILLS THE OBVIOUS FIX

Before designing anything, the obvious design must be ruled out, because it is the one the
project would otherwise build.

**The obvious fix.** Add a per-residue side-chain descriptor block — volume, dG_tr, a
packing proxy — as `[N, K]` columns between Fb and emb, exactly as W5 and W6 do.

**Why it cannot work.** The feature vector ends in `one_hot(20)`, the destination residue,
exact and lossless. Any per-residue scalar table `T ∈ R^20` enters as `one_hot @ T`. That
is a **rank-1 linear projection of a block `fc1_gcn` and `fc1_gat` already receive on every
forward pass**. Verified directly: `one_hot @ w_T` reproduces KD, dG_tr and VOL with
`max|err| = 0.0e+00`, and the saturated 20-parameter one-hot model interpolates every
table exactly.

> **A per-residue scalar descriptor block adds ZERO INFORMATION to either branch.** It
> cannot change the hypothesis class — only the optimisation (a better-conditioned,
> lower-dimensional parameterisation, i.e. regularisation).

This is the **same conclusion FINDINGS §8.1 reached for W6 by a completely different
route** (ProtT5 decodes residue identity at accuracy 1.000, and extrapolates held-out
hydropathy at R² = 0.704). Two independent arguments, one verdict. It also explains why
the compression exists *at all* despite the model having every residue property available:
**the information is present and the model is not using it**, which is an optimisation
failure, not a representation failure — for the per-residue part.

**What this leaves.** A fix must add something that is **not** a function of residue
identity alone. Given N/CA/C/CB, exactly one such class of quantity exists: a **geometric**
one, computed from coordinates, that **differs between the folded and unfolded passes**.
Part II says precisely which: a **burial-weighted desolvation term**, because the deficit
is a *buried*-hydrophobic deficit and burial is geometry.

### And this is exactly what W5 already is — which is why the design is a W5 arm, not a new block

W5's solvation block is `[burial, hydropathy, burial × hydropathy]`, with **burial ≡ 0 in
the unfolded state**, so the delta between passes is the hydrophobic driving force. The
product term exists precisely because *a linear layer cannot construct a product from its
factors*. That is structurally the right shape for this finding. **W5 was designed for this
problem before this problem was measured.** What the measurements now say is that W5 is
under-parameterised in two specific, testable ways.

---

# PART IV — THE DESIGN

## 4.1 Architectural facts the design must respect (two of them corrected here)

Verified by direct construction of `PEM` (`model/hydro_net.py`), lever off and on:

```
OFF   fc1_gcn.in = 52   fc1_gat.in = 36   solv_start = 48   fc1.in = 1096
W5 ON fc1_gcn.in = 55   fc1_gat.in = 39   inst_norm1 = (36,) UNCHANGED   fc1.in = 1096
```

1. Layout is `[ D(16) | Fb(32) | <new blocks> | emb(1024) | one_hot(20) ] = 1092`. New
   blocks go between Fb and emb so the **right-anchored** emb and one-hot slices never
   move. `hydro_net` slices emb and one-hot right-anchored, so deleting the one-hot columns
   would silently re-point `x[:, -20:]` into the embedding — never delete, only zero.
2. **Only `fc1_gcn` and `fc1_gat` grow.** `fc2_*` project back to fixed widths, so
   `inst_norm1 (36,)`, `inst_norm2 (72,)` and `fc_in_dim (1096)` must **NOT** change.
   Widening them was a real bug in this tree.
3. **`_sibling_block_dims` arithmetic is unforgiving**: a wrong offset makes the model read
   the wrong columns **without crashing**, sliding a block into the LLM embedding. Any new
   block must place itself through one authority, and the gate must test on/off
   combinations, not just shapes.

### CORRECTION to FINDINGS §6.2 — the GCN branch DOES see new blocks

FINDINGS §6.2 states: *"The GCN branch reads `x[:, :32]` only … any new block inserted at
offset 48 is invisible to it. W5/W6/W9/W11 reach the GAT branch alone."*

**This is wrong.** `hydro_net.py` builds both branch inputs by **concatenation**, not by a
single slice:

```python
_extra = []
if self.solv_dim:  _extra.append(x[:, self.solv_start : self.solv_start + self.solv_dim])
if self.desc_dim:  _extra.append(x[:, self.desc_start : self.desc_start + self.desc_dim])
if self.lig_dim:   _extra.append(x[:, self.lig_start  : self.lig_start  + self.lig_dim])
if _extra:
    x_gcn = torch.cat((x[:, :32], *_extra, x[:, -20:]), dim=-1)   # 52 -> 55 with W5
    x_gat = torch.cat((x[:, :16], *_extra, x[:, -20:]), dim=-1)   # 36 -> 39 with W5
```

`x[:, :32]` is the **D(16) + half-Fb prefix**, not the whole GCN input. The new blocks are
appended *after* that prefix and **before** the right-anchored one-hot. Both `fc1_gcn`
(52→55) and `fc1_gat` (36→39) grow by the block width — measured above. **New blocks reach
BOTH branches.** The true §6.2 constraint is the surviving half: neither branch sees the
1024-dim embedding (only the post-GNN concat does), so a feature that must interact with
sequence context still cannot.

This correction removes what would otherwise be the design's biggest obstacle, and it
should be propagated to FINDINGS §6.2.

## 4.2 What can be DERIVED from N/CA/C/CB

Glycine has no CB and the loader stores CA there — a virtual-CB construction is therefore
**mandatory**, not optional, or every glycine term is silently zero.

| derived quantity | from | why it is not `one_hot @ T` |
|---|---|---|
| **virtual side-chain centroid** `P_i = CB_i + (CB_i − CA_i)/‖·‖ × L(a_i)` | CA, CB, residue type | depends on **coordinates**; its neighbour counts are structural |
| **CB-direction packing count** — neighbours within 10 Å of `P_i` | above | geometric; **zero in the unfolded state** |
| **directional (HSE) burial** | CA→CB hemisphere | already in-tree as `compute_hse` |
| **desolvation free energy** `dG_tr(a_i)` in kcal/mol | table | is `one_hot @ T` — **only useful multiplied by a geometric term** |

`L(a)` is the CA→centroid distance per residue type: G 0.0, A 1.5, S/C 1.9, T/V 2.0,
D/N/L/I 2.4, P 1.9, E/Q/M 3.0, H 3.2, K 3.5, R 4.2, F 3.4, Y 3.9, W 3.9 Å. This is
**rotamer-free**: the direction is fixed by the backbone, only the reach is by type. It
cannot recover a real rotamer and does not claim to; it makes a large residue occupy more
space *along the direction its side chain actually points*, which is the entire content of
"burying a large hydrophobic".

## 4.3 The proposed block — `W12`, an arm of W5, not a new offset

**Do not add a new block at a new offset.** The measurements say the missing quantity is a
*better solvation block*, and W5 already owns offset 48 with the correct folded/unfolded
semantics and a passing gate. Adding a fourth block would create a fourth
`_sibling_block_dims` term — the exact arithmetic that §6.2's own comment warns silently
mis-reads columns. **Extend W5 in place, behind a new mode flag.**

```
--burial_mode {count, hse, packing}      # 'packing' is new; count/hse unchanged
--solvation_scale {kd, dgtr}             # which desolvation table; kd is the current default
```

### `--burial_mode packing`: the block becomes `[N, 5]` instead of `[N, 3]`

```
col 0   b_i        directional (HSE) burial of the backbone CB          [0,1]   0 unfolded
col 1   h_i        desolvation scale of residue i (kd or dgtr, normed)  [0,1]   SAME both states
col 2   b_i · h_i  the existing driving-force product                           0 unfolded
col 3   q_i        PACKING COUNT at the virtual centroid P_i            [0,1]   0 unfolded   <- NEW
col 4   q_i · h_i  bulk-weighted desolvation                                    0 unfolded   <- NEW
```

`q_i` = neighbours' virtual centroids within 10 Å of `P_i`, divided by the **constant** 30.0
— never by N. Dividing a local quantity by chain length injects a per-protein length
confound into the one feature meant to fix a per-protein problem; that was a real bug in
this tree and W5's own docstring records it.

**Why these two columns and not others.**

* `q_i` is the only genuinely new *information*: it differs from `b_i` exactly when a large
  side chain points into a crowded region — which Part II says is where the model fails.
  `corr(q, b)` across the test set must be reported; if it exceeds ~0.95 the column is
  redundant and the arm should be retired.
* `q_i · h_i` is mandatory because **a linear layer cannot construct a product from its
  factors**, and Part I §1.5 says the effect is precisely multiplicative: KD alone gives
  LORO 0.666, KD+VOL gives 0.801 — bulk acts as a *modifier of* desolvation, and only a
  product expresses that. Column 4 **is** the +0.135 LORO finding, written as a feature.
* Both are **zero in the unfolded state**, so they do not cancel in `E_u − E_f`.

### `--solvation_scale dgtr`: a one-line change with a measured justification

Replace `_KD_NORM` with the normalised Fauchère–Pliska table. This is a pure
reparameterisation with **no width change and no gate risk**, and Part I §1.6 predicts it
improves the fit by 29% in SSE and LORO 0.666 → 0.753. It is the **cheapest testable
prediction in this document** and should be run first, as its own arm, before any width
change at all.

### Where it enters, and which layers grow

Offset **48**, inside the existing W5 slot, ahead of W6/W11. Width 1092 → **1097**.
`fc1_gcn` 52 → 57, `fc1_gat` 36 → 41. `fc2_*`, `inst_norm1 (36,)`, `inst_norm2 (72,)` and
`fc_in_dim (1096)` **unchanged**. Because `solv_dim` is read through the existing
`self.solv_dim` and `desc_start = solv_start + solv_dim`, W6 and W11 relocate correctly
**with no edit to their code** — which is the whole reason for extending W5 rather than
adding a block.

Default `--burial_mode count` keeps the tree **bit-identical**.

## 4.4 What the gate must assert

`scripts/gate_g4_cpu.py` must still print **`baseline … dG=-0.0030 width=1092`** — verified
clean immediately before this work. Add to `scripts/gate_w5.py`:

1. **Width.** off 1092; `count`/`hse` 1095; `packing` **1097**; `fc1_gcn` +5 and `fc1_gat`
   +5 versus off; `inst_norm1`, `inst_norm2`, `fc_in_dim` **byte-identical** across all
   four modes. *(Catches the widening-the-wrong-layer bug.)*
2. **Unfolded zeroing.** Columns 0, 2, 3, 4 of the block are **exactly 0** in
   `get_unfolded_graph`, and column 1 is **bit-identical** between the two passes.
   *(Without this the block cancels in `E_u − E_f` and the lever is unfalsifiable — the
   exact failure W5 already suffered once.)*
3. **Glycine.** With a poly-G input, `P_i == CB_i == CA_i` and `q_i` is finite and equal to
   the plain CB count. *(No NaN from a zero-length CB−CA vector; the loader stores CA at
   the CB slot for G.)*
4. **Column placement, not just width.** With `packing` + W6 + W11 all on, assert the model
   reads the *solvation* columns at 48..52, W6 at 53.., W11 last — by feeding a graph whose
   block columns carry known sentinel values and checking `x_gcn`/`x_gat` contain them.
   *(A wrong ORDER passes every shape check; this is the documented silent-wrong-column
   failure mode.)*
5. **Both branches.** Assert `fc1_gcn.in_features` and `fc1_gat.in_features` **both** grow
   by 5 — the standing check that §4.1's correction stays true.
6. **Non-degeneracy.** On the 28 test proteins, `std(q) > 0`, `corr(q, b) < 0.95`, and
   `q` is not a function of residue identity alone: regress `q ~ one_hot`, assert
   **R² < 0.9**. *(This is the SIGNATURE-FAILURE-MODE guard: it proves the column carries
   structural information rather than being a disguised `one_hot @ T`, which Part III
   proves would be worthless. **No other assertion in this list catches that.**)*

## 4.5 How to score it — the metric rule applied

`q_i` and `q_i·h_i` are **zero in the unfolded state and non-zero in the folded state**, so
they do **not** cancel in ddG. Unlike the coil, BSA, ligands and W5's original framing,
this lever is **legitimately ddG-measurable**. But the specific prediction is sharper than
"pooled PCC goes up", and it must be pre-registered so the arm is falsifiable:

| quantity | prediction | note |
|---|---|---|
| per-destination slope of F, I, L, M, W, C | **rises** toward the polar 0.278 | the targeted effect |
| **(polar − big-phobic) gap at BURIED positions** | **falls from 0.0947** | **the primary endpoint** |
| the same gap at exposed positions | ~unchanged (already 0.024) | specificity control |
| `corr(slope, dG_tr)` across 20 destinations | **weakens from −0.731** | the axis is being removed |
| `a_p` median | rises from 0.499 | secondary; `a_p` is within-protein and ddG-valid |
| `std(b_p)` | **no prediction** | this lever is not aimed at the offset |
| pooled ddG PCC | weak prediction only | §7 — pooled PCC is dominated by the offset, and the best factorial cell had the **highest a_p but not the highest pooled PCC** |

**Do not score this on pooled ddG PCC alone.** That metric is offset-dominated and has
already produced one unfalsifiable 40-run sweep in this project.

## 4.6 Order of work, cheapest falsification first

1. **`--solvation_scale dgtr`** — no width change, no gate risk, tests §1.6's 29%-SSE
   prediction. If the predicted direction does not appear, the desolvation-axis claim is
   wrong and steps 2–3 should not be built.
2. **`--burial_mode packing` with column 3 only** (width 1096). Isolates the packing count
   from the product.
3. **Full 5-column block** (width 1097). The difference between 2 and 3 *is* the
   multiplicative claim, measured.

Each arm is one cell; all three are cheaper than the ~190 GPU-hours the D1 half of the
factorial spent confirming a negative visible at n = 5.

---

# PART V — HONEST LIMITS

* **n = 20 destination residues.** The n = 28 rule (|r| < 0.374 ≈ 0) does not transfer;
  the n = 20 two-sided 5% threshold is |r| = 0.444. Volume's marginal −0.372 is **below
  it**, KD's −0.765 far above. LORO and the label-shuffle nulls were used precisely
  because |r| at n = 20 is not self-certifying.
* **The 10 checkpoints are not independent.** They share a training set and several share
  seeds, so "sign-consistent 10/10" is a **stability** statement, not 10 independent
  replications. The permutation nulls are the inferential claims; the 10/10 counts are not.
* **`abl_anchor_w3.0` is an outlier** on the mutation-level error correlations
  (r(err,d_KD) = −0.340 vs ≈ −0.053 elsewhere), consistent with §3.6's finding that the
  anchor suppresses the exposure/slope coupling. It is included, not dropped, and the
  spread is reported.
* **Burial is computed on the wild-type AlphaFold model**, not per variant, so it is
  identical between WT and mutant at a given position — correct for *bucketing* mutations,
  but it means the burial analysis describes where mutations sit, not how burial changes.
* **The virtual centroid is not a rotamer.** It cannot represent a side chain that swings
  away from a clash, which is a real mechanism it will miss. If the arm fails, that is the
  first alternative explanation, and it is not fixable within N/CA/C/CB.
* **Nothing here has been trained.** Every number is an analysis of existing predictions.
  Part IV is a design and a set of pre-registered predictions, **not a result**.

---

# APPENDIX — REPRODUCTION

```bash
cd /home/nissimb/DeepPEF
export PATH="/home/nissimb/.conda/envs/esm2_env_py38/bin:$PATH"
export WANDB_MODE=disabled
python scripts/sidechain_mech.py    # Q1 marginals + Q2 burial, uncentred
python scripts/sidechain_mech2.py   # within-protein centring; within-class volume; interaction
python scripts/sidechain_mech3.py   # 10 property tables head to head; W/Y diagnostic
python scripts/sidechain_mech4.py   # LORO model comparison; one-hot arithmetic; permutation null
python scripts/gate_g4_cpu.py       # MUST print: baseline ... dG=-0.0030 width=1092
```

Outputs: `results/sidechain_mech{,2,3,4}.{out,json}`.
**G4 verified clean at `dG=-0.0030 width=1092` before and after this work — no source file
was modified; all four scripts are read-only analyses.**
