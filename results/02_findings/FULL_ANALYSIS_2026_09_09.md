# Full analysis of all 79 eval CSVs — what is true, what was wrong

**Date:** 2026-09-09. 16 agents, 4 independent slices, every novel claim adversarially re-derived
from the raw CSVs. Two claims were REFUTED by verification; both corrections are recorded here.

---

## 1. The pooled/per-protein gap is TWO PROTEINS, not a broad property

Dropping just 2KVS and HEEH_KT_rd6_0793 lifts control pooled **0.5635 -> 0.6837 (+0.120)** —
3.5x the seed bar. The effect holds above the seed bar in **82.3% of all 79 runs**. Beyond k=5
the drop-worst curve is flat (k=5 0.7334, k=8 0.7351, k=10 0.7342): a few-outlier phenomenon.

Their offsets are enormous against an inter-protein spread of only sd 0.38:

    2KVS  mean b_p +1.29        HEEH_KT_rd6_0793  mean b_p +1.73

**The mechanism is offset, not ranking.** Recentring every protein to zero mean — removing b_p
only — lifts pooled **0.5635 -> 0.6878**, recovering **76%** of the whole gap to ppmean 0.7262.
Adding per-protein rescaling reaches only 0.7180.

**The drop-worst-k experiment and the offset-removal oracle are the same finding seen twice.**

## 2. Protein difficulty is a property of the DATA

Two-way decomposition of the complete 27x79 grid (0 NaN, 0 missing):

| source | share |
|---|---|
| protein identity | 15.4% |
| run identity | 57.8% |
| residual | 26.8% |

But the run effect is **almost pure level shift** (run mean r spans −0.015 to 0.749). Once run
level is removed, **protein identity holds 36.5%**; ICC(3,1) = 0.357, mean pairwise Spearman
between runs' protein profiles = 0.603, and the 79-run mean profile has **ICC(3,k) = 0.978**.

Rank stability: HEEH_KT_rd6_0793 is bottom-5 in **82.3%** of runs, 2KVS in **81.0%**;
2WXC is top-5 in **91.1%**.

**Hardest:** 2KVS 0.276, HEEH_KT_rd6_0793 0.321, 1QKH 0.441, 1QP2 0.466, 3DKM 0.468.
**Easiest:** 2WXC 0.750, 2K28 0.718, 1W4H 0.716, r18_3_TrROS_Hall 0.694, 2K1B 0.686.

**Model configuration moves every protein up and down together; it does not change which
proteins are hard.**

## 3. REFUTED — "offset beats slope as the predictor of per-protein quality"

The first pass claimed `corr(mean_r, |b_p|) = −0.696` beats `corr(mean_r, a_p) = +0.613`, and
that slope does not survive partialling. **This was an artifact of a mislabeled column.**

`results/perprot_all79.csv`'s `b` column is **not** the regression intercept — it is the mean
residual `mean(pred − true)` (exact identity to 6.7e-16; differs from the true intercept by up
to 1.777). Since `b_p = mean(y) − a_p·mean(x)` and a_p spans 0.09–0.80, the mean residual has
slope shrinkage baked into it, which artificially coupled the two channels.

| quantity | claimed | **correct** |
|---|---|---|
| corr(mean_r, mean\|b\|) | −0.696 | **−0.583** |
| corr(mean_a, mean\|b\|) | −0.636 (p=4e-4) | **−0.259 (p=0.19, n.s.)** |
| partial(r, a \| b) | +0.307 (n.s.) | **+0.588 (p=0.0016) — SURVIVES** |
| partial(r, b \| a) | −0.502 | **−0.556 (p=0.0032)** |

**Corrected conclusion: slope and offset are near-equal, largely INDEPENDENT predictors**
(+0.588 vs −0.556, joint R² = 0.568). The "one channel dominates" story is dead.

## 4. Ensembling: significant at k=20, null on pooled

Confirmed independently (63 healthy runs). **Pooled: no ensemble beats the best single run**
(best N=5 gives 0.6373 vs 0.6382); honest 13/14 protein splits put the true gain at +0.005 to
+0.010, ~4x below the seed bar, winning only 83/200 splits.

**At k=20 it is real:** top-5 ensemble 0.7661 vs best single 0.7591 vs control 0.7513; paired
CIs [+0.0052,+0.0092] and [+0.0092,+0.0225], both P=1.000. See `ENSEMBLE_MEASURED.md`.

Runs correlate at **0.946** pairwise — little diversity to exploit. More seeds will not help.

## 5. METHOD: the eval CSVs are not row-aligned, and two schemas exist

**47 files carry a `variant_idx` column, 32 do not, and row order differs between schemas.**
78 of 78 comparison files are positionally misaligned; sorting on `(protein, deltaG, ddG)`
aligns all of them.

**The operational stake, measured:** an 8-file ensemble scores pooled **0.2993 naive-positional
vs 0.5967 correctly aligned** — naive averaging lands *below every strong member*. The gap is
8.6x the seed bar.

**Warning about a null test:** "sorting reproduces all 79 pooled scores" proves nothing — pooled
corrcoef is permutation-invariant, so sorted and unsorted agree to 5e-05 either way. Only the
ensemble test has power here.

## 6. METHOD: k=20 must be aggregated POOLED, and the seed bar is channel-specific

A verifier argued k=20 should average per-protein r instead of pooling. **That would be a null
metric.** Affine calibration cannot change a within-protein correlation, so per-protein-mean
after calibration equals the raw per-protein mean — measured on the control: **0.7256 after
calibration vs 0.7262 raw.** It answers no question.

**Pooled-after-calibration (0.7524) is the correct quantity**: it asks whether k anchors put the
proteins on a common scale, which IS the few-shot question. The convention in `SCORE_ALL76.tsv`
is right and all k=20 numbers in this project stand.

**But a real correction:** the **0.0344 seed sd is a POOLED-channel bar and is the wrong
denominator for k=20.** Measured in the 6-run control band:

    pooled seed sd  0.0314        k=20 seed sd  0.0066        per-protein r seed sd  0.0065

k=20 is ~5x tighter. Expressing k=20 gains in pooled units shrinks them 5x by construction.
On its own scale, **W12's k=20 gain is +1.56 sd** (not +0.30), and the anchor's loss is **−6.7 sd**.

## 7. Epochs

Only 2 runs are scored at >=3 epochs, so this is n=1-2 throughout. For `gld_slope1.0_s42`
pooled peaks at **e4** while k=20, per-protein mean and median all rise to **e13/e14**; selecting
on pooled costs **+0.0060** k=20 (paired, 400 common draws, 97-98% of draws favour e14).

**The honest statement is stronger than "pooled picks the wrong epoch":** pooled's range across
all 7 checkpoints is 0.0189 — *smaller than the seed sd* — and the e4-vs-e14 pooled difference
has CI [−0.029,+0.056]. **Pooled cannot discriminate checkpoints at all; ranking by it is
ranking noise.**

`a_p` is **not** monotone in epoch (4/6 and 4/5 steps up), so the slope lever's a_p is not just
"more training": at matched epochs the slope run beats dg_coil by ~0.62 absolute, and
`gld_slope1.0_s42` already starts at a_p 0.480 at e0 versus dg_coil's all-epoch max of 0.103.

**Epoch explains none of the between-run variance:** pooled~epoch R² = 0.0043, p = 0.568, n=79.
No overfitting clears the noise bar (largest late drop −0.0270).

## 8. The ceiling

k=20 calibration already closes **87.7%** of the raw->affine-oracle gap. Remaining headroom
(+0.167 to a theoretical 0.8049 for the best run) is **entirely within-protein ranking quality**,
plus ~0.11 lost to the shallow median slope a_p = 0.410.

**There is almost nothing left in calibration. The rest is §3.4's hydrophobic ranking deficit.**

---

**Confidence: 93%.** Every headline reproduced by an independent agent from raw CSVs; the two
refutations were themselves re-derived; §6's defence of the pooled convention is a direct
measurement (0.7256 vs 0.7262), not an argument.
