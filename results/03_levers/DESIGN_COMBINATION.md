# DESIGN_COMBINATION — the optimal combination of levers

**Status:** written design, precise enough to implement. No GPU job submitted. No push. No scancel.
**Gate re-run after this work** (`scripts/gate_g4_cpu.py`, SLURM CPU job 21107718, `ise-cpu-intl-01`):

```
baseline               forward ok, dG finite   PASS dG=-0.0030 width=1092
unfolded_emb=zero      forward ok, dG finite   PASS dG=0.0040  width=1092
flory_unfolded         forward ok, dG finite   PASS dG=-0.0058 width=1092
burial_features        forward ok, dG finite   PASS dG=0.0033  width=1095
burial hse             forward ok, dG finite   PASS dG=0.0017  width=1095
burial + uemb zero     forward ok, dG finite   PASS dG=0.0023  width=1095
width: levers that add no dims keep width      PASS 1092
width: burial adds exactly 3                   PASS 1095
G4-CPU: ALL PASS
```

`dG=-0.0030 width=1092` as required.

---

## 0. Three corrections to the brief, established by measurement before anything else

The brief contains three statements that the code and the eval CSVs contradict. They are load-bearing
for the plan, so they are settled first rather than inherited.

**0.1 — The W6 flag values in the brief do not exist and will raise.**
The brief lists `--aa_descriptors {none,pca16,curated12,pca16_only}`. Those names are **RETIRED**.
`Megascale-fineTuning/train.py:87` declares
`choices=['none','mordred726','mordred_pca16','mordred_pca16_only']`, and the loader asserts the CSV's
provenance header and column count against the name. Reason recorded in the help text: `curated12` had
come to load a 726-column Mordred matrix. Every command in this document uses the live names.

**0.2 — `--slope_weight` has already been run, and it failed backwards.**
The brief's plan treats factor C as unrun. `abl_p3_slope3.0_s42_e8` is a real `--slope_weight 3.0` run
(`cluster_run/scripts/submit_factorial.sh:84` passes `--slope_weight ${CW}`), its eval CSV is on disk,
and it has been inside the 10-checkpoint replication table all along. Against its own seed-matched
control:

| run | sd_ratio (target 1.0) | a_p median | pooled ddG PCC |
|---|---|---|---|
| `--slope_weight 3.0` | **0.4489** | 0.3253 | **0.5624** |
| control, same seed 42 | 0.5573 | 0.4366 | 0.5929 |

It produced **more** compression and cost pooled PCC. The mechanism is in the term itself: it is an L1
objective whose gradient magnitude is constant (±1.5) and does not anneal near the optimum, so a large
weight chatters across the optimum rather than converging.

**0.3 — `--slope_weight` with `--loss_mode dg` RAISES.** (`train.py:231`.) This is a hard constraint on
the design: **no single training run can carry both the dG loss and the slope term.** Any plan that
puts the coil's dG objective and the slope objective in one cell is unimplementable. This single fact
forces the two-track structure in §6.

---

## 1. What exactly we see — the two b_p's, and why the distinction decides the plan

The per-protein CSVs (`results/_lc_merged_abl_*.csv`, 28 rows each) carry **two different columns that
are both called an offset**, and conflating them is the fastest way to mis-score a lever.

| column | definition | sd on reference ckpt | what it is |
|---|---|---|---|
| `b_p` | intercept of `polyfit(ddg_true, ddg_pred, 1)` | **0.2273** | the **ddG-side** intercept |
| `b_p_wt_error` | `pred_dG(WT) − exp_dG(WT)` | **1.6030** | the **dG-side** absolute error |

The headline `std(b_p)=1.5741` is the **`b_p_wt_error` family** (mean 1.60 on the reference; range
1.17–1.78 across the 10 checkpoints). It is roughly **7×** the ddG-side intercept. Whenever this
document says "the b_p channel" it means `b_p_wt_error` — the absolute-dG offset — because that is the
quantity the coil, unfolded_emb, ligands and the anchor actually move, and it is the quantity worth
1 kcal/mol.

**Measured cross-tabulation** (all 10 checkpoints, per-protein, n=28 each; sign-consistency counted):

| feature | vs `a_p` | vs `b_p` (ddG intercept) | vs `b_p_wt_error` (dG offset) |
|---|---|---|---|
| `mean_rel_SASA` (exposure) | **+0.624 mean, 10/10** | −0.011, 6/10 — **noise** | +0.233, 10/10 |
| `frac_buried_rel_lt_0.25` | **−0.410, 10/10** | +0.146, 3/10 — **noise** | **−0.475, 10/10** |
| `length` | −0.215, 10/10 | +0.040, 4/10 — **noise** | −0.137, 7/10 |

Two things fall out that the brief's table does not have:

- The ddG-side `b_p` is **not predicted by anything** — every structural feature is sign-inconsistent
  against it. All the real offset structure lives in `b_p_wt_error`. This is why the b_p corrector
  fails out of sample: it was chasing a column with no reproducible structural signal.
- **Burial is dual-channel, and the brief classifies it as b_p-only.** `frac_buried` is 10/10
  sign-consistent against **both** `a_p` (−0.410) and `b_p_wt_error` (−0.475), and the two are not the
  same story — burial and exposure are near-mirror images, and exposure is the established a_p driver.
  W5 must therefore be scored on both channels, and it is the one lever that can legitimately be
  credited on ddG.

---

## 2. The spine: every built lever, classified by channel

Channel assignment is by **mechanism**, then checked against measurement. The governing rule: ddG
cancels anything identical between WT and mutant, so a lever that shifts a whole protein by a constant
is invisible on ddG **by construction**.

| # | lever | flag(s) | Δwidth | channel | basis |
|---|---|---|---|---|---|
| U2 | unfolded ProtT5 | `--unfolded_emb {full,zero,mean}` | 0 | **b_p** | value-only in unfolded pass; identical WT/mut → cancels in ddG |
| U3 | coil channel map | `--coil_channels {broadcast,ca_only,offset}` | 0 | **b_p** | function of \|i−j\| only |
| U4 | coil segment length | `--coil_b {fitted,fixed}` | 0 | **b_p** | ditto; **measured**: fixed −1.01 MAE, fitted +0.78 |
| U5 | coil edge topology | `--coil_edges` | 0 | **b_p** | unfolded graph topology, mutation-invariant |
| U6 | per-half ca_coords | (guard, auto) | 0 | **b_p** | correctness precondition for U5+W7-edge |
| D | Flory coil | `--flory_unfolded --flory_nu` | 0 | **b_p** | **the lever**: dG MAE 4.9650 → 3.9521 |
| W5 | burial / solvation | `--burial_features --burial_mode {count,hse}` | **+3** | **BOTH** | zero in unfolded → dG; but −0.410 vs a_p 10/10 |
| W6 | AA descriptors | `--aa_descriptors mordred_pca16` | **+16** | **a_p** | per-residue, differs WT vs mutant → survives ddG |
| W7 | GCN span / bidir | `--gcn_span N --gcn_bidir` | 0 | **a_p** | changes which contacts reach a mutated residue |
| W7e | GAT edge attrs | `--edge_features` | 0 (edge_dim=56) | **a_p** | pairwise, mutation-sensitive via src/dst one-hot |
| W11 | ligand nodes | `--ligand_nodes --ligand_annotations` | **+10** | **b_p** | zero unfolded; cancels in ddG when ligand in both |
| A | WT anchor | `--wt_anchor_weight` | 0 | **b_p by construction** | `L1(pred_dG(WT), exp_dG(WT))` — a pure offset term |
| B | designed oversample | `--designed_weight` | 0 | **a_p** | reweights which proteins' within-protein spread is fit |
| C | slope term | `--slope_weight` | 0 | **a_p** | `\|std(pred_ddg) − std(true_ddg)\|` within protein |
| — | holdout | `--holdout_residues` | 0 | **neither** | a generalisation probe, not a lever |
| W9 | metal | *(retired)* | — | — | `_sibling_block_dims` **raises**; superseded by W11 |

**Width arithmetic, exactly as the code computes it** (`model/hydro_net.py:449-471`). Blocks are
inserted between Fb and emb so the right-anchored emb/one_hot slices never move:

```
solv_start = 48                                        # after D(16) + Fb(32)
desc_start = solv_start + solv_dim                     # 48 + {0,3}
lig_start  = desc_start + desc_dim + _sibling_block_dims(CFG)
```

Only `fc1_gcn` and `fc1_gat` grow:

```
fc1_gcn = Linear(52 + solv + desc + lig + proj_extra, 64)
fc1_gat = Linear(36 + solv + desc + lig + proj_extra, 64)
```

`fc2_*`, `inst_norm1` (36+proj_extra), `inst_norm2` (2×(36+proj_extra)) and
`fc_in_dim = 2×(36+proj_extra) + post_gnn_emb` are **fixed** and must not change — widening them was a
real bug. Total widths: baseline 1092; +W5 → 1095; +W5+W6 → 1111; +W5+W6+W11 → 1121.

---

## 3. Interactions

### 3.1 Coil × unfolded_emb — they do NOT compete; they act in opposite directions

The brief asks what `r_noemb=0.331` and `r_coil=1.175` imply jointly. First, what those numbers are:
they are **variance ratios of the unfolded energy across proteins**, `var(E_u)/var(E_u,base)` from
`results/w0.json` — not ddG correlations. The full row set:

| condition | var(E_u) | r vs base | corr(E_u, wt_err) | corr(E_u, length) |
|---|---|---|---|---|
| base | 0.9583 | 1.000 | 0.4201 | 0.4029 |
| `unfolded_emb=zero` | 0.3170 | **0.331** | **0.119** | **0.895** |
| `flory_unfolded` | 1.1263 | **1.175** | **0.507** | **0.144** |

They move **every diagnostic in opposite directions**. Zeroing the embedding *removes* reference-state
variance (0.331) and collapses the reference state onto **length** (0.895) — E_u becomes essentially a
chain-length counter. The coil *adds* variance (1.175) and **decouples** the reference state from
length (0.4029 → 0.144), which is exactly what a random-coil reference should do: a homopolymer
reference whose energy is not merely proportional to N.

So they are **not competing for the same variance**. They are two different reference states:
`unfolded_emb=zero` is a *lower-information* reference, the coil is a *differently-structured* one. The
joint prediction follows, and it is the one place a naive reading goes wrong:

- On raw dG MAE, `noemb` is the best single number in the table (3.1315 vs base 4.9650 vs
  `coil_fixed_b` 3.9521). **This is a trap.** Its `corr` with true dG collapses to **0.0510** (base
  0.3464, coil 0.3404). It wins MAE by killing the across-protein signal and landing near the mean —
  a lower error with no ranking left. `std_err` barely moves (0.9999 vs 1.0385).
- The coil buys its 1.01 kcal/mol **while keeping corr at 0.3404**, and it is the only condition that
  breaks the length coupling.

**Prediction, stated as a falsifiable claim:** `--flory_unfolded --coil_b fixed --unfolded_emb zero`
combined will show **sub-additive dG MAE gain and a corr collapse**. The two levers overlap in what
they remove (both strip folded-structure leakage from the reference) but `zero` removes the
per-residue identity the coil still needs to discriminate proteins. Expected joint MAE ≈ 3.0–3.3 with
corr < 0.15 — a *worse model* than the coil alone despite a better MAE. **The acceptance criterion must
therefore be a joint one (MAE and corr), never MAE alone.** This is precisely the metric-rule error the
project already made once, one metric later.

**Decision:** run the coil at `--unfolded_emb full`. Carry `mean` — not `zero` — as the only U2 arm
worth a cell, because `mean` preserves global scale while removing per-residue identity, and is the one
U2 setting not yet measured on dG. `zero` is a **diagnostic, not a candidate**.

### 3.2 Anchor × slope — they fight, and the mechanism is now identified

The brief predicts a fight from a 3-point trend. The prediction is **confirmed, and the mechanism is
stronger than a trend.** Measured from the per-protein CSVs, anchor ladder against its seed-42 control:

| anchor w | sd(`b_p_wt_error`) | median a_p | sd(`b_p`) | mean per-protein PCC |
|---|---|---|---|---|
| 0.0 (control) | 1.6737 | 0.4366 | 0.1947 | 0.7305 |
| 0.3 | 1.7795 | 0.5802 | 0.2656 | 0.7314 |
| 1.0 | 1.7701 | 0.5889 | 0.2767 | 0.7065 |
| 3.0 | **1.1704** | **0.1535** | 0.0621 | **0.6441** |

Read this carefully, because it contains a result that **reverses part of the brief's premise**:

- At w=0.3 and w=1.0 the anchor **does not shrink the offset at all** — sd(`b_p_wt_error`) goes *up*
  (1.674 → 1.780, 1.770). The anchor is b_p-side by construction, and at usable weights **it is not
  delivering on its own channel.**
- Only at w=3.0 does the offset shrink (1.170), and it does so by **crushing the slope**: median a_p
  collapses 0.44 → **0.15**, and per-protein PCC falls 0.731 → 0.644.
- That w=3.0 checkpoint is independently **pathological**: `a_min = 0.0187` (1/a_p = 53×) and **25 of
  its 28 proteins** fall below the 0.25 slope-oracle clip.

So the exposure→a_p decay (0.587 → 0.471 → 0.400) is not the anchor "suppressing a coupling" in a
neutral way. It is the visible edge of the anchor **destroying the slope channel** to buy offset. The
anchor converts a_p into b_p at a bad exchange rate.

**Therefore anchor and slope do not merely fight — they are antagonistic by construction**, and the
antagonism is measurable in a quantity neither was scored on. Both terms call the *same*
`get_wt_deltaG(batch)` (train.py:707 and :733) on the same minibatch: the anchor pins that scalar to
the label while the slope term needs the spread *around* that scalar to grow. Adding a shared-scalar
pin and a spread-expander to one loss is a tug-of-war on one number.

**Design consequence:** the A×C cell of the existing factorial is not an interesting interaction to
resolve — it is a **predictable loss**, and it is already half-observed. Do not spend seeds on A=1,C=1.

### 3.3 Slope (C) is not an accuracy lever at all — drop it from the accuracy track

Two independent results retire factor C as an accuracy lever:

- **Algebraic ceiling.** `a_p = r_p × sd(q)/sd(t)` exactly (verified to 3e-16 across 280
  protein-checkpoint cells). The std-matching term's unique zero is `sd(q)=sd(t)`, so **at its own
  optimum `a_p = r_p`**. It cannot see `a_p` except through `r_p`.
- **Per-protein PCC is exactly invariant to `pred/a_p`** (max |ΔPCC| = 7.77e-16). A perfect slope
  correction changes within-protein ranking by **exactly zero**. On pooled PCC the slope oracle is
  actively **destructive**: 0.599 → 0.470 on 10/10 checkpoints, worse than raw, and even the affine
  oracle (0.649) sits below plain offset removal (0.712).

Fixing a_p is a **calibration** result, not an accuracy result. It belongs in the thesis as a
calibration figure scored on `sd_ratio`, at a weight far below 3.0, and it must not be sold as a PCC
lever.

### 3.4 Burial (W5) × edge features (W7-edge) — related, not redundant

Both encode contacts, but at different orders and on different channels:

- W5 is a **node-level scalar**: how many Cβ neighbours residue *i* has. It is zero in the unfolded
  state, so the folded−unfolded difference is a hydrophobic burial term → **dG/b_p**.
- W7-edge is a **pairwise attribute** on the GAT graph: src one-hot(20) + dst one-hot(20) + 16
  **concatenated** RBF channels = `edge_dim` 56. The one-hot halves make it mutation-sensitive → **a_p**.

The overlap is that W5's neighbour count is (approximately) a function the GAT could compute by summing
over W7-edge's RBF channels. So they are **partially redundant in information, disjoint in channel**.
Keep both, but never claim them as independent evidence, and expect their *joint* effect to be
sub-additive on any shared channel.

One hard constraint the code enforces: `--edge_features` with `--flory_unfolded` **raises** unless the
U6 per-half `ca_coords` fix is importable (train.py:1073-1085). W7-edge and the coil are therefore
coupled through a correctness guard — the coil track must not casually add `--edge_features`.

### 3.5 The redundancy list

| pair | verdict |
|---|---|
| U2 `zero` × coil | **overlapping, opposite-signed.** Do not combine. `zero` kills corr. |
| A × C | **antagonistic by construction** (shared `get_wt_deltaG` scalar). Do not combine. |
| W5 × W7-edge | partially redundant in information, disjoint in channel. Keep both, expect sub-additivity. |
| W7 span × W7 bidir | **confounded by design** — `--gcn_span N` alone changes reach *and* direction. `--gcn_bidir` at span 1 is the honest control and must be run if span is run. |
| W6 × one-hot | `mordred_pca16` **appends**; `mordred_pca16_only` zeroes the trailing 20. These are alternatives, never both. |
| W11 × the 28 test proteins | **inert by construction.** All 28 are single-chain, ligand-free, metal-free monomers. W11 is a **generality** lever; scoring it here can only produce a null. |

---

## 4. Is the planned W5 × W6 × W7 factorial (2³ × 3 seeds = 24 runs) right?

**No. It is the wrong design, for four separable reasons, and it should be cut to 8 runs.**

**(a) Its acceptance criterion is on the wrong metric for one third of its own factors.** `EFFECT_MIN`
in the S7 factorial is defined on **pooled ddG**. W5 is dual-channel and its dominant, largest effect
is on dG/`b_p_wt_error` (−0.475, 10/10). Scoring W5 on pooled ddG is the Flory-coil error one lever
later — the project's signature failure mode, already documented against this exact factorial.

**(b) W6 is the weakest-motivated factor in the set, and its own author's argument does not apply
here.** Ofir concedes physicochemical properties are already implicit in LM embeddings, so his case for
descriptors reduces to **coverage of non-canonical residues**. **All 28 test proteins are canonical.**
W6 therefore has no mechanism to help on this eval set: 16 columns derived deterministically from the
one-hot (`one_hot @ table`) are a **fixed linear re-encoding of information already in the input**, fed
to a layer that immediately learns a linear map of it. On canonical-only data W6 is close to a
capacity-only change. It is a **thesis-completeness** lever, not an accuracy lever.

**(c) W7 as specified is internally confounded.** `--gcn_span N` changes reach and directionality
simultaneously; without the `--gcn_bidir`-at-span-1 control arm, any W7 effect is uninterpretable. A
2³ design has no room for that control, so the W7 axis as drawn cannot produce a clean answer.

**(d) A 2³ × 3-seed design is under-powered for what it is being asked to detect.** Seed-to-seed spread
of pooled PCC across the five sigma seeds is 0.5799–0.6575 — a **range of 0.078**. With 3 seeds per
cell, the standard error on a cell mean is roughly 0.03/√3 ≈ 0.017, so a main effect must be ≳0.04
pooled PCC to be seen. Neither W6 nor W7 has any prior predicting an effect that large. The design
would spend 24 GPU runs to return three nulls that cannot be distinguished from seed noise.

**Verdict: drop W6 and W7 from the confirmatory factorial.** Replace the 2³×3 = 24 runs with a
**one-factor-at-a-time W5 arm on the correct metric, 4 runs**, plus a **4-run W6/W7 screening arm** that
is explicitly labelled a screen, not a test.

---

## 5. What replaces it — the ordered plan

Everything below is built on the **winning calibration cell**, never the bare baseline, or the two
result sets cannot be combined (this is already the project's stated rule).

### Track 1 — the b_p / dG track (this is where the kcal/mol is)

**S-B1 — the coil confirmation, 4 runs (2 arms × 2 seeds).**
The 3.9521 number is an **evaluation-time** measurement on a checkpoint trained *without* the coil. It
has never been trained. That is the single largest untested claim in the project.

```
# arm 1 (coil), seeds 42,1
python Megascale-fineTuning/train.py \
  --full_data --no_pretrain --no_freeze \
  --loss_mode dg \
  --flory_unfolded --flory_nu 0.5 --coil_b fixed --coil_channels broadcast \
  --unfolded_emb full \
  --val_frac 0.1 --epochs 15 --seed {42,1} --run_tag sB1_coil_seed{42,1}

# arm 2 (control), seeds 42,1 — identical minus the coil
python Megascale-fineTuning/train.py \
  --full_data --no_pretrain --no_freeze \
  --loss_mode dg \
  --unfolded_emb full \
  --val_frac 0.1 --epochs 15 --seed {42,1} --run_tag sB1_ctrl_seed{42,1}
```

Note `--loss_mode dg`. This is mandatory for the coil track and it is *why* the slope term cannot ride
along (train.py:231 raises). Scored on **dG MAE and `std(b_p_wt_error)` and corr** — jointly.

**S-B2 — W5 burial on the correct metric, 4 runs (2 arms × 2 seeds).** Same `--loss_mode dg`
skeleton, `--burial_features --burial_mode hse` vs off. `hse` over `count` because half-sphere exposure
is direction-aware and needs only CA and CB — exactly what this backbone-only dataset has. Width
1092 → 1095; `fc1_gcn` 52→55 inputs, `fc1_gat` 36→39. Scored on **dG MAE, `std(b_p_wt_error)`, and
additionally on `a_p`** because W5 is dual-channel.

**S-B3 — the coil × W5 cell, 2 runs.** Only if S-B1 and S-B2 both clear their gates. Both are b_p-side
and both strip the same hydrophobic/reference confound, so the prediction is **sub-additive**. Stating
that prediction in advance is what makes the cell worth running.

### Track 2 — the a_p / ddG track

**S-A1 — the W6/W7 screen, 4 runs, labelled a screen.** `--loss_mode ddg`, on the winning calibration
cell, 1 seed each: (i) `--aa_descriptors mordred_pca16`, (ii) `--gcn_span 4`, (iii) `--gcn_bidir` at
span 1 *(the honest control for (ii))*, (iv) `--edge_features`. Any arm not clearing seed noise
(≳0.04 pooled PCC) is reported as **"screened, no effect detected at this power"** — never as a null
result. `--gcn_span 4` is chosen over 2 because with three layers the baseline reach is three residues,
so an α-helix i→i+4 contact is currently unrepresentable; span 4 is the smallest span that makes it
representable.

**S-A2 — the slope calibration figure, 0 new runs.** The weight ladder {0.3, 1.0, 3.0} is already
trained (15 checkpoints each) and the two missing eval jobs `ev_p3_slope0.3_s42` and
`ev_p3_slope1.0_s42` are already queued. **Wait for those CSVs; submit nothing.** Report on `sd_ratio`
as calibration.

### Not run

- **A×C cells** — antagonistic by construction (§3.2), predictable loss.
- **W11** — inert on 28 ligand-free monomers; a generality lever, documented as such, and its
  header-only `ligand_sites.csv` is a legitimate "this dataset is ligand-free" delivery.
- **`--unfolded_emb zero`** as a candidate — diagnostic only; it destroys corr (0.0510).
- **W9** — retired; `_sibling_block_dims` raises rather than silently shifting the W11 slice 9 columns
  into the embedding.

**Total: 14 GPU runs, against the 24 originally planned for W5×W6×W7 alone** — and the 14 answer
questions on the metrics that can move, while the 24 would have answered one question on the wrong
metric and two on under-powered axes.

---

## 6. The two bets — and they are NOT the same combination

This is the point of the whole document, so it is stated without hedging.

### Bet 1 — best pooled ddG PCC

```
--loss_mode ddg \
--wt_anchor_weight 0.3 \
--designed_weight 3 \
--slope_weight 0 \
--unfolded_emb full \
--aa_descriptors none \
--gcn_span 1
```

**Rationale.** Pooled PCC is dominated by *between*-protein offset (offset-removal oracle 0.712 vs raw
0.599; the ranking is already at per-protein PCC 0.798). So the winner is the cell that removes offset
**without** paying for it in slope. The ladder says w=0.3 is the only anchor weight that improves mean
per-protein PCC (0.7314 vs control 0.7305) while the offset term is active at all; w=1.0 costs PCC
(0.7065) and w=3.0 is pathological (0.6441). `--slope_weight 0` because C is provably
ranking-neutral and empirically destructive. `--designed_weight 3` is the one a_p-side lever with no
measured downside.

**Honest expected value: pooled PCC ≈ 0.60–0.62, i.e. roughly the current 0.599 ± seed noise.** No
built lever has a mechanism to reach the 0.712 oracle, because that oracle is fitted on test labels.
**Anyone promising a large pooled-PCC gain from these levers is over-claiming.**

### Bet 2 — smallest std(b_p)

```
--loss_mode dg \
--flory_unfolded --flory_nu 0.5 --coil_b fixed --coil_channels broadcast \
--unfolded_emb full \
--burial_features --burial_mode hse \
--wt_anchor_weight 3.0 \
--slope_weight 0        # mandatory: --slope_weight with --loss_mode dg RAISES
```

**Rationale.** Every term here is b_p-side. The coil at fixed b is the largest measured single lever
(4.9650 → 3.9521). W5 adds the hydrophobic burial term that is zero in the unfolded state. The anchor
at **3.0** — the one weight that actually shrinks the offset (1.674 → 1.170) — is acceptable *here*
precisely because this run is not being judged on slope.

**They are different combinations, and they differ in the direction of one shared lever.** The anchor
is at 0.3 in Bet 1 and 3.0 in Bet 2. That is not a tuning detail; it is the a_p↔b_p exchange rate of
§3.2 being resolved in opposite directions because the two bets are scored on opposite channels. **A
single "best model" does not exist here**, and the thesis should present the trade-off curve rather
than pretend one cell wins both.

---

## 7. The gate

`scripts/gate_g4_cpu.py` must assert, and does assert today:

1. `dG=-0.0030 width=1092` on the baseline cell — the exact numeric anchor.
2. Every lever that adds no dimensions leaves width at **1092** (U2, coil, U3/U4/U5, W7 span/bidir,
   W7-edge, and every loss-side factor A/B/C).
3. `--burial_features` adds **exactly 3** → 1095. By extension, to be added with the arms above:
   `mordred_pca16` adds exactly 16 → 1111; `--ligand_nodes` adds exactly 10.
4. A real `PEM` forward on a real graph for **every** lever combination, so a wrong `fc1_*` width raises.
5. `dG` finite in every cell.

**The anti-silent-no-op check.** A wrong *width* raises; a wrong *order* does not. Order errors are the
dangerous class here, because `hydro_net` slices right-anchored and a mis-ordered block silently shifts
the read into the ProtT5 embedding. The gate must therefore additionally assert, and this is the
central requirement:

- **Block-position assertions, not just widths.** With W5 on, the 3 solvation columns must be readable
  at exactly `[48:51]`; with W5+W6, the descriptor block at `[51:67]`; with W5+W6+W11, the ligand block
  at `[67:77]`. `gate_ligand.py` gate E already checks all sixteen on/off combinations of the sibling
  blocks — that pattern is the model.
- **Right-anchored invariance.** `emb` must remain at `[-1044:-20]` and `one_hot` at `[-20:]` under
  **every** combination. This is the assertion that catches an inserted block that silently ate into
  the embedding.
- **Non-degeneracy of every new column.** For each added block, assert `std > 0` across nodes on a real
  graph. A block that is constant does not raise — it trains a bias and reports as a null. This is the
  documented W5 units bug: scaling Å coordinates by 0.1 saturated every neighbour count to exactly 1.0
  with std 0.0, producing `nan`/meaningless-zero rather than an error.
- **Off ⇒ byte-identical.** Every lever's default-off path must produce bitwise-identical output to
  baseline, so an unset flag can never be a silent partial application.
- **Guards fire.** `--slope_weight > 0` with `--loss_mode dg` raises; `--coil_edges` without
  `--flory_unfolded` raises; `--ligand_nodes` without `--ligand_annotations` raises; `--metal_features`
  raises; `PEMGraphTransformer` raises when any block is inserted; `--edge_features` with
  `--flory_unfolded` raises absent the U6 fix.
- **Zero-variance refusal at scoring time.** Refuse to score any lever whose feature is constant on the
  eval set, rather than reporting the resulting ~0 as a null.

---

## 8. Cost

| item | runs | GPU |
|---|---|---|
| S-B1 coil confirmation | 4 | ~4 × 8 h |
| S-B2 W5 on dG | 4 | ~4 × 8 h |
| S-B3 coil × W5 | 2 | ~2 × 8 h |
| S-A1 W6/W7 screen | 4 | ~4 × 8 h |
| S-A2 slope calibration | **0** | already queued |
| **total** | **14** | **~112 GPU-hours** |

Against the original W5×W6×W7 (24 runs, ~192 GPU-hours) plus the unrun A×C cells: **this plan is
~40% cheaper and answers the questions on metrics that can actually move.**

Site rules that are already paid for in wasted runs: `export WANDB_MODE=disabled` before every
submission; `--qos normal --gres=gpu:rtx_6000:1`, **never** select a GPU by partition name (the submit
filter reroutes onto a GTX 1080 and OOMs); exclude `ise-pheno-*`. **A finished training run is not a
result — completion is a non-empty CSV in `eval_results/`.**

---

## 9. Verdict

**DO-IT**, with the plan as amended: 14 runs, not 24.

The single highest-value item is **S-B1**, the coil confirmation, because the 1 kcal/mol headline has
only ever been measured at evaluation time on a checkpoint that was never trained with the coil. It is
simultaneously the project's largest claim and its least-tested one. Everything else in this document
is subordinate to getting that one number trained.

The single most important *negative* result is that **factor C (slope) is not an accuracy lever** — it
is ranking-neutral by algebra, destructive by oracle, and already failed once by experiment. Retiring
it from the accuracy track is worth more than any of the additive levers, because it stops the project
spending GPU on a channel that provably cannot buy the metric it was being spent for.
