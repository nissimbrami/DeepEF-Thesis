# INFO_LEVERS — the four information levers that had never been run

Collected 2026-09-08. All four trained 15 epochs (0–14), seed 42, `--full_data --no_pretrain
--no_freeze --pooled_corr_weight 0 --dg_length_norm none --affine_calib --val_frac 0.1`,
scored on the untouched 28-protein / 28,314-variant test set through the trusted
`evaluate.py` path with each arm's own val-fitted affine sidecar.

**Headline: all four are negative on information. None of them adds ranking power.**
Three of the four *look* positive on pooled ddG PCC; that gain is rescaling, not information,
and §3 shows the arithmetic that separates the two.

---

## 1. What was run, and proof the flag was actually read

The signature failure mode of this project is a flag that is parsed, printed, and never
reaches the model. Each arm was verified at the level of *artifacts*, not success lines.

| arm | flag (from the submit script) | proof it changed the model |
|---|---|---|
| `gld_w7span4` | `--gcn_span 4` | `gate_w7`: span 4 builds **70 directed edges** vs baseline **19** on N=20; reaches i→i+2 and i→i+4, baseline reaches **only i→i+1** |
| `gld_u10bidir` | `--gcn_bidir` | `gate_w7`: bidirectional gives **38** edges vs **19** (exactly 2(N−1)) |
| `gld_w7edge` | `--edge_features` | `gate_w7_edge`: ON constructs **92,064** edge parameters, OFF constructs **0**; checkpoint carries `GAT_layers.*.gat*.lin_edge.weight` |
| `gld_w5_dg` | `--burial_features --loss_mode dg --flory_unfolded --coil_b fixed` | checkpoint `fc1_gcn` is **[64,55]** vs default **[64,52]** — burial's exactly 3 columns (bur, hyd, bur×hyd) |

Submit scripts: `/home/nissimb/bs_2108459{5,6,7,8}.sh`. Flags reach `CFG` in `train.py:125–132`
and are read in `model/hydro_net.py:690–691` (span/bidir), `:448` (burial `solv_dim=3`), and
`PEM.__init__` (edge channel).

**Two arms could not be scored by the standard path at all.** `evaluate.py` has no
`--burial_features` and no `--edge_features` flag, so it rebuilt the *default* architecture and
died on a state-dict mismatch (`[64,55]` vs `[64,52]`; `Unexpected key … lin_edge.weight`).
Fixed with two read-only wrappers, `scripts/eval_w5.py` and `scripts/eval_w7edge.py`, that set the
CFG flags the arm was trained with before the model is constructed. **Nothing in the shared tree
was modified**; `gate_g4_cpu` prints `dG=-0.0030 width=1092` before and after.

A third failure is worth recording because it is the exact signature mode: the first u10bidir
scoring attempt was **`Terminated`** by the login node after one protein at 247 s/it, and
`run_calib_eval.sh` still printed `DONE … -> abl_gld_u10bidir_s42_e14.csv` **with no CSV on disk.**
All scoring was moved to the CPU partition and every result below was confirmed by the presence
and row count of the CSV (28 proteins / 28,314 rows, identical to control), never by a log line.

---

## 2. The numbers

Control = `abl_calib_ctrl_repro2_e14`. `a_p`, `r`, `s` are medians over the 28 proteins;
`a_p = r · s` by construction.

| arm | epoch | pooled ddG PCC | per-protein PCC | a_p | r | s | std(b_p) | dG MAE |
|---|---|---|---|---|---|---|---|---|
| **control** | e14 | 0.5910 | 0.7955 | 0.4975 | 0.7955 | 0.6342 | 1.5741 | 1.3029 |
| W7 span 4 | e12 | 0.6107 | 0.7933 | 0.3204 | 0.7933 | 0.4410 | 1.2405 | 1.0534 |
| U10 bidir | e14 | 0.6246 | 0.7891 | 0.2463 | 0.7891 | 0.3108 | 1.0964 | 1.0309 |
| W7 edge | e14 | 0.6006 | **0.8017** | 0.3284 | **0.8017** | 0.4181 | 1.2473 | 1.1176 |
| W5 burial (dG) | e14 | 0.5956 | 0.7794 | **0.6497** | 0.7794 | **0.8552** | 1.5549 | 1.1880 |

Epochs differ because `run_calib_eval.sh` selects the best epoch by **val ddG PCC**; W7span's
best was e12, the others e14.

### The seed-noise band — what any of this has to clear

Five same-config seeds (the `abl_sigma_seed*` replicates behind the ICC(b_p)=0.898 result)
give, on identical data:

| metric | mean | sd | **2σ band** |
|---|---|---|---|
| pooled ddG PCC | 0.6052 | 0.0302 | **±0.060** |
| per-protein PCC | 0.7916 | 0.0024 | **±0.005** |
| a_p | 0.3879 | 0.0515 | **±0.103** |
| s | 0.5155 | 0.0711 | **±0.142** |
| std(b_p) | 1.3919 | 0.1530 | **±0.306** |
| dG MAE | 1.1168 | 0.2049 | **±0.410** |

*Caveat:* those seeds are unanchored and lack `--affine_calib`, so this is the right order of
magnitude rather than an exact null. It is used below only to reject effects, never to claim one.

---

## 3. The metric rule, applied arm by arm

**W7span, U10, W7-edge — judged on ddG.** All three change the *folded* contact graph, which
genuinely differs between WT and mutant, so the WT/mutant difference does not cancel and ddG is a
legitimate metric for them. **W5 burial — judged on dG and std(b_p), never pooled ddG.** It acts
on the reference state and was deliberately trained with `--loss_mode dg --flory_unfolded`;
its ddG numbers are reported above for completeness but are *not* the basis of its verdict.

### 3.1 The ddG arms: pooled PCC went up, and it means nothing

W7span +0.020, U10 +0.034, W7-edge +0.010 on pooled ddG PCC — all **inside the ±0.060 seed band**
before any further argument. But the decisive test is *where* the movement came from.

Because `a_p = r·s`, a lever can raise pooled PCC either by improving ranking (`r`) or merely by
rescaling predictions (`s`). Paired over the same 28 proteins:

| arm | Δr | t | arm>ctrl | Δs | t | arm>ctrl |
|---|---|---|---|---|---|---|
| W7span 4 | **+0.0029** | +0.58 | 14/28 | −0.2092 | **−10.73** | **0/28** |
| U10 bidir | **+0.0091** | +1.29 | 13/28 | −0.3581 | **−12.49** | **0/28** |
| W7 edge | **+0.0039** | +0.70 | 11/28 | −0.1986 | **−11.15** | **0/28** |
| W5 burial | **+0.0035** | +0.23 | 11/28 | +0.1969 | +4.20 | 22/28 |

**Δr is indistinguishable from zero in every arm** — every t below 1.3, every win-rate a coin
flip. Meanwhile Δs is enormous and *unanimous*: 0 of 28 proteins improved for the three ddG arms.
That is a systematic compression of the prediction spread, not noise, and it is the only thing
these levers did.

Confirming it directly — recompute pooled PCC after z-scoring each protein's predictions
(removes per-protein scale and offset, preserves ranking):

| arm | pooled | after per-protein z-score | surviving gain |
|---|---|---|---|
| control | 0.5910 | 0.5812 | — |
| W7span 4 | 0.6107 | 0.5854 | **+0.004** |
| U10 bidir | 0.6246 | 0.5887 | **+0.008** |
| W7 edge | 0.6006 | 0.5852 | **+0.004** |
| W5 burial | 0.5956 | 0.5826 | **+0.001** |

Gains of +0.010…+0.034 collapse to **+0.001…+0.008** once rescaling is removed. The pooled
improvement was a scale artifact.

**Degenerate baseline.** Predicting each protein's **mean ddG** for every one of its mutations —
using zero within-protein information — already scores **pooled 0.5574**. The control's 0.5910 sits
only +0.034 above that. Pooled ddG PCC is dominated by between-protein structure and is a weak
discriminator; this is precisely why per-protein PCC (0.798) is the number that carries the thesis.

### 3.2 W5 burial, on its own metric

- **dG MAE 1.1880 vs control 1.3029** — better by 0.115, but that is **0.28σ** of the ±0.410 band,
  and it is **still worse than predicting a constant (1.0931)**. The degenerate baseline defeats
  both the lever and the control.
- **std(b_p) 1.5549 vs control 1.5741** — a change of **0.019**, against a ±0.306 band. The lever
  built to attack the per-protein offset **did not move it**.
- Its affine sidecar is near-identity (a=0.9724, b=0.0212) versus the control's a=0.6976, i.e. the
  dG loss did put the predictions on the right scale — it simply bought no accuracy.
- Its a_p of 0.6497 is the largest in the table, and under the metric rule it is **not** evidence
  for this arm: a_p is a ddG-side quantity and W5 was trained on dG. Read as an observation only,
  it is consistent with §3.1 — s rose to 0.8552 while r stayed at 0.7794.

---

## 4. Ranking by effect size, and the call

Ranked by *real* effect — surviving pooled gain and Δr, not raw pooled PCC:

| rank | arm | real effect | verdict |
|---|---|---|---|
| 1 | **W7 edge** (`--edge_features`) | per-protein PCC **0.8017 vs 0.7955 (+0.006)**, the only arm above control; Δr +0.0039 (n.s.); +0.004 surviving | **Marginal — the only one worth a second look.** Its +0.006 sits just past the ±0.005 band, and its val curve (0.808 final vs control 0.797) agrees. But Δr is not significant and it costs 92k params and ~3 min/protein at inference. **Do not carry into the final model on this evidence; re-run at 2–3 seeds if anything is carried at all.** |
| 2 | **U10 bidir** (`--gcn_bidir`) | largest raw pooled gain (+0.034) and the largest s collapse (−0.358, 0/28) | **Drop.** The pooled gain is the rescaling artifact in its purest form; per-protein PCC actually *falls* (0.7891). |
| 3 | **W7 span 4** (`--gcn_span 4`) | +0.020 pooled, Δr +0.003 (n.s.), s −0.209 (0/28) | **Drop.** Same mechanism as U10, smaller. Reaching i→i+2/i→i+4 is representationally real (gate-verified) and bought nothing. |
| 4 | **W5 burial on dG** (`--burial_features --loss_mode dg`) | dG MAE −0.115 (0.28σ, still beaten by a constant); std(b_p) −0.019 on a ±0.306 band | **Drop.** Judged on the metric it acts on, it is a clean null: it did not shrink the offset spread it was designed to shrink. |

**Nothing here should be carried into the final model.** W7-edge is the sole arm that is not
flatly negative, and it is one marginal seed.

---

## 5. Why this is a coherent result, not four accidents

The four arms attack the folded contact representation (span, bidirectionality, edge distances)
and the reference state (burial). Every one of them left `r` — the within-protein ranking
correlation — **statistically unchanged**: +0.003, +0.009, +0.004, +0.004, no t above 1.3.

That is the same wall the project has already documented from a different direction: the model is
worst at burying hydrophobics, with corr(slope, KD) = −0.734 **and** corr(spearman, KD) = −0.740
moving in lockstep, i.e. lost information rather than rescalable miscalibration — because only 4
backbone atoms are stored and **there are no side chains** in the input. Richer *backbone* topology
cannot recover side-chain packing information that was never in the tensors. These four negatives
are that constraint showing up four more times, and they localize the remaining headroom: the
per-protein offset b_p (std 1.5741, ICC 0.898, still unexplained by every cheap hypothesis) and
the input representation itself — not the graph wiring.

---

## 6. Provenance

- CSVs: `eval_results/abl_gld_{w7span4_s42_e12,u10bidir_s42_e14,w7edge_s42_e14,w5_dg_s42_e14}.csv`
- Control: `eval_results/abl_calib_ctrl_repro2_e14.csv`
- Scorers (all new, all read-only w.r.t. the shared tree): `scripts/score_levers.py` (reproduces the
  published control triple exactly: pooled 0.5910, per-protein 0.7955, a_p 0.4975, r 0.7955,
  s 0.6342, std(b_p) 1.5741), `scripts/perprot_compare.py`, `scripts/pooled_decomp.py`,
  `scripts/degen_baseline.py`, `scripts/degen_ddg.py`
- Eval wrappers: `scripts/eval_w5.py`, `scripts/eval_w7edge.py`
- Gate: `scripts/gate_g4_cpu.py` → `baseline … PASS dG=-0.0030 width=1092`, re-run after every step
- n=28 throughout; |r| < 0.374 is indistinguishable from zero at this n
