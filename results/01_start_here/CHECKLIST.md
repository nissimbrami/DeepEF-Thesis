# THE CHECKLIST — built from the FULL review (Parts A-G), not just Part E

**Rebuilt 2026-09-10** after the first version was found to cover only Part E's work order and to
drop Parts A-D and F-G entirely. **One task active at a time. GPU jobs in the background are fine.**

Inner checklist for every task:
`[ ] PLAN` (question / hypothesis / falsifier / metric / config / "if X then Y" written in advance)
`[ ] EXECUTE` (end to end, no deviation)
`[ ] VERIFY` (criterion met? did it train? did the predicted mechanism occur?)
`[ ] DONE` (mean AND median + training-health line)

Status: `[ ]` open · `[x]` done · `[~]` blocked · `[>]` active

---

# PART A — the four structural errors (must be fixed as POLICY, not as tasks)

- [x] **A1 screen vs confirm never mixed.** A screen is 1 seed, NO multiplicity correction, and
      yields a ranking only. Confirmation is 5 seeds with one pre-registered test at 0.05.
      -> in `PROTOCOL.md` §2 Q5. **Consequence: `slope 0.7` was a screen and must NOT have been
      BH-corrected. Its verdict is "leading candidate", not "miss".**
- [x] **A2 power before submission.** Control SE 0.0027, single-arm SE 0.0065, difference SE
      ~0.0070; a 0.011 effect is 1.6σ at n=1. **3 seeds minimum for a claim, 5 for a headline,
      7 for 3σ.** -> `PROTOCOL.md` §2 Q6.
- [x] **A3 "collapsed" and "harmful" are different verdicts.** Every arm reports a training-health
      line before its score; a frozen loss is a build failure, not a result. -> `PROTOCOL.md` §4.
- [x] **A4 rejections are written "rejected in configuration X".** -> `PROTOCOL.md` §3.

---

# PART B — the six levers, each with the review's own "what to do"

## B1 Flory / the coil
- [x] read the loss curves of both Flory arms  -> HEALTHY (1.450->1.086, 2.313->1.628)
- [x] confirm only the map was swapped -> docstring "value-only lever, no new parameters";
      map is `b*|i-j|^nu`, never reads one_hot; zero learnable params -> `FLORY_QUARTER_TESTED.md`
- [x] **B1c DONE V** - ratio is 1.45 (nu=.5) and 1.29 (nu=.588): NOT a scale problem, `b`
      recalibration ruled out by measurement. Real mechanism found: SPARSITY. Baseline is a hard
      tridiagonal mask (3.3% of pairs nonzero at 0.3150); the coil smears the same mass over 39%
      of pairs at 0.0391 each - 12x denser, 8x weaker per contact. Next config: a TRUNCATED coil
      (cutoff restores sparsity, keeps the distance profile). -> `B1C_RESULT.md`
- [x] **B1d DONE V** - the flag was a FIFTH SILENT NO-OP: DistogramHead was absent from the
      live model (grep count 0 in hydro_net.py / train.py / train_utils.py) and
      --distogram_weight was only parsed, stored and logged. Proof: three arms spanning a
      100x weight range gave identical RMSE trajectories. ~32 GPU-hours trained a baseline
      under four names. BUILT IT: head copied into hydro_net.py, instantiated in
      PEM.__init__ only when weight>0 (SLOPE_WEIGHT guard pattern), loss computed on the
      UNFOLDED graph inside get_deltaG, consumed at the loss site, weight 0.1 not IFUM's 100.
      Four integration faults fixed: indentation, out-of-scope var, missing n_folded=0,
      CUDA OOM (fixed by a single-protein forward - all variants share the unfolded state).
      VERIFIED: gate_g4_cpu ALL PASS after every fix; B=64 shape test loss 3.4536 finite,
      |grad| 31.60; untrained loss 3.4800 vs ln(32)=3.4657 = random guess, as expected;
      job 21175008 running clean at 19/306 steps. -> `B1D_DISTOGRAM_NOOP.md`
      CAVEAT: target is the coords the model receives. If the ens* arms show sampling works,
      the target should move to SAMPLED conformations (predicting a formula teaches nothing).
- [x] representation mismatch recorded: our coil broadcasts one CA distance across all 16
      atom-pair channels, making them identical. -> `FLORY_QUARTER_TESTED.md`

## B2 descriptors
- [x] `torch.equal(folded, unfolded)` -> **True**, max|d|=0.000e+00 = ROOT CAUSE
- [ ] **B2b confirmation run: zero the block in the UNFOLDED pass only, one epoch. If RMSE moves
      off 2.529, proved.**
- [x] the rule it generalises to -> `gate_state_dependence.py`
- [x] **B2d DONE V** - an edge path ALREADY exists: edge_features.py builds
      [src_onehot(20) | dst_onehot(20) | rbf(16)] fed to GATv2Conv's LINEAR lin_edge.
      Measured: hydropathy DIFFERENCE R2=1.0000 and volume SUM R2=1.0000 from the existing
      edge_attr -> both REDUNDANT (lin_edge absorbs any table T). Distance is already the
      RBF block. Only a PRODUCT h_src*h_dst is outside the span (R2=0.1970).
      Corrected item: the gap is NOT 'edge descriptors' but 'no MULTIPLICATIVE pair term
      exists anywhere'. Same shape as W5 (bur*hyd), W15 col2 (reach*env) and W12 - every
      mechanism with signal in this project is multiplicative. Arm specified, not run:
      w7edge is null across 2 seeds, so the edge channel is inert today.
      -> `B2D_EDGE_DESCRIPTORS.md`

## B3 W5 burial
- [~] **B3a NOTHING may be concluded about W5 until `pub_w5_ddg_s42` lands.** All four W5 seeds
      ran `--loss_mode dg`; every dG-trained arm has a negative median, so the objective and the
      block are confounded. (arm training now)
- [x] **B3b DONE V** - prediction HALF confirmed. 19/27 proteins, 76,416 mutation-seed pairs.
      Buried mean gain +0.03697 vs exposed +0.01051; difference +0.02647, Welch t=4.933,
      p=1.1e-06 -> the direction is real. BUT the buried MEDIAN is -0.00406 against a mean of
      +0.037 (the same split as W5 overall); corr(burial,gain)=+0.065 explains 0.4% of the
      variance; and it helps POLAR destinations MORE than hydrophobic (+0.047 vs +0.024), the
      opposite of desolvation. The mechanism story does NOT hold. Rejection stands, better
      characterised: W5 rescues a tail at a cost to the majority, at every stratum.
      Found a real join bug: eval CSVs are float32, mutation files float64 - an exact deltaG
      join returns 4 of 1703 rows. B3a still binds (all 4 seeds are loss_mode dg).
      -> `B3B_BURIED_VS_EXPOSED.md`
- [x] report mean AND median always -> `PROTOCOL.md` §4

## B4 W7 edges
- [x] verify the kernel is monotone -> **it is** (0.726 -> 1.5e-08), so the nulls are genuine
- [x] the span/bidir confound claim -> **WRONG**, flags are independent; u10bidir IS the control
- [x] **B4c DONE V (same as B2d)** - closed by measurement: hydropathy difference and
      volume sum are fit at R2=1.0000 from the existing edge_attr, so lin_edge absorbs
      any table T; distance is already the RBF block. Only a PRODUCT h_src*h_dst is
      outside the span (R2=0.1970). Arm specified, not run - w7edge is null across 2
      seeds so the edge channel is inert today. -> `B2D_EDGE_DESCRIPTORS.md`
- [x] correlate BSA against `b_p`/`a_p` on the 27 test proteins, not against the pre-training loss
      -> interface NULL for b_p; **packing_frac -0.662 and void_vol_per_res +0.652 vs a_p survive
      Bonferroni** -> `T4_INTERFACE_NULL.md`

## B6 ligands / metals
- [x] recorded as "not measurable on this dataset", never "dead"
- [ ] **B6b decide explicitly:** either evaluate on ProTherm/FireProtDB (noting S669 leakage binds
      only if S669 stays the test set — a decision, not a fact), or record the direction as
      untestable here and say so in the thesis. **This is Nissim's call, not mine.**

---

# PART E — the work order

- [x] E1 training curves (= B1a, B2a)
- [x] E2 descriptor block folded vs unfolded (= B2a)
- [x] E3 kernel monotonicity (= B4a)
- [x] E4 BSA vs b_p/a_p (= B5)
- [~] E5 `pub_w5_ddg_s42` — training on GPU
- [ ] **E6 `slope 0.7` at 5 seeds** — the only positive candidate the wave produced. **It is a
      SCREEN result; treat as leading candidate (A1).**
- [x] **E7 DONE V** - falsifier FIRED. W12 pooled 0.6175 (2 seeds) vs control family
      0.5757 +/- 0.0314 = **+1.33 sd only**, INSIDE the pooled seed band, while the ranking
      channel is +3.19 sd. W12 is an information gain WITHOUT a headline pooled number,
      because pooled is dominated by b_p (96.3% one global constant, unpredictable by 10
      methods). Do NOT headline 0.6175: it is below the best single canonical run 0.6382,
      which is a control seed with no lever. Report as: ranking +0.0207 (+3.19 sd, Wilcoxon
      p=0.0104), pooled +0.042 NOT established.
      ALSO FLAGGED: recomputing the declared 9-CSV canonical population gives oracle
      0.7327 +/- 0.0126 against the recorded 0.7156 +/- 0.0474 - a 4x sd discrepancy, not
      rounding. Not fixed here (separate decision, gate_headline.py enforces the basis).
      -> `E7_W12_CANONICAL.md`
- [x] **E8 DONE V** - MECHANISM CONFIRMED on both predicted axes.
      PER PROTEIN (n=27): corr(W12 gain, packing_frac)=+0.4126 p=0.0325; void -0.4138 p=0.0319.
      Tightly packed proteins gain +0.0366 vs +0.0035 loose = 10x. T4 predicted this in advance
      (packing predicts slope collapse -0.662), so it is confirmation, not discovery.
      PER MUTATION (37,852 pairs, 19 proteins): to HYDROPHOBIC +0.02894 (median +0.0023) vs to
      POLAR +0.00341 (median -0.0060) = 8.5x. Buried+hydrophobic (the FINDINGS 3.4 deficit)
      +0.02687. Burial ALONE is not the discriminator (p=0.36) - destination chemistry is.
      DECISIVE CONTRAST: W5 helps POLAR more (+0.047 vs +0.024); W12 helps HYDROPHOBIC 8.5x
      more. Both were built to attack burial; only W12's gain matches its intended mechanism.
      -> `E8_W12_MECHANISM.md`
- [~] E9 W15 — `pub_w15b_s42` resuming on GPU
- [~] E10 distogram head at 0.1 (= B1d) — `disto0.01/0.1/1.0` on GPU

---

# PART F — the 19 lessons about the MODEL (kept because they survive the levers)

Recorded in `results/02_findings/MODEL_LESSONS.md`. These are what remains when a lever falls.

---

# PART G — protocol

- [x] one active task; GPU background jobs are not tasks
- [x] plan / execute / verify, uninterrupted, with a written definition of DONE
- [x] the 8-question pre-flight test (4 from the review + 4 the wave exposed)
- [x] written to memory so it survives compaction

---

**Rule: nothing below the active task is touched until it is DONE.**
