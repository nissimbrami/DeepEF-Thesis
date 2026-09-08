# TASKS — main session only, no background agents. Loop: task -> test -> record -> next.

## OPEN
> **CANONICAL BASIS (P8).** Every headline number in this document is computed on:
> **27 test proteins (2K5H excluded) | ddG metric | the 9 original-population eval CSVs |
> the val-selected epoch only.** That basis is `pooled 0.5772 / oracle 0.7156 / gain +0.1384`.
> Membership, exclusions and provenance: `results/05_infrastructure/HEADLINE_BASIS.md`.
> Enforced by `scripts/gate_headline.py`. **Never average across runs that differ in factor D**
> (`--unfolded_emb zero`) — D0 and D1 are different models; report them separately, always.


- [x] P0a **DONE — the headline a_p was PEEKED.** Canonical epoch e10 gives a_p 0.782, not the
      0.857 quoted from e13 (best of 7 TEST-scored epochs). Correction: 0.496 -> 0.782, not 0.857.
      NEW: s crosses 1.0 between e4 and e8, so the arm OVERSHOOTS calibration and a_p rising past
      s=1 is a defect reported as a gain. Read the 3 running seeds on s, not a_p.
      -> results/02_findings/SLOPE_CANONICAL.md
- [x] P0b **DONE.** OPEN_PROBLEMS.md P0 step 3 claimed canonical e14 for BOTH trajectory runs.
      Verified from the val logs: gld_slope1.0_s42 -> e10 (argmax 0.740, tied at e12, first-max
      wins; e14 is 8th-best at 0.722) and gld_dg_coil_s42 -> e14 (0.769, unique). Text corrected.
- [ ] P0c Rescore loroW_onehot_s42 and p3_a1_d1_s0_D1_uemb_seed42 at e14 (both were scored
          at e12, not their val argmax; both near-zero arms, error runs against them).
          Detector: scripts/audit_epoch_provenance.py.

- [ ] K8  Read out the 8 golden-lane arms when they finish (now at epoch 7-11 of 14):
          dg_coil, slope, w5_dg, w7edge, u10bidir, w7span4, loroW_onehot, loroW_desc.
          Each answers a lever that had NEVER been run.
- [ ] K9  LORO prediction ON RECORD before readout: the descriptor arm should NOT beat one-hot
          by much, because ProtT5 already predicts held-out-residue hydropathy at R^2 0.704.
- [ ] K10 Decide on the D1 half of the factorial: D0 coil scores 0.58-0.62, D1 uemb -0.08 to 0.31,
          no overlap, a_p 0.37-0.72 vs 0.005-0.020. ~190 GPU-h would confirm a visible negative.
- [x] K11 **CANCELLED — same refuted claim as K22.** See CHECKPOINT 25: the 2,356 2-point
      rows are all 2K5H single mutations on the _G11S/_G23A backgrounds, not double mutants.
- [ ] K15 results/FINDINGS.md — the single thesis-facing document, updated with corrected numbers.
- [ ] K16 CONTEXT.md checkpoint + push after every batch (running continuously).

## DONE — verified by the main session, not by agent report
V K1  2K5H reference row fixed. Its file concatenates 3 backgrounds and the mutant _G11S sorts
      first; true WT is at index 2738. Shift +3.0824. Corrected copy in data_fixed/ (Shahar's
      tree is read-only). CANONICAL HEADLINE (9 original-population CSVs, 27 proteins):
      pooled 0.5772, oracle 0.7156, gain +0.1384. The earlier, now-retired figures over 22
      mixed CSVs (0.4899 / 0.6443 / +0.1544) were the wrong population — see
      results/05_infrastructure/HEADLINE_BASIS.md.
V K2  r18_3_TrROS_Hall is CLEAN: single background, row 0 is the true WT; its low percentile is
      legitimate for a designed protein with many stabilising mutations.
V K3  gate_refrow hardened: counts distinct WT backgrounds per protein, which is the decisive
      test. Reports "3 distinct backgrounds" for 2K5H, no longer false-flags r18_3.
V K4  The "1/sqrt(N) embedding artifact" does NOT reproduce: claimed -0.9938, measured -0.3927
      (20 proteins) and -0.2946 (our 28). Below the n=28 threshold of 0.392. Lever dropped.
V K5  Dropped as a consequence of K4.
V K6  --dg_length_norm is DEGENERATE: /n converges to 0.905 for every checkpoint because
      std(true WT dG) = 0.9042 — it deletes the prediction rather than removing a bias. And
      corr(N, b_p) = +0.0252, so b_p is not length-driven.
V K7  Scoring pipeline unblocked (3 bugs) and running on the CPU partition at zero GPU cost.
      10/48 factorial CSVs; job 21108561 scoring 4 more.
V K12 a_p = r*s decomposed: a_p 0.364 = r 0.653 x s 0.504. Ranking error is 54.6% of the gap
      (unfixable by the slope term), spread 77.9% (fixable). Ceiling a_p = r. The three slope
      arms confirm the mechanism: r flat at 0.795-0.809 while s swings 0.47-0.76; weight 1.0 is
      best (+0.063 over control), weight 3.0 OVERSHOOTS and is worse than no lever.
V K13 The model is worst at burying hydrophobics: slope 0.28-0.31 to hydrophobic destinations vs
      0.53-0.57 to charged/polar. corr(slope, KD) = -0.734, sign-consistent 10/10. And
      corr(spearman, KD) = -0.740 — ranking degrades in LOCKSTEP, so it is LOST INFORMATION, not
      a rescalable calibration error. Mechanism: only 4 backbone atoms are stored, so side-chain
      packing is invisible. W and Y are the telling exceptions (bulk, not hydropathy).
V K14 Distribution shift verified: our regime was 0.1206% of pre-training RESIDUES (1 in 830);
      NMR 77.4% vs 8.6%, monomer 88.2% vs 31.9%, complexes 11.8% vs 68.1%. Real and large — but
      NOT the cause of b_p (r = +0.043, p = 0.829; nothing survives Bonferroni over 25 features).
V     Attenuation DEAD by 200x: sigma = 0.0327 kcal/mol over 28,312 mutations predicts a_p
      = 0.9985 against measured ~0.50 (28/28 proteins joined via ThermoMPNN/mega_test.csv).
V     ProtT5 is CONTEXTUAL and per-variant; it predicts held-out-residue hydropathy at R^2 0.704,
      so W6 descriptors are largely redundant with what the embedding already carries.
V     a_p logging added to validate() — 40 slope runs had been unfalsifiable because pooled dG
      PCC is scale-invariant and cannot see a spread change.
V     13 gate suites green throughout; gate_g4_cpu prints dG=-0.0030 width=1092 after every edit.

## THE STANDING RULE THAT KEEPS PAYING
Score a lever on the metric it acts on. It has now caught FIVE: the coil, BSA, W5 burial,
ligands, and our own model-selection metric. And: always compute the degenerate baseline —
K6 looked like a 31% improvement and was the model predicting nothing.

## Added 2026-09-08 — after the ν sweep and the b_p decomposition

- [x] K17 ν sweep (0.5→0.65 × b) — **DONE, REJECTED.** All 10 cells worse than no coil on
      std(b_p): best 1.1036 vs base 0.9972. Correct physics does not rescue the coil.
      → `results/02_findings/NU_SWEEP.md`
- [x] K18 b_p global-vs-spread decomposition — **DONE.** b_p is **96.3 % one global constant**;
      `MAE ≡ |global|` exactly. The thesis target is the residual spread **1.0155**.
      → `results/02_findings/BP_DECOMPOSITION.md`
- [x] K19 Is the coil failure a length-normalisation bug? — **DONE, YES.** corr(b_p,N) goes
      −0.215 → −0.507 monotonically in ν. The coil INDUCES a length artifact. Closed at zero cost.
- [ ] K20 **T-E1 unfolded ENSEMBLE** — k sampled coil conformations, cached per protein (the coil
      map never reads one_hot, so it is variant-independent). Sample COORDINATES not distances.
      Run both `mean_k E_k` and `−log Σ exp(−E_k)`; their difference is the conformational entropy.
      **Falsifier: if k=8 does not beat std(b_p) 0.9972, the reference-state programme ENDS.**
- [ ] K21 T-E2 distogram head, **FOLDED state only**. Attach at `train_utils.py:607`, before
      `D.sum(dim=1)` — the `[N,N,16]` pair tensor still exists there. NOT on the unfolded state
      (its map is an analytic function of |i−j| and predicting it teaches nothing). After K20.
- [x] K22 **CANCELLED — the 26,315 double mutants DO NOT EXIST.** CHECKPOINT 25 already measured
      it: 2-point rows number 2,356, ALL belonging to 2K5H, and they are single mutations on the
      _G11S / _G23A backgrounds parsed as two codes — the same concatenation artifact K1 fixed.
      There is no held-out double-mutant set. Do not re-plan this.
- [ ] K23 W12 side-chain readout: 5 golden arms queued (3 seeds ddG, 1 dG, 1 ×slope).
      Score on ddG AND dG/std(b_p) — scoring on ddG alone would repeat the W5 mistake.

- [x] W13 **DONE — THE LARGEST RESULT IN THE PROJECT.** Ofir's MEASURED offset, zero GPU.
      k=3 -> 0.6813 (+0.060), k=5 -> 0.7047, k=10 -> 0.7247, k=20 -> **0.7330** vs oracle 0.7468
      = 98% of the recoverable gain. Every lever ever tested combined gave +0.075; this gives +0.112.
      k=1 HURTS (-0.044): noise/signal = 0.6801/0.4615 = 1.47, break-even k=2.2, so never use k<3.
      Leakage blocked by disjoint index sets. MUST be reported as FEW-SHOT, not zero-shot.
      -> results/02_findings/W13_MEASURED_OFFSET.md

- [x] CATALOGUE RE-CHECK **— MY EARLIER CLAIM WAS WRONG.** I recorded "ligands/complexes have zero
      variance" from an agent report without verifying. The 100k catalogue actually holds
      67,856 complexes vs 31,852 monomers (100% coverage of Is Complex?/Oligomeric State).
      BUT the decisive fact is different: **0 of our 28 test proteins AND 0 of 247 PDB-like
      training proteins appear in the catalogue at all.** Catalogue median length 477aa, p5=94;
      our proteins are 42-72aa, below its 5th percentile. Ligands_x and BSA_Percentage are
      populated in only 4 and 5 rows of 100,246 - genuinely empty columns.
      -> results/04_data_integrity/CATALOGUE_RECHECK.md
- [ ] K21 **PLANNED IN FULL** (not yet built). Attach at train_utils.py:379, FOLDED ONLY, before
      D.sum(dim=1) at 381. Correction: my earlier "line 607" was wrong - that is inside
      _coil_expand_channels. Adds a LOSS not columns, so width stays 1092.
      -> results/03_levers/K21_DISTOGRAM_PLAN.md
- [ ] ANNOTATE OUR 21 PDB PROTEINS DIRECTLY from the PDB (oligomeric state, HETATM, interface
      area). Bounded job, 21 structures. This is the only honest route to testing the
      ligand/complex ideas and it has never been attempted.

- [x] W13b **DONE — slope and the measured offset do NOT compose.** At k=20 the arms converge to
      0.7106-0.7337 and the ORDER SCRAMBLES: the best-slope arm (a_p 0.782) finishes LAST, the
      worst (0.448) ties for first. They overlap rather than add.
      BUT across all 35 runs the oracle spread does NOT collapse (sd 0.0237 -> 0.0261), and
      corr(zero-shot, oracle) = +0.548 -- so the model still matters, and zero-shot ranking is a
      POOR predictor of post-calibration ranking. Select on the oracle, not on zero-shot.
      Best k=20 so far: 0.7337 (p3_slope1.0_s42_e13). -> results/02_findings/W13B_COMPOSITION.md

- [x] K21 **BUILT, VERIFIED, SUBMITTED.** --distogram_weight wired into train.py (flag 96, CFG 144,
      run-config 299). Self-test 6/6 PASS including overfit-one-example 3.488 -> 0.009, which proves
      the gradient path is real. gate_g4_cpu ALL PASS width 1092 (it adds a LOSS, not columns).
      Flag proven parsed by importing train.py with argv --distogram_weight 0.25 -> CFG = 0.25.
      Queued: 21144938/39/40 (weights 0.01/0.1/1.0) + 21144941 (x W12). Weight is 0.01-1.0, NOT
      IFUM's 100: our CE is order ln(32)~3.5 vs an MSE of order 1.
- [x] 12 DELIVERED SCRIPTS **RUN ON REAL DATA** (job 21144901, CPU). 4 artifacts verified with ls:
      BIAS_VS_DISPERSION.tsv, W13_ROBUST.tsv, W13_SELECT.tsv, EPOCH_PROVENANCE.tsv.
      Excluded: double_mutant_test (premise refuted), hydrophobic_failure (needs absent 'mut' column).
- [x] BOTH [VERIFY] MARKERS CLOSED. f_type='features' EXISTS but is FOLDED-ONLY and n_folded must be
      passed explicitly. coil_b_fixed=3.8 is WRONG: ours is 5.82A * 0.1 = 0.582 model units; using
      3.8 would be ~6.5x off and saturate the Gaussian kernel -- the w5_dg.py units bug again.

- [x] K20 **BUILT, UNITS BUG FIXED, VERIFIED, SUBMITTED.** The delivered script defaulted to
      coil_b_fixed=3.8; ours is _COIL_B_FIXED = 5.82A * 0.1 = 0.582 MODEL UNITS. 3.8 is ~6.5x too
      large and would saturate the Gaussian kernel -- the w5_dg.py units bug again. Fixed to 0.582.
      Self-verify after the fix: triangle inequality 0.00% violations (proves COORDINATES are
      sampled, not distances), Flory scaling worst error 14.8%, cache determinism, ensemble
      diversity 2.138 A. gate_g4_cpu ALL PASS width 1092. Flags proven parsed:
      --unfolded_ensemble_k 8 -> CFG=8, reduce -> logsumexp.
      Queued 21144960 (k=8 mean), 21144961 (k=8 logsumexp), 21144962 (k=3 mean).
      Their difference IS the conformational entropy.
      FALSIFIER: if k=8 does not push std(b_p) below 0.9972, the reference-state programme CLOSES.
- [x] CLARIFIED: only 3 of the 12 delivered scripts are GPU levers (distogram_head,
      unfolded_ensemble, coil_sign_check). The other 9 are CPU analysis tools and have already RUN
      on real data (job 21144901, 4 artifacts verified). They do not belong on a GPU.

- [~] W15 SIDE-CHAIN RECONSTRUCTION **IN PROGRESS.** FASPR downloaded, compiled, smoke-tested
      (783 atoms in -> 782 out, 0.08 s). 368/368 backbones written. Repack job 21145042 running.
      TWO BLOCKERS FOUND BY MEASUREMENT:
      (1) THE CLUSTER HAS NO C++ COMPILER. gcc reports 11.5 and ls shows /usr/bin/g++, but
          cc1plus is missing on BOTH login and compute nodes, so gcc cannot compile C++ at all.
          Fixed with conda gxx_linux-64 -> x86_64-conda-linux-gnu-g++.
      (2) data/MsDs IS GEOMETRICALLY CORRUPT for packing: |CB-CA|/CA-CA = 0.0039 across 40
          proteins vs the real 1.53/3.80 = 0.403. CB sits ~100x too close to CA, so the CB
          direction vector a packer uses is numerical noise. Processed_K50 is clean (0.3990-0.4025)
          and is the tree the model actually trains on. Also: aa_seq.pt is an EMPTY LIST, so the
          first run wrote all 368 as poly-alanine; sequence now comes from one_hot_encodings.pt.
      Writer verified on real output: 18 residue types present, |CB-CA| median 1.534 A,
      scale correctly detected as 1.0 (already Angstrom) on all 368, glycine CB dropped not faked.
      -> results/03_levers/W15_PROGRESS.md

- [x] W15 FEATURES **BUILT AND VALIDATED** for all 368 proteins (19,512 residues, 12 MB).
      CHECK A -- reconstruction is real: corr(sc_sasa, existing burial) = -0.6820, exactly the
      strong negative physics requires; a wrong reconstruction would give ~0. contacts +0.6994,
      clash +0.0853 (nearly orthogonal to burial -- information the model has no access to).
      CHECK B -- THE DECISIVE TEST. Any per-residue descriptor table is one_hot @ T and cannot
      add information (this killed LORO, W6, and 2 of W12's 4 columns). R2 explainable from
      one-hot alone: sc_sasa 0.443, buried_frac 0.668, contacts 0.488, clash 0.245.
      W15 PASSES: 33-76% of each column is geometry residue identity cannot express. Residual sd
      after removing everything one-hot predicts: 75%, 58%, 72%, 87% of raw sd survives.
      CAVEAT: FASPR PREDICTS the mutant conformation; nobody measured it. Inference, not observation.
      Confidence it carries new information 90%; confidence it improves the score 55%.
      -> results/03_levers/W15_VALIDATION.md
