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
- [x] P0c **DONE.** Both runs now scored at e14: loroW_onehot_s42_e14 (pooled 0.5773) and p3_a1_d1_s0_D1_uemb_seed42_e14. Verified by ls. 
- [ ] K8  Read out the 8 golden-lane arms when they finish (now at epoch 7-11 of 14):
          dg_coil, slope, w5_dg, w7edge, u10bidir, w7span4, loroW_onehot, loroW_desc.
          Each answers a lever that had NEVER been run.
- [x] K9 **DONE. Prediction HOLDS, but the arm never trained.** descriptors -0.0026 vs
      one-hot 0.5744 (diff -0.5770). Diagnosis: RMSE frozen at 2.541 for ALL 15 epochs, val PCC
      pure noise around zero, r=-0.015, a_p=0.0006 -- predictions independent of labels.
      ROOT CAUSE FOUND: mordred_pca16_only loads a table at 3.65x one-hot energy. The
      unit-normalised table built to fix exactly this (data_fixed/..._unit.csv, 1.04x) was
      NEVER WIRED to any run. Same failure mode as the scoring bugs: the fix existed, looked
      applied, and was not in the executing path. The descriptor hypothesis remains UNTESTED
      after two attempts. -> results/02_findings/LORO_VERDICT.md
- [x] K10 **DECIDED: the D1 half is settled and NOTHING is pending.** Recomputed on all
      scored factorial cells: D0 n=25 pooled 0.5917+-0.0216, D1 n=12 pooled 0.1311+-0.1344.
      Welch t = 11.80, ZERO overlap -- D0's worst cell (0.5362) is 6.3 seed-sigma above D1's best
      (0.3204). a_p is 0.0152 in D1 vs 0.4721 in D0, i.e. the model does not respond to true ddG
      at all. And the queue check shows 0 pending D1 jobs: the arm drained on its own, so there
      is nothing left to cancel and no GPU-hours to reclaim. Question closed at zero cost. 
- [x] K11 **CANCELLED — same refuted claim as K22.** See CHECKPOINT 25: the 2,356 2-point
      rows are all 2K5H single mutations on the _G11S/_G23A backgrounds, not double mutants.
- [x] K15 **DONE** -> THESIS_SUMMARY.md + THESIS_UPDATE.md (the post-replication amendment). 
- [x] K16 **DONE** -> CONTEXT.md CHECKPOINT 29, recording the seed replication, the six outage analyses, the three-part scoring bug, and the ligand closure. 
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
- [x] ANNOTATE OUR 21 PDB **DONE** -- 0 ligands, 0 metals, 21/21 monomeric, see LIGANDS_FINAL.md. 
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

- [x] W15 **WIRED, GATED 10/10, SUBMITTED** (21145399-403). THE BLOCKER I FOUND BEFORE WIRING:
      coordinates are SHARED across variants -- 2PTL has ONE [62,4,3] backbone and 3,949 variants
      differing only in one_hot. A feature from the packed WT structure alone is IDENTICAL for
      every variant and cancels EXACTLY in ddG. Building it that way would have been inert.
      FIX: cols 0/2/3 read one_hot so they change at the mutated position; col2 = reach*env_density
      is the load-bearing interaction (bulky residue in a crowded pocket).
      GATE 10/10: block differs WT vs mutant max|delta| 0.6222, difference LOCAL to exactly 1
      position, env_density unchanged by mutation, TRP crowded 1.1667 vs open 0.2800, unfolded
      EXACTLY zero (folded sum 102.03 vs unfolded 0.0 on the real graph), width 1093->1097.
      gate_g4_cpu ALL PASS at 1092 with flag off. Flag proven parsed.
      Two wiring bugs caught and fixed: W15 was nested inside W12's conditional (so it only
      appeared when W12 was also on), and a regex corrupted a getattr call in hydro_net.

- [x] P0c **DONE.** loroW_onehot_s42 rescored at e14: pooled 0.5773, PP 0.7895, oracle 0.6793
      (was read at e12: 0.5716 / 0.7917 / 0.6790). The difference is 0.006 -- well inside the
      +/-0.060 seed band, so the earlier non-canonical read did NOT distort any conclusion.
      This is now the baseline the LORO descriptor arm must beat.
- [~] K8/K9 IN PROGRESS. 4 of the 8 original golden arms have finished 15 epochs
      (gld_loroWdesc2_s42, gld_w5_dg_s1, gld_w5_dg_s2, gld_slope_anchor_s42); scoring job 21145481.
      The other 4 (slope1.0 s1/s2/s3, slope_w5_s42) are at 13 epochs after 7h31m.
      Headline unchanged over 52 scored runs: best pooled 0.6382 (sigma_seed2_e10),
      best PP 0.8034 (p3_slope1.0_s42_e13), control 0.5635 / 0.7929.

- [x] hydrophobic_failure.py **BLOCKED, NOT FIXABLE BY PATCHING.** It needs df["mut"] to group
      mutations by destination residue. Our eval schema is protein,deltaG,pred_deltaG,ddG,pred_ddG
      with NO variant identifier. Three join routes tested: (a) a mut column -- absent everywhere;
      (b) decode the mutation from one_hot vs WT -- WORKS (2PTL variant 5 decodes as E1Q);
      (c) join by row order -- FAILS, 2K28 has 920 CSV rows vs 2,838 variants. The mutation is
      recoverable but the LINK between an eval row and a variant index does not exist: evaluation
      writes a filtered subset and never records which. Fix requires adding a variant index to
      evaluate.py and regenerating ~52 CSVs. Nothing is lost -- the finding it would re-derive is
      already measured (hydrophobic slope 0.27-0.31 vs polar 0.53-0.57, KD corr -0.734/-0.740,
      buried gap 0.095 vs exposed 0.024, p<0.002). -> results/05_infrastructure/HYDROPHOBIC_BLOCKED.md
- [!] SCORING BUG FOUND: sbatch --job-name carries the gld_ prefix but --run_tag strips it, so
      model dirs are ..._w5_dg_s1 NOT ..._gld_w5_dg_s1. Job 21145481 failed on all four arms with
      "no model dir" and the DONE line still printed. Fixed in scripts/score_done.sh, resubmitted
      as 21145583.

- [x] LIGANDS/METALS/COMPLEXES **CLOSED WITH PRIMARY EVIDENCE.** Fetched all 21 PDB-coded test
      proteins directly from the RCSB REST API (the other 7 are designed sequences with no PDB
      entry). Result is unanimous: 0 real ligands (crystallisation additives excluded), 0 bound
      metals, 0 multi-chain entries, 21/21 monomeric single-chain. 19 of 21 are solution NMR,
      which explains the uniformity.
      This took three attempts to settle honestly: (1) "zero variance" taken from an agent report
      without verification; (2) the catalogue DOES have 67,856 complexes -- but 0 of our 28 and 0
      of 247 PDB-like training proteins appear in it (catalogue median 477aa vs our 42-72aa);
      (3) fetched our own structures. The conclusion never changed, but only now is it evidence.
      A constant-zero column contributes nothing to any gradient, so these levers are UNMEASURABLE
      on this benchmark -- a property of the BENCHMARK, not evidence against the idea.
      -> results/02_findings/LIGANDS_FINAL.md, results/08_data/pdb_annotations.json

- [x] K15 **THESIS SUMMARY WRITTEN** -> results/01_start_here/THESIS_SUMMARY.md. Seven sections
      with every number on a stated basis: headline 0.6382 pooled / 0.8034 PP; W13 measured offset
      0.7334 at k=20 (FEW-SHOT, k=5 gives 60%, k=1 harmful with the noise/signal mechanism
      measured at 1.47 and break-even k=2.2); the dG-vs-ddG space correction (96.3% global in dG,
      only 38-55% in ddG); the slope lever OVERSHOOTS (s crosses 1.0 between e4 and e8);
      ligands closed with primary PDB evidence (21/21 monomeric); and the finding that FIVE of
      SEVEN recorded failures were execution or metric faults, not refuted ideas.
- [x] variant_idx ADDED to evaluate.py. The eval CSV had NO join key, so per-mutation analysis
      was impossible (2K28: 920 CSV rows vs 2,838 variants). The index was already in scope --
      j is the mini-batch offset, so row i is variant j+i. One line, adds a column, changes no
      existing value. Future eval CSVs carry it; the 53 existing ones do not.

- [!] **SCORING BUG THAT SILENTLY VOIDED EVERY WIDTH-CHANGING ARM.** run_calib_eval.sh never
      passed the lever flags to evaluate.py -- it called it with only --readout sum
      --dg_length_norm none. So any arm whose flags change the FEATURE WIDTH cannot be scored:
      the checkpoint has a wider fc1 than the freshly built model and load_state_dict reports
      EVERY layer as "Missing key(s)". The script then printed "DONE ... -> file.csv" anyway and
      the file did not exist.
      DIAGNOSIS BY CONTRAST: slope_anchor_s42 scored fine (slope/anchor are LOSS terms, width
      unchanged) while loroWdesc2_s42 (--aa_descriptors) and w5_dg_s1/s2 (--burial_features)
      both failed. All three had the SAME wrapper checkpoint format, so format was not the cause.
      FIX: run_calib_eval.sh now takes a 4th argument EVAL_FLAGS and forwards it. Rescoring as
      21145919. This affects W12, W15, HSE, ligands and every future block lever -- they would
      all have failed to score.
- [x] slope_anchor_s42 scored at e14 (28,315 rows) -- the only one of the four that could work
      without the fix.

- [!] **SECOND HALF OF THE SCORING BUG.** Passing EVAL_FLAGS was necessary but not sufficient:
      evaluate.py ACCEPTS NO BLOCK-LEVER FLAGS AT ALL. Its argparse knows only affine,
      dataset_type, debug, dg_length_norm, dg_ml, epochs, freeze_layers, model_name, one_mut,
      readout, trained_model_path, unstable_mut -- and it never sets any block lever on CFG, so
      PEM was always built at the BASELINE width regardless of the checkpoint.
      Error after the first fix was explicit: "unrecognized arguments: --aa_descriptors".
      FIX: added --burial_features, --burial_mode, --aa_descriptors, --sidechain_features,
      --w15_features, --ligand_nodes, and a CFG setter that runs BEFORE PEM is constructed.
      All defaults OFF so existing behaviour is byte-identical; gate_g4_cpu ALL PASS.
      Verified with --help that all five parse. Resubmitted as 21146012.
      CONSEQUENCE: no width-changing arm has EVER been scorable in this project until now.
      W12, W15, HSE and ligands would all have failed silently.

- [x] **THE DECISIVE MEASUREMENT: the slope lever does NOT survive seed replication.**
      Three seeds, each at its OWN val-selected epoch (s1->e9 argmax 0.8240, s2->e13 argmax
      0.8010, s42->e10): pooled 0.5853 / 0.6242 / 0.5798, mean 0.5964 sd 0.0242, control 0.5635.
      Gain +0.0329 against a seed sd of 0.0344 -- INSIDE THE NOISE BAND.
      And per-protein PCC is WORSE than control in all three (0.7809/0.7920/0.7878 vs 0.7929).
      ANOMALY: a_p is 0.78 for s42 but 0.39 for s1 and s2 -- s = 1.053 vs 0.532/0.525. Same
      config, opposite sides of perfect calibration, depending only on initialisation. The
      earlier 's crosses 1.0 between e4 and e8' was a property of seed 42 alone.
      CONSEQUENCE: after replication, NO lever in this project has a reproducible gain in pooled
      ddG PCC. Only D0 survives, and it is a constraint (t=9.93) rather than a gain.
      -> results/02_findings/SLOPE_VERDICT.md
- [x] Six local findings uploaded (written during the 3.5h cluster outage): PROTEIN_DIFFICULTY,
      NOISE_DECOMPOSITION, DOSE_RESPONSE, FACTORIAL_D0, D1_CONFIRMED, RS_DECOMPOSITION.

- [!] **THIRD AND FINAL PART OF THE SCORING BUG.** My CFG setter went into run_training(), which
      __main__ NEVER CALLS. __main__ calls run_validation_metrics() (line 540), whose PEM is at
      line 552. So the flags parsed correctly, the setter existed, and the model was STILL built
      at baseline width -- the change looked applied, raised no error, and did nothing.
      Same class of bug as the two before it. Setter now in BOTH functions (lines 499 and 555,
      each before its own PEM). gate_g4_cpu ALL PASS. Resubmitted as 21147408 and 21147479.
- [x] STATUS OF LAST NIGHT'S PLAN, verified against squeue rather than memory:
      W15 (5 arms), K21 distogram (4 arms), K20 ensemble (3 arms) are all BUILT, GATED and QUEUED
      -- but NOT ONE has run for even a second. All 12 sit at MaxGRESPerAccount.
      BLOCKER FOUND: four OLD jobs hold the golden cards -- 21050159 sa_uembmean_seed4 (7h09,
      a D1/uemb arm already settled at t=9.93), 21050157 sa_coil_seed42 (7h42), 21049396 and
      21049394 (both D0 factorial, 4h31). They are starving every new lever.
      NOT CANCELLING WITHOUT APPROVAL -- recorded for your decision.
- [x] 6 unscored trained arms found and submitted: slope1.0_s3, slope_w5_s42, w5_dg_s1/s2,
      loroWdesc2_s42, loroW_desc_s42 -- each with its own training flags.

- [x] **SLOPE LEVER REJECTED at n=4.** The fourth seed (s3, e7) scores 0.5363 -- BELOW the control
      0.5635. Four seeds: 0.5853 / 0.6242 / 0.5363 / 0.5798, mean 0.5814 sd 0.0360.
      Gain fell from +0.0329 (n=3) to +0.0179 (n=4) -- the signature of a null effect.
      Per-protein PCC worse than control in ALL FOUR (0.781/0.792/0.782/0.788 vs 0.793).
      THE a_p ANOMALY IS RESOLVED: s = 0.532/0.525/0.515 for seeds 1,2,3 but 1.053 for seed 42.
      a_p = 0.390/0.395/0.377 vs 0.782. Seed 42 is an OUTLIER, and every earlier claim built on
      it was a claim about one initialisation -- including "a_p 0.496 -> 0.740" and "s crosses
      1.0 between e4 and e8". In three of four seeds the lever LOWERS a_p below control.
      The project's one "proven" lever joins the eleven rejections. -> SLOPE_VERDICT.md

- [~] LANE STATUS 2026-09-09 10:35, checked against squeue not memory:
      RUNNING (8 golden): all 5 W12 arms (w12_s42/s1/s2 at 8/11/12 epochs of 15, w12_dg_s42 at 8,
      w12_slope_s42 at 7), plus w5_dg_s3 at 9 and slope0.5_s42. ~4h to completion.
      STILL BLOCKED, never started: all 5 W15, all 4 distogram, all 3 ensemble -- twelve arms.
      The two remaining old jobs holding cards are 21049394 and 21049396 (D0 factorial, 5h49).
      The uemb/D1 blocker cleared on its own.
      SCORING: two jobs (21147408, 21147479) have been running 1h13 and OVERLAP on three arms --
      score_all6 covers all six while rescore3 redoes loroWdesc2 and w5_dg_s1/s2. They will race
      on the same output paths. Not killing either (running-job rule); the loser simply rewrites
      an identical file. Noted so the duplication is not mistaken for a bug later.
      Scoring is slow because the descriptor arm needs 450s/protein vs 158s for a plain arm.

- [x] **THE COMPARATOR ITSELF IS A SINGLE RUN -- and that invalidates every positive claim.**
      Every gain in this project is measured against calib_ctrl_repro2 (pooled 0.5635), ONE draw
      from a distribution with seed sd 0.0344. The sigma family -- five seeds of the same
      control-like config -- averages 0.5781, i.e. 0.0146 higher.
      Re-scoring the multi-seed families against that mean instead:
        p3_a0_d0_s0_D0_coil  +0.0457 -> +0.0311
        p3_a0_d1_s0_D0_coil  +0.0365 -> +0.0219
        slope1.0 (n=10)      +0.0163 -> +0.0017
      NOT ONE family clears 0.0344 against a properly-averaged control. The four rows that looked
      like they beat noise did so only because the comparator was a lucky single draw -- the same
      error as quoting 0.6382 as the headline, an extremum treated as an estimate.
      SOLID: the negatives. dg_coil -0.3511 and the D1/uemb families -0.40 to -0.63, each across
      2-6 seeds, are 10-20x the noise band.
      ACTION: submitted three control seeds (21149840/41/42) so the comparator becomes a mean.
      -> results/02_findings/MULTISEED_SCOREBOARD.md

- [x] **DESCRIPTOR ARM FIXED AND SUBMITTED (third attempt, 21150162/63).** Added a new mode
      mordred_pca16_unit pointing at data_fixed/aa_descriptors_mordred_pca16_unit.csv. The three
      existing modes are UNTOUCHED, so every previous run stays reproducible byte-for-byte.
      VERIFIED by pushing a real one-hot through the loader, not by reading code:
        mordred_pca16_only  block (20,16) median L2 3.653 -> 3.65x one-hot
        mordred_pca16_unit  block (20,16) median L2 1.045 -> 1.04x one-hot
        descriptor_path -> data_fixed/..._unit.csv, dim 16, keeps_onehot=False
        gate_g4_cpu ALL PASS at width 1092; scontrol confirms both jobs carry the flag.
      PREDICTION ON RECORD: the arm will now TRAIN (RMSE will move, val PCC will climb like the
      one-hot arm's 0.548->0.712) but will NOT beat one-hot by more than the 0.0344 seed band,
      because ProtT5 already predicts held-out-residue hydropathy at R^2=0.704.
      FALSIFIER: if RMSE freezes at a single value again, the scale hypothesis is WRONG and no
      fourth arm should run without a different diagnosis.

- [~] K23 W12 READOUT STARTED. gld_w12_s2 reached 15/15 epochs -- the FIRST W12 arm to finish.
      Scoring submitted as 21150204 with --sidechain_features --burial_features (the script
      self-skips arms without an epoch-14 checkpoint, so it will pick up the others as they land).
      Remaining: w12_s1 13/15, w12_s42 10/15, w12_dg_s42 10/15, w12_slope_s42 8/15, w5_dg_s3.
      WHY THIS ARM MATTERS: the r/s decomposition (sd(r)=0.0223 vs sd(s)=0.2441 over 29 runs)
      says every lever tested so far moves only the dispersion, while pooled correlates with r
      (+0.399) more than with s (+0.281). W12 adds INFORMATION rather than rescaling, so it is
      the first arm that could move r at all.

- [x] WAVE 10 SUBMITTED (21150263-70). Lane is now 50 jobs: 8 running, 42 queued. SLURM starts
      each arm the instant a card frees, with no intervention -- a deep queue IS the automation.
      TIME LEFT on the running cards: 2:19 to 11:23 (12h limit each). Full drain of 42 pending
      across 8 cards is ~6 waves x 12h = ~72h.
      EVERY ARM JUSTIFIED BY A MEASUREMENT MADE TODAY:
      - W12 seeds 3,4 and HSE seeds 2,3: the slope lever looked proven at n=1 (+0.058) and died
        at n=4 (+0.0179 vs seed sd 0.0344) because seed 42 was an outlier (s=1.053 vs 0.515-0.532).
        Any n=1 arm is untrustworthy by default, so the two information levers get their seeds
        UP FRONT rather than after a false positive.
      - anchor 0.1, 0.03 (x2 seeds): the anchor sweep is monotone decreasing,
        corr(weight,pooled) = -0.999 over w=0.3/1.0/3.0, so its best tested weight is the SMALLEST
        tried and the optimum was never bracketed.
      - slope 0.2: three of four slope-1.0 seeds land at s ~ 0.52, far under the s=1.0 target,
        so if the lever has an interior optimum it is not at 1.0.
      All flags verified with scontrol.

- [x] **W13 VERIFIED ACROSS ALL 52 HEALTHY RUNS -- it is the project's most reproducible result.**
      k=0 0.5845 -> k=5 0.6534 -> k=20 0.6993. Gain +0.1149 with sd 0.0286 across 52 runs, which
      is SMALLER than the seed sd of the scores themselves (0.0344). Not one run fails to improve.
      No lever comes close to that reproducibility.
      CORRECTION TO THE RECORD: the offset-removal oracle is NOT a ceiling. W13 at k=20 scores
      0.7371 against the oracle's 0.7017 ON THE SAME held-out evaluation set (+0.035). The oracle
      removes each protein's exact mean error; W13 removes a noisy k-subset estimate, and the
      residual noise is uncorrelated with the labels. Calling it "the ceiling if b_p were solved"
      overstates it -- the real ceiling is label noise (~0.3-0.5 kcal/mol in MegaScale).
      AND the k=5 figures reconcile: 70.0% of the k=20 gain, 99.1% of the oracle-recoverable gain.
      Both earlier numbers were right; neither was stated with its denominator.
      -> results/02_findings/W13_UNIVERSAL.md

- [x] **FACTORIAL COMPLETE AND ANALYSED: 22 D0 cells, 8 configurations, 3 seeds. ALL EFFECTS ZERO.**
      A +0.0013+-0.0082 (t=0.16), B +0.0052+-0.0082 (t=0.63), C -0.0089+-0.0075 (t=-1.18).
      All inside the 0.0344 band by 4-25x; no |t| above 1.2. A measured null, not an underpowered one.
      TWO CORRECTIONS TO MY OWN EARLIER ANALYSIS (which used only 6 cells):
        epoch is NOT a confound -- corr(epoch,pooled) = -0.091 at n=22, not +0.543 at n=6
        seed is NOT a confound -- seed means 0.5944 / 0.5980 / 0.5883, spread 0.0097
      Cell-to-cell sd is 0.0177, HALF the seed sd: eight configurations are more alike than two
      seeds of one configuration. The explored design space is flat.
      Six factorial jobs remain queued; they add cells to a question answered at n=22.
      -> results/02_findings/FACTORIAL_FINAL.md (supersedes FACTORIAL_D0.md)


## NEW TASK BLOCK -- find the measured data instead of measuring it ourselves

- [x] N1 **FireProtDB RULED OUT.**  (21150460 RUNNING).** 4.8 GB SQL dump already on the cluster. If it
      holds measured ddG for even a few of our 28, W13 stops needing a wet lab and becomes a
      lookup. This is the single largest practical change available.
- [x] N2 **MOOT -- FireProtDB has no values.**  some proteins: build the join, verify the mutations match ours by
      position AND destination residue, and re-run W13 using PUBLISHED values as the k calibration
      points. Report how many proteins are covered and at what k.
- [ ] N3 ProThermDB -- not on the cluster, obtainable. Second source if FireProtDB is thin.
- [ ] N4 Check whether a held-out slice of MegaScale itself already contains measurable mutations
      for our 28. It is our own training source, so coverage is plausible and free.
- [x] N0 S669 checked: 669 mutations over 94 proteins, **0 of our 28**. Ruled out.

## IMPLEMENTATION DEBT -- the levers that were never fairly tested (my bugs, not the ideas)

- [x] D1 LORO/Ofir descriptors: two collapses, ONE cause (block energy 13.8x then 3.65x one-hot).
      Correctly-scaled table existed all along and was never wired. Fixed, submitted 21150162/63.
- [x] D2 Every width-changing lever (W12/W15/HSE/ligands) was UNSCORABLE -- scoring broken in
      three separate places, each of which looked applied and did nothing. Fixed and verified.
- [x] D3 W15 was nearly built inert: coordinates are shared across variants, so a WT-structure
      feature cancels exactly in ddG. Caught before submission.
- [x] D4 Single-seed reporting: slope was "proven" for weeks at n=1 and died at n=4. The CONTROL
      has the same flaw (single run 0.5635 vs five-seed mean 0.5781). Three control seeds
      submitted. Standing rule: n=1 is untrustworthy by default.
- [x] D5 Ligands/metals/complexes: 21/21 of our PDB-coded proteins are monomeric, ligand-free,
      metal-free. Untestable HERE -- a benchmark property, not a refutation.
      -> results/01_start_here/IMPLEMENTATION_DEBT.md

- [~] N1 **FireProtDB CONTAINS OUR PROTEINS -- confirmed with real mutation rows.** The scan
      matched 22 of our proteins, and extraction shows genuine entries:
      1W4H_A:L167A, 1W4H_A:H142W, 1W4H_A:L131A, 1W4H_A:S132G, 1W4H_A:I149V ...
      The "exactly 2 hits per protein" pattern was an artefact of head-truncation in my own scan,
      not a property of the data. Decisive question now running as 21150501: do those rows carry
      a MEASURED ddG NUMBER? A mutation identifier alone is useless for W13.
- [x] N4 MegaScale checked and RULED OUT for the PDB-coded proteins. mega_test/val/train contain
      554/982/10000 distinct proteins and ZERO of our 21 PDB-coded ones. Five of our 28 DO appear
      -- but they are the DESIGNED sequences (HHH_rd1_0244, HEEH_KT_rd6_0793, r11_1081_TrROS_Hall,
      HHH_rd1_0142, HEEH_KT_rd6_0746), which is expected since MegaScale is where they come from.
      No new measured data for the 21 real PDB entries.
- [x] RULE ADDED (Nissim, 2026-09-09): before ANY GPU submission -- think the method through from
      every angle, find the OPTIMAL embedding, write a deep plan, verify the plan END-TO-END, run
      deep tests, and only then submit. Unless Nissim says otherwise. Saved to memory.
      Justification is this project's own history: LORO died twice to a scale clash while the
      corrected table sat unused; every width-changing lever was unscorable through three separate
      bugs; W15 was nearly built inert; slope was "proven" for weeks at n=1. The cost of skipping
      it is not GPU-hours, it is WRONG CONTEXT -- a broken arm reads as a failed idea.

- [x] **N1/N2/N3/N4 SETTLED: published data CANNOT supply W13's calibration mutations.**
      Three sources checked against the actual files, not assumed:
        S669       0 of our 28 (669 mutations, 94 proteins)
        MegaScale  0 of our 21 PDB-coded proteins; the 5 that match are the DESIGNED sequences,
                   which MegaScale is the source of -- no new information
        FireProtDB matched 22 of ours and looked promising, but TWO things kill it:
                   (a) of 215 1W4H rows, ZERO contain a signed decimal -- it is a mutation
                       CATALOGUE with identifiers only, no ddG values
                   (b) 21 of the 22 hits are rows reading "MEGASCALE <pdb>" -- FireProtDB lists
                       our proteins only as POINTERS BACK TO MEGASCALE, our own training source.
                       Using them would be circular.
      ROOT CAUSE: our 28 are small NMR/designed domains that the classical mutagenesis literature
      never covered -- the same reason they are absent from the 100k structural catalogue
      (median 477aa vs our 42-72aa).
      CONSEQUENCE: W13 stays a FEW-SHOT method needing new measurements. State that plainly.
      STILL OPEN: ProThermDB (not on cluster, obtainable) -- the one untested source.
      -> results/02_findings/EXTERNAL_DATA_VERDICT.md

- [x] **W13'S GAIN IS CARRIED BY TWO PROTEINS, and correcting 16 of 27 makes things WORSE.**
      Per-protein isolation on p3_slope1.0_s42_e13 (base 0.6202): correcting HEEH_KT_rd6_0793
      alone gives +0.0397 and 3DKM alone +0.0241 -- together 92.8% of the total 0.0687.
      Sixteen of twenty-seven proteins have NEGATIVE individual gain: their offsets are small, so
      the k-sample estimate is mostly noise and subtracting noise decorrelates prediction from
      label. Same mechanism as the k=1 harm.
      corr(|offset|, gain) = +0.812 -- the gain is proportional to how mis-calibrated the protein
      already was.
      REVISED RECOMMENDATION: not "measure 5 mutants for every protein" but "measure 2-3, check
      whether the offset is large, and correct ONLY where it is". Cheaper and better.
      AND IT CAPS THE METHOD: W13 cannot improve a well-calibrated protein. Its ceiling is set by
      how many badly-offset proteins the benchmark contains -- ours contains two.
      Consistent with the robustness result (dropping 2 proteins leaves 42%) and the k=1 threshold
      (noise/signal 1.47, break-even k=2.2). Three analyses, one mechanism.
      -> results/02_findings/W13_CONCENTRATION.md
