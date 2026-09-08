# DeepEF — CONSOLIDATED AGENT REPORT

Every background-agent result, collected into one file. 28 agents across four workflows.
From here the work continues as TASKS executed by the main session, with no background agents.

Status legend: verdicts are the agents' own. Anything marked VERIFIED was independently
re-derived by the main session; anything else is agent-reported and still needs checking.


====================================================================================================
# WORKFLOW: DEEP-VERIFY  (10 results)
====================================================================================================

## Coil / unfolded-state levers U3 (coil_channels), U4 (coil_b), U5 (coil_edges), U6 (per-half ca_coords) — end-to-end verification
**Verdict:** NEEDS-FIX

**DATA USED OPTIMALLY:** The dG-side measurement is the correct and well-used one: w0_dg.json scores the coil on ABSOLUTE wild-type dG, which is exactly what THE METRIC RULE demands for a reference-state lever, and it is the measurement that reversed the ddG-based "coil is harmful" verdict (r=1.175 in w0.json is genuinely a var(E_u) ratio, not a dG result — both objects are present and correctly labelled). All 28 test proteins are used; per-protein rows are retained so every headline number is re-derivable, which is how I could confirm them exactly rather than trusting the summary block. What is left on the table: (1) w0_dg.json has only n=28 and reports MAE/corr/std_err but NO paired significance test — base 4.9650 vs coil_fixed_b 3.9521 is a 1.01 kcal/mol improvement on 28 points with no CI or paired test, and the bias column shows MAE == -bias for every single arm, i.e. every arm under-predicts EVERY protein, so this "MAE" is measuring a pure offset, not a fit; the honest framing is that coil_fixed_b shrinks a uniform offset, which is precisely a b_p effect and consistent with the b_p story, but it should be stated as offset-shrinkage, not accuracy. (2) corr is FLAT (0.3464 -> 0.3404), confirming the gain is offset-only — this is worth saying explicitly in the thesis rather than letting "MAE improved 20%" imply better ranking. (3) The U5/U6 arm has ZERO runs: no submitted script anywhere passes --coil_edges (grep over all .sh/.slurm returned nothing), and the 113 queued jobs are ddG-factorial coil cells, so the strongest dG finding still has no training run testing it, and U5/U6 are entirely untested in flight. (4) The b-mismatch defect below means the one flag combination the thesis actually recommends (--flory_unfolded --coil_b fixed --coil_edges) has never been measured coherently.

**EVIDENCE:** ALL work read-only; the only file I wrote was /home/nissimb/vfy_coil.py, outside the repo.

[1] GATES — all three pass.
  python scripts/gate_g4_cpu.py -> "baseline forward ok, dG finite PASS dG=-0.0030 width=1092" ... "G4-CPU: ALL PASS". Re-run at the END of the session, identical. G4 UNMOVED.
  python scripts/gate_u3u4.py -> "GATE PASSED: 32 checks"
  python scripts/gate_u5u6.py -> "GATE PASSED: U5 + U6 ready. Byte-identical when off; G2-U5 and G2-U6 hold."
  Byte-identity is asserted by torch.equal on the REAL PEM.get_edge_index / PEM.forward, not a fixture — gate_u5u6 section [10] "INTEGRATION the REAL PEM.get_edge_index, not a paraphrase of it": "PEM GAT edges byte-identical with coil off (torch.equal)", "PEM GCN edges byte-identical with coil off (torch.equal)", "PEM.forward byte-identical with/without n_folded when coil off". Claim SUBSTANTIATED.

[2] RESULTS JSON — re-derived from per_protein, not read off the summary.
  python: for each arm, mae = mean(|pred_arm - true|) over the 28 per_protein rows:
    base          re-derived MAE=4.9649982971  stored=4.9649982971  match=True
    coil          re-derived MAE=5.7445897191  stored=5.7445897191  match=True
    coil_fixed_b  re-derived MAE=3.9520752425  stored=3.9520752425  match=True
    coil_ca_only  re-derived MAE=5.5462562362  stored=5.5462562362  match=True
    noemb         re-derived MAE=3.1315321908  stored=3.1315321908  match=True
    base 4.9650 -> coil_fixed_b 3.9521 : delta = 1.0129
  Correlations re-derived too: base 0.346442 == stored 0.346442; coil_fixed_b 0.340410 == stored 0.340410.
  w0.json coil r_vs_base = 1.1753293925623431, re-derived as var_Eu(coil)/var_Eu(base) = 1.1753293925623431 — so the "1.175" is a variance ratio on E_u, NOT a ddG accuracy number; CONTEXT.md:265,271,383 use it correctly.
  results/CONTEXT.md:397-403 states base 4.9650 / coil-b-fixed 3.9521 / coil-fitted 5.7446 / ca_only 5.5463 — matches the JSON exactly. CONTEXT.md CONFIRMED.
  NOTE: for all five arms MAE == -bias exactly, i.e. every arm under-predicts all 28 proteins; the improvement is offset shrinkage.

[3] CRITICAL WIRING CHECK — confirmed EMPIRICALLY on the real coil_topology, N=10, PEM's real fully-connected branch (no self-loops):
    batched (n_folded=1,B=2): 108 edges  [folded 90 + coil 18 = 108]   <- lever APPLIED
    SEPARATE pass (n_folded==B==1): 90 edges [fully-connected baseline = 90]
      returns the IDENTICAL object (is): True   torch.equal: True
    n_folded=None: 180 edges, identical object: True, torch.equal: True
    >>> SILENT NO-OP CONFIRMED: True
  Mechanism: coil_topology.py:168 `n_f, n_u = split_folded_rows(B, n_folded)`; :169 `if n_u <= 0: return edge_index_full` — returns the input OBJECT, no exception, no warning.
  TRAINING PATH IS CORRECT: Megascale-fineTuning/train.py:963-964 `self.model(all_graph_minibatch, n_folded=folded_graph_minibatch.size(0))` and train.py:1000 `self.model(torch.cat([folded, unfolded], dim=0), n_folded=folded.size(0))`. Both batch the two halves into ONE forward and pass n_folded. VERIFIED.
  BUT the EVAL path is NOT: Megascale-fineTuning/evaluate.py:463 `minibatch_energy = self.model(all_graph_minibatch)` — no n_folded, so n_folded=None -> split_folded_rows returns (B,0) -> n_u=0 -> coil edges silently skipped. evaluate.py IS live: Megascale-fineTuning/run_calib_eval.sh:45 invokes it (the same script named in the brief for printing DONE after a CUDA crash).

[4] U4 UNITS — verified by measurement, not by comment.
  train_utils.py:445-447: _COIL_B_FIXED_ANGSTROM=5.82, _COIL_COORD_SCALE=0.1, _COIL_B_FIXED=5.82*0.1.
  Measured: _COIL_B_FIXED (model) = 0.5820000000000001; "converted exactly ONCE (5.82*0.1): True".
  Scale chain confirmed against the REAL normalize_batch: Megascale-fineTuning/train.py:35 NANO_TO_ANGSTROM=0.1 and train.py:318 `batch['coords'] = batch['coords'] * NANO_TO_ANGSTROM` inside normalize_batch (identical pair at evaluate.py:51 and :137). So 0.1 is right and applied once. gate_u3u4 independently pins it: "PASS _COIL_COORD_SCALE matches train.NANO_TO_ANGSTROM (0.1)" and "PASS fixed-b coil does not saturate the kernel to ~0". U4 UNITS CORRECT.
  On realistic model-unit coords (CA-CA = 0.380): NODE b fitted=0.380000, fixed=0.582000, ratio 1.532 — same order, no saturation. Confirms gate_u3u4's "ratio 1.52".

[5] NEW DEFECT FOUND — U4 x U6 b-mismatch, still live (previously logged in results/VERIFY_9_ITEMS.md:65 and NOT fixed).
  coil_topology.py:211-226 `def coil_bond_length(ca_folded, mask=None)` hardcodes the fitted arithmetic and never reads CFG.coil_b; its own docstring (:218-221) says "A4's fix (b = 5.82 A fixed) is a separate arm and is deliberately NOT applied here ... If the A4 arm lands, _flory_unfolded_graph and this function must change TOGETHER." The A4 arm HAS landed (train_utils.py:521 _coil_bond_length branches on CFG.coil_b; --coil_b is live argparse at Megascale-fineTuning/train.py:82, bound at :120).
  Measured:
    CFG.coil_b=fitted   NODE b=0.380000   EDGE(U6) b=0.380000   equal=True   ratio=1.000
    CFG.coil_b=fixed    NODE b=0.582000   EDGE(U6) b=0.380000   equal=False  ratio=1.532
    node-feature coil d(0,10) = 1.840446 ; U6 edge coil d(0,10) = 1.201666  => DIFFERENT coils, factor 1.53x
  Live consumption path confirmed: model/hydro_net.py:529 and :884 both call `_CT.per_half_ca_coords(...)`, and coil_topology.py:303 inside per_half_ca_coords calls `b = coil_bond_length(src_row)`. So under the exact flag set CONTEXT.md recommends (--flory_unfolded --coil_b fixed --coil_edges) the node features use b=0.582 while the edge geometry uses b=0.380. No crash, plausible number — the precise failure mode U6 exists to prevent.
  gate_u5u6.py cannot catch it: its check "coil_bond_length reproduces the coil's own b (torch.equal)" hardcodes the same pre-U4 formula as its reference, so the gate PASSES while the code is wrong. Confirmed the gate passes above.

[6] SCOPE — no --coil_edges run exists.
  grep -rn "coil_edges" --include=*.sh --include=*.slurm . -> NO HITS.
  squeue -u nissimb -h | wc -l -> 113 queued; the coil-named jobs are all ddG-factorial cells (DeepEF_p3_a*_d*_s*_D0_coil_seed42) plus DeepEF_sa_coil_seed42. So U5/U6 are unexercised in flight and the U4xU6 defect is not currently corrupting any running job — it blocks the arm that has yet to launch.

[7] No repo mutation by me. git -C /home/nissimb/DeepPEF status --porcelain shows only pre-existing " M Megascale-fineTuning/evaluate.py", " M train_utils.py" and untracked .bak files, all predating this session. No sbatch/scancel/kill/push issued.

**DEFECTS:**
- FATAL (blocks the U5/U6 arm launch): coil_topology.py:211 coil_bond_length() hardcodes the 'fitted' b and never reads CFG.coil_b, while train_utils.py:521 _coil_bond_length() does branch on it. Under the recommended flag set --flory_unfolded --coil_b fixed --coil_edges the NODE-feature coil uses b=0.582 and the U6 EDGE coil uses the fitted b (measured 0.380): a 1.53x divergence, so node and edge halves describe two different coils. Measured d(0,10): 1.840446 (node) vs 1.201666 (edge). Live via model/hydro_net.py:529 and :884 -> coil_topology.py:303. Silent: no crash, plausible output. Previously reported in results/VERIFY_9_ITEMS.md:65 and STILL NOT FIXED. Fix: make coil_bond_length take the same CFG.coil_b branch (or accept b as a parameter supplied by the caller from the single source of truth in train_utils).
- GATE BLIND SPOT: scripts/gate_u5u6.py test [7] 'coil_bond_length reproduces the coil's own b (torch.equal)' uses its own hardcoded copy of the pre-U4 fitted formula as the reference, so it passes while the code is wrong. The gate cannot catch defect 1 and gives false assurance. Fix: assert coil_bond_length against train_utils._coil_bond_length under BOTH CFG.coil_b settings, not against a local re-implementation.
- LIVE SILENT NO-OP in the eval path: Megascale-fineTuning/evaluate.py:463 calls self.model(all_graph_minibatch) with NO n_folded, so n_folded=None -> split_folded_rows returns (B,0) -> apply_coil_edges and per_half_ca_coords return their input object UNCHANGED. Any --coil_edges evaluation run through evaluate.py silently scores the coil-OFF topology while logging the coil-ON arm name. evaluate.py is live (Megascale-fineTuning/run_calib_eval.sh:45). train.py:964/:1000 pass n_folded correctly; evaluate.py was never updated. Fix: pass n_folded=folded_graph_minibatch.size(0) at evaluate.py:463, or have the model RAISE when CFG.coil_edges is on and n_folded is None (the project's own 'raise, never no-op' rule, which apply_coil_edges follows for the flag combination but not for a missing n_folded).
- MEASUREMENT FRAMING (not a code bug, but it will not survive a viva as written): in results/w0_dg.json every arm has MAE == -bias exactly, i.e. all 28 proteins are under-predicted by all five arms. The 4.9650 -> 3.9521 gain is therefore pure OFFSET shrinkage, and correlation is flat (0.3464 -> 0.3404, re-derived). This is fully consistent with — and good evidence for — a b_p effect, but CONTEXT.md:403 'improves absolute dG MAE by 1.01 (20%)' reads as an accuracy claim. State it as offset/b_p shrinkage with flat ranking, and add a paired test/CI on n=28.
- COVERAGE: no --coil_edges training or eval run exists anywhere (grep over all .sh/.slurm returns nothing) and none of the 113 queued jobs sets it. The strongest dG result (coil_b fixed) and the entire U5/U6 lever are untested in flight, so 'the coil is only half applied' (the confound --coil_edges exists to remove) is still unmeasured.

## W5 burial lever — end-to-end verification (gate mechanics, folded/unfolded physics, dG-side scoring, missing-tensor recovery, CB-only adequacy)
**Verdict:** NEEDS-FIX

**DATA USED OPTIMALLY:** NO — this was the highest-value finding and it is now fixed at the measurement level. w5_dg.py hardcodes TDIR=/groups/keasar_group/casp15/meytav/protein_tensors, which holds only 20 dirs, one of which (2K5H) is a STUB containing nothing but prott5_embeddings/ and no coords_tensor.pt. That is exactly 19 usable -> the reported n=19. The proteins are NOT lost: /groups/keasar_group/casp15/meytav/MutationFineTuning/test_protein_tensors holds 76 proteins and covers 27 of our 28 (all of 2K5H, 2K1B, 2K28, 2KXD, 1W4H, 2L33, 6EWT, r11_1081, r12_757); the 28th, r18_3_TrROS_Hall, is in the OLD root, so the UNION gives a clean 28/28. I proved the two roots are the same dataset rather than different builds: for all 18 proteins present in both, coords_tensor.pt is bit-for-bit allclose (18 identical, 0 differing). I proved the recovered coords are real Angstrom: CA-CA virtual-bond median is 3.82-3.84 A for every recovered protein (canonical 3.8). Re-deriving on the union at n=28 REPLICATES the direction and, importantly, WEAKENS the headline: mean_burial~b_p goes -0.343 (n=19) -> -0.265 (n=28), against a significance threshold that tightens 0.490 -> 0.392. So it remains NOT SIGNIFICANT, and the "not a random third" caveat is resolved rather than left hanging. The length confound also persists and still dominates burial in magnitude (length~b_p = -0.303 vs burial -0.265), so burial and length are still not disentangled. What is still left on the table: this whole exercise is a b_p-side signal check, NOT a measurement of the lever. W5 widens 1092->1095 and no existing checkpoint has weights for those 3 columns, so the only thing that can actually score W5 is a TRAINING run with --burial_features under --loss_mode dg, scored on dG MAE and std(b_p). That is correctly identified in the .md and correctly NOT faked. Also untested: the HSE arm (burial_mode='hse') has never been scored on anything — it passes only mechanical gates.

**EVIDENCE:** GATE + BASELINE (cluster, esm2_env_py38, WANDB_MODE=disabled):
$ python scripts/gate_w5.py  -> 18/18 PASS, "W5 GATE: ALL PASS"
$ python scripts/gate_g4_cpu.py -> "baseline forward ok, dG finite PASS dG=-0.0030 width=1092" ... "G4-CPU: ALL PASS"; also "burial_features PASS dG=0.0033 width=1095", "width: burial adds exactly 3 PASS 1095". Re-run at end of session: still ALL PASS. G4 UNMOVED.

(1) ALL 18 CHECKS ARE MECHANICAL — read every check() call. md5sum scripts/gate_w5.py = f3e1a7afc3e30be218240c2f29c833b2 (5352 bytes), byte-identical to my local copy. grep -c "^check(" = 18 (19th hit is the `def check` line). Inventory: 4 shape/width/reproducibility/byte-identity (G1); 3 block-position (D+Fb block unchanged, one-hot right-anchored, emb right-anchored); 5 folded-vs-unfolded semantics (burial differs, burial==0 unfolded, burial!=0 folded, hydropathy identical, product column == burial*hydropathy); 1 length-confound correlation (r=0.351, threshold |r|<0.5); 3 hydropathy-table ordering (I argmax, R argmin, F>D); 2 HSE arm (differs from count and in [0,1]; HSE <= count). NOT ONE scores model performance — no MAE, no PCC, no loss, no held-out metric anywhere. Stated plainly: gate_w5 proves the feature is WIRED, never that it HELPS.

(2) PHYSICS VERIFIED AT SOURCE, not just via gate. train_utils.py burial block: `if folded: bur = fn(x, mask)` else `bur = torch.zeros_like(hyd)`; `hyd = one_hot @ _KD_NORM...` computed OUTSIDE the branch. So burial DIFFERS folded/unfolded (does not cancel in dG -> lever is live) and hydropathy is IDENTICAL in both (it is sequence, not structure) — both by construction, and both re-confirmed by gate checks. _BURIAL_R=10.0, _BURIAL_CAP=30.0 (constant, never N — docstring explicitly records that an earlier version divided by N and injected a length confound).

(3) UNITS BUG — CONFIRMED ABSENT FROM CURRENT SCRIPT. grep -n "0\.1|scale|\*=|/ *10" scripts/w5_dg.py returns ONLY lines 69/71, which are COMMENTS documenting the bug ("Scaling by 0.1 here saturates... angstrom -> mean 0.48/0.41 std 0.18/0.16; 0.1x -> 1.0/0.0"). The live call is bare: `b = tu.compute_burial(x, m)` on raw loaded coords. No scaling. Re-ran scripts/w5_dg.py: reproduced -0.343/-0.282, n=19, threshold 0.490 EXACTLY, matching results/w5_on_dg.json and results/W5_ON_DG.md line for line.

(4) MISSING-PROTEIN ROOT CAUSE. `ls $TDIR | wc -l` = 20. Per-dir classification: 19 have coords_tensor.pt; 2K5H has ONLY prott5_embeddings/ (STUB). Against the 28 test proteins: 19 COORDS, 1 STUB (2K5H), 8 NO-DIR (1W4H, 2K1B, 2K28, 2KXD, 2L33, 6EWT, r11_1081_TrROS_Hall, r12_757_TrROS_Hall). Recovery root found via `find /groups/keasar_group/casp15 -maxdepth 3 -type d -name "*tensor*"`. Cross-root identity: "overlap identical=18 differ=0". Angstrom check printed above. Union re-derivation at n=28: mean_burial~b_p pearson -0.265 spearman -0.194; frac_buried~b_p -0.060; length~b_p -0.303; "n=28: |r| < 0.392 indistinguishable from zero at p=0.05"; "2K5H present: True | b_p=2.522".

(5) B_P EXTRACTION SELF-VALIDATED: std(b_p) over the 28 = 1.5741, reproducing the established measured fact exactly. Top |b_p|: 2KVS -4.704, r12_757_TrROS_Hall +2.747, 2K5H +2.522, r18_3 +2.500. Note the two proteins carrying 78% of the oracle gain (2KVS, r12_757) were SPLIT across the two roots — 2KVS in old, r12_757 in new — so no single root could ever have seen both.

NO MUTATIONS: git status --porcelain shows only pre-existing modifications from earlier work (train_utils.py, evaluate.py, various .bak). I ran only read-only scripts plus w5_dg.py, which rewrote results/w5_on_dg.json to byte-identical values. No sbatch/scancel/kill/push; autopilot untouched.

**DEFECTS:**
- scripts/w5_dg.py:57 hardcodes TDIR='/groups/keasar_group/casp15/meytav/protein_tensors', a root that contains only 20 of the 28 test proteins. This silently discarded 9 proteins (32% of the test set) and the loss was reported as an unexplained data caveat rather than diagnosed. The proteins were never missing — 27/28 are in /groups/keasar_group/casp15/meytav/MutationFineTuning/test_protein_tensors and the 28th (r18_3_TrROS_Hall) is in the old root; the union covers 28/28. FIX: search a root LIST, not a single hardcoded path.
- 2K5H's directory EXISTS in the old root but contains only prott5_embeddings/ with no coords_tensor.pt. A directory-existence check would have called it present; only the file-level check caught it. Any other script that tests os.path.exists(dir) rather than os.path.exists(dir/coords_tensor.pt) will silently mis-handle 2K5H — and 2K5H is one of the high-|b_p| proteins.
- The headline number in results/W5_ON_DG.md and results/w5_on_dg.json (-0.343, n=19, threshold 0.490) is computed on a non-random 68% subset and is now superseded: at the full n=28 it is -0.265 against threshold 0.392. Both are non-significant so the CONCLUSION is unchanged, but the published figure and the 'n=19 of 28' caveat are stale and should be regenerated on the union.
- The length confound is not resolved and gets no weaker with more data: length~b_p (-0.303) still exceeds burial~b_p (-0.265) in magnitude at n=28, and burial correlates with length. Neither is significant, so burial cannot be claimed as an independent b_p signal on this dataset.
- The HSE arm (CFG.burial_mode='hse') is exercised by exactly two mechanical gate assertions (differs from count, HSE<=count) and has never been scored against any target — not b_p, not dG, not ddG. It is an unscored second lever riding along inside a lever that is itself unscored.
- W5 still has NO performance measurement of any kind. It widens 1092->1095, so no existing checkpoint has weights for those 3 columns and no forward-pass evaluation can score it. Its only scheduled scoring is the S7 factorial whose EFFECT_MIN is defined on POOLED ddG — the wrong metric, since burial is folded-only and acts on dG/b_p. This is the mis-scoring the task set out to find, and it is real: correctly diagnosed in the .md, correctly NOT faked, and still unfixed. AUTOPILOT.md needs a dG-side acceptance criterion or a lever that shrinks std(b_p) without moving pooled ddG will keep being recorded as noise.

## Factor C: --slope_weight (the a_p decompression lever) — end-to-end verification
**Verdict:** NEEDS-FIX

**DATA USED OPTIMALLY:** Partially. USED WELL: the term does real work — data loss alone minimises at s=r giving a_p=0.637; the term lifts the equilibrium to a_p=0.798, a genuine +0.16. It correctly exploits the within-protein ddG contrast (WS-1 WT=row 0, verified 368/368 in data), and it is correctly shift-invariant so it does not contaminate b_p. LEFT ON THE TABLE, in order of value: (1) The selection/scheduler metric is pooled-dG PCC, which is mathematically INVARIANT to s — so all 40 queued jobs will produce numbers that cannot distinguish slope_weight>0 from slope_weight=0. Logging per-protein a_p and slope_loss in validate() is a ~5-line change that converts 40 unfalsifiable runs into a measurable result; this is by far the highest-value unused data, since the ingredients (ddg_e, ddg_p, perprot_pcc) are ALREADY computed in validate() at lines 878-890 and simply discarded. (2) a_p = r*s means the (1-r) half of the compression is untouched and needs a different lever entirely — the thesis should decompose measured a_p=0.4990 into its r component (0.798) and s component (0.6253) rather than treating a_p as one number. (3) The block-spread ratio (0.9008) is a known, measured, correctable 10% target bias that nothing currently corrects. (4) n-weighting of blocks is free and unused.

**EVIDENCE:** All cluster commands via `python cluster.py exec` into /home/nissimb/DeepPEF, conda esm2_env_py38, WANDB_MODE=disabled.

1) `python scripts/gate_slope.py` -> 22 checks, "SLOPE GATE: ALL PASS", EXIT=0. true ddG pop-std=1.683734; check 8: "spread 0.8419 -> 1.6794 (target 1.6837)".

2) INDEPENDENT RE-DERIVATION (local fresh session, slope_theory.py, float64, seed 1234, n=37, different data from the gate):
 - shift-invariance: shifts -3.7/0.0/+1.9/+100.0 all give loss 0.6683377510, max delta 1.11e-16. PASS.
 - gradient mean-freeness (NOT in the gate): mean of pred during 400 descent steps 4.810647 -> 4.810647, drift 0.00e+00. Proves the term cannot move b_p.
 - descent: spread 0.4678 -> 0.6841 (target 1.3367), monotone-toward=True. PASS.

3) THE CORE ALGEBRA (slope_theory.py part B / slope_theory2.py part C):
 identity a_p = r * s, s = std(pred)/std(true). Constructed preds with std exactly matched and r forced:
   r=1.00 -> slope_loss=0.00e+00 -> a_p=1.0000
   r=0.90 -> slope_loss=0.00e+00 -> a_p=0.9000
   r=0.80 -> slope_loss=0.00e+00 -> a_p=0.8000
   r=0.60 -> slope_loss=0.00e+00 -> a_p=0.6000
 => zero slope-loss does NOT imply a_p=1; it implies a_p=r.
 L_std minimised at s=1 -> a_p=r ; L_slope=(1-a_p)^2 minimised at s=1/r -> a_p=1.
 At our r=0.798 they demand s=1.000 vs s=1.2531 -> 25.3% disagreement in requested spread.

4) EQUILIBRIUM SWEEP (slope_theory2.py part E), total = MSE(s) + W*|s-1|*sd, r=0.798:
   W=0.00 -> s*=0.7980 a_p*=0.6368
   W=0.25 -> s*=0.8954 a_p*=0.7145
   W=0.50 -> s*=0.9929 a_p*=0.7923
   W=1.00 -> s*=1.0000 a_p*=0.7980
   W=2.00 -> s*=1.0000 a_p*=0.7980
   W=5.00 -> s*=1.0000 a_p*=0.7980   <-- saturates; a_p is hard-capped at r.
 Also part D: PCC is CONSTANT across s=0.5..1.4 (0.7980 at every s) while RMSE is minimised at s=0.800=r.

5) BLOCK BIAS, re-measured live: `python cluster_run/code/fix3_block_std.py --out /home/nissimb/fix3_rerun.json`
   -> proteins=332 skipped=530 mb=64; ratio median=0.901 IQR=[0.863,0.927] min=0.691; position-ordered (median monotonicity 1.000): True.
   Ceiling: a_p <= 0.798 * 0.9008 = 0.7188.

6) CODE THAT RUNS (Megascale-fineTuning/train.py): slope built at 732-737, added at 740-741, INSIDE `for j in range(0, batch['prott5'].size(1), self.mini_batch_size)` (line 685) with MINI_BATCH_SIZE=64 (line 41). Line 737 is verbatim what the gate tests.

7) GUARD FIRES LIVE (with PYTHONPATH=/home/nissimb/DeepPEF; without it an ImportError masked the test and it falsely "passed"):
   `--slope_weight 0.5 --loss_mode dg` -> "File train.py, line 232, in <module> raise ValueError: --slope_weight 0.5 requires --loss_mode ddg|joint|ddg_head..."
   `--slope_weight 0 --loss_mode dg` -> 0 matches (no raise). `--slope_weight 0.5 --loss_mode ddg` -> 0 matches (passes guard).

8) WS-1 VERIFIED IN DATA (not just code): scanned all 862 mutation CSVs -> "row0 looks like WT: 368  NOT-wt: 0  empty:494". Zero violations.
   Last-block sizes V mod 64: ==1:5 (correctly skipped by n>=2 guard), ==2:2, (1<=x<=4):15 of 368.

9) ddg_head DEAD PATH: ddg_head `continue` at line 700; slope build at lines 732 and 740; all(s>c)=True -> slope term never built under --loss_mode ddg_head, which the guard explicitly advertises as valid.

10) SCORING: validate() line 884 `pc_corr = corrcoef(val_dg, val_dg_pred)` = pooled dG. Consumed by `self.scheduler.step(pc_corr)` (808) and epoch selection (823 val_pc_corr, 832 return). Line 896 overrides to ddg_pcc ONLY when LOSS_MODE=='ddg_head'; submit_factorial.sh line 82 uses `--loss_mode ddg`, so the override never fires. `grep -n slope_loss train.py` -> only 737 and 741: never logged. No a_p/slope computed in validate().

11) G4 GATE after all work (no code changed): "baseline forward ok, dG finite PASS dG=-0.0030 width=1092" / "G4-CPU: ALL PASS". Matches g4_final.txt line 1 exactly.

**DEFECTS:**
- SCORED ON THE WRONG METRIC (most serious). validate() returns pc_corr = pooled dG correlation (train.py:884), which drives scheduler.step() (808) and epoch/checkpoint selection (823, 832). The ddg_pcc override at line 896 fires only under --loss_mode ddg_head, but submit_factorial.sh:82 runs --loss_mode ddg, so it never fires. Proven in evidence part D: PCC is mathematically INVARIANT to the spread ratio s (0.7980 at s=0.5 through s=1.4), so the model-selection metric cannot see what this lever does. Violates the metric rule: a whole-arm effect scored on a quantity that provably cancels it.
- THE TERM'S OWN OPTIMUM IS NOT THE STATED TARGET. Because a_p = r*s, driving |std(pred)-std(true)| to exactly 0 yields a_p = r, not 1. At our measured per-protein r=0.798 the term is fully satisfied at a_p=0.798 and exerts zero further pull. Confirmed by construction (r=0.9->a_p=0.900, r=0.6->a_p=0.600 at slope_loss=0.00e+00) and by the equilibrium sweep, where s* saturates at 1.0000 and a_p* at 0.7980 for W=1,2,5. Raising slope_weight cannot fix this; the (1-r) component of compression needs a different lever. std-matching and slope-matching diverge whenever r<1, demanding spreads 25.3% apart at r=0.798.
- NOT OBSERVABLE: slope_loss appears only at train.py:737 and 741 and is never passed to wandb_log (unlike l1_loss, reg_loss, energy_reg, wt_anchor_loss, ddg_loss which all are). Per-protein a_p is never computed or logged in validate() either, even though ddg_e/ddg_p/perprot_pcc are already built there (878-890). The single lever aimed at a_p reports neither its own loss nor its target quantity, so the 40 queued jobs cannot demonstrate whether it worked.
- BLOCK-FRAGMENTATION BIAS: the term is inside the mini-batch loop (train.py:685, MINI_BATCH_SIZE=64), so it matches the spread of ONE contiguous 64-row block, while a_p is defined over the whole protein. CSVs are position-ordered (monotonicity 1.000), so blocks are narrow sequence windows. Live re-measurement: mean_block_std/protein_std median 0.9008 (min 0.691). The term therefore targets a spread ~10% too small, lowering the ceiling to a_p <= 0.798*0.9008 = 0.7188. Prior fix3 verdict 'fine as written' rests on a 0.9 threshold the value only barely clears (0.9008).
- ddg_head IS A DEAD PATH: the argparse guard (train.py:232-235) explicitly advertises --loss_mode ddg|joint|ddg_head as valid for --slope_weight, and --loss_mode ddg_head passes the guard, but the ddg_head branch executes `continue` at line 700 while the slope term is built at 732/740. The combination runs, completes and prints success while the feature is never read — the project's signature failure mode, currently reachable via a documented flag combination.
- abs() DOES NOT ANNEAL: d/ds |s-1|*sd has constant magnitude and flips sign discontinuously at s=1, so near the optimum the term chatters rather than converging (no vanishing gradient), and under Adam's normalisation it applies a persistent unit-scale pull. A squared or Huber form would settle. Minor relative to the above but affects stability at larger slope_weight.
- n=2 BLOCKS GET FULL WEIGHT: the n>=2 guard correctly skips n=1 (5 of 368 proteins have V mod 64 == 1), but a 2-row block's std is just |x1-x2|/2 — a single pairwise gap — and contributes the same slope gradient as a 64-row block, since the term is unweighted by block size. 2 of 368 proteins hit exactly n=2; 15 have final blocks of 1-4 rows.

## Catalogue schema guard + het ligand classification + catalogue_vs_bp correlation scan (items 1-5)
**Verdict:** NEEDS-FIX

**DATA USED OPTIMALLY:** The 90-test scan is scored correctly per the METRIC RULE: whole-protein features go to b_p/pcc, and the one real hit (mean_rel_SASA) lands on a_p, which is a within-protein slope and does NOT cancel in ddG. That is the right pairing and it is the honest result.

LEFT ON THE TABLE - the 100,246 catalogue rows. Honest answer: near-zero for THIS thesis, narrow but real for a future one.

Worth nothing for our 28: the join is 0/28 BY CONSTRUCTION (catalogue = PDB pre-training decoy set, split is 100,086 training / 160 validation; our 28 are held-out MegaScale). 7 of the 28 are de-novo designs that CANNOT exist in a PDB decoy set. More fundamentally the catalogue carries NO dG, NO ddG, NO b_p - only loss/lossd/lossc, which are decoy-discrimination and are barred by the METRIC RULE. So there is no label to regress a b_p corrector against. A "b_p corrector trained on catalogue features" is not merely unjoinable, it is unsupervised - there is no b_p in this file at all. That idea should be dropped, not deferred.

What it IS good for, concretely, and only if a ligand-bearing benchmark ever arrives: it is a well-characterised FEATURE-ENGINEERING and STRATIFICATION substrate. 43,832 rows carry >=1 genuine bound ligand (45,328 after the NA fix), 30,292 metal, 9,529 cofactor, and critically 20,151 rows carry ONLY crystallization additives - that additives-only stratum is the exact confounder that makes a naive "has ligand" feature learn crystallography instead of biochemistry, and this catalogue is where you would learn to separate them. The het table (100% occurrence coverage once NA is fixed, 87.7% of occurrences hand-curated) is genuinely reusable IP independent of the catalogue.

Second real use, available today: it is a NEGATIVE CONTROL / scope boundary. It is the cleanest available evidence for the thesis claim that ligands, complexes and interface BSA are UNMEASURABLE on our test set rather than absent - 43.7% of PDB entries bear a bound ligand while 28/28 of ours bear none. That is a defensible limitations-section paragraph, not a result.

Weak spot worth stating: only 1,843 rows (1.84%) fall in our 43-72aa window, and 1,070 are small+monomeric+ligand-free, so even as a transfer source the size-matched overlap is thin.

**EVIDENCE:** ALL VIA cluster.py exec, conda esm2_env_py38, /home/nissimb/DeepPEF. No file modified (timestamps re-checked at end: het_classification.csv 14:37, catalogue_schema.py 14:37, catalogue_vs_bp.json 10:20, het_freq.tsv 14:29 - all unchanged).

MANDATORY GATE: `python scripts/gate_g4_cpu.py` -> "baseline forward ok, dG finite PASS dG=-0.0030 width=1092 ... G4-CPU: ALL PASS", exit 0. UNMOVED. (Nothing to revert - I changed nothing.)

ITEM 1 - re-derived from raw CSV independently (pd.read_csv FINAL_DATASET_100k_030926.csv, shape 100246 x 21):
  Ligands_x       nonnull=4      nuniq=4      <- CONFIRMS "4 non-null of 100,246"
  BSA_Numeric_x   nonnull=100246 nuniq=4      values [0.0, 18.76, 26.99, 35.95]  <- CONFIRMS "4 unique"
  BSA_Percentage  nonnull=5      nuniq=4
  lossg           nonnull=100246 nuniq=1      value [0]
  Global Symmetry nonnull=100246 nuniq=1      'Asymmetric'
Loader: load_catalogue() -> shape (100246, 18). Arithmetic is 21 - 5 dead + 2 derived (BSA_parsed_pct, BSA_inconsistent) = 18, NOT "21->18 = 3 dropped".
All 5 dead cols absent from frame; d[col] raises DeadColumnError naming live sibling (5/5); getattr raises; list-select d[["AA Length","Ligands_x"]] raises.
assert_not_decoy_loss RAISES MetricRuleError on all 4 of loss/lossd/lossg/lossc; returns True on "AA Length"; safe_correlate(d,"AA Length","loss") -> MetricRuleError. CONFIRMED.
BSA policy: bsa_inconsistent_rows = 48 (NOT 294). Decomposition: raw BSA_Numeric_y NaN = 246, flagged = 48, overlap = 0, 246+48 = 294 total NaN after policy. policy='drop' -> shape (100198, 18); 'keep_zero' -> emits RuntimeWarning. bsa_max_abs_disagreement_elsewhere = 0.000000 over 99,721 rows, which is what justifies the parse-failure reading.

ITEM 2 - het table. Re-derived occurrence coverage myself with split_het_codes over Ligands_y:
  total occurrences 205,648; distinct codes observed 8,187; codes in het table 8,186; covered 201,776 -> 98.1172%  <- matches the claim
  BUT the ENTIRE 3,872-occurrence gap is ONE code: [('NA', 3872)]. Nothing else uncovered.
  Root cause, from disk not from pandas: `awk -F, 'NR>1 && $1==""'` -> `,10,3872,metal,yes,yes,`  i.e. het_code field is EMPTY ON DISK for sodium.
  `sed -n '11p'` shows the row between EDO(rank 9) and PO4(rank 11) with a blank first field. csv module read: "'NA' present as literal text: False".
  Source of truth is FINE: data/het_freq.tsv line 10 = `NA<TAB>3872`, and build_het_classification.py line 22 METAL list contains NA.
  Rebuilt to scratch: `python scripts/build_het_classification.py data/het_freq.tsv /home/nissimb/het_rebuild_check.csv` -> fresh line 11 = `NA,10,3872,metal,yes,yes,`. diff ignoring CR vs shipped = EXACTLY 2 lines (one <, one >), the NA row only. So shipped file is a pandas round-trip corruption; builder is correct. (scratch file deleted.)
  Downstream impact via classify_ligands: shipped n_metal=37,002 / n_unknown=31,734 / n_bound_ligand=71,411 / rows>=1 bound=43,832.  Repaired: 40,874 / 27,862 / 75,283 / 45,328. Delta metal +3,872; 3,872 rows change bound-count; 1,496 rows FLIP from 0 bound to >0 bound. Repaired coverage = 100.0000%.
  Counts as stated otherwise verified: 8,187 rows; classification value_counts unknown 7,735 / additive 106 / substrate 90 / modres 80 / metal 61 / polyres 58 / cofactor 57; bound (metal+cofactor+substrate) = 61+57+90 = 208 CONFIRMED. counts_as_bound_ligand vs classification: 0 inconsistent rows. curated: 456 codes = 87.7% of occurrences.
  CHEMISTRY SPOT-CHECK, 45 highest-frequency + 62 probed codes. The three named in the task are RIGHT: SO4 -> crystallization_additive ("sulfate - crystallization precipitant"); NAD -> cofactor; MSE -> modified_residue with note "selenomethionine - amino acid IN the chain (MAD/SAD phasing), not a ligand", bound=no. Also correct: GOL/EDO/PEG/MPD/ACT/CL/PO4 additive; MG/ZN/CA/MN/K/FE/CU metal; HEM/FAD/FMN/NAP/SAH/SAM/COA/PLP/TPP/GSH cofactor; A/U/C/G/DA/DT/DC/DG polymer_residue; SEP/TPO/PTR/KCX/MLY/M3L/PCA/FME/CSO/CSD/OCS modified_residue; SF4 "iron-sulfur [4Fe-4S] cluster" metal; UNK/UNX/UNL unknown.
  Errors found (all curated=yes unless noted): BR(260)+IOD(475) classified METAL and counted as BOUND - they are halide ANIONS, phasing soaks, not metals. ACE(768)+NH2(526) classified additive - they are N-acetyl / C-amide TERMINAL CAPS covalently in the chain (same error class as MSE). MYR(167)/PLM(159)/OLC(86) additive - myristate/palmitate are lipidation mods / genuine bound lipids. ABA(67)+NLE(56) classified SUBSTRATE and counted BOUND - non-canonical amino acids IN the chain; caused by MODRES listing the typo "ABA1" so first-wins put() let SUBSTRATE take them. BTN(53) biotin -> unknown, absent from both lists, should be cofactor (curated=no). F(27) fluoride -> unknown (curated=no).

ITEM 3 - results/catalogue_vs_bp.json. n_tests=90, len(correlations)=90, 15 features x 6 targets (a_p, b_p, abs_b_p, pcc, b_p_wt_error, abs_b_p_wt_error) = 90. bonferroni_alpha = 0.05/90 = 5.5556e-4.
  Recomputed survivors across ALL 90 myself: clearing BOTH Pearson AND Spearman = EXACTLY 1 -> mean_rel_SASA vs a_p, r=+0.7139 p=1.99e-05, rho=+0.6229 p=3.999e-04. CLAIM CONFIRMED.
  plddt_mean vs pcc: pearson_p = 0.01509756 = 1.5e-2, and 0.0151/5.5556e-4 = 27.2x above the bar. CONFIRMED EXACTLY as stated (spearman_p 2.44e-4 does pass, pearson fails).
  Nothing survives against b_p: best b_p p-values are frac_charged p=0.157, mean_rel_SASA_hydrophobic p=0.638, plddt_mean p=0.247 - none near alpha. CONFIRMED.
  mean_rel_SASA r=+0.7139 matches the established fact r=+0.714. LOO range 0.642-0.790; natural-only n=21 r=0.767.

ITEM 4 - join. JSON: n_of_28_matched=0, 28 per_protein entries, all matched=False, sum n_exact_pdb4=0, sum substring on both id columns=0.
  Re-derived independently: 66,809 distinct 4-char PDB ids in catalogue; exact 4-char hits = 0/28. My looser full-substring test flagged 5, and I ran them down - all FALSE POSITIVES: they are 4-char PREFIXES of DESIGNED names (R12_, HHH_, R18_, R11_) matching unrelated PDB ids (1R12_1_A, 3HHH_1_A, ...); full-name membership = False for all 5. Join is genuinely 0/28.
  BY CONSTRUCTION: split = {training: 100086, validation: 160}; 7 of 28 are de-novo designs (r12_757_TrROS_Hall, HHH_rd1_0244/0142, HEEH_KT_rd6_0746/0793, r18_3_TrROS_Hall, r11_1081_TrROS_Hall) which cannot appear in a PDB decoy set; the other 21 are PDB-like natives and are all absent, consistent with a held-out split. CONFIRMED.

ITEM 5 evidence: catalogue columns after load = [rank, protein_id, split, loss, lossd, lossc, PDB_ID_and_Entity, Method, Resolution, AA Length, Oligomeric State, Is Complex?, Chain Composition, BSA, Ligands_y, BSA_Numeric_y, +2 derived] - NO dG/ddG/b_p label anywhere. AA Length numeric min/med/max = 20/477/89160; rows in 43-72aa = 1,843 (1.84%); small+monomeric+ligand-free = 1,070. Is Complex? = {Yes 67856, No 31852, Unknown 538}. Ligand strata: >=1 bound 43,832 (43.7%); metal 30,292; cofactor 9,529; additives-only 20,151; no het at all 24,782. Method = {X-RAY 88879, NMR 8545, EM 2161, Unknown 538}.

GATE CROSS-CHECK: `python scripts/gate_catalogue_integrity.py` -> "CATALOGUE-INTEGRITY: ALL 24 PASS", exit 0 - WITH the NA bug present. Its own het line prints "all-occurrence coverage 0.8770" (that is CURATION coverage, a different quantity from the 98.1% classification coverage) and no assert fires on either. grep for isna/notna/nunique/duplicat on het_code in that gate: no check exists.

**DEFECTS:**
- CRITICAL / data corruption: data/het_classification.csv has the sodium row's het_code field EMPTY ON DISK (`,10,3872,metal,yes,yes,` at line 11). Verified with awk/sed/csv-module, so it is not a pandas read artifact. NA is the 10th most frequent het code in the catalogue (3,872 occurrences). Consequence via classify_ligands: 3,872 sodium ions are scored 'unknown' instead of 'metal', and 1,496 catalogue rows are wrongly recorded as having ZERO bound ligand when they have one. True occurrence coverage is 100.0%, not the claimed 98.1% - the entire 1.9% 'gap' is this single corrupted key. Root cause is a pandas round-trip that wrote NaN as an empty string; data/het_freq.tsv (line 10 `NA<TAB>3872`) and the METAL list in build_het_classification.py are both CORRECT, and re-running the builder reproduces `NA,10,3872,metal,yes,yes,` with an exactly-2-line diff vs shipped. Fix = re-run the builder, or read with keep_default_na=False. This is the signature failure mode: the artifact is wrong while every success message is green.
- CRITICAL / gate blindness: scripts/gate_catalogue_integrity.py prints 'ALL 24 PASS' with the above corruption present. It has NO assertion on het_code being non-null or unique. Worse, its own emitted line says 'all-occurrence coverage 0.8770' while the documented claim is 98.1% - two different quantities (curation vs classification coverage), neither asserted, and the discrepancy is printed and ignored. load_het_classification() also uses a bare pd.read_csv, so it would re-introduce the NaN key even from a correct file. Needs: assert het_code.notna().all(), assert nunique == len, assert classification-coverage >= 0.99, and keep_default_na=False in the loader.
- Chemistry misclassification - BR (260 occ) and IOD (475 occ) are classified 'metal' and COUNT AS BOUND LIGAND. Bromide and iodide are halide ANIONS introduced as heavy-atom phasing soaks; they are neither metals nor biological ligands. ~735 occurrences inflate n_metal and n_bound_ligand. Both curated=yes, so this is a curation error, not heuristic fallout.
- Chemistry misclassification - ACE (768 occ) and NH2 (526 occ) are classified 'crystallization_additive'. They are N-terminal acetyl and C-terminal amide CAPPING GROUPS, covalently part of the polypeptide - exactly the MSE error class the table was built to avoid. Should be modified_residue. Currently harmless to n_bound_ligand (both already excluded) but wrong in the additive stratum, which is the stratum the catalogue's main future use depends on.
- Chemistry misclassification - ABA (67) and NLE (56) are classified 'substrate' and COUNT AS BOUND LIGAND. Both are non-canonical amino acids incorporated IN the chain (alpha-aminobutyric acid, norleucine). Cause is a typo: the MODRES list contains 'ABA1' instead of 'ABA', and since put() is first-wins with SUBSTRATE applied last, SUBSTRATE captured them. Also ORN/AIB/DVA are correctly modres but ORN appears in SUBSTRATE too.
- Chemistry misclassification - MYR (167), PLM (159), OLC (86), OLA classified 'crystallization_additive'. Myristate and palmitate are lipidation modifications / genuine bound lipids, not precipitants. Under-counts real ligand-bearing structures.
- Chemistry gap - BTN (biotin, 53 occ) falls through to 'unknown'; it is a textbook cofactor and is in neither COFACTOR nor any other list. F (fluoride, 27 occ) also falls through to 'unknown'. Both curated=no.
- Framing error in the brief being verified (not a code defect): 'the BSA policy nulls 294 rows' is wrong. The policy nulls 48 rows (the self-inconsistent ones). 294 is the TOTAL NaN count afterward = 246 pre-existing + 48 nulled, overlap 0. The code is right; the stated number conflates two things.
- Framing error - 'quarantines all 5 dead columns (21 -> 18)' invites the reading that 3 columns are dropped. Five are dropped and two derived columns are added: 21 - 5 + 2 = 18. The coincidence makes a real regression (e.g. a dead column silently surviving while a derived one is missing) invisible to a shape check alone.
- Misleading field name in results/catalogue_vs_bp.json: multiple_testing.survivors contains 5 entries, but only 1 (mean_rel_SASA vs a_p) actually clears Bonferroni on both tests. The other 4 pass on only ONE of Pearson/Spearman. The adjacent 'note' field does say 'Only these clear it', which is ambiguous and, read with the field name, overstates the result 5x. The headline text is correct and honest. Rename to 'candidates' with an explicit clears_both boolean.
- Unvalidated dtype: 'AA Length' is an object column containing the string 'Unknown' in 538 rows (0.54%), and catalogue_schema.py lists it as required but never checks its type. A plain .median()/.mean() on it raises TypeError (I hit this). 'Is Complex?' likewise carries 'Unknown' in the same 538 rows, and Method/Resolution too (9,162 null resolutions). Any future ligand/size analysis needs an explicit numeric coercion guard the schema does not currently provide.

## Autopilot (scripts/autopilot.py) end-to-end verification: liveness, D1 slope guard, empty-CSV refusal, D2 zero-variance guard, dG-side acceptance crit
**Verdict:** NOT-READY

**DATA USED OPTIMALLY:** POORLY. Nine factorial trainings have COMPLETED (5x D0_coil, 4x D1_uemb, seed42) at 6-11 GPU-hours each -- roughly 70 GPU-hours already spent -- and the autopilot has scored exactly zero of them, 150 consecutive passes. The data is on the table three separate ways. (1) PATH: collect_results scans ROOT/eval_results, but the eval jobs write to Megascale-fineTuning/models/eval_results. Those cell directories exist and are EMPTY (size=2, only . and ..) -- so even the right path would find nothing; the eval jobs that would fill them (ev_p3_a*, gev_p3_a*) are all still PENDING behind ~108 queued jobs. The trainings finished; the scoring never ran. This is the signature failure mode again: COMPLETED training, no artifact. (2) METRIC: std_b is parsed by calib_diag, carried in TSV_COLS and written to RESULTS.tsv, then never read by any decision. main_effects and select_best rank on pooled ddG alone. REF_STD_B=0.223 is defined at line 35 and referenced nowhere else. So the one measurement that could detect a dG-side/reference-state lever is collected and discarded -- the richest column in the table is inert. (3) DESIGN: the queue holds BOTH factor-D arms (48 D0_coil + 47 D1_uemb) while state.json records D0 as decided for --unfolded_emb {full,zero}. Either the D0 decision is not being honored or the factorial is 2x its documented size; and the D0 evidence itself says r_coil=1.178, coil INCREASES unfolded variance, i.e. the coil arm is the one D0 argued against. Also cells_total=16 in state.json vs '0/48' in the log, and state.json's submitted[] holds 16 placeholder job_ids 18000001-18000016 that do not exist in sacct, while the real jobs are 21049400-21050135. What is left on the table: the 9 finished checkpoints could be scored on CPU right now (a completed cpu_evals job 21085470 is RUNNING, and abl_*_e14.csv files of ~1.4MB were produced this way for the sigma/anchor arms), which would give 9 of 16 seed-42 cells today instead of waiting behind a 108-job queue.

**EVIDENCE:** ALL commands run via cluster.py exec (script piped on stdin; NOTE the documented heredoc form flattens to one line through the Bash tool and silently produces garbage -- I wrote script files instead).

1. LIVENESS. `pgrep -af autopilot.py` / `ps -eo pid,etimes,cmd | grep autopilot` -> "NO AUTOPILOT/WRAPPER ON THIS LOGIN NODE" (hostname slurm-login-01). Lock pidfile ~/auto/.autopilot.lock/pid = 958995; `ps -p 958995` -> "PID 958995 NOT RUNNING". Log IS advancing (stat: 16:34:34, then 16:42:02, date 16:42:28) so it lives on another round-robin login node. But: `grep -c 'wrapper: driver exited'` = 148; every one is rc=143 preceded by "Terminated". Restart cadence is exactly ~9 min (15:13, 15:22, 15:31, 15:40, 15:49, 15:58, 16:07, 16:16, 16:25, 16:34) = POLL_SECONDS(600) + startup, i.e. it is SIGTERM'd during time.sleep(POLL_SECONDS) on every single cycle by a login-node reaper. It is not alive-and-working; it is a crash-restart loop.

2. NEVER PROGRESSED. `grep -o 'S3: [0-9]*/48 cells scored' | sort | uniq -c` -> "150  S3: 0/48 cells scored" -- one distinct message, 150 times. `grep 'D1 |D2 |halt|recorded'` -> nothing. It has never reached D1 or D2.

3. D1 SLOPE GUARD -- PASSES. Synthetic rows where the HIGHEST-pooled cell is slope-collapsed, run against the LIVE module (sys.path -> /home/nissimb/DeepPEF/scripts, import autopilot):
   rows: CELL_HIPOOL_COLLAPSED pooled=0.700 slope_med=0.12; CELL_NOSLOPE pooled=0.680 slope_med=None; CELL_GOOD_MID 0.612/0.51; CELL_GOOD_LOW 0.590/0.62
   ->"D1 REJECT CELL_HIPOOL_COLLAPSED: pooled=0.7000 but slope_med=0.120 < 0.30 -- slope collapse; not selected (AUTOPILOT.md D1)"
   ->"D1 REJECT CELL_NOSLOPE: pooled=0.6800 but slope_med=missing < 0.30" (unknown slope correctly not a passing slope)
   ->"D1 GUARD BINDING: CELL_HIPOOL_COLLAPSED had the highest pooled (0.7000) but was rejected for slope collapse; selecting CELL_GOOD_MID (pooled 0.6120, slope_med 0.510) instead"
   BEST = CELL_GOOD_MID. Rejected AND logged, both. All-collapsed case returns None (caller halts H2). Signature check: select_best reads r.get('tag') and rows carry 'tag' (parse_tag returns dict(...tag=tag)) -- NO mismatch. SLOPE_MED_MIN=0.3 matches AUTOPILOT.md lines 128-130 verbatim.

4. EMPTY-CSV REFUSAL -- PASSES ON A REAL FILE. Wrote a 23-byte header-only abl_p3_a0_d0_s0_D0_uemb_seed42_e14.csv plus a real 1,412,228-byte sibling into the scanned dir, ran collect_results():
   -> "SKIP p3_a0_d0_s0_D0_uemb_seed42: eval CSV is 23 bytes (< 1000) -- not a result, not scored"
   -> rows returned: 1 ['p3_a1_d1_s1_D0_uemb_seed42']; stub in rows = False. MIN_EVAL_CSV_BYTES=1000. Both temp files deleted afterwards (verified [False, False]).

5. D2 ZERO-VARIANCE GUARD -- LOGIC CORRECT, BUT DEAD CODE. lever_testable/admit_lever behave exactly right: 28 rows with n_het_residues=0,n_metal_residues=0 -> False + "driving feature 'n_het_residues' is CONSTANT across all 28 eval proteins (spread 0) ... any result would be a manufactured negative, not evidence"; varying input -> True; absent column -> False with "Absent is NOT constant -- refusing to score rather than assume". BUT: grep for callsites -> "CALLSITES of admit_lever: NONE -- DEAD CODE"; lever_testable's only callsite is line 256 inside admit_lever itself. Nothing in one_pass or main ever calls it. results/UNTESTABLE.md did not exist before my test and does not exist after (the 16:38 copy was mine; reverted).

6. dG-SIDE ACCEPTANCE CRITERION -- ABSENT. "EFFECT_MIN_STD_B in source: False". REF_STD_B appears on exactly one line (35, the definition). std_b appears only at line 49 (TSV_COLS), 167 (calib_diag regex), 419 (TSV write), 437-438 (self_audit sanity std_b<=0). main_effects is 100% pooled: hi/lo/same/diff all built from r['pooled']. select_best ranks max(eligible, key=pooled). So a lever that shrinks std(b_p) without moving pooled ddG is recorded as NOISE and routed to the F1 "genuine negative result and a thesis finding" branch.

7. STATE. results/state.json: state=D0_DECIDED, cells_done=0, cells_total=16, blocked=[], retries={}, two decisions (D0 factor_D := --unfolded_emb {full,zero}; FIX3 slope kept). submitted[] = 16 entries with job_id 18000001-18000016, run_tags p3_a*_D0_uemb_seed42. `sacct -j 18000001,18000002` returns nothing -- those job ids do not exist. Real factorial jobs are 21049400-21050135 with tags D0_coil and D1_uemb. results/RESULTS.tsv is header-only (123 bytes).

8. QUEUE/ARTIFACT TRUTH. squeue -u nissimb: 108 PENDING, 6 RUNNING. `ls -d checkpoints/*p3_a*` -> 0; checkpoints/ does not exist at ROOT. sacct COMPLETED factorial trainings (JobName%60): DeepEF_p3_a0_d1_s1_D0_coil_seed42, a1_d0_s0_D0_coil, a1_d0_s1_D0_coil, a1_d1_s0_D0_coil, a1_d1_s1_D0_coil, a0_d0_s0_D1_uemb, a0_d0_s1_D1_uemb, a0_d1_s0_D1_uemb, a0_d1_s1_D1_uemb = 9 completed. Their eval dirs under Megascale-fineTuning/models/eval_results/ (abl_p3_a1_d0_s0_D0_coil_seed42_e14 etc.) are all DIRECTORIES of size=2, i.e. EMPTY. All ev_p3_a*/gev_p3_a* eval jobs: PENDING. ROOT/eval_results holds only the 10 older sigma/anchor/slope CSVs; collect_results() on the live tree returns 0 rows and 0 filenames match CELL_RE.

9. G4 GATE AFTER MY CHANGES -- PASS, UNCHANGED. `python scripts/gate_g4_cpu.py` -> "baseline forward ok, dG finite PASS dG=-0.0030 width=1092" ... "G4-CPU: ALL PASS". Exactly the required dG=-0.0030 width=1092.

10. REVERTED. Removed results/UNTESTABLE.md, deleted the 'untestable' key my admit_lever test added to results/state.json, removed the two stub CSVs and /home/nissimb/d1test.py, /home/nissimb/t8.py. No sbatch/scancel/kill issued; autopilot untouched.

**DEFECTS:**
- BLOCKER -- autopilot is in a 148-cycle SIGTERM crash-loop. Every pass ends rc=143 during time.sleep(POLL_SECONDS=600); restart cadence is a metronomic ~9 min from 15:13 to 16:34. A login-node reaper kills it mid-sleep on every cycle. The wrapper masks this by restarting, and the advancing log makes it LOOK alive. It has completed 150 passes and 0 units of work. Fix: run it under the existing tmux (a 'collect' session already exists) or as a batch job, and/or poll in short sleeps so it is not idle for 10 minutes at a stretch.
- BLOCKER -- collect_results() reads the wrong directory. It scans os.path.join(ROOT,'eval_results'); the eval jobs write to ROOT/Megascale-fineTuning/models/eval_results. Zero filenames in the scanned dir match CELL_RE, so 9 completed factorial trainings are invisible. 'S3: 0/48 cells scored' x150 is this bug, not a genuine wait.
- BLOCKER -- the 9 finished trainings have EMPTY eval output directories (size=2). Their ev_p3_a*/gev_p3_a* scoring jobs are all still PENDING behind ~108 queued jobs. Even with the path fixed there is nothing to score. Training COMPLETED with no artifact -- the signature failure mode, third occurrence.
- MAJOR -- NO dG-SIDE ACCEPTANCE CRITERION. EFFECT_MIN_STD_B does not exist in the source. REF_STD_B=0.223 is defined once and never used. main_effects() and select_best() operate exclusively on pooled ddG. std_b is parsed, carried in TSV_COLS and written to RESULTS.tsv, then dropped. Per the METRIC RULE this is the fourth-lever failure in structural form: any reference-state or whole-protein lever that shrinks std(b_p) while leaving pooled ddG flat gets classified by the F1 branch as 'a genuine negative result and a thesis finding'. That is a manufactured negative on the dG side, and it is exactly the failure the rule exists to prevent.
- MAJOR -- the D2 zero-variance guard is DEAD CODE. admit_lever() has zero callsites; lever_testable() is called only from inside admit_lever. The logic is correct in isolation (verified) but nothing in one_pass() or main() invokes it, and results/UNTESTABLE.md has never been written by the autopilot. Item 4 does NOT fire. If ligand_nodes ever entered the factorial it would be scored and given a zero -- the exact manufactured negative the guard was written to stop.
- MAJOR -- state.json is stale/fabricated relative to reality. submitted[] lists 16 job_ids 18000001-18000016 that do not exist in sacct (real ids are 21049400-21050135), and its run_tags are p3_*_D0_uemb_seed42, a combination that appears nowhere in the queue. cells_total=16 contradicts the log's '/48'. Nothing in the file reflects the 9 completed trainings.
- MAJOR -- the D0 decision is not being honored by the queue. state.json records D0 as '--unfolded_emb {full,zero}', yet the queue holds 48 D0_coil AND 47 D1_uemb jobs, i.e. both arms. The D0 evidence itself (r_coil=1.178, coil INCREASES variance; corr(E_u,wt_err) 0.419->0.507 under coil) argues against the coil arm, and 5 of the 9 completed trainings are coil cells. Either ~48 jobs and ~5 finished trainings are burning the cluster on an arm D0 rejected, or the recorded D0 decision is wrong.
- MINOR -- CEILING=0.711 is checked only as an upper H4 halt (pooled > CEILING+0.05). The measured offset-removal ORACLE band is 0.70-0.72, so a cell landing at 0.75 with a legitimately good slope halts as 'contradicting a recorded fact' -- defensible, but note H4 never fires downward, so a catastrophically collapsed pooled is never halted, only silently ranked.
- MINOR -- self_audit's only std_b check is 'std_b <= 0', a type sanity check, not a scientific one. It cannot distinguish std_b=0.223 (reference) from std_b=1.5 (broken).

## Queue and resource-use end-to-end audit (public gpu lane, golden rtx6000 lane, cpu lane); epoch-rate measurement; 48-cell factorial ETA; ordering; gol
**Verdict:** NEEDS-FIX

**DATA USED OPTIMALLY:** NO - two clear wins are being left on the table.

(1) PREEMPTION SAFETY IS OFF WHERE IT MATTERS. train.py has a fully-built, e2e-tested --resume (RESUME_MARKER_V1, scripts/e2e_resume.py) whose own help text says "Without it a requeued job silently retrains from epoch 0." The 3 preempted factorial jobs (21049394/96/98) were submitted WITHOUT it: `grep -c -- "--resume"` on their extracted batch scripts = 0. They had reached epochs 10.6/10.6/12.9 and hold 10/10/12 checkpoints on disk RIGHT NOW; on restart they begin at epoch 0 and overwrite those files. 4.6 GPU-h already discarded + ~24.8 GPU-h to redo. Note the newer golden jobs (21084179/180/595/596/597/598) DO pass --resume - so the fix is known and already applied elsewhere; only the public factorial submitter omits it. Highest-value action: ensure the 3 restarts (and the 31 unstarted trainings) carry --resume. Because the existing checkpoints are bare state_dicts, they will also need --resume_allow_partial_state, which is NOT numerically equivalent to an uninterrupted run - for a factorial cell used in a seed-averaged comparison, the honest choice is to let those 3 retrain clean and instead add --resume to the 31 unstarted cells so the NEXT preemption is free.

(2) EVALS DO NOT BELONG ON THE GPU AT ALL. The cpu partition has ~1,500 idle cores and a 7-day limit; scripts/run_cpu_evals.sh already scores cells there (job 21085470 running now). Yet 12 dependency-free ev_* jobs sit in the gpu queue holding gres/gpu:1 apiece. Scoring is a 28-protein forward pass. Moving every eval to the cpu lane frees GPU slots and removes the starvation class entirely.

Correctly used already: the metric rule is respected throughout (burial-on-dG, not ddG); a construction-time guard in hydro_net.py:808-830 refuses inserted feature blocks the graph_transformer cannot read, which directly blocks the signature failure mode; the holdout guard at train.py:375 refuses a no-op LORO.

**EVIDENCE:** QUEUE (squeue -u nissimb): 5 R + 89 PD on gpu, 19 PD on rtx6000, plus 2 R + 1 PD on cpu. Totals: 108 PD / 5 R GPU-side.
- Public pending reason = (QOSMaxGRESPerUser); `sacctmgr show qos gpu-part` -> MaxTRESPU=gres/gpu=5. Confirmed cap.
- Golden pending reason = (MaxGRESPerAccount); `sacctmgr show qos keasar` -> MaxTRESPA=...rtx_6000=8. `squeue -j 21067517` -> bo.hassonof birder_training_franca, R 8:19:53, gres/gpu:rtx_6000:8. All 8 held by one labmate job. Confirmed exactly as briefed.
- 26 of the 89 public PD are (Dependency) afterok on their trainings; the rest are QOS-capped.

RATE (tqdm parsed via `tr '\r' '\n' | grep -o 'Training Epoch: *[0-9]*' | tail -1`):
  21050063 2.00h @ ep 3.90 -> 0.514 h/ep
  21050059 4.15h @ ep 8.00 -> 0.519 h/ep
  21050057 4.49h @ ep 9.00 -> 0.499 h/ep
  21050055 4.76h @ ep 8.34 -> 0.571 h/ep
  21050053 5.74h @ ep 6.35 -> 0.904 h/ep
  mean 0.601 h/epoch -> 15-epoch training = 9.02 h. All 5 fit inside their 24h limit.

GAPS (integrated concurrency from sacct Start/End over 2026-09-06T21:31:31 -> now, 19.14h):
  capacity 95.72 slot-h, used 93.07, IDLE 2.65 slot-h = 2.8%.
  ALL of that idle is one 2.645h window where only 4 of 5 slots were filled. Handoffs are 1,7,6,2,4,5,1,8,1 SECONDS. Backfill is effectively perfect; there is no scheduling gap worth chasing.

PREEMPTION (the real waste, invisible to the occupancy math):
  `grep -l PREEMPTION logs/*.out` -> exactly 3 files. All: "CANCELLED AT 2026-09-06T21:31:23 DUE TO PREEMPTION" on ise-6000-03.
  `scontrol show job` -> Requeue=1 Restarts=1 on all three.
  `scontrol write batch_script 21049394` -> command line contains no --resume; `grep -c -- "--resume"` = 0 (also 0 for unstarted 21050065).
  Checkpoints still on disk: seed42 a0d0s0=10 (max ep 9), a0d0s1=10 (max ep 9), a0d1s0=12 (max ep 11).

CHECKPOINT CENSUS (Megascale-fineTuning/models/, NOT checkpoints/ which does not exist):
  9 factorial cells at max_epoch=14 (complete); 3 preempted short; 5 in progress; 31 never started.

ETA for the 48-cell factorial: 23.7 GPU-h to finish the 5 running + 34 x 9.02 = 306.5 GPU-h fresh = 330.2 GPU-h; on the hard 5-slot cap = 66.0 h wall clock = 2.8 days (assuming no further preemption; each preemption of a late-stage job adds up to ~9h since there is no resume).

SIGNATURE FAILURE MODE CAUGHT LIVE: logs/cpueval/p3_a1_d1_s1_D0_coil_seed42.log ends with
  "ModuleNotFoundError: No module named 'pandas'" (x2), then
  "DONE: ... scored on 28-test -> eval_results/abl_p3_a1_d1_s1_D0_coil_seed42_e13.csv"
  and `ls eval_results/abl_p3_a*.csv | wc -l` = 0. The success line is a lie; the 16:25 pass produced 11 logs and ZERO CSVs. The 16:38 relaunch (21085470) fixed PATH and is genuinely producing output (affine loaded, model loaded from kf_all_epoch_9.pt, validation progressing). Artifact verified, not the message.

G4 GATE (mandatory, run after all work; I changed nothing so this is a no-change confirmation):
  `python scripts/gate_g4_cpu.py` ->
  baseline               forward ok, dG finite  PASS dG=-0.0030 width=1092   <-- EXACT required value
  unfolded_emb=zero      PASS dG=0.0040 width=1092
  flory_unfolded         PASS dG=-0.0058 width=1092
  burial_features        PASS dG=0.0033 width=1095
  burial hse             PASS dG=0.0017 width=1095
  burial + uemb zero     PASS dG=0.0023 width=1095
  width checks PASS 1092 / 1095;  G4-CPU: ALL PASS
  Gate holds. No revert required.

READ-ONLY CONFIRMED: no sbatch, scancel, kill, or push issued. Only squeue/sacct/sacctmgr/sinfo/scontrol-show/grep/ls plus the read-only G4 gate.

**DEFECTS:**
- HIGH / ACTIVE DATA LOSS: the 3 preempted factorial jobs (21049394, 21049396, 21049398) were submitted without --resume, which train.py's own help text says causes a requeued job to 'silently retrain from epoch 0'. They currently hold 10, 10 and 12 real checkpoints (max epochs 9, 9, 11) which the restart will OVERWRITE. 4.6 GPU-h already lost, ~24.8 GPU-h to redo. The capability exists, is e2e-tested (scripts/e2e_resume.py), and is already used by the newer golden jobs 21084179/180/595/596/597/598 - only the public factorial submitter omits it. Any future preemption of a late-stage job costs up to a full 9h cell.
- HIGH / WRONG LANE: 12 dependency-free ev_* jobs occupy the 5-wide GPU QOS while scoring is a pure 28-protein forward pass that needs no GPU. The cpu partition has ~1,500 idle cores and a 7-day limit and is ALREADY running the equivalent work (21085470 via scripts/run_cpu_evals.sh). Keeping evals in the GPU pool is what created the 12-completed-cells-no-CSV situation in the first place.
- MEDIUM / PRIORITY INVERSION: within the public queue the long 24h trainings carry priority 6637 while the short 6h evals carry 6603, so evals sort BEHIND every training. Combined with the 5-slot cap this guarantees eval starvation for the life of the factorial. (Note: the 6637-vs-6603 split, not the FIFO job-id order, is the operative cause.)
- MEDIUM / STALE-ARTIFACT RISK: gld_severing (21080889) pins --ckpt .../p3_a0_d0_s0_D0_coil_seed42/kf_all_epoch_9.pt. That file belongs to preempted job 21049394 and will be DELETED/overwritten when 21049394 restarts from epoch 0. If the restart lands before gld_severing gets a GPU, severing either fails or silently scores a different, half-trained model. Ordering dependency nobody declared.
- LOW / MISLEADING SUCCESS LINE (pre-existing, known): run_calib_eval.sh prints its 'DONE ... -> abl_<tag>.csv' banner unconditionally, after a ModuleNotFoundError, with no CSV written. Reproduced today in logs/cpueval/*.log from the 16:25 pass (11 logs, 0 CSVs). The wrapper run_cpu_evals.sh does verify the artifact afterwards, but the inner script's banner remains a trap for anyone reading logs directly.

## Scoring pipeline (evaluate.py / train_utils.load_checkpoint / run_calib_eval.sh) for the 12 completed factorial cells
**Verdict:** NEEDS-FIX

**DATA USED OPTIMALLY:** Fixes 1-3 are real and verified; they were necessary but NOT sufficient. Left on the table: (a) the 12 completed cells are still unscored - 0 of 12 CSVs exist, and they cannot be scored from a login node at all (proved below), so the factorial is still un-analyzable; the only path is an sbatch to rtx6000, which I am forbidden to do, so this must go back to the manager/autopilot. (b) run_calib_eval.sh step 4 calls validation/diag_unfolded_offset.py, which DOES NOT EXIST in validation/ (only benchmark_metrics.py, score_runs.py, validation.py). The "Plot B offset diagnostic" - the diagnostic that is supposed to test whether lever 1 shrank std(b_p) - has therefore NEVER run, on any run, including the one successful eval on 2026-09-06. It is masked by `|| true` and the script still claims a plot was written. That is a second silent-success channel, and it is aimed squarely at the b_p/offset question. (c) METRIC RULE issue on lever 3: the affine sidecar applies pred := a*pred + b in validate() (evaluate.py ~line 425), but pred_ddG is computed as a within-protein difference (line 432), so b CANCELS EXACTLY in ddG and only a survives. The affine b is a dG/b_p-space object being carried through a ddG-scored pipeline; affine.json values differ wildly across cells (b=0.2512, 3.4032, 3.5165) yet none of that b can move any ddG metric. Any claim that "lever 3 / affine calibration changed ddG PCC" is measuring `a` alone. b must be scored on dG or b_p. (d) The bare `except:` at evaluate.py 493-494 and 542-543 swallows EVERY exception from load_checkpoint and silently falls back to a raw load_state_dict - so a genuinely corrupt checkpoint or a real KeyError would be masked rather than surfaced. It should be `except KeyError` or at minimum log the exception.

**EVIDENCE:** ALL commands run via cluster.py exec on slurm-login-02, cwd /home/nissimb/DeepPEF, env esm2_env_py38.

ITEM 1 - CONFIRMED. `grep -n -i device Megascale-fineTuning/evaluate.py`:
  63:DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
No hard-coded 'cuda', no commented-out fallback. Also verified CFG.device (model/model_cfg.py:5) is itself conditional. Verified the fallback actually fires: `python -c "import torch;print(torch.cuda.is_available())"` -> False on login node, and evaluate.py then ran on CPU (it got as far as loading the checkpoint and starting inference).

ITEM 3 - CONFIRMED, 9 of 9. `grep -n "torch.load" Megascale-fineTuning/evaluate.py`:
  192,193,194,195 and 247,248,249,250 -> all `weights_only=True, map_location=DEVICE` (8)
  296: embedding load -> `weights_only=True, map_location=DEVICE).to('cpu')` (9)
Exactly the 9 preprocessed-tensor loads claimed. NOTE the 2 remaining bare loads at 494 and 543 (`model.load_state_dict(torch.load(TRAINED_MODEL_PATH))`) have NO map_location - see defects.

ITEM 2 - CONFIRMED, BOTH FORMATS, EMPIRICALLY. train_utils.py:93-118 branches on `'model_state_dict' in model_dict`. I tested by round-trip, not by reading:
  BARE   (models/...kf_p3_slope1.0_s42/kf_all_epoch_14.pt, 80 keys, first key GAT_layers.0.gat1.att):
         -> OK epoch=-1 loss=nan first_w=[-0.054817624390125275, 0.1050737053155899, -0.21189697086811066]
  WRAPPER(synthesized {'model_state_dict':sd,'optimizer_state_dict':{},'epoch':7,'loss':0.5,'valid_loss':0.6}):
         -> OK epoch=7 loss=0.5 first_w=[-0.054817624390125275, 0.1050737053155899, -0.21189697086811066]
  Identical weights from both paths. No wrapper checkpoint exists in the tree any more, so I built one.
  Anti-no-op check (this is the "feature was never read" guard):
    weights CHANGED from random init: True
    weights EQUAL to checkpoint     : True
    strict load OK, missing= [] unexpected= []
  So the checkpoint genuinely reaches the model.

ITEM 4 - DEFECT CONFIRMED, STILL PRESENT, AND REPRODUCED LIVE 3x.
`grep -n "set -\|\-f \|\-s \|exit\|DONE" Megascale-fineTuning/run_calib_eval.sh`:
  7:set -uo pipefail            <- no -e
  39:if [ ! -f "${CKPT}" ]...   <- guards the INPUT checkpoint only
  62:echo "DONE: ... -> eval_results/abl_${RUN_TAG}_e${BEST_E}.csv ..."   <- UNCONDITIONAL
There is NO test of the output CSV anywhere in the script. I then reproduced it: I ran
`bash Megascale-fineTuning/run_calib_eval.sh <log> <model_dir> <tag>` for THREE completed cells
(p3_a1_d1_s1_D0_coil_seed42, p3_a0_d0_s0_D1_uemb_seed42, p3_a0_d1_s1_D0_coil_seed42). All three printed:
  "DONE: <tag> best epoch N scored on 28-test -> eval_results/abl_<tag>_eN.csv"
and `ls eval_results/abl_<tag>_eN.csv` -> "No such file or directory" for all three. The bug is live and unfixed.
Second layer: validation/score_runs.py main() does `rows = score(tag); if not rows: continue` - it exits 0
silently when the CSV is absent, so step 3 cannot fail the script either.

ITEM 5 - ACTUALLY SCORED. RESULT: NO CSV. NOT a code bug - a login-node execution-environment bug.
Confirmed the 12 tags first: `ls -d Megascale-fineTuning/models/*kf_p3_* | while read d; do [ -f "$d/kf_all_epoch_14.pt" ] && basename "$d"; done` -> exactly 12.
Best-epoch selection works (parses the val log correctly), e.g. for p3_a1_d1_s1_D0_coil_seed42 it printed the
full e0..e14 curve and "BEST val epoch = 13 (PCC=0.691)"; for p3_a0_d0_s0_D1_uemb_seed42 "BEST val epoch = 9 (PCC=0.809)".
Checkpoint loads fine ("Loaded model from .../kf_all_epoch_13.pt"), affine loads fine.
Then it dies at "Validation Epoch: 0: 0%| | 0/28" every time.
To get the real exit status I re-ran evaluate.py directly under `setsid nohup` (to rule out SSH-teardown
SIGHUP) writing `echo "EXITCODE=$?"` to a file:
  EXITCODE=143   (= 128+15 = SIGTERM), CSV=none, PROG=0%, after ~7 min wall, RSS stable at 1.36GB
Ruled out OOM explicitly: /sys/fs/cgroup/user.slice/user-<uid>.slice/memory.events shows `oom_kill 0`
(and `oom 0`), with memory.max = 8589934592 (8GB) and memory.current only ~3.5GB. So not memory.
143 = an external reaper killing long CPU jobs on the login node.
The contrast that proves it: the ONE eval that ever succeeded, logs/ev_p3_slope3.0_s42_21024110.out,
ran under SLURM - `sacct -j 21024110` -> Partition=rtx6000, State=COMPLETED, Elapsed=00:02:47, NodeList=ise-6000-09.
`nvidia-smi -L` on the login node -> "command not found" (no GPU). GPU does 28 proteins in ~2:47
(~5s/protein); on the login node CPU it did not finish protein 1 in 7 minutes before being SIGTERMed.
CONCLUSION: scoring these 12 cells on a login node is not possible; it needs an sbatch to rtx6000, which I am forbidden to do.

G4 GATE - PASS, EXACT. `python scripts/gate_g4_cpu.py`:
  baseline               forward ok, dG finite         PASS dG=-0.0030 width=1092
  unfolded_emb=zero      ... PASS dG=0.0040 width=1092
  flory_unfolded         ... PASS dG=-0.0058 width=1092
  burial_features        ... PASS dG=0.0033 width=1095
  burial hse             ... PASS dG=0.0017 width=1095
  burial + uemb zero     ... PASS dG=0.0023 width=1095
  width: levers that add no dims keep width  PASS 1092
  width: burial adds exactly 3               PASS 1095
  G4-CPU: ALL PASS
Required value dG=-0.0030 width=1092 reproduced exactly on the baseline row. Ran it twice (before and after
all my activity), identical both times. No revert needed.

STATE LEFT CLEAN: eval_results still holds exactly its original 10 CSVs (none of the 12 factorial tags among
them except the pre-existing abl_p3_slope3.0_s42_e8.csv); 0 stray evaluate.py processes; my temp files
(/home/nissimb/run_one.sh, exitcode.txt, exittest.log) removed. `git status --porcelain` shows only the
pre-existing ` M scripts/autopilot.py` and untracked dirs - I modified nothing, sbatch'd nothing, killed nothing.

**DEFECTS:**
- run_calib_eval.sh:62 - THE WORST BUG IS NOT FIXED. The final `echo "DONE: ... -> eval_results/abl_<tag>_e<N>.csv"` is unconditional. Line 7 is `set -uo pipefail` with NO `-e`, and the only existence test in the whole script (line 39) guards the INPUT checkpoint, never the OUTPUT CSV. Reproduced live 3x today: three completed cells each printed the DONE line naming a CSV that does not exist on disk. FIX: before the DONE echo, add `CSV="eval_results/abl_${RUN_TAG}_e${BEST_E}.csv"; if [ ! -s "$CSV" ]; then echo "FAIL: $CSV missing or empty"; exit 1; fi`.
- validation/diag_unfolded_offset.py DOES NOT EXIST (validation/ contains only benchmark_metrics.py, score_runs.py, validation.py, __pycache__). run_calib_eval.sh steps at lines ~55 and ~59 invoke it twice; both fail with "can't open file", both are swallowed by `|| true`, and line 62 still advertises `plot -> eval_results/figures/plotB_<tag>.png`. The Plot B std(b_p) offset diagnostic has therefore NEVER produced output on ANY run, including the one successful GPU eval of 2026-09-06. This is a second silent-success channel and it disables exactly the b_p/offset diagnostic that lever 1 is supposed to be judged by.
- validation/score_runs.py main(): `rows = score(tag); if not rows: continue` - silently skips and exits 0 when no CSV matches the tag. Step 3 of run_calib_eval.sh therefore cannot signal the missing artifact either. It should print a warning and return non-zero when an explicitly-requested tag has no CSV.
- evaluate.py:493-494 and 542-543 - bare `except:` around `load_checkpoint(...)` falling back to `model.load_state_dict(torch.load(TRAINED_MODEL_PATH))`. This catches EVERYTHING (including a genuine KeyError, a corrupt file, or an OOM) and silently substitutes a different load path, so a real failure looks like a clean load. Note this fallback is what made the 2026-09-06 eval appear to work. Narrow it to `except KeyError` and log the exception.
- evaluate.py:494 and 543 - the two `torch.load(TRAINED_MODEL_PATH)` calls in that except-fallback have NO map_location, unlike the 9 that were fixed. A CUDA-serialized checkpoint hitting this fallback on a CPU-only node raises RuntimeError. Add map_location=DEVICE to both.
- METRIC RULE violation in the affine sidecar (lever 3). evaluate.py ~line 425 applies pred_deltaG := a*pred + b, but line ~432 computes pred_ddG as a within-protein difference from the first row, so the additive b CANCELS EXACTLY and cannot move any ddG metric. affine.json b values vary widely across cells (0.2512, 3.4032, 3.5165) while only `a` can affect ddG PCC/RMSE. b is a dG/b_p-space quantity and MUST be scored on dG or b_p, never on ddG.
- ENVIRONMENT BLOCKER for item 5: the 12 completed cells cannot be scored on a login node. Login nodes have no GPU (`nvidia-smi` not found) and kill long CPU jobs with SIGTERM (measured EXITCODE=143 at ~7 min, with cgroup oom_kill=0 ruling out memory). The only eval that ever succeeded ran via SLURM on partition rtx6000 (job 21024110, node ise-6000-09, COMPLETED 00:02:47). Scoring these 12 requires an sbatch to a GPU partition, which is outside my permissions - this needs to go back to the manager/autopilot.

## Adversarial verification of the four scientific claims (IDENTIFIABILITY / SLOPE_ORIGIN / OFFSET_CORRECTOR / exposure-burial replication) + n=28 power 
**Verdict:** NEEDS-FIX

**DATA USED OPTIMALLY:** The 5-seed abl_sigma replicate set is genuinely the right experiment and is used well: identical 28 proteins / 28,314 mutations / identical deltaG multiset across all 5 (I verified the multiset equality myself, not just the row count). Three things are left on the table. (1) The epoch confound is acknowledged but only partly exploited: seeds 1 and 3 are BOTH e13, so a clean epoch-matched ICC exists (I computed it: 0.9722 for b_p) — that is the honest identifiability number, and it is stronger than the headline, so the doc is under-claiming on its best-controlled estimate while over-claiming elsewhere. corr(seed mean, epoch) = -0.81 for b_p confirms epoch drift is real and large. (2) The whole a_p/b_p literature here rests on 10 eval CSVs that are NOT 10 independent experiments — they are 5 sigma seeds plus 5 different ablation configs (anchor_w0.3/1.0/3.0, p3_slope3.0, calib_ctrl). Calling "10/10 sign-consistent" a replication overstates independence; the anchor arm shares seed 42 and differs only in a loss weight. An effective-n closer to 5 is defensible, not 10. (3) The Tsuboyama per-mutation sigma exists in 5 files on disk but joins 0/28 onto the test proteins — verified genuinely disjoint via a positive control (1ENH/1VII found by aa_seq). That is a real dead end, correctly reported, and it is the single thing that would most sharpen the attenuation question if a join could be found.

**EVIDENCE:** All work CPU-only via `python cluster.py exec`, no sbatch/scancel, no repo files modified (only read-only vfy*.py added). Every number below is from MY OWN from-scratch reimplementation (own pearson/fit/ICC), never by rerunning their scripts.

G4 GATE (mandatory, run last, read from ARTIFACT not exit code):
  nohup python scripts/gate_g4_cpu.py > /home/nissimb/g4_verify.log
  -> "baseline  forward ok, dG finite  PASS dG=-0.0030 width=1092" ... "G4-CPU: ALL PASS"
  UNMOVED. Note: two earlier foreground runs returned exit 0 with EMPTY output — the exact
  signature failure mode. Only the logfile artifact confirmed the gate.

CLAIM 1 — ICC. REPRODUCES EXACTLY.
  ICC(1,1) b_p = 0.8978 (MSB=9.7488 MSW=0.2170); a_p = 0.8450; b_ddg = 0.8780.
  Integrity: all 5 CSVs 28 proteins / 28,314 rows / identical deltaG multiset = True.
  Epoch confound REAL: per-seed b_p means -0.175,-0.183,-0.205,-0.396,+0.515;
  corr(seed mean, epoch) = -0.8053. Epoch-matched (seeds 1,3 both e13, k=2) ICC = 0.9722;
  seed-main-effect-removed (consistency) ICC = 0.9532. So 0.898 IS a lower bound, as claimed.
  Ceiling arithmetic reproduces (raw 0.6052, oracle 0.7181, -> 0.705) BUT uses ICC(b_ddg)=0.878,
  not the headline 0.898 — the headline ICC is not the number driving the ceiling.

CLAIM 2 — SLOPE_ORIGIN. SIMULATION REPLICATES; INFERENCE IS WEAKER THAN STATED.
  My independent rerun: mean r = -0.3873, max -0.3271, P(r >= +0.7139) = 0/2000 (theirs: -0.3871, 0/2000).
  BUT I showed the sim is near-deterministic arithmetic, not an independent test: with ONE global
  sigma, a_p = var/(var+sigma^2) is a monotone function of var(ddG_true) alone, so sign(sim r) is
  FORCED to equal sign(r(var_true,SASA)) = -0.3712. My closed-form no-noise calculation gives
  r = -0.3896, essentially the entire "simulated" result. The 2000 trials add ~0.018 sd of jitter.
  P=0/2000 therefore measures the tightness of that jitter, NOT the improbability of attenuation.
  Fair version (per-protein sigma allowed to vary): required sigma vs SASA r = -0.5038, n=26,
  median required sigma 1.1098 (range 0.33-3.56) — 1.1 kcal/mol is a strained but not absurd assay noise.

CLAIM 3 — OFFSET_CORRECTOR. NUMBERS REPRODUCE; ONE REAL LABELLING DEFECT.
  raw/oracle per CSV reproduce to 4dp (e.g. calib_ctrl 0.5910 -> 0.7113; sigma_seed1 0.5948 -> 0.7005).
  Shift-invariance CONFIRMED EXACTLY: subtracting -1.0/0.0/0.5/1.0/12345.0 all give pooled
  PCC = 0.590991 identically. So the null IS the raw number. Doc states this correctly.
  78% top-2 share: reproduces ONLY when top-2 is ranked by |b_ddg| -> mean 78.0%, range 60-88%
  (matches doc's "78%, range 60-88%" exactly). Ranked by the CANONICAL |b_p| (dG WT error) it is
  mean 35.8%, range -6% to +74% — and NEGATIVE on 2 CSVs.
  Outliers: NEITHER is a structural outlier. 2K5H max|z| = 1.73 (mean_hydropathy_KD), 2KVS max|z|
  = 1.92 (frac_buried); 0/16 features exceed |z|>2 for either. Claim CONFIRMED (doc says 2.03/1.88;
  I get 1.73/1.92 — same conclusion, minor numeric drift from feature-set choice).

CLAIM 4 — REPLICATIONS. REPRODUCE EXACTLY (10 CSVs each, own recomputation).
  mean_rel_SASA vs a_p: mean r = +0.6239, sign-consistent 10/10, range +0.3997..+0.7139.
  frac_buried vs b_p(WT error): mean r = -0.4746, sign-consistent 10/10, range -0.3055..-0.5615.

CLAIM 5 — n=28 POWER. Uncorrected |r|>0.374: exposure 10/10, burial 8/10.
  Bonferroni (alpha=0.05/90 = 5.6e-4, df=26): exposure passes 7/10; burial passes 0/10.

HEADLINE FACTS (primary CSV, my recomputation): std(b_p)=1.5741 EXACT; a_p median 0.4975
  (briefed 0.4990 — that value comes from a slightly different fit path), min 0.0789 max 1.2566 EXACT;
  per-protein PCC median 0.7955; pooled 0.5910; corr(a_p,PCC)=+0.5714 EXACT.
  std(b_ddg)=0.2232, NOT the briefed 0.2273. corr(b_ddg, b_wt) = -0.7060 (OPPOSITE sign).
METRIC RULE AUDIT: injecting a per-protein constant of +/-2.0 leaves median a_p at 0.4990 unchanged
  while std(b_p) moves 1.5741 -> 2.8343. Rule HOLDS exactly as stated.
ZERO-VARIANCE: n_chains distinct={1}, n_het={0}, n_metal={0}, interchain_BSA={0.0} across all 28. EXACT.

**DEFECTS:**
- OFFSET_CORRECTOR.md: the 78% headline is computed on |b_ddg| (ddG intercept) but the supporting table is captioned 'true b_p' and the doc's own Section-2-style distinction insists b_p is the dG-space WT error (std 1.5741). On the CANONICAL b_p the top-2 share is 35.8% (range -6% to +74%), not 78%. The two objects have corr = -0.706 and 2KVS/2K5H flip sign between them (b_ddg +0.7680/-0.7199 vs b_wt -4.7043/+2.5215), so this is not cosmetic. The quoted values -0.7199/+0.7680 are b_ddg. FIX: relabel the column 'b_ddg (ddG-space intercept)' and state that the 78% is a ddG-intercept statistic. The corrector's own code (offset_corrector.py L5, L252) is internally consistent in using the ddG intercept — the defect is in the prose/table labelling, not the analysis.
- SLOPE_ORIGIN.md: 'P(r >= +0.714) = 0/2000' is presented as strong independent evidence but the simulation is near-deterministic by construction. A single global sigma makes a_p a monotone function of var(ddG_true) alone, so the negative sign is forced by the pre-existing r(var_true, SASA) = -0.3712. My closed-form zero-noise calculation reproduces essentially the whole result (-0.3896 vs sim mean -0.3873). The p-value quantifies simulation jitter, not the odds against attenuation. The verdict (attenuation predicts the wrong sign) still stands on the SIGN argument in 3a/3b, which is sound and independent — but the '0/2000' should be retired or reported as 'deterministic consequence', and the doc should concede it only refutes STRUCTURE-INDEPENDENT (global-sigma) attenuation. Heteroscedastic attenuation with per-protein sigma is NOT refuted: required sigma vs SASA is r = -0.5038 with median 1.11 kcal/mol.
- burial -> b_p (mean r = -0.475) passes Bonferroni on 0/10 checkpoints (needs |r| ~ 0.60 at alpha = 5.6e-4, df = 26; observed max |r| = 0.5615). Under the project's own stated honest bar this is NOT established. It should be reported as suggestive/uncorrected-only. The exposure -> a_p result is materially stronger (7/10 pass Bonferroni) and the two should not be presented as equally solid, as the '10/10 sign-consistent' framing implies.
- '10/10 checkpoints' overstates independence. The 10 eval CSVs are 5 sigma seeds + 5 ablation configs (anchor_w0.3/w1.0/w3.0 all seed 42, p3_slope3.0, calib_ctrl), not 10 independent replicates. Sign-consistency across correlated configs of the same model is much weaker evidence than 10/10 suggests; effective n is nearer 5.
- IDENTIFIABILITY.md headline is ICC(b_p) = 0.898, but the 0.705 ceiling is computed from ICC(b_ddg) = 0.878 (icc3.py bound(icc_bd)). The headline number is not the one carrying the conclusion; the doc should say which ICC drives the ceiling.
- Briefed fact 'ddG-space intercept has std 0.2273' does not reproduce on the primary CSV: I measure 0.2232. Minor, but it is a stated MEASURED FACT and should be re-sourced to whichever CSV produced it.
- Cannot verify 'reaches_model': every claim here is a post-hoc analysis of eval CSVs on disk. None of a_p/b_p/exposure/burial is a feature read by the model at training time, so the READY criterion 'actually read by the model at training time' is not applicable-and-satisfied — it is untested. Nothing in this verification licenses a training-time lever.

## W6 descriptors + W10 open alphabet + Mordred matrix (end-to-end verification)
**Verdict:** READY

**DATA USED OPTIMALLY:** The descriptor block is a WITHIN-protein, per-residue quantity that differs between WT and mutant, so it does NOT cancel in ddG -- it is correctly ddG/a_p-measurable and the METRIC RULE does not demote it to a dG-only lever. That is the right scoring target.

What the data says about VALUE (not correctness): the residue-disjoint probe puts the descriptor block at mean R2=+0.18 / median +0.28 from ProtT5, i.e. only partially redundant -- 7 of 19 residue types are NEGATIVE (K -0.334, H -0.265, M -0.255, T -0.126, R -0.045, Q -0.011, plus P ~0.000). ProtT5 linearly recovers the polar/charged and aromatic axes (D +0.69, Y +0.63, E +0.46, A +0.43, I +0.43, V +0.40) but NOT the basic residues or sulphur-containing M. So the block carries real non-redundant signal concentrated exactly where ProtT5 is weakest. This is a genuine finding either way, as the brief anticipated.

LEFT ON THE TABLE (three things, all honest limits rather than defects):
1. n=19 residue types, not 20 -- CYSTEINE IS ENTIRELY ABSENT (count 0) from the 19 usable protein_tensors. C cannot be held out, and C is chemically distinctive (the only sulphur/disulphide residue). The 20-fold claim is really 19-fold.
2. Only 1056 residues from 19 proteins were used (the loop caps at 40 and only 19 dirs carry both prott5_embeddings and one_hot_encodings). Rare residues are thin -- M n=12, H n=20, W n=23 -- and the three worst-fitting types are among the rarest, so some negative R2 is small-sample noise, not proven irreducibility. Pooling the full 28-protein union via tensor_paths.py ROOTS (which exists in the scratchpad precisely because protein_tensors alone is incomplete) would tighten this materially and could recover C.
3. Probe is LINEAR ridge (lam=1.0, unswept). "Not linearly predictable" is a weaker claim than "not predictable" -- the network is nonlinear, so a linear probe UNDERSTATES redundancy. The R2=+0.18 is therefore an upper bound on the block's unique contribution, and should be reported as such rather than as proof of independent information.
4. Not yet measured: whether the 726 columns beat their own 16-dim PCA projection (mordred_pca16 exists and is loadable), which is the cheap test of whether K=726 is buying anything over K=16 at 1/45 the width.

**EVIDENCE:** ALL commands run on BGU cluster via `python cluster.py exec` in /home/nissimb/DeepPEF, conda env esm2_env_py38. (Note: the `source ~/miniconda3/...` line errors "No such file"; the fallback path activates and python works -- harmless, but the heredoc's first conda line is dead on this node.)

--- ITEM 1: PROVENANCE + SHAPE ---
$ ls -la data/aa_descriptors.csv ; md5sum
  -rw-r--r-- 199228 bytes, Sep 7 10:17
  c0f35dfb30a49798a205b9f233d506f5
$ grep '^#' data/aa_descriptors.csv   -> 25 provenance lines, including:
  # DeepEF amino-acid physicochemical descriptor matrix -- REAL Mordred pipeline
  # generated_utc=2026-09-07T07:17:48Z
  # rdkit=2025.09.2 / mordred=1.2.0 / python=3.9.25
  # smiles_source=PubChem canonical isomeric SMILES, neutral free L-amino acids
  # smiles_gate=rdkit molecular formula + stereocentre count; ENFORCED, 20/20 pass
  # raw_descriptors=1613 -> filter1 dropna 1613->1278 -> filter2 nunique>=15 1278->726
  # final_shape=20 x 726
  NO "synthetic replica for verification" string present anywhere.
$ python: shape (20, 726); index ACDEFGHIKLMNPQRSTVWY; dtypes {float64}; any NaN: False
  col mean absmax 1.350e-10, col std range 1.0000..1.0000 (properly z-scored)
  n unique cols by content: 714 (12 duplicate-valued columns, benign)

--- ITEM 2: CHEMICAL VALIDITY (euclidean NN in z-space, computed by me) ---
  F: Y(14.3)  H(22.3)  W(25.6)  K(26.3)      <- top-3 all aromatic
  Y: F(14.3)  W(22.7)  H(23.2)
  W: Y(22.7)  F(25.6)  H(34.5)
  D: N(13.6)  E(21.7)  Q(25.3)
  E: Q(12.4)  D(21.7)  N(22.7)
  N: D(13.6)  Q(19.3)  E(22.7)
  Q: E(12.4)  N(19.3)  H(22.6)                <- D/E/N/Q closed cluster
  I: V(17.2)  L(21.3)  ; L: V(20.9) I(21.3) ; V: I(17.2) L(20.9)
  S: T(18.9) ; T: S(18.9)                     <- mutual rank-1
  G: A(22.1)  S(32.8)
  Targeted ranks: F->Y r1, F->H r2, F->W r3, D->E r2, N->Q r2, I->V r1, I->L r2,
                  V->L r2, S->T r1, K->R r3, L->M r5
  Outlier ranking (mean dist to others): G 51.3 (HIGHEST), W 50.8, A 41.6 ... Q 30.7
  VERDICT: chemistry is correct. Two honest caveats: P is NOT a distance outlier
  (rank 9/20, mean 37.1) -- its oddity shows only as having no strong partner
  (V/I/T/L), because ignore_3D=True cannot see backbone rigidity; and L->M is rank 5,
  weaker than expected. Neither is a defect in the matrix.

--- ITEM 3: HISTIDINE (recovered from COMMITTED matrix, not the header) ---
  nN/nHetero/nBase were removed by filter2; MW survived. Fitted MW = alpha*z + beta on
  the 19 residues whose true MW I supplied independently:
  fit MW = 30.0817*z + 136.9003 ; max residual on 19 knowns = 0.0674
  RECOVERED MW of H = 155.166
    L-histidine    C6H9N3O2  = 155.155 -> delta +0.011   <-- MATCH
    wrong plan-doc C7H10N2O2 = 154.167 -> delta +0.999   (90x the fit error)
  VERDICT: CORRECT imidazole. Header line 12 independently documents the correction.

--- ITEM 4: W10 REFUSAL ON THE REAL FILE ---
  data/aa_descriptors_open25.csv exists, 257967 bytes, header says final_shape=25 x 756,
  filter2 nunique>=15 of 25 (0.600) -- a SEPARATE filter pass, hence a different column set.
$ python: real file abspath /home/nissimb/DeepPEF/data/aa_descriptors_open25.csv 257967 bytes
  A.load_descriptor_table('data/aa_descriptors_open25.csv')  -> raised through the NORMAL
  entry point (traceback: load_descriptor_table:457 -> _load_table_obj:368 ->
  _validate_open_alphabet:419), NOT a hand-called helper, NOT a /tmp fixture:
  ValueError: W10: open-alphabet table /home/nissimb/DeepPEF/data/aa_descriptors_open25.csv
  has K=756 columns but the canonical table .../aa_descriptors_mordred.csv has K=726, and
  the column NAMES differ (27 canonical columns dropped, 57 new columns added; e.g.
  dropped ATSC0c,ATSC1c,ATSC2c / added ATS5dv,ATS6d,ATS7s).
  Exactly the stated 756/726, 27 dropped, 57 added. Guard also checks rows 0..19 are
  byte-identical to canonical (second raise, not reached here).

--- ITEM 5: RESIDUE-DISJOINT PROBE (leave-one-residue-type-out, ridge lam=1.0) ---
  Data: 19 proteins, 1056 residues, emb_dim=1024, K=726, real prott5_embeddings from
  /groups/keasar_group/casp15/meytav/protein_tensors. Baseline = mean descriptor of the
  19 TRAINING residue types (held-out row never seen).
  counts: A79 C0 D52 E115 F33 G74 H20 I64 K91 L97 M12 N55 P33 Q36 R69 S47 T42 V83 W23 Y31
  D +0.6905  Y +0.6322  E +0.4580  A +0.4349  I +0.4322  V +0.4047  G +0.3632
  S +0.3036  W +0.2943  F +0.2848  N +0.1244  L +0.0446  P +0.0003  Q -0.0108
  R -0.0447  T -0.1260  M -0.2548  H -0.2654  K -0.3341
  MEAN R2 = +0.1806   MEDIAN R2 = +0.2848   positive 13/20 (of 19 types present)
  CONTAMINATED row-random split for contrast: R2 = +0.9170  <-- exactly the trap the
  brief flagged; it would have falsely "proven" the block redundant.

--- REACHES MODEL ---
  MODES = ('none','mordred726','mordred_pca16','mordred_pca16_only')
  descriptor_dim: none=0 (width 1092), mordred726=726 (width 1818), mordred_pca16=16 (1108)
  residue_descriptors(one_hot F/H/G, mode='mordred726') returns rows matching the CSV
  exactly: F/H/G all allclose=True, maxabsdiff 1.16e-07 (float32 rounding).
  So the block is genuinely gathered into the feature vector, not merely parsed.

--- G4 GATE (after all work; I changed nothing in the repo) ---
$ python scripts/gate_g4_cpu.py
  baseline  forward ok, dG finite  PASS dG=-0.0030 width=1092   <-- REQUIRED VALUE HELD
  ... burial adds exactly 3 PASS 1095 ... G4-CPU: ALL PASS
  git status: only PRE-EXISTING modifications (scripts/block_ablation.py, _noop_work/,
  *.bak) from earlier work. My only write was /home/nissimb/w6probe.py, outside the repo.
  No sbatch/scancel/kill/push issued. Password never written to disk.

**DEFECTS:**
- NOT A DEFECT IN THE ARTIFACT, BUT A LIMIT ON ITEM 5: cysteine is completely absent (count 0) from the 1056 residues, so the 'leave-one-residue-type-out' probe covers 19 types, not 20. C is the single most chemically distinctive residue (sulphur/disulphide). Any writeup must say n=19 types; '13/20 positive' is really 13 of 19 present.
- Item 5 sample is thin and unnecessarily so: only 19 proteins / 1056 residues, because the loop reads only /groups/keasar_group/casp15/meytav/protein_tensors and requires both prott5_embeddings/ and one_hot_encodings.pt. The scratchpad's own tensor_paths.py documents that this root holds only 20 of 28 test proteins and that the MutationFineTuning/test_protein_tensors root is needed for full coverage. The three most negative R2 residues (K n=91 is fine, but H n=20, M n=12) include the two rarest types, so part of the negative signal is small-sample noise rather than proven non-redundancy.
- Item 5 probe is LINEAR (ridge, lam=1.0, never swept). It therefore gives an UPPER bound on the block's unique contribution -- a nonlinear network could extract more of the descriptor block from ProtT5 than a linear map can. Reporting mean R2=+0.18 as evidence the block is non-redundant overstates it; the defensible claim is 'not LINEARLY recoverable'. lam was also not tuned, so the R2 values are not the best achievable linear fit.
- Chemistry caveat (minor, expected from ignore_3D=True): proline is NOT a distance outlier as the brief predicted (rank 9/20 by mean distance, 37.1 vs G 51.3). Its distinctiveness appears only as the absence of a close partner. Mordred with ignore_3D cannot represent backbone ring constraint, so the matrix under-represents P's conformational effect. Also L->M is only rank 5. Neither invalidates the matrix but both should be stated rather than claiming all predicted groupings held.
- Housekeeping (no impact on results): the standard heredoc's first line `source ~/miniconda3/etc/profile.d/conda.sh` fails with 'No such file or directory' on this login node every time; only the /opt/conda fallback works. Harmless because of the `||`, but the documented CLUSTER snippet has a dead first branch on this node.
- 12 of the 726 columns are exact duplicates by content (714 unique). Harmless for a linear layer but means the effective descriptor rank is slightly below 726.

## ?

====================================================================================================
# WORKFLOW: FRONTIER  (6 results)
====================================================================================================

## LINE 2 — recover the 9 missing test proteins
**Verdict:** PROMISING

**HEADLINE:** All 9 recovered — 28/28 now resolvable — but W5 got WEAKER at full n: mean_burial~b_p went -0.343 (n=19) -> -0.265 (n=28, p=0.17), never crossing the 0.392 threshold. The bigger finding: the missing third was NOT random — it was significantly biased toward high-a_p (0.668 vs 0.451, p=0.029), well-fit (PCC 0.813 vs 0.692, p=0.037), positively-offset (b_p +1.14 vs -0.02, p=0.052) proteins.

**NUMBERS:** RECOVERY: protein_tensors has 20 dirs but only 19 with coords_tensor.pt. MutationFineTuning/test_protein_tensors (75 dirs) has 27/28 incl. all 9 unusable ones, under IDENTICAL names. Union of both roots = 28/28 (r18_3_TrROS_Hall only in the old root). Provenance verified: 18 proteins carry coords in BOTH roots, 18/18 BIT-IDENTICAL (torch.equal), so the union is one preprocessing, no batch effect. Recovered tensors well-formed: (L,4,3), full masks, L=44-71, ranges +/-13-23 Angstrom.

W5-ON-dG at n=28 (threshold |r|=0.392): mean_burial~b_p -0.265 (p=0.174); mean_burial~abs_b_p +0.233 (p=0.233); frac_buried~b_p -0.060 (p=0.760); frac_buried~abs_b_p -0.090 (p=0.647); length~b_p -0.303 (p=0.117); length~abs_b_p +0.264 (p=0.175). NOTHING crosses 0.392. Control: n=19 subset reproduces -0.343 EXACTLY, so the resolver reproduces the historical computation and only the sample changed.

BIAS TEST (Mann-Whitney, present n=19 vs missing n=9): b_p -0.019 vs +1.143 p=0.052; abs_b_p 1.102 vs 1.587 p=0.081; a_p 0.451 vs 0.668 p=0.029; per-protein PCC 0.692 vs 0.813 p=0.037; length 55.3 vs 56.3 p=0.806; mean_burial 0.449 vs 0.440 p=0.658; designed 5/19 (26%) vs 2/9 (22%). Model-behaviour axes differ; structural axes are all null.

TOP-6 |b_p|: 2KVS 4.704 (present), r12_757_TrROS_Hall 2.747 (WAS MISSING), 2K5H 2.522 (WAS MISSING), r18_3_TrROS_Hall 2.500 (present), 1TUC 2.022 (present), 2KXD 1.935 (WAS MISSING). Three of the top six were invisible, including #2.

**METHOD:** 1) Enumerated the true 28 from eval_results/abl_calib_ctrl_repro2_e14.csv and tested dir-presence vs coords-presence separately in protein_tensors. 2) Traced the pipeline: model/data_loader.py reads crd_backbone.pt/mask.pt from CFG.data_path+set_type; the string "protein_tensors" occurs in exactly ONE repo file (scripts/w5_dg.py), so it was never train.py's resolution path. 3) find over /groups/keasar_group depth-6 for the 8 absent names -> MutationFineTuning/test_protein_tensors. 4) Provenance: torch.equal on all 18 both-root overlaps. 5) Wrote scripts/tensor_paths.py (union resolver keyed on the COORDS FILE not the directory; raises FileNotFoundError so n can never silently shrink). 6) scripts/w5_dg28.py re-ran W5 at n=28 with an exact-Student-t p-value and an embedded n=19-subset control. 7) scripts/missbias.py: Mann-Whitney (ties-corrected normal approx) on b_p, |b_p|, a_p, per-protein PCC, length, burial, designed-fraction. 8) Verified every artifact by ls -l + reading the JSON back, not by trusting success lines. 9) Gate: run in background to home dir (the first /tmp attempt vanished — different node's /tmp, a live instance of the signature failure mode).

**NEXT ACTION:** Re-run the two load-bearing structural correlations at n=28 through scripts/tensor_paths.py, because BOTH were estimated on the now-demonstrated biased subsample and both underpin the "one structural axis, both calibration channels" claim: (a) mean_rel_SASA ~ a_p (r=+0.714 at n=19) — this is the one most at risk, since the missing 9 are exactly the high-a_p proteins that were absent from the fit; (b) frac_buried ~ b_p (mean r=-0.475 over 10 checkpoints). Both are pure CPU re-analyses over existing eval CSVs, no training. Then re-derive the b_p-corrector post-mortem at n=28: its conclusion ("neither top-2 protein is a structural outlier") was reached without structural access to r12_757_TrROS_Hall, the actual #2 carrier.

**CAVEATS:** TWO PREMISES IN THE BRIEF WERE WRONG AND ARE CORRECTED. (1) 2K5H was NOT absent — it HAS a directory in protein_tensors; that directory contains only prott5_embeddings and no coords_tensor.pt. Any isdir-based check calls it present. This is why the "9 missing" list circulated wrong: the real split is 8 absent dirs + 1 coords-less dir. (2) On this checkpoint 2K5H is the #3 |b_p| carrier, not one of the top-2; the top-2 are 2KVS (4.704) and r12_757_TrROS_Hall (2.747). The "78% of oracle gain in the top-2" framing should be re-checked at n=28.

THE W5 RESULT IS A CLEAN NEGATIVE, NOT A WIN. Recovering the sample moved mean_burial~b_p TOWARD zero. The n=19 value was not an underpowered true effect. W5 as a b_p lever now has no supporting measurement at full n on the metric it acts on.

NOT IN TENSION with the known frac_buried~b_p mean r=-0.475: that is a 10-checkpoint average of a different statistic; frac_buried on this single checkpoint is -0.060. The 10-checkpoint replication is still the stronger evidence and should be redone at n=28 before anything is concluded.

STATISTICS. All numbers come from ONE checkpoint (abl_calib_ctrl_repro2_e14.csv). The bias tests are n=19 vs n=9 — underpowered; p=0.029/0.037 are nominal and uncorrected across six quantities (Bonferroni alpha=0.0083 retains none). The defensible claim is the PATTERN — three model-behaviour axes differ, three structural axes are null — not any single p-value. b_p at p=0.052 does not clear 0.05.

UNRESOLVED. Why the two roots diverge is not established; transfer_files.py / transfer_training_proteins.sh in that directory are where to look. I did not modify any existing script — tensor_paths.py, w5_dg28.py and missbias.py are all new files, and the original scripts/w5_dg.py is untouched, so nothing needed reverting. Gate re-run after all changes: G4-CPU ALL PASS, baseline dG=-0.0030 width=1092. Generalisation beyond these 28 single-chain 43-72aa monomers is not testable here.

**GPU HOURS:** 0 — all work was CPU-only re-analysis over existing eval CSVs and stored tensors. The recommended next actions are also 0 GPU-hours.

**FILES:**
- /home/nissimb/DeepPEF/results/MISSING_PROTEINS.md
- /home/nissimb/DeepPEF/scripts/tensor_paths.py
- /home/nissimb/DeepPEF/scripts/w5_dg28.py
- /home/nissimb/DeepPEF/scripts/missbias.py
- /home/nissimb/DeepPEF/results/w5_on_dg_n28.json
- /home/nissimb/DeepPEF/results/missing_bias.json
- C:\Users\User\AppData\Local\Temp\claude\C--Users-User\eee8e8e1-58ee-4bbf-8e95-538e7a9fb0ba\scratchpad\MISSING_PROTEINS.md

## LINE 6
**Verdict:** NEGATIVE

**HEADLINE:** The headline is NOT length: a_p ~ mean_rel_SASA controlling for length = +0.6623 mean, significant in 10/10 checkpoints, while length alone is r=-0.2149 and significant in 0/10. CHECKPOINT 8 stands, no correction required.

**NUMBERS:** CONFOUND IS REAL: corr(length, mean_rel_SASA) = -0.6662 (p=1.1e-4), spearman -0.6280. Length range 43-72aa. n=28, |r|<0.374 at p=0.05 (partials df=n-3, |r|<0.381).

RAW vs a_p over 10 checkpoints (mean / sign-consistency / n sig):
  mean_rel_SASA +0.6239, 10/10, sig 10/10 (range +0.400..+0.714)
  length        -0.2149, 10/10, sig 0/10  (range -0.286..-0.082)
  n_mut         -0.1960, 10/10, sig 0/10
  sd(true ddG)  -0.2939, 10/10, sig 0/10

PARTIALS:
  a_p ~ SASA | length = +0.6623 mean (range +0.408..+0.740), sig 10/10 -- RISES vs raw
  a_p ~ length | SASA = +0.3533 mean, sig 5/10; SIGN FLIPS from raw -0.215 to +0.353 (suppressor signature)
  a_p ~ SASA | n_mut  = +0.6882, sig 10/10
  a_p ~ SASA | sd_ddG = +0.5796, sig 9/10

MULTIVARIATE OLS (standardised, a_p averaged over 10 ckpts): SASA beta=+0.9123 t=+4.860 p=5.4e-05; length beta=+0.3904 t=+2.080 p=0.048; R2=0.5101; VIF both 1.80.
SEMIPARTIAL unique variance: SASA sr=+0.6803 (46.3%), length sr=+0.2911 (8.5%).
BOOTSTRAP (10k) sr2(SASA)-sr2(length): point 0.378, 95% CI [0.030, 0.634], P(SASA wins)=0.984.
LOPO (28 folds): partial SASA|len range +0.6456..+0.7921, p<0.05 in 28/28 (worst p=3.7e-4). partial len|SASA range +0.268..+0.460, p<0.05 in 9/28.

WITHIN-LENGTH-STRATUM (length has no variance inside band):
  short 43-56aa n=15: mean r(SASA,a_p)=+0.6766 over 10 ckpts, 10/10 sign-consistent (r_crit 0.514); r(length,a_p)=+0.037
  long 57-72aa n=13: mean r(SASA,a_p)=+0.4411, 10/10 sign-consistent (r_crit 0.553); r(length,a_p)=-0.154

DESIGNED-vs-NATURAL: mean a_p designed 0.3420 (n=7) vs natural 0.4713 (n=21), Welch p=0.101 (not a class effect); mean SASA 0.3892 vs 0.3868. partial SASA|(length,designed) = +0.5876 p=0.0016 (df=24); partial length|(SASA,designed) = +0.1579 p=0.441.

NATURAL-ONLY n=21 CAVEAT: r(SASA,length)=-0.884 there; partial SASA|len=+0.4109 p=0.072; partial len|SASA=+0.0014 p=0.995. Power simulated on the real n=21 design matrix: true partial 0.30->28.4%, 0.40->46.4%, 0.50->70.5%, 0.60->90.3%, 0.70->99.1%. Observed +0.411 sits at ~47% power, so p=0.072 is the EXPECTED outcome for a real moderate effect.

OTHER PROXIES (raw / |length / r with length):
  mean_rel_SASA_hydrophobic +0.649 / +0.640 / -0.141  <-- near length-independent
  frac_exposed_rel>0.5      +0.575 / +0.554 / -0.517
  SASA_over_len^0.73        +0.581 / +0.568 / -0.562
  SASA_per_residue          +0.493 / +0.585 / -0.839
  frac_buried_rel<0.25      -0.419 / -0.369 / +0.591
  mean_hydropathy_KD        +0.200 / +0.385 / +0.537 (raw NOT sig)
  frac_hydrophobic          +0.233 / +0.263 / +0.104 (raw NOT sig)
  plddt_mean (n=24)         -0.426 / -0.023 / +0.712 <-- DIES, as in W8

b_p CHANNEL (clean of length): raw r(frac_buried, b_p_wt_error) = -0.4969 p=0.0071; raw r(length, b_p) = -0.1376 p=0.485; partial frac_buried|len = -0.5202 p=0.0054; partial len|frac_buried = +0.2231 p=0.263.

GATE: scripts/gate_g4_cpu.py -> "baseline ... PASS dG=-0.0030 width=1092", G4-CPU ALL PASS.

**METHOD:** Reused scripts/catalogue_vs_bp.py's recover_bp() and structural_features() (identical SASA/Shrake-Rupley code path as CHECKPOINT 8, Tien 2013 max-SASA for relative SASA) so structural numbers match what was already reported. Wrote three new scripts run on all 10 eval CSVs in eval_results/.

(1) length_confound.py: per-protein a_p via the calib_diag convention (polyfit of ddG_pred on ddG_true, row 0 of each protein group = WT). Added two statistical covariates per protein: n_mut (rows the slope was fitted on) and sd_ddg_true / range / mean|ddG| (regressor spread) -- both confounds for any fitted slope a_p=cov/var independent of structure. Pearson + Spearman, and partial correlations (df=n-3) of a_p~SASA|length, a_p~length|SASA, a_p~SASA|n_mut, a_p~SASA|sd_ddG, per checkpoint.

(2) length_confound2.py: standardised multivariate OLS with t-stats and VIF; semipartial (part) correlations for unique variance; leave-one-protein-out on both partials (28 folds); drop-the-designed-proteins subset; median-split length stratification; a sweep of 9 alternative exposure/composition proxies raw vs controlling length; and the b_p channel (frac_buried vs b_p_wt_error controlling length) as a cross-check.

(3) length_confound3.py: to resolve the n=21 ambiguity -- a power simulation that generates y with a KNOWN true partial effect on the length-residualised SASA using the REAL (SASA,length) design matrix so the collinearity is exact; designed-status as a third covariate (residualising on [length,designed], df=n-4); within-stratum r(SASA,a_p) recomputed on each of the 10 checkpoints separately rather than on the mean; and a 10k bootstrap CI on the sr2 difference.

Verification discipline: two nohup launches died silently with zero-byte logs (the known failure signature). Relaunched with setsid nohup ... < /dev/null & and confirmed every number against the on-disk JSON/log artifact, never a success message. CPU only, no GPU, no sbatch.

**NEXT ACTION:** Do NOT rewrite CHECKPOINT 8 -- the exposure finding survives and length is a suppressor, not a cause. Two concrete edits to the writeup instead: (1) report mean_rel_SASA_hydrophobic (+0.640 controlling for length, r with length only -0.141) alongside mean_rel_SASA, since it makes the claim without needing the confound argument at all; (2) state explicitly that raw hydropathy (+0.200) and frac_hydrophobic (+0.233) are NOT significant, so the mechanism is geometric EXPOSURE and not compositional hydrophobicity -- this pre-empts the obvious examiner question. Add the length partial-correlation table and the within-stratum table to the thesis as the confound-control evidence, and carry the n=21 power caveat as a stated limitation. The pLDDT contrast (dies at -0.023 in the same test that leaves SASA at +0.680) is worth including as the internal positive control showing the test has teeth.

**CAVEATS:** 1. THE REAL CAVEAT: on the natural-only n=21 subset, r(SASA,length)=-0.884 and the SASA partial falls to +0.4109 (p=0.072) -- above the 0.05 bar. I argue this is a power failure (simulated power ~47% at the observed effect size on the real design matrix), and length is +0.0014 p=0.995 there so the subset does not favour length either, but a hostile examiner can point at that p=0.072. The defence is the within-stratum result and mean_rel_SASA_hydrophobic, not the n=21 partial.

2. The 10 "checkpoints" are not 10 independent training runs -- they are 5 sigma seeds plus 3 anchor weights plus a calib control plus a p3 slope variant, all on the same 28 test proteins with the same structures. Sign-consistency 10/10 therefore measures robustness to training config and seed, NOT independent replication of the structural effect. The structural covariates are literally identical across all 10, so the 10 tests are strongly dependent and 10/10 is not 10 independent confirmations.

3. Structures are AlphaFold models, not experimental. SASA is computed on predicted coordinates, so exposure error correlates with prediction confidence -- and pLDDT itself correlates with length at +0.712. Mitigated by pLDDT dying under the length control (-0.023) rather than competing with SASA, but not eliminated.

4. a_p is itself a fitted quantity with error; mean across-checkpoint SD of a_p is 0.1545 against a between-protein SD of 0.2017, so a substantial fraction of a_p's spread is estimation/seed noise. This attenuates all correlations toward zero, meaning the true |r| values are likely LARGER than reported -- it biases against the finding, not for it, but it also means the point estimates should not be quoted as precise.

5. Length range is only 43-72aa on single-chain ligand-free monomers. Nothing here tests whether the exposure-slope relation survives at 150aa+, where length and exposure decouple and multi-domain structure appears. GENERALISATION CLAIM this benchmark cannot settle.

6. Multiple comparisons: item 7 sweeps 9 proxies x (raw, partial) and the per-checkpoint tables are 10 correlated tests each. No formal family-wise correction was applied to the proxy sweep; the primary SASA result (p=1.3e-5 on the reference checkpoint) clears any reasonable Bonferroni bar, but the weaker proxies (mean_hydropathy_KD |length = +0.385 p=0.047) should not be treated as established.

7. The designed/natural split was made by a filename heuristic (first character not a digit), giving 7 designed. It matches the DESIGNED_RE intent in catalogue_vs_bp.py but was not independently verified protein by protein.

**GPU HOURS:** 0 (CPU only; ~3 min total wall clock across three scripts on the login node)

**FILES:**
- /home/nissimb/DeepPEF/results/LENGTH_CONFOUND.md
- /home/nissimb/DeepPEF/scripts/length_confound.py
- /home/nissimb/DeepPEF/scripts/length_confound2.py
- /home/nissimb/DeepPEF/scripts/length_confound3.py
- /home/nissimb/DeepPEF/results/length_confound.json
- /home/nissimb/DeepPEF/results/length_confound2.json
- /home/nissimb/DeepPEF/results/length_confound3.json
- C:\Users\User\AppData\Local\Temp\claude\C--Users-User\eee8e8e1-58ee-4bbf-8e95-538e7a9fb0ba\scratchpad\LENGTH_CONFOUND.md
- C:\Users\User\AppData\Local\Temp\claude\C--Users-User\eee8e8e1-58ee-4bbf-8e95-538e7a9fb0ba\scratchpad\length_confound.py
- C:\Users\User\AppData\Local\Temp\claude\C--Users-User\eee8e8e1-58ee-4bbf-8e95-538e7a9fb0ba\scratchpad\length_confound2.py
- C:\Users\User\AppData\Local\Temp\claude\C--Users-User\eee8e8e1-58ee-4bbf-8e95-538e7a9fb0ba\scratchpad\length_confound3.py

## LINE 5 — free accuracy from existing checkpoints (ensembling), CPU only, no training
**Verdict:** NEGATIVE

**HEADLINE:** Seed ensembling gives +0.0095 pooled ddG PCC (0.6050 -> 0.6145) — statistically certain but negligible — and provably CANNOT change a_p at all: a_p and b_p are linear functionals of the predictions, so ensemble a_p equals mean-member a_p to 6.66e-16 (exact). The brief's prediction that ensembling would worsen the compression is refuted by algebra, not just by measurement.

**NUMBERS:** ALIGNMENT (trap cleared first): the 10 eval CSVs share an identical label multiset but are ROW-SHUFFLED (naive mean would be garbage; row-wise label maxdiff 5.95 kcal/mol). Joined on (protein, deltaG) — identical key multisets, ddG labels agree to 0.000e+00, 0 NaNs, 5 dup keys dropped -> 28,304 rows, 28 proteins.

1. SIGMA-5 SEED ENSEMBLE. Members pooled ddG PCC: 0.5947 / 0.6574 / 0.5797 / 0.5927 / 0.6007, mean 0.6050. ENSEMBLE 0.6145 (+0.0095). Per-protein PCC median 0.7916 -> 0.7982. dG MAE 1.1167 -> 1.0729. ddG RMSE 1.0380 -> 1.0296. Size sweep k=1..5: 0.6050, 0.6110, 0.6130, 0.6139, 0.6145 (saturates by k=3). Cluster bootstrap over the 28 proteins (2000 reps): gain +0.0085, 95% CI [+0.0044, +0.0132], P(gain>0)=1.000; but the ensemble PCC LEVEL has 95% CI [0.4809, 0.7286] — the level CI is ~100x wider than the gain. Per-protein: ensemble better on 27/28, Wilcoxon p=2.76e-06.

2. a_p — THE KEY RESULT. mean a_p ensemble = 0.4287, mean-of-members = 0.4287, max deviation 6.66e-16 (b_p: 4.44e-16). Exact identity because a_p = cov(y,pred)/var(y) is linear in pred with a label-only denominator, and averaging is linear => slope-of-mean == mean-of-slopes. Median a_p ensemble 0.3939 sits INSIDE member range [0.3271, 0.4366]. std(a_p) across proteins 0.1996 (members) -> 0.1967 (ensemble). Compression a_p~0.5 is immune to ensembling.

3. b_p — PREDICTION STATED BEFORE MEASURING. Predicted from ICC=0.898: std(b_p) x sqrt(0.898+0.102/5)=0.958x, ~4% reduction, "barely moves". MEASURED 1.0982 -> 1.0722 = 2.36% reduction. PREDICTION CORRECT in sign and magnitude. Seed-noise SD within protein 0.4051; ICC(b_p) re-estimated on these 5 seeds = 0.8560 (independently reproduces the known 0.898).

4. WHY THE GAIN IS SMALL. Error decomposition: mean member MSE 1.07857, ensemble/common MSE 1.05998, diversity (reducible) 0.01859 = 1.7% of member MSE. 98.3% of member error is SHARED bias. Inter-seed pred correlation 0.9619 (0.925-0.985).

5. EPOCH ENSEMBLING: NOT POSSIBLE from disk. All 15 epochs (0-14) exist as WEIGHT checkpoints for ~25 runs (382 .pt, 88MB each), but per-epoch prediction dumps = NONE, *.npy/*.npz = NONE. Each eval CSV is a single epoch.

6. HONEST CEILING. all-10 ensemble 0.6197; all-10 z-avg 0.6206; sigma-5 0.6145; all-10 median 0.6116; mean single member 0.5993; rank-avg WORSE (0.5930 / 0.5852). Test-selected best single member 0.6574 = ORACLE (exhaustive search over all 1023 subsets picks k=1, i.e. no ensemble beats one member). LOPO-honest subset selection 0.6224 — beats fixed all-10 by only +0.0027, recovers 40% of the gap to the oracle. Ensemble + oracle per-protein mean removal 0.8047; + full oracle affine 0.8396. Spearman: 0.5723 -> 0.5838.

**METHOD:** Joined the 10 eval CSVs on (protein, deltaG-label) after proving the naive row-order join is invalid (files are shuffled permutations of one label multiset); validated the join as a bijection (identical sorted label vectors bitwise, ddG label agreement 0.000e+00, no NaNs). Computed pooled/per-protein ddG PCC, Spearman, dG MAE, ddG RMSE, per-protein a_p (OLS slope of pred_ddG on true ddG) and b_p (mean dG-space residual) for each member, for the 5-seed ensemble, the all-10 ensemble, and every subset size. Significance by CLUSTER bootstrap resampling the 28 proteins (mutations within a protein are not independent) plus Wilcoxon signed-rank on per-protein PCC. Proved and numerically verified the linearity identity a_p(mean)=mean(a_p). Decomposed member squared error into shared (common) and seed-specific (diversity) components. Tested honest selection by leave-one-protein-out choice of the member subset vs test-set selection. Checked disk directly for per-epoch prediction artifacts. Scripts ens.py/ens2.py/ens3.py run on the cluster login node, CPU only, no GPU, no training. Verified the output artifact on disk (size, line count, md5, key numbers) rather than trusting the success message.

**NEXT ACTION:** Bank the all-10 ensemble as a free +0.02 (0.5993 -> 0.6197) and close Line 5 — it is exhausted and cannot address calibration. Do NOT spend GPU on more seeds: with 98.3% of error shared and inter-seed correlation 0.96, more seeds of the same config are asymptotically worthless (the k-sweep already saturates at k=3). The one remaining nearly-free item is a within-run weight-space soup over the last k epochs, which needs ONE CPU forward pass over 28k mutations per soup (submit as a --partition=cpu job) — worth a single cheap test, but the linearity result means even if it helps PCC it will not move a_p. Redirect effort to the slope channel: a_p is the strongest per-protein quality predictor (r=+0.5714 with per-protein PCC) and mean_rel_SASA predicts it at r=+0.714, and unlike b_p the slope does NOT cancel in ddG — so a per-protein slope corrector is the only lever here with headroom to the 0.80-0.84 oracle.

**CAVEATS:** 1. n=28 proteins. The gain (+0.0095) is far smaller than the uncertainty on the LEVEL (95% CI [0.481, 0.729]). The gain is significant only because it is a paired within-protein comparison; the absolute pooled PCC is not pinned to better than ~0.1 by this benchmark.
2. The 0.6574 "best member" and the 0.6574 best subset are selected ON THE TEST SET and are oracles, not methods. Reporting either as an achievable result would be test-set selection. The honest number is ~0.62.
3. My LOPO selection holds out proteins, not the training set — the members themselves were all trained on the same training data, so LOPO tests only the honesty of the SUBSET CHOICE, not of the models. It is the right null for "which ensemble do I pick", not a full generalisation estimate.
4. The 5 abl_sigma runs differ in eval epoch (e13/e10/e13/e9/e14), so "seed ensemble" is not a pure seed-only contrast; epoch choice is partially confounded with seed.
5. 10 of 28,314 rows (0.035%) were dropped as ambiguous join keys. Immaterial but disclosed.
6. The oracle numbers I compute (0.8047 offset / 0.8396 affine) are NOT the same object as the 0.70-0.72 offset oracle in CONTEXT: mine removes the per-protein ddG-space mean, whereas a dG-space b_p constant cancels exactly in ddG by the metric rule and cannot change pooled ddG PCC at all. Do not compare the two figures directly.
7. The algebraic claims (linearity => exact invariance of mean a_p) are dataset-independent and hold generally. The empirical claims (1.7% diversity, 0.96 correlation, +0.02 ceiling) are specific to these 5 seeds of one architecture on these 28 canonical monomers; a pool diverse in ARCHITECTURE rather than seed would likely ensemble better — though by the same algebra it still would not change a_p.
8. Weight-space averaging across seeds was not attempted and would be invalid (independent runs are not mode-connected).
9. Gate re-verified: scripts/gate_g4_cpu.py baseline line reads PASS dG=-0.0030 width=1092, G4-CPU ALL PASS. Note the gate's LAST line is the burial sub-case (dG=0.0023 width=1095, which adds 3 dims by design) — reading `tail` alone would falsely suggest a gate failure. No repo code was modified; only standalone analysis scripts added.

**GPU HOURS:** 0. All work was CPU-only on the login node (a few minutes total). No training, no GPU, no jobs submitted. Follow-up epoch-soup test would need ~0 GPU hours but ~1 CPU job (one forward pass over 28k mutations per soup).

**FILES:**
- /home/nissimb/DeepPEF/results/ENSEMBLE.md
- /home/nissimb/DeepPEF/results/ensemble_stats.json
- /home/nissimb/DeepPEF/ens.py
- /home/nissimb/DeepPEF/ens2.py
- /home/nissimb/DeepPEF/ens3.py
- C:\Users\User\AppData\Local\Temp\claude\C--Users-User\eee8e8e1-58ee-4bbf-8e95-538e7a9fb0ba\scratchpad\ENSEMBLE.md

## LINE 1
**Verdict:** PROMISING

**HEADLINE:** The std-matching objective has a hard algebraic CEILING at a_p = r = 0.7930: since a_p = r * sd(pred)/sd(true), driving sd(pred)->sd(true) pins the slope to the correlation, never to 1.0. But the ceiling is worth having -- a_p median 0.4119 is 62.8% spread compression and only 36.8% ranking error, so a spread objective can recover 64.8% of the gap to a_p=1.

**NUMBERS:** IDENTITY (exact, 280 protein-checkpoint cells, 10 CSVs x 28 proteins): max|a_p - r*ratio| = 1e-16 to 6.7e-16. Not an approximation.

DECOMPOSITION, 6 slope-free replicates (sigma_seed1/2/3/4/42 + calib_ctrl), n=28 proteins:
  a_p median 0.4119 | r median 0.7930 | ratio median 0.5370
  Log budget: -log a_p = 0.9316 = -log r (0.3429, 36.8%) + -log ratio (0.5848, 62.8%)
  Per-checkpoint mean over all 10: r's share = 39.0%
  In slope units: recoverable by a spread objective 0.4119->0.7930 = 0.3811 = 64.8% of gap to 1.0;
  structurally unreachable 0.2070 = 35.2% of gap.

WHICH FACTOR DRIVES a_p (n=28, p=0.05 threshold |r|=0.374):
  corr(a_p, ratio) = +0.9436, p=3e-14   <- a_p IS essentially the ratio
  corr(a_p, r)     = +0.5773, p=0.0013
  corr(r, ratio)   = +0.3007, p=0.120   <- NOT distinguishable from zero; channels independent

CEILING IS HARD, COMPRESSION IS SOFT (ICC(1) over 6 replicates):
  r     median 6-seed SD 0.0098, ICC 0.960
  a_p   median 6-seed SD 0.0730, ICC 0.839
  ratio median 6-seed SD 0.1003, ICC 0.780

BLINDNESS DEMO (synthetic, n=2000): loss = 0 achievable at a_p = 0.276 / 0.497 / 0.798 / 1.000, all with sd(q)/sd(t) = 1.000000. a_p > 1 is UNREACHABLE under std-matching (needs r>1).

THE ONE slope_weight=3.0 RUN DID NOTHING. Naive Wilcoxon vs its seed-42 control says ratio 0.5573->0.4489, p<1e-4 -- but that is PSEUDO-REPLICATION (28 proteins treated as replicates of an n=1 run-level effect). Against the correct 6-replicate null: ratio z=-1.08 (rank 1/6), a_p z=-1.26 (rank 0/6), r z=+0.77. INSIDE the seed band [0.4220, 0.6342].

WHY IT FAILED (not the ceiling -- it stopped far short of it):
  (a) Sign is right: sd_pred<sd_true for 25/28 proteins; P(sd_pred>sd_true) in a minibatch = 0.217/0.178/0.157/0.144 at B=4/8/16/32, so 78-86% of the time it pushes UP correctly.
  (b) ||grad |sd_q-sd_t| ||_2 = 1/sqrt(n) EXACTLY, constant regardless of mismatch size, non-vanishing and non-smooth at its own optimum -> chatter.
  (c) The data loss WANTS compression: MSE-optimal rescale k* = r*sd_t/sd_p median 1.2028 drives a_p -> r^2 = 0.6274, BELOW the std target r = 0.7930. The two terms genuinely disagree.
  (d) Under-determined: sd_q^2 = a_p^2 sd_t^2 + sd_resid^2; noise share of predictive variance 0.373. sd-matching is satisfiable by inflating noise alone.

MSE PRICE (median centred per-protein): now (a_p=0.437) 0.4296 | std-matched (a_p=r) 0.3365 | slope-forced (a_p=1) 0.6236. Std-matching is CHEAPER in MSE than the status quo -- the models are over-shrunk even by MSE's own criterion, so the first part of decompression is free. Forcing a_p=1 costs 1.85x.

ALTERNATIVES:
  A |sd_q-sd_t| (current): fixed pt sd_q=sd_t -> a_p=0.793 ceiling; grad 1/sqrt(n) constant; blind to a_p.
  B (sd_q/sd_t - 1)^2: SAME ceiling 0.793, but smooth, grad vanishes at optimum, scale-free across sd_true range 0.4767-1.5847 (3.3x). Strictly dominates A; free upgrade; does not raise the ceiling.
  C (1 - a_hat)^2, a_hat = <q',t'>/<t',t'>: fixed pt a_p = 1, NO CEILING. ||grad|| = 2|1-a_hat|/(sqrt(n) sd_t) = 1.356x the L_A gradient at our operating point (identical ratio at n=8/16/32). Projects error onto t', so noise inflation earns nothing -- closes hole (d). Cost: needs sd_q = sd_t/r, i.e. OVER-spread by 1/r = 1.2624 (26.2% wider than truth), so C and A/B are in direct conflict and must not be summed.
  D C + lam(1-r): same a_p fixed point plus pressure on r; but r has ICC 0.960 and moved <0.003 across 6 seeds -- looks like a capacity limit, not loss-shiftable. Not expected to earn its hyperparameter.
  RECOMMENDED: C, low weight, per-protein, clipped as max(0, 1-a_hat)^2 (2WXC a_p=0.924 and 1W4H a_p=0.990 are already at slope 1 and would be pushed past it).

INCIDENTAL: abl_anchor_w3.0 has ratio 0.2272 / a_p 0.1535 / r 0.6669 -- the WT anchor at weight 3.0 takes over the optimisation and wrecks r (0.667 vs the 0.792 +- 0.0027 replicate band). But anchor at w=0.3/1.0 gives ratio 0.8142/0.8454, the two HIGHEST in the entire set. The strongest observed spread lever here is the WT anchor at LOW weight, not slope_weight at all (n=1 per weight; suggestive only).

**METHOD:** All CPU, on the 10 existing eval CSVs in /home/nissimb/DeepPEF/eval_results/ (28,314 rows each; 28 test proteins; 771-1334 variants per protein). No training run, no GPU, no sbatch.

1. Read the actual loss term first: Megascale-fineTuning/train.py:736 is slope_loss = torch.abs(slope_pred_ddg.std(unbiased=False) - slope_true_ddg.std(unbiased=False)), an L1 marginal-spread mismatch on a per-protein minibatch (WT = row 0, ddg = output - wt_dg).
2. Per protein per checkpoint computed a_p = cov(q,t)/var(t), r = corr, ratio = sd(q)/sd(t), and verified the identity a_p = r*ratio to floating point.
3. Decomposed the deficit additively in logs (-log a_p = -log r + -log ratio), which is the only honest split of a multiplicative identity.
4. Built the null correctly: the 6 slope-free replicates (5 seeds + calib control) as the run-level null band, rather than the 28 proteins, which are pseudo-replicates of an n=1 training effect. Reported z and rank against that band.
5. ICC(1) over the 6 replicates for r, a_p, ratio to separate seed noise from genuine between-protein signal.
6. Gradient norms derived analytically and evaluated at the measured operating point (sd_pred=0.5998, sd_true=0.8309, r=0.7921, a_p=0.4366) for n=8/16/32.
7. Minibatch simulation (200 draws per protein at B=4/8/16/32) to establish the sign the term actually sees during training.
8. Constructive synthetic demo (n=2000) that loss=0 is attainable at every a_p in (0,1].
9. Verified every artifact on disk by ls/wc/head/tail/json-parse rather than trusting the success line, per the known failure mode.
10. scripts/gate_g4_cpu.py re-run after all work.

**NEXT ACTION:** Implement objective C as a new flag (do NOT edit --slope_weight, keep it for comparability): per-protein slope_loss = torch.clamp(1.0 - a_hat, min=0.0)**2 where a_hat = ((q-q.mean())*(t-t.mean())).sum() / ((t-t.mean())**2).sum().detach(). The .detach() on the denominator is important -- it is constant w.r.t. the model and keeps the gradient exactly the derived 2|1-a_hat|*t'_i/(n var t). Guard it the same way --slope_weight is guarded (requires --loss_mode ddg|joint|ddg_head, no-op at weight 0 so baseline stays bit-identical), and add it to scripts/gate_slope.py. Sweep weight 0.1/0.3/1.0 -- NOT 3.0, which is where abl_anchor_w3.0 shows an auxiliary term hijacking the optimisation and destroying r. Because the run-level SD of the ratio is 0.0800, a single run cannot be read: budget >=3 seeds per weight and test against the existing 6-replicate band, not against 28 proteins. Cheap parallel win with no new training: --wt_anchor_weight 0.3 and 1.0 already produce the two highest ratios in the whole collection (0.8142, 0.8454) -- replicate those two across 3 seeds before building anything new, since the lever may already exist.

**CAVEATS:** - n = 28 proteins. The p=0.05 two-sided threshold is |r| = 0.374. corr(r, ratio) = +0.3007 is NOT distinguishable from zero, so "the two channels are independent" is a failure to reject, not a demonstration. If they are in fact coupled, the 64.8% recoverable figure is optimistic.
- n = 6 training runs for the replicate null and n = 1 for slope_weight=3.0. "slope_weight did nothing" means "no effect detectable at |z| < 1.1 with n=6"; an effect of ~0.08 in the ratio could hide inside that band. It is NOT proof the term is inert.
- Epoch is confounded with condition across the 10 checkpoints (e8 through e14) and cannot be separated at n=6: corr(epoch, ratio) = -0.295, p=0.57.
- I initially misread the slope3.0 comparison as "the lever made compression 31% WORSE" on the strength of Wilcoxon p<1e-4. That was wrong -- pseudo-replication. Corrected to "no detectable effect" once the 6-run null was built. Flagging because the wrong version is the more dramatic claim and must not propagate.
- The CEILING result (section 1) is algebra and is not subject to any of the above -- it holds identically. Only the SIZES (r=0.793, 64.8%, 62.8%) are benchmark estimates.
- Sections 4c and 5C PREDICT the consequences of alternative objectives from the algebra and from current predictions. No alternative objective has been trained. The 1.85x MSE cost and the 1.356x gradient ratio are derived, not observed.
- All 28 test proteins are single-chain, ligand-free, metal-free monomers 43-72aa. Whether r = 0.79 -- and hence the ceiling's height -- transfers to larger or multi-chain targets is a GENERALISATION CLAIM this benchmark cannot test. On a harder target set r would likely be lower, which LOWERS the ceiling and makes objective C relatively more valuable.
- pct_deficit_from_r is ill-defined for 2WXC and 1W4H (a_p ~ 1 puts -log a_p near zero, so the ratio explodes); those two rows are excluded from that column's interpretation but included in all medians and correlations.
- Gate verified: scripts/gate_g4_cpu.py -> dG=-0.0030 width=1092, G4-CPU ALL PASS. Nothing in the repo's model/feature path was modified; only new files were added under results/ and scratch_slopeobj/.

**GPU HOURS:** 0 GPU-hours consumed (this was entirely CPU analysis on existing eval CSVs). Follow-up cost: objective C at 3 weights x 3 seeds = 9 runs; at the ~14-epoch scale of the existing checkpoints that is roughly 9 x the per-run cost of the abl_* family. The WT-anchor replication (2 weights x 3 seeds, minus the 2 runs that already exist) is 4 runs and should be done first as it needs no code change.

**FILES:**
- /home/nissimb/DeepPEF/results/SLOPE_OBJECTIVE.md
- /home/nissimb/DeepPEF/results/_slope_decomposition.csv
- /home/nissimb/DeepPEF/results/_slope_objective.json
- /home/nissimb/DeepPEF/scratch_slopeobj/slope_obj.py
- /home/nissimb/DeepPEF/scratch_slopeobj/slope_obj2.py
- /home/nissimb/DeepPEF/scratch_slopeobj/slope_obj3.py
- /home/nissimb/DeepPEF/scratch_slopeobj/slope_obj4.py
- C:\Users\User\AppData\Local\Temp\claude\C--Users-User\eee8e8e1-58ee-4bbf-8e95-538e7a9fb0ba\scratchpad\slopeobj\SLOPE_OBJECTIVE.md

## LINE 4
**Verdict:** PROMISING

**HEADLINE:** a_p IS driven by a chemical class: mutations TO hydrophobic residues are compressed ~49% harder than TO polar (within-protein slope 0.334 vs 0.497; corr(slope, Kyte-Doolittle) r=-0.833, 10/10 checkpoints; 26/28 proteins) — but rank accuracy is degraded in lockstep (Spearman rho 0.490 vs 0.737), so it is LOST INFORMATION, not a rescalable calibration error.

**NUMBERS:** PROVENANCE (new capability): mutation identity recovered for all 28,314 eval rows by replaying evaluate.py's filter chain; max |reconstructed dg - CSV deltaG| = 0.0 exactly, all 28 proteins. 28,172 single point mutants, 142 wt, 0 multi-mutants. Sign convention verified independently: mean ddG where Stabilizing_mut=True = +1.474 vs False = -1.153 => positive ddG = stabilising.

Q1 MUTATION TYPE (353 of 380 observed). Worst mean signed error: F>E +1.441, F>A +1.420, F>Q +1.412, F>R +1.399, F>G +1.389 (all 10/10 sign-consistent, t=5.5-7.1). By SOURCE: K +0.098, E +0.130 ... Y +0.810, W +0.840, F +1.185. By DESTINATION: C +0.162 ... G +0.495, P +0.735. All 19 sources and 20 destinations 10/10 sign-consistent.
BUT this is ~90% compression, not chemistry: corr(mean_true, mean_err) = -0.906 (p=9.6e-8) over 19 sources, -0.919 (p=1.0e-8) over 20 destinations. Residual SD collapses 0.273->0.116 (source), 0.107->0.042 (dest). Surviving residuals are small: H +0.223, F +0.204, V -0.245, L -0.142.

Q2 POSITION/BURIAL. Sequence position is a CLEAN NULL: corr(seq_frac, signed err) = -0.005, p=0.42 (n=28,172); slope wanders 0.38-0.53 with no monotone trend. Burial: corr(rel_sasa, signed err) = -0.289; within-protein slope core(<0.1) 0.427 vs exposed(>0.5) 0.304, 10/10 checkpoints. Note sign flip vs the known between-protein mean_rel_SASA/a_p result (+0.714) — different objects.

Q3 STAB vs DESTAB. Imbalance: 78.27% destabilising (22,049), 21.73% stabilising (6,122), only 6.82% strongly stabilising (>+0.5), 3.72% > +1. True ddG skew -0.06 vs predicted skew -1.46. Pooled PCC stab 0.570 vs destab 0.535 (deceptively similar). WITHIN-PROTEIN: stab PCC 0.425 vs destab 0.678; stab slope 0.268 vs destab 0.423. Signed errors equal and opposite: -0.579 stab, +0.655 destab. Design utility survives: top-10 per protein = 55.4% true-stabilising precision vs 21.7% base = 2.55x enrichment (top-50 2.30x, top-100 2.15x).

Q4 THE BIG QUESTION — YES, with a hard honest limit. TO-hydrophobic (AVILMFWCY) slope 0.334 vs TO-polar 0.497; per-residue C 0.212, F 0.247, L 0.251 ... N 0.445. corr(slope, KD hydropathy) r=-0.855 p=1.6e-6 (n=20); over 10 ckpts mean r=-0.833, 10/10 negative. Polar-minus-hydrophobic gap +0.149, 10/10 positive. t=-13.4.
NOT BURIAL: mean rel_SASA 0.3898 (hyd) vs 0.3904 (pol) — identical; gap +0.106 to +0.135 inside EVERY burial stratum.
UNIVERSAL: gap>0 in 26/28 proteins (binomial p=3.0e-6), 26-28/28 in all 10 ckpts. Median a_p: all 0.409, TO-hyd 0.251, TO-pol 0.439; TO-pol>TO-hyd 10/10.
LIMIT 1 — does NOT explain between-protein spread: corr(frac_toh, a_p) = -0.243, p=0.21 NS (n=28, threshold 0.374). OLS a_p ~ frac_toh + frac_from_arom + frac_stab: R2=0.206, adj R2=0.106. corr(a_p, gap)=+0.741 but corr(a_p, a_hyd/a_pol)=+0.059 p=0.77, ratio median 0.655 => defect is MULTIPLICATIVE and universal. mean_rel_SASA (+0.708) remains the only real a_p predictor.
LIMIT 2 — NOT rescalable: LOPO class rescale 0.5917 vs raw 0.5987, beats raw 0/10. Even the global per-class affine ORACLE (fitted on test) gives 0.5999, +0.0012. Reason: rank is degraded too. Within-protein Spearman rho: all 0.651, TO-hyd 0.490, TO-pol 0.737 — PERFECT class separation, zero overlap (F 0.414 < L 0.424 < W 0.447 < C 0.481 < I 0.492 < Y 0.506 < V 0.515 < M 0.515 < A 0.639 | H 0.652 < T 0.678 < ... < D 0.749). Two variance controls: decile-stratified rho_hyd 0.125 vs rho_pol 0.169 (10/10); histogram-matched resampling rho_hyd 0.490 vs rho_pol 0.642, gap +0.152 (10/10).

**METHOD:** Mutation identity did not exist in the eval CSVs (columns are only protein/deltaG/pred_deltaG/ddG/pred_ddG), so I first reconstructed it by replaying the exact filter chain in Megascale-fineTuning/evaluate.py::load_test_protein_data: drop ins/del + reset_index, attach deltaG.pt positionally, drop ddG_ML=='-' (UNSTABLE_MUT=False default), drop names not in ThermoMPNN/mega_test.csv (DS_TYPE='pnas'), then list(set(ints)) which CPython iterates ascending. Validated by joining PER PROTEIN (the eval DataLoader uses shuffle=True so whole-file positional join is wrong) and checking max |dg - deltaG| = 0.0 over all 28,314 rows — the artifact, not a success message. Per-residue rel SASA via Bio.PDB ShrakeRupley over the AlphaFold PDBs reusing the MAXASA table from scripts/bp_replication.py, with a wt-identity check at each mutated position. All slopes fitted WITHIN-PROTEIN-CENTRED so b_p is stripped and the quantity is a genuine within-protein slope (ddG is the right metric for a_p per the metric rule; I deliberately did not use dG here). Every claim required 10/10 checkpoint sign-consistency rather than a p-value, since n=10^4 makes any effect nominally significant. Two independent controls for the variance confound (decile stratification and histogram-matched resampling), a bootstrap CI on every class slope, and an honest LOPO test of the implied fix against the raw null. Sign convention derived from the data (Stabilizing_mut column), not assumed. CPU only; no training, no sbatch.

**NEXT ACTION:** The class defect is real and universal but is lost information, not miscalibration, so DO NOT build an output-side class rescaler — that is already tested and dead (LOPO 0/10, oracle +0.0012). The finding points at the INPUT side: the model cannot see hydrophobic packing. Concrete next test, CPU-cheap: check whether the existing burial-features lever (the +3-dim block that gate_g4_cpu already exercises at width=1095) closes the TO-hydrophobic rho gap of 0.490 vs 0.737 — score an existing burial-lever checkpoint with scratch_nb/q4c.py and compare that single number. That is a sharp, pre-registered, falsifiable target: if burial features are the right representation fix, rho_toHyd must rise. If a GPU run is warranted afterwards, the training-side variant is a per-class loss reweighting (upweight TO-hydrophobic), which unlike rescaling can actually recover rank information. Also worth 10 minutes: the 2KVS anomaly now has a partial explanation — its TO-hydrophobic slope is -0.009 (sign-inverted), so the model is not merely weak there, it is anti-correlated on hydrophobic substitutions.

**CAVEATS:** 1. GATE: scripts/gate_g4_cpu.py run after all work -> "G4-CPU: ALL PASS", baseline line exactly "dG=-0.0030 width=1092". No repo code modified; everything written under scratch_nb/ and results/. Nothing sbatched, killed, or pushed; autopilot untouched.

2. Mutation identity is RECONSTRUCTED, not stored. It relies on CPython's set-of-small-ints iterating in ascending order. This is validated to max |dG diff| = 0.0 on all 28,314 rows, which is extremely strong evidence, but it is a reconstruction and any future change to evaluate.py's filter chain silently invalidates it. build_master.py re-asserts the check on every run.

3. PRE-EXISTING EVAL BUG FOUND (not introduced by me, and I did not fix it): ddG is referenced to row 0, a SINGLE wt replicate, not the wt mean. Each protein has 5 wt replicates (2K5H has 7). Replicate SD is normally 0.05-0.31 kcal/mol, but 2K5H has wt SD = 1.498 (range 1.56-4.81), so that protein's entire ddG column carries an arbitrary ~1 kcal/mol reference offset. This plausibly contributes to its b_p and should be checked before any further b_p work. I kept the standard reference for consistency with all prior results.

4. The hydrophobic/polar split (AVILMFWCY vs rest) is a conventional partition chosen a priori, not fitted. The continuous version (Kyte-Doolittle, r=-0.833) is partition-free and agrees, and the per-residue rho ranking separates the two classes with zero overlap, so the conclusion does not depend on where the line is drawn. A is the borderline case (slope 0.332, rho 0.639) and sits at the boundary as expected.

5. n=28 for everything per-protein; I applied the |r|<0.374 threshold and explicitly marked frac_toh (-0.243) and frac_stab (+0.018) as NOT significant. The claim that composition drives the between-protein a_p spread is REJECTED, not supported — only the universal within-protein defect is supported.

6. The per-protein per-class affine number (0.8349) is an ORACLE fitted on test labels, not a method. Quoted only as an upper bound.

7. Generalisation: 28 single-chain, ligand-free, metal-free monomers of 43-72 aa, canonical residues only. Whether hydrophobic compression persists on larger, multi-chain, or ligand-bound proteins cannot be tested on this benchmark.

**GPU HOURS:** 0 GPU-hours used; entire analysis was CPU-only through the exec channel, longest single call well under the ~120s limit, no SLURM job needed. The recommended next check (rescoring an existing burial-lever checkpoint's eval CSV with q4c.py) is also 0 GPU-hours. Only the optional per-class loss-reweighting training arm would need GPU: ~1 run x ~15 epochs, comparable to the existing abl_* arms.

**FILES:**
- /home/nissimb/DeepPEF/results/PER_MUTATION.md
- /home/nissimb/DeepPEF/scratch_nb/build_master.py
- /home/nissimb/DeepPEF/scratch_nb/master_mut.pkl
- /home/nissimb/DeepPEF/scratch_nb/single_mut.pkl
- /home/nissimb/DeepPEF/scratch_nb/q1.py
- /home/nissimb/DeepPEF/scratch_nb/q1b.py
- /home/nissimb/DeepPEF/scratch_nb/q1c.py
- /home/nissimb/DeepPEF/scratch_nb/q2.py
- /home/nissimb/DeepPEF/scratch_nb/q3.py
- /home/nissimb/DeepPEF/scratch_nb/q4.py
- /home/nissimb/DeepPEF/scratch_nb/q4b.py
- /home/nissimb/DeepPEF/scratch_nb/q4c.py
- /home/nissimb/DeepPEF/scratch_nb/q1_subst.csv
- /home/nissimb/DeepPEF/scratch_nb/q1_dest.csv
- /home/nissimb/DeepPEF/scratch_nb/q1_src.csv
- /home/nissimb/DeepPEF/scratch_nb/q1b_slopes.csv
- /home/nissimb/DeepPEF/scratch_nb/q1c_to_slopes.csv
- /home/nissimb/DeepPEF/scratch_nb/q1c_from_slopes.csv
- /home/nissimb/DeepPEF/scratch_nb/q4_ap.csv
- /home/nissimb/DeepPEF/scratch_nb/recon_muts.csv
- C:\Users\User\AppData\Local\Temp\claude\C--Users-User\eee8e8e1-58ee-4bbf-8e95-538e7a9fb0ba\scratchpad\PER_MUTATION.md

## LINE 3: the two proteins that carry 78% of the oracle gain
**Verdict:** PROMISING

**HEADLINE:** 2K5H is not a model failure — it is a data bug. Its eval reference row is a mutant background (2K5H.pdb_G11S, dG=1.723) not the true WT (2K5H.pdb, dG=4.806), shifting all 1125 of its ddG labels by +3.0123 kcal/mol. Fixing the reference row destroys 46% of the entire oracle gain: +0.1125 -> +0.0611.

**NUMBERS:** REFERENCE BUG (the find): corr(source ddG_ML, our ddG) for 2K5H = 0.9967 with mean offset +3.0123; all 27 other proteins have r=1.0000 and offset in -0.647..+0.634. 2K5H's mutation file has 15 'wt'-labelled rows at 3 distinct dG levels (1.56, 1.72, 4.81) — the true WT sits at eval-order row 2, not row 0. Its WT dG is 2.80 kcal/mol BELOW its own median mutant; 85.2% of its mutants score as more stable than "WT".

FIX-AND-RECOMPUTE (means over all 10 eval CSVs):
  raw pooled PCC   0.5994 -> 0.5953  (-0.0041; Pearson shift-invariance, as predicted)
  oracle pooled    0.7119 -> 0.6564  (-0.0554)
  ORACLE GAIN      +0.1125 -> +0.0611  (46% of the gain was the bug)
  std(b_p)         0.1893 -> 0.1348  (-29%)
  b_p(2K5H)        -0.6891 -> -0.1020  (-85%)
  b_p(2KVS)        +0.3761 -> +0.3761  (unchanged)
  top-2 share      78.0% -> 59.2%
2K5H leaves the top-2 |b_p| list on 10/10 CSVs after the fix (replaced by 3DKM or r18_3_TrROS_Hall).

ROBUSTNESS as published (means over 10 CSVs):
  all 28      raw 0.5994  oracle 0.7119  gain +0.1125  (100%)
  drop 2K5H   raw 0.5731  oracle 0.6391  gain +0.0659  (59%)
  drop 2KVS   raw 0.6504  oracle 0.7367  gain +0.0863  (77%)
  drop BOTH   raw 0.6293  oracle 0.6685  gain +0.0392  (35% — 65% of the gain destroyed by dropping 2 of 28)
  correcting only the 2: 0.5994 -> 0.6873 (78% of gain, range 60-88%); only the other 26: -> 0.6152.
Pattern holds 10/10 CSVs.

2KVS — NARROW-RANGE HYPOTHESIS FALSIFIED. Its true ddG sd = 1.1474, rank 22/28, z = +0.78 (ABOVE average); IQR 1.8845 is the 2nd largest of 28. a_p = 0.0939 with SE 0.0166: t=+5.64 vs 0 (significantly non-zero) and t=-24.25 vs the cohort median 0.4975 (significantly flatter). Not ill-conditioned. corr(true_sd, a_p) over 28 = -0.2516, p=0.197 (below the |r|=0.374 threshold at n=28). So 2KVS should NOT be excluded.

2KVS MECHANISM — the burial channel is lost:
  corr(rel_SASA, pred) = -0.054 on 2KVS vs +0.367 on the other 26, while corr(rel_SASA, true) = +0.615 on 2KVS (STRONGER than the cohort's +0.452). The prediction instead tracks sequence position (corr(pos,pred) = -0.411) which is anti-correlated with truth (corr(pos,true) = +0.196).
  buried  n=538  a=0.241 PCC=0.355 meanErr=+2.719  (vs cohort a=0.458 PCC=0.593 meanErr=+0.646)
  exposed n=664  a=0.138 PCC=0.156 meanErr=+1.162
  41.8% of its 67 positions have per-position slope a<0.2; 20.9% have a<0 (model moves the wrong way). 2K5H: 19.4% and 9.7%.
  Replicated 10/10 CSVs: a_p 0.019-0.207 (median 0.100), PCC 0.121-0.510 (median 0.216).

ddG DISTRIBUTIONS (primary CSV):
  2K5H  n=1118 true mean +2.044 sd 1.586, 83.2% stabilizing; pred mean +0.298; mean err -1.746 MAE 1.848
  2KVS  n=1202 true mean -1.200 sd 1.148, 0.0% stabilizing; pred mean +0.659 (WRONG SIGN); mean err +1.859 MAE 1.880
  other26 n=25852 true mean -0.781 sd 1.018, 3.8% stabilizing; pred mean -0.414; mean err +0.367 MAE 0.600

NEITHER IS A STRUCTURAL OUTLIER (confirms prior): max|z| over 14 features = 1.26 for 2K5H (frac_G), 1.88 for 2KVS (frac_buried). Nothing reaches |z|=2.

2K5H IS OTHERWISE ONE OF THE MODEL'S BEST PROTEINS: a_p = 0.4990 (cohort median 0.4975) and per-protein PCC = 0.8610 (cohort median 0.798).

SCOPE OF THE BUG: 13 of 28 test proteins have >=2 WT dG clusters. Across all 862 mutation files: 367 have >=2 'wt' rows; 37 (10.1%) have row 0 on a mutant background; 37 (10.1%) have WT dG spread >1.0 kcal/mol, 29 (7.9%) >2.0. Worst: 5GU9 11.03, 1UBQ 9.70, 2KT8 8.63, 2LSS 7.90, 1SIF 7.85.

**METHOD:** Reconstructed exact per-row mutation identity for all 10 eval CSVs by replaying the selection in Megascale-fineTuning/evaluate.py::load_test_protein_data (drop ins|del, drop ddG_ML=='-', intersect name with data/ThermoMPNN/mega_test.csv, then list(set(...))). Verified exactly on all 28 proteins: row counts match to the row, max|deltaG_recon - deltaG_eval| < 4.7e-07 (float32 round-trip). This gave mutation identity, position, wt/mut residue for 28,172 of 28,314 rows.

Then: (a) computed SASA (Shrake-Rupley on the AlphaFold PDBs), CA-geometry secondary structure, radius of gyration, contact order, pLDDT and composition for all 28, z-scored within the cohort; (b) stratified per-mutation error by burial/SS/position; (c) cross-checked our ddG (= dG - dG[row0]) against the INDEPENDENT source ddG_ML column in the mutation files — this is what exposed the 2K5H offset; (d) re-referenced both deltaG and pred_deltaG to the true WT construct row (source name == '<PROTEIN>.pdb', else the wt row with min |ddG_ML|) and recomputed raw/oracle/std(b_p)/top-2 across all 10 CSVs; (e) audited all 862 mutation files for the same multi-WT pattern. All numbers are re-scorings of the existing eval CSVs; no model was retrained. Verified every artifact on disk (CSV files, byte counts), not success messages.

**NEXT ACTION:** Fix the reference row in Megascale-fineTuning/evaluate.py::load_test_protein_data: replace deltaG.iloc[0] with the row whose source name is the bare '<PROTEIN>.pdb' construct (or the mut_type=='wt' row with |ddG_ML| nearest zero). Then re-run every calibration number downstream of b_p on the corrected reference — the b_p corrector, the anchor levers, and the dG-space std(b_p)=1.5741 all inherit this origin. This changes the thesis headline and must be done before writing. Report 0.5953 -> 0.6564 as the honest number with drop-both (+0.0392) as a robustness row. Keep 2KVS in the benchmark.

**CAVEATS:** 1. The fix treats the source ddG_ML column and the construct name as ground truth for which row is WT. If the MegaScale release itself mislabels 2K5H, the fix inherits that error. The +3.0123 offset with r=0.9967 against an independent column makes this unlikely but it is not independently confirmed.
2. Fixing the reference re-references the PREDICTIONS too, so this is a re-scoring of existing predictions, not a re-training. Whether a model TRAINED with corrected 2K5H labels behaves differently is unmeasured and needs a GPU run.
3. The "under-represented topology" explanation for 2KVS (high burial z=+1.88, low contact order z=-1.12, 95.5% helical, fewest charged residues z=-1.36) is a HYPOTHESIS, not a measured result. At n=28, |r|<0.374 is indistinguishable from zero at p=0.05, so it cannot be tested on this benchmark. It is a generalisation claim requiring a larger held-out set.
4. This does NOT overturn the calibration story: after the fix std(b_p)=0.1348 still, the oracle still gains +0.0611, ICC(b_p)=0.898 was measured across seeds and is unaffected by a constant per-protein shift, and the mean_rel_SASA<->a_p (r=+0.714) and frac_buried<->b_p findings are untouched by 2K5H's origin. The MAGNITUDE was inflated roughly two-fold, not the direction.
5. It STRENGTHENS the corrector negative result: the corrector was being asked to predict b_p(2K5H)=-0.72 from structure. No structural feature could ever have predicted it because it was not structural — it was a mis-chosen row. The corrector's failure was correct behaviour.
6. 28,172 of 28,314 eval rows matched a single-point mutation regex; the 142 unmatched are multi-point (':') and wt rows, excluded only from the per-mutation stratification, never from any pooled PCC.
7. Gate verified: scripts/gate_g4_cpu.py prints "baseline ... PASS dG=-0.0030 width=1092" and "G4-CPU: ALL PASS". No tracked repo file was modified (the one "M scripts/autopilot.py" is pre-existing and was never touched); all work is in the new untracked twoprot/ directory plus results/TWO_PROTEINS.md.

**GPU HOURS:** 0 — all work was CPU re-scoring of the 10 existing eval CSVs plus Biopython SASA on 28 small PDBs; total cluster time under 10 minutes. The recommended fix and downstream recomputation are also CPU-only. The one open question that DOES need GPU is whether a model retrained with corrected 2K5H labels behaves differently (~1 training run); that is optional and not required for the correction.

**FILES:**
- /home/nissimb/DeepPEF/results/TWO_PROTEINS.md
- /home/nissimb/DeepPEF/twoprot/recon.py
- /home/nissimb/DeepPEF/twoprot/analyse.py
- /home/nissimb/DeepPEF/twoprot/charac.py
- /home/nissimb/DeepPEF/twoprot/wtref.py
- /home/nissimb/DeepPEF/twoprot/wtsrc.py
- /home/nissimb/DeepPEF/twoprot/fixref.py
- /home/nissimb/DeepPEF/twoprot/kvs.py
- /home/nissimb/DeepPEF/twoprot/permut.csv
- /home/nissimb/DeepPEF/twoprot/fixref.csv
- /home/nissimb/DeepPEF/twoprot/robustness.csv
- /home/nissimb/DeepPEF/twoprot/src_vs_our_ddg.csv
- /home/nissimb/DeepPEF/twoprot/wt_audit_all.csv
- /home/nissimb/DeepPEF/twoprot/structural.csv
- /home/nissimb/DeepPEF/twoprot/wtref.csv
- /home/nissimb/DeepPEF/twoprot/medref.csv
- /home/nissimb/DeepPEF/twoprot/perprotein_primary.csv
- /home/nissimb/DeepPEF/twoprot/pos_2K5H.csv
- /home/nissimb/DeepPEF/twoprot/pos_2KVS.csv
- /home/nissimb/DeepPEF/twoprot/bp_vs_ddgdist.csv
- C:\Users\User\AppData\Local\Temp\claude\C--Users-User\eee8e8e1-58ee-4bbf-8e95-538e7a9fb0ba\scratchpad\twoprot\TWO_PROTEINS.md

====================================================================================================
# WORKFLOW: DATA-OPTIMIZATION  (5 results)
====================================================================================================

## data/FINAL_DATASET_100k_030926.csv (the 100k PDB pre-training decoy catalogue), joined/compared against the 340 training + 28 test MegaScale proteins 
**Verdict:** NEGATIVE

**HEADLINE:** The catalogue can never be joined to our data - it has no sequence column at all, and 0/226 real PDB ids in the TRAINING set match - but it is the only record of what pre-training saw, and it says our entire problem domain was 0.1206% of pre-training residues. Both attempts to turn that shift into an explanation of b_p came back null, and Ofir's train/test mean-gap mechanism is dead: the gap is -0.103 kcal/mol (p=0.588) against std(b_p)=1.5741.

**NUMBERS:** JOIN (Q1) - DEAD. Traced train.py L1198 tensor_root_dir, L245 TM_PATH, L388 os.listdir, L392-395 test/train split: 368 dirs -> 28 test, 340 train, all 368 with a real mutation CSV. 226 of the 340 train proteins are genuine 4-char PDB ids (114 are Rocklin/Baker designed names). Catalogue holds 66,945 unique PDB ids (~1/3 of the PDB). Hits: 0/226 by PDB id, 0/21 test PDB-like, 0/340 by full protein_id. SEQUENCE JOIN IS IMPOSSIBLE, not merely empty: the file's 21 columns contain no sequence field (verified against the actual header; catalogue_has_no_sequence_column=true). So "structural properties predict TRAINING error at large n" is dead - it required the join.

DISTRIBUTION SHIFT (Q2) - THE ONE LIVE USE, needs no join. Catalogue length: n=99,708, median 477, mean 1167.7, max 89,160. Ours: train 340 proteins 32-74 aa (median 55), test 28 proteins 43-72 aa (median 56). Catalogue mass at/below our longest (74 aa) = 3.05%; inside our full 32-74 range = 2.53% (2,526 rows). THE HEADLINE - gradient share, since pre-training loss is per-residue: our regime supplied 140,383 of 116,431,172 residues = 0.1206%, i.e. one residue in 830. The short tail is also a different KIND of structure: NMR 8.56% overall vs 77.43% in our regime (9.05x enriched); Is Complex? Yes 67.7% vs 11.8%; Monomer 31.8% vs 88.2%; BSA=0 31.2% vs 87.8%; mean BSA 13.02 vs 2.64. MegaScale-like stratum (Monomer AND not complex AND ligand-free) = 10.33% of catalogue, and only 1.64% when also 32-74 aa. Composition train(340) vs test(28), Welch, Bonferroni 0.05/20=0.0025: Threonine 0.0660 -> 0.0395, p=2.73e-07 SURVIVES; M p=0.0113 nominal only; all other 18 > 0.07.

MEAN GAP (Q3) - HONEST NEGATIVE. Per-protein WT dG: train 3.0930+/-1.3362 (n=340) vs test 2.9895+/-0.9263 (n=28), gap = -0.1034, Cohen d=-0.090, Welch p=0.588, MWU p=0.992. Mean dG over variants: gap -0.0797, p=0.683. Pooled over mutations (670,565 train rows vs 63,380 test rows): train 1.7387, test 1.3632, gap -0.3756. Against std(b_p)=1.5741 the per-protein gap is 6.6% of one SD (24% for the generous pooled figure) - and a shared constant cannot generate per-protein DISPERSION anyway. Ofir's mechanism does not carry over.

b_p LINKAGE (Q4) - NULL. Fitted pred_deltaG ~ a_p*deltaG + b_p per protein per checkpoint over all 10 eval CSVs. n=28, threshold |r|>0.374 at p=0.05. Catalogue length-percentile vs b_p r=+0.0427 (p=0.829); Mahalanobis distance from the 340-protein composition centroid r=+0.0277 (p=0.889); L2 distance r=+0.1904 (p=0.332); f_T (the Bonferroni-surviving composition hit) r=-0.1189 (p=0.547). Across m=25 tests, nominal hits were f_H -0.522/0.0044, f_R -0.448/0.0169, f_K +0.390/0.0400 (b_p) and f_F -0.510/0.0056, wt_dG -0.421/0.0259 (a_p); under Bonferroni (alpha=0.0020) NONE survive and under BH-FDR 5% NONE survive, for either b_p or a_p. Expected ~1.25 nominal hits by chance, observed 3.

**NEXT ACTION:** Quote the 0.1206% residue-share and the 9.05x NMR / 5.7x complex-depletion table in the thesis as a standalone distribution-shift characterisation of pre-training - it is well-powered (n=99,708) and needs no join - but write it as a SHIFT, never as the cause of b_p, since r=+0.043 against b_p. Formally retire the catalogue for every join-based purpose on the ground that no sequence column exists (this is a permanent, in-principle result, not a failed attempt). Retire Ofir's mean-gap hypothesis for b_p on the -0.103/p=0.588 result. b_p must now be hunted on the input/representation side, not the label side; the 2b table (interface-rich X-ray pre-training vs interface-free monomeric test) is the concrete pointer and is consistent with BSA and ligands having already registered as a_p levers.

**CAVEATS:** GATE: scripts/gate_g4_cpu.py run after all work prints "baseline forward ok, dG finite PASS dG=-0.0030 width=1092" and "G4-CPU: ALL PASS" (saved at scratch_cat/gate.txt). No repo file was modified - all new files are under scratch_cat/ plus results/CATALOGUE_USES.md. The pre-existing "M scripts/autopilot.py" in git status predates this task. Nothing was committed or pushed; three CPU-only SLURM jobs (21085626, 21085645, 21085673), no GPU.

CONVENTION MISMATCH, stated openly: my per-protein regression over the 10 eval CSVs, averaged across checkpoints, gives std(b_p)=0.7914 and median a_p=0.4148, versus the briefed std(b_p)=1.5741 and median a_p=0.4990. Averaging slopes/intercepts across checkpoints shrinks the spread, so the two are not directly comparable in scale. Only the CORRELATIONS are used for the section-4 conclusion and those are scale-free, so the null result stands regardless; but the 6.6%-of-one-SD comparison in section 3 deliberately uses the BRIEFED 1.5741 to avoid flattering the negative.

"AA Length" in the catalogue is the entity/assembly length, not per-chain, so the 0.1206% residue share is computed on assembly residues. This makes the shift estimate CONSERVATIVE in the direction that matters least - per-chain lengths would be smaller, moving catalogue mass slightly toward our regime - so the honest reading is that 0.1206% is a lower bound on the mismatch severity, not an exact per-chain figure. The 32-74 aa regime characterisation (NMR/complex/BSA) is unaffected since those are entity-level attributes anyway.

The 2b interface/method story is a plausible mechanism for b_p that I did NOT demonstrate - the two direct tests I could run against b_p were null. It should be presented as a hypothesis consistent with prior BSA/ligand lever hits, not as an established cause. Section 2d's Threonine result is a genuine train/test asymmetry (p=2.7e-07) but carries no predictive power over b_p (r=-0.119), so it is a descriptive fact, not a lever.

**FILES:**
- C:\Users\User\AppData\Local\Temp\claude\C--Users-User\eee8e8e1-58ee-4bbf-8e95-538e7a9fb0ba\scratchpad\results\CATALOGUE_USES.md
- /home/nissimb/DeepPEF/results/CATALOGUE_USES.md
- /home/nissimb/DeepPEF/scratch_cat/cat_analysis.py
- /home/nissimb/DeepPEF/scratch_cat/cat_out.json
- /home/nissimb/DeepPEF/scratch_cat/cat_perprotein.csv
- /home/nissimb/DeepPEF/scratch_cat/cat_comp.csv
- /home/nissimb/DeepPEF/scratch_cat/shift.py
- /home/nissimb/DeepPEF/scratch_cat/shift.json
- /home/nissimb/DeepPEF/scratch_cat/bp_link.py
- /home/nissimb/DeepPEF/scratch_cat/bp_link.json
- /home/nissimb/DeepPEF/scratch_cat/bp_features.csv
- /home/nissimb/DeepPEF/scratch_cat/gate.txt

## MegaScale raw data on BGU cluster: /groups/keasar_group/casp15/Shahar/DeepPEF/data (repo data/ is symlinks into it). Key files: Processed_K50_dG_datas
**Verdict:** PROMISING

**HEADLINE:** Per-mutation uncertainty for all 28 test proteins IS available (in ThermoMPNN/mega_test.csv, not the pnas file the earlier attempt searched) — and it kills the attenuation explanation of a_p: EIV correction factor is 1.0015. Separately, 26,315 double mutants for our 28 sit in no split at all.

**NUMBERS:** UNCERTAINTY JOIN (SOLVED): mega_test.csv matches 28/28 by name and 28/28 by exact WT sequence; per-protein counts match 27/28 exactly (2K5H 1123 vs 1125). 100% non-null on deltaG, deltaG_95CI, fitting_error_t, ddG_ML. Row join on protein+round(deltaG,5): 28,021/28,314 = 99.0% matched, 64 ambiguous keys; after dropping ambiguous, 27,895 rows (98.5%) carry a sigma.

PNAS NEGATIVE (PROVEN, not assumed): pnas_mutations.csv 0/28 by exact name, by PDB-id normalisation, by exact WT seq, by substring both directions, AND by nearest-neighbour difflib over all 270 WTs — max ratio 0.741 (1TUC vs 1PWT), zero >= 0.95. The 28 are genuinely held out from that file.

ERRORS-IN-VARIABLES (the payoff): sigma = deltaG_95CI/(2*1.96); median sigma = 0.0372 kcal/mol vs std(deltaG) = 1.3275. Label noise = 0.30% of var(deltaG) (max 0.86% on any protein). a_eiv = a_ols*var(x)/(var(x)-mean(sigma^2)) over all 10 checkpoints (n=27,895): a_ols median 0.4088 -> a_eiv median 0.4106, correction factor 1.0015 (0.15% on a_p median 0.4990). Reference ckpt abl_calib_ctrl_repro2_e14: 0.4969 -> 0.4976.

MULTI-POINT (28 proteins, raw 63,380 rows vs 28,314 evaluated = 44.7% used): single 31,873, multi 26,315, ins 3,366, del 1,676, wt 150. All multi are exactly double (0 triples). 20/28 have doubles (3DKM 7,448; 1QP2 3,600; 4C26 2,285; 2K28 1,884; 2KVS 1,601; 1QKH 1,483). All 26,315 have finite measured deltaG. Confirmed in NO split: mega_test multi=0, mega_train multi=0, ins/del=0 in both. Additivity baseline: 23,652/26,315 (89.9%) decompose into two measured singles; r(additive,observed) = 0.7108 weighted; mean epistasis +1.235 kcal/mol, weighted sd 1.311; per-protein r 0.34 (2WXC) to 0.91 (6EWT, 2KXD).

TRAINING FRACTION (all 862 proteins, 733,945 rows): single 451,514 (61.5%), multi 210,118 (28.6%), ins 46,805 (6.4%), del 23,359 (3.2%), wt 2,149 (0.3%). Splits total 216,919+27,481+28,312 = 272,712 = 37.2%. Excluded 62.8% = 461,233 rows. 862 mutation_datasets CSVs vs only 368 training_data tensor dirs = 494 proteins with labels but no tensors.

GATE: scripts/gate_g4_cpu.py prints dG=-0.0030 width=1092, G4-CPU: ALL PASS (read-only work, no code changed).

**NEXT ACTION:** Report the EIV negative in the thesis immediately: a_ols 0.4088 -> a_eiv 0.4106 (factor 1.0015) across 10 checkpoints proves label measurement error explains essentially none of a_p ~= 0.50, so the regression-dilution hypothesis is dead and a_p is a real model property (consistent with exposure -> a_p r=+0.714 on 10/10). Then run the double-mutant generalisation test: 26,315 held-out doubles on the 28, tensors already present for all 28, no new labels or preprocessing, with the additivity baseline r=0.711 / epistasis +1.235 kcal/mol already established as the number to beat. Note the loader at data_loaders/dg_data_loader.py:34 drops only ins|del, so doubles need no filter change — they were removed upstream when the ThermoMPNN split was built.

**CAVEATS:** 1) The eval-to-mega_test join is a VALUE join on rounded deltaG, not a key join: 419/28,314 rows (1.5%) dropped as ambiguous at 5 dp. mega_test.csv does carry mut_type, so regenerating the eval CSVs to include mut_type would give an exact key and remove this caveat entirely — worth doing before the numbers go in the thesis.

2) Unexplained 2-row discrepancy: eval CSVs have 28,314 rows, mega_test has 28,312, entirely from 2K5H (1,125 vs 1,123). Small, but resolve it before quoting exact per-protein n.

3) ddG_ML in mutation_datasets/*.csv is STRING-typed with '-' as a sentinel for unreliable values (2,033 of 9,037 for 3DKM). Naive numeric use silently fails or crashes. Always pd.to_numeric(..., errors='coerce'). deltaG is fully numeric — my additivity analysis uses deltaG minus WT, not ddG_ML.

4) Epistasis mean uses a WT reference = mean of the per-protein 'wt' rows; a different reference convention shifts the +1.235 mean but not the r=0.711.

5) The EIV correction is small BECAUSE measured label noise is small (lambda=0.003) — this is a genuine negative result about attenuation, not a failed measurement. It does not rule out other explanations for a_p; it rules out exactly one (regression dilution from label noise).

6) Sigma-weighted metrics are now possible but at lambda=0.003 should move pooled PCC by <0.01 — report as robustness, do not oversell as a lever.

7) Insertions/deletions (70,164 rows across all 862) are NOT usable without an architecture change: they alter sequence length and the loader explicitly drops them.

8) I did not train anything or verify that a model actually predicts doubles well — the r=0.711 additivity figure is a data-side baseline, not a model result.

**FILES:**
- C:\Users\User\AppData\Local\Temp\claude\C--Users-User\eee8e8e1-58ee-4bbf-8e95-538e7a9fb0ba\scratchpad\MEGASCALE_RAW.md
- /home/nissimb/DeepPEF/results/MEGASCALE_RAW.md
- /groups/keasar_group/casp15/Shahar/DeepPEF/data/ThermoMPNN/mega_test.csv
- /groups/keasar_group/casp15/Shahar/DeepPEF/data/Processed_K50_dG_datasets/mutation_datasets/
- /groups/keasar_group/casp15/Shahar/DeepPEF/data/Processed_K50_dG_datasets/training_data/
- /home/nissimb/DeepPEF/data_loaders/dg_data_loader.py

## Ofir Ezrielev M.Sc. thesis (C:/Users/User/Downloads/Thesis Ofir Ezrielev.pdf, 56pp, read directly via extracted text) + our 10-11 eval CSVs, frozen ch
**Verdict:** PROMISING

**HEADLINE:** Ofir's leave-one-AA-out offset identity replicates on our data at r=+0.9895 (n=28, his +0.9883 at n=18), and his remedy — MEASURE the offset from a few samples instead of PREDICTING it — lifts pooled dG PCC 0.369 -> 0.753 with 3 measured mutations per protein (oracle 0.817), all three negative controls passing. Separately: enrichment, never computed here before, shows the model is 10.37x chance at finding destabilizing mutations but only 4.69x at stabilizing ones, and corr(ddG PCC, top-5% enrichment) across 10 checkpoints is +0.0021 — we have been selecting checkpoints on a metric carrying zero information about the design use case.

**NUMBERS:** ENRICHMENT (formula p.21: |argmax_k(y) ∩ argmax_k(yhat)| / (N*(k/N)^2); chance=1.0, max=N/k). Null validated: shuffle gives E5top=1.006, E5bot=0.998 vs theory 1.000. Per-protein (28 proteins x 10 ckpts, 28,285 non-WT rows): E5bot=10.37 (hit rate 51.4% vs 5.0% chance, 1.9 screened/hit); E5top=4.69 (23.2%, 4.3 screened/hit); E15bot=4.45; E15top=2.48. 97.1% of cells beat chance on E5top, 100% on E5bot, zero total misses. Budget: 10 true top-5% stabilizers需 43 model-ranked vs 202 random = 4.7x saving. Correlates (n=28, thr 0.392): a_p->E5top +0.490 (FIFTH a_p lever hit); ddG skew->E5bot -0.567 (reproduces his CP2-vs-PUMA distribution-shape explanation pp.32-33); PCC->E15bot +0.938 but PCC->E5top only +0.431. Checkpoint selection (n=10, thr 0.632): corr(pooled ddG PCC, E5top)=+0.0021, corr(...,E5bot)=+0.2852; best-by-PCC (sigma_seed2_e10) != best-by-E5top (sigma_seed3_e13). Offset damage: removing b_p multiplies pooled dG bottom-5% enrichment 4.06->9.55 (2.35x).
SEQUENCE ABLATION (7 modes, 28,314 rows each, PATCH_CALLS=28314 verified every mode). Intact: pooled ddG PCC 0.5912 / per-prot 0.7310 / E5top 4.99 / E5bot 10.51 / med a_p 0.714 / sd(pred) 1.068. Permuted (residue axis of one_hot+prott5, coords+mask fixed, same perm for WT and mutant): pooled 0.6047 (102% retained), per-prot 0.6668 (91%), E5top 4.77 (96%), E5bot 9.69 (92%), RMSE -3%; second seed s7 confirms (0.5945/0.6673/4.74/9.34). Permute one_hot only: 0.7266 per-prot (99.5% — nearly free). Permute prott5 only: 0.6709 (reproduces nearly the whole effect). Collapse prott5 to sequence mean: per-prot 0.5297 (72%), E5top 3.23 (65%), E5bot 7.24 (69%), RMSE +82%, sd(pred) 2.05. Collapse one_hot to mean composition: 0.7265 (no cost). Median a_p drops 0.714->0.554-0.593 under permutation but stays 0.714 under collapse.
OFIR IDENTITY + FEW-SHOT. His Table 3.4 recomputed (n=18, thr 0.500): corr(RMSE,|Δmean|)=+0.9883, r flat 0.858-0.948 while RMSE swings 2.74x; de-offsetting in quadrature collapses spread to 1.71x and corr to +0.2337. Ours (n=28, thr 0.392): corr(RMSE_dG,|mean gap|)=+0.9895; offset = 80.3% of dG MSE; RMSE range 9.08x -> 3.01x de-offset. Few-shot measured b_p (k points excluded from scoring), mean over 11 ckpts: raw 0.369, k=1 0.680, k=3 0.753, k=5 0.794, k=10 0.796-0.813, k=20 0.822, oracle 0.817-0.831. Controls all pass: random constant same magnitude 0.266 (<raw, hurts); swapped protein offsets 0.245 (<<raw); ddG PCC under k=3 = 0.557 = untouched 0.557 (b_p cancels in ddG, exactly the required no-op per the metric rule).

**NEXT ACTION:** Adopt few-shot measured-offset calibration as the stated deployment protocol for new proteins (measure 3-10 mutations, subtract mean residual: dG PCC ~0.39 -> ~0.75-0.81) — no retraining, and it reframes the failed LOPO b_p regression honestly: the offset is not a function of the protein features we have, but it is trivially measurable. Then add E5top/E5bot to validation/score_runs.py and re-rank the 361 checkpoints by top-5% enrichment rather than PCC, since the two are uncorrelated (+0.0021, n=10). Untested lever worth a GPU cell: Ofir p.17 uses batch normalization deliberately to suppress the shared-WT component with protein-homogeneous batches — a direct architectural attack on b_p we have never tried.

**CAVEATS:** (1) The sequence permutation is applied at INFERENCE on a frozen checkpoint. Ofir also RETRAINED on permuted data ("Permuted Model" column of his Table 3.2); we did not — that needs GPU training. So the honest claim is "the trained model does not use sequence order at inference", not "sequence order is unlearnable here". (2) The ablation reproduction gate initially read FAIL for two separate reasons I had to chase down and both are benign: row ORDER differs because the dataloader shuffles proteins (fixed by comparing on a (protein, within-protein index) key — deltaG and ddG then match at exactly 0.000e+00), and pred_deltaG differs because the shipped CSVs have an affine.json sidecar applied that my runner omits. I recovered that map empirically from my raw predictions as a=0.697576 b=3.983037 vs the sidecar's a=0.6975855 b=3.9830644, corr=0.99999878 — agreement to 5 decimals, residual 1.5e-2 being float32 accumulation. The affine is a pure monotone rescale so it cannot move any enrichment or PCC number in the ablation table, but every ablation figure quoted is on RAW (un-affined) predictions and is therefore internally consistent but NOT directly comparable row-for-row against the shipped eval_results CSVs. (3) The few-shot k-sweep used 60 draws in one script and 30 in the verification script, hence two slightly different figures quoted at k=0/10 (0.393/0.369 and 0.813/0.796); the gap is draw noise, not disagreement. (4) The few-shot result is measured on the SAME 28 test proteins with no held-out protein set — it is a calibration protocol, not a generalization claim, and its validity rests on the k calibration points being excluded from scoring (they are) plus the three controls. (5) n=28 throughout for per-protein correlates, so a_p->E5top at +0.490 clears the 0.392 threshold but not by a wide margin; E5top->n_mut at -0.397 is right at the threshold and I would not lean on it. Across-checkpoint claims are n=10 (threshold 0.632), so the +0.0021 decoupling is a strong null but the +0.2852 for E5bot is simply indeterminate at that n, not evidence of a relationship. (6) Ofir's Table 3.4 numbers were transcribed by hand from the PDF text extraction into loao_check.py; I did not re-derive them from his data (which we do not have). (7) scripts/block_ablation.py shows as git-modified on the cluster with mtime 2026-09-08 00:30, after my last write at 22:49 — that is a parallel workstream, not mine; I edited no tracked source, pushed nothing, submitted no GPU job and cancelled nothing. (8) gate_g4_cpu.py re-run after all work: "baseline ... PASS dG=-0.0030 width=1092", G4-CPU ALL PASS.

**FILES:**
- C:/Users/User/AppData/Local/Temp/claude/C--Users-User/eee8e8e1-58ee-4bbf-8e95-538e7a9fb0ba/scratchpad/results/OFIR_TRANSFER.md
- C:/Users/User/AppData/Local/Temp/claude/C--Users-User/eee8e8e1-58ee-4bbf-8e95-538e7a9fb0ba/scratchpad/loao_check.py
- /home/nissimb/DeepPEF/results/OFIR_TRANSFER.md
- /home/nissimb/DeepPEF/scripts/enrich.py
- /home/nissimb/DeepPEF/scripts/enrich2.py
- /home/nissimb/DeepPEF/scripts/enrich3.py
- /home/nissimb/DeepPEF/scripts/seqabl.py
- /home/nissimb/DeepPEF/scripts/seqabl_job.sh
- /home/nissimb/DeepPEF/scripts/seqabl_an.py
- /home/nissimb/DeepPEF/scripts/gate_none.py
- /home/nissimb/DeepPEF/scripts/gate_none2.py
- /home/nissimb/DeepPEF/scripts/fewshot.py
- /home/nissimb/DeepPEF/scripts/fewshot_vfy.py
- /home/nissimb/DeepPEF/results/enrichment_pooled.csv
- /home/nissimb/DeepPEF/results/enrichment_perprotein.csv
- /home/nissimb/DeepPEF/results/enrichment_offset.csv
- /home/nissimb/DeepPEF/results/enrichment_bp.csv
- /home/nissimb/DeepPEF/results/enrichment_hitrate.csv
- /home/nissimb/DeepPEF/results/seqabl_summary.csv
- /home/nissimb/DeepPEF/results/ofir_identity.csv
- /home/nissimb/DeepPEF/results/fewshot_calib.csv
- /home/nissimb/DeepPEF/results/seqabl/abl_none_s42.csv
- /home/nissimb/DeepPEF/results/seqabl/abl_perm_s42.csv
- /home/nissimb/DeepPEF/results/seqabl/abl_perm_s7.csv
- /home/nissimb/DeepPEF/results/seqabl/abl_perm_emb_s42.csv
- /home/nissimb/DeepPEF/results/seqabl/abl_perm_oh_s42.csv
- /home/nissimb/DeepPEF/results/seqabl/abl_collapse_s42.csv
- /home/nissimb/DeepPEF/results/seqabl/abl_collapse_oh_s42.csv

## Cluster accounting (`sacct -u nissimb`, 163 jobs, 2026-09-05T19:09→2026-09-08T00:21) + the 22 eval CSVs in `/home/nissimb/DeepPEF/eval_results/` + val
**Verdict:** PROMISING

**HEADLINE:** The factorial is scored on the wrong metric: on pooled ddG PCC it provably cannot resolve A/B/C (would need k=12 seeds = 1,708 GPU-h), while factor D is already decided at |z|=29 from one seed — making ~147 of the 286 queued GPU-hours uninformative.

**NUMBERS:** SPEND: 287.10 GPU-h burned in 52.9 wall-h (COMPLETED 235.95 / RUNNING-so-far 50.73 / CANCELLED 0.38 / FAILED 0.04). Factorial trains 166.74, other trains 84.14, golden probes 35.10. Mean train elapsed 8.6 h (n=26, 6:21–14:41). REMAINING: 285.8 GPU-h (D=1 arm 137.6, D=0 arm 111.8, running remainder 17.2, side arms 17.2, all 48 evals 1.9).

SEED NOISE (5 abl_sigma replicates = cell A0/B1/C0/D0, df=4): sd_seed pooled ddG PCC 0.0302 [CI 0.018–0.087]; mean per-prot PCC 0.0069; median a_p 0.0515; std(b_p) 0.1369; dG PCC 0.0470. Decomposition: across-seed sd at fixed epoch (e8–14) 0.0285 vs within-seed epoch jitter 0.0135 — noise is seed-driven, ~2x epoch jitter. Chosen epoch wanders e5–e14.

POWER (SE = sd_seed*sqrt(2/(8k))): pooled MDE80 = 0.0319 (k=2), 0.0252 (k=3). median a_p MDE80 = 0.0543 (k=2), 0.0430 (k=3).

EFFECTS, D=0 subset (honest, n=5): pooled A=-0.0120 B=-0.0040 C=-0.0115 → ALL below MDE80 0.0252. median a_p A=+0.2962 C=+0.1153 → both DETECTABLE. Required k to detect pooled 0.0120 = 12 seeds/cell = 1,708 GPU-h.

FACTOR D: all 5 D=1 cells pooled PCC 0.10–0.31 (one −0.078); all 5 D=0 cells 0.583–0.618. Complete separation, gap 0.443 = 15 seed-sigmas, |z|=29 at k=1 (still |z|=10 at the upper-95% sd bound). A×D interaction −0.147 (A effect −0.004 at D=0 vs −0.298 at D=1).

HALF-FRACTION: 2^(4-1), I=ABCD, res IV — all 4 main effects clean. Half at 6 seeds = full at 3 seeds: 48 runs, 413 GPU-h, SE 0.0087 IDENTICAL. Full design's only premium is separating AB/CD, AC/BD, AD/BC.

CORRECTIONS: eval is 0.040 GPU-h (145 s, n=17), not 0.5. Golden lane is held by US (8 gld_* running), not a labmate. Public cap confirmed gres/gpu=5.

**NEXT ACTION:** Two zero-GPU changes first, before any more cells land: (1) score the factorial on median a_p / std(b_p) instead of pooled ddG PCC — on pooled the design cannot resolve A/B/C at any affordable k, on a_p it already does; (2) replace best-val-epoch selection with a fixed epoch or mean(e10–e14), removing 0.0135 sd for free using the 513 checkpoints already on disk. Then recommend (do not execute) scancel of the 16 pending DeepEF_p3_*_D1_uemb_seed{1,2} trainings + 24 dependent evals (~138 GPU-h) and sa_coil (8.6 GPU-h, redundant); keep sa_uembmean (the one unfalsified D variant). Resubmit all 48 evals with --partition=cpu (evaluate.py:63 already has the CPU fallback) so 2.4-minute jobs stop holding 1 of only 5 RTX-6000 slots ahead of 8.6 h trainings.

**CAVEATS:** sd_seed rests on n=5 replicates (df=4); its own 95% CI spans a 2.87x range. I re-ran every power claim at the pessimistic upper bound and both conclusions survive (D stays |z|=10 at k=1; A/B/C on pooled stay below MDE at 0.0724). The 10 cell-level effect estimates are UNREPLICATED (one seed each) and the observed subset is non-orthogonal — corr(A,D) = −0.6, which is why the marginal A effect on pooled (+0.169) flips sign to −0.160 once D is adjusted for. That is why I report the D=0-subset effects as the honest ones; A/B/C numbers will move when the pending seeds land. The D conclusion does NOT depend on this — it is complete separation, not a contrast. abl_sigma replicates only cell A0/B1/C0/D0, so homogeneous seed variance across cells is an assumption; the D=1 cells are visibly more dispersed (pooled range 0.39 vs 0.035), which strengthens rather than weakens the case for cancelling that arm. n=28 proteins throughout, |r|<0.374 indistinguishable from zero at p=0.05 — no claim here rests near that threshold.

VERIFICATION / process notes: I hit the exact signature failure the brief warns about — `put.py` printed "uploaded ... -> results/COMPUTE_OPTIMISATION.md" but SFTP is home-relative, so the file went to /home/nissimb/results/, NOT /home/nissimb/DeepPEF/results/. Caught it by listing the artifact rather than trusting the success line; moved it and re-verified by md5sum plus grepping for seven specific numbers. Also removed a pre-existing landmine, /tmp/inspect.py, which shadowed the stdlib and broke any numpy/pandas import run from /tmp (not mine; it silently breaks other jobs). gate_g4_cpu.py re-run after all work: PASS dG=-0.0030 width=1092, ALL PASS. Nothing cancelled, killed, sbatched or pushed — queue still 98 jobs; all §4 items are recommendations only.

Two counts in the brief are stale: 513 checkpoints on disk (not 361) and 22 eval CSVs (not 10).

**FILES:**
- C:\Users\User\AppData\Local\Temp\claude\C--Users-User\eee8e8e1-58ee-4bbf-8e95-538e7a9fb0ba\scratchpad\results\COMPUTE_OPTIMISATION.md
- /home/nissimb/DeepPEF/results/COMPUTE_OPTIMISATION.md
- /home/nissimb/DeepPEF/analysis_co/reg2.py
- /home/nissimb/DeepPEF/analysis_co/design.py
- /home/nissimb/DeepPEF/analysis_co/valan.py
- /home/nissimb/DeepPEF/analysis_co/dint.py
- /home/nissimb/DeepPEF/analysis_co/robust.py
- /home/nissimb/DeepPEF/analysis_co/alloc2.py
- /home/nissimb/DeepPEF/analysis_co/ep.py
- /home/nissimb/DeepPEF/analysis_co/alloc.py

## ?

====================================================================================================
# WORKFLOW: INTEGRATION-DESIGN  (7 results)
====================================================================================================

## How chemistry enters the model: the aa_descriptors / Mordred block (W6)
**Verdict:** MAYBE

**WHAT WE SEE:** data/aa_descriptors_mordred.csv is a 20 x 726 float matrix, row-keyed by one-letter code in AA_MAP order ACDEFGHIKLMNPQRSTVWY, so row i IS one-hot index i. Provenance header records the full pipeline: 1613 Mordred descriptors registered -> 1278 after dropping any column with a missing value (-335) -> 726 after nunique>=15 of 20 (-552). Inputs are PubChem canonical isomeric SMILES for NEUTRAL FREE L-amino acids (not zwitterionic), smiles_sha256_16=b3f4304c226f896f, Calculator(descriptors, ignore_3D=True), rdkit 2025.09.2 / mordred 1.2.0. Histidine was corrected against the PHASES table (which encoded a pyridine-type ring C7H10N2O2 instead of imidazol-4-yl C6H9N3O2); an RDKit formula + stereocentre gate enforces 20/20.

SCALES: settled. Normalisation is a z-score down each column across the 20 residues (ddof=0). Verified live: column means are 1.35e-10 and EVERY column std is exactly 1.0000. There is no scale heterogeneity left to fix.

WHAT THE COLUMNS ARE: the names expose the redundancy. SpAbs_A, SpMax_A, SpDiam_A, SpAD_A, SpMAD_A, LogEE_A, VE1_A, VE2_A, VE3_A, VR1_A... are all functionals of the SAME adjacency-matrix eigenspectrum. ATS0dv..ATS4dv, ATS0d..ATS5d, ATS0s..ATS6s, ATS0Z..ATS6Z, ATS0m..ATS6m, ATS0v.. are Moreau-Broto autocorrelations of the same molecular graph at lags 0-6 weighted by six atomic properties (valence-electron count, degree, intrinsic state, atomic number, mass, volume). These are ~a dozen underlying graph quantities read out through many near-equivalent functionals.

CORRELATION STRUCTURE (measured over 263,175 column pairs): mean |r|=0.423, median |r|=0.377, frac|r|>0.90 = 8.3%, frac|r|>0.99 = 0.7%. Average-linkage clustering on 1-|r|: 278 clusters at |r|>0.95, 194 at |r|>0.90, 109 at |r|>0.80.

EIGENVALUE SPECTRUM (SVD of column-centred 20x726): RANK = 19 EXACTLY (singular values beyond 19 at machine zero). Variance fractions PC1=0.4627 PC2=0.1775 PC3=0.0814 PC4=0.0524 PC5=0.0517 PC6=0.0431 PC7=0.0313 PC8=0.0244 PC9=0.0178 PC10=0.0145 PC11=0.0101 PC12=0.0083 PC13=0.0067 PC14=0.0062 PC15=0.0046 PC16=0.0035 PC17=0.0020 PC18=0.0017 PC19=0.0002. Cumulative: PC1..3=0.7216, PC1..7=0.9001, PC1..10=0.9568, PC1..16=0.9962. Participation ratio (sum L)^2/sum(L^2) = 3.82; exp(spectral entropy) = 6.45.

THE COMPONENTS ARE CHEMICALLY LEGIBLE (so the matrix is not noise): PC1 r(volume)=+0.966, r(Kyte-Doolittle)=-0.152 -> PC1 IS SIZE. PC2 r(KD)=+0.695, r(volume)=+0.172 -> PC2 IS HYDROPHOBICITY. Together 64% of variance -- the same two axes every AA property scale since Kyte-Doolittle 1982 recovers.

WHAT 726-ON-20 MEANS: 14,520 entries but at most 20*19=380 degrees of freedom and effectively ~4-6. 726 is NOT a dimensionality, it is a REDUNDANCY FACTOR of ~38-190x. "The block adds 726 features" is false; it adds at most 19 linearly independent numbers per residue, ~7 of which carry 90% of variance. No estimation problem (it is a fixed lookup table, not fitted data) -- but a severe CONDITIONING problem, which is the whole story.

DEVIATION TO REPORT: Ofir kept nunique>=40 across 58 residues; at 20 residues 40 is unsatisfiable (max 20), so it was rescaled 40/58=0.690 -> 15/20=0.750. Defensible but OURS, and it sets how many of the 726 survive.

**WHAT CAN BE EXTRACTED:** Genuinely recoverable: ~19 linearly independent numbers per residue, effectively ~4-6 (PR=3.82), dominated by size (PC1, 46%) and hydrophobicity (PC2, 18%). A meaningful chemical METRIC on the alphabet -- measured closest pairs E->Q 12.4, D->N 13.6, F->Y 14.3, I->V 17.2, S->T 18.9, L->V 20.9, I->L 21.3 against alphabet mean 37.5; farthest G->W 82.0, A->W 71.2, G->Y 68.4. That similarity structure is real and is the ONLY thing the block can contribute.

WHAT CANNOT BE EXTRACTED -- the honest problem, stated without hedging: the block is computed as Desc = one_hot @ T with T a FIXED [20,K] table (aa_descriptors.residue_descriptors, a matmul). It is therefore a DETERMINISTIC LINEAR FUNCTION of the one-hot. Concatenating gives [one_hot | one_hot @ T] = one_hot @ [I | T], which spans EXACTLY the same column space as one_hot alone. For the first Linear (fc1_gcn/fc1_gat), whose relevant slice acts as one_hot @ W: for EVERY W_desc there exists W_oh = T @ W_desc giving IDENTICAL outputs for all inputs. The function classes are EQUAL. A model with a one-hot and a free first layer CANNOT be improved by adding descriptors -- at any width, for any K, for any descriptor set. Not "unlikely to be": cannot be, as linear algebra. This holds through the nonlinear network too, because the one-hot enters ONLY through that first Linear and everything after is shared. Descriptors do not enlarge the hypothesis space by one function. INFORMATION ADDED = ZERO.

So the case rests on exactly two things and must be argued/scored on those:
(a) REGULARISATION / inductive bias. The parameterisation changes though the function class does not. Under one-hot the 20 residue vectors are free and mutually orthogonal -- evidence about leucine tells the model NOTHING about isoleucine. Under descriptors they are tied (|L-I|=21.3 vs mean 37.5). With ~10 residues of a type per protein and 28 proteins, that sharing is the ONLY mechanism by which W6 can help. This is a variance-reduction claim and IS falsifiable.
(b) GENERALISATION to unseen residues -- Ofir's real contribution; aa_descriptors.py already implements the open alphabet (label-keyed gather, RAISES on unknown labels rather than returning zeros, since a zero row is indistinguishable from a legitimately-zero descriptor and trains to garbage without failing a shape check). But it is WORTH NOTHING ON OUR BENCHMARK: all 28 test proteins are single-chain, ligand-free, metal-free monomers 43-72aa, every residue canonical. Ofir concedes physicochemical properties are already implicit in LM embeddings, so his case narrows to non-canonical coverage -- which our evaluation set cannot express. MSE (selenomethionine) is exactly that case and is an amino acid IN the chain, not a ligand, but it is not in the 28.

Thesis sentence: W6 cannot add information; it can only add bias. We test the bias, report the result, and keep the open alphabet as the honest generalisation while stating the benchmark cannot reward it.

**HOW IT ENTERS:** LAYOUT (as built): Fh = [ D(16) | Fb(32) | S(solv_dim) | Desc(K) | <siblings> | emb(1024) | one_hot(20) ], desc_start = solv_start + solv_dim = 48 + solv_dim.

model/hydro_net.py verified line numbers: self.desc_mode=_aa_desc_mode(CFG) (l.447), self.desc_dim (l.448), self.desc_start = self.solv_start + self.solv_dim (l.449), self.lig_start = self.desc_start + self.desc_dim + _sibling_block_dims(CFG) (l.458). Right-anchored indices self.one_hot_index = -20 (l.481) and x_emb_features = x[:, self.llm_index:self.one_hot_index] (l.549) are unmoved by construction.

LAYERS THAT GROW -- ONLY these two:
  fc1_gcn: Linear(52 + solv + K + lig + proj_extra, 64)   # 52 = 32 dist + 20 one-hot  (l.462)
  fc1_gat: Linear(36 + solv + K + lig + proj_extra, 64)   # 36 = 16 dist + 20 one-hot  (l.465)
LAYERS THAT MUST NOT CHANGE (l.468-472): inst_norm1 = Normalization_layer(36 + proj_extra), inst_norm2 = Normalization_layer(2*(36+proj_extra)), fc_in_dim = 2*(36+proj_extra) + post_gnn_emb, fc1 = Linear(fc_in_dim, 128).

BOTH BRANCHES: the GCN slices x[:, :32] only, so the descriptor block must be APPENDED into _extra for both branches (l.536-537: if getattr(self,'desc_dim',0): _extra.append(x[:, self.desc_start:self.desc_start+self.desc_dim]); l.544-545 concat for gcn/gat). W6 must EXTEND that branch, never bypass it. PEMGraphTransformer slices LEFT-anchored (l.850-855) and would silently ignore any inserted block; it now RAISES -- keep it raising.

WIDTHS: mordred_pca16 1092 -> 1108; mordred726 1092 -> 1818 (+66% on the first layer, to carry <=19 independent numbers); mordred_pca16_only stays 1092 (one-hot ZEROED in place, not deleted).

THE ZEROING DETAIL (already correct, train_utils._onehot_block l.324-333, gated by keeps_one_hot at l.331): under replace the one-hot is zeroed rather than deleted because hydro_net slices emb and one_hot RIGHT-ANCHORED. Deleting 20 columns would silently re-point x[:,-20:] into the tail of ProtT5 and slide the 1024-wide LLM window left into Fb. Every shape check would still pass and every number would be wrong. Zeroing costs nothing -- a constant column is a bias the following Linear already has.

ARM C (THE RECOMMENDED CHANGE, NOT YET BUILT): ~15 lines in PEM.__init__ after fc1_gcn/fc1_gat are constructed, behind a NEW flag --aa_desc_init {none,pca16}, default none => byte-identical.
  1. T = residue_descriptors_table('mordred_pca16') -> [20,16], z-scored.
  2. R ~ N(0,1) shape [16,64], seeded from CFG.seed.
  3. W_init = T @ R -> [20,64]; rescale W_init *= (w_std / W_init.std()) where w_std is the std of the EXISTING initialised one-hot slice, so overall init scale is unchanged and only the 20-row CORRELATION STRUCTURE differs.
  4. Write into the one-hot slice of both first layers, transposed for nn.Linear's [out,in]: fc1_gcn.weight[:, 32:52] = W_init.T ; fc1_gat.weight[:, 16:36] = W_init.T. Those ranges hold when _sd=_dd=_ld=proj_extra=0; DERIVE them from self.one_hot_index and each branch's own concat order rather than hard-coding, so they stay correct when a sibling block is enabled.
  5. Biases untouched. No width change, no new parameters.
Do NOT combine Arm C with Arm A or B in one run -- C presumes the one-hot is present and carrying identity.

**OPTIMAL USE:** THE CRUX -- WHERE THE MUTATION ENTERS. ddG is a difference of two forward passes over graphs identical except at ONE residue's one-hot row, so the descriptor block changes in exactly one row and the model sees a K-dim delta at one position. Measured over all 190 residue pairs:
  one_hot   dim 20  |d| mean 1.4142 min 1.4142 max 1.4142  max/min 1.00  CV 0.000  rank 19  eff.dim(PR) 19.00
  pca16     dim 16  |d| mean 5.568  min 1.814  max 12.194  max/min 6.72  CV 0.294  rank 16  eff.dim 3.79
  mordred726 dim 726 |d| mean 37.530 min 12.359 max 81.980 max/min 6.63  CV 0.292  rank 19  eff.dim 3.82
Gram G = D^T D / n of those 190 deltas (sets per-direction curvature of a linear head):
  one_hot     lmax 0.105 lmin 0.105  cond 1.0     sqrt(cond) 1.00
  pca16       lmax 15.644 lmin 0.118 cond 132.2   sqrt(cond) 11.50
  mordred726  lmax 707.1 lmin 0.260  cond 2724.5  sqrt(cond) 52.20
ANSWER: STRICTLY WORSE, by a large factor.

(i) The one-hot mutation delta is PERFECTLY conditioned and this is not an accident: every delta is e_mut - e_wt, always norm sqrt(2), always orthogonal to unrelated pairs' deltas; the Gram is a scaled identity, cond = 1.0 EXACTLY; all 190 directions span 19 dims ISOTROPICALLY (PR = 19.00, the maximum). No representation can beat cond=1, so ANY chemical block we add can only DEGRADE it. That is the sentence the chapter was missing.
(ii) Descriptors collapse the mutation space from 19 effective directions to 3.8 while keeping nominal rank 16-19. Because PC1 (volume) and PC2 (hydrophobicity) dominate, gradient descent with one LR moves ~52x faster along "bigger/greasier" than along the other 17 directions -> the model learns a size-and-polarity rule fast and is effectively BLIND to the rest. The fine distinctions compressed into the slow directions are exactly D vs N (13.6), E vs Q (12.4), F vs Y (14.3) -- precisely the mutations where ddG is hard and interesting.
(iii) The 6.6x spread in |delta| is an implicit, UNREQUESTED reweighting of the loss. G->W arrives with |d|=82.0, I->V with 17.2; a fixed-norm head yields ~5x larger ddG for G->W BEFORE seeing data. This is a prior that conservative mutations have small ddG -- often true, which is what makes it DANGEROUS: it looks like it is working while re-encoding the size/polarity heuristic, and it fails precisely on the stabilising conservative mutations a ddG predictor exists to catch.
(iv) Why this kills CONCAT specifically: the model has both routes to the same function, cond 1 via one-hot and cond 2724 via descriptors. Gradient descent is conditioning-greedy; it fits through the one-hot route and leaves descriptor weights near init, consuming weight decay and contributing nothing. Concat is not merely a null, it is a PREDICTABLE null.

THE THREE ARMS:
A CONCAT (mordred726 / mordred_pca16, width 1108/1818): PROVABLY cannot help (function classes equal). Worse, a harmful reparameterisation -- two redundant routes, weight decay splits burden by norm not usefulness, extra parameters receiving gradient for nothing = pure variance. mordred726's +66% first layer to carry <=19 numbers is indefensible on its face. Run once as a NEGATIVE CONTROL, then drop.
B REPLACE (mordred_pca16_only, width 1092): the ONLY arm where the regularisation claim is actually TESTED, because it is the only one where descriptor geometry is load-bearing. If chemical similarity is a good prior, replace ~= baseline with 16 columns instead of 20; if bad, worse -- and that is a real publishable NEGATIVE result, unlike concat's guaranteed null. Caveat to state: rank 16 < 19 merges the alphabet along PC17-19, but those carry only 0.0020+0.0017+0.0002 = 0.39% of variance and no pair collides (min pairwise distance 1.81 in PCA space, well clear of zero). The trade: pca16 loses 0.39% of alphabet resolution to buy a 4.5x better-conditioned delta (2724 -> 132).
C DESCRIPTOR-INITIALISED EMBEDDING (width 1092, NOT YET BUILT): OPTIMAL. The claim is that chemical similarity is a PRIOR; section 2 proves a prior CANNOT live in the function class, so it belongs in the INITIALISATION. Arm C puts it exactly there: zero added width, zero parameters, zero FLOPs; retains the cond=1 isotropic 19-effective-dimension geometry the optimiser sees; the prior is ESCAPABLE (unlike replace, which imposes the descriptor metric for the whole run and cannot recover if wrong); degrades gracefully toward the one-hot baseline with more data, so its downside is BOUNDED whereas replace's is not. Soft variant if a stronger prior is wanted: add lambda*||W_res - W_res_init||^2 (L2-to-descriptors rather than L2-to-zero); lambda=0 recovers C, lambda->inf recovers B's rigidity.
VERDICT: C optimal, B the informative experiment, A a one-off negative control. One line: PUT THE CHEMISTRY IN THE STARTING POINT, NOT IN THE COORDINATE SYSTEM.
If B is run anyway, the mitigation is to WHITEN (use U[:,:16] not U[:,:16]*S[:16]). The CSV deliberately does not whiten (header pca_scaling=NATURAL COMPONENT SCALE PRESERVED, PC1 std 2.726 -> PC16 0.237, ratio 11.4986). That is right for REPORTING/interpretability and wrong for OPTIMISATION; whitening takes cond 132 -> ~1 at the cost of amplifying the 0.39%-variance noise directions. State the trade, do not silently pick.
HONEST EXPECTATION: no significant dG improvement from any arm, because ProtT5 already encodes residue chemistry and the net already has a free 20-row embedding. The value of the section is that it says which experiment DISTINGUISHES them and predicts the answer in advance.

**METRIC:** ddG PCC per-protein is PRIMARY, with a_p secondary. This is the rare lever that does NOT have to fall back to dG. The metric rule says ddG cancels anything IDENTICAL between WT and mutant -- a descriptor block is not such a thing: it changes in exactly the mutated row (the whole of item 4), so W6 IS ddG-measurable.

- PRIMARY: per-protein ddG PCC, baseline 0.798 (vs pooled ~0.59), because the effect is within-protein.
- SECONDARY: a_p. Item (iii) shows the 6.6x |delta| spread acts as an AMPLITUDE prior, which is a within-protein SLOPE effect and therefore lands on a_p -- and a_p is within-protein and IS ddG-measurable. a_p median 0.4990; corr(a_p, per-protein PCC) = +0.5714. If W6 does anything at all, this is where it shows.
- NOT b_p. The block is per-residue-TYPE and averages to a near-constant over a whole protein, so it is a weak whole-protein lever. b_p belongs to the coil (dG MAE 4.9650 -> 3.9521), not here. Do NOT claim W6 as a b_p lever -- and note the b_p corrector already FAILS out of sample.
- Report dG too, so the arms stay comparable to every other lever on the board.
PRE-REGISTERED PREDICTION, written BEFORE the runs: Arm A null (proved, items 2 and 4iv); Arm B neutral-to-slightly-worse (prior imposed irrevocably, cond 132); Arm C neutral-to-slightly-better with the SMALLEST seed variance. If Arm A shows a "gain" it is seed noise and must be checked against the seed spread (std(b_p)=1.5741, ICC(b_p)=0.898) before it goes anywhere near the thesis.

**GATE:** Extend scripts/gate_g4_cpu.py with a "descriptors" section; every assertion mechanical. Default path must still print G4-CPU: ALL PASS, dG=-0.0030, width=1092 (confirmed green after this work).

1. DEFAULT BYTE-IDENTICAL: --aa_descriptors none and --aa_desc_init none -> dG == -0.0030, width == 1092; compare a full forward against a stored baseline tensor with torch.equal, NOT allclose.
2. WIDTHS EXACT: mordred_pca16 -> 1108; mordred726 -> 1818; mordred_pca16_only -> 1092. Assert ==, not >=.
3. ANTI-SILENT-NO-OP -- THE ASSERTION THAT ACTUALLY MATTERS. For each non-none arm assert the forward output DIFFERS from baseline: assert not torch.equal(out_arm, out_base) AND assert (out_arm - out_base).abs().max() > 1e-6. A descriptor block that is inserted but NEVER READ would otherwise pass every width and finiteness check while changing nothing. This is exactly the PEMGraphTransformer left-anchored-slice failure mode, and it is the one that costs months.
4. BLOCK NON-CONSTANT AND CORRECTLY PLACED: extract x[:, desc_start:desc_start+K] from a real get_graph call on a real sequence and assert (a) x_desc.std(dim=0).min() > 0 -- varies across residues, so neither zero nor constant; (b) two positions with the SAME residue letter give torch.equal rows; (c) two positions with DIFFERENT letters do not; (d) torch.allclose(x_desc, one_hot @ T) -- it is the table, at the offset we claim.
5. RIGHT-ANCHORED SLICES UNMOVED: for every arm assert x[:,-20:] equals the one-hot (or is exactly zero under mordred_pca16_only) and x[:, llm_index:one_hot_index] is torch.equal to the baseline ProtT5 block. This is what catches a left-shifted LLM window.
6. REPLACE REALLY REPLACES: under mordred_pca16_only assert x[:,-20:].abs().max() == 0 AND the forward differs from baseline -- zeroed AND consequential.
7. FROZEN LAYERS STAYED FROZEN: inst_norm1.weight.shape, inst_norm2.weight.shape and fc1.in_features IDENTICAL across all arms.
8. ARM C CHANGES INIT ONLY: with --aa_desc_init pca16 assert width == 1092; assert sum(p.numel()) UNCHANGED vs baseline; assert the forward DIFFERS at step 0; assert the one-hot weight slice has rank <= 16; then assert the REST of the parameters are torch.equal to the baseline init -- only the one-hot slice moved.
9. PROVENANCE ENFORCED: assert the loaded CSV's header final_shape and column count match the mode name (already implemented via descriptor_provenance -- keep it), and keep the retired names pca16 / curated12 / pca16_only RAISING, since curated12 had come to load a 726-column Mordred matrix. Two runs with identical logged config and different tensors is the worst failure mode in the repo.

**COST:** Arm C implementation: ~15 lines in PEM.__init__ + one flag + a gate section. Arm C RUNTIME: ZERO -- init-only, width stays 1092, no parameters added, no FLOPs, no new tensors at train time. Arm B runtime: zero added width; fc1_* input unchanged because the one-hot is zeroed in place rather than deleted. Arm A runtime: fc1_gcn/fc1_gat input +16 (pca16, 1092->1108) or +726 (mordred726, 1092->1818, a +66% widening of the first layer to carry at most 19 independent numbers). CSV build: already done, offline, one-off; rdkit/mordred are NEVER imported on the training path (dependency rule in aa_descriptors.py -- esm2_env_py38 must stay exactly as the runs in flight saw it). Experiments: 4 runs (baseline, A-pca16, B, C) x seeds, scored on per-protein ddG PCC and a_p. Analysis cost already paid: the spectrum, correlation structure and delta-conditioning numbers in this design were measured read-only on the cluster, no GPU jobs submitted. Documentation: results/DESIGN_CHEMISTRY.md, 28,445 bytes, md5 ffc55d7a5577f1e6b839dea5287d480b, verified byte-identical local vs cluster.

## Hadar's 100k report (data/FINAL_DATASET_100k_030926.csv) - full column-by-column read, distribution-shift extraction, and honest assessment of whether
**Verdict:** MAYBE

**WHAT WE SEE:** 100,246 rows x 21 columns; one row = one CHAIN/DOMAIN (66,945 distinct PDB entries, mean 1.50 chains/entry, max 77). Written to results/HADAR_REPORT.md (26,447 bytes, cluster + local).

IDENTITY: rank (100% nonnull, 100,246 unique) is ONLY the sort order of loss - Spearman(rank,loss) = -1.0000 exactly; range 2..100,247 vs N=100,246 means exactly one rank was dropped. protein_id = {PDB}_{entity}_{chain} (SCOP domains as 2CRV_d2crva1). PDB_ID_and_Entity is the same string upper-cased - PURE DUPLICATE. split = training 100,086 / validation 160.

LOSS COLUMNS - three corrections to the existing schema doc:
(1) loss = lossd + lossc EXACTLY. OLS of loss on (lossd, lossc, 1) returns [1.000000, 1.000000, 0.000000], R2 = 1.0, max|residual| = 1e-6. Two free numbers, not four.
(2) lossc is NOT dead (mean +20.10, sd 6.44, range 0.12-67.29). From evaluate_train.py:232-237 it is reg_alpha * mean(E^2) over 9 energies with reg_alpha=0.1, so RMS|E| = sqrt(lossc/0.1) recovers the network's own raw energy scale for all 100,246 rows. Most useful quantity in the file.
(3) lossd is InfoNCE over 9 states (native folded/unfolded, decoy seq, decoy struct, decoy unfolded, 4 cyclic perms), tau=1.0. Textbook InfoNCE is >=0 but 99.7% of observed values are NEGATIVE -> generating version used an unnormalised log-ratio form. USE ORDINALLY ONLY. corr(lossc,lossd) = -0.953, so effectively ONE degree of freedom.
lossg = constant 0 (dead, confirmed) - it is a gradient penalty, zero because these are with_grad=False validation-path evals.
PROVENANCE: the CSV is NOT reproducible with today's train.py, which hard-zeroes lossc at line 686. It predates that commit.

METADATA: Method 10 values (X-RAY 88,879 / SOLUTION NMR 8,545 / EM 2,161 / Unknown 538 / 6 rare). Resolution 91,084 nonnull (nulls are exactly NMR+Unknown), mean 2.33 median 2.10, max 70.0 A (an EM entry - clip at 5 if ever used); Spearman(Resolution,loss)=+0.186, strongest single covariate. Oligomeric State 104 values = 102 distinct integer chain counts + Monomer + Unknown (parse the integer, do NOT one-hot 104 levels); Monomer 31,852 (31.8%), 2ch 29,253, tail is real - 182 rows at 86 chains, 147 at 162, 41 at 184 (ribosomes), 1 at 480 (viral capsid). Is Complex? is exactly (Oligomeric State != Monomer) - REDUNDANT. Chain Composition 6 values and NOT redundant, it names the partner type: homomeric 66,336 / heteromeric 22,775 / protein-NA 5,978 / protein-oligosaccharide 4,591 / Unknown 538 / protein-NA-oligosaccharide 28 -> 6,006 rows (6.0%) are protein-nucleic-acid, which is where the A/C/G/U/DA/DC/DG/DT "ligands" come from (partner chain residues, not small molecules). Global Symmetry constant 'Asymmetric' - dead, and suspicious rather than merely uninformative: a set containing 480-chain capsids with ZERO C/D/I/O symmetry is a FAILED EXTRACTION.

THE BOOBY TRAP: AA Length is stored as STRING ('Unknown' for 538) and is ASSEMBLY length, not chain length. Median 477, p75 920, p99 19,110, max 89,160. Proof: median AA Length / n_chains is ~200 aa for EVERY oligomeric state 1-25 chains (216, 229, 194, 220, 177, 205, 196, 212, 229, 156...). Row 4V8P_45_BX reads 43,352 but is one ribosome chain. Monomers-only gives a believable single-chain distribution: n=31,852, median 216, p25 124, p75 342, min 20, max 1,520.

LIGANDS: Ligands_x 4 nonnull (dead, confirmed). Ligands_y 75,464 nonnull, 8,186 distinct het codes, 212 literal 'Error'; 75.1% carry >=1 code. But the 75% figure is an ILLUSION - classifying rows by whether any het code is not a nucleic-acid chain residue, crystallisation additive, bare metal, or modified residue gives real cofactor/substrate 43,798 (43.7%), junk-only 31,454 (31.4%), no het record 24,994 (24.9%). MSE at 9,072 occurrences - the largest "ligand" after additives - is SELENOMETHIONINE, an amino acid inside the chain.

BSA - three DIFFERENT failure modes that must not be collapsed: BSA_Percentage 5 nonnull (dead), BSA_Numeric_x 4 distinct values recycled over 100,246 rows (dead merge artefact), BSA string has 233 'Error'. But BSA_Numeric_y is ALIVE: 100,000 nonnull (99.75%), 4,867 distinct, units percent, range 0.00-70.08, and internally consistent to a degree that rules out corruption - median BSA is monotone in chain count (0.00 / 12.44 / 19.68 / 23.35 / 27.72 / 28.00 / 32.56) and 30,536 of 31,852 monomers are exactly 0.

VALIDATION SPLIT IS METADATA-BLIND: all 160 validation rows have Method/AA Length/Oligomeric State/Chain Composition/Is Complex? = 'Unknown' and Resolution NaN; their ids carry a numeric prefix (20#2Y1B_1_A, 10#2FK5_1_A) that broke the join. Of the 538 Unknown rows, 160 are exactly the whole validation split and 378 are training. Any stratified analysis must run on TRAINING ONLY.

**WHAT CAN BE EXTRACTED:** The join is empty BY CONSTRUCTION: 0 of 28 test proteins appear, under exact PDB-4 match and case-insensitive substring scan of both id columns. Nothing per-protein transfers. What IS extractable is the shape of the pre-training distribution - and it yields a mechanism for b_p that needs no join at all.

MISMATCH, MARGINALS (pre-training vs our 28): monomeric 31.8% vs 100%; 43<=L<=72 aa 1.84% vs 100%; no real ligand 57.6% vs 100%; BSA=0 31.2% vs 100%; X-ray 88.7% vs mixed NMR/design.

MISMATCH, JOINT: Monomer 31,852 (P=0.3177); + 43-72aa 1,588 (P=0.01584); + no real ligand 1,504 (P=0.01500); + BSA=0 1,500 (P=0.01496, 1 in 67). Independence product 0.00337 vs observed 0.01496 -> ratio 4.45, i.e. the properties are strongly positively associated (small => monomeric => ligand-free), so the true cell is 4.45x larger than independence predicts. Still 1.5%.

THE HEADLINE NUMBER: the cell is 1.584% of ROWS but its residues are 93,686 of 116,431,172 = 0.0805% of all residues - 1 residue in 1,250. Because AA Length is assembly length and loss is per-structure, a 43,352-residue ribosome chain and a 43-residue designed bundle each count as ONE example, but message passing, ProtT5 embedding statistics and every per-residue normalisation see them 1000:1.

THE LOSS GAP: cell (n=1,500) loss -14.82, median -16.13, sd 4.64, lossd -26.04, lossc 11.22. Rest (n=98,746) loss -18.16, median -19.26, sd 3.42, lossd -38.40, lossc 20.24. Cohen d = +0.964, Welch t = 27.7. Mean rank-percentile of the cell = 0.189 (0.5 = typical) - our test regime sits in the WORST FIFTH. Cell loss sd is also 36% larger, so the model is not merely worse there, it is less consistent there.
Monomer loss vs length (real chain length): <=30 -5.68, 30-43 -9.94, 43-50 -13.05, 50-60 -14.09, 60-72 -16.30, 72-85 -16.87, 85-100 -17.88, 100-120 -18.34, 120-150 -18.68, 150-200 -18.95, 200-300 -19.12, 300-500 -19.08, 500-1000 -18.71. Steep to ~200 aa, flat thereafter.

IT IS LENGTH, NOT METHOD, NOT LIGANDS: restricting to 43-150 aa and splitting by method gives near-identical curves - 43-60: NMR -12.87 (n=733) vs X-ray -12.75 (n=161); 60-72: -15.88 vs -14.50; 72-100: -16.94 vs -16.59; 100-150: -17.49 vs -18.17. ANCOVA loss ~ log10(L) + is_NMR on 43-300 aa (n=28,957): beta_log10L = -4.211, beta_NMR = +0.485. Over 43->300 aa length moves loss 3.56 units, method 0.49 -> LENGTH IS 7.3x THE METHOD EFFECT. Ligand/BSA conditions barely move the cell mean at all (-14.86 with ligands allowed -> -14.82 without; -14.80 for rows with no het record).

THE b_p MECHANISM (Finding 2.4): decoding RMS|E| = sqrt(lossc/0.1) by length gives <=43: 6.37, 43-72: 9.85, 72-100: 12.30, 100-150: 13.51, 150-250: 13.80, 250-500: 14.33, 500-1000: 14.35, >1000: 14.01. The energy magnitude SATURATES at ~14.8 above 250 aa - correct for a contrastive ranking objective (it only ever needed energy DIFFERENCES to have the right sign) and WRONG for a model that must emit an absolute dG. On monomers L<=1000: log10(RMS|E|) = 0.2006*log10(L) + 0.6735, r=0.563, n=31,782. Sanity check: predicted at L=57 is 10.61, observed cell median 10.69 - 0.8% error. At 43-72 aa the scale is at ~67% of asymptote and still moving; dG = E_unfolded - E_folded is a difference of two such mis-scaled energies, so a PROTEIN-SPECIFIC ADDITIVE OFFSET is exactly what this predicts. That is b_p: std 1.5741, ICC 0.898 (a stable property of the learned function, not noise), 10/10 checkpoints.
This reconciles three things previously unexplained together: the b_p corrector fails out of sample (no length variation left within 43-72 aa); a_p is fine (median 0.4990, per-protein PCC 0.798) because a within-protein slope largely cancels multiplicative mis-scaling; and the coil is the best b_p lever (MAE 4.9650 -> 3.9521) because it moves the unfolded reference state, the exact term pre-training never constrained in absolute scale.

**HOW IT ENTERS:** IT DOES NOT ENTER THE MODEL. That is the designed answer, and it is backed by a measurement rather than an assumption.

No tensor changes, no insertion offset, no layer growth. Node width stays 1092 = [D(16) | Fb(32) | emb(1024) | one_hot(20)]; fc1_gcn / fc1_gat do not grow; fc2_*, inst_norm1, inst_norm2, fc_in_dim untouched. Gate re-run before and after this work: dG=-0.0030 width=1092, G4-CPU ALL PASS, git status shows no model-path modification.

I TESTED the obvious integration (turn the length/energy-scale law into a per-protein correction) and it FAILS on the 28. r_crit at p=0.05, n=28 is 0.3739:
  length            -> b_p +0.085, abs_b_p +0.363, a_p -0.284, pcc -0.292, b_p_wt_error -0.315
  log10 length      -> b_p +0.066, abs_b_p +0.360, a_p -0.276, pcc -0.278, b_p_wt_error -0.292
  predicted RMS     -> b_p +0.070, abs_b_p +0.361, a_p -0.277, pcc -0.281, b_p_wt_error -0.297
  saturation deficit-> b_p -0.070, abs_b_p -0.361, a_p +0.277, pcc +0.281, b_p_wt_error +0.297
NOTHING clears r_crit. Best is abs_b_p at r=+0.363, p=0.059 - and it is the same correlation four times, because all four predictors are monotone functions of length.
WHY, precisely: our 28 span 43-72 aa, a 1.67x range at the BOTTOM of a curve whose interesting variation is 20-250 aa. On the fitted law the predicted RMS scale runs only 9.71 (L=43) to 10.85 (L=72) - an 11% spread, against a b_p spread of +/-1.57.

The crucial distinction, which the report states explicitly: the length effect explains why b_p EXISTS (a between-regime effect, measured on 100k) but cannot predict WHICH protein gets WHICH b_p (a within-regime effect, needing variation we do not have).

Any "insert a length / n_chains / BSA / ligand-count column between Fb and emb" proposal is therefore DEAD ON BOTH METRICS AT ONCE: constant within a protein so it cancels in ddG by the metric rule, AND near-constant across the 28 so it cannot move b_p either. Explicitly rejected, not merely unimplemented.

The one integration that WOULD be correct is upstream of the feature vector entirely - see optimal_use section B: a per-example sampling weight in the pre-training DataLoader's WeightedRandomSampler, w_i = 1.9 for monomeric L<=150 and 1.0 otherwise, epoch size fixed. That touches no tensor shape and no layer width; it changes only which examples the gradient sees.

**OPTIMAL USE:** FOUR-WAY SPLIT. The optimal use is as EVIDENCE, not as features.

A. REPORT THE DISTRIBUTION SHIFT AS A THESIS RESULT - DO-IT, zero cost, zero risk, no training run, no join, no code in the model path. This converts "the model has an unexplained per-protein offset" into "the model has a per-protein offset BECAUSE it was pre-trained on a distribution where our test regime is 0.0805% of residues, with a contrastive objective that never constrained absolute energy scale, and it is measurably worst exactly there (d=0.964, t=27.7, rank-percentile 0.189)." That is the difference between a limitation and a finding. Ship: Finding 2.1 (0.0805% of residues), the monomer loss-vs-length table, the ANCOVA ruling out the method confound (7.3x), and the power law + saturation as the b_p mechanism. State the 3.1 negative control in the same breath - it is what makes the claim honest rather than convenient.

B. LENGTH-STRATIFIED RE-PRE-TRAINING - MAYBE, correct root-cause fix, out of scope. Feasibility is real: monomer 20-72aa n=2,604 needs 7.7x upsampling to reach 20% of a 100k epoch; 20-100aa n=5,186 needs 3.9x; 20-150aa n=10,771 needs only 1.9x - a mild reweight, not a fabrication. Implementation: per-example sampling weight w_i = 1.9 for monomeric L<=150, 1.0 otherwise, in the pre-training DataLoader's WeightedRandomSampler, epoch size fixed. WHY STILL MAYBE: requires a GPU pre-training run (explicitly out of bounds), and it moves the checkpoint that ICC 0.898, the 10-checkpoint replications and every b_p number are anchored to; fine-tuning may already absorb part of it. Recommend writing it up as identified root cause + principled fix, scoped as future work, with a SHARP FALSIFIABLE PREDICTION: length-balanced pre-training should reduce std(b_p) below 1.5741 while leaving a_p (median 0.4990) and per-protein PCC (0.798) essentially unchanged, because the shift is a reference-state effect.

C. FIX scripts/catalogue_schema.py - DO-IT, small, prevents a real error. (1) Document lossc's exact decode RMS|E| = sqrt(lossc/0.1) from evaluate_train.py:232-237 and note the CSV predates the commit that zeroed it. (2) Document loss = lossd + lossc exactly (R2=1, residual 1e-6) so nobody treats them as three independent signals, and Spearman(rank,loss) = -1.0 so nobody analyses rank. (3) HIGHEST-VALUE CHANGE IN THE WHOLE REPORT - hard guard on AA Length: expose n_chains parsed from Oligomeric State, and chain_length defined ONLY where Oligomeric State == 'Monomer', NaN elsewhere. Without it the next person silently correlates against ribosome assembly sizes. Also record that the 160 validation rows are metadata-blind, and split the BSA verdict THREE ways instead of calling all four columns dead.

D. DROP everything else, explicitly: length/n_chains/BSA/ligand-count as node features (dead on both metrics, per how_it_enters); pre-training loss as a per-protein prior (needs the join; join is 0/28); ligand features from Ligands_y (all 28 ligand-free, column constant on test set); anything using rank, lossg, Global Symmetry, Ligands_x, BSA_Numeric_x, BSA_Percentage, Is Complex?, PDB_ID_and_Entity (dead or duplicate).

WHY ALTERNATIVES ARE WORSE: the naive move is to join and build features - impossible, 0/28. The next naive move is to build a length-conditioned b_p corrector - I ran it, nothing clears r_crit=0.3739, and Section 3.1 explains mechanistically why (11% predictor spread vs +/-1.57 b_p spread). Reporting the shift is the ONLY use that extracts real value at zero risk, and it happens to be the highest-value one because it supplies the mechanism for the project's central unexplained quantity.

**METRIC:** dG and b_p - never pooled ddG, and never the pre-training decoy loss as a proxy for either.

The catalogue describes a WHOLE-PROTEIN, REFERENCE-STATE property (the absolute energy scale the pre-trained network assigns to a chain of a given length). By the metric rule, ddG = dG_mut - dG_wt cancels anything identical between WT and mutant, and a per-protein energy-scale offset is identical between them by construction, so it cancels EXACTLY. Scoring any catalogue-derived quantity on pooled ddG is guaranteed to return a null that means nothing - this is precisely the error that produced the "-0.001 BSA null" (a dead column scored against the wrong metric, void twice over).

a_p is NOT the right metric here either, and for an interesting reason: a_p is a within-protein slope, so a multiplicative mis-scaling of the energy scale largely CANCELS in it. That is not a defect of the analysis - it is the positive prediction that explains why a_p is healthy (median 0.4990, per-protein PCC 0.798) while b_p is not. The measured table confirms it: predicted RMS scale -> a_p gives r=-0.277, p=0.153, comfortably null.

So the scoring is two-tier:
(1) ON THE CATALOGUE ITSELF (n=100,246, no join needed): the metric is the pre-training decoy loss and its decoded components - mean loss in the cell vs rest (-14.82 vs -18.16, Cohen d=0.964, Welch t=27.7, rank-percentile 0.189), and RMS|E| = sqrt(lossc/0.1) as a function of chain length (power law slope 0.2006, r=0.563, saturating at ~14.8). This is legitimate because the question being asked is "what did pre-training optimise, and where was it worst" - which IS a question about the decoy objective. It is NOT being used as a ddG proxy.
(2) ON THE 28 (any proposed feature): the metric is b_p and abs_b_p, with r_crit = 0.3739 at p=0.05, n=28. Measured result: length +0.085, log10 length +0.066, predicted RMS scale +0.070, saturation deficit -0.070 against b_p; best anywhere is abs_b_p at +0.363, p=0.059. All fail.

The distinction that keeps this honest: tier (1) is a between-regime measurement and it succeeds decisively; tier (2) is a within-regime measurement and it fails. Reporting only tier (1) would overclaim; reporting only tier (2) would discard the file's one real finding. The report states both.

**GATE:** Existing gate, unchanged and already verified: scripts/gate_g4_cpu.py must assert dG=-0.0030 width=1092. Run BEFORE and AFTER; both runs done, both G4-CPU ALL PASS (baseline dG=-0.0030 width=1092; unfolded_emb=zero 0.0040/1092; flory_unfolded -0.0058/1092; burial_features 0.0033/1095; burial hse 0.0017/1095; burial+uemb zero 0.0023/1095; width invariants PASS). git status confirms no model-path file modified.

THE ANTI-SILENT-NO-OP CHECK for this topic is unusual and must be stated explicitly, because the deliverable adds NO tensor and therefore the ordinary "did the block actually reach fc1?" check does not apply. The failure mode here is not a silent no-op in the network - it is a SILENT WRONG NUMBER in the analysis. So the gate must assert data-layer invariants, not forward-pass ones. If scripts/catalogue_schema.py is amended per recommendation C, gate_catalogue_integrity.py must additionally assert:

1. ASSEMBLY-LENGTH GUARD (the one that actually prevents a wrong result). Assert that reading a bare `AA Length` as chain length raises, and that `chain_length` is NaN for every row where Oligomeric State != 'Monomer'. Positive control: assert max(chain_length) <= 1600 (observed 1,520) while max(AA Length) == 89160. Without this the next analysis silently correlates against ribosome assembly sizes and returns a number that looks fine and is meaningless.
2. LOSS IDENTITY. Assert max|loss - (lossd + lossc)| < 1e-5 (observed 9.6e-7). If this ever fails, the CSV was regenerated by a different code version and every decoded RMS|E| is invalid.
3. lossc DECODE LIVENESS. Assert lossc is non-constant, min > 0, and nunique > 90000 (observed 99,536). Guards against a future regeneration by today's train.py, which hard-zeroes lossc at line 686 - that would make sqrt(lossc/0.1) identically 0 and the b_p mechanism silently vanish rather than error.
4. RANK IS NOT A SIGNAL. Assert |Spearman(rank, loss) + 1.0| < 1e-6, so any future use of rank as a covariate is caught as the tautology it is.
5. VALIDATION BLINDNESS. Assert all 160 validation rows have Method == 'Unknown', so no stratified analysis silently includes covariate-free rows.
6. BSA THREE-WAY. Assert BSA_Numeric_y has > 4000 distinct values and is NOT quarantined, while BSA_Numeric_x (<= 8 distinct) and BSA_Percentage (<= 16 nonnull) remain quarantined - the three failure modes must stay distinguished.
7. JOIN EMPTINESS. Assert n_of_28_matched == 0, so nobody later believes a per-protein transfer became possible.

If recommendation B (length-stratified re-pre-training) were ever executed, its gate would be different and must assert the FALSIFIABLE prediction, not merely that training ran: std(b_p) < 1.5741 with a_p median and per-protein PCC (0.798) unchanged within noise. A run that moved all three equally would indicate a generic capacity change, not the reference-state fix claimed, and must fail.

**COST:** ANALYSIS (done): ~15 min wall clock. Read-only inspection only - no GPU job submitted, no sbatch, no scancel, no push, nothing written to the model path. Local pandas hit an out-of-memory error on the 18 MB CSV so the heavy passes were moved to the cluster login node (each well under the ~120 s exec timeout; no SLURM job needed). Deliverable: results/HADAR_REPORT.md, 26,447 bytes, uploaded base64 and verified byte-identical on both cluster and local scratchpad.

A. REPORT THE SHIFT: zero marginal cost. The measurements are already made and in the file. Pure writing time into the thesis. Zero risk - touches nothing executable.

C. FIX catalogue_schema.py: ~1-2 hours. Roughly 40 lines of schema-doc corrections plus ~7 assertions in gate_catalogue_integrity.py. CPU-only, no training. Risk is confined to the analysis layer; the G4 gate is unaffected because no model-path file changes. Highest value-per-hour item in the report - the AA Length guard prevents a whole class of silently-wrong future results.

B. LENGTH-STRATIFIED RE-PRE-TRAINING: the expensive one, and the reason its verdict is MAYBE rather than DO-IT. The code change itself is trivial - a WeightedRandomSampler with w_i = 1.9 for monomeric L<=150, 1.0 otherwise, epoch size fixed, maybe 10 lines. The cost is everything downstream: a full pre-training run on GPU (explicitly out of bounds for this session), and then RE-DERIVATION of every number anchored to the current checkpoint - ICC 0.898, std(b_p)=1.5741, a_p median 0.4990, per-protein PCC 0.798, the 10-checkpoint replications, the exposure->a_p r=+0.714 and burial->b_p r=-0.475 results, and the coil's 4.9650->3.9521 MAE gain would all need re-measuring against the new base. That is the real bill, not the GPU hours. It also risks invalidating a coherent body of measurement in exchange for an effect that fine-tuning may already partly absorb.

D. DROPPED ITEMS: zero cost, and that is the point - the ~11% predictor spread against a +/-1.57 b_p spread means any effort spent on catalogue-derived node features is spent for a guaranteed null. Establishing that cheaply (one correlation table, r_crit=0.3739) is itself part of the value delivered.

## THE OPTIMAL COMBINATION — how the built levers should be combined (results/DESIGN_COMBINATION.md)
**Verdict:** DO-IT

**WHAT WE SEE:** Written to /home/nissimb/DeepPEF/results/DESIGN_COMBINATION.md (28927 bytes, sha256[:32]=acfe94eb0536933e294ab50c48fc1d73, local copy at C:\Users\User\AppData\Local\Temp\claude\C--Users-User\eee8e8e1-58ee-4bbf-8e95-538e7a9fb0ba\scratchpad\design_out\DESIGN_COMBINATION.md).

DATA INSPECTED (read-only): 10 per-protein CSVs results/_lc_merged_abl_*.csv, 28 rows each, columns [protein,n,a_p,b_p,abs_b_p,pcc,b_p_wt_error,abs_b_p_wt_error,designed,n_chains,n_het_residues,n_metal_residues,length,mean_hydropathy_KD,frac_hydrophobic,frac_charged,frac_glycine,frac_proline,total_SASA,SASA_per_residue,mean_rel_SASA,frac_buried_rel_lt_0.25,frac_exposed_rel_gt_0.5,mean_rel_SASA_hydrophobic,SASA_over_len_pow_073,interchain_BSA,plddt_mean,plddt_min,n_mut,sd_ddg_true,range_ddg_true,mean_abs_ddg_true]; results/w0.json (var(E_u) ratios, unitless); results/w0_dg.json (dG MAE, kcal/mol); results/SLOPE_ARM_READY.md; results/SLOPE_OBJECTIVE.md; results/W5_ON_DG.md; model/hydro_net.py:400-500; Megascale-fineTuning/train.py:53-300,700-745,1073-1085; cluster_run/scripts/submit_factorial.sh.

CRITICAL DISCOVERY — "b_p" IS TWO DIFFERENT COLUMNS, units kcal/mol both:
  b_p            = intercept of polyfit(ddg_true,ddg_pred,1), sd = 0.2273 on reference ckpt  (ddG-side)
  b_p_wt_error   = pred_dG(WT) - exp_dG(WT),                   sd = 1.6030 on reference ckpt  (dG-side)
The 1.5741 headline is the b_p_wt_error family (range 1.17-1.78 across 10 ckpts), ~7x the ddG intercept.

ARCHITECTURE, verbatim from hydro_net.py:449-471:
  solv_start=48; desc_start=48+solv_dim; lig_start=desc_start+desc_dim+_sibling_block_dims(CFG)
  fc1_gcn=Linear(52+solv+desc+lig+proj_extra,64); fc1_gat=Linear(36+solv+desc+lig+proj_extra,64)
  fc2_*, inst_norm1(36+proj_extra), inst_norm2(2*(36+proj_extra)), fc_in_dim FIXED.

**WHAT CAN BE EXTRACTED:** MEASURED THIS SESSION (all 10 ckpts, per-protein n=28, sign-consistency counted):

  feature                  vs a_p            vs b_p(ddG)        vs b_p_wt_error(dG)
  mean_rel_SASA            +0.624 10/10      -0.011 6/10 NOISE  +0.233 10/10
  frac_buried_rel_lt_0.25  -0.410 10/10      +0.146 3/10 NOISE  -0.475 10/10
  length                   -0.215 10/10      +0.040 4/10 NOISE  -0.137 7/10

TWO CORRECTIONS TO THE BRIEF'S SPINE TABLE:
1. The ddG-side b_p is predicted by NOTHING structural (sign-inconsistent 6/10,3/10,4/10). All real
   offset structure lives in b_p_wt_error. This explains why the b_p corrector fails out of sample.
2. Burial (W5) is DUAL-CHANNEL, not b_p-only: 10/10 sign-consistent on BOTH a_p (-0.410) and
   b_p_wt_error (-0.475). It is the one lever legitimately creditable on ddG.

ANCHOR LADDER vs its seed-42 control — REVERSES PART OF THE BRIEF'S PREMISE:
  anchor w   sd(b_p_wt_err)   median a_p   sd(b_p)   mean per-protein PCC
  0.0        1.6737           0.4366       0.1947    0.7305
  0.3        1.7795           0.5802       0.2656    0.7314
  1.0        1.7701           0.5889       0.2767    0.7065
  3.0        1.1704           0.1535       0.0621    0.6441
At w=0.3/1.0 the anchor does NOT shrink the offset — it RISES. Only w=3.0 shrinks it, by crushing
a_p 0.44->0.15 and PCC 0.731->0.644. w3.0 is independently pathological: a_min=0.0187 (1/a_p=53x),
25/28 proteins below the 0.25 slope-oracle clip. The anchor converts a_p into b_p at a bad rate.

COIL x UNFOLDED_EMB — r_noemb=0.331 and r_coil=1.175 are var(E_u) RATIOS, not ddG correlations:
  cond        var(E_u)  r_vs_base  corr(E_u,wt_err)  corr(E_u,length)
  base        0.9583    1.000      0.4201            0.4029
  uemb=zero   0.3170    0.331      0.119             0.895
  coil        1.1263    1.175      0.507             0.144
They move EVERY diagnostic in OPPOSITE directions. They do not compete for the same variance:
zero = lower-information reference collapsing onto length; coil = differently-structured reference
that DECOUPLES from length (0.403->0.144).

**HOW IT ENTERS:** No new tensor block is proposed; this is a combination design over built levers. Exact arithmetic as the code computes it (hydro_net.py:449-471), blocks inserted BETWEEN Fb and emb so right-anchored emb/one_hot never move:

  solv_start = 48                                  # after D(16)+Fb(32)
  desc_start = solv_start + solv_dim               # 48 + {0,3}
  lig_start  = desc_start + desc_dim + _sibling_block_dims(CFG)

ONLY fc1_gcn/fc1_gat grow:
  fc1_gcn = Linear(52 + solv + desc + lig + proj_extra, 64)   # 52 = 32 dist + 20 one-hot
  fc1_gat = Linear(36 + solv + desc + lig + proj_extra, 64)   # 36 = 16 dist + 20 one-hot
FIXED, must not change: fc2_*, inst_norm1(36+proj_extra), inst_norm2(2*(36+proj_extra)),
fc_in_dim = 2*(36+proj_extra)+post_gnn_emb.

WIDTHS: baseline 1092; +W5 1095 (fc1_gcn 52->55, fc1_gat 36->39); +W5+W6 1111; +W5+W6+W11 1121.
BLOCK POSITIONS to assert: W5 [48:51]; W6 [51:67]; W11 [67:77]; emb [-1044:-20]; one_hot [-20:].
W7-edge adds NO node dims: edge_dim=56 = 20 src one-hot + 20 dst one-hot + 16 CONCATENATED RBF.

HARD CODE CONSTRAINTS FOUND (each RAISES, none is a silent no-op):
 - train.py:231  --slope_weight>0 with --loss_mode dg RAISES. No single run can carry the coil's dG
   objective AND the slope term. This forces the two-track plan.
 - train.py:1073-1085  --edge_features with --flory_unfolded RAISES absent the U6 per-half ca_coords fix.
 - _sibling_block_dims RAISES on --metal_features (W9 retired); reserving 9 cols would shift the W11
   slice into the embedding WITHOUT crashing.
 - --coil_edges without --flory_unfolded RAISES; --ligand_nodes without --ligand_annotations RAISES.

BRIEF CORRECTION: --aa_descriptors {pca16,curated12,pca16_only} are RETIRED and RAISE. Live choices are
{none,mordred726,mordred_pca16,mordred_pca16_only} (train.py:87); the loader asserts the CSV provenance
header and column count against the name.

**OPTIMAL USE:** THE TWO BETS ARE DIFFERENT COMBINATIONS, and they differ in the DIRECTION of one shared lever:

BET 1 — best pooled ddG PCC:
  --loss_mode ddg --wt_anchor_weight 0.3 --designed_weight 3 --slope_weight 0
  --unfolded_emb full --aa_descriptors none --gcn_span 1
  Why: pooled PCC is dominated by BETWEEN-protein offset (offset oracle 0.712 vs raw 0.599; ranking
  already 0.798 per-protein). w=0.3 is the ONLY anchor weight improving mean per-protein PCC (0.7314 vs
  0.7305); w=1.0 costs PCC (0.7065); w=3.0 is pathological (0.6441). Honest EV: 0.60-0.62, i.e. current
  performance +/- seed noise. NO built lever can reach the 0.712 oracle — it is fitted on test labels.

BET 2 — smallest std(b_p):
  --loss_mode dg --flory_unfolded --flory_nu 0.5 --coil_b fixed --coil_channels broadcast
  --unfolded_emb full --burial_features --burial_mode hse --wt_anchor_weight 3.0 --slope_weight 0
  (slope_weight 0 is MANDATORY, not a choice: the flag combination raises.)
  Anchor 3.0 is acceptable HERE precisely because this run is not judged on slope.

Anchor 0.3 vs 3.0 is not a tuning detail — it is the a_p<->b_p exchange rate resolved in opposite
directions because the bets are scored on opposite channels. A single "best model" does not exist;
present the trade-off curve.

WHY ALTERNATIVES ARE WORSE:
 - coil + unfolded_emb=zero: PREDICTED sub-additive with corr COLLAPSE. uemb=zero has the best raw dG
   MAE in the table (3.1315 vs coil 3.9521 vs base 4.9650) but corr collapses to 0.0510 (base 0.3464).
   It wins MAE by killing across-protein signal and landing near the mean. Acceptance must be JOINT
   (MAE and corr), never MAE alone — that is the metric-rule error one lever later. Use `mean`, not
   `zero`, as the only U2 candidate arm; `zero` is a DIAGNOSTIC.
 - A x C cells: antagonistic BY CONSTRUCTION. Both terms call the SAME get_wt_deltaG(batch)
   (train.py:707 and :733); the anchor pins that scalar while the slope term needs spread around it to
   grow. Predictable loss — do not spend seeds.
 - slope_weight as an accuracy lever: RETIRED (see metric).
 - W11: inert on 28 ligand-free, metal-free, single-chain monomers 43-72aa. Generality lever only.
 - gcn_span without gcn_bidir-at-span-1: confounds reach with directionality; uninterpretable.
 - mordred_pca16 vs mordred_pca16_only: alternatives, never both (append vs zero trailing 20).

W5 x W7-edge: partially redundant in information (W5's neighbour count is ~ a sum over W7-edge RBF
channels) but DISJOINT in channel (node-scalar/dG vs pairwise/ddG). Keep both; never claim as
independent evidence; expect sub-additivity.

**METRIC:** CHANNEL ASSIGNMENT (mechanism first, checked against measurement):
 b_p (dG-side, must NOT be scored on ddG): U2 unfolded_emb, U3 coil_channels, U4 coil_b, U5 coil_edges,
   U6 ca_coords, D flory_unfolded, W11 ligand_nodes, A wt_anchor_weight (b_p by construction —
   L1(pred_dG(WT),exp_dG(WT)) is a pure offset term).
 a_p (ddG-measurable): W6 aa_descriptors, W7 gcn_span/gcn_bidir, W7-edge edge_features,
   B designed_weight, C slope_weight.
 BOTH: W5 burial (zero in unfolded -> dG; AND -0.410 vs a_p 10/10). Brief had this as b_p-only.
 NEITHER: --holdout_residues (generalisation probe, not a lever).

SCORE Bet 1 on pooled ddG PCC; Bet 2 on dG MAE + std(b_p_wt_error) + corr JOINTLY. Score W5 on both
channels. Never score coil/U2/W11/anchor on ddG — ddG cancels anything identical WT vs mutant.

FACTOR C IS RETIRED FROM THE ACCURACY TRACK — three independent proofs:
 1. ALGEBRA: a_p = r_p * sd(q)/sd(t) EXACTLY (verified to 3e-16 across 280 protein-ckpt cells). The
    std-matching term's unique zero is sd(q)=sd(t), so AT ITS OWN OPTIMUM a_p = r_p. It cannot see a_p
    except through r_p. Ceiling = 0.793, never 1.0.
 2. ORACLE: per-protein PCC is EXACTLY invariant to pred/a_p (max |dPCC| = 7.772e-16). A perfect slope
    correction changes within-protein ranking by exactly ZERO. On pooled, the slope oracle is
    DESTRUCTIVE: 0.599 -> 0.470 on 10/10; even the affine oracle (0.649) sits below plain offset
    removal (0.712).
 3. EXPERIMENT (brief's premise "never run once" is FALSE): abl_p3_slope3.0_s42_e8 is a real
    --slope_weight 3.0 run. sd_ratio moved AWAY from 1 (0.5573 control -> 0.4489) and pooled PCC fell
    0.5929 -> 0.5624. Mechanism: the L1 term has CONSTANT gradient (+/-1.5), does not anneal near the
    optimum, so large weights chatter across it. Fixing a_p is a CALIBRATION result, not an accuracy
    result; report on sd_ratio at a weight far below 3.0.

**GATE:** scripts/gate_g4_cpu.py RE-RUN AFTER THIS WORK — SLURM CPU job 21107718 (--partition=cpu --qos normal),
node ise-cpu-intl-01, ALL PASS, baseline dG=-0.0030 width=1092 as required:
  baseline            PASS dG=-0.0030 width=1092
  unfolded_emb=zero   PASS dG=0.0040  width=1092
  flory_unfolded      PASS dG=-0.0058 width=1092
  burial_features     PASS dG=0.0033  width=1095
  burial hse          PASS dG=0.0017  width=1095
  burial + uemb zero  PASS dG=0.0023  width=1095
  width: levers that add no dims keep width  PASS 1092
  width: burial adds exactly 3               PASS 1095
Temp dir _g4tmp removed afterwards; repo left clean.

WHAT THE GATE MUST ASSERT:
 1. dG=-0.0030 width=1092 on baseline — the exact numeric anchor.
 2. Every zero-dim lever leaves width 1092 (U2, coil, U3/U4/U5, W7 span/bidir, W7-edge, A/B/C).
 3. burial adds exactly 3 -> 1095; mordred_pca16 exactly 16 -> 1111; ligand_nodes exactly 10.
 4. A real PEM forward on a real graph for EVERY combination, so a wrong fc1_* width raises.
 5. dG finite in every cell.

ANTI-SILENT-NO-OP CHECK (the central requirement — a wrong WIDTH raises, a wrong ORDER does not, and
hydro_net slices right-anchored so a mis-ordered block silently reads into the ProtT5 embedding):
 a. BLOCK-POSITION assertions, not just widths: W5 columns readable at exactly [48:51]; W6 at [51:67];
    W11 at [67:77]. gate_ligand.py gate E already checks all 16 sibling on/off combinations — that is
    the pattern to follow.
 b. RIGHT-ANCHORED INVARIANCE: emb stays at [-1044:-20] and one_hot at [-20:] under EVERY combination.
    This is the assertion that catches a block eating into the embedding.
 c. NON-DEGENERACY: for each added block assert std > 0 across nodes on a real graph. A constant block
    does not raise — it trains a bias and reports as a null. This is the documented W5 units bug:
    scaling Angstrom coords by 0.1 saturated every neighbour count to exactly 1.0 with std 0.0,
    yielding nan/meaningless-zero rather than an error.
 d. OFF => BYTE-IDENTICAL for every lever's default-off path, so an unset flag can never be a silent
    partial application.
 e. GUARDS FIRE: slope_weight+loss_mode dg; coil_edges without flory_unfolded; ligand_nodes without
    ligand_annotations; metal_features; PEMGraphTransformer with any inserted block; edge_features with
    flory_unfolded absent U6.
 f. ZERO-VARIANCE REFUSAL at scoring time: refuse to score a lever whose feature is constant on the
    eval set rather than reporting the resulting ~0 as a null.

**COST:** 14 GPU runs total, ~112 GPU-hours (~8h/run), replacing the planned 24-run W5xW6xW7 (~192 GPU-hours):
  S-B1 coil confirmation (2 arms x 2 seeds)        4 runs  — THE priority
  S-B2 W5 burial on dG   (2 arms x 2 seeds)        4 runs
  S-B3 coil x W5 cell    (only if B1,B2 both pass) 2 runs
  S-A1 W6/W7 SCREEN      (1 seed each: mordred_pca16 / gcn_span 4 / gcn_bidir@span1 control /
                          edge_features)           4 runs
  S-A2 slope calibration                           0 runs — weight ladder {0.3,1.0,3.0} ALREADY trained
        (15 ckpts each); eval jobs ev_p3_slope0.3_s42 and ev_p3_slope1.0_s42 ALREADY QUEUED. Wait for
        those CSVs; submit nothing.
~40% cheaper than the original plan and scored on metrics that can actually move.
All runs on top of the winning calibration cell, never the bare baseline, or the two result sets
cannot be combined.

CPU-only cost already paid this session: 1 CPU gate job. No GPU job submitted, no scancel, no push,
password never written anywhere.

SITE RULES (already paid for in wasted runs): export WANDB_MODE=disabled before EVERY submission;
--qos normal --gres=gpu:rtx_6000:1, NEVER select a GPU by partition name (submit filter reroutes onto
a GTX 1080 and OOMs); exclude ise-pheno-*. A finished training run is NOT a result — completion is a
non-empty CSV in eval_results/.

## WHAT EXACTLY DO WE SEE? An honest inventory of the model's actual inputs (results/WHAT_WE_SEE.md)
**Verdict:** DO-IT

**WHAT WE SEE:** Measured by inspection on 28 test proteins, 1563 pooled residues. Layout [D(16)|Fb(32)|<new@48>|emb(1024)|one_hot(20)] = 1092.

RAW PROVENANCE: coords_tensor.pt [N,4,3] holds N,CA,C AND CB (CB is stored, not recomputed); multiplied by NANO_TO_ANGSTROM=0.1 in normalize_batch BEFORE get_graph. one_hot_encodings.pt is [M,N,21] — the 21st column is dropped at train.py:317 ([:,:,:,:-1]); row 0 = WT. prott5_embeddings/*.pt are [128,N,1024] chunks, one row per mutation. mask is all 1.0 for all 28.

D(16) = atom_i*4+atom_j over [N,CA,C,CB]; cdist -> exp(g*d^2) -> mask -> sum_j -> F.normalize(p=2,dim=0). THE KERNEL IS SATURATED: because coords are scaled 0.1x, exp(-0.08*d_model^2)=exp(-0.0008*d_A^2). Over the whole physical range 0-30A the kernel moves only 1.000 -> 0.487; at contact (3.8A) it is 0.9885. All 16 channel means agree to 4 decimals (0.13287-0.13291), std ~0.0159. Off-diagonal corr median 0.994, PC1=99.22%, EFFECTIVE RANK 1.016 — 16 columns carrying ONE number. Diagonal artefact: true-diagonal channels 0/5/10/15 are ~5e-5, but cross-atom diagonals are genuine intra-residue bonds (N-CA 1.47A, N-C 2.48A, CA-CB 1.51A) — correct geometry, not artefact.

Fb(32) is NOT 32 bonded/geometric features. get_bonded_features slices two off-diagonals of the ALREADY-GAUSSIAN D: cols 16:32 = the 16 atom-pair values of the FORWARD bond (i,i+1), last row zero-padded; cols 32:48 = the same for the BACKWARD bond (i,i-1), first row zero-padded. Within each half the 16 channels correlate at median 0.999899, effective rank 1.000 each; whole block effective rank 1.958. Fb carries exactly TWO numbers. Two-thirds of its apparent variance is the ZERO PAD: corr(col16, terminus flag) = -0.7055, and interior-only std drops to 0.343x of pooled std.

emb(1024): F.normalize(p=2,dim=0) is DOWN THE RESIDUE AXIS, so every COLUMN has L2 exactly 1.0 and per-residue norms vary (3.11-6.39). min -0.558 max 0.610 mean -0.0035 std 0.130. NOT sparse: 0/60416 exact zeros, 1 entry below 1e-6.

one_hot(20): {0,1}, row sums exactly 1, AA_MAP alphabetical ACDEFGHIKLMNPQRSTVWY. Decoded 1GYZ sequence verified correct.

**WHAT CAN BE EXTRACTED:** VARIANCE (the question as posed): by the literal zero-variance test only ONE of 1092 dims is dead — a one-hot column for a residue type absent from all 28. That threshold is stable from 0 to 1e-2. This is the honest answer and it is NOT the interesting one.

RANK (the real answer): per-dimension variance cannot see duplication. Effective rank can. D=1.016, Fb=1.958, one_hot=19. THE 48 STRUCTURAL DIMENSIONS CARRY ABOUT 3 NUMBERS: a length proxy, forward-bond geometry, backward-bond geometry. ~45 of 48 are redundant copies.

D IS A LENGTH READOUT, not a structure readout: corr(N, per-protein mean D) = -0.9960, and corr(mean D, 1/sqrt(N)) = +0.9989. Mechanism: with the kernel pinned near 1, sum_j exp(g d^2) ~ 0.95N for every residue, so the column is near-constant; F.normalize(dim=0) then divides by 0.95*N*sqrt(N), leaving every entry ~ 1/sqrt(N).

WHAT dG SEES: with default flags, folded and unfolded differ in EXACTLY 16 of 1092 dims — all inside D (dims [0..15]). Fb, emb, one_hot are bit-identical between the two graphs (zero_except_udiagonal keeps the tridiagonal, which is precisely what Fb reads). So dG is driven by ONE effective scalar that is mostly a length proxy. Per-protein sum(D_f - D_u) ranges -0.212..+0.424, corr with N = -0.5742.

WHAT ddG SEES: coordinates do NOT change between WT and mutant — the mutant is scored on WT coordinates. So D and Fb are bit-identical across a mutation and cancel EXACTLY. One-hot changes 2 numbers out of 59x1092 (||d||=1.4142). The embedding changes at ALL 59/59 residues because ProtT5 is re-run on the mutant sequence: per-residue |demb|_1 for W1Q decays 98.4 -> 75.8 -> 38.9 -> 21.4 but never reaches zero (~7-15 at the far terminus); ||demb|| = 6.35, 4.5x larger than the one-hot edit. ddG is carried almost entirely by the embedding.

**HOW IT ENTERS:** CORRECTION TO THE BRIEF: the GCN does NOT read x[:,:32] only. It reads x[:,0:32] CONCATENATED WITH x[:,-20:], width 52. The GAT reads x[:,0:16] ++ x[:,-20:], width 36. Both branches DO see the one-hot; NEITHER sees the embedding.

Live PEM constants: non_bonded_index=16, bonded_index=48, one_hot_index=-20, llm_index=-1044, solv/desc/lig_start=48 with dim 0. fc1_gcn (64,52), fc1_gat (64,36), fc2_gcn (36,64), fc2_gat (36,64), fc1 (128,1096) where 1096 = 72 + 1024.

The embedding BYPASSES message passing entirely — it is concatenated AFTER both GNNs at torch.cat((x, x_emb_features)) into fc1. ProtT5 is never propagated along an edge; it is a per-residue lookup.

THE DEAD 16, PROVEN NOT ARGUED: x[:,32:48] (the backward half of Fb) appears in NO slice — GCN stops at 32, GAT stops at 16, the emb window starts at 48. Ablation on an untrained net, +100 per block: D +3.33e-03, Fb-fwd +1.14e-04, Fb-bwd EXACTLY +0.000000e+00, emb +9.80e-01, one_hot +4.60e-03. get_bonded_features computes 32 columns; the network consumes 16. The backward bond is computed and discarded. Note the emb term is ~300x larger than D — the model is dominated by the residual embedding path.

CONSEQUENCE FOR EVERY BLOCK DESIGN: a block at offset 48 is invisible to both GNNs UNLESS fc1_gcn/fc1_gat are widened AND forward() is taught to splice it in. The code already does this via _extra (which is why fc1_gcn is 52+_sd+_dd+_ld). Omit the splice and the block lands between 48 and -1044, where the GCN slice has stopped and the emb window has not started: a silent no-op with every shape check passing. Any new block needs THREE edits — _blocks in BOTH get_graph and get_unfolded_graph; the fc1_* widths; the _extra splice. fc2_*, inst_norm1, inst_norm2, fc_in_dim must not move.

**OPTIMAL USE:** Energy head: h_i = relu(fc1([gcn_out(36); gat_out(36); emb(1024)])) -> e_i = fc2(h_i) -> E = sum_i e_i (plain sum, EXTENSIVE). dG = E_folded - E_unfolded.

WHERE b_p ENTERS, reasoned from architecture not correlations — four reasons in order of force:
(a) EXTENSIVITY x a near-constant per-residue term. E ~ N*<e>, and N spans 43-72 over the 28 (a 52% spread). A per-residue bias contributes beta*N to E.
(b) D IS THE ONLY BLOCK THAT DIFFERS between the two graphs (measured: differing dims = exactly [0..15]), AND D is 99.2% one direction, AND that direction is length (corr -0.9960). So dG is a function of a length-dominated scalar.
(c) THE 1/sqrt(N) NORMALIZATION couples every protein's D to its own length. F.normalize(p=2,dim=0) normalizes WITHIN a protein, down the residue axis; its divisor is a function of N. The same local geometry in a 43-mer and a 72-mer yields DIFFERENT D values — that is the definition of a per-protein offset in the input.
(d) BETWEEN/WITHIN SPLIT: B^2/(B^2+W^2) is D 0.5573, Fb 0.0084, emb 0.0597, one_hot 0.0291. D is the ONLY block whose variance is majority between-protein. A per-protein offset needs a per-protein-constant carrier and D is the only candidate.

This mechanistically explains the measured facts: ICC(b_p)=0.898 (a stable protein property); the b_p corrector failing out of sample (it is fitting N, which does not generalize); and the coil being the best b_p lever (U3-U6 act on the unfolded D, the one channel family that actually moves). a_p differs because exposure varies WITHIN a protein, which is where emb and one_hot live.

DESIGN IMPLICATIONS: (1) Do NOT add another distance-derived Gaussian-kernelled block — it lands in the same saturated rank-1 regime. Fixing gaussian_coef or the 0.1 scaling recovers more information than any new block; the kernel currently discriminates 3.8A from 8A by 0.038. (2) Reclaim the dead 16: widen the GCN slice to x[:,:48] and fc1_gcn to (64,68), costing 1024 parameters — the cheapest real information gain available, but NOT bit-identical, so it needs its own arm. (3) Alternatives are worse because they add width to a model whose structural view is already 3 numbers wide.

**METRIC:** dG or b_p for anything touching D, the reference state, or the whole protein — now confirmed at the TENSOR level rather than by argument: coordinates are shared between WT and mutant, so D and Fb are bit-identical across a mutation and cancel EXACTLY in ddG. Only emb and one_hot survive a ddG.

Conversely, ddG is the right metric for anything acting through the embedding or one-hot, since those are the only blocks a mutation moves. a_p is within-protein and remains ddG-measurable.

b_p specifically is carried by D -> 1/sqrt(N) -> the extensive sum, so any b_p lever must be scored on dG. This is why the coil (which changes unfolded D) is the only lever that has moved dG MAE (4.9650 -> 3.9521) and why the b_p corrector, which fits N, fails out of sample.

**GATE:** scripts/gate_g4_cpu.py re-run after this work: ALL PASS, dG=-0.0030 width=1092 (plus unfolded_emb=zero 0.0040, flory -0.0058, burial 0.0033 width 1095, hse 0.0017, burial+uemb 0.0023).

The gate a new block must pass:
1. WIDTH GREW ONLY VIA fc1_*: x.shape[-1] == 1092+K; fc1_gcn.in_features == 52+K; fc1_gat.in_features == 36+K; fc2_gcn.out_features == 36; fc2_gat.out_features == 36; inst_norm1.num_features == 36; inst_norm2.num_features == 72; fc1.in_features == 1096.
2. RIGHT-ANCHORED SLICES DID NOT MOVE: torch.equal(x[:,-20:], one_hot) and torch.equal(x[:,-1044:-20], F.normalize(emb,p=2,dim=0)).
3. ANTI-SILENT-NO-OP (the load-bearing one): z = x.clone(); z[:,:,48:48+K] += 100.0; assert abs(float(m(z)-m(x))) > 1e-6, "block inserted but read by nothing". THIS IS THE CHECK THAT WOULD HAVE CAUGHT THE DEAD 16 — cols 32:48 are present, correctly shaped, correctly ordered, and move the energy by exactly 0.0. Shape assertions cannot see this; only a value perturbation can.
4. Lever off => baseline unchanged: gate_g4_cpu.py -> dG=-0.0030 width=1092.

**COST:** Zero GPU. All read-only CPU inspection on the login node, well within the ~120s limit; the only long job (gate_g4_cpu.py, ~90s building six PEMs) was run detached with setsid and its output collected. No training, no sbatch, no scancel, no push, no config change.

Deliverable: results/WHAT_WE_SEE.md on the cluster, 418 lines, md5 ddbfd5eb2412089635dc7297b696486f, verified byte-identical to the local copy at C:\Users\User\AppData\Local\Temp\claude\C--Users-User\eee8e8e1-58ee-4bbf-8e95-538e7a9fb0ba\scratchpad\wws\WHAT_WE_SEE.md. Reproduction scripts left at /home/nissimb/DeepPEF/scratch_wws3.py .. scratch_wws11.py plus cached matrices results/_wws_folded.npy, _wws_unfolded.npy, _wws_lens.npy (1563x1092 float32, ~6.8MB each).

Cost of the two follow-on actions this implies: reclaiming the dead 16 costs 1024 parameters and one training arm; re-tuning gaussian_coef costs one sweep and is value-only (no width change, no gate risk).

## W10 open-alphabet / non-canonical residue descriptors (Ofir Ezrielev's contribution), implemented properly and assessed honestly. Design written to /h
**Verdict:** MAYBE

**WHAT WE SEE:** Descriptor tables are z-scored Mordred 2D descriptors from chiral neutral-free-form L-amino-acid PubChem isomeric SMILES (histidine corrected: the plan's table had c1cc(nc1)... = C7H10N2O2, a pyridine-type ring; L-histidine is imidazol-4-yl C6H9N3O2, CID 6274). Pipeline: SMILES -> Mordred -> to_numeric(coerce) -> dropna(axis=1) -> unique-value filter -> z-score. Provenance ('#' lines: generator, smiles_sha256_16, final_shape, arm, threshold) is written into each CSV and asserted at load.

MEASURED FILE STATE (cluster, /home/nissimb/DeepPEF/data/):
- aa_descriptors_mordred.csv: M=20, K=726, filter2 "1278 -> 726", threshold_used=15 of 20 (0.750), arm=full. This is the canonical reference.
- aa_descriptors_mordred_pca16.csv: M=20, K=16.
- aa_descriptors_open25.csv: M=25, K=756, labels A..Y + SEP,TPO,PTR,MSE,HYP; filter2 "1249 -> 756", threshold_used=15 of 25 (0.600), arm=full.

THE CLOSURE, verified as two INDEPENDENT tensor-level mechanisms (not one policy):
1. ORDER = list('ACDEFGHIKLMNPQRSTVWY') and the loader RAISED on any extra row label.
2. Lookup was one_hot @ table with one_hot width 20 -- a matmul against [N,20] can only ever address the first 20 rows. Even a 25-row table leaves rows 20..24 ARITHMETICALLY UNREACHABLE.
Fixing only (1) would have produced unreachable rows and a lever that is silently a no-op. Both are now fixed in aa_descriptors.py.

WHY open25 IS REFUSED -- the brief's K-mismatch is the SYMPTOM; I traced the CAUSE and it is worse. Both the unique-value filter AND the z-score run over the WHOLE residue set in scripts/build_aa_descriptors_mordred.py:
  Mechanism A: df.loc[:, df.nunique() >= min_unique] with min_unique pinned at 15 while n_res grew 20->25, so the effective threshold fraction moved 0.750 -> 0.600 (laxer, admits 57 new columns) and the five new molecules changed per-column unique counts (drops 27). Measured: dropped=27, added=57, prefix identical? False.
  Mechanism B (the serious one): sd = df.std(ddof=0); dfn = (df - df.mean())/sd, computed over all rows present. SEP carries a phosphate and is a large outlier, so every canonical residue's z-score moves.
DECISIVE MEASUREMENT, on the 699 columns the two files SHARE (column-set change factored out, leaving only the normalisation effect): ALL 20/20 canonical rows differ, max abs diff = 3.2406 z-units. So open25 is NOT "canonical + 5 rows"; it is a DIFFERENT matrix in which the canonical twenty are different numbers on a different column basis. The loader is right; the FILE is broken.

Live modes: none | mordred726 | mordred_pca16 | mordred_pca16_only. 'pca16', 'pca16_only', 'curated12' are RETIRED and raise (curated12 had come to load a 726-column Mordred matrix, so identical logged configs could mean different matrices).

**WHAT CAN BE EXTRACTED:** 1. A METRIC ON THE ALPHABET. One-hot asserts all twenty residues are mutually equidistant; descriptors replace that with a geometry. Measured via --descriptor_neighbours: W -> Y(22.72), F(25.64), H(34.53), R(38.25), K(43.06) -- chemically correct, the aromatics. P -> V(26.29), I(27.79), T(27.97), L(29.50), K(30.24) -- aliphatics. 5/5 nearest neighbours survive in training for both, so interpolation is possible.

2. COVERAGE of residues outside the twenty -- the actual claim, and precisely what the closed tensor made impossible. A residue absent from the LM's training distribution has no good embedding; descriptors are the only thing that places it in a space the model already understands.

3. WHAT IS *NOT* EXTRACTABLE: anything the ProtT5 embedding does not already carry, for canonical residues. Ofir concedes this. ProtT5 has seen every canonical residue in hundreds of millions of contexts and encodes their chemistry better than 726 z-scored 2D Mordred columns. This is the crux of the verdict.

Note the one non-canonical code that appears in the het analysis at 6.3% is MSE (selenomethionine), which is an amino acid IN THE CHAIN, not a ligand -- exactly the case the open alphabet exists for, and exactly the case that does not occur in our 28.

**HOW IT ENTERS:** FEATURE VECTOR (live layout, verified by running get_graph):
  Fh = [ D(16) | Fb(32) | S(solv_dim) | Desc(K) | Lig(10) | emb(1024) | one_hot(20) ]
       0..15    16..47    48..48+S      -> K      -> 10     right-anchored, unmoved
desc_start = 48 + solv_dim. Insertion is BETWEEN Fb and emb so the right-anchored emb and one_hot slices never move. Baseline width 1092 (K=0); mordred_pca16 -> 1108; mordred726 -> 1818 (measured: get_graph returned torch.Size([40, 1818])).

LAYERS: only fc1_gcn and fc1_gat grow. fc2_*, inst_norm1, inst_norm2, fc_in_dim must NOT. Worth stating explicitly in the thesis: the GCN branch reads x[:, :32] only, so the descriptor block reaches the GAT BRANCH ONLY.

DATA STRUCTURE: _Table with values [M,K] float32 CPU (rows 0..19 = ORDER, rows 20..M-1 = extras), labels list[str] len M, index dict[str,int], columns list[str] len K, plus path/provenance/md5. Cached per resolved path under a lock; get_graph never touches disk.

THE LOAD-ORDER INVARIANT, enforced by construction and never trusted from the file:
  extra  = [a for a in file_order if a not in ORDER]
  labels = list(ORDER) + extra
  values = torch.tensor([rows[a] for a in labels], dtype=torch.float32)
Canonical rows placed by ORDER, not file order => row i == one-hot index i for every i < 20, so values[:20] IS the old [20,K] matrix. This single invariant is what keeps the matmul path bit-identical while the alphabet above row 19 is open.

TWO LOOKUPS:
  canonical (unchanged): residue_descriptors(one_hot, mode) -> tab.values[:N_CANONICAL] then one_hot @ table. The [:N_CANONICAL] slice is load-bearing -- it is WHY widening the table cannot change this path. load_descriptor_table(..., canonical_only=True) DEFAULTS to True so every layer-sizing caller keeps the [20,K] view.
  open (new, a GATHER not a matmul): residue_descriptors_by_label(labels, mode) builds rows=[tab.index[lab] ...], idx=torch.as_tensor(rows, dtype=torch.long), returns tab.values.index_select(0, idx) -> [M',K]. index_select over the FULL [M,K] reaches any row. 'ACD' iterates per char; ['A','SEP','W'] is taken literally, so multi-char codes work.
  Also one_hot_from_labels(labels) -> [M',M]: width M, so a non-canonical residue gets ITS OWN COLUMN rather than an all-zero row; first 20 columns coincide with the canonical one-hot.

UNDER mordred_pca16_only the one-hot is removed by ZEROING the trailing 20 columns, never deleting them: hydro_net slices right-anchored, so deletion would re-point x[:, -20:] into ProtT5 and slide the 1024-wide window left into Fb, with every shape check still passing.

CRUCIALLY: M is a ROW count, K is a COLUMN count. Adding rows moves NO layer width. That is why the open alphabet is safe to land ahead of any result. PEMGraphTransformer slices left-anchored and now raises rather than silently ignoring the block.

REBUILD REQUIRED FOR A VALID OPEN TABLE (fixes both mechanisms): persist the canonical run's surviving 726 column names AND its per-column mean/std; compute Mordred for the new residues; REINDEX to the frozen 726 columns (do not re-run either filter; a NaN on a frozen column is a hard error); z-score the new rows with the FROZEN mean/std; append verbatim with canonical rows byte-unchanged. Implement as --freeze_basis <canonical.csv>, writing basis_frozen_from=/basis_md5= into provenance. NOTE: the mean/std vectors are NOT currently emitted by the canonical build, so this is real (small) work, not a config change.

**OPTIMAL USE:** OPTIMAL: land the mechanism as INFRASTRUCTURE (default-off, byte-identical when off), run the leave-one-residue-type-out proxy on CPU, and report it as future work with the mechanism in place. Do NOT put it in the results chapter.

Rationale, and why the alternatives are worse:

- vs. "turn on --aa_descriptors for the 28 and report the delta": worse, and close to dishonest. Ofir CONCEDES physicochemical properties are implicit in LM embeddings, so the only surviving argument is COVERAGE of non-canonical residues -- and that has NO INSTANCE in this corpus (all 28 are canonical; MegaScale has zero non-canonical residues). A small positive delta on canonical residues would NOT vindicate the coverage claim because coverage is not what was tested. It would be extra free parameters, not chemistry.

- vs. "just fix ORDER and declare the alphabet open": worse, and is the original bug. A width-20 matmul cannot reach rows 20+, so the lever would be ON and arithmetically a NO-OP while every shape check passed.

- vs. "loosen the guard so open25 loads": much worse. It would silently move K 726->756 (resizing fc1_gcn/fc1_gat) AND silently retrain the canonical twenty on numbers shifted by up to 3.24 z-units, while passing every shape and finiteness check. That is the exact shape of a run that completes, reports a number, and is not what it claims to be.

- vs. "map unknown labels to zeros" (the old matmul behaviour): worse. A zero row is INDISTINGUISHABLE from a legitimately-zero descriptor (a z-scored column is zero at its mean), so it trains to garbage without ever failing. The old docstring defended zeros as "the honest encoding of no information" and as better than an argmax gather that would silently assign to alanine -- correct as far as it goes, and the canonical matmul still behaves that way, but it is a defence of a FAILURE MODE, not a feature. Phosphoserine is not the zero vector; it is a specific point. Hence KeyError.

- For the descriptor width when on: mordred_pca16 (K=16, 1092->1108) is the sane default; mordred726 (1092->1818) buys width, not information, and only reaches the GAT branch.

RECOMMENDED FRAMING: implementation + future work, containing (a) the two-mechanism closure and its removal, (b) the torch.equal bit-identity proof on real get_graph output, (c) the open25 refusal as a worked example of the silent-no-op failure class, (d) the LORO result whichever way it falls, (e) an explicit statement that the coverage claim is untested here. Claim the engineering; do not claim the science. The strongest honest sentence: "The alphabet is no longer closed, the canonical path is provably unchanged, and the claim that motivated opening it cannot be tested on this corpus -- so it is left as future work with the mechanism in place." That is defensible under questioning; a manufactured win is not.

**METRIC:** ddG, and it is legitimate here -- this is the one place the metric rule permits it.

The metric rule says ddG cancels anything IDENTICAL between WT and mutant. In leave-one-residue-type-out the held-out residue is exactly what DIFFERS between WT and mutant, so it does not cancel. This is a within-protein, a_p-type quantity and is ddG-measurable. No dG or b_p framing is needed or appropriate, and none should be used: the open alphabet is not a reference-state or whole-protein lever.

Statistic: dPCC = PCC(desc arm) - PCC(onehot arm) on the held-out rows, same estimator, same fit, same split; dMAE secondary.

MEASURED SPLITS over 862 protein CSVs / 733,945 rows:
  hold-out W (test):    TRAIN 421,268 | HELD-OUT 23,531 | DROPPED (WT side) 6,715  | 368 proteins
  hold-out P (control): TRAIN 413,319 | HELD-OUT 23,068 | DROPPED (WT side) 15,127 | 368 proteins
n ~ 23k per arm: statistical power is NOT the limiting factor; interpretation is.

Rows whose WT is the held-out residue are DROPPED, not merely untrained -- leaving X->W rows in training would show the model tryptophan as the residue being mutated AWAY FROM and the "never seen" premise would be false. This is what makes the split honest.

**GATE:** scripts/gate_open_alphabet.py (exists, passes) plus scripts/gate_g4_cpu.py. Required assertions:

(a) BIT-IDENTITY OF THE CANONICAL PATH -- anti-regression
- torch.equal(get_graph(x,oh,emb,mask) with the canonical [20,K] table, get_graph(...) with an M>20 table) on a REAL get_graph call, not on the table alone. VERIFIED LIVE: True, both torch.Size([40, 1818]).
- torch.equal(residue_descriptors_by_label(ORDER), residue_descriptors(I20)) -- matmul and gather agree on the canonical twenty. VERIFIED: row A gather == matmul -> True.
- list(ORDER) == list(train_utils.AA_MAP).

(b) THE ANTI-SILENT-NO-OP CHECK -- the one that matters, because every failure here is silent. It must assert BOTH:
- a >20-row CSV LOADS rather than raising (mechanism 1 gone); AND
- the gather RETURNS ROW 20's ACTUAL VALUES: torch.equal(residue_descriptors_by_label(['SEP']), table.values[20:21]) and tab.index['SEP'] == 20. VERIFIED: index 20, gather shape (3,726).
  WITHOUT the second assertion the lever could be "on" and arithmetically unreachable -- which IS the pre-W10 state. Asserting only that the CSV loads would let the original bug pass. This is the single most important line in the gate.
- one_hot_from_labels(seq)[:, :20] == get_one_hot(seq) for canonical seq; width is M, not 20. VERIFIED: shape (2,22) for a 22-row table.

(c) FAILURE IS LOUD
- unknown label -> KeyError naming the label, listing known labels, stating why not zeros. VERIFIED.
- duplicate row label -> raises (open != unvalidated).
- data/aa_descriptors_open25.csv STILL RAISES, message naming both the column-count and column-name mismatch. Pin as a regression test -- it is the worked example. VERIFIED.
- a table with correct columns but a perturbed canonical block raises via torch.equal(tab.values[:20], ref.values[:20]).
- retired mode names (pca16, pca16_only, curated12) raise.

(d) WIDTH INVARIANCE -- the open alphabet moves nothing
- descriptor_dim(mode) unchanged by adding rows; adding rows changes NO layer width.
- fc2_*, inst_norm1, inst_norm2, fc_in_dim unchanged.
- scripts/gate_g4_cpu.py -> dG=-0.0030 width=1092. VERIFIED AFTER THE CHANGE: "baseline forward ok, dG finite PASS dG=-0.0030 width=1092", "G4-CPU: ALL PASS".

(e) LORO HARNESS
- split non-empty; NO TRAIN row has the held-out residue as mutant OR as WT.
- parse_mutation handles multi-char codes (S7SEP).
- ridge penalty selected by TRAIN-ONLY CV WITHIN EACH ARM (arms are width 40 vs 3K; without this, P3 measures capacity rather than chemistry).

**COST:** GPU cost: ZERO. Nothing in this design requires a training run.
Runtime cost with --aa_descriptors none (the default): ZERO, byte-identical.
Runtime cost when on: mordred_pca16 +16 columns (1092->1108); mordred726 +726 (1092->1818), GAT branch only. Only fc1_gcn/fc1_gat grow.

Work status:
- Loader / label-keyed gather / silent-no-op guard: DONE.
- scripts/gate_open_alphabet.py: DONE, passes.
- LORO harness + W and P splits: DONE, measured.
- --freeze_basis rebuild (the fix for open25): ~half a day, and it BLOCKS on persisting the canonical build's per-column mean/std, which are NOT currently emitted.
- LORO reference probe for W and P: CPU only, no GPU job.

Risk is bounded: default-off, byte-identical when off, no layer width moves when the alphabet opens, and the guard refuses any table that would silently change K or the canonical block.

## THE EMBEDDINGS — what the stored ProtT5 block actually is, what is recoverable from it, and the optimal way to feed it into DeepEF
**Verdict:** DO-IT

**WHAT WE SEE:** Path: /groups/keasar_group/casp15/meytav/{protein_tensors,MutationFineTuning/test_protein_tensors}/<P>/prott5_embeddings/prott5_embedding_<k>.pt (union covers 28/28; 27 carry embeddings, r18_3_TrROS_Hall only in root 1). Loaded by train.py:520 load_embedding_tensor(), numeric-sorted glob + torch.vstack.

SHAPE: [128, 59, 1024] float32 for 1GYZ shard 0. The FIRST axis is VARIANTS, not proteins — 128 sequence variants per shard. 1GYZ = 12 shards = 1526 variants, matching deltaG.pt [1526] and one_hot_encodings.pt [1526,59,21] exactly. EVERY MUTANT GOT ITS OWN ProtT5 FORWARD PASS. This was never written down and it reframes the problem: we hold a per-variant contextual embedding for ~1500 variants per protein, not one WT embedding reused.

Variant composition (1GYZ): 1159 single mutants, 361 doubles, 6 WT replicates (rows 0-4 bit-identical, torch.equal). 1483 unique sequences of 1526. dG range -5.096..+4.550, no NaNs.

VALUES: dtype float32; min -0.994, max +1.172; mean -0.0041 (centered); element std 0.1767; per-residue L2 norm ~5.6 (5.2-9.0). Raw prot_t5_xl_uniref50 per-residue encoder hidden states, unpooled, unnormalized at rest.

PROVENANCE: these are per-residue (not pooled) states; the model consumes them as x[:, -1044:-20] in a width-1092 node vector [D(16)|Fb(32)|<new>|emb(1024)|one_hot(20)].

**WHAT CAN BE EXTRACTED:** DECISIVE TEST — CONTEXTUAL, NOT A LOOKUP. Within variant 0 of 1GYZ, all 13 residue types occurring at >=2 positions: cosines 0.0895-0.7140, L2 distances 4.13-9.14 against vector norms ~5.6. Two valines (pos 8 vs 39) have cosine 0.0895 — essentially ORTHOGONAL. A lookup would give cosine exactly 1.0000 and L2 0.0000 on every row. So the descriptor question does NOT collapse to linear algebra over 20 points.

LINEAR PROBES, LEAVE-ONE-PROTEIN-OUT (stricter than residue-disjoint; 1501 residue rows, 27 proteins, scaler+PCA fit on train only; SLURM job 21107785):
- Residue identity (20-way): acc 0.9913 (majority 0.1146, chance 0.0500); per-protein mean 0.9921, min 0.9403, max 1.0000.
- Kyte-Doolittle hydropathy: R2 0.9809 (r 0.9904)
- Formal charge pH7: R2 0.9767 (r 0.9884)
- vdW volume: R2 0.9710 (r 0.9854)
=> A descriptor block is one_hot @ table, i.e. a LINEAR function of identity; identity is linearly recoverable at 99.13%. W6 is provably redundant. This is the measurement of exactly what Ofir conceded, and his residual case (non-canonical coverage) is void: one_hot column 20 has total mass 0.0 across all 28.

MUTATION RIPPLE (ddG-relevance): per-position L2 delta vs WT, binned by |pos - mutsite|: at-site 4.3064, |dp|=1 1.5161, |dp|=5 0.6733, |dp|>=15 ~0.35. Ratio at-site/far = 12.3x. The far field is NON-ZERO (~6% of ||emb||), so the embedding does NOT cancel in ddG — it is a legitimate a_p lever, unlike the state-independent W6 block which cancels exactly.

VARIANCE DECOMPOSITION: total 37.1347; BETWEEN-protein 2.2477 (6.1%) = the offset/b_p channel; WITHIN-protein 34.8870 (93.9%) = what a_p/ddG can see. Reconciles W0 (zeroing emb unfolded collapses var(E_u) 67%): the offset rides a small, low-dimensional, high-leverage component — isolable by projection/gating, which is why zeroing the whole block is a sledgehammer.

INTRINSIC DIM: top16 PCs 21.8%, top128 68.0%, top256 86.4%; 95% needs 412 PCs, 99% needs 640. Genuinely high-dimensional — the existing CFG.emb_proj_dim=16 would keep only 21.8%.

**HOW IT ENTERS:** TODAY (read from model/hydro_net.py PEM.forward ~544-580, __init__ ~403-478): x_gcn = cat(x[:,:32], *_extra, x[:,-20:]) -> 52 dims; x_gat = cat(x[:,:16], *_extra, x[:,-20:]) -> 36 dims; BOTH EXCLUDE emb. x_emb_features = x[:, llm_index:one_hot_index] = [B*N,1024]. GNN branches give 36+36=72, pass inst_norm2, then `if self.emb_projector is None: x = cat((x, x_emb_features))` -> [B*N,1096] -> fc1 = Linear(1096,128) -> fc2 = Linear(128,1). fc_in_dim = 2*(36+proj_extra) + post_gnn_emb (line 477).

THREE FACTS NOT PREVIOUSLY STATED: (1) the embedding NEVER enters the graph — no message passing, no edge, no neighbor; it is a post-GNN per-residue side-channel, so all structural reasoning happens in 72 dims that never saw the LM. (2) inst_norm2 normalizes the 72 GNN dims and NOT the 1024 emb dims — they meet un-comparable. (3) PEM.__init__ defaults emb_projection="none" and Megascale-fineTuning/train.py (lines 1110,1166,1212,1250) constructs PEM(...) WITHOUT passing it, so MLPProjection/LowRankProjection (lines 292-310) are fully wired and completely unused.

E1 --emb_norm_axis {residue,feature} (bug fix, b_p). train_utils.py:620,684 + folded site. residue = F.normalize(emb,p=2,dim=0) (today, bit-identical); feature = F.normalize(emb,p=2,dim=1) * _EMB_SCALE with _EMB_SCALE = 1/sqrt(1024) as a MODULE-LEVEL CONSTANT, never a function of N. emb stays [N,1024]; NO shape change anywhere; fc1 stays Linear(1096,128). Must apply to folded AND unfolded identically or it contaminates the U2 factor.

E2 --emb_gate {none,scalar,film} (accuracy, a_p). In __init__: emb_ln = nn.LayerNorm(1024); emb_proj = nn.Linear(1024, emb_proj_dim_v2=128); scalar -> emb_gate_p = Parameter(zeros(1)); film -> emb_film = nn.Linear(72,128). fc_in_dim = 72+128 = 200 when on, 1096 when off. In forward: e = emb_proj(emb_ln(x_emb_features)); e = sigmoid(emb_gate_p)*e OR sigmoid(emb_film(x))*e where x is the 72-dim GNN output; x = cat((x,e)) -> [B*N,200]. ONLY fc1 changes, and it SHRINKS Linear(1096,128)->Linear(200,128). fc2, inst_norm1, inst_norm2 untouched. New params: emb_ln 2048 + emb_proj 131,200 + 1 (scalar) or 9,344 (film); net vs old fc1 (140,288->25,600) is +19k/+29k. light_attention rebuilds automatically since fc_in_dim is computed before line 502. Node width stays 1092.

E3 extend --unfolded_emb with `wt` (use variant-0 WT embedding in the unfolded pass; already plumbed at train.py:986,995) and `proj` (emb.mean(dim=1,keepdim=True).expand_as(emb) — removes feature-direction identity, complementing existing `mean` which does dim=0). [N,1024]->[N,1024], no width or layer change.

E4 (MAYBE) --emb_site_delta: 1 column |emb_mut[i]-emb_wt[i]|_2 / 4.3064 inserted between the last new block and emb, width 1092->1093, only fc1_gcn/fc1_gat grow by 1. Needs a get_graph signature change (WT emb not currently passed) touching every caller and gate.

**OPTIMAL USE:** FOR b_p: E1 first, then E3. b_p is a whole-protein additive constant. The embedding contributes to it via (i) the 1/sqrt(N) scale artifact — corr(N, per-residue emb norm) = -0.9938 — and (ii) the 6.1% between-protein component that W0 showed drives 67% of var(E_u). E1 removes (i) at the source in one line and CANNOT hurt a_p because it rescales WT and mutant identically. E3's wt/proj arms attack (ii) surgically instead of destroying 93.9% of the block as U2=zero does. The coil stays the best measured b_p lever (4.9650->3.9521); E1 is COMPLEMENTARY, not competing — the coil fixes the reference state's geometry, E1 fixes the emb block's extensivity. Judge out-of-fold on dG MAE and corr(b_p,N), never an in-sample offset fit, since the b_p corrector is known to fail out of sample.

FOR a_p: E2, specifically `film`. Only E2 and E4 change within-protein discrimination; E1/E3 are scale/reference-state levers and are a_p-neutral by construction. The measured pathology is exactly a within-protein discrimination problem: 72 geometry dims encoding THIS residue's structural environment are outvoted 14:1 by a block whose within-protein variance is largely mutation-irrelevant context. Projecting 1024->128 and gating moves fc1's emb share from 93.4% to 64% while retaining 68% of variance. `film` beats `scalar` because exposure->a_p r=+0.714 says the STRUCTURAL state should modulate trust in the LM (more ProtT5 in loops, less in a determined core), and film is exactly that function.

WHY ALTERNATIVES ARE WORSE: (a) U2=zero — already measured by W0, collapses var(E_u) 67%, but throws away 93.9% of variance to remove a 6.1% component; a sledgehammer where E3 is the scalpel. (b) Project to 16 dims (the EXISTING CFG.emb_proj_dim scaffold) — retains only 21.8% of variance; it would fix the drowning by destroying the signal. Hence proj_dim=128 and a NEW key emb_proj_dim_v2, because re-tuning emb_proj_dim would silently alter PEMSM and pnas_train*.py which read it. (c) Feed emb into the GNN branches (what emb_projection was built for, lines 558-561) — changes gcn_dim_in/out and therefore fc_in_dim=2*(36+proj_extra), growing the GNN layers, perturbing working message passing, and violating the rule that only fc1_gcn/fc1_gat may grow. (d) LayerNorm alone — fixes scale parity but not the column-count mechanism F4 identifies; subsumed by E2, which includes it.

W6 DESCRIPTORS: DROP, do not tune. Section-2 probes show a descriptor block is already in the span of the 1024 columns handed to the model. Report the probe as the REASON — stronger than a null ablation because it explains why rather than merely observing that it didn't help.

**METRIC:** E1 -> dG and b_p. It is a whole-protein/reference-state scale lever; the change is near-identical between WT and mutant so ddG would show almost nothing. Report dG MAE, var(E_u), std(b_p) (baseline 1.5741), and corr(b_p,N) before/after. Expected: corr(b_p,N) falls, dG MAE improves or holds.

E2 -> a_p and per-protein ddG PCC FIRST, then dG/b_p. This is the one lever here that is legitimately ddG-measurable, and F5 is the justification: the mutation ripple is 12.3x concentrated at the site but the far field is non-zero (~6% of ||emb||), so the embedding does NOT cancel in ddG. Report per-protein ddG PCC (baseline 0.798), pooled PCC (~0.59), a_p median (0.4990), then dG MAE.

E3 -> dG, var(E_u), b_p ONLY. Reference-state lever, ddG-invisible by construction. Never score on ddG.

E4 -> a_p (zero unfolded, non-zero folded, so it survives the ddG difference).

ProstT5 swap -> dG and b_p, and only after E1 is in.

Metric-rule compliance: the only ddG-scored lever is E2 (and E4), both of which are genuinely within-protein. Every reference-state and whole-protein lever here is explicitly routed to dG/b_p/var(E_u).

**GATE:** Extend scripts/gate_g4_cpu.py, which must keep reporting dG=-0.0030 width=1092 (re-verified GREEN this session: baseline PASS dG=-0.0030 width=1092, all 8 checks pass).

A. DEFAULT-OFF BYTE-IDENTITY (anti-silent-no-op, part 1). With --emb_norm_axis residue, --emb_gate none, --unfolded_emb full: width == 1092 exactly; fc1.weight.shape == (128,1096); dG == -0.0030 to committed tolerance; and torch.equal (BIT-IDENTICAL, not allclose) between baseline Fh and flagged-default Fh.

B. THE FLAG ACTUALLY DOES SOMETHING (anti-silent-no-op, part 2) — catches the class of bug W9's silent column shift and PEMGraphTransformer's left-anchored slice belong to.
 - E1: assert NOT torch.equal(Fh_res, Fh_feat); assert the emb slice Fh[:,-1044:-20] DIFFERS while Fh[:,:48] and Fh[:,-20:] are torch.equal. Then assert the ARTIFACT IS SPECIFICALLY REMOVED: over >=6 synthetic proteins with N in {43..72}, abs(corr(N, per-residue emb norm)) < 0.10 under `feature` and < -0.95 under `residue`. A test that only checks "output changed" is INSUFFICIENT.
 - E2: fc1.in_features == 200 when emb_gate != none with emb_proj_dim_v2=128, == 1096 when none; fc2.in_features == 128; inst_norm1/inst_norm2 parameter shapes torch.equal to baseline; forward runs and output is finite; sigmoid(emb_gate_p) in (0,1).
 - E3: for each of full|zero|mean|wt|proj, the UNFOLDED Fh emb slice differs pairwise across arms while the FOLDED Fh is torch.equal across all five — proving E3 is reference-state-only and has not leaked into the folded pass.

C. SLICE INTEGRITY UNDER EVERY COMBINATION. For all on/off combinations of --burial_features, --aa_descriptors, --ligand_nodes crossed with the new flags, assert llm_index and one_hot_index still select slices torch.equal to independently-constructed emb and one_hot. Reuse the 16-combination sweep from scripts/gate_ligand.py gate E. This is the wrong-column-without-crashing risk.

D. CHECKPOINT-SHAPE SAFETY. Loading a BASELINE checkpoint into a model built with emb_gate != none must RAISE on fc1.weight shape mismatch, never silently strict=False-load — a partial load would yield a plausible-looking garbage model, exactly the failure PEMGraphTransformer was made to raise on.

E. PEMGraphTransformer still raises. E1-E4 must not re-enable it.

**COST:** E1 norm axis: ~10 lines in train_utils.py (3 call sites) + flag + gate; 1 retrain per arm (2 arms). RISK LOW — value-only, no shape change, default bit-identical.
E2 gate/project: ~25 lines in hydro_net.py + flag + gate; 1 retrain per arm, realistically 3 arms (none / scalar@128 / film@128) out of a 3x3 grid. RISK MEDIUM — the only lever that changes fc1's shape, cannot be bit-identical when on, needs gate D.
E3 unfolded arms: ~8 lines in _unfolded_emb; WT tensor already plumbed at train.py:986,995; 1 retrain per new arm (2 new). RISK LOW — same shape, unfolded-only.
E4 site delta: get_graph signature change + all callers + all gates; 1 retrain. RISK HIGH — width 1092->1093 touches every slice. DEFER.
ProstT5 swap: ~0 code (dimension-identical 1024, no shape or gate changes); compute = re-embed ~40k variant-sequences on GPU. RISK LOW code / MEDIUM compute. Only after E1.
Non-1024 swaps (ESM-2/ESM-1v/SaProt 1280, ESM-IF 512) additionally shift llm_index and total width (1092->1348), touching every right-anchored slice and every gate.

RECOMMENDED ORDER: E1 -> E2(film,128) -> E3 -> (ProstT5) -> E4 only if a_p is still binding. E1 first because it is a bug fix, it is cheap, and leaving it in place biases every subsequent measurement — including any embedding comparison, which would otherwise measure which 1024-dim block best survives a bad normalization.

Work done this session was read-only inspection plus ONE CPU SLURM job (21107785, --partition=cpu --qos=normal). No GPU job submitted, nothing trained, no repo file modified, nothing cancelled or pushed.

## Attacking b_p architecturally rather than by post-hoc correction: a design for --dg_scale_channels, grounded in a measured diagnosis that b_p is not a
**Verdict:** DO-IT

**WHAT WE SEE:** b_p is defined in scripts/calc_bp.py as the wild-type residual b_p = pred_deltaG(WT) - deltaG(WT): one scalar per protein from the single row with ddG==0, not an OLS intercept. On abl_calib_ctrl_repro2_e14, n=28: std(b_p)=1.5741 (matches the brief), mean=+0.3542, range -4.704 (2KVS) to +2.747 (r12_757_TrROS_Hall). std(true WT dG)=0.9042 but std(pred WT dG)=1.3530 -- the predictions have MORE spread than the truth.

Across all 10 eval CSVs, corr(true WT dG, pred WT dG) averages -0.0244 and never exceeds |r|=0.12 (worst -0.1188, best +0.0696, all p>0.5). Five seeds, three anchor weights, one slope arm: the model has ZERO cross-protein signal on absolute stability.

Architecturally, from the live tree: the node vector is width 1092 = [D(16)|Fb(32)|emb(1024)|one_hot(20)]. Diffing get_graph vs get_unfolded_graph at N=60, ONLY columns 0-15 differ; Fb, emb and one_hot are bit-identical (torch.allclose True). This follows from source: get_bonded_features reads only D[i,i+1] and D[i+1,i], and zero_except_udiagonal preserves exactly that tridiagonal. Then F.normalize(D,p=2,dim=0) sets every one of the 16 channels to unit L2 norm over residues in BOTH passes -- verified colnorm [1.0000]x4 folded and unfolded -- destroying a raw block-sum difference of 96.4 vs 58.1 (ratio 0.60).

At random init: |E_f|/N = 0.0650 for N=43,50,60,72, corr(N,E_f) = -1.0000 (perfectly extensive), while dG ~ 0.003, i.e. 3 orders of magnitude smaller than the quantities differenced, and corr(N,dG)=+0.263. Live gate: dG=-0.0030 width=1092.

**WHAT CAN BE EXTRACTED:** The decisive extraction is the variance decomposition on b_p averaged over 6 checkpoints: var(b_p) = var(pred) + var(true) - 2cov = 2.0243 = 1.1971 + 0.8479 - 2(0.0104). cov(pred,true) = 0.0104, i.e. ZERO. b_p is not a bias sitting on top of a correct prediction; it is the sum of two independent variances. This is why the post-hoc corrector failed out of sample and always would: you cannot subtract a bias that carries no information about the target. The oracle (0.7119) works only because it uses the held-out protein's own b_p, which encodes the true dG (r=-0.6393, p=0.0003) -- information no test-time observable can supply.

b_p IS stable and therefore attackable: mean pairwise corr of b_p vectors across 10 checkpoints = +0.8844, between-protein var/total = 0.8375 (consistent with ICC 0.898). It is a deterministic property of the (protein, architecture) pair.

Of test-time-observable features, only frac_buried_rel_lt_0.25 clears p<0.01 (r=-0.5337, R^2=0.285). Everything else is ns: SASA_per_residue +0.354, frac_charged +0.353, mean_rel_SASA +0.322, length -0.261.

CORRECTION TO THE BRIEF: b_p does NOT correlate with length. Across 10 checkpoints mean r = -0.1371, sign flips 7/10 negative vs 3/10 POSITIVE (+0.160, +0.093, +0.090), and not one reaches p<0.05. The brief's -0.368 reproduces on exactly one checkpoint (abl_calib_ctrl_repro2_e14, where I measure -0.315). All 28 proteins span 43-72aa, a 1.67x range -- insufficient leverage. Contrast burial->b_p, which is 10/10 consistent.

**HOW IT ENTERS:** --dg_scale_channels adds a 4-column per-residue block computed from the SAME D tensor already built in get_graph/get_unfolded_graph, taken AFTER the Gaussian kernel and masking but BEFORE D.sum(dim=1) and F.normalize:
  col0 log1p(D[i,:,5].sum())   CA-CA contact mass (CA-CA is channel 5)
  col1 log1p(D[i].sum())       total mass over all 16 channels
  col2 (D[i,:,5]>0.5).sum()/50.0   contact count
  col3 log1p(N_valid)/5.0      chain length, broadcast
Cols 0-2 DIFFER between passes (the unfolded path zeroes off-tridiagonal entries, collapsing contact mass to ~2 neighbours); col3 is identical and cancels, present so the net can form contact DENSITY internally. log1p keeps it O(1) without destroying ratio information the way L2 normalization does. It is deliberately NOT normalized over residues -- that is the entire point.

Width 1092 -> 1096: [D(16)|Fb(32)|SCALE(4)|emb(1024)|one_hot(20)], block at 48-51, emb 52-1075, one_hot 1076-1095. Inserted BETWEEN Fb and emb so right-anchored llm_index=-(1024+20) and one_hot_index=-20 never move.

In train_utils.get_graph the block takes a slot in _blocks immediately after W5 solv and before W6 desc. In PEM.__init__: scale_dim = 4 if CFG.dg_scale_channels else 0; scale_start = solv_start + solv_dim (48, or 51 with W5); desc_start = scale_start + scale_dim. CRITICAL: _sibling_block_dims(CFG) at model/hydro_net.py:470 is the ONE authority for lig_start arithmetic and MUST be updated to include scale_dim, or W11 silently reads wrong columns and never crashes -- the single highest-risk line.

Layers that GROW (only these): fc1_gcn 68->72, fc1_gat 52->56 (verified live at CFG defaults with emb_projection="mlp", proj_extra=16). Layers that must NOT change: fc2_gcn, fc2_gat, inst_norm1 (36+proj), inst_norm2 (2*(36+proj)), fc_in_dim, fc1, fc2. PEM.forward extends the existing _extra list between the solv and desc appends; since _extra is non-empty, forward_gat takes the already-existing wider-input branch, so no new branch. PEMGraphTransformer already raises on inserted blocks -- nothing to do.

**OPTIMAL USE:** Run it as a 2x2 factorial with U2 --unfolded_emb zero, NOT sequentially. The two attack the two independent mechanisms identified in the analysis: (i) the erased contact magnitude, and (ii) the embedding leak -- the common 1076-dim part bleeding into E_u - E_f through relu, GAT softmax and two InstanceNorm layers, which makes df/dd a function of the ProtT5 fingerprint and acts as a per-protein gain. That leak explains why b_p is reproducible at ICC 0.898 yet uncorrelated with true dG, and W0 already confirmed it from the other side (zeroing unfolded emb drops var(E_u) to 0.331 and corr(E_u,wt_err) 0.420->0.119).

Why the alternatives are worse:
(a) learned b_p head -- DROP. Two independent kills. It can only regress on observables that top out at R^2=0.285, i.e. the exact ceiling the corrector already failed at (LOPO 0.6049 vs raw 0.5994, a gain of 0.0055). And it is unidentifiable: conditioning on the folded protein makes it constant across variants, so under --loss_mode dg the gradient trades it freely against a global E_f shift, and under --loss_mode ddg it cancels exactly and gets ZERO gradient. No regime where it is both identifiable and useful.
(b) --dg_length_norm -- DROP, and actively harmful, not merely useless. Its premise is false (b_p vs length is ns, sign-flipping). Worse, the extensive part ALREADY cancels: corr(N,E_f)=-1.0000 but corr(N,dG)=+0.263 at init, before training. Dividing an already length-independent quantity by per-protein N or sqrt(N) INJECTS a per-protein scale error (up to 1.67x across 43-72aa) and rescales every ddG, corrupting a_p as collateral.
(c) Siamese -- DROP. Already satisfied trivially: one PEM, one call on torch.cat([folded,unfolded]), weights fully shared, inputs identical in 1076/1092 columns. Maximally Siamese already; no untied parameter remains. The non-trivial reading (penalize ||h_u - h_f||) is degenerate: since dG = sum(e_u - e_f), driving latents together drives dG->0 for every protein, and exempting the signal subspace IS the original problem.
(d) ddg_head -- MAYBE, as a control arm only. See caveat.

(e) is the only option that addresses the MEASURED cause (cov=0.0104, a missing input) rather than the assumed cause (a bias to subtract), is cheap, gated, default-off, metric-legal, consistent with the one 10/10-reproducible structural finding (burial->b_p r=-0.475), and adds inputs rather than a correction term -- so dG stays E_u - E_f from one network on two graphs.

**METRIC:** Score on dG / b_p, never ddG -- cols 0-2 differ between folded and unfolded so they survive E_u - E_f, but are near-constant across the point mutants of one protein (a single substitution barely moves total contact mass), so they largely cancel in ddG. Correct b_p-lever signature under THE METRIC RULE.

PRIMARY endpoint: corr(true WT dG, pred WT dG) over the 28, currently -0.0244 averaged over 10 checkpoints. SECONDARY: std(b_p), currently 1.5741.

The primary must be corr, not std(b_p), and this is a real trap: because cov(pred,true)=0.0104 and var(b_p)=var(pred)+var(true)-2cov, a lever that merely shrinks predictions toward their common mean will REDUCE std(b_p) toward std(true)=0.9042 while leaving corr at zero -- looking like success while the model still knows nothing about absolute stability. Report both, never std(b_p) alone. Honest expected outcome: corr moves to 0.3-0.5, std(b_p) 1.57 -> ~1.2.

**GATE:** scripts/gate_dg_scale.py before any training; scripts/gate_g4_cpu.py must still print dG=-0.0030 width=1092 on its baseline row.

A. OFF is bit-identical: width==1092; torch.equal(graph_off, baseline) folded AND unfolded; g4 baseline row unchanged.
B. Widths: ON width==1096 exactly; fc1_gcn.in_features==72; fc1_gat.in_features==56; fc2_gcn/fc2_gat UNCHANGED; inst_norm1/inst_norm2/fc1.in_features UNCHANGED; with --burial_features width==1099 and scale_start==51.
C. Placement: allclose(graph[:, scale_start:scale_start+4], block_built_directly); emb slice x[:,-1044:-20] identical ON vs OFF; one_hot x[:,-20:] identical ON vs OFF; C.4 with --dg_scale_channels AND --ligand_nodes, ligand_features.ligand_start == PEM.lig_start -- the assertion that catches the W11 silent-corruption bug.

D. ANTI-SILENT-NO-OP (every failure mode here is silent: wrong offset feeds wrong columns to fc1; an all-zero block trains happily; a block identical in both passes cancels in E_u - E_f and does nothing while looking fine):
 D.1 block.abs().sum() > 0, folded AND unfolded (not all-zero)
 D.2 block.std(dim=0) > 1e-6 for cols 0,1,2 (not constant)
 D.3 THE CORE ONE: (blk_f[:,:3] - blk_u[:,:3]).abs().max() > 0.1 -- if this fails the block cancels in the difference and the lever is a no-op
 D.4 torch.equal(blk_f[:,3], blk_u[:,3]) -- length col IS identical, as designed
 D.5 abs(blk[:,1].norm() - 1.0) > 0.01 -- proves it was NOT L2-normalised, the whole point
 D.6 monotonicity: compact vs extended structure at same N, blk_compact[:,1].mean() > blk_extended[:,1].mean() -- the analogue of W7's k(2A)>k(8A)>k(15A) assertion that caught the flat-kernel bug; proves the channel encodes what it claims
 D.7 dG.backward(); fc1_gcn.weight.grad and fc1_gat.weight.grad over the scale columns both nonzero -- gradient actually reaches it
 D.8 N=1 and all-zero mask produce finite output, no NaN/Inf
E. Metric legality: on a WT/mutant pair of the SAME protein the block changes by <5% of its folded/unfolded difference -- confirms b_p lever, not a disguised a_p lever.
Post-run (on the eval CSV, not the gate): corr(true WT dG, pred WT dG) must move decisively off -0.024.

**COST:** Implementation ~40 lines across three files: a block builder in train_utils.py wired into the _blocks list of both get_graph and get_unfolded_graph; three lines in PEM.__init__ (scale_dim, scale_start, shifted desc_start) plus the two fc1_* width expressions; one append in PEM.forward's _extra list; the _sibling_block_dims update; one argparse flag in Megascale-fineTuning/train.py mirroring the W5 pattern. Plus ~120 lines of gate. No new parameters beyond 4+4=8 extra input weights per fc1 branch. One training run to evaluate; the 2x2 with U2 is four runs. Default-off and bit-identical when off, so it cannot disturb any existing arm.