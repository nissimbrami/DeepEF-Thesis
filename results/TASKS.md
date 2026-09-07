# TASKS — main session only, no background agents.
# Read this every round. Do not stop until every line is DONE or BLOCKED-with-reason.
# Source: results/AGENT_REPORT.md (28 agent results, 286KB) + CONTEXT.md checkpoints 1-19.

## P0 — CORRECTNESS. These invalidate published numbers if left.
- [ ] K1  Fix 2K5H: its eval reference row is the mutant background 2K5H.pdb_G11S (dG 1.723,
          14.6th percentile) instead of the true WT 2K5H.pdb (dG 4.806). All 1125 of its ddG
          labels are shifted ~3.0 kcal/mol. Dropping it alone removes 36% of the offset-removal
          gain (+0.2373 -> +0.1528 over 22 CSVs). Rebuild its ddG column, re-run every
          calibration number, and re-state CONTEXT.md.
- [ ] K2  Audit r18_3_TrROS_Hall (row0 at 33.3rd percentile) the same way.
- [ ] K3  Wire gate_refrow.py into the eval path so a bad reference row can never ship again.
          It currently FAILS 22/22 CSVs on 2K5H.

## P1 — THE EMBEDDING SCALE ARTIFACT. Cheapest real b_p lever found.
- [ ] K4  corr(N, per-residue embedding norm) = -0.9938. The stored ProtT5 block carries a
          1/sqrt(N) length artifact, and E = sum_i e_i is EXTENSIVE, so this injects a
          length-dependent per-protein offset directly into b_p. Verify the correlation myself.
- [ ] K5  Implement the one-line rescale (agent design E1). It rescales WT and mutant identically,
          so it CANNOT hurt a_p. Gate: byte-identical when off; b_p spread shrinks when on.
- [ ] K6  --dg_length_norm exists and is set to 'none'. b_p ~ length is r=-0.368. Test the
          existing flag before writing new code.

## P2 — MEASURE WHAT IS ALREADY RUNNING.
- [ ] K7  Score the remaining factorial cells as they finish (cpu_evals pattern, CPU partition,
          zero GPU). 10/48 done.
- [ ] K8  Read out the 8 golden-lane arms when they finish: dg_coil, slope, w5_dg, w7edge,
          u10bidir, w7span4, loroW_onehot, loroW_desc. Each answers a lever that had NEVER run.
- [ ] K9  LORO prediction on record BEFORE readout: the descriptor arm should NOT beat one-hot
          by much, because ProtT5 already predicts held-out-residue hydropathy at R^2 0.704.
- [ ] K10 Factor D is decisive (D0 coil 0.58-0.62 vs D1 uemb -0.08 to 0.31, no overlap; a_p
          0.37-0.72 vs 0.005-0.020). Decide whether to keep spending ~190 GPU-h on the D1 half.

## P3 — SCIENCE STILL OPEN.
- [ ] K11 26,315 double mutants for our 28 sit in NO split. Held-out generalisation test that
          needs no new labels. Nobody has touched it.
- [ ] K12 a_p = r * s: the slope term's ceiling is a_p = r = 0.798, and measured a_p decomposes
          into 62.8% spread compression (fixable) + 36.8% ranking error (not). Write this up.
- [ ] K13 Per-mutation analysis: mutations TO hydrophobic residues are compressed ~49% harder
          (slope 0.334 vs 0.497, corr with Kyte-Doolittle -0.833, 10/10 checkpoints) but rank
          accuracy degrades in lockstep, so it is LOST INFORMATION, not a rescalable error.
          Verify and write up.
- [ ] K14 Distribution shift as a thesis result: our regime was 0.1206% of pre-training residues,
          NMR 77.4% vs 8.6%, monomer 88.2% vs 31.8%. Report as characterisation, NOT as the cause
          of b_p (r=+0.043 against b_p, p=0.829).

## P4 — REPORTING.
- [ ] K15 Keep results/FINDINGS.md current as the single thesis-facing document.
- [ ] K16 CONTEXT.md checkpoint + push after every batch.

## DONE (verified by the main session, not by agent report)
V  13 gate suites green; gate_g4_cpu prints dG=-0.0030 width=1092 throughout
V  gate_slope 21/21 (gradient moves spread 0.8419 -> 1.6794, target 1.6837)
V  gate_loro 13/13 (filter keys on DESTINATION residue, not source)
V  gate_refrow written; fails 22/22 on 2K5H (the bug it was built to catch)
V  Scoring unblocked: 3 bugs (DEVICE='cuda' hard-coded, load_checkpoint format, map_location)
   -> 0/48 CSVs became 10/48, scored on CPU at zero GPU cost
V  a_p logging added to validate() -- 40 slope runs were previously unfalsifiable because
   pooled dG PCC is scale-invariant and cannot see a spread change
V  Attenuation DEAD by 200x: sigma=0.0327 kcal/mol over 28,312 mutations predicts a_p=0.9985
   against measured ~0.50. a_p is a real model failure. (28/28 proteins joined.)
V  ProtT5 is CONTEXTUAL (cosine 0.13-0.18 same residue, different position) and per-variant
   (one mutation changes all 52 positions, so it does NOT cancel in ddG)
V  ProtT5 predicts held-out-residue hydropathy at R^2=0.704 -> W6 is largely redundant
V  2K5H reference-row bug found and quantified
V  Exposure->a_p survives the length confound: +0.6623 partial, 10/10 checkpoints
V  100k catalogue retired for joins: it has NO sequence column (0/226, 0/21, 0/340)
V  Ofir's mean-gap mechanism does not carry over (gap -0.1034, p=0.588)
V  Autopilot: load_state hardened against '{}', wrapper stops only on rc=3, now a 7-day SLURM job
