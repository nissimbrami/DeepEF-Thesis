# Action D — instrument validation (calib_diag.py) — GATE FAILED

Date: 2026-09-05. Operator: cluster agent. Machine: slurm.bgu.ac.il (login node, anaconda base python).

## What was run
    python cluster_run/code/calib_diag.py --csv <eval.csv>
CSVs: /groups/keasar_group/casp15/Shahar/DeepPEF/eval_results/*.csv
(NOTE: docs point at /mnt/new_groups/keasar_group/... which no longer exists; live mount is /groups/keasar_group/...)

## Result table (all [ALL] scope, 28 proteins, groups len>=3)

| checkpoint | pooled | PP | offset-removed | AFFINE-oracle | in 0.77-0.81? |
|---|---|---|---|---|---|
| abl_calib_ctrl_e12   | 0.6062 | 0.7109 | 0.7006 | 0.7136 | NO |
| abl_ddg_nopretrain_e11 | 0.6471 | 0.7376 | 0.7395 | 0.7486 | NO |
| abl_randscratch_e12  | 0.6237 | 0.7386 | 0.7287 | 0.7061 | NO |
| abl_calib_combo_e13  | 0.6313 | 0.7412 | 0.7338 | 0.7485 | NO |
| abl_designed_w3_e8   | 0.6401 | 0.7388 | 0.7323 | 0.7742 | YES |
| abl_readout_attn_e12 | 0.6055 | 0.6838 | 0.6740 | 0.6456 | NO |
| abl_ddg_core_e9      | 0.5992 | 0.7067 | 0.7101 | 0.6721 | NO |

1 of 7 real evaluated checkpoints lands in the target range.

## The decisive detail
calib_ctrl reproduces the reference EXACTLY: pooled 0.6062 vs 0.606, PP 0.7109 vs 0.711,
std(b) 0.2850 vs the 0.29 quoted in STATE.md section 3. So the CSV is the right one and
calib_diag's pooled/PP path is correct. The disagreement is confined to the affine-oracle
claim ("all 52 checkpoints reach 0.77-0.81"), which these seven CSVs do not support.

## Status
GATE FAILED per HANDOVER Part C6 / PROMPT section 8D -> STOPPED, reported to Nissim.
No training submitted. Awaiting decision.
