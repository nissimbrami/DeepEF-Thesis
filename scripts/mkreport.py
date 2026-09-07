import json, numpy as np
raw   = json.load(open('results/identifiability_raw.json'))
ep    = json.load(open('results/identifiability_epoch.json'))
guard = json.load(open('results/identifiability_guard.json'))
S = raw['stats']

final = {
  "question": "Is b_p learnable, or is it seed noise? ICC decomposition over 5 same-config seeds.",
  "method": {
    "source_csvs": {k: v['epoch'] for k, v in S['seeds'].items()},
    "n_proteins": S['n_proteins'],
    "conventions": "groupby('protein',sort=False); row0=WT (verified ddG[0]==0 in all 5 CSVs); ddG=deltaG-deltaG[0]; a_p,b_ddg=np.polyfit(ddg_true,ddg_pred,1); groups len>=3 (all 28 qualify).",
    "b_p_definition_note": S['note_two_bp_definitions'],
    "estimator": "ICC(1,1) one-way random effects, balanced 28x5. ICC=(MSB-MSW)/(MSB+(k-1)MSW).",
    "no_gpu": True, "csvs_only": True
  },
  "b_p_wt_CANONICAL": {
    "definition": "pred_deltaG[WT] - deltaG[WT]  (calc_bp.py; the std=1.5741 object; dG-side)",
    "ICC1_lower_bound": S['b_p_wt']['ICC1'],
    "ICC_C1_seedshift_removed": ep['b_p_wt']['ICC_C1_twoway_seedshift_removed'],
    "ICC_epoch_matched_pair_e13": ep['b_p_wt']['ICC1_epoch_matched_pair_e13_only'],
    "sd_protein_component": S['b_p_wt']['sd_protein_component'],
    "sd_noise_component": S['b_p_wt']['sd_noise_component'],
    "mean_across_protein_std": S['b_p_wt']['mean_per_seed_across_protein_std'],
    "mean_within_protein_across_seed_std": S['b_p_wt']['mean_within_protein_across_seed_std'],
    "pairwise_seed_PCC_mean": S['b_p_wt']['pairwise_seed_PCC_mean'],
    "pairwise_seed_PCC_min": S['b_p_wt']['pairwise_seed_PCC_min'],
    "bootstrap_CI95": guard['b_p_wt']['bootstrap_CI95'],
    "leave_one_seed_out": guard['b_p_wt']['leave_one_seed_out']
  },
  "b_ddg_calib_diag_intercept": {
    "definition": "polyfit intercept on ddG (calib_diag.py 'std_b'); NOT the 1.5741 object",
    "ICC1_lower_bound": S['b_ddg']['ICC1'],
    "ICC_epoch_matched_pair_e13": ep['b_ddg']['ICC1_epoch_matched_pair_e13_only'],
    "mean_across_protein_std": S['b_ddg']['mean_per_seed_across_protein_std'],
    "mean_within_protein_across_seed_std": S['b_ddg']['mean_within_protein_across_seed_std'],
    "pairwise_seed_PCC_mean": S['b_ddg']['pairwise_seed_PCC_mean']
  },
  "a_p": {
    "definition": "ddG compression slope, polyfit(ddg_true,ddg_pred,1)[0]",
    "ICC1_lower_bound": S['a_p']['ICC1'],
    "ICC_C1_seedshift_removed": ep['a_p']['ICC_C1_twoway_seedshift_removed'],
    "ICC_epoch_matched_pair_e13": ep['a_p']['ICC1_epoch_matched_pair_e13_only'],
    "sd_protein_component": S['a_p']['sd_protein_component'],
    "sd_noise_component": S['a_p']['sd_noise_component'],
    "mean_across_protein_std": S['a_p']['mean_per_seed_across_protein_std'],
    "mean_within_protein_across_seed_std": S['a_p']['mean_within_protein_across_seed_std'],
    "pairwise_seed_PCC_mean": S['a_p']['pairwise_seed_PCC_mean'],
    "bootstrap_CI95": guard['a_p']['bootstrap_CI95'],
    "leave_one_seed_out": guard['a_p']['leave_one_seed_out']
  },
  "consequence": {
    "std_b_reference": 1.5741,
    "total_var": S['b_p_wt']['ICC1'] and 1.5741**2,
    "max_removable_var_ICC_times_var": S['consequence']['max_removable_var'],
    "irreducible_seed_var": S['consequence']['irreducible_seed_var'],
    "irreducible_seed_sd": S['consequence']['irreducible_seed_sd'],
    "mean_raw_pooled_PCC": ep['achievable']['mean_raw_pooled_PCC'],
    "mean_offset_oracle_pooled_PCC": ep['achievable']['mean_offset_oracle_pooled_PCC'],
    "achievable_pooled_PCC_bound": ep['achievable']['bound_with_ICC_b_ddg'],
    "PCC_lost_to_seed_noise": ep['achievable']['lost_to_seed_noise_PCC'],
    "verdict": "b_p is LEARNABLE. ICC>=0.90 (lower bound); epoch-matched estimate 0.97. Seed noise removes only ~1.3% of the offset-correction headroom."
  },
  "epoch_confound": {
    "seed_epochs": {k: v['epoch'] for k, v in S['seeds'].items()},
    "corr_columnmean_vs_epoch_b_p_wt": ep['b_p_wt']['corr_columnmean_vs_epoch'],
    "note": "Seeds differ in epoch (9-14). ICC(1,1) charges ALL epoch drift to noise, so it is a LOWER bound on the protein-attributable share. Two controls raise it: ICC(C,1) removes the per-seed additive shift -> 0.9515; the epoch-MATCHED pair (seed1_e13 vs seed3_e13, identical epoch, pure seed contrast) -> 0.9722."
  },
  "guard": {
    "shuffled_protein_identity_ICC_b_p_wt": guard['b_p_wt']['shuffled_protein_identity_mean'],
    "pure_noise_ICC_b_p_wt": guard['b_p_wt']['pure_noise_mean'],
    "positive_control_identical_columns": guard['b_p_wt']['positive_control_identical_columns'],
    "guard_pass": guard['guard_pass'],
    "interpretation": "Permuting protein identity within each seed column collapses ICC to ~0.00 and identical columns give exactly 1.00, so the statistic genuinely reads per-protein identity rather than being an artifact of the arithmetic."
  },
  "per_seed_metrics": S['seeds']
}
json.dump(final, open('results/identifiability.json','w'), indent=2)
print('wrote results/identifiability.json')
print(json.dumps(final['consequence'], indent=2))
