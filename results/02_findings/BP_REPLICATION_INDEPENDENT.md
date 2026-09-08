# b_p-side replication sweep - INDEPENDENT re-derivation (scripts/indep_bp.py)

SASA quadrature: Bio.PDB ShrakeRupley n_points=960 (accurate). Note that the
`frac_buried_rel_lt_0.25` row is quadrature-sensitive; at Biopython's default
n_points=100 it reads -0.475 instead of -0.327. See CHECKPOINT 11 DEFECT 1.

Sign-consistency of 10/10 has a ~60% false-positive rate here (effective N of the
10 checkpoints = 1.24). Treat the SUGGESTIVE class as noise. See DEFECT 2.

| # | feature | target | mean r | sd | signs | sig p<0.05 | range | class |
|---|---|---|---|---|---|---|---|---|
| 1 | frac_buried_rel_lt_0.25 | b_p_wt_error | -0.327 | 0.117 | 10/10- | 6/10 | -0.448..-0.120 | ESTABLISHED |
| 2 | SASA_per_residue | abs_b_p | -0.337 | 0.096 | 10/10- | 3/10 | -0.439..-0.096 | SUGGESTIVE |
| 3 | SASA_over_len_pow_073 | b_p_wt_error | +0.303 | 0.091 | 10/10+ | 3/10 | +0.155..+0.415 | SUGGESTIVE |
| 4 | mean_rel_SASA | abs_b_p_wt_error | -0.312 | 0.099 | 10/10- | 2/10 | -0.461..-0.168 | SUGGESTIVE |
| 5 | SASA_over_len_pow_073 | abs_b_p | -0.303 | 0.100 | 10/10- | 2/10 | -0.388..-0.036 | SUGGESTIVE |
| 6 | SASA_per_residue | abs_b_p_wt_error | -0.301 | 0.093 | 10/10- | 2/10 | -0.436..-0.144 | SUGGESTIVE |
| 7 | frac_charged | b_p_wt_error | +0.262 | 0.144 | 10/10+ | 2/10 | +0.002..+0.427 | SUGGESTIVE |
| 8 | SASA_per_residue | b_p_wt_error | +0.254 | 0.144 | 10/10+ | 2/10 | +0.008..+0.398 | SUGGESTIVE |
| 9 | length | abs_b_p | +0.294 | 0.081 | 10/10+ | 1/10 | +0.135..+0.401 | SUGGESTIVE |
| 10 | mean_rel_SASA | abs_b_p | -0.287 | 0.096 | 10/10- | 1/10 | -0.403..-0.067 | SUGGESTIVE |
| 11 | SASA_over_len_pow_073 | abs_b_p_wt_error | -0.272 | 0.092 | 10/10- | 1/10 | -0.382..-0.085 | SUGGESTIVE |
| 12 | frac_buried_rel_lt_0.25 | abs_b_p | +0.220 | 0.137 | 10/10+ | 1/10 | +0.025..+0.379 | SUGGESTIVE |
| 13 | mean_rel_SASA_hydrophobic | b_p_wt_error | +0.280 | 0.052 | 10/10+ | 0/10 | +0.183..+0.349 | SUGGESTIVE |
| 14 | mean_hydropathy_KD | abs_b_p | +0.280 | 0.034 | 10/10+ | 0/10 | +0.212..+0.315 | SUGGESTIVE |
| 15 | mean_rel_SASA | b_p_wt_error | +0.238 | 0.125 | 10/10+ | 0/10 | +0.013..+0.352 | SUGGESTIVE |
| 16 | frac_exposed_rel_gt_0.5 | abs_b_p | -0.235 | 0.053 | 10/10- | 0/10 | -0.293..-0.112 | SUGGESTIVE |
| 17 | frac_buried_rel_lt_0.25 | abs_b_p_wt_error | +0.232 | 0.079 | 10/10+ | 0/10 | +0.103..+0.347 | SUGGESTIVE |
| 18 | plddt_mean | b_p | +0.216 | 0.062 | 10/10+ | 0/10 | +0.117..+0.291 | SUGGESTIVE |
| 19 | frac_hydrophobic | abs_b_p | +0.213 | 0.074 | 10/10+ | 0/10 | +0.106..+0.306 | SUGGESTIVE |
| 20 | frac_charged | b_p | -0.208 | 0.094 | 10/10- | 0/10 | -0.307..-0.071 | SUGGESTIVE |
| 21 | total_SASA | abs_b_p | +0.156 | 0.052 | 10/10+ | 0/10 | +0.083..+0.229 | SUGGESTIVE |
| 22 | mean_hydropathy_KD | b_p | -0.151 | 0.086 | 10/10- | 0/10 | -0.274..-0.024 | SUGGESTIVE |
| 23 | frac_glycine | abs_b_p | +0.141 | 0.107 | 10/10+ | 0/10 | +0.025..+0.346 | SUGGESTIVE |
| 24 | frac_proline | abs_b_p_wt_error | +0.115 | 0.077 | 10/10+ | 0/10 | +0.009..+0.242 | SUGGESTIVE |
| 25 | frac_proline | b_p_wt_error | -0.113 | 0.052 | 10/10- | 0/10 | -0.175..-0.034 | SUGGESTIVE |
| 26 | frac_proline | b_p | +0.102 | 0.070 | 10/10+ | 0/10 | +0.018..+0.211 | SUGGESTIVE |
| 27 | plddt_min | b_p | +0.084 | 0.026 | 10/10+ | 0/10 | +0.054..+0.137 | SUGGESTIVE |
| 28 | frac_exposed_rel_gt_0.5 | b_p | +0.084 | 0.061 | 10/10+ | 0/10 | +0.022..+0.193 | SUGGESTIVE |
| 29 | mean_rel_SASA_hydrophobic | abs_b_p | +0.030 | 0.027 | 10/10+ | 0/10 | +0.004..+0.084 | SUGGESTIVE |
| 30 | length | abs_b_p_wt_error | +0.260 | 0.139 | 9/10+ | 2/10 | -0.070..+0.424 | DEAD |
| 31 | frac_exposed_rel_gt_0.5 | abs_b_p_wt_error | -0.257 | 0.111 | 9/10- | 2/10 | -0.387..+0.013 | DEAD |
| 32 | plddt_mean | abs_b_p_wt_error | +0.218 | 0.112 | 9/10+ | 0/10 | -0.020..+0.359 | DEAD |
| 33 | mean_rel_SASA_hydrophobic | abs_b_p_wt_error | -0.179 | 0.127 | 9/10- | 0/10 | -0.346..+0.092 | DEAD |
| 34 | frac_charged | abs_b_p | -0.148 | 0.107 | 9/10- | 0/10 | -0.246..+0.034 | DEAD |
| 35 | frac_glycine | abs_b_p_wt_error | -0.137 | 0.109 | 9/10- | 0/10 | -0.258..+0.103 | DEAD |
| 36 | plddt_min | abs_b_p_wt_error | +0.116 | 0.090 | 9/10+ | 0/10 | -0.059..+0.244 | DEAD |
| 37 | frac_charged | abs_b_p_wt_error | -0.090 | 0.072 | 9/10- | 0/10 | -0.201..+0.009 | DEAD |
| 38 | mean_hydropathy_KD | abs_b_p_wt_error | +0.074 | 0.083 | 9/10+ | 0/10 | -0.102..+0.171 | DEAD |
| 39 | frac_proline | abs_b_p | -0.070 | 0.070 | 9/10- | 0/10 | -0.131..+0.115 | DEAD |
| 40 | plddt_min | b_p_wt_error | +0.063 | 0.057 | 9/10+ | 0/10 | -0.045..+0.157 | DEAD |
| 41 | mean_rel_SASA_hydrophobic | b_p | +0.035 | 0.029 | 9/10+ | 0/10 | -0.022..+0.077 | DEAD |
| 42 | total_SASA | abs_b_p_wt_error | +0.144 | 0.181 | 8/10+ | 0/10 | -0.272..+0.297 | DEAD |
| 43 | plddt_mean | b_p_wt_error | -0.143 | 0.111 | 8/10- | 0/10 | -0.259..+0.039 | DEAD |
| 44 | mean_hydropathy_KD | b_p_wt_error | +0.099 | 0.111 | 8/10+ | 0/10 | -0.031..+0.279 | DEAD |
| 45 | frac_hydrophobic | b_p_wt_error | +0.088 | 0.096 | 8/10+ | 0/10 | -0.020..+0.264 | DEAD |
| 46 | length | b_p_wt_error | -0.137 | 0.182 | 7/10- | 0/10 | -0.318..+0.160 | DEAD |
| 47 | frac_buried_rel_lt_0.25 | b_p | +0.131 | 0.118 | 7/10+ | 0/10 | -0.071..+0.253 | DEAD |
| 48 | frac_glycine | b_p_wt_error | +0.074 | 0.079 | 7/10+ | 0/10 | -0.068..+0.157 | DEAD |
| 49 | frac_exposed_rel_gt_0.5 | b_p_wt_error | +0.069 | 0.119 | 7/10+ | 0/10 | -0.127..+0.214 | DEAD |
| 50 | frac_hydrophobic | abs_b_p_wt_error | +0.059 | 0.116 | 7/10+ | 0/10 | -0.144..+0.238 | DEAD |
| 51 | total_SASA | b_p | +0.032 | 0.070 | 7/10+ | 0/10 | -0.092..+0.158 | DEAD |
| 52 | SASA_over_len_pow_073 | b_p | -0.011 | 0.090 | 7/10- | 0/10 | -0.100..+0.136 | DEAD |
| 53 | plddt_mean | abs_b_p | +0.075 | 0.110 | 6/10+ | 0/10 | -0.112..+0.191 | DEAD |
| 54 | length | b_p | +0.040 | 0.094 | 6/10+ | 0/10 | -0.089..+0.175 | DEAD |
| 55 | total_SASA | b_p_wt_error | +0.038 | 0.175 | 6/10- | 0/10 | -0.136..+0.315 | DEAD |
| 56 | mean_rel_SASA | b_p | -0.014 | 0.098 | 6/10- | 0/10 | -0.116..+0.145 | DEAD |
| 57 | SASA_per_residue | b_p | -0.014 | 0.101 | 6/10- | 0/10 | -0.119..+0.145 | DEAD |
| 58 | frac_hydrophobic | b_p | -0.057 | 0.118 | 5/10- | 0/10 | -0.244..+0.112 | DEAD |
| 59 | plddt_min | abs_b_p | -0.006 | 0.051 | 5/10- | 0/10 | -0.101..+0.053 | DEAD |
| 60 | frac_glycine | b_p | +0.006 | 0.068 | 5/10- | 0/10 | -0.076..+0.144 | DEAD |
