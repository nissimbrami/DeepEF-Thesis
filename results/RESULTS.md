# DeepEF results - auto-generated 2026-09-09 01:50

GPU constraint: keasar QOS allows ONLY rtx_6000, max 8 per ACCOUNT (shared with the lab).
All runs: train.py, ddG objective, random init, mini-batch 64 (hardcoded), --val_frac 0.1,
held-out 28 test proteins, epoch selected on validation by run_calib_eval.sh.

## Phase 1 - reference reproduction   GATE: pooled ~0.606, PP ~0.711

| tag | pooled | PP | std(b) | slope min/med/max | offset-rm | affine |
|---|---|---|---|---|---|---|
| calib_ctrl_repro2 | 0.5910 | 0.7310 | 0.2232 | 0.079 / 0.498 / 1.257 | 0.7113 | 0.5694 |

## Phase 2 - seed noise (sigma)

| tag | pooled | PP | std(b) | slope min/med/max | offset-rm | affine |
|---|---|---|---|---|---|---|
| sigma_seed42 | 0.5929 | 0.7305 | 0.1912 | 0.099 / 0.437 / 1.193 | 0.7016 | 0.6576 |
| sigma_seed1 | 0.5948 | 0.7229 | 0.1862 | 0.057 / 0.407 / 1.055 | 0.7005 | 0.4876 |
| sigma_seed2 | 0.6575 | 0.7422 | 0.1717 | 0.082 / 0.430 / 0.850 | 0.7503 | 0.7705 |
| sigma_seed3 | 0.5799 | 0.7332 | 0.1748 | 0.073 / 0.339 / 0.803 | 0.7178 | 0.5706 |
| sigma_seed4 | 0.6008 | 0.7315 | 0.1535 | 0.070 / 0.327 / 0.784 | 0.7204 | 0.6440 |

POOLED mean=0.6052 sigma=0.0270  (n=5)
PP     mean=0.7321 sigma=0.0062
SIGMA RULE: sigma=0.0270 -> Phase-3 seeds per cell = 5

## Phase 4 - WT-anchor trade-off (seed 42)

| tag | pooled | PP | std(b) | slope min/med/max | offset-rm | affine |
|---|---|---|---|---|---|---|
| anchor_w0.3_s42 | 0.6157 | 0.7314 | 0.2608 | 0.109 / 0.580 / 1.272 | 0.7294 | 0.7672 |
| anchor_w1.0_s42 | 0.6018 | 0.7065 | 0.2718 | 0.123 / 0.589 / 1.209 | 0.7066 | 0.7435 |
| anchor_w3.0_s42 | 0.5972 | 0.6441 | 0.0610 | 0.019 / 0.153 / 0.275 | 0.6834 | 0.6384 |

POOLED mean=0.6049 sigma=0.0079  (n=3)
PP     mean=0.6940 sigma=0.0367
SIGMA RULE: sigma=0.0079 -> Phase-3 seeds per cell = 3

## Queue
```
     JOBID                     NAME     STATE    TIME NODELIST(REASON)
  21107688                autopilot   RUNNING 1-01:23:01 ise-cpu-intl-14
  21144483                   p0c_v2   RUNNING 1:02:39 ise-cpu-intl-26
  21136488               score_all1   RUNNING 5:26:58 ise-cpu-intl-09
  21136489               score_all2   RUNNING 5:26:58 ise-cpu-intl-14
  21136490               score_all3   RUNNING 5:26:58 ise-cpu-intl-14
  21134976               cpu_evals3   RUNNING 5:56:06 ise-cpu-intl-25
  21050603              G4_w5_smoke   PENDING    0:00 (QOSMaxGRESPer
  21050159 DeepEF_sa_uembmean_seed4   PENDING    0:00 (QOSMaxGRESPer
  21024050       ev_p3_slope1.0_s42   PENDING    0:00 (QOSMaxGRESPer
  21049396 DeepEF_p3_a0_d0_s1_D0_co   PENDING    0:00 (QOSMaxGRESPer
  21049394 DeepEF_p3_a0_d0_s0_D0_co   PENDING    0:00 (QOSMaxGRESPer
  21024048       ev_p3_slope0.3_s42   PENDING    0:00 (QOSMaxGRESPer
  21049416 ev_p3_a1_d1_s0_D0_coil_s   PENDING    0:00 (QOSMaxGRESPer
  21049401 ev_p3_a0_d1_s1_D0_coil_s   PENDING    0:00 (QOSMaxGRESPer
  21049414 ev_p3_a1_d0_s1_D0_coil_s   PENDING    0:00 (QOSMaxGRESPer
  21049412 ev_p3_a1_d0_s0_D0_coil_s   PENDING    0:00 (QOSMaxGRESPer
  21049418 ev_p3_a1_d1_s1_D0_coil_s   PENDING    0:00 (QOSMaxGRESPer
  21050050 ev_p3_a0_d1_s0_D1_uemb_s   PENDING    0:00 (QOSMaxGRESPer
  21050046 ev_p3_a0_d0_s0_D1_uemb_s   PENDING    0:00 (QOSMaxGRESPer
  21050052 ev_p3_a0_d1_s1_D1_uemb_s   PENDING    0:00 (QOSMaxGRESPer
  21050048 ev_p3_a0_d0_s1_D1_uemb_s   PENDING    0:00 (QOSMaxGRESPer
  21050058 ev_p3_a1_d1_s0_D1_uemb_s   PENDING    0:00 (QOSMaxGRESPer
  21050060 ev_p3_a1_d1_s1_D1_uemb_s   PENDING    0:00 (QOSMaxGRESPer
  21050056 ev_p3_a1_d0_s1_D1_uemb_s   PENDING    0:00 (QOSMaxGRESPer
  21050064 ev_p3_a0_d0_s0_D0_coil_s   PENDING    0:00 (QOSMaxGRESPer
  21050054 ev_p3_a1_d0_s0_D1_uemb_s   PENDING    0:00 (QOSMaxGRESPer
  21050072 ev_p3_a0_d0_s1_D0_coil_s   PENDING    0:00 (QOSMaxGRESPer
  21050066 ev_p3_a0_d0_s0_D0_coil_s   PENDING    0:00 (QOSMaxGRESPer
  21050075 ev_p3_a0_d1_s0_D0_coil_s   PENDING    0:00 (QOSMaxGRESPer
  21050069 ev_p3_a0_d0_s1_D0_coil_s   PENDING    0:00 (QOSMaxGRESPer
  21050079 ev_p3_a0_d1_s0_D0_coil_s   PENDING    0:00 (QOSMaxGRESPer
  21050082 ev_p3_a0_d1_s1_D0_coil_s   PENDING    0:00 (QOSMaxGRESPer
  21050085 ev_p3_a0_d1_s1_D0_coil_s   PENDING    0:00 (QOSMaxGRESPer
  21050087 ev_p3_a1_d0_s0_D0_coil_s   PENDING    0:00 (QOSMaxGRESPer
  21050093 ev_p3_a1_d0_s1_D0_coil_s   PENDING    0:00 (QOSMaxGRESPer
  21050090 ev_p3_a1_d0_s0_D0_coil_s   PENDING    0:00 (QOSMaxGRESPer
  21050095 ev_p3_a1_d0_s1_D0_coil_s   PENDING    0:00 (QOSMaxGRESPer
  21050100 ev_p3_a1_d1_s0_D0_coil_s   PENDING    0:00 (QOSMaxGRESPer
  21050098 ev_p3_a1_d1_s0_D0_coil_s   PENDING    0:00 (QOSMaxGRESPer
  21050160    ev_sa_uembmean_seed42   PENDING    0:00   (Dependency)
  21050158        ev_sa_coil_seed42   PENDING    0:00   (Dependency)
  21050110 ev_p3_a0_d0_s1_D1_uemb_s   PENDING    0:00   (Dependency)
  21050108 ev_p3_a0_d0_s0_D1_uemb_s   PENDING    0:00   (Dependency)
  21050106 ev_p3_a0_d0_s0_D1_uemb_s   PENDING    0:00   (Dependency)
  21050104 ev_p3_a1_d1_s1_D0_coil_s   PENDING    0:00   (Dependency)
  21050102 ev_p3_a1_d1_s1_D0_coil_s   PENDING    0:00 (QOSMaxGRESPer
  21049399 ev_p3_a0_d1_s0_D0_coil_s   PENDING    0:00   (Dependency)
  21049397 ev_p3_a0_d0_s1_D0_coil_s   PENDING    0:00   (Dependency)
  21049395 ev_p3_a0_d0_s0_D0_coil_s   PENDING    0:00   (Dependency)
  21049398 DeepEF_p3_a0_d1_s0_D0_co   PENDING    0:00  (JobHeldUser)
  21050103 DeepEF_p3_a1_d1_s1_D0_co   RUNNING 7:57:36    ise-6000-08
  21050157    DeepEF_sa_coil_seed42   RUNNING   15:53  ise-cpu256-17
  21050109 DeepEF_p3_a0_d0_s1_D1_ue   RUNNING 5:57:43     cs-6000-01
  21050107 DeepEF_p3_a0_d0_s0_D1_ue   RUNNING 6:18:18  ise-cpu256-07
  21050105 DeepEF_p3_a0_d0_s0_D1_ue   RUNNING 6:55:53     cs-6000-02
  21137938          gld_slope_w5_s1   PENDING    0:00 (MaxGRESPerAcc
  21137937             gld_w5_dg_s3   PENDING    0:00 (MaxGRESPerAcc
  21137936        gld_w12_slope_s42   PENDING    0:00 (MaxGRESPerAcc
  21137933           gld_w12_dg_s42   PENDING    0:00 (MaxGRESPerAcc
  21137932               gld_w12_s2   PENDING    0:00 (MaxGRESPerAcc
  21137931               gld_w12_s1   PENDING    0:00 (MaxGRESPerAcc
  21137930              gld_w12_s42   PENDING    0:00 (MaxGRESPerAcc
  21144538          gld_u10bidir_s1   PENDING    0:00 (MaxGRESPerAcc
  21144537           gld_w7span4_s1   PENDING    0:00 (MaxGRESPerAcc
  21144536            gld_w7edge_s1   PENDING    0:00 (MaxGRESPerAcc
  21144535           gld_ligand_s42   PENDING    0:00 (MaxGRESPerAcc
  21144534      gld_coiledge_dg_s42   PENDING    0:00 (MaxGRESPerAcc
  21144462       gld_coil_nu588_s42   PENDING    0:00 (MaxGRESPerAcc
  21144461        gld_w12_w5_dg_s42   PENDING    0:00 (MaxGRESPerAcc
  21144460         gld_slope0.7_s42   PENDING    0:00 (MaxGRESPerAcc
  21144459          gld_slope0.5_s1   PENDING    0:00 (MaxGRESPerAcc
  21144458         gld_slope0.5_s42   PENDING    0:00 (MaxGRESPerAcc
  21144774     gld_pooled_slope_s42   PENDING    0:00 (MaxGRESPerAcc
  21144773        gld_pooled1.0_s42   PENDING    0:00 (MaxGRESPerAcc
  21144772        gld_pooled0.3_s42   PENDING    0:00 (MaxGRESPerAcc
  21144771      gld_readgate_dg_s42   PENDING    0:00 (MaxGRESPerAcc
  21144770       gld_readatt_dg_s42   PENDING    0:00 (MaxGRESPerAcc
  21136225          gld_slope1.0_s1   RUNNING 5:50:31    ise-6000-04
  21136226          gld_slope1.0_s2   RUNNING 5:50:31    ise-6000-04
  21136227          gld_slope1.0_s3   RUNNING 5:50:31    ise-6000-04
  21136228       gld_loroWdesc2_s42   RUNNING 5:50:31    ise-6000-05
  21136229             gld_w5_dg_s1   RUNNING 5:50:31    ise-6000-05
  21136230             gld_w5_dg_s2   RUNNING 5:50:31    ise-6000-05
  21136231         gld_slope_w5_s42   RUNNING 5:50:31    ise-6000-05
  21136232     gld_slope_anchor_s42   RUNNING 5:50:31    ise-6000-05
```
