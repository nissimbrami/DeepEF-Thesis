# SLOPE ARM (factor C, `--slope_weight`) — READY TO SUBMIT, NOT SUBMITTED

Prepared 2026-09-07. CPU only. **Nothing was submitted. No job was cancelled. No push.**
Everything below was measured by running it; the scripts are listed in section 7.

---

## 0. HEADLINE — read before spending GPU

Two things changed from the brief:

1. **The premise "`--slope_weight` has NEVER been run once" is FALSE.**
   `abl_p3_slope3.0_s42_e8` is a real `--slope_weight 3.0` run (Phase-3 pilot;
   `cluster_run/scripts/submit_factorial.sh:84` passes `--slope_weight ${CW}`). Its eval CSV is on
   disk and has been sitting in the 10-checkpoint replication table all along.
2. **That run FAILED its own pre-registered gate, and failed it backwards.**
   The gate (`results/CONTEXT.md:201`) was: *move `std(pred_ddG)/std(true_ddG)` toward 1 without
   hurting pooled.* It moved the ratio AWAY from 1 and hurt pooled.

| run | sd_ratio (target 1.0) | a_p median | pooled ddG PCC | per-protein PCC |
|---|---|---|---|---|
| `--slope_weight 3.0` (`abl_p3_slope3.0_s42_e8`) | **0.4489** | 0.3253 | **0.5624** | 0.7943 |
| control, same seed (`abl_sigma_seed42_e9`) | 0.5573 | 0.4366 | 0.5929 | 0.7921 |
| reference control (`abl_calib_ctrl_repro2_e14`) | 0.6342 | 0.4975 | 0.5910 | 0.7955 |

Weight 3.0 produced **more** compression, not less. Pooled fell 0.593 -> 0.562. Per-protein PCC
did not move (0.7921 -> 0.7943).

3. **The oracle says the slope channel cannot buy ddG PCC even in principle** (section 3).
   Per-protein PCC is *exactly* invariant to a per-protein rescale, so a perfect slope correction
   changes ranking by **zero**.

**Recommendation: this arm is NOT worth GPU as an accuracy lever, and the command in section 5
should not be run to chase pooled PCC.** It is defensible only as a *calibration* arm scored on
`sd_ratio`, at a weight far below 3.0.

**And the section-5 command is very likely unnecessary anyway.** All three pilot weights
{0.3, 1.0, 3.0} are already trained (15 checkpoints each), and the two missing eval jobs
`ev_p3_slope0.3_s42` and `ev_p3_slope1.0_s42` are **already sitting in the queue** (verified with
`squeue`, read-only). The full weight ladder is arriving for zero additional GPU. Wait for those
two CSVs before submitting anything — see section 6.

---

## 1. What the term actually computes (`Megascale-fineTuning/train.py:519-527`)

```python
if SLOPE_WEIGHT > 0 and output.numel() >= 2:
    wt_dg_slope, _, _ = self.get_wt_deltaG(batch)
    delta_g_wt_slope  = batch['delta_g'][0, 0].to(self.device)
    slope_pred_ddg    = output - wt_dg_slope
    slope_true_ddg    = delta_g - delta_g_wt_slope
    slope_loss = torch.abs(slope_pred_ddg.std(unbiased=False)
                           - slope_true_ddg.std(unbiased=False))
...
loss = data_loss + reg_loss + energy_reg + WT_ANCHOR_WEIGHT * wt_anchor_loss
if SLOPE_WEIGHT > 0 and output.numel() >= 2:
    loss = loss + SLOPE_WEIGHT * slope_loss
```

In words: **an L1 penalty on the difference between the population standard deviation of the
predicted ddG and of the true ddG, within the current mini-batch of one protein.** WT is row 0.
It is a *spread-matching* term. Verified on synthetic tensors (TEST 1-5):

- **It has a true minimum in the right place.** True sd fixed at 1.500, pred = a x true:
  a=0.10 -> loss 1.350; a=0.4990 -> 0.7515; **a=1.0 -> 0.000**; a=1.25 -> 0.375; a=2.0 -> 1.500.
- **The gradient points the right way.** Starting at the measured median compression a=0.4990 and
  running SGD on the slope term alone, a climbs 0.499 -> 0.874 -> 0.949 -> ~1.0.
- **It supplies NO ranking signal.** slope_loss = 0.00000000 for pred = true, for pred = **-true**
  (PCC -1.0), and for a **shuffled** pred (PCC -0.30). It only matches spread, so it is useless
  alone and must ride on a data loss — which is why the guard in section 2 matters.

Two implementation facts that are not obvious and that change how the arm should be read:

- **The WT subtraction is arithmetically inert.** `get_wt_deltaG` (train.py:698-709) returns a
  1-element tensor, i.e. a per-protein *scalar*, and `std(x - c) == std(x)` for scalar c. Measured:
  loss with the WT subtraction 0.5999999642 vs `|std(output)-std(delta_g)|` 0.5999999642,
  difference **0.000e+00**; gradients through a shared parameter agree to 1.2e-07. So the term is
  really `| std(pred_dG) - std(true_dG) |` over the minibatch, and the extra WT forward pass it
  triggers is **wasted compute** (one folded+unfolded forward per minibatch).
- **The gradient magnitude is constant** (an L1 term): d(loss)/da = ±1.5 regardless of how close a
  is to 1. It does not anneal near the optimum — it oscillates across it (observed: a bounces
  0.949 <-> 1.024 indefinitely at lr 0.05). **A large weight will chatter rather than converge**,
  which is a mechanical explanation for why weight 3.0 overshot into *more* compression. This
  argues for a small weight, not the anchor ladder {0.3, 1.0, 3.0} borrowed from factor A.
- `unbiased=False` makes a 1-element std `0.0`, not `nan`, so the `numel() >= 2` guard is
  belt-and-braces. Verified.

---

## 2. THE MISSING GUARD — added and verified firing

`--slope_weight` with `--loss_mode dg` was previously accepted and would train happily: the slope
term would ride as an unscored side-objective on a loss that never forms a within-protein
contrast. That is this project's signature failure mode. **Now it raises.**

Inserted in `Megascale-fineTuning/train.py` immediately after the verbatim anchor
`LOSS_MODE = _a.loss_mode` (anchor asserted to match **exactly once** before writing; backup at
`train.py.bak_slopeguard`; `ast.parse` OK). It sits after `LOSS_MODE` and after `SLOPE_WEIGHT`
(line 97) — that ordering is asserted by the patch script, else the guard would `NameError`:

```python
if SLOPE_WEIGHT > 0 and LOSS_MODE == 'dg':
    raise ValueError(
        '--slope_weight %g requires --loss_mode ddg|joint|ddg_head; got --loss_mode dg. '
        'The slope term penalises |std(pred_ddG) - std(true_ddG)| within a protein, which '
        'is undefined under a pure dG loss. Use --loss_mode ddg (or joint) for the slope arm.'
        % SLOPE_WEIGHT)
```

**Verified 7/7.** Note the first attempt at this test was itself a false pass: run from
`Megascale-fineTuning/`, the script died at `ModuleNotFoundError: No module named 'model'` on line
17, so all seven cases reported "did not raise" without ever reaching the guard. train.py must be
run from the repo root. The harness was rewritten to detect early-import death explicitly.

| case | expected | got |
|---|---|---|
| `--slope_weight 3.0 --loss_mode dg` | RAISED | **RAISED** |
| `--slope_weight 3.0` (dg is the default) | RAISED | **RAISED** |
| `--slope_weight 0.5 --loss_mode dg` | RAISED | **RAISED** |
| `--slope_weight 3.0 --loss_mode ddg` | runs | ran past guard |
| `--slope_weight 3.0 --loss_mode joint` | runs | ran past guard |
| `--slope_weight 0.0 --loss_mode dg` | runs | ran past guard |
| no slope flag (baseline) | runs | ran past guard |

---

## 3. ORACLE: what a slope correction could buy — **nothing, on ddG PCC**

**These are ORACLES, not methods.** `a_p` and `b_p` are fitted by `np.polyfit(ddg_true, ddg_pred, 1)`
on the **test labels of the very proteins being scored** (calib_diag / catalogue_vs_bp convention;
groupby protein, WT = row 0). Dividing by an `a_p` fitted on test labels is not reproducible and
cannot be claimed as model performance. All 10 eval CSVs, 28 proteins / 28,314 variants each.

Pooled ddG PCC:

| checkpoint | raw | offset (`p-b_p`) | **slope (`p/a_p`)** | slope clip 0.25 | affine | a_min | a_med |
|---|---|---|---|---|---|---|---|
| abl_anchor_w0.3_s42_e14 | 0.6157 | 0.7294 | 0.6473 | 0.6606 | 0.7672 | 0.1095 | 0.5802 |
| abl_anchor_w1.0_s42_e14 | 0.6018 | 0.7066 | 0.6369 | 0.6475 | 0.7435 | 0.1228 | 0.5889 |
| abl_anchor_w3.0_s42_e13 | 0.5972 | 0.6834 | 0.5840 | 0.6009 | 0.6384 | 0.0187 | 0.1535 |
| abl_calib_ctrl_repro2_e14 | 0.5910 | 0.7113 | **0.3219** | 0.5472 | 0.5694 | 0.0789 | 0.4975 |
| abl_p3_slope3.0_s42_e8 | 0.5624 | 0.6974 | 0.4239 | 0.5744 | 0.6407 | 0.1050 | 0.3253 |
| abl_sigma_seed1_e13 | 0.5948 | 0.7005 | **0.2421** | 0.5809 | 0.4876 | 0.0565 | 0.4068 |
| abl_sigma_seed2_e10 | 0.6575 | 0.7503 | 0.6556 | 0.6870 | 0.7705 | 0.0823 | 0.4301 |
| abl_sigma_seed3_e13 | 0.5799 | 0.7178 | 0.3075 | 0.5768 | 0.5706 | 0.0733 | 0.3387 |
| abl_sigma_seed42_e9 | 0.5929 | 0.7016 | 0.4548 | 0.5938 | 0.6576 | 0.0986 | 0.4366 |
| abl_sigma_seed4_e14 | 0.6008 | 0.7204 | 0.4217 | 0.6125 | 0.6440 | 0.0703 | 0.3271 |
| **mean** | **0.5994** | **0.7119** | **0.4696** | **0.6082** | **0.6490** | | |

**The slope oracle makes pooled ddG PCC WORSE on 10/10 checkpoints: 0.599 -> 0.470.**
Offset removal lifts it to 0.712, replicating the established number. Even the *affine* oracle
(0.649) sits below plain offset removal (0.712) — the `/a_p` step actively destroys what `-b_p`
gained.

### Why — this is the part that decides the arm

**Per-protein PCC is EXACTLY invariant to `pred/a_p`.** Dividing one protein's predictions by a
positive constant is a positive affine map and cannot change that protein's own Pearson r.
Measured over all 28 proteins: **max |PCC_raw - PCC_after| = 7.772e-16** (floating-point zero).

So a slope correction has **exactly zero** ability to improve within-protein ranking. Its only
effect on a *pooled* number is to rescale proteins relative to one another, and that is
destructive: pooled sd goes 0.744 -> 2.559, with the per-protein inflation factor `1/a_p` ranging
0.80 to 12.67, so badly-calibrated proteins are blown up ~16x more than well-calibrated ones and
the cross-protein comparison that pooled PCC measures gets scrambled.

**Honest conclusion: a_p is real (median sd_ratio 0.6342; 6/28 proteins below 0.5 — the model
genuinely compresses ddG spread ~2x), but correcting it cannot buy ddG PCC, pooled or
per-protein. Fixing a_p is a CALIBRATION result, not an accuracy result.**

### Which proteins blow up (unclipped `pred/a_p`), reference checkpoint

| protein | n | a_p | 1/a_p | max abs(pred/a_p) |
|---|---|---|---|---|
| HEEH_KT_rd6_0793 | 796 | **0.0789** | 12.67 | 13.65 |
| 2KVS | 1207 | 0.0939 | 10.65 | **30.42** |
| HHH_rd1_0142 | 804 | 0.2638 | 3.79 | 4.93 |
| 1QKH | 1272 | 0.2670 | 3.74 | 6.56 |
| HHH_rd1_0244 | 771 | 0.3141 | 3.18 | 4.83 |
| 3DKM | 1239 | 0.3177 | 3.15 | 8.32 |
| 2KXD | 1173 | 0.3965 | 2.52 | 8.95 |
| r18_3_TrROS_Hall | 990 | 0.3986 | 2.51 | 7.10 |

`2KVS` is the worst blow-up: a predicted ddG of magnitude 30.4 kcal/mol, physically absurd.
`abl_anchor_w3.0_s42_e13` is the pathological checkpoint — **a_min = 0.0187 (1/a_p = 53x) and 25
of its 28 proteins fall below the 0.25 clip.**

**With vs without clipping.** Clipping repairs most but not all of the damage and never recovers
to raw:

| clip | mean pooled slope PCC |
|---|---|
| 0.00 (none) | +0.4696 |
| 0.05 | +0.4753 |
| 0.10 | +0.5048 |
| 0.20 | +0.5882 |
| 0.25 | +0.6082 |
| 0.30 | +0.6199 |
| **0.40** | **+0.6264** (best) |
| 0.50 | +0.6237 |

Exact comparison at the best clip: clipped slope oracle **+0.6264**, raw **+0.5994**, offset oracle
**+0.7119**. So the best-case clipped slope oracle beats raw by only **+0.027** and still loses to
plain offset removal by **-0.086** — and that +0.027 requires knowing every test-set `a_p` in
advance *and* hand-tuning the clip on the same data, which is two layers of oracle.

Leave-one-out on the single worst protein is **not** the story: dropping `HEEH_KT_rd6_0793`
(a_p=0.0789) moves the reference checkpoint only 0.3219 -> 0.3356. **The damage is broad, not one
outlier** — which is why clipping (which touches many proteins) helps and LOO does not.

---

## 4. Scoring rule for this arm (metric discipline)

Per the project metric rule, ddG cancels anything identical between WT and mutant, so a
reference-state / whole-protein lever must be scored on dG or b_p — but **a_p is a within-protein
slope and does NOT cancel**, so it is legitimately measurable on ddG. That licence is what makes
this arm scoreable at all. Section 3 nevertheless shows the *ddG PCC* metric is blind to it.

**Therefore the slope arm MUST be scored primarily on `std(pred_ddG)/std(true_ddG)` per protein
(target 1.0), with pooled and per-protein ddG PCC reported as GUARDRAILS that must not degrade.**
Reporting only pooled PCC would show a loss and hide whether the term did its job — exactly the
trap `results/CONTEXT.md:202-204` warned about.

---

## 5. THE EXACT SBATCH COMMAND — **DO NOT RUN. The user decides when this goes in.**

Follows every site rule from `cluster_run/scripts/submit_factorial.sh`: `--qos normal` (keasar caps
the lab at 8), `--gres=gpu:rtx_6000:1` (**never a partition name** — the submit filter reroutes
onto a 1080 and OOMs), `export WANDB_MODE=disabled` (omitting it kills the run at wandb login), the
pheno exclude list, a unique run_tag, and a chained eval with `--dependency afterok`.

Run tag `p3_slopeW0.3_s42` is **unique**: existing tags are `p3_slope0.3_s42`, `p3_slope1.0_s42`,
`p3_slope3.0_s42` (all already trained), so this does not collide and the `[ -d "$MODEL_DIR" ]`
idempotence check will not skip it.

Weight **0.3**, not 3.0: 3.0 has been run and failed backwards (section 0), and the
constant-magnitude L1 gradient (section 1) predicts large weights chatter rather than converge.

```bash
cd /home/nissimb/DeepPEF

RUN_TAG=p3_slopeW0.3_s42
MODEL_DIR=./Megascale-fineTuning/models/PEM_fine_tuned-trianed_models-light_attentionkf_${RUN_TAG}
EXCLUDE="--exclude=ise-pheno-01,ise-pheno-02,ise-pheno-03,ise-pheno-04,ise-pheno-05,ise-pheno-06,ise-pheno-07,ise-pheno-08,ise-pheno-09,ise-pheno-10,ise-pheno-11,ise-pheno-12"

# ---- TRAIN ----
JID=$(sbatch --parsable --job-name "DeepEF_${RUN_TAG}" \
  --qos normal --gres=gpu:rtx_6000:1 --cpus-per-task=8 --time 1-00:00:00 $EXCLUDE \
  --output "logs/${RUN_TAG}_%j.out" --error "logs/${RUN_TAG}_%j.out" \
  --wrap "module load anaconda; source activate esm2_env_py38; \
export WANDB_MODE=disabled; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
python Megascale-fineTuning/train.py \
  --full_data --no_pretrain --no_freeze --loss_mode ddg \
  --pooled_corr_weight 0 --dg_length_norm none --affine_calib \
  --wt_anchor_weight 0 --designed_weight 1 --slope_weight 0.3 \
  --val_frac 0.1 --epochs 15 --seed 42 --run_tag ${RUN_TAG}")
echo "train job: $JID"

# ---- EVAL, chained afterok ----
sbatch --parsable --job-name "ev_${RUN_TAG}" \
  --qos normal --gres=gpu:rtx_6000:1 --cpus-per-task=8 --time 06:00:00 $EXCLUDE \
  --dependency "afterok:${JID}" \
  --output "logs/ev_${RUN_TAG}_%j.out" --error "logs/ev_${RUN_TAG}_%j.out" \
  --wrap "module load anaconda; source activate esm2_env_py38; \
export WANDB_MODE=disabled; mkdir -p eval_results; \
bash Megascale-fineTuning/run_calib_eval.sh 'logs/${RUN_TAG}_${JID}.out' '${MODEL_DIR}' '${RUN_TAG}'; \
ls -s eval_results/abl_${RUN_TAG}_e*.csv >/dev/null 2>&1 || { echo 'EVAL PRODUCED NO CSV'; exit 1; }"
```

`--loss_mode ddg` is mandatory — with `dg` the new guard raises immediately (section 2).
The eval must produce a CSV or fail loudly: three 7h+ runs previously reached COMPLETED with no
CSV. A finished training run is not a result.

---

## 6. What would have to be true for this arm to be worth GPU

Given section 3, do **not** submit this expecting a pooled-PCC win. Submit only to answer the
calibration question, and only if these hold:

1. `sd_ratio -> 1.0` is accepted as the primary endpoint, with ddG PCC as a non-degradation
   guardrail.
2. A low weight is used. 3.0 is refuted; 0.3 is the proposal, and even that is a guess — the
   constant-gradient analysis suggests the useful range is below 1.0.
3. **The weight ladder is already arriving — wait for it, and submit NOTHING.** Verified on the
   cluster:

   | tag | trained checkpoints | eval CSV | eval job |
   |---|---|---|---|
   | `p3_slope0.3_s42` | 15 | **0** | **`ev_p3_slope0.3_s42` ALREADY QUEUED** |
   | `p3_slope1.0_s42` | 15 | **0** | **`ev_p3_slope1.0_s42` ALREADY QUEUED** |
   | `p3_slope3.0_s42` | 15 | 1 | done (analysed in section 0) |

   All three weights are trained, and the two missing evals are **already in the queue**.
   Submitting anything for them would be a duplicate. When those two CSVs land, re-run
   `scratch_slope/slope_ran.py` extended to all three weights and the ladder
   {0.3, 1.0, 3.0} is complete **for zero additional GPU**.
   **That is the correct next step, and it requires no submission at all** — which makes the
   section-5 command redundant unless the ladder shows a weight below 0.3 is wanted.

The larger point for the thesis: the exposure -> a_p finding (r = +0.714, replicated 10/10) is a
finding about *why the model is miscalibrated*, and it stands. What section 3 removes is the hope
that fixing a_p raises the headline ddG correlation. Those are different claims, and only the
first is supported.

---

## 7. Verification — every script run, on the cluster, CPU only

```
scratch_slope/slope_synth.py         slope term on synthetic tensors (TEST 1-5)
scratch_slope/patch_slope_guard.py   anchor asserted exactly once, backup, ast.parse OK
scratch_slope/slope_oracle.py        10 eval CSVs, oracle table + clip sweep + blow-ups
scratch_slope/slope_why.py           per-protein PCC invariance (max diff 7.772e-16)
scratch_slope/slope_ran.py           the p3_slope3.0 run vs its controls
scripts/gate_g4_cpu.py               REQUIRED GATE
```

`python scripts/gate_g4_cpu.py` (run twice, identical both times):

```
baseline               forward ok, dG finite         PASS dG=-0.0030 width=1092
unfolded_emb=zero      forward ok, dG finite         PASS dG=0.0040 width=1092
flory_unfolded         forward ok, dG finite         PASS dG=-0.0058 width=1092
burial_features        forward ok, dG finite         PASS dG=0.0033 width=1095
burial hse             forward ok, dG finite         PASS dG=0.0017 width=1095
burial + uemb zero     forward ok, dG finite         PASS dG=0.0023 width=1095
width: levers that add no dims keep width            PASS 1092
width: burial adds exactly 3                         PASS 1095

G4-CPU: ALL PASS
```

Baseline `dG=-0.0030 width=1092`, all 8 PASS — unchanged by the guard, as expected: the guard is
argument validation and touches no feature vector.

**Not done, deliberately: nothing submitted, nothing cancelled, no sbatch, no push.**

---

## 8. Correction to the project record

`results/ctx8.md` and the task brief both state that `--slope_weight` "has still never been run".
**That is false and should be corrected wherever it is repeated.** `abl_p3_slope3.0_s42_e8` is a
`--slope_weight 3.0` run; it is row 5 of the 10-checkpoint `mean_rel_SASA` replication table
(r=0.676), so the replication has been quoting a slope-arm checkpoint as if it were a control.

That does not damage the exposure -> a_p finding: the correlation holds at 0.400-0.714 across all
ten checkpoints including the anchor arms, so it is not an artifact of any one training condition.
But the claim "nothing has ever attacked a_p" is wrong, and the honest version is stronger and
more useful: **a_p HAS been attacked once, at weight 3.0, and the attack made a_p worse**
(sd_ratio 0.4489 vs 0.5573 control). That is a real, if negative, result about factor C, and it
should be reported as one rather than left as an untried lever.
