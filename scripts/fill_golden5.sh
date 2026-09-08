#!/bin/bash
# WAVE 5 - two levers that EXIST in train.py and have NEVER been run once.
#
# 1) --readout {sum,attention,gated}. Default 'sum'. Every run in this project used 'sum'.
#    WHY IT MATTERS: dG = E_unfolded - E_folded is a SUM of per-residue energies. A sum over
#    N residues scales with N, which is exactly the shape of a per-protein offset. b_p is
#    96.3% one global constant with a residual spread of 1.0155, and corr(b_p,N) = -0.215.
#    'attention'/'gated' learn a weighted aggregation instead, so the readout can normalise
#    itself rather than accumulating N terms. This is the FIRST lever that acts directly on
#    the aggregation that manufactures b_p -- every other lever acted on features or on scale.
#    SCORED ON: std(b_p) and dG (metric rule: this is a whole-protein reference-state lever,
#    ddG would cancel most of it). Degenerate check: report std(pred WT) against 0.9239.
#
# 2) --pooled_corr_weight. Default 0. THE PROJECT HAS NEVER TURNED THIS ON --
#    BASE carries "--pooled_corr_weight 0" in every single arm ever submitted.
#    It is a pooled CROSS-PROTEIN Pearson loss, i.e. a loss that optimises the exact quantity
#    the thesis reports (pooled ddG PCC 0.6382). Training on the metric you are judged by is
#    the most direct possible intervention and it has sat switched off the entire time.
#    RISK, stated up front: optimising pooled correlation directly can degrade per-protein
#    ranking (currently 0.798, already good). Report BOTH pooled and per-protein.
cd /home/nissimb/DeepPEF
SUB="module load anaconda; source activate esm2_env_py38; export WANDB_MODE=disabled;"
BASE="--full_data --no_pretrain --no_freeze --dg_length_norm none --affine_calib --val_frac 0.1 --epochs 15 --resume"
G="--partition=rtx6000 --qos keasar --gres=gpu:rtx_6000:1 --cpus-per-task=8 --time 12:00:00"
sub () {
  local N="$1"; shift
  squeue -u nissimb -h -o "%j" | grep -qx "$N" && { echo "skip $N (queued)"; return; }
  ls eval_results/abl_${N#gld_}_e*.csv >/dev/null 2>&1 && { echo "skip $N (scored)"; return; }
  local J=$(sbatch --parsable --job-name "$N" $G --output logs/${N}_%j.out --error logs/${N}_%j.out \
    --wrap "$SUB python Megascale-fineTuning/train.py $BASE $* --run_tag ${N#gld_}" 2>&1 | tail -1)
  echo "SUBMIT $N -> $J"
}
# --- readout: the aggregation that MAKES b_p. Scored on dG. ---
sub gld_readatt_dg_s42  --loss_mode dg --pooled_corr_weight 0 --readout attention --seed 42
sub gld_readgate_dg_s42 --loss_mode dg --pooled_corr_weight 0 --readout gated     --seed 42
# --- pooled correlation loss: trains the metric we report. Never once enabled. ---
sub gld_pooled0.3_s42 --loss_mode ddg --pooled_corr_weight 0.3 --seed 42
sub gld_pooled1.0_s42 --loss_mode ddg --pooled_corr_weight 1.0 --seed 42
# --- the two strongest levers combined with the pooled loss ---
sub gld_pooled_slope_s42 --loss_mode ddg --pooled_corr_weight 0.3 --slope_weight 1.0 --seed 42
echo "--- lane: $(squeue -u nissimb -h -o '%P' | grep -c rtx6000) ---"
