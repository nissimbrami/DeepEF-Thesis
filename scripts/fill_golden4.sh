#!/bin/bash
# WAVE 4 - the levers that have CODE but were NEVER submitted end-to-end.
# Verified by scontrol scan: coil_edges=0, ligand_nodes=0, edge_features=0, gcn_span=0,
# gcn_bidir=0 jobs carry these flags anywhere in the queue.
#
# W7edge / W7span / U10bidir DID run once each (scored CSVs exist) and all landed inside the
# +/-0.060 seed band at n=1. n=1 inside a noise band is NOT a rejection - it is an untested
# arm. A second seed is the cheapest way to turn "inside noise" into a real answer.
#
# U5 coil_edges has NEVER run at all and is the missing half of the coil: the coil currently
# changes node distances but NOT the graph edges, so the unfolded state is only half applied.
cd /home/nissimb/DeepPEF
SUB="module load anaconda; source activate esm2_env_py38; export WANDB_MODE=disabled;"
BASE="--full_data --no_pretrain --no_freeze --pooled_corr_weight 0 --dg_length_norm none --affine_calib --val_frac 0.1 --epochs 15 --resume"
G="--partition=rtx6000 --qos keasar --gres=gpu:rtx_6000:1 --cpus-per-task=8 --time 12:00:00"
sub () {
  local N="$1"; shift
  squeue -u nissimb -h -o "%j" | grep -qx "$N" && { echo "skip $N (queued)"; return; }
  ls eval_results/abl_${N#gld_}_e*.csv >/dev/null 2>&1 && { echo "skip $N (scored)"; return; }
  local J=$(sbatch --parsable --job-name "$N" $G --output logs/${N}_%j.out --error logs/${N}_%j.out \
    --wrap "$SUB python Megascale-fineTuning/train.py $BASE $* --run_tag ${N#gld_}" 2>&1 | tail -1)
  echo "SUBMIT $N -> $J"
}
# --- U5: the coil's MISSING HALF. Never run. Trained on the unfolded chain (loss_mode dg). ---
sub gld_coiledge_dg_s42 --loss_mode dg --flory_unfolded --flory_nu 0.588 --coil_edges --seed 42
# --- W11 ligands: 659 lines of code, NEVER submitted once ---
sub gld_ligand_s42 --loss_mode ddg --ligand_nodes --seed 42
# --- second seeds for the three information levers stuck at n=1 inside the noise band ---
sub gld_w7edge_s1   --loss_mode ddg --edge_features --seed 1
sub gld_w7span4_s1  --loss_mode ddg --gcn_span 4 --seed 1
sub gld_u10bidir_s1 --loss_mode ddg --gcn_bidir --seed 1
echo "--- lane: $(squeue -u nissimb -h -o '%P' | grep -c rtx6000) ---"
